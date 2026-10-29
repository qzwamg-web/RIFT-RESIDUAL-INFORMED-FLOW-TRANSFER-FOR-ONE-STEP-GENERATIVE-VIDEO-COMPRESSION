"""Versioned ELIC-style channel context for the I-frame entropy model.

This module deliberately does *not* replace the accepted DMCI analysis,
hyperprior, quantizer, or synthesis transforms.  It only predicts bounded
corrections to the Gaussian parameters of one channel group from information
that a real decoder already has: the group's current hyper/spatial-prior
parameters, QP, and completed earlier channel groups.

The final projection of every head is zero initialized.  Enabling a lane on a
legacy checkpoint therefore starts from the exact legacy reconstruction and
coder-aligned Gaussian distribution.  The ``legacy`` lane is handled by
:mod:`raqvc.codec.rt_two_codec` without creating this module at all, so old
configurations keep an identical state dict and bitstream path.
"""

from __future__ import annotations

import hashlib
import json
from typing import Any, Optional, Sequence

import torch
from torch import nn

I_GROUP_CONTEXT_CONTRACT_VERSION = 1
I_GROUP_CONTEXT_LANES = (
    "legacy",
    "control",
    "even8",
    "uneven-index",
    "uneven-permuted",
)
I_GROUP_CONTEXT_PARAMETERIZATIONS = ("scale", "mean_scale")


def normalize_i_group_context_lane(value: str) -> str:
    """Normalize the public lane name while retaining an explicit legacy mode."""

    lane = str(value).strip().lower().replace("_", "-")
    aliases = {
        "none": "legacy",
        "off": "legacy",
        "disabled": "legacy",
        "even-8": "even8",
        "uneven": "uneven-index",
        "permuted": "uneven-permuted",
    }
    lane = aliases.get(lane, lane)
    if lane not in I_GROUP_CONTEXT_LANES:
        raise ValueError(f"Unknown I group-context lane {value!r}; expected one of " f"{I_GROUP_CONTEXT_LANES}.")
    return lane


def normalize_i_group_context_parameterization(value: str) -> str:
    """Normalize scale-only versus bounded mean-and-scale correction mode."""

    mode = str(value).strip().lower().replace("-", "_")
    aliases = {
        "scale_only": "scale",
        "sigma": "scale",
        "mu_sigma": "mean_scale",
        "mean+scale": "mean_scale",
    }
    mode = aliases.get(mode, mode)
    if mode not in I_GROUP_CONTEXT_PARAMETERIZATIONS:
        raise ValueError("I group-context parameterization must be 'scale' or " f"'mean_scale', got {value!r}.")
    return mode


def _group_sizes_for_lane(lane: str, channels: int) -> tuple[int, ...]:
    if lane == "control":
        return (channels,)
    if lane == "even8":
        if channels % 8:
            raise ValueError(f"even8 I group context requires channels divisible by 8, got {channels}.")
        return (channels // 8,) * 8
    if lane in {"uneven-index", "uneven-permuted"}:
        if channels != 256:
            raise ValueError("The Task218 uneven I-y contract is defined for exactly 256 " f"channels, got {channels}.")
        return (16, 16, 32, 64, 128)
    raise ValueError(f"Lane {lane!r} does not instantiate a group context module.")


def _validate_permutation(
    permutation: Optional[Sequence[int]],
    *,
    channels: int,
    required: bool,
) -> tuple[int, ...]:
    if permutation is None:
        if required:
            raise ValueError(
                "uneven-permuted requires an explicit fixed 256-entry "
                "permutation; refusing to invent an unvalidated channel order."
            )
        return tuple(range(channels))
    values = tuple(int(value) for value in permutation)
    if len(values) != channels:
        raise ValueError(f"I group permutation must contain {channels} entries, got {len(values)}.")
    if sorted(values) != list(range(channels)):
        raise ValueError(f"I group permutation must contain every index in [0,{channels - 1}] exactly once.")
    return values


class _IGroupEntropyDeltaHead(nn.Module):
    """One zero-init Gaussian correction head for a fixed channel group."""

    def __init__(
        self,
        *,
        group_channels: int,
        previous_channels: int,
        hidden_channels: int,
        parameterization: str,
        max_log_scale: float,
        max_mean_delta: float,
    ) -> None:
        super().__init__()
        self.group_channels = int(group_channels)
        self.previous_channels = int(previous_channels)
        self.parameterization = normalize_i_group_context_parameterization(parameterization)
        self.max_log_scale = float(max_log_scale)
        self.max_mean_delta = float(max_mean_delta)
        if self.group_channels <= 0:
            raise ValueError("I group size must be positive.")
        if self.previous_channels < 0:
            raise ValueError("Previous-group channel count must be non-negative.")
        if self.max_log_scale <= 0:
            raise ValueError("I group max_log_scale must be positive.")
        if self.max_mean_delta <= 0:
            raise ValueError("I group max_mean_delta must be positive.")

        hidden_channels = int(hidden_channels)
        if hidden_channels <= 0:
            raise ValueError("I group hidden_channels must be positive.")
        # Decoder-known QP and spatial-pass scalars remove two otherwise
        # ambiguous cases from the context tensor: a genuine decoded zero and
        # a position which has not been visited yet.  Availability itself is
        # fixed by the versioned pass/group traversal contract.
        input_channels = 2 * self.group_channels + self.previous_channels + 2
        output_channels = self.group_channels * (2 if self.parameterization == "mean_scale" else 1)
        self.net = nn.Sequential(
            nn.Conv2d(input_channels, hidden_channels, 1),
            nn.GELU(),
            nn.Conv2d(hidden_channels, hidden_channels, 3, padding=1),
            nn.GELU(),
            nn.Conv2d(hidden_channels, output_channels, 1),
        )
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    @staticmethod
    def _qp_map(qp: int, like: torch.Tensor) -> torch.Tensor:
        # I QP is decoder-known and defined over [0,63].
        qp_value = 2.0 * max(0, min(int(qp), 63)) / 63.0 - 1.0
        return like.new_full((like.shape[0], 1, like.shape[2], like.shape[3]), qp_value)

    @staticmethod
    def _pass_map(pass_index: int, like: torch.Tensor) -> torch.Tensor:
        index = int(pass_index)
        if not 0 <= index < 4:
            raise ValueError(f"I spatial pass must be in [0,3], got {pass_index}.")
        return like.new_full((like.shape[0], 1, like.shape[2], like.shape[3]), index / 3.0)

    def forward(
        self,
        scales: torch.Tensor,
        means: torch.Tensor,
        previous: Optional[torch.Tensor],
        qp: int,
        pass_index: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if scales.shape != means.shape or scales.shape[1] != self.group_channels:
            raise ValueError(
                "I group context received incompatible scale/mean tensors: "
                f"scales={tuple(scales.shape)} means={tuple(means.shape)} "
                f"expected_channels={self.group_channels}."
            )
        # Match DCVC's real Gaussian index builder: signed predictions are
        # clamped directly to its representable [0.11,16] sigma table.  In
        # particular, ``abs(scales)`` would select the wrong CDF for a negative
        # inherited prediction.  A positive floor also leaves the new delta
        # head trainable at those locations.
        base_sigma = scales.clamp(0.11, 16.0)
        features = [
            torch.log(base_sigma),
            means,
        ]
        if self.previous_channels:
            if previous is None or previous.shape[1] != self.previous_channels:
                actual = None if previous is None else tuple(previous.shape)
                raise ValueError(
                    "I group context requires completed previous groups with "
                    f"{self.previous_channels} channels, got {actual}."
                )
            if previous.shape[0] != scales.shape[0] or previous.shape[-2:] != scales.shape[-2:]:
                raise ValueError("Previous I group context must match batch and spatial shape.")
            # Bound only the conditioner feature, never the decoded symbol used
            # by synthesis.  Encoder and decoder apply this deterministic map.
            features.append(torch.tanh(previous / 8.0))
        elif previous is not None and previous.shape[1] != 0:
            raise ValueError("The first/control I group cannot consume future context.")
        features.append(self._qp_map(qp, scales))
        features.append(self._pass_map(pass_index, scales))
        raw = self.net(torch.cat(features, dim=1))

        if self.parameterization == "mean_scale":
            raw_log_scale, raw_mean = raw.chunk(2, dim=1)
            mean_delta = torch.tanh(raw_mean) * self.max_mean_delta
            adjusted_means = means + mean_delta
        else:
            raw_log_scale = raw
            adjusted_means = means
        log_scale_delta = torch.tanh(raw_log_scale) * self.max_log_scale
        adjusted_scales = (base_sigma * torch.exp(log_scale_delta)).clamp(0.11, 16.0)
        return adjusted_scales, adjusted_means


class IGroupEntropyContext(nn.Module):
    """Fixed-group causal entropy context for DMCI's 256-channel I-y tensor.

    ``permutation`` maps group-order positions to original DMCI channel indices:
    ``group_order = original[:, permutation]``.  It is a registered buffer and
    is fingerprinted into every new bitstream contract.
    """

    def __init__(
        self,
        *,
        channels: int = 256,
        lane: str = "control",
        parameterization: str = "scale",
        hidden_channels: int = 96,
        max_log_scale: float = 0.75,
        max_mean_delta: float = 2.0,
        permutation: Optional[Sequence[int]] = None,
    ) -> None:
        super().__init__()
        self.channels = int(channels)
        self.lane = normalize_i_group_context_lane(lane)
        if self.lane == "legacy":
            raise ValueError("The legacy lane must bypass IGroupEntropyContext entirely.")
        self.parameterization = normalize_i_group_context_parameterization(parameterization)
        self.group_sizes = _group_sizes_for_lane(self.lane, self.channels)
        if sum(self.group_sizes) != self.channels:
            raise AssertionError("I group sizes do not cover the y channels.")
        if any(size % 4 for size in self.group_sizes):
            raise ValueError(
                "Every I group must be divisible by four to retain the DMCI "
                f"four-part spatial prior, got {self.group_sizes}."
            )
        values = _validate_permutation(
            permutation,
            channels=self.channels,
            required=self.lane == "uneven-permuted",
        )
        if self.lane != "uneven-permuted" and values != tuple(range(self.channels)):
            raise ValueError(
                f"Lane {self.lane!r} has a fixed identity order; use "
                "'uneven-permuted' for a non-identity permutation."
            )
        permutation_tensor = torch.tensor(values, dtype=torch.long)
        inverse = torch.empty_like(permutation_tensor)
        inverse[permutation_tensor] = torch.arange(self.channels, dtype=torch.long)
        self.register_buffer("permutation", permutation_tensor, persistent=True)
        self.register_buffer("inverse_permutation", inverse, persistent=True)

        # Control deliberately has no decoded-group input.  Every other lane is
        # causal over all completed groups; the current group is handled only by
        # the inherited four-pass spatial prior.
        self.uses_previous_groups = self.lane != "control"
        heads = []
        previous_channels = 0
        for group_channels in self.group_sizes:
            heads.append(
                _IGroupEntropyDeltaHead(
                    group_channels=group_channels,
                    previous_channels=(previous_channels if self.uses_previous_groups else 0),
                    hidden_channels=int(hidden_channels),
                    parameterization=self.parameterization,
                    max_log_scale=float(max_log_scale),
                    max_mean_delta=float(max_mean_delta),
                )
            )
            previous_channels += group_channels
        self.heads = nn.ModuleList(heads)

    @property
    def group_count(self) -> int:
        return len(self.group_sizes)

    def permutation_fingerprint(self) -> str:
        payload = json.dumps(
            [int(value) for value in self.permutation.detach().cpu().tolist()],
            separators=(",", ":"),
        ).encode("ascii")
        return hashlib.sha256(payload).hexdigest()[:24]

    def contract_payload(self) -> dict[str, Any]:
        return {
            "version": I_GROUP_CONTEXT_CONTRACT_VERSION,
            "lane": self.lane,
            "channels": self.channels,
            "group_sizes": list(self.group_sizes),
            "permutation_fingerprint": self.permutation_fingerprint(),
            "parameterization": self.parameterization,
            "scale_contract": "dcvc_signed_clamp_0.11_16",
            "context_source": (
                "earlier_groups_hard_prefix_by_spatial_pass" if self.uses_previous_groups else "hyper_spatial_only"
            ),
            "order": "spatial_pass_major_group_minor_v1",
            "spatial_passes": 4,
        }

    def validate_contract(self, payload: Any) -> None:
        if not isinstance(payload, dict):
            raise ValueError("I group-context bitstream is missing its versioned contract.")
        expected = self.contract_payload()
        mismatches = {
            key: (payload.get(key), expected_value)
            for key, expected_value in expected.items()
            if payload.get(key) != expected_value
        }
        if mismatches:
            raise ValueError(
                "I group-context bitstream contract mismatch: "
                + ", ".join(
                    f"{key}=stream:{actual!r}/decoder:{wanted!r}" for key, (actual, wanted) in mismatches.items()
                )
            )

    def to_group_order(self, value: torch.Tensor) -> torch.Tensor:
        if value.shape[1] != self.channels:
            raise ValueError(f"I group context expected {self.channels} channels, got {value.shape[1]}.")
        return value.index_select(1, self.permutation)

    def to_original_order(self, value: torch.Tensor) -> torch.Tensor:
        if value.shape[1] != self.channels:
            raise ValueError(f"I group context expected {self.channels} channels, got {value.shape[1]}.")
        return value.index_select(1, self.inverse_permutation)

    def split_group_order(self, value: torch.Tensor) -> tuple[torch.Tensor, ...]:
        if value.shape[1] != self.channels:
            raise ValueError(f"I group context expected {self.channels} channels, got {value.shape[1]}.")
        return tuple(value.split(self.group_sizes, dim=1))

    def group_indices(self, group_index: int) -> torch.Tensor:
        index = int(group_index)
        if not 0 <= index < self.group_count:
            raise IndexError(f"I group index {group_index} outside [0,{self.group_count - 1}].")
        start = sum(self.group_sizes[:index])
        return self.permutation[start : start + self.group_sizes[index]]

    def adjust(
        self,
        group_index: int,
        scales: torch.Tensor,
        means: torch.Tensor,
        completed_groups: Sequence[torch.Tensor],
        qp: int,
        pass_index: int = 0,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Correct one group's Gaussian parameters from hard prior groups only."""

        index = int(group_index)
        if len(completed_groups) != index:
            raise ValueError(
                "I group context is causal: completed group count must equal "
                f"the current index ({len(completed_groups)} != {index})."
            )
        previous: Optional[torch.Tensor]
        if self.uses_previous_groups and completed_groups:
            previous = torch.cat(tuple(completed_groups), dim=1)
        else:
            previous = None
        return self.heads[index](scales, means, previous, qp, pass_index)
