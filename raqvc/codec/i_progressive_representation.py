"""Constrained progressive I-frame representation for Wan-latent coding.

This module deliberately separates two operations that were conflated by
earlier entropy-only screens:

* a strictly invertible, identity-initialized channel transform that can move
  information between fixed decoder-known channel layers; and
* a bounded QP/group/channel precision offset around the accepted UF-style
  channel quantization tables.

Neither path is enabled by default.  The legacy codec and its checkpoints keep
their exact state-dict and bitstream behaviour when both features are off.
"""

from __future__ import annotations

import hashlib
import json
from typing import Any, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

PROGRESSIVE_REPRESENTATION_VERSION = 1
DEFAULT_PROGRESSIVE_GROUP_SIZES = (64, 64, 128)


def normalize_progressive_group_sizes(
    group_sizes: Sequence[int] | None,
    *,
    channels: int,
) -> tuple[int, ...]:
    values = DEFAULT_PROGRESSIVE_GROUP_SIZES if group_sizes is None else tuple(int(value) for value in group_sizes)
    if not values or any(value <= 0 for value in values):
        raise ValueError(f"progressive group sizes must be positive, got {values}")
    if sum(values) != int(channels):
        raise ValueError(f"progressive group sizes sum to {sum(values)}, expected {channels}")
    if len(values) < 3:
        raise ValueError(
            "progressive representation requires at least base, critical-detail, " "and optional-detail groups"
        )
    return values


def _contract_fingerprint(payload: dict[str, Any]) -> str:
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


class IProgressiveGivensTransform(nn.Module):
    """Efficient orthogonal 1x1 channel mixing built from Givens stages.

    Every stage applies independent two-channel rotations under a fixed
    permutation.  Zero angles are the exact identity.  The inverse traverses
    the stages in reverse with transposed rotations, so no learned inverse or
    numerical matrix inversion is required.
    """

    def __init__(
        self,
        *,
        channels: int = 256,
        stages: int = 6,
        max_angle: float = 0.35,
        group_sizes: Sequence[int] | None = None,
    ) -> None:
        super().__init__()
        self.channels = int(channels)
        self.stages = int(stages)
        self.max_angle = float(max_angle)
        self.group_sizes = normalize_progressive_group_sizes(group_sizes, channels=self.channels)
        if self.channels % 2:
            raise ValueError("Givens channel transform requires an even channel count")
        if self.stages < 1:
            raise ValueError("Givens channel transform requires at least one stage")
        if not 0.0 < self.max_angle <= 1.0:
            raise ValueError("max_angle must be in (0, 1] radians")

        # Odd multipliers are permutations modulo a power-of-two channel count.
        # The changing multiplier and shift expose every progressive layer to
        # cross-layer pairs without learning or signalling a permutation.
        odd_multipliers = (1, 73, 151, 199, 43, 109, 181, 229)
        base = torch.arange(self.channels, dtype=torch.long)
        permutations = []
        for stage in range(self.stages):
            multiplier = odd_multipliers[stage % len(odd_multipliers)]
            shift = (stage * 37) % self.channels
            permutation = (base * multiplier + shift) % self.channels
            if torch.unique(permutation).numel() != self.channels:
                raise RuntimeError("invalid fixed Givens channel permutation")
            permutations.append(permutation)
        self.register_buffer("permutations", torch.stack(permutations, dim=0), persistent=True)
        self.angle_logits = nn.Parameter(torch.zeros(self.stages, self.channels // 2))

    def _stage(
        self,
        value: torch.Tensor,
        stage: int,
        *,
        inverse: bool,
    ) -> torch.Tensor:
        permutation = self.permutations[int(stage)]
        ordered = value.index_select(1, permutation)
        first, second = ordered.chunk(2, dim=1)
        angle = (torch.tanh(self.angle_logits[int(stage)]) * self.max_angle).to(device=value.device, dtype=value.dtype)
        cosine = torch.cos(angle).view(1, -1, 1, 1)
        sine = torch.sin(angle).view(1, -1, 1, 1)
        if inverse:
            rotated_first = cosine * first + sine * second
            rotated_second = -sine * first + cosine * second
        else:
            rotated_first = cosine * first - sine * second
            rotated_second = sine * first + cosine * second
        rotated = torch.cat((rotated_first, rotated_second), dim=1)
        restored = torch.empty_like(value)
        restored.index_copy_(1, permutation, rotated)
        return restored

    def forward(self, value: torch.Tensor, *, inverse: bool = False) -> torch.Tensor:
        if value.ndim != 4 or int(value.shape[1]) != self.channels:
            raise ValueError(f"progressive transform expects [B,{self.channels},H,W], " f"got {tuple(value.shape)}")
        if inverse:
            for stage in reversed(range(self.stages)):
                value = self._stage(value, stage, inverse=True)
            return value
        for stage in range(self.stages):
            value = self._stage(value, stage, inverse=False)
        return value

    def contract_payload(self) -> dict[str, Any]:
        core = {
            "version": PROGRESSIVE_REPRESENTATION_VERSION,
            "kind": "fixed_permutation_givens_v1",
            "channels": self.channels,
            "stages": self.stages,
            "max_angle": self.max_angle,
            "group_sizes": list(self.group_sizes),
            "permutations": self.permutations.detach().cpu().tolist(),
        }
        return {**core, "fingerprint": _contract_fingerprint(core)}


class IProgressiveChannelQuantizer(nn.Module):
    """Bounded group plus zero-centred per-channel precision allocation.

    The multiplier is applied to q_enc and its reciprocal to q_dec.  Therefore
    the continuous (unquantized) analysis/synthesis scale is unchanged while
    the discrete precision and symbol rate can move between groups/channels.
    """

    def __init__(
        self,
        *,
        channels: int = 256,
        num_qp: int = 64,
        group_sizes: Sequence[int] | None = None,
        max_group_log: float = 0.35,
        max_channel_log: float = 0.20,
        center_group_log: bool = False,
    ) -> None:
        super().__init__()
        self.channels = int(channels)
        self.num_qp = int(num_qp)
        self.group_sizes = normalize_progressive_group_sizes(group_sizes, channels=self.channels)
        self.max_group_log = float(max_group_log)
        self.max_channel_log = float(max_channel_log)
        self.center_group_log = bool(center_group_log)
        if self.num_qp < 1:
            raise ValueError("num_qp must be positive")
        if self.max_group_log <= 0 or self.max_channel_log <= 0:
            raise ValueError("progressive q log bounds must be positive")
        self.group_logits = nn.Parameter(torch.zeros(self.num_qp, len(self.group_sizes)))
        self.channel_logits = nn.Parameter(torch.zeros(self.num_qp, self.channels))

    def log_precision(self, qp: int) -> torch.Tensor:
        qp_index = max(0, min(int(qp), self.num_qp - 1))
        group_log = self.group_log_precision(qp_index)
        channel_log = torch.tanh(self.channel_logits[qp_index]) * self.max_channel_log
        pieces = []
        group_pieces = []
        start = 0
        for group_index, size in enumerate(self.group_sizes):
            stop = start + int(size)
            local = channel_log[start:stop]
            local = local - local.mean()
            pieces.append(local)
            group_pieces.append(group_log[group_index].expand(int(size)))
            start = stop
        return torch.cat(group_pieces) + torch.cat(pieces)

    def precision(self, qp: int, like: torch.Tensor) -> torch.Tensor:
        value = torch.exp(self.log_precision(qp)).view(1, self.channels, 1, 1)
        return value.to(device=like.device, dtype=like.dtype)

    def group_log_precision(self, qp: int) -> torch.Tensor:
        qp_index = max(0, min(int(qp), self.num_qp - 1))
        raw = torch.tanh(self.group_logits[qp_index])
        if not self.center_group_log:
            return raw * self.max_group_log
        # Force the group allocation to be relative rather than learning a
        # duplicated global QP shift.  Weighting by channel count keeps the
        # geometric-mean precision over all 256 symbols exactly one.  The 1/2
        # factor retains the public max_group_log as a strict absolute bound:
        # raw and its weighted mean both lie in [-1, 1].
        weights = raw.new_tensor(self.group_sizes)
        weighted_mean = (raw * weights).sum() / weights.sum()
        return 0.5 * self.max_group_log * (raw - weighted_mean)

    def contract_payload(self) -> dict[str, Any]:
        core = {
            "version": 2 if self.center_group_log else 1,
            "kind": (
                "bounded_centered_group_channel_reciprocal_q_v2"
                if self.center_group_log
                else "bounded_group_channel_reciprocal_q_v1"
            ),
            "channels": self.channels,
            "num_qp": self.num_qp,
            "group_sizes": list(self.group_sizes),
            "max_group_log": self.max_group_log,
            "max_channel_log": self.max_channel_log,
            "center_group_log": self.center_group_log,
        }
        return {**core, "fingerprint": _contract_fingerprint(core)}


def progressive_representation_losses(
    value: torch.Tensor,
    group_sizes: Sequence[int],
) -> dict[str, torch.Tensor]:
    """Return differentiable energy-compaction/detail-separation diagnostics."""

    groups = normalize_progressive_group_sizes(group_sizes, channels=int(value.shape[1]))
    work = value.float()
    energy = work.square().mean(dim=(0, 2, 3))
    total_energy = energy.sum().clamp_min(1.0e-12)
    base_stop = groups[0]
    critical_stop = groups[0] + groups[1]
    base_fraction = energy[:base_stop].sum() / total_energy
    optional_fraction = energy[critical_stop:].sum() / total_energy

    lowpass = F.avg_pool2d(work, kernel_size=3, stride=1, padding=1)
    lowpass_energy = lowpass.square().mean(dim=(0, 2, 3))
    total_lowpass = lowpass_energy.sum().clamp_min(1.0e-12)
    base_lowpass_fraction = lowpass_energy[:base_stop].sum() / total_lowpass
    highpass_energy = (work - lowpass).square().mean(dim=(0, 2, 3))
    total_highpass = highpass_energy.sum().clamp_min(1.0e-12)
    critical_highpass_fraction = highpass_energy[base_stop:critical_stop].sum() / total_highpass
    return {
        # Do not ask the base layer to absorb the high frequencies that the
        # critical layer is explicitly meant to isolate.  Compaction acts on
        # the low-pass component; the complementary objective below acts on
        # the high-pass residual.
        "compact_loss": 1.0 - base_lowpass_fraction,
        "critical_highpass_loss": 1.0 - critical_highpass_fraction,
        "base_energy_fraction": base_fraction,
        "base_lowpass_fraction": base_lowpass_fraction,
        "critical_highpass_fraction": critical_highpass_fraction,
        "optional_energy_fraction": optional_fraction,
    }


def validate_progressive_contract(
    observed: dict[str, Any] | None,
    expected: dict[str, Any] | None,
) -> None:
    if observed is None and expected is None:
        return
    if observed is None or expected is None:
        raise ValueError("I progressive representation bitstream/config contract mismatch")
    if observed != expected:
        raise ValueError("I progressive representation contract mismatch: " f"stream={observed} decoder={expected}")
