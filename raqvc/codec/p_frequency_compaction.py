"""Contextual frequency-compaction coordinate for the latent P codec.

The accepted P codec already produces a 128-channel residual representation
``y`` and has a decoder-known temporal context.  This module inserts an
identity-initialized analysis/synthesis pair around entropy coding so the
jointly trained P core can learn three different channel roles:

* a compact base group for predictable low-frequency structure;
* a critical-detail group for spatial high-frequency residuals;
* an optional texture group for the remaining information.

Both directions use only QP, P-position type and the decoded-reference
context, all of which are available to encoder and decoder.  The output heads
are zero initialized, so enabling the module is exactly equivalent to the
parent codec before training.  The feature is opt-in and carries a versioned
bitstream contract.
"""

from __future__ import annotations

import hashlib
import json
from typing import Any, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

P_FREQUENCY_COMPACTION_VERSION = 1
P_FREQUENCY_LIFTING_VERSION = 2
DEFAULT_P_GROUP_SIZES = (32, 48, 48)


def normalize_p_group_sizes(values: Sequence[int] | None, *, channels: int) -> tuple[int, int, int]:
    sizes = DEFAULT_P_GROUP_SIZES if values is None else tuple(int(v) for v in values)
    if len(sizes) != 3 or any(v <= 0 for v in sizes):
        raise ValueError("P frequency group sizes must contain three positive integers, " f"got {sizes}.")
    if sum(sizes) != int(channels):
        raise ValueError(f"P frequency group sizes must sum to {channels}, got {sizes}.")
    return sizes


def _fingerprint(payload: dict[str, Any]) -> str:
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


class _ResidualBlock(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        groups = min(32, int(channels))
        while int(channels) % groups:
            groups -= 1
        self.norm = nn.GroupNorm(groups, channels)
        self.expand = nn.Conv2d(channels, channels * 2, 1)
        self.depthwise = nn.Conv2d(
            channels * 2,
            channels * 2,
            3,
            padding=1,
            groups=channels * 2,
        )
        self.project = nn.Conv2d(channels * 2, channels, 1)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        residual = self.norm(value)
        residual = F.silu(self.expand(residual))
        residual = F.silu(self.depthwise(residual))
        return value + self.project(residual)


class _ContextualFrequencyPath(nn.Module):
    """One identity-initialized direction of the P coordinate transform."""

    def __init__(
        self,
        *,
        channels: int,
        context_channels: int,
        hidden_channels: int,
        blocks: int,
        num_qp: int,
        p_type_count: int,
        residual_scale: float,
        group_sizes: tuple[int, int, int],
    ) -> None:
        super().__init__()
        self.channels = int(channels)
        self.context_channels = int(context_channels)
        self.hidden_channels = int(hidden_channels)
        self.residual_scale = float(residual_scale)
        self.group_sizes = tuple(group_sizes)

        # Low/high decomposition is explicit.  The network cannot accidentally
        # treat this as a plain 1x1 channel rotation.
        self.value_input = nn.Conv2d(self.channels * 2, self.hidden_channels, 1)
        self.context_input = nn.Conv2d(self.context_channels * 2, self.hidden_channels, 1)
        self.qp_embedding = nn.Embedding(int(num_qp), self.hidden_channels)
        self.type_embedding = nn.Embedding(int(p_type_count), self.hidden_channels)
        self.blocks = nn.ModuleList([_ResidualBlock(self.hidden_channels) for _ in range(int(blocks))])
        self.group_outputs = nn.ModuleList([nn.Conv2d(self.hidden_channels, size, 1) for size in self.group_sizes])

        nn.init.zeros_(self.qp_embedding.weight)
        nn.init.zeros_(self.type_embedding.weight)
        for output in self.group_outputs:
            nn.init.zeros_(output.weight)
            nn.init.zeros_(output.bias)

    @staticmethod
    def _frequency_pair(value: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        low = F.avg_pool2d(value, kernel_size=3, stride=1, padding=1)
        return low, value - low

    def forward(
        self,
        value: torch.Tensor,
        context: torch.Tensor,
        qp: int,
        frame_type_idx: int,
    ) -> torch.Tensor:
        if context.shape[-2:] != value.shape[-2:]:
            context = F.interpolate(
                context,
                size=value.shape[-2:],
                mode="bilinear",
                align_corners=False,
            )
        context = context.to(device=value.device, dtype=value.dtype)
        value_low, value_high = self._frequency_pair(value)
        context_low, context_high = self._frequency_pair(context)
        hidden = self.value_input(torch.cat((value_low, value_high), dim=1))
        hidden = hidden + self.context_input(torch.cat((context_low, context_high), dim=1))
        qp_idx = max(0, min(int(qp), self.qp_embedding.num_embeddings - 1))
        type_idx = max(0, min(int(frame_type_idx), self.type_embedding.num_embeddings - 1))
        condition = self.qp_embedding.weight[qp_idx] + self.type_embedding.weight[type_idx]
        hidden = hidden + condition.to(device=value.device, dtype=value.dtype).view(1, -1, 1, 1)
        for block in self.blocks:
            hidden = block(hidden)
        correction = torch.cat([head(hidden) for head in self.group_outputs], dim=1)
        return value + self.residual_scale * correction


class _ContextualLiftingCorrection(nn.Module):
    """Zero-initialized coupling correction from decoder-known conditions."""

    def __init__(
        self,
        *,
        input_channels: int,
        output_channels: int,
        context_channels: int,
        hidden_channels: int,
        blocks: int,
        num_qp: int,
        p_type_count: int,
        residual_scale: float,
    ) -> None:
        super().__init__()
        self.residual_scale = float(residual_scale)
        self.value_input = nn.Conv2d(int(input_channels) * 2, int(hidden_channels), 1)
        self.context_input = nn.Conv2d(int(context_channels) * 2, int(hidden_channels), 1)
        self.qp_embedding = nn.Embedding(int(num_qp), int(hidden_channels))
        self.type_embedding = nn.Embedding(int(p_type_count), int(hidden_channels))
        self.blocks = nn.ModuleList([_ResidualBlock(int(hidden_channels)) for _ in range(int(blocks))])
        self.output = nn.Conv2d(int(hidden_channels), int(output_channels), 1)
        nn.init.zeros_(self.qp_embedding.weight)
        nn.init.zeros_(self.type_embedding.weight)
        nn.init.zeros_(self.output.weight)
        nn.init.zeros_(self.output.bias)

    @staticmethod
    def _frequency_pair(value: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        low = F.avg_pool2d(value, kernel_size=3, stride=1, padding=1)
        return low, value - low

    def forward(
        self,
        value: torch.Tensor,
        context: torch.Tensor,
        qp: int,
        frame_type_idx: int,
    ) -> torch.Tensor:
        if context.shape[-2:] != value.shape[-2:]:
            context = F.interpolate(
                context,
                size=value.shape[-2:],
                mode="bilinear",
                align_corners=False,
            )
        context = context.to(device=value.device, dtype=value.dtype)
        value_low, value_high = self._frequency_pair(value)
        context_low, context_high = self._frequency_pair(context)
        hidden = self.value_input(torch.cat((value_low, value_high), dim=1))
        hidden = hidden + self.context_input(torch.cat((context_low, context_high), dim=1))
        qp_idx = max(0, min(int(qp), self.qp_embedding.num_embeddings - 1))
        type_idx = max(0, min(int(frame_type_idx), self.type_embedding.num_embeddings - 1))
        condition = self.qp_embedding.weight[qp_idx] + self.type_embedding.weight[type_idx]
        hidden = hidden + condition.to(device=value.device, dtype=value.dtype).view(1, -1, 1, 1)
        for block in self.blocks:
            hidden = block(hidden)
        return self.residual_scale * self.output(hidden)


class _ContextualFrequencyLifting(nn.Module):
    """Two-stage exactly invertible base/detail/optional lifting transform."""

    def __init__(
        self,
        *,
        context_channels: int,
        hidden_channels: int,
        blocks: int,
        num_qp: int,
        p_type_count: int,
        residual_scale: float,
        group_sizes: tuple[int, int, int],
    ) -> None:
        super().__init__()
        self.group_sizes = tuple(int(value) for value in group_sizes)
        base, detail, optional = self.group_sizes
        rest = detail + optional
        common = dict(
            context_channels=int(context_channels),
            hidden_channels=int(hidden_channels),
            blocks=int(blocks),
            num_qp=int(num_qp),
            p_type_count=int(p_type_count),
            residual_scale=float(residual_scale),
        )
        # Stage one exchanges information between the compact base and the
        # combined detail payload. Stage two separates critical from optional
        # detail. Applying the same functions in reverse order with subtraction
        # is an exact inverse, unlike two independently learned residual paths.
        self.predict_rest = _ContextualLiftingCorrection(input_channels=base, output_channels=rest, **common)
        self.update_base = _ContextualLiftingCorrection(input_channels=rest, output_channels=base, **common)
        self.predict_optional = _ContextualLiftingCorrection(input_channels=detail, output_channels=optional, **common)
        self.update_detail = _ContextualLiftingCorrection(input_channels=optional, output_channels=detail, **common)

    def analysis(
        self,
        value: torch.Tensor,
        context: torch.Tensor,
        qp: int,
        frame_type_idx: int,
    ) -> torch.Tensor:
        base, detail, optional = value.split(self.group_sizes, dim=1)
        rest = torch.cat((detail, optional), dim=1)
        rest = rest + self.predict_rest(base, context, qp, frame_type_idx)
        base = base + self.update_base(rest, context, qp, frame_type_idx)
        detail, optional = rest.split(self.group_sizes[1:], dim=1)
        optional = optional + self.predict_optional(detail, context, qp, frame_type_idx)
        detail = detail + self.update_detail(optional, context, qp, frame_type_idx)
        return torch.cat((base, detail, optional), dim=1)

    def synthesis(
        self,
        value: torch.Tensor,
        context: torch.Tensor,
        qp: int,
        frame_type_idx: int,
    ) -> torch.Tensor:
        base, detail, optional = value.split(self.group_sizes, dim=1)
        detail = detail - self.update_detail(optional, context, qp, frame_type_idx)
        optional = optional - self.predict_optional(detail, context, qp, frame_type_idx)
        rest = torch.cat((detail, optional), dim=1)
        base = base - self.update_base(rest, context, qp, frame_type_idx)
        rest = rest - self.predict_rest(base, context, qp, frame_type_idx)
        detail, optional = rest.split(self.group_sizes[1:], dim=1)
        return torch.cat((base, detail, optional), dim=1)


class PContextualFrequencyCompaction(nn.Module):
    """Learned P analysis/synthesis coordinate with semantic channel groups."""

    def __init__(
        self,
        *,
        channels: int = 128,
        context_channels: int = 256,
        hidden_channels: int = 160,
        blocks: int = 3,
        num_qp: int = 72,
        p_type_count: int = 3,
        residual_scale: float = 0.25,
        group_sizes: Sequence[int] | None = None,
        mode: str = "residual",
    ) -> None:
        super().__init__()
        self.channels = int(channels)
        self.context_channels = int(context_channels)
        self.hidden_channels = int(hidden_channels)
        self.blocks_count = int(blocks)
        self.num_qp = int(num_qp)
        self.p_type_count = int(p_type_count)
        self.residual_scale = float(residual_scale)
        self.group_sizes = normalize_p_group_sizes(group_sizes, channels=self.channels)
        self.mode = str(mode).strip().lower()
        if self.mode not in {"residual", "lifting"}:
            raise ValueError("P frequency mode must be 'residual' or 'lifting', got " f"{mode!r}.")
        if self.hidden_channels < 32:
            raise ValueError("P frequency hidden_channels must be at least 32.")
        if self.blocks_count < 1:
            raise ValueError("P frequency blocks must be positive.")
        if self.num_qp < 1 or self.p_type_count < 1:
            raise ValueError("P frequency QP/type counts must be positive.")
        if not 0.0 < self.residual_scale <= 1.0:
            raise ValueError("P frequency residual_scale must be in (0, 1].")

        common = dict(
            channels=self.channels,
            context_channels=self.context_channels,
            hidden_channels=self.hidden_channels,
            blocks=self.blocks_count,
            num_qp=self.num_qp,
            p_type_count=self.p_type_count,
            residual_scale=self.residual_scale,
            group_sizes=self.group_sizes,
        )
        self.analysis_path: _ContextualFrequencyPath | None = None
        self.synthesis_path: _ContextualFrequencyPath | None = None
        self.lifting_path: _ContextualFrequencyLifting | None = None
        if self.mode == "residual":
            self.analysis_path = _ContextualFrequencyPath(**common)
            self.synthesis_path = _ContextualFrequencyPath(**common)
        else:
            common.pop("channels")
            self.lifting_path = _ContextualFrequencyLifting(**common)

    def _validate(self, value: torch.Tensor, context: torch.Tensor) -> None:
        if value.ndim != 4 or int(value.shape[1]) != self.channels:
            raise ValueError(f"P frequency transform expects [B,{self.channels},H,W], " f"got {tuple(value.shape)}.")
        if context.ndim != 4 or int(context.shape[1]) != self.context_channels:
            raise ValueError(
                "P frequency context expects " f"[B,{self.context_channels},H,W], got {tuple(context.shape)}."
            )

    def analysis(
        self,
        value: torch.Tensor,
        context: torch.Tensor,
        qp: int,
        frame_type_idx: int,
    ) -> torch.Tensor:
        self._validate(value, context)
        if self.lifting_path is not None:
            return self.lifting_path.analysis(value, context, qp, frame_type_idx)
        assert self.analysis_path is not None
        return self.analysis_path(value, context, qp, frame_type_idx)

    def synthesis(
        self,
        value: torch.Tensor,
        context: torch.Tensor,
        qp: int,
        frame_type_idx: int,
    ) -> torch.Tensor:
        self._validate(value, context)
        if self.lifting_path is not None:
            return self.lifting_path.synthesis(value, context, qp, frame_type_idx)
        assert self.synthesis_path is not None
        return self.synthesis_path(value, context, qp, frame_type_idx)

    def progressive_auxiliary(
        self,
        source: torch.Tensor,
        representation: torch.Tensor,
        context: torch.Tensor,
        qp: int,
        frame_type_idx: int,
    ) -> dict[str, torch.Tensor]:
        """Measure and supervise base/detail ownership in the P coordinate."""
        self._validate(source, context)
        self._validate(representation, context)
        base_stop = int(self.group_sizes[0])
        detail_stop = base_stop + int(self.group_sizes[1])

        base_symbols = torch.zeros_like(representation)
        base_symbols[:, :base_stop] = representation[:, :base_stop]
        detail_symbols = torch.zeros_like(representation)
        detail_symbols[:, :detail_stop] = representation[:, :detail_stop]
        base_recon = self.synthesis(base_symbols, context, qp, frame_type_idx).float()
        detail_recon = self.synthesis(detail_symbols, context, qp, frame_type_idx).float()

        source_work = source.float()
        source_low = F.avg_pool2d(source_work, 3, stride=1, padding=1)
        source_high = source_work - source_low
        base_low = F.avg_pool2d(base_recon, 3, stride=1, padding=1)
        detail_high = detail_recon - F.avg_pool2d(detail_recon, 3, stride=1, padding=1)
        work = representation.float()
        energy = work.square().mean(dim=(0, 2, 3))
        low = F.avg_pool2d(work, 3, stride=1, padding=1)
        high = work - low
        low_energy = low.square().mean(dim=(0, 2, 3))
        high_energy = high.square().mean(dim=(0, 2, 3))
        base_lowpass_fraction = low_energy[:base_stop].sum() / low_energy.sum().clamp_min(1.0e-12)
        critical_highpass_fraction = high_energy[base_stop:detail_stop].sum() / high_energy.sum().clamp_min(1.0e-12)

        # Prefix reconstruction makes synthesis useful when groups are
        # truncated.  The ownership terms additionally reach analysis at step
        # zero, preventing an identity-initialized path from waiting for the
        # synthesis heads to move before it can start redistributing energy.
        compact_loss = (
            F.mse_loss(base_low, source_low) / source_low.square().mean().clamp_min(1.0e-6)
            + 1.0
            - base_lowpass_fraction
        )
        critical_highpass_loss = (
            F.mse_loss(detail_high, source_high) / source_high.square().mean().clamp_min(1.0e-6)
            + 1.0
            - critical_highpass_fraction
        )
        return {
            "compact_loss": compact_loss,
            "critical_highpass_loss": critical_highpass_loss,
            "base_energy_fraction": energy[:base_stop].sum() / energy.sum().clamp_min(1.0e-12),
            "base_lowpass_fraction": base_lowpass_fraction,
            "critical_highpass_fraction": critical_highpass_fraction,
            "optional_energy_fraction": energy[detail_stop:].sum() / energy.sum().clamp_min(1.0e-12),
        }

    def contract_payload(self) -> dict[str, Any]:
        if self.mode == "lifting":
            core = {
                "version": P_FREQUENCY_LIFTING_VERSION,
                "kind": "contextual_exact_lifting_p_analysis_synthesis_v2",
                "mode": self.mode,
                "channels": self.channels,
                "context_channels": self.context_channels,
                "hidden_channels": self.hidden_channels,
                "blocks": self.blocks_count,
                "num_qp": self.num_qp,
                "p_type_count": self.p_type_count,
                "residual_scale": self.residual_scale,
                "group_sizes": list(self.group_sizes),
            }
            return {**core, "fingerprint": _fingerprint(core)}
        core = {
            "version": P_FREQUENCY_COMPACTION_VERSION,
            "kind": "contextual_low_high_p_analysis_synthesis_v1",
            "channels": self.channels,
            "context_channels": self.context_channels,
            "hidden_channels": self.hidden_channels,
            "blocks": self.blocks_count,
            "num_qp": self.num_qp,
            "p_type_count": self.p_type_count,
            "residual_scale": self.residual_scale,
            "group_sizes": list(self.group_sizes),
        }
        return {**core, "fingerprint": _fingerprint(core)}


def validate_p_frequency_contract(
    observed: dict[str, Any] | None,
    expected: dict[str, Any] | None,
) -> None:
    if observed is None and expected is None:
        return
    if observed is None or expected is None or observed != expected:
        raise ValueError("P frequency-compaction contract mismatch: " f"stream={observed} decoder={expected}")
