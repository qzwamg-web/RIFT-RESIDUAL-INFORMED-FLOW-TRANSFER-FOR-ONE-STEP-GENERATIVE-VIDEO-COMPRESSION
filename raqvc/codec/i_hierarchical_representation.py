"""Learned hierarchical analysis/synthesis coordinate for the I codec.

The Task219/220 Givens screens could only rotate the accepted ``y`` tensor
while the hyperprior and decoder remained frozen.  That is a useful negative
control, but it cannot create a new lossy representation.  This module instead
provides independent, identity-initialized analysis and synthesis networks.
It is intended to be trained jointly with the I hyperprior, spatial prior and
decoder while the accepted QP/channel quantization tables remain frozen.

The feature is opt-in.  A legacy config does not instantiate the module and
therefore retains its exact state-dict and bitstream behaviour.
"""

from __future__ import annotations

import hashlib
import json
from typing import Any, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

from raqvc.codec.i_progressive_representation import (
    DEFAULT_PROGRESSIVE_GROUP_SIZES,
    normalize_progressive_group_sizes,
)

HIERARCHICAL_REPRESENTATION_VERSION = 1


def _fingerprint(payload: dict[str, Any]) -> str:
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


class _HierarchyResidualBlock(nn.Module):
    """Spatial/channel residual block with bounded memory at 256 channels."""

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
        residual = F.gelu(self.expand(residual))
        residual = F.gelu(self.depthwise(residual))
        return value + self.project(residual)


class _HierarchyPath(nn.Module):
    """One direction of the learned hierarchy, exactly identity at init."""

    def __init__(
        self,
        *,
        channels: int,
        hidden_channels: int,
        blocks: int,
        num_qp: int,
        residual_scale: float,
    ) -> None:
        super().__init__()
        self.channels = int(channels)
        self.hidden_channels = int(hidden_channels)
        self.residual_scale = float(residual_scale)
        self.input = nn.Conv2d(self.channels * 2, self.hidden_channels, 1)
        self.qp_embedding = nn.Embedding(int(num_qp), self.hidden_channels)
        self.blocks = nn.ModuleList([_HierarchyResidualBlock(self.hidden_channels) for _ in range(int(blocks))])
        self.output = nn.Conv2d(self.hidden_channels, self.channels, 1)

        # The surrounding accepted codec is reproduced bit-for-bit at step 0.
        nn.init.zeros_(self.qp_embedding.weight)
        nn.init.zeros_(self.output.weight)
        nn.init.zeros_(self.output.bias)

    def forward(self, value: torch.Tensor, qp: int) -> torch.Tensor:
        lowpass = F.avg_pool2d(value, kernel_size=3, stride=1, padding=1)
        highpass = value - lowpass
        hidden = self.input(torch.cat((lowpass, highpass), dim=1))
        qp_index = max(0, min(int(qp), self.qp_embedding.num_embeddings - 1))
        condition = self.qp_embedding.weight[qp_index].to(device=value.device, dtype=value.dtype)
        hidden = hidden + condition.view(1, -1, 1, 1)
        for block in self.blocks:
            hidden = block(hidden)
        return value + self.residual_scale * self.output(hidden)


class IHierarchicalAnalysisSynthesis(nn.Module):
    """Independent learned I analysis/synthesis maps with progressive layers.

    Unlike an invertible 1x1 transform, the synthesis map need not algebraically
    invert analysis.  This is deliberate: quantized base/detail/texture symbols
    should be mapped to the reconstruction that minimizes the end-to-end RD
    objective, and the jointly trained hyperprior must model that new symbol
    coordinate directly.
    """

    def __init__(
        self,
        *,
        channels: int = 256,
        hidden_channels: int = 256,
        blocks: int = 3,
        num_qp: int = 64,
        residual_scale: float = 0.25,
        group_sizes: Sequence[int] | None = None,
    ) -> None:
        super().__init__()
        self.channels = int(channels)
        self.hidden_channels = int(hidden_channels)
        self.blocks_count = int(blocks)
        self.num_qp = int(num_qp)
        self.residual_scale = float(residual_scale)
        self.group_sizes = normalize_progressive_group_sizes(
            DEFAULT_PROGRESSIVE_GROUP_SIZES if group_sizes is None else group_sizes,
            channels=self.channels,
        )
        if self.hidden_channels < 32:
            raise ValueError("hierarchical hidden_channels must be at least 32")
        if self.blocks_count < 1:
            raise ValueError("hierarchical blocks must be positive")
        if self.num_qp < 1:
            raise ValueError("hierarchical num_qp must be positive")
        if not 0.0 < self.residual_scale <= 1.0:
            raise ValueError("hierarchical residual_scale must be in (0, 1]")

        common = dict(
            channels=self.channels,
            hidden_channels=self.hidden_channels,
            blocks=self.blocks_count,
            num_qp=self.num_qp,
            residual_scale=self.residual_scale,
        )
        self.analysis_path = _HierarchyPath(**common)
        self.synthesis_path = _HierarchyPath(**common)

    def analysis(self, value: torch.Tensor, qp: int) -> torch.Tensor:
        self._validate(value)
        return self.analysis_path(value, qp)

    def synthesis(self, value: torch.Tensor, qp: int) -> torch.Tensor:
        self._validate(value)
        return self.synthesis_path(value, qp)

    def _validate(self, value: torch.Tensor) -> None:
        if value.ndim != 4 or int(value.shape[1]) != self.channels:
            raise ValueError(
                f"hierarchical representation expects [B,{self.channels},H,W], " f"got {tuple(value.shape)}"
            )

    def progressive_auxiliary(
        self,
        source: torch.Tensor,
        representation: torch.Tensor,
        qp: int,
    ) -> dict[str, torch.Tensor]:
        """Supervise semantic base/detail ownership using prefix reconstructions.

        The base prefix must reproduce low frequency structure.  The base plus
        critical-detail prefix must reproduce spatial high frequency.  The
        optional texture suffix remains available to the ordinary full RD loss.
        Losses are normalized by target energy so their weights transfer across
        QPs and batches.
        """

        self._validate(source)
        self._validate(representation)
        base_stop = int(self.group_sizes[0])
        critical_stop = base_stop + int(self.group_sizes[1])

        base_symbols = torch.zeros_like(representation)
        base_symbols[:, :base_stop] = representation[:, :base_stop]
        critical_symbols = torch.zeros_like(representation)
        critical_symbols[:, :critical_stop] = representation[:, :critical_stop]
        base_reconstruction = self.synthesis(base_symbols, qp)
        critical_reconstruction = self.synthesis(critical_symbols, qp)

        source_work = source.float()
        source_low = F.avg_pool2d(source_work, 3, stride=1, padding=1)
        base_low = F.avg_pool2d(base_reconstruction.float(), 3, stride=1, padding=1)
        low_energy = source_low.square().mean().clamp_min(1.0e-6)
        compact_loss = F.mse_loss(base_low, source_low) / low_energy

        source_high = source_work - source_low
        critical_work = critical_reconstruction.float()
        critical_high = critical_work - F.avg_pool2d(critical_work, 3, stride=1, padding=1)
        high_energy = source_high.square().mean().clamp_min(1.0e-6)
        critical_highpass_loss = F.mse_loss(critical_high, source_high) / high_energy

        energy = representation.float().square().mean(dim=(0, 2, 3))
        total_energy = energy.sum().clamp_min(1.0e-12)
        low_channels = F.avg_pool2d(representation.float(), 3, stride=1, padding=1).square().mean(dim=(0, 2, 3))
        high_channels = (
            (representation.float() - F.avg_pool2d(representation.float(), 3, stride=1, padding=1))
            .square()
            .mean(dim=(0, 2, 3))
        )
        return {
            "compact_loss": compact_loss,
            "critical_highpass_loss": critical_highpass_loss,
            "base_energy_fraction": energy[:base_stop].sum() / total_energy,
            "base_lowpass_fraction": low_channels[:base_stop].sum() / low_channels.sum().clamp_min(1.0e-12),
            "critical_highpass_fraction": high_channels[base_stop:critical_stop].sum()
            / high_channels.sum().clamp_min(1.0e-12),
            "optional_energy_fraction": energy[critical_stop:].sum() / total_energy,
        }

    def contract_payload(self) -> dict[str, Any]:
        core = {
            "version": HIERARCHICAL_REPRESENTATION_VERSION,
            "kind": "learned_residual_low_high_analysis_synthesis_v1",
            "channels": self.channels,
            "hidden_channels": self.hidden_channels,
            "blocks": self.blocks_count,
            "num_qp": self.num_qp,
            "residual_scale": self.residual_scale,
            "group_sizes": list(self.group_sizes),
        }
        return {**core, "fingerprint": _fingerprint(core)}


def validate_hierarchical_contract(
    observed: dict[str, Any] | None,
    expected: dict[str, Any] | None,
) -> None:
    if observed is None and expected is None:
        return
    if observed is None or expected is None or observed != expected:
        raise ValueError("I hierarchical representation contract mismatch: " f"stream={observed} decoder={expected}")
