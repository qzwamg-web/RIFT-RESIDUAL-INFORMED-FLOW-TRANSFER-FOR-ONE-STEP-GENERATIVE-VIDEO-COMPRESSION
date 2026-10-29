"""Fixed-symbol P-transform capacity used by Task713.

The module is deliberately P-specific.  It changes neither the 128-channel
``y``/``y_hat`` ABI nor the bitstream topology, and it consumes only the QP
that is already known to both encoder and decoder.  A zero-initialized output
projection makes an enabled module an exact identity before its first update.
"""

from __future__ import annotations

import hashlib
import json
from typing import Any

import torch
from torch import nn

P_TRANSFORM_CAPACITY_CONTRACT_VERSION = 1


class _PTransformResidualBlock(nn.Module):
    """Depthwise spatial residual block with decoder-known QP FiLM."""

    def __init__(self, channels: int, num_qp: int) -> None:
        super().__init__()
        self.norm = nn.GroupNorm(1, channels)
        self.depthwise = nn.Conv2d(channels, channels, kernel_size=3, padding=1, groups=channels)
        self.expand = nn.Conv2d(channels, 2 * channels, kernel_size=1)
        self.project = nn.Conv2d(channels, channels, kernel_size=1)
        self.qp_film = nn.Embedding(num_qp, 2 * channels)
        nn.init.zeros_(self.qp_film.weight)

    def forward(self, value: torch.Tensor, qp: int) -> torch.Tensor:
        qp_index = max(0, min(int(qp), self.qp_film.num_embeddings - 1))
        indices = torch.full((value.shape[0],), qp_index, device=value.device, dtype=torch.long)
        gain, bias = self.qp_film(indices).to(value.dtype).chunk(2, dim=1)
        hidden = self.norm(value)
        hidden = hidden * (1.0 + gain[:, :, None, None])
        hidden = hidden + bias[:, :, None, None]
        hidden = self.depthwise(hidden)
        gate, content = self.expand(hidden).chunk(2, dim=1)
        hidden = torch.nn.functional.silu(gate) * content
        return value + self.project(hidden)


class PTransformCapacityCore(nn.Module):
    """Same-shape QP-conditioned capacity around the coded P ``y`` tensor."""

    def __init__(
        self,
        *,
        channels: int,
        hidden_channels: int,
        blocks: int,
        residual_scale: float,
        role: str,
        num_qp: int = 72,
    ) -> None:
        super().__init__()
        self.channels = int(channels)
        self.hidden_channels = int(hidden_channels)
        self.blocks = int(blocks)
        self.residual_scale = float(residual_scale)
        self.role = str(role).strip().lower()
        self.num_qp = int(num_qp)
        if self.channels <= 0 or self.hidden_channels <= 0:
            raise ValueError("P transform capacity channels must be positive")
        if self.blocks <= 0:
            raise ValueError("p_core_blocks must be positive when a core is enabled")
        if not 0.0 < self.residual_scale <= 1.0:
            raise ValueError("p_core_residual_scale must be in (0, 1]")
        if self.role not in {"analysis", "synthesis"}:
            raise ValueError("P transform capacity role must be analysis or synthesis")
        if self.num_qp <= 0:
            raise ValueError("P transform capacity num_qp must be positive")

        self.input_proj = nn.Conv2d(self.channels, self.hidden_channels, 1)
        self.body = nn.ModuleList(
            _PTransformResidualBlock(self.hidden_channels, self.num_qp) for _ in range(self.blocks)
        )
        self.output_proj = nn.Conv2d(self.hidden_channels, self.channels, 1)
        nn.init.zeros_(self.output_proj.weight)
        nn.init.zeros_(self.output_proj.bias)

    def forward(self, value: torch.Tensor, qp: int) -> torch.Tensor:
        if value.ndim != 4 or int(value.shape[1]) != self.channels:
            raise ValueError(
                "P transform capacity expects [B,C,H,W] with C=" f"{self.channels}, got {tuple(value.shape)}"
            )
        hidden = self.input_proj(value)
        for block in self.body:
            hidden = block(hidden, qp)
        return value + self.residual_scale * self.output_proj(hidden)

    def contract_payload(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "version": P_TRANSFORM_CAPACITY_CONTRACT_VERSION,
            "role": self.role,
            "channels": self.channels,
            "hidden_channels": self.hidden_channels,
            "blocks": self.blocks,
            "residual_scale": self.residual_scale,
            "num_qp": self.num_qp,
            "condition_inputs": ["decoder_known_qp"],
            "additional_streams": 0,
        }
        encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
        payload["fingerprint"] = hashlib.sha256(encoded).hexdigest()
        return payload


def p_transform_capacity_contract(
    analysis: PTransformCapacityCore | None,
    synthesis: PTransformCapacityCore | None,
) -> dict[str, Any] | None:
    """Return ``None`` for the old/default codec ABI, else a full contract."""

    if analysis is None and synthesis is None:
        return None
    return {
        "schema": "task713_p_transform_capacity_v1",
        "analysis": None if analysis is None else analysis.contract_payload(),
        "synthesis": None if synthesis is None else synthesis.contract_payload(),
        "y_channels": 128,
        "z_channels": 128,
        "symbol_topology": "unchanged",
        "rans_order": "unchanged",
    }


__all__ = [
    "P_TRANSFORM_CAPACITY_CONTRACT_VERSION",
    "PTransformCapacityCore",
    "p_transform_capacity_contract",
]
