"""Decoder-closed causal recurrent synthesis for sequential latent P coding."""

from __future__ import annotations

from typing import Optional

import torch
from torch import nn


class _ResidualStateBlock(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        groups = min(16, int(channels))
        while int(channels) % groups != 0:
            groups -= 1
        self.norm1 = nn.GroupNorm(groups, int(channels))
        self.conv1 = nn.Conv2d(int(channels), int(channels), 3, padding=1)
        self.norm2 = nn.GroupNorm(groups, int(channels))
        self.conv2 = nn.Conv2d(int(channels), int(channels), 3, padding=1)
        self.activation = nn.SiLU(inplace=False)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        residual = self.conv1(self.activation(self.norm1(value)))
        residual = self.conv2(self.activation(self.norm2(residual)))
        return value + residual


class PRecurrentSynthesisCore(nn.Module):
    """Update one compact decoded-feature state at every P position.

    The state is produced entirely from already decoded reconstruction
    features. It is available identically to encoder and decoder and carries
    no payload. The final projection is zero initialized for an exact parent
    codec at step zero.
    """

    def __init__(
        self,
        *,
        feature_channels: int = 320,
        state_channels: int = 96,
        blocks: int = 3,
        residual_scale: float = 0.20,
        max_position: int = 24,
        num_qp: int = 64,
    ) -> None:
        super().__init__()
        if min(feature_channels, state_channels, blocks, max_position, num_qp) <= 0:
            raise ValueError("recurrent synthesis dimensions must be positive")
        if float(residual_scale) <= 0:
            raise ValueError("recurrent synthesis residual scale must be positive")
        self.input = nn.Conv2d(feature_channels + 2, state_channels, 3, padding=1)
        self.gates = nn.Conv2d(2 * state_channels, 2 * state_channels, 3, padding=1)
        self.candidate = nn.Conv2d(2 * state_channels, state_channels, 3, padding=1)
        self.blocks = nn.ModuleList([_ResidualStateBlock(state_channels) for _ in range(blocks)])
        self.output = nn.Conv2d(state_channels, feature_channels, 3, padding=1)
        self.activation = nn.SiLU(inplace=False)
        self.feature_channels = int(feature_channels)
        self.state_channels = int(state_channels)
        self.block_count = int(blocks)
        self.residual_scale = float(residual_scale)
        self.max_position = int(max_position)
        self.num_qp = int(num_qp)
        nn.init.zeros_(self.output.weight)
        nn.init.zeros_(self.output.bias)

    def forward(
        self,
        feature: torch.Tensor,
        state: Optional[torch.Tensor],
        *,
        qp: int,
        position: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if feature.ndim != 4 or int(feature.shape[1]) != self.feature_channels:
            raise ValueError(f"expected feature [B,{self.feature_channels},H,W], got {tuple(feature.shape)}")
        batch, _, height, width = feature.shape
        qp_value = float(max(0, min(int(qp), self.num_qp - 1))) / float(max(1, self.num_qp - 1))
        position_value = float(max(1, min(int(position), self.max_position))) / float(self.max_position)
        condition = feature.new_empty((batch, 2, height, width))
        condition[:, 0].fill_(qp_value)
        condition[:, 1].fill_(position_value)
        current = self.activation(self.input(torch.cat((feature, condition), dim=1)))
        if state is None:
            state = current.new_zeros((batch, self.state_channels, height, width))
        if tuple(state.shape) != tuple(current.shape):
            raise ValueError(
                f"recurrent state shape {tuple(state.shape)} does not match current feature {tuple(current.shape)}"
            )
        reset, update = self.gates(torch.cat((current, state), dim=1)).chunk(2, dim=1)
        reset = torch.sigmoid(reset)
        update = torch.sigmoid(update)
        candidate = torch.tanh(self.candidate(torch.cat((current, reset * state), dim=1)))
        next_state = update * state + (1.0 - update) * candidate
        for block in self.blocks:
            next_state = block(next_state)
        refined = feature + self.residual_scale * self.output(next_state)
        return refined, next_state

    def contract_payload(self) -> dict[str, object]:
        return {
            "version": 1,
            "role": "in_loop_causal_recurrent_p_synthesis",
            "feature_channels": self.feature_channels,
            "state_channels": self.state_channels,
            "blocks": self.block_count,
            "residual_scale": self.residual_scale,
            "max_position": self.max_position,
            "payload_bits": 0,
            "updates_every_p": True,
        }
