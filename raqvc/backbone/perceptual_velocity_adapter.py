"""Lightweight perceptual correction around a frozen generative velocity field.

The adapter is deliberately decoder-only: every input is available after
entropy decoding and a normal DiT forward.  It therefore changes reconstruction
quality without changing the bitstream.  The last convolution is zero
initialized, making an enabled, untrained adapter an exact identity.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def _group_count(channels: int) -> int:
    for groups in (16, 8, 4, 2, 1):
        if channels % groups == 0:
            return groups
    return 1


class _ConditionedResidualBlock(nn.Module):
    """Factorized space/time residual block with QP/time FiLM."""

    def __init__(self, channels: int, condition_dim: int):
        super().__init__()
        groups = _group_count(channels)
        self.norm1 = nn.GroupNorm(groups, channels)
        self.spatial = nn.Conv3d(
            channels,
            channels,
            kernel_size=(1, 3, 3),
            padding=(0, 1, 1),
        )
        self.norm2 = nn.GroupNorm(groups, channels)
        self.temporal = nn.Conv3d(
            channels,
            channels,
            kernel_size=(3, 1, 1),
            padding=(1, 0, 0),
        )
        self.film = nn.Linear(condition_dim, 2 * channels)

    def forward(self, x: torch.Tensor, condition: torch.Tensor) -> torch.Tensor:
        scale, bias = self.film(condition).chunk(2, dim=-1)
        scale = scale[:, :, None, None, None]
        bias = bias[:, :, None, None, None]
        h = self.spatial(F.silu(self.norm1(x)))
        h = self.norm2(h)
        h = h * (1.0 + 0.1 * scale) + 0.1 * bias
        h = self.temporal(F.silu(h))
        return x + h / math.sqrt(2.0)


class PerceptualVelocityAdapter(nn.Module):
    """Correct a frozen DiT velocity using codec-artifact evidence.

    Inputs are the current restoration state, the frozen base velocity, a
    causal temporal residual, and a spatial high-pass residual.  QP and solver
    progress condition every residual block.  Lower-rate operating points get
    a larger fixed residual budget, while the zero output projection preserves
    the parent checkpoint exactly at initialization.
    """

    def __init__(
        self,
        latent_channels: int = 48,
        hidden_channels: int = 96,
        num_blocks: int = 6,
        num_qp_points: int = 4,
        residual_scale: float = 0.25,
        qp_strengths: tuple[float, ...] = (0.25, 0.5, 0.85, 1.0),
    ):
        super().__init__()
        if latent_channels <= 0 or hidden_channels <= 0 or num_blocks <= 0:
            raise ValueError("adapter channel/block counts must be positive")
        if num_qp_points <= 0:
            raise ValueError("num_qp_points must be positive")
        if len(qp_strengths) != num_qp_points:
            raise ValueError(f"qp_strengths has {len(qp_strengths)} values, expected {num_qp_points}")

        self.latent_channels = int(latent_channels)
        self.num_qp_points = int(num_qp_points)
        self.residual_scale = float(residual_scale)
        self._last_correction_rms: float | None = None
        self._last_relative_rms: float | None = None
        self._last_direction_cosine: float | None = None
        condition_dim = 64

        self.qp_embedding = nn.Embedding(num_qp_points, 32)
        self.time_mlp = nn.Sequential(
            nn.Linear(6, 32),
            nn.SiLU(),
            nn.Linear(32, 32),
        )
        self.input_projection = nn.Conv3d(
            4 * latent_channels,
            hidden_channels,
            kernel_size=3,
            padding=1,
        )
        self.blocks = nn.ModuleList(
            _ConditionedResidualBlock(hidden_channels, condition_dim) for _ in range(num_blocks)
        )
        self.output_norm = nn.GroupNorm(_group_count(hidden_channels), hidden_channels)
        self.output_projection = nn.Conv3d(
            hidden_channels,
            latent_channels,
            kernel_size=3,
            padding=1,
        )
        nn.init.zeros_(self.output_projection.weight)
        nn.init.zeros_(self.output_projection.bias)
        self.register_buffer(
            "qp_strengths",
            torch.tensor(qp_strengths, dtype=torch.float32),
            persistent=True,
        )

    @staticmethod
    def _time_features(progress: torch.Tensor) -> torch.Tensor:
        progress = progress.float().reshape(-1).clamp(0.0, 1.0)
        return torch.stack(
            (
                progress,
                progress.square(),
                torch.sin(math.pi * progress),
                torch.cos(math.pi * progress),
                torch.sin(2.0 * math.pi * progress),
                torch.cos(2.0 * math.pi * progress),
            ),
            dim=-1,
        )

    def forward(
        self,
        z_state: torch.Tensor,
        base_velocity: torch.Tensor,
        *,
        progress: torch.Tensor,
        schedule_idx: int | torch.Tensor,
    ) -> torch.Tensor:
        if z_state.shape != base_velocity.shape or z_state.ndim != 5:
            raise ValueError(
                "adapter expects matching [B,C,T,H,W] state/velocity tensors; "
                f"got {tuple(z_state.shape)} and {tuple(base_velocity.shape)}"
            )
        if z_state.size(1) != self.latent_channels:
            raise ValueError(f"adapter expected {self.latent_channels} latent channels, " f"got {z_state.size(1)}")

        batch = z_state.size(0)
        if torch.is_tensor(schedule_idx):
            qp_index = schedule_idx.to(device=z_state.device, dtype=torch.long).reshape(-1)
            if qp_index.numel() == 1 and batch > 1:
                qp_index = qp_index.expand(batch)
        else:
            qp_index = torch.full((batch,), int(schedule_idx), device=z_state.device, dtype=torch.long)
        if qp_index.numel() != batch:
            raise ValueError(
                f"schedule_idx must contain one value per batch item, got {qp_index.numel()} for B={batch}"
            )
        if bool(((qp_index < 0) | (qp_index >= self.num_qp_points)).any()):
            raise ValueError(f"schedule_idx outside [0, {self.num_qp_points - 1}]")

        progress = progress.to(device=z_state.device).reshape(-1)
        if progress.numel() == 1 and batch > 1:
            progress = progress.expand(batch)
        if progress.numel() != batch:
            raise ValueError(f"progress must contain one value per batch item, got {progress.numel()} for B={batch}")

        previous = torch.zeros_like(z_state)
        if z_state.size(2) > 1:
            previous[:, :, 1:] = z_state[:, :, :-1]
        temporal_residual = z_state - previous
        smooth = F.avg_pool3d(
            z_state.float(),
            kernel_size=(1, 3, 3),
            stride=1,
            padding=(0, 1, 1),
        ).to(z_state.dtype)
        spatial_highpass = z_state - smooth

        adapter_dtype = self.input_projection.weight.dtype
        features = torch.cat((z_state, base_velocity, temporal_residual, spatial_highpass), dim=1).to(
            dtype=adapter_dtype
        )
        condition = torch.cat(
            (
                self.qp_embedding(qp_index),
                self.time_mlp(self._time_features(progress).to(adapter_dtype)),
            ),
            dim=-1,
        )
        h = self.input_projection(features)
        for block in self.blocks:
            h = block(h, condition)
        delta = self.output_projection(F.silu(self.output_norm(h)))
        strength = self.qp_strengths[qp_index].to(device=delta.device, dtype=delta.dtype)[:, None, None, None, None]
        corrected = base_velocity.to(delta.dtype) + self.residual_scale * strength * delta
        return corrected.to(base_velocity.dtype)


class Task438IndependentQPSpeedResidual(nn.Module):
    """Four genuinely independent QP velocity-direction residual heads.

    This is deliberately separate from :class:`PerceptualVelocityAdapter`.
    The latter has one shared trunk and only injects a QP embedding/strength;
    it is therefore QP-conditioned, not QP-independent.  Each head below has
    its own convolutional parameters and receives no shared trainable tensor.
    The last projection is zero initialized, so adding the module to a
    historical Task840 checkpoint is exactly identity until its own optimizer
    group is trained.
    """

    def __init__(
        self,
        latent_channels: int = 48,
        qp_count: int = 4,
        hidden_channels: int = 64,
        residual_scale: float = 0.25,
    ) -> None:
        super().__init__()
        if min(int(latent_channels), int(qp_count), int(hidden_channels)) < 1:
            raise ValueError("independent QP residual dimensions must be positive")
        if float(residual_scale) <= 0.0:
            raise ValueError("independent QP residual scale must be positive")
        self.latent_channels = int(latent_channels)
        self.qp_count = int(qp_count)
        self.hidden_channels = int(hidden_channels)
        self.residual_scale = float(residual_scale)
        # State, base velocity, temporal difference, spatial high-pass, and
        # two scalar maps (progress and solver horizon) are all head inputs.
        input_channels = 4 * self.latent_channels + 2
        heads: list[nn.Module] = []
        for _ in range(self.qp_count):
            head = nn.Sequential(
                nn.Conv3d(input_channels, self.hidden_channels, 3, padding=1),
                nn.SiLU(),
                nn.Conv3d(self.hidden_channels, self.hidden_channels, 3, padding=1),
                nn.SiLU(),
                nn.Conv3d(self.hidden_channels, self.latent_channels, 3, padding=1),
            )
            nn.init.zeros_(head[-1].weight)
            nn.init.zeros_(head[-1].bias)
            heads.append(head)
        self.heads = nn.ModuleList(heads)

    def contract(self) -> dict[str, object]:
        return {
            "enabled": True,
            "kind": "independent_qp_velocity_direction_residual_v1",
            "qp_count": self.qp_count,
            "latent_channels": self.latent_channels,
            "hidden_channels": self.hidden_channels,
            "residual_scale": self.residual_scale,
            "zero_init": True,
            "shared_trainable_parameters": False,
            "inputs": [
                "state",
                "base_velocity",
                "temporal_residual",
                "spatial_highpass",
                "restoration_progress",
                "solver_horizon",
            ],
        }

    def forward(
        self,
        z_state: torch.Tensor,
        base_velocity: torch.Tensor,
        *,
        progress: torch.Tensor,
        qp_index: int,
        solver_steps: int | None = None,
    ) -> torch.Tensor:
        if z_state.shape != base_velocity.shape or z_state.ndim != 5:
            raise ValueError("independent QP residual expects matching [B,C,T,H,W] tensors")
        if int(z_state.size(1)) != self.latent_channels:
            raise ValueError(
                f"independent QP residual expected {self.latent_channels} channels, " f"got {z_state.size(1)}"
            )
        index = int(qp_index)
        if index < 0 or index >= self.qp_count:
            raise ValueError(f"QP index {index} is outside [0,{self.qp_count})")
        progress = progress.to(device=z_state.device, dtype=z_state.dtype).reshape(-1)
        batch = int(z_state.size(0))
        if progress.numel() == 1 and batch > 1:
            progress = progress.expand(batch)
        if progress.numel() != batch:
            raise ValueError(f"progress must contain one value per batch item, got {progress.numel()}")
        previous = torch.zeros_like(z_state)
        if z_state.size(2) > 1:
            previous[:, :, 1:] = z_state[:, :, :-1]
        temporal_residual = z_state - previous
        smooth = F.avg_pool3d(z_state.float(), kernel_size=(1, 3, 3), stride=1, padding=(0, 1, 1)).to(z_state.dtype)
        spatial_highpass = z_state - smooth
        horizon_value = 0.0 if solver_steps is None else math.log2(max(int(solver_steps), 1)) / 2.0
        scalar_shape = (batch, 1, z_state.size(2), z_state.size(3), z_state.size(4))
        progress_map = progress[:, None, None, None, None].expand(scalar_shape)
        horizon_map = torch.full(
            scalar_shape,
            float(horizon_value),
            device=z_state.device,
            dtype=z_state.dtype,
        )
        features = torch.cat(
            (z_state, base_velocity, temporal_residual, spatial_highpass, progress_map, horizon_map),
            dim=1,
        )
        head = self.heads[index]
        delta = torch.tanh(head(features.to(dtype=next(head.parameters()).dtype)))
        # Bound the new direction relative to the current velocity magnitude;
        # this makes the high short-train LR safe while still allowing a real
        # direction change rather than only a scalar speed adjustment.
        base_rms = (
            base_velocity.detach()
            .float()
            .square()
            .mean(dim=(1, 2, 3, 4), keepdim=True)
            .sqrt()
            .clamp_min(1.0e-4)
            .to(delta.dtype)
        )
        correction = self.residual_scale * base_rms * delta
        corrected = base_velocity.to(delta.dtype) + correction
        with torch.no_grad():
            correction_flat = correction.float().flatten(1)
            base_flat = base_velocity.float().flatten(1)
            correction_rms = correction_flat.square().mean().sqrt()
            base_rms_observed = base_flat.square().mean().sqrt().clamp_min(1.0e-8)
            self._last_correction_rms = float(correction_rms.item())
            self._last_relative_rms = float((correction_rms / base_rms_observed).item())
            self._last_direction_cosine = float(F.cosine_similarity(correction_flat, base_flat, dim=1).mean().item())
        return corrected.to(base_velocity.dtype)
