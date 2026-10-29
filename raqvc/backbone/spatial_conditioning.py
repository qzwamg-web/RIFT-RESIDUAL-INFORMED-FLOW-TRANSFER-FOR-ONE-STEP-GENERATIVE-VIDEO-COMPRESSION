"""C512 deterministic spatial conditioning from decoded latent evidence.
This module adds a learned conditioning residual; it does not inject random noise.
"""

from __future__ import annotations
from dataclasses import dataclass
import torch
from torch import Tensor, nn
import torch.nn.functional as F


@dataclass(frozen=True)
class Task614SpatialNoiseConfig:
    """Fixed spatial evidence map; no stochastic noise is sampled."""

    minimum_scale: float = 0.65
    maximum_scale: float = 1.35
    detail_weight: float = 0.35
    temporal_weight: float = 0.65
    smoothing_kernel: int = 3


def _standardized_energy(value: Tensor) -> Tensor:
    """Return a stable per-sample standardized log-energy field."""
    if value.ndim != 5 or int(value.shape[1]) != 1:
        raise ValueError("Task614 energy must be [B,1,T,H,W]")
    dimensions = (2, 3, 4)
    mean = value.mean(dim=dimensions, keepdim=True).clamp_min(1e-06)
    logged = torch.log1p(value / mean)
    centered = logged - logged.mean(dim=dimensions, keepdim=True)
    scale = centered.square().mean(dim=dimensions, keepdim=True).add(1e-06).sqrt()
    return centered / scale


def decoded_spatial_noise_scale_map(source: Tensor, config: Task614SpatialNoiseConfig) -> Tensor:
    """Build ``[B,1,T,H,W]`` noise scales from a decoded latent.

    Spatial high-pass energy identifies texture/detail demand.  Temporal
    innovation identifies locations whose decoded evidence is unstable across
    neighboring frames.  The second term receives the larger default weight,
    avoiding the unsafe rule "all strong edges should receive maximum noise".

    The bounded result has per-sample mean one.  Consequently Task614 changes
    *where* the existing anchored-innovation budget is spent without learning
    a hidden global rate/QP multiplier.
    """
    if source.ndim != 5:
        raise ValueError("Task614 decoded source must be [B,C,T,H,W]")
    if not bool(torch.isfinite(source).all()):
        raise FloatingPointError("Task614 decoded source contains NaN or Inf")
    value = source.float()
    batch, _channels, temporal, height, width = value.shape
    flattened = value.transpose(1, 2).reshape(batch * temporal, -1, height, width)
    local_mean = F.avg_pool2d(flattened, kernel_size=3, stride=1, padding=1)
    detail = (flattened - local_mean).square().mean(dim=1, keepdim=True).add(1e-08).sqrt()
    detail = detail.reshape(batch, temporal, 1, height, width).transpose(1, 2)
    if temporal > 1:
        difference = (value[:, :, 1:] - value[:, :, :-1]).square().mean(dim=1, keepdim=True).add(1e-08).sqrt()
        temporal_energy = torch.cat((difference[:, :, :1], difference), dim=2)
    else:
        temporal_energy = torch.zeros((batch, 1, 1, height, width), device=value.device, dtype=value.dtype)
    detail_score = _standardized_energy(detail)
    temporal_score = _standardized_energy(temporal_energy)
    weight_sum = float(config.detail_weight) + float(config.temporal_weight)
    score = (float(config.detail_weight) * detail_score + float(config.temporal_weight) * temporal_score) / weight_sum
    kernel = int(config.smoothing_kernel)
    if kernel > 1:
        pad = kernel // 2
        score_2d = score.transpose(1, 2).reshape(batch * temporal, 1, height, width)
        score = (
            F.avg_pool2d(score_2d, kernel_size=kernel, stride=1, padding=pad)
            .reshape(batch, temporal, 1, height, width)
            .transpose(1, 2)
        )
    allocation = torch.tanh(score)
    allocation = allocation - allocation.mean(dim=(2, 3, 4), keepdim=True)
    positive_span = float(config.maximum_scale) - 1.0
    negative_span = 1.0 - float(config.minimum_scale)
    positive = allocation.clamp_min(0.0) * positive_span
    negative = allocation.clamp_max(0.0) * negative_span
    scale_map = (1.0 + positive + negative).clamp(float(config.minimum_scale), float(config.maximum_scale))
    scale_map = scale_map - scale_map.mean(dim=(2, 3, 4), keepdim=True) + 1.0
    scale_map = scale_map.clamp(float(config.minimum_scale), float(config.maximum_scale))
    if not bool(torch.isfinite(scale_map).all()):
        raise FloatingPointError("Task614 spatial noise map contains NaN or Inf")
    return scale_map.to(dtype=source.dtype)


class Task614SpatialNoiseConditioner(nn.Module):
    """Explicitly inject the spatial scale map into the DiT latent grid.

    The last convolution is zero-initialized.  Loading a pre-Task614 teacher
    therefore preserves its original velocity function at initialization,
    while the map gets a dedicated, spatially aligned trainable path.
    """

    def __init__(self, latent_channels: int, hidden_channels: int = 32) -> None:
        super().__init__()
        if int(latent_channels) < 1 or int(hidden_channels) < 1:
            raise ValueError("Task614 conditioner channels must be positive")
        self.net = nn.Sequential(
            nn.Conv3d(1, int(hidden_channels), kernel_size=3, padding=1),
            nn.SiLU(),
            nn.Conv3d(int(hidden_channels), int(latent_channels), kernel_size=3, padding=1),
        )
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, state: Tensor, scale_map: Tensor) -> Tensor:
        if state.ndim != 5 or scale_map.ndim != 5:
            raise ValueError("Task614 state/map must be five-dimensional")
        expected = (state.shape[0], 1, state.shape[2], state.shape[3], state.shape[4])
        if tuple(scale_map.shape) != tuple(expected):
            raise ValueError(f"Task614 spatial map shape {tuple(scale_map.shape)} != {tuple(expected)}")
        parameter = next(self.net.parameters())
        condition = (scale_map.float() - 1.0).to(device=parameter.device, dtype=parameter.dtype)
        correction = self.net(condition)
        return state + correction.to(device=state.device, dtype=state.dtype)
