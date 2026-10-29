"""Decoder-visible, learnable regional quantization widths for Task938.

The predictor only consumes tensors available at the decoder.  Its output is
therefore safe to use as part of a deterministic codec contract: no source
image, clean latent, or encoder-only side input is accepted by ``forward``.
"""

from __future__ import annotations

import math
from typing import Union

import torch
from torch import nn

CanonicalQP = Union[int, torch.Tensor]


def quantize_with_width_ste(residual: torch.Tensor, width: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Quantize ``residual`` with a differentiable width-aware STE.

    Forward values are ``round(residual / width) * width``.  The backward
    derivative with respect to ``residual`` is one, while the derivative with
    respect to ``width`` is the LSQ term ``round(residual / width) -
    residual / width``.  In particular, width gradients are not accidentally
    cancelled by the usual ``q * width`` construction.
    """
    if residual.shape != width.shape and not (width.ndim == 0 or width.shape == residual.shape):
        try:
            torch.broadcast_shapes(residual.shape, width.shape)
        except RuntimeError as exc:
            raise ValueError("residual and width must be broadcastable") from exc
    if torch.any(width <= 0):
        raise ValueError("width must be positive")
    normalized = residual / width
    symbols = torch.round(normalized)
    # Multiplying the detached residual correction by width preserves the
    # desired LSQ width derivative while retaining an identity residual STE.
    quantized = residual + (symbols - normalized).detach() * width
    return quantized, symbols.to(torch.int32)


class RegionalQuantizer(nn.Module):
    """Predict one bounded quantization width per channel group and pixel.

    ``scales`` and ``means`` are decoder-known ``[B,C,H,W]`` tensors.  A group
    shares one width across its channels; a final partial group is supported.
    The zero-initialized final projection makes construction exactly
    ``base_width`` everywhere, while subsequent optimization can specialize
    widths by region and group.
    """

    def __init__(
        self,
        group_size: int = 16,
        hidden_dim: int = 32,
        base_width: float = 1.0,
        max_log: float = math.log(2.0),
        use_group_embedding: bool = False,
        max_groups: int = 256,
        embedding_dim: int = 8,
        eps: float = 1e-6,
    ) -> None:
        super().__init__()
        if group_size <= 0 or hidden_dim <= 0:
            raise ValueError("group_size and hidden_dim must be positive")
        if not math.isfinite(base_width) or base_width <= 0:
            raise ValueError("base_width must be finite and positive")
        if not math.isfinite(max_log) or max_log < 0:
            raise ValueError("max_log must be finite and nonnegative")
        if eps <= 0:
            raise ValueError("eps must be positive")
        self.group_size = int(group_size)
        self.base_width = float(base_width)
        self.max_log = float(max_log)
        self.eps = float(eps)
        self.use_group_embedding = bool(use_group_embedding)
        self.max_groups = int(max_groups)
        self.embedding_dim = int(embedding_dim)
        in_dim = 3
        if self.use_group_embedding:
            if self.max_groups <= 0 or self.embedding_dim <= 0:
                raise ValueError("max_groups and embedding_dim must be positive")
            self.group_embedding = nn.Embedding(self.max_groups, self.embedding_dim)
            in_dim += self.embedding_dim
        else:
            self.group_embedding = None
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 1),
        )
        nn.init.zeros_(self.mlp[-1].weight)
        nn.init.zeros_(self.mlp[-1].bias)

    @staticmethod
    def _qp_tensor(qp: CanonicalQP, batch: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        q = torch.as_tensor(qp, device=device)
        if q.ndim == 0:
            q = q.expand(batch)
        elif q.ndim == 1 and q.numel() == batch:
            pass
        else:
            raise ValueError("qp must be a scalar or a [B] tensor")
        if not torch.all((q == 0) | (q == 16) | (q == 39) | (q == 63)):
            raise ValueError("canonical QP must be one of 0, 16, 39, or 63")
        return q.to(dtype=dtype)

    def forward(
        self,
        scales: torch.Tensor,
        means: torch.Tensor,
        qp: CanonicalQP,
        base_width: float | None = None,
    ) -> torch.Tensor:
        if scales.ndim != 4 or means.shape != scales.shape:
            raise ValueError("scales and means must have matching [B,C,H,W] shapes")
        if not torch.isfinite(scales).all() or not torch.isfinite(means).all():
            raise ValueError("scales and means must be finite")
        b, channels, height, width = scales.shape
        if channels == 0:
            raise ValueError("channel dimension must be nonzero")
        base = self.base_width if base_width is None else float(base_width)
        if not math.isfinite(base) or base <= 0:
            raise ValueError("base_width must be finite and positive")
        q = self._qp_tensor(qp, b, scales.device, scales.dtype)
        groups = (channels + self.group_size - 1) // self.group_size
        pad = groups * self.group_size - channels
        scale_feature = torch.log(scales.abs().clamp_min(self.eps))
        energy_feature = means.square()
        if pad:
            scale_feature = torch.nn.functional.pad(scale_feature, (0, 0, 0, 0, 0, pad))
            energy_feature = torch.nn.functional.pad(energy_feature, (0, 0, 0, 0, 0, pad))
            valid = torch.nn.functional.pad(
                torch.ones((b, channels, height, width), device=scales.device, dtype=scales.dtype),
                (0, 0, 0, 0, 0, pad),
            )
        else:
            valid = torch.ones_like(scale_feature)
        shape = (b, groups, self.group_size, height, width)
        scale_feature = (scale_feature.reshape(shape) * valid.reshape(shape)).sum(2) / valid.reshape(shape).sum(
            2
        ).clamp_min(1)
        energy_feature = (energy_feature.reshape(shape) * valid.reshape(shape)).sum(2) / valid.reshape(shape).sum(
            2
        ).clamp_min(1)
        qp_feature = (q / 63.0).view(b, 1, 1, 1).expand(b, groups, height, width)
        features = torch.stack((scale_feature, energy_feature, qp_feature), dim=-1)
        if self.group_embedding is not None:
            if groups > self.max_groups:
                raise ValueError(f"{groups} groups exceed max_groups={self.max_groups}")
            ids = torch.arange(groups, device=scales.device)
            emb = self.group_embedding(ids).view(1, groups, 1, 1, self.embedding_dim).expand(b, -1, height, width, -1)
            features = torch.cat((features, emb), dim=-1)
        feature_map = features
        log_multiplier = torch.tanh(self.mlp(feature_map)).squeeze(-1)
        group_width = base * torch.exp(self.max_log * log_multiplier)
        return (
            group_width.unsqueeze(2)
            .expand(b, groups, self.group_size, height, width)
            .reshape(b, groups * self.group_size, height, width)[:, :channels]
        )


__all__ = ["RegionalQuantizer", "quantize_with_width_ste"]
