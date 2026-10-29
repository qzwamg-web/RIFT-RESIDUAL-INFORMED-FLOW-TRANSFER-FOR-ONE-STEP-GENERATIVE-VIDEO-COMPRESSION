"""Decoder-known soft reliability for Task438/453 window attention.

The score deliberately uses no clean latent, RGB target, or encoder cache.  It
combines coded operating-point metadata with a per-position quantization
survivability proxy computed from the decoded source latent.  The resulting
frame score is used in two different ways:

* a soft, directional key bias inside the current restoration window;
* a stricter eligibility gate when a restored frame is written to the bounded
  cross-window memory.

This is intentionally not a fixed temporal hierarchy.  A low-reliability
query can still read a reliable key, while a reliable query only weakly reads
an unreliable key.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor, nn
import torch.nn.functional as F


@dataclass(frozen=True)
class SoftReliabilityConfig:
    floor: float = 0.08
    write_threshold: float = 0.52
    write_epsilon: float = 1.0e-4
    query_floor: float = 0.20
    max_log_bias: float = 6.0

    def __post_init__(self) -> None:
        if not 0.0 < self.floor < 1.0:
            raise ValueError("reliability floor must be in (0,1)")
        if not self.floor < self.write_threshold < 1.0:
            raise ValueError("write threshold must be above the reliability floor")
        if not 0.0 < self.write_epsilon < self.floor:
            raise ValueError("write epsilon must be positive and below the floor")
        if not 0.0 <= self.query_floor < 1.0:
            raise ValueError("query floor must be in [0,1)")
        if self.max_log_bias <= 0.0:
            raise ValueError("maximum reliability log bias must be positive")


class Task456SoftReliability(nn.Module):
    """Monotone metadata fusion plus decoded-latent position quality.

    Positive softplus weights preserve the intended monotonic directions:
    lower QP, higher actual bpp, lower residual RMS and stronger decoded
    quantization survivability all increase reliability.  Two small position
    tables allow reset-T7 and continuation-T6 roles to calibrate their local
    positions without forcing a global hard order.
    """

    def __init__(self, config: SoftReliabilityConfig | None = None) -> None:
        super().__init__()
        self.config = config or SoftReliabilityConfig()
        # softplus([-0.4, -1.0, -0.8, -0.5]) ~= [0.51, 0.31, 0.37, 0.47].
        self.raw_weights = nn.Parameter(torch.tensor([-0.4, -1.0, -0.8, -0.5], dtype=torch.float32))
        self.bias = nn.Parameter(torch.tensor([0.15], dtype=torch.float32))
        self.position_bias = nn.Parameter(torch.zeros(2, 7, dtype=torch.float32))
        self._last_snapshot: dict[str, Tensor] = {}

    @staticmethod
    def _batch_vector(value: Tensor, batch: int, name: str) -> Tensor:
        if value.ndim == 0:
            value = value.expand(batch)
        if value.ndim == 2 and value.shape[1] == 1:
            value = value[:, 0]
        if value.shape != (batch,):
            raise ValueError(f"{name} must be scalar/[B]/[B,1], got {tuple(value.shape)}")
        return value.float()

    @staticmethod
    def _position_quality(source: Tensor, log_q_y: Tensor) -> Tensor:
        """Return a decoder-known [B,T] quantization-survivability proxy."""

        amplitude = source.float().square().mean(dim=(1, 3, 4)).clamp_min(1e-10).sqrt()
        temporal = int(source.shape[2])
        if temporal == 1:
            innovation = torch.zeros_like(amplitude)
        else:
            delta = source[:, :, 1:].float() - source[:, :, :-1].float()
            delta = delta.square().mean(dim=(1, 3, 4)).clamp_min(1e-10).sqrt()
            innovation = torch.empty_like(amplitude)
            innovation[:, 0] = delta[:, 0]
            innovation[:, -1] = delta[:, -1]
            if temporal > 2:
                innovation[:, 1:-1] = 0.5 * (delta[:, :-1] + delta[:, 1:])
        q_step = log_q_y.float().exp().reshape(-1, 1).clamp_min(1e-6)
        survivability = torch.log1p(amplitude / q_step)
        unpredictability = torch.log1p(innovation / amplitude.clamp_min(1e-6))
        quality = survivability - 0.25 * unpredictability
        # Preserve relative position evidence while keeping a stable range.
        quality = quality - quality.mean(dim=1, keepdim=True)
        scale = quality.square().mean(dim=1, keepdim=True).add(1e-4).sqrt()
        return torch.tanh(quality / scale)

    def forward(
        self,
        source: Tensor,
        *,
        qp_value: Tensor,
        log_q_y: Tensor,
        actual_bpp: Tensor,
        residual_rms: Tensor,
        role_index: int,
    ) -> Tensor:
        if source.ndim != 5:
            raise ValueError("reliability source must be [B,C,T,H,W]")
        batch, temporal = int(source.shape[0]), int(source.shape[2])
        role = int(role_index)
        # A short independent GOP keeps reset role 0: 21 real RGB frames
        # produce six latents, whose learned positions are the reset-table
        # prefix. Do not relabel this as a continuation or replicate content.
        # Full reset-T7 and continuation-T6 arithmetic is unchanged.
        allowed = tuple(range(1, 8)) if role == 0 else tuple(range(1, 7))
        if role not in (0, 1) or temporal not in allowed:
            raise ValueError(f"reliability role {role} expects T in {allowed}, got T={temporal}")
        qp = self._batch_vector(qp_value, batch, "qp_value")
        bpp = self._batch_vector(actual_bpp, batch, "actual_bpp").clamp_min(1e-8)
        rms = self._batch_vector(residual_rms, batch, "residual_rms").clamp_min(1e-8)
        qy = self._batch_vector(log_q_y, batch, "log_q_y")
        position = self._position_quality(source, qy)

        qp_quality = 0.5 - qp / 63.0
        bpp_quality = (bpp.log() - torch.tensor(0.006, device=bpp.device).log()).clamp(-2.0, 2.0)
        rms_quality = (-rms.log()).clamp(-2.0, 2.0)
        weights = F.softplus(self.raw_weights.float())
        logit = (
            self.bias.float()
            + weights[0] * qp_quality[:, None]
            + weights[1] * bpp_quality[:, None]
            + weights[2] * rms_quality[:, None]
            + weights[3] * position
            + self.position_bias[role, :temporal][None]
        )
        floor = float(self.config.floor)
        score = floor + (1.0 - floor) * torch.sigmoid(logit)
        score = score.to(device=source.device, dtype=torch.float32)
        self._last_snapshot = {
            "mean": score.detach().mean().reshape(1),
            "min": score.detach().amin().reshape(1),
            "max": score.detach().amax().reshape(1),
        }
        return score

    def write_reliability(self, score: Tensor) -> Tensor:
        """Keep only reliable frames, with one deterministic fallback per sample."""

        if score.ndim != 2:
            raise ValueError("frame reliability must have shape [B,T]")
        eligible = score >= float(self.config.write_threshold)
        # Avoid an empty external-memory slot at initialization/high QP.  The
        # best decoded position remains available; every other weak position is
        # suppressed to an effectively absent key prior.
        best = score.argmax(dim=1, keepdim=True)
        eligible = eligible.scatter(1, best, True)
        epsilon = torch.full_like(score, float(self.config.write_epsilon))
        written = torch.where(eligible, score, epsilon)
        self._last_snapshot["write_fraction"] = eligible.detach().float().mean().reshape(1)
        return written

    def contract(self) -> dict[str, object]:
        return {
            "enabled": True,
            "inputs": ["qp", "actual_bpp", "qp_residual_rms_ema", "decoded_position_quality"],
            "clean_or_encoder_input": False,
            "within_window": "directional_soft_log_key_bias",
            "cross_window_write": "threshold_with_best_position_fallback",
            "floor": self.config.floor,
            "write_threshold": self.config.write_threshold,
            "write_epsilon": self.config.write_epsilon,
            "query_floor": self.config.query_floor,
            "max_log_bias": self.config.max_log_bias,
        }


__all__ = ["SoftReliabilityConfig", "Task456SoftReliability"]
