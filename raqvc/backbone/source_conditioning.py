"""Decoder-available QP/source conditioning for Task438 DiT."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor, nn

QP_VALUES = (0, 16, 39, 63)


def _vector(value: Tensor, batch: int, name: str) -> Tensor:
    if value.ndim == 0:
        value = value.expand(batch)
    if value.ndim == 2 and value.shape[1] == 1:
        value = value[:, 0]
    if value.shape != (batch,):
        raise ValueError(f"{name} must be scalar/[B]/[B,1], got {tuple(value.shape)}")
    return value


@dataclass(frozen=True)
class Task438SourceObservation:
    """Information present in the coded stream or reconstructed decoder state."""

    qp_value: Tensor
    log_q_y: Tensor
    log_q_z: Tensor
    actual_bpp: Tensor
    window_role: Tensor  # reset=0, continuation=1
    temporal_offset: Tensor


class Task438SourceConditioner(nn.Module):
    """Build a single Wan-width token for a conditional source distribution."""

    def __init__(self, output_dim: int, hidden_dim: int = 256) -> None:
        super().__init__()
        self.qp_embedding = nn.Embedding(4, 32)
        self.role_embedding = nn.Embedding(2, 16)
        self.net = nn.Sequential(
            nn.Linear(32 + 16 + 6, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, output_dim),
        )
        # Additive Task438 source conditioning must be a functional no-op at
        # Task297 initialization. T0 learns this residual without randomly
        # perturbing the proven parent QP/transition token.
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    @staticmethod
    def qp_index(qp_value: Tensor) -> Tensor:
        values = torch.tensor(QP_VALUES, device=qp_value.device, dtype=torch.long)
        qp_value = qp_value.long()
        matches = qp_value[:, None].eq(values[None])
        if not bool(matches.any(1).all()):
            invalid = qp_value[~matches.any(1)].detach().cpu().tolist()
            raise ValueError(f"unsupported Task438 QP values {invalid}")
        return matches.long().argmax(1)

    def forward(self, observation: Task438SourceObservation, residual_rms: Tensor) -> Tensor:
        candidates = (
            observation.qp_value,
            observation.log_q_y,
            observation.log_q_z,
            observation.actual_bpp,
            observation.window_role,
            observation.temporal_offset,
            residual_rms,
        )
        batch = max(int(value.numel()) for value in candidates)
        qp = _vector(observation.qp_value, batch, "qp_value").long()
        role = _vector(observation.window_role, batch, "window_role").long()
        if bool((role < 0).any() or (role > 1).any()):
            raise ValueError("window_role must be reset=0 or continuation=1")
        q_y = _vector(observation.log_q_y, batch, "log_q_y").float()
        q_z = _vector(observation.log_q_z, batch, "log_q_z").float()
        bpp = _vector(observation.actual_bpp, batch, "actual_bpp").float()
        rms = _vector(residual_rms, batch, "residual_rms").float()
        offset = _vector(observation.temporal_offset, batch, "temporal_offset").float()
        if bool((bpp <= 0).any() or (rms <= 0).any()):
            raise ValueError("actual_bpp and residual_rms must be positive")
        phase = offset / 32.0
        scalars = torch.stack((q_y, q_z, bpp.log(), rms.log(), phase.sin(), phase.cos()), dim=-1)
        parameter = next(self.net.parameters())
        features = torch.cat(
            (
                self.qp_embedding(self.qp_index(qp)),
                self.role_embedding(role),
                scalars,
            ),
            dim=-1,
        ).to(device=parameter.device, dtype=parameter.dtype)
        return self.net(features).unsqueeze(1)

    def contract(self) -> dict[str, object]:
        return {
            "conditions": [
                "qp_value",
                "actual_log_q_y",
                "actual_log_q_z",
                "actual_bpp",
                "qp_residual_rms_ema",
                "window_role",
                "absolute_latent_offset",
            ],
            "clean_target_condition": False,
            "encoder_only_condition": False,
            "initialization": "zero_residual_over_task297_qp_transition_token",
        }


__all__ = ["Task438SourceConditioner", "Task438SourceObservation"]
