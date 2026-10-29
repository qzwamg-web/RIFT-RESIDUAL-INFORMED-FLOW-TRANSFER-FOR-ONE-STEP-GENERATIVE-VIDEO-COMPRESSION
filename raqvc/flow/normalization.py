"""Fixed QP normalization used by the RIFT velocity interface."""

from __future__ import annotations
from dataclasses import dataclass
from typing import Mapping, Sequence
import torch
import torch.nn.functional as F
from torch import Tensor, nn

TASK438_QP_VALUES = (0, 16, 39, 63)


def _broadcast_like(vector: Tensor, reference: Tensor) -> Tensor:
    return vector.reshape(vector.shape[0], *[1] * (reference.ndim - 1))


class QPConditionalFlowPath(nn.Module):

    def __init__(self):
        super().__init__()
        self.register_buffer("qp_values", torch.tensor(TASK438_QP_VALUES, dtype=torch.long))
        self.register_buffer("gamma", torch.ones(4))
        self.register_buffer("residual_rms", torch.ones(4))
        self.register_buffer("residual_updates", torch.zeros(4, dtype=torch.long))
        self.register_buffer("position_residual_rms", torch.ones(4, 6))
        self.minimum_rms = 1e-4

    def qp_indices(self, qp: Tensor | int) -> Tensor:
        value = torch.as_tensor(qp, device=self.qp_values.device, dtype=torch.long)
        original_shape = value.shape
        value = value.reshape(-1)
        matches = value[:, None].eq(self.qp_values[None])
        if not bool(matches.any(dim=1).all()):
            bad = value[~matches.any(dim=1)].detach().cpu().tolist()
            raise ValueError(f"unsupported QP values: {bad}")
        return matches.long().argmax(dim=1).reshape(original_shape)

    def _parameter(self, table: Tensor, qp: Tensor | int, *, batch: int) -> Tensor:
        indices = self.qp_indices(qp).reshape(-1)
        if indices.numel() == 1:
            indices = indices.expand(batch)
        if indices.numel() != batch:
            raise ValueError(f"QP batch has {indices.numel()} values, expected {batch}")
        return table[indices]

    def sigma(self, qp: Tensor | int, *, batch: int) -> Tensor:
        return self._parameter(self.residual_rms, qp, batch=batch).clamp_min(self.minimum_rms)

    def sigma_like(self, qp, reference):
        scalar = self.sigma(qp, batch=int(reference.shape[0])).to(reference)
        return _broadcast_like(scalar, reference)

    def _load_from_state_dict(
        self,
        state_dict: Mapping[str, Tensor],
        prefix: str,
        local_metadata: Mapping[str, object],
        strict: bool,
        missing_keys: list[str],
        unexpected_keys: list[str],
        error_msgs: list[str],
    ) -> None:
        """Strictly load legacy parents that predate the opt-in P table."""
        key = prefix + "position_residual_rms"
        legacy = key not in state_dict
        super()._load_from_state_dict(
            state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs
        )
        if legacy and key in missing_keys:
            missing_keys.remove(key)

    def denormalize_velocity(self, normalized_velocity: Tensor, qp: Tensor | int) -> Tensor:
        sigma = self.sigma_like(qp, normalized_velocity)
        return normalized_velocity * sigma
