"""Decoder-visible H2 condition record (shared train/eval)."""

from __future__ import annotations
from dataclasses import dataclass
from torch import Tensor


@dataclass
class MainlineCondition:
    source: Tensor
    sigma: Tensor
    width: Tensor
    actual_bpp: Tensor
    log_q_y: Tensor
    log_q_z: Tensor
    qp: int
