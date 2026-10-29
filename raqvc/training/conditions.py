"""Shared runtime contract for Task938 mainline train/eval."""

from __future__ import annotations

import torch
import torch.nn.functional as F

from raqvc.training.types import MainlineCondition
from raqvc.final_support import q_summary

QPS = (0, 16, 39, 63)


def build_condition(codec, source, sigma, width, actual, qp, device) -> MainlineCondition:
    """Build the decoder-visible condition used by both train and eval."""
    log_y, log_z = q_summary(codec, int(qp))
    batch = int(source.shape[0])
    log_q_y = source.new_full((batch,), float(log_y), device=device)
    log_q_z = source.new_full((batch,), float(log_z), device=device)
    return MainlineCondition(
        source=source, sigma=sigma, width=width, actual_bpp=actual, log_q_y=log_q_y, log_q_z=log_q_z, qp=int(qp)
    )
