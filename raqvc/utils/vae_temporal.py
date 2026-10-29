"""Temporal accounting helpers for Wan-style causal 3D VAEs.

Wan encodes frame 0 separately and then consumes subsequent frames in groups of
four.  A raw clip is exactly aligned when its length is ``1 + 4k``.
"""

from __future__ import annotations

from typing import Sequence, TypeVar

WAN_TEMPORAL_GROUP = 4
_Scalar = TypeVar("_Scalar")


def latent_frames_from_raw(raw_frames: int) -> int:
    """Return Wan latent frames produced by ``raw_frames`` input frames."""
    if raw_frames <= 0:
        raise ValueError(f"raw_frames must be positive, got {raw_frames}")
    return 1 + (int(raw_frames) - 1) // WAN_TEMPORAL_GROUP


def effective_raw_frames_from_latent(latent_frames: int) -> int:
    """Return the raw-frame span represented by ``latent_frames`` latents."""
    if latent_frames <= 0:
        raise ValueError(f"latent_frames must be positive, got {latent_frames}")
    return 1 + WAN_TEMPORAL_GROUP * (int(latent_frames) - 1)


def is_aligned_raw_length(raw_frames: int) -> bool:
    """Return whether ``raw_frames`` lands exactly on the Wan temporal grid."""
    if raw_frames <= 0:
        return False
    return (int(raw_frames) - 1) % WAN_TEMPORAL_GROUP == 0


def trim_to_aligned_length(raw_frames: int) -> int:
    """Trim ``raw_frames`` down to the nearest positive aligned length."""
    if raw_frames <= 0:
        raise ValueError(f"raw_frames must be positive, got {raw_frames}")
    return effective_raw_frames_from_latent(latent_frames_from_raw(raw_frames))


def next_aligned_length(raw_frames: int) -> int:
    """Return the smallest aligned length greater than or equal to ``raw_frames``."""
    if raw_frames <= 0:
        raise ValueError(f"raw_frames must be positive, got {raw_frames}")
    if is_aligned_raw_length(raw_frames):
        return int(raw_frames)
    return 1 + WAN_TEMPORAL_GROUP * (((int(raw_frames) - 1) // WAN_TEMPORAL_GROUP) + 1)


def legacy_raw_frames_from_latent(latent_frames: int) -> int:
    """Return the legacy denominator used by older scripts: ``4 * T_latent``."""
    if latent_frames <= 0:
        raise ValueError(f"latent_frames must be positive, got {latent_frames}")
    return WAN_TEMPORAL_GROUP * int(latent_frames)


def effective_bpp_from_per_latent(per_latent_bpp: Sequence[_Scalar]) -> _Scalar:
    """Aggregate per-latent rates on Wan's causal effective-frame grid.

    Codec frame-level rates are normalized by ``4 * H * W``.  Their payloads
    therefore add as ``4 * sum(bpp_t)``, while the clip represents
    ``1 + 4*(T-1)`` raw frames.  The function intentionally accepts tensors as
    well as floats so training gradients are preserved.
    """
    if len(per_latent_bpp) == 0:
        raise ValueError("per_latent_bpp must contain at least one value")
    effective_frames = effective_raw_frames_from_latent(len(per_latent_bpp))
    return sum(per_latent_bpp) * (WAN_TEMPORAL_GROUP / float(effective_frames))
