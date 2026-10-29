"""Small, explicit helpers for paper benchmark YUV420 inputs.

The standard HEVC-B, MCL-JCV and UVG files used by this repository are raw
8-bit planar YUV420.  Their geometry is encoded in the filename.  RGB
conversion follows the BT.709 full-range convention used by the official
DCVC evaluation utilities: byte values are divided by 255, chroma is
nearest-neighbour upsampled, and the BT.709 YCbCr matrix is applied.
"""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import torch

_GEOMETRY_RE = re.compile(r"_(?P<width>\d+)x(?P<height>\d+)(?:_|\.)")


def geometry_from_filename(path: str | Path) -> tuple[int, int]:
    """Return ``(height, width)`` parsed from a standard YUV filename."""
    match = _GEOMETRY_RE.search(Path(path).name)
    if match is None:
        raise ValueError(f"cannot infer raw YUV geometry from filename: {path}")
    return int(match.group("height")), int(match.group("width"))


def frame_count(path: str | Path, height: int, width: int) -> int:
    frame_bytes = height * width * 3 // 2
    size = Path(path).stat().st_size
    if size % frame_bytes:
        raise ValueError(f"raw YUV size is not frame aligned: path={path} size={size} " f"frame_bytes={frame_bytes}")
    return size // frame_bytes


def read_yuv420p8(
    path: str | Path,
    *,
    max_frames: int | None = None,
    height: int | None = None,
    width: int | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Read planar YUV420p8 as three float32 arrays in ``[0, 1]``."""
    path = Path(path)
    if height is None or width is None:
        height, width = geometry_from_filename(path)
    available = frame_count(path, height, width)
    count = available if max_frames is None else min(available, int(max_frames))
    frame_samples = height * width * 3 // 2
    raw = np.fromfile(path, dtype=np.uint8, count=count * frame_samples)
    raw = raw.reshape(count, frame_samples)
    y_end = height * width
    uv_samples = y_end // 4
    y = raw[:, :y_end].reshape(count, height, width)
    u = raw[:, y_end : y_end + uv_samples].reshape(count, height // 2, width // 2)
    v = raw[:, y_end + uv_samples :].reshape(count, height // 2, width // 2)
    return tuple(value.astype(np.float32) / 255.0 for value in (y, u, v))


def yuv420p8_to_rgb_bt709(y: np.ndarray, u: np.ndarray, v: np.ndarray) -> torch.Tensor:
    """Convert YUV420 arrays to RGB ``[3,T,H,W]`` with DCVC's BT.709 rule."""
    if y.ndim != 3 or u.ndim != 3 or v.ndim != 3:
        raise ValueError("expected Y/U/V arrays with shapes [T,H,W], [T,H/2,W/2]")
    if u.shape != v.shape or y.shape[0] != u.shape[0]:
        raise ValueError(f"incompatible YUV shapes: y={y.shape} u={u.shape} v={v.shape}")
    cb = np.repeat(np.repeat(u, 2, axis=1), 2, axis=2)
    cr = np.repeat(np.repeat(v, 2, axis=1), 2, axis=2)
    kr, kg, kb = 0.2126, 0.7152, 0.0722
    red = y + (2.0 - 2.0 * kr) * (cr - 0.5)
    blue = y + (2.0 - 2.0 * kb) * (cb - 0.5)
    green = (y - kr * red - kb * blue) / kg
    rgb = np.stack((red, green, blue), axis=0)
    return torch.from_numpy(np.clip(rgb, 0.0, 1.0)).float()
