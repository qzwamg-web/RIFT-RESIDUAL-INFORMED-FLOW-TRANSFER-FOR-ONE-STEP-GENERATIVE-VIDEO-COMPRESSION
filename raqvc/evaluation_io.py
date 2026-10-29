"""Native RGB source and atomic output helpers."""

import os, json, re, contextlib
from pathlib import Path
from typing import Any
from raqvc.training.c512 import QPS, TAU_TABLE as _TAU_TABLE, DEFAULT_WAN as WAN

TAU_TABLE = list(_TAU_TABLE)

DATASETS = ("HEVC_B", "MCL-JCV", "UVG")


def atomic_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n")
    os.replace(temporary, path)


def file_identity(path: Path) -> dict[str, Any]:
    stat = path.stat()
    return dict(path=str(path.resolve()), bytes=stat.st_size, mtime_ns=stat.st_mtime_ns)


def open_video_writer(path: Path, fps: int):
    import imageio.v2 as imageio

    path.parent.mkdir(parents=True, exist_ok=True)
    return imageio.get_writer(str(path), fps=fps, codec="libx264", quality=8, macro_block_size=1, pixelformat="yuv420p")


def append_video(writer, value) -> None:
    import torch

    # One frame conversion at a time bounds the additional CPU memory.
    for frame in value.unbind(1):
        writer.append_data((frame.clamp(0, 1) * 255).round().to(torch.uint8).permute(1, 2, 0).numpy())


def padded_shape(height: int, width: int) -> tuple[int, int]:
    # Wan latent H/W must be even for the DiT patch embed. 720 -> 736, not 720.
    return ((height + 31) // 32) * 32, ((width + 31) // 32) * 32


class SourceReader:
    """Map raw YUV and materialize only one RGB GOP, using canonical BT.709."""

    def __init__(self, video: Path, dataset: str, fps: int, frames: int):
        import numpy as np
        from raqvc.data.raw_yuv import frame_count, geometry_from_filename

        if dataset not in DATASETS or geometry_from_filename(video) != (1080, 1920):
            raise ValueError(f"invalid dataset/geometry: {dataset} {video}")
        match = re.search(r"_(\d+)(?:fps)?(?:_420.*)?\.yuv$", video.name)
        if match is None or int(match.group(1)) != fps:
            raise ValueError(f"FPS does not match source filename: {video}")
        if frame_count(video, 1080, 1920) < frames:
            raise ValueError(f"source has fewer than {frames} frames: {video}")
        self.frames = frames
        self.raw = np.memmap(video, dtype=np.uint8, mode="r", shape=(frames, 1080 * 1920 * 3 // 2))

    def read(self, start: int, stop: int, resolution: int):
        import numpy as np
        from raqvc.data.raw_yuv import yuv420p8_to_rgb_bt709

        if not 0 <= start < stop <= self.frames:
            raise ValueError("invalid source frame range")
        block = self.raw[start:stop]
        pixels, count = 1080 * 1920, stop - start
        y = block[:, :pixels].reshape(count, 1080, 1920).astype(np.float32) / 255.0
        u = block[:, pixels : pixels + pixels // 4].reshape(count, 540, 960).astype(np.float32) / 255.0
        v = block[:, pixels + pixels // 4 :].reshape(count, 540, 960).astype(np.float32) / 255.0
        if resolution != 1080:
            raise ValueError("use paper_protocol.coding_crop for 720p")
        return yuv420p8_to_rgb_bt709(y, u, v)


def atomic_npz(path: Path, arrays: dict) -> None:
    import numpy as np

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("wb") as stream:
        np.savez(stream, **arrays)
    os.replace(temporary, path)
