"""Aligned original-RGB supervision for the preserved latent datasets."""

from __future__ import annotations
from pathlib import Path
from PIL import Image
import torch


class _SequentialSourceVideoCache:
    """Reuse one ffmpeg reader for adjacent windows of the same video.

    Expanded hard-source shards store consecutive windows (for example c0000
    followed by c0001) from one source video.  Opening a new reader for every
    window makes imageio decode from frame zero repeatedly.  This cache keeps a
    single ``iter_data`` generator and advances it to the requested window;
    non-monotonic windows or a video change safely reopen the reader.
    """

    def __init__(self) -> None:
        self.path: str | None = None
        self.reader = None
        self.frames = None
        self.next_frame = 0

    def close(self) -> None:
        reader = self.reader
        self.path = None
        self.reader = None
        self.frames = None
        self.next_frame = 0
        if reader is not None:
            reader.close()

    def read(self, path: Path, start_frame: int, window_frames: int):
        path_value = str(path)
        start = int(start_frame)
        count = int(window_frames)
        if count <= 0:
            raise ValueError("source video window length must be positive")
        if self.path != path_value or self.reader is None or self.frames is None or start < self.next_frame:
            self.close()
            import imageio.v2 as iio

            self.reader = iio.get_reader(path_value, format="ffmpeg")
            self.frames = iter(self.reader.iter_data())
            self.path = path_value
            self.next_frame = 0
        try:
            while self.next_frame < start:
                next(self.frames)
                self.next_frame += 1
            values = []
            for _ in range(count):
                values.append(Image.fromarray(next(self.frames)).convert("RGB"))
                self.next_frame += 1
            return values
        except BaseException:
            # Do not retain a partially consumed/broken decoder for the next
            # sample.  The caller receives the original exception.
            self.close()
            raise


def _load_aligned_source_video(
    meta: dict,
    target_size: tuple[int, int],
    *,
    sequential_cache: _SequentialSourceVideoCache | None = None,
) -> torch.Tensor:
    """Reload the raw window used to create a latent shard sample.

    The resize/crop path is shared with latent generation.  New shards record
    the exact window-level spatial transform; historical shards fall back to
    the original centre crop.  The returned ``[3,T,H,W]`` tensor is therefore
    an aligned original-RGB target rather than a VAE-decoded proxy target.
    """
    from PIL import Image

    from raqvc.data.video_dataset import frames_to_video_tensor

    video_path = str(meta.get("video_path", ""))
    if not video_path:
        raise ValueError("source-video supervision requires meta.video_path")
    start_frame = int(meta.get("start_frame", 0) or 0)
    window_frames = int(meta.get("raw_window_frames", 0) or 0)
    if window_frames <= 0:
        raise ValueError("source-video supervision requires positive meta.raw_window_frames")
    meta_size = meta.get("target_size", target_size)
    target_h, target_w = (int(meta_size[0]), int(meta_size[1]))

    frames = []
    end_frame = start_frame + window_frames
    path = Path(video_path)
    if path.is_dir():
        import re

        frame_paths = list(path.glob("im*.png")) or list(path.glob("*.png"))

        def numeric_key(frame_path: Path) -> tuple[int, str]:
            matches = re.findall(r"\d+", frame_path.stem)
            return (int(matches[-1]) if matches else 10**9, frame_path.name)

        ordered = sorted(frame_paths, key=numeric_key)
        frames = [
            Image.open(frame_path).convert("RGB") for frame_path in ordered[start_frame : start_frame + window_frames]
        ]
        if frames and len(frames) < window_frames and bool(meta.get("pad_last_frame_to_window", False)):
            frames.extend([frames[-1].copy() for _ in range(window_frames - len(frames))])
    else:
        if sequential_cache is not None:
            frames = sequential_cache.read(path, start_frame, window_frames)
        else:
            import imageio.v2 as iio

            reader = iio.get_reader(video_path, format="ffmpeg")
            try:
                for index, frame in enumerate(reader):
                    if index < start_frame:
                        continue
                    if index >= end_frame:
                        break
                    frames.append(Image.fromarray(frame).convert("RGB"))
            finally:
                reader.close()
    if len(frames) != window_frames:
        raise RuntimeError(
            f"source video {video_path} yielded {len(frames)} frames for "
            f"[{start_frame},{end_frame}), expected {window_frames}"
        )
    spatial_transform = meta.get("spatial_transform")
    return frames_to_video_tensor(
        frames,
        target_h,
        target_w,
        spatial_transform=spatial_transform if isinstance(spatial_transform, dict) else None,
    )
