"""Video frame loading and preprocessing utilities.

All functions accept plain `Path` + shape arguments; no hidden state.
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any, Iterable, List

import numpy as np
import torch
from PIL import Image


def load_image_rgb(path: Path) -> Image.Image:
    """Open an image file and force RGB mode."""
    return Image.open(path).convert("RGB")


def make_spatial_transform(
    image_size: tuple[int, int],
    target_h: int,
    target_w: int,
    *,
    crop_mode: str = "center",
    crop_seed: int = 0,
    crop_key: str = "",
) -> dict[str, Any]:
    """Build a serializable resize/crop transform for an entire video window.

    Random coordinates are derived from a stable SHA-256 key rather than the
    process RNG.  This makes preprocessing independent of worker count and
    allows the exact crop to be replayed when loading source-RGB supervision.
    """
    w, h = image_size
    if crop_mode not in {"center", "random"}:
        raise ValueError(f"Unsupported crop_mode={crop_mode!r}; use 'center' or 'random'")
    if target_h <= 0 or target_w <= 0:
        raise ValueError(f"target size must be positive, got {(target_h, target_w)}")
    scale = max(target_w / w, target_h / h)
    new_w = max(int(round(w * scale)), target_w)
    new_h = max(int(round(h * scale)), target_h)
    max_left = new_w - target_w
    max_top = new_h - target_h
    if crop_mode == "center":
        left = max_left // 2
        top = max_top // 2
    else:
        material = (f"raqvc-spatial-crop-v1\0{int(crop_seed)}\0{crop_key}\0" f"{w}x{h}\0{target_w}x{target_h}").encode(
            "utf-8"
        )
        digest = hashlib.sha256(material).digest()
        left = int.from_bytes(digest[:8], "big") % (max_left + 1)
        top = int.from_bytes(digest[8:16], "big") % (max_top + 1)
    return {
        "version": 1,
        "mode": crop_mode,
        "source_width": int(w),
        "source_height": int(h),
        "resize_width": int(new_w),
        "resize_height": int(new_h),
        "crop_left": int(left),
        "crop_top": int(top),
        "target_width": int(target_w),
        "target_height": int(target_h),
        "crop_seed": int(crop_seed),
        "crop_key": str(crop_key),
    }


def resize_and_crop(
    img: Image.Image,
    target_h: int,
    target_w: int,
    *,
    spatial_transform: dict[str, Any] | None = None,
) -> Image.Image:
    """Resize to fill the target, then apply a centre or recorded crop."""
    transform = spatial_transform or make_spatial_transform(img.size, target_h, target_w, crop_mode="center")
    # Task889 and later cleaned indexes record an explicit source-coordinate
    # edge crop (version 2).  The latent materializer applies this transform
    # before encoding, so the RGB-supervision replay path must dispatch to the
    # same implementation rather than looking for the version-1
    # ``resize_width``/``crop_top`` fields.  Keeping the dispatch here makes
    # ``frames_to_video_tensor`` and all historical callers share one exact
    # preprocessing contract.
    if str(transform.get("mode", "")).strip().lower() == "edge_crop_then_resize":
        return edge_crop_then_resize(
            img,
            target_h=int(target_h),
            target_w=int(target_w),
            spatial_transform=transform,
        )
    expected_size = (
        int(transform.get("source_width", img.size[0])),
        int(transform.get("source_height", img.size[1])),
    )
    if img.size != expected_size:
        raise ValueError(f"frame size {img.size} does not match recorded source size {expected_size}")
    new_w = int(transform["resize_width"])
    new_h = int(transform["resize_height"])
    left = int(transform["crop_left"])
    top = int(transform["crop_top"])
    recorded_target = (
        int(transform.get("target_height", target_h)),
        int(transform.get("target_width", target_w)),
    )
    if recorded_target != (target_h, target_w):
        raise ValueError(f"recorded target {recorded_target} does not match requested {(target_h, target_w)}")
    if left < 0 or top < 0 or left + target_w > new_w or top + target_h > new_h:
        raise ValueError(f"invalid crop ({left}, {top}, {target_w}, {target_h}) for resized {(new_w, new_h)}")
    img = img.resize((new_w, new_h), Image.BICUBIC)
    return img.crop((left, top, left + target_w, top + target_h))


def make_edge_crop_resize_transform(
    image_size: tuple[int, int],
    target_h: int = 256,
    target_w: int = 256,
    *,
    rotation_degrees: int = 0,
) -> dict[str, Any]:
    """Create one square edge-crop/resize transform for a GOP pair.

    The crop is computed in source coordinates by removing the excess long
    edge, then the resulting square is resized to the requested training
    size.  The returned transform is serializable and must be reused for all
    frames in both consecutive GOPs.  Only 0/180 degree rotations are
    accepted so the output geometry remains square and deterministic.
    """
    w, h = (int(image_size[0]), int(image_size[1]))
    if w <= 0 or h <= 0 or target_h <= 0 or target_w <= 0:
        raise ValueError("image and target sizes must be positive")
    rotation = int(rotation_degrees) % 360
    if rotation not in {0, 180}:
        raise ValueError("rotation_degrees must be 0 or 180 for square training crops")
    side = min(w, h)
    left = (w - side) // 2
    top = (h - side) // 2
    return {
        "version": 2,
        "mode": "edge_crop_then_resize",
        "source_width": w,
        "source_height": h,
        "crop_left": int(left),
        "crop_top": int(top),
        "crop_width": int(side),
        "crop_height": int(side),
        "target_width": int(target_w),
        "target_height": int(target_h),
        "rotation_degrees": rotation,
        "transform_scope": "two_consecutive_gops",
    }


def edge_crop_then_resize(
    img: Image.Image,
    target_h: int = 256,
    target_w: int = 256,
    *,
    spatial_transform: dict[str, Any],
) -> Image.Image:
    """Apply a recorded edge crop, optional rotation, and final resize."""
    expected = (
        int(spatial_transform.get("source_width", img.size[0])),
        int(spatial_transform.get("source_height", img.size[1])),
    )
    if img.size != expected:
        raise ValueError(f"frame size {img.size} does not match recorded source size {expected}")
    if spatial_transform.get("mode") != "edge_crop_then_resize":
        raise ValueError("edge_crop_then_resize requires an edge-crop transform")
    left = int(spatial_transform["crop_left"])
    top = int(spatial_transform["crop_top"])
    width = int(spatial_transform["crop_width"])
    height = int(spatial_transform["crop_height"])
    if left < 0 or top < 0 or left + width > img.width or top + height > img.height:
        raise ValueError("recorded edge crop is outside the source image")
    recorded_target = (
        int(spatial_transform.get("target_height", target_h)),
        int(spatial_transform.get("target_width", target_w)),
    )
    if recorded_target != (int(target_h), int(target_w)):
        raise ValueError(f"recorded target {recorded_target} does not match requested {(target_h, target_w)}")
    out = img.crop((left, top, left + width, top + height))
    rotation = int(spatial_transform.get("rotation_degrees", 0)) % 360
    if rotation:
        out = out.rotate(rotation, resample=Image.Resampling.BICUBIC, expand=False)
    return out.resize((int(target_w), int(target_h)), Image.Resampling.BICUBIC)


def pil_to_normalized_tensor(img: Image.Image) -> torch.Tensor:
    """RGB PIL → [3, H, W] float tensor in [-1, 1]."""
    arr = torch.from_numpy(np.array(img, dtype=np.float32))  # [H, W, 3]
    t = arr.permute(2, 0, 1) / 127.5 - 1.0
    return t.contiguous()


def frames_to_video_tensor(
    frames: Iterable[Image.Image],
    target_h: int,
    target_w: int,
    *,
    crop_mode: str = "center",
    crop_seed: int = 0,
    crop_key: str = "",
    spatial_transform: dict[str, Any] | None = None,
    return_spatial_transform: bool = False,
) -> torch.Tensor | tuple[torch.Tensor, dict[str, Any]]:
    """Stack frames to ``[C,T,H,W]`` using one crop for the whole window."""
    frame_list = list(frames)
    if not frame_list:
        raise ValueError("frames_to_video_tensor requires at least one frame")
    transform = spatial_transform or make_spatial_transform(
        frame_list[0].size,
        target_h,
        target_w,
        crop_mode=crop_mode,
        crop_seed=crop_seed,
        crop_key=crop_key,
    )
    tensors: List[torch.Tensor] = []
    for f in frame_list:
        f = resize_and_crop(
            f,
            target_h,
            target_w,
            spatial_transform=transform,
        )
        tensors.append(pil_to_normalized_tensor(f))
    stacked = torch.stack(tensors, dim=1)  # [C, T, H, W]
    if return_spatial_transform:
        return stacked, transform
    return stacked


def two_gops_to_video_tensor(
    gops: Iterable[Iterable[Image.Image]],
    *,
    gop_length: int = 25,
    target_h: int = 256,
    target_w: int = 256,
    rotation_degrees: int = 0,
    spatial_transform: dict[str, Any] | None = None,
    return_spatial_transform: bool = False,
) -> torch.Tensor | tuple[torch.Tensor, dict[str, Any]]:
    """Stack exactly two consecutive GOPs with one shared transform.

    ``gops`` must contain two GOPs of exactly ``gop_length`` frames.  A
    transform is derived from the first frame once and applied to every frame
    in both GOPs, preventing crop/rotation drift across the temporal pair.
    """
    gop_list = [list(gop) for gop in gops]
    if len(gop_list) != 2 or any(len(gop) != int(gop_length) for gop in gop_list):
        raise ValueError(f"two_gops_to_video_tensor requires two GOPs of {gop_length} frames")
    first = gop_list[0][0]
    transform = spatial_transform or make_edge_crop_resize_transform(
        first.size, target_h, target_w, rotation_degrees=rotation_degrees
    )
    tensors: List[torch.Tensor] = []
    for frame in gop_list[0] + gop_list[1]:
        transformed = edge_crop_then_resize(frame, target_h, target_w, spatial_transform=transform)
        tensors.append(pil_to_normalized_tensor(transformed))
    stacked = torch.stack(tensors, dim=1)
    if return_spatial_transform:
        return stacked, transform
    return stacked
