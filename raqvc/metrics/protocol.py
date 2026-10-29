"""VoRTeC A.2.2 evaluation geometry and mergeable torchmetrics patch FID.

The shifted grid deliberately uses (floor(H/f)-1)*(floor(W/f)-1) patches,
exactly as stated in the paper, rather than every shifted crop that fits.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

DEPENDENCIES = Path(
    os.environ.get("TASK938_METRIC_DEPS", "/qzwang/data/raqvc_artifacts/C512_FINAL_v1/runtime_assets/paper_metric_deps")
)
if DEPENDENCIES.is_dir() and str(DEPENDENCIES) not in sys.path:
    sys.path.insert(0, str(DEPENDENCIES))

STATES = tuple(f"{side}_features_{kind}" for side in ("real", "fake") for kind in ("sum", "cov_sum", "num_samples"))


def center_crop(rgb, resolution):
    h, w = rgb.shape[-2:]
    oh, ow = {720: (720, 1280), 1080: (1080, 1920)}[resolution]
    if h < oh or w < ow:
        raise ValueError("paper protocol requires a native source at least as large as the crop")
    top, left = (h - oh) // 2, (w - ow) // 2
    return rgb[..., top : top + oh, left : left + ow].contiguous()


def coding_crop(rgb, resolution):
    """720 metrics use the center of a real 736x1280 coding crop."""
    if resolution == 1080:
        return center_crop(rgb, 1080)
    if resolution != 720:
        raise ValueError("unsupported resolution")
    h, w = rgb.shape[-2:]
    if h < 736 or w < 1280:
        raise ValueError("720 protocol requires a real 736x1280 source crop")
    top, left = (h - 736) // 2, (w - 1280) // 2
    return rgb[..., top : top + 736, left : left + 1280].contiguous()


def metric_crop(rgb, resolution):
    """Use identical scoring coordinates for decoded RGB and source RGB."""
    h, w = rgb.shape[-2:]
    if resolution == 720:
        if (h, w) != (736, 1280):
            raise ValueError(f"expected 736x1280 coding canvas, got {(h,w)}")
        return rgb[..., 8:728, :].contiguous()
    if resolution == 1080 and (h, w) in ((1080, 1920), (1088, 1920)):
        return rgb[..., :1080, :1920].contiguous()
    raise ValueError(f"invalid scoring canvas: resolution={resolution}, shape={(h,w)}")


def prepare_gop(rgb, frames=25):
    h, w = rgb.shape[-2:]
    if not 0 < rgb.shape[1] <= frames:
        raise ValueError("invalid source GOP length")
    # Keep native H2 reflect padding at 1080; the 736x1280 crop needs no padding.
    padded = F.pad(rgb, (0, (-w) % 32, 0, (-h) % 32), mode="reflect")
    if padded.shape[1] < frames:
        padded = torch.cat([padded, padded[:, -1:].expand(-1, frames - padded.shape[1], -1, -1)], 1)
    return padded


def patch_coordinates(height, width, size=256):
    nh, nw = height // size, width // size
    if min(nh, nw) < 1:
        raise ValueError("image smaller than FID patch")
    return [(y * size, x * size) for y in range(nh) for x in range(nw)] + [
        (size // 2 + y * size, size // 2 + x * size) for y in range(nh - 1) for x in range(nw - 1)
    ]


def iter_patch_batches(rgb, batch=32):
    if rgb.ndim != 4 or rgb.shape[0] != 3 or batch < 1:
        raise ValueError("expected RGB [3,T,H,W] and positive patch batch")
    pending = []
    for frame in range(rgb.shape[1]):
        for y, x in patch_coordinates(*rgb.shape[-2:]):
            pending.append(rgb[:, frame, y : y + 256, x : x + 256])
            if len(pending) == batch:
                yield torch.stack(pending)
                pending = []
    if pending:
        yield torch.stack(pending)


class PatchFID:
    def __init__(self, device="cuda:0", batch=32):
        from torchmetrics.image.fid import FrechetInceptionDistance
        from importlib.metadata import version

        self.device = torch.device(device)
        self.batch = batch
        self.metric = FrechetInceptionDistance(feature=2048, normalize=True).to(self.device).eval()
        self.metric.requires_grad_(False)
        self.metadata = dict(
            implementation="torchmetrics.image.fid.FrechetInceptionDistance",
            torchmetrics=version("torchmetrics"),
            torch_fidelity=version("torch-fidelity"),
            dimensions=2048,
            patch=256,
            shifted_origin=[128, 128],
            grid_counts="floor(H/256)*floor(W/256)+(floor(H/256)-1)*(floor(W/256)-1)",
            input="RGB_float32_0_1; official_normalize_True_multiplies_255_then_uint8",
            feature_precision="float32_no_autocast",
            moment_precision="float64",
            aggregation="dataset_pooled_patch_moments; not_mean_video_FID",
            patch_batch=batch,
        )

    def reset(self, real=None):
        self.metric.reset()
        if real is not None:
            for key in STATES:
                if key.startswith("real_"):
                    getattr(self.metric, key).copy_(torch.as_tensor(real[key], device=self.device))

    @torch.inference_mode()
    def update(self, rgb, real):
        with torch.autocast(device_type=self.device.type, enabled=False):
            for patches in iter_patch_batches(rgb, self.batch):
                self.metric.update(patches.to(self.device).float(), real=real)

    def state(self):
        return {name: getattr(self.metric, name).detach().cpu().numpy().copy() for name in STATES}


class IdentityFeatures(torch.nn.Module):
    """Only used to compute FID from saved official sufficient statistics."""

    num_features = 2048

    def forward(self, x):
        return x


def compute_pooled_fid(states, device="cpu"):
    from torchmetrics.image.fid import FrechetInceptionDistance

    if not states:
        raise ValueError("no complete FID states")
    metric = FrechetInceptionDistance(feature=IdentityFeatures()).to(device)
    for state in states:
        for key in STATES:
            getattr(metric, key).add_(torch.as_tensor(state[key], device=device))
    # Public compute() applies the identical torchmetrics covariance/FID formula.
    metric.orig_dtype = torch.float64
    metric._update_count = len(states)
    value = float(metric.compute().cpu())
    if not np.isfinite(value):
        raise ValueError("nonfinite pooled FID")
    return value
