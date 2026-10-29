"""Streaming native-float metrics for the Task938 H2 benchmark.

The stream keeps only the preceding RGB frame, per-frame scalar scores, and
Inception pool3 features. FloLPIPS includes pairs crossing GOP boundaries.
"""

from __future__ import annotations

from importlib.metadata import PackageNotFoundError, version
from typing import Any

import numpy as np
import torch

DEFAULT_METRICS = {
    720: ("psnr", "lpips_vgg", "dists", "lpips_alex", "fid", "flolpips"),
    1080: ("psnr", "lpips_alex", "fid", "flolpips"),
}
ALL_METRICS = tuple(dict.fromkeys(DEFAULT_METRICS[720]))


def package_versions() -> dict[str, str | None]:
    result = {}
    for name in ("torch", "torchvision", "lpips", "DISTS-pytorch", "pytorch-fid"):
        try:
            result[name] = version(name)
        except PackageNotFoundError:
            result[name] = None
    return result


class NativeMetricEngine:
    """Load each feature network once, reused by all videos, QPs, and sizes."""

    def __init__(self, device: str, metrics=ALL_METRICS, batch_size: int = 1):
        self.device = torch.device(device)
        self.names = tuple(metrics)
        self.batch_size = int(batch_size)
        if self.batch_size < 1 or set(self.names) - set(ALL_METRICS):
            raise ValueError("invalid metrics or metric batch size")
        self.lpips = {}
        for key, backbone in (("lpips_vgg", "vgg"), ("lpips_alex", "alex")):
            if key in self.names:
                import lpips

                self.lpips[key] = lpips.LPIPS(net=backbone, version="0.1").to(self.device).eval().requires_grad_(False)
        if "dists" in self.names:
            from raqvc.metrics.perceptual import DISTSMetric

            self.dists = DISTSMetric(device=str(self.device))
        if "fid" in self.names:
            from raqvc.metrics.perceptual import FIDMetric

            self.fid = FIDMetric(device=str(self.device), dims=2048)
        if "flolpips" in self.names:
            from raqvc.metrics.flolpips import NativeFloLPIPS

            self.flo = NativeFloLPIPS(device=str(self.device))
        self.metadata = dict(
            packages=package_versions(),
            rgb="float32_0_1_native_cropped",
            lpips="official_lpips_v0.1; input_m1_p1; frame_mean",
            dists="DISTS_pytorch; input_0_1; frame_mean",
            fid="pytorch_fid_pool3_2048; internal_299x299_resize; dataset_pooled_frames",
            flolpips=getattr(getattr(self, "flo", None), "metadata", None),
        )

    @torch.inference_mode()
    def frames(self, reference, reconstruction, names) -> dict[str, list[float]]:
        if reference.shape != reconstruction.shape or reference.ndim != 4 or reference.shape[0] != 3:
            raise ValueError("metrics require matched [3,T,H,W] RGB")
        errors = (reference.float() - reconstruction.float()).square().mean((0, 2, 3))
        result = {"rgb_mse": errors.tolist()}
        if "psnr" in names:
            result["psnr"] = (-10 * errors.clamp_min(1e-12).log10()).tolist()
        for key in names:
            if key in self.lpips or key == "dists":
                result[key] = []
        for start in range(0, reference.shape[1], self.batch_size):
            left = reference[:, start : start + self.batch_size].permute(1, 0, 2, 3).to(self.device).float()
            right = reconstruction[:, start : start + self.batch_size].permute(1, 0, 2, 3).to(self.device).float()
            for key, metric in self.lpips.items():
                if key in names:
                    result[key].extend(metric(left * 2 - 1, right * 2 - 1).reshape(-1).float().cpu().tolist())
            if "dists" in names:
                # The wrapper deliberately averages a batch. Score each frame
                # separately so saved frame scores remain auditable.
                for index in range(left.shape[0]):
                    result["dists"].append(float(self.dists(left[index : index + 1], right[index : index + 1])))
        return result

    @torch.inference_mode()
    def features(self, rgb):
        return self.fid.extract_features(rgb.permute(1, 0, 2, 3), batch_size=self.batch_size, input_range="zero_one")


class MetricStream:
    """Frame and pair accounting independent of how RGB is split into GOPs."""

    def __init__(self, engine: NativeMetricEngine, names):
        self.engine, self.names = engine, tuple(names)
        self.frame_scores: dict[str, list[float]] = {}
        self.pair_scores: list[float] = []
        self.pair_status: list[str] = []
        self.real_features: list[np.ndarray] = []
        self.fake_features: list[np.ndarray] = []
        self.previous_reference = self.previous_reconstruction = None
        self.frames = 0
        self.boundary_indices: list[int] = []

    @torch.inference_mode()
    def add(self, reference, reconstruction, reference_features=None):
        scores = self.engine.frames(reference, reconstruction, self.names)
        for key, values in scores.items():
            if len(values) != reference.shape[1] or not np.isfinite(values).all():
                raise ValueError(f"invalid per-frame {key}")
            self.frame_scores.setdefault(key, []).extend(values)
        if "fid" in self.names:
            if reference_features is None:
                reference_features = self.engine.features(reference)
            self.real_features.append(reference_features)
            self.fake_features.append(self.engine.features(reconstruction))
        if "flolpips" in self.names:
            for index in range(reference.shape[1]):
                current_ref = reference[:, index].unsqueeze(0)
                current_rec = reconstruction[:, index].unsqueeze(0)
                if self.previous_reference is not None:
                    value = self.engine.flo.per_pair(
                        self.previous_reference, current_ref, self.previous_reconstruction, current_rec
                    )
                    value = float(value.reshape(-1).mean())
                    statuses = getattr(self.engine.flo, "last_pair_status", ["ok"])
                    status = statuses[0]
                    if not np.isfinite(value) and not status.startswith("undefined_official_"):
                        raise ValueError("unexplained nonfinite FloLPIPS pair")
                    self.pair_scores.append(value)
                    self.pair_status.append(status)
                    if index == 0:
                        self.boundary_indices.append(self.frames)
                # Clone only one frame: a view would retain the whole previous GOP.
                self.previous_reference = current_ref.clone()
                self.previous_reconstruction = current_rec.clone()
        self.frames += int(reference.shape[1])
        return reference_features

    def result(self) -> dict[str, Any]:
        if self.frames < 1:
            raise ValueError("no metric frames")
        result = {key: float(np.mean(values)) for key, values in self.frame_scores.items()}
        if "flolpips" in self.names:
            if len(self.pair_scores) != self.frames - 1:
                raise RuntimeError("FloLPIPS must include every adjacent pair")
            finite = np.isfinite(self.pair_scores)
            values = np.asarray(self.pair_scores)[finite]
            status_counts = {status: self.pair_status.count(status) for status in sorted(set(self.pair_status))}
            result.update(
                flolpips=float(np.mean(values)) if finite.all() else None,
                flolpips_pairs=len(self.pair_scores),
                flolpips_undefined_pairs=int((~finite).sum()),
                flolpips_finite_pair_mean=float(np.mean(values)) if len(values) else None,
                flolpips_pair_status_counts=status_counts,
                flolpips_gop_boundary_indices=self.boundary_indices,
            )
        if "fid" in self.names:
            result.update(
                fid_frames=self.frames,
                fid_feature_dimensions=2048,
                fid_aggregation="pooled_by_original_dataset_resolution_QP_route; not_mean_video_FID",
            )
        return result

    def arrays(self, route: str) -> dict[str, np.ndarray]:
        result = {
            f"{route}_{key}_frames": np.asarray(values, dtype=np.float32) for key, values in self.frame_scores.items()
        }
        if "flolpips" in self.names:
            result[f"{route}_flolpips_pairs"] = np.asarray(self.pair_scores, dtype=np.float32)
            result[f"{route}_flolpips_status"] = np.asarray(self.pair_status, dtype="U80")
        if "fid" in self.names:
            real, fake = np.concatenate(self.real_features), np.concatenate(self.fake_features)
            if real.shape != (self.frames, 2048) or fake.shape != real.shape:
                raise ValueError("invalid FID feature shape")
            if not np.isfinite(real).all() or not np.isfinite(fake).all():
                raise ValueError("nonfinite FID features")
            result.update(reference=real.astype(np.float32), **{route: fake.astype(np.float32)})
        return result
