"""Shared final-release helpers; numerical formulas retained from validated H2."""

from __future__ import annotations
import os, re, json, tempfile
from pathlib import Path
from typing import Any, Mapping
import torch
from torch import Tensor
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint
from raqvc.metrics.perceptual import DISTSMetric
from raqvc.data.raw_yuv import frame_count, geometry_from_filename, read_yuv420p8, yuv420p8_to_rgb_bt709

DATASET_CONTRACT = {"HEVC_B": (60, 1080, 1920), "MCL-JCV": (30, 1080, 1920), "UVG": (120, 1080, 1920)}


def _atomic_torch(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=path.parent, prefix=path.name + ".", delete=False) as f:
        tmp = Path(f.name)
    try:
        torch.save(payload, tmp)
        os.replace(tmp, path)
    finally:
        if tmp.exists():
            tmp.unlink()


def _atomic_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=path.parent, prefix=path.name + ".", mode="w", delete=False) as f:
        json.dump(payload, f, indent=2, default=str)
        f.flush()
        os.fsync(f.fileno())
        tmp = Path(f.name)
    os.replace(tmp, path)


def _rgb_metrics(pred, target, lpips_model, dists_model, metric_batch: int):
    """Score every frame equally, with metric graphs checkpointed by chunks."""
    b, c, t, h, w = pred.shape
    x = pred.permute(0, 2, 1, 3, 4).reshape(b * t, c, h, w).float()
    y = target.permute(0, 2, 1, 3, 4).reshape(b * t, c, h, w).float().detach()
    lvals, dvals, counts = [], [], []
    for start in range(0, x.shape[0], max(1, metric_batch)):
        xx, yy = x[start : start + metric_batch], y[start : start + metric_batch]
        counts.append(xx.shape[0])
        lvals.append(checkpoint(lambda a, z: lpips_model(a, z), xx, yy, use_reentrant=False))
        if dists_model is not None:
            # DISTS consumes [0, 1], whereas LPIPS consumes [-1, 1].
            dvals.append(
                checkpoint(
                    lambda a, z: dists_model(a, z, require_grad=True, batch_average=True),
                    (xx + 1) / 2,
                    (yy + 1) / 2,
                    use_reentrant=False,
                )
            )
    weights = torch.tensor(counts, device=pred.device, dtype=pred.dtype)
    lp = (torch.stack([v.mean() for v in lvals]) * weights).sum() / weights.sum().clamp_min(1)
    ds = (
        (
            (torch.stack([v.mean() for v in dvals]) * weights[: len(dvals)]).sum()
            / weights[: len(dvals)].sum().clamp_min(1)
        )
        if dvals
        else pred.new_zeros(())
    )
    return lp, ds


def _losses(restored, target, clean, source, aux, lpips_model, dists_model, metric_batch):
    if restored.ndim != 5 or target.ndim != 5 or restored.shape != target.shape or restored.shape[1] != 3:
        raise ValueError(f"RGB decode/target shape mismatch: {tuple(restored.shape)} vs {tuple(target.shape)}")
    restored = restored.float()
    lp, dists = _rgb_metrics(restored, target, lpips_model, dists_model, metric_batch)
    l1 = F.l1_loss(restored.float(), target.float().detach())
    temporal = (
        F.l1_loss(restored[:, :, 1:] - restored[:, :, :-1], target[:, :, 1:].detach() - target[:, :, :-1].detach())
        if restored.shape[2] > 1
        else restored.new_zeros(())
    )
    latent = F.smooth_l1_loss(aux["output"].float(), clean.float().detach())
    rhat = aux["rhat"].float().clamp_min(1e-8)
    error_sq = (clean.float() - source.float()).square().mean(1, keepdim=True)
    target_rms = F.adaptive_avg_pool3d(error_sq, rhat.shape[2:]).sqrt().clamp_min(1e-8)
    rms_log = F.smooth_l1_loss(rhat.log(), target_rms.log().detach())
    # This is a bounded trust penalty on the predicted correction itself.  It
    # does not invent an optimal tau and does not use gain=rhat as a target.
    tau_delta = aux.get("tau_delta", rhat.new_zeros(()))
    tau_trust = (tau_delta.float() / 100.0).square().mean()
    total = 0.1 * l1 + lp + dists + 0.05 * temporal + 0.1 * latent + 0.1 * rms_log + 0.001 * tau_trust
    return total, {
        "rgb_l1": l1,
        "lpips": lp,
        "dists": dists,
        "temporal": temporal,
        "latent": latent,
        "rms_log": rms_log,
        "tau_trust": tau_trust,
    }


def frame_mean_psnr(target: Tensor, reconstruction: Tensor) -> float:
    mse = (target - reconstruction).square().mean(dim=(0, 2, 3))
    return float((-10.0 * torch.log10(mse.clamp_min(1.0e-12))).mean())


def metric_lpips(
    target: Tensor,
    reconstruction: Tensor,
    metric: Any,
    *,
    device: torch.device,
    batch_size: int,
) -> float:
    values: list[float] = []
    with torch.inference_mode():
        for start in range(0, target.shape[1], batch_size):
            stop = min(target.shape[1], start + batch_size)
            left = target[:, start:stop].permute(1, 0, 2, 3).to(device) * 2.0 - 1.0
            right = reconstruction[:, start:stop].permute(1, 0, 2, 3).to(device) * 2.0 - 1.0
            values.extend(float(value) for value in metric(left, right).reshape(-1).cpu())
    return sum(values) / len(values)


def metric_dists(
    target: Tensor,
    reconstruction: Tensor,
    metric: DISTSMetric,
    *,
    device: torch.device,
    batch_size: int,
) -> float:
    total = 0.0
    count = 0
    with torch.inference_mode():
        for start in range(0, target.shape[1], batch_size):
            stop = min(target.shape[1], start + batch_size)
            left = target[:, start:stop].permute(1, 0, 2, 3).to(device)
            right = reconstruction[:, start:stop].permute(1, 0, 2, 3).to(device)
            amount = stop - start
            total += float(metric(left, right).cpu()) * amount
            count += amount
    return total / count


def gop_groups(frame_count_value: int, gop_frames: int) -> tuple[tuple[int, int], ...]:
    if gop_frames < 5 or (gop_frames - 1) % 4:
        raise ValueError("gop-frames must be 1+4*t with t>=1")
    if frame_count_value < gop_frames:
        raise ValueError("frames must be at least gop-frames")
    return tuple(
        (start, min(start + gop_frames, frame_count_value)) for start in range(0, frame_count_value, gop_frames)
    )


def load_source(path: Path, dataset: str, fps: int, frames: int) -> torch.Tensor:
    if dataset not in DATASET_CONTRACT:
        raise ValueError(f"unknown dataset {dataset}")
    _, height, width = DATASET_CONTRACT[dataset]
    match = re.search(r"_(\d+)(?:fps)?(?:_420.*)?\.yuv$", path.name)
    if match is None:
        raise ValueError(f"cannot infer source FPS: {path}")
    expected_fps = int(match.group(1))
    if int(fps) != expected_fps or geometry_from_filename(path) != (height, width):
        raise ValueError(f"{dataset} expects {width}x{height}@{expected_fps}: {path}")
    if frame_count(path, height, width) < frames:
        raise ValueError(f"{path} has fewer than {frames} frames")
    y, u, v = read_yuv420p8(path, max_frames=frames, height=height, width=width)
    result = yuv420p8_to_rgb_bt709(y, u, v).contiguous()
    if result.shape[1] != frames:
        raise RuntimeError(f"expected {frames} source frames, got {result.shape[1]}")
    return result


def pad_spatial(video: torch.Tensor) -> tuple[torch.Tensor, int, int]:
    height, width = int(video.shape[-2]), int(video.shape[-1])
    padded_height, padded_width = ((height + 15) // 16) * 16, ((width + 15) // 16) * 16
    if (height, width) == (padded_height, padded_width):
        return video, height, width
    return F.pad(video, (0, padded_width - width, 0, padded_height - height), mode="reflect"), height, width


def rgb_metrics(
    target: torch.Tensor,
    restored: torch.Tensor,
    lpips_metric: Any,
    dists_metric: Any,
    device: torch.device,
    batch_size: int,
) -> dict[str, float]:

    if target.shape != restored.shape:
        raise RuntimeError(f"RGB shape mismatch {target.shape} != {restored.shape}")
    delta_t, delta_r = target[:, 1:] - target[:, :-1], restored[:, 1:] - restored[:, :-1]
    mse = (target.float() - restored.float()).square().mean().clamp_min(1e-12)
    return {
        "psnr": frame_mean_psnr(target, restored),
        "rgb_mse": float(mse),
        "lpips_vgg": float(metric_lpips(target, restored, lpips_metric, device=device, batch_size=batch_size)),
        "dists": float(metric_dists(target, restored, dists_metric, device=device, batch_size=batch_size)),
        "temporal_delta_l1": float((delta_t - delta_r).abs().mean()),
    }


def json_safe(value: Any) -> Any:
    if torch.is_tensor(value):
        value = value.detach().float().cpu()
        if value.numel() == 1:
            return float(value.item())
        return {
            "shape": list(value.shape),
            "mean": float(value.mean()),
            "std": float(value.std(unbiased=False)),
            "min": float(value.min()),
            "max": float(value.max()),
        }
    if isinstance(value, dict):
        return {str(k): json_safe(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [json_safe(v) for v in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def normalize_codec_state(state: Mapping[str, Any]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in state.items():
        normalized = str(key).removeprefix("module.")
        if normalized in result:
            raise RuntimeError(f"codec state collision at {normalized}")
        result[normalized] = value
    return result


def q_summary(codec: Any, qp: int) -> tuple[float, float]:
    qe = codec.p_codec._get_quant_step(codec.p_codec.qe_log_inter, int(qp))
    return float(torch.log(qe.float().clamp_min(1e-8)).mean().cpu()), 0.0
