"""Stored C512 anchor lookup and monotone PCHIP BD-rate."""

import json
from pathlib import Path
from scripts import evaluate as paper

BASELINE = Path("/qzwang/data/raqvc_artifacts/C512_FINAL_v1/results/C512_first96/formal")


def existing_full(job, qp):
    path = paper.result_path(BASELINE, job, qp)
    doc = json.loads(path.read_text())
    if not doc.get("complete") or doc["qp"] != qp or doc["encoded_frames"] != 96:
        raise ValueError(f"incomplete C512 baseline: {path}")
    original = doc["identity"]["job"]
    for key in ("dataset", "video", "resolution", "frames"):
        if original[key] != job[key]:
            raise ValueError(f"baseline job mismatch: {key}")
    return doc, path


def bd_rate(full, variant, metric):
    """PCHIP integral on common quality interval only; reject nonmonotone curves."""
    import numpy as np
    from scipy.interpolate import PchipInterpolator

    def curve(rows):
        ordered = sorted(rows, key=lambda r: r["actual_bpp"])
        rate = np.asarray([r["actual_bpp"] for r in ordered], float)
        quality = np.asarray([r[metric] for r in ordered], float) * (1 if metric == "psnr" else -1)
        if not np.isfinite(rate).all() or not np.isfinite(quality).all() or (rate <= 0).any():
            raise ValueError("nonfinite_or_nonpositive_curve")
        if (np.diff(rate) <= 0).any() or (np.diff(quality) <= 0).any():
            raise ValueError("nonmonotonic_curve")
        return quality, np.log(rate)

    try:
        x, y = curve(full)
        u, v = curve(variant)
        lo = max(x[0], u[0])
        hi = min(x[-1], u[-1])
        if hi <= lo:
            raise ValueError("no_common_quality_interval")
        delta = (PchipInterpolator(u, v).integrate(lo, hi) - PchipInterpolator(x, y).integrate(lo, hi)) / (hi - lo)
        return dict(
            value=float(100 * np.expm1(delta)),
            status="ok",
            quality_interval=[float(lo), float(hi)],
            baseline_quality_coverage=float((hi - lo) / (x[-1] - x[0])),
        )
    except ValueError as e:
        return dict(value=None, status=str(e))
