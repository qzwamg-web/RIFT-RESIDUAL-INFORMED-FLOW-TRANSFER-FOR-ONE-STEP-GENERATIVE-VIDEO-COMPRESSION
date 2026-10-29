"""H2 replay under the published crop/frame/patch-FID protocol; no training.

Full 42-video runs resume at completed QPs and finish with dataset-separated
plots. Inference uses the exact shared H2 load_bundle / endpoint / hard codec.
"""

from __future__ import annotations
import argparse
import contextlib
import csv
import fcntl
import json
import math
import os
import sys
import time
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from raqvc.metrics.protocol import (
    PatchFID,
    coding_crop,
    metric_crop,
    prepare_gop,
    patch_coordinates,
    compute_pooled_fid,
)
from raqvc.evaluation_io import SourceReader, padded_shape, atomic_npz
from raqvc.evaluation_io import atomic_json, file_identity, open_video_writer, append_video, TAU_TABLE, QPS

SCHEMA = "task938_h2_paper_protocol_v2_crop736_reflect1080"
ART = Path("/qzwang/data/raqvc_artifacts")
DEFAULT_MANIFEST = Path(__file__).resolve().parents[1] / "configs/evaluation.json"
METRICS = ["psnr", "lpips_vgg", "dists", "lpips_alex"]


def ensure_plan(output, plan):
    """Serialize first creation; completed workers never rewrite the shared plan."""
    output.mkdir(parents=True, exist_ok=True)
    path = output / "PLAN.json"
    with (output / "PLAN.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        try:
            existing = json.loads(path.read_text())
        except FileNotFoundError:
            atomic_json(path, plan)
        else:
            if existing != plan:
                raise RuntimeError("output belongs to another evaluation protocol")


def metric_names(a):
    return METRICS + (["flolpips"] if getattr(a, "flolpips", False) else [])


def shard_jobs(jobs, count, pixel_weighted=False):
    """Balance full videos by frame count without splitting temporal pairs."""
    if count < 1:
        raise ValueError("shard count must be positive")
    shards = [[] for _ in range(count)]
    loads = [0] * count

    def cost(job):
        pixels = (736 * 1280 if job["resolution"] == 720 else 1088 * 1920) if pixel_weighted else 1
        return job["frames"] * pixels

    for job in sorted(jobs, key=lambda j: (-cost(j), j["resolution"], j["dataset"], j["video"])):
        index = min(range(count), key=lambda i: (loads[i], i))
        shards[index].append(job)
        loads[index] += cost(job)
    return shards


def log(event, **kwargs):
    print(json.dumps(dict(event=event, time_utc=time.strftime("%FT%TZ", time.gmtime()), **kwargs)), flush=True)


class PaperReader(SourceReader):

    def read(self, start, stop, resolution):
        return coding_crop(super().read(start, stop, 1080), resolution)


def build_jobs(a):
    from raqvc.data.raw_yuv import frame_count

    original = json.loads(a.manifest.read_text())["jobs"]
    jobs = []
    for resolution in a.resolutions:
        for job in original:
            available = frame_count(job["video"], 1080, 1920)
            requested = getattr(a, "frames", None)
            frames = requested if requested is not None else available if resolution == 720 else 96
            if frames < 1 or available < frames:
                raise ValueError(f"insufficient source frames: {job['video']}: {available} < {frames}")
            if a.smoke:
                if not Path(job["video"]).name.startswith("BQTerrace_"):
                    continue
                frames = getattr(a, "smoke_frames", 25)
                if frames > available:
                    raise ValueError("insufficient smoke frames")
            reader = SourceReader(Path(job["video"]), job["dataset"], job["fps"], frames)
            del reader
            jobs.append(dict(**job, resolution=resolution, frames=frames, available_frames=available))
    order = {"HEVC_B": 0, "UVG": 1, "MCL-JCV": 2}
    return sorted(jobs, key=lambda j: (-j["resolution"], order[j["dataset"]], j["video"]))


def make_identity(job, a):
    identity = dict(
        schema=SCHEMA,
        checkpoint=file_identity(a.checkpoint),
        source=file_identity(Path(job["video"])),
        job=job,
        metrics=metric_names(a),
        source_transform="center_crop_736x1280_no_resampling" if job["resolution"] == 720 else "native",
        metric_crop="rows8:728_cols0:1280" if job["resolution"] == 720 else "rows0:1080_cols0:1920",
        coding_resolution=[736, 1280] if job["resolution"] == 720 else [1088, 1920],
        frame_protocol=(
            f"smoke{job['frames']}"
            if a.smoke
            else (
                f"first{a.frames}"
                if getattr(a, "frames", None) is not None
                else "all_available" if job["resolution"] == 720 else "first96"
            )
        ),
        mcl_frame_policy=(
            "first96_user_aligned_across_datasets"
            if getattr(a, "frames", None) == 96
            else "720_full_sequence_assumption; paper_not_explicit"
        ),
        rgb_metric_range="native_float32_0_1",
        color="BT709_full_range_nearest_chroma",
        padding="none_real_source_crop" if job["resolution"] == 720 else "RGB_reflect_right_bottom_multiple32",
        tail="native_1plus4k_no_temporal_padding",
        gop=25,
        tau_table=TAU_TABLE,
        velocity_bridge="unit",
        noise_alpha=0.0,
        endpoint="raqvc.training.c512.endpoint",
        fid="torchmetrics1.8.2_torch_fidelity0.3.0_2048_patch256_dual_grid_float32_to_uint8",
        fid_batch=a.fid_batch,
        metric_batch=a.metric_batch,
        bpp_primary="requested_frames*padded_height*padded_width",
        bpp_secondary="requested_frames*original_height*original_width",
    )
    if getattr(a, "save_all_videos", False) or getattr(a, "save_lossless_qps", []):
        identity["saved_artifacts"] = dict(
            all_video_previews=bool(getattr(a, "save_all_videos", False)),
            lossless_qps=list(getattr(a, "save_lossless_qps", [])),
            lossless_format="PNG_RGB_uint8_round",
        )
    return identity


def result_path(out, job, qp):
    return (
        out / "metrics" / str(job["resolution"]) / job["dataset"] / Path(job["video"]).stem / f"qp{qp}" / "RESULT.json"
    )


def read_complete(path, identity, qp):
    import numpy as np

    d = json.loads(path.read_text())
    if d.get("complete") is not True or d.get("identity") != identity or d.get("qp") != qp:
        raise RuntimeError(f"incompatible/incomplete result: {path}")
    if identity.get("tail") == "native_1plus4k_no_temporal_padding":
        frames = identity["job"]["frames"]
        expected_groups = [(i, min(i + 25, frames)) for i in range(0, frames, 25)]
        stats = d.get("gop_stats", [])
        if (
            d.get("encoded_frames") != frames
            or d.get("evaluated_frames") != frames
            or [tuple(g["frames"]) for g in stats] != expected_groups
            or any(
                (
                    g.get("encoded_rgb_frames") != b - a
                    or g.get("latent_frames") != 1 + (b - a - 1) // 4
                    or len(g["frame_bits_actual"]) != g.get("latent_frames")
                    for (g, (a, b)) in zip(stats, expected_groups)
                )
            )
        ):
            raise RuntimeError(f"invalid native GOP accounting: {path}")
        bits = sum((sum(g["frame_bits_actual"]) for g in stats))
        if bits != d["actual_bits"] or d["rate_pixels"] != frames * math.prod(identity["coding_resolution"]):
            raise RuntimeError(f"invalid native rate accounting: {path}")
    artifacts = identity.get("saved_artifacts", {})
    if artifacts.get("all_video_previews") and (not Path(d.get("video") or "/missing_video").is_file()):
        raise RuntimeError(f"missing video preview: {path}")
    if qp in artifacts.get("lossless_qps", []):
        folder = Path(d.get("lossless_frames_dir") or "/missing_frames")
        if not all(((folder / f"{i:06d}.png").is_file() for i in range(identity["job"]["frames"]))):
            raise RuntimeError(f"missing lossless frames: {path}")
    count = identity["job"]["frames"] * len(patch_coordinates(*d["rgb_resolution"]))
    with np.load(d["fid_state"]) as f:
        for side in ["real", "fake"]:
            if int(f[f"{side}_features_num_samples"]) != count:
                raise RuntimeError(f"wrong FID patch count: {path}")
            if f[f"{side}_features_cov_sum"].shape != (2048, 2048):
                raise RuntimeError(f"wrong FID covariance: {path}")
        for key in f.files:
            if not np.isfinite(f[key]).all():
                raise RuntimeError(f"nonfinite FID state: {path}")
    if any((not math.isfinite(d["metrics"][k]) for k in METRICS + ["actual_bpp", "actual_bpp_original"])):
        raise RuntimeError(f"nonfinite metric: {path}")
    if "flolpips" in identity["metrics"]:
        m = d["metrics"]
        pairs = identity["job"]["frames"] - 1
        if m.get("flolpips_pairs") != pairs:
            raise RuntimeError(f"missing temporal pairs: {path}")
        value = m.get("flolpips")
        undefined = m.get("flolpips_undefined_pairs", 0)
        if not 0 <= undefined <= pairs:
            raise RuntimeError(f"invalid undefined FloLPIPS count: {path}")
        if value is None:
            if not undefined:
                raise RuntimeError(f"unexplained null FloLPIPS: {path}")
        elif not math.isfinite(value) or undefined:
            raise RuntimeError(f"inconsistent FloLPIPS: {path}")
    return d


class Evaluator:

    def restore(self, condition):
        """Shared production endpoint; research evaluators may override this seam."""
        from raqvc.training.c512 import endpoint

        return endpoint(self.model, self.heads, condition)

    def __init__(self, a):
        import torch
        from raqvc.training.c512 import load_bundle
        from raqvc.backbone.vae import load_wan_vae
        from raqvc.metrics.perceptual_video import NativeMetricEngine
        from raqvc.evaluation_io import WAN
        from raqvc.final_support import json_safe

        self.device = torch.device(a.device)
        self.model, self.heads, self.codec, self.meta = load_bundle(a.checkpoint, self.device)
        for module in [self.model, self.heads, self.codec]:
            module.eval().requires_grad_(False)
        self.vae = load_wan_vae(WAN / "vae.pth", device=str(self.device), dtype=torch.bfloat16)
        self.vae.model.eval().requires_grad_(False)
        if (
            self.meta["tau_table"] != TAU_TABLE
            or self.meta["velocity_scale_mode"] != "unit"
            or self.meta["noise_alpha"] != 0
        ):
            raise RuntimeError("H2 inference contract mismatch")
        self.names = metric_names(a)
        self.rgb = NativeMetricEngine(str(self.device), self.names, a.metric_batch)
        self.fid = PatchFID(str(self.device), a.fid_batch)
        self.rgb.metadata["fid"] = "computed_separately_by_PatchFID; see fid metadata"
        self.a = a
        runtime = (
            "METRIC_RUNTIME.json"
            if getattr(a, "shard_index", None) is None
            else f"METRIC_RUNTIME_shard{a.shard_index}.json"
        )
        atomic_json(a.output / runtime, dict(fid=self.fid.metadata, rgb=self.rgb.metadata, meta=json_safe(self.meta)))

    @staticmethod
    def groups(frames):
        return [(i, min(i + 25, frames)) for i in range(0, frames, 25)]

    def evaluate(self, job, identity, qps):
        import numpy as np
        import torch
        from raqvc.backbone.vae import encode_video, decode_latent
        from raqvc.training.c512 import endpoint
        from raqvc.training.codec_training import online_reconstruction
        from raqvc.training.conditions import build_condition
        from raqvc.metrics.perceptual_video import MetricStream
        from raqvc.final_support import json_safe

        a = self.a
        reader = PaperReader(Path(job["video"]), job["dataset"], job["fps"], job["frames"])
        groups = self.groups(job["frames"])
        if any(((stop - start - 1) % 4 for (start, stop) in groups)):
            raise ValueError("native temporal GOPs must contain 1+4K real frames")
        clean_groups = []
        self.fid.reset()
        for gi, (start, stop) in enumerate(groups):
            coding_rgb = reader.read(start, stop, job["resolution"])
            target = metric_crop(coding_rgb, job["resolution"])
            with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
                clean = encode_video(self.vae, prepare_gop(coding_rgb, frames=stop - start).to(self.device) * 2 - 1)
            if clean.shape[1] != 1 + (stop - start - 1) // 4:
                raise RuntimeError(f"VAE temporal shape mismatch: {clean.shape}")
            clean_groups.append(clean.float().cpu())
            del clean
            self.fid.update(target, real=True)
            del target, coding_rgb
            log(
                "reference_gop",
                video=Path(job["video"]).stem,
                resolution=job["resolution"],
                gop=gi + 1,
                total=len(groups),
            )
        real = self.fid.state()
        h, w = {720: (720, 1280), 1080: (1080, 1920)}[job["resolution"]]
        ph, pw = padded_shape(h, w)
        docs = []
        for qp in qps:
            started = time.monotonic()
            dest = result_path(a.output, job, qp).parent
            dest.mkdir(parents=True, exist_ok=True)
            self.fid.reset(real)
            stream = MetricStream(self.rgb, self.names)
            bits = 0
            stats = []
            save = getattr(a, "save_all_videos", False) or (
                a.save_videos
                and any((Path(job["video"]).name.startswith(s + "_") for s in ["BQTerrace", "Beauty", "videoSRC01"]))
            )
            lossless = qp in getattr(a, "save_lossless_qps", [])
            frame_dir = dest / "frames_lossless"
            if lossless:
                frame_dir.mkdir(exist_ok=True)
            video = dest / "reconstruction.mp4"
            temp = dest / "reconstruction.inprogress.mp4"
            with contextlib.ExitStack() as stack:
                writer = stack.enter_context(open_video_writer(temp, job["fps"])) if save else None
                for gi, ((start, stop), clean_cpu) in enumerate(zip(groups, clean_groups)):
                    target = metric_crop(reader.read(start, stop, job["resolution"]), job["resolution"])
                    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
                        clean = clean_cpu.to(self.device).unsqueeze(0)
                        source, _, actual, sigma, width, metadata = online_reconstruction(
                            self.codec, clean, qp, verify=True, return_metadata=True
                        )
                        fb = metadata["frame_bits_actual"]
                        if not fb or any((int(b) <= 0 for b in fb)):
                            raise RuntimeError("invalid actual rANS bits")
                        bits += sum(map(int, fb))
                        condition = build_condition(self.codec, source, sigma, width, actual, qp, self.device)
                        restored, aux = self.restore(condition)
                        stats.append(
                            dict(
                                group=gi,
                                frames=[start, stop],
                                encoded_rgb_frames=stop - start,
                                latent_frames=int(source.shape[2]),
                                frame_bits_actual=fb,
                                aux=json_safe(aux),
                            )
                        )
                        del clean, source, actual, sigma, width, metadata, condition, aux
                        decoded = ((decode_latent(self.vae, restored[0].float()) + 1) * 0.5).clamp(0, 1)
                        if decoded.shape[1] != stop - start:
                            raise RuntimeError(f"decoded temporal shape mismatch: {decoded.shape}")
                        value = metric_crop(decoded[:, : stop - start], job["resolution"]).float().cpu()
                        del decoded, restored
                    if value.shape != target.shape:
                        raise RuntimeError("cropped RGB shape mismatch")
                    stream.add(target, value)
                    self.fid.update(value, real=False)
                    if writer is not None:
                        append_video(writer, value)
                    if lossless:
                        from PIL import Image

                        for fi, frame in enumerate(value.unbind(1), start=start):
                            pixels = (frame.clamp(0, 1) * 255).round().to(torch.uint8).permute(1, 2, 0).numpy()
                            Image.fromarray(pixels).save(frame_dir / f"{fi:06d}.png", compress_level=1)
                    del target, value
                    log(
                        "gop_complete",
                        video=Path(job["video"]).stem,
                        resolution=job["resolution"],
                        qp=qp,
                        gop=gi + 1,
                        total=len(groups),
                        seconds=time.monotonic() - started,
                    )
                if stream.frames != job["frames"]:
                    raise RuntimeError("wrong scored frame count")
            if save:
                os.replace(temp, video)
            fidstate = self.fid.state()
            expected = job["frames"] * len(patch_coordinates(h, w))
            assert int(fidstate["real_features_num_samples"]) == int(fidstate["fake_features_num_samples"]) == expected
            moments = dest / "FID_MOMENTS.npz"
            atomic_npz(moments, fidstate)
            atomic_npz(dest / "FRAME_METRICS.npz", stream.arrays("mainline"))
            metrics = dict(
                **stream.result(),
                actual_bpp=bits / (job["frames"] * ph * pw),
                actual_bpp_original=bits / (job["frames"] * h * w),
            )
            doc = dict(
                complete=True,
                identity=identity,
                qp=qp,
                metrics=metrics,
                rgb_resolution=[h, w],
                padded_resolution=[ph, pw],
                actual_bits=bits,
                metric_crop=identity["metric_crop"],
                rate_pixels=job["frames"] * ph * pw,
                evaluated_frames=stream.frames,
                encoded_frames=job["frames"],
                patches=expected,
                fid_state=str(moments.resolve()),
                gop_stats=stats,
                seconds=time.monotonic() - started,
                video=str(video.resolve()) if save else None,
                lossless_frames_dir=str(frame_dir.resolve()) if lossless else None,
            )
            atomic_json(dest / "RESULT.json", doc)
            docs.append(doc)
            log("qp_complete", result=str(dest / "RESULT.json"), metrics=metrics, patches=expected)
        return docs


def aggregate(a, jobs, require_all=True):
    import numpy as np

    buckets = defaultdict(list)
    expected = defaultdict(int)
    for job in jobs:
        for qp in a.qps:
            key = (job["resolution"], job["dataset"], qp)
            expected[key] += 1
            path = result_path(a.output, job, qp)
            if path.exists():
                buckets[key].append(read_complete(path, make_identity(job, a), qp))
    rows = []
    for key, number in expected.items():
        docs = buckets[key]
        if len(docs) != number:
            if require_all:
                raise RuntimeError(f"incomplete dataset {key}: {len(docs)}/{number}")
            continue

        def states():
            for d in docs:
                with np.load(d["fid_state"]) as f:
                    yield {k: f[k] for k in f.files}

        total = None
        for state in states():
            if total is None:
                total = state
            else:
                for name in total:
                    total[name] += state[name]
        fid = compute_pooled_fid([total])
        row = dict(
            resolution=key[0],
            dataset=key[1],
            qp=key[2],
            route="mainline",
            videos=number,
            frame_count=sum((d["evaluated_frames"] for d in docs)),
            fid_patches=int(total["real_features_num_samples"]),
            fid=fid,
            fid_protocol="torchmetrics2048_patch256_dualgrid_dataset_pool",
            **{
                k: float(np.mean([d["metrics"][k] for d in docs]))
                for k in METRICS + ["rgb_mse", "actual_bpp", "actual_bpp_original"]
            },
        )
        if getattr(a, "flolpips", False):
            row.update(aggregate_flo(docs))
        rows.append(row)
        log("dataset_complete", **row)
    atomic_json(
        a.output / "SUMMARY.json",
        dict(
            complete=len(rows) == len(expected),
            dataset_metrics=rows,
            protocol=SCHEMA,
            smoke=a.smoke,
            checkpoint=file_identity(a.checkpoint),
            metrics_aggregation="equal_video_scalar_mean; dataset_pooled_patch_FID; FloLPIPS_all_adjacent_pairs_including_GOP_boundaries",
        ),
    )
    if rows:
        with (a.output / "DATASET_METRICS.csv").open("w") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)
    per_video = []
    for docs in buckets.values():
        for d in docs:
            job = d["identity"]["job"]
            per_video.append(
                dict(
                    resolution=job["resolution"],
                    dataset=job["dataset"],
                    video=Path(job["video"]).stem,
                    qp=d["qp"],
                    frames=d["evaluated_frames"],
                    actual_bits=d["actual_bits"],
                    **d["metrics"],
                )
            )
    if per_video:
        with (a.output / "VIDEO_METRICS.csv").open("w") as f:
            w = csv.DictWriter(f, fieldnames=list(per_video[0]))
            w.writeheader()
            w.writerows(per_video)


def aggregate_flo(docs):
    """Never turn undefined official flow weights into a fabricated zero score."""
    values = [d["metrics"]["flolpips"] for d in docs]
    undefined = sum((d["metrics"]["flolpips_undefined_pairs"] for d in docs))
    pairs = sum((d["metrics"]["flolpips_pairs"] for d in docs))
    finite_pairs = pairs - undefined
    diagnostic = sum(
        (
            (d["metrics"]["flolpips_finite_pair_mean"] or 0.0)
            * (d["metrics"]["flolpips_pairs"] - d["metrics"]["flolpips_undefined_pairs"])
            for d in docs
        )
    )
    return dict(
        flolpips=sum(values) / len(values) if all((v is not None for v in values)) else None,
        flolpips_pairs=pairs,
        flolpips_undefined_pairs=undefined,
        flolpips_defined_videos=sum((v is not None for v in values)),
        flolpips_finite_pair_mean=diagnostic / finite_pairs if finite_pairs else None,
    )


def main():
    from raqvc.training.c512 import DEFAULT_CHECKPOINT
    import torch

    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    p.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    p.add_argument("--resolutions", type=int, nargs="+", choices=[720, 1080], default=[1080, 720])
    p.add_argument("--frames", type=int, default=96, help="Exact first N source frames at every resolution")
    p.add_argument("--qps", type=int, nargs="+", choices=QPS, default=list(QPS))
    p.add_argument("--fid-batch", type=int, default=32)
    p.add_argument("--metric-batch", type=int, default=1)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--smoke", action="store_true")
    p.add_argument("--smoke-frames", type=int, default=25)
    p.add_argument("--preflight", action="store_true")
    p.add_argument("--aggregate-only", action="store_true")
    p.add_argument("--save-videos", action="store_true")
    p.add_argument("--save-all-videos", action="store_true")
    p.add_argument("--save-lossless-qps", type=int, nargs="+", choices=QPS, default=[])
    p.add_argument("--balance-pixels", action="store_true")
    p.add_argument("--flolpips", action="store_true")
    p.add_argument("--num-shards", type=int, default=1)
    p.add_argument("--shard-index", type=int)
    a = p.parse_args()
    torch.set_num_threads(8)
    if a.fid_batch < 1 or a.metric_batch < 1:
        raise ValueError("batch must be positive")
    if len(set(a.qps)) != len(a.qps) or len(set(a.resolutions)) != len(a.resolutions):
        raise ValueError("duplicate jobs")
    if a.num_shards < 1 or (a.shard_index is not None and (not 0 <= a.shard_index < a.num_shards)):
        raise ValueError("invalid shard assignment")
    jobs = build_jobs(a)
    plan = dict(
        schema=SCHEMA,
        jobs=jobs,
        qps=a.qps,
        checkpoint=file_identity(a.checkpoint),
        identities=[make_identity(j, a) for j in jobs],
        smoke=a.smoke,
    )
    if a.preflight:
        print(json.dumps(plan, indent=2))
        return
    ensure_plan(a.output, plan)
    if a.aggregate_only:
        aggregate(a, jobs)
        return
    if a.shard_index is not None:
        jobs = shard_jobs(jobs, a.num_shards, pixel_weighted=a.balance_pixels)[a.shard_index]
    completed = 0
    pending = []
    for job in jobs:
        todo = []
        identity = make_identity(job, a)
        for qp in a.qps:
            path = result_path(a.output, job, qp)
            if path.exists():
                read_complete(path, identity, qp)
                completed += 1
            else:
                todo.append(qp)
        if todo:
            pending.append((job, identity, todo))
    if pending:
        log("loading_models", jobs=len(pending), qps=sum((len(q) for (_, _, q) in pending)))
        evaluator = Evaluator(a)
        for job, identity, todo in pending:
            completed += len(evaluator.evaluate(job, identity, todo))
            progress = "PROGRESS.json" if a.shard_index is None else f"PROGRESS_shard{a.shard_index}.json"
            atomic_json(
                a.output / progress,
                dict(
                    completed_qps=completed,
                    expected_qps=len(jobs) * len(a.qps),
                    last_job=job,
                    time_utc=time.strftime("%FT%TZ", time.gmtime()),
                ),
            )
        del evaluator
        torch.cuda.empty_cache()
    if a.shard_index is None:
        aggregate(a, jobs)
    log("evaluation_complete", output=str(a.output), completed_qps=completed)


if __name__ == "__main__":
    main()
