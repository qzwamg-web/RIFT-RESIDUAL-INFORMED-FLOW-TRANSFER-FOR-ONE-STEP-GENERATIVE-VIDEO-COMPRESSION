#!/usr/bin/env python3
"""Native42 evaluation of a fine-tuned headless bundle, same endpoint as training."""

import argparse
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ.setdefault("TORCH_HOME", "/qzwang/data/raqvc_artifacts/C512_FINAL_v1/runtime_assets/torch_cache")
from scripts import evaluate as paper
from raqvc import rd_analysis as baseline
from raqvc.training import rift_headless as runtime


def identity(job, a):
    d = paper.make_identity(job, a)
    d.update(
        endpoint=f"{runtime.__name__}.endpoint",
        model_schema=runtime.SCHEMA,
        connection_heads=False,
        training_variant="codec+rank8LoRA+inherited_context_adapters; all connection heads removed",
    )
    return d


class Evaluator(paper.Evaluator):
    def __init__(self, a):
        import torch
        from raqvc.backbone.vae import load_wan_vae
        from raqvc.metrics.perceptual_video import NativeMetricEngine
        from raqvc.training.c512 import DEFAULT_WAN

        self.device = torch.device(a.device)
        self.model, self.codec, self.meta = runtime.load_bundle(a.checkpoint, self.device)
        if self.meta.get("schema") != runtime.SCHEMA:
            raise ValueError("not a headless bundle")
        self.vae = load_wan_vae(DEFAULT_WAN / "vae.pth", device=str(self.device), dtype=torch.bfloat16)
        self.vae.model.eval().requires_grad_(False)
        self.names = paper.metric_names(a)
        self.a = a
        self.rgb = NativeMetricEngine(str(self.device), self.names, a.metric_batch)
        self.fid = paper.PatchFID(str(self.device), a.fid_batch)
        paper.atomic_json(
            a.output / f"METRIC_RUNTIME_shard{a.shard_index}.json",
            dict(rgb=self.rgb.metadata, fid=self.fid.metadata, model=self.meta),
        )

    def restore(self, c):
        return runtime.endpoint(self.model, c, audit=True)


def aggregate(a, jobs):
    import numpy as np
    import csv

    allrows = []
    curves = {}
    bd = {}
    for dataset in ["All42", "HEVC_B", "UVG", "MCL-JCV"]:
        subset = jobs if dataset == "All42" else [j for j in jobs if j["dataset"] == dataset]
        curves[dataset] = {}
        for arm in ["C512", "headless_finetuned"]:
            curve = []
            for qp in a.qps:
                docs = []
                for job in subset:
                    if arm == "C512":
                        d, p = baseline.existing_full(job, qp)
                    else:
                        p = paper.result_path(a.output, job, qp)
                        d = paper.read_complete(p, identity(job, a), qp)
                    docs.append(d)
                    if dataset == "All42":
                        allrows.append(
                            dict(
                                arm=arm,
                                dataset=job["dataset"],
                                video=Path(job["video"]).stem,
                                qp=qp,
                                source=str(p),
                                **{
                                    k: d["metrics"][k]
                                    for k in ["actual_bpp", "psnr", "lpips_vgg", "dists", "lpips_alex"]
                                },
                            )
                        )
                curve.append(
                    dict(
                        qp=qp,
                        videos=len(docs),
                        **{
                            k: float(np.mean([d["metrics"][k] for d in docs]))
                            for k in ["actual_bpp", "psnr", "lpips_vgg", "dists", "lpips_alex"]
                        },
                    )
                )
            curves[dataset][arm] = curve
        bd[dataset] = {
            m: baseline.bd_rate(curves[dataset]["C512"], curves[dataset]["headless_finetuned"], m)
            for m in ["lpips_vgg", "dists"]
        }
    paper.atomic_json(
        a.output / "SUMMARY.json",
        dict(
            complete=True,
            curves=curves,
            bd_rate=bd,
            baseline="C512",
            scope="C512 warm-started headless joint fine-tuning; not a from-scratch matched-budget architecture comparison",
            aggregation="equal-video mean RD curves; common-quality PCHIP; no extrapolation",
            checkpoint=str(a.checkpoint),
        ),
    )
    with (a.output / "VIDEO_METRICS.csv").open("w") as f:
        w = csv.DictWriter(f, fieldnames=list(allrows[0]))
        w.writeheader()
        w.writerows(allrows)
    tex = [
        r"\begin{table}[t]",
        r"\centering\small",
        r"\caption{Head-free joint fine-tuning on 42 native1080p videos. BD-rate is relative to C512; positive values indicate a rate penalty.}",
        r"\label{tab:rift_headless_finetuned}",
        r"\begin{tabular}{lrr}",
        r"\toprule",
        r"Dataset & BD-rate LPIPS (\%) & BD-rate DISTS (\%) \\",
        r"\midrule",
    ]
    for dataset in bd:
        vals = [bd[dataset][m]["value"] for m in ["lpips_vgg", "dists"]]
        tex.append(
            dataset.replace("_", r"\_") + " & " + " & ".join("--" if v is None else f"{v:+.2f}" for v in vals) + r" \\"
        )
    tex += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    (a.output / "ABLATION_TABLE.tex").write_text("\n".join(tex) + "\n")
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axs = plt.subplots(2, 4, figsize=(15, 7), layout="constrained")
    for col, dataset in enumerate(curves):
        for row, metric in enumerate(["lpips_vgg", "dists"]):
            ax = axs[row, col]
            for arm, color, marker in [("C512", "#005A9C", "o"), ("headless_finetuned", "#C43C39", "s")]:
                points = sorted(curves[dataset][arm], key=lambda p: p["actual_bpp"])
                ax.plot(
                    [p["actual_bpp"] for p in points],
                    [p[metric] for p in points],
                    label=arm,
                    color=color,
                    marker=marker,
                )
            ax.set_title(dataset)
            ax.set_xlabel("Actual bpp")
            ax.set_ylabel(metric)
            ax.grid(alpha=0.2)
    axs[0, 0].legend(fontsize=8)
    fig.savefig(a.output / "HEADLESS_FINETUNED_RD.pdf")
    plt.close(fig)


def main():
    import torch

    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--shard-index", type=int, default=0)
    p.add_argument("--num-shards", type=int, default=6)
    p.add_argument("--smoke", action="store_true")
    p.add_argument("--smoke-frames", type=int, choices=[21, 25], default=21)
    p.add_argument("--aggregate-only", action="store_true")
    a = p.parse_args()
    torch.set_num_threads(4)
    if not 0 <= a.shard_index < a.num_shards:
        raise ValueError("invalid shard")
    a.manifest = paper.DEFAULT_MANIFEST
    a.resolutions = [1080]
    a.frames = 96
    a.native_tail = True
    a.qps = [63] if a.smoke else [0, 16, 39, 63]
    a.device = "cuda:0"
    a.fid_batch = 32
    a.metric_batch = 1
    a.flolpips = False
    a.save_videos = False
    a.save_all_videos = True
    a.save_lossless_qps = [63]
    jobs = paper.build_jobs(a)
    if not a.smoke and len(jobs) != 42:
        raise ValueError("requires exactly42 videos")
    plan = dict(
        checkpoint=paper.file_identity(a.checkpoint), jobs=jobs, qps=a.qps, identities=[identity(j, a) for j in jobs]
    )
    paper.ensure_plan(a.output, plan)
    if a.aggregate_only:
        if a.smoke:
            raise ValueError("do not aggregate smoke as formal")
        return aggregate(a, jobs)
    selected = jobs if a.smoke else paper.shard_jobs(jobs, a.num_shards)[a.shard_index]
    evaluator = None
    completed = 0
    for job in selected:
        ident = identity(job, a)
        todo = []
        for qp in a.qps:
            path = paper.result_path(a.output, job, qp)
            if path.exists():
                paper.read_complete(path, ident, qp)
                completed += 1
            else:
                todo.append(qp)
        if todo:
            if evaluator is None:
                evaluator = Evaluator(a)
            docs = evaluator.evaluate(job, ident, todo)
            completed += len(docs)
        paper.atomic_json(
            a.output / f"PROGRESS_shard{a.shard_index}.json",
            dict(completed=completed, expected=len(selected) * len(a.qps), last_video=job["video"]),
        )
    paper.atomic_json(a.output / f"WORKER_{a.shard_index}.json", dict(complete=True, points=completed, smoke=a.smoke))


if __name__ == "__main__":
    main()
