#!/usr/bin/env python3
"""C512: one command for preflight, smoke, training, evaluation and timing."""

import argparse
import datetime
import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
ART = Path(os.environ.get("RAQVC_ARTIFACTS", "/qzwang/data/raqvc_artifacts"))
FINAL = ART / "C512_FINAL_v1"
DEFAULT = FINAL / "model/C512_original_bundle.pt"


def run(command, env=None):
    subprocess.run([sys.executable, "-u", *map(str, command)], cwd=ROOT, env=env, check=True)


def preflight():
    paths = [
        DEFAULT,
        FINAL / "model/stage3_dit_reproduction.pt",
        FINAL / "model/wan/vae.pth",
        FINAL / "model/wan/low_noise_model/diffusion_pytorch_model.bin",
        FINAL / "model/codec_initializers/cvpr2025_image.pth.tar",
        FINAL / "model/codec_initializers/cvpr2025_video.pth.tar",
    ]
    for resolution in (256, 512):
        index = FINAL / f"datasets/{resolution}/index.json"
        paths.append(index)
        data = json.loads(index.read_text())
        paths += [Path(p) for p in {row["shard"] for row in data["rows"]}]
        paths.append(Path(data["source_manifest"]))
    for row in json.loads((ROOT / "configs/evaluation.json").read_text())["jobs"]:
        paths.append(Path(row["video"]))
    missing = [str(p) for p in paths if not p.is_file()]
    if missing:
        raise FileNotFoundError("\n".join(missing))
    from raqvc.metrics.flolpips import prepare_flolpips_assets

    prepare_flolpips_assets(download=False)
    print(json.dumps(dict(preflight=True, checked_files=len(paths), checkpoint=str(DEFAULT))))


def evaluate(a, checkpoint, output):
    devices = a.devices.split(",")
    common = [
        "scripts/evaluate.py",
        "--checkpoint",
        checkpoint,
        "--output",
        output,
        "--manifest",
        ROOT / "configs/evaluation.json",
        "--resolutions",
        *a.resolutions,
        "--frames",
        96,
        "--flolpips",
        "--save-all-videos",
        "--balance-pixels",
        "--num-shards",
        len(devices),
    ]
    output.mkdir(parents=True, exist_ok=True)
    children = []
    try:
        for i, device in enumerate(devices):
            env = os.environ.copy()
            env["CUDA_VISIBLE_DEVICES"] = device
            log = (output / f"worker{i}.log").open("a")
            p = subprocess.Popen(
                [sys.executable, "-u", *map(str, common), "--shard-index", str(i)],
                cwd=ROOT,
                env=env,
                stdout=log,
                stderr=subprocess.STDOUT,
            )
            children.append((i, p, log))
        errors = []
        for i, p, log in children:
            code = p.wait()
            log.close()
            if code:
                errors.append((i, code))
        if errors:
            raise RuntimeError(f"Workers failed: {errors}; inspect worker logs; rerun to resume completed QPs")
    finally:
        for _, p, log in children:
            if p.poll() is None:
                p.terminate()
                p.wait()
            if not log.closed:
                log.close()
    run([*common, "--aggregate-only"])


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("action", choices=["preflight", "smoke", "train", "resume", "evaluate", "speed"])
    p.add_argument("--checkpoint", type=Path, default=DEFAULT)
    p.add_argument("--output", type=Path)
    p.add_argument("--devices", default="0")
    p.add_argument("--steps", type=int, default=512)
    p.add_argument("--batch", type=int, default=1, help="Per-GPU batch size")
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--resolutions", type=int, nargs="+", choices=[720, 1080], default=[720, 1080])
    p.add_argument("--evaluate-after", action="store_true")
    a = p.parse_args()
    os.environ["PYTHONPATH"] = str(ROOT) + os.pathsep + os.environ.get("PYTHONPATH", "")
    os.environ.setdefault("TORCH_HOME", str(FINAL / "runtime_assets/torch_cache"))
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    for key in ["OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"]:
        os.environ.setdefault(key, "4")
    if a.action == "preflight":
        preflight()
        return
    if a.action == "resume" and a.output is None:
        p.error("resume requires the original --output")
    output = (
        a.output or FINAL / "new_runs" / f"{a.action}_{datetime.datetime.now(datetime.timezone.utc):%Y%m%dT%H%M%S%fZ}"
    )
    if a.action in ["smoke", "train", "resume"]:
        preflight()
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = a.devices
        run(
            [
                "-m",
                "torch.distributed.run",
                "--standalone",
                f'--nproc_per_node={len(a.devices.split(","))}',
                "scripts/train.py",
                "--phase",
                "C",
                "--checkpoint",
                a.checkpoint,
                "--output",
                output,
                "--steps",
                a.steps,
                "--batch",
                a.batch,
                "--workers",
                a.workers,
                *(["--smoke"] if a.action == "smoke" else []),
                *(["--resume"] if a.action == "resume" else []),
            ],
            env=env,
        )
        if a.evaluate_after:
            result = json.loads((output / "RESULT.json").read_text())
            ckpt = Path(result.get("final_checkpoint", output / f"step_{a.steps:06d}.pt"))
            evaluate(a, ckpt, output / "evaluation")
    elif a.action == "evaluate":
        evaluate(a, a.checkpoint, output)
    elif a.action == "speed":
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = a.devices.split(",")[0]
        run(
            ["scripts/benchmark.py", "--checkpoint", a.checkpoint, "--output", output, "--resolutions", *a.resolutions],
            env=env,
        )


if __name__ == "__main__":
    main()
