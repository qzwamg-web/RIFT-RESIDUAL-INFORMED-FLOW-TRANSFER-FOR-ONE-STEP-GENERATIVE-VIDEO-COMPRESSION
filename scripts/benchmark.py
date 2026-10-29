#!/usr/bin/env python3
"""Measure RIFT encode/decode time and opt-in VAE throughput variants."""

from __future__ import annotations
import argparse
import io
import json
import statistics
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import torch
import torch.nn.functional as F
from raqvc.backbone.vae import load_wan_vae, encode_video, decode_latent
from raqvc.backbone.vae_memory import decode_latent_batch
from raqvc.training.c512 import DEFAULT_CHECKPOINT, DEFAULT_WAN, load_bundle, endpoint
from raqvc.training.conditions import build_condition
from raqvc.final_support import load_source


def atomic(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(".tmp")
    temp.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temp.replace(path)


def timed(fn):
    torch.cuda.synchronize()
    start = time.perf_counter()
    result = fn()
    torch.cuda.synchronize()
    return result, time.perf_counter() - start


def source_for(source, height):
    from raqvc.metrics.protocol import coding_crop, prepare_gop

    crop = coding_crop(source, height)
    return prepare_gop(crop, frames=25), height, (1280 if height == 720 else 1920)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    p.add_argument("--repeats", type=int, default=3)
    p.add_argument("--resolutions", type=int, nargs="+", choices=(720, 1080), default=[720, 1080])
    p.add_argument("--qps", type=int, nargs="+", default=[0, 16, 39, 63])
    a = p.parse_args()
    if a.repeats < 1:
        raise ValueError("positive repeats required")
    torch.set_num_threads(4)
    torch.cuda.set_device(0)
    torch.backends.cudnn.benchmark = False
    device = torch.device("cuda:0")
    print(json.dumps(dict(event="loading", checkpoint=str(a.checkpoint))), flush=True)
    model, heads, codec, meta = load_bundle(a.checkpoint, device)
    vae = load_wan_vae(DEFAULT_WAN / "vae.pth", device=device, dtype=torch.bfloat16)
    model.eval().requires_grad_(False)
    heads.eval().requires_grad_(False)
    codec.eval().requires_grad_(False)
    source = load_source(
        Path(__import__("os").environ.get("RAQVC_DATA", "/qzwang/data"))
        / "UVG/Beauty_1920x1080_120fps_420_8bit_YUV.yuv",
        "UVG",
        120,
        25,
    )
    report = dict(
        checkpoint=str(a.checkpoint),
        hardware=torch.cuda.get_device_name(),
        frames=25,
        repeats=a.repeats,
        protocol="steady state component-sum throughput, one reset GOP, all components resident; CUDA synchronized; excludes bitstream metadata inspection/cloning, metrics/disk/resize/model load",
        samples="first25 Beauty frames repeated; timing diagnostic, not whole-dataset FPS",
        resize="none; center736/crop720 or reflect1088/crop1080",
        dtype="bfloat16 VAE/DiT, float32 codec",
        rows=[],
        vae_variants=[],
    )
    for resolution in a.resolutions:
        target, height, width = source_for(source, resolution)
        target = target.to(device) * 2 - 1
        with torch.inference_mode():
            for qp in a.qps:
                records = []
                saved = None
                for repeat in range(a.repeats + 1):
                    torch.cuda.reset_peak_memory_stats()
                    clean, tvae = timed(lambda: encode_video(vae, target))
                    (stream, encoded), tenc = timed(lambda: codec.encode(clean[None].float(), qp=qp, p_shift_qp=False))
                    hard, tdec = timed(lambda: codec.decode(stream))
                    if not torch.equal(encoded, hard):
                        raise RuntimeError("independent decoder mismatch")
                    payload = torch.load(io.BytesIO(stream), map_location="cpu", weights_only=False)
                    bits = sum(8 * len(f["bit_stream"]) for batch in payload["batches"] for f in batch)
                    actual = hard.new_tensor([bits / (25 * target.shape[-2] * target.shape[-1])])
                    sigma = codec.get_last_entropy_scale_sequence().detach().clone()
                    width_map = codec.get_last_quantization_width_sequence().detach().clone()
                    cond, tcond = timed(lambda: build_condition(codec, hard, sigma, width_map, actual, qp, device))
                    with torch.autocast("cuda", dtype=torch.bfloat16):
                        (out, aux), tdit = timed(lambda: endpoint(model, heads, cond))
                        rgb, trgb = timed(lambda: decode_latent(vae, out[0].float()))
                    row = dict(
                        resolution=resolution,
                        qp=qp,
                        repeat=repeat,
                        warmup=repeat == 0,
                        vae_encode_s=tvae,
                        codec_encode_s=tenc,
                        codec_decode_s=tdec,
                        condition_s=tcond,
                        dit_heads_s=tdit,
                        vae_decode_s=trgb,
                        encoder_fps=25 / (tvae + tenc),
                        decoder_fps=25 / (tdec + tcond + tdit + trgb),
                        end_to_end_fps=25 / (tvae + tenc + tdec + tcond + tdit + trgb),
                        actual_bits=bits,
                        peak_gib=torch.cuda.max_memory_allocated() / 2**30,
                    )
                    print(json.dumps(dict(event="timing", **row)), flush=True)
                    records.append(row)
                    if qp == 39 and repeat == a.repeats:
                        saved = (out.detach().clone(), rgb.detach().float().cpu())
                    del clean, stream, encoded, hard, cond, out, aux, rgb, sigma, width_map, actual, payload
                kept = records[1:]
                times = {
                    key: statistics.median(r[key] for r in kept)
                    for key in (
                        "vae_encode_s",
                        "codec_encode_s",
                        "codec_decode_s",
                        "condition_s",
                        "dit_heads_s",
                        "vae_decode_s",
                    )
                }
                enc = times["vae_encode_s"] + times["codec_encode_s"]
                dec = sum(times[k] for k in ("codec_decode_s", "condition_s", "dit_heads_s", "vae_decode_s"))
                report["rows"].append(
                    dict(
                        resolution=resolution,
                        qp=qp,
                        median_seconds=times,
                        encoder_fps=25 / enc,
                        decoder_fps=25 / dec,
                        end_to_end_fps=25 / (enc + dec),
                        vae_decoder_fraction=times["vae_decode_s"] / dec,
                        measurements=records,
                    )
                )
                atomic(a.output / "SPEED.json", report)
        del target
    report["complete"] = True
    atomic(a.output / "SPEED.json", report)
    print(json.dumps(dict(event="complete", output=str(a.output))), flush=True)


if __name__ == "__main__":
    main()
