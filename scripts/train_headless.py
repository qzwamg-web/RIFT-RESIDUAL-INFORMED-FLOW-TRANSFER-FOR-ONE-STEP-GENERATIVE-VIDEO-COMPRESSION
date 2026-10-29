#!/usr/bin/env python3
"""512/T7 codec + rank8 LoRA joint fine-tuning with no connection heads."""

import argparse
import json
import math
import os
from pathlib import Path
import random
import sys
import time
from datetime import timedelta
import zlib

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ.setdefault("TORCH_HOME", "/qzwang/data/raqvc_artifacts/C512_FINAL_v1/runtime_assets/torch_cache")
import torch
import torch.distributed as dist
import torch.nn.functional as F
from raqvc.training import rift_headless as runtime
from raqvc.training.c512 import invalidate_entropy_coders
from raqvc.training.codec_training import online_reconstruction
from raqvc.training.conditions import build_condition
from raqvc.training.distributed import reduce_gradients
from raqvc.final_support import _atomic_json, _atomic_torch, _rgb_metrics
from scripts.train import GlobalBatches, schedule, DATA, WAN, restore_optimizer
from raqvc.data.clean_tar import CleanTar

DEFAULT_CKPT = Path(
    "/qzwang/data/raqvc_artifacts/C512_FINAL_v1/model/headless_step_002048.pt"
)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", type=Path, default=DEFAULT_CKPT)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--steps", type=int, default=2048)
    p.add_argument("--batch", type=int, default=1)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--data", type=Path, default=DATA[512])
    p.add_argument("--save-every", type=int, default=512)
    p.add_argument("--seed", type=int, default=938)
    p.add_argument("--smoke", action="store_true")
    p.add_argument("--resume", action="store_true")
    a = p.parse_args()
    if min(a.steps, a.batch, a.workers, a.save_every) < 1:
        raise ValueError("invalid counts")
    rank, world, local = [int(os.environ.get(n, d)) for n, d in [("RANK", 0), ("WORLD_SIZE", 1), ("LOCAL_RANK", 0)]]
    torch.cuda.set_device(local)
    device = torch.device("cuda", local)
    torch.set_num_threads(4)
    if world > 1:
        dist.init_process_group("nccl", device_id=device, timeout=timedelta(minutes=45))

    def barrier():
        if world > 1:
            dist.barrier(device_ids=[local])

    def emit(event, **kw):
        if rank == 0:
            print(json.dumps(dict(event=event, **kw)), flush=True)

    torch.manual_seed(a.seed)
    random.seed(a.seed)
    total = 4 if a.smoke else a.steps
    out = a.output
    out.mkdir(parents=True, exist_ok=True)
    identity = dict(
        schema=runtime.SCHEMA,
        initial_checkpoint=str(a.checkpoint.resolve()),
        data=str(a.data.resolve()),
        steps=total,
        batch=a.batch,
        world_size=world,
        resolution=512,
        latent_frames=7,
        rgb_frames=25,
        seed=a.seed,
        smoke=a.smoke,
        rates=runtime.RATES,
        loss="rate+lambda*(.1L1+LPIPS+DISTS+.05temporal+.1latent+.05codec_visual)",
    )
    config = out / "CONFIG.json"
    resume_file = out / "resume_latest.pt"
    if config.exists():
        if json.loads(config.read_text())["identity"] != identity:
            raise ValueError("training plan changed")
        if not a.resume:
            raise FileExistsError("use --resume with existing training output")
    if (out / "RESULT.json").exists():
        result = json.loads((out / "RESULT.json").read_text())
        if result.get("complete"):
            if not config.exists():
                raise RuntimeError("completed result without config")
            emit("already_complete", output=str(out))
            if world > 1:
                dist.destroy_process_group()
            return
    saved = (
        torch.load(resume_file, map_location="cpu", weights_only=False) if a.resume and resume_file.exists() else None
    )
    if saved and saved["identity"] != identity:
        raise ValueError("resume identity mismatch")
    source = Path(saved["checkpoint"]) if saved else a.checkpoint
    emit("loading", checkpoint=str(source))
    model, codec, meta = runtime.load_bundle(source, device)
    model.eval()
    codec.eval()
    codec._task938_failure_dir = out / f"parity_rank{rank}"
    Path(codec._task938_failure_dir).mkdir(exist_ok=True)
    groups, inventory = runtime.configure_groups(model, codec)
    params = [p for g in groups for p in g["params"]]
    optimizer = torch.optim.AdamW(groups, weight_decay=0.01)
    if saved:
        groups = restore_optimizer(optimizer, saved["optimizer"])
    from raqvc.backbone.vae import load_wan_vae
    from raqvc.backbone.vae_memory import decode_latent_batch, configure_vae_stateless_checkpointing
    from raqvc.metrics.perceptual import LPIPSMetric
    from DISTS_pytorch import DISTS

    vae = load_wan_vae(WAN / "vae.pth", device=device, dtype=torch.bfloat16)
    vae.model.eval().requires_grad_(False)
    configure_vae_stateless_checkpointing(vae)
    lp = LPIPSMetric(net="vgg", device=str(device))
    ds = DISTS().to(device).eval().requires_grad_(False)
    data = CleanTar(a.data)
    if data.resolution != 512 or data.latent_shape != (48, 7, 32, 32):
        raise ValueError("requires clean512/T7 dataset")
    train = [i for i, r in enumerate(data.rows) if zlib.crc32(r["meta"]["video_path"].encode()) % 20]
    if len(train) < world * a.batch:
        raise ValueError("insufficient training data")
    meta.update(
        training=identity,
        phase="headless_joint",
        resolution=512,
        world_size=world,
        global_batch=world * a.batch,
        data=str(a.data.resolve()),
        optimizer_state_reused=bool(saved),
        lambda_qp=list(runtime.LAMBDAS),
        noise_alpha=0.0,
        connection_heads=False,
        codec_frozen=False,
        selection="terminal requested step; no test-set checkpoint selection",
    )
    for key in ["head_contract", "heads_checkpoint", "head_state"]:
        meta.pop(key, None)
    if rank == 0:
        _atomic_json(
            config,
            dict(
                identity=identity,
                groups=inventory,
                meta=meta,
                train_windows=len(train),
                train_videos=len({data.rows[i]["meta"]["video_path"] for i in train}),
                frozen="Wan base weights and VAE",
                trainable="codec, regional quantizer, rank8 LoRA, inherited context adapters",
            ),
        )
    emit(
        "training_config",
        **identity,
        parameters={k: v["parameters"] for k, v in inventory.items()},
        global_batch=world * a.batch,
    )
    start_step = saved["step"] + 1 if saved else 1
    if saved:
        state = saved["rng"][rank]
        torch.set_rng_state(state["torch"])
        torch.cuda.set_rng_state(state["cuda"], device)
        random.setstate(state["python"])
    if rank == 0 and (out / "train.jsonl").exists():
        lines = [s for s in (out / "train.jsonl").read_text().splitlines() if json.loads(s)["step"] < start_step]
        (out / "train.jsonl").write_text("\n".join(lines) + ("\n" if lines else ""))
    del saved

    def save(step):
        path = out / f"step_{step:06d}.pt"
        if rank == 0:
            state = runtime.bundle_state(model, codec, meta)
            state["step"] = step
            if "heads" in state:
                raise RuntimeError("unexpected head state")
            _atomic_torch(path, state)
        own = dict(torch=torch.get_rng_state(), cuda=torch.cuda.get_rng_state(device), python=random.getstate())
        states = [own]
        if world > 1:
            states = [None] * world
            dist.all_gather_object(states, own)
        if rank == 0:
            _atomic_torch(
                resume_file,
                dict(
                    identity=identity,
                    step=step,
                    checkpoint=str(path.resolve()),
                    optimizer=optimizer.state_dict(),
                    rng=states,
                ),
            )
        barrier()
        return path

    def prepare(row, qp, training):
        codec.train(training)
        clean = row["clean"].to(device)
        target = row["source_video"].to(device)
        if clean.ndim == 4:
            clean = clean[None]
            target = target[None]
        if tuple(clean.shape[1:]) != (48, 7, 32, 32) or tuple(target.shape[1:]) != (3, 25, 512, 512):
            raise ValueError("whole GOP geometry")
        source, rate, actual, sigma, width = online_reconstruction(codec, clean, qp, verify=True, hard_eval=True)
        c = build_condition(codec, source.float(), sigma.float(), width.float(), actual.float(), qp, device)
        return clean, target, c, rate, actual

    audit_norms = {g["name"]: 0.0 for g in groups}
    initial = {g["name"]: [v.detach().cpu().clone() for v in g["params"]] for g in groups} if rank == 0 else {}
    batches = GlobalBatches(data, train, a.batch, rank, world, a.seed, start_step, total, a.workers)
    started = time.monotonic()
    last = started
    final_path = source
    for step, row in batches:
        tick = time.monotonic()
        waited = tick - last
        qp = runtime.QPS[(step - 1) % 4]
        lam = runtime.LAMBDAS[(step - 1) % 4]
        optimizer.zero_grad(set_to_none=True)
        clean, target, c, rate, actual = prepare(row, qp, True)
        if not c.source.requires_grad or not rate.requires_grad:
            raise RuntimeError("codec joint gradient disconnected")
        # Same codec-only safeguard as RIFT. Backprop separately to release its
        # VAE/metric graph before constructing the larger restoration graph.
        with torch.autocast("cuda", dtype=torch.bfloat16):
            codec_rgb = decode_latent_batch(vae, c.source, batch_size=1)
        clp, cds = _rgb_metrics(codec_rgb, target, lp, ds, 1)
        codec_loss = 0.1 * F.l1_loss(codec_rgb.float(), target.float()) + clp + cds
        if not torch.isfinite(codec_loss):
            raise FloatingPointError("codec loss")
        codec_value = float(codec_loss.detach())
        (0.05 * lam * codec_loss).backward(retain_graph=True)
        del codec_rgb, codec_loss, clp, cds
        source_gradient = []
        handle = c.source.register_hook(lambda grad: source_gradient.append(float(grad.float().norm())))
        with torch.autocast("cuda", dtype=torch.bfloat16):
            restored, aux = runtime.endpoint(model, c, audit=a.smoke or step == start_step)
            rgb = decode_latent_batch(vae, restored, batch_size=1)
        distortion, terms = runtime.distortion(rgb, target, restored, clean, lp, ds)
        loss = rate.mean() + lam * distortion
        if not torch.isfinite(loss):
            raise FloatingPointError("restoration loss")
        loss.backward()
        handle.remove()
        if not any(math.isfinite(v) and v > 0 for v in source_gradient):
            raise RuntimeError("restoration did not train codec latent")
        reduce_gradients(params, world)
        norms = {}
        for g in groups:
            gg = [p.grad for p in g["params"] if p.grad is not None]
            norms[g["name"]] = float(torch.stack([v.float().square().sum() for v in gg]).sum().sqrt()) if gg else 0.0
            audit_norms[g["name"]] += norms[g["name"]]
        torch.nn.utils.clip_grad_norm_(params, 1.0, error_if_nonfinite=True)
        for g in groups:
            g["lr"] = g["base_lr"] * schedule(step, total, 64)
        optimizer.step()
        invalidate_entropy_coders(codec)
        names = list(terms)
        values = torch.tensor(
            [float(loss.detach()) + 0.05 * lam * codec_value, float(actual.mean()), codec_value]
            + [float(terms[k].detach()) for k in names],
            device=device,
        )
        if world > 1:
            dist.all_reduce(values)
            values /= world
        rec = dict(
            step=step,
            qp=qp,
            loss=float(values[0]),
            actual_bpp=float(values[1]),
            codec_visual=float(values[2]),
            **{k: float(values[3 + i]) for i, k in enumerate(names)},
            elapsed=time.monotonic() - started,
            step_seconds=time.monotonic() - tick,
            data_wait_seconds=waited,
            peak_gib=torch.cuda.max_memory_allocated() / 2**30,
            global_batch=world * a.batch,
            samples=step * world * a.batch,
            gradient_norms=norms,
            lr={g["name"]: g["lr"] for g in groups},
            tau=runtime.TAU[(step - 1) % 4],
            source_gradient=max(source_gradient),
            connection_heads=0,
        )
        if rank == 0:
            with (out / "train.jsonl").open("a") as f:
                f.write(json.dumps(rec) + "\n")
            _atomic_json(out / "PROGRESS.json", rec)
        if step <= start_step + 3 or step % 25 == 0 or step == total:
            emit("train", **rec)
        if step == min(start_step + 3, total):
            if any(not math.isfinite(v) or v <= 0 for v in audit_norms.values()):
                raise RuntimeError("inactive group " + str(audit_norms))
            if rank == 0:
                changes = {
                    g["name"]: sum(
                        float((p.detach().cpu() - old).square().sum())
                        for p, old in zip(g["params"], initial[g["name"]])
                    )
                    ** 0.5
                    for g in groups
                }
                if any(v == 0 for v in changes.values()):
                    raise RuntimeError("optimizer did not change group")
                _atomic_json(
                    out / "GRADIENT_AUDIT.json",
                    dict(complete=True, group_norms=audit_norms, changes=changes, heads=0, lora_rank=8),
                )
            initial.clear()
        del clean, target, c, rate, actual, restored, aux, rgb, distortion, terms, loss, values, row
        optimizer.zero_grad(set_to_none=True)
        if step % a.save_every == 0 or step == total:
            final_path = save(step)
        last = time.monotonic()
    if a.smoke:
        with torch.no_grad():
            clean, target, c, _, _ = prepare(data[train[rank]], 63, False)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                before, _ = runtime.endpoint(model, c, audit=True)
            bundle = torch.load(final_path, map_location="cpu", weights_only=False)
            assert bundle["meta"]["schema"] == runtime.SCHEMA and "heads" not in bundle
            model.load_state_dict(bundle["model_adaptation"], strict=False)
            codec.load_state_dict(bundle["codec"], strict=True)
            del bundle
            invalidate_entropy_coders(codec)
            clean, target, c, _, _ = prepare(data[train[rank]], 63, False)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                after, _ = runtime.endpoint(model, c, audit=True)
            error = float((before - after).abs().max())
            if error > 2e-5:
                raise RuntimeError("save/reload endpoint drift " + str(error))
            emit("save_reload_pass", max_abs=error)
    if rank == 0:
        _atomic_json(
            out / "RESULT.json",
            dict(
                complete=True,
                steps=total,
                smoke=a.smoke,
                final_checkpoint=str(final_path.resolve()),
                elapsed=time.monotonic() - started,
                identity=identity,
                selection="terminal checkpoint",
            ),
        )
    barrier()
    emit("training_complete", checkpoint=str(final_path))
    if world > 1:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
