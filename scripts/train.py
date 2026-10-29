#!/usr/bin/env python3
"""H2 full-chain continuation; shared runtime, online codec, resumable schedule."""

from __future__ import annotations
import argparse
import collections
import contextlib
import json
import math
import os
import random
import sys
import time
import zlib
from concurrent.futures import ThreadPoolExecutor
from datetime import timedelta
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.utils.data import default_collate
from raqvc.final_support import _atomic_json, _atomic_torch, _losses, _rgb_metrics
from raqvc.data.clean_tar import CleanTar
from raqvc.training.codec_training import online_reconstruction
from raqvc.training.conditions import build_condition
from raqvc.training.distributed import reduce_gradients

ART = Path(__import__("os").environ.get("RAQVC_ARTIFACTS", "/qzwang/data/raqvc_artifacts"))
DATA = {256: ART / "C512_FINAL_v1/datasets/256/index.json", 512: ART / "C512_FINAL_v1/datasets/512/index.json"}
WAN = Path(__import__("os").environ.get("RAQVC_WAN", str(ART / "C512_FINAL_v1/model/wan")))
QPS = (0, 16, 39, 63)
LAMBDAS = (0.15, 0.08, 0.05, 0.02)
RESOLUTIONS = dict(A=512, B=256, C=512, D=512)


class GlobalBatches:
    """Shuffled global epochs; deterministic random access supports exact resume."""

    def __init__(self, data, indices, batch, rank, world, seed, start, stop, workers, origin_step=0, origin_samples=0):
        self.data, self.indices, self.batch = data, indices, batch
        self.rank, self.world, self.seed = rank, world, seed
        self.start, self.stop, self.workers = start, stop, max(1, workers)
        self.permutations = {}
        self.origin_step, self.origin_samples = origin_step, origin_samples

    def consumed_before(self, step):
        return self.origin_samples + (step - 1 - self.origin_step) * self.world * self.batch

    def positions(self, step):
        n = len(self.indices)
        offset = self.consumed_before(step) + self.rank * self.batch
        result = []
        for position in range(offset, offset + self.batch):
            epoch, slot = divmod(position, n)
            if epoch not in self.permutations:
                self.permutations[epoch] = torch.randperm(
                    n, generator=torch.Generator().manual_seed(self.seed + epoch)
                ).tolist()
                for old in list(self.permutations):
                    if old < epoch - 1:
                        del self.permutations[old]
            result.append(self.indices[self.permutations[epoch][slot]])
        return result

    def __iter__(self):
        def fetch(step, positions):
            return step, default_collate([self.data[ix] for ix in positions])

        with ThreadPoolExecutor(max_workers=self.workers) as pool:
            pending = collections.deque()
            steps = iter(range(self.start, self.stop + 1))
            for _ in range(self.workers):
                step = next(steps, None)
                if step is not None:
                    pending.append(pool.submit(fetch, step, self.positions(step)))
            while pending:
                yield pending.popleft().result()
                step = next(steps, None)
                if step is not None:
                    pending.append(pool.submit(fetch, step, self.positions(step)))


def schedule(step, total, warmup):
    warmup = min(warmup, max(1, total // 8))
    if step <= warmup:
        return step / warmup
    position = (step - warmup) / max(1, total - warmup)
    return 0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * min(1.0, position)))


def continuation_lr(step, total, continuation, group):
    """Re-anchor cosine at the saved optimizer LR, without a warmup or LR jump."""
    anchor = continuation["anchor_step"]
    fraction = min(1.0, max(0.0, (step - anchor) / max(1, total - anchor)))
    start = continuation["anchor_lrs"][group]
    return start * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * fraction)))


def qp_for_step(step, epoch, train_size, continuation=None):
    # Batch migration changes sample exposure, not the original QP sequence.
    if continuation:
        source = continuation["source_identity"]
        epoch = ((step - 1) * source["batch"] * source["world_size"]) // train_size
    return QPS[(step - 1 + epoch) % 4]


def restore_optimizer(optimizer, saved):
    # load_state_dict replaces parameter-group dicts. Scheduling stale `groups`
    # only changes the log while optimizer.step silently retains the old LR.
    expected = [(g["name"], len(g["params"])) for g in optimizer.param_groups]
    actual = [(g["name"], len(g["params"])) for g in saved["param_groups"]]
    if actual != expected:
        raise ValueError("saved optimizer parameter-group layout changed")
    optimizer.load_state_dict(saved)
    return optimizer.param_groups


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--phase", choices=list(RESOLUTIONS), required=True)
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--steps", type=int, required=True)
    p.add_argument("--batch", type=int, default=1)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--validation-windows", type=int, default=128)
    p.add_argument("--metric-batch", type=int, default=1)
    p.add_argument("--seed", type=int, default=938)
    p.add_argument("--smoke", action="store_true")
    p.add_argument("--resume", action="store_true")
    a = p.parse_args()
    if min(a.steps, a.batch, a.validation_windows, a.metric_batch) < 1 or a.workers < 0:
        p.error("invalid counts")
    rank, world, local = (int(os.environ.get(n, d)) for n, d in [("RANK", 0), ("WORLD_SIZE", 1), ("LOCAL_RANK", 0)])
    torch.cuda.set_device(local)
    device = torch.device("cuda", local)
    torch.set_num_threads(4)
    if world > 1:
        dist.init_process_group("nccl", timeout=timedelta(minutes=60), device_id=device)

    def barrier():
        if world > 1:
            dist.barrier(device_ids=[local])

    def emit(event, **values):
        if rank == 0:
            print(json.dumps(dict(event=event, **values)), flush=True)

    torch.manual_seed(a.seed)
    random.seed(a.seed)
    resolution = RESOLUTIONS[a.phase]
    total = 4 if a.smoke else a.steps
    joint = a.phase != "A"
    out = a.output
    if rank == 0:
        out.mkdir(parents=True, exist_ok=True)
        if (out / "RESULT.json").exists():
            if not a.resume:
                raise FileExistsError("completed stage already exists: " + str(out))
            if json.loads((out / "RESULT.json").read_text()).get("complete"):
                emit("already_complete", output=str(out))
    barrier()
    if a.resume and (out / "RESULT.json").is_file() and json.loads((out / "RESULT.json").read_text()).get("complete"):
        if world > 1:
            dist.destroy_process_group()
        return
    resume_file = out / "resume_latest.pt"
    resume = (
        torch.load(resume_file, map_location="cpu", weights_only=False) if a.resume and resume_file.is_file() else None
    )
    identity = dict(
        phase=a.phase,
        steps=total,
        batch=a.batch,
        world_size=world,
        seed=a.seed,
        initial_checkpoint=str(a.checkpoint.resolve()),
        smoke=a.smoke,
        resolution=resolution,
    )
    if resume and resume["identity"] != identity:
        raise ValueError("resume must preserve phase/batch/world/steps/seed and initial checkpoint")
    continuation = resume.get("continuation") if resume else None
    if not a.resume and ((out / "CONFIG.json").exists() or resume_file.exists()):
        raise FileExistsError("stage has training state; use --resume")
    from raqvc.training.c512 import load_bundle, configure_groups, bundle_state, invalidate_entropy_coders
    from raqvc.training.restoration import endpoint, auxiliary_loss

    checkpoint = Path(resume["bundle_checkpoint"]) if resume else a.checkpoint
    emit("loading_bundle", checkpoint=str(checkpoint), phase=a.phase)
    model, heads, codec, meta = load_bundle(checkpoint, device)
    model.eval()
    heads.eval()
    codec.eval()
    codec._task938_failure_dir = out / f"parity_rank{rank}"
    Path(codec._task938_failure_dir).mkdir(parents=True, exist_ok=True)
    groups, inventory = configure_groups(model, heads, codec, a.phase)
    parameters = [v for group in groups for v in group["params"]]
    if len({id(v) for v in parameters}) != len(parameters):
        raise RuntimeError("duplicate optimizer parameters")
    optimizer = torch.optim.AdamW(groups, weight_decay=0.01)
    if resume:
        groups = restore_optimizer(optimizer, resume["optimizer"])
    from raqvc.backbone.vae import load_wan_vae
    from raqvc.backbone.vae_memory import decode_latent_batch, configure_vae_stateless_checkpointing
    from raqvc.metrics.perceptual import LPIPSMetric
    from DISTS_pytorch import DISTS

    vae = load_wan_vae(WAN / "vae.pth", device=device, dtype=torch.bfloat16)
    vae.model.eval().requires_grad_(False)
    configure_vae_stateless_checkpointing(vae)
    lp = LPIPSMetric(net="vgg", device=str(device))
    ds = DISTS().to(device).eval().requires_grad_(False)
    data = CleanTar(DATA[resolution])
    validation_data = data if resolution == 512 else CleanTar(DATA[512])
    train, valid, seen = [], [], set()
    for i, row in enumerate(data.rows):
        name = row["meta"]["video_path"]
        if zlib.crc32(name.encode()) % 20:
            train.append(i)
    for i, row in enumerate(validation_data.rows):
        name = row["meta"]["video_path"]
        if not zlib.crc32(name.encode()) % 20 and name not in seen:
            valid.append(i)
            seen.add(name)
    nvalid = 1 if a.smoke else a.validation_windows
    valid = valid[128 : 128 + nvalid]
    if len(valid) != nvalid:
        raise ValueError("not enough distinct heldout videos")
    stage_seed = a.seed + (ord(a.phase) - ord("A")) * 100003
    state_meta = dict(
        meta,
        phase=a.phase,
        codec_frozen=not joint,
        resolution=resolution,
        data=str(DATA[resolution]),
        world_size=world,
        global_batch=a.batch * world,
        lambda_qp=list(LAMBDAS),
        initial_checkpoint=str(a.checkpoint),
        noise_alpha=0.0,
        selection="mean actual_bpp/lambda_q + RGB perceptual distortion on fixed heldout512",
        optimizer_state_reused=bool(resume),
        continuation=continuation,
    )
    state_meta["head_contract"] = dict(state_meta.get("head_contract", {}), condition_grad_enabled=joint)
    if rank == 0:
        _atomic_json(
            out / "CONFIG.json",
            dict(
                identity=identity,
                args=vars(a),
                meta=state_meta,
                groups=inventory,
                validation_keys=[validation_data.rows[i]["key"] for i in valid],
                train_windows=len(train),
                train_videos=len({data.rows[i]["meta"]["video_path"] for i in train}),
            ),
        )
    emit(
        "training_config",
        **identity,
        learning_rates={g["name"]: g["base_lr"] for g in groups},
        parameters={g["name"]: sum(x.numel() for x in g["params"]) for g in groups},
        tau_table=heads.base.bases.tolist(),
        output_bridge="unit",
        lambda_qp=list(LAMBDAS),
        codec_frozen=not joint,
        optimizer_state_reused=bool(resume),
    )

    def prepare(row, qp, training=False):
        # The configured I/P reference BPTT is gated by codec.training.
        # Actual rANS always uses eval arithmetic, then restores this mode.
        codec.train(bool(training and joint))
        clean, target = row["clean"].to(device), row["source_video"].to(device)
        if clean.ndim == 4:
            clean, target = clean[None], target[None]
        scope = contextlib.nullcontext() if training and joint else torch.no_grad()
        with scope:
            source, rate, actual, sigma, width = online_reconstruction(codec, clean, qp, verify=True, hard_eval=True)
            cond = build_condition(codec, source.float(), sigma.float(), width.float(), actual.float(), qp, device)
        return clean, target, cond, rate, actual

    def codec_quality(source, target, validation=False):
        with torch.autocast("cuda", dtype=torch.bfloat16):
            codec_rgb = decode_latent_batch(vae, source.float(), batch_size=1)
        if validation:
            codec_rgb = codec_rgb.clamp(-1, 1)
        clp, cds = _rgb_metrics(codec_rgb, target, lp, ds, a.metric_batch)
        return 0.1 * F.l1_loss(codec_rgb.float(), target.float()) + clp + cds

    def compute(clean, target, cond, rate, include_codec_visual=True, validation=False):
        with torch.autocast("cuda", dtype=torch.bfloat16):
            restored, aux = endpoint(model, heads, cond)
            rgb = decode_latent_batch(vae, restored.float(), batch_size=1)
        if validation:
            rgb = rgb.clamp(-1, 1)
        base_loss, terms = _losses(rgb, target, clean, cond.source, aux, lp, ds, a.metric_batch)
        # The independent strength auxiliary already owns RMS supervision.
        distortion = base_loss - 0.1 * terms["rms_log"] - 0.001 * terms["tau_trust"]
        extra, extra_terms = auxiliary_loss(heads, aux, clean, cond.source)
        terms.update(extra_terms)
        codec_visual = distortion.new_zeros(())
        if joint and include_codec_visual:
            codec_visual = codec_quality(cond.source, target, validation=validation)
        lam = LAMBDAS[QPS.index(cond.qp)]
        loss = (rate.mean() + lam * (distortion + 0.05 * codec_visual + extra)) if joint else distortion + extra
        terms.update(codec_visual=codec_visual, distortion=distortion, rate_proxy=rate.mean())
        return loss, terms, aux, rgb

    def validate(step, full=True):
        selected = valid if full else valid[: min(32, len(valid))]
        rows = []
        with torch.no_grad():
            for ix in selected[rank::world]:
                for qp in QPS:
                    clean, target, cond, rate, actual = prepare(validation_data[ix], qp)
                    _, terms, aux, rgb = compute(clean, target, cond, rate, validation=True)
                    mse = ((rgb.float().clamp(-1, 1) - target) / 2).square().mean((1, 3, 4))
                    row = dict(
                        key=validation_data.rows[ix]["key"],
                        qp=qp,
                        psnr=float((-10 * mse.clamp_min(1e-12).log10()).mean()),
                        actual_bpp=float(actual.mean()),
                        **{k: float(v) for k, v in terms.items()},
                    )
                    # Keep the loss term in [-1,1] units, and expose a separate
                    # [0,1] temporal metric matching the native evaluator.
                    row["temporal_delta_l1"] = 0.5 * row["temporal"]
                    from raqvc.training.residual_heads import strength_target

                    truth = strength_target(clean - cond.source)
                    pred = aux["rhat"].float().clamp_min(1e-8)
                    for label, sl in [("I", slice(0, 1)), ("P", slice(1, None))]:
                        pr, gt = pred[:, :, sl], truth[:, :, sl]
                        row[label + "_rms_ratio"] = float((pr / gt.clamp_min(1e-8)).mean())
                        row[label + "_log_rmse"] = float((pr.log() - gt.log()).square().mean().sqrt())
                        prc, gtc = pr.flatten(3), gt.flatten(3)
                        prc = prc - prc.mean(-1, keepdim=True)
                        gtc = gtc - gtc.mean(-1, keepdim=True)
                        denom = (prc.square().sum(-1) * gtc.square().sum(-1)).sqrt()
                        valid_corr = denom > 1e-10
                        row[label + "_within_frame_corr"] = (
                            float(((prc * gtc).sum(-1) / denom.clamp_min(1e-10))[valid_corr].mean())
                            if valid_corr.any()
                            else None
                        )
                    rows.append(row)
        if world > 1:
            all_rows = [None] * world
            dist.all_gather_object(all_rows, rows)
            rows = [x for part in all_rows for x in part]
        rows.sort(key=lambda x: (x["key"], x["qp"]))
        if len(rows) != len(selected) * 4:
            raise RuntimeError("incomplete validation rows")
        # Include actual rate once codec is unlocked. A uses perceptual selection.
        score = sum(
            (
                (r["actual_bpp"] / LAMBDAS[QPS.index(r["qp"])] + r["distortion"] + 0.05 * r["codec_visual"])
                if joint
                else r["lpips"] + r["dists"]
            )
            for r in rows
        ) / len(rows)
        qp_means = {
            str(q): {
                k: sum(r[k] for r in rows if r["qp"] == q) / len(selected)
                for k in ["actual_bpp", "psnr", "lpips", "dists", "temporal", "temporal_delta_l1"]
            }
            for q in QPS
        }
        rates = [qp_means[str(q)]["actual_bpp"] for q in QPS]
        monotonic = all(x >= y for x, y in zip(rates, rates[1:]))
        if rank == 0:
            _atomic_json(
                out / f'validation_{step:06d}{"" if full else "_quick"}.json',
                dict(step=step, score=score, rows=rows, resolution=512, qp_means=qp_means, rate_monotonic=monotonic),
            )
        emit(
            "validation",
            step=step,
            full=full,
            score=score,
            videos=len(selected),
            resolution=512,
            rate_monotonic=monotonic,
            qp_means=qp_means,
        )
        if step == 0 and not monotonic:
            raise RuntimeError("initial codec has non-monotonic heldout rate points")
        return score if monotonic else float("inf")

    def save_bundle(step, score):
        dest = out / f"step_{step:06d}.pt"
        if rank == 0:
            payload = bundle_state(model, heads, codec, state_meta)
            payload.update(step=step, validation_score=score)
            _atomic_torch(dest, payload)
        barrier()
        return dest

    def rng_state():
        return dict(torch=torch.get_rng_state(), cuda=torch.cuda.get_rng_state(device), python=random.getstate())

    def save_resume(step, checkpoint, best_path, best_score, initial_score):
        own = rng_state()
        rngs = [own]
        if world > 1:
            rngs = [None] * world
            dist.all_gather_object(rngs, own)
        if rank == 0:
            _atomic_torch(
                resume_file,
                dict(
                    identity=identity,
                    step=step,
                    bundle_checkpoint=str(checkpoint.resolve()),
                    optimizer=optimizer.state_dict(),
                    rng=rngs,
                    best_path=str(best_path.resolve()),
                    best_score=best_score,
                    initial_score=initial_score,
                    continuation=continuation,
                ),
            )
        barrier()

    start_step = int(resume["step"]) + 1 if resume else 1
    if resume:
        best, best_path = float(resume["best_score"]), Path(resume["best_path"])
        initial_score = float(resume["initial_score"])
        r = resume["rng"][rank]
        torch.set_rng_state(r["torch"])
        torch.cuda.set_rng_state(r["cuda"], device)
        random.setstate(r["python"])
    else:
        initial_score = validate(0)
        best, best_path = initial_score, save_bundle(0, initial_score)
        save_resume(0, best_path, best_path, best, initial_score)

    # Audits capture the first four QPs after start; mean gradients synchronized first.
    audits = {g["name"]: 0.0 for g in groups}
    tensor_used = {id(v): False for v in parameters}
    tensor_nonzero = {id(v): False for v in parameters}
    original = {g["name"]: [(v, v.detach().cpu().clone()) for v in g["params"]] for g in groups}
    if rank != 0:
        original = {}
    iterator = GlobalBatches(
        data,
        train,
        a.batch,
        rank,
        world,
        stage_seed,
        start_step,
        total,
        a.workers,
        origin_step=continuation["anchor_step"] if continuation else 0,
        origin_samples=continuation["consumed_samples"] if continuation else 0,
    )
    start = time.monotonic()
    last = start
    history_path = out / "train.jsonl"
    final_path = Path(resume["bundle_checkpoint"]) if resume else best_path
    for step, row in iterator:
        waited = time.monotonic() - last
        epoch = iterator.consumed_before(step) // len(train)
        qp = qp_for_step(step, epoch, len(train), continuation)
        optimizer.zero_grad(set_to_none=True)
        clean, target, cond, rate, actual = prepare(row, qp, training=True)
        source_grad_norm = []
        if joint:
            if not cond.source.requires_grad or not rate.requires_grad:
                raise RuntimeError("joint codec source/rate lost the autograd graph")
            cond.source.register_hook(lambda grad: source_grad_norm.append(float(grad.detach().float().norm())))
            if a.smoke and step == start_step:
                regional = next(g["params"] for g in groups if g["name"] == "regional")
                rate_grad = torch.autograd.grad(rate.mean(), regional, retain_graph=True, allow_unused=True)
                rate_norm = sum(float(g.detach().float().square().sum()) for g in rate_grad if g is not None) ** 0.5
                if not math.isfinite(rate_norm) or rate_norm <= 0:
                    raise RuntimeError("rate proxy does not train the regional quantizer")
                emit("rate_regional_gradient", l2=rate_norm)
                del rate_grad
        # Release the codec-only VAE/metric graph before building the much
        # larger DiT-restoration graph. Gradients sum to the same joint loss.
        codec_visual_value = rate.new_zeros(())
        if joint:
            codec_aux = codec_quality(cond.source, target)
            if not torch.isfinite(codec_aux):
                raise FloatingPointError("nonfinite codec RGB auxiliary")
            codec_visual_value = codec_aux.detach()
            (0.05 * LAMBDAS[QPS.index(qp)] * codec_aux).backward(retain_graph=True)
            del codec_aux
            source_grad_norm.clear()
        loss, terms, aux, rgb = compute(clean, target, cond, rate, include_codec_visual=False)
        if not torch.isfinite(loss):
            raise FloatingPointError("nonfinite training loss")
        loss.backward()
        if joint and (not source_grad_norm or not any(v > 0 and math.isfinite(v) for v in source_grad_norm)):
            raise RuntimeError("restoration distortion did not backpropagate to compressed latent")
        reduce_gradients(parameters, world)
        norms = {}
        for g in groups:
            ps = [v for v in g["params"] if v.grad is not None]
            norms[g["name"]] = (
                float(torch.stack([v.grad.float().square().sum() for v in ps]).sum().sqrt()) if ps else 0.0
            )
            audits[g["name"]] += norms[g["name"]]
            if step <= start_step + 3:
                magnitudes = (
                    torch.stack([v.grad.detach().float().square().sum() for v in ps]).cpu().tolist() if ps else []
                )
                for v, magnitude in zip(ps, magnitudes):
                    tensor_used[id(v)] = True
                    tensor_nonzero[id(v)] |= magnitude > 0
        torch.nn.utils.clip_grad_norm_(parameters, 1.0, error_if_nonfinite=True)
        factor = schedule(step, total, 128 if a.phase in ("A", "D") else 256)
        for g in groups:
            g["lr"] = continuation_lr(step, total, continuation, g["name"]) if continuation else g["base_lr"] * factor
        optimizer.step()
        if joint:
            invalidate_entropy_coders(codec)
        terms["codec_visual"] = codec_visual_value
        reported_loss = loss.detach() + (0.05 * LAMBDAS[QPS.index(qp)] * codec_visual_value if joint else 0.0)
        values = torch.tensor(
            [float(reported_loss), float(actual.mean())] + [float(v.detach()) for v in terms.values()], device=device
        )
        if world > 1:
            dist.all_reduce(values)
            values /= world
        rec = dict(
            step=step,
            qp=qp,
            epoch=epoch,
            sample_presentations=iterator.consumed_before(step + 1),
            loss=float(values[0]),
            actual_bpp=float(values[1]),
            data_wait_seconds=waited,
            elapsed=time.monotonic() - start,
            peak_gib=torch.cuda.max_memory_allocated() / 2**30,
            learning_rates={g["name"]: g["lr"] for g in groups},
            gradient_norms=norms,
            tau_mean=float(aux["tau"].detach().mean()),
            bridge=float(aux["bridge"].mean()),
            codec_source_gradient_norm=max(source_grad_norm, default=0.0),
            **{k: float(values[i + 2]) for i, k in enumerate(terms)},
        )
        if rank == 0:
            with history_path.open("a") as f:
                f.write(json.dumps(rec) + "\n")
            if step == start_step or step % 50 == 0 or step == total:
                _atomic_json(out / "PROGRESS.json", rec)
        if step == start_step or step % 50 == 0 or step == total:
            emit("train", **rec)
        if step == min(start_step + 3, total):
            if any(not math.isfinite(v) or v == 0 for v in audits.values()):
                raise RuntimeError("inactive/nonfinite trainable group over four QPs: " + str(audits))
            if rank == 0:
                changed = {
                    k: sum(float((p.detach().cpu() - before).square().sum()) for p, before in items) ** 0.5
                    for k, items in original.items()
                }
                if any(v == 0 for v in changed.values()):
                    raise RuntimeError("parameter group did not update: " + str(changed))
                detail = {
                    g["name"]: dict(
                        no_gradient=[
                            n for n, v in zip(inventory[g["name"]]["names"], g["params"]) if not tensor_used[id(v)]
                        ],
                        zero_gradient=[
                            n
                            for n, v in zip(inventory[g["name"]]["names"], g["params"])
                            if tensor_used[id(v)] and not tensor_nonzero[id(v)]
                        ],
                    )
                    for g in groups
                }
                _atomic_json(
                    out / "GRADIENT_AUDIT.json",
                    dict(
                        group_gradient_norm_sum=audits,
                        parameter_change_l2=changed,
                        frozen_codec=not joint,
                        per_tensor=detail,
                    ),
                )
                emit("gradient_audit_pass", change_l2=changed)
            original.clear()
        del rgb, aux, loss, terms, values, cond, clean, target, row
        optimizer.zero_grad(set_to_none=True)
        if step % 1024 == 0 or step == total:
            full = step % 4096 == 0 or step == total
            score = validate(step, full=full)
            final_path = save_bundle(step, score)
            # Never compare a 32-video quick score with 128-video full score.
            if full and score < best:
                best, best_path = score, final_path
            save_resume(step, final_path, best_path, best, initial_score)
        last = time.monotonic()
    if a.smoke:
        # Roundtrip actual persisted parameters without keeping a second 5B model.
        clean, target, cond, rate, actual = prepare(validation_data[valid[0]], 0)
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            before, _ = endpoint(model, heads, cond)
        from raqvc.training.c512 import restore_bundle_state

        restore_bundle_state(model, heads, codec, torch.load(final_path, map_location="cpu", weights_only=False))
        clean, target, cond, rate, actual = prepare(validation_data[valid[0]], 0)
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            after, _ = endpoint(model, heads, cond)
        difference = float((before - after).abs().max())
        if difference > 2e-5:
            raise RuntimeError(f"save/reload endpoint differs: {difference}")
        emit("save_reload_pass", max_abs=difference)
    if rank == 0:
        _atomic_json(
            out / "RESULT.json",
            dict(
                complete=True,
                phase=a.phase,
                steps=total,
                smoke=a.smoke,
                selected_checkpoint=str(best_path.resolve()),
                final_checkpoint=str(final_path.resolve()),
                initial_score=initial_score,
                selected_score=best,
                meta=state_meta,
            ),
        )
    barrier()
    emit("stage_complete", output=str(out), selected=str(best_path))
    if world > 1:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
