"""Head-free RIFT ablation: one shared differentiable train/eval endpoint.

Task938 connection heads are not retained in the saved bundle. Inherited
DiT-wrapper context adapters remain separate from the removed interface.
"""

import copy
from pathlib import Path
import torch
import torch.nn.functional as F
from raqvc.training.c512 import (
    load_bundle as load_h2,
    DEFAULT_PARENT,
    DEFAULT_WAN,
    _is_adaptation,
    _is_lora,
    _cpu_state,
    invalidate_entropy_coders,
)

SCHEMA = "rift_headless_joint_v1"
TAU = (520.0, 640.0, 800.0, 960.0)
QPS = (0, 16, 39, 63)
LAMBDAS = (0.15, 0.08, 0.05, 0.02)
RATES = dict(codec_backbone=1e-6, regional=5e-6, lora=3e-5, context_adapter=3e-6)


def endpoint(model, condition, audit=False):
    c = condition
    expected = TAU[QPS.index(c.qp)]
    tau = torch.full_like(c.sigma, expected, dtype=torch.float32).flatten(1)
    calls = []

    def hook(module, inputs, out):
        t = inputs[1].detach().float()
        if not torch.allclose(t, torch.full_like(t, expected)):
            raise RuntimeError("actual Wan timestep does not match fixed table")
        calls.append(1)

    handle = model.backbone.register_forward_hook(hook) if audit else None
    try:
        raw = model(c.source, qp=c.qp, log_q_y=c.log_q_y, log_q_z=c.log_q_z, actual_bpp=c.actual_bpp)
        velocity = model.path.denormalize_velocity(raw, c.qp)
        # Preserve the rounding boundary used in the completed inference ablation.
        velocity = ((c.source + velocity) - c.source).float()
        output = c.source.float() + velocity
    finally:
        if handle is not None:
            handle.remove()
    if audit and len(calls) != 1:
        raise RuntimeError("expected exactly one DiT forward")
    return output, dict(
        output=output, tau=tau, unit_gain=True, connection_heads=0, update_rms=velocity.square().mean().sqrt()
    )


def load_bundle(checkpoint, device="cpu"):
    from raqvc.codec import create_latent_codec
    from raqvc.training.model_loader import load_mainline_dit

    checkpoint = Path(checkpoint)
    saved = torch.load(checkpoint, map_location="cpu", weights_only=False, mmap=True)
    if saved["meta"].get("schema") != SCHEMA:
        del saved
        model, heads, codec, meta = load_h2(checkpoint, device)
        del heads  # Never registered, called, optimized, or saved by this runtime.
        meta = copy.deepcopy(meta)
        meta.update(
            schema=SCHEMA,
            initial_checkpoint=str(checkpoint.resolve()),
            base_dit_checkpoint=str(DEFAULT_PARENT / "stage3_dit_reproduction.pt"),
            connection_heads=False,
            endpoint="raqvc.training.rift_headless.endpoint",
            output_rule="zq + physical_DiT_velocity; unit gain",
            tau_table=list(TAU),
            velocity_scale_mode="unit",
            noise_alpha=0.0,
            lora_rank=8,
        )
    else:
        meta = copy.deepcopy(saved["meta"])
        if "heads" in saved:
            raise ValueError("headless bundle contains heads")
        model, _ = load_mainline_dit(DEFAULT_PARENT / "stage3_dit_reproduction.pt", DEFAULT_WAN, torch.device(device))
        expected = {k for k in model.state_dict() if _is_adaptation(k)}
        if set(saved["model_adaptation"]) != expected:
            raise ValueError("adapter state incomplete")
        mismatch = model.load_state_dict(saved["model_adaptation"], strict=False)
        if mismatch.unexpected_keys:
            raise ValueError(mismatch)
        kwargs = copy.deepcopy(meta["codec_config_dict"])
        kind = kwargs.pop("type")
        kwargs.pop("enabled", None)
        kwargs["device"] = str(device)
        codec = create_latent_codec(kind.replace("-", "_"), **kwargs).to(device).float()
        codec.load_state_dict(saved["codec"], strict=True)
        codec.enable_entropy_scale_export(True)
    if meta["tau_table"] != list(TAU) or meta["noise_alpha"] != 0 or meta["velocity_scale_mode"] != "unit":
        raise ValueError("incompatible headless contract")
    ranks = {int(v.shape[0]) for n, v in model.named_parameters() if n.endswith(".lora_A")}
    if ranks != {8}:
        raise ValueError(f"requires rank8 LoRA, got {ranks}")
    # This mapper is installed only to load the immutable historical base.
    # Fixed-time headless execution must not retain an unused time head.
    if hasattr(model, "task938_time_mapper"):
        delattr(model, "task938_time_mapper")
    for key in ["head_contract", "heads_checkpoint", "head_state", "historical_head_contract"]:
        meta.pop(key, None)
    meta.update(
        runtime="rift_headless_joint",
        arm="without_connection_heads",
        run_contract=dict(
            schema=SCHEMA,
            endpoint="raqvc.training.rift_headless.endpoint",
            codec_state="embedded",
            adapter_state="model_adaptation",
            connection_heads=False,
            tau_table=list(TAU),
            output_bridge="unit",
            noise_alpha=0.0,
            solver_steps=1,
            lora_rank=8,
            restore_i=True,
            time_mapper=False,
        ),
    )
    for m in [model, codec]:
        m.eval().requires_grad_(False)
    invalidate_entropy_coders(codec)
    return model, codec, meta


def configure_groups(model, codec):
    model.requires_grad_(False)
    codec.requires_grad_(False)
    selected = {k: [] for k in RATES}
    for n, p in model.named_parameters():
        if _is_lora(n):
            selected["lora"].append(("model." + n, p))
        elif _is_adaptation(n):
            selected["context_adapter"].append(("model." + n, p))
    dormant = getattr(codec, "_is_dormant_adapted_q_parameter", lambda n: False)
    for n, p in codec.named_parameters():
        if dormant(n):
            continue
        if n in ("latent_scale", "latent_bias") and not getattr(codec, "_normalize_latent", False):
            continue
        pc = getattr(codec, "p_codec", None)
        if (
            n.startswith("p_codec.model.feature_adaptor_p.")
            and getattr(codec, "_p_reference_mode", None) == "latent"
            and getattr(pc, "p_decoded_feature_fusion", None) is None
            and getattr(pc, "p_htcm_long_fusion", None) is None
        ):
            continue
        key = "regional" if ".task938_region_quantizer." in n else "codec_backbone"
        selected[key].append(("codec." + n, p))
    groups = []
    inventory = {}
    seen = set()
    for name, entries in selected.items():
        if not entries:
            raise ValueError("empty parameter group " + name)
        for n, p in entries:
            if id(p) in seen:
                raise ValueError("duplicated optimizer parameter " + n)
            seen.add(id(p))
            p.data = p.data.float()
            p.requires_grad_(True)
        groups.append(dict(name=name, params=[p for _, p in entries], lr=RATES[name], base_lr=RATES[name]))
        inventory[name] = dict(
            lr=RATES[name], names=[n for n, _ in entries], parameters=sum(p.numel() for _, p in entries)
        )
    return groups, inventory


def bundle_state(model, codec, meta):
    if meta["schema"] != SCHEMA or meta["connection_heads"] is not False:
        raise ValueError("invalid schema")
    return dict(
        meta=copy.deepcopy(meta),
        codec=_cpu_state(codec.state_dict()),
        model_adaptation=_cpu_state({n: v for n, v in model.state_dict().items() if _is_adaptation(n)}),
    )


def distortion(rgb, target, restored, clean, lp, ds, metric_batch=1):
    from raqvc.final_support import _rgb_metrics

    if rgb.shape != target.shape:
        raise ValueError("RGB geometry differs")
    lps, dists = _rgb_metrics(rgb, target, lp, ds, metric_batch)
    l1 = F.l1_loss(rgb.float(), target.float())
    temporal = F.l1_loss(rgb[:, :, 1:] - rgb[:, :, :-1], target[:, :, 1:] - target[:, :, :-1])
    latent = F.smooth_l1_loss(restored.float(), clean.float())
    loss = 0.1 * l1 + lps + dists + 0.05 * temporal + 0.1 * latent
    return loss, dict(rgb_l1=l1, lpips=lps, dists=dists, temporal=temporal, latent=latent)
