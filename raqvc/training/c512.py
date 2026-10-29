"""H2 online joint runtime and complete mutable-weight inference bundles.

The pretrained Wan/Stage3 base is shared and immutable. Every tensor which
this training can change is stored in each bundle, including frozen groups
from earlier phases. The inference endpoint is the original H2 endpoint.
"""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Mapping

import torch
from torch import nn

from raqvc.training.restoration import (
    FinalHeads,
    endpoint,
)
from raqvc.training.base_heads import Task938CorrectedHeads
from raqvc.training.base_restoration import SelectionHeads
from raqvc.training.residual_heads import ResidualStrengthHead, DirectionRefinementHead

ART = Path(__import__("os").environ.get("RAQVC_ARTIFACTS", "/qzwang/data/raqvc_artifacts"))
DEFAULT_PARENT = ART / "C512_FINAL_v1/model"
DEFAULT_CFG = Path(__file__).resolve().parents[2] / "configs/codec.yaml"
DEFAULT_WAN = Path(__import__("os").environ.get("RAQVC_WAN", str(DEFAULT_PARENT / "wan")))
DEFAULT_CHECKPOINT = DEFAULT_PARENT / "C512_original_bundle.pt"
QPS = (0, 16, 39, 63)
TAU_TABLE = (520.0, 640.0, 800.0, 960.0)
SCHEMA = "task938_h2_joint_v1"
PHASE_LRS = {
    "A": dict(
        codec_backbone=0.0, regional=0.0, strength=2e-5, gain=3e-5, direction=1e-5, legacy_adapter=0.0, lora=5e-6
    ),
    "B": dict(
        codec_backbone=2e-6, regional=1e-5, strength=3e-5, gain=3e-5, direction=1e-5, legacy_adapter=5e-6, lora=1e-5
    ),
    "C": dict(
        codec_backbone=1e-6, regional=5e-6, strength=2e-5, gain=2e-5, direction=1e-5, legacy_adapter=3e-6, lora=5e-6
    ),
    "D": dict(
        codec_backbone=2e-7, regional=1e-6, strength=4e-6, gain=4e-6, direction=2e-6, legacy_adapter=6e-7, lora=1e-6
    ),
}
_ADAPTER_PREFIXES = (
    "backbone.codec_context_mlp.",
    "backbone.spatial_anchor_fusion.",
    "backbone.perceptual_adapter.",
    "backbone.rate_mlp.",
    "backbone.transition_mlp.",
    "backbone.restoration_gate.",
    "solver_step_calibrator.",
)


def _is_lora(name):
    return name.endswith((".lora_A", ".lora_B"))


def _is_adaptation(name):
    return _is_lora(name) or name.startswith(_ADAPTER_PREFIXES) or name == "backbone.velocity_scale"


def _cpu_state(state):
    return {name: value.detach().cpu().clone() for name, value in state.items()}


def invalidate_entropy_coders(codec):
    """Invalidate derived CDFs after optimizer updates or checkpoint restore."""
    for module in codec.modules():
        if hasattr(module, "_entropy_coder_ready"):
            module._entropy_coder_ready = False


def _active_heads(state):
    """Drop only explicitly identified inactive tensors in historical C512 files."""
    retired = (
        "base.tau_global",
        "base.tau_local.",
        "strength.direction_",
        "direction.strength_",
        "direction.log_calibration",
        "direction.feature_mean",
        "direction.feature_std",
    )
    return {k: v for k, v in state.items() if not k.startswith(retired)}


def _heads_from_state(state):
    """Rebuild the saved H2 shape without loading historical predictor files."""
    prefix = "base.legacy."
    legacy = Task938CorrectedHeads(
        error_channels=int(state[prefix + "error.project.0.weight"].shape[0]),
        gain_channels=int(state[prefix + "gain_body.0.weight"].shape[0]),
        feature_channels=int(state[prefix + "error.project.0.weight"].shape[1]),
    )
    base = SelectionHeads(legacy, state["base.bases"], "E")
    # FinalHeads.__init__ reads old strength/direction files. The architecture
    # below is identical, with its dimensions derived from this bundle.
    heads = FinalHeads.__new__(FinalHeads)
    nn.Module.__init__(heads)
    heads.base = base
    heads.arm, heads.level, heads.time = "H2_old", 2, "old"
    heads.strength = ResidualStrengthHead(state["strength.feature_mean"], state["strength.feature_std"])
    heads.direction = DirectionRefinementHead()
    hidden = int(state["extra_gain.0.weight"].shape[0])
    heads.extra_gain = nn.Sequential(
        nn.Conv3d(51, hidden, 1),
        nn.SiLU(),
        nn.Conv3d(hidden, hidden, (1, 3, 3), padding=(0, 1, 1)),
        nn.SiLU(),
        nn.Conv3d(hidden, 1, 1),
    )
    heads.direction_gate = nn.Parameter(torch.zeros_like(state["direction_gate"]))
    heads.load_state_dict(_active_heads(state), strict=True)
    return heads


def _validate_contract(heads, meta):
    if meta.get("arm") != "H2_old" or heads.arm != "H2_old":
        raise ValueError("joint runtime requires the H2_old architecture")
    if heads.base.velocity_scale_mode != "unit" or heads.base.use_tau:
        raise ValueError("joint runtime requires unit output and fixed time")
    if list(meta.get("tau_table", ())) != list(TAU_TABLE):
        raise ValueError("checkpoint must retain the original H2 timestep table")
    if heads.base.bases.detach().cpu().tolist() != list(TAU_TABLE):
        raise ValueError("head timestep values disagree with H2_old")
    if float(meta.get("noise_alpha", 0.0)) != 0.0:
        raise ValueError("joint runtime is noise free")


def _active_meta(meta, heads):
    result = copy.deepcopy(meta)
    if result.get("schema") != SCHEMA:
        result["historical_head_contract"] = result.get("head_contract")
        result["historical_parent_run_contract"] = result.get("run_contract")
    result.update(
        schema=SCHEMA,
        runtime="h2_joint",
        arm="H2_old",
        velocity_scale_mode="unit",
        tau_enabled=False,
        tau_table=list(TAU_TABLE),
        noise_alpha=0.0,
        actual_lora_rank=8,
        model_base_checkpoint=str(DEFAULT_PARENT / "stage3_dit_reproduction.pt"),
        parent_dir=str(DEFAULT_PARENT),
        wan_dir=str(DEFAULT_WAN),
        codec_checkpoint="embedded:codec",
        head_contract=dict(
            schema="task938_h2_joint_heads_v1",
            bridge_amplitude="unit",
            tau_trainable=False,
            condition_grad_enabled=bool(getattr(heads.base, "condition_grad_enabled", False)),
            strength_reconstruction_gradient=False,
            dit_calls_per_gop=1,
            restore_i=True,
            lora_rank=8,
        ),
    )
    result["run_contract"] = dict(
        schema=SCHEMA,
        endpoint="raqvc.training.restoration.endpoint",
        codec_state="embedded",
        adapter_state="model_adaptation",
        task938_time_bridge="current_token_tau",
        output_bridge="unit",
        tau_table=list(TAU_TABLE),
        noise_alpha=0.0,
        solver_steps=1,
        restore_i=True,
        task938_stage0=dict(rank=8, alpha=8.0, time_mode="scalar"),
        condition_grad_enabled=bool(getattr(heads.base, "condition_grad_enabled", False)),
    )
    return result


def restore_bundle_state(model, heads, codec, payload):
    """Strictly restore mutable weights into already allocated model objects."""
    if payload.get("meta", {}).get("schema") != SCHEMA:
        raise ValueError("restore_bundle_state requires a complete H2 joint bundle")
    expected = {name for name in model.state_dict() if _is_adaptation(name)}
    state = payload["model_adaptation"]
    if set(state) != expected:
        raise ValueError(
            f"adaptation keys differ: missing={sorted(expected-set(state))}, "
            f"unexpected={sorted(set(state)-expected)}"
        )
    result = model.load_state_dict(state, strict=False)
    if result.unexpected_keys:
        raise ValueError(result)
    heads.load_state_dict(_active_heads(payload["heads"]), strict=True)
    codec.load_state_dict(payload["codec"], strict=True)
    invalidate_entropy_coders(codec)
    if hasattr(codec, "reset_temporal_state"):
        codec.reset_temporal_state()
    _validate_contract(heads, payload["meta"])
    return copy.deepcopy(payload["meta"])


def load_bundle(checkpoint=DEFAULT_CHECKPOINT, device="cpu"):
    """Load old H2 initialization or a bundle with its own updated codec."""
    from omegaconf import OmegaConf
    from raqvc.codec import create_latent_codec
    from raqvc.training.codec_training import load_regional_codec
    from raqvc.training.model_loader import load_mainline_dit

    checkpoint = Path(checkpoint)
    device = torch.device(device)
    saved = torch.load(checkpoint, map_location="cpu", mmap=True, weights_only=False)
    meta = saved.get("meta", {})
    if meta.get("schema") == SCHEMA:
        if not isinstance(meta.get("codec_config_dict"), Mapping):
            raise ValueError("joint bundle must embed its codec configuration")
        # Recreate the same immutable Stage3 rank16 merge and rank8 modules
        # used by the legacy H2 loader, then restore every trained increment.
        model, _ = load_mainline_dit(DEFAULT_PARENT / "stage3_dit_reproduction.pt", DEFAULT_WAN, torch.device(device))
        heads = _heads_from_state(saved["heads"]).to(device)
        kwargs = copy.deepcopy(dict(meta["codec_config_dict"]))
        kind = kwargs.pop("type")
        kwargs.pop("enabled", None)
        kwargs["device"] = str(device)
        codec = create_latent_codec(kind.replace("-", "_"), **kwargs).to(device).float()
        codec.enable_entropy_scale_export(True)
        restore_bundle_state(model, heads, codec, saved)
    else:
        raise ValueError("unsupported H2 checkpoint schema")
    _validate_contract(heads, meta)
    ranks = {int(p.shape[0]) for name, p in model.named_parameters() if name.endswith(".lora_A")}
    if ranks != {8}:
        raise ValueError(f"H2 joint runtime requires rank8 LoRA, got {ranks}")
    if not getattr(model, "wan_native_timestep_bridge", False):
        raise ValueError("H2 requires the Wan native timestep bridge")
    model._stage3_pixel_calibration = copy.deepcopy(meta.get("calibration", {}))
    for module in (model, heads, codec):
        module.eval().requires_grad_(False)
    heads.base.condition_grad_enabled = False
    heads.base.legacy.condition_grad_enabled = False
    invalidate_entropy_coders(codec)
    return model, heads, codec, _active_meta(meta, heads)


def configure_groups(model, heads, codec, phase):
    """Select disjoint active parameters and return JSON-friendly inventory."""
    phase = str(phase).upper()
    if phase not in PHASE_LRS:
        raise ValueError(f"unknown H2 phase: {phase}")
    rates = PHASE_LRS[phase]
    for module in (model, heads, codec):
        module.requires_grad_(False)
    heads.base.condition_grad_enabled = phase != "A"
    heads.base.legacy.condition_grad_enabled = phase != "A"
    selected = {name: [] for name in rates}
    for name, parameter in model.named_parameters():
        if _is_lora(name):
            selected["lora"].append(("model." + name, parameter))
        elif _is_adaptation(name):
            selected["legacy_adapter"].append(("model." + name, parameter))
    for name, parameter in heads.named_parameters():
        if name == "base.gain_global" or name.startswith(("base.gain_local.", "extra_gain.")):
            key = "gain"
        elif name.startswith("strength.") and not name.startswith("strength.direction_"):
            key = "strength"
        elif name == "direction_gate" or name.startswith("direction.direction_"):
            key = "direction"
        elif name.startswith("base.legacy.") and ".tau_" not in name:
            key = "legacy_adapter"
        else:
            continue
        selected[key].append(("heads." + name, parameter))
    dormant = getattr(codec, "_is_dormant_adapted_q_parameter", lambda name: False)
    for name, parameter in codec.named_parameters():
        if dormant(name):
            continue
        if name in ("latent_scale", "latent_bias") and not getattr(codec, "_normalize_latent", False):
            continue
        p_codec = getattr(codec, "p_codec", None)
        if (
            name.startswith("p_codec.model.feature_adaptor_p.")
            and getattr(codec, "_p_reference_mode", None) == "latent"
            and getattr(p_codec, "p_decoded_feature_fusion", None) is None
            and getattr(p_codec, "p_htcm_long_fusion", None) is None
        ):
            continue
        key = "regional" if ".task938_region_quantizer." in name else "codec_backbone"
        selected[key].append(("codec." + name, parameter))
    groups, inventory, seen = [], {}, set()
    for name, entries in selected.items():
        if not rates[name]:
            continue
        if not entries:
            raise ValueError(f"empty active H2 parameter group: {name}")
        for parameter_name, parameter in entries:
            if id(parameter) in seen:
                raise ValueError("duplicate optimizer parameter: " + parameter_name)
            seen.add(id(parameter))
            parameter.data = parameter.data.float()
            parameter.requires_grad_(True)
        params = [p for _, p in entries]
        groups.append(dict(name=name, params=params, lr=rates[name], base_lr=rates[name]))
        inventory[name] = dict(
            lr=rates[name],
            parameter_count=sum(p.numel() for p in params),
            tensor_count=len(params),
            names=[n for n, _ in entries],
        )
    return groups, inventory


def bundle_state(model, heads, codec, meta):
    """Snapshot the entire mutable inference state, independent of phase."""
    meta = _active_meta(meta, heads)
    _validate_contract(heads, meta)
    if not isinstance(meta.get("codec_config_dict"), Mapping):
        raise ValueError("codec configuration is required in a joint bundle")
    return dict(
        meta=meta,
        codec=_cpu_state(codec.state_dict()),
        heads=_cpu_state(heads.state_dict()),
        model_adaptation=_cpu_state(
            {name: value for name, value in model.state_dict().items() if _is_adaptation(name)}
        ),
    )
