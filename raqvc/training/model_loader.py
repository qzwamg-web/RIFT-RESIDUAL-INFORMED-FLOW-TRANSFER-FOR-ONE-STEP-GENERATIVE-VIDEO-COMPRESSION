"""Import the immutable Stage3 base and install RIFT rank-8 LoRA.

Dropped historical tensors belong exclusively to unused cache writers/clocks.
All active tensors are required; an unknown key is an error.
"""

from pathlib import Path
import torch
from raqvc.backbone.model_factory import build_task438_windowed_dit
from raqvc.training.lora import install_stage0

INACTIVE_PREFIXES = ("cache_writer.", "wan_clock.", "task938_time_mapper.", "backbone.flow_mlp.")


def load_mainline_dit(checkpoint, wan_dir, device):
    payload = torch.load(checkpoint, map_location="cpu", mmap=True, weights_only=False)
    model = build_task438_windowed_dit(wan_dir, device)
    state = payload["model_state_dict"]
    architecture = payload.get("architecture", {})
    old_lora = [k for k in state if k.endswith(".lora_A")]
    rank = int(state[old_lora[0]].shape[0])
    alpha = float(architecture.get("stage0", {}).get("alpha", rank))
    filtered = {
        k: v
        for k, v in state.items()
        if not k.startswith(INACTIVE_PREFIXES)
        and not k.endswith(".residual_memory_gate_raw")
        and not k.endswith((".lora_A", ".lora_B"))
    }
    for key in old_lora:
        prefix = key[: -len(".lora_A")]
        update = (state[prefix + ".lora_B"].float() @ state[key].float()) * (alpha / rank)
        base = filtered[prefix + ".base.weight"]
        filtered[prefix + ".base.weight"] = base.float() + update.to(base)
    install_stage0(model)
    missing, unexpected = model.load_state_dict(filtered, strict=False)
    allowed = {k for k in model.state_dict() if k.endswith((".lora_A", ".lora_B"))}
    if set(missing) != allowed or unexpected:
        raise RuntimeError(f"Active backbone state mismatch: missing={missing}, unexpected={unexpected}")
    return model, payload
