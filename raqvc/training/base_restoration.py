"""H2 fixed-time unit-bridge connector; legacy state names retained."""

import math
from contextlib import nullcontext
from pathlib import Path
import torch
from torch import nn
from torch.nn import functional as F

QPS = (0, 16, 39, 63)


class SelectionHeads(nn.Module):
    def __init__(self, legacy, bases, arm):
        super().__init__()
        if arm != "E":
            raise ValueError("the cleaned runtime only supports the H2/E unit head")
        self.arm = "E"
        self.velocity_scale_mode = "unit"
        self.use_tau = False
        self.legacy = legacy.requires_grad_(False)
        self.register_buffer("bases", bases.detach().float().clone())

        def branch():
            n = nn.Sequential(
                nn.Conv3d(54, 64, 1),
                nn.SiLU(),
                nn.Conv3d(64, 64, (1, 3, 3), padding=(0, 1, 1)),
                nn.SiLU(),
                nn.Conv3d(64, 1, 1),
            )
            nn.init.zeros_(n[-1].weight)
            nn.init.zeros_(n[-1].bias)
            return n

        self.gain_global = nn.Parameter(torch.zeros(4))
        self.gain_local = branch()

    def conditions(self, c):
        # Preserve the parent's ambient-autocast estimate path. Switching its
        # frozen convolutions to FP32 would change even zero-increment outputs.
        with nullcontext() if getattr(self, "condition_grad_enabled", False) else torch.no_grad():
            p, r, s, _ = self.legacy.estimate(c.source.float(), c.sigma.float(), c.width.float(), c.qp)
            src = F.adaptive_avg_pool3d(c.source.float(), s.shape[2:])
            q = torch.full_like(s, c.qp / 63.0)
            ip = torch.zeros_like(s)
            ip[:, :, :1] = 1
            features = torch.cat((src, p.float().log(), c.sigma.float().log(), c.width.float().log(), q, ip), 1)
        qi = QPS.index(c.qp)
        with torch.autocast(c.source.device.type, enabled=False):
            gain_log = math.log(2) * torch.tanh(self.gain_global[qi] + self.gain_local(features))
            delta = torch.zeros_like(s)  # fixed Wan time; historical placeholders are never read
            tau = (self.bases[qi] + delta).clamp(10, 990).flatten(1)
        return p, r, s, tau, gain_log


def endpoint(model, heads, c):
    p, r, s, tau, gain_log = heads.conditions(c)
    bridge = torch.ones(c.source.shape[0], device=c.source.device, dtype=c.source.dtype)
    raw = model(c.source, qp=c.qp, log_q_y=c.log_q_y, log_q_z=c.log_q_z, actual_bpp=c.actual_bpp)
    v = model.path.denormalize_velocity(raw, c.qp)
    v = ((c.source + v) - c.source).float()
    with torch.autocast(c.source.device.type, enabled=False):
        direction = heads.legacy.direction(c.source.float(), v, p, r, c.qp)
        parent_gain = heads.legacy.gain(c.source.float(), direction, p, s, c.qp)
        gain = F.interpolate(
            parent_gain * gain_log.exp(), size=c.source.shape[2:], mode="trilinear", align_corners=False
        )
        output = c.source.float() + gain * direction
    return output, dict(
        output=output,
        tau=tau,
        tau_delta=tau - heads.bases[QPS.index(c.qp)],
        rhat=p.square().mean(1, keepdim=True).sqrt(),
        gain_log_delta=gain_log,
        gain=gain,
        bridge=bridge,
        velocity_rms=v.square().mean().sqrt(),
        direction_rms=direction.square().mean().sqrt(),
        update_rms=(output - c.source.float()).square().mean().sqrt(),
    )


def lora_state(model):
    return {k: v.detach().cpu().clone() for k, v in model.state_dict().items() if k.endswith((".lora_A", ".lora_B"))}
