"""Single shared endpoint for final-candidate training and native evaluation."""

import math
import torch
from torch import nn
import torch.nn.functional as F
from raqvc.training.base_restoration import endpoint as e_endpoint, lora_state
from raqvc.training.residual_heads import strength_target


class FinalHeads(nn.Module):

    def __init__(self, base, arm="H2_old", *, strength=None, direction=None):
        super().__init__()
        self.base = base
        self.arm = arm
        if arm != "H2_old" or strength is None or direction is None:
            raise ValueError("the cleaned runtime only supports state-reconstructed H2_old heads")
        self.level, self.time = (2, "old")
        self.strength = strength
        self.direction = direction
        self.base.use_tau = False
        self.base.velocity_scale_mode = "unit"
        self.base.bases.copy_(self.base.bases.new_tensor([520, 640, 800, 960]))
        self.extra_gain = nn.Sequential(
            nn.Conv3d(51, 64, 1),
            nn.SiLU(),
            nn.Conv3d(64, 64, (1, 3, 3), padding=(0, 1, 1)),
            nn.SiLU(),
            nn.Conv3d(64, 1, 1),
        )
        nn.init.zeros_(self.extra_gain[-1].weight)
        nn.init.zeros_(self.extra_gain[-1].bias)
        self.direction_gate = nn.Parameter(torch.zeros(2))


def endpoint(model, heads, c):
    if heads.base.velocity_scale_mode != "unit" or heads.base.use_tau:
        raise RuntimeError("final candidate requires unit bridge and fixed tau")
    if not model.wan_native_timestep_bridge:
        raise RuntimeError("Wan native bridge disabled")
    expected = float(heads.base.bases[(0, 16, 39, 63).index(c.qp)])
    observed = []

    def check_time(m, inputs, out):
        t = inputs[1].detach().float()
        if not torch.allclose(t, torch.full_like(t, expected)):
            raise RuntimeError("actual Wan timestep differs")
        observed.append((float(t.min()), float(t.max())))

    handle = model.backbone.register_forward_hook(check_time)
    try:
        base, aux = e_endpoint(model, heads.base, c)
    finally:
        handle.remove()
    if len(observed) != 1:
        raise RuntimeError("Wan backbone was not called")
    aux.update(actual_wan_tau_min=observed[0][0], actual_wan_tau_max=observed[0][1])
    with torch.autocast(c.source.device.type, enabled=False):
        rhat = heads.strength.strength(c.source.float(), c.sigma.float(), c.width.float(), c.qp)
        context = F.adaptive_avg_pool3d(c.source.float(), rhat.shape[2:])
        ip = torch.zeros_like(rhat)
        ip[:, :, 0] = 1
        gain_delta = math.log(2) * torch.tanh(
            heads.extra_gain(
                torch.cat([context, rhat.detach().clamp_min(1e-08).log(), torch.full_like(ip, c.qp / 63.0), ip], 1)
            )
        )
        gain_delta = F.interpolate(gain_delta, size=c.source.shape[2:], mode="trilinear", align_corners=False)
        u0 = base.float() - c.source.float()
        norm = u0.square().sum(1, keepdim=True).sqrt()
        e0 = u0 / norm.clamp_min(1e-08)
        unit = e0
        dn = heads.direction.direction(c.source.float(), u0, c.sigma, c.width, rhat.detach(), c.qp)
        beta = torch.ones_like(norm) * heads.direction_gate[1].tanh()
        beta[:, :, :1] = heads.direction_gate[0].tanh()
        unit = F.normalize(e0 + beta * (dn - e0), dim=1, eps=1e-08)
        delta = norm * gain_delta.exp() * unit - norm * e0
        delta = torch.where(norm > 1e-08, delta, torch.zeros_like(delta))
        out = base.float() + delta
        aux.update(
            output=out,
            rhat=rhat,
            gain=aux["gain"] * gain_delta.exp(),
            extra_gain_log=gain_delta,
            update_rms=(out - c.source.float()).square().mean().sqrt(),
            final_unit=unit,
        )
    return (out, aux)


def auxiliary_loss(heads, aux, clean, source):
    zero = aux["output"].sum() * 0
    target = strength_target(clean - source).detach()
    pred = aux["rhat"].clamp_min(1e-08)
    losses = []
    for sl in [slice(0, 1), slice(1, None)]:
        x = pred[:, :, sl].log()
        y = target[:, :, sl].log()
        diff = y - y.roll(1, -1)
        mask = diff.abs() > 0.05
        order = F.relu(0.02 - diff.sign() * (x - x.roll(1, -1)))
        rank = order[mask].mean() if mask.any() else x.sum() * 0
        energy = F.smooth_l1_loss(
            pred[:, :, sl].square().mean((-2, -1)).clamp_min(1e-12).log(),
            target[:, :, sl].square().mean((-2, -1)).clamp_min(1e-12).log(),
        )
        losses.append(F.smooth_l1_loss(x, y) + 0.1 * rank + 0.05 * energy)
    strength = torch.stack(losses).mean()
    direction = zero
    r = (clean - source).detach().float()
    valid = r.square().sum(1) > 1e-08
    cos = 1 - F.cosine_similarity(aux["final_unit"], r, dim=1)
    dd = []
    for sl in [slice(0, 1), slice(1, None)]:
        m = valid[:, sl]
        dd.append(cos[:, sl][m].mean() if m.any() else zero)
    direction = torch.stack(dd).mean()
    return (0.1 * strength + 0.01 * direction, dict(strength_aux=strength, direction_aux=direction))
