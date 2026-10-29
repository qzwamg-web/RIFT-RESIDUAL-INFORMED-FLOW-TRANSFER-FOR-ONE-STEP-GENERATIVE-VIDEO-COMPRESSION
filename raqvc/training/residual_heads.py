"""Research-only full-residual RMS and unit-direction heads; no LF/HF semantics."""

import torch
from torch import nn
import torch.nn.functional as F
from raqvc.training.multiscale_error import decoder_features


class SpatialContext(nn.Module):

    def __init__(self, inputs, width=64):
        super().__init__()
        self.project = nn.Conv2d(inputs, width, 1)
        self.local = nn.Sequential(
            nn.Conv2d(width, width, 3, padding=1), nn.SiLU(), nn.Conv2d(width, width, 3, padding=1)
        )
        self.mid = nn.Sequential(
            nn.Conv2d(width, width, 3, padding=1), nn.SiLU(), nn.Conv2d(width, width, 3, padding=1)
        )
        self.fuse = nn.Sequential(
            nn.Conv2d(width * 3, width, 1), nn.SiLU(), nn.Conv2d(width, width, 3, padding=1), nn.SiLU()
        )

    def forward(self, x):
        b, c, t, h, w = x.shape
        x = x.permute(0, 2, 1, 3, 4).reshape(b * t, c, h, w)
        x = F.silu(self.project(x))
        local = self.local(x)
        mid = self.mid(F.adaptive_avg_pool2d(x, (max(1, h // 2), max(1, w // 2))))
        mid = F.interpolate(mid, size=(h, w), mode="bilinear", align_corners=False)
        y = self.fuse(torch.cat((local, mid, x.mean((-2, -1), keepdim=True).expand_as(x)), 1))
        return y.reshape(b, t, -1, h, w).permute(0, 2, 1, 3, 4)


def strength_target(residual):
    return F.avg_pool3d(residual.float().square().mean(1, keepdim=True), (1, 2, 2)).clamp_min(1e-12).sqrt()


class ResidualStrengthHead(nn.Module):

    def __init__(self, feature_mean, feature_std):
        super().__init__()
        self.register_buffer("feature_mean", feature_mean[:, :100].clone().float())
        self.register_buffer("feature_std", feature_std[:, :100].clone().float())
        self.strength_context = SpatialContext(101)
        self.strength_ip = nn.ModuleList([nn.Conv3d(64, 1, 1), nn.Conv3d(64, 1, 1)])
        self.log_calibration = nn.Parameter(torch.zeros(4, 2))

    def strength(self, source, sigma, width, qp):
        features = decoder_features(source, sigma, width, qp)
        features = ((features - self.feature_mean) / self.feature_std.clamp_min(0.05)).clamp(-8, 8)
        pos = torch.linspace(0, 1, source.shape[2], device=source.device).view(1, 1, -1, 1, 1).expand_as(sigma)
        ctx = self.strength_context(torch.cat((features, pos), 1))
        ip = torch.zeros_like(sigma)
        ip[:, :, 0] = 1
        qi = (0, 16, 39, 63).index(qp)
        delta = ip * self.strength_ip[0](ctx) + (1 - ip) * self.strength_ip[1](ctx)
        calibration = ip * self.log_calibration[qi, 0] + (1 - ip) * self.log_calibration[qi, 1]
        return (sigma.clamp_min(1e-08).log() + calibration + 3 * torch.tanh(delta / 3)).clamp(-12, 5).exp()


class DirectionRefinementHead(nn.Module):

    def __init__(self):
        super().__init__()
        self.direction_context = SpatialContext(102)
        self.direction_ip = nn.ModuleList([nn.Conv3d(64, 48, 1), nn.Conv3d(64, 48, 1)])

    def direction(self, source, base_direction, sigma, width, strength, qp):
        base = F.normalize(base_direction.detach().float(), dim=1, eps=1e-08)

        def up(x):
            return F.interpolate(x.detach().float(), size=source.shape[2:], mode="nearest")

        scalar = up(sigma)
        ip = torch.zeros_like(scalar)
        ip[:, :, 0] = 1
        pos = torch.linspace(0, 1, source.shape[2], device=source.device).view(1, 1, -1, 1, 1).expand_as(scalar)
        features = torch.cat(
            (
                source.detach().float().clamp(-8, 8),
                base,
                up(sigma).clamp_min(1e-08).log(),
                up(width).clamp_min(1e-08).log(),
                up(strength).clamp_min(1e-08).log(),
                torch.full_like(ip, qp / 63),
                ip,
                pos,
            ),
            1,
        )
        ctx = self.direction_context(features)
        correction = ip * self.direction_ip[0](ctx) + (1 - ip) * self.direction_ip[1](ctx)
        return F.normalize(base + correction, dim=1, eps=1e-08)
