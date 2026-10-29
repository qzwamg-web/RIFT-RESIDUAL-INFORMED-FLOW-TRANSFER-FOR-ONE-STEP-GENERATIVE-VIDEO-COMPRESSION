"""Multiscale full-band residual-strength prediction from decoder-visible features.

The two output channels are complementary receptive-field estimates (local and
contextual), not a hard low/high-frequency decomposition.  Their squared RMS is
used as the full-band quantization-residual strength.  This distinction matters
at very low rates, where both low-frequency structure and texture can be lost.
"""

import torch
from torch import nn
import torch.nn.functional as F


def decoder_features(source, sigma, width, qp):
    if source.ndim != 5 or source.shape[1] != 48:
        raise ValueError("expected B48THW")
    if (
        tuple(sigma.shape) != (source.shape[0], 1, source.shape[2], source.shape[3] // 2, source.shape[4] // 2)
        or width.shape != sigma.shape
    ):
        raise ValueError("map geometry")
    if source.shape[-1] % 2 or source.shape[-2] % 2 or (sigma <= 0).any() or (width <= 0).any():
        raise ValueError("invalid decoder features")
    x = source.detach().float()
    mean = F.avg_pool3d(x, (1, 2, 2))
    var = (F.avg_pool3d(x.square(), (1, 2, 2)) - mean.square()).clamp_min(0)
    q = torch.full_like(sigma, float(qp) / 63)
    ip = torch.zeros_like(q)
    ip[:, :, 0] = 1
    return torch.cat([mean, (var + 1e-8).sqrt(), sigma.float().log(), width.float().log(), q, ip], 1)


class Residual(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.net = nn.Sequential(nn.Conv2d(c, c, 3, padding=1), nn.SiLU(), nn.Conv2d(c, c, 3, padding=1))

    def forward(self, x):
        return F.silu(x + self.net(x))


class MultiScaleErrorHead(nn.Module):
    def __init__(self, channels=64, ip_adapters=False, feature_channels=100):
        super().__init__()
        c = int(channels)
        self.channels = c
        self.feature_channels = int(feature_channels)
        if self.feature_channels != 100:
            raise ValueError("C512 uses exactly 100 decoder features")
        self.register_buffer("feature_mean", torch.zeros(1, self.feature_channels, 1, 1, 1))
        self.register_buffer("feature_std", torch.ones(1, self.feature_channels, 1, 1, 1))
        self.project = nn.Sequential(nn.Conv2d(self.feature_channels, c, 1), nn.SiLU())
        self.local = Residual(c)
        self.mid = Residual(c)
        self.global_context = nn.Sequential(nn.Conv2d(c, c, 1), nn.SiLU(), nn.Conv2d(c, c, 1))
        # Two complementary spatial scales.  They are deliberately not named
        # LF/HF: a low-rate quantizer can damage both bands at once.
        self.fuse = nn.Sequential(
            nn.Conv2d(3 * c, c, 1), nn.SiLU(), nn.Conv2d(c, c, 3, padding=1), nn.SiLU(), nn.Conv2d(c, 2, 1)
        )
        self.ip_adapters = nn.ModuleList()
        if ip_adapters:
            for _ in range(2):
                adapter = nn.Sequential(nn.Conv2d(c, 16, 1), nn.SiLU(), nn.Conv2d(16, c, 1))
                nn.init.zeros_(adapter[-1].weight)
                nn.init.zeros_(adapter[-1].bias)
                self.ip_adapters.append(adapter)
        nn.init.normal_(self.fuse[-1].weight, std=0.001)
        nn.init.constant_(self.fuse[-1].bias, -2.0)

    def forward_with_features(self, features):
        if features.ndim != 5 or features.shape[1] != self.feature_channels:
            raise ValueError(f"expected {self.feature_channels} features")
        b, c, t, h, w = features.shape
        x = ((features.float() - self.feature_mean) / self.feature_std.clamp_min(0.05)).clamp(-8, 8)
        x = x.permute(0, 2, 1, 3, 4).reshape(b * t, c, h, w).to(self.project[0].weight.dtype)
        z = self.project(x)
        local = self.local(z)
        mid = self.mid(F.adaptive_avg_pool2d(z, (max(1, h // 2), max(1, w // 2))))
        mid = F.interpolate(mid, size=(h, w), mode="bilinear", align_corners=False)
        global_ = self.global_context(z.mean((-2, -1), keepdim=True)).expand_as(z)
        shared = self.fuse[:-1](torch.cat([local, mid, global_], 1))
        if len(self.ip_adapters):
            # Feature layout is [mean(48), rms(48), sigma, width, qp, ip,
            # alpha, zero_mask].  After adding alpha/zero_mask, the last
            # channel is no longer the I/P flag.  Reading it here silently
            # routes every zero-symbol region through the I adapter, which
            # corrupts the I/P separation while all tensor shapes still pass.
            is_i = features[:, 99:100].permute(0, 2, 1, 3, 4).reshape(b * t, 1, h, w).to(shared.dtype)
            shared = shared + is_i * self.ip_adapters[0](shared) + (1 - is_i) * self.ip_adapters[1](shared)
        y = self.fuse[-1](shared)
        return y.reshape(b, t, 2, h, w).permute(0, 2, 1, 3, 4), shared.reshape(b, t, self.channels, h, w).permute(
            0, 2, 1, 3, 4
        )

    def forward(self, features):
        return self.forward_with_features(features)[0]

    def predict_with_features(self, source, sigma, width, qp):
        logits, shared = self.forward_with_features(decoder_features(source, sigma, width, qp))
        return F.softplus(logits) + 1e-6, shared

    def predict(self, source, sigma, width, qp):
        return self.predict_with_features(source, sigma, width, qp)[0]
