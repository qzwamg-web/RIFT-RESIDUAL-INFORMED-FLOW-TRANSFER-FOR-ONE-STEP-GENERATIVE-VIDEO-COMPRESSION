"""Inherited C512 error, scale correction, direction and token gain heads."""

from __future__ import annotations
import torch
import torch.nn.functional as F
from torch import Tensor, nn
from raqvc.training.multiscale_error import MultiScaleErrorHead, decoder_features


class Task938CorrectedHeads(nn.Module):

    def __init__(self, *, error_channels=128, gain_channels=128, feature_channels=100):
        super().__init__()
        self.error = MultiScaleErrorHead(error_channels, ip_adapters=True, feature_channels=feature_channels)
        self.error_channels = int(error_channels)
        self.feature_channels = int(feature_channels)
        self.register_buffer("scale_reference", torch.ones(1))
        self.register_buffer("error_reference", torch.ones(1))
        self.scale_residual = nn.Sequential(nn.Conv3d(error_channels + 2, 128, 1), nn.SiLU(), nn.Conv3d(128, 2, 1))
        nn.init.zeros_(self.scale_residual[-1].weight)
        nn.init.zeros_(self.scale_residual[-1].bias)
        self.direction_shared = nn.Sequential(
            nn.Conv3d(48 + 48 + 2 + 2 + 2, 128, 1),
            nn.SiLU(),
            nn.Conv3d(128, 128, (1, 3, 3), padding=(0, 1, 1)),
            nn.SiLU(),
        )
        self.direction_i = nn.Conv3d(128, 48, 1)
        self.direction_p = nn.Conv3d(128, 48, 1)
        for layer in (self.direction_i, self.direction_p):
            nn.init.zeros_(layer.weight)
            nn.init.zeros_(layer.bias)
        self.gain_body = nn.Sequential(
            nn.Conv3d(48 + 48 + 1 + 2 + 1 + 1, gain_channels, 1),
            nn.SiLU(),
            nn.Conv3d(gain_channels, gain_channels, (1, 3, 3), padding=(0, 1, 1)),
            nn.SiLU(),
        )
        self.gain_residual = nn.Conv3d(gain_channels, 1, 1)
        nn.init.zeros_(self.gain_residual.weight)
        nn.init.zeros_(self.gain_residual.bias)
        self.register_buffer("gain_table", torch.tensor([0.25, 0.2, 0.15, 0.1]))

    def estimate(self, source: Tensor, sigma: Tensor, width: Tensor, qp: int):
        """Return prediction, scale residual and a shared positive strength."""
        prediction, shared = self.error.predict_with_features(source, sigma, width, qp)
        pred_strength = prediction.square().mean(1, keepdim=True).sqrt()
        codec_strength = sigma.float()
        common = (
            (
                codec_strength
                / self.scale_reference.clamp_min(1e-06)
                * (pred_strength / self.error_reference.clamp_min(1e-06))
            )
            .clamp_min(1e-08)
            .sqrt()
        )
        shared_up = F.interpolate(shared, size=common.shape[2:], mode="trilinear", align_corners=False)
        residual = self.scale_residual(torch.cat((shared_up, prediction.log()), 1))
        corrected = common * torch.exp(0.25 * residual.mean(1, keepdim=True)).clamp(0.5, 2.0)
        return (prediction, residual, corrected, shared)

    @torch.no_grad()
    def set_references(self, sigma_mean: Tensor, error_mean: Tensor) -> None:
        if sigma_mean.numel() != 1 or error_mean.numel() != 1:
            raise ValueError("references must be scalar")
        if not torch.isfinite(sigma_mean + error_mean) or bool((sigma_mean <= 0) | (error_mean <= 0)):
            raise ValueError("references must be finite and positive")
        self.scale_reference.copy_(sigma_mean.detach().float().reshape_as(self.scale_reference))
        self.error_reference.copy_(error_mean.detach().float().reshape_as(self.error_reference))

    def direction(
        self, source: Tensor, velocity: Tensor, prediction: Tensor, scale_residual: Tensor, qp: int
    ) -> Tensor:
        pred = F.interpolate(prediction.float(), size=source.shape[2:], mode="nearest")
        resid = F.interpolate(scale_residual.float(), size=source.shape[2:], mode="nearest")
        q = torch.full_like(source[:, :1], float(qp) / 63.0)
        ip = torch.zeros_like(q)
        ip[:, :, :1] = 1.0
        if not getattr(self, "condition_grad_enabled", False):
            pred, resid = (pred.detach(), resid.detach())
        z = self.direction_shared(torch.cat((source.detach(), velocity.detach(), pred, resid, q, ip), 1))
        delta = self.direction_i(z) * ip + self.direction_p(z) * (1.0 - ip)
        return velocity + delta

    def gain(self, source: Tensor, velocity: Tensor, prediction: Tensor, common_strength: Tensor, qp: int) -> Tensor:
        target = common_strength.shape[2:]
        src = F.interpolate(source.detach().float(), size=target, mode="trilinear", align_corners=False)
        vel = F.interpolate(velocity.detach().float(), size=target, mode="trilinear", align_corners=False)
        if not getattr(self, "condition_grad_enabled", False):
            prediction, common_strength = (prediction.detach(), common_strength.detach())
        pred = F.interpolate(prediction.float(), size=target, mode="trilinear", align_corners=False)
        strength = F.interpolate(common_strength.float(), size=target, mode="trilinear", align_corners=False)
        q = torch.full_like(strength, float(qp) / 63.0)
        ip = torch.zeros_like(strength)
        ip[:, :, :1] = 1.0
        z = self.gain_body(torch.cat((src, vel, strength.log(), pred, q, ip), 1).to(self.gain_body[0].weight.dtype))
        residual = self.gain_residual(z).float()
        base = self.gain_table[{0: 0, 16: 1, 39: 2, 63: 3}[int(qp)]].to(residual)
        log_residual = 0.25 * torch.tanh(residual)
        return (base * torch.exp(log_residual)).clamp(0.03, 0.8)
