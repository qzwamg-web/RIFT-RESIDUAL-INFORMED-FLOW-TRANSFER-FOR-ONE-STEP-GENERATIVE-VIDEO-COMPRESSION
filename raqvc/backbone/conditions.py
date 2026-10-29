from dataclasses import dataclass
from typing import Optional
import torch


@dataclass(frozen=True)
class DiTConditions:
    """Decoder-visible context for an independent whole-GOP DiT call."""

    text_ctx: torch.Tensor
    rate_noise_tok: torch.Tensor
    codec_ctx: Optional[torch.Tensor] = None
    spatial_anchor: Optional[torch.Tensor] = None

    def __post_init__(self) -> None:
        if self.rate_noise_tok.ndim != 3 or self.rate_noise_tok.shape[1] != 1:
            raise ValueError(f"rate_noise_tok must be [B, 1, d], got {tuple(self.rate_noise_tok.shape)}")
        if self.codec_ctx is not None and self.codec_ctx.ndim != 3:
            raise ValueError(f"codec_ctx must be [B, L, d], got {tuple(self.codec_ctx.shape)}")
        if self.spatial_anchor is not None and self.spatial_anchor.ndim != 5:
            raise ValueError(f"spatial_anchor must be [B, C, T, H, W], got {tuple(self.spatial_anchor.shape)}")
