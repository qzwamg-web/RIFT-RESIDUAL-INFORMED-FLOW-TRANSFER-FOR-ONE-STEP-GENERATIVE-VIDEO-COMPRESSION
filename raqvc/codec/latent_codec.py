"""Latent codec interface for rate-adaptive video compression.

This module provides a simplified codec interface for training Phase A.1,
complementing the production LatentEntropyCodec interface in base.py.

The key difference:
- LatentCodec: Training-focused, returns (bitstream, quantized_latent) tuples
- LatentEntropyCodec: Production-focused, returns CodecResult with metadata

Both interfaces can coexist. DCVC implementations may implement both.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Tuple

import torch


class LatentCodec(ABC):
    """Abstract base for latent entropy codecs used in training.

    Encodes VAE latents to bitstreams with rate control.
    Used in Phase A.1 training to replace uniform noise simulation.

    This is a simplified interface compared to LatentEntropyCodec in base.py,
    focused on training workflows where we need (bitstream, quantized_latent)
    pairs without the full CodecResult metadata.
    """

    @abstractmethod
    def encode(
        self,
        latent: torch.Tensor,
        rate: float,
        *,
        return_quantized: bool = True,
    ) -> Tuple[bytes, torch.Tensor]:
        """Encode latent to bitstream at target rate.

        Args:
            latent: [B, C, T, H, W] VAE latent (typically [B, 48, 1, H', W'])
            rate: Target bitrate in bpp
            return_quantized: If True, return quantized latent for training

        Returns:
            bitstream: Compressed bytes
            latent_q: Quantized latent (for training loss computation)
        """
        ...

    @abstractmethod
    def decode(self, bitstream: bytes) -> torch.Tensor:
        """Decode bitstream to latent.

        Args:
            bitstream: Compressed bytes

        Returns:
            latent: [B, C, T, H, W] reconstructed latent
        """
        ...

    @abstractmethod
    def get_actual_rate(self, bitstream: bytes, num_pixels: int) -> float:
        """Calculate actual bitrate in bpp.

        Args:
            bitstream: Compressed bytes
            num_pixels: Total number of pixels (B * T * H * W)

        Returns:
            Actual bitrate in bpp
        """
        ...
