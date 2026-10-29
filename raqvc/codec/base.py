"""Abstract interfaces for latent entropy and intra image codecs (spec §5.2)."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional

import torch


@dataclass(frozen=True)
class CodecResult:
    """Result of a single codec encode-or-decode call.

    Invariant (unit-tested in tests/integration/test_eem_bit_exact_roundtrip.py):
        encode_inter(z, r, prev).quantized_latent
        bit-equals
        decode_inter(encode_inter(...).bitstream, prev).quantized_latent
    """

    bitstream: bytes
    quantized_latent: torch.Tensor
    delta: float
    bpp: float

    def __post_init__(self) -> None:
        if not isinstance(self.bitstream, (bytes, bytearray)):
            raise TypeError(f"bitstream must be bytes, got {type(self.bitstream).__name__}")
        if not isinstance(self.quantized_latent, torch.Tensor):
            raise TypeError(f"quantized_latent must be a torch.Tensor, got " f"{type(self.quantized_latent).__name__}")
        if self.delta < 0:
            raise ValueError(f"delta must be >= 0, got {self.delta}")
        if self.bpp < 0:
            raise ValueError(f"bpp must be >= 0, got {self.bpp}")


class LatentEntropyCodec(ABC):
    """Entropy codec operating on Wan VAE latents (spec §5.2).

    Implementations wrap a learned entropy model (EEM, DCVC, ...) and an
    optional pair of `proj_in` / `proj_out` conv layers to adapt the
    latent channel count. All four methods return a `CodecResult`.
    """

    @abstractmethod
    def encode_intra(self, latent: torch.Tensor, rate: float) -> CodecResult: ...

    @abstractmethod
    def decode_intra(self, bitstream: bytes) -> CodecResult: ...

    @abstractmethod
    def encode_inter(
        self,
        latent: torch.Tensor,
        rate: float,
        prev_quant_latent: Optional[torch.Tensor],
    ) -> CodecResult: ...

    @abstractmethod
    def decode_inter(
        self,
        bitstream: bytes,
        prev_quant_latent: Optional[torch.Tensor],
    ) -> CodecResult: ...


@dataclass(frozen=True)
class IntraResult:
    """Result of encoding a single intra frame with a generative image codec."""

    bitstream: bytes
    bpp: float

    def __post_init__(self) -> None:
        if not isinstance(self.bitstream, (bytes, bytearray)):
            raise TypeError(f"bitstream must be bytes, got {type(self.bitstream).__name__}")
        if self.bpp < 0:
            raise ValueError(f"bpp must be >= 0, got {self.bpp}")


class GenerativeImageCodec(ABC):
    """Generative image codec for intra frames (spec §5.2)."""

    @abstractmethod
    def encode(self, frame: torch.Tensor, rate: float) -> IntraResult: ...

    @abstractmethod
    def decode(self, bitstream: bytes) -> torch.Tensor: ...
