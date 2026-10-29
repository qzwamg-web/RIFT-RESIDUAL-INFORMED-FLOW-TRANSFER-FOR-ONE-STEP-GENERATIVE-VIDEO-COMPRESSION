"""Task938 opt-in centered dyadic scalar quantization."""

from __future__ import annotations

import math
from typing import Optional, Sequence

import torch


class DyadicScalarQuantizer:
    """A decoder-closed scalar lattice for one entropy coordinate.

    QP normally selects the dyadic bin width; callers may opt into four
    explicit widths with ``widths_by_qp``. Explicit widths are fixed codec
    configuration: they are neither nested/successive-refinement bins nor
    learnable parameters. Hyper-z and motion syntax are intentionally outside
    this class.
    """

    VERSION = 1

    def __init__(
        self,
        *,
        fine_step: float = 1.0,
        drop_bits_by_qp: Sequence[int] = (0, 0, 1, 1),
        widths_by_qp: Optional[Sequence[float]] = None,
        enabled: bool = False,
    ) -> None:
        self.fine_step = float(fine_step)
        self.drop_bits_by_qp = tuple(int(v) for v in drop_bits_by_qp)
        self.widths_by_qp = None if widths_by_qp is None else tuple(float(v) for v in widths_by_qp)
        self.enabled = bool(enabled)
        if not math.isfinite(self.fine_step) or self.fine_step <= 0.0:
            raise ValueError("fine_step must be finite and positive")
        if len(self.drop_bits_by_qp) != 4 or any(v < 0 or v > 8 for v in self.drop_bits_by_qp):
            raise ValueError("drop_bits_by_qp must be nonempty and nonnegative")
        if tuple(sorted(self.drop_bits_by_qp)) != self.drop_bits_by_qp:
            raise ValueError("drop_bits_by_qp must be nondecreasing")
        if self.widths_by_qp is not None:
            if len(self.widths_by_qp) != 4:
                raise ValueError("widths_by_qp must contain exactly 4 values")
            if any(not math.isfinite(value) or value < 0.25 or value > 4.0 for value in self.widths_by_qp):
                raise ValueError("widths_by_qp values must be finite and bounded in [0.25, 4]")

    def _index(self, qp: int) -> int:
        if int(qp) not in (0, 16, 39, 63):
            raise ValueError("Task938 quantizer requires canonical QP 0/16/39/63")
        return {0: 0, 16: 1, 39: 2, 63: 3}[int(qp)]

    def drop_bits(self, qp: int) -> int:
        return self.drop_bits_by_qp[self._index(qp)]

    def width(self, qp: int) -> float:
        if self.widths_by_qp is not None:
            return self.widths_by_qp[self._index(qp)]
        return self.fine_step * float(1 << self.drop_bits(qp))

    def quantize_residual(self, residual: torch.Tensor, qp: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Return `(reconstruction, integer_symbol)` for a centered residual."""
        if not self.enabled:
            symbol = torch.round(residual)
            reconstruction = residual + (symbol - residual).detach()
            return reconstruction, symbol.to(torch.int32)
        width = self.width(qp)
        normalized = residual / width
        symbol = torch.round(normalized)
        # STE in normalized coordinates gives d(reconstruction)/d(residual)=1.
        symbol_ste = normalized + (symbol - normalized).detach()
        reconstruction = symbol_ste * width
        return reconstruction, symbol.to(torch.int32)

    def quantize(self, residual: torch.Tensor, qp: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.quantize_residual(residual, qp)

    def symbol_coordinate(self, symbol: torch.Tensor, qp: int) -> torch.Tensor:
        """Convert the integer symbol into the unit-width Gaussian coordinate."""
        return symbol.to(torch.float32)

    def corrected_distribution(
        self, scale: torch.Tensor, mean: torch.Tensor, qp: int
    ) -> tuple[torch.Tensor, torch.Tensor, float]:
        """Return `(scale/width, mean, width)` for a centered residual coder."""
        width = self.width(qp)
        return scale / width, mean, width

    def contract(self) -> dict:
        if self.widths_by_qp is not None:
            return {
                "schema": "task938_dyadic_scalar_v2",
                "version": 2,
                "enabled": self.enabled,
                "fine_step": self.fine_step,
                "drop_bits_by_qp": list(self.drop_bits_by_qp),
                "widths_by_qp": list(self.widths_by_qp),
            }
        return {
            "schema": "task938_dyadic_scalar_v1",
            "version": self.VERSION,
            "enabled": self.enabled,
            "fine_step": self.fine_step,
            "drop_bits_by_qp": list(self.drop_bits_by_qp),
        }


__all__ = ["DyadicScalarQuantizer"]
