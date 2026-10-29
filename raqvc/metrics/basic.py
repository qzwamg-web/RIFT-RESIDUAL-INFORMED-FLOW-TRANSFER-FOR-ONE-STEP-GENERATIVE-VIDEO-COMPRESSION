"""Basic evaluation metrics for RAQVC.

Provides lightweight helpers for PSNR, MSE, SSIM, and bitrate calculations.
Heavy perceptual metrics (LPIPS, FID) live in :mod:`raqvc.metrics.perceptual`.

All tensor operations assume float32 inputs unless otherwise noted.
"""

from __future__ import annotations

import math
from typing import Sequence

import torch
import torch.nn.functional as F

# ---------------------------------------------------------------------------
# Pixel-domain distortion
# ---------------------------------------------------------------------------


def mse(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """Mean squared error between two tensors.

    Args:
        x: Reference tensor, arbitrary shape.
        y: Reconstructed tensor, same shape as *x*.

    Returns:
        Scalar tensor containing the mean squared error.

    Raises:
        ValueError: If *x* and *y* have different shapes.
    """
    if x.shape != y.shape:
        raise ValueError(f"x and y must have the same shape, got {tuple(x.shape)} vs {tuple(y.shape)}")
    return torch.mean((x.float() - y.float()) ** 2)


def psnr(x: torch.Tensor, y: torch.Tensor, data_range: float = 2.0) -> torch.Tensor:
    """Peak signal-to-noise ratio (dB).

    The default *data_range* of 2.0 matches tensors normalised to [-1, 1].
    Use 1.0 for [0, 1] tensors or 255.0 for uint8 images.

    Args:
        x: Reference tensor, arbitrary shape.
        y: Reconstructed tensor, same shape as *x*.
        data_range: Dynamic range of the signal (max - min).  Defaults to 2.0.

    Returns:
        Scalar tensor with PSNR in dB.  Returns ``inf`` when MSE is zero.

    Raises:
        ValueError: If *x* and *y* have different shapes.
        ValueError: If *data_range* is not positive.
    """
    if data_range <= 0:
        raise ValueError(f"data_range must be positive, got {data_range}")
    err = mse(x, y)
    if err == 0:
        return torch.tensor(float("inf"), device=err.device, dtype=err.dtype)
    peak = torch.tensor(data_range**2, device=err.device, dtype=err.dtype)
    return 10.0 * torch.log10(peak / err)


# ---------------------------------------------------------------------------
# Bitrate helpers
# ---------------------------------------------------------------------------


def bitrate_bpp(num_bits: int, num_pixels: int) -> float:
    """Bits per pixel from a raw bit count.

    Args:
        num_bits: Total number of bits in the compressed representation.
        num_pixels: Total number of spatial pixels (H × W × frames).

    Returns:
        Bitrate in bits-per-pixel (bpp).

    Raises:
        ValueError: If *num_pixels* is zero or negative.
    """
    if num_pixels <= 0:
        raise ValueError(f"num_pixels must be positive, got {num_pixels}")
    return num_bits / num_pixels


def bitrate_from_bytes(num_bytes: int, num_pixels: int) -> float:
    """Bits per pixel from a byte count.

    Convenience wrapper around :func:`bitrate_bpp` that converts bytes to bits.

    Args:
        num_bytes: Total number of bytes in the compressed representation.
        num_pixels: Total number of spatial pixels (H × W × frames).

    Returns:
        Bitrate in bits-per-pixel (bpp).

    Raises:
        ValueError: If *num_pixels* is zero or negative.
    """
    return bitrate_bpp(num_bytes * 8, num_pixels)


# ---------------------------------------------------------------------------
# Aggregation helpers
# ---------------------------------------------------------------------------


def average_meter(values: Sequence[float]) -> float:
    """Arithmetic mean of a sequence of floats.

    Useful for accumulating per-frame or per-chunk metric values and
    reporting a single epoch-level number.

    Args:
        values: Non-empty sequence of float values.

    Returns:
        Arithmetic mean.

    Raises:
        ValueError: If *values* is empty.
    """
    if not values:
        raise ValueError("values must be non-empty")
    return math.fsum(values) / len(values)


# ---------------------------------------------------------------------------
# SSIM
# ---------------------------------------------------------------------------


def _fspecial_gauss_1d(size: int, sigma: float) -> torch.Tensor:
    """Create a 1-D Gaussian kernel (normalised to sum=1).

    Args:
        size: Kernel width (must be odd and positive).
        sigma: Standard deviation of the Gaussian.

    Returns:
        1-D tensor of shape ``(1, 1, size)``.
    """
    coords = torch.arange(size, dtype=torch.float32) - size // 2
    g = torch.exp(-(coords**2) / (2 * sigma**2))
    g /= g.sum()
    return g.view(1, 1, -1)


def ssim(
    x: torch.Tensor,
    y: torch.Tensor,
    data_range: float = 2.0,
    win_size: int = 11,
    win_sigma: float = 1.5,
    k1: float = 0.01,
    k2: float = 0.03,
) -> torch.Tensor:
    """Structural Similarity Index (SSIM).

    Computes single-scale SSIM between two image tensors using a Gaussian
    sliding window.

    Example::

        >>> x = torch.rand(1, 3, 256, 256)
        >>> y = torch.rand(1, 3, 256, 256)
        >>> score = ssim(x, y, data_range=1.0)

    Args:
        x: Reference images, shape ``(N, C, H, W)`` in float32.
        y: Reconstructed images, same shape as *x*.
        data_range: Dynamic range of the signal (max - min). Defaults to 2.0
            for tensors normalised to ``[-1, 1]``.
        win_size: Side length of the Gaussian window. Must be odd. Defaults to 11.
        win_sigma: Standard deviation of the Gaussian window. Defaults to 1.5.
        k1: Stability constant for luminance. Defaults to 0.01.
        k2: Stability constant for contrast. Defaults to 0.03.

    Returns:
        Scalar tensor with the mean SSIM value.

    Raises:
        ValueError: If *x* and *y* have different shapes.
        ValueError: If *x* is not 4-D.
        ValueError: If *win_size* is even or non-positive.
        ValueError: If *data_range* is not positive.
    """
    if x.shape != y.shape:
        raise ValueError(f"x and y must have the same shape, got {tuple(x.shape)} vs {tuple(y.shape)}")
    if x.ndim != 4:
        raise ValueError(f"Expected 4-D input (N, C, H, W), got {x.ndim}-D")
    if win_size <= 0 or win_size % 2 == 0:
        raise ValueError(f"win_size must be a positive odd integer, got {win_size}")
    if data_range <= 0:
        raise ValueError(f"data_range must be positive, got {data_range}")

    x = x.float()
    y = y.float()

    c1 = (k1 * data_range) ** 2
    c2 = (k2 * data_range) ** 2

    # Build 2-D Gaussian kernel
    g1d = _fspecial_gauss_1d(win_size, win_sigma)  # (1, 1, K)
    g1d_flat = g1d.squeeze()  # (K,)
    g2d = g1d_flat.unsqueeze(1) @ g1d_flat.unsqueeze(0)  # (K, K)
    g2d = g2d.unsqueeze(0).unsqueeze(0)  # (1, 1, K, K)

    c = x.shape[1]
    window = g2d.expand(c, 1, -1, -1).to(x.device, x.dtype)
    pad = win_size // 2

    mu_x = F.conv2d(x, window, padding=pad, groups=c)
    mu_y = F.conv2d(y, window, padding=pad, groups=c)

    mu_x_sq = mu_x**2
    mu_y_sq = mu_y**2
    mu_xy = mu_x * mu_y

    sigma_x_sq = F.conv2d(x * x, window, padding=pad, groups=c) - mu_x_sq
    sigma_y_sq = F.conv2d(y * y, window, padding=pad, groups=c) - mu_y_sq
    sigma_xy = F.conv2d(x * y, window, padding=pad, groups=c) - mu_xy

    numerator = (2 * mu_xy + c1) * (2 * sigma_xy + c2)
    denominator = (mu_x_sq + mu_y_sq + c1) * (sigma_x_sq + sigma_y_sq + c2)

    ssim_map = numerator / denominator
    return ssim_map.mean()


def ms_ssim(
    x: torch.Tensor,
    y: torch.Tensor,
    data_range: float = 2.0,
    win_size: int = 11,
    win_sigma: float = 1.5,
    weights: Sequence[float] | None = None,
) -> torch.Tensor:
    """Multi-Scale Structural Similarity Index (MS-SSIM).

    Computes MS-SSIM as described in Wang et al. (2003) by evaluating SSIM
    components at multiple downsampled scales.

    Example::

        >>> x = torch.rand(1, 3, 256, 256)
        >>> y = torch.rand(1, 3, 256, 256)
        >>> score = ms_ssim(x, y, data_range=1.0)

    Args:
        x: Reference images, shape ``(N, C, H, W)`` in float32.
        y: Reconstructed images, same shape as *x*.
        data_range: Dynamic range of the signal. Defaults to 2.0.
        win_size: Side length of the Gaussian window. Must be odd. Defaults to 11.
        win_sigma: Standard deviation of the Gaussian window. Defaults to 1.5.
        weights: Per-scale weights. Defaults to the 5-scale weights from the
            original MS-SSIM paper.

    Returns:
        Scalar tensor with the mean MS-SSIM value.

    Raises:
        ValueError: If *x* and *y* have different shapes.
        ValueError: If *x* is not 4-D.
        ValueError: If spatial dimensions are too small for the number of scales.
    """
    if x.shape != y.shape:
        raise ValueError(f"x and y must have the same shape, got {tuple(x.shape)} vs {tuple(y.shape)}")
    if x.ndim != 4:
        raise ValueError(f"Expected 4-D input (N, C, H, W), got {x.ndim}-D")

    if weights is None:
        weights = [0.0448, 0.2856, 0.3001, 0.2363, 0.1333]
    weights_t = torch.tensor(weights, dtype=torch.float32, device=x.device)
    levels = len(weights)

    x = x.float()
    y = y.float()

    k1, k2 = 0.01, 0.03
    c1 = (k1 * data_range) ** 2
    c2 = (k2 * data_range) ** 2

    # Build 2-D Gaussian kernel
    g1d = _fspecial_gauss_1d(win_size, win_sigma)
    g1d_flat = g1d.squeeze()  # (K,)
    g2d = g1d_flat.unsqueeze(1) @ g1d_flat.unsqueeze(0)  # (K, K)
    g2d = g2d.unsqueeze(0).unsqueeze(0)
    c = x.shape[1]
    window = g2d.expand(c, 1, -1, -1).to(x.device, x.dtype)
    pad = win_size // 2

    cs_vals: list[torch.Tensor] = []
    ssim_val: torch.Tensor | None = None

    for i in range(levels):
        min_dim = min(x.shape[2], x.shape[3])
        if min_dim < win_size:
            raise ValueError(
                f"Spatial dimensions too small at scale {i}: " f"{x.shape[2]}x{x.shape[3]} < win_size={win_size}"
            )

        mu_x = F.conv2d(x, window, padding=pad, groups=c)
        mu_y = F.conv2d(y, window, padding=pad, groups=c)

        mu_x_sq = mu_x**2
        mu_y_sq = mu_y**2
        mu_xy = mu_x * mu_y

        sigma_x_sq = F.conv2d(x * x, window, padding=pad, groups=c) - mu_x_sq
        sigma_y_sq = F.conv2d(y * y, window, padding=pad, groups=c) - mu_y_sq
        sigma_xy = F.conv2d(x * y, window, padding=pad, groups=c) - mu_xy

        cs = (2 * sigma_xy + c2) / (sigma_x_sq + sigma_y_sq + c2)
        cs_vals.append(cs.mean())

        if i == levels - 1:
            lum = (2 * mu_xy + c1) / (mu_x_sq + mu_y_sq + c1)
            ssim_val = lum.mean()
        else:
            # Downsample by 2
            x = F.avg_pool2d(x, kernel_size=2)
            y = F.avg_pool2d(y, kernel_size=2)

    # Combine: product of cs^weight at each scale, times luminance^weight at last scale
    result = torch.ones(1, device=x.device, dtype=x.dtype)
    for i in range(levels - 1):
        result = result * torch.clamp(cs_vals[i], min=0.0) ** weights_t[i]
    assert ssim_val is not None
    result = result * torch.clamp(ssim_val, min=0.0) ** weights_t[-1]
    return result.squeeze()
