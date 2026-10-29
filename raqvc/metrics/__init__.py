"""Metrics utilities for evaluation and reporting."""

from raqvc.metrics.basic import average_meter, bitrate_bpp, bitrate_from_bytes, mse, ms_ssim, psnr, ssim

__all__ = [
    "mse",
    "psnr",
    "ssim",
    "ms_ssim",
    "bitrate_bpp",
    "bitrate_from_bytes",
    "average_meter",
    "DISTSMetric",
    "LPIPSMetric",
    "FIDMetric",
]


# Lazy imports for optional dependencies
def __getattr__(name):
    if name == "DISTSMetric":
        from raqvc.metrics.perceptual import DISTSMetric

        return DISTSMetric
    elif name == "LPIPSMetric":
        from raqvc.metrics.perceptual import LPIPSMetric

        return LPIPSMetric
    elif name == "FIDMetric":
        from raqvc.metrics.perceptual import FIDMetric

        return FIDMetric
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
