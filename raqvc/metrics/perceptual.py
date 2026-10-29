"""Perceptual evaluation metrics for RAQVC.

Provides LPIPS (Learned Perceptual Image Patch Similarity) and FID (Fréchet
Inception Distance) metrics for evaluating perceptual quality of reconstructed
video frames.

All metrics accept torch.Tensor inputs and return scalar tensors.
"""

from __future__ import annotations

import math
from typing import Literal, Sequence

import numpy as np
import torch
import torch.nn as nn

try:
    import lpips
except ImportError:
    lpips = None  # type: ignore

try:
    from pytorch_fid import fid_score
    from pytorch_fid.inception import InceptionV3
except ImportError:
    fid_score = None  # type: ignore
    InceptionV3 = None  # type: ignore

try:
    from DISTS_pytorch import DISTS
except ImportError:
    DISTS = None  # type: ignore


# ---------------------------------------------------------------------------
# LPIPS
# ---------------------------------------------------------------------------


class LPIPSMetric:
    """Learned Perceptual Image Patch Similarity (LPIPS) metric.

    Wraps the ``lpips`` library to provide a consistent interface for
    computing perceptual distance between image pairs.

    Example::

        >>> metric = LPIPSMetric(net="vgg", device="cuda")
        >>> x = torch.rand(1, 3, 256, 256, device="cuda")
        >>> y = torch.rand(1, 3, 256, 256, device="cuda")
        >>> distance = metric(x, y)

    Args:
        net: Backbone network. One of ``"alex"``, ``"vgg"``, or ``"squeeze"``.
            Defaults to ``"vgg"``.
        device: Device to run the model on. Defaults to ``"cpu"``.

    Raises:
        ImportError: If the ``lpips`` library is not installed.
    """

    def __init__(
        self,
        net: Literal["alex", "vgg", "squeeze"] = "vgg",
        device: str = "cpu",
    ):
        if lpips is None:
            raise ImportError("lpips library is required for LPIPS metric. " "Install with: pip install lpips")
        self.model = lpips.LPIPS(net=net).to(device)
        self.model.eval()
        # LPIPS is a fixed perceptual feature extractor.  Freezing its weights
        # avoids parameter gradients while deliberately preserving gradients
        # with respect to ``x``/``y`` for training-time perceptual losses.
        self.model.requires_grad_(False)
        self.device = device

    def __call__(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Compute LPIPS distance between two image tensors.

        Args:
            x: Reference images, shape ``(N, C, H, W)`` in range ``[-1, 1]``.
            y: Reconstructed images, same shape as *x*.

        Returns:
            Scalar tensor with the mean LPIPS distance.

        Raises:
            ValueError: If *x* and *y* have different shapes.
        """
        if x.shape != y.shape:
            raise ValueError(f"x and y must have the same shape, got {tuple(x.shape)} vs {tuple(y.shape)}")
        return self.per_sample(x, y).mean()

    def per_sample(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Return one LPIPS value per image without changing normalization."""
        if x.shape != y.shape:
            raise ValueError(f"x and y must have the same shape, got {tuple(x.shape)} vs {tuple(y.shape)}")
        x = x.to(self.device)
        y = y.to(self.device)
        dist = self.model(x, y)
        return dist.reshape(int(dist.shape[0]), -1).mean(dim=1)


def lpips_distance(
    x: torch.Tensor,
    y: torch.Tensor,
    net: Literal["alex", "vgg", "squeeze"] = "vgg",
) -> torch.Tensor:
    """Compute LPIPS distance between two image tensors.

    Convenience function that creates a temporary :class:`LPIPSMetric` instance.
    For repeated evaluations, prefer creating a persistent instance to avoid
    reloading the model.

    Example::

        >>> x = torch.rand(1, 3, 256, 256)
        >>> y = torch.rand(1, 3, 256, 256)
        >>> distance = lpips_distance(x, y, net="vgg")

    Args:
        x: Reference images, shape ``(N, C, H, W)`` in range ``[-1, 1]``.
        y: Reconstructed images, same shape as *x*.
        net: Backbone network. One of ``"alex"``, ``"vgg"``, or ``"squeeze"``.
            Defaults to ``"vgg"``.

    Returns:
        Scalar tensor with the mean LPIPS distance.

    Raises:
        ImportError: If the ``lpips`` library is not installed.
        ValueError: If *x* and *y* have different shapes.
    """
    device = x.device if x.is_cuda else "cpu"
    metric = LPIPSMetric(net=net, device=str(device))
    return metric(x, y)


# ---------------------------------------------------------------------------
# FID
# ---------------------------------------------------------------------------


class FIDMetric:
    """Fréchet Inception Distance (FID) metric.

    Computes FID between two sets of images by extracting Inception-v3 features
    and measuring the Fréchet distance between their distributions.

    Example::

        >>> metric = FIDMetric(device="cuda")
        >>> real_images = torch.rand(100, 3, 256, 256, device="cuda")
        >>> fake_images = torch.rand(100, 3, 256, 256, device="cuda")
        >>> fid = metric(real_images, fake_images)

    Args:
        device: Device to run the model on. Defaults to ``"cpu"``.
        dims: Dimensionality of Inception features. Defaults to 2048.

    Raises:
        ImportError: If the ``pytorch-fid`` library is not installed.
    """

    def __init__(self, device: str = "cpu", dims: int = 2048):
        if InceptionV3 is None:
            raise ImportError(
                "pytorch-fid library is required for FID metric. " "Install with: pip install pytorch-fid"
            )
        block_idx = InceptionV3.BLOCK_INDEX_BY_DIM[dims]
        self.model = InceptionV3([block_idx]).to(device)
        self.model.eval()
        self.model.requires_grad_(False)
        self.device = device
        self.dims = dims

    def extract_features(
        self,
        images: torch.Tensor,
        *,
        batch_size: int = 8,
        input_range: Literal["minus_one_one", "zero_one"] = "minus_one_one",
    ) -> np.ndarray:
        """Extract standard pytorch-fid pool features without retaining images.

        Args:
            images: Image tensor with shape ``(N,3,H,W)``.
            batch_size: Inception forward microbatch.
            input_range: Explicit tensor range. ``pytorch-fid`` itself expects
                ``[0,1]`` before its internal ImageNet normalization.

        Returns:
            CPU float32 array with shape ``(N,dims)``.
        """
        if images.ndim != 4 or images.shape[1] != 3:
            raise ValueError(f"FID expects [N,3,H,W] images, got {tuple(images.shape)}")
        if batch_size <= 0:
            raise ValueError(f"batch_size must be positive, got {batch_size}")
        if input_range not in {"minus_one_one", "zero_one"}:
            raise ValueError(f"unsupported FID input_range={input_range!r}")

        chunks: list[np.ndarray] = []
        with torch.no_grad():
            for start in range(0, int(images.shape[0]), int(batch_size)):
                batch = images[start : start + batch_size].to(self.device, dtype=torch.float32)
                if input_range == "minus_one_one":
                    batch = (batch + 1.0) / 2.0
                batch = batch.clamp(0.0, 1.0)
                features = self.model(batch)[0]
                if features.shape[2:] != (1, 1):
                    features = torch.nn.functional.adaptive_avg_pool2d(features, output_size=(1, 1))
                chunks.append(features.squeeze(3).squeeze(2).float().cpu().numpy())
        if not chunks:
            raise ValueError("FID requires at least one image")
        return np.concatenate(chunks, axis=0)

    @staticmethod
    def calculate_from_features(real: np.ndarray, fake: np.ndarray) -> float:
        """Calculate FID from two feature matrices.

        For video benchmarks the number of sampled frames is normally much
        smaller than the 2048-dimensional pool3 feature.  In that regime an
        exact sample-space identity avoids materializing singular 2048x2048
        covariances: ``tr(sqrt(Cr Cf))`` is the nuclear norm of the centered
        real/fake cross-product after covariance normalization.
        """
        real = np.asarray(real, dtype=np.float64)
        fake = np.asarray(fake, dtype=np.float64)
        if real.ndim != 2 or fake.ndim != 2 or real.shape[1] != fake.shape[1]:
            raise ValueError(f"FID feature mismatch: real={real.shape}, fake={fake.shape}")
        if real.shape[0] < 2 or fake.shape[0] < 2:
            raise ValueError("FID requires at least two real and two fake features")
        mu_real = np.mean(real, axis=0)
        mu_fake = np.mean(fake, axis=0)
        centered_real = real - mu_real
        centered_fake = fake - mu_fake
        mean_term = float(np.square(mu_real - mu_fake).sum())

        if real.shape[0] * fake.shape[0] <= real.shape[1] * real.shape[1]:
            real_denom = real.shape[0] - 1
            fake_denom = fake.shape[0] - 1
            trace_real = float(np.square(centered_real).sum() / real_denom)
            trace_fake = float(np.square(centered_fake).sum() / fake_denom)
            cross = centered_real @ centered_fake.T
            cross /= math.sqrt(real_denom * fake_denom)
            trace_covmean = float(np.linalg.svd(cross, compute_uv=False).sum())
            return max(0.0, mean_term + trace_real + trace_fake - 2.0 * trace_covmean)

        sigma_real = np.cov(real, rowvar=False)
        sigma_fake = np.cov(fake, rowvar=False)
        value = fid_score.calculate_frechet_distance(
            mu_real,
            sigma_real,
            mu_fake,
            sigma_fake,
        )
        return max(0.0, float(np.real(value)))

    def __call__(self, real: torch.Tensor, fake: torch.Tensor) -> torch.Tensor:
        """Compute FID between real and generated images.

        Args:
            real: Real images, shape ``(N, C, H, W)`` in range ``[-1, 1]``.
            fake: Generated images, shape ``(M, C, H, W)`` in range ``[-1, 1]``.

        Returns:
            Scalar tensor with the FID score.

        Raises:
            ValueError: If images have incompatible shapes.
        """
        if real.ndim != 4 or fake.ndim != 4 or real.shape[1] != fake.shape[1]:
            raise ValueError(
                f"real and fake must be 4-D with matching channels, " f"got {tuple(real.shape)} vs {tuple(fake.shape)}"
            )
        real_features = self.extract_features(real, input_range="minus_one_one")
        fake_features = self.extract_features(fake, input_range="minus_one_one")
        return torch.tensor(
            self.calculate_from_features(real_features, fake_features),
            dtype=torch.float64,
        )


def fid_score_tensors(real: torch.Tensor, fake: torch.Tensor, device: str = "cpu") -> torch.Tensor:
    """Compute FID between real and generated image tensors.

    Convenience function that creates a temporary :class:`FIDMetric` instance.
    For repeated evaluations, prefer creating a persistent instance to avoid
    reloading the model.

    Example::

        >>> real = torch.rand(100, 3, 256, 256)
        >>> fake = torch.rand(100, 3, 256, 256)
        >>> fid = fid_score_tensors(real, fake, device="cuda")

    Args:
        real: Real images, shape ``(N, C, H, W)`` in range ``[-1, 1]``.
        fake: Generated images, shape ``(M, C, H, W)`` in range ``[-1, 1]``.
        device: Device to run the model on. Defaults to ``"cpu"``.

    Returns:
        Scalar tensor with the FID score.

    Raises:
        ImportError: If the ``pytorch-fid`` library is not installed.
        ValueError: If images have incompatible shapes.
    """
    metric = FIDMetric(device=device)
    return metric(real, fake)


# ---------------------------------------------------------------------------
# DISTS
# ---------------------------------------------------------------------------


class DISTSMetric:
    """Deep Image Structure and Texture Similarity (DISTS) metric.

    Wraps the ``DISTS_pytorch`` library to provide a consistent interface for
    computing perceptual distance between image pairs.

    Example::

        >>> metric = DISTSMetric(device="cuda")
        >>> x = torch.rand(1, 3, 256, 256, device="cuda")
        >>> y = torch.rand(1, 3, 256, 256, device="cuda")
        >>> distance = metric(x, y)

    Args:
        device: Device to run the model on. Defaults to ``"cpu"``.

    Raises:
        ImportError: If the ``DISTS_pytorch`` library is not installed.
    """

    def __init__(self, device: str = "cpu"):
        if DISTS is None:
            raise ImportError(
                "DISTS_pytorch library is required for DISTS metric. " "Install with: pip install DISTS-pytorch"
            )
        # Bug workaround: DISTS_pytorch hardcodes weights.pt path to sys.prefix,
        # but the file actually ships inside the package directory.
        # Load without auto-loading weights, then load manually from package dir.
        import os
        import DISTS_pytorch

        self._model = DISTS(load_weights=False).to(device)
        weights_path = os.path.join(os.path.dirname(DISTS_pytorch.__file__), "weights.pt")
        if os.path.exists(weights_path):
            weights = torch.load(weights_path, map_location=device, weights_only=True)
            self._model.alpha.data = weights["alpha"].to(device)
            self._model.beta.data = weights["beta"].to(device)
        else:
            raise FileNotFoundError(
                f"DISTS weights.pt not found at {weights_path}. " f"Please reinstall DISTS-pytorch."
            )
        self._model.eval()
        # The metric is fixed, but its input gradient is a useful training
        # signal.  Freezing parameters instead of wrapping the forward in
        # no_grad matches the LPIPS training contract.
        self._model.requires_grad_(False)
        self._device = device

    def __call__(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Compute DISTS distance between two image tensors.

        Args:
            x: Reference images, shape ``(N, 3, H, W)`` in range ``[0, 1]``.
            y: Reconstructed images, same shape as *x*.

        Returns:
            Scalar tensor with the mean DISTS distance.

        Raises:
            ValueError: If *x* and *y* have different shapes.
        """
        if x.shape != y.shape:
            raise ValueError(f"x and y must have the same shape, got {tuple(x.shape)} vs {tuple(y.shape)}")
        x = x.to(self._device).float()
        y = y.to(self._device).float()
        return self._model(x, y).mean()

    def components(self, x: torch.Tensor, y: torch.Tensor) -> dict[str, torch.Tensor]:
        """Return the exact DISTS texture/structure distance decomposition.

        DISTS normalizes ``alpha`` and ``beta`` jointly.  Consequently the
        total distance is exactly the sum of the missing weighted texture
        similarity and missing weighted structure similarity returned here.
        One value per image is retained so callers can aggregate videos
        without changing frame weighting.
        """

        if x.shape != y.shape:
            raise ValueError(f"x and y must have the same shape, got {tuple(x.shape)} vs {tuple(y.shape)}")
        x = x.to(self._device).float()
        y = y.to(self._device).float()
        with torch.no_grad():
            features_x = self.extract_features(x)
            features_y = self.extract_features(y)
            return self.components_from_features(features_x, features_y)

    @torch.no_grad()
    def extract_features(self, value: torch.Tensor) -> tuple[torch.Tensor, ...]:
        """Extract frozen DISTS features for exact no-grad cache reuse."""

        value = value.to(self._device).float()
        return tuple(self._model.forward_once(value))

    @torch.no_grad()
    def components_from_features(
        self,
        features_x: Sequence[torch.Tensor],
        features_y: Sequence[torch.Tensor],
    ) -> dict[str, torch.Tensor]:
        """Evaluate the exact DISTS formula from already frozen features."""

        if len(features_x) != len(self._model.chns) or len(features_y) != len(self._model.chns):
            raise ValueError("DISTS cached feature level count changed")
        batch = int(features_x[0].shape[0])
        if any(int(left.shape[0]) != batch or left.shape != right.shape for left, right in zip(features_x, features_y)):
            raise ValueError("DISTS cached feature shapes differ")
        with torch.no_grad():
            weight_sum = self._model.alpha.sum() + self._model.beta.sum()
            alpha = torch.split(self._model.alpha / weight_sum, self._model.chns, dim=1)
            beta = torch.split(self._model.beta / weight_sum, self._model.chns, dim=1)
            base = features_x[0]
            texture_similarity = base.new_zeros((batch, 1, 1, 1))
            structure_similarity = base.new_zeros((batch, 1, 1, 1))
            alpha_mass = base.new_zeros(())
            beta_mass = base.new_zeros(())
            for index in range(len(self._model.chns)):
                mean_x = features_x[index].mean((2, 3), keepdim=True)
                mean_y = features_y[index].mean((2, 3), keepdim=True)
                texture = (2.0 * mean_x * mean_y + 1.0e-6) / (mean_x.square() + mean_y.square() + 1.0e-6)
                var_x = (features_x[index] - mean_x).square().mean((2, 3), keepdim=True)
                var_y = (features_y[index] - mean_y).square().mean((2, 3), keepdim=True)
                covariance = (features_x[index] * features_y[index]).mean((2, 3), keepdim=True) - mean_x * mean_y
                structure = (2.0 * covariance + 1.0e-6) / (var_x + var_y + 1.0e-6)
                texture_similarity = texture_similarity + (alpha[index] * texture).sum(1, keepdim=True)
                structure_similarity = structure_similarity + (beta[index] * structure).sum(1, keepdim=True)
                alpha_mass = alpha_mass + alpha[index].sum()
                beta_mass = beta_mass + beta[index].sum()
            texture_distance = (alpha_mass - texture_similarity).flatten(1).mean(1)
            structure_distance = (beta_mass - structure_similarity).flatten(1).mean(1)
        return {
            "texture": texture_distance,
            "structure": structure_distance,
            "total": texture_distance + structure_distance,
        }
