"""Thin wrapper around the Wan 2.2 VAE.

The VAE itself is third-party (frozen). This module provides:
  - a stable `load_wan_vae()` entry point that takes explicit paths,
  - encode/decode helpers that accept the raqvc tensor convention
    ([C, T, H, W] in [-1, 1]).

No hardcoded paths — the checkpoint path comes from the caller (typically
loaded via `raqvc.utils.paths.RAQVCPaths.checkpoints_root`).
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

import torch

from raqvc.utils.paths import repo_root

_WAN_ADDED = False


def _ensure_wan_on_path() -> None:
    global _WAN_ADDED
    if _WAN_ADDED:
        return
    wan_root = repo_root() / "third_party" / "Wan2.2"
    if not wan_root.exists():
        raise RuntimeError(f"third_party/Wan2.2 not found at {wan_root}. " "See third_party/VERSIONS.md for setup.")
    if str(wan_root) not in sys.path:
        sys.path.insert(0, str(wan_root))
    _WAN_ADDED = True


def _convert_diffusers_vae_keys(diffusers_sd: dict) -> dict:
    """Convert Diffusers VAE state_dict keys to native Wan format."""
    import re

    native_sd = {}
    for key, tensor in diffusers_sd.items():
        new_key = key

        # Conv in/out
        if key.startswith("encoder.conv_in."):
            new_key = key.replace("encoder.conv_in.", "encoder.conv1.")
        elif key.startswith("decoder.conv_in."):
            new_key = key.replace("decoder.conv_in.", "decoder.conv1.")
        elif key.startswith("encoder.conv_out."):
            new_key = key.replace("encoder.conv_out.", "encoder.head.2.")
        elif key.startswith("decoder.conv_out."):
            new_key = key.replace("decoder.conv_out.", "decoder.head.2.")
        elif ".norm_out." in key:
            new_key = new_key.replace(".norm_out.", ".head.0.")
        elif ".down_blocks." in key:
            match = re.match(r"(encoder|decoder)\.down_blocks\.(\d+)\.resnets\.(\d+)\.(.*)", key)
            if match:
                prefix, block_idx, resnet_idx, rest = match.groups()
                new_key = f"{prefix}.downsamples.{block_idx}.downsamples.{resnet_idx}.{rest}"
            match = re.match(r"(encoder|decoder)\.down_blocks\.(\d+)\.downsampler\.(.*)", key)
            if match:
                prefix, block_idx, rest = match.groups()
                new_key = f"{prefix}.downsamples.{block_idx}.downsamples.2.{rest}"
        elif ".up_blocks." in key:
            match = re.match(r"(encoder|decoder)\.up_blocks\.(\d+)\.resnets\.(\d+)\.(.*)", key)
            if match:
                prefix, block_idx, resnet_idx, rest = match.groups()
                new_key = f"{prefix}.upsamples.{block_idx}.upsamples.{resnet_idx}.{rest}"
            match = re.match(r"(encoder|decoder)\.up_blocks\.(\d+)\.upsampler\.(.*)", key)
            if match:
                prefix, block_idx, rest = match.groups()
                new_key = f"{prefix}.upsamples.{block_idx}.upsamples.3.{rest}"
        elif ".mid_block.resnets." in key:
            match = re.match(r"(encoder|decoder)\.mid_block\.resnets\.(\d+)\.(.*)", key)
            if match:
                prefix, resnet_idx, rest = match.groups()
                middle_idx = int(resnet_idx) * 2
                new_key = f"{prefix}.middle.{middle_idx}.{rest}"
        elif ".mid_block.attentions." in key:
            match = re.match(r"(encoder|decoder)\.mid_block\.attentions\.(\d+)\.(.*)", key)
            if match:
                prefix, attn_idx, rest = match.groups()
                middle_idx = int(attn_idx) * 2 + 1
                new_key = f"{prefix}.middle.{middle_idx}.{rest}"

        # Residual block internals
        if ".conv1." in new_key and ("downsamples" in new_key or "upsamples" in new_key or "middle" in new_key):
            new_key = new_key.replace(".conv1.", ".residual.2.")
        if ".conv2." in new_key and ("downsamples" in new_key or "upsamples" in new_key or "middle" in new_key):
            new_key = new_key.replace(".conv2.", ".residual.6.")
        if ".norm1." in new_key and ("downsamples" in new_key or "upsamples" in new_key or "middle" in new_key):
            new_key = new_key.replace(".norm1.", ".residual.0.")
        if ".norm2." in new_key and ("downsamples" in new_key or "upsamples" in new_key or "middle" in new_key):
            new_key = new_key.replace(".norm2.", ".residual.3.")

        # conv_shortcut → shortcut
        new_key = new_key.replace(".conv_shortcut.", ".shortcut.")

        # quant_conv → conv1, post_quant_conv → conv2
        if key == "quant_conv.weight" or key == "quant_conv.bias":
            new_key = key.replace("quant_conv", "conv1")
        elif key == "post_quant_conv.weight" or key == "post_quant_conv.bias":
            new_key = key.replace("post_quant_conv", "conv2")

        native_sd[new_key] = tensor
    return native_sd


def _load_safetensors_vae(safetensors_path: Path) -> dict:
    """Load VAE weights from a safetensors file and convert to native format."""
    from safetensors.torch import load_file

    diffusers_sd = load_file(str(safetensors_path))
    return _convert_diffusers_vae_keys(diffusers_sd)


def load_wan_vae(
    checkpoint_path: Path,
    device: str = "cuda:0",
    dtype: torch.dtype = torch.bfloat16,
) -> torch.nn.Module:
    """Load a frozen Wan 2.2 3D-VAE.

    Supports both native .pth format and Diffusers safetensors format.
    For Diffusers format, pass the path to the vae/ subdirectory containing
    diffusion_pytorch_model.safetensors.
    """
    _ensure_wan_on_path()
    from wan.modules.vae2_2 import Wan2_2_VAE

    checkpoint_path = Path(checkpoint_path)

    # Check if this is a Diffusers-format directory
    safetensors_path = checkpoint_path / "diffusion_pytorch_model.safetensors"
    if checkpoint_path.is_dir() and safetensors_path.exists():
        logging.info(f"Loading Diffusers VAE from {safetensors_path}")
        native_sd = _load_safetensors_vae(safetensors_path)

        # Create VAE on meta device, then load weights
        with torch.device("meta"):
            from wan.modules.vae2_2 import WanVAE_

            model = WanVAE_(
                dim=160,
                z_dim=48,
                dim_mult=[1, 2, 4, 4],
                num_res_blocks=2,
                attn_scales=[],
                temperal_downsample=[False, True, True],
                dropout=0.0,
            )
        model.load_state_dict(native_sd, assign=True)
        model = model.eval().requires_grad_(False).to(device)

        vae = Wan2_2_VAE.__new__(Wan2_2_VAE)
        vae.dtype = dtype
        vae.device = device
        vae.model = model

        # Set up scale factors (same as Wan2_2_VAE.__init__)
        mean = torch.tensor(
            [
                -0.2289,
                -0.0052,
                -0.1323,
                -0.2339,
                -0.2799,
                0.0174,
                0.1838,
                0.1557,
                -0.1382,
                0.0542,
                0.2813,
                0.0891,
                0.1570,
                -0.0098,
                0.0375,
                -0.1825,
                -0.2246,
                -0.1207,
                -0.0698,
                0.5109,
                0.2665,
                -0.2108,
                -0.2158,
                0.2502,
                -0.2055,
                -0.0322,
                0.1109,
                0.1567,
                -0.0729,
                0.0899,
                -0.2799,
                -0.1230,
                -0.0313,
                -0.1649,
                0.0117,
                0.0723,
                -0.2839,
                -0.2083,
                -0.0520,
                0.3748,
                0.0152,
                0.1957,
                0.1433,
                -0.2944,
                0.3573,
                -0.0548,
                -0.1681,
                -0.0667,
            ],
            dtype=dtype,
            device=device,
        )
        std = torch.tensor(
            [
                0.4765,
                1.0364,
                0.4514,
                1.1677,
                0.5313,
                0.4990,
                0.4818,
                0.5013,
                0.8158,
                1.0344,
                0.5894,
                1.0901,
                0.6885,
                0.6165,
                0.8454,
                0.4978,
                0.5759,
                0.3523,
                0.7135,
                0.6804,
                0.5833,
                1.4146,
                0.8986,
                0.5659,
                0.7069,
                0.5338,
                0.4889,
                0.4917,
                0.4069,
                0.4999,
                0.6866,
                0.4093,
                0.5709,
                0.6065,
                0.6415,
                0.4944,
                0.5726,
                1.2042,
                0.5458,
                1.6887,
                0.3971,
                1.0600,
                0.3943,
                0.5537,
                0.5444,
                0.4089,
                0.7468,
                0.7744,
            ],
            dtype=dtype,
            device=device,
        )
        vae.scale = [mean, 1.0 / std]
        return vae

    # Native .pth format (original path)
    vae = Wan2_2_VAE(vae_pth=str(checkpoint_path), device=device, dtype=dtype)
    for p in vae.model.parameters():
        p.requires_grad_(False)
    vae.model.eval()
    return vae


def encode_video(
    vae: torch.nn.Module,
    video: torch.Tensor,
) -> torch.Tensor:
    """Encode a [C, T, H, W] video in [-1, 1] to Wan latent [z_dim, T', H', W']."""
    return vae.encode([video])[0]


def decode_latent(
    vae: torch.nn.Module,
    latent: torch.Tensor,
) -> torch.Tensor:
    """Decode a latent [z_dim, T', H', W'] to video [C, T, H, W] in [-1, 1]."""
    return vae.decode([latent])[0]
