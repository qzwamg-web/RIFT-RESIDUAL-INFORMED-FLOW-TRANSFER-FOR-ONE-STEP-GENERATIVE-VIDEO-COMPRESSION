"""C512 fixed 48-channel I/P latent codec with actual rANS coding.
DCVC-RT analysis/synthesis and the selected regional quantizer are retained.
Unsupported historical model configurations are rejected by the factory.
"""

from __future__ import annotations
import io
import hashlib
import json
import math
import sys
from pathlib import Path
from typing import Any, Iterator, Optional, Sequence, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
from raqvc.codec.latent_codec import LatentCodec
from raqvc.codec.p_recurrent_synthesis import PRecurrentSynthesisCore
from raqvc.codec.elic_group_context import IGroupEntropyContext, normalize_i_group_context_lane
from raqvc.codec.i_progressive_representation import DEFAULT_PROGRESSIVE_GROUP_SIZES, validate_progressive_contract
from raqvc.codec.i_hierarchical_representation import validate_hierarchical_contract
from raqvc.codec.p_frequency_compaction import validate_p_frequency_contract
from raqvc.codec.p_transform_capacity import PTransformCapacityCore, p_transform_capacity_contract
from raqvc.utils.vae_temporal import effective_bpp_from_per_latent

_VAE_CH = 48
_RT_D = 256
_RT_Y = 128
_RT_Z = 128
_RT_RECON = 320
_DMCI_ENC_DEC = 368
_DMCI_N = 256
_DMCI_Z = 128
_DMCI_NATIVE_CHANNEL_CONTRACT_VERSION = 1
_NUM_QP = 64
_EXTRA_QP = 8
_INTER_QP_NUM = _NUM_QP + _EXTRA_QP
_QP_MAX = _NUM_QP - 1
_DCVC_RT_ADDED = False
_TYPE_NORM_NAMES = ("I0", "P1", "P2+")


def _aggregate_decoder_entropy_scales(scales: Sequence[torch.Tensor], masks: Sequence[torch.Tensor]) -> torch.Tensor:
    """Merge causal-pass Gaussian scales in their original ``[B,C,H,W]`` grid.

    Every channel/location is assigned by one checkerboard pass.  The exported
    map is channel RMS per spatial location; it is diagnostic state only.
    """
    if len(scales) != len(masks) or not scales:
        raise ValueError("entropy scale export requires one scale tensor per mask")
    ref = scales[0]
    coverage = ref.new_zeros(ref.shape)
    sum_sq = ref.new_zeros((ref.shape[0], 1, ref.shape[-2], ref.shape[-1]))
    count = ref.new_zeros(sum_sq.shape)
    for scale, mask in zip(scales, masks):
        if scale.shape != ref.shape or mask.shape != ref.shape:
            raise ValueError("entropy scale/mask grids must match")
        scale = scale.clamp(min=0.11, max=16.0)
        active = mask.to(dtype=ref.dtype)
        coverage = coverage + active
        sum_sq = sum_sq + (scale * active).square().sum(dim=1, keepdim=True)
        count = count + active.sum(dim=1, keepdim=True)
    if not torch.allclose(coverage, torch.ones_like(coverage)):
        raise RuntimeError("entropy scale masks must cover every channel/location exactly once")
    return (sum_sq / count).clamp_min(0).sqrt().detach()


def _expand_type_channel_values(values: Sequence[Any], *, latent_channels: int, field_name: str) -> torch.Tensor:
    """Return fixed type/channel affine values as ``[3, C, 1, 1]``.

    The historical Task119 form used three scalars (one for I0/P1/P2+).
    Accepting both ``[3]`` and ``[3, C]`` keeps old configs/checkpoints usable
    while making the new per-channel contract explicit and fail-fast.
    """
    tensor = torch.as_tensor(values, dtype=torch.float32)
    expected_types = len(_TYPE_NORM_NAMES)
    if tensor.ndim == 1 and tuple(tensor.shape) == (expected_types,):
        tensor = tensor.view(expected_types, 1).expand(expected_types, latent_channels)
    elif tensor.ndim == 2 and tuple(tensor.shape) == (expected_types, latent_channels):
        pass
    elif tensor.ndim == 4 and tuple(tensor.shape) == (expected_types, latent_channels, 1, 1):
        tensor = tensor[:, :, 0, 0]
    else:
        raise ValueError(
            f"{field_name} must have shape [3], [3,{latent_channels}], or [3,{latent_channels},1,1] for I0/P1/P2+; got {tuple(tensor.shape)}."
        )
    if not torch.isfinite(tensor).all():
        raise ValueError(f"{field_name} contains NaN or Inf values.")
    return tensor.reshape(expected_types, latent_channels, 1, 1).clone()


def _ensure_dcvc_rt_on_path() -> None:
    """Add DCVC-RT root + C++ extension to sys.path."""
    global _DCVC_RT_ADDED
    if _DCVC_RT_ADDED:
        return
    dcvc_rt_root = Path(
        __import__("os").environ.get("RAQVC_DCVC_ROOT", str(Path(__file__).resolve().parents[2] / "third_party/DCVC"))
    )
    if not dcvc_rt_root.exists():
        raise RuntimeError(f"DCVC-RT root not found: {dcvc_rt_root}")
    root_path = str(dcvc_rt_root)
    cpp_path = str(dcvc_rt_root / "src" / "cpp")
    inference_path = str(dcvc_rt_root / "src" / "layers" / "extensions" / "inference")
    for p in (inference_path, cpp_path, root_path):
        if p not in sys.path:
            sys.path.insert(0, p)
    _DCVC_RT_ADDED = True
    try:
        import src.layers.cuda_inference as _ci

        if getattr(_ci, "CUSTOMIZED_CUDA_INFERENCE", False):
            _ci.CUSTOMIZED_CUDA_INFERENCE = False
            if "src.layers.cuda_inference" in sys.modules:
                sys.modules["src.layers.cuda_inference"] = _ci
    except Exception:
        pass


_ensure_dcvc_rt_on_path()
from src.layers.cuda_inference import restore_y_2x, restore_y_4x, round_and_to_int8
from raqvc.codec.task938_quantizer import DyadicScalarQuantizer


def _probs_to_bits(probs: torch.Tensor) -> torch.Tensor:
    """Convert probabilities to bits (-log2)."""
    probs = probs.clamp(1e-05, 1.0)
    return -torch.log2(probs)


def _get_y_gaussian_bits(
    y: torch.Tensor, sigma: torch.Tensor, mask: Optional[torch.Tensor] = None, *, coder_aligned: bool = False
) -> torch.Tensor:
    """Differentiable y bits, optionally aligned to the rANS scale range.

    DCVC-RT's hard Gaussian coder indexes a 128-level table spanning
    ``[0.11, 16]``.  Using a wider range here lets the noise path optimize a
    likelihood which cannot be represented by the real bitstream.  Masking is
    explicit because checkerboard positions outside the active pass are not
    coded and a scale clamped to 0.11 otherwise contributes a small fake rate.
    """
    mu = torch.zeros_like(sigma)
    if coder_aligned:
        sigma = sigma.clamp(0.11, 16.0)
    else:
        sigma = sigma.clamp(1e-05, 10000000000.0)
    dist = torch.distributions.normal.Normal(mu, sigma)
    probs = dist.cdf(y + 0.5) - dist.cdf(y - 0.5)
    bits = _probs_to_bits(probs)
    if mask is not None:
        bits = bits * mask
    return bits


def _get_z_bits(z: torch.Tensor, bit_estimator, qp: int) -> torch.Tensor:
    """Differentiable bit estimate for z-channel (factorized entropy)."""
    device = z.device
    qp_index = torch.tensor([qp], device=device, dtype=torch.int32)
    probs = bit_estimator.get_cdf(z + 0.5, qp_index) - bit_estimator.get_cdf(z - 0.5, qp_index)
    return _probs_to_bits(probs)


def _uf_quantize_train(x: torch.Tensor, *, hard_rate: bool = False) -> tuple[torch.Tensor, torch.Tensor]:
    """Return DCVC-UF-style reconstruction and entropy quantization paths.

    The reconstruction path is hard in the forward pass and identity in the
    backward pass (round STE).  The entropy path keeps the classical additive
    uniform-noise relaxation.  Keeping both tensors explicit prevents a noisy
    reconstruction surrogate from leaking into the hyper/spatial prior state.
    """
    hard = x + (torch.round(x) - x).detach()
    noisy = hard if hard_rate else x + torch.empty_like(x).uniform_(-0.5, 0.5)
    return (hard, noisy)


def _task938_width(owner: Any, qp: int | None = None) -> float:
    q = getattr(owner, "_task938_quantizer", None)
    if q is None or not q.enabled:
        return 1.0
    return q.width(int(qp if qp is not None else owner._active_task938_qp))


def _task938_effective_width(owner, scales, means, qp=None):
    base = _task938_width(owner, qp)
    regional = getattr(owner, "task938_region_quantizer", None)
    if regional is None:
        return base
    qp = int(qp if qp is not None else owner._active_task938_qp)
    with torch.autocast(device_type=scales.device.type, enabled=False):
        return regional(scales.detach().float(), means.detach().float(), qp, base_width=base)


def _task938_publish_prior(owner, scales, widths, masks):
    if not getattr(owner, "_export_entropy_scales", False):
        return
    owner._last_entropy_scale_map = _aggregate_decoder_entropy_scales(scales, masks)
    tensors = [torch.ones_like(scale) * width for (scale, width) in zip(scales, widths)]
    squared = sum(((width * mask).square().sum(1, keepdim=True) for (width, mask) in zip(tensors, masks)))
    count = sum((mask.sum(1, keepdim=True) for mask in masks))
    owner._last_quantization_width_map = (squared / count.clamp_min(1)).sqrt()


def _task938_quantize_train(x, owner, *, hard_rate=False, scales=None, mask=None, width=None):
    q = getattr(owner, "_task938_quantizer", None)
    if q is None or not q.enabled:
        return _uf_quantize_train(x, hard_rate=hard_rate)
    width = _task938_width(owner) if width is None else width
    coordinate = x / width
    if scales is None or mask is None:
        raise ValueError("Task938 y quantization needs coder scale/mask")
    with torch.no_grad():
        _, coded, _, _ = owner.model.process_with_mask(coordinate, scales / width, torch.zeros_like(x), mask)
    symbol = coordinate + (coded - coordinate).detach()
    rate_value = symbol if hard_rate else coordinate + torch.empty_like(coordinate).uniform_(-0.5, 0.5)
    return (symbol * width, rate_value)


def _task938_process_mask(owner, y, scales, means, mask, qp, width=None):
    width = _task938_effective_width(owner, scales, means, qp) if width is None else width
    _, symbol, recon, coded_scales = owner.model.process_with_mask(
        (y - means) / width, scales / width, torch.zeros_like(means), mask
    )
    return (symbol, (symbol * width + means) * mask, coded_scales)


def _task938_p_normalize(y, q_dec):
    """Coder-exact P coordinate, with ordinary autograd through both inputs."""
    return y * torch.reciprocal(q_dec)


def _task938_p_prior(owner, params, spatial_prior, qp, y=None):
    """Symmetric two-pass y coder. Hyper-z and all prior networks are inherited."""
    model = owner.model
    if y is None:
        q_dec, scales, means = model.separate_prior_for_video_decoding(params)
    else:
        y, q_dec, scales, means = model.separate_prior_for_video_encoding(params, y)
    masks = model.get_mask_2x(*means.shape, means.dtype, means.device)
    sofar = torch.zeros_like(means)
    symbols, coded_scales = ([], [])
    exported_scales, exported_widths = ([], [])
    for index, mask in enumerate(masks):
        if index:
            scales, means = spatial_prior(torch.cat((sofar, params), 1)).chunk(2, 1)
        width = _task938_effective_width(owner, scales, means, qp)
        exported_widths.append(width)
        if getattr(owner, "_export_entropy_scales", False):
            exported_scales.append((scales / width).detach())
        if y is None:
            scales_read = model.single_part_for_writing_2x(scales * mask / width)
            symbol = model.gaussian_encoder.decode_and_get_y(scales_read, means.dtype, means.device)
            reconstructed = (torch.cat((symbol, symbol), 1) * width + means) * mask
        else:
            symbol, reconstructed, scale = _task938_process_mask(owner, y, scales, means, mask, qp, width=width)
            symbols.append(model.single_part_for_writing_2x(symbol))
            coded_scales.append(model.single_part_for_writing_2x(scale))
        sofar = sofar + reconstructed
    _task938_publish_prior(owner, exported_scales, exported_widths, masks)
    return sofar * q_dec if y is None else (*symbols, *coded_scales, sofar * q_dec)


def _detach_leading_prior_channels(params: torch.Tensor, channels: int, *, enabled: bool) -> torch.Tensor:
    """Stop dynamic-q gradients without detaching scale/mean channels.

    DCVC-RT packs dynamic quantization and entropy-distribution parameters in
    one prior tensor.  Freezing only the final q projection rows is insufficient:
    q gradients can still modify their shared upstream features.  Rate-bridge
    phases use this split/concat barrier so q values remain available in the
    forward pass while scale/mean and spatial-prior gradients remain intact.
    """
    if not enabled:
        return params
    channels = int(channels)
    if channels <= 0 or channels >= int(params.shape[1]):
        raise ValueError(f"detached prior channels must be in [1, {params.shape[1] - 1}], got {channels}")
    return torch.cat((params[:, :channels].detach(), params[:, channels:]), dim=1)


def _initialize_neutral_prior_output_(conv: nn.Conv2d, *, q_channels: int, scale_channels: int, q_bias: float) -> None:
    """Give a scratch entropy head a finite, non-saturated starting point.

    Output layout is ``[q, scale, mean]`` when ``q_channels > 0`` and
    ``[scale, mean]`` otherwise.  Only the final projection is neutralized;
    the transform and prior trunks retain their standard random initialization.
    """
    expected = int(q_channels) + 2 * int(scale_channels)
    if conv.out_channels != expected or conv.bias is None:
        raise ValueError(
            f"neutral prior initialization shape mismatch: out={conv.out_channels}, expected={expected}, bias={conv.bias is not None}"
        )
    q_end = int(q_channels)
    scale_end = q_end + int(scale_channels)
    with torch.no_grad():
        if q_end:
            conv.weight[:q_end].zero_()
            conv.bias[:q_end].fill_(float(q_bias))
        conv.weight[q_end:scale_end].zero_()
        conv.bias[q_end:scale_end].fill_(1.0)
        conv.weight[scale_end:].zero_()
        conv.bias[scale_end:].zero_()


def _image_q_to_logit(q_value: float) -> float:
    """Map an image-prior q value in ``(0.5, 2.0)`` to its raw logit.

    DCVC-RT's image prior uses ``q = sigmoid(raw) * 1.5 + 0.5``.  Scratch
    training previously used raw zero (q=1.25), even though q=1 is the
    identity quantization coordinate.  Keeping the inverse explicit prevents
    future initializers from silently choosing a non-identity q.
    """
    q_value = float(q_value)
    if not 0.5 < q_value < 2.0:
        raise ValueError(f"image q must be in (0.5, 2.0), got {q_value}")
    probability = (q_value - 0.5) / 1.5
    return math.log(probability / (1.0 - probability))


def _rate_to_qp(
    rate: float,
    rate_qp_table: Optional[list] = None,
    rate_max: float = 0.02,
    rate_min: float = 0.003,
    num_qp: int = _NUM_QP,
) -> int:
    """Map target bpp to DCVC-RT qp (0..num_qp-1)."""
    rate = max(float(rate), 0.0001)
    if rate_qp_table is not None and len(rate_qp_table) > 0:
        best_qp = rate_qp_table[0][1]
        best_diff = abs(rate_qp_table[0][0] - rate)
        for r, q in rate_qp_table:
            diff = abs(r - rate)
            if diff < best_diff:
                best_diff = diff
                best_qp = q
        return best_qp
    log_min = math.log(rate_min)
    log_max = math.log(rate_max)
    log_rate = math.log(rate)
    qp_max = num_qp - 1
    qp = int(qp_max - qp_max * (log_rate - log_min) / (log_max - log_min))
    return max(0, min(qp_max, qp))


def _normalize_latent_init_mode(mode: str) -> str:
    """Normalize codec initialization mode names."""
    mode = str(mode).lower().strip()
    aliases = {
        "default": "pretrained_adapted",
        "pretrained": "pretrained_adapted",
        "adapted": "pretrained_adapted",
        "latent_native": "latent_io_random",
        "latent-random": "latent_io_random",
        "random_io": "latent_io_random",
        "io_random": "latent_io_random",
        "from_scratch": "scratch",
    }
    mode = aliases.get(mode, mode)
    if mode not in {"pretrained_adapted", "latent_io_random", "scratch"}:
        raise ValueError(
            f"Unknown latent_init_mode={mode!r}; expected 'pretrained_adapted', 'latent_io_random', or 'scratch'."
        )
    return mode


def _normalize_latent_spatial_mode(mode: str) -> str:
    """Normalize latent spatial transform mode names."""
    mode = str(mode or "dcvc_stride2").lower().strip()
    aliases = {
        "default": "dcvc_stride2",
        "stride2": "dcvc_stride2",
        "dcvc": "dcvc_stride2",
        "dcvc_rt": "dcvc_stride2",
        "preserve": "preserve",
        "native": "preserve",
        "no_down": "preserve",
        "no_downsample": "preserve",
    }
    mode = aliases.get(mode, mode)
    if mode not in {"dcvc_stride2", "preserve"}:
        raise ValueError(f"Unknown latent_spatial_mode={mode!r}; expected 'dcvc_stride2' or 'preserve'.")
    return mode


def _remap_legacy_no_upsample_refine_keys(state_dict: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    """Support old spatial-preserve checkpoints saved before refine became Sequential.

    Earlier task097-style checkpoints used ``NoUpsampleBlock.refine`` directly as
    a DepthConvBlock.  Current checkpoints use ``refine.0`` when there is a
    single block, so old keys need a mechanical rename at load time.
    """
    remapped: dict[str, torch.Tensor] = {}
    for key, value in state_dict.items():
        new_key = key
        for marker in (".model.dec.dec_1.0.refine.", ".model.decoder.up.refine."):
            if marker in new_key:
                suffix = new_key.split(marker, 1)[1]
                first = suffix.split(".", 1)[0]
                if not first.isdigit():
                    new_key = new_key.replace(marker, f"{marker}0.")
        remapped[new_key] = value
    return remapped


class LatentInputStem(nn.Module):
    """Latent-native input stem for Wan latent tensors.

    DCVC-RT's original 192-channel input is RGB pixel-unshuffle specific.  This
    stem maps Wan's 48-channel latent directly into the codec's internal feature
    width with several depthwise residual blocks, avoiding a mechanical
    48->192 compatibility expansion.
    """

    def __init__(self, in_ch: int, out_ch: int, num_blocks: int = 2):
        super().__init__()
        from src.layers.layers import DepthConvBlock

        num_blocks = max(1, int(num_blocks))
        blocks: list[nn.Module] = [DepthConvBlock(in_ch, out_ch)]
        for _ in range(num_blocks - 1):
            blocks.append(DepthConvBlock(out_ch, out_ch, shortcut=True))
        self.blocks = nn.ModuleList(blocks)

    def forward(self, x: torch.Tensor, quant_step: Optional[torch.Tensor] = None) -> torch.Tensor:
        for block in self.blocks:
            x = block(x)
        if quant_step is not None:
            x = x * quant_step
        return x


class PContextPyramidAnalysis(nn.Module):
    """Early decoder-known reference interaction inside the P analysis path.

    The parent DMC transform does not expose temporal context until its second
    encoder block.  This module lets the first latent feature interact with the
    hard-decoded reference context at both the native and one-octave-lower
    spatial scales before that block.  It emits only a bounded residual on the
    existing feature: no side symbols or parallel reconstruction path are
    introduced.  The output projection is zero initialized, making a newly
    enabled module exactly equivalent to the parent codec at step zero.
    """

    def __init__(
        self,
        feature_channels: int = _RT_D,
        context_channels: int = _RT_D,
        hidden_channels: int = 192,
        blocks: int = 3,
        residual_scale: float = 0.25,
        qp_count: int = _INTER_QP_NUM,
        position_count: int = 3,
        qp_gate_max_log_gain: float = 0.75,
        qp_gate_max_bias: float = 0.25,
        affine_modulation: bool = False,
        affine_max_log_scale: float = 0.15,
        content_gate: bool = False,
    ) -> None:
        super().__init__()
        hidden_channels = int(hidden_channels)
        blocks = max(1, int(blocks))
        if hidden_channels <= 0 or residual_scale <= 0:
            raise ValueError("P context-pyramid width and residual scale must be positive")
        self.hidden_channels = hidden_channels
        self.blocks_count = blocks
        self.residual_scale = float(residual_scale)
        self.position_count = max(1, int(position_count))
        self.qp_gate_values = (0, 16, 39, 63)
        self.qp_gate_max_log_gain = float(qp_gate_max_log_gain)
        self.qp_gate_max_bias = float(qp_gate_max_bias)
        self.affine_modulation = bool(affine_modulation)
        self.affine_max_log_scale = float(affine_max_log_scale)
        self.content_gate_enabled = bool(content_gate)
        if self.qp_gate_max_log_gain <= 0 or self.qp_gate_max_bias <= 0:
            raise ValueError("P context-pyramid QP gate bounds must be positive")
        if self.affine_max_log_scale <= 0:
            raise ValueError("P context-pyramid affine bound must be positive")
        self.input = nn.Conv2d(int(feature_channels) + int(context_channels), hidden_channels, 1)
        self.qp_embedding = nn.Embedding(max(1, int(qp_count)), hidden_channels)
        self.position_embedding = nn.Embedding(self.position_count, hidden_channels)

        def residual_block() -> nn.Module:
            return nn.Sequential(
                nn.Conv2d(hidden_channels, hidden_channels, 3, padding=1, groups=hidden_channels),
                nn.GELU(),
                nn.Conv2d(hidden_channels, hidden_channels, 1),
                nn.GELU(),
            )

        self.full_blocks = nn.ModuleList((residual_block() for _ in range(blocks)))
        self.coarse_blocks = nn.ModuleList((residual_block() for _ in range(max(1, blocks // 2))))
        self.fuse = nn.Conv2d(2 * hidden_channels, hidden_channels, 1)
        self.content_gate = None
        if self.content_gate_enabled:
            self.content_gate = nn.Conv2d(hidden_channels, 1, 1)
            nn.init.zeros_(self.content_gate.weight)
            nn.init.zeros_(self.content_gate.bias)
        self.output = nn.Conv2d(hidden_channels, int(feature_channels), 3, padding=1)
        nn.init.zeros_(self.output.weight)
        nn.init.zeros_(self.output.bias)
        if self.affine_modulation:
            self.affine_scale = nn.Conv2d(hidden_channels, int(feature_channels), 1)
            nn.init.zeros_(self.affine_scale.weight)
            nn.init.zeros_(self.affine_scale.bias)
            self.affine_qp_gain_logit = nn.Parameter(torch.zeros(len(self.qp_gate_values), 1, 1, 1))
        else:
            self.affine_scale = None
            self.affine_qp_gain_logit = None
        self.qp_output_log_gain = nn.Parameter(torch.zeros(len(self.qp_gate_values), int(feature_channels), 1, 1))
        self.qp_output_bias = nn.Parameter(torch.zeros(len(self.qp_gate_values), int(feature_channels), 1, 1))

    def contract_payload(self) -> dict[str, Any]:
        return {
            "schema": "p_context_pyramid_analysis_v5",
            "hidden_channels": self.hidden_channels,
            "blocks": self.blocks_count,
            "residual_scale": self.residual_scale,
            "position_count": self.position_count,
            "qp_gate_values": list(self.qp_gate_values),
            "qp_gate_max_log_gain": self.qp_gate_max_log_gain,
            "qp_gate_max_bias": self.qp_gate_max_bias,
            "affine_modulation": self.affine_modulation,
            "affine_max_log_scale": self.affine_max_log_scale,
            "affine_qp_gate": "one_plus_tanh_decoder_known_qp" if self.affine_modulation else None,
            "content_gate": "one_plus_tanh_spatial_fused_feature" if self.content_gate_enabled else None,
        }

    def _qp_gate_index(self, qp: int) -> int:
        value = int(qp)
        return min(range(len(self.qp_gate_values)), key=lambda index: abs(value - self.qp_gate_values[index]))

    def forward(self, feature: torch.Tensor, context: torch.Tensor, qp: int, frame_type_idx: int) -> torch.Tensor:
        if feature.shape != context.shape:
            raise ValueError(
                f"P context-pyramid expects matching feature/context shapes, got {tuple(feature.shape)} and {tuple(context.shape)}"
            )
        qp_index = max(0, min(int(qp), self.qp_embedding.num_embeddings - 1))
        position_index = max(0, min(int(frame_type_idx), self.position_count - 1))
        condition = (self.qp_embedding.weight[qp_index] + self.position_embedding.weight[position_index]).to(
            device=feature.device, dtype=feature.dtype
        )[None, :, None, None]
        full = self.input(torch.cat((feature, context), dim=1)) + condition
        for block in self.full_blocks:
            full = full + block(full)
        coarse = F.avg_pool2d(full, kernel_size=2, stride=2, ceil_mode=True)
        for block in self.coarse_blocks:
            coarse = coarse + block(coarse)
        coarse = F.interpolate(coarse, size=full.shape[-2:], mode="bilinear", align_corners=False)
        fused = self.fuse(torch.cat((full, coarse), dim=1))
        content_gain: torch.Tensor | float = 1.0
        if self.content_gate is not None:
            content_gain = 1.0 + torch.tanh(self.content_gate(fused))
        delta = torch.tanh(self.output(fused))
        gate_index = self._qp_gate_index(qp)
        gain = torch.exp(torch.tanh(self.qp_output_log_gain[gate_index]) * self.qp_gate_max_log_gain).to(
            device=delta.device, dtype=delta.dtype
        )
        bias = (torch.tanh(self.qp_output_bias[gate_index]) * self.qp_gate_max_bias).to(
            device=delta.device, dtype=delta.dtype
        )
        delta = (delta * gain + bias) * content_gain
        if self.affine_scale is None:
            return feature + self.residual_scale * delta
        assert self.affine_qp_gain_logit is not None
        affine_gain = 1.0 + torch.tanh(self.affine_qp_gain_logit[gate_index]).to(
            device=feature.device, dtype=feature.dtype
        )
        log_scale = torch.tanh(self.affine_scale(fused)) * self.affine_max_log_scale * affine_gain * content_gain
        return feature * torch.exp(log_scale) + self.residual_scale * delta


class PContextPyramidSynthesis(PContextPyramidAnalysis):
    """Decoder-known context transform on the main P synthesis feature.

    This is the synthesis half of the symmetric Task487 transform.  It runs
    before the shared reconstruction core, so its result is used both for the
    displayed latent and for the next hard-decoded P reference.  It introduces
    no side symbols and cannot access encoder-private features.
    """

    def contract_payload(self) -> dict[str, Any]:
        payload = super().contract_payload()
        payload["schema"] = "p_context_pyramid_synthesis_v1"
        return payload


class PLatentLiftingSubnet(nn.Module):
    """Conditioned update used by an exactly invertible additive coupling."""

    def __init__(self, channels: int, hidden_channels: int) -> None:
        super().__init__()
        self.input = nn.Conv2d(channels, hidden_channels, 3, padding=1)
        self.depthwise = nn.Conv2d(hidden_channels, hidden_channels, 3, padding=1, groups=hidden_channels)
        self.output = nn.Conv2d(hidden_channels, channels, 1)
        nn.init.zeros_(self.output.weight)
        nn.init.zeros_(self.output.bias)

    def forward(self, x: torch.Tensor, condition: torch.Tensor) -> torch.Tensor:
        hidden = self.input(x) + condition
        hidden = hidden + self.depthwise(F.silu(hidden))
        return torch.tanh(self.output(F.silu(hidden)))


class PInvertibleLatentLifting(nn.Module):
    """Identity-initialized latent preconditioner around the mature P core.

    The 48 Wan channels are split in half and transformed by two additive
    coupling steps.  The inverse executes the same steps in reverse, so this
    module cannot create an out-of-loop display path or require side bits.  It
    changes only the coordinate presented to the proven Task297 transform,
    allowing rate-distortion training to expose spatial/channel dependencies
    that its fixed first layer did not align with.
    """

    def __init__(
        self,
        *,
        latent_channels: int = _VAE_CH,
        hidden_channels: int = 96,
        qp_count: int = _INTER_QP_NUM,
        position_count: int = 3,
        update_scale: float = 0.25,
        perceptual_channel_weights: Optional[Sequence[float]] = None,
        companding_strength: float = 0.0,
        companding_max_log_scale: float = 0.75,
    ) -> None:
        super().__init__()
        if latent_channels % 2 != 0:
            raise ValueError("invertible latent lifting requires even channels")
        if hidden_channels <= 0 or update_scale <= 0:
            raise ValueError("latent lifting width/scale must be positive")
        self.latent_channels = int(latent_channels)
        self.half_channels = self.latent_channels // 2
        self.hidden_channels = int(hidden_channels)
        self.position_count = max(1, int(position_count))
        self.update_scale = float(update_scale)
        self.companding_strength = float(companding_strength)
        self.companding_max_log_scale = float(companding_max_log_scale)
        if self.companding_strength < 0 or self.companding_max_log_scale <= 0:
            raise ValueError("invalid latent companding strength/log-scale")
        companding_scale: Optional[torch.Tensor] = None
        if perceptual_channel_weights is not None and self.companding_strength > 0:
            weights = torch.as_tensor(list(perceptual_channel_weights), dtype=torch.float32)
            if weights.numel() != self.latent_channels:
                raise ValueError(
                    f"perceptual latent companding requires one positive weight per channel, got {weights.numel()} for {self.latent_channels}"
                )
            if not torch.isfinite(weights).all() or not bool((weights > 0).all()):
                raise ValueError("perceptual latent companding weights must be positive")
            log_scale = 0.5 * self.companding_strength * torch.log(weights)
            log_scale = log_scale - log_scale.mean()
            log_scale = log_scale.clamp(-self.companding_max_log_scale, self.companding_max_log_scale)
            companding_scale = log_scale.exp().view(1, self.latent_channels, 1, 1)
        elif self.companding_strength != 0:
            raise ValueError("non-zero companding strength requires channel weights")
        self.register_buffer("companding_scale", companding_scale)
        self.qp_embedding = nn.Embedding(max(1, int(qp_count)), hidden_channels)
        self.position_embedding = nn.Embedding(self.position_count, hidden_channels)
        nn.init.zeros_(self.qp_embedding.weight)
        nn.init.zeros_(self.position_embedding.weight)
        self.update_b = PLatentLiftingSubnet(self.half_channels, hidden_channels)
        self.update_a = PLatentLiftingSubnet(self.half_channels, hidden_channels)

    def contract_payload(self) -> dict[str, Any]:
        payload = {
            "schema": (
                "p_invertible_latent_lifting_companded_v2"
                if self.companding_scale is not None
                else "p_invertible_latent_lifting_v1"
            ),
            "latent_channels": self.latent_channels,
            "hidden_channels": self.hidden_channels,
            "position_count": self.position_count,
            "update_scale": self.update_scale,
            "coupling": "two_step_additive_exact_inverse",
            "side_information": False,
        }
        if self.companding_scale is not None:
            scale = self.companding_scale.detach().float().cpu()
            payload.update(
                {
                    "companding_coordinate": "previous_hard_reference_centered",
                    "companding_strength": self.companding_strength,
                    "companding_max_log_scale": self.companding_max_log_scale,
                    "companding_scale_sha256": hashlib.sha256(scale.numpy().tobytes()).hexdigest(),
                    "companding_scale_min": float(scale.min()),
                    "companding_scale_max": float(scale.max()),
                    "companding_volume_normalized": True,
                }
            )
        return payload

    def _condition(self, x: torch.Tensor, qp: int, frame_type_idx: int) -> torch.Tensor:
        qp_index = max(0, min(int(qp), self.qp_embedding.num_embeddings - 1))
        position_index = max(0, min(int(frame_type_idx), self.position_count - 1))
        value = self.qp_embedding.weight[qp_index] + self.position_embedding.weight[position_index]
        return value.to(device=x.device, dtype=x.dtype)[None, :, None, None]

    def analysis(
        self, x: torch.Tensor, qp: int, frame_type_idx: int, reference: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        if self.companding_scale is not None:
            if reference is None:
                raise ValueError("latent companding requires a hard decoded reference")
            scale = self.companding_scale.to(device=x.device, dtype=x.dtype)
            x = reference + (x - reference) * scale
        condition = self._condition(x, qp, frame_type_idx)
        a, b = x.chunk(2, dim=1)
        b = b + self.update_scale * self.update_b(a, condition)
        a = a + self.update_scale * self.update_a(b, condition)
        return torch.cat((a, b), dim=1)

    def synthesis(
        self, x: torch.Tensor, qp: int, frame_type_idx: int, reference: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        condition = self._condition(x, qp, frame_type_idx)
        a, b = x.chunk(2, dim=1)
        a = a - self.update_scale * self.update_a(b, condition)
        b = b - self.update_scale * self.update_b(a, condition)
        x = torch.cat((a, b), dim=1)
        if self.companding_scale is not None:
            if reference is None:
                raise ValueError("latent companding requires a hard decoded reference")
            scale = self.companding_scale.to(device=x.device, dtype=x.dtype)
            x = reference + (x - reference) / scale
        return x


class PReferenceTranslationLifting(nn.Module):
    """Decoder-closed predictive coordinate around the mature P codec.

    The same prediction made solely from the previous hard reconstruction is
    subtracted before analysis and added after synthesis.  A zero-initialized
    per-QP/role/channel gate makes step zero exactly the mature parent while
    still giving the gate a direct gradient.  Once the gate opens, the small
    reference filter can learn a better decoder-known predictor.  There is no
    side stream and no display-only correction path.
    """

    def __init__(
        self,
        *,
        latent_channels: int = _VAE_CH,
        hidden_channels: int = 96,
        qp_count: int = _INTER_QP_NUM,
        position_count: int = 3,
        max_alpha: float = 1.0,
        predictor_scale: float = 0.25,
        base_alpha: float = 0.0,
    ) -> None:
        super().__init__()
        if hidden_channels <= 0 or max_alpha <= 0 or predictor_scale <= 0:
            raise ValueError("reference translation widths/scales must be positive")
        if not 0.0 <= float(base_alpha) < 1.0:
            raise ValueError("reference translation base_alpha must be in [0, 1)")
        self.latent_channels = int(latent_channels)
        self.hidden_channels = int(hidden_channels)
        self.position_count = max(1, int(position_count))
        self.max_alpha = float(max_alpha)
        self.predictor_scale = float(predictor_scale)
        self.base_alpha = float(base_alpha)
        self.qp_count = max(1, int(qp_count))
        self.gate = nn.Parameter(torch.zeros(self.position_count, self.qp_count, self.latent_channels, 1, 1))
        self.qp_embedding = nn.Embedding(self.qp_count, hidden_channels)
        self.position_embedding = nn.Embedding(self.position_count, hidden_channels)
        nn.init.zeros_(self.qp_embedding.weight)
        nn.init.zeros_(self.position_embedding.weight)
        self.input = nn.Conv2d(self.latent_channels, hidden_channels, 3, padding=1)
        self.depthwise = nn.Conv2d(hidden_channels, hidden_channels, 5, padding=2, groups=hidden_channels)
        self.output = nn.Conv2d(hidden_channels, self.latent_channels, 1)
        nn.init.zeros_(self.output.weight)
        nn.init.zeros_(self.output.bias)

    def contract_payload(self) -> dict[str, Any]:
        payload = {
            "schema": (
                "p_reference_translation_lifting_v1" if self.base_alpha == 0.0 else "p_reference_translation_lifting_v2"
            ),
            "latent_channels": self.latent_channels,
            "hidden_channels": self.hidden_channels,
            "position_count": self.position_count,
            "max_alpha": self.max_alpha,
            "predictor_scale": self.predictor_scale,
            "coordinate": "normalized_hard_previous_latent",
            "operation": "analysis_subtract_synthesis_add",
            "side_information": False,
        }
        if self.base_alpha != 0.0:
            payload.update(
                {
                    "base_alpha": self.base_alpha,
                    "gate_parameterization": "base_logit_plus_trainable_delta",
                    "initial_coordinate": "partial_hard_reference_residual",
                }
            )
        return payload

    def prediction(self, reference: torch.Tensor, qp: int, frame_type_idx: int) -> torch.Tensor:
        if int(reference.shape[1]) != self.latent_channels:
            raise ValueError(
                f"reference translation latent channel mismatch: {reference.shape[1]} != {self.latent_channels}"
            )
        qp_index = max(0, min(int(qp), self.qp_count - 1))
        position_index = max(0, min(int(frame_type_idx), self.position_count - 1))
        condition = (self.qp_embedding.weight[qp_index] + self.position_embedding.weight[position_index]).to(
            device=reference.device, dtype=reference.dtype
        )[None, :, None, None]
        hidden = self.input(reference) + condition
        hidden = hidden + self.depthwise(F.silu(hidden))
        residual = torch.tanh(self.output(F.silu(hidden)))
        predictor = reference + self.predictor_scale * residual
        gate = self.gate[position_index, qp_index]
        if self.base_alpha == 0.0:
            alpha = torch.tanh(gate) * self.max_alpha
        else:
            base_logit = math.log(self.base_alpha / (1.0 - self.base_alpha))
            alpha = torch.sigmoid(gate + base_logit) * self.max_alpha
        alpha = alpha.to(device=reference.device, dtype=reference.dtype)
        return predictor * alpha

    def analysis(self, x: torch.Tensor, reference: torch.Tensor, qp: int, frame_type_idx: int) -> torch.Tensor:
        return x - self.prediction(reference, qp, frame_type_idx)

    def synthesis(self, x: torch.Tensor, reference: torch.Tensor, qp: int, frame_type_idx: int) -> torch.Tensor:
        return x + self.prediction(reference, qp, frame_type_idx)


class PSharedQueryMotionResidual(nn.Module):
    """Shared motion-compensated high-frequency coordinate around P coding.

    Analysis queries the hard reference with the current latent; synthesis uses
    the mature decoded latent as the query.  One shared matcher predicts flow
    and confidence, so no flow side stream or encoder-private feature reaches
    the decoder.  A zero QP/position gate is exactly the parent codec.
    """

    def __init__(
        self,
        *,
        latent_channels: int = _VAE_CH,
        hidden_channels: int = 96,
        blocks: int = 3,
        qp_count: int = _INTER_QP_NUM,
        position_count: int = 3,
        max_displacement: float = 2.0,
        max_alpha: float = 0.5,
        highpass_kernel: int = 3,
        reliability_temperature: float = 0.0,
        channel_indices: Optional[Sequence[int]] = None,
    ) -> None:
        super().__init__()
        if hidden_channels <= 0 or blocks <= 0:
            raise ValueError("shared-query motion residual width/blocks must be positive")
        if max_displacement <= 0 or max_alpha <= 0:
            raise ValueError("shared-query motion residual bounds must be positive")
        if highpass_kernel < 3 or highpass_kernel % 2 == 0:
            raise ValueError("shared-query highpass kernel must be odd and >=3")
        self.latent_channels = int(latent_channels)
        self.hidden_channels = int(hidden_channels)
        self.blocks_count = int(blocks)
        self.qp_count = max(1, int(qp_count))
        self.position_count = max(1, int(position_count))
        self.max_displacement = float(max_displacement)
        self.max_alpha = float(max_alpha)
        self.highpass_kernel = int(highpass_kernel)
        self.reliability_temperature = float(reliability_temperature)
        if self.reliability_temperature < 0:
            raise ValueError("shared-query reliability temperature must be non-negative")
        if channel_indices is None:
            selected = tuple(range(self.latent_channels))
        else:
            selected = tuple(sorted({int(value) for value in channel_indices}))
            if not selected or selected[0] < 0 or selected[-1] >= self.latent_channels:
                raise ValueError("shared-query motion channel indices are invalid")
        channel_gain = torch.zeros(1, self.latent_channels, 1, 1)
        channel_gain[:, list(selected)] = 1.0
        self.register_buffer("channel_gain", channel_gain, persistent=True)
        self.channel_indices = selected
        self.input = nn.Conv2d(3 * self.latent_channels, self.hidden_channels, 1)
        self.qp_embedding = nn.Embedding(self.qp_count, self.hidden_channels)
        self.position_embedding = nn.Embedding(self.position_count, self.hidden_channels)
        self.blocks = nn.ModuleList(
            (
                nn.Sequential(
                    nn.Conv2d(self.hidden_channels, self.hidden_channels, 3, padding=1, groups=self.hidden_channels),
                    nn.GELU(),
                    nn.Conv2d(self.hidden_channels, self.hidden_channels, 1),
                    nn.GELU(),
                )
                for _ in range(self.blocks_count)
            )
        )
        self.flow_confidence = nn.Conv2d(self.hidden_channels, 3, 3, padding=1)
        nn.init.normal_(self.flow_confidence.weight, mean=0.0, std=0.001)
        nn.init.zeros_(self.flow_confidence.bias)
        self.gate = nn.Parameter(torch.zeros(self.position_count, self.qp_count, 1, 1, 1))

    def contract_payload(self) -> dict[str, Any]:
        return {
            "schema": "p_shared_query_motion_residual_v1",
            "analysis_query": "current_latent",
            "synthesis_query": "mature_hard_decoded_latent",
            "reference": "immediate_hard_decoded_latent",
            "prediction": "flow_aligned_reference_highpass",
            "highpass_kernel": self.highpass_kernel,
            "max_displacement": self.max_displacement,
            "max_alpha": self.max_alpha,
            "reliability": (
                "aligned_lowpass_exponential_agreement"
                if self.reliability_temperature > 0
                else "learned_confidence_only"
            ),
            "reliability_temperature": self.reliability_temperature,
            "prediction_channel_indices": list(self.channel_indices),
            "shared_matcher": True,
            "side_information": False,
        }

    @staticmethod
    def _base_grid(value: torch.Tensor) -> torch.Tensor:
        batch, _, height, width = value.shape
        ys = torch.linspace(-1.0, 1.0, height, device=value.device, dtype=value.dtype)
        xs = torch.linspace(-1.0, 1.0, width, device=value.device, dtype=value.dtype)
        yy, xx = torch.meshgrid(ys, xs, indexing="ij")
        return torch.stack((xx, yy), dim=-1)[None].expand(batch, -1, -1, -1)

    def prediction(self, query: torch.Tensor, reference: torch.Tensor, qp: int, frame_type_idx: int) -> torch.Tensor:
        if query.shape != reference.shape:
            raise ValueError("shared-query motion residual requires equal latent shapes")
        qp_index = max(0, min(int(qp), self.qp_count - 1))
        position_index = max(0, min(int(frame_type_idx), self.position_count - 1))
        hidden = self.input(torch.cat((query, reference, query - reference), dim=1))
        condition = (self.qp_embedding.weight[qp_index] + self.position_embedding.weight[position_index]).to(
            device=query.device, dtype=query.dtype
        )[None, :, None, None]
        hidden = hidden + condition
        for block in self.blocks:
            hidden = hidden + block(hidden)
        flow_raw, confidence_raw = self.flow_confidence(hidden).split((2, 1), dim=1)
        flow_pixels = torch.tanh(flow_raw) * self.max_displacement
        height, width = reference.shape[-2:]
        normalizer = reference.new_tensor([2.0 / max(width - 1, 1), 2.0 / max(height - 1, 1)])[None, :, None, None]
        grid = self._base_grid(reference) + (flow_pixels * normalizer).permute(0, 2, 3, 1)
        lowpass = F.avg_pool2d(reference, kernel_size=self.highpass_kernel, stride=1, padding=self.highpass_kernel // 2)
        highpass = reference - lowpass
        aligned = F.grid_sample(highpass, grid, mode="bilinear", padding_mode="border", align_corners=True)
        confidence = torch.sigmoid(confidence_raw)
        if self.reliability_temperature > 0:
            aligned_lowpass = F.grid_sample(lowpass, grid, mode="bilinear", padding_mode="border", align_corners=True)
            query_lowpass = F.avg_pool2d(
                query, kernel_size=self.highpass_kernel, stride=1, padding=self.highpass_kernel // 2
            )
            disagreement = (query_lowpass - aligned_lowpass).abs().mean(dim=1, keepdim=True)
            confidence = confidence * torch.exp(-self.reliability_temperature * disagreement)
        alpha = (torch.tanh(self.gate[position_index, qp_index]) * self.max_alpha).to(
            device=query.device, dtype=query.dtype
        )
        return alpha * confidence * aligned * self.channel_gain.to(device=aligned.device, dtype=aligned.dtype)

    def analysis(self, x: torch.Tensor, reference: torch.Tensor, qp: int, frame_type_idx: int) -> torch.Tensor:
        return x - self.prediction(x, reference, qp, frame_type_idx)

    def synthesis(self, x: torch.Tensor, reference: torch.Tensor, qp: int, frame_type_idx: int) -> torch.Tensor:
        return x + self.prediction(x, reference, qp, frame_type_idx)


class PYConditionedContextWarp(nn.Module):
    """Warp hard-reference context in a shared analysis/synthesis coordinate.

    On synthesis the query is carried by decoded ``y`` and therefore requires
    no flow side stream.  The optional analysis half uses the current-frame
    stem feature as its query before the parent's first context fusion.  Both
    halves share the same matcher, which makes the emitted symbols learn a
    bottleneck-compatible motion coordinate instead of adding an out-of-loop
    display correction.  A zero gate is an exact parent model.
    """

    def __init__(
        self,
        *,
        channels: int = _RT_D,
        hidden_channels: int = 128,
        blocks: int = 3,
        qp_count: int = _INTER_QP_NUM,
        position_count: int = 3,
        max_displacement: float = 4.0,
        warp_mode: str = "full",
        lowpass_kernel: int = 3,
        analysis_enabled: bool = False,
        base_y_channels: int = 0,
    ) -> None:
        super().__init__()
        if hidden_channels <= 0 or blocks <= 0 or max_displacement <= 0:
            raise ValueError("y-conditioned context warp values must be positive")
        self.channels = int(channels)
        self.hidden_channels = int(hidden_channels)
        self.blocks_count = int(blocks)
        self.position_count = max(1, int(position_count))
        self.qp_count = max(1, int(qp_count))
        self.max_displacement = float(max_displacement)
        self.warp_mode = str(warp_mode).lower().strip()
        if self.warp_mode not in {"full", "lowpass"}:
            raise ValueError("y-conditioned context warp mode must be full or lowpass")
        self.lowpass_kernel = int(lowpass_kernel)
        if self.lowpass_kernel < 3 or self.lowpass_kernel % 2 == 0:
            raise ValueError("context warp lowpass kernel must be odd and >= 3")
        self.analysis_enabled = bool(analysis_enabled)
        self.base_y_channels = int(base_y_channels)
        if self.base_y_channels not in {0, _RT_Y // 4}:
            raise ValueError("base-y warp channels must be 0 or the first 32-channel group")
        if self.analysis_enabled and self.base_y_channels:
            raise ValueError("base-y context warp is decoder-led and cannot use the private analysis query")
        self.input = nn.Conv2d(2 * self.channels, hidden_channels, 1)
        self.qp_embedding = nn.Embedding(self.qp_count, hidden_channels)
        self.position_embedding = nn.Embedding(self.position_count, hidden_channels)
        self.gate = nn.Parameter(torch.zeros(self.position_count, self.qp_count, 1, 1, 1))
        self.blocks = nn.ModuleList(
            (
                nn.Sequential(
                    nn.Conv2d(hidden_channels, hidden_channels, 3, padding=1, groups=hidden_channels),
                    nn.GELU(),
                    nn.Conv2d(hidden_channels, hidden_channels, 1),
                    nn.GELU(),
                )
                for _ in range(self.blocks_count)
            )
        )
        self.flow = nn.Conv2d(hidden_channels, 2, 3, padding=1)
        nn.init.normal_(self.flow.weight, mean=0.0, std=0.001)
        nn.init.zeros_(self.flow.bias)

    def contract_payload(self) -> dict[str, Any]:
        payload = {
            "schema": "p_shared_query_context_warp_v2" if self.analysis_enabled else "p_y_conditioned_context_warp_v1",
            "channels": self.channels,
            "hidden_channels": self.hidden_channels,
            "blocks": self.blocks_count,
            "position_count": self.position_count,
            "max_displacement": self.max_displacement,
            "warp_mode": self.warp_mode,
            "lowpass_kernel": self.lowpass_kernel,
            "motion_source": "decoded_main_y_feature",
            "reference_source": "immediate_hard_context",
            "identity_gate": "qp_position_zero_initialized_tanh",
            "side_information": False,
            "query_source": (
                "first_even4_decoded_y_group_32_channels" if self.base_y_channels else "full_decoded_main_y_feature"
            ),
        }
        if self.analysis_enabled:
            payload.update(
                {
                    "analysis_query": "current_latent_stem_feature",
                    "synthesis_query": "decoded_main_y_feature",
                    "shared_matcher": True,
                }
            )
        return payload

    def decoded_query(self, y_hat: torch.Tensor, decoder_up: nn.Module, full_feature: torch.Tensor) -> torch.Tensor:
        """Build the decoder-known motion query from one mandatory y namespace.

        With ``base_y_channels>0`` the first channel group is decoded before the
        later groups by the group prior and becomes the only alignment query.
        The remaining y channels still share the same transform and bitstream;
        this is a semantic ordering inside y, not an auxiliary motion payload.
        """
        if self.base_y_channels == 0:
            return full_feature
        mask = torch.zeros_like(y_hat)
        mask[:, 0::4] = y_hat[:, 0::4]
        return decoder_up(mask)

    @staticmethod
    def _base_grid(context: torch.Tensor) -> torch.Tensor:
        batch, _, height, width = context.shape
        ys = torch.linspace(-1.0, 1.0, height, device=context.device, dtype=context.dtype)
        xs = torch.linspace(-1.0, 1.0, width, device=context.device, dtype=context.dtype)
        yy, xx = torch.meshgrid(ys, xs, indexing="ij")
        return torch.stack((xx, yy), dim=-1)[None].expand(batch, -1, -1, -1)

    def forward(
        self, decoded_y_feature: torch.Tensor, context: torch.Tensor, qp: int, frame_type_idx: int
    ) -> torch.Tensor:
        if decoded_y_feature.shape != context.shape:
            raise ValueError(
                f"y-conditioned context warp expects equal feature/context shapes, got {tuple(decoded_y_feature.shape)} and {tuple(context.shape)}"
            )
        qp_index = max(0, min(int(qp), self.qp_count - 1))
        position_index = max(0, min(int(frame_type_idx), self.position_count - 1))
        condition = (self.qp_embedding.weight[qp_index] + self.position_embedding.weight[position_index]).to(
            device=context.device, dtype=context.dtype
        )[None, :, None, None]
        hidden = self.input(torch.cat((decoded_y_feature, context), dim=1)) + condition
        for block in self.blocks:
            hidden = hidden + block(hidden)
        flow_pixels = torch.tanh(self.flow(hidden)) * self.max_displacement
        height, width = context.shape[-2:]
        normalizer = context.new_tensor([2.0 / max(width - 1, 1), 2.0 / max(height - 1, 1)])[None, :, None, None]
        flow = (flow_pixels * normalizer).permute(0, 2, 3, 1)
        base = self._base_grid(context)
        warp_source = context
        if self.warp_mode == "lowpass":
            warp_source = F.avg_pool2d(
                context, kernel_size=self.lowpass_kernel, stride=1, padding=self.lowpass_kernel // 2
            )
        sampled = F.grid_sample(warp_source, base + flow, mode="bilinear", padding_mode="border", align_corners=True)
        sampled_zero = F.grid_sample(warp_source, base, mode="bilinear", padding_mode="border", align_corners=True)
        gate = torch.tanh(self.gate[position_index, qp_index]).to(device=context.device, dtype=context.dtype)
        return context + gate * (sampled - sampled_zero)


class PReferenceAffineLifting(nn.Module):
    """Shared decoder-closed affine coordinate for P analysis and synthesis.

    A single reference-only predictor produces a bounded mean and log scale.
    Analysis normalizes the current feature into that coordinate and synthesis
    applies the exact inverse coordinate to the decoded-y feature.  Both sides
    call this same module, so this is a paired transform rather than two
    independently learned correction heads.
    """

    def __init__(
        self,
        channels: int = _RT_D,
        context_channels: int = _RT_D,
        hidden_channels: int = 160,
        blocks: int = 3,
        qp_count: int = _INTER_QP_NUM,
        position_count: int = 3,
        max_log_scale: float = 0.2,
        max_mean: float = 0.35,
    ) -> None:
        super().__init__()
        if hidden_channels <= 0 or blocks <= 0:
            raise ValueError("reference affine-lifting width/blocks must be positive")
        if max_log_scale <= 0 or max_mean <= 0:
            raise ValueError("reference affine-lifting bounds must be positive")
        self.channels = int(channels)
        self.context_channels = int(context_channels)
        self.hidden_channels = int(hidden_channels)
        self.blocks_count = int(blocks)
        self.position_count = max(1, int(position_count))
        self.max_log_scale = float(max_log_scale)
        self.max_mean = float(max_mean)
        self.input = nn.Conv2d(self.context_channels, self.hidden_channels, 1)
        self.qp_embedding = nn.Embedding(max(1, int(qp_count)), self.hidden_channels)
        self.position_embedding = nn.Embedding(self.position_count, self.hidden_channels)

        def block() -> nn.Module:
            return nn.Sequential(
                nn.Conv2d(self.hidden_channels, self.hidden_channels, 3, padding=1, groups=self.hidden_channels),
                nn.GELU(),
                nn.Conv2d(self.hidden_channels, self.hidden_channels, 1),
                nn.GELU(),
            )

        self.full_blocks = nn.ModuleList((block() for _ in range(self.blocks_count)))
        self.coarse_blocks = nn.ModuleList((block() for _ in range(max(1, self.blocks_count // 2))))
        self.fuse = nn.Conv2d(2 * self.hidden_channels, self.hidden_channels, 1)
        self.output = nn.Conv2d(self.hidden_channels, 2 * self.channels, 1)
        nn.init.zeros_(self.output.weight)
        nn.init.zeros_(self.output.bias)
        self._analysis_parameter_cache: Optional[tuple[tuple[int, int, int, str], torch.Tensor, torch.Tensor]] = None

    def contract_payload(self) -> dict[str, Any]:
        return {
            "schema": "p_reference_affine_lifting_v1",
            "channels": self.channels,
            "context_channels": self.context_channels,
            "hidden_channels": self.hidden_channels,
            "blocks": self.blocks_count,
            "position_count": self.position_count,
            "max_log_scale": self.max_log_scale,
            "max_mean": self.max_mean,
        }

    @staticmethod
    def _reference(context: torch.Tensor, mode: str) -> torch.Tensor:
        mode = str(mode).lower().strip()
        if mode == "correct":
            return context
        if mode == "zero":
            return torch.zeros_like(context)
        if mode == "wrong":
            shift_h = max(1, int(context.shape[-2]) // 2)
            shift_w = max(1, int(context.shape[-1]) // 2)
            return torch.roll(context, shifts=(shift_h, shift_w), dims=(-2, -1))
        raise ValueError(f"unknown reference affine-lifting mode {mode!r}")

    def parameters_from_reference(
        self, context: torch.Tensor, qp: int, frame_type_idx: int, mode: str
    ) -> tuple[torch.Tensor, torch.Tensor]:
        context = self._reference(context, mode)
        qp_index = max(0, min(int(qp), self.qp_embedding.num_embeddings - 1))
        position_index = max(0, min(int(frame_type_idx), self.position_count - 1))
        condition = (self.qp_embedding.weight[qp_index] + self.position_embedding.weight[position_index]).to(
            device=context.device, dtype=context.dtype
        )[None, :, None, None]
        full = self.input(context) + condition
        for block in self.full_blocks:
            full = full + block(full)
        coarse = F.avg_pool2d(full, kernel_size=2, stride=2, ceil_mode=True)
        for block in self.coarse_blocks:
            coarse = coarse + block(coarse)
        coarse = F.interpolate(coarse, size=full.shape[-2:], mode="bilinear", align_corners=False)
        fused = self.fuse(torch.cat((full, coarse), dim=1))
        mean_raw, log_scale_raw = self.output(fused).chunk(2, dim=1)
        mean = torch.tanh(mean_raw) * self.max_mean
        log_scale = torch.tanh(log_scale_raw) * self.max_log_scale
        return (mean, log_scale)

    def analysis(
        self,
        feature: torch.Tensor,
        context: torch.Tensor,
        qp: int,
        frame_type_idx: int,
        mode: str,
        reference_latent: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        mean, log_scale = self.parameters_from_reference(context, qp, frame_type_idx, mode)
        self._analysis_parameter_cache = ((id(context), int(qp), int(frame_type_idx), str(mode)), mean, log_scale)
        return (feature - mean) * torch.exp(-log_scale)

    def synthesis(
        self,
        feature: torch.Tensor,
        context: torch.Tensor,
        qp: int,
        frame_type_idx: int,
        mode: str,
        reference_latent: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        key = (id(context), int(qp), int(frame_type_idx), str(mode))
        cached = self._analysis_parameter_cache
        self._analysis_parameter_cache = None
        if cached is not None and cached[0] == key:
            mean, log_scale = (cached[1], cached[2])
        else:
            mean, log_scale = self.parameters_from_reference(context, qp, frame_type_idx, mode)
        return feature * torch.exp(log_scale) + mean


class PContextFirstBlock(nn.Module):
    """Reference-conditioned residual block used by the Task498 P core.

    The reference is not appended once and then forgotten: every block sees
    the current innovation, the decoded-reference coordinate, and their
    difference.  QP/P-role conditioning is supplied as a spatially broadcast
    bias by the owning core.
    """

    def __init__(self, channels: int) -> None:
        super().__init__()
        self.fuse = nn.Conv2d(3 * channels, channels, 1)
        self.spatial = nn.Conv2d(channels, channels, 3, padding=1, groups=channels)
        self.ffn = nn.Sequential(nn.Conv2d(channels, 2 * channels, 1), nn.SiLU(), nn.Conv2d(2 * channels, channels, 1))

    def forward(self, feature: torch.Tensor, reference: torch.Tensor, condition: torch.Tensor) -> torch.Tensor:
        update = self.fuse(torch.cat((feature, reference, feature - reference), dim=1))
        update = update + condition
        update = update + self.spatial(F.silu(update))
        return feature + self.ffn(F.silu(update))


class PContextFirstResidualCore(nn.Module):
    """Full replacement P analysis/synthesis transform for Wan latents.

    This core codes an innovation coordinate relative to the immediate
    hard-decoded reference.  It replaces, rather than augments, the inherited
    DCVC-RT encoder/decoder transform while retaining the proven one-y/one-z
    entropy syntax.  Analysis and synthesis have separate capacity but share
    the reference projector and QP/P-role coordinate.
    """

    def __init__(
        self,
        *,
        latent_channels: int = _VAE_CH,
        channels: int = _RT_D,
        y_channels: int = _RT_Y,
        blocks: int = 4,
        qp_count: int = _INTER_QP_NUM,
        position_count: int = 3,
    ) -> None:
        super().__init__()
        if channels <= 0 or y_channels <= 0 or blocks <= 0:
            raise ValueError("context-first core widths/blocks must be positive")
        self.latent_channels = int(latent_channels)
        self.channels = int(channels)
        self.y_channels = int(y_channels)
        self.blocks_count = int(blocks)
        self.position_count = max(1, int(position_count))
        self.current_stem = nn.Conv2d(self.latent_channels, self.channels, 3, padding=1)
        self.reference_stem = nn.Sequential(
            nn.Conv2d(_RT_D, self.channels, 3, padding=1),
            nn.SiLU(),
            nn.Conv2d(self.channels, self.channels, 3, padding=1, groups=self.channels),
            nn.Conv2d(self.channels, self.channels, 1),
        )
        self.qp_embedding = nn.Embedding(max(1, int(qp_count)), self.channels)
        self.position_embedding = nn.Embedding(self.position_count, self.channels)
        self.analysis_blocks = nn.ModuleList((PContextFirstBlock(self.channels) for _ in range(self.blocks_count)))
        self.synthesis_blocks = nn.ModuleList((PContextFirstBlock(self.channels) for _ in range(self.blocks_count)))
        self.down = nn.Conv2d(self.channels, self.y_channels, 3, stride=2, padding=1)
        self.up = nn.Sequential(nn.Conv2d(self.y_channels, 4 * self.channels, 3, padding=1), nn.PixelShuffle(2))

    def contract_payload(self) -> dict[str, Any]:
        return {
            "schema": "p_context_first_residual_core_v1",
            "latent_channels": self.latent_channels,
            "channels": self.channels,
            "y_channels": self.y_channels,
            "blocks": self.blocks_count,
            "position_count": self.position_count,
            "reference": "immediate_hard_decoded_P_or_I",
            "entropy_payload": "one_y_one_z_unchanged",
        }

    @staticmethod
    def _reference(context: torch.Tensor, mode: str) -> torch.Tensor:
        mode = str(mode).lower().strip()
        if mode == "correct":
            return context
        if mode == "zero":
            return torch.zeros_like(context)
        if mode == "wrong":
            return torch.roll(
                context, shifts=(max(1, context.shape[-2] // 2), max(1, context.shape[-1] // 2)), dims=(-2, -1)
            )
        raise ValueError(f"unknown context-first reference mode {mode!r}")

    def _coordinate(
        self, context: torch.Tensor, qp: int, frame_type_idx: int, mode: str
    ) -> tuple[torch.Tensor, torch.Tensor]:
        context = self._reference(context, mode)
        reference = self.reference_stem(context)
        qp_index = max(0, min(int(qp), self.qp_embedding.num_embeddings - 1))
        position_index = max(0, min(int(frame_type_idx), self.position_count - 1))
        condition = (self.qp_embedding.weight[qp_index] + self.position_embedding.weight[position_index]).to(
            device=reference.device, dtype=reference.dtype
        )[None, :, None, None]
        return (reference, condition)

    def analysis(
        self,
        x: torch.Tensor,
        context: torch.Tensor,
        quant_step: torch.Tensor,
        qp: int,
        frame_type_idx: int,
        mode: str,
        reference_latent: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        reference, condition = self._coordinate(context, qp, frame_type_idx, mode)
        feature = self.current_stem(x) - reference
        for block in self.analysis_blocks:
            feature = block(feature, reference, condition)
        return self.down(feature * quant_step)

    def synthesis(
        self,
        y_hat: torch.Tensor,
        context: torch.Tensor,
        quant_step: torch.Tensor,
        qp: int,
        frame_type_idx: int,
        mode: str,
        reference_latent: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        reference, condition = self._coordinate(context, qp, frame_type_idx, mode)
        feature = self.up(y_hat)
        for block in self.synthesis_blocks:
            feature = block(feature, reference, condition)
        return (feature + reference) * quant_step


class PContextFirstSharedLatentCore(PContextFirstResidualCore):
    """Task498-v2 core with a shared current/reference latent coordinate.

    The v1 core subtracted features produced by unrelated stems.  Here the
    current latent and immediate hard reference pass through exactly the same
    stem.  The richer temporal context may only provide a zero-initialized
    correction to that shared coordinate.
    """

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        del self.reference_stem
        self.context_correction = nn.Sequential(
            nn.Conv2d(_RT_D, self.channels, 3, padding=1), nn.SiLU(), nn.Conv2d(self.channels, self.channels, 1)
        )
        nn.init.zeros_(self.context_correction[-1].weight)
        nn.init.zeros_(self.context_correction[-1].bias)
        nn.init.zeros_(self.qp_embedding.weight)
        nn.init.zeros_(self.position_embedding.weight)

    def contract_payload(self) -> dict[str, Any]:
        payload = super().contract_payload()
        payload.update(
            {
                "schema": "p_context_first_shared_latent_core_v2",
                "coordinate": "shared_current_reference_latent_stem",
                "context_correction_init": "zero",
                "condition_init": "zero",
            }
        )
        return payload

    def _coordinate_v2(
        self, context: torch.Tensor, reference_latent: Optional[torch.Tensor], qp: int, frame_type_idx: int, mode: str
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if reference_latent is None:
            raise ValueError("shared-latent context-first core requires the immediate hard decoded reference latent")
        mode = str(mode).lower().strip()
        if mode == "correct":
            ref_latent = reference_latent
            ref_context = context
        elif mode == "zero":
            ref_latent = torch.zeros_like(reference_latent)
            ref_context = torch.zeros_like(context)
        elif mode == "wrong":
            shifts = (max(1, reference_latent.shape[-2] // 2), max(1, reference_latent.shape[-1] // 2))
            ref_latent = torch.roll(reference_latent, shifts=shifts, dims=(-2, -1))
            ref_context = torch.roll(context, shifts=shifts, dims=(-2, -1))
        else:
            raise ValueError(f"unknown context-first reference mode {mode!r}")
        reference = self.current_stem(ref_latent) + self.context_correction(ref_context)
        qp_index = max(0, min(int(qp), self.qp_embedding.num_embeddings - 1))
        position_index = max(0, min(int(frame_type_idx), self.position_count - 1))
        condition = (self.qp_embedding.weight[qp_index] + self.position_embedding.weight[position_index]).to(
            device=reference.device, dtype=reference.dtype
        )[None, :, None, None]
        return (reference, condition)

    def analysis(
        self,
        x: torch.Tensor,
        context: torch.Tensor,
        quant_step: torch.Tensor,
        qp: int,
        frame_type_idx: int,
        mode: str,
        reference_latent: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        reference, condition = self._coordinate_v2(context, reference_latent, qp, frame_type_idx, mode)
        feature = self.current_stem(x) - reference
        for block in self.analysis_blocks:
            feature = block(feature, reference, condition)
        return self.down(feature * quant_step)

    def synthesis(
        self,
        y_hat: torch.Tensor,
        context: torch.Tensor,
        quant_step: torch.Tensor,
        qp: int,
        frame_type_idx: int,
        mode: str,
        reference_latent: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        reference, condition = self._coordinate_v2(context, reference_latent, qp, frame_type_idx, mode)
        feature = self.up(y_hat)
        for block in self.synthesis_blocks:
            feature = block(feature, reference, condition)
        return (feature + reference) * quant_step


class PInnovationFusionBlock(nn.Module):
    """Keep innovation as the main state while conditioning on reconstruction."""

    def __init__(self, channels: int) -> None:
        super().__init__()
        groups = min(16, int(channels))
        while int(channels) % groups != 0:
            groups -= 1
        self.feature_norm = nn.GroupNorm(groups, int(channels))
        self.reference_norm = nn.GroupNorm(groups, int(channels))
        self.fuse = nn.Conv2d(4 * int(channels), int(channels), 1)
        self.spatial = nn.Conv2d(int(channels), int(channels), 3, padding=1, groups=int(channels))
        self.condition = nn.Conv2d(int(channels), 2 * int(channels), 1)
        self.ffn = nn.Sequential(
            nn.Conv2d(int(channels), 2 * int(channels), 1), nn.SiLU(), nn.Conv2d(2 * int(channels), int(channels), 1)
        )

    def forward(self, innovation: torch.Tensor, reference: torch.Tensor, condition: torch.Tensor) -> torch.Tensor:
        feature = self.feature_norm(innovation)
        ref = self.reference_norm(reference)
        fused = self.fuse(torch.cat((feature, ref, feature + ref, feature.abs()), dim=1))
        scale, bias = self.condition(condition).chunk(2, dim=1)
        fused = fused * (1.0 + 0.25 * torch.tanh(scale)) + bias
        fused = fused + self.spatial(F.silu(fused))
        return innovation + self.ffn(F.silu(fused))


class PContextFirstInnovationCoreV3(PContextFirstSharedLatentCore):
    """Single-coordinate current/reference innovation transform.

    V2 accidentally fed ``innovation-reference`` to every interaction block,
    which repeatedly forms a current-minus-two-reference coordinate.  V3 keeps
    innovation as the only evolving state.  The immediate hard reference is
    injected as the shared stem coordinate, as blockwise reconstruction
    context (innovation+reference), and as the final synthesis skip.
    """

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.current_stem = nn.Conv2d(self.latent_channels, self.channels, 3, padding=1, bias=False)
        self.analysis_blocks = nn.ModuleList((PInnovationFusionBlock(self.channels) for _ in range(self.blocks_count)))
        self.synthesis_blocks = nn.ModuleList((PInnovationFusionBlock(self.channels) for _ in range(self.blocks_count)))
        nn.init.zeros_(self.qp_embedding.weight)
        nn.init.zeros_(self.position_embedding.weight)

    def contract_payload(self) -> dict[str, Any]:
        payload = super().contract_payload()
        payload.update(
            {
                "schema": "p_context_first_innovation_core_v3",
                "coordinate": "shared_stem_innovation",
                "block_inputs": ["innovation", "reference", "innovation_plus_reference", "innovation_abs"],
                "reference_injection": ["input", "every_block", "output_skip"],
                "current_minus_two_reference_removed": True,
            }
        )
        return payload


class PContextFirstLatentResidualCoreV4(PContextFirstInnovationCoreV3):
    """Code an exact latent innovation and close the reference loop explicitly.

    Earlier context-first cores reconstructed an *absolute* latent through the
    random 256/320-channel synthesis path.  Even when every transmitted symbol
    was zero, the decoder therefore had to relearn the identity mapping from
    the reference feature to the Wan latent.  V4 keeps the transmitted signal
    and the prediction structurally separate:

    ``analysis_input = current_latent - hard_decoded_reference``
    ``output = hard_decoded_reference + decoded_latent_residual``

    The reference remains available to every interaction block as conditioning,
    but it is never folded into the coded innovation coordinate.  The final
    residual projection is zero initialized, giving an exact reference-copy
    reconstruction at step zero without an optional display-only branch.
    """

    outputs_latent_residual = True

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        groups = min(16, _RT_RECON)
        while _RT_RECON % groups != 0:
            groups -= 1
        hidden = max(self.latent_channels * 4, self.channels)
        self.latent_residual_head = nn.Sequential(
            nn.GroupNorm(groups, _RT_RECON),
            nn.Conv2d(_RT_RECON, hidden, 1),
            nn.SiLU(),
            nn.Conv2d(hidden, hidden, 3, padding=1, groups=hidden),
            nn.SiLU(),
            nn.Conv2d(hidden, self.latent_channels, 1),
        )
        nn.init.zeros_(self.latent_residual_head[-1].weight)
        nn.init.zeros_(self.latent_residual_head[-1].bias)
        anchors = ((0, 3.0), (16, 2.0), (39, 0.5), (63, 0.125))
        anchor_logs = [0.25 * math.log(value / 2.0) for (_, value) in anchors]
        ladder = torch.empty(_INTER_QP_NUM, dtype=torch.float32)
        for left in range(len(anchors) - 1):
            q0, _ = anchors[left]
            q1, _ = anchors[left + 1]
            for q in range(q0, q1 + 1):
                alpha = float(q - q0) / float(q1 - q0)
                ladder[q] = math.exp((1.0 - alpha) * anchor_logs[left] + alpha * anchor_logs[left + 1])
        self.register_buffer("qp_analysis_base_gain", ladder, persistent=True)

    def contract_payload(self) -> dict[str, Any]:
        payload = super().contract_payload()
        payload.update(
            {
                "schema": "p_context_first_latent_residual_core_v4",
                "coded_coordinate": "current_minus_immediate_hard_latent",
                "synthesis_coordinate": "hard_latent_plus_decoded_residual",
                "zero_symbol_reconstruction": "exact_reference_copy_at_initialization",
                "display_only_branch": False,
                "qp_base_ladder": "lambda_quarter_power_log_interpolation",
            }
        )
        return payload

    @staticmethod
    def _latent_reference(reference_latent: torch.Tensor, mode: str) -> torch.Tensor:
        mode = str(mode).lower().strip()
        if mode == "correct":
            return reference_latent
        if mode == "zero":
            return torch.zeros_like(reference_latent)
        if mode == "wrong":
            return torch.roll(
                reference_latent,
                shifts=(max(1, reference_latent.shape[-2] // 2), max(1, reference_latent.shape[-1] // 2)),
                dims=(-2, -1),
            )
        raise ValueError(f"unknown context-first reference mode {mode!r}")

    def analysis(
        self,
        x: torch.Tensor,
        context: torch.Tensor,
        quant_step: torch.Tensor,
        qp: int,
        frame_type_idx: int,
        mode: str,
        reference_latent: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if reference_latent is None:
            raise ValueError("latent-residual core requires a hard reference")
        latent_reference = self._latent_reference(reference_latent, mode)
        reference, condition = self._coordinate_v2(context, reference_latent, qp, frame_type_idx, mode)
        innovation = self.current_stem(x - latent_reference)
        for block in self.analysis_blocks:
            innovation = block(innovation, reference, condition)
        qp_index = max(0, min(int(qp), self.qp_analysis_base_gain.numel() - 1))
        base_gain = self.qp_analysis_base_gain[qp_index].to(device=innovation.device, dtype=innovation.dtype)
        return self.down(innovation * quant_step * base_gain)

    def synthesis(
        self,
        y_hat: torch.Tensor,
        context: torch.Tensor,
        quant_step: torch.Tensor,
        qp: int,
        frame_type_idx: int,
        mode: str,
        reference_latent: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        reference, condition = self._coordinate_v2(context, reference_latent, qp, frame_type_idx, mode)
        innovation = self.up(y_hat)
        for block in self.synthesis_blocks:
            innovation = block(innovation, reference, condition)
        return innovation * quant_step

    def reconstruct_latent(
        self, reconstruction_feature: torch.Tensor, reference_latent: Optional[torch.Tensor], mode: str
    ) -> torch.Tensor:
        if reference_latent is None:
            raise ValueError("latent-residual reconstruction requires a hard reference")
        reference = self._latent_reference(reference_latent, mode)
        return reference + self.latent_residual_head(reconstruction_feature)


class PMatureContextCouplingBlock(nn.Module):
    """One decoder-closed interaction block for the mature Task297 transform."""

    def __init__(self, channels: int) -> None:
        super().__init__()
        channels = int(channels)
        self.fuse = nn.Conv2d(4 * channels, channels, 1)
        self.spatial = nn.Conv2d(channels, channels, 3, padding=1, groups=channels)
        self.ffn = nn.Sequential(nn.Conv2d(channels, 2 * channels, 1), nn.SiLU(), nn.Conv2d(2 * channels, channels, 1))

    def forward(
        self, feature: torch.Tensor, reference: torch.Tensor, context: torch.Tensor, condition: torch.Tensor
    ) -> torch.Tensor:
        update = self.fuse(torch.cat((feature, reference, feature - reference, context), dim=1))
        update = update + condition
        update = update + self.spatial(F.silu(update))
        return feature + self.ffn(F.silu(update))


class PMatureContextFirstCoupling(nn.Module):
    """Identity-safe context-first coupling inside the mature P transform.

    The inherited Task297 analysis, synthesis, quantization, hyperprior and
    four-pass spatial prior remain intact.  This module only changes the
    features entering ``encoder.conv2`` and ``decoder.conv1``.  Both halves
    receive a reference feature made by the *same parent encoder.conv1* from
    the immediate hard-decoded latent, plus the existing decoder-known DMC
    context.  Zero output projections make enablement exactly equal to the
    mature parent before training.
    """

    def __init__(
        self,
        *,
        channels: int = _RT_D,
        blocks: int = 4,
        qp_count: int = _INTER_QP_NUM,
        position_count: int = 3,
        residual_scale: float = 0.5,
    ) -> None:
        super().__init__()
        channels = int(channels)
        blocks = int(blocks)
        if channels <= 0 or blocks <= 0 or residual_scale <= 0:
            raise ValueError("mature context coupling widths/blocks/scale must be positive")
        self.channels = channels
        self.blocks_count = blocks
        self.position_count = max(1, int(position_count))
        self.residual_scale = float(residual_scale)
        self.qp_embedding = nn.Embedding(max(1, int(qp_count)), channels)
        self.position_embedding = nn.Embedding(self.position_count, channels)
        nn.init.zeros_(self.qp_embedding.weight)
        nn.init.zeros_(self.position_embedding.weight)
        self.analysis_blocks = nn.ModuleList((PMatureContextCouplingBlock(channels) for _ in range(blocks)))
        self.synthesis_blocks = nn.ModuleList((PMatureContextCouplingBlock(channels) for _ in range(blocks)))
        self.analysis_output = nn.Conv2d(channels, channels, 3, padding=1)
        self.synthesis_output = nn.Conv2d(channels, channels, 3, padding=1)
        nn.init.zeros_(self.analysis_output.weight)
        nn.init.zeros_(self.analysis_output.bias)
        nn.init.zeros_(self.synthesis_output.weight)
        nn.init.zeros_(self.synthesis_output.bias)

    def contract_payload(self) -> dict[str, Any]:
        return {
            "schema": "p_mature_context_first_coupling_v1",
            "channels": self.channels,
            "blocks": self.blocks_count,
            "position_count": self.position_count,
            "residual_scale": self.residual_scale,
            "reference": "immediate_hard_latent_shared_parent_encoder_conv1",
            "analysis_insertion": "before_parent_encoder_conv2",
            "synthesis_insertion": "after_parent_decoder_up_before_conv1",
            "entropy_payload": "parent_one_y_one_z_unchanged",
            "initial_state": "exact_parent_zero_outputs",
        }

    def _condition(self, feature: torch.Tensor, qp: int, frame_type_idx: int) -> torch.Tensor:
        qp_index = max(0, min(int(qp), self.qp_embedding.num_embeddings - 1))
        position_index = max(0, min(int(frame_type_idx), self.position_count - 1))
        return (self.qp_embedding.weight[qp_index] + self.position_embedding.weight[position_index]).to(
            device=feature.device, dtype=feature.dtype
        )[None, :, None, None]

    def _forward_half(
        self,
        feature: torch.Tensor,
        reference: torch.Tensor,
        context: torch.Tensor,
        qp: int,
        frame_type_idx: int,
        blocks: nn.ModuleList,
        output: nn.Conv2d,
    ) -> torch.Tensor:
        if feature.shape != reference.shape or feature.shape != context.shape:
            raise ValueError(
                f"mature context coupling requires equal feature/reference/context shapes, got {tuple(feature.shape)}, {tuple(reference.shape)}, {tuple(context.shape)}"
            )
        hidden = feature
        condition = self._condition(feature, qp, frame_type_idx)
        for block in blocks:
            hidden = block(hidden, reference, context, condition)
        delta = torch.tanh(output(F.silu(hidden)))
        return feature + self.residual_scale * delta

    def analysis(
        self, feature: torch.Tensor, reference: torch.Tensor, context: torch.Tensor, qp: int, frame_type_idx: int
    ) -> torch.Tensor:
        return self._forward_half(
            feature, reference, context, qp, frame_type_idx, self.analysis_blocks, self.analysis_output
        )

    def synthesis(
        self, feature: torch.Tensor, reference: torch.Tensor, context: torch.Tensor, qp: int, frame_type_idx: int
    ) -> torch.Tensor:
        return self._forward_half(
            feature, reference, context, qp, frame_type_idx, self.synthesis_blocks, self.synthesis_output
        )


class PContextNativeStage(nn.Module):
    """One identity-initialized stage of the context-native P transform."""

    def __init__(
        self,
        *,
        feature_channels: int,
        hidden_channels: int,
        blocks: int,
        qp_count: int,
        position_count: int,
        residual_scale: float,
    ) -> None:
        super().__init__()
        feature_channels = int(feature_channels)
        hidden_channels = int(hidden_channels)
        blocks = int(blocks)
        if min(feature_channels, hidden_channels, blocks) <= 0:
            raise ValueError("context-native stage dimensions must be positive")
        self.feature_channels = feature_channels
        self.hidden_channels = hidden_channels
        self.residual_scale = float(residual_scale)
        self.feature_proj = nn.Conv2d(feature_channels, hidden_channels, 1)
        self.reference_proj = nn.Conv2d(_RT_D, hidden_channels, 1)
        self.context_proj = nn.Conv2d(_RT_D, hidden_channels, 1)
        self.difference_proj = nn.Conv2d(feature_channels, hidden_channels, 1)
        self.reference_to_feature = (
            nn.Identity() if feature_channels == _RT_D else nn.Conv2d(_RT_D, feature_channels, 1)
        )
        self.qp_embedding = nn.Embedding(int(qp_count), hidden_channels)
        self.position_embedding = nn.Embedding(max(1, int(position_count)), hidden_channels)
        nn.init.zeros_(self.qp_embedding.weight)
        nn.init.zeros_(self.position_embedding.weight)
        self.blocks = nn.ModuleList()
        for _ in range(blocks):
            self.blocks.append(
                nn.Sequential(
                    nn.GroupNorm(max(1, math.gcd(hidden_channels, 16)), hidden_channels),
                    nn.SiLU(),
                    nn.Conv2d(hidden_channels, hidden_channels, 3, padding=1, groups=hidden_channels),
                    nn.Conv2d(hidden_channels, 2 * hidden_channels, 1),
                    nn.SiLU(),
                    nn.Conv2d(2 * hidden_channels, hidden_channels, 1),
                )
            )
        self.output = nn.Conv2d(hidden_channels, feature_channels, 1)
        nn.init.zeros_(self.output.weight)
        nn.init.zeros_(self.output.bias)

    def forward(
        self, feature: torch.Tensor, reference: torch.Tensor, context: torch.Tensor, *, qp: int, position: int
    ) -> torch.Tensor:
        if reference.shape[-2:] != feature.shape[-2:]:
            reference = F.interpolate(reference, size=feature.shape[-2:], mode="bilinear", align_corners=False)
        if context.shape[-2:] != feature.shape[-2:]:
            context = F.interpolate(context, size=feature.shape[-2:], mode="bilinear", align_corners=False)
        reference_feature = self.reference_to_feature(reference)
        hidden = (
            self.feature_proj(feature)
            + self.reference_proj(reference)
            + self.context_proj(context)
            + self.difference_proj(feature - reference_feature)
        )
        qp_index = max(0, min(int(qp), self.qp_embedding.num_embeddings - 1))
        position_index = max(0, min(int(position), self.position_embedding.num_embeddings - 1))
        condition = (self.qp_embedding.weight[qp_index] + self.position_embedding.weight[position_index]).to(
            device=hidden.device, dtype=hidden.dtype
        )[None, :, None, None]
        hidden = hidden + condition
        for block in self.blocks:
            hidden = hidden + block(hidden)
        return feature + self.residual_scale * self.output(F.silu(hidden))


class PContextNativeDeepTransform(nn.Module):
    """Inject one hard-decoded reference throughout the mature P backbone.

    This is a single-path transform: no alternate reconstruction, flow payload,
    cache payload, or optional stream is created.  Seven zero-output stages
    make construction exactly equal to Task297 and then add capacity at every
    important analysis/synthesis depth.
    """

    STAGE_CHANNELS = {
        "analysis_pre": _RT_D,
        "analysis_mid": _RT_D,
        "analysis_tail": _RT_D,
        "synthesis_pre": _RT_D,
        "synthesis_mid": _RT_D,
        "synthesis_tail": _RT_D,
        "reconstruction": _RT_RECON,
    }

    def __init__(
        self,
        *,
        hidden_channels: int = 128,
        blocks: int = 2,
        qp_count: int = _INTER_QP_NUM,
        position_count: int = 3,
        residual_scale: float = 0.25,
    ) -> None:
        super().__init__()
        if hidden_channels <= 0 or blocks <= 0 or residual_scale <= 0:
            raise ValueError("invalid context-native deep-transform settings")
        self.hidden_channels = int(hidden_channels)
        self.blocks_count = int(blocks)
        self.position_count = max(1, int(position_count))
        self.residual_scale = float(residual_scale)
        self.stages = nn.ModuleDict(
            {
                name: PContextNativeStage(
                    feature_channels=channels,
                    hidden_channels=self.hidden_channels,
                    blocks=self.blocks_count,
                    qp_count=int(qp_count),
                    position_count=self.position_count,
                    residual_scale=self.residual_scale,
                )
                for (name, channels) in self.STAGE_CHANNELS.items()
            }
        )

    def contract_payload(self) -> dict[str, Any]:
        return {
            "schema": "p_context_native_deep_transform_v1",
            "hidden_channels": self.hidden_channels,
            "blocks_per_stage": self.blocks_count,
            "position_count": self.position_count,
            "residual_scale": self.residual_scale,
            "stages": list(self.STAGE_CHANNELS),
            "reference": "immediate_hard_decoded_latent_via_parent_stem",
            "payload": "parent_one_y_one_z_only",
            "initial_state": "exact_Task297_zero_outputs",
        }

    def forward(
        self,
        stage: str,
        feature: torch.Tensor,
        reference: torch.Tensor,
        context: torch.Tensor,
        *,
        qp: int,
        position: int,
    ) -> torch.Tensor:
        if stage not in self.stages:
            raise KeyError(f"unknown context-native stage {stage!r}")
        return self.stages[stage](feature, reference, context, qp=qp, position=position)


class PLocalReferenceTransform(nn.Module):
    """Decoder-closed 5x5 local-reference main-fusion transform.

    The module receives a current/decoded-y feature and the decoder-known
    reference context ``ctx``.  It projects a low-dimensional query from the
    feature and key/value pairs from ``ctx``, computes a fixed-window local
    attention over a 5x5 neighborhood, and fuses the current feature, the
    ordinary context and the aligned reference value into a single coordinate.
    The output projection and QP/position gates are zero-initialized, so the
    module is exactly an identity when first enabled and preserves any parent
    checkpoint bit-exactly at step zero.
    """

    def __init__(
        self,
        feature_channels: int,
        context_channels: int,
        query_key_channels: int = 64,
        value_channels: int = 128,
        num_heads: int = 4,
        window: int = 5,
        out_channels: Optional[int] = None,
        residual_target: str = "context",
        qp_count: int = _INTER_QP_NUM,
        position_count: int = 3,
        residual_scale: float = 0.25,
        qp_gate_max_log_gain: float = 0.75,
        qp_gate_max_bias: float = 0.25,
    ) -> None:
        super().__init__()
        if window % 2 == 0:
            raise ValueError(f"local-reference window must be odd, got {window}")
        if query_key_channels % num_heads != 0:
            raise ValueError(f"query/key channels {query_key_channels} not divisible by {num_heads} heads")
        if value_channels % num_heads != 0:
            raise ValueError(f"value channels {value_channels} not divisible by {num_heads} heads")
        self.feature_channels = int(feature_channels)
        self.context_channels = int(context_channels)
        self.query_key_channels = int(query_key_channels)
        self.value_channels = int(value_channels)
        self.num_heads = int(num_heads)
        self.head_dim_qk = self.query_key_channels // self.num_heads
        self.head_dim_v = self.value_channels // self.num_heads
        self.window = int(window)
        self.padding = self.window // 2
        self.out_channels = int(out_channels or context_channels)
        self.residual_target = str(residual_target).lower().strip()
        if self.residual_target not in {"context", "feature"}:
            raise ValueError(f"residual_target must be 'context' or 'feature', got {residual_target!r}")
        self.residual_scale = float(residual_scale)
        self.qp_gate_values = (0, 16, 39, 63)
        self.qp_gate_max_log_gain = float(qp_gate_max_log_gain)
        self.qp_gate_max_bias = float(qp_gate_max_bias)
        if self.qp_gate_max_log_gain <= 0 or self.qp_gate_max_bias <= 0:
            raise ValueError("local-reference QP gate bounds must be positive")
        self.query_proj = nn.Conv2d(self.feature_channels, self.query_key_channels, 1)
        self.key_proj = nn.Conv2d(self.context_channels, self.query_key_channels, 1)
        self.value_proj = nn.Conv2d(self.context_channels, self.value_channels, 1)
        self.fuse = nn.Conv2d(self.feature_channels + self.context_channels + self.value_channels, self.out_channels, 1)
        nn.init.zeros_(self.fuse.weight)
        nn.init.zeros_(self.fuse.bias)
        self.qp_embedding = nn.Embedding(max(1, int(qp_count)), self.out_channels)
        self.position_embedding = nn.Embedding(max(1, int(position_count)), self.out_channels)
        nn.init.zeros_(self.qp_embedding.weight)
        nn.init.zeros_(self.position_embedding.weight)
        self.qp_output_log_gain = nn.Parameter(torch.zeros(len(self.qp_gate_values), self.out_channels, 1, 1))
        self.qp_output_bias = nn.Parameter(torch.zeros(len(self.qp_gate_values), self.out_channels, 1, 1))

    def contract_payload(self) -> dict[str, Any]:
        return {
            "schema": "p_local_reference_transform_v1",
            "feature_channels": self.feature_channels,
            "context_channels": self.context_channels,
            "query_key_channels": self.query_key_channels,
            "value_channels": self.value_channels,
            "num_heads": self.num_heads,
            "window": self.window,
            "out_channels": self.out_channels,
            "residual_target": self.residual_target,
            "residual_scale": self.residual_scale,
            "qp_gate_values": list(self.qp_gate_values),
            "qp_gate_max_log_gain": self.qp_gate_max_log_gain,
            "qp_gate_max_bias": self.qp_gate_max_bias,
        }

    def _qp_gate_index(self, qp: int) -> int:
        value = int(qp)
        return min(range(len(self.qp_gate_values)), key=lambda index: abs(value - self.qp_gate_values[index]))

    def forward(
        self,
        feature: torch.Tensor,
        context: torch.Tensor,
        qp: int,
        frame_type_idx: int,
        reference_mode: str = "correct",
    ) -> torch.Tensor:
        if feature.shape[0] != context.shape[0] or feature.shape[-2:] != context.shape[-2:]:
            raise ValueError(
                f"local-reference transform expects matching batch and spatial shapes, got {tuple(feature.shape)} and {tuple(context.shape)}"
            )
        match_context = context
        if reference_mode == "zero":
            match_context = torch.zeros_like(context)
        elif reference_mode == "wrong":
            match_context = torch.roll(
                context, shifts=(max(1, context.shape[-2] // 2), max(1, context.shape[-1] // 2)), dims=(-2, -1)
            )
        elif reference_mode != "correct":
            raise ValueError(f"reference_mode must be 'correct', 'zero', or 'wrong', got {reference_mode!r}")
        B, _, H, W = feature.shape
        q = self.query_proj(feature)
        k = self.key_proj(match_context)
        v = self.value_proj(match_context)
        q = q.reshape(B, self.num_heads, self.head_dim_qk, H, W)
        k_unf = F.unfold(k, kernel_size=self.window, padding=self.padding, stride=1)
        v_unf = F.unfold(v, kernel_size=self.window, padding=self.padding, stride=1)
        win2 = self.window * self.window
        k_unf = k_unf.reshape(B, self.num_heads, self.head_dim_qk, win2, H * W)
        v_unf = v_unf.reshape(B, self.num_heads, self.head_dim_v, win2, H * W)
        q_flat = q.reshape(B, self.num_heads, self.head_dim_qk, H * W)
        scores = (q_flat[:, :, :, None, :] * k_unf).sum(dim=2) / self.head_dim_qk**0.5
        attn = torch.softmax(scores, dim=3)
        aligned = (attn[:, :, None, :, :] * v_unf).sum(dim=3)
        aligned = aligned.reshape(B, self.value_channels, H, W)
        fused = torch.cat((feature, match_context, aligned), dim=1)
        delta = torch.tanh(self.fuse(fused))
        qp_index = max(0, min(int(qp), self.qp_embedding.num_embeddings - 1))
        position_index = max(0, min(int(frame_type_idx), self.position_embedding.num_embeddings - 1))
        condition = (
            torch.tanh(self.qp_embedding.weight[qp_index] + self.position_embedding.weight[position_index])
            * self.qp_gate_max_bias
        )
        condition = condition.to(device=delta.device, dtype=delta.dtype)[None, :, None, None]
        gate_index = self._qp_gate_index(qp)
        gain = torch.exp(torch.tanh(self.qp_output_log_gain[gate_index]) * self.qp_gate_max_log_gain).to(
            device=delta.device, dtype=delta.dtype
        )
        bias = (torch.tanh(self.qp_output_bias[gate_index]) * self.qp_gate_max_bias).to(
            device=delta.device, dtype=delta.dtype
        )
        delta = delta * gain + bias + condition
        if self.residual_target == "context":
            return context + self.residual_scale * delta
        return feature + self.residual_scale * delta


class DMCIAdapter(nn.Module):
    """DMCI-based I-frame codec adapted for Wan VAE latents (48ch Wan2.2 / 16ch Wan2.1)."""

    def __init__(self, checkpoint_path: Optional[str] = None, device: str = "cuda:0"):
        latent_init_mode = "scratch"
        latent_stem_blocks = 4
        latent_spatial_mode = "dcvc_stride2"
        latent_preserve_blocks = 1
        latent_channels = 48
        bpp_spatial_factor = 32
        i_y_gain_enabled = False
        i_y_gain_max_log = 1.0
        i_y_balance_enabled = False
        i_y_balance_max_log = 0.5
        i_y_balance_max_bias = 0.5
        i_y_coupling_enabled = False
        i_y_coupling_hidden = 256
        i_y_coupling_blocks = 4
        i_y_coupling_max_log_scale = 0.35
        i_y_coupling_max_shift = 1.0
        i_adaptive_y_gain_enabled = False
        i_adaptive_y_gain_hidden = 128
        i_adaptive_y_gain_max_log = 0.12
        i_post_y_gain_enabled = False
        i_post_y_gain_max_log = 0.35
        i_synthesis_head_enabled = False
        i_synthesis_head_hidden = 192
        i_synthesis_head_scale = 0.1
        i_synthesis_head_arch = "multiscale_reference"
        i_synthesis_head_blocks = 4
        i_synthesis_head_out_of_loop = False
        i_entropy_q_mode = "dynamic"
        i_group_context_lane = "legacy"
        i_group_context_parameterization = "scale"
        i_group_context_hidden = 96
        i_group_context_max_log_scale = 0.75
        i_group_context_max_mean_delta = 2.0
        i_group_context_permutation = None
        i_transform_channels = 512
        i_y_channels = 320
        i_z_channels = 160
        i_analysis_core_hidden = 0
        i_synthesis_core_hidden = 0
        i_core_blocks = 2
        i_core_residual_scale = 1.0
        i_progressive_transform_enabled = False
        i_progressive_transform_stages = 6
        i_progressive_transform_max_angle = 0.35
        i_progressive_q_enabled = False
        i_progressive_q_max_group_log = 0.35
        i_progressive_q_max_channel_log = 0.2
        i_progressive_q_center_groups = False
        i_hierarchical_transform_enabled = False
        i_hierarchical_hidden = 256
        i_hierarchical_blocks = 3
        i_hierarchical_residual_scale = 0.25
        i_hierarchical_aux_enabled = False
        super().__init__()
        from src.models.image_model import DMCI

        self._device = device
        self.latent_channels = int(48)
        self.bpp_spatial_factor = int(32)
        self.latent_init_mode = _normalize_latent_init_mode("scratch")
        self.latent_spatial_mode = _normalize_latent_spatial_mode("dcvc_stride2")
        self.latent_preserve_blocks = max(1, int(1))
        self._entropy_coder_ready = False
        self._capture_exact_entropy_symbols = False
        self._last_entropy_symbol_tensors: dict[str, Any] = {}
        self._last_bpp_y = 0.0
        self._last_bpp_z = 0.0
        self._last_q_enc_mean = 0.0
        self._last_q_enc_floor_fraction = 0.0
        self._last_q_dec_mean = 0.0
        self.i_y_gain_enabled = bool(False)
        self.i_y_gain_max_log = float(1.0)
        self.i_y_balance_enabled = bool(False)
        self.i_y_balance_max_log = float(0.5)
        self.i_y_balance_max_bias = float(0.5)
        self.i_synthesis_head_enabled = bool(False)
        self.i_synthesis_head_out_of_loop = bool(False)
        self.i_synthesis_head_arch = str("multiscale_reference").strip().lower()
        self.i_y_coupling_enabled = bool(False)
        self.i_adaptive_y_gain_enabled = bool(False)
        self.i_post_y_gain_enabled = bool(False)
        self.i_entropy_q_mode = str("dynamic").strip().lower()
        self.i_group_context_lane = normalize_i_group_context_lane("legacy")
        self.i_transform_channels = int(512)
        self.i_y_channels = int(320)
        self.i_z_channels = int(160)
        self._native_i_channels = (self.i_transform_channels, self.i_y_channels, self.i_z_channels)
        self._default_native_i_channels = (_DMCI_ENC_DEC, _DMCI_N, _DMCI_Z)
        if self._native_i_channels != self._default_native_i_channels and self.latent_init_mode != "scratch":
            raise ValueError(
                "Non-default I transform/y/z widths require latent_init_mode='scratch'; pretrained Task297/DMCI tensors cannot be shape-adapted into a fair native-width comparison."
            )
        self.i_analysis_core_hidden = int(0)
        self.i_synthesis_core_hidden = int(0)
        self.i_core_blocks = int(2)
        self.i_core_residual_scale = float(1.0)
        self.i_progressive_transform_enabled = bool(False)
        self.i_progressive_q_enabled = bool(False)
        self.i_hierarchical_transform_enabled = bool(False)
        self.i_hierarchical_aux_enabled = bool(False)
        self.i_progressive_group_sizes = tuple((int(value) for value in DEFAULT_PROGRESSIVE_GROUP_SIZES))
        self.i_post_y_gain_max_log = float(0.35)
        self.model = DMCI(N=self.i_y_channels, z_channel=self.i_z_channels)
        from src.layers.layers import DepthConvBlock, ResidualBlockUpsample

        if self.i_transform_channels != _DMCI_ENC_DEC:
            self.model.enc.enc_2 = nn.Sequential(
                *[DepthConvBlock(self.i_transform_channels, self.i_transform_channels) for _ in range(6)],
                nn.Conv2d(self.i_transform_channels, self.i_y_channels, 3, stride=2, padding=1),
            )
            self.model.dec.dec_1 = nn.Sequential(
                ResidualBlockUpsample(self.i_y_channels, self.i_transform_channels),
                *[DepthConvBlock(self.i_transform_channels, self.i_transform_channels) for _ in range(12)],
            )
            self.model.dec.dec_2 = DepthConvBlock(self.i_transform_channels, _VAE_CH * 4)
            self.model.q_scale_enc = nn.Parameter(torch.ones((_NUM_QP, self.i_transform_channels, 1, 1)))
            self.model.q_scale_dec = nn.Parameter(torch.ones((_NUM_QP, self.i_transform_channels, 1, 1)))
        self.model.enc.enc_1 = LatentInputStem(self.latent_channels, self.i_transform_channels, num_blocks=int(4))
        sd: dict[str, torch.Tensor] = {}
        _initialize_neutral_prior_output_(
            self.model.y_prior_fusion[-1], q_channels=2, scale_channels=self.i_y_channels, q_bias=_image_q_to_logit(1.0)
        )
        _initialize_neutral_prior_output_(
            self.model.y_spatial_prior[-1], q_channels=0, scale_channels=self.i_y_channels, q_bias=0.0
        )
        self.model.to(device)
        self.model.eval()
        self.output_proj = nn.Conv2d(_VAE_CH * 4, self.latent_channels, 1).to(device)
        self.i_y_gain_logit = None
        self.i_y_balance_gain_logit = None
        self.i_y_balance_bias = None
        self.i_y_coupling = None
        self.i_adaptive_y_gain = None
        self.i_post_y_gain_logit = None
        self.i_channel_q_enc_log = None
        self.i_channel_q_dec_log = None
        self.i_group_context = None
        self.i_progressive_transform = None
        self.i_progressive_q = None
        self.i_hierarchical_transform = None
        self._last_i_progressive_aux: dict[str, torch.Tensor] = {}
        self._last_i_progressive_q_group_log_mean = 0.0
        self._last_i_progressive_q_group_log_spread = 0.0
        self._last_i_progressive_q_group_logs: tuple[float, ...] = ()
        self._last_i_progressive_q_channel_log_mean_abs = 0.0
        self.i_analysis_core = None
        self.i_synthesis_core = None
        self.i_synthesis_head = None
        self.qe_log_intra = nn.Parameter(torch.zeros(_NUM_QP, self.i_transform_channels, 1, 1, device=device))
        self.qd_log_intra = nn.Parameter(torch.zeros(_NUM_QP, self.i_transform_channels, 1, 1, device=device))
        with torch.no_grad():
            if "q_scale_enc" in sd and "q_scale_dec" in sd:
                qe = sd["q_scale_enc"].clamp(min=0.001).to(device)
                qd = sd["q_scale_dec"].clamp(min=0.001).to(device)
                self.qe_log_intra.data.copy_(torch.log(qe))
                self.qd_log_intra.data.copy_(torch.log(qd))
        try:
            self.model.update(force_zero_thres=None)
            self._entropy_coder_ready = True
        except ImportError:
            pass

    def _ensure_entropy_coder(self) -> None:
        if self._entropy_coder_ready:
            return
        try:
            self.model.update(force_zero_thres=None)
            self._entropy_coder_ready = True
        except ImportError as exc:
            raise RuntimeError(
                "The DCVC-RT C++ rANS extension (MLCodec_extensions_cpp) is not available. Training-only paths do not require it, but real entropy coding does."
            ) from exc

    def _load_from_state_dict(self, *args, **kwargs) -> None:
        """Invalidate learned z-CDFs whenever codec weights are restored."""
        super()._load_from_state_dict(*args, **kwargs)
        self._entropy_coder_ready = False

    def _get_quant_step(self, log_param: nn.Parameter, qp: int) -> torch.Tensor:
        return torch.exp(log_param[qp : qp + 1]) + 0.001

    def _y_gain(self, qp: int, like: torch.Tensor) -> torch.Tensor:
        return torch.ones((1, self.i_y_channels, 1, 1), device=like.device, dtype=like.dtype)

    def _y_analysis_coordinate(self, y_raw: torch.Tensor, qp: int) -> torch.Tensor:
        value = y_raw
        progressive_transform = None
        hierarchical_transform = None
        return value

    def _y_synthesis_coordinate(self, y: torch.Tensor, qp: int) -> torch.Tensor:
        value = y
        hierarchical_transform = None
        progressive_transform = None
        return value

    def _adaptive_entropy_gain(self, params: torch.Tensor, qp: int, like: torch.Tensor) -> torch.Tensor:
        return torch.ones_like(like)

    def _post_y_gain(self, qp: int, like: torch.Tensor) -> torch.Tensor:
        return torch.ones((1, self.i_y_channels, 1, 1), device=like.device, dtype=like.dtype)

    def _separate_i_entropy_prior(
        self, params: torch.Tensor, qp: int
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return q/scales/means under the selected I entropy-q contract."""
        return self.model.separate_prior(params, is_video=False)

    def _compress_channel_prior_4x(self, y: torch.Tensor, params: torch.Tensor, qp: int) -> tuple[torch.Tensor, ...]:
        """Hard rANS preparation for the QP/channel I quantization tables."""
        q_enc, q_dec, scales, means = self._separate_i_entropy_prior(params, qp)
        common_params = self.model.y_spatial_prior_reduction(params)
        B, C, H, W = y.shape
        masks = self.model.get_mask_4x(B, C, H, W, y.dtype, y.device)
        adaptors = (
            self.model.y_spatial_prior_adaptor_1,
            self.model.y_spatial_prior_adaptor_2,
            self.model.y_spatial_prior_adaptor_3,
        )
        y_scaled = y * q_enc
        y_hat_so_far = torch.zeros_like(y_scaled)
        y_q_writes: list[torch.Tensor] = []
        scale_writes: list[torch.Tensor] = []
        exported_scales: list[torch.Tensor] = []
        exported_widths = []
        for index, mask in enumerate(masks):
            if index > 0:
                context = torch.cat((y_hat_so_far, common_params), dim=1)
                scales, means = self.model.y_spatial_prior(adaptors[index - 1](context)).chunk(2, 1)
            width = _task938_effective_width(self, scales, means, qp)
            exported_widths.append(width)
            if getattr(self, "_task938_quantizer", None) is not None and self._task938_quantizer.enabled:
                y_q, y_hat, scales_hat = _task938_process_mask(self, y_scaled, scales, means, mask, qp, width=width)
            else:
                _, y_q, y_hat, scales_hat = self.model.process_with_mask(y_scaled, scales, means, mask)
            y_hat_so_far = y_hat_so_far + y_hat
            if getattr(self, "_export_entropy_scales", False):
                exported_scales.append((scales / width).detach())
            y_q_writes.append(self.model.single_part_for_writing_4x(y_q))
            scale_writes.append(self.model.single_part_for_writing_4x(scales_hat))
        _task938_publish_prior(self, exported_scales, exported_widths, masks)
        return (*y_q_writes, *scale_writes, y_hat_so_far * q_dec)

    def _decompress_channel_prior_4x(self, params: torch.Tensor, qp: int) -> torch.Tensor:
        """Decode the QP/channel I quantization-table bitstream."""
        _, q_dec, scales, means = self._separate_i_entropy_prior(params, qp)
        common_params = self.model.y_spatial_prior_reduction(params)
        B, C, H, W = means.shape
        masks = self.model.get_mask_4x(B, C, H, W, means.dtype, means.device)
        adaptors = (
            self.model.y_spatial_prior_adaptor_1,
            self.model.y_spatial_prior_adaptor_2,
            self.model.y_spatial_prior_adaptor_3,
        )
        y_hat_so_far = torch.zeros_like(means)
        exported_scales: list[torch.Tensor] = []
        exported_widths = []
        for index, mask in enumerate(masks):
            if index > 0:
                context = torch.cat((y_hat_so_far, common_params), dim=1)
                scales, means = self.model.y_spatial_prior(adaptors[index - 1](context)).chunk(2, 1)
            width = _task938_effective_width(self, scales, means, qp)
            exported_widths.append(width)
            if getattr(self, "_export_entropy_scales", False):
                exported_scales.append((scales / width).detach())
            scales_read = self.model.single_part_for_writing_4x(scales * mask / width)
            y_q_read = self.model.gaussian_encoder.decode_and_get_y(scales_read, means.dtype, means.device)
            expanded = restore_y_4x(y_q_read, torch.zeros_like(means), mask)
            y_hat_so_far = y_hat_so_far + (expanded * width + means) * mask
        _task938_publish_prior(self, exported_scales, exported_widths, masks)
        return y_hat_so_far * q_dec

    def _train_channel_prior_4x(
        self,
        y_scaled: torch.Tensor,
        params_for_context: torch.Tensor,
        initial_scales: torch.Tensor,
        initial_means: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Legacy four-pass UF helper, factored out for equivalence gates."""
        B, C, H, W = y_scaled.shape
        common_params = self.model.y_spatial_prior_reduction(params_for_context)
        masks = self.model.get_mask_4x(B, C, H, W, y_scaled.dtype, y_scaled.device)
        adaptors = (
            self.model.y_spatial_prior_adaptor_1,
            self.model.y_spatial_prior_adaptor_2,
            self.model.y_spatial_prior_adaptor_3,
        )
        y_hat_so_far = torch.zeros_like(y_scaled)
        bits_y = y_scaled.new_zeros(())
        scales, means = (initial_scales, initial_means)
        exported_scales, exported_widths = ([], [])
        for pass_index, mask in enumerate(masks):
            if pass_index > 0:
                spatial_input = torch.cat((y_hat_so_far, common_params), dim=1)
                scales, means = self.model.y_spatial_prior(adaptors[pass_index - 1](spatial_input)).chunk(2, dim=1)
            width = _task938_effective_width(self, scales, means)
            exported_scales.append((scales / width).detach())
            exported_widths.append(width)
            scales_hat = scales * mask
            means_hat = means * mask
            y_res = (y_scaled - means_hat) * mask
            y_q_hard, y_for_bits = _task938_quantize_train(y_res, self, scales=scales, mask=mask, width=width)
            y_q_hard = y_q_hard * mask
            y_for_bits = y_for_bits * mask
            y_hat_so_far = y_hat_so_far + y_q_hard + means_hat
            bits_y = bits_y + _get_y_gaussian_bits(y_for_bits, scales_hat / width, mask=mask, coder_aligned=True).sum()
        _task938_publish_prior(self, exported_scales, exported_widths, masks)
        return (y_hat_so_far, bits_y)

    def _group_context_contract_payload(self) -> Optional[dict[str, Any]]:
        return None

    def _validate_group_context_contract(self, payload: Any) -> None:
        if payload is not None:
            raise ValueError(
                "I bitstream uses a Task218 group-context contract but the decoder is configured for the legacy entropy path."
            )
        return

    def _core_capacity_contract_payload(self) -> Optional[dict[str, Any]]:
        analysis = None
        synthesis = None
        return None

    def _validate_core_capacity_contract(self, payload: Any) -> None:
        analysis = None
        synthesis = None
        if payload is not None:
            raise ValueError(
                "I bitstream uses a Task438 core-capacity contract but the decoder has both I cores disabled."
            )
        return

    def _native_channel_contract_payload(self) -> Optional[dict[str, Any]]:
        """Describe non-default native I widths without changing legacy streams."""
        if self._native_i_channels == self._default_native_i_channels:
            return None
        payload: dict[str, Any] = {
            "version": _DMCI_NATIVE_CHANNEL_CONTRACT_VERSION,
            "transform_channels": self.i_transform_channels,
            "y_channels": self.i_y_channels,
            "z_channels": self.i_z_channels,
            "scratch_only": True,
        }
        canonical = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
        payload["fingerprint"] = hashlib.sha256(canonical).hexdigest()[:16]
        return payload

    def _validate_native_channel_contract(self, payload: Any) -> None:
        expected = self._native_channel_contract_payload()
        if expected is None:
            if payload is not None:
                raise ValueError(
                    "I bitstream uses non-default native channel widths but the decoder is configured for the legacy 368/256/128 path."
                )
            return
        if payload is None:
            raise ValueError("Missing non-default I native-channel bitstream contract.")
        if not isinstance(payload, dict):
            raise ValueError("Malformed I native-channel bitstream contract.")
        if payload != expected:
            stream_widths = (payload.get("transform_channels"), payload.get("y_channels"), payload.get("z_channels"))
            raise ValueError(
                f"I native-channel bitstream contract mismatch: stream={stream_widths} decoder={self._native_i_channels}; stream_fingerprint={payload.get('fingerprint')!r} decoder_fingerprint={expected['fingerprint']!r}."
            )

    def _progressive_representation_contract_payload(self) -> Optional[dict[str, Any]]:
        progressive_transform = None
        progressive_q = None
        return None

    def _hierarchical_representation_contract_payload(self) -> Optional[dict[str, Any]]:
        module = None
        return None

    def _validate_progressive_representation_contract(self, payload: Any) -> None:
        validate_progressive_contract(payload, self._progressive_representation_contract_payload())

    def _validate_hierarchical_representation_contract(self, payload: Any) -> None:
        validate_hierarchical_contract(payload, self._hierarchical_representation_contract_payload())

    def _y_balance(self, qp: int, y: torch.Tensor) -> torch.Tensor:
        """Channel-wise affine energy rebalance before entropy coding.

        Identity at initialization (gain=1, bias=0).  The decoder applies the
        inverse transform so no side information is transmitted.
        """
        return y

    def _y_balance_inverse(self, qp: int, y_balanced: torch.Tensor) -> torch.Tensor:
        return y_balanced

    def eval(self) -> "DMCIAdapter":
        super().eval()
        self.model.eval()
        self.output_proj.eval()
        return self

    def train(self, mode: bool = True) -> "DMCIAdapter":
        super().train(mode)
        if mode:
            self._entropy_coder_ready = False
        self.model.train(mode)
        self.output_proj.train(mode)
        return self

    def to(self, device: str) -> "DMCIAdapter":
        super().to(device)
        self._device = device
        self.model.to(device)
        self.output_proj.to(device)
        self.qe_log_intra.data = self.qe_log_intra.data.to(device)
        self.qd_log_intra.data = self.qd_log_intra.data.to(device)
        return self

    def _encode_input(self, x: torch.Tensor, qe: torch.Tensor, qp: Optional[int] = None) -> torch.Tensor:
        """Encode [B,C,H,W] latent to y.

        ``qp`` remains optional for legacy diagnostics when the Task438
        analysis core is disabled; an enabled QP-conditioned core requires it.

        Default DCVC-RT mode emits ``H/2,W/2``.  The opt-in preserve mode keeps
        ``H,W`` so the Wan latent is not spatially downsampled a second time.
        """
        out = self.model.enc.enc_1(x, quant_step=qe)
        out = self.model.enc.enc_2(out)
        return out

    def _decode_output(self, y_hat: torch.Tensor, qd: torch.Tensor, qp: Optional[int] = None) -> torch.Tensor:
        """Decode y_hat to latent [B,C,H,W]."""
        out = self.model.dec.dec_1(y_hat)
        out = out * qd
        out = self.model.dec.dec_2(out)
        z_out = self.output_proj(out)
        self._last_pre_synthesis_reconstruction = z_out
        return z_out

    def _bitstream_shapes(self, input_h: int, input_w: int) -> tuple[int, int, int, int]:
        y_h, y_w = (input_h // 2, input_w // 2)
        z_h, z_w = ((input_h + 7) // 8, (input_w + 7) // 8)
        return (y_h, y_w, z_h, z_w)

    def encode_train(self, x: torch.Tensor, qp: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Differentiable I-frame encode. Returns (z_hat, bpp)."""
        self._active_task938_qp = int(qp)
        if self.training:
            self._entropy_coder_ready = False
        input_h, input_w = x.shape[-2:]
        qe = self._get_quant_step(self.qe_log_intra, qp)
        y_raw = self._encode_input(x, qe, qp)
        y_coordinate = self._y_analysis_coordinate(y_raw, qp)
        progressive_transform = None
        hierarchical_transform = None
        if None is not None and self.i_hierarchical_aux_enabled:
            if None is not None or self.i_y_coupling is not None:
                raise RuntimeError(
                    "hierarchical prefix supervision currently requires the legacy pre-hierarchy coordinate (no coupling/Givens transform)"
                )
            self._last_i_progressive_aux = None.progressive_auxiliary(y_raw, y_coordinate, qp)
        else:
            self._last_i_progressive_aux = {}
        y_gain = self._y_gain(qp, y_coordinate)
        y = y_coordinate * y_gain
        y = self._y_balance(qp, y)
        B, C, H, W = y.shape
        y_pad = self.model.pad_for_y(y)
        z = self.model.hyper_enc(y_pad)
        z_hat_hard, z_for_bits = _uf_quantize_train(z)
        params = self.model.hyper_dec(z_hat_hard)
        params = self.model.y_prior_fusion(params)
        params = params[:, :, :H, :W].contiguous()
        q_enc, q_dec, scales, means = self._separate_i_entropy_prior(params, qp)
        if getattr(self, "_export_entropy_scales", False):
            self._last_entropy_scales = scales.detach()
        detach_dynamic_q = bool(getattr(self, "_detach_dynamic_q_for_training", False))
        params_for_context = _detach_leading_prior_channels(params, 2, enabled=detach_dynamic_q)
        if detach_dynamic_q and self.i_entropy_q_mode == "dynamic":
            q_enc = q_enc.detach()
            q_dec = q_dec.detach()
        adaptive_y_gain = self._adaptive_entropy_gain(params_for_context, qp, y)
        post_y_gain = self._post_y_gain(qp, y)
        y_entropy = y * adaptive_y_gain * post_y_gain
        self._last_q_enc_mean = float(q_enc.detach().mean().item())
        self._last_q_enc_floor_fraction = float((q_enc.detach() <= 0.5001).float().mean().item())
        self._last_q_dec_mean = float(q_dec.detach().mean().item())
        progressive_q = None
        self._last_i_progressive_q_group_log_mean = 0.0
        self._last_i_progressive_q_group_log_spread = 0.0
        self._last_i_progressive_q_group_logs = ()
        self._last_i_progressive_q_channel_log_mean_abs = 0.0
        y_scaled = y_entropy * q_enc
        y_hat_so_far, bits_y = self._train_channel_prior_4x(y_scaled, params_for_context, scales, means)
        y_hat = y_hat_so_far * q_dec
        y_hat = y_hat / post_y_gain
        y_hat = y_hat / adaptive_y_gain
        y_hat = self._y_balance_inverse(qp, y_hat)
        y_hat = y_hat / y_gain
        y_hat = self._y_synthesis_coordinate(y_hat, qp)
        bits_z = _get_z_bits(z_for_bits, self.model.bit_estimator_z, qp)
        pix = float(self.bpp_spatial_factor * self.bpp_spatial_factor)
        num_pixels = float(B * H * W * 4) * pix
        bpp_y = bits_y / max(num_pixels, 1.0)
        bpp_z = bits_z.sum() / max(num_pixels, 1.0)
        bpp = bpp_y + bpp_z
        self._last_bpp_y = float(bpp_y.detach().item())
        self._last_bpp_z = float(bpp_z.detach().item())
        qd = self._get_quant_step(self.qd_log_intra, qp)
        z_out = self._decode_output(y_hat, qd, qp=qp)
        return (z_out, bpp)

    def encode(self, x: torch.Tensor, qp: int) -> Tuple[bytes, torch.Tensor]:
        """Real rANS I-frame encode. Returns (bitstream, reconstructed latent)."""
        self._active_task938_qp = int(qp)
        self._ensure_entropy_coder()
        qe = self._get_quant_step(self.qe_log_intra, qp)
        y_raw = self._encode_input(x, qe, qp)
        y_coordinate = self._y_analysis_coordinate(y_raw, qp)
        y_gain = self._y_gain(qp, y_coordinate)
        y = y_coordinate * y_gain
        y = self._y_balance(qp, y)
        B, C, H, W = y.shape
        y_pad = self.model.pad_for_y(y)
        z = self.model.hyper_enc(y_pad)
        z_hat, z_hat_write = round_and_to_int8(z)
        params = self.model.hyper_dec(z_hat)
        params = self.model.y_prior_fusion(params)
        params = params[:, :, :H, :W].contiguous()
        adaptive_y_gain = self._adaptive_entropy_gain(params, qp, y)
        post_y_gain = self._post_y_gain(qp, y)
        y_entropy = y * adaptive_y_gain * post_y_gain
        if (
            self.i_entropy_q_mode == "channel"
            or _task938_width(self, qp) != 1.0
            or getattr(self, "task938_region_quantizer", None) is not None
            or getattr(self, "_export_entropy_scales", False)
        ):
            prior_result = self._compress_channel_prior_4x(y_entropy, params, qp)
        else:
            prior_result = self.model.compress_prior_4x(
                y_entropy,
                params,
                self.model.y_spatial_prior_reduction,
                self.model.y_spatial_prior_adaptor_1,
                self.model.y_spatial_prior_adaptor_2,
                self.model.y_spatial_prior_adaptor_3,
                self.model.y_spatial_prior,
            )
        y_q_w_0, y_q_w_1, y_q_w_2, y_q_w_3, s_w_0, s_w_1, s_w_2, s_w_3, y_hat = prior_result
        y_q_writes = [y_q_w_0, y_q_w_1, y_q_w_2, y_q_w_3]
        scale_writes = [s_w_0, s_w_1, s_w_2, s_w_3]
        self.model.entropy_coder.reset()
        self.model.bit_estimator_z.encode_z(z_hat_write, qp)
        for y_q_write, scale_write in zip(y_q_writes, scale_writes):
            self.model.gaussian_encoder.encode_y(y_q_write, scale_write)
        if self._capture_exact_entropy_symbols:
            self._last_entropy_symbol_tensors = {
                "z": z_hat_write.detach().cpu().clone(),
                "y": tuple((value.detach().cpu().clone() for value in y_q_writes)),
                "y_scales": tuple((value.detach().cpu().clone() for value in scale_writes)),
            }
        else:
            self._last_entropy_symbol_tensors = {}
        self.model.entropy_coder.flush()
        bit_stream = self.model.entropy_coder.get_encoded_stream()
        y_hat = y_hat / post_y_gain
        y_hat = y_hat / adaptive_y_gain
        y_hat = self._y_balance_inverse(qp, y_hat)
        y_hat = y_hat / y_gain
        y_hat = self._y_synthesis_coordinate(y_hat, qp)
        qd = self._get_quant_step(self.qd_log_intra, qp)
        z_out = self._decode_output(y_hat, qd, qp=qp)
        buf = io.BytesIO()
        payload = {
            "codec": "dmci",
            "bit_stream": bit_stream,
            "qp": qp,
            "shape": list(x.shape),
            "i_y_gain_enabled": self.i_y_gain_enabled,
            "i_y_balance_enabled": self.i_y_balance_enabled,
            "i_y_coupling_enabled": self.i_y_coupling_enabled,
            "i_adaptive_y_gain_enabled": self.i_adaptive_y_gain_enabled,
            "i_post_y_gain_enabled": self.i_post_y_gain_enabled,
            "i_entropy_q_mode": self.i_entropy_q_mode,
        }
        group_contract = self._group_context_contract_payload()
        if group_contract is not None:
            payload["i_group_context_contract"] = group_contract
        native_channel_contract = self._native_channel_contract_payload()
        if native_channel_contract is not None:
            payload["i_native_channel_contract"] = native_channel_contract
        core_contract = self._core_capacity_contract_payload()
        if core_contract is not None:
            payload["i_core_capacity_contract"] = core_contract
        progressive_contract = self._progressive_representation_contract_payload()
        if progressive_contract is not None:
            payload["i_progressive_representation_contract"] = progressive_contract
        hierarchical_contract = self._hierarchical_representation_contract_payload()
        if hierarchical_contract is not None:
            payload["i_hierarchical_representation_contract"] = hierarchical_contract
        torch.save(payload, buf)
        return (buf.getvalue(), z_out.detach())

    def decode(self, bitstream: bytes) -> torch.Tensor:
        """Decode a DMCI bitstream to latent."""
        self._ensure_entropy_coder()
        payload = torch.load(io.BytesIO(bitstream), map_location="cpu", weights_only=False)
        bit_stream = payload["bit_stream"]
        qp = payload["qp"]
        bitstream_y_gain = bool(payload.get("i_y_gain_enabled", False))
        if bitstream_y_gain != self.i_y_gain_enabled:
            raise ValueError(
                f"I bitstream y-gain contract mismatch: stream={bitstream_y_gain} decoder={self.i_y_gain_enabled}."
            )
        bitstream_y_balance = bool(payload.get("i_y_balance_enabled", False))
        if bitstream_y_balance != self.i_y_balance_enabled:
            raise ValueError(
                f"I bitstream y-balance contract mismatch: stream={bitstream_y_balance} decoder={self.i_y_balance_enabled}."
            )
        bitstream_y_coupling = bool(payload.get("i_y_coupling_enabled", False))
        if bitstream_y_coupling != self.i_y_coupling_enabled:
            raise ValueError(
                f"I bitstream y-coupling contract mismatch: stream={bitstream_y_coupling} decoder={self.i_y_coupling_enabled}."
            )
        bitstream_adaptive_y_gain = bool(payload.get("i_adaptive_y_gain_enabled", False))
        if bitstream_adaptive_y_gain != self.i_adaptive_y_gain_enabled:
            raise ValueError(
                f"I bitstream adaptive y-gain contract mismatch: stream={bitstream_adaptive_y_gain} decoder={self.i_adaptive_y_gain_enabled}."
            )
        bitstream_post_y_gain = bool(payload.get("i_post_y_gain_enabled", False))
        if bitstream_post_y_gain != self.i_post_y_gain_enabled:
            raise ValueError(
                f"I bitstream post-hyper y-gain contract mismatch: stream={bitstream_post_y_gain} decoder={self.i_post_y_gain_enabled}."
            )
        bitstream_entropy_q_mode = str(payload.get("i_entropy_q_mode", "dynamic")).strip().lower()
        if bitstream_entropy_q_mode != self.i_entropy_q_mode:
            raise ValueError(
                f"I bitstream entropy-q contract mismatch: stream={bitstream_entropy_q_mode} decoder={self.i_entropy_q_mode}."
            )
        self._validate_native_channel_contract(payload.get("i_native_channel_contract"))
        self._validate_group_context_contract(payload.get("i_group_context_contract"))
        self._validate_core_capacity_contract(payload.get("i_core_capacity_contract"))
        self._validate_progressive_representation_contract(payload.get("i_progressive_representation_contract"))
        self._validate_hierarchical_representation_contract(payload.get("i_hierarchical_representation_contract"))
        B, C, H, W = payload["shape"]
        assert C == self.latent_channels, f"DMCIAdapter.decode expects {self.latent_channels} channels, got {C}"
        y_h, y_w, z_h, z_w = self._bitstream_shapes(H, W)
        device = next(self.model.parameters()).device
        dtype = next(self.model.parameters()).dtype
        self.model.entropy_coder.set_use_two_entropy_coders(False)
        self.model.entropy_coder.set_stream(bit_stream)
        self.model.bit_estimator_z.decode_z((z_h, z_w), qp)
        z_hat = self.model.bit_estimator_z.get_z((z_h, z_w), device, dtype)
        params = self.model.hyper_dec(z_hat)
        params = self.model.y_prior_fusion(params)
        params = params[:, :, :y_h, :y_w].contiguous()
        adaptive_y_gain = self._adaptive_entropy_gain(params, qp, params[:, : self.i_y_channels])
        post_y_gain = self._post_y_gain(qp, params[:, : self.i_y_channels])
        if (
            self.i_entropy_q_mode == "channel"
            or _task938_width(self, qp) != 1.0
            or getattr(self, "task938_region_quantizer", None) is not None
            or getattr(self, "_export_entropy_scales", False)
        ):
            y_hat = self._decompress_channel_prior_4x(params, qp)
        else:
            y_hat = self.model.decompress_prior_4x(
                params,
                self.model.y_spatial_prior_reduction,
                self.model.y_spatial_prior_adaptor_1,
                self.model.y_spatial_prior_adaptor_2,
                self.model.y_spatial_prior_adaptor_3,
                self.model.y_spatial_prior,
            )
        y_hat = y_hat / post_y_gain
        y_hat = y_hat / adaptive_y_gain
        y_hat = self._y_balance_inverse(qp, y_hat)
        y_hat = y_hat / self._y_gain(qp, y_hat)
        y_hat = self._y_synthesis_coordinate(y_hat, qp)
        qd = self._get_quant_step(self.qd_log_intra, qp)
        z_out = self._decode_output(y_hat, qd, qp=qp)
        if z_out.shape[-2] > H or z_out.shape[-1] > W:
            z_out = z_out[:, :, :H, :W]
        return z_out.detach()


class _PResidualBlock(nn.Module):
    """Pre-activation residual block used by the high-capacity P predictors."""

    def __init__(self, channels: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1), nn.GELU(), nn.Conv2d(channels, channels, 3, padding=1)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.net(x)


class LatentResidualPredictor(nn.Module):
    """Decoded-reference predictor for the explicit P residual mode.

    ``legacy`` preserves the historical two-layer Task37/57 module and its
    checkpoint key layout. ``multiscale_v2`` adds enough spatial capacity to
    model motion/deformation in Wan latents while remaining strictly causal:
    every input is reconstructed by the decoder before the current P frame.
    """

    def __init__(
        self,
        latent_channels: int = _VAE_CH,
        context_channels: int = _RT_D,
        hidden_channels: int = 96,
        use_context: bool = True,
        architecture: str = "legacy",
        num_blocks: int = 6,
        p_type_count: int = 3,
    ):
        super().__init__()
        self.use_context = bool(use_context)
        self.architecture = str(architecture).strip().lower()
        self.p_type_count = max(1, int(p_type_count))
        if self.architecture not in {"legacy", "multiscale_v2"}:
            raise ValueError(f"residual predictor architecture must be legacy|multiscale_v2, got {architecture!r}.")
        in_channels = latent_channels + 1
        if self.use_context:
            in_channels += context_channels
        if self.architecture == "legacy":
            self.net = nn.Sequential(
                nn.Conv2d(in_channels, hidden_channels, 3, padding=1),
                nn.GELU(),
                nn.Conv2d(hidden_channels, hidden_channels, 3, padding=1),
                nn.GELU(),
                nn.Conv2d(hidden_channels, latent_channels, 3, padding=1),
            )
            output_layer = self.net[-1]
        else:
            in_channels += 1
            hidden_channels = max(int(hidden_channels), latent_channels)
            block_count = max(2, int(num_blocks))
            self.stem = nn.Conv2d(in_channels, hidden_channels, 3, padding=1)
            self.local_blocks = nn.ModuleList([_PResidualBlock(hidden_channels) for _ in range(block_count)])
            self.global_down = nn.Conv2d(hidden_channels, hidden_channels, 3, stride=2, padding=1)
            self.global_blocks = nn.ModuleList(
                [_PResidualBlock(hidden_channels) for _ in range(max(2, block_count // 2))]
            )
            self.fusion = nn.Sequential(
                nn.Conv2d(2 * hidden_channels, hidden_channels, 1),
                nn.GELU(),
                nn.Conv2d(hidden_channels, hidden_channels, 3, padding=1),
                nn.GELU(),
            )
            self.output = nn.Conv2d(hidden_channels, latent_channels, 3, padding=1)
            output_layer = self.output
        nn.init.zeros_(output_layer.weight)
        nn.init.zeros_(output_layer.bias)

    def forward(self, z_prev: torch.Tensor, ctx: torch.Tensor, qp: int, frame_type_idx: int = 0) -> torch.Tensor:
        qp_value = float(qp) / max(float(_INTER_QP_NUM - 1), 1.0)
        qp_map = torch.full(
            (z_prev.shape[0], 1, z_prev.shape[-2], z_prev.shape[-1]), qp_value, device=z_prev.device, dtype=z_prev.dtype
        )
        inputs = [z_prev, qp_map]
        if self.use_context:
            inputs.append(ctx)
        if self.architecture == "legacy":
            delta = self.net(torch.cat(inputs, dim=1))
        else:
            type_value = float(max(0, min(int(frame_type_idx), self.p_type_count - 1)))
            type_value /= max(float(self.p_type_count - 1), 1.0)
            inputs.append(torch.full_like(qp_map, type_value))
            local = F.gelu(self.stem(torch.cat(inputs, dim=1)))
            for block in self.local_blocks:
                local = block(local)
            global_features = F.gelu(self.global_down(local))
            for block in self.global_blocks:
                global_features = block(global_features)
            global_features = F.interpolate(
                global_features, size=local.shape[-2:], mode="bilinear", align_corners=False
            )
            delta = self.output(self.fusion(torch.cat((local, global_features), dim=1)))
        return z_prev + delta


class LatentPRefiner(nn.Module):
    """Temporal reconstruction-side refiner for P-frame decoded latents.

    Unlike explicit residual mode, this module does not change the entropy-coded
    input.  It only predicts a small post-decode correction from the decoded
    P latent, the previous reconstructed latent, temporal context, and QP.
    """

    def __init__(
        self,
        latent_channels: int = _VAE_CH,
        context_channels: int = _RT_D,
        hidden_channels: int = 128,
        use_context: bool = True,
        residual_scale: float = 0.1,
    ):
        super().__init__()
        self.use_context = bool(use_context)
        self.residual_scale = float(residual_scale)
        in_channels = latent_channels * 2 + 1
        if self.use_context:
            in_channels += context_channels
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, hidden_channels, 3, padding=1),
            nn.GELU(),
            nn.Conv2d(hidden_channels, hidden_channels, 3, padding=1),
            nn.GELU(),
            nn.Conv2d(hidden_channels, hidden_channels, 3, padding=1),
            nn.GELU(),
            nn.Conv2d(hidden_channels, latent_channels, 3, padding=1),
        )
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, z_hat: torch.Tensor, z_prev: torch.Tensor, ctx: torch.Tensor, qp: int) -> torch.Tensor:
        qp_value = float(qp) / max(float(_INTER_QP_NUM - 1), 1.0)
        qp_map = torch.full(
            (z_hat.shape[0], 1, z_hat.shape[-2], z_hat.shape[-1]), qp_value, device=z_hat.device, dtype=z_hat.dtype
        )
        inputs = [z_hat, z_prev, qp_map]
        if self.use_context:
            inputs.append(ctx)
        delta = self.net(torch.cat(inputs, dim=1))
        return z_hat + self.residual_scale * delta


class AnchorMultiscaleCorrection(nn.Module):
    """Zero-initialized local/global correction for decoder-known references."""

    def __init__(self, in_channels: int, hidden_channels: int, out_channels: int, blocks: int = 4):
        super().__init__()
        hidden_channels = max(int(hidden_channels), int(out_channels))
        block_count = max(2, int(blocks))
        self.stem = nn.Conv2d(in_channels, hidden_channels, 3, padding=1)
        self.local_blocks = nn.ModuleList([_PResidualBlock(hidden_channels) for _ in range(block_count)])
        self.global_down = nn.Conv2d(hidden_channels, hidden_channels, 3, stride=2, padding=1)
        self.global_blocks = nn.ModuleList([_PResidualBlock(hidden_channels) for _ in range(max(2, block_count // 2))])
        self.fusion = nn.Sequential(
            nn.Conv2d(2 * hidden_channels, hidden_channels, 1),
            nn.GELU(),
            nn.Conv2d(hidden_channels, hidden_channels, 3, padding=1),
            nn.GELU(),
        )
        self.output = nn.Conv2d(hidden_channels, out_channels, 3, padding=1)
        nn.init.zeros_(self.output.weight)
        nn.init.zeros_(self.output.bias)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        local = F.gelu(self.stem(inputs))
        for block in self.local_blocks:
            local = block(local)
        global_features = F.gelu(self.global_down(local))
        for block in self.global_blocks:
            global_features = block(global_features)
        global_features = F.interpolate(global_features, size=local.shape[-2:], mode="bilinear", align_corners=False)
        return self.output(self.fusion(torch.cat((local, global_features), dim=1)))


class AnchorBroadcastAdapter(nn.Module):
    """Decoder-known I0 anchor adapter for P-frame temporal context.

    The first reconstructed latent in a GOP is available to both encoder and
    decoder.  This adapter uses that anchor to adjust the P reference before
    DCVC-RT's feature extractor/context model sees it.  It does not add side
    information to the bitstream; the same adjusted reference is recomputed at
    decode time from the decoded anchor and decoded previous latent.
    """

    def __init__(
        self,
        channels: int = _VAE_CH,
        hidden_channels: int = 128,
        residual_scale: float = 0.1,
        use_type: bool = True,
        multiscale_enabled: bool = False,
        multiscale_hidden: int = 192,
        multiscale_blocks: int = 4,
        multiscale_scale: float = 0.25,
    ):
        super().__init__()
        hidden_channels = max(int(hidden_channels), int(channels))
        self.residual_scale = float(residual_scale)
        self.use_type = bool(use_type)
        self.multiscale_scale = float(multiscale_scale)
        extra_channels = 2 if self.use_type else 1
        self.net = nn.Sequential(
            nn.Conv2d(channels * 3 + extra_channels, hidden_channels, 3, padding=1),
            nn.GELU(),
            nn.Conv2d(hidden_channels, hidden_channels, 3, padding=1),
            nn.GELU(),
            nn.Conv2d(hidden_channels, channels, 3, padding=1),
        )
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)
        self.multiscale: Optional[AnchorMultiscaleCorrection] = None
        if bool(multiscale_enabled):
            self.multiscale = AnchorMultiscaleCorrection(
                in_channels=channels * 3 + extra_channels,
                hidden_channels=int(multiscale_hidden),
                out_channels=channels,
                blocks=int(multiscale_blocks),
            )

    def forward(self, z_prev: torch.Tensor, z_anchor: torch.Tensor, qp: int, frame_type_idx: int) -> torch.Tensor:
        qp_value = float(qp) / max(float(_INTER_QP_NUM - 1), 1.0)
        qp_map = torch.full(
            (z_prev.shape[0], 1, z_prev.shape[-2], z_prev.shape[-1]), qp_value, device=z_prev.device, dtype=z_prev.dtype
        )
        pieces = [z_prev, z_anchor, z_anchor - z_prev, qp_map]
        if self.use_type:
            type_value = float(max(0, frame_type_idx)) / max(float(3 - 1), 1.0)
            type_map = torch.full_like(qp_map, type_value)
            pieces.append(type_map)
        inputs = torch.cat(pieces, dim=1)
        delta = self.net(inputs)
        output = z_prev + self.residual_scale * delta
        if self.multiscale is not None:
            output = output + self.multiscale_scale * self.multiscale(inputs)
        return output


class ThirdOrderReferenceAdapter(nn.Module):
    """Decoder-known acceleration correction from three reconstructed frames."""

    def __init__(
        self,
        channels: int = _VAE_CH,
        hidden_channels: int = 96,
        blocks: int = 2,
        residual_scale: float = 0.05,
        position_count: int = 13,
    ) -> None:
        super().__init__()
        channels = int(channels)
        hidden_channels = max(int(hidden_channels), channels)
        if blocks <= 0 or residual_scale <= 0 or position_count <= 1:
            raise ValueError("third-order reference configuration is invalid")
        self.residual_scale = float(residual_scale)
        self.position_count = int(position_count)
        self.stem = nn.Conv2d(channels * 6 + 2, hidden_channels, 3, padding=1)
        self.blocks = nn.ModuleList([_PResidualBlock(hidden_channels) for _ in range(int(blocks))])
        self.output = nn.Conv2d(hidden_channels, channels + 1, 3, padding=1)
        nn.init.zeros_(self.output.weight)
        nn.init.zeros_(self.output.bias)

    def forward(
        self,
        z_prev: torch.Tensor,
        z_prev2: torch.Tensor,
        z_prev3: torch.Tensor,
        z_anchor: torch.Tensor,
        qp: int,
        position: int,
    ) -> torch.Tensor:
        velocity_1 = z_prev - z_prev2
        velocity_2 = z_prev2 - z_prev3
        acceleration = velocity_1 - velocity_2
        qp_map = torch.full_like(z_prev[:, :1], float(qp) / max(float(_INTER_QP_NUM - 1), 1.0))
        position_map = torch.full_like(
            qp_map, float(max(0, min(int(position), self.position_count - 1))) / float(self.position_count - 1)
        )
        pieces = (
            z_prev,
            velocity_1,
            velocity_2,
            acceleration,
            acceleration.abs(),
            z_anchor - z_prev,
            qp_map,
            position_map,
        )
        feature = F.gelu(self.stem(torch.cat(pieces, dim=1)))
        for block in self.blocks:
            feature = block(feature)
        delta, confidence_logit = self.output(feature).split((z_prev.shape[1], 1), dim=1)
        correction = torch.sigmoid(confidence_logit) * torch.tanh(delta)
        return z_prev + self.residual_scale * correction


class PAdaptiveEntropyPrior(nn.Module):
    """Sample-adaptive P entropy correction from decoder-known context.

    Task136/137 showed that a static per-type affine CDF correction is too weak.
    This module predicts bounded mean/log-scale residuals from the decoded
    temporal context, QP and P position.  A separate second-pass head also sees
    the already decoded checkerboard half ``y_hat_0``.  Both heads are
    zero-initialized, making an enabled module bit-exact to Task123 before
    optimization.
    """

    def __init__(
        self,
        context_channels: int = _RT_D,
        y_channels: int = _RT_Y,
        hidden_channels: int = 96,
        p_type_count: int = 3,
        max_log_scale: float = 0.75,
        max_mean_delta: float = 0.5,
    ):
        super().__init__()
        hidden_channels = max(int(hidden_channels), 16)
        self.p_type_count = max(1, int(p_type_count))
        self.max_log_scale = float(max_log_scale)
        self.max_mean_delta = float(max_mean_delta)
        if self.max_log_scale <= 0 or self.max_mean_delta <= 0:
            raise ValueError("Adaptive entropy correction bounds must be positive.")
        self.context_proj = nn.Conv2d(context_channels, hidden_channels, 1)
        self.y_hat_proj = nn.Conv2d(y_channels, hidden_channels, 1)
        self.qp_embedding = nn.Embedding(_INTER_QP_NUM, hidden_channels)
        self.type_embedding = nn.Embedding(self.p_type_count, hidden_channels)
        self.trunk = nn.Sequential(
            nn.Conv2d(hidden_channels, hidden_channels, 3, padding=1),
            nn.GELU(),
            nn.Conv2d(hidden_channels, hidden_channels, 3, padding=1),
            nn.GELU(),
        )
        self.common_head = nn.Conv2d(hidden_channels, 2 * y_channels, 1)
        self.spatial_head = nn.Conv2d(hidden_channels, 2 * y_channels, 1)
        for head in (self.common_head, self.spatial_head):
            nn.init.zeros_(head.weight)
            nn.init.zeros_(head.bias)

    def forward(
        self,
        ctx_t: torch.Tensor,
        qp: int,
        frame_type_idx: int,
        *,
        y_size: tuple[int, int],
        y_hat_0: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if ctx_t.shape[-2:] != y_size:
            ctx_t = F.interpolate(ctx_t, size=y_size, mode="bilinear", align_corners=False)
        features = self.context_proj(ctx_t)
        qp_idx = max(0, min(int(qp), _INTER_QP_NUM - 1))
        type_idx = max(0, min(int(frame_type_idx), self.p_type_count - 1))
        condition = self.qp_embedding.weight[qp_idx] + self.type_embedding.weight[type_idx]
        features = features + condition.to(device=features.device, dtype=features.dtype).view(1, -1, 1, 1)
        if y_hat_0 is not None:
            if y_hat_0.shape[-2:] != y_size:
                y_hat_0 = F.interpolate(y_hat_0, size=y_size, mode="bilinear", align_corners=False)
            features = features + self.y_hat_proj(y_hat_0)
        features = self.trunk(F.gelu(features))
        head = self.spatial_head if y_hat_0 is not None else self.common_head
        log_scale_delta, mean_delta = head(features).chunk(2, dim=1)
        return (torch.tanh(log_scale_delta) * self.max_log_scale, torch.tanh(mean_delta) * self.max_mean_delta)


class PSecondPassSpatialScaleRefiner(nn.Module):
    """Split the mature second checkerboard pass into two scale-only passes."""

    def __init__(
        self,
        *,
        y_channels: int = _RT_Y,
        hidden_channels: int = 128,
        blocks: int = 2,
        p_type_count: int = 3,
        max_log_scale: float = 0.75,
    ) -> None:
        super().__init__()
        self.y_channels = int(y_channels)
        self.hidden_channels = int(hidden_channels)
        self.blocks_count = int(blocks)
        self.p_type_count = max(1, int(p_type_count))
        self.max_log_scale = float(max_log_scale)
        if self.hidden_channels <= 0 or self.blocks_count <= 0:
            raise ValueError("P spatial scale-refiner width/blocks must be positive")
        if self.max_log_scale <= 0:
            raise ValueError("P spatial scale-refiner bound must be positive")
        self.input = nn.Conv2d(4 * self.y_channels, self.hidden_channels, 1)
        self.qp_embedding = nn.Embedding(_INTER_QP_NUM, self.hidden_channels)
        self.position_embedding = nn.Embedding(self.p_type_count, self.hidden_channels)
        self.blocks = nn.ModuleList(
            (
                nn.Sequential(
                    nn.Conv2d(self.hidden_channels, self.hidden_channels, 3, padding=1, groups=self.hidden_channels),
                    nn.GELU(),
                    nn.Conv2d(self.hidden_channels, self.hidden_channels, 1),
                    nn.GELU(),
                )
                for _ in range(self.blocks_count)
            )
        )
        self.heads = nn.ModuleList((nn.Conv2d(self.hidden_channels, self.y_channels, 1) for _ in range(2)))
        for head in self.heads:
            nn.init.zeros_(head.weight)
            nn.init.zeros_(head.bias)

    def contract_payload(self) -> dict[str, Any]:
        return {
            "schema": "p_second_checkerboard_scale_refiner_v1",
            "passes": 3,
            "split": "parent_mask1_even_row_then_odd_row",
            "reconstruction": "parent_means_and_symbols_bit_exact",
            "parameterization": "scale_only",
            "max_log_scale": self.max_log_scale,
            "side_information": False,
        }

    def forward(
        self, y_hat_prefix: torch.Tensor, common_params: torch.Tensor, qp: int, frame_type_idx: int, subpass: int
    ) -> torch.Tensor:
        index = int(subpass)
        if index not in (0, 1):
            raise ValueError("P spatial scale-refiner subpass must be 0 or 1")
        features = self.input(torch.cat((y_hat_prefix, common_params), dim=1))
        qp_index = max(0, min(int(qp), _INTER_QP_NUM - 1))
        position_index = max(0, min(int(frame_type_idx), self.p_type_count - 1))
        condition = (self.qp_embedding.weight[qp_index] + self.position_embedding.weight[position_index]).to(
            device=features.device, dtype=features.dtype
        )[None, :, None, None]
        features = features + condition
        for block in self.blocks:
            features = features + block(features)
        return torch.tanh(self.heads[index](features)) * self.max_log_scale


class PResidualSynthesisHead(nn.Module):
    """Decoder-only quality head on DCVC-RT's 320-channel reconstruction feature.

    The last layer is zero initialized, so enabling the module before loading a
    Task123 checkpoint is exactly the legacy synthesis transform.  It consumes
    no encoder-only state and is therefore symmetric in real encode/decode.
    """

    def __init__(
        self,
        latent_channels: int = _VAE_CH,
        hidden_channels: int = 160,
        residual_scale: float = 0.1,
        position_conditioned: bool = False,
        position_count: int = 6,
        position_hidden_channels: int = 128,
        qp_conditioned: bool = False,
        qp_values: Optional[Sequence[int]] = None,
        qp_hidden_channels: int = 128,
        qp_position_factorized: bool = False,
    ):
        super().__init__()
        self.residual_scale = float(residual_scale)
        self.position_conditioned = bool(position_conditioned)
        self.position_count = max(1, int(position_count))
        self.net = nn.Sequential(
            nn.Conv2d(_RT_RECON, int(hidden_channels), 3, padding=1),
            nn.GELU(),
            nn.Conv2d(int(hidden_channels), latent_channels, 3, padding=1),
        )
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)
        self.position_branches: Optional[nn.ModuleList] = None
        if self.position_conditioned:
            self.position_branches = nn.ModuleList()
            for _ in range(self.position_count):
                branch = nn.Sequential(
                    nn.Conv2d(_RT_RECON + 1, int(position_hidden_channels), 3, padding=1),
                    nn.GELU(),
                    nn.Conv2d(int(position_hidden_channels), latent_channels, 3, padding=1),
                )
                nn.init.zeros_(branch[-1].weight)
                nn.init.zeros_(branch[-1].bias)
                self.position_branches.append(branch)
        self.qp_values: tuple[int, ...] = ()
        self.qp_position_factorized = bool(qp_position_factorized)
        self.qp_branches: Optional[nn.ModuleList] = None
        if self.qp_position_factorized and (not bool(qp_conditioned)):
            raise ValueError("p_synthesis_qp_position_factorized requires p_synthesis_qp_conditioned=true")
        if bool(qp_conditioned):
            parsed_qps = tuple((int(value) for value in ((0, 16, 39, 63) if qp_values is None else qp_values)))
            if not parsed_qps or len(set(parsed_qps)) != len(parsed_qps):
                raise ValueError("p_synthesis_qp_values must be a non-empty list of unique QPs")
            invalid_qps = [value for value in parsed_qps if value < 0 or value >= _INTER_QP_NUM]
            if invalid_qps:
                raise ValueError(f"p_synthesis_qp_values must be in [0, {_INTER_QP_NUM - 1}], got {invalid_qps}")
            self.qp_values = parsed_qps
            self.qp_branches = nn.ModuleList()
            branch_count = len(self.qp_values)
            if self.qp_position_factorized:
                branch_count *= self.position_count
            for _ in range(branch_count):
                branch = nn.Sequential(
                    nn.Conv2d(_RT_RECON + 1, int(qp_hidden_channels), 3, padding=1),
                    nn.GELU(),
                    nn.Conv2d(int(qp_hidden_channels), latent_channels, 3, padding=1),
                )
                nn.init.zeros_(branch[-1].weight)
                nn.init.zeros_(branch[-1].bias)
                self.qp_branches.append(branch)

    def forward(
        self,
        feature: torch.Tensor,
        shape_like: torch.Tensor,
        *,
        position_idx: Optional[int] = None,
        qp: Optional[int] = None,
        reference_latent: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        del reference_latent
        delta = self.net(feature)
        if self.position_branches is not None and position_idx is not None:
            branch_idx = max(0, min(int(position_idx) - 1, len(self.position_branches) - 1))
            qp_value = float(0 if qp is None else qp) / max(float(_INTER_QP_NUM - 1), 1.0)
            qp_map = torch.full(
                (feature.shape[0], 1, feature.shape[-2], feature.shape[-1]),
                qp_value,
                device=feature.device,
                dtype=feature.dtype,
            )
            delta = delta + self.position_branches[branch_idx](torch.cat((feature, qp_map), dim=1))
        if self.qp_branches is not None and qp is not None:
            qp_branch_idx = min(range(len(self.qp_values)), key=lambda idx: abs(self.qp_values[idx] - int(qp)))
            position_branch_idx = max(0, min(int(position_idx or 1) - 1, self.position_count - 1))
            position_value = float(position_branch_idx) / max(float(self.position_count - 1), 1.0)
            position_map = torch.full(
                (feature.shape[0], 1, feature.shape[-2], feature.shape[-1]),
                position_value,
                device=feature.device,
                dtype=feature.dtype,
            )
            if self.qp_position_factorized:
                qp_branch_idx = qp_branch_idx * self.position_count + position_branch_idx
            delta = delta + self.qp_branches[qp_branch_idx](torch.cat((feature, position_map), dim=1))
        if delta.shape[-2:] != shape_like.shape[-2:]:
            delta = F.interpolate(delta, size=shape_like.shape[-2:], mode="bilinear", align_corners=False)
        return shape_like + self.residual_scale * delta


class PContinuousDisplaySynthesisHead(nn.Module):
    """Shared out-of-loop P decoder with continuous GOP-age conditioning.

    Unlike the historical six position experts, this module has one parameter
    set for every P age.  Its optional temporal coordinate is built only from
    the immediate hard-decoded reference, so both encoder and decoder can
    reproduce it without payload bits.  The output is display/DiT condition
    only; :class:`RTTwoCodecPipeline` keeps ``shape_like`` as the causal P
    reference when the out-of-loop contract is enabled.
    """

    def __init__(
        self,
        latent_channels: int = _VAE_CH,
        hidden_channels: int = 192,
        blocks: int = 4,
        residual_scale: float = 0.25,
        max_position: int = 24,
        reference_mode: str = "correct",
        base_head: Optional[nn.Module] = None,
    ) -> None:
        super().__init__()
        hidden_channels = max(int(hidden_channels), int(latent_channels))
        blocks = max(1, int(blocks))
        self.latent_channels = int(latent_channels)
        self.hidden_channels = hidden_channels
        self.blocks_count = blocks
        self.residual_scale = float(residual_scale)
        self.max_position = max(2, int(max_position))
        self.legacy_head = base_head
        if self.residual_scale <= 0:
            raise ValueError("continuous display residual scale must be positive")
        self.feature_projection = nn.Conv2d(_RT_RECON, hidden_channels, 1)
        self.base_projection = nn.Conv2d(latent_channels, hidden_channels, 3, padding=1)
        self.reference_projection = nn.Conv2d(3 * latent_channels, hidden_channels, 3, padding=1)
        self.qp_embedding = nn.Embedding(_INTER_QP_NUM, hidden_channels)
        self.age_projection = nn.Sequential(
            nn.Linear(6, hidden_channels), nn.SiLU(), nn.Linear(hidden_channels, hidden_channels)
        )
        self.local_blocks = nn.ModuleList([_PResidualBlock(hidden_channels) for _ in range(blocks)])
        self.global_down = nn.Conv2d(hidden_channels, hidden_channels, 3, stride=2, padding=1)
        self.global_blocks = nn.ModuleList([_PResidualBlock(hidden_channels) for _ in range(max(1, blocks // 2))])
        self.fusion = nn.Sequential(
            nn.Conv2d(2 * hidden_channels, hidden_channels, 1),
            nn.SiLU(),
            nn.Conv2d(hidden_channels, hidden_channels, 3, padding=1),
            nn.SiLU(),
        )
        self.output = nn.Conv2d(hidden_channels, latent_channels, 3, padding=1)
        nn.init.zeros_(self.output.weight)
        nn.init.zeros_(self.output.bias)
        self.set_reference_mode(reference_mode)

    def set_reference_mode(self, mode: str) -> None:
        mode = str(mode).lower().strip()
        if mode not in {"correct", "zero", "wrong"}:
            raise ValueError("continuous display reference mode must be correct|zero|wrong")
        self.reference_mode = mode

    def contract_payload(self) -> dict[str, Any]:
        return {
            "version": 1,
            "architecture": "continuous_age_v1",
            "hidden_channels": self.hidden_channels,
            "blocks": self.blocks_count,
            "residual_scale": self.residual_scale,
            "max_position": self.max_position,
            "mature_display_base": self.legacy_head is not None,
        }

    def _reference_coordinate(self, base: torch.Tensor, reference: Optional[torch.Tensor]) -> torch.Tensor:
        if reference is None:
            reference = torch.zeros_like(base)
        if reference.shape[-2:] != base.shape[-2:]:
            reference = F.interpolate(reference, size=base.shape[-2:], mode="bilinear", align_corners=False)
        if self.reference_mode == "zero":
            reference = torch.zeros_like(reference)
        elif self.reference_mode == "wrong":
            reference = torch.roll(
                reference, shifts=(max(1, reference.shape[-2] // 2), max(1, reference.shape[-1] // 2)), dims=(-2, -1)
            )
        innovation = base - reference
        return torch.cat((reference, innovation, innovation.abs()), dim=1)

    def _age_features(self, position_idx: int, *, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        position = max(1, int(position_idx))
        u = min(float(position) / float(self.max_position), 2.0)
        log_age = math.log1p(float(position)) / math.log1p(float(self.max_position))
        reciprocal = 1.0 / float(position)
        return torch.tensor([u, u * u, u * u * u, log_age, reciprocal, 1.0], device=device, dtype=dtype).view(1, 6)

    def forward(
        self,
        feature: torch.Tensor,
        shape_like: torch.Tensor,
        *,
        position_idx: Optional[int] = None,
        qp: Optional[int] = None,
        reference_latent: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if position_idx is None or qp is None:
            raise RuntimeError("continuous display synthesis requires decoder-known QP and P age")
        if self.legacy_head is None:
            base = shape_like
        else:
            base = self.legacy_head(
                feature, shape_like, position_idx=position_idx, qp=qp, reference_latent=reference_latent
            )
        reference_coordinate = self._reference_coordinate(base, reference_latent)
        local = (
            self.feature_projection(feature)
            + self.base_projection(base)
            + self.reference_projection(reference_coordinate)
        )
        qp_idx = max(0, min(int(qp), _INTER_QP_NUM - 1))
        condition = self.qp_embedding.weight[qp_idx].to(device=local.device, dtype=local.dtype)
        age = self.age_projection(self._age_features(int(position_idx), device=local.device, dtype=local.dtype))[0]
        local = F.silu(local + condition.view(1, -1, 1, 1) + age.view(1, -1, 1, 1))
        for block in self.local_blocks:
            local = block(local)
        global_feature = F.silu(self.global_down(local))
        for block in self.global_blocks:
            global_feature = block(global_feature)
        global_feature = F.interpolate(global_feature, size=local.shape[-2:], mode="bilinear", align_corners=False)
        delta = self.output(self.fusion(torch.cat((local, global_feature), dim=1)))
        if delta.shape[-2:] != base.shape[-2:]:
            delta = F.interpolate(delta, size=base.shape[-2:], mode="bilinear", align_corners=False)
        return base + self.residual_scale * delta


class LatentYContextPredictor(nn.Module):
    """Predict the P-frame entropy bottleneck from temporal context.

    This module is used by the experimental ``y_centered`` P mode.  The decoder
    can reconstruct the same prediction from the reference state, so the
    bitstream only needs to code the residual around this context-predicted y.
    """

    def __init__(self, context_channels: int = _RT_D, y_channels: int = _RT_Y, hidden_channels: int = 192):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(context_channels + 1, hidden_channels, 3, padding=1),
            nn.GELU(),
            nn.Conv2d(hidden_channels, hidden_channels, 3, stride=2, padding=1),
            nn.GELU(),
            nn.Conv2d(hidden_channels, hidden_channels, 3, padding=1),
            nn.GELU(),
            nn.Conv2d(hidden_channels, y_channels, 3, padding=1),
        )
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, ctx: torch.Tensor, qp: int, shape_like: torch.Tensor) -> torch.Tensor:
        qp_value = float(qp) / max(float(_INTER_QP_NUM - 1), 1.0)
        qp_map = torch.full(
            (ctx.shape[0], 1, ctx.shape[-2], ctx.shape[-1]), qp_value, device=ctx.device, dtype=ctx.dtype
        )
        pred = self.net(torch.cat((ctx, qp_map), dim=1))
        if pred.shape[-2:] != shape_like.shape[-2:]:
            pred = F.interpolate(pred, size=shape_like.shape[-2:], mode="bilinear", align_corners=False)
        return pred


class PTask452MotionContextAdapter(nn.Module):
    """Decoder-closed residual reference adapter for quantized cache motion."""

    def __init__(
        self,
        *,
        latent_channels: int,
        hidden_channels: int = 64,
        blocks: int = 3,
        residual_scale: float = 0.25,
        qp_position_conditioned: bool = False,
        p_type_count: int = 3,
        bounded_output: bool = False,
    ) -> None:
        super().__init__()
        if hidden_channels <= 0 or blocks <= 0 or residual_scale <= 0:
            raise ValueError("Task452 motion-context adapter values must be positive")
        self.residual_scale = float(residual_scale)
        self.qp_position_conditioned = bool(qp_position_conditioned)
        self.p_type_count = max(1, int(p_type_count))
        self.bounded_output = bool(bounded_output)
        condition_channels = 2 if self.qp_position_conditioned else 0
        self.stem = nn.Conv2d(2 * latent_channels + 1 + condition_channels, hidden_channels, 3, padding=1)
        self.blocks = nn.ModuleList(
            [
                nn.Sequential(
                    nn.GroupNorm(max(1, math.gcd(hidden_channels, 8)), hidden_channels),
                    nn.SiLU(),
                    nn.Conv2d(hidden_channels, hidden_channels, 3, padding=1),
                    nn.GroupNorm(max(1, math.gcd(hidden_channels, 8)), hidden_channels),
                    nn.SiLU(),
                    nn.Conv2d(hidden_channels, hidden_channels, 3, padding=1),
                )
                for _ in range(int(blocks))
            ]
        )
        self.head = nn.Conv2d(hidden_channels, latent_channels, 3, padding=1)
        nn.init.zeros_(self.head.weight)
        nn.init.zeros_(self.head.bias)

    def forward(
        self, z_prev: torch.Tensor, warped: torch.Tensor, confidence: torch.Tensor, qp: int = 0, frame_type_idx: int = 0
    ) -> torch.Tensor:
        motion_delta = confidence * (warped - z_prev)
        inputs = [z_prev, motion_delta, confidence]
        if self.qp_position_conditioned:
            qp_value = float(qp) / max(float(_INTER_QP_NUM - 1), 1.0)
            type_value = float(max(0, min(int(frame_type_idx), self.p_type_count - 1))) / max(
                float(self.p_type_count - 1), 1.0
            )
            inputs.extend([torch.full_like(confidence, qp_value), torch.full_like(confidence, type_value)])
        feature = self.stem(torch.cat(inputs, dim=1))
        for block in self.blocks:
            feature = feature + block(feature)
        correction = self.head(F.silu(feature))
        if self.bounded_output:
            correction = torch.tanh(correction)
        correction = confidence * correction
        return z_prev + self.residual_scale * correction


class DMCAdapter(nn.Module):
    """DMC-based P-frame codec adapted for Wan VAE latents (48ch Wan2.2 / 16ch Wan2.1)."""

    def __init__(self, checkpoint_path: Optional[str] = None, device: str = "cuda:0"):
        latent_init_mode = "scratch"
        latent_stem_blocks = 4
        latent_spatial_mode = "dcvc_stride2"
        latent_preserve_blocks = 1
        latent_channels = 48
        bpp_spatial_factor = 32
        residual_mode = "raw"
        residual_predictor_hidden = 96
        residual_predictor_use_context = True
        residual_predictor_arch = "legacy"
        residual_predictor_blocks = 6
        explicit_residual_scale = 1.0
        refiner_hidden = 128
        refiner_use_context = True
        refiner_residual_scale = 0.1
        y_context_hidden = 192
        p_type_conditioning = True
        p_type_count = 3
        p_type_input_scale = 0.1
        p_type_quant_scale = 0.1
        p_entropy_scale_calibration = False
        p_entropy_scale_max_log = 1.0
        p_entropy_mean_calibration = False
        p_entropy_mean_max_delta = 0.5
        p_entropy_q_mode = "channel_scale"
        p_hard_rate_training = True
        p_channel_q_max_log = 2.5
        p_position_q_gain_enabled = False
        p_position_q_gain_max_log = 0.25
        p_base_allocator_mode = "off"
        p_base_allocator_position_count = 6
        p_base_allocator_rank = 4
        p_base_allocator_max_log = 0.25
        p_base_allocator_importance_path = None
        p_progressive_canvas_enabled = False
        p_progressive_canvas_group_sizes = None
        p_progressive_canvas_max_log = 0.2
        p_progressive_rate_bias_max_log = 0.2
        p_factorized_q_spatial_max_log = 0.25
        p_factorized_q_mix = 1.0
        p_adaptive_entropy_enabled = True
        p_adaptive_entropy_hidden = 256
        p_adaptive_entropy_max_log_scale = 1.5
        p_adaptive_entropy_max_mean_delta = 2.0
        p_decoded_dpb_mode = "off"
        p_decoded_dpb_hidden = 96
        p_decoded_dpb_blocks = 2
        p_decoded_dpb_max_feature_delta = 0.5
        p_decoded_dpb_max_log_scale = 0.5
        p_decoded_dpb_max_mean_delta = 0.5
        p_htcm_enabled = False
        p_htcm_hidden = 128
        p_htcm_blocks = 3
        p_htcm_max_delta = 0.5
        p_htcm_memory_mode = "correct"
        p1_context_adapter_enabled = False
        p1_context_adapter_hidden = 64
        p1_context_adapter_scale = 0.1
        p_residual_blend_init = 0.02
        p_residual_blend_max = 1.0
        p_residual_blend_typewise = True
        p_y_gain_enabled = False
        p_y_gain_max_log = 1.0
        p_y_balance_enabled = False
        p_y_balance_max_log = 0.5
        p_y_balance_max_bias = 0.5
        p_synthesis_head_enabled = True
        p_synthesis_head_hidden = 256
        p_synthesis_head_scale = 0.25
        p_synthesis_position_conditioned = True
        p_synthesis_position_count = 6
        p_synthesis_position_hidden = 128
        p_synthesis_qp_conditioned = False
        p_synthesis_qp_values = None
        p_synthesis_qp_hidden = 128
        p_synthesis_qp_position_factorized = False
        p_synthesis_head_arch = "legacy"
        p_continuous_display_blocks = 4
        p_continuous_display_max_position = 24
        p_continuous_display_reference_mode = "correct"
        p_z_context_enabled = False
        p_z_context_hidden = 192
        p_z_context_blocks = 4
        p_z_context_scale = 1.0
        p_frequency_compaction_enabled = False
        p_frequency_compaction_hidden = 160
        p_frequency_compaction_blocks = 3
        p_frequency_compaction_residual_scale = 0.25
        p_frequency_compaction_group_sizes = None
        p_frequency_compaction_mode = "residual"
        p_frequency_compaction_aux_enabled = False
        p_analysis_core_hidden = 0
        p_synthesis_core_hidden = 0
        p_core_blocks = 2
        p_core_residual_scale = 1.0
        super().__init__()
        from src.models.video_model import DMC

        self._device = device
        self.latent_channels = int(48)
        self.bpp_spatial_factor = int(32)
        self.latent_init_mode = _normalize_latent_init_mode("scratch")
        self.latent_spatial_mode = _normalize_latent_spatial_mode("dcvc_stride2")
        self.latent_preserve_blocks = max(1, int(1))
        self.residual_mode = str("raw").lower()
        self._entropy_coder_ready = False
        self._last_bpp_y = 0.0
        self._last_bpp_z = 0.0
        self._last_q_dec_mean = 0.0
        self._last_q_dec_gt4_fraction = 0.0
        self._last_q_dec_gt10_fraction = 0.0
        self._last_decoded_feature = None
        self._last_entropy_y_hat = None
        self._export_entropy_scales = False
        self._last_entropy_scale_map = None
        self._last_dpb_feature_delta_rms = 0.0
        self._last_dpb_prior_log_scale_rms = 0.0
        self._last_dpb_prior_mean_rms = 0.0
        self.p_type_conditioning = bool(True)
        self.p_type_count = max(1, int(3))
        self.p_type_input_scale = float(0.1)
        self.p_type_quant_scale = float(0.1)
        self.p_entropy_scale_calibration = bool(False)
        self.p_entropy_scale_max_log = float(1.0)
        self.p_entropy_mean_calibration = bool(False)
        self.p_entropy_mean_max_delta = float(0.5)
        self.p_entropy_q_mode = str("channel_scale").strip().lower()
        self.p_hard_rate_training = bool(True)
        self.p_channel_q_max_log = float(2.5)
        self.p_position_q_gain_enabled = bool(False)
        self.p_position_q_gain_max_log = float(0.25)
        self.p_base_allocator_mode = str("off").strip().lower()
        self.p_factorized_q_spatial_max_log = float(0.25)
        factorized_mix = float(1.0)
        if not 0.0 <= factorized_mix <= 1.0:
            raise ValueError("p_factorized_q_mix must be in [0, 1].")
        self.p_adaptive_entropy_enabled = bool(True)
        self.p_decoded_dpb_mode = str("off").strip().lower()
        self.p_htcm_enabled = bool(False)
        self._active_htcm_long_feature = None
        self.set_p_htcm_memory_mode("correct")
        self.p1_context_adapter_enabled = bool(False)
        self.p_residual_blend_max = max(float(1.0), 1e-06)
        self.p_residual_blend_typewise = bool(True)
        self.explicit_residual_scale = max(float(1.0), 1e-06)
        self.explicit_residual_scales = None
        self.p_y_gain_enabled = bool(False)
        self.p_y_gain_max_log = float(1.0)
        self.p_y_balance_enabled = bool(False)
        self.p_y_balance_max_log = float(0.5)
        self.p_y_balance_max_bias = float(0.5)
        self.p_synthesis_head_enabled = bool(True)
        self.p_synthesis_head_arch = str("legacy").lower().strip()
        if self.p_synthesis_head_arch not in {"legacy", "continuous_age_v1"}:
            raise ValueError("p_synthesis_head_arch must be legacy|continuous_age_v1")
        self._synthesis_position_idx = None
        self._synthesis_qp = None
        self.p_z_context_enabled = bool(False)
        self.p_frequency_compaction_enabled = bool(False)
        self.p_frequency_compaction_aux_enabled = bool(False)
        analysis_hidden = int(0)
        synthesis_hidden = int(0)
        core_blocks = int(2)
        core_scale = float(1.0)
        if analysis_hidden < 0 or synthesis_hidden < 0:
            raise ValueError("P transform capacity hidden widths must be non-negative")
        if (analysis_hidden or synthesis_hidden) and core_blocks <= 0:
            raise ValueError("p_core_blocks must be positive when a P core is enabled")
        if (analysis_hidden or synthesis_hidden) and (not 0.0 < core_scale <= 1.0):
            raise ValueError("p_core_residual_scale must be in (0, 1]")
        self.model = DMC()
        self.model.encoder.conv1 = LatentInputStem(self.latent_channels, _RT_D, num_blocks=int(4))
        self.model.encoder.fuse_conv1_flag = False
        from src.layers.layers import DepthConvBlock

        self.model.feature_adaptor_i = LatentInputStem(self.latent_channels, _RT_D, num_blocks=int(4))
        sd: dict[str, torch.Tensor] = {}
        _initialize_neutral_prior_output_(
            self.model.y_prior_fusion.conv[-1], q_channels=_RT_Y, scale_channels=_RT_Y, q_bias=1.0
        )
        _initialize_neutral_prior_output_(
            self.model.y_spatial_prior.conv[-1], q_channels=0, scale_channels=_RT_Y, q_bias=0.0
        )
        self.model.to(device)
        self.model.eval()
        old_head = self.model.recon_generation_net.head
        new_head = nn.Conv2d(_RT_RECON, self.latent_channels, 1).to(device)
        self.model.recon_generation_net.head = new_head
        self.residual_predictor: Optional[LatentResidualPredictor]
        self.residual_predictor = None
        self.residual_refiner: Optional[LatentPRefiner]
        self.residual_refiner = None
        self.y_context_predictor: Optional[LatentYContextPredictor]
        self.y_context_predictor = None
        self.qe_log_inter = nn.Parameter(torch.zeros(_INTER_QP_NUM, _RT_D, 1, 1).to(device))
        self.qd_log_inter = nn.Parameter(torch.zeros(_INTER_QP_NUM, _RT_D, 1, 1).to(device))
        self.qf_log_inter = nn.Parameter(torch.zeros(_INTER_QP_NUM, _RT_D, 1, 1).to(device))
        self.qr_log_inter = nn.Parameter(torch.zeros(_INTER_QP_NUM, _RT_RECON, 1, 1).to(device))
        with torch.no_grad():
            if all((k in sd for k in ("q_encoder", "q_decoder", "q_feature", "q_recon"))):
                self.qe_log_inter.data.copy_(torch.log(sd["q_encoder"].clamp(min=0.001)).to(device))
                self.qd_log_inter.data.copy_(torch.log(sd["q_decoder"].clamp(min=0.001)).to(device))
                self.qf_log_inter.data.copy_(torch.log(sd["q_feature"].clamp(min=0.001)).to(device))
                self.qr_log_inter.data.copy_(torch.log(sd["q_recon"].clamp(min=0.001)).to(device))
        self.p_type_input_bias = nn.Parameter(torch.zeros(self.p_type_count, self.latent_channels, 1, 1, device=device))
        self.p_type_input_gain = nn.Parameter(torch.zeros(self.p_type_count, self.latent_channels, 1, 1, device=device))
        self.p_type_qe_delta = nn.Parameter(torch.zeros(self.p_type_count, _RT_D, 1, 1, device=device))
        self.p_type_qd_delta = nn.Parameter(torch.zeros(self.p_type_count, _RT_D, 1, 1, device=device))
        self.p_type_qf_delta = nn.Parameter(torch.zeros(self.p_type_count, _RT_D, 1, 1, device=device))
        self.p_type_qr_delta = nn.Parameter(torch.zeros(self.p_type_count, _RT_RECON, 1, 1, device=device))
        self.p_entropy_log_scale_delta = None
        self.p_entropy_mean_delta = None
        self.p_channel_q_logit = None
        self.p_channel_q_logit = nn.Parameter(torch.zeros(self.p_type_count, _INTER_QP_NUM, _RT_Y, 1, 1, device=device))
        self.p_position_q_gain_logit = None
        self.p_base_allocator = None
        self.p_q_base_log = None
        self.p_adaptive_entropy = None
        self.p_adaptive_entropy = PAdaptiveEntropyPrior(
            context_channels=_RT_D,
            y_channels=_RT_Y,
            hidden_channels=int(256),
            p_type_count=self.p_type_count,
            max_log_scale=float(1.5),
            max_mean_delta=float(2.0),
        ).to(device)
        self.p_decoded_feature_fusion = None
        self.p_decoded_y_prior = None
        self.p_decoded_y_scale_only = False
        self.p_decoded_y_mean_max_qp = _INTER_QP_NUM - 1
        self.p_decoded_y_current_pass = False
        self.p_htcm_long_fusion = None
        self.p1_context_adapter = None
        self.p_residual_blend_logit = None
        self.p_y_gain_logit = None
        self.p_y_balance_gain_logit = None
        self.p_y_balance_bias = None
        self.p_frequency_compaction = None
        self._last_p_frequency_aux: dict[str, torch.Tensor] = {}
        self.p_analysis_core = None
        if analysis_hidden > 0:
            self.p_analysis_core = PTransformCapacityCore(
                channels=_RT_Y,
                hidden_channels=analysis_hidden,
                blocks=core_blocks,
                residual_scale=core_scale,
                role="analysis",
                num_qp=_INTER_QP_NUM,
            ).to(device)
        self.p_synthesis_core = None
        if synthesis_hidden > 0:
            self.p_synthesis_core = PTransformCapacityCore(
                channels=_RT_Y,
                hidden_channels=synthesis_hidden,
                blocks=core_blocks,
                residual_scale=core_scale,
                role="synthesis",
                num_qp=_INTER_QP_NUM,
            ).to(device)
        self._last_y_symbol_nonzero_fraction = 0.0
        self._last_entropy_symbol_tensors: dict[str, Any] = {}
        self._capture_exact_entropy_symbols = False
        self._last_entropy_symbol_tensors = {}
        self.p_synthesis_head = None
        if self.p_synthesis_head_enabled:
            if self.p_synthesis_head_arch == "continuous_age_v1":
                self.p_synthesis_head = PContinuousDisplaySynthesisHead(
                    latent_channels=self.latent_channels,
                    hidden_channels=int(256),
                    blocks=int(4),
                    residual_scale=float(0.25),
                    max_position=int(24),
                    reference_mode=str("correct"),
                ).to(device)
            else:
                self.p_synthesis_head = PResidualSynthesisHead(
                    latent_channels=self.latent_channels,
                    hidden_channels=int(256),
                    residual_scale=float(0.25),
                    position_conditioned=bool(True),
                    position_count=int(6),
                    position_hidden_channels=int(128),
                    qp_conditioned=bool(False),
                    qp_values=None,
                    qp_hidden_channels=int(128),
                    qp_position_factorized=bool(False),
                ).to(device)
        self.p_group_context = None
        self.p_second_pass_spatial_refiner = None
        self.p_context_pyramid_analysis = None
        self.p_context_pyramid_synthesis = None
        self.p_invertible_latent_lifting = None
        self.p_reference_translation_lifting = None
        self.p_shared_query_motion_residual = None
        self.p_y_conditioned_context_warp = None
        self.p_reference_affine_lifting = None
        self._p_reference_affine_lifting_mode = "correct"
        self.p_context_first_core = None
        self._p_context_first_mode = "correct"
        self.p_shared_latent_lifting_alpha = None
        self.p_shared_latent_lifting_max_alpha = 1.0
        self._p_shared_latent_lifting_mode = "correct"
        self._p_shared_latent_lifting_cache = None
        self.p_mature_context_coupling = None
        self._p_mature_context_coupling_mode = "correct"
        self.p_context_native_deep_transform = None
        self._p_context_native_deep_mode = "correct"
        self.p_recurrent_synthesis_core = None
        self._p_recurrent_synthesis_state = None
        self._p_recurrent_synthesis_mode = "correct"
        self.p_local_reference_transform_analysis = None
        self.p_local_reference_transform_synthesis = None
        self._p_local_reference_mode = "correct"
        self.p_z_context_predictor = None
        self.p_motion_relation_bit_estimator = None
        self.p_motion_context_scale = None
        self.p_motion_context_adapter = None
        self.p_motion_relation_flow_step = 0.25
        self.p_motion_relation_confidence_step = 0.125
        self.p_motion_relation_cache_size = 8
        self._last_bpp_motion = 0.0
        self._last_motion_reference_delta_rms = 0.0
        try:
            self.model.update(force_zero_thres=None)
            self._entropy_coder_ready = True
        except ImportError:
            pass

    def enable_p_group_context(
        self,
        *,
        hidden_channels: int = 96,
        max_log_scale: float = 0.75,
        symbol_center: bool = False,
        mean_scale: bool = False,
        max_symbol_center: float = 4.0,
        permutation: Optional[list[int]] = None,
    ) -> None:
        if self.p_group_context is not None:
            raise RuntimeError("P group context is already enabled.")
        if symbol_center and mean_scale:
            raise ValueError("P group context symbol-center and mean-scale modes are alternatives")
        self.p_group_context = IGroupEntropyContext(
            channels=_RT_Y,
            lane="even4-permuted" if permutation is not None else "even4",
            parameterization="scale_symbol_center" if symbol_center else "mean_scale" if mean_scale else "scale",
            hidden_channels=int(hidden_channels),
            max_log_scale=float(max_log_scale),
            max_mean_delta=float(max_symbol_center),
            spatial_passes=2,
            permutation=permutation,
        ).to(next(self.parameters()).device)

    def enable_p_second_pass_spatial_refiner(
        self, *, hidden_channels: int = 128, blocks: int = 2, max_log_scale: float = 0.75
    ) -> None:
        if self.p_second_pass_spatial_refiner is not None:
            raise RuntimeError("P second-pass spatial refiner is already enabled")
        if self.p_group_context is not None:
            raise RuntimeError("P channel-group context and spatial refiner are alternative priors")
        self.p_second_pass_spatial_refiner = PSecondPassSpatialScaleRefiner(
            y_channels=_RT_Y,
            hidden_channels=int(hidden_channels),
            blocks=int(blocks),
            p_type_count=self.p_type_count,
            max_log_scale=float(max_log_scale),
        ).to(next(self.parameters()).device)
        self._entropy_coder_ready = False

    def enable_p_context_pyramid_analysis(
        self,
        *,
        hidden_channels: int = 192,
        blocks: int = 3,
        residual_scale: float = 0.25,
        affine_modulation: bool = False,
        affine_max_log_scale: float = 0.15,
        content_gate: bool = False,
    ) -> None:
        if self.p_context_pyramid_analysis is not None:
            raise RuntimeError("P context-pyramid analysis is already enabled.")
        self.p_context_pyramid_analysis = PContextPyramidAnalysis(
            feature_channels=_RT_D,
            context_channels=_RT_D,
            hidden_channels=int(hidden_channels),
            blocks=int(blocks),
            residual_scale=float(residual_scale),
            qp_count=_INTER_QP_NUM,
            position_count=self.p_type_count,
            affine_modulation=bool(affine_modulation),
            affine_max_log_scale=float(affine_max_log_scale),
            content_gate=bool(content_gate),
        ).to(next(self.parameters()).device)

    def enable_p_context_pyramid_synthesis(
        self,
        *,
        hidden_channels: int = 192,
        blocks: int = 3,
        residual_scale: float = 0.15,
        affine_max_log_scale: float = 0.15,
    ) -> None:
        if self.p_context_pyramid_synthesis is not None:
            raise RuntimeError("P context-pyramid synthesis is already enabled.")
        self.p_context_pyramid_synthesis = PContextPyramidSynthesis(
            feature_channels=_RT_D,
            context_channels=_RT_D,
            hidden_channels=int(hidden_channels),
            blocks=int(blocks),
            residual_scale=float(residual_scale),
            qp_count=_INTER_QP_NUM,
            position_count=self.p_type_count,
            affine_modulation=True,
            affine_max_log_scale=float(affine_max_log_scale),
        ).to(next(self.parameters()).device)

    def enable_p_reference_affine_lifting(
        self, *, hidden_channels: int = 160, blocks: int = 3, max_log_scale: float = 0.2, max_mean: float = 0.35
    ) -> None:
        if self.p_reference_affine_lifting is not None:
            raise RuntimeError("P reference affine lifting is already enabled.")
        if (
            self.p_context_pyramid_analysis is not None
            or self.p_context_pyramid_synthesis is not None
            or self.p_local_reference_transform_analysis is not None
            or (self.p_local_reference_transform_synthesis is not None)
        ):
            raise RuntimeError(
                "P reference affine lifting is an alternative main transform; context-pyramid/local-reference transforms must be disabled."
            )
        self.p_reference_affine_lifting = PReferenceAffineLifting(
            channels=_RT_D,
            context_channels=_RT_D,
            hidden_channels=int(hidden_channels),
            blocks=int(blocks),
            qp_count=_INTER_QP_NUM,
            position_count=self.p_type_count,
            max_log_scale=float(max_log_scale),
            max_mean=float(max_mean),
        ).to(next(self.parameters()).device)
        self._p_reference_affine_lifting_mode = "correct"
        self._entropy_coder_ready = False

    def enable_p_invertible_latent_lifting(
        self,
        *,
        hidden_channels: int = 96,
        update_scale: float = 0.25,
        perceptual_channel_weights: Optional[Sequence[float]] = None,
        companding_strength: float = 0.0,
        companding_max_log_scale: float = 0.75,
    ) -> None:
        if self.p_invertible_latent_lifting is not None:
            raise RuntimeError("P invertible latent lifting is already enabled")
        self.p_invertible_latent_lifting = PInvertibleLatentLifting(
            latent_channels=self.latent_channels,
            hidden_channels=int(hidden_channels),
            qp_count=_INTER_QP_NUM,
            position_count=self.p_type_count,
            update_scale=float(update_scale),
            perceptual_channel_weights=perceptual_channel_weights,
            companding_strength=float(companding_strength),
            companding_max_log_scale=float(companding_max_log_scale),
        ).to(next(self.parameters()).device)
        self._entropy_coder_ready = False

    def enable_p_reference_translation_lifting(
        self,
        *,
        hidden_channels: int = 96,
        max_alpha: float = 1.0,
        predictor_scale: float = 0.25,
        base_alpha: float = 0.0,
    ) -> None:
        if self.p_reference_translation_lifting is not None:
            raise RuntimeError("P reference translation lifting is already enabled")
        if self.p_invertible_latent_lifting is not None:
            raise RuntimeError("P reference translation and channel lifting are alternative coordinates")
        self.p_reference_translation_lifting = PReferenceTranslationLifting(
            latent_channels=self.latent_channels,
            hidden_channels=int(hidden_channels),
            qp_count=_INTER_QP_NUM,
            position_count=self.p_type_count,
            max_alpha=float(max_alpha),
            predictor_scale=float(predictor_scale),
            base_alpha=float(base_alpha),
        ).to(next(self.parameters()).device)
        self._entropy_coder_ready = False

    def enable_p_shared_query_motion_residual(
        self,
        *,
        hidden_channels: int = 96,
        blocks: int = 3,
        max_displacement: float = 2.0,
        max_alpha: float = 0.5,
        highpass_kernel: int = 3,
        reliability_temperature: float = 0.0,
        channel_indices: Optional[Sequence[int]] = None,
    ) -> None:
        if self.p_shared_query_motion_residual is not None:
            raise RuntimeError("P shared-query motion residual is already enabled")
        if self.p_reference_translation_lifting is not None:
            raise RuntimeError("P reference translation and shared-query motion residual are alternatives")
        self.p_shared_query_motion_residual = PSharedQueryMotionResidual(
            latent_channels=self.latent_channels,
            hidden_channels=int(hidden_channels),
            blocks=int(blocks),
            qp_count=_INTER_QP_NUM,
            position_count=self.p_type_count,
            max_displacement=float(max_displacement),
            max_alpha=float(max_alpha),
            highpass_kernel=int(highpass_kernel),
            reliability_temperature=float(reliability_temperature),
            channel_indices=channel_indices,
        ).to(next(self.parameters()).device)
        self._entropy_coder_ready = False

    def enable_p_y_conditioned_context_warp(
        self,
        *,
        hidden_channels: int = 128,
        blocks: int = 3,
        max_displacement: float = 4.0,
        warp_mode: str = "full",
        lowpass_kernel: int = 3,
        analysis_enabled: bool = False,
        base_y_channels: int = 0,
    ) -> None:
        if self.p_y_conditioned_context_warp is not None:
            raise RuntimeError("P y-conditioned context warp is already enabled")
        if self.p_local_reference_transform_synthesis is not None:
            raise RuntimeError("P y-conditioned warp and local-reference synthesis are alternatives")
        self.p_y_conditioned_context_warp = PYConditionedContextWarp(
            channels=_RT_D,
            hidden_channels=int(hidden_channels),
            blocks=int(blocks),
            qp_count=_INTER_QP_NUM,
            position_count=self.p_type_count,
            max_displacement=float(max_displacement),
            warp_mode=str(warp_mode),
            lowpass_kernel=int(lowpass_kernel),
            analysis_enabled=bool(analysis_enabled),
            base_y_channels=int(base_y_channels),
        ).to(next(self.parameters()).device)
        self._entropy_coder_ready = False

    def enable_p_context_first_core(
        self, *, channels: int = _RT_D, blocks: int = 4, version: str = "context_v1"
    ) -> None:
        if self.p_context_first_core is not None:
            raise RuntimeError("P context-first core is already enabled.")
        if int(channels) != _RT_D:
            raise ValueError(
                f"context-first core currently requires channels={_RT_D} to preserve the reconstruction/temporal-prior ABI"
            )
        incompatible = (
            self.p_context_pyramid_analysis,
            self.p_context_pyramid_synthesis,
            self.p_reference_affine_lifting,
            self.p_local_reference_transform_analysis,
            self.p_local_reference_transform_synthesis,
        )
        if any((module is not None for module in incompatible)):
            raise RuntimeError(
                "P context-first core replaces the main transform and cannot be combined with legacy transform adapters."
            )
        version = str(version).lower().strip()
        core_class: type[PContextFirstResidualCore]
        if version == "context_v1":
            core_class = PContextFirstResidualCore
        elif version == "shared_latent_v2":
            core_class = PContextFirstSharedLatentCore
        elif version == "innovation_v3":
            core_class = PContextFirstInnovationCoreV3
        elif version == "latent_residual_v4":
            core_class = PContextFirstLatentResidualCoreV4
        else:
            raise ValueError(
                f"context-first core version must be context_v1, shared_latent_v2, innovation_v3, or latent_residual_v4, got {version!r}"
            )
        self.p_context_first_core = core_class(
            latent_channels=self.latent_channels,
            channels=int(channels),
            y_channels=_RT_Y,
            blocks=int(blocks),
            qp_count=_INTER_QP_NUM,
            position_count=self.p_type_count,
        ).to(next(self.parameters()).device)
        self._p_context_first_mode = "correct"
        self._entropy_coder_ready = False

    def enable_p_shared_latent_lifting(self, *, max_alpha: float = 1.0) -> None:
        if self.p_shared_latent_lifting_alpha is not None:
            raise RuntimeError("P shared-latent lifting is already enabled.")
        if max_alpha <= 0:
            raise ValueError("P shared-latent lifting max alpha must be positive")
        if self.p_context_first_core is not None or any(
            (
                module is not None
                for module in (
                    self.p_context_pyramid_analysis,
                    self.p_context_pyramid_synthesis,
                    self.p_reference_affine_lifting,
                    self.p_local_reference_transform_analysis,
                    self.p_local_reference_transform_synthesis,
                )
            )
        ):
            raise RuntimeError("P shared-latent lifting is an alternative main transform")
        self.p_shared_latent_lifting_alpha = nn.Parameter(
            torch.zeros(self.p_type_count, _INTER_QP_NUM, 1, 1, 1, device=next(self.parameters()).device)
        )
        self.p_shared_latent_lifting_max_alpha = float(max_alpha)
        self._p_shared_latent_lifting_mode = "correct"
        self._entropy_coder_ready = False

    def enable_p_mature_context_coupling(self, *, blocks: int = 4, residual_scale: float = 0.5) -> None:
        if self.p_mature_context_coupling is not None:
            raise RuntimeError("P mature context coupling is already enabled")
        if int(blocks) <= 0 or float(residual_scale) <= 0:
            raise ValueError("P mature context coupling blocks/scale must be positive")
        if (
            self.p_context_first_core is not None
            or any(
                (
                    module is not None
                    for module in (
                        self.p_context_pyramid_analysis,
                        self.p_context_pyramid_synthesis,
                        self.p_reference_affine_lifting,
                        self.p_local_reference_transform_analysis,
                        self.p_local_reference_transform_synthesis,
                    )
                )
            )
            or self.p_shared_latent_lifting_alpha is not None
        ):
            raise RuntimeError("P mature context coupling is an alternative main transform")
        self.p_mature_context_coupling = PMatureContextFirstCoupling(
            channels=_RT_D,
            blocks=int(blocks),
            qp_count=_INTER_QP_NUM,
            position_count=self.p_type_count,
            residual_scale=float(residual_scale),
        ).to(next(self.parameters()).device)
        self._p_mature_context_coupling_mode = "correct"
        self._entropy_coder_ready = False

    def enable_p_context_native_deep_transform(
        self, *, hidden_channels: int = 128, blocks: int = 2, residual_scale: float = 0.25
    ) -> None:
        if self.p_context_native_deep_transform is not None:
            raise RuntimeError("P context-native deep transform is already enabled")
        if (
            self.p_context_first_core is not None
            or any(
                (
                    module is not None
                    for module in (
                        self.p_mature_context_coupling,
                        self.p_context_pyramid_analysis,
                        self.p_context_pyramid_synthesis,
                        self.p_reference_affine_lifting,
                        self.p_local_reference_transform_analysis,
                        self.p_local_reference_transform_synthesis,
                    )
                )
            )
            or self.p_shared_latent_lifting_alpha is not None
        ):
            raise RuntimeError("P context-native deep transform is an alternative main transform")
        self.p_context_native_deep_transform = PContextNativeDeepTransform(
            hidden_channels=int(hidden_channels),
            blocks=int(blocks),
            qp_count=_INTER_QP_NUM,
            position_count=self.p_type_count,
            residual_scale=float(residual_scale),
        ).to(next(self.parameters()).device)
        self._p_context_native_deep_mode = "correct"
        self._entropy_coder_ready = False

    def enable_p_continuous_display_synthesis(
        self,
        *,
        hidden_channels: int = 192,
        blocks: int = 4,
        residual_scale: float = 0.25,
        max_position: int = 24,
        reference_mode: str = "correct",
    ) -> None:
        """Add a continuous-age correction on the restored mature display head."""
        if isinstance(self.p_synthesis_head, PContinuousDisplaySynthesisHead):
            raise RuntimeError("continuous display synthesis is already enabled")
        mature_display_head = self.p_synthesis_head
        self.p_synthesis_head = PContinuousDisplaySynthesisHead(
            latent_channels=self.latent_channels,
            hidden_channels=int(hidden_channels),
            blocks=int(blocks),
            residual_scale=float(residual_scale),
            max_position=int(max_position),
            reference_mode=str(reference_mode),
            base_head=mature_display_head,
        ).to(next(self.parameters()).device)
        self.p_synthesis_head_enabled = True
        self.p_synthesis_head_arch = "continuous_age_v1"

    def enable_p_recurrent_synthesis_core(
        self,
        *,
        state_channels: int = 96,
        blocks: int = 3,
        residual_scale: float = 0.2,
        max_position: int = 24,
        mode: str = "correct",
    ) -> None:
        if self.p_recurrent_synthesis_core is not None:
            raise RuntimeError("P recurrent synthesis core is already enabled")
        self.p_recurrent_synthesis_core = PRecurrentSynthesisCore(
            feature_channels=_RT_RECON,
            state_channels=int(state_channels),
            blocks=int(blocks),
            residual_scale=float(residual_scale),
            max_position=int(max_position),
            num_qp=_INTER_QP_NUM,
        ).to(next(self.parameters()).device)
        self.set_p_recurrent_synthesis_mode(mode)
        self._p_recurrent_synthesis_state = None

    def reset_p_recurrent_synthesis_state(self) -> None:
        self._p_recurrent_synthesis_state = None

    def set_p_recurrent_synthesis_mode(self, mode: str) -> None:
        mode = str(mode).lower().strip()
        if mode not in {"correct", "zero", "wrong"}:
            raise ValueError("p_recurrent_synthesis_mode must be correct, zero, or wrong")
        self._p_recurrent_synthesis_mode = mode

    def _p_recurrent_synthesis_contract_payload(self) -> Optional[dict[str, Any]]:
        module = self.p_recurrent_synthesis_core
        if module is None:
            return None
        return {**module.contract_payload(), "mode": self._p_recurrent_synthesis_mode}

    def _validate_p_recurrent_synthesis_contract(self, payload: Any) -> None:
        expected = self._p_recurrent_synthesis_contract_payload()
        if payload != expected:
            raise ValueError(f"P recurrent synthesis contract mismatch: stream={payload} decoder={expected}")

    def set_p_mature_context_coupling_mode(self, mode: str) -> None:
        mode = str(mode).lower().strip()
        if mode not in {"correct", "zero", "wrong"}:
            raise ValueError("p_mature_context_coupling_mode must be correct, zero, or wrong")
        self._p_mature_context_coupling_mode = mode

    def _p_mature_context_coupling_contract_payload(self) -> Optional[dict[str, Any]]:
        module = self.p_mature_context_coupling
        return None if module is None else module.contract_payload()

    def _validate_p_mature_context_coupling_contract(self, payload: Any) -> None:
        expected = self._p_mature_context_coupling_contract_payload()
        if payload != expected:
            raise ValueError(f"P mature context coupling contract mismatch: stream={payload} decoder={expected}")

    def set_p_context_native_deep_mode(self, mode: str) -> None:
        mode = str(mode).lower().strip()
        if mode not in {"correct", "zero", "wrong"}:
            raise ValueError("p_context_native_deep_mode must be correct, zero, or wrong")
        self._p_context_native_deep_mode = mode

    def _p_context_native_deep_contract_payload(self) -> Optional[dict[str, Any]]:
        module = self.p_context_native_deep_transform
        return None if module is None else module.contract_payload()

    def _validate_p_context_native_deep_contract(self, payload: Any) -> None:
        expected = self._p_context_native_deep_contract_payload()
        if payload != expected:
            raise ValueError(f"P context-native deep contract mismatch: stream={payload} decoder={expected}")

    def _p_context_native_reference(
        self, reference_latent: Optional[torch.Tensor], context: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if reference_latent is None:
            raise ValueError("P context-native transform requires immediate hard reference latent")
        mode = self._p_context_native_deep_mode
        if mode == "correct":
            reference = reference_latent
            branch_context = context
        elif mode == "zero":
            reference = torch.zeros_like(reference_latent)
            branch_context = torch.zeros_like(context)
        else:
            shifts = (max(1, reference_latent.shape[-2] // 2), max(1, reference_latent.shape[-1] // 2))
            reference = torch.roll(reference_latent, shifts=shifts, dims=(-2, -1))
            branch_context = torch.roll(context, shifts=shifts, dims=(-2, -1))
        return (self.model.encoder.conv1(reference), branch_context)

    def _p_mature_context_reference(
        self, reference_latent: Optional[torch.Tensor], context: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if reference_latent is None:
            raise ValueError("P mature context coupling requires immediate hard reference latent")
        mode = self._p_mature_context_coupling_mode
        if mode == "correct":
            reference = reference_latent
            branch_context = context
        elif mode == "zero":
            reference = torch.zeros_like(reference_latent)
            branch_context = torch.zeros_like(context)
        else:
            shifts = (max(1, reference_latent.shape[-2] // 2), max(1, reference_latent.shape[-1] // 2))
            reference = torch.roll(reference_latent, shifts=shifts, dims=(-2, -1))
            branch_context = torch.roll(context, shifts=shifts, dims=(-2, -1))
        return (self.model.encoder.conv1(reference), branch_context)

    def set_p_shared_latent_lifting_mode(self, mode: str) -> None:
        mode = str(mode).lower().strip()
        if mode not in {"correct", "zero", "wrong"}:
            raise ValueError("p_shared_latent_lifting_mode must be correct, zero, or wrong")
        self._p_shared_latent_lifting_mode = mode

    def _p_shared_latent_lifting_contract_payload(self) -> Optional[dict[str, Any]]:
        if self.p_shared_latent_lifting_alpha is None:
            return None
        return {
            "schema": "p_shared_latent_lifting_v1",
            "coordinate": "shared_parent_encoder_conv1",
            "gate": "qp_position_bounded_tanh",
            "max_alpha": self.p_shared_latent_lifting_max_alpha,
            "initial_state": "exact_parent_alpha_zero",
            "payload": "none",
        }

    def _validate_p_shared_latent_lifting_contract(self, payload: Any) -> None:
        expected = self._p_shared_latent_lifting_contract_payload()
        if payload != expected:
            raise ValueError(f"P shared-latent lifting contract mismatch: stream={payload} decoder={expected}")

    def _shared_latent_reference_feature(
        self, reference_latent: Optional[torch.Tensor], qp: int, frame_type_idx: int, *, allow_cache: bool
    ) -> torch.Tensor:
        alpha_table = self.p_shared_latent_lifting_alpha
        if alpha_table is None:
            raise RuntimeError("P shared-latent lifting is disabled")
        if reference_latent is None:
            raise ValueError("P shared-latent lifting requires immediate hard reference latent")
        mode = self._p_shared_latent_lifting_mode
        key = (id(reference_latent), int(qp), int(frame_type_idx), mode)
        cached = self._p_shared_latent_lifting_cache
        if allow_cache and cached is not None and (cached[0] == key):
            self._p_shared_latent_lifting_cache = None
            return cached[1]
        if mode == "correct":
            reference = reference_latent
        elif mode == "zero":
            reference = torch.zeros_like(reference_latent)
        else:
            reference = torch.roll(
                reference_latent,
                shifts=(max(1, reference_latent.shape[-2] // 2), max(1, reference_latent.shape[-1] // 2)),
                dims=(-2, -1),
            )
        feature = self.model.encoder.conv1(reference)
        type_index = self._type_index(frame_type_idx)
        qp_index = max(0, min(int(qp), _INTER_QP_NUM - 1))
        alpha = (torch.tanh(alpha_table[type_index, qp_index]) * self.p_shared_latent_lifting_max_alpha).to(
            device=feature.device, dtype=feature.dtype
        )
        weighted = feature * alpha
        if not allow_cache:
            self._p_shared_latent_lifting_cache = (key, weighted)
        return weighted

    def set_p_context_first_mode(self, mode: str) -> None:
        mode = str(mode).lower().strip()
        if mode not in {"correct", "zero", "wrong"}:
            raise ValueError(f"p_context_first_mode must be correct, zero, or wrong, got {mode!r}")
        self._p_context_first_mode = mode

    def _p_context_first_contract_payload(self) -> Optional[dict[str, Any]]:
        module = self.p_context_first_core
        return None if module is None else module.contract_payload()

    def _validate_p_context_first_contract(self, payload: Any) -> None:
        expected = self._p_context_first_contract_payload()
        if payload != expected:
            raise ValueError(
                f"P context-first core contract mismatch between bitstream and decoder: stream={payload} decoder={expected}"
            )

    def _p_invertible_latent_lifting_contract_payload(self) -> Optional[dict[str, Any]]:
        module = self.p_invertible_latent_lifting
        return None if module is None else module.contract_payload()

    def _validate_p_invertible_latent_lifting_contract(self, payload: Any) -> None:
        expected = self._p_invertible_latent_lifting_contract_payload()
        if payload != expected:
            raise ValueError(
                f"P invertible latent lifting contract mismatch between bitstream and decoder: stream={payload} decoder={expected}"
            )

    def _p_reference_translation_lifting_contract_payload(self) -> Optional[dict[str, Any]]:
        module = self.p_reference_translation_lifting
        return None if module is None else module.contract_payload()

    def _validate_p_reference_translation_lifting_contract(self, payload: Any) -> None:
        expected = self._p_reference_translation_lifting_contract_payload()
        if payload != expected:
            raise ValueError(
                f"P reference translation lifting contract mismatch between bitstream and decoder: stream={payload} decoder={expected}"
            )

    def _reference_translation_analysis(
        self, x: torch.Tensor, reference: Optional[torch.Tensor], qp: int, frame_type_idx: int
    ) -> torch.Tensor:
        module = self.p_reference_translation_lifting
        if module is None:
            return x
        if reference is None:
            raise ValueError("P reference translation requires a hard reference")
        return module.analysis(x, reference, qp, frame_type_idx)

    def _reference_translation_synthesis(
        self, x: torch.Tensor, reference: Optional[torch.Tensor], qp: int, frame_type_idx: int
    ) -> torch.Tensor:
        module = self.p_reference_translation_lifting
        if module is None:
            return x
        if reference is None:
            raise ValueError("P reference translation requires a hard reference")
        return module.synthesis(x, reference, qp, frame_type_idx)

    def _shared_query_motion_analysis(
        self, x: torch.Tensor, reference: Optional[torch.Tensor], qp: int, frame_type_idx: int
    ) -> torch.Tensor:
        module = self.p_shared_query_motion_residual
        if module is None:
            return x
        if reference is None:
            raise ValueError("P shared-query motion residual requires a hard reference")
        return module.analysis(x, reference, qp, frame_type_idx)

    def _shared_query_motion_synthesis(
        self, x: torch.Tensor, reference: Optional[torch.Tensor], qp: int, frame_type_idx: int
    ) -> torch.Tensor:
        module = self.p_shared_query_motion_residual
        if module is None:
            return x
        if reference is None:
            raise ValueError("P shared-query motion residual requires a hard reference")
        return module.synthesis(x, reference, qp, frame_type_idx)

    def _p_shared_query_motion_contract_payload(self) -> Optional[dict[str, Any]]:
        module = self.p_shared_query_motion_residual
        return None if module is None else module.contract_payload()

    def _validate_p_shared_query_motion_contract(self, payload: Any) -> None:
        expected = self._p_shared_query_motion_contract_payload()
        if payload != expected:
            raise ValueError(
                f"P shared-query motion-residual contract mismatch: stream={payload!r} decoder={expected!r}"
            )

    def _p_y_conditioned_context_warp_contract_payload(self) -> Optional[dict[str, Any]]:
        module = self.p_y_conditioned_context_warp
        return None if module is None else module.contract_payload()

    def _validate_p_y_conditioned_context_warp_contract(self, payload: Any) -> None:
        expected = self._p_y_conditioned_context_warp_contract_payload()
        if payload != expected:
            raise ValueError(
                f"P y-conditioned context warp contract mismatch between bitstream and decoder: stream={payload} decoder={expected}"
            )

    def _latent_lifting_analysis(
        self, x: torch.Tensor, qp: int, frame_type_idx: int, reference: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        module = self.p_invertible_latent_lifting
        if module is None:
            return x
        return module.analysis(x, qp, frame_type_idx, reference=reference)

    def _latent_lifting_synthesis(
        self, x: torch.Tensor, qp: int, frame_type_idx: int, reference: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        module = self.p_invertible_latent_lifting
        if module is None:
            return x
        return module.synthesis(x, qp, frame_type_idx, reference=reference)

    def set_p_reference_affine_lifting_mode(self, mode: str) -> None:
        mode = str(mode).lower().strip()
        if mode not in {"correct", "zero", "wrong"}:
            raise ValueError(f"p_reference_affine_lifting_mode must be correct, zero, or wrong, got {mode!r}")
        self._p_reference_affine_lifting_mode = mode

    def _p_reference_affine_lifting_contract_payload(self) -> Optional[dict[str, Any]]:
        module = self.p_reference_affine_lifting
        return None if module is None else module.contract_payload()

    def _validate_p_reference_affine_lifting_contract(self, payload: Any) -> None:
        expected = self._p_reference_affine_lifting_contract_payload()
        if payload != expected:
            raise ValueError(
                f"P reference affine-lifting contract mismatch between bitstream and decoder: stream={payload} decoder={expected}"
            )

    def enable_p_local_reference_transform(
        self,
        *,
        query_key_channels: int = 64,
        value_channels: int = 128,
        num_heads: int = 4,
        window: int = 5,
        residual_scale: float = 0.25,
        qp_gate_max_log_gain: float = 0.75,
        qp_gate_max_bias: float = 0.25,
    ) -> None:
        """Enable the Task493 decoder-closed local-reference main-fusion transform."""
        if self.p_local_reference_transform_analysis is not None:
            raise RuntimeError("P local-reference transform analysis is already enabled.")
        if self.p_local_reference_transform_synthesis is not None:
            raise RuntimeError("P local-reference transform synthesis is already enabled.")
        device = next(self.parameters()).device
        common = {
            "feature_channels": _RT_D,
            "context_channels": _RT_D,
            "query_key_channels": int(query_key_channels),
            "value_channels": int(value_channels),
            "num_heads": int(num_heads),
            "window": int(window),
            "qp_count": _INTER_QP_NUM,
            "position_count": self.p_type_count,
            "residual_scale": float(residual_scale),
            "qp_gate_max_log_gain": float(qp_gate_max_log_gain),
            "qp_gate_max_bias": float(qp_gate_max_bias),
        }
        self.p_local_reference_transform_analysis = PLocalReferenceTransform(
            out_channels=_RT_D, residual_target="context", **common
        ).to(device)
        self.p_local_reference_transform_synthesis = PLocalReferenceTransform(
            out_channels=_RT_D, residual_target="context", **common
        ).to(device)
        self._p_local_reference_mode = "correct"
        self._entropy_coder_ready = False

    def set_p_local_reference_mode(self, mode: str) -> None:
        """Switch matcher input (correct / zero / evaluation-only wrong)."""
        mode = str(mode).lower().strip()
        if mode not in {"correct", "zero", "wrong"}:
            raise ValueError(f"p_local_reference_mode must be correct, zero, or wrong, got {mode!r}")
        self._p_local_reference_mode = mode

    def _p_local_reference_transform_contract_payload(self) -> Optional[dict[str, Any]]:
        analysis = self.p_local_reference_transform_analysis
        synthesis = self.p_local_reference_transform_synthesis
        if analysis is None or synthesis is None:
            return None
        return {
            "schema": "p_local_reference_transform_pair_v1",
            "analysis": analysis.contract_payload(),
            "synthesis": synthesis.contract_payload(),
        }

    def _validate_p_local_reference_transform_contract(self, payload: Any) -> None:
        expected = self._p_local_reference_transform_contract_payload()
        if expected is None:
            if payload is not None:
                raise ValueError(
                    "P bitstream carries a local-reference transform contract, but decoder has it disabled."
                )
            return
        if payload is None:
            raise ValueError("P bitstream lacks a local-reference transform contract, but decoder has it enabled.")
        if payload.get("analysis") != expected["analysis"] or payload.get("synthesis") != expected["synthesis"]:
            raise ValueError(
                f"P local-reference transform contract mismatch between bitstream and decoder: stream={payload} decoder={expected}"
            )

    def _p_group_contract_payload(self) -> Optional[dict[str, Any]]:
        context = self.p_group_context
        return None if context is None else context.contract_payload()

    def _validate_p_group_contract(self, payload: Any) -> None:
        context = self.p_group_context
        if context is None:
            if payload is not None:
                raise ValueError("P bitstream uses channel-group context but decoder has it disabled.")
            return
        context.validate_contract(payload)

    def _p_second_pass_spatial_contract_payload(self) -> Optional[dict[str, Any]]:
        module = self.p_second_pass_spatial_refiner
        return None if module is None else module.contract_payload()

    def _validate_p_second_pass_spatial_contract(self, payload: Any) -> None:
        expected = self._p_second_pass_spatial_contract_payload()
        if payload != expected:
            raise ValueError(
                f"P second-pass spatial-refiner contract mismatch: stream={payload!r} decoder={expected!r}"
            )

    def enable_task452_motion_relation(
        self,
        *,
        flow_step: float = 0.25,
        confidence_step: float = 0.125,
        cache_size: int = 8,
        context_scale_init: float = 0.0,
        adapter_hidden: int = 0,
        adapter_blocks: int = 3,
        adapter_residual_scale: float = 0.25,
        adapter_qp_position_conditioned: bool = False,
        adapter_bounded_output: bool = False,
    ) -> None:
        """Enable a decoder-closed 3-channel relation code before P hyper-z."""
        if flow_step <= 0 or confidence_step <= 0 or cache_size <= 0:
            raise ValueError("Task452 motion relation steps/size must be positive")
        if self.p_motion_relation_bit_estimator is not None:
            expected = (
                self.p_motion_relation_flow_step,
                self.p_motion_relation_confidence_step,
                self.p_motion_relation_cache_size,
            )
            requested = (float(flow_step), float(confidence_step), int(cache_size))
            if expected != requested:
                raise RuntimeError(f"Task452 motion relation already enabled as {expected}, requested {requested}")
            if bool(adapter_hidden) != (self.p_motion_context_adapter is not None):
                raise RuntimeError("Task452 motion relation adapter mode cannot be changed after enable")
            return
        from src.models.entropy_models import BitEstimator

        self.p_motion_relation_flow_step = float(flow_step)
        self.p_motion_relation_confidence_step = float(confidence_step)
        self.p_motion_relation_cache_size = int(cache_size)
        self.p_motion_relation_bit_estimator = BitEstimator(_INTER_QP_NUM, 3).to(self._device)
        self.p_motion_context_scale = nn.Parameter(torch.tensor(float(context_scale_init), device=self._device))
        if int(adapter_hidden) > 0:
            self.p_motion_context_adapter = PTask452MotionContextAdapter(
                latent_channels=self.latent_channels,
                hidden_channels=int(adapter_hidden),
                blocks=int(adapter_blocks),
                residual_scale=float(adapter_residual_scale),
                qp_position_conditioned=bool(adapter_qp_position_conditioned),
                p_type_count=self.p_type_count,
                bounded_output=bool(adapter_bounded_output),
            ).to(self._device)
        self._entropy_coder_ready = False

    def _task452_motion_symbols_train(self, relation: torch.Tensor, qp: int) -> tuple[torch.Tensor, torch.Tensor]:
        estimator = self.p_motion_relation_bit_estimator
        if estimator is None:
            raise RuntimeError("Task452 motion relation codec is not enabled")
        if relation.ndim != 4 or relation.shape[1] != 3:
            raise ValueError("Task452 motion relation must be [B,3,h,w]")
        steps = relation.new_tensor(
            [self.p_motion_relation_flow_step, self.p_motion_relation_flow_step, self.p_motion_relation_confidence_step]
        ).view(1, 3, 1, 1)
        normalized = relation / steps
        if estimator.training:
            hard, value_for_bits = _uf_quantize_train(normalized, hard_rate=self.p_hard_rate_training)
        else:
            hard = torch.round(normalized)
            value_for_bits = hard
        bits = _get_z_bits(value_for_bits, estimator, int(qp)).sum()
        return (hard, bits)

    def _task452_motion_reference(
        self, z_prev: torch.Tensor, symbols: torch.Tensor, qp: int = 0, frame_type_idx: int = 0
    ) -> torch.Tensor:
        if self.p_motion_context_scale is None:
            raise RuntimeError("Task452 motion context scale is missing")
        height, width = z_prev.shape[-2:]
        flow = symbols[:, :2].float() * self.p_motion_relation_flow_step
        confidence = (symbols[:, 2:3].float() * self.p_motion_relation_confidence_step).clamp(0.0, 1.0)
        flow = F.interpolate(flow, size=(height, width), mode="bilinear", align_corners=False)
        confidence = F.interpolate(confidence, size=(height, width), mode="bilinear", align_corners=False).to(
            dtype=z_prev.dtype
        )
        flow_x = flow[:, 0] * (float(width) / float(self.p_motion_relation_cache_size))
        flow_y = flow[:, 1] * (float(height) / float(self.p_motion_relation_cache_size))
        y = torch.linspace(-1.0, 1.0, height, device=z_prev.device)
        x = torch.linspace(-1.0, 1.0, width, device=z_prev.device)
        grid_y, grid_x = torch.meshgrid(y, x, indexing="ij")
        grid = torch.stack((grid_x, grid_y), dim=-1)[None].expand(z_prev.shape[0], -1, -1, -1).clone()
        if width > 1:
            grid[..., 0] += 2.0 * flow_x / float(width - 1)
        if height > 1:
            grid[..., 1] += 2.0 * flow_y / float(height - 1)
        warped = F.grid_sample(z_prev.float(), grid, mode="bilinear", padding_mode="border", align_corners=True).to(
            dtype=z_prev.dtype
        )
        if self.p_motion_context_adapter is not None:
            adapted = self.p_motion_context_adapter(z_prev, warped, confidence, qp=qp, frame_type_idx=frame_type_idx)
            self._last_motion_reference_delta_rms = float(
                (adapted - z_prev).detach().float().square().mean().sqrt().item()
            )
            return adapted
        scale = torch.tanh(self.p_motion_context_scale).to(dtype=z_prev.dtype)
        adapted = z_prev + scale * confidence * (warped - z_prev)
        self._last_motion_reference_delta_rms = float((adapted - z_prev).detach().float().square().mean().sqrt().item())
        return adapted

    def _ensure_entropy_coder(self) -> None:
        if self._entropy_coder_ready:
            return
        try:
            self.model.update(force_zero_thres=None)
            if self.p_motion_relation_bit_estimator is not None:
                self.p_motion_relation_bit_estimator.update(self.model.entropy_coder)
            self._entropy_coder_ready = True
        except ImportError as exc:
            raise RuntimeError(
                "The DCVC-RT C++ rANS extension (MLCodec_extensions_cpp) is not available. Training-only paths do not require it, but real entropy coding does."
            ) from exc

    def _load_from_state_dict(self, *args, **kwargs) -> None:
        """Invalidate derived P-frame CDFs whenever codec weights are restored."""
        super()._load_from_state_dict(*args, **kwargs)
        self._entropy_coder_ready = False

    def _get_quant_step(self, log_param: nn.Parameter, qp: int) -> torch.Tensor:
        return torch.exp(log_param[qp : qp + 1]) + 0.001

    def _type_index(self, frame_type_idx: int) -> int:
        return max(0, min(int(frame_type_idx), self.p_type_count - 1))

    @staticmethod
    def _apply_legacy_stream_q_dec(q_dec: torch.Tensor, stream_policy: str) -> torch.Tensor:
        """Decode an old stream without re-exposing its policy to training.

        ``legacy`` and ``native`` had the same hard-path lower bound.  The
        bounded experimental stream is still readable with its historical
        fixed cap.  Learned-modulation streams require the removed experiment
        checkpoint and are rejected instead of being decoded incorrectly.
        """
        policy = str(stream_policy).strip().lower()
        if policy in {"legacy", "native"}:
            return torch.clamp_min(q_dec, 0.5)
        if policy == "cap10":
            return torch.clamp(q_dec, min=0.5, max=10.0)
        if policy == "softcap10":
            q_floor = 0.5 + F.softplus(q_dec - 0.5)
            q_max = q_dec.new_tensor(10.0)
            return q_max - F.softplus(q_max - q_floor)
        raise ValueError(
            f"Unsupported legacy P bitstream q_dec policy {stream_policy!r}; decode it with its archived experiment code."
        )

    def _apply_hard_q_policy_to_params(
        self, params: torch.Tensor, *, qp: int, frame_type_idx: int, stream_policy: str = "native"
    ) -> torch.Tensor:
        q_dec, scales, means = params.chunk(3, 1)
        q_dec = self._effective_p_q_dec(q_dec, qp, frame_type_idx, stream_policy=stream_policy)
        return torch.cat((q_dec, scales, means), dim=1)

    def _apply_channel_q_scale(self, q_dec: torch.Tensor, qp: int, frame_type_idx: int) -> torch.Tensor:
        if self.p_channel_q_logit is None:
            return q_dec
        qp_idx = max(0, min(int(qp), _INTER_QP_NUM - 1))
        type_idx = self._type_index(frame_type_idx)
        log_scale = torch.tanh(self.p_channel_q_logit[type_idx, qp_idx]) * self.p_channel_q_max_log
        scale = torch.exp(log_scale).to(device=q_dec.device, dtype=q_dec.dtype)
        return torch.clamp_min(q_dec * scale.unsqueeze(0), 0.5)

    def _apply_position_q_gain(self, q_dec: torch.Tensor, qp: int, frame_type_idx: int) -> torch.Tensor:
        parameter = None
        return q_dec

    def _apply_task694_base_allocator(self, q_dec: torch.Tensor, qp: int) -> torch.Tensor:
        allocator = None
        return q_dec

    def _effective_p_q_dec(
        self, q_dec_raw: torch.Tensor, qp: int, frame_type_idx: int, *, stream_policy: str = "native"
    ) -> torch.Tensor:
        """Return the one q tensor shared by train, hard encode, and decode.

        Existing dynamic/channel_scale modes intentionally execute their exact
        historical operations.  Factorized mode geometrically migrates from
        that old channel_scale coordinate (mix=0) to an absolute
        QP/type/channel base times a normalized spatial residual (mix=1).
        """
        q_native = self._apply_legacy_stream_q_dec(q_dec_raw, stream_policy)
        effective = self._apply_position_q_gain(
            self._apply_channel_q_scale(q_native, qp, frame_type_idx), qp, frame_type_idx
        )
        return self._apply_task694_base_allocator(effective, qp)

    def _entropy_q_contract_payload(self) -> dict[str, Any]:
        """Serialize every non-checkpoint scalar that changes P symbols."""
        payload: dict[str, Any] = {"p_entropy_q_mode": self.p_entropy_q_mode}
        payload["p_channel_q_max_log"] = float(self.p_channel_q_max_log)
        payload["p_position_q_gain_enabled"] = bool(None is not None)
        if payload["p_position_q_gain_enabled"]:
            payload["p_position_q_gain_max_log"] = float(self.p_position_q_gain_max_log)
        allocator = None
        payload["p_base_allocator_contract"] = None
        return payload

    def _validate_entropy_q_contract(self, payload: dict[str, Any]) -> None:
        """Reject a P bitstream whose q coordinate cannot be reproduced."""
        stream_mode = str(payload.get("p_entropy_q_mode", "dynamic")).strip().lower()
        if stream_mode != self.p_entropy_q_mode:
            raise ValueError(
                f"P bitstream entropy-q contract mismatch: stream={stream_mode} decoder={self.p_entropy_q_mode}."
            )
        encoded_allocator = payload.get("p_base_allocator_contract")
        allocator = None
        expected_allocator = None
        if encoded_allocator != None:
            raise ValueError(
                f"P bitstream Task694 base-allocator contract mismatch: stream={encoded_allocator} decoder={None}."
            )
        contract_version = int(payload.get("codec_contract_version", 1))
        if contract_version < 3:
            if stream_mode == "factorized":
                raise ValueError(
                    "Factorized P-q bitstream lacks its spatial/mix contract; codec_contract_version>=3 is required."
                )
            return

        def require_close(key: str, expected: float) -> None:
            if key not in payload:
                raise ValueError(f"P bitstream q contract is missing {key}.")
            actual = float(payload[key])
            if not math.isclose(actual, expected, rel_tol=0.0, abs_tol=1e-07):
                raise ValueError(f"P bitstream q contract mismatch for {key}: stream={actual} decoder={expected}.")

        if stream_mode in {"channel_scale", "factorized"}:
            require_close("p_channel_q_max_log", float(self.p_channel_q_max_log))
        stream_position_gain = bool(payload.get("p_position_q_gain_enabled", False))
        decoder_position_gain = None is not None
        if stream_position_gain != decoder_position_gain:
            raise ValueError(
                f"P bitstream position-q contract mismatch: stream={stream_position_gain} decoder={decoder_position_gain}."
            )
        if decoder_position_gain:
            require_close("p_position_q_gain_max_log", float(self.p_position_q_gain_max_log))
        if stream_mode == "factorized":
            spatial_policy = payload.get("p_factorized_q_spatial_policy")
            if spatial_policy != "zero_log_mean_strict_bound_v1":
                raise ValueError(
                    f"P bitstream q contract mismatch for p_factorized_q_spatial_policy: stream={spatial_policy!r} decoder='zero_log_mean_strict_bound_v1'."
                )
            require_close("p_factorized_q_spatial_max_log", float(self.p_factorized_q_spatial_max_log))
            require_close("p_factorized_q_mix", float(self.p_factorized_q_mix.detach().cpu().item()))

    def _y_gain(self, qp: int, frame_type_idx: int, like: torch.Tensor) -> torch.Tensor:
        return torch.ones((1, _RT_Y, 1, 1), device=like.device, dtype=like.dtype)

    def _y_balance(self, qp: int, frame_type_idx: int, y: torch.Tensor) -> torch.Tensor:
        """Channel-wise affine energy rebalance before P entropy coding."""
        return y

    def _y_balance_inverse(self, qp: int, frame_type_idx: int, y_balanced: torch.Tensor) -> torch.Tensor:
        return y_balanced

    def _frequency_analysis(self, y: torch.Tensor, ctx_t: torch.Tensor, qp: int, frame_type_idx: int) -> torch.Tensor:
        module = self.p_frequency_compaction
        if module is None:
            return y
        return module.analysis(y, ctx_t, qp, frame_type_idx)

    def _frequency_synthesis(
        self, y_hat: torch.Tensor, ctx_t: torch.Tensor, qp: int, frame_type_idx: int
    ) -> torch.Tensor:
        module = self.p_frequency_compaction
        if module is None:
            return y_hat
        return module.synthesis(y_hat, ctx_t, qp, frame_type_idx)

    def _frequency_contract_payload(self) -> Optional[dict[str, Any]]:
        module = self.p_frequency_compaction
        return None if module is None else module.contract_payload()

    def _p_transform_capacity_contract_payload(self) -> Optional[dict[str, Any]]:
        return p_transform_capacity_contract(self.p_analysis_core, self.p_synthesis_core)

    def _validate_p_transform_capacity_contract(self, payload: Any) -> None:
        expected = self._p_transform_capacity_contract_payload()
        if payload != expected:
            raise ValueError(
                f"P transform-capacity bitstream contract mismatch: stream={payload!r} decoder={expected!r}"
            )

    def _apply_p_analysis_core(self, y: torch.Tensor, qp: int) -> torch.Tensor:
        module = self.p_analysis_core
        return y if module is None else module(y, qp)

    def _continuous_display_contract_payload(self) -> Optional[dict[str, Any]]:
        module = self.p_synthesis_head
        if not isinstance(module, PContinuousDisplaySynthesisHead):
            return None
        return module.contract_payload()

    def _validate_continuous_display_contract(self, encoded: Optional[dict[str, Any]]) -> None:
        expected = self._continuous_display_contract_payload()
        if encoded != expected:
            raise ValueError(
                f"P continuous-display bitstream contract mismatch: stream={encoded!r} decoder={expected!r}"
            )

    def set_p_continuous_display_reference_mode(self, mode: str) -> None:
        module = self.p_synthesis_head
        if not isinstance(module, PContinuousDisplaySynthesisHead):
            raise RuntimeError("continuous display synthesis is not enabled")
        module.set_reference_mode(mode)

    def _apply_synthesis_head(
        self, z_out: torch.Tensor, feature: torch.Tensor, *, reference_latent: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        if self.p_synthesis_head is None:
            return z_out
        return self.p_synthesis_head(
            feature,
            z_out,
            position_idx=self._synthesis_position_idx,
            qp=self._synthesis_qp,
            reference_latent=reference_latent,
        )

    def _explicit_scale(self, frame_type_idx: int) -> float:
        return self.explicit_residual_scale

    def _condition_p_input(self, x: torch.Tensor, frame_type_idx: int) -> torch.Tensor:
        type_idx = self._type_index(frame_type_idx)
        assert self.p_type_input_gain is not None
        assert self.p_type_input_bias is not None
        gain = torch.tanh(self.p_type_input_gain[type_idx : type_idx + 1]) * self.p_type_input_scale
        bias = torch.tanh(self.p_type_input_bias[type_idx : type_idx + 1]) * self.p_type_input_scale
        return x * (1.0 + gain) + bias

    def _condition_quant_step(
        self, q: torch.Tensor, delta: Optional[nn.Parameter], frame_type_idx: int
    ) -> torch.Tensor:
        if not self.p_type_conditioning or delta is None:
            return q
        type_idx = self._type_index(frame_type_idx)
        scale = torch.exp(torch.tanh(delta[type_idx : type_idx + 1]) * self.p_type_quant_scale)
        return q * scale

    def _entropy_scale_factor(self, frame_type_idx: int, *, like: torch.Tensor) -> torch.Tensor:
        return torch.ones((1, _RT_Y, 1, 1), device=like.device, dtype=like.dtype)

    def _condition_entropy_scales(self, scales: torch.Tensor, frame_type_idx: int) -> torch.Tensor:
        return scales * self._entropy_scale_factor(frame_type_idx, like=scales)

    def _entropy_mean_offset(self, frame_type_idx: int, *, like: torch.Tensor) -> torch.Tensor:
        return torch.zeros((1, _RT_Y, 1, 1), device=like.device, dtype=like.dtype)

    def _condition_entropy_means(self, means: torch.Tensor, frame_type_idx: int) -> torch.Tensor:
        return means + self._entropy_mean_offset(frame_type_idx, like=means)

    def _apply_decoded_y_prior(
        self,
        scales: torch.Tensor,
        means: torch.Tensor,
        frame_type_idx: int,
        *,
        ref_y_prev: Optional[torch.Tensor],
        ctx_t: Optional[torch.Tensor],
        qp: Optional[int],
        spatial_pass: bool,
        current_y: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        prior = self.p_decoded_y_prior
        if prior is None or ref_y_prev is None:
            return (scales, means)
        if ctx_t is None or qp is None:
            raise ValueError("Decoded-y DPB entropy requires ctx_t and qp.")
        log_delta, mean_delta = prior(
            ref_y_prev,
            ctx_t,
            qp,
            frame_type_idx,
            y_size=scales.shape[-2:],
            spatial_pass=spatial_pass,
            current_y=current_y,
        )
        if self.p_decoded_y_scale_only or int(qp) > int(self.p_decoded_y_mean_max_qp):
            mean_delta = torch.zeros_like(mean_delta)
        self._last_dpb_prior_log_scale_rms = float(log_delta.detach().float().square().mean().sqrt().item())
        self._last_dpb_prior_mean_rms = float(mean_delta.detach().float().square().mean().sqrt().item())
        return (scales * torch.exp(log_delta.to(dtype=scales.dtype)), means + mean_delta.to(dtype=means.dtype))

    def _condition_common_entropy_params(
        self,
        params: torch.Tensor,
        frame_type_idx: int,
        *,
        ctx_t: Optional[torch.Tensor] = None,
        qp: Optional[int] = None,
        ref_y_prev: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Apply static and sample-adaptive common-prior corrections."""
        q_dec, scales, means = params.chunk(3, 1)
        scales = self._condition_entropy_scales(scales, frame_type_idx)
        means = self._condition_entropy_means(means, frame_type_idx)
        adaptive_entropy = getattr(self, "p_adaptive_entropy", None)
        if adaptive_entropy is not None:
            if ctx_t is None or qp is None:
                raise ValueError("Adaptive P entropy requires ctx_t and qp.")
            log_delta, mean_delta = adaptive_entropy(ctx_t, qp, frame_type_idx, y_size=params.shape[-2:])
            scales = scales * torch.exp(log_delta.to(dtype=scales.dtype))
            means = means + mean_delta.to(dtype=means.dtype)
        scales, means = self._apply_decoded_y_prior(
            scales, means, frame_type_idx, ref_y_prev=ref_y_prev, ctx_t=ctx_t, qp=qp, spatial_pass=False
        )
        return torch.cat((q_dec, scales, means), dim=1)

    def _condition_entropy_distribution(
        self,
        scales: torch.Tensor,
        means: torch.Tensor,
        frame_type_idx: int,
        *,
        ctx_t: Optional[torch.Tensor],
        qp: Optional[int],
        y_hat_0: Optional[torch.Tensor] = None,
        ref_y_prev: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        scales = self._condition_entropy_scales(scales, frame_type_idx)
        means = self._condition_entropy_means(means, frame_type_idx)
        adaptive_entropy = getattr(self, "p_adaptive_entropy", None)
        if adaptive_entropy is not None:
            if ctx_t is None or qp is None:
                raise ValueError("Adaptive P entropy requires ctx_t and qp.")
            log_delta, mean_delta = adaptive_entropy(
                ctx_t, qp, frame_type_idx, y_size=scales.shape[-2:], y_hat_0=y_hat_0
            )
            scales = scales * torch.exp(log_delta.to(dtype=scales.dtype))
            means = means + mean_delta.to(dtype=means.dtype)
        return self._apply_decoded_y_prior(
            scales,
            means,
            frame_type_idx,
            ref_y_prev=ref_y_prev,
            ctx_t=ctx_t,
            qp=qp,
            spatial_pass=y_hat_0 is not None,
            current_y=y_hat_0,
        )

    def _hard_spatial_prior(
        self,
        cat_params: torch.Tensor,
        frame_type_idx: int,
        *,
        ctx_t: Optional[torch.Tensor] = None,
        qp: Optional[int] = None,
        raw_common_params: Optional[torch.Tensor] = None,
        ref_y_prev: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Preserve the native spatial-prior input while conditioning its CDF.

        DCVC's hard helper concatenates the already-conditioned common params
        into the spatial-prior input.  Undo the static/adaptive common-prior
        corrections before calling the native network, then condition its
        second-pass distribution.  This mirrors ``encode_train`` and keeps the
        real encoder and decoder causal and symmetric.
        """
        channels = _RT_Y
        y_hat_0 = cat_params[:, :channels]
        if raw_common_params is not None:
            native_out = self.model.y_spatial_prior(torch.cat((y_hat_0, raw_common_params), dim=1))
            scales, spatial_means = native_out.chunk(2, 1)
            scales, spatial_means = self._condition_entropy_distribution(
                scales, spatial_means, frame_type_idx, ctx_t=ctx_t, qp=qp, y_hat_0=y_hat_0, ref_y_prev=ref_y_prev
            )
            return torch.cat((scales, spatial_means), dim=1)
        q_dec, scales_calibrated, means = cat_params[:, channels:].chunk(3, 1)
        adaptive_entropy = getattr(self, "p_adaptive_entropy", None)
        if adaptive_entropy is not None:
            if ctx_t is None or qp is None:
                raise ValueError("Adaptive P entropy requires ctx_t and qp.")
            common_log_delta, common_mean_delta = adaptive_entropy(
                ctx_t, qp, frame_type_idx, y_size=scales_calibrated.shape[-2:]
            )
            scales_calibrated = scales_calibrated / torch.exp(common_log_delta.to(dtype=scales_calibrated.dtype))
            means = means - common_mean_delta.to(dtype=means.dtype)
        factor = self._entropy_scale_factor(frame_type_idx, like=scales_calibrated)
        mean_offset = self._entropy_mean_offset(frame_type_idx, like=means)
        native_params = torch.cat((y_hat_0, q_dec, scales_calibrated / factor, means - mean_offset), dim=1)
        scales, spatial_means = self.model.y_spatial_prior(native_params).chunk(2, 1)
        if adaptive_entropy is None:
            scales = self._condition_entropy_scales(scales, frame_type_idx)
            spatial_means = self._condition_entropy_means(spatial_means, frame_type_idx)
            scales, spatial_means = self._apply_decoded_y_prior(
                scales,
                spatial_means,
                frame_type_idx,
                ref_y_prev=ref_y_prev,
                ctx_t=ctx_t,
                qp=qp,
                spatial_pass=True,
                current_y=y_hat_0,
            )
        else:
            assert ctx_t is not None and qp is not None
            scales, spatial_means = self._condition_entropy_distribution(
                scales, spatial_means, frame_type_idx, ctx_t=ctx_t, qp=qp, y_hat_0=y_hat_0, ref_y_prev=ref_y_prev
            )
        return torch.cat((scales, spatial_means), dim=1)

    def shift_qp(self, qp: int, fa_idx: int) -> int:
        """Apply DCVC-RT P-frame QP offset."""
        qp_shift = [0, 8, 4]
        return max(0, min(qp + qp_shift[fa_idx], _INTER_QP_NUM - 1))

    def eval(self) -> "DMCAdapter":
        super().eval()
        self.model.eval()
        if self.p_adaptive_entropy is not None:
            self.p_adaptive_entropy.eval()
        if self.p_synthesis_head is not None:
            self.p_synthesis_head.eval()
        if getattr(self, "p_invertible_latent_lifting", None) is not None:
            self.p_invertible_latent_lifting.eval()
        if getattr(self, "p_reference_translation_lifting", None) is not None:
            self.p_reference_translation_lifting.eval()
        if getattr(self, "p_y_conditioned_context_warp", None) is not None:
            self.p_y_conditioned_context_warp.eval()
        if getattr(self, "p_mature_context_coupling", None) is not None:
            self.p_mature_context_coupling.eval()
        if getattr(self, "p_context_native_deep_transform", None) is not None:
            self.p_context_native_deep_transform.eval()
        if getattr(self, "p_local_reference_transform_analysis", None) is not None:
            self.p_local_reference_transform_analysis.eval()
        if getattr(self, "p_local_reference_transform_synthesis", None) is not None:
            self.p_local_reference_transform_synthesis.eval()
        return self

    def train(self, mode: bool = True) -> "DMCAdapter":
        super().train(mode)
        self.model.train(mode)
        if self.p_adaptive_entropy is not None:
            self.p_adaptive_entropy.train(mode)
        if self.p_synthesis_head is not None:
            self.p_synthesis_head.train(mode)
        if getattr(self, "p_context_first_core", None) is not None:
            self.p_context_first_core.train(mode)
        if getattr(self, "p_invertible_latent_lifting", None) is not None:
            self.p_invertible_latent_lifting.train(mode)
        if getattr(self, "p_reference_translation_lifting", None) is not None:
            self.p_reference_translation_lifting.train(mode)
        if getattr(self, "p_y_conditioned_context_warp", None) is not None:
            self.p_y_conditioned_context_warp.train(mode)
        if getattr(self, "p_mature_context_coupling", None) is not None:
            self.p_mature_context_coupling.train(mode)
        if getattr(self, "p_context_native_deep_transform", None) is not None:
            self.p_context_native_deep_transform.train(mode)
        if getattr(self, "p_local_reference_transform_analysis", None) is not None:
            self.p_local_reference_transform_analysis.train(mode)
        if getattr(self, "p_local_reference_transform_synthesis", None) is not None:
            self.p_local_reference_transform_synthesis.train(mode)
        return self

    def to(self, device: str) -> "DMCAdapter":
        super().to(device)
        self._device = device
        self.model.to(device)
        if self.p_adaptive_entropy is not None:
            self.p_adaptive_entropy.to(device)
        if self.p_synthesis_head is not None:
            self.p_synthesis_head.to(device)
        if getattr(self, "p_context_first_core", None) is not None:
            self.p_context_first_core.to(device)
        if getattr(self, "p_invertible_latent_lifting", None) is not None:
            self.p_invertible_latent_lifting.to(device)
        if getattr(self, "p_reference_translation_lifting", None) is not None:
            self.p_reference_translation_lifting.to(device)
        if getattr(self, "p_y_conditioned_context_warp", None) is not None:
            self.p_y_conditioned_context_warp.to(device)
        if getattr(self, "p_mature_context_coupling", None) is not None:
            self.p_mature_context_coupling.to(device)
        if getattr(self, "p_context_native_deep_transform", None) is not None:
            self.p_context_native_deep_transform.to(device)
        if getattr(self, "p_reference_affine_lifting", None) is not None:
            self.p_reference_affine_lifting.to(device)
        if getattr(self, "p_local_reference_transform_analysis", None) is not None:
            self.p_local_reference_transform_analysis.to(device)
        if getattr(self, "p_local_reference_transform_synthesis", None) is not None:
            self.p_local_reference_transform_synthesis.to(device)
        for p in (self.qe_log_inter, self.qd_log_inter, self.qf_log_inter, self.qr_log_inter):
            p.data = p.data.to(device)
        for p in (
            self.p_type_input_bias,
            self.p_type_input_gain,
            self.p_type_qe_delta,
            self.p_type_qd_delta,
            self.p_type_qf_delta,
            self.p_type_qr_delta,
            self.p_entropy_log_scale_delta,
            self.p_entropy_mean_delta,
            self.p_residual_blend_logit,
            self.p_y_gain_logit,
            self.p_y_balance_gain_logit,
            self.p_y_balance_bias,
        ):
            if p is not None:
                p.data = p.data.to(device)
        return self

    def _encode_input(
        self,
        x: torch.Tensor,
        ctx: torch.Tensor,
        qe: torch.Tensor,
        qp: int,
        frame_type_idx: int,
        analysis_feature_hint: Optional[torch.Tensor] = None,
        reference_latent: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Encode [B,C,H,W] latent to y.

        Default DCVC-RT mode emits ``H/2,W/2``.  The opt-in preserve mode keeps
        ``H,W`` to avoid a second spatial downsample on Wan latents.
        """
        if self.p_context_first_core is not None:
            if analysis_feature_hint is not None:
                raise ValueError("context-first core does not accept encoder-private feature hints")
            return self.p_context_first_core.analysis(
                x, ctx, qe, qp, frame_type_idx, self._p_context_first_mode, reference_latent=reference_latent
            )
        feature = self.model.encoder.conv1(x)
        native_reference = None
        native_context = None
        if self.p_context_native_deep_transform is not None:
            native_reference, native_context = self._p_context_native_reference(reference_latent, ctx)
            feature = self.p_context_native_deep_transform(
                "analysis_pre", feature, native_reference, native_context, qp=qp, position=frame_type_idx
            )
        if self.p_y_conditioned_context_warp is not None and self.p_y_conditioned_context_warp.analysis_enabled:
            ctx = self.p_y_conditioned_context_warp(feature, ctx, qp, frame_type_idx)
        if self.p_mature_context_coupling is not None:
            reference_feature, branch_context = self._p_mature_context_reference(reference_latent, ctx)
            feature = self.p_mature_context_coupling.analysis(
                feature, reference_feature, branch_context, qp, frame_type_idx
            )
        if self.p_shared_latent_lifting_alpha is not None:
            feature = feature - self._shared_latent_reference_feature(
                reference_latent, qp, frame_type_idx, allow_cache=False
            )
        if self.p_reference_affine_lifting is not None:
            feature = self.p_reference_affine_lifting.analysis(
                feature, ctx, qp, frame_type_idx, self._p_reference_affine_lifting_mode
            )
        if self.p_context_pyramid_analysis is not None:
            feature = self.p_context_pyramid_analysis(feature, ctx, qp, frame_type_idx)
        if self.p_local_reference_transform_analysis is not None:
            ctx = self.p_local_reference_transform_analysis(
                feature, ctx, qp, frame_type_idx, reference_mode=self._p_local_reference_mode
            )
        if analysis_feature_hint is not None:
            if analysis_feature_hint.shape != feature.shape:
                raise ValueError(
                    f"P analysis feature hint must match the first encoder feature: hint={tuple(analysis_feature_hint.shape)} feature={tuple(feature.shape)}"
                )
            feature = feature + analysis_feature_hint.to(device=feature.device, dtype=feature.dtype)
        feature = self.model.encoder.conv2(torch.cat((feature, ctx), dim=1))
        if self.p_context_native_deep_transform is not None:
            assert native_reference is not None and native_context is not None
            feature = self.p_context_native_deep_transform(
                "analysis_mid", feature, native_reference, native_context, qp=qp, position=frame_type_idx
            )
        feature = self.model.encoder.conv3(feature)
        if self.p_context_native_deep_transform is not None:
            assert native_reference is not None and native_context is not None
            feature = self.p_context_native_deep_transform(
                "analysis_tail", feature, native_reference, native_context, qp=qp, position=frame_type_idx
            )
        feature = feature * qe
        y = self.model.encoder.down(feature)
        return self._apply_p_analysis_core(y, qp)

    def _decode_output(
        self,
        y_hat: torch.Tensor,
        ctx: torch.Tensor,
        qd: torch.Tensor,
        qr: torch.Tensor,
        return_feature: bool = False,
        reference_latent: Optional[torch.Tensor] = None,
        *,
        p_core_qp: Optional[int] = None,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        """Decode y_hat with context to latent [B,C,H,W]."""
        if self.p_context_first_core is not None:
            if self._synthesis_qp is None or self._synthesis_position_idx is None:
                raise RuntimeError("P context-first synthesis requires decoder-known QP and position")
            frame_type_idx = max(0, int(self._synthesis_position_idx) - 1)
            feature = self.p_context_first_core.synthesis(
                y_hat,
                ctx,
                qd,
                int(self._synthesis_qp),
                frame_type_idx,
                self._p_context_first_mode,
                reference_latent=reference_latent,
            )
        elif self.p_mature_context_coupling is not None:
            if self._synthesis_qp is None or self._synthesis_position_idx is None:
                raise RuntimeError("P mature context coupling synthesis requires decoder-known QP and position")
            frame_type_idx = max(0, int(self._synthesis_position_idx) - 1)
            feature = self.model.decoder.up(y_hat)
            reference_feature, branch_context = self._p_mature_context_reference(reference_latent, ctx)
            feature = self.p_mature_context_coupling.synthesis(
                feature, reference_feature, branch_context, int(self._synthesis_qp), frame_type_idx
            )
            feature = self.model.decoder.conv1(torch.cat((feature, ctx), dim=1))
            feature = self.model.decoder.conv2(feature)
            feature = feature * qd
        elif self.p_context_native_deep_transform is not None:
            if self._synthesis_qp is None or self._synthesis_position_idx is None:
                raise RuntimeError("P context-native synthesis requires decoder-known QP and position")
            frame_type_idx = max(0, int(self._synthesis_position_idx) - 1)
            native_reference, native_context = self._p_context_native_reference(reference_latent, ctx)
            feature = self.model.decoder.up(y_hat)
            feature = self.p_context_native_deep_transform(
                "synthesis_pre",
                feature,
                native_reference,
                native_context,
                qp=int(self._synthesis_qp),
                position=frame_type_idx,
            )
            feature = self.model.decoder.conv1(torch.cat((feature, ctx), dim=1))
            feature = self.p_context_native_deep_transform(
                "synthesis_mid",
                feature,
                native_reference,
                native_context,
                qp=int(self._synthesis_qp),
                position=frame_type_idx,
            )
            feature = self.model.decoder.conv2(feature)
            feature = self.p_context_native_deep_transform(
                "synthesis_tail",
                feature,
                native_reference,
                native_context,
                qp=int(self._synthesis_qp),
                position=frame_type_idx,
            )
            feature = feature * qd
        elif self.p_reference_affine_lifting is not None:
            if self._synthesis_qp is None or self._synthesis_position_idx is None:
                raise RuntimeError("P reference affine lifting synthesis requires decoder-known QP and position")
            frame_type_idx = max(0, int(self._synthesis_position_idx) - 1)
            feature = self.model.decoder.up(y_hat)
            feature = self.p_reference_affine_lifting.synthesis(
                feature, ctx, int(self._synthesis_qp), frame_type_idx, self._p_reference_affine_lifting_mode
            )
            feature = self.model.decoder.conv1(torch.cat((feature, ctx), dim=1))
            feature = self.model.decoder.conv2(feature)
            feature = feature * qd
        elif self.p_y_conditioned_context_warp is not None:
            if self._synthesis_qp is None or self._synthesis_position_idx is None:
                raise RuntimeError("P y-conditioned context warp requires decoder-known QP and position")
            frame_type_idx = max(0, int(self._synthesis_position_idx) - 1)
            feature = self.model.decoder.up(y_hat)
            warp_query = self.p_y_conditioned_context_warp.decoded_query(y_hat, self.model.decoder.up, feature)
            warped_ctx = self.p_y_conditioned_context_warp(warp_query, ctx, int(self._synthesis_qp), frame_type_idx)
            feature = self.model.decoder.conv1(torch.cat((feature, warped_ctx), dim=1))
            feature = self.model.decoder.conv2(feature)
            feature = feature * qd
        elif self.p_local_reference_transform_synthesis is not None:
            if self._synthesis_qp is None or self._synthesis_position_idx is None:
                raise RuntimeError("P local-reference synthesis requires decoder-known QP and position")
            frame_type_idx = max(0, int(self._synthesis_position_idx) - 1)
            feature = self.model.decoder.up(y_hat)
            ctx_fused = self.p_local_reference_transform_synthesis(
                feature, ctx, int(self._synthesis_qp), frame_type_idx, reference_mode=self._p_local_reference_mode
            )
            feature = self.model.decoder.conv1(torch.cat((feature, ctx_fused), dim=1))
            feature = self.model.decoder.conv2(feature)
            feature = feature * qd
        else:
            feature = self.model.decoder(y_hat, ctx, qd)
        if self.p_shared_latent_lifting_alpha is not None:
            if self._synthesis_qp is None or self._synthesis_position_idx is None:
                raise RuntimeError("P shared-latent synthesis requires decoder-known QP and position")
            frame_type_idx = max(0, int(self._synthesis_position_idx) - 1)
            feature = (
                feature
                + self._shared_latent_reference_feature(
                    reference_latent, int(self._synthesis_qp), frame_type_idx, allow_cache=True
                )
                * qd
            )
        if self.p_context_pyramid_synthesis is not None:
            if self._synthesis_qp is None or self._synthesis_position_idx is None:
                raise RuntimeError("P symmetric synthesis requires decoder-known QP and position")
            frame_type_idx = max(0, int(self._synthesis_position_idx) - 1)
            feature = self.p_context_pyramid_synthesis(feature, ctx, int(self._synthesis_qp), frame_type_idx)
        out = self.model.recon_generation_net.conv(feature)
        if self.p_context_native_deep_transform is not None:
            if self._synthesis_qp is None or self._synthesis_position_idx is None:
                raise RuntimeError("P context-native reconstruction requires QP and position")
            frame_type_idx = max(0, int(self._synthesis_position_idx) - 1)
            native_reference, native_context = self._p_context_native_reference(reference_latent, ctx)
            out = self.p_context_native_deep_transform(
                "reconstruction",
                out,
                native_reference,
                native_context,
                qp=int(self._synthesis_qp),
                position=frame_type_idx,
            )
        out = out * qr
        if self.p_recurrent_synthesis_core is not None:
            if self._synthesis_qp is None or self._synthesis_position_idx is None:
                raise RuntimeError("P recurrent synthesis requires decoder-known QP and position")
            recurrent_state = self._p_recurrent_synthesis_state
            if recurrent_state is not None:
                if self._p_recurrent_synthesis_mode == "zero":
                    recurrent_state = torch.zeros_like(recurrent_state)
                elif self._p_recurrent_synthesis_mode == "wrong":
                    recurrent_state = torch.roll(
                        recurrent_state,
                        shifts=(max(1, recurrent_state.shape[-2] // 2), max(1, recurrent_state.shape[-1] // 2)),
                        dims=(-2, -1),
                    )
            out, next_recurrent_state = self.p_recurrent_synthesis_core(
                out, recurrent_state, qp=int(self._synthesis_qp), position=int(self._synthesis_position_idx)
            )
            self._p_recurrent_synthesis_state = next_recurrent_state
        if self.p_context_first_core is not None and bool(
            getattr(self.p_context_first_core, "outputs_latent_residual", False)
        ):
            z_out = self.p_context_first_core.reconstruct_latent(out, reference_latent, self._p_context_first_mode)
        else:
            z_out = self.model.recon_generation_net.head(out)
        self._last_pre_synthesis_reconstruction = z_out
        self._last_reconstruction_feature = out
        z_out = self._apply_synthesis_head(z_out, out, reference_latent=reference_latent)
        if return_feature:
            return (z_out, feature)
        return z_out

    def _match_prior_params(self, params: torch.Tensor, y_like: torch.Tensor) -> torch.Tensor:
        """Resize prior parameters when preserve mode keeps y at latent H/W."""
        if params.shape[-2:] == y_like.shape[-2:]:
            return params
        return F.interpolate(params, size=y_like.shape[-2:], mode="bilinear", align_corners=False).contiguous()

    def _bitstream_shapes(self, input_h: int, input_w: int) -> tuple[int, int, int, int]:
        y_h, y_w = (input_h // 2, input_w // 2)
        z_h, z_w = ((input_h + 7) // 8, (input_w + 7) // 8)
        return (y_h, y_w, z_h, z_w)

    def get_temporal_context(
        self,
        z_prev: torch.Tensor,
        qp: int,
        feature_prev: Optional[torch.Tensor] = None,
        feature_blend: float = 1.0,
        frame_type_idx: int = 0,
        dpb_feature_prev: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute (ctx, ctx_t) from a reference latent [B,C,H,W]."""
        qf = self._get_quant_step(self.qf_log_inter, qp)
        qf = self._condition_quant_step(qf, self.p_type_qf_delta, frame_type_idx)
        context_reference = z_prev
        if self.p1_context_adapter is not None and self._type_index(frame_type_idx) == 0:
            context_reference = self.p1_context_adapter(z_prev, qp)
        feature_from_latent = self.model.feature_adaptor_i(context_reference)
        if feature_prev is None:
            feature = feature_from_latent
        else:
            feature_from_state = self.model.feature_adaptor_p(feature_prev)
            blend = float(feature_blend)
            if blend <= 0.0:
                feature = feature_from_latent
            elif blend >= 1.0:
                feature = feature_from_state
            else:
                feature = feature_from_latent + blend * (feature_from_state - feature_from_latent)
        if self.p_decoded_feature_fusion is not None and dpb_feature_prev is not None:
            feature_from_dpb = self.model.feature_adaptor_p(dpb_feature_prev)
            parent_feature = feature
            feature = self.p_decoded_feature_fusion(parent_feature, feature_from_dpb, qp, frame_type_idx)
            self._last_dpb_feature_delta_rms = float(
                (feature - parent_feature).detach().float().square().mean().sqrt().item()
            )
        else:
            self._last_dpb_feature_delta_rms = 0.0
        if self.p_htcm_long_fusion is not None and self._active_htcm_long_feature is not None:
            long_feature = self._active_htcm_long_feature
            if self._p_htcm_memory_mode == "zero":
                long_feature = torch.zeros_like(long_feature)
            elif self._p_htcm_memory_mode == "wrong":
                long_feature = torch.roll(
                    long_feature,
                    shifts=(max(1, long_feature.shape[-2] // 2), max(1, long_feature.shape[-1] // 2)),
                    dims=(-2, -1),
                )
            feature_from_long = self.model.feature_adaptor_p(long_feature)
            feature = self.p_htcm_long_fusion(feature, feature_from_long, qp, frame_type_idx)
        return self.model.feature_extractor(feature, qf)

    def set_p_htcm_memory_mode(self, mode: str) -> None:
        mode = str(mode).lower().strip()
        if mode not in {"correct", "zero", "wrong"}:
            raise ValueError("P HTCM memory mode must be correct|zero|wrong")
        self._p_htcm_memory_mode = mode

    def _p_htcm_contract_payload(self) -> Optional[dict[str, Any]]:
        module = None
        return None

    def _validate_p_htcm_contract(self, payload: Any) -> None:
        expected = self._p_htcm_contract_payload()
        if payload != expected:
            raise ValueError(f"P HTCM bitstream contract mismatch: stream={payload!r} decoder={expected!r}")

    def _predict_from_reference(
        self,
        z_prev: Optional[torch.Tensor],
        ctx: torch.Tensor,
        qp: int,
        shape_like: torch.Tensor,
        frame_type_idx: int = 0,
    ) -> torch.Tensor:
        return torch.zeros_like(shape_like)

    def _predict_z_from_context(
        self, ctx_t: torch.Tensor, qp: int, frame_type_idx: int, shape_like: torch.Tensor
    ) -> torch.Tensor:
        return torch.zeros_like(shape_like)

    def _refine_reconstruction(
        self,
        z_out: torch.Tensor,
        z_prev: Optional[torch.Tensor],
        ctx: torch.Tensor,
        qp: int,
        residual_mode: Optional[str] = None,
    ) -> torch.Tensor:
        mode = self.residual_mode if residual_mode is None else str(residual_mode).lower()
        if mode != "hybrid_refine":
            return z_out
        if z_prev is None:
            raise ValueError("hybrid_refine P mode requires z_prev for reconstruction refinement.")
        raise ValueError("hybrid_refine bitstream requires DMCAdapter.residual_refiner.")

    def _predict_y_from_context(
        self, ctx: torch.Tensor, qp: int, shape_like: torch.Tensor, residual_mode: Optional[str] = None
    ) -> torch.Tensor:
        mode = self.residual_mode if residual_mode is None else str(residual_mode).lower()
        if mode != "y_centered":
            return torch.zeros_like(shape_like)
        raise ValueError("y_centered bitstream requires DMCAdapter.y_context_predictor.")

    def _residual_blend_prediction(
        self, z_prev: Optional[torch.Tensor], frame_type_idx: int, shape_like: torch.Tensor
    ) -> torch.Tensor:
        return torch.zeros_like(shape_like)

    @staticmethod
    def _split_second_checkerboard_mask(mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        row_even = torch.zeros_like(mask[:, :1])
        row_even[:, :, 0::2, :] = 1
        first = mask * row_even
        second = mask - first
        return (first, second)

    def _p_spatial_refined_scales(
        self,
        base_scales: torch.Tensor,
        y_hat_prefix: torch.Tensor,
        common_params: torch.Tensor,
        qp: int,
        frame_type_idx: int,
        subpass: int,
    ) -> torch.Tensor:
        module = self.p_second_pass_spatial_refiner
        if module is None:
            raise RuntimeError("P second-pass spatial refiner is disabled")
        log_delta = module(y_hat_prefix, common_params, qp, frame_type_idx, subpass)
        return (base_scales.clamp(0.11, 16.0) * torch.exp(log_delta)).clamp(0.11, 16.0)

    def _train_p_second_pass_spatial_prior(
        self,
        y_scaled: torch.Tensor,
        params_for_context: torch.Tensor,
        initial_scales: torch.Tensor,
        initial_means: torch.Tensor,
        *,
        ctx_t: torch.Tensor,
        qp: int,
        frame_type_idx: int,
        ref_y_prev: Optional[torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        B, C, H, W = y_scaled.shape
        mask_0, mask_1 = self.model.get_mask_2x(B, C, H, W, y_scaled.dtype, y_scaled.device)
        means_hat_0 = initial_means * mask_0
        y_res_0 = (y_scaled - means_hat_0) * mask_0
        y_q_0, y_bits_0 = _uf_quantize_train(y_res_0, hard_rate=self.p_hard_rate_training)
        y_hat = y_q_0 * mask_0 + means_hat_0
        bits = _get_y_gaussian_bits(y_bits_0 * mask_0, initial_scales * mask_0, mask=mask_0, coder_aligned=True).sum()
        base_scales, base_means = self.model.y_spatial_prior(torch.cat((y_hat, params_for_context), dim=1)).chunk(2, 1)
        base_scales, base_means = self._condition_entropy_distribution(
            base_scales, base_means, frame_type_idx, ctx_t=ctx_t, qp=qp, y_hat_0=y_hat, ref_y_prev=ref_y_prev
        )
        for subpass, mask in enumerate(self._split_second_checkerboard_mask(mask_1)):
            scales = self._p_spatial_refined_scales(base_scales, y_hat, params_for_context, qp, frame_type_idx, subpass)
            means_hat = base_means * mask
            residual = (y_scaled - means_hat) * mask
            y_q, y_bits = _uf_quantize_train(residual, hard_rate=self.p_hard_rate_training)
            y_hat = y_hat + y_q * mask + means_hat
            bits = bits + _get_y_gaussian_bits(y_bits * mask, scales * mask, mask=mask, coder_aligned=True).sum()
        return (y_hat, bits)

    def _compress_p_second_pass_spatial_prior(
        self, y: torch.Tensor, params: torch.Tensor, spatial_prior: Any, *, qp: int, frame_type_idx: int
    ) -> tuple[list[torch.Tensor], list[torch.Tensor], torch.Tensor]:
        y_scaled, q_dec, scales, means = self.model.separate_prior_for_video_encoding(params, y)
        B, C, H, W = y_scaled.shape
        mask_0, mask_1 = self.model.get_mask_2x(B, C, H, W, y_scaled.dtype, y_scaled.device)
        writes: list[torch.Tensor] = []
        scale_writes: list[torch.Tensor] = []
        _, y_q, y_hat, scales_hat = self.model.process_with_mask(y_scaled, scales, means, mask_0)
        active = mask_0.to(torch.bool)
        writes.append(y_q.masked_select(active))
        scale_writes.append(scales_hat.masked_select(active))
        base_scales, base_means = spatial_prior(torch.cat((y_hat, params), dim=1)).chunk(2, 1)
        for subpass, mask in enumerate(self._split_second_checkerboard_mask(mask_1)):
            refined = self._p_spatial_refined_scales(base_scales, y_hat, params, qp, frame_type_idx, subpass)
            _, y_q, y_part, scales_hat = self.model.process_with_mask(y_scaled, refined, base_means, mask)
            y_hat = y_hat + y_part
            active = mask.to(torch.bool)
            writes.append(y_q.masked_select(active))
            scale_writes.append(scales_hat.masked_select(active))
        return (writes, scale_writes, y_hat * q_dec)

    def _decompress_p_second_pass_spatial_prior(
        self, params: torch.Tensor, spatial_prior: Any, *, qp: int, frame_type_idx: int
    ) -> torch.Tensor:
        q_dec, scales, means = self.model.separate_prior_for_video_decoding(params)
        B, C, H, W = means.shape
        mask_0, mask_1 = self.model.get_mask_2x(B, C, H, W, means.dtype, means.device)

        def decode_part(part_scales: torch.Tensor, part_means: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
            active = mask.to(torch.bool)
            scales_read = (part_scales * mask).masked_select(active)
            symbols = self.model.gaussian_encoder.decode_and_get_y(scales_read, part_means.dtype, part_means.device)
            residual = torch.zeros_like(part_means).masked_scatter(active, symbols.reshape(-1))
            return residual + part_means * mask

        y_hat = decode_part(scales, means, mask_0)
        base_scales, base_means = spatial_prior(torch.cat((y_hat, params), dim=1)).chunk(2, 1)
        for subpass, mask in enumerate(self._split_second_checkerboard_mask(mask_1)):
            refined = self._p_spatial_refined_scales(base_scales, y_hat, params, qp, frame_type_idx, subpass)
            y_hat = y_hat + decode_part(refined, base_means, mask)
        return y_hat * q_dec

    def _p_group_parameters(
        self,
        *,
        base_scales: torch.Tensor,
        base_means: torch.Tensor,
        y_hat_so_far: torch.Tensor,
        group_index: int,
        pass_index: int,
        qp: int,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        context = self.p_group_context
        if context is None:
            raise RuntimeError("P group parameters requested while disabled.")
        indices = context.group_indices(group_index)
        ordered = context.to_group_order(y_hat_so_far)
        completed = context.split_group_order(ordered)[: int(group_index)]
        scales = base_scales.index_select(1, indices)
        means = base_means.index_select(1, indices)
        scales, adjusted_means, symbol_center = context.adjust_with_symbol_center(
            group_index, scales, means, completed, qp, pass_index
        )
        if context.parameterization == "scale_symbol_center":
            hard_center = torch.round(symbol_center)
            if self.training:
                symbol_center = symbol_center + (hard_center - symbol_center).detach()
            else:
                symbol_center = hard_center
        else:
            symbol_center = torch.zeros_like(means)
        return (indices, scales, means, symbol_center)

    def _train_p_group_prior_2x(
        self,
        y_scaled: torch.Tensor,
        params_for_context: torch.Tensor,
        initial_scales: torch.Tensor,
        initial_means: torch.Tensor,
        *,
        ctx_t: torch.Tensor,
        qp: int,
        frame_type_idx: int,
        ref_y_prev: Optional[torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        context = self.p_group_context
        if context is None:
            raise RuntimeError("P group training requested while disabled.")
        B, C, H, W = y_scaled.shape
        masks = self.model.get_mask_2x(B, C, H, W, y_scaled.dtype, y_scaled.device)
        y_hat_so_far = torch.zeros_like(y_scaled)
        bits_y = y_scaled.new_zeros(())
        group_bits = [y_scaled.new_zeros(()) for _ in context.group_sizes]
        for pass_index, full_mask in enumerate(masks):
            if pass_index == 0:
                base_scales, base_means = (initial_scales, initial_means)
            else:
                cat_params = torch.cat((y_hat_so_far, params_for_context), dim=1)
                base_scales, base_means = self.model.y_spatial_prior(cat_params).chunk(2, 1)
                base_scales, base_means = self._condition_entropy_distribution(
                    base_scales,
                    base_means,
                    frame_type_idx,
                    ctx_t=ctx_t,
                    qp=qp,
                    y_hat_0=y_hat_so_far,
                    ref_y_prev=ref_y_prev,
                )
            for group_index in range(context.group_count):
                indices, scales, means, symbol_center = self._p_group_parameters(
                    base_scales=base_scales,
                    base_means=base_means,
                    y_hat_so_far=y_hat_so_far,
                    group_index=group_index,
                    pass_index=pass_index,
                    qp=qp,
                )
                y_group = y_scaled.index_select(1, indices)
                mask = full_mask.index_select(1, indices)
                means_hat = means * mask
                scales_hat = scales * mask
                y_res = (y_group - means_hat) * mask
                y_q_hard, y_for_bits = _uf_quantize_train(y_res, hard_rate=self.p_hard_rate_training)
                y_q_hard = y_q_hard * mask
                y_for_bits = y_for_bits * mask
                group_hat = y_q_hard + means_hat
                previous = y_hat_so_far.index_select(1, indices)
                y_hat_so_far = y_hat_so_far.index_copy(1, indices, previous + group_hat)
                group_rate = _get_y_gaussian_bits(
                    y_for_bits - symbol_center * mask, scales_hat, mask=mask, coder_aligned=True
                ).sum()
                bits_y = bits_y + group_rate
                group_bits[group_index] = group_bits[group_index] + group_rate
        self._last_p_group_soft_bits = tuple((float(value.detach().item()) for value in group_bits))
        return (y_hat_so_far, bits_y)

    def _compress_p_group_prior_2x(
        self, y: torch.Tensor, params: torch.Tensor, spatial_prior: Any, *, qp: int
    ) -> tuple[list[torch.Tensor], list[torch.Tensor], torch.Tensor]:
        context = self.p_group_context
        if context is None:
            raise RuntimeError("P group compression requested while disabled.")
        y_scaled, q_dec, scales, means = self.model.separate_prior_for_video_encoding(params, y)
        B, C, H, W = y_scaled.shape
        masks = self.model.get_mask_2x(B, C, H, W, y_scaled.dtype, y_scaled.device)
        y_hat_so_far = torch.zeros_like(y_scaled)
        symbol_writes: list[torch.Tensor] = []
        scale_writes: list[torch.Tensor] = []
        for pass_index, full_mask in enumerate(masks):
            if pass_index == 0:
                base_scales, base_means = (scales, means)
            else:
                base_scales, base_means = spatial_prior(torch.cat((y_hat_so_far, params), dim=1)).chunk(2, 1)
            for group_index in range(context.group_count):
                indices, group_scales, group_means, symbol_center = self._p_group_parameters(
                    base_scales=base_scales,
                    base_means=base_means,
                    y_hat_so_far=y_hat_so_far,
                    group_index=group_index,
                    pass_index=pass_index,
                    qp=qp,
                )
                y_group = y_scaled.index_select(1, indices)
                mask = full_mask.index_select(1, indices)
                _, y_q, y_hat, scales_hat = self.model.process_with_mask(y_group, group_scales, group_means, mask)
                previous = y_hat_so_far.index_select(1, indices)
                y_hat_so_far = y_hat_so_far.index_copy(1, indices, previous + y_hat)
                active = mask.to(dtype=torch.bool)
                centered_symbol = y_q - symbol_center * mask
                symbol_writes.append(centered_symbol.masked_select(active))
                scale_writes.append(scales_hat.masked_select(active))
        return (symbol_writes, scale_writes, y_hat_so_far * q_dec)

    def _decompress_p_group_prior_2x(self, params: torch.Tensor, spatial_prior: Any, *, qp: int) -> torch.Tensor:
        context = self.p_group_context
        if context is None:
            raise RuntimeError("P group decompression requested while disabled.")
        q_dec, scales, means = self.model.separate_prior_for_video_decoding(params)
        B, C, H, W = means.shape
        masks = self.model.get_mask_2x(B, C, H, W, means.dtype, means.device)
        y_hat_so_far = torch.zeros_like(means)
        for pass_index, full_mask in enumerate(masks):
            if pass_index == 0:
                base_scales, base_means = (scales, means)
            else:
                base_scales, base_means = spatial_prior(torch.cat((y_hat_so_far, params), dim=1)).chunk(2, 1)
            for group_index in range(context.group_count):
                indices, group_scales, group_means, symbol_center = self._p_group_parameters(
                    base_scales=base_scales,
                    base_means=base_means,
                    y_hat_so_far=y_hat_so_far,
                    group_index=group_index,
                    pass_index=pass_index,
                    qp=qp,
                )
                mask = full_mask.index_select(1, indices)
                active = mask.to(dtype=torch.bool)
                scales_read = (group_scales * mask).masked_select(active)
                y_q_read = self.model.gaussian_encoder.decode_and_get_y(
                    scales_read, group_means.dtype, group_means.device
                )
                residual = torch.zeros_like(group_means).masked_scatter(active, y_q_read.reshape(-1))
                residual = residual + symbol_center * mask
                group_hat = residual + group_means * mask
                previous = y_hat_so_far.index_select(1, indices)
                y_hat_so_far = y_hat_so_far.index_copy(1, indices, previous + group_hat)
        return y_hat_so_far * q_dec

    def encode_train(
        self,
        x: torch.Tensor,
        qp: int,
        ctx: Optional[torch.Tensor] = None,
        ctx_t: Optional[torch.Tensor] = None,
        z_prev: Optional[torch.Tensor] = None,
        feature_prev: Optional[torch.Tensor] = None,
        feature_blend: float = 1.0,
        frame_type_idx: int = 0,
        dpb_feature_prev: Optional[torch.Tensor] = None,
        ref_y_prev: Optional[torch.Tensor] = None,
        analysis_feature_hint: Optional[torch.Tensor] = None,
        motion_relation: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Differentiable P-frame encode. Returns (z_hat, bpp)."""
        self._active_task938_qp = int(qp)
        B, _, H, W = x.shape
        input_h, input_w = (H, W)
        motion_bits = x.new_zeros(())
        effective_z_prev = z_prev
        if motion_relation is not None:
            if z_prev is None:
                raise ValueError("Task452 motion relation requires z_prev")
            if ctx is not None or ctx_t is not None:
                raise ValueError("Task452 motion relation must recompute ctx/ctx_t from its decoder-closed reference")
            motion_symbols, motion_bits = self._task452_motion_symbols_train(motion_relation, qp)
            effective_z_prev = self._task452_motion_reference(
                z_prev, motion_symbols, qp=qp, frame_type_idx=frame_type_idx
            )
        if ctx is None and effective_z_prev is not None:
            ctx, ctx_t = self.get_temporal_context(
                effective_z_prev,
                qp,
                feature_prev=feature_prev,
                feature_blend=feature_blend,
                frame_type_idx=frame_type_idx,
                dpb_feature_prev=dpb_feature_prev,
            )
        if ctx is None:
            ctx = torch.zeros(B, _RT_D, H, W, device=x.device, dtype=x.dtype)
        if ctx_t is None:
            ctx_t = torch.zeros(B, _RT_D, H, W, device=x.device, dtype=x.dtype)
        pred = torch.zeros_like(x)
        encode_input = x
        encode_input = self._reference_translation_analysis(encode_input, effective_z_prev, qp, frame_type_idx)
        encode_input = self._shared_query_motion_analysis(encode_input, effective_z_prev, qp, frame_type_idx)
        encode_input = self._condition_p_input(encode_input, frame_type_idx)
        encode_input = self._latent_lifting_analysis(encode_input, qp, frame_type_idx, reference=effective_z_prev)
        qe = self._get_quant_step(self.qe_log_inter, qp)
        qe = self._condition_quant_step(qe, self.p_type_qe_delta, frame_type_idx)
        y_raw = self._encode_input(
            encode_input,
            ctx,
            qe,
            qp,
            frame_type_idx,
            analysis_feature_hint=analysis_feature_hint,
            reference_latent=effective_z_prev,
        )
        y_pred = self._predict_y_from_context(ctx, qp, y_raw)
        y = y_raw
        y_gain = self._y_gain(qp, frame_type_idx, y)
        y = y * y_gain
        y = self._y_balance(qp, frame_type_idx, y)
        y_frequency_source = y
        y = self._frequency_analysis(y, ctx_t, qp, frame_type_idx)
        self._last_p_frequency_aux = {}
        B, C, H, W = y.shape
        y_pad = self.model.pad_for_y(y)
        z = self.model.hyper_encoder(y_pad)
        z_pred = self._predict_z_from_context(ctx_t, qp, frame_type_idx, z)
        z_residual = z - z_pred
        z_residual_hat_hard, z_residual_for_bits = _uf_quantize_train(z_residual, hard_rate=self.p_hard_rate_training)
        z_hat_hard = z_residual_hat_hard + z_pred
        params = self.model.res_prior_param_decoder(z_hat_hard, ctx_t)
        params = self._match_prior_params(params, y)
        q_dec_raw, scales, means = params.chunk(3, 1)
        detach_dynamic_q = bool(getattr(self, "_detach_dynamic_q_for_training", False))
        params_for_context = _detach_leading_prior_channels(params, q_dec_raw.shape[1], enabled=detach_dynamic_q)
        q_dec_input = q_dec_raw.detach() if detach_dynamic_q else q_dec_raw
        q_dec = self._effective_p_q_dec(q_dec_input, qp, frame_type_idx)
        q_dec_detached = q_dec.detach()
        self._last_q_dec_mean = float(q_dec_detached.mean().item())
        self._last_q_dec_gt4_fraction = float((q_dec_detached > 4.0).float().mean().item())
        self._last_q_dec_gt10_fraction = float((q_dec_detached > 10.0).float().mean().item())
        y_scaled = (
            _task938_p_normalize(y, q_dec) if getattr(self, "task938_region_quantizer", None) is not None else y / q_dec
        )
        scales, means = self._condition_entropy_distribution(
            scales, means, frame_type_idx, ctx_t=ctx_t, qp=qp, ref_y_prev=ref_y_prev
        )
        if self.p_group_context is not None:
            y_hat_unscaled, bits_y = self._train_p_group_prior_2x(
                y_scaled,
                params_for_context,
                scales,
                means,
                ctx_t=ctx_t,
                qp=qp,
                frame_type_idx=frame_type_idx,
                ref_y_prev=ref_y_prev,
            )
            y_hat = y_hat_unscaled * q_dec
        elif self.p_second_pass_spatial_refiner is not None:
            y_hat_unscaled, bits_y = self._train_p_second_pass_spatial_prior(
                y_scaled,
                params_for_context,
                scales,
                means,
                ctx_t=ctx_t,
                qp=qp,
                frame_type_idx=frame_type_idx,
                ref_y_prev=ref_y_prev,
            )
            y_hat = y_hat_unscaled * q_dec
        else:
            mask_0, mask_1 = self.model.get_mask_2x(B, C, H, W, y.dtype, y.device)
            width_0 = _task938_effective_width(self, scales, means)
            scales_hat_0 = scales * mask_0
            means_hat_0 = means * mask_0
            y_res_0 = (y_scaled - means_hat_0) * mask_0
            y_q_hard_0, y_for_bits_0 = _task938_quantize_train(
                y_res_0, self, hard_rate=self.p_hard_rate_training, scales=scales, mask=mask_0, width=width_0
            )
            y_q_hard_0 = y_q_hard_0 * mask_0
            y_for_bits_0 = y_for_bits_0 * mask_0
            y_hat_0 = y_q_hard_0 + means_hat_0
            cat_params = torch.cat((y_hat_0, params_for_context), dim=1)
            scales_1, means_1 = self.model.y_spatial_prior(cat_params).chunk(2, 1)
            scales_1, means_1 = self._condition_entropy_distribution(
                scales_1, means_1, frame_type_idx, ctx_t=ctx_t, qp=qp, y_hat_0=y_hat_0, ref_y_prev=ref_y_prev
            )
            width_1 = _task938_effective_width(self, scales_1, means_1)
            scales_hat_1 = scales_1 * mask_1
            means_hat_1 = means_1 * mask_1
            y_res_1 = (y_scaled - means_hat_1) * mask_1
            y_q_hard_1, y_for_bits_1 = _task938_quantize_train(
                y_res_1, self, hard_rate=self.p_hard_rate_training, scales=scales_1, mask=mask_1, width=width_1
            )
            y_q_hard_1 = y_q_hard_1 * mask_1
            y_for_bits_1 = y_for_bits_1 * mask_1
            y_hat_1 = y_q_hard_1 + means_hat_1
            y_hat = (y_hat_0 + y_hat_1) * q_dec
            bits_y = _get_y_gaussian_bits(y_for_bits_0, scales_hat_0 / width_0, mask=mask_0, coder_aligned=True).sum()
            bits_y = (
                bits_y
                + _get_y_gaussian_bits(y_for_bits_1, scales_hat_1 / width_1, mask=mask_1, coder_aligned=True).sum()
            )
        if getattr(self, "task938_region_quantizer", None) is not None:
            _task938_publish_prior(
                self, [(scales / width_0).detach(), (scales_1 / width_1).detach()], [width_0, width_1], [mask_0, mask_1]
            )
        self._last_entropy_y_hat = y_hat
        y_hat = self._frequency_synthesis(y_hat, ctx_t, qp, frame_type_idx)
        y_hat = self._y_balance_inverse(qp, frame_type_idx, y_hat)
        y_hat = y_hat / y_gain
        bits_z = _get_z_bits(z_residual_for_bits, self.model.bit_estimator_z, qp)
        pix = float(self.bpp_spatial_factor * self.bpp_spatial_factor)
        num_pixels = float(B * H * W * 4) * pix
        bpp_y = bits_y / max(num_pixels, 1.0)
        bpp_z = bits_z.sum() / max(num_pixels, 1.0)
        bpp_motion = motion_bits / max(num_pixels, 1.0)
        bpp = bpp_y + bpp_z + bpp_motion
        self._last_bpp_y = float(bpp_y.detach().item())
        self._last_bpp_z = float(bpp_z.detach().item())
        self._last_bpp_motion = float(bpp_motion.detach().item())
        qd = self._get_quant_step(self.qd_log_inter, qp)
        qd = self._condition_quant_step(qd, self.p_type_qd_delta, frame_type_idx)
        qr = self._get_quant_step(self.qr_log_inter, qp)
        qr = self._condition_quant_step(qr, self.p_type_qr_delta, frame_type_idx)
        z_out, feature_out = self._decode_output(
            y_hat, ctx, qd, qr, return_feature=True, reference_latent=effective_z_prev, p_core_qp=qp
        )
        self._last_decoded_feature = feature_out
        z_out = self._reference_translation_synthesis(z_out, effective_z_prev, qp, frame_type_idx)
        z_out = self._shared_query_motion_synthesis(z_out, effective_z_prev, qp, frame_type_idx)
        z_out = self._latent_lifting_synthesis(z_out, qp, frame_type_idx, reference=effective_z_prev)
        return (z_out, bpp)

    def encode(
        self,
        x: torch.Tensor,
        qp: int,
        ctx: Optional[torch.Tensor] = None,
        ctx_t: Optional[torch.Tensor] = None,
        z_prev: Optional[torch.Tensor] = None,
        feature_prev: Optional[torch.Tensor] = None,
        feature_blend: float = 1.0,
        frame_type_idx: int = 0,
        dpb_feature_prev: Optional[torch.Tensor] = None,
        ref_y_prev: Optional[torch.Tensor] = None,
        analysis_feature_hint: Optional[torch.Tensor] = None,
        motion_relation: Optional[torch.Tensor] = None,
    ) -> Tuple[bytes, torch.Tensor]:
        """Real rANS P-frame encode. Returns (bitstream, reconstructed latent)."""
        self._active_task938_qp = int(qp)
        self._ensure_entropy_coder()
        motion_symbols_write = None
        effective_z_prev = z_prev
        if motion_relation is not None:
            if z_prev is None:
                raise ValueError("Task452 motion relation requires z_prev")
            if ctx is not None or ctx_t is not None:
                raise ValueError("Task452 motion relation must recompute ctx/ctx_t from its decoder-closed reference")
            if self.p_motion_relation_bit_estimator is None:
                raise RuntimeError("Task452 motion relation codec is not enabled")
            steps = motion_relation.new_tensor(
                [
                    self.p_motion_relation_flow_step,
                    self.p_motion_relation_flow_step,
                    self.p_motion_relation_confidence_step,
                ]
            ).view(1, 3, 1, 1)
            motion_symbols, motion_symbols_write = round_and_to_int8(motion_relation / steps)
            effective_z_prev = self._task452_motion_reference(
                z_prev, motion_symbols, qp=qp, frame_type_idx=frame_type_idx
            )
        if (ctx is None or ctx_t is None) and effective_z_prev is not None:
            ctx, ctx_t = self.get_temporal_context(
                effective_z_prev,
                qp,
                feature_prev=feature_prev,
                feature_blend=feature_blend,
                frame_type_idx=frame_type_idx,
                dpb_feature_prev=dpb_feature_prev,
            )
        if ctx is None or ctx_t is None:
            raise ValueError("DMCAdapter.encode requires ctx and ctx_t (or use pipeline).")
        B, _, H, W = x.shape
        pred = torch.zeros_like(x)
        encode_input = x
        encode_input = self._reference_translation_analysis(encode_input, effective_z_prev, qp, frame_type_idx)
        encode_input = self._shared_query_motion_analysis(encode_input, effective_z_prev, qp, frame_type_idx)
        encode_input = self._condition_p_input(encode_input, frame_type_idx)
        encode_input = self._latent_lifting_analysis(encode_input, qp, frame_type_idx, reference=effective_z_prev)
        qe = self._get_quant_step(self.qe_log_inter, qp)
        qe = self._condition_quant_step(qe, self.p_type_qe_delta, frame_type_idx)
        y_raw = self._encode_input(
            encode_input,
            ctx,
            qe,
            qp,
            frame_type_idx,
            analysis_feature_hint=analysis_feature_hint,
            reference_latent=effective_z_prev,
        )
        y_pred = self._predict_y_from_context(ctx, qp, y_raw)
        y = y_raw
        y_gain = self._y_gain(qp, frame_type_idx, y)
        y = y * y_gain
        y = self._y_balance(qp, frame_type_idx, y)
        y = self._frequency_analysis(y, ctx_t, qp, frame_type_idx)
        B, C, H, W = y.shape
        y_pad = self.model.pad_for_y(y)
        z = self.model.hyper_encoder(y_pad)
        z_pred = self._predict_z_from_context(ctx_t, qp, frame_type_idx, z)
        z_residual_hat, z_hat_write = round_and_to_int8(z - z_pred)
        z_hat = z_residual_hat + z_pred
        params_raw = self.model.res_prior_param_decoder(z_hat, ctx_t)
        params_raw = self._match_prior_params(params_raw, y)
        params = self._condition_common_entropy_params(
            params_raw, frame_type_idx, ctx_t=ctx_t, qp=qp, ref_y_prev=ref_y_prev
        )
        params = self._apply_hard_q_policy_to_params(params, qp=qp, frame_type_idx=frame_type_idx)
        spatial_prior = lambda value: self._hard_spatial_prior(
            value, frame_type_idx, ctx_t=ctx_t, qp=qp, raw_common_params=params_raw, ref_y_prev=ref_y_prev
        )
        if self.p_group_context is not None:
            y_q_writes, scale_writes, y_hat = self._compress_p_group_prior_2x(y, params, spatial_prior, qp=qp)
        elif self.p_second_pass_spatial_refiner is not None:
            y_q_writes, scale_writes, y_hat = self._compress_p_second_pass_spatial_prior(
                y, params, spatial_prior, qp=qp, frame_type_idx=frame_type_idx
            )
        else:
            if getattr(self, "_task938_quantizer", None) is not None and self._task938_quantizer.enabled:
                result = _task938_p_prior(self, params, spatial_prior, qp, y=y)
            else:
                result = self.model.compress_prior_2x(y, params, spatial_prior)
            y_q_w_0, y_q_w_1, s_w_0, s_w_1, y_hat = result
            y_q_writes = [y_q_w_0, y_q_w_1]
            scale_writes = [s_w_0, s_w_1]
        symbol_count = sum((int(value.numel()) for value in y_q_writes))
        nonzero_count = sum((int(torch.count_nonzero(value).detach().cpu().item()) for value in y_q_writes))
        self._last_y_symbol_nonzero_fraction = float(nonzero_count) / float(symbol_count) if symbol_count else 0.0
        if self._capture_exact_entropy_symbols:
            self._last_entropy_symbol_tensors = {
                "z": z_hat_write.detach().cpu().clone(),
                "y": tuple((value.detach().cpu().clone() for value in y_q_writes)),
                "y_scales": tuple((value.detach().cpu().clone() for value in scale_writes)),
                "motion": None if motion_symbols_write is None else motion_symbols_write.detach().cpu().clone(),
            }
        else:
            self._last_entropy_symbol_tensors = {}
        self.model.entropy_coder.reset()
        if motion_symbols_write is not None:
            assert self.p_motion_relation_bit_estimator is not None
            self.p_motion_relation_bit_estimator.encode_z(motion_symbols_write, qp)
        self.model.bit_estimator_z.encode_z(z_hat_write, qp)
        for y_q_write, scale_write in zip(y_q_writes, scale_writes):
            self.model.gaussian_encoder.encode_y(y_q_write, scale_write)
        self.model.entropy_coder.flush()
        bit_stream = self.model.entropy_coder.get_encoded_stream()
        self._last_entropy_y_hat = y_hat
        y_hat = self._frequency_synthesis(y_hat, ctx_t, qp, frame_type_idx)
        y_hat = self._y_balance_inverse(qp, frame_type_idx, y_hat)
        y_hat = y_hat / y_gain
        qd = self._get_quant_step(self.qd_log_inter, qp)
        qd = self._condition_quant_step(qd, self.p_type_qd_delta, frame_type_idx)
        qr = self._get_quant_step(self.qr_log_inter, qp)
        qr = self._condition_quant_step(qr, self.p_type_qr_delta, frame_type_idx)
        z_out, feature_out = self._decode_output(
            y_hat, ctx, qd, qr, return_feature=True, reference_latent=effective_z_prev, p_core_qp=qp
        )
        self._last_decoded_feature = feature_out
        z_out = self._reference_translation_synthesis(z_out, effective_z_prev, qp, frame_type_idx)
        z_out = self._shared_query_motion_synthesis(z_out, effective_z_prev, qp, frame_type_idx)
        z_out = self._latent_lifting_synthesis(z_out, qp, frame_type_idx, reference=effective_z_prev)
        buf = io.BytesIO()
        torch.save(
            {
                "codec": "dmc",
                "bit_stream": bit_stream,
                "qp": qp,
                "shape": list(x.shape),
                "codec_contract_version": 6,
                "p_motion_relation_codec": self.p_motion_relation_bit_estimator is not None,
                "p_motion_relation_present": motion_symbols_write is not None,
                "p_motion_context_adapter": self.p_motion_context_adapter is not None,
                "p_motion_context_contract": (
                    None
                    if self.p_motion_context_adapter is None
                    else {
                        "qp_position_conditioned": self.p_motion_context_adapter.qp_position_conditioned,
                        "bounded_output": self.p_motion_context_adapter.bounded_output,
                    }
                ),
                "p_motion_relation_flow_step": self.p_motion_relation_flow_step,
                "p_motion_relation_confidence_step": self.p_motion_relation_confidence_step,
                "p_motion_relation_cache_size": self.p_motion_relation_cache_size,
                "p_decoded_dpb_mode": self.p_decoded_dpb_mode,
                "p_decoded_y_scale_only": self.p_decoded_y_scale_only,
                "p_decoded_y_mean_max_qp": int(self.p_decoded_y_mean_max_qp),
                "p_decoded_y_current_pass": bool(self.p_decoded_y_current_pass),
                "residual_mode": self.residual_mode,
                "explicit_residual_scale": self._explicit_scale(frame_type_idx),
                "frame_type_idx": int(frame_type_idx),
                "q_dec_policy": "native",
                **self._entropy_q_contract_payload(),
                "p_y_gain_enabled": self.p_y_gain_enabled,
                "p_y_balance_enabled": self.p_y_balance_enabled,
                "p_spatial_prior_raw_context": True,
                "p_z_context_enabled": self.p_z_context_enabled,
                "p_group_context_contract": self._p_group_contract_payload(),
                "p_second_pass_spatial_contract": self._p_second_pass_spatial_contract_payload(),
                "p_frequency_compaction_contract": self._frequency_contract_payload(),
                **(
                    {"p_transform_capacity_contract": self._p_transform_capacity_contract_payload()}
                    if self._p_transform_capacity_contract_payload() is not None
                    else {}
                ),
                "p_local_reference_transform_contract": self._p_local_reference_transform_contract_payload(),
                "p_reference_affine_lifting_contract": self._p_reference_affine_lifting_contract_payload(),
                "p_context_first_core_contract": self._p_context_first_contract_payload(),
                "p_invertible_latent_lifting_contract": self._p_invertible_latent_lifting_contract_payload(),
                "p_reference_translation_lifting_contract": self._p_reference_translation_lifting_contract_payload(),
                "p_y_conditioned_context_warp_contract": self._p_y_conditioned_context_warp_contract_payload(),
                "p_shared_query_motion_residual_contract": self._p_shared_query_motion_contract_payload(),
                "p_shared_latent_lifting_contract": self._p_shared_latent_lifting_contract_payload(),
                "p_mature_context_coupling_contract": self._p_mature_context_coupling_contract_payload(),
                "p_context_native_deep_contract": self._p_context_native_deep_contract_payload(),
                "p_continuous_display_contract": self._continuous_display_contract_payload(),
                "p_htcm_contract": self._p_htcm_contract_payload(),
                "p_recurrent_synthesis_contract": self._p_recurrent_synthesis_contract_payload(),
            },
            buf,
        )
        return (buf.getvalue(), z_out.detach())

    def _decompress_native_prior_2x_export(self, params: torch.Tensor, spatial_prior: Any) -> torch.Tensor:
        """Native two-pass decoder with a decoder-known scale-grid export."""
        q_dec, scales0, means0 = self.model.separate_prior_for_video_decoding(params)
        B, C, H, W = means0.shape
        masks = self.model.get_mask_2x(B, C, H, W, means0.dtype, means0.device)
        scales1_read = self.model.single_part_for_writing_2x(scales0 * masks[0])
        y_q0 = self.model.gaussian_encoder.decode_and_get_y(scales1_read, means0.dtype, means0.device)
        y_hat0 = restore_y_2x(y_q0, means0, masks[0])
        next_scales, next_means = spatial_prior(torch.cat((y_hat0, params), dim=1)).chunk(2, 1)
        scales2_read = self.model.single_part_for_writing_2x(next_scales * masks[1])
        y_q1 = self.model.gaussian_encoder.decode_and_get_y(scales2_read, next_means.dtype, next_means.device)
        y_hat1 = restore_y_2x(y_q1, next_means, masks[1])
        self._last_entropy_scale_map = _aggregate_decoder_entropy_scales((scales0, next_scales), masks)
        return (y_hat0 + y_hat1) * q_dec

    def decode(
        self,
        bitstream: bytes,
        ctx: Optional[torch.Tensor] = None,
        ctx_t: Optional[torch.Tensor] = None,
        z_prev: Optional[torch.Tensor] = None,
        feature_prev: Optional[torch.Tensor] = None,
        feature_blend: float = 1.0,
        frame_type_idx: Optional[int] = None,
        dpb_feature_prev: Optional[torch.Tensor] = None,
        ref_y_prev: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Decode a DMC bitstream to latent."""
        self._ensure_entropy_coder()
        payload = torch.load(io.BytesIO(bitstream), map_location="cpu", weights_only=False)
        bit_stream = payload["bit_stream"]
        qp = payload["qp"]
        residual_mode = str(payload.get("residual_mode", self.residual_mode)).lower()
        bitstream_q_policy = str(payload.get("q_dec_policy", "legacy")).lower()
        self._validate_entropy_q_contract(payload)
        self._validate_p_group_contract(payload.get("p_group_context_contract"))
        self._validate_p_second_pass_spatial_contract(payload.get("p_second_pass_spatial_contract"))
        validate_p_frequency_contract(
            payload.get("p_frequency_compaction_contract"), self._frequency_contract_payload()
        )
        self._validate_p_transform_capacity_contract(payload.get("p_transform_capacity_contract"))
        self._validate_p_local_reference_transform_contract(payload.get("p_local_reference_transform_contract"))
        self._validate_p_reference_affine_lifting_contract(payload.get("p_reference_affine_lifting_contract"))
        self._validate_p_context_first_contract(payload.get("p_context_first_core_contract"))
        self._validate_p_invertible_latent_lifting_contract(payload.get("p_invertible_latent_lifting_contract"))
        self._validate_p_reference_translation_lifting_contract(payload.get("p_reference_translation_lifting_contract"))
        self._validate_p_y_conditioned_context_warp_contract(payload.get("p_y_conditioned_context_warp_contract"))
        self._validate_p_shared_query_motion_contract(payload.get("p_shared_query_motion_residual_contract"))
        self._validate_p_shared_latent_lifting_contract(payload.get("p_shared_latent_lifting_contract"))
        self._validate_p_mature_context_coupling_contract(payload.get("p_mature_context_coupling_contract"))
        self._validate_p_context_native_deep_contract(payload.get("p_context_native_deep_contract"))
        self._validate_continuous_display_contract(payload.get("p_continuous_display_contract"))
        self._validate_p_htcm_contract(payload.get("p_htcm_contract"))
        self._validate_p_recurrent_synthesis_contract(payload.get("p_recurrent_synthesis_contract"))
        bitstream_y_gain = bool(payload.get("p_y_gain_enabled", False))
        if bitstream_y_gain != self.p_y_gain_enabled:
            raise ValueError(
                f"P bitstream y-gain contract mismatch: stream={bitstream_y_gain} decoder={self.p_y_gain_enabled}."
            )
        bitstream_y_balance = bool(payload.get("p_y_balance_enabled", False))
        if bitstream_y_balance != self.p_y_balance_enabled:
            raise ValueError(
                f"P bitstream y-balance contract mismatch: stream={bitstream_y_balance} decoder={self.p_y_balance_enabled}."
            )
        contract_version = int(payload.get("codec_contract_version", 1))
        encoded_dpb_mode = str(payload.get("p_decoded_dpb_mode", "off")).strip().lower()
        if encoded_dpb_mode != self.p_decoded_dpb_mode:
            raise ValueError(
                f"P bitstream decoded-DPB contract mismatch: stream={encoded_dpb_mode} decoder={self.p_decoded_dpb_mode}."
            )
        encoded_scale_only = bool(payload.get("p_decoded_y_scale_only", False))
        if encoded_scale_only != self.p_decoded_y_scale_only:
            raise ValueError(
                f"P decoded-y scale-only contract mismatch: stream={encoded_scale_only} decoder={self.p_decoded_y_scale_only}."
            )
        encoded_mean_max_qp = int(payload.get("p_decoded_y_mean_max_qp", _INTER_QP_NUM - 1))
        if encoded_mean_max_qp != int(self.p_decoded_y_mean_max_qp):
            raise ValueError(
                f"P decoded-y mean-QP contract mismatch: stream={encoded_mean_max_qp} decoder={self.p_decoded_y_mean_max_qp}."
            )
        encoded_current_pass = bool(payload.get("p_decoded_y_current_pass", False))
        if encoded_current_pass != bool(self.p_decoded_y_current_pass):
            raise ValueError(
                f"P decoded-y current-pass contract mismatch: stream={encoded_current_pass} decoder={self.p_decoded_y_current_pass}."
            )
        bitstream_raw_context = (
            True if contract_version >= 2 else bool(payload.get("p_spatial_prior_raw_context", False))
        )
        bitstream_z_context = bool(payload.get("p_z_context_enabled", False))
        if bitstream_z_context != self.p_z_context_enabled:
            raise ValueError(
                f"P bitstream z-context contract mismatch: stream={bitstream_z_context} decoder={self.p_z_context_enabled}."
            )
        if frame_type_idx is None:
            frame_type_idx = int(payload.get("frame_type_idx", 0))
        bitstream_motion_codec = bool(payload.get("p_motion_relation_codec", False))
        decoder_motion_codec = self.p_motion_relation_bit_estimator is not None
        if bitstream_motion_codec != decoder_motion_codec:
            raise ValueError(
                f"P bitstream motion-relation contract mismatch: stream={bitstream_motion_codec} decoder={decoder_motion_codec}."
            )
        bitstream_motion_adapter = bool(payload.get("p_motion_context_adapter", False))
        decoder_motion_adapter = self.p_motion_context_adapter is not None
        if bitstream_motion_adapter != decoder_motion_adapter:
            raise ValueError(
                f"P bitstream motion-context adapter mismatch: stream={bitstream_motion_adapter} decoder={decoder_motion_adapter}."
            )
        if bitstream_motion_adapter:
            assert self.p_motion_context_adapter is not None
            expected_context_contract = {
                "qp_position_conditioned": self.p_motion_context_adapter.qp_position_conditioned,
                "bounded_output": self.p_motion_context_adapter.bounded_output,
            }
            if payload.get("p_motion_context_contract") != expected_context_contract:
                raise ValueError(
                    f"P bitstream motion-context contract mismatch: stream={payload.get('p_motion_context_contract')} decoder={expected_context_contract}."
                )
        motion_present = bool(payload.get("p_motion_relation_present", False))
        if motion_present:
            expected_motion_contract = (
                self.p_motion_relation_flow_step,
                self.p_motion_relation_confidence_step,
                self.p_motion_relation_cache_size,
            )
            stream_motion_contract = (
                float(payload.get("p_motion_relation_flow_step", -1.0)),
                float(payload.get("p_motion_relation_confidence_step", -1.0)),
                int(payload.get("p_motion_relation_cache_size", -1)),
            )
            if stream_motion_contract != expected_motion_contract:
                raise ValueError(
                    f"P bitstream motion-relation quantization mismatch: stream={stream_motion_contract} decoder={expected_motion_contract}."
                )
        B, C, H, W = payload["shape"]
        assert C == self.latent_channels, f"DMCAdapter.decode expects {self.latent_channels} channels, got {C}"
        y_h, y_w, z_h, z_w = self._bitstream_shapes(H, W)
        device = next(self.model.parameters()).device
        dtype = next(self.model.parameters()).dtype
        self.model.entropy_coder.set_use_two_entropy_coders(False)
        self.model.entropy_coder.set_stream(bit_stream)
        effective_z_prev = z_prev
        if motion_present:
            if z_prev is None:
                raise ValueError("Task452 motion relation requires z_prev")
            if ctx is not None or ctx_t is not None:
                raise ValueError("Task452 motion relation must recompute ctx/ctx_t from its decoder-closed reference")
            assert self.p_motion_relation_bit_estimator is not None
            self.p_motion_relation_bit_estimator.decode_z((2, 2), qp)
            motion_symbols = self.p_motion_relation_bit_estimator.get_z((2, 2), device, dtype)
            effective_z_prev = self._task452_motion_reference(
                z_prev, motion_symbols, qp=qp, frame_type_idx=frame_type_idx
            )
        if (ctx is None or ctx_t is None) and effective_z_prev is not None:
            ctx, ctx_t = self.get_temporal_context(
                effective_z_prev,
                qp,
                feature_prev=feature_prev,
                feature_blend=feature_blend,
                frame_type_idx=frame_type_idx,
                dpb_feature_prev=dpb_feature_prev,
            )
        if ctx is None or ctx_t is None:
            raise ValueError("DMCAdapter.decode requires ctx and ctx_t (or use pipeline).")
        self.model.bit_estimator_z.decode_z((z_h, z_w), qp)
        z_residual_hat = self.model.bit_estimator_z.get_z((z_h, z_w), device, dtype)
        z_pred = self._predict_z_from_context(ctx_t, qp, frame_type_idx, z_residual_hat)
        z_hat = z_residual_hat + z_pred
        params_raw = self.model.res_prior_param_decoder(z_hat, ctx_t)
        if params_raw.shape[-2:] != (y_h, y_w):
            params_raw = F.interpolate(params_raw, size=(y_h, y_w), mode="bilinear", align_corners=False).contiguous()
        params = self._condition_common_entropy_params(
            params_raw, frame_type_idx, ctx_t=ctx_t, qp=qp, ref_y_prev=ref_y_prev
        )
        params = self._apply_hard_q_policy_to_params(
            params, qp=qp, frame_type_idx=frame_type_idx, stream_policy=bitstream_q_policy
        )
        spatial_prior = lambda value: self._hard_spatial_prior(
            value,
            frame_type_idx,
            ctx_t=ctx_t,
            qp=qp,
            raw_common_params=params_raw if bitstream_raw_context else None,
            ref_y_prev=ref_y_prev,
        )
        if self.p_group_context is not None:
            y_hat = self._decompress_p_group_prior_2x(params, spatial_prior, qp=qp)
        elif self.p_second_pass_spatial_refiner is not None:
            y_hat = self._decompress_p_second_pass_spatial_prior(
                params, spatial_prior, qp=qp, frame_type_idx=frame_type_idx
            )
        elif getattr(self, "_task938_quantizer", None) is not None and self._task938_quantizer.enabled:
            y_hat = _task938_p_prior(self, params, spatial_prior, qp)
        elif getattr(self, "_export_entropy_scales", False):
            y_hat = self._decompress_native_prior_2x_export(params, spatial_prior)
        else:
            infos = self.model.decompress_prior_2x_part1(params)
            y_hat = self.model.decompress_prior_2x_part2(params, spatial_prior, infos)
        self._last_entropy_y_hat = y_hat
        y_hat = self._frequency_synthesis(y_hat, ctx_t, qp, frame_type_idx)
        y_hat = self._y_balance_inverse(qp, frame_type_idx, y_hat)
        y_hat = y_hat / self._y_gain(qp, frame_type_idx, y_hat)
        if residual_mode == "y_centered":
            y_pred = self._predict_y_from_context(ctx, qp, y_hat, residual_mode=residual_mode)
            y_hat = y_hat + y_pred
        qd = self._get_quant_step(self.qd_log_inter, qp)
        qd = self._condition_quant_step(qd, self.p_type_qd_delta, frame_type_idx)
        qr = self._get_quant_step(self.qr_log_inter, qp)
        qr = self._condition_quant_step(qr, self.p_type_qr_delta, frame_type_idx)
        z_out, feature_out = self._decode_output(
            y_hat, ctx, qd, qr, return_feature=True, reference_latent=effective_z_prev, p_core_qp=qp
        )
        self._last_decoded_feature = feature_out
        explicit_residual_scale = float(payload.get("explicit_residual_scale", self.explicit_residual_scale))
        explicit_residual_scale = max(explicit_residual_scale, 1e-06)
        if residual_mode == "explicit":
            pred = self._predict_from_reference(effective_z_prev, ctx, qp, z_out, frame_type_idx)
            z_out = pred + z_out / explicit_residual_scale
        elif residual_mode == "residual_blend":
            pred = self._residual_blend_prediction(effective_z_prev, frame_type_idx, z_out)
            z_out = pred + z_out
        elif residual_mode == "hybrid_refine":
            z_out = self._refine_reconstruction(z_out, effective_z_prev, ctx, qp, residual_mode)
        z_out = self._reference_translation_synthesis(z_out, effective_z_prev, qp, frame_type_idx)
        z_out = self._shared_query_motion_synthesis(z_out, effective_z_prev, qp, frame_type_idx)
        z_out = self._latent_lifting_synthesis(z_out, qp, frame_type_idx, reference=effective_z_prev)
        if z_out.shape[-2] > H or z_out.shape[-1] > W:
            z_out = z_out[:, :, :H, :W]
        return z_out.detach()


class RTTwoCodecPipeline(nn.Module, LatentCodec):
    """Two-codec RT-style pipeline with shared latent normalization."""

    def enable_entropy_scale_export(self, enabled: bool = True) -> None:
        """Opt in to decoder-known per-frame Gaussian scale export.

        The result is exposed after :meth:`decode` through
        ``_last_entropy_scale_sequence`` as ``[B,1,T,H_y,W_y]``.  It is not
        serialized and does not alter the default bitstream path.
        """
        flag = bool(enabled)
        self._export_entropy_scales = flag
        for codec in (self.i_codec, self.p_codec, self.p1_codec):
            if codec is not None:
                codec._export_entropy_scales = flag
        if not flag:
            self._last_entropy_scale_sequence = None
            self._last_quantization_width_sequence = None
            for module in self.modules():
                if hasattr(module, "_last_entropy_scale_map"):
                    module._last_entropy_scale_map = None
                if hasattr(module, "_last_quantization_width_map"):
                    module._last_quantization_width_map = None

    def task938_quantizer_contract(self):
        contract = self._task938_quantizer.contract()
        regions = {}
        for name, module in self.named_modules():
            if name.endswith("task938_region_quantizer"):
                digest = hashlib.sha256()
                for key, value in sorted(module.state_dict().items()):
                    digest.update(key.encode())
                    digest.update(value.detach().float().cpu().contiguous().numpy().tobytes())
                regions[name] = {
                    "group_size": module.group_size,
                    "max_log": module.max_log,
                    "weights_sha256": digest.hexdigest(),
                }
        if regions:
            contract = dict(contract, schema="task938_regional_quantization_v3", version=3, regions=regions)
        return contract

    def get_last_quantization_width_sequence(self):
        value = getattr(self, "_last_quantization_width_sequence", None)
        if value is None:
            raise RuntimeError("no decoder quantization-width sequence")
        return value

    def get_last_entropy_scale_sequence(self) -> torch.Tensor:
        value = getattr(self, "_last_entropy_scale_sequence", None)
        if value is None:
            raise RuntimeError(
                "No entropy scale sequence is available; call enable_entropy_scale_export() before decode()."
            )
        return value

    _SEMANTIC_HEADER_BYTES = 10

    def __init__(
        self,
        device: str = "cuda:0",
        image_checkpoint_path: Optional[str] = None,
        video_checkpoint_path: Optional[str] = None,
    ):
        normalize_latent = False
        p_residual_mode = "raw"
        p_residual_predictor_hidden = 96
        p_residual_predictor_use_context = True
        p_residual_predictor_arch = "legacy"
        p_residual_predictor_blocks = 6
        p_explicit_residual_scale = 1.0
        p_explicit_residual_scales = None
        p_refiner_hidden = 128
        p_refiner_use_context = True
        p_refiner_residual_scale = 0.1
        p_y_context_hidden = 192
        p_reference_mode = "latent"
        p_feature_blend = 1.0
        p_hierarchical_reference_enabled = False
        p_hierarchical_quality_enabled = False
        split_z1_separate_i_codec = False
        p_type_conditioning = True
        p_type_count = 3
        p_type_input_scale = 0.1
        p_type_quant_scale = 0.1
        p_entropy_scale_calibration = False
        p_entropy_scale_max_log = 1.0
        p_entropy_mean_calibration = False
        p_entropy_mean_max_delta = 0.5
        p_entropy_q_mode = "channel_scale"
        p_hard_rate_training = True
        task938_nested_quantization_enabled = True
        task938_nested_quantization_fine_step = 1.0
        task938_nested_quantization_drop_bits_by_qp = (0, 0, 1, 1)
        task938_quantization_widths_by_qp = [1.0, 1.0, 1.0, 1.0]
        task938_regional_group_size = 16
        task938_regional_max_log = 0.6931471805599453
        p_channel_q_max_log = 2.5
        p_position_q_gain_enabled = False
        p_position_q_gain_max_log = 0.25
        p_base_allocator_mode = "off"
        p_base_allocator_position_count = 6
        p_base_allocator_rank = 4
        p_base_allocator_max_log = 0.25
        p_base_allocator_importance_path = None
        p_progressive_canvas_enabled = False
        p_progressive_canvas_group_sizes = None
        p_progressive_canvas_max_log = 0.2
        p_progressive_rate_bias_max_log = 0.2
        p_factorized_q_spatial_max_log = 0.25
        p_factorized_q_mix = 1.0
        p_adaptive_entropy_enabled = True
        p_adaptive_entropy_hidden = 256
        p_adaptive_entropy_max_log_scale = 1.5
        p_adaptive_entropy_max_mean_delta = 2.0
        p_decoded_dpb_mode = "off"
        p_decoded_dpb_hidden = 96
        p_decoded_dpb_blocks = 2
        p_decoded_dpb_max_feature_delta = 0.5
        p_decoded_dpb_max_log_scale = 0.5
        p_decoded_dpb_max_mean_delta = 0.5
        p_htcm_enabled = False
        p_htcm_hidden = 128
        p_htcm_blocks = 3
        p_htcm_max_delta = 0.5
        p_htcm_memory_mode = "correct"
        p1_context_adapter_enabled = False
        p1_context_adapter_hidden = 64
        p1_context_adapter_scale = 0.1
        p_residual_blend_init = 0.02
        p_residual_blend_max = 1.0
        p_residual_blend_typewise = True
        p_y_gain_enabled = False
        p_y_gain_max_log = 1.0
        p_y_balance_enabled = False
        p_y_balance_max_log = 0.5
        p_y_balance_max_bias = 0.5
        p_synthesis_head_enabled = True
        p_synthesis_head_hidden = 256
        p_synthesis_head_scale = 0.25
        p_synthesis_head_out_of_loop = True
        p_synthesis_head_min_position = 1
        p_synthesis_position_conditioned = True
        p_synthesis_position_count = 6
        p_synthesis_position_cycle = True
        p_synthesis_position_hidden = 128
        p_synthesis_qp_conditioned = False
        p_synthesis_qp_values = None
        p_synthesis_qp_hidden = 128
        p_synthesis_qp_position_factorized = False
        p_synthesis_head_arch = "legacy"
        p_continuous_display_blocks = 4
        p_continuous_display_max_position = 24
        p_continuous_display_reference_mode = "correct"
        p_reference_state_head_enabled = False
        p_reference_state_bptt = True
        i_reference_bptt_enabled = True
        p_z_context_enabled = False
        p_z_context_hidden = 192
        p_z_context_blocks = 4
        p_z_context_scale = 1.0
        p_frequency_compaction_enabled = False
        p_frequency_compaction_hidden = 160
        p_frequency_compaction_blocks = 3
        p_frequency_compaction_residual_scale = 0.25
        p_frequency_compaction_group_sizes = None
        p_frequency_compaction_mode = "residual"
        p_frequency_compaction_aux_enabled = False
        p_analysis_core_hidden = 0
        p_synthesis_core_hidden = 0
        p_core_blocks = 2
        p_core_residual_scale = 1.0
        i_y_gain_enabled = False
        i_y_gain_max_log = 1.0
        i_y_balance_enabled = False
        i_y_balance_max_log = 0.5
        i_y_balance_max_bias = 0.5
        i_y_coupling_enabled = False
        i_y_coupling_hidden = 256
        i_y_coupling_blocks = 4
        i_y_coupling_max_log_scale = 0.35
        i_y_coupling_max_shift = 1.0
        i_adaptive_y_gain_enabled = False
        i_adaptive_y_gain_hidden = 128
        i_adaptive_y_gain_max_log = 0.12
        i_post_y_gain_enabled = False
        i_post_y_gain_max_log = 0.35
        i_synthesis_head_enabled = False
        i_synthesis_head_hidden = 192
        i_synthesis_head_scale = 0.1
        i_synthesis_head_arch = "multiscale_reference"
        i_synthesis_head_blocks = 4
        i_synthesis_head_out_of_loop = False
        i_entropy_q_mode = "dynamic"
        i_group_context_lane = "legacy"
        i_group_context_parameterization = "scale"
        i_group_context_hidden = 96
        i_group_context_max_log_scale = 0.75
        i_group_context_max_mean_delta = 2.0
        i_group_context_permutation = None
        i_transform_channels = 512
        i_y_channels = 320
        i_z_channels = 160
        i_analysis_core_hidden = 0
        i_synthesis_core_hidden = 0
        i_core_blocks = 2
        i_core_residual_scale = 1.0
        i_progressive_transform_enabled = False
        i_progressive_transform_stages = 6
        i_progressive_transform_max_angle = 0.35
        i_progressive_q_enabled = False
        i_progressive_group_sizes = None
        i_progressive_q_max_group_log = 0.35
        i_progressive_q_max_channel_log = 0.2
        i_progressive_q_center_groups = False
        i_hierarchical_transform_enabled = False
        i_hierarchical_hidden = 256
        i_hierarchical_blocks = 3
        i_hierarchical_residual_scale = 0.25
        i_hierarchical_aux_enabled = False
        p_codec_arch = "dcvc_rt"
        p1_separate_codec = False
        p_type_separate_codecs = False
        latent_init_mode = "scratch"
        latent_stem_blocks = 4
        latent_spatial_mode = "dcvc_stride2"
        latent_preserve_blocks = 1
        codec_proxy_mode = "none"
        latent_channels = 48
        latent_canvas_protection_enabled = False
        latent_canvas_importance_path = None
        latent_canvas_max_log_gain = 0.35
        latent_canvas_temporal_log_gain = None
        bpp_spatial_factor = 32
        semantic_anchor_enabled = False
        semantic_anchor_analysis_hidden = 96
        semantic_anchor_analysis_blocks = 3
        semantic_anchor_synthesis_hidden = 128
        semantic_anchor_synthesis_blocks = 4
        semantic_anchor_detail_hidden = 128
        semantic_anchor_detail_blocks = 4
        semantic_anchor_analysis_scale = 0.25
        semantic_anchor_synthesis_scale = 0.25
        semantic_anchor_detail_scale = 0.25
        semantic_anchor_predict_detail = True
        type_aware_norm = True
        type_norm_scales = [0.6591, 0.8553, 0.9778]
        type_norm_biases = [0.0, 0.0, 0.0]
        type_norm_learnable = False
        anchor_context_mode = "reference_residual"
        anchor_context_hidden = 192
        anchor_context_scale = 0.25
        anchor_context_use_type = True
        anchor_context_multiscale_enabled = False
        anchor_context_multiscale_hidden = 192
        anchor_context_multiscale_blocks = 4
        anchor_context_multiscale_scale = 0.25
        ring_anchor_required_enabled = False
        ring_anchor_hidden = 128
        ring_anchor_blocks = 3
        ring_anchor_max_mix = 0.5
        ring_anchor_initial_mix = 0.05
        ring_anchor_start_position = 4
        ring_anchor_normal_probability = 0.5
        ring_anchor_required_probability = 0.35
        ring_anchor_negative_probability = 0.15
        ring_anchor_block_size = 4
        ring_anchor_topk_ratio = 0.2
        ring_anchor_benefit_margin = 0.02
        p_reference_reliability_enabled = False
        p_second_order_reference_enabled = False
        p_second_order_motion_alignment_enabled = False
        p_second_order_display_enabled = False
        latent_tail_compand = False
        tail_compand_scale = 2.0
        tail_compand_clamp = 8.0
        pixel_postfilter_enabled = False
        pixel_postfilter_hidden = 32
        pixel_postfilter_blocks = 6
        pixel_postfilter_residual_scale = 0.1
        latent_postfilter_enabled = False
        latent_postfilter_hidden = 128
        latent_postfilter_blocks = 8
        latent_postfilter_residual_scale = 0.25
        latent_postfilter_position_experts_enabled = False
        latent_postfilter_position_count = 7
        latent_postfilter_position_residual_scale = 0.1
        latent_postfilter_qp_position_experts_enabled = False
        latent_postfilter_qp_values = (0, 16, 39, 63)
        latent_postfilter_qp_position_residual_scale = 0.1
        cell_grid_smoother_enabled = False
        cell_grid_smoother_gate_hidden = 16
        cell_grid_smoother_stats_kernel = 3
        cell_grid_smoother_max_strength = 1.0
        conditioned_pixel_refiner_enabled = False
        conditioned_pixel_refiner_condition_hidden = 64
        conditioned_pixel_refiner_condition_channels = 32
        conditioned_pixel_refiner_condition_blocks = 4
        conditioned_pixel_refiner_pixel_hidden = 48
        conditioned_pixel_refiner_pixel_blocks = 6
        conditioned_pixel_refiner_residual_scale = 0.1
        pyramid_pixel_refiner_enabled = False
        pyramid_pixel_refiner_condition_hidden = 128
        pyramid_pixel_refiner_condition_blocks = 4
        pyramid_pixel_refiner_hidden = 192
        pyramid_pixel_refiner_blocks_per_scale = 2
        pyramid_pixel_refiner_residual_scale = 0.1
        super().__init__()
        self._device = device
        self._qp = None
        self._rate_qp_table = None
        self._temporal_state = None
        self._hierarchical_reference_states: dict[int, dict] = {}
        self._p_htcm_long_feature_state = None
        self._anchor_state = None
        self._ring_anchor_state = None
        self._ring_anchor_training_target = None
        self._normalize_latent = bool(False)
        self.latent_channels = int(48)
        self.latent_canvas_protection_enabled = bool(False)
        self.latent_canvas_protection = None
        self.bpp_spatial_factor = int(32)
        self._semantic_anchor_predict_detail = bool(True)
        self.semantic_anchor = None
        self._p_reference_mode = str("latent").lower()
        self._p_feature_blend = float(1.0)
        self._p_hierarchical_reference_enabled = bool(False)
        self._p_hierarchical_quality_enabled = bool(False)
        self._p_decoded_dpb_mode = str("off").strip().lower()
        self._p_htcm_enabled = bool(False)
        self._p_synthesis_head_out_of_loop = bool(True)
        self._i_synthesis_head_out_of_loop = bool(False)
        self._p_synthesis_head_min_position = int(1)
        self._p_synthesis_position_count = int(6)
        self._p_synthesis_position_cycle = bool(True)
        self._p_synthesis_head_arch = str("legacy").lower().strip()
        self._p_reference_state_head_enabled = bool(False)
        self._p_reference_state_bptt = bool(True)
        self._i_reference_bptt_enabled = bool(True)
        if self._p_synthesis_head_out_of_loop and str("raw").lower() != "raw":
            raise ValueError("p_synthesis_head_out_of_loop currently requires p_residual_mode='raw'.")
        self._split_z1_separate_i_codec = bool(False)
        self._p1_separate_codec = bool(False)
        self._p_type_separate_codecs = bool(False)
        self._p_codec_arch = str("dcvc_rt").lower().strip()
        self._latent_init_mode = _normalize_latent_init_mode("scratch")
        self._latent_stem_blocks = int(4)
        self._latent_spatial_mode = _normalize_latent_spatial_mode("dcvc_stride2")
        self._latent_preserve_blocks = max(1, int(1))
        self._codec_proxy_mode = str("none").lower()
        self._anchor_context_mode = str("reference_residual").lower()
        self._anchor_context_multiscale_enabled = bool(False)
        self._p_second_order_reference_enabled = bool(False)
        self._p_second_order_motion_alignment_enabled = bool(False)
        self._p_second_order_display_enabled = bool(False)
        self._latent_tail_compand = bool(False)
        self._tail_compand_clamp = float(8.0)
        self.num_qp = _NUM_QP
        self.i_codec = DMCIAdapter(checkpoint_path=image_checkpoint_path, device=device)
        self.z1_i_codec = None
        self.p_codec = DMCAdapter(checkpoint_path=video_checkpoint_path, device=device)
        self.p1_codec = None
        self.p_type_codecs = None
        self._task938_quantizer = DyadicScalarQuantizer(
            enabled=bool(True),
            fine_step=float(1.0),
            drop_bits_by_qp=tuple(task938_nested_quantization_drop_bits_by_qp or (0, 0, 1, 1)),
            widths_by_qp=task938_quantization_widths_by_qp,
        )
        _codec_instances = [self.i_codec, self.p_codec]
        if self._task938_quantizer.enabled:
            for _codec in _codec_instances:
                for _feature in ("i_group_context", "p_group_context", "p_second_pass_spatial_refiner"):
                    if getattr(_codec, _feature, None) is not None:
                        raise ValueError("Task938 quantization unsupported with " + _feature)
        for _codec in _codec_instances:
            _codec._task938_quantizer = self._task938_quantizer
            _codec._active_task938_qp = 0
            if not self._task938_quantizer.enabled:
                raise ValueError("regional quantization requires Task938 quantizer enabled")
            from raqvc.codec.task938_regional_quantizer import RegionalQuantizer

            _codec.task938_region_quantizer = RegionalQuantizer(group_size=16, max_log=0.6931471805599453).to(
                device=device, dtype=torch.float32
            )
        self.latent_scale = nn.Parameter(torch.ones(1, self.latent_channels, 1, 1, device=device))
        self.latent_bias = nn.Parameter(torch.zeros(1, self.latent_channels, 1, 1, device=device))
        self.codec_proxy = None
        self.anchor_context_adapter = None
        self.anchor_context_adapter = AnchorBroadcastAdapter(
            channels=self.latent_channels,
            hidden_channels=int(192),
            residual_scale=float(0.25),
            use_type=bool(True),
            multiscale_enabled=bool(False),
            multiscale_hidden=int(192),
            multiscale_blocks=int(4),
            multiscale_scale=float(0.25),
        ).to(device)
        self.ring_anchor_required_mixer = None
        self.p_reference_reliability_mixer = None
        self._p_reference_reliability_enabled = bool(False)
        self.second_order_reference_adapter = None
        self.second_order_display_adapter = None
        self.third_order_reference_adapter = None
        self._p_third_order_reference_enabled = False
        self._last_third_order_reference_delta = 0.0
        self.second_order_position_gain = None
        self._p_second_order_position_gain_enabled = False
        self.second_order_reliability_gate = None
        self._p_second_order_reliability_gate_enabled = False
        self.p_reference_state_head = None
        self.p_display_tokenizer_calibrator = None
        self._last_p_display_tokenizer: dict[str, Any] = {}
        self._capture_exact_p_entropy_symbols = False
        self.register_buffer(
            "tail_compand_scale",
            torch.full((1, self.latent_channels, 1, 1), float(2.0), dtype=torch.float32, device=device),
        )
        with torch.no_grad():
            self.latent_scale.fill_(1.0)
            self.latent_bias.fill_(0.0)
        self.latent_scale.requires_grad_(False)
        self.latent_bias.requires_grad_(False)
        self._type_aware_norm = bool(True)
        self._type_norm_learnable = bool(False)
        scales = type_norm_scales
        biases = type_norm_biases
        scale_init = _expand_type_channel_values(
            scales, latent_channels=self.latent_channels, field_name="type_norm_scales"
        ).to(device)
        bias_init = _expand_type_channel_values(
            biases, latent_channels=self.latent_channels, field_name="type_norm_biases"
        ).to(device)
        if torch.any(scale_init <= 0):
            raise ValueError("type_norm_scales must be strictly positive for an invertible whitening transform.")
        self.register_buffer("type_norm_scale", scale_init)
        self.register_buffer("type_norm_bias", bias_init)
        self.pixel_postfilter = None
        self.latent_postfilter = None
        self.cell_grid_smoother = None
        self.conditioned_pixel_refiner = None
        self.pyramid_pixel_refiner = None

    def _semantic_analysis(self, latent: torch.Tensor) -> tuple[torch.Tensor, tuple[int, int] | None]:
        """Map a full Wan latent to the native half-grid entropy domain."""
        semantic_anchor = None
        return (latent, None)

    def _semantic_reconstruct(
        self, decoded_payload: torch.Tensor, original_size: tuple[int, int] | None
    ) -> torch.Tensor:
        """Return a decoder-only full-grid reconstruction."""
        semantic_anchor = None
        if original_size is not None:
            raise ValueError("semantic size metadata present while module is disabled")
        return decoded_payload

    def _annotate_semantic_bitstream(self, bitstream: bytes, original_size: tuple[int, int] | None) -> bytes:
        semantic_anchor = None
        return bitstream

    def postprocess_latents(
        self, latent: torch.Tensor, qp: int, *, gradient_checkpointing: bool = False
    ) -> torch.Tensor:
        """Apply optional GOP refinement to ``[C,T,H,W]`` or ``[B,C,T,H,W]``."""
        return latent

    def postprocess_pixels(
        self,
        video: torch.Tensor,
        *,
        latent_condition: Optional[torch.Tensor] = None,
        qp: Optional[int] = None,
        frame_batch_size: int = 1,
        gradient_checkpointing: bool = False,
    ) -> torch.Tensor:
        """Apply the optional filter to ``[C,T,H,W]`` pixels in ``[-1,1]``.

        Frames are processed independently in small batches so native 1080p
        evaluation stays within one-GPU memory.  This operation is strictly
        out-of-loop and therefore cannot change payload bits.
        """
        return video

    def _type_norm_index(self, t: int) -> int:
        """Latent-position type index: 0=I0, 1=P1, 2+=P2plus (decoder-known)."""
        return max(0, min(int(t), int(self.type_norm_scale.shape[0]) - 1))

    def _apply_type_norm(self, z: torch.Tensor, t: int) -> torch.Tensor:
        idx = self._type_norm_index(t)
        scale = self.type_norm_scale[idx]
        return (z - self.type_norm_bias[idx]) / scale

    def _apply_type_denorm(self, z: torch.Tensor, t: int) -> torch.Tensor:
        idx = self._type_norm_index(t)
        scale = self.type_norm_scale[idx]
        return z * scale + self.type_norm_bias[idx]

    def _type_norm_fingerprint(self) -> str:
        """Identify the exact decoder-side affine contract in a bitstream."""
        digest = hashlib.sha256()
        digest.update(b"raqvc-type-channel-whitening-v1\x00")
        for tensor in (self.type_norm_bias, self.type_norm_scale):
            value = tensor.detach().to(device="cpu", dtype=torch.float32).contiguous()
            digest.update(value.numpy().tobytes())
        return digest.hexdigest()[:24]

    def _validate_type_norm_fingerprint(self, payload: dict) -> None:
        """Reject new bitstreams decoded with different whitening constants.

        Historical payloads did not carry this field and remain readable.  All
        newly encoded payloads carry it, so a separate decoder cannot silently
        apply the wrong inverse transform.
        """
        encoded = payload.get("type_norm_fingerprint")
        if encoded is None:
            return
        expected = self._type_norm_fingerprint()
        if str(encoded) != expected:
            raise ValueError(
                f"Bitstream/type-channel-whitening mismatch: encoded={encoded!r}, decoder={expected!r}. Load the exact codec checkpoint/config used by the encoder."
            )

    def _latent_canvas_contract(self) -> Optional[dict[str, Any]]:
        module = self.latent_canvas_protection
        return None if module is None else module.contract()

    def _validate_latent_canvas_contract(self, payload: dict) -> None:
        observed = payload.get("latent_canvas_protection_contract")
        expected = self._latent_canvas_contract()
        if observed != expected:
            raise ValueError(
                f"Bitstream/latent-canvas-protection mismatch: encoded={observed!r}, decoder={expected!r}."
            )

    def _apply_tail_compand(self, z: torch.Tensor) -> torch.Tensor:
        return z

    def _apply_tail_decompand(self, z: torch.Tensor) -> torch.Tensor:
        return z

    def _codec_input_transform(self, z: torch.Tensor, t: int) -> torch.Tensor:
        value = self._apply_type_norm(z, t)
        return self._apply_tail_compand(value)

    def _codec_output_transform(self, z: torch.Tensor, t: int) -> torch.Tensor:
        value = self._apply_tail_decompand(z)
        return self._apply_type_denorm(value, t)

    def _select_i_codec(self, role: str = "default") -> DMCIAdapter:
        if role == "z1" and self.z1_i_codec is not None:
            return self.z1_i_codec
        return self.i_codec

    def _select_p_codec(self, frame_type_idx: int = 0, role: Optional[str] = None) -> DMCAdapter:
        if (role == "p1" or int(frame_type_idx) == 0) and self.p1_codec is not None:
            return self.p1_codec
        return self.p_codec

    def _proxy_analysis(self, z: torch.Tensor) -> torch.Tensor:
        return z

    def _proxy_synthesis(self, z: torch.Tensor) -> torch.Tensor:
        return z

    def enable_task452_third_order_reference(
        self, *, hidden_channels: int = 96, blocks: int = 2, residual_scale: float = 0.05, position_count: int = 13
    ) -> None:
        """Enable a zero-init, decoder-known three-reference correction."""
        if self.third_order_reference_adapter is not None:
            expected = (
                self.third_order_reference_adapter.stem.out_channels,
                len(self.third_order_reference_adapter.blocks),
                self.third_order_reference_adapter.residual_scale,
                self.third_order_reference_adapter.position_count,
            )
            requested = (int(hidden_channels), int(blocks), float(residual_scale), int(position_count))
            if expected != requested:
                raise RuntimeError(f"third-order reference already enabled as {expected}, requested {requested}")
            return
        self.third_order_reference_adapter = ThirdOrderReferenceAdapter(
            channels=self.latent_channels,
            hidden_channels=int(hidden_channels),
            blocks=int(blocks),
            residual_scale=float(residual_scale),
            position_count=int(position_count),
        ).to(self._device)
        self._p_third_order_reference_enabled = True

    def enable_task452_second_order_position_gain(self, *, position_count: int = 13, max_delta: float = 0.5) -> None:
        """Enable exact-position calibration without changing stream syntax."""
        raise RuntimeError("second-order position gain requires second-order reference")

    def enable_task452_second_order_reliability_gate(
        self, *, hidden_channels: int = 64, blocks: int = 2, position_count: int = 13, max_delta: float = 0.5
    ) -> None:
        """Enable local confidence gating without changing stream syntax."""
        raise RuntimeError("second-order reliability gate requires second-order reference")

    def _anchor_reference(
        self, z_prev: torch.Tensor, qp: int, *, pos_type_idx: int, frame_type_idx: int
    ) -> torch.Tensor:
        z_ref = z_prev
        z_anchor = z_prev
        if self._anchor_state is not None:
            z_anchor = self._proxy_analysis(self._codec_input_transform(self._anchor_state, pos_type_idx))
            if z_anchor.shape[-2:] != z_prev.shape[-2:]:
                z_anchor = F.interpolate(z_anchor, size=z_prev.shape[-2:], mode="bilinear", align_corners=False)
        if self.anchor_context_adapter is not None and self._anchor_state is not None:
            z_ref = self.anchor_context_adapter(z_prev, z_anchor, qp, frame_type_idx=frame_type_idx)
            with torch.no_grad():
                self._last_anchor_context_delta = float((z_ref - z_prev).pow(2).mean().sqrt().detach().item())
        self._last_ring_anchor_diagnostics = {}
        if (
            self.ring_anchor_required_mixer is not None
            and self._ring_anchor_state is not None
            and (int(pos_type_idx) >= self.ring_anchor_required_mixer.start_position)
        ):
            p3_anchor = self._proxy_analysis(self._codec_input_transform(self._ring_anchor_state, int(pos_type_idx)))
            if p3_anchor.shape[-2:] != z_ref.shape[-2:]:
                p3_anchor = F.interpolate(p3_anchor, size=z_ref.shape[-2:], mode="bilinear", align_corners=False)
            z_ref = self.ring_anchor_required_mixer(
                z_ref,
                p3_anchor,
                qp=int(qp),
                position=int(pos_type_idx),
                target=self._ring_anchor_training_target,
                negative_anchor=z_anchor,
            )
            self._last_ring_anchor_diagnostics = dict(self.ring_anchor_required_mixer.last_diagnostics)
        self._last_second_order_reference_delta = 0.0
        self._last_third_order_reference_delta = 0.0
        state = self._temporal_state
        if (
            self.second_order_reference_adapter is not None
            and state is not None
            and (int(state.get("history_depth", 1)) >= 2)
            and (state.get("z_prev2") is not None)
        ):
            z_prev2 = self._proxy_analysis(self._codec_input_transform(state["z_prev2"], pos_type_idx))
            if z_prev2.shape[-2:] != z_ref.shape[-2:]:
                z_prev2 = F.interpolate(z_prev2, size=z_ref.shape[-2:], mode="bilinear", align_corners=False)
            second_order_base = z_ref
            second_order_ref = self.second_order_reference_adapter(
                z_ref, z_prev2, z_anchor, qp, frame_type_idx=frame_type_idx
            )
            if self.second_order_position_gain is not None:
                second_order_ref = self.second_order_position_gain(
                    second_order_base, second_order_ref, qp, pos_type_idx
                )
            if self.second_order_reliability_gate is not None:
                second_order_ref = self.second_order_reliability_gate(
                    second_order_base, second_order_ref, z_prev2, z_anchor, qp, pos_type_idx
                )
            with torch.no_grad():
                self._last_second_order_reference_delta = float(
                    (second_order_ref - z_ref).pow(2).mean().sqrt().detach().item()
                )
            z_ref = second_order_ref
        if (
            self.third_order_reference_adapter is not None
            and state is not None
            and (int(state.get("history_depth", 1)) >= 3)
            and (state.get("z_prev2") is not None)
            and (state.get("z_prev3") is not None)
        ):
            z_prev2 = self._proxy_analysis(self._codec_input_transform(state["z_prev2"], pos_type_idx))
            z_prev3 = self._proxy_analysis(self._codec_input_transform(state["z_prev3"], pos_type_idx))
            if z_prev2.shape[-2:] != z_ref.shape[-2:]:
                z_prev2 = F.interpolate(z_prev2, size=z_ref.shape[-2:], mode="bilinear", align_corners=False)
            if z_prev3.shape[-2:] != z_ref.shape[-2:]:
                z_prev3 = F.interpolate(z_prev3, size=z_ref.shape[-2:], mode="bilinear", align_corners=False)
            third_order_ref = self.third_order_reference_adapter(z_ref, z_prev2, z_prev3, z_anchor, qp, pos_type_idx)
            with torch.no_grad():
                self._last_third_order_reference_delta = float(
                    (third_order_ref - z_ref).pow(2).mean().sqrt().detach().item()
                )
            z_ref = third_order_ref
        return z_ref

    def _normalize(self, z: torch.Tensor) -> torch.Tensor:
        return z

    def _denormalize(self, z: torch.Tensor) -> torch.Tensor:
        return z

    def reset_temporal_state(self) -> None:
        self._temporal_state = None
        self._hierarchical_reference_states = {}
        self._p_htcm_long_feature_state = None
        self._anchor_state = None
        self._ring_anchor_state = None
        self._ring_anchor_training_target = None
        self._last_ring_anchor_diagnostics = {}
        seen: set[int] = set()
        recurrent_codecs = [
            getattr(self, name, None) for name in ("p_codec", "p1_codec", "p_codec_p1", "p_codec_p2", "p_codec_p3")
        ]
        recurrent_codecs.extend(list(None or ()))
        for codec in recurrent_codecs:
            if codec is None or id(codec) in seen:
                continue
            seen.add(id(codec))
            if hasattr(codec, "reset_p_recurrent_synthesis_state"):
                codec.reset_p_recurrent_synthesis_state()

    def _hierarchical_quality_qp(self, base_qp: int, position_idx: int) -> int:
        """Map one operating point to decoder-known layer quality banks.

        Only the four Task297-calibrated banks are used.  Lower-layer anchors
        receive finer quantization and higher-layer disposable frames receive
        coarser quantization, coupling reference and quality topology without
        side information.
        """
        return int(base_qp)

    def _activate_temporal_reference(self, position_idx: int) -> None:
        """Select the hierarchical short reference before coding one P."""
        return

    def _commit_temporal_state(self, state: dict, *, is_intra: bool, position_idx: int) -> None:
        """Commit one decoder-known state to the sequential/hierarchical DPB."""
        self._temporal_state = state
        if self.ring_anchor_required_mixer is not None and (not is_intra) and (int(position_idx) == 3):
            self._ring_anchor_state = state["z_out"].detach()
        return

    def _temporal_feature_ref(self) -> Optional[torch.Tensor]:
        if self._p_reference_mode not in {"feature", "blend"} or self._temporal_state is None:
            return None
        return self._temporal_state.get("feature")

    def _temporal_dpb_feature_ref(self) -> Optional[torch.Tensor]:
        if self._p_decoded_dpb_mode not in {"feature", "full"} or self._temporal_state is None:
            return None
        return self._temporal_state.get("dpb_feature")

    def _temporal_dpb_y_ref(self) -> Optional[torch.Tensor]:
        if self._p_decoded_dpb_mode not in {"y", "full"} or self._temporal_state is None:
            return None
        return self._temporal_state.get("dpb_y")

    def _temporal_feature_blend(self) -> float:
        return 0.0

    def _make_temporal_state(
        self, z_out: torch.Tensor, *, is_intra: bool, p_codec: Optional[DMCAdapter] = None
    ) -> dict:
        keep_state_graph = bool(
            self.training and (self._p_reference_state_bptt or (is_intra and self._i_reference_bptt_enabled))
        )
        current = z_out if keep_state_graph else z_out.detach()
        previous_state = self._temporal_state
        if is_intra or previous_state is None:
            previous = current
            previous2 = current
            history_depth = 1
        else:
            previous_raw = previous_state.get("z_out", z_out)
            previous = previous_raw if keep_state_graph else previous_raw.detach()
            previous2_raw = previous_state.get("z_prev2", previous_raw)
            previous2 = previous2_raw if keep_state_graph else previous2_raw.detach()
            history_depth = min(int(previous_state.get("history_depth", 1)) + 1, 3)
        state = {"z_out": current, "z_prev2": previous, "z_prev3": previous2, "history_depth": history_depth}
        if not is_intra:
            feature_codec = p_codec if p_codec is not None else self.p_codec
            feature = getattr(feature_codec, "_last_decoded_feature", None)
        return state

    def _p_reference_codec_output(self, p_codec: DMCAdapter, refined_codec_output: torch.Tensor) -> torch.Tensor:
        """Return the reconstruction that is allowed into the prediction loop."""
        reference = getattr(p_codec, "_last_pre_synthesis_reconstruction", None)
        if reference is None:
            raise RuntimeError(
                "Out-of-loop P synthesis requested but the P decoder did not publish its pre-synthesis reconstruction."
            )
        return reference

    def _i_reference_codec_output(self, i_codec: DMCIAdapter, refined_codec_output: torch.Tensor) -> torch.Tensor:
        """Return the I reconstruction allowed into the causal P loop.

        Task268 showed that optimizing a single enhanced I tensor against an
        I->P objective can lower P rate by reshaping the reference while
        degrading displayed I quality.  The out-of-loop contract separates
        those responsibilities: the quality head remains decoder-visible and
        zero-side-bit, but the stable pre-head I reconstruction is the only
        tensor used by P prediction and the anchor state.
        """
        return refined_codec_output

    def _p_reference_state_codec_output(
        self, p_codec: DMCAdapter, reference_codec_output: torch.Tensor, *, pos_type_idx: int, qp: int
    ) -> torch.Tensor:
        """Restore the decoder-known state used by the following P frame."""
        head = self.p_reference_state_head
        if head is None:
            return reference_codec_output
        feature = getattr(p_codec, "_last_reconstruction_feature", None)
        if feature is None:
            raise RuntimeError("P reference-state head requires the decoded reconstruction feature.")
        if self._anchor_state is None:
            raise RuntimeError("P reference-state head requires a decoded I0 anchor.")
        anchor = self._proxy_analysis(self._codec_input_transform(self._anchor_state, pos_type_idx))
        repaired = head(reference_codec_output, feature, anchor, qp=int(qp), position_idx=int(pos_type_idx))
        self._last_reference_state_rms_delta = float(
            (repaired - reference_codec_output).pow(2).mean().sqrt().detach().item()
        )
        return repaired

    def _p_metric_codec_output(
        self, refined_codec_output: torch.Tensor, reference_codec_output: torch.Tensor, *, pos_type_idx: int
    ) -> torch.Tensor:
        """Gate an out-of-loop quality head by decoder-known GOP position.

        Positions before ``p_synthesis_head_min_position`` use the exact
        pre-head reconstruction.  The recurrent reference already uses that
        reconstruction for every P frame, so this gate changes neither later
        symbols nor payload bits.
        """
        if self._p_synthesis_head_out_of_loop and int(pos_type_idx) < self._p_synthesis_head_min_position:
            return reference_codec_output
        return refined_codec_output

    def _p_display_tokenizer_output(
        self, displayed: torch.Tensor, pre_display: torch.Tensor, p_codec: DMCAdapter, *, qp: int, pos_type_idx: int
    ) -> torch.Tensor:
        """Apply an optional display-only tokenizer calibration.

        All inputs are decoder-known.  In particular, ``pre_display`` is the
        frozen pre-synthesis reconstruction in the common Wan latent domain
        and ``feature`` is the decoder synthesis feature.  The returned tensor
        is used only for the current displayed P frame; callers separately
        retain the unmodified recurrent state.
        """
        calibrator = self.p_display_tokenizer_calibrator
        if calibrator is None:
            self._last_p_display_tokenizer = {}
            return displayed
        feature = getattr(p_codec, "_last_reconstruction_feature", None)
        if feature is None:
            raise RuntimeError("P display tokenizer calibrator requires the decoded reconstruction feature.")
        calibrated, diagnostics = calibrator(
            displayed.detach(), pre_display.detach(), feature.detach(), qp=int(qp), position_idx=int(pos_type_idx)
        )
        self._last_p_display_tokenizer = diagnostics
        return calibrated

    def _second_order_display(
        self, current: torch.Tensor, *, qp: int, pos_type_idx: int, frame_type_idx: int
    ) -> torch.Tensor:
        adapter = self.second_order_display_adapter
        state = self._temporal_state
        self._last_second_order_display_delta = 0.0
        if (
            adapter is None
            or state is None
            or int(state.get("history_depth", 1)) < 2
            or (state.get("z_prev2") is None)
            or (self._anchor_state is None)
        ):
            return current
        values = []
        for raw in (state["z_out"], state["z_prev2"], self._anchor_state):
            value = self._proxy_analysis(self._codec_input_transform(raw, pos_type_idx))
            if value.shape[-2:] != current.shape[-2:]:
                value = F.interpolate(value, size=current.shape[-2:], mode="bilinear", align_corners=False)
            values.append(value)
        repaired = adapter(current, values[0], values[1], values[2], int(qp), int(frame_type_idx))
        with torch.no_grad():
            self._last_second_order_display_delta = float((repaired - current).pow(2).mean().sqrt().detach().item())
        return repaired

    def _set_p_synthesis_condition(self, p_codec: DMCAdapter, *, pos_type_idx: int, qp: int) -> None:
        """Publish decoder-known position/QP to the out-of-loop head.

        Task245 trained six display-only P experts on a reset T7 window.  A
        continuous T13/T25 stream must keep one causal P state across the
        entire sequence, but its display role repeats every six P latents.
        The historical clamp sent every P7+ to expert six.  The opt-in cycle
        keeps entropy/state recurrence continuous while mapping display roles
        P1..P6, P1..P6, ... identically in encode, decode, and training.
        """
        position = int(pos_type_idx)
        if self._p_synthesis_position_cycle and position > 0:
            position = 1 + (position - 1) % self._p_synthesis_position_count
        p_codec._synthesis_position_idx = position
        p_codec._synthesis_qp = int(qp)
        p_codec._active_htcm_long_feature = self._p_htcm_long_feature_state

    @staticmethod
    def _is_dormant_adapted_q_parameter(name: str) -> bool:
        dormant_suffixes = (
            ".model.q_scale_enc",
            ".model.q_scale_dec",
            ".model.q_encoder",
            ".model.q_decoder",
            ".model.q_feature",
            ".model.q_recon",
        )
        return name.endswith(dormant_suffixes)

    def freeze_dormant_adapted_q_parameters(self) -> None:
        """Freeze vendored q tables bypassed by the latent wrapper."""
        for name, parameter in self.named_parameters():
            if self._is_dormant_adapted_q_parameter(name):
                parameter.requires_grad_(False)

    def parameters_for_finetuning(self) -> Iterator[nn.Parameter]:
        """Yield parameters used by the adapted I/P forward paths.

        The vendored DCVC modules retain their original RGB-codec q tables,
        but this wrapper uses the outer ``*_log_intra/inter`` banks instead.
        Keeping the dormant tables in AdamW gave them optimizer state despite
        never producing a gradient and obscured q-head diagnostics.
        """
        for name, parameter in self.named_parameters():
            if self._is_dormant_adapted_q_parameter(name):
                parameter.requires_grad_(False)
                continue
            yield parameter

    def parameters_for_module_bank_training(self) -> Iterator[nn.Parameter]:
        """Adapter / bank parameters only (no pretrained entropy model)."""
        for n, p in self.named_parameters():
            if "i_codec.model." in n or "p_codec.model." in n:
                continue
            yield p

    def load_state_dict(self, state_dict, *args, **kwargs):
        state_dict = _remap_legacy_no_upsample_refine_keys(state_dict)
        had_z1_i = any((str(k).startswith("z1_i_codec.") for k in state_dict.keys()))
        had_p1 = any((str(k).startswith("p1_codec.") for k in state_dict.keys()))
        had_p_type = any((str(k).startswith("p_type_codecs.") for k in state_dict.keys()))
        strict = bool(kwargs.get("strict", args[0] if args else True))
        core_prefixes: list[str] = []
        for module_name in ("p_codec", "p1_codec"):
            module = getattr(self, module_name, None)
            if module is None:
                continue
            for core_name in ("p_analysis_core", "p_synthesis_core"):
                if getattr(module, core_name, None) is not None:
                    core_prefixes.append(f"{module_name}.{core_name}.")
        for index, module in enumerate(self.p_type_codecs or ()):
            for core_name in ("p_analysis_core", "p_synthesis_core"):
                if getattr(module, core_name, None) is not None:
                    core_prefixes.append(f"p_type_codecs.{index}.{core_name}.")
        legacy_core_load = (
            strict and core_prefixes and (not any((str(key).startswith(tuple(core_prefixes)) for key in state_dict)))
        )
        if legacy_core_load:
            call_kwargs = dict(kwargs)
            call_kwargs["strict"] = False
            call_args = args[1:] if args else ()
            result = super().load_state_dict(state_dict, *call_args, **call_kwargs)
            illegal_missing = [key for key in result.missing_keys if not str(key).startswith(tuple(core_prefixes))]
            if illegal_missing or result.unexpected_keys:
                raise RuntimeError(
                    f"Legacy C1 strict load may omit only Task713 P-core keys; missing={illegal_missing[:8]} unexpected={list(result.unexpected_keys)[:8]}"
                )
        else:
            result = super().load_state_dict(state_dict, *args, **kwargs)
        if self.z1_i_codec is not None and (not had_z1_i):
            self.z1_i_codec.load_state_dict(self.i_codec.state_dict(), strict=True)
        if self.p1_codec is not None and (not had_p1):
            self.p1_codec.load_state_dict(self.p_codec.state_dict(), strict=True)
        if self.p_type_codecs is not None and (not had_p_type):
            for codec in self.p_type_codecs:
                codec.load_state_dict(self.p_codec.state_dict(), strict=True)
        return result

    def train(self, mode: bool = True) -> "RTTwoCodecPipeline":
        super().train(mode)
        self.i_codec.train(mode)
        self.p_codec.train(mode)
        return self

    def eval(self) -> "RTTwoCodecPipeline":
        super().eval()
        self.i_codec.eval()
        self.p_codec.eval()
        return self

    def to(self, device: str) -> "RTTwoCodecPipeline":
        super().to(device)
        self._device = device
        self.i_codec.to(device)
        self.p_codec.to(device)
        if self.anchor_context_adapter is not None:
            self.anchor_context_adapter.to(device)
        if self.third_order_reference_adapter is not None:
            self.third_order_reference_adapter.to(device)
        if self.second_order_position_gain is not None:
            self.second_order_position_gain.to(device)
        if self.second_order_reliability_gate is not None:
            self.second_order_reliability_gate.to(device)
        self.latent_scale.data = self.latent_scale.data.to(device)
        self.latent_bias.data = self.latent_bias.data.to(device)
        if hasattr(self, "type_norm_scale"):
            self.type_norm_scale.data = self.type_norm_scale.data.to(device)
        if hasattr(self, "type_norm_bias"):
            self.type_norm_bias.data = self.type_norm_bias.data.to(device)
        if hasattr(self, "tail_compand_scale"):
            self.tail_compand_scale.data = self.tail_compand_scale.data.to(device)
        return self

    def forward(
        self,
        latent: torch.Tensor,
        rate: Optional[float] = None,
        qp: Optional[int] = None,
        *,
        temporal_chunk: bool = False,
        p_rate: Optional[float] = None,
        p_shift_qp: bool = True,
        reset_temporal_state: bool = True,
        split_after_first_latent: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """DDP-friendly differentiable training entrypoint.

        Stage A0 wraps this module in DDP for multi-GPU training. Calling the
        module's ``forward`` keeps the active computation graph on the wrapped
        module while preserving the existing encode_train APIs.
        """
        if temporal_chunk:
            return self.encode_train_temporal_chunk(
                latent,
                rate=rate,
                qp=qp,
                p_rate=p_rate,
                p_shift_qp=p_shift_qp,
                reset_temporal_state=reset_temporal_state,
                split_after_first_latent=split_after_first_latent,
            )
        return self.encode_train(latent, rate=rate, qp=qp)

    def encode_train(
        self, latent: torch.Tensor, rate: Optional[float] = None, qp: Optional[int] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Differentiable intra-only encoding.

        Args:
            latent: Input latent tensor.
            rate: Target bpp used to derive a QP.  Deprecated; prefer ``qp``.
            qp: Explicit QP index.  If provided, ``rate`` is ignored.
        """
        self._last_bpps_per_frame = None
        if qp is None:
            if rate is None:
                raise ValueError("encode_train requires either rate or qp")
            qp = _rate_to_qp(rate, self._rate_qp_table)
        latent = latent.to(self._device)
        latent, semantic_size = self._semantic_analysis(latent)
        latent_norm = self._normalize(latent)
        frames: list[torch.Tensor]
        restore_layout: str
        if latent_norm.ndim == 5:
            B, C, T, H, W = latent_norm.shape
            frames = [latent_norm[:, :, t, :, :] for t in range(T)]
            restore_layout = "bcthw"
        elif latent_norm.ndim == 4 and latent_norm.shape[0] == self.latent_channels:
            C, T, H, W = latent_norm.shape
            frames = [latent_norm[:, t, :, :].unsqueeze(0) for t in range(T)]
            restore_layout = "cthw"
        else:
            frames = [latent_norm]
            restore_layout = "frame"
        reconstructed: list[torch.Tensor] = []
        references: list[torch.Tensor] = []
        bpps: list[torch.Tensor] = []
        bpp_y_values: list[float] = []
        bpp_z_values: list[float] = []
        for t, frame in enumerate(frames):
            frame_codec = self._proxy_analysis(self._codec_input_transform(frame, t))
            z_hat_t, bpp_t = self.i_codec.encode_train(frame_codec, qp)
            reference_t = self._i_reference_codec_output(self.i_codec, z_hat_t)
            reference_t = self._codec_output_transform(self._proxy_synthesis(reference_t), t)
            z_hat_t = self._codec_output_transform(self._proxy_synthesis(z_hat_t), t)
            reconstructed.append(z_hat_t)
            references.append(reference_t)
            bpps.append(bpp_t)
            bpp_y_values.append(float(self.i_codec._last_bpp_y))
            bpp_z_values.append(float(self.i_codec._last_bpp_z))
        if restore_layout == "bcthw":
            z_hat = torch.stack(reconstructed, dim=2)
            reference_hat = torch.stack(references, dim=2)
        elif restore_layout == "cthw":
            z_hat = torch.cat(reconstructed, dim=0).permute(1, 0, 2, 3)
            reference_hat = torch.cat(references, dim=0).permute(1, 0, 2, 3)
        else:
            z_hat = reconstructed[0]
            reference_hat = references[0]
        self._last_bpp_y = 4.0 * sum(bpp_y_values) / len(bpp_y_values)
        self._last_bpp_z = 4.0 * sum(bpp_z_values) / len(bpp_z_values)
        self._last_i_progressive_aux = dict(getattr(self.i_codec, "_last_i_progressive_aux", {}))
        self._last_i_progressive_q_group_log_mean = float(
            getattr(self.i_codec, "_last_i_progressive_q_group_log_mean", 0.0)
        )
        self._last_i_progressive_q_group_log_spread = float(
            getattr(self.i_codec, "_last_i_progressive_q_group_log_spread", 0.0)
        )
        self._last_i_progressive_q_group_logs = tuple(getattr(self.i_codec, "_last_i_progressive_q_group_logs", ()))
        self._last_i_progressive_q_channel_log_mean_abs = float(
            getattr(self.i_codec, "_last_i_progressive_q_channel_log_mean_abs", 0.0)
        )
        z_hat = self._semantic_reconstruct(self._denormalize(z_hat), semantic_size)
        self._last_reference_output = self._semantic_reconstruct(self._denormalize(reference_hat), semantic_size)
        return (z_hat, 4.0 * torch.stack(bpps).mean())

    def encode_train_temporal_chunk(
        self,
        latent: torch.Tensor,
        rate: Optional[float] = None,
        qp: Optional[int] = None,
        *,
        p_rate: Optional[float] = None,
        p_shift_qp: bool = True,
        reset_temporal_state: bool = True,
        split_after_first_latent: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Differentiable temporal encoding.

        Default policy is frame 0 intra, frames 1+ inter.  When
        ``split_after_first_latent`` is enabled, z0 is coded as a standalone I
        frame, then temporal state is reset and z1 starts an independent I/P/P
        chain.  This matches Wan's asymmetric latent layout where z0 comes from
        raw frame 0 while z1+ come from 4-frame temporal VAE groups.

        Args:
            rate: Target bpp used to derive a QP.  Deprecated; prefer ``qp``.
            qp: Explicit QP index.  If provided, ``rate`` is ignored.
            p_rate: Optional separate target bpp for P-frames.  This is only a
                hint for loss computation; P-frames still use ``shift_qp`` to
                select a coarser QP than I-frames unless ``p_shift_qp`` is False.
                Passing ``p_rate`` does NOT override the QP heuristic.
            p_shift_qp: If False, P-frames use the same QP as I-frames.  This
                supports the "I/P same rate / same lambda" training setting.
            split_after_first_latent: If True, use I-I-P-P for a 4-latent GOP:
                z0 is standalone, z1 is the anchor for the later latent video
                sequence, and z2+ are P frames.
        """
        if self._task938_quantizer.enabled and (p_shift_qp or split_after_first_latent):
            raise ValueError("Task938 quantization requires p_shift_qp=False and independent I/P6")
        del p_rate
        if qp is None:
            if rate is None:
                raise ValueError("encode_train_temporal_chunk requires either rate or qp")
            qp = _rate_to_qp(rate, self._rate_qp_table)
        if reset_temporal_state:
            self.reset_temporal_state()
        latent = latent.to(self._device)
        latent, semantic_size = self._semantic_analysis(latent)
        latent_norm = self._normalize(latent)
        if latent_norm.ndim == 5:
            B, C, T, H, W = latent_norm.shape
            batch_mode = True
        elif latent_norm.ndim == 4 and latent_norm.shape[0] == self.latent_channels:
            C, T, H, W = latent_norm.shape
            latent_norm = latent_norm.unsqueeze(0)
            B, C, T, H, W = latent_norm.shape
            batch_mode = False
        else:
            raise ValueError(f"Unsupported latent shape for temporal chunk: {latent.shape}")
        index_map = [0, 1, 0, 2, 0, 2, 0, 2]
        frames_z_hat: list[torch.Tensor] = []
        self._last_i_entropy_scales = None
        self._last_entropy_scale_sequence = None
        self._last_quantization_width_sequence = None
        task938_scales, task938_widths = ([], [])
        bpps: list[torch.Tensor] = []
        bpp_y_values: list[float] = []
        bpp_z_values: list[float] = []
        reference_repair_losses: list[torch.Tensor] = []
        reference_repair_rms_deltas: list[float] = []
        reference_state_losses: list[torch.Tensor] = []
        reference_state_rms_deltas: list[float] = []
        p_frequency_aux_values: list[dict[str, torch.Tensor]] = []
        ring_gate_losses: list[torch.Tensor] = []
        ring_benefit_losses: list[torch.Tensor] = []
        ring_alignment_losses: list[torch.Tensor] = []
        ring_anchor_usages: list[torch.Tensor] = []
        ring_valid_fractions: list[torch.Tensor] = []
        ring_modes: list[str] = []
        for t in range(T):
            frame = self._codec_input_transform(latent_norm[:, :, t, :, :], t)
            frame_codec = self._proxy_analysis(frame)
            state_p_codec = None
            state_z_hat_t = None
            is_intra = t == 0 or (split_after_first_latent and t == 1)
            if is_intra or not p_shift_qp:
                frame_qp = qp
            else:
                chain_t = t - 1 if split_after_first_latent else t
                frame_qp = self.p_codec.shift_qp(qp, index_map[chain_t % 8])
            if not is_intra and self._p_hierarchical_quality_enabled:
                frame_qp = self._hierarchical_quality_qp(qp, t)
            if is_intra:
                if split_after_first_latent and t == 1:
                    self.reset_temporal_state()
                intra_role = "z1" if split_after_first_latent and t == 1 else "default"
                i_codec = self._select_i_codec(intra_role)
                z_hat_codec, bpp_t = i_codec.encode_train(frame_codec, frame_qp)
                if t == 0:
                    self._last_i_entropy_scales = getattr(i_codec, "_last_entropy_scales", None)
                state_codec = self._i_reference_codec_output(i_codec, z_hat_codec)
                z_hat_t = self._proxy_synthesis(z_hat_codec)
                z_hat_t = self._codec_output_transform(z_hat_t, t)
                state_z_hat_t = self._codec_output_transform(self._proxy_synthesis(state_codec), t)
                bpp_y_values.append(float(i_codec._last_bpp_y))
                bpp_z_values.append(float(i_codec._last_bpp_z))
            else:
                self._activate_temporal_reference(t)
                z_prev = None
                if self._temporal_state is not None:
                    z_prev = self._proxy_analysis(self._codec_input_transform(self._temporal_state["z_out"], t))
                feature_prev = self._temporal_feature_ref()
                dpb_feature_prev = self._temporal_dpb_feature_ref()
                ref_y_prev = self._temporal_dpb_y_ref()
                chain_t = t - 1 if split_after_first_latent else t
                frame_type_idx = max(0, min(chain_t - 1, getattr(self.p_codec, "p_type_count", 1) - 1))
                p_codec = self._select_p_codec(frame_type_idx)
                self._set_p_synthesis_condition(p_codec, pos_type_idx=t, qp=frame_qp)
                if z_prev is not None:
                    self._ring_anchor_training_target = (
                        frame_codec.detach() if self.training and self.ring_anchor_required_mixer is not None else None
                    )
                    z_prev = self._anchor_reference(z_prev, frame_qp, pos_type_idx=t, frame_type_idx=frame_type_idx)
                    self._ring_anchor_training_target = None
                    ring_diagnostics = self._last_ring_anchor_diagnostics
                    if ring_diagnostics:
                        ring_gate_losses.append(ring_diagnostics["gate_loss"])
                        ring_benefit_losses.append(ring_diagnostics["benefit_loss"])
                        ring_alignment_losses.append(ring_diagnostics["alignment_loss"])
                        ring_anchor_usages.append(ring_diagnostics["anchor_usage"])
                        ring_valid_fractions.append(ring_diagnostics["valid_fraction"])
                        ring_modes.append(str(ring_diagnostics["mode"]))
                    if self.anchor_context_adapter is not None:
                        reference_repair_rms_deltas.append(float(getattr(self, "_last_anchor_context_delta", 0.0)))
                    if self.anchor_context_adapter is not None:
                        clean_reference = self._proxy_analysis(
                            self._codec_input_transform(latent_norm[:, :, t - 1, :, :], t)
                        )
                        reference_repair_losses.append(F.mse_loss(z_prev.float(), clean_reference.detach().float()))
                state_p_codec = p_codec
                z_hat_codec, bpp_t = p_codec.encode_train(
                    frame_codec,
                    frame_qp,
                    z_prev=z_prev,
                    feature_prev=feature_prev,
                    feature_blend=self._temporal_feature_blend(),
                    frame_type_idx=frame_type_idx,
                    dpb_feature_prev=dpb_feature_prev,
                    ref_y_prev=ref_y_prev,
                )
                p_frequency_aux = getattr(p_codec, "_last_p_frequency_aux", {})
                if p_frequency_aux:
                    p_frequency_aux_values.append(dict(p_frequency_aux))
                base_state_codec = self._p_reference_codec_output(p_codec, z_hat_codec)
                metric_codec = self._p_metric_codec_output(z_hat_codec, base_state_codec, pos_type_idx=t)
                metric_codec = self._second_order_display(
                    metric_codec, qp=frame_qp, pos_type_idx=t, frame_type_idx=frame_type_idx
                )
                state_codec = self._p_reference_state_codec_output(
                    p_codec, base_state_codec, pos_type_idx=t, qp=frame_qp
                )
                if self.training:
                    reference_state_losses.append(F.mse_loss(state_codec.float(), frame_codec.detach().float()))
                z_hat_t = self._proxy_synthesis(metric_codec)
                z_hat_t = self._codec_output_transform(z_hat_t, t)
                pre_display_t = self._codec_output_transform(self._proxy_synthesis(base_state_codec), t)
                state_z_hat_t = self._codec_output_transform(self._proxy_synthesis(state_codec), t)
                z_hat_t = self._p_display_tokenizer_output(z_hat_t, pre_display_t, p_codec, qp=frame_qp, pos_type_idx=t)
                bpp_y_values.append(float(p_codec._last_bpp_y))
                bpp_z_values.append(float(p_codec._last_bpp_z))
            if (
                getattr(self, "_export_entropy_scales", False)
                and getattr(self.i_codec, "task938_region_quantizer", None) is not None
            ):
                active_codec = i_codec if is_intra else p_codec
                task938_scales.append(active_codec._last_entropy_scale_map)
                task938_widths.append(active_codec._last_quantization_width_map)
            frames_z_hat.append(z_hat_t)
            bpps.append(bpp_t)
            next_state = self._make_temporal_state(
                z_hat_t if state_z_hat_t is None else state_z_hat_t, is_intra=is_intra, p_codec=state_p_codec
            )
            self._commit_temporal_state(next_state, is_intra=is_intra, position_idx=t)
            if is_intra:
                self._anchor_state = (z_hat_t if state_z_hat_t is None else state_z_hat_t).detach()
        if task938_scales:
            self._last_entropy_scale_sequence = torch.stack(task938_scales, dim=2)
            self._last_quantization_width_sequence = torch.stack(task938_widths, dim=2)
        z_hat_norm = torch.stack(frames_z_hat, dim=2)
        self._last_bpp_y = float(effective_bpp_from_per_latent(bpp_y_values))
        self._last_bpp_z = float(effective_bpp_from_per_latent(bpp_z_values))
        self._last_bpps_per_frame = torch.stack(bpps)
        self._last_reference_repair_aux_loss = (
            torch.stack(reference_repair_losses).mean() if reference_repair_losses else latent_norm.new_zeros(())
        )
        self._last_reference_repair_rms_delta = (
            sum(reference_repair_rms_deltas) / len(reference_repair_rms_deltas) if reference_repair_rms_deltas else 0.0
        )
        self._last_reference_state_aux_loss = (
            torch.stack(reference_state_losses).mean() if reference_state_losses else latent_norm.new_zeros(())
        )
        self._last_reference_state_rms_delta = (
            sum(reference_state_rms_deltas) / len(reference_state_rms_deltas) if reference_state_rms_deltas else 0.0
        )
        self._last_ring_anchor_gate_loss = (
            torch.stack(ring_gate_losses).mean() if ring_gate_losses else latent_norm.new_zeros(())
        )
        self._last_ring_anchor_benefit_loss = (
            torch.stack(ring_benefit_losses).mean() if ring_benefit_losses else latent_norm.new_zeros(())
        )
        self._last_ring_anchor_alignment_loss = (
            torch.stack(ring_alignment_losses).mean() if ring_alignment_losses else latent_norm.new_zeros(())
        )
        self._last_ring_anchor_usage = (
            torch.stack(ring_anchor_usages).mean() if ring_anchor_usages else latent_norm.new_zeros(())
        )
        self._last_ring_anchor_valid_fraction = (
            torch.stack(ring_valid_fractions).mean() if ring_valid_fractions else latent_norm.new_zeros(())
        )
        self._last_ring_anchor_modes = tuple(ring_modes)
        if p_frequency_aux_values:
            keys = set(p_frequency_aux_values[0])
            if any((set(item) != keys for item in p_frequency_aux_values[1:])):
                raise RuntimeError("P frequency-compaction metrics changed across one GOP.")
            self._last_p_frequency_aux = {
                key: torch.stack([item[key] for item in p_frequency_aux_values]).mean() for key in sorted(keys)
            }
        else:
            self._last_p_frequency_aux = {}
        self._last_i_progressive_aux = dict(getattr(self.i_codec, "_last_i_progressive_aux", {}))
        self._last_i_progressive_q_group_log_mean = float(
            getattr(self.i_codec, "_last_i_progressive_q_group_log_mean", 0.0)
        )
        self._last_i_progressive_q_group_log_spread = float(
            getattr(self.i_codec, "_last_i_progressive_q_group_log_spread", 0.0)
        )
        self._last_i_progressive_q_group_logs = tuple(getattr(self.i_codec, "_last_i_progressive_q_group_logs", ()))
        self._last_i_progressive_q_channel_log_mean_abs = float(
            getattr(self.i_codec, "_last_i_progressive_q_channel_log_mean_abs", 0.0)
        )
        z_hat = self._denormalize(z_hat_norm)
        z_hat = self.postprocess_latents(z_hat, int(qp), gradient_checkpointing=bool(self.training))
        z_hat = self._semantic_reconstruct(z_hat, semantic_size)
        if not batch_mode:
            z_hat = z_hat.squeeze(0)
        return (z_hat, effective_bpp_from_per_latent(bpps))

    def encode(
        self,
        latent: torch.Tensor,
        rate: Optional[float] = None,
        qp: Optional[int] = None,
        *,
        p_shift_qp: bool = True,
        split_after_first_latent: bool = False,
        return_quantized: bool = True,
    ) -> Tuple[bytes, torch.Tensor]:
        """Real rANS encode supporting [C,T,H,W] and [B,C,T,H,W] latent layouts.

        Args:
            rate: Target bpp used to derive a QP.  Deprecated; prefer ``qp``.
            qp: Explicit QP index.  If provided, ``rate`` is ignored.
            p_shift_qp: If False, P-frames use the same QP as I-frames.  This
                must match the training setting for a fair evaluation.
            split_after_first_latent: If True, encode z0 as a standalone I
                latent and z1 as the anchor I latent for the subsequent P chain.
        """
        if self._task938_quantizer.enabled and (p_shift_qp or split_after_first_latent):
            raise ValueError("Task938 quantization requires p_shift_qp=False and independent I/P6")
        self.i_codec._ensure_entropy_coder()
        self.p_codec._ensure_entropy_coder()
        if qp is None:
            if rate is None:
                raise ValueError("encode requires either rate or qp")
            qp = _rate_to_qp(rate, self._rate_qp_table)
        latent = latent.to(self._device)
        latent, semantic_size = self._semantic_analysis(latent)
        latent_norm = self._normalize(latent)
        if latent_norm.ndim == 4 and latent_norm.shape[0] == self.latent_channels:
            bitstream, decoded = self._encode_temporal_4d(
                latent_norm, qp, p_shift_qp=p_shift_qp, split_after_first_latent=split_after_first_latent
            )
            bitstream = self._annotate_semantic_bitstream(bitstream, semantic_size)
            decoded = self.postprocess_latents(decoded, int(qp))
            return (bitstream, self._semantic_reconstruct(decoded, semantic_size))
        if latent_norm.ndim == 5:
            bitstream, decoded = self._encode_temporal_5d(
                latent_norm, qp, p_shift_qp=p_shift_qp, split_after_first_latent=split_after_first_latent
            )
            bitstream = self._annotate_semantic_bitstream(bitstream, semantic_size)
            decoded = self.postprocess_latents(decoded, int(qp))
            return (bitstream, self._semantic_reconstruct(decoded, semantic_size))
        raise ValueError(f"Expected 4D [C,T,H,W] or 5D [B,C,T,H,W] tensor, got {latent.ndim}D")

    def _encode_single_frame(
        self,
        frame: torch.Tensor,
        qp: int,
        is_intra: bool,
        intra_role: str = "default",
        frame_type_idx: int = 0,
        pos_type_idx: int = 0,
    ) -> Tuple[bytes, torch.Tensor]:
        frame_codec = self._proxy_analysis(self._codec_input_transform(frame, pos_type_idx))
        if is_intra:
            i_codec = self._select_i_codec(intra_role)
            bitstream, z_out_codec = i_codec.encode(frame_codec, qp)
            state_codec = self._i_reference_codec_output(i_codec, z_out_codec)
            z_out = self._proxy_synthesis(z_out_codec)
            z_out = self._codec_output_transform(z_out, pos_type_idx)
            if intra_role != "default":
                payload = torch.load(io.BytesIO(bitstream), map_location="cpu", weights_only=False)
                payload["i_codec_role"] = intra_role
                buf = io.BytesIO()
                torch.save(payload, buf)
                bitstream = buf.getvalue()
            self._last_reference_output = self._codec_output_transform(self._proxy_synthesis(state_codec), pos_type_idx)
            return (bitstream, z_out)
        self._activate_temporal_reference(pos_type_idx)
        qp = self._hierarchical_quality_qp(qp, pos_type_idx)
        z_prev = None
        if self._temporal_state is not None:
            z_prev = self._proxy_analysis(self._codec_input_transform(self._temporal_state["z_out"], pos_type_idx))
        feature_prev = self._temporal_feature_ref()
        dpb_feature_prev = self._temporal_dpb_feature_ref()
        ref_y_prev = self._temporal_dpb_y_ref()
        if z_prev is None:
            raise RuntimeError("P-frame encoding requires a previous reconstructed latent.")
        z_prev = self._anchor_reference(z_prev, qp, pos_type_idx=pos_type_idx, frame_type_idx=frame_type_idx)
        p_codec = self._select_p_codec(frame_type_idx)
        self._set_p_synthesis_condition(p_codec, pos_type_idx=pos_type_idx, qp=qp)
        ctx, ctx_t = p_codec.get_temporal_context(
            z_prev,
            qp,
            feature_prev=feature_prev,
            feature_blend=self._temporal_feature_blend(),
            frame_type_idx=frame_type_idx,
            dpb_feature_prev=dpb_feature_prev,
        )
        bitstream, z_out_codec = p_codec.encode(
            frame_codec,
            qp,
            ctx=ctx,
            ctx_t=ctx_t,
            z_prev=z_prev,
            feature_prev=feature_prev,
            feature_blend=self._temporal_feature_blend(),
            frame_type_idx=frame_type_idx,
            dpb_feature_prev=dpb_feature_prev,
            ref_y_prev=ref_y_prev,
        )
        base_state_codec = self._p_reference_codec_output(p_codec, z_out_codec)
        metric_codec = self._p_metric_codec_output(z_out_codec, base_state_codec, pos_type_idx=pos_type_idx)
        metric_codec = self._second_order_display(
            metric_codec, qp=qp, pos_type_idx=pos_type_idx, frame_type_idx=frame_type_idx
        )
        state_codec = self._p_reference_state_codec_output(p_codec, base_state_codec, pos_type_idx=pos_type_idx, qp=qp)
        z_out = self._proxy_synthesis(metric_codec)
        z_out = self._codec_output_transform(z_out, pos_type_idx)
        pre_display = self._codec_output_transform(self._proxy_synthesis(base_state_codec), pos_type_idx)
        self._last_reference_output = self._codec_output_transform(self._proxy_synthesis(state_codec), pos_type_idx)
        z_out = self._p_display_tokenizer_output(z_out, pre_display, p_codec, qp=qp, pos_type_idx=pos_type_idx)
        payload = torch.load(io.BytesIO(bitstream), map_location="cpu", weights_only=False)
        if p_codec is self.p1_codec:
            payload["p_codec_role"] = "p1"
        buf = io.BytesIO()
        torch.save(payload, buf)
        return (buf.getvalue(), z_out)

    def _encode_temporal_4d(
        self, latent: torch.Tensor, qp: int, p_shift_qp: bool = True, split_after_first_latent: bool = False
    ) -> Tuple[bytes, torch.Tensor]:
        C, T, H, W = latent.shape
        self.reset_temporal_state()
        frame_payloads = []
        decoded_frames = []
        self._last_hard_p_symbol_tensors: list[dict[str, Any]] = []
        self._last_hard_i_symbol_tensors: list[dict[str, Any]] = []
        index_map = [0, 1, 0, 2, 0, 2, 0, 2]
        for t in range(T):
            frame = latent[:, t, :, :].unsqueeze(0)
            is_intra = t == 0 or (split_after_first_latent and t == 1)
            if split_after_first_latent and t == 1:
                self.reset_temporal_state()
            chain_t = t - 1 if split_after_first_latent else t
            frame_qp = qp if is_intra or not p_shift_qp else self.p_codec.shift_qp(qp, index_map[chain_t % 8])
            frame_type_idx = max(0, min(chain_t - 1, getattr(self.p_codec, "p_type_count", 1) - 1))
            intra_role = "z1" if split_after_first_latent and t == 1 else "default"
            bitstream, z_out = self._encode_single_frame(
                frame, frame_qp, is_intra=is_intra, intra_role=intra_role, frame_type_idx=frame_type_idx, pos_type_idx=t
            )
            frame_payloads.append(torch.load(io.BytesIO(bitstream), map_location="cpu", weights_only=False))
            if is_intra and self._capture_exact_p_entropy_symbols:
                i_codec = self._select_i_codec(intra_role)
                symbols = getattr(i_codec, "_last_entropy_symbol_tensors", None)
                if not symbols:
                    raise RuntimeError("Hard I encoder did not publish exact symbols")
                self._last_hard_i_symbol_tensors.append(symbols)
            if not is_intra and self._capture_exact_p_entropy_symbols:
                p_codec = self._select_p_codec(frame_type_idx)
                symbols = getattr(p_codec, "_last_entropy_symbol_tensors", None)
                if not symbols:
                    raise RuntimeError("Hard P encoder did not publish exact symbols")
                self._last_hard_p_symbol_tensors.append(symbols)
            state_out = self._last_reference_output.detach()
            if z_out.ndim == 4 and z_out.shape[0] == 1:
                z_out = z_out.squeeze(0)
            decoded_frames.append(z_out)
            next_state = self._make_temporal_state(
                state_out, is_intra=is_intra, p_codec=None if is_intra else self._select_p_codec(frame_type_idx)
            )
            self._commit_temporal_state(next_state, is_intra=is_intra, position_idx=t)
            if is_intra:
                self._anchor_state = state_out.detach()
        decoded = torch.stack(decoded_frames, dim=1)
        buf = io.BytesIO()
        torch.save(
            {
                "format": "rt_two_temporal_4d",
                **(
                    {"task938_nested_quantizer_contract": self.task938_quantizer_contract()}
                    if self._task938_quantizer.enabled
                    else {}
                ),
                "frames": frame_payloads,
                "shape": [C, T, H, W],
                "qp": qp,
                "split_after_first_latent": bool(split_after_first_latent),
                "type_norm_fingerprint": self._type_norm_fingerprint(),
                "latent_canvas_protection_contract": self._latent_canvas_contract(),
                "i_synthesis_head_out_of_loop": self._i_synthesis_head_out_of_loop,
                "p_synthesis_head_out_of_loop": self._p_synthesis_head_out_of_loop,
                "p_reference_state_head_enabled": self._p_reference_state_head_enabled,
                "p_decoded_dpb_mode": self._p_decoded_dpb_mode,
                "p_hierarchical_reference_enabled": self._p_hierarchical_reference_enabled,
                "p_hierarchical_quality_enabled": self._p_hierarchical_quality_enabled,
                "latent_postfilter_enabled": self.latent_postfilter is not None,
                "decoder_cell_grid_smoother_enabled": self.cell_grid_smoother is not None,
                "entropy_bitstream_cell_grid_smoother": False,
                "anchor_context_multiscale_enabled": self._anchor_context_multiscale_enabled,
                "ring_anchor_required_contract": None,
                "p_reference_reliability_contract": None,
                "p_second_order_reference_enabled": self._p_second_order_reference_enabled,
                "p_second_order_motion_alignment_enabled": self._p_second_order_motion_alignment_enabled,
                "p_second_order_display_enabled": self._p_second_order_display_enabled,
                "p_third_order_reference_enabled": self._p_third_order_reference_enabled,
                "p_second_order_position_gain_enabled": self._p_second_order_position_gain_enabled,
                "p_second_order_reliability_gate_enabled": self._p_second_order_reliability_gate_enabled,
            },
            buf,
        )
        return (buf.getvalue(), self._denormalize(decoded))

    def _encode_temporal_5d(
        self, latent: torch.Tensor, qp: int, p_shift_qp: bool = True, split_after_first_latent: bool = False
    ) -> Tuple[bytes, torch.Tensor]:
        B, C, T, H, W = latent.shape
        self.reset_temporal_state()
        batch_payloads = []
        decoded_batches = []
        self._last_hard_p_symbol_tensors = []
        self._last_hard_i_symbol_tensors = []
        index_map = [0, 1, 0, 2, 0, 2, 0, 2]
        for b in range(B):
            frame_payloads = []
            decoded_frames = []
            self.reset_temporal_state()
            for t in range(T):
                frame = latent[b, :, t, :, :].unsqueeze(0)
                is_intra = t == 0 or (split_after_first_latent and t == 1)
                if split_after_first_latent and t == 1:
                    self.reset_temporal_state()
                chain_t = t - 1 if split_after_first_latent else t
                frame_qp = qp if is_intra or not p_shift_qp else self.p_codec.shift_qp(qp, index_map[chain_t % 8])
                frame_type_idx = max(0, min(chain_t - 1, getattr(self.p_codec, "p_type_count", 1) - 1))
                intra_role = "z1" if split_after_first_latent and t == 1 else "default"
                bitstream, z_out = self._encode_single_frame(
                    frame,
                    frame_qp,
                    is_intra=is_intra,
                    intra_role=intra_role,
                    frame_type_idx=frame_type_idx,
                    pos_type_idx=t,
                )
                frame_payloads.append(torch.load(io.BytesIO(bitstream), map_location="cpu", weights_only=False))
                if is_intra and self._capture_exact_p_entropy_symbols:
                    i_codec = self._select_i_codec(intra_role)
                    symbols = getattr(i_codec, "_last_entropy_symbol_tensors", None)
                    if not symbols:
                        raise RuntimeError("Hard I encoder did not publish exact symbols")
                    self._last_hard_i_symbol_tensors.append(symbols)
                if not is_intra and self._capture_exact_p_entropy_symbols:
                    p_codec = self._select_p_codec(frame_type_idx)
                    symbols = getattr(p_codec, "_last_entropy_symbol_tensors", None)
                    if not symbols:
                        raise RuntimeError("Hard P encoder did not publish exact symbols")
                    self._last_hard_p_symbol_tensors.append(symbols)
                state_out = self._last_reference_output.detach()
                if z_out.ndim == 4 and z_out.shape[0] == 1:
                    z_out = z_out.squeeze(0)
                decoded_frames.append(z_out)
                next_state = self._make_temporal_state(
                    state_out, is_intra=is_intra, p_codec=None if is_intra else self._select_p_codec(frame_type_idx)
                )
                self._commit_temporal_state(next_state, is_intra=is_intra, position_idx=t)
                if is_intra:
                    self._anchor_state = state_out.detach()
            batch_payloads.append(frame_payloads)
            decoded_batches.append(torch.stack(decoded_frames, dim=1))
        decoded = torch.stack(decoded_batches, dim=0)
        buf = io.BytesIO()
        torch.save(
            {
                "format": "rt_two_temporal_5d",
                **(
                    {"task938_nested_quantizer_contract": self.task938_quantizer_contract()}
                    if self._task938_quantizer.enabled
                    else {}
                ),
                "batches": batch_payloads,
                "shape": [B, C, T, H, W],
                "qp": qp,
                "split_after_first_latent": bool(split_after_first_latent),
                "type_norm_fingerprint": self._type_norm_fingerprint(),
                "latent_canvas_protection_contract": self._latent_canvas_contract(),
                "i_synthesis_head_out_of_loop": self._i_synthesis_head_out_of_loop,
                "p_synthesis_head_out_of_loop": self._p_synthesis_head_out_of_loop,
                "p_reference_state_head_enabled": self._p_reference_state_head_enabled,
                "p_decoded_dpb_mode": self._p_decoded_dpb_mode,
                "p_hierarchical_reference_enabled": self._p_hierarchical_reference_enabled,
                "p_hierarchical_quality_enabled": self._p_hierarchical_quality_enabled,
                "latent_postfilter_enabled": self.latent_postfilter is not None,
                "decoder_cell_grid_smoother_enabled": self.cell_grid_smoother is not None,
                "entropy_bitstream_cell_grid_smoother": False,
                "anchor_context_multiscale_enabled": self._anchor_context_multiscale_enabled,
                "ring_anchor_required_contract": None,
                "p_reference_reliability_contract": None,
                "p_second_order_reference_enabled": self._p_second_order_reference_enabled,
                "p_second_order_motion_alignment_enabled": self._p_second_order_motion_alignment_enabled,
                "p_second_order_display_enabled": self._p_second_order_display_enabled,
                "p_third_order_reference_enabled": self._p_third_order_reference_enabled,
                "p_second_order_position_gain_enabled": self._p_second_order_position_gain_enabled,
                "p_second_order_reliability_gate_enabled": self._p_second_order_reliability_gate_enabled,
            },
            buf,
        )
        return (buf.getvalue(), self._denormalize(decoded))

    def build_rate_qp_table(self, reference_latent: torch.Tensor, num_pixels: int) -> list[tuple[float, int]]:
        """Build an I-frame (rate, qp) lookup table using real rANS encoding.

        Args:
            reference_latent: A reference latent in the original Wan VAE domain.
                Supported shapes: ``[C, T, H, W]`` or ``[B, C, T, H, W]``.
            num_pixels: Number of raw video pixels represented by the whole
                reference latent (used to compute bpp).

        Returns:
            A list of ``(actual_bpp, qp)`` sorted by ``actual_bpp`` ascending.
        """
        self.eval()
        reference_latent = reference_latent.to(self._device)
        reference_latent, _ = self._semantic_analysis(reference_latent)
        latent_norm = self._normalize(reference_latent)
        if latent_norm.ndim == 4 and latent_norm.shape[0] == self.latent_channels:
            T = latent_norm.shape[1]
            frames = [(latent_norm[:, t : t + 1, :, :].permute(1, 0, 2, 3), t) for t in range(T)]
        elif latent_norm.ndim == 5:
            B, C, T, H, W = latent_norm.shape
            frames = [(latent_norm[b, :, t : t + 1, :, :].permute(1, 0, 2, 3), t) for b in range(B) for t in range(T)]
        else:
            raise ValueError(f"build_rate_qp_table expects [C,T,H,W] or [B,C,T,H,W], got {latent_norm.shape}")
        table: list[tuple[float, int]] = []
        with torch.no_grad():
            for qp in range(_NUM_QP):
                total_bits = 0
                for frame, t in frames:
                    frame_codec = self._proxy_analysis(self._codec_input_transform(frame, t))
                    bitstream, _ = self.i_codec.encode(frame_codec, qp=qp)
                    payload = torch.load(io.BytesIO(bitstream), map_location="cpu", weights_only=False)
                    total_bits += len(payload["bit_stream"]) * 8
                actual_bpp = total_bits / max(num_pixels, 1.0)
                table.append((actual_bpp, qp))
        table.sort(key=lambda x: x[0])
        self._rate_qp_table = table
        print(
            f"[Rate-QP] I-frame table built: {len(table)} entries, rate range [{table[0][0]:.6f}, {table[-1][0]:.6f}]"
        )
        return table

    def build_gop_rate_qp_table(
        self, reference_latent: torch.Tensor, num_pixels: int, p_shift_qp: bool = True
    ) -> list[tuple[float, int]]:
        """Build a GOP-level (rate, qp) lookup table using real rANS encoding.

        This is meant for temporal eval: it encodes the entire reference GOP
        (frame 0 intra, remaining frames inter) for each candidate I-frame QP
        and records the total GOP bitrate. The resulting table maps target
        **average GOP bpp** to the I-frame QP that should be passed to
        ``encode(..., rate=target)``.

        Args:
            reference_latent: A reference GOP in the original Wan VAE domain.
                Shape must be ``[C, T, H, W]`` with ``T`` equal to the desired
                GOP length (e.g. 2 or 4 latent frames).
            num_pixels: Number of raw video pixels represented by the whole GOP.

        Returns:
            A list of ``(actual_bpp, qp)`` sorted by ``actual_bpp`` ascending.
        """
        self.eval()
        reference_latent = reference_latent.to(self._device)
        if not (reference_latent.ndim == 4 and reference_latent.shape[0] == self.latent_channels):
            raise ValueError(f"build_gop_rate_qp_table expects [C,T,H,W], got {reference_latent.shape}")
        reference_latent, _ = self._semantic_analysis(reference_latent)
        latent_norm = self._normalize(reference_latent)
        table: list[tuple[float, int]] = []
        with torch.no_grad():
            for qp in range(_NUM_QP):
                bitstream, _ = self._encode_temporal_4d(latent_norm, qp, p_shift_qp=p_shift_qp)
                payload = torch.load(io.BytesIO(bitstream), map_location="cpu", weights_only=False)
                raw_len = sum((len(f.get("bit_stream", b"")) for f in payload["frames"]))
                header_len = 0
                actual_bpp = 8 * (raw_len + header_len) / max(num_pixels, 1.0)
                table.append((actual_bpp, qp))
        table.sort(key=lambda x: x[0])
        self._rate_qp_table = table
        print(f"[Rate-QP] GOP table built: {len(table)} entries, rate range [{table[0][0]:.6f}, {table[-1][0]:.6f}]")
        return table

    def decode(self, bitstream: bytes) -> torch.Tensor:
        """Decode a bitstream produced by ``encode``."""
        self.i_codec._ensure_entropy_coder()
        self.p_codec._ensure_entropy_coder()
        payload = torch.load(io.BytesIO(bitstream), map_location="cpu", weights_only=False)
        encoded_nested = payload.get("task938_nested_quantizer_contract")
        expected_nested = self.task938_quantizer_contract() if self._task938_quantizer.enabled else None
        if encoded_nested != expected_nested:
            raise ValueError(
                f"Bitstream/Task938 quantizer contract mismatch: stream={encoded_nested} decoder={expected_nested}"
            )
        semantic_meta = payload.get("semantic_anchor")
        encoded_semantic = semantic_meta is not None
        if encoded_semantic != (self.semantic_anchor is not None):
            raise ValueError(
                f"Bitstream/semantic-anchor policy mismatch: encoded={encoded_semantic} decoder={self.semantic_anchor is not None}."
            )
        semantic_size = None
        if encoded_semantic:
            assert self.semantic_anchor is not None
            if int(semantic_meta.get("version", -1)) != int(self.semantic_anchor.bitstream_version):
                raise ValueError("unsupported semantic-anchor bitstream version")
            values = semantic_meta.get("original_size", ())
            if len(values) != 2:
                raise ValueError("semantic-anchor bitstream lacks original_size")
            semantic_size = (int(values[0]), int(values[1]))
            encoded_detail = bool(semantic_meta.get("predict_detail", False))
            if encoded_detail != self._semantic_anchor_predict_detail:
                raise ValueError(
                    f"Bitstream/semantic detail-predictor policy mismatch: encoded={encoded_detail} decoder={self._semantic_anchor_predict_detail}."
                )
        self._validate_type_norm_fingerprint(payload)
        self._validate_latent_canvas_contract(payload)
        encoded_i_out_of_loop = bool(payload.get("i_synthesis_head_out_of_loop", False))
        if encoded_i_out_of_loop != self._i_synthesis_head_out_of_loop:
            raise ValueError(
                f"Bitstream/I synthesis loop-policy mismatch: encoded={encoded_i_out_of_loop} decoder={self._i_synthesis_head_out_of_loop}."
            )
        encoded_out_of_loop = bool(payload.get("p_synthesis_head_out_of_loop", False))
        if encoded_out_of_loop != self._p_synthesis_head_out_of_loop:
            raise ValueError(
                f"Bitstream/P synthesis loop-policy mismatch: encoded={encoded_out_of_loop} decoder={self._p_synthesis_head_out_of_loop}."
            )
        encoded_dpb_mode = str(payload.get("p_decoded_dpb_mode", "off")).strip().lower()
        if encoded_dpb_mode != self._p_decoded_dpb_mode:
            raise ValueError(
                f"Bitstream/decoded-DPB policy mismatch: encoded={encoded_dpb_mode} decoder={self._p_decoded_dpb_mode}."
            )
        encoded_hierarchical_reference = bool(payload.get("p_hierarchical_reference_enabled", False))
        if encoded_hierarchical_reference != self._p_hierarchical_reference_enabled:
            raise ValueError(
                f"Bitstream/hierarchical-reference policy mismatch: encoded={encoded_hierarchical_reference} decoder={self._p_hierarchical_reference_enabled}."
            )
        encoded_reference_state = bool(payload.get("p_reference_state_head_enabled", False))
        if encoded_reference_state != self._p_reference_state_head_enabled:
            raise ValueError(
                f"Bitstream/P reference-state policy mismatch: encoded={encoded_reference_state} decoder={self._p_reference_state_head_enabled}."
            )
        encoded_latent_postfilter = bool(payload.get("latent_postfilter_enabled", False))
        if encoded_latent_postfilter != (self.latent_postfilter is not None):
            raise ValueError(
                f"Bitstream/latent-postfilter policy mismatch: encoded={encoded_latent_postfilter} decoder={self.latent_postfilter is not None}."
            )
        encoded_cell_grid_smoother = bool(
            payload.get("decoder_cell_grid_smoother_enabled", payload.get("cell_grid_smoother_enabled", False))
        )
        if encoded_cell_grid_smoother != (self.cell_grid_smoother is not None):
            raise ValueError(
                f"Bitstream/cell-grid-smoother policy mismatch: encoded={encoded_cell_grid_smoother} decoder={self.cell_grid_smoother is not None}."
            )
        encoded_anchor_multiscale = bool(payload.get("anchor_context_multiscale_enabled", False))
        if encoded_anchor_multiscale != self._anchor_context_multiscale_enabled:
            raise ValueError(
                f"Bitstream/anchor multiscale policy mismatch: encoded={encoded_anchor_multiscale} decoder={self._anchor_context_multiscale_enabled}."
            )
        encoded_ring_anchor = payload.get("ring_anchor_required_contract")
        expected_ring_anchor = None
        if encoded_ring_anchor != None:
            raise ValueError(f"Bitstream/ring-anchor contract mismatch: encoded={encoded_ring_anchor} decoder={None}.")
        encoded_reliability = payload.get("p_reference_reliability_contract")
        expected_reliability = None
        if encoded_reliability != None:
            raise ValueError(
                f"Bitstream/Task694 reference-reliability contract mismatch: encoded={encoded_reliability} decoder={None}."
            )
        encoded_second_order = bool(payload.get("p_second_order_reference_enabled", False))
        if encoded_second_order != self._p_second_order_reference_enabled:
            raise ValueError(
                f"Bitstream/second-order reference policy mismatch: encoded={encoded_second_order} decoder={self._p_second_order_reference_enabled}."
            )
        encoded_second_order_motion = bool(payload.get("p_second_order_motion_alignment_enabled", False))
        if encoded_second_order_motion != self._p_second_order_motion_alignment_enabled:
            raise ValueError(
                f"Bitstream/second-order motion-alignment policy mismatch: encoded={encoded_second_order_motion} decoder={self._p_second_order_motion_alignment_enabled}."
            )
        encoded_second_order_display = bool(payload.get("p_second_order_display_enabled", False))
        if encoded_second_order_display != self._p_second_order_display_enabled:
            raise ValueError(
                f"Bitstream/second-order display policy mismatch: encoded={encoded_second_order_display} decoder={self._p_second_order_display_enabled}."
            )
        encoded_third_order = bool(payload.get("p_third_order_reference_enabled", False))
        if encoded_third_order != self._p_third_order_reference_enabled:
            raise ValueError(
                f"Bitstream/third-order reference policy mismatch: encoded={encoded_third_order} decoder={self._p_third_order_reference_enabled}."
            )
        encoded_second_order_position_gain = bool(payload.get("p_second_order_position_gain_enabled", False))
        if encoded_second_order_position_gain != self._p_second_order_position_gain_enabled:
            raise ValueError(
                f"Bitstream/second-order position-gain policy mismatch: encoded={encoded_second_order_position_gain} decoder={self._p_second_order_position_gain_enabled}."
            )
        encoded_second_order_reliability_gate = bool(payload.get("p_second_order_reliability_gate_enabled", False))
        if encoded_second_order_reliability_gate != self._p_second_order_reliability_gate_enabled:
            raise ValueError(
                f"Bitstream/second-order reliability-gate policy mismatch: encoded={encoded_second_order_reliability_gate} decoder={self._p_second_order_reliability_gate_enabled}."
            )
        fmt = payload.get("format", "single_frame")
        if fmt == "rt_two_temporal_4d":
            decoded = self._decode_temporal_4d(payload)
            decoded = self.postprocess_latents(decoded, int(payload["qp"]))
            return self._semantic_reconstruct(decoded, semantic_size)
        if fmt == "rt_two_temporal_5d":
            decoded = self._decode_temporal_5d(payload)
            decoded = self.postprocess_latents(decoded, int(payload["qp"]))
            return self._semantic_reconstruct(decoded, semantic_size)
        raise ValueError(f"Unknown RTTwoCodecPipeline bitstream format: {fmt}")

    def _decode_single_frame_payload(self, payload: dict, is_intra: bool, pos_type_idx: int = 0) -> torch.Tensor:
        buf = io.BytesIO()
        torch.save(payload, buf)
        wrapped = buf.getvalue()
        if is_intra:
            role = str(payload.get("i_codec_role", "default"))
            i_codec = self._select_i_codec(role)
            z_out_codec = i_codec.decode(wrapped)
            state_codec = self._i_reference_codec_output(i_codec, z_out_codec)
            z_out = self._proxy_synthesis(z_out_codec)
            z_out = self._codec_output_transform(z_out, pos_type_idx)
            self._last_reference_output = self._codec_output_transform(self._proxy_synthesis(state_codec), pos_type_idx)
            return z_out
        self._activate_temporal_reference(pos_type_idx)
        z_prev = None
        if self._temporal_state is not None:
            z_prev = self._proxy_analysis(self._codec_input_transform(self._temporal_state["z_out"], pos_type_idx))
        feature_prev = self._temporal_feature_ref()
        dpb_feature_prev = self._temporal_dpb_feature_ref()
        ref_y_prev = self._temporal_dpb_y_ref()
        if z_prev is None:
            raise RuntimeError("P-frame decode requires a previous reconstructed latent.")
        frame_type_idx = int(payload.get("frame_type_idx", 0))
        z_prev = self._anchor_reference(z_prev, payload["qp"], pos_type_idx=pos_type_idx, frame_type_idx=frame_type_idx)
        p_codec = self._select_p_codec(frame_type_idx, role=str(payload.get("p_codec_role", "")))
        self._set_p_synthesis_condition(p_codec, pos_type_idx=pos_type_idx, qp=int(payload["qp"]))
        ctx, ctx_t = p_codec.get_temporal_context(
            z_prev,
            payload["qp"],
            feature_prev=feature_prev,
            feature_blend=self._temporal_feature_blend(),
            frame_type_idx=frame_type_idx,
            dpb_feature_prev=dpb_feature_prev,
        )
        z_out_codec = p_codec.decode(
            wrapped,
            ctx=ctx,
            ctx_t=ctx_t,
            z_prev=z_prev,
            feature_prev=feature_prev,
            feature_blend=self._temporal_feature_blend(),
            frame_type_idx=frame_type_idx,
            dpb_feature_prev=dpb_feature_prev,
            ref_y_prev=ref_y_prev,
        )
        base_state_codec = self._p_reference_codec_output(p_codec, z_out_codec)
        metric_codec = self._p_metric_codec_output(z_out_codec, base_state_codec, pos_type_idx=pos_type_idx)
        metric_codec = self._second_order_display(
            metric_codec, qp=int(payload["qp"]), pos_type_idx=pos_type_idx, frame_type_idx=frame_type_idx
        )
        state_codec = self._p_reference_state_codec_output(
            p_codec, base_state_codec, pos_type_idx=pos_type_idx, qp=int(payload["qp"])
        )
        z_out = self._codec_output_transform(self._proxy_synthesis(metric_codec), pos_type_idx)
        pre_display = self._codec_output_transform(self._proxy_synthesis(base_state_codec), pos_type_idx)
        self._last_reference_output = self._codec_output_transform(self._proxy_synthesis(state_codec), pos_type_idx)
        z_out = self._p_display_tokenizer_output(
            z_out, pre_display, p_codec, qp=int(payload["qp"]), pos_type_idx=pos_type_idx
        )
        return z_out

    def _decode_temporal_4d(self, payload: dict) -> torch.Tensor:
        self._last_entropy_scale_sequence = None
        self._last_quantization_width_sequence = None
        C, T, H, W = payload["shape"]
        self.reset_temporal_state()
        scale_frames: list[torch.Tensor] = []
        width_frames = []
        decoded_frames = []
        for t, frame_payload in enumerate(payload["frames"]):
            is_intra = str(frame_payload.get("codec", "")) == "dmci"
            z_out = self._decode_single_frame_payload(frame_payload, is_intra=is_intra, pos_type_idx=t)
            if getattr(self, "_export_entropy_scales", False):
                codec = (
                    self._select_i_codec(str(frame_payload.get("i_codec_role", "default")))
                    if is_intra
                    else self._select_p_codec(
                        int(frame_payload.get("frame_type_idx", 0)), role=str(frame_payload.get("p_codec_role", ""))
                    )
                )
                scale_map = getattr(codec, "_last_entropy_scale_map", None)
                if scale_map is None:
                    raise RuntimeError(f"decoder did not publish entropy scale map at frame {t}")
                scale_frames.append(scale_map)
                width_frames.append(getattr(codec, "_last_quantization_width_map", torch.ones_like(scale_map)))
            state_out = self._last_reference_output.detach()
            if z_out.ndim == 4 and z_out.shape[0] == 1:
                z_out = z_out.squeeze(0)
            decoded_frames.append(z_out)
            p_codec = (
                None
                if is_intra
                else self._select_p_codec(
                    int(frame_payload.get("frame_type_idx", 0)), role=str(frame_payload.get("p_codec_role", ""))
                )
            )
            next_state = self._make_temporal_state(state_out, is_intra=is_intra, p_codec=p_codec)
            self._commit_temporal_state(next_state, is_intra=is_intra, position_idx=t)
            if is_intra:
                self._anchor_state = state_out.detach()
        if getattr(self, "_export_entropy_scales", False):
            shapes = {tuple(v.shape[-2:]) for v in scale_frames}
            if len(shapes) != 1:
                raise ValueError(f"I/P entropy scale grids differ across frames: {sorted(shapes)}")
            self._last_entropy_scale_sequence = torch.stack(scale_frames, dim=2)
            self._last_quantization_width_sequence = torch.stack(width_frames, dim=2)
        return self._denormalize(torch.stack(decoded_frames, dim=1))

    def _decode_temporal_5d(self, payload: dict) -> torch.Tensor:
        self._last_entropy_scale_sequence = None
        self._last_quantization_width_sequence = None
        B, C, T, H, W = payload["shape"]
        decoded_batches = []
        scale_batches: list[torch.Tensor] = []
        width_batches = []
        for batch_payloads in payload["batches"]:
            self.reset_temporal_state()
            scale_frames: list[torch.Tensor] = []
            width_frames = []
            decoded_frames = []
            for t, frame_payload in enumerate(batch_payloads):
                is_intra = str(frame_payload.get("codec", "")) == "dmci"
                z_out = self._decode_single_frame_payload(frame_payload, is_intra=is_intra, pos_type_idx=t)
                if getattr(self, "_export_entropy_scales", False):
                    codec = (
                        self._select_i_codec(str(frame_payload.get("i_codec_role", "default")))
                        if is_intra
                        else self._select_p_codec(
                            int(frame_payload.get("frame_type_idx", 0)), role=str(frame_payload.get("p_codec_role", ""))
                        )
                    )
                    scale_map = getattr(codec, "_last_entropy_scale_map", None)
                    if scale_map is None:
                        raise RuntimeError(f"decoder did not publish entropy scale map at batch frame {t}")
                    scale_frames.append(scale_map)
                    width_frames.append(getattr(codec, "_last_quantization_width_map", torch.ones_like(scale_map)))
                state_out = self._last_reference_output.detach()
                if z_out.ndim == 4 and z_out.shape[0] == 1:
                    z_out = z_out.squeeze(0)
                decoded_frames.append(z_out)
                p_codec = (
                    None
                    if is_intra
                    else self._select_p_codec(
                        int(frame_payload.get("frame_type_idx", 0)), role=str(frame_payload.get("p_codec_role", ""))
                    )
                )
                next_state = self._make_temporal_state(state_out, is_intra=is_intra, p_codec=p_codec)
                self._commit_temporal_state(next_state, is_intra=is_intra, position_idx=t)
                if is_intra:
                    self._anchor_state = state_out.detach()
            decoded_batches.append(torch.stack(decoded_frames, dim=1))
            if getattr(self, "_export_entropy_scales", False):
                shapes = {tuple(v.shape[-2:]) for v in scale_frames}
                if len(shapes) != 1 or len(scale_frames) != T:
                    raise ValueError(
                        f"I/P entropy scale grids invalid in batch {len(decoded_batches) - 1}: frames={len(scale_frames)} shapes={sorted(shapes)}"
                    )
                scale_batches.append(torch.stack(scale_frames, dim=2))
                width_batches.append(torch.stack(width_frames, dim=2))
        if getattr(self, "_export_entropy_scales", False):
            self._last_entropy_scale_sequence = torch.cat(scale_batches, dim=0)
            self._last_quantization_width_sequence = torch.cat(width_batches, dim=0)
        return self._denormalize(torch.stack(decoded_batches, dim=0))

    def get_actual_rate(self, bitstream: bytes, num_pixels: int) -> float:
        """Return actual bitrate in bpp from the raw entropy-coded streams."""
        if num_pixels <= 0:
            raise ValueError(f"num_pixels must be positive, got {num_pixels}")
        payload = torch.load(io.BytesIO(bitstream), map_location="cpu", weights_only=False)
        fmt = payload.get("format", "single_frame")
        if fmt == "rt_two_temporal_4d":
            raw_len = sum((len(f.get("bit_stream", b"")) for f in payload["frames"]))
        elif fmt == "rt_two_temporal_5d":
            raw_len = sum((len(f.get("bit_stream", b"")) for batch in payload["batches"] for f in batch))
        else:
            raw_len = len(bitstream)
        if payload.get("semantic_anchor") is not None:
            raw_len += self._SEMANTIC_HEADER_BYTES
        return 8 * raw_len / num_pixels
