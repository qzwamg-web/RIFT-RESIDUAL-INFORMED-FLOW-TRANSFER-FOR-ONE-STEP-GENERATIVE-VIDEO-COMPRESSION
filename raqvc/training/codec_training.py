"""Actual-rANS-aligned online codec -> one-step DiT joint training helpers."""

import io
from contextlib import contextmanager
from pathlib import Path
import torch
from omegaconf import OmegaConf
from raqvc.codec import create_latent_codec
from raqvc.final_support import normalize_codec_state


def load_regional_codec(config, checkpoint, device, *, initialize_new=False):
    cfg = OmegaConf.load(config)
    kwargs = OmegaConf.to_container(cfg.codec, resolve=True)
    kind = kwargs.pop("type")
    kwargs.pop("enabled", None)
    kwargs["device"] = str(device)
    codec = create_latent_codec(kind.replace("-", "_"), **kwargs)
    payload = torch.load(checkpoint, map_location="cpu", mmap=True, weights_only=False)
    missing, unexpected = codec.load_state_dict(normalize_codec_state(payload["model_state_dict"]), strict=False)
    allowed = {k for k in codec.state_dict() if ".task938_region_quantizer." in k} if initialize_new else set()
    if unexpected or set(missing) - allowed:
        raise ValueError(f"codec initialization mismatch missing={missing} unexpected={unexpected}")
    if initialize_new and set(missing) != allowed:
        raise ValueError("regional initialization must start from a parent without regional weights")
    codec = codec.to(device).float()
    codec.enable_entropy_scale_export(True)
    return codec


def regional_parameters(codec):
    return [p for (n, p) in codec.named_parameters() if ".task938_region_quantizer." in n]


def _stream_frame_bits(payload):
    """Return actual encoded bits per temporal frame from the rANS payload."""
    if payload.get("format") == "rt_two_temporal_5d":
        batches = payload.get("batches", [])
        if len(batches) != 1:
            raise ValueError("metadata capture expects a singleton codec batch")
        return [8 * len(frame.get("bit_stream", b"")) for frame in batches[0]]
    if payload.get("format") == "rt_two_temporal_4d":
        return [8 * len(frame.get("bit_stream", b"")) for frame in payload.get("frames", [])]
    return [8 * len(payload.get("bit_stream", b""))]


def _copy_symbol_payload(value):
    if value is None:
        return None
    if isinstance(value, (tuple, list)):
        return tuple((_copy_symbol_payload(item) for item in value))
    if torch.is_tensor(value):
        return value.detach().cpu().clone()
    return value


def _decoder_metadata(codec, stream, width):
    """Collect only decoder-visible/bitstream-derived cache fields.

    Entropy means are not currently published by the codec and are explicitly
    marked unavailable instead of being reconstructed from clean tensors.
    """
    payload = torch.load(io.BytesIO(stream), map_location="cpu", weights_only=False)
    p_symbols = _copy_symbol_payload(getattr(codec, "_last_hard_p_symbol_tensors", []))
    i_symbols = _copy_symbol_payload(getattr(codec, "_last_hard_i_symbol_tensors", []))

    def zero_masks(items):
        result = []
        for item in items or []:
            if not isinstance(item, dict):
                result.append(None)
                continue
            ys = item.get("y")
            result.append(None if ys is None else tuple((torch.as_tensor(y == 0) for y in ys)))
        return tuple(result)

    zero_mask_i = zero_masks(i_symbols)
    zero_mask_p = zero_masks(p_symbols)
    groups = list(zero_mask_i) + list(zero_mask_p)
    zero_mask_grid = None
    if groups and all((group for group in groups)):
        planes = []
        for group in groups:
            tensors = [torch.as_tensor(item, dtype=torch.float32) for item in group]
            planes.append(torch.cat(tensors, dim=1).mean(1, keepdim=True))
        zero_mask_grid = torch.stack(planes, dim=2)
    return {
        "frame_bits_actual": _stream_frame_bits(payload),
        "hard_entropy_symbols_p": p_symbols,
        "hard_entropy_symbols_i": i_symbols,
        "zero_mask_p": zero_mask_p,
        "zero_mask_i": zero_mask_i,
        "zero_mask_grid": zero_mask_grid,
        "quantization_delta_proxy": width.detach().cpu().clone(),
        "entropy_mean": None,
        "entropy_mean_available": False,
        "metadata_schema": "task938_decoder_visible_cache_v2",
    }


@contextmanager
def _hard_codec_mode(codec, enabled):
    training = codec.training
    if enabled:
        codec.eval()
    try:
        yield
    finally:
        if enabled:
            codec.train(training)


def online_reconstruction(codec, clean, qp, *, verify=True, return_metadata=False, hard_eval=False):
    if len(clean) > 1:
        parts = [
            online_reconstruction(
                codec, x[None], qp, verify=verify, return_metadata=return_metadata, hard_eval=hard_eval
            )
            for x in clean
        ]
        base = (
            torch.cat([v[0] for v in parts]),
            torch.stack([v[1] for v in parts]).mean(),
            torch.cat([v[2] for v in parts]),
            torch.cat([v[3] for v in parts]),
            torch.cat([v[4] for v in parts]),
        )
        return base + ([v[5] for v in parts],) if return_metadata else base
    with (
        torch.no_grad(),
        torch.autocast(device_type=clean.device.type, enabled=False),
        _hard_codec_mode(codec, hard_eval),
    ):
        if return_metadata:
            codec._capture_exact_p_entropy_symbols = True
            for module in codec.modules():
                if hasattr(module, "_capture_exact_entropy_symbols"):
                    module._capture_exact_entropy_symbols = True
        stream, encoded = codec.encode(clean.float(), qp=qp, p_shift_qp=False)
        hard = codec.decode(stream)
        if not torch.equal(encoded, hard):
            raise ValueError("independent decoder mismatch")
        decoder_sigma = codec.get_last_entropy_scale_sequence().detach().clone()
        decoder_width = codec.get_last_quantization_width_sequence().detach().clone()
        payload = torch.load(io.BytesIO(stream), map_location="cpu", weights_only=False)
        pixels = (1 + 4 * (clean.shape[2] - 1)) * clean.shape[-2] * 16 * clean.shape[-1] * 16
        actual = torch.tensor(
            [sum((8 * len(f["bit_stream"]) for f in fs)) / pixels for fs in payload["batches"]], device=clean.device
        )
    with torch.autocast(device_type=clean.device.type, enabled=False):
        source, rate = codec(clean.float(), qp=qp, temporal_chunk=True, p_shift_qp=False, reset_temporal_state=True)
    if verify and (not torch.allclose(source.detach(), hard, atol=2e-05, rtol=2e-05)):
        failure_dir = getattr(codec, "_task938_failure_dir", None)
        if failure_dir is not None:
            from pathlib import Path

            dest = Path(failure_dir) / "CODEC_PARITY_FAILURE.pt"
            if not dest.exists():
                torch.save(
                    {
                        "qp": int(qp),
                        "clean": clean.detach().cpu(),
                        "hard": hard.detach().cpu(),
                        "train_source": source.detach().cpu(),
                        "stream": stream,
                        "codec_state": {k: v.detach().cpu() for (k, v) in codec.state_dict().items()},
                        "cpu_rng": torch.get_rng_state(),
                        "cuda_rng": torch.cuda.get_rng_state_all(),
                        "codec_training": codec.training,
                    },
                    dest,
                )
        raise ValueError(f"train/hard mismatch: {float((source.detach() - hard).abs().max())}")
    if return_metadata:
        return (source, rate, actual, decoder_sigma, decoder_width, _decoder_metadata(codec, stream, decoder_width))
    return (source, rate, actual, decoder_sigma, decoder_width)
