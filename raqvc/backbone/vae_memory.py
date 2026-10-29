"""Opt-in activation recomputation with explicit native Wan decoder cache IO.

Each temporal decoder call becomes a pure function of its input and incoming
cache. Recomputations mutate private lists only; the public cache is updated
once by the original forward. Cache tensors stay in autograd (never detach).
"""

from types import MethodType
import torch
from torch.utils.checkpoint import checkpoint


def configure_vae_stateless_checkpointing(vae):
    """Checkpoint the native Decoder3d; weights and inference stay unchanged."""
    model = getattr(vae, "model", vae)
    decoder = getattr(model, "decoder", None)
    if decoder is None or type(decoder).__name__ != "Decoder3d":
        raise TypeError("Task938 recomputation requires the native Wan Decoder3d")
    if hasattr(decoder, "_task938_original_forward"):
        return 0
    decoder._task938_original_forward = decoder.forward

    def forward(self, x, feat_cache=None, feat_idx=None, first_chunk=False):
        original = self._task938_original_forward
        if not torch.is_grad_enabled() or not x.requires_grad:
            return original(x, feat_cache, feat_idx, first_chunk)
        if feat_cache is None or feat_idx is None:
            raise ValueError("native temporal decoder requires explicit cache and index")
        # Tuples snapshot list membership without cloning/detaching tensors.
        incoming = tuple(feat_cache)
        start = int(feat_idx[0])

        def functional(value, cache, index, first):
            private_cache = list(cache)
            private_index = [index]
            output = original(value, private_cache, private_index, first)
            return output, tuple(private_cache), private_index[0]

        output, outgoing, end = checkpoint(
            functional,
            x,
            incoming,
            start,
            first_chunk,
            use_reentrant=False,
        )
        feat_cache[:] = outgoing
        feat_idx[0] = end
        return output

    decoder.forward = MethodType(forward, decoder)
    return 1


def decode_latent_batch(vae, latent, batch_size=1):
    """Decode independent [B,C,T,H,W] windows with native batched cache reset.

    The high-level Wan list API loops over batch elements. The native model
    already carries an independent leading batch dimension through all caches.
    Only same-shaped windows are accepted. Gradients remain enabled by caller.
    """
    if latent.ndim != 5 or int(batch_size) < 1:
        raise ValueError("expected [B,C,T,H,W] and positive decoder batch size")
    if not hasattr(vae, "scale") or not hasattr(vae, "model"):
        raise TypeError("batched decode requires the native Wan VAE wrapper")
    outputs = []
    for start in range(0, latent.shape[0], int(batch_size)):
        with torch.autocast(latent.device.type, dtype=vae.dtype):
            value = vae.model.decode(latent[start : start + int(batch_size)], vae.scale)
            outputs.append(value.float().clamp(-1, 1))
    return torch.cat(outputs, dim=0)
