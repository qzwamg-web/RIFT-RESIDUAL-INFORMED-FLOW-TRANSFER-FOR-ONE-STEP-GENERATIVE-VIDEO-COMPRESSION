"""Within-GOP reliability attention. No cross-GOP cache or receiver branches."""

import torch
from torch import nn


def reliability_biased_attention(
    attention_fn,
    *,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    q_reliability: torch.Tensor | None,
    k_reliability: torch.Tensor | None,
    query_floor: float,
    max_log_bias: float,
    **kwargs,
) -> torch.Tensor:
    """Apply an exact separable soft reliability bias with FlashAttention.

    The desired logit term is ``query_strength * log(key_reliability)``.
    Appending a small padded feature block to Q/K realizes that rank-one term
    exactly while retaining the fused attention kernel.  V is zero padded and
    sliced back to its original width.  Setting ``softmax_scale`` to the
    original head scale keeps the content logits bit-for-bit calibrated.
    """

    if q_reliability is None or k_reliability is None:
        return attention_fn(q=q, k=k, v=v, **kwargs)
    if q.ndim != 4 or k.ndim != 4 or v.ndim != 4:
        raise ValueError("reliability-biased attention expects [B,L,H,D]")
    if tuple(q_reliability.shape) != (int(q.shape[0]), int(q.shape[1])):
        raise ValueError("query reliability does not match attention queries")
    if tuple(k_reliability.shape) != (int(k.shape[0]), int(k.shape[1])):
        raise ValueError("key reliability does not match attention keys")
    original_dim = int(q.shape[-1])
    if int(k.shape[-1]) != original_dim or int(v.shape[-1]) != original_dim:
        raise ValueError("Q/K/V head widths must match before reliability padding")
    padded_dim = ((original_dim + 1 + 7) // 8) * 8
    if padded_dim > 256:
        raise ValueError("reliability padding exceeds FlashAttention head limit")
    auxiliary = padded_dim - original_dim
    scale = original_dim**-0.5
    query_strength = float(query_floor) + (1.0 - float(query_floor)) * q_reliability.float()
    key_log = k_reliability.float().clamp_min(1e-8).log().clamp_min(-float(max_log_bias))
    q_extra = query_strength[:, :, None, None].expand(-1, -1, q.shape[2], auxiliary)
    k_extra = (key_log[:, :, None, None] / (float(auxiliary) * scale)).expand(-1, -1, k.shape[2], auxiliary)
    v_extra = v.new_zeros((*v.shape[:-1], auxiliary))
    output = attention_fn(
        q=torch.cat((q, q_extra.to(q)), dim=-1),
        k=torch.cat((k, k_extra.to(k)), dim=-1),
        v=torch.cat((v, v_extra), dim=-1),
        softmax_scale=scale,
        **kwargs,
    )
    return output[..., :original_dim]


class CachingWanSelfAttention(nn.Module):
    """Name retained for original projection keys; this module does not cache."""

    def __init__(self, original_attn):
        super().__init__()
        self._original = original_attn
        self._current_reliability = None
        self._reliability_query_floor = 0.2
        self._reliability_max_log_bias = 6.0

    def __getattr__(self, name):
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self._original, name)

    def set_reliability_bias(self, reliability, *, query_floor=0.2, max_log_bias=6.0):
        self._current_reliability = reliability
        self._reliability_query_floor = query_floor
        self._reliability_max_log_bias = max_log_bias

    def forward(self, x, seq_lens, grid_sizes, freqs):
        from wan.modules.model import rope_apply
        from wan.modules.attention import attention

        orig = self._original
        b, s, n, d = *x.shape[:2], orig.num_heads, orig.head_dim
        rel = self._current_reliability
        if rel is not None:
            if tuple(rel.shape) != (b, s):
                raise ValueError("Reliability shape must match token grid")
            rel = rel.to(device=x.device, dtype=torch.float32)
        q = orig.norm_q(orig.q(x)).view(b, s, n, d)
        k = orig.norm_k(orig.k(x)).view(b, s, n, d)
        v = orig.v(x).view(b, s, n, d)
        q = rope_apply(q, grid_sizes, freqs)
        k = rope_apply(k, grid_sizes, freqs)
        out = reliability_biased_attention(
            attention,
            q=q,
            k=k,
            v=v,
            q_reliability=rel,
            k_reliability=rel,
            query_floor=self._reliability_query_floor,
            max_log_bias=self._reliability_max_log_bias,
            k_lens=seq_lens,
            window_size=orig.window_size,
        )
        return orig.o(out.flatten(2))
