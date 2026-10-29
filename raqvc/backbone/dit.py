"""Wan backbone with the context and velocity adapters used by C512."""

from pathlib import Path
import sys
from typing import List
import torch
from torch import nn
from torch.nn import functional as F
from raqvc.backbone.conditions import DiTConditions
from raqvc.backbone.rate_embedding import RateEmbeddingMLP
from raqvc.utils.paths import repo_root

_WAN_ADDED = False


def _stack_preembedded_context(
    context: List[torch.Tensor],
    *,
    hidden_dim: int,
    max_context_len: int,
) -> torch.Tensor:
    """Stack already-projected context without introducing padding tokens.

    Wan's native raw-text path pads caption embeddings to ``text_len`` before
    projecting them.  RAQ-Flow's no-text/rate/codec path is different: every
    token is already in Wan's hidden dimension and the model does not pass an
    attention mask (``context_lens`` is ``None``).  Padding that path with
    hundreds of zero tokens therefore changes cross-attention instead of being
    a harmless batching operation.

    The current A1 batches construct the same number of conditioning tokens for
    every sample.  Stack that true length directly.  If a future caller builds
    ragged hidden-width contexts, fail explicitly until a real attention-mask
    path is implemented rather than silently reintroducing unmasked padding.
    """
    if not context:
        raise ValueError("pre-embedded context batch must contain at least one sample")

    lengths = []
    for index, sample in enumerate(context):
        if not torch.is_tensor(sample) or sample.dim() != 2:
            shape = tuple(sample.shape) if torch.is_tensor(sample) else type(sample).__name__
            raise ValueError(
                "pre-embedded context samples must have shape [L, hidden_dim]; " f"sample {index} has {shape}"
            )
        if sample.size(-1) != hidden_dim:
            raise ValueError(
                f"pre-embedded context sample {index} has width {sample.size(-1)}, " f"expected hidden_dim={hidden_dim}"
            )
        lengths.append(int(sample.size(0)))

    if len(set(lengths)) != 1:
        raise ValueError(
            "ragged pre-embedded context is unsupported without an attention mask; " f"got per-sample lengths {lengths}"
        )
    if lengths[0] > max_context_len:
        raise ValueError(f"pre-embedded context length {lengths[0]} exceeds Wan text_len={max_context_len}")

    return torch.stack(context, dim=0)


def _ensure_wan_on_path() -> None:
    """Add Wan2.2 to sys.path (lazy, once)."""
    global _WAN_ADDED
    if _WAN_ADDED:
        return
    wan_root = repo_root() / "third_party" / "Wan2.2"
    if not wan_root.exists():
        raise RuntimeError(f"third_party/Wan2.2 not found at {wan_root}. " "See third_party/VERSIONS.md for setup.")
    if str(wan_root) not in sys.path:
        sys.path.insert(0, str(wan_root))
    _WAN_ADDED = True


class SpatialAnchorFusion(nn.Module):
    """Inject decoder-known anchor geometry into the DiT latent grid.

    Cross-attention codec tokens deliberately compress spatial information.
    Native-LL detail synthesis instead needs a position-aligned path.  This
    residual 3-D adapter preserves the pretrained DiT input at initialization
    (zero final projection) while allowing local space-time anchor evidence to
    enter the native Wan patch embedding.
    """

    def __init__(
        self,
        channels: int,
        *,
        hidden_channels: int = 96,
        residual_scale: float = 1.0,
    ) -> None:
        super().__init__()
        if channels < 1 or hidden_channels < 1:
            raise ValueError("fusion channel counts must be positive")
        self.residual_scale = float(residual_scale)
        self.net = nn.Sequential(
            nn.Conv3d(channels, hidden_channels, kernel_size=3, padding=1),
            nn.SiLU(),
            nn.Conv3d(hidden_channels, channels, kernel_size=3, padding=1),
        )
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, state: torch.Tensor, anchor: torch.Tensor) -> torch.Tensor:
        if state.ndim != 5 or anchor.ndim != 5:
            raise ValueError("state and anchor must be [B, C, T, H, W]")
        if state.shape != anchor.shape:
            raise ValueError(f"state/anchor shape mismatch {tuple(state.shape)} vs {tuple(anchor.shape)}")
        param = next(self.net.parameters())
        correction = self.net(anchor.to(device=param.device, dtype=param.dtype))
        return state + self.residual_scale * correction.to(state.dtype)


class RAQFlowDiT(nn.Module):
    """One whole-GOP Wan call, with fixed active adapter structure."""

    def __init__(self, wan_checkpoint_dir, device="cuda"):
        super().__init__()
        _ensure_wan_on_path()
        from wan.modules.model import WanModel
        from raqvc.backbone.perceptual_velocity_adapter import PerceptualVelocityAdapter

        self.dit = WanModel(
            model_type="ti2v",
            patch_size=(1, 2, 2),
            text_len=512,
            in_dim=48,
            dim=3072,
            ffn_dim=14336,
            freq_dim=256,
            text_dim=4096,
            out_dim=48,
            num_heads=24,
            num_layers=30,
            window_size=(-1, -1),
            qk_norm=True,
            cross_attn_norm=True,
            eps=1e-6,
        )
        path = Path(wan_checkpoint_dir) / "low_noise_model/diffusion_pytorch_model.bin"
        state = torch.load(path, map_location="cpu", mmap=True, weights_only=True)
        defaults = self.dit.state_dict()
        for k in defaults:
            if "eem_latent_embedding" in k and k not in state:
                state[k] = defaults[k].clone()
        self.dit.load_state_dict(state, strict=True)
        self.dit.eval()
        self.dit.prepare_text_context = lambda context: _stack_preembedded_context(
            context, hidden_dim=3072, max_context_len=512
        )
        self.rate_mlp = RateEmbeddingMLP(hidden_dim=3072)
        self.transition_feature_dim = 8
        self.transition_mlp = nn.Sequential(nn.Linear(8, 512), nn.SiLU(), nn.Linear(512, 3072))
        self.codec_context_feature_dim = 48
        self.codec_context_mlp = nn.Sequential(nn.Linear(48, 512), nn.SiLU(), nn.Linear(512, 3072))
        self._caching_attns = self._install_caching_attention()
        self.device = device
        self.dtype = torch.bfloat16
        self.parameter_dtype = torch.float32
        self.velocity_scale = nn.Parameter(torch.ones(1, device=device, dtype=torch.float32))
        self.perceptual_adapter = PerceptualVelocityAdapter(
            latent_channels=48,
            hidden_channels=96,
            num_blocks=6,
            num_qp_points=4,
            residual_scale=0.25,
            qp_strengths=(0.25, 0.5, 0.85, 1.0),
        )
        self.spatial_anchor_fusion = SpatialAnchorFusion(channels=48, hidden_channels=96, residual_scale=1.0)
        self.to(device=device, dtype=torch.float32)

    def build_transition_token(self, features: torch.Tensor) -> torch.Tensor:
        """Embed explicit rate-ladder transition features for A1.

        ``features`` is expected to be ``[B, 8]``. The returned tensor is
        shaped like ``rate_noise_tok``. Historical callers add it to their
        rate token; F2 uses it as the sole global condition token without
        changing Wan's context interface.
        """
        if features.dim() != 2 or features.size(-1) != self.transition_feature_dim:
            raise ValueError(
                f"transition features must have shape [B, {self.transition_feature_dim}], "
                f"got {tuple(features.shape)}"
            )
        param = next(self.transition_mlp.parameters())
        x = features.to(device=param.device, dtype=param.dtype)
        token = self.transition_mlp(x)
        return token.reshape(token.size(0), 1, token.size(-1))

    def build_codec_context_tokens(
        self,
        z_cond: torch.Tensor,
        pool_shape: tuple[int, int, int] = (1, 4, 4),
    ) -> torch.Tensor:
        """Embed a compressed latent into compact context tokens.

        ``z_cond`` is expected to be ``[B, 48, T, H, W]``.  The latent is
        adaptively pooled over time/space, flattened to ``[B, L, 48]``, and
        projected to Wan hidden size so it can be appended to text/rate tokens.
        """
        if z_cond.dim() != 5 or z_cond.size(1) != self.codec_context_feature_dim:
            raise ValueError(
                f"codec context latent must have shape [B, {self.codec_context_feature_dim}, T, H, W], "
                f"got {tuple(z_cond.shape)}"
            )
        if len(pool_shape) != 3:
            raise ValueError(f"pool_shape must be (T, H, W), got {pool_shape}")
        param = next(self.codec_context_mlp.parameters())
        z = z_cond.to(device=param.device, dtype=param.dtype)
        pooled = F.adaptive_avg_pool3d(z, output_size=tuple(int(x) for x in pool_shape))
        tokens = pooled.flatten(2).transpose(1, 2).contiguous()
        return self.codec_context_mlp(tokens)

    def _install_caching_attention(self) -> List:
        """Replace WanSelfAttention with CachingWanSelfAttention in all blocks.

        Returns:
            List of CachingWanSelfAttention instances (for mode control)
        """
        from wan.modules.model import WanSelfAttention
        from raqvc.backbone.reliability_attention import CachingWanSelfAttention

        caching_attns = []
        for block in self.dit.blocks:
            if isinstance(block.self_attn, WanSelfAttention):
                # Wrap the original attention
                caching_attn = CachingWanSelfAttention(block.self_attn)
                block.self_attn = caching_attn
                caching_attns.append(caching_attn)
        return caching_attns

    def set_self_attention_reliability(
        self,
        reliability: torch.Tensor | None,
        *,
        query_floor: float = 0.20,
        max_log_bias: float = 6.0,
    ) -> None:
        """Install decoder-known token reliability for the next Wan forward."""

        for attn in self._caching_attns:
            attn.set_reliability_bias(
                reliability,
                query_floor=query_floor,
                max_log_bias=max_log_bias,
            )

    def forward(self, z_t, t, conditions, *, schedule_idx, restoration_progress):
        adapter_input = z_t
        z_t = self.spatial_anchor_fusion(z_t, conditions.spatial_anchor)
        context = torch.cat([conditions.text_ctx, conditions.rate_noise_tok], dim=1)
        context = torch.cat([context, conditions.codec_ctx.to(context)], dim=1)
        if z_t.ndim != 5:
            raise ValueError("Expected [B,48,T,H,W] latent")
        x_list = list(z_t.unbind(0))
        ps = self.dit.patch_size
        seq_len = max((x.shape[1] // ps[0]) * (x.shape[2] // ps[1]) * (x.shape[3] // ps[2]) for x in x_list)
        with torch.amp.autocast("cuda", dtype=self.dtype):
            velocity = self.dit(
                x=x_list, t=t, context=list(context.unbind(0)), seq_len=seq_len, y=None, eem_latent=None
            )
        velocity = torch.stack(velocity, dim=0) * self.velocity_scale
        return self.perceptual_adapter(
            adapter_input, velocity, progress=restoration_progress, schedule_idx=schedule_idx
        )
