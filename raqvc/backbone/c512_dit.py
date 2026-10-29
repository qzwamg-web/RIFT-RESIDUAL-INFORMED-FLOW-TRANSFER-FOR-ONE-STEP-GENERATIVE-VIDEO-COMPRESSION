"""C512 single-call restoration. Fixed QP time; independent GOPs.

Only decoder-visible context used by the selected C512 checkpoint is retained.
There is no dynamic time mapper, time-scaled velocity, or cross-GOP KV input.
Legacy class names remain solely for checkpoint/import compatibility.
"""

from __future__ import annotations
import torch
from torch import Tensor, nn
from raqvc.backbone.conditions import DiTConditions
from raqvc.backbone.source_conditioning import Task438SourceConditioner, Task438SourceObservation
from raqvc.backbone.reliability import Task456SoftReliability, SoftReliabilityConfig
from raqvc.backbone.spatial_conditioning import (
    Task614SpatialNoiseConfig,
    Task614SpatialNoiseConditioner,
    decoded_spatial_noise_scale_map,
)
from raqvc.flow.normalization import QPConditionalFlowPath

_TASK438_QPS = (0, 16, 39, 63)
_TASK438_TASK297_RATES = (0.020, 0.012, 0.006, 0.003)
TAU_TABLE = (520.0, 640.0, 800.0, 960.0)


class Task438QPSolverCalibrator(nn.Module):
    """Inherited QP calibration; only the single-step column is evaluated."""

    def __init__(self):
        super().__init__()
        self.raw = nn.Parameter(torch.zeros(4, 3))

    def scale(self, qp_index, device, dtype):
        return torch.exp(0.12 * torch.tanh(self.raw[qp_index, 0])).to(device=device, dtype=dtype)


class Task438WindowedDiT(nn.Module):
    """The deployed C512 backbone and its active inherited conditioning."""

    def __init__(self, backbone):
        super().__init__()
        self.backbone = backbone
        self.path = QPConditionalFlowPath()
        self.conditioner = Task438SourceConditioner(output_dim=int(backbone.dit.dim), hidden_dim=256)
        self.reliability = Task456SoftReliability(
            SoftReliabilityConfig(floor=0.08, write_threshold=0.7, query_floor=0.2, max_log_bias=6.0)
        )
        self.spatial_noise_config = Task614SpatialNoiseConfig()
        self.spatial_noise_conditioner = Task614SpatialNoiseConditioner(int(backbone.dit.in_dim))
        self.solver_step_calibrator = Task438QPSolverCalibrator()
        self.wan_native_timestep_bridge = True
        self.gop_contract = "independent_reset_i_p6"

    @staticmethod
    def _direct_clean_transition_features(batch: int, qp_index: int, device: torch.device) -> Tensor:
        """Reproduce Task297 direct-codec-to-clean transition features."""

        index = int(qp_index)
        if index < 0 or index >= len(_TASK438_QPS):
            raise ValueError(f"Task438 QP index outside [0,3]: {index}")
        qp = float(_TASK438_QPS[index])
        rate = float(_TASK438_TASK297_RATES[index])
        maximum_rate = float(max(_TASK438_TASK297_RATES))
        values = (
            float(index) / 3.0,
            0.0,
            1.0,
            rate / maximum_rate,
            1.0,
            qp / 63.0,
            0.0,
            (maximum_rate - rate) / maximum_rate,
        )
        return torch.tensor(values, device=device, dtype=torch.float32).unsqueeze(0).expand(batch, -1).contiguous()

    def _base_conditions(self, state, source_anchor, qp):
        batch = state.shape[0]
        qi = int(self.path.qp_indices(qp).item())
        text = state.new_zeros((batch, 1, int(self.backbone.dit.dim)))
        features = self._direct_clean_transition_features(batch, qi, state.device)
        transition = self.backbone.build_transition_token(features)
        indices = torch.full((batch,), qi, device=state.device, dtype=torch.long)
        rate = self.backbone.rate_mlp(None, None, qp_idx=indices)
        rate = rate + transition.to(rate.dtype)
        codec_ctx = self.backbone.build_codec_context_tokens(source_anchor, pool_shape=(4, 2, 2)) * 0.6
        return DiTConditions(text_ctx=text, rate_noise_tok=rate, codec_ctx=codec_ctx, spatial_anchor=source_anchor)

    def forward(self, state, *, qp, log_q_y, log_q_z, actual_bpp):
        if state.ndim != 5 or state.shape[1] != 48:
            raise ValueError("Expected latent [B,48,T,H,W]")
        qi = _TASK438_QPS.index(int(qp))
        batch = state.shape[0]
        t = state.new_zeros(batch)
        qps = torch.full((batch,), int(qp), device=state.device, dtype=torch.long)
        roles = torch.zeros(batch, device=state.device, dtype=torch.long)
        observation = Task438SourceObservation(
            qp_value=qps,
            log_q_y=log_q_y,
            log_q_z=log_q_z,
            actual_bpp=actual_bpp,
            window_role=roles,
            temporal_offset=roles,
        )
        rms = self.path.sigma(qp, batch=batch).to(device=state.device, dtype=torch.float32)
        reliability = self.reliability(
            state.detach(), qp_value=qps, log_q_y=log_q_y, actual_bpp=actual_bpp, residual_rms=rms, role_index=0
        )
        pt, ph, pw = map(int, self.backbone.dit.patch_size)
        if any(n % p for n, p in zip(state.shape[2:], (pt, ph, pw))):
            raise ValueError("Latent dimensions must align to DiT patches")
        if pt > 1:
            reliability = reliability.reshape(batch, -1, pt).mean(2)
        reliability = reliability.repeat_interleave((state.shape[3] // ph) * (state.shape[4] // pw), dim=1)
        self.backbone.set_self_attention_reliability(reliability, query_floor=0.2, max_log_bias=6.0)
        conditions = self._base_conditions(state, state, qp)
        source_token = self.conditioner(observation, rms)
        conditions = DiTConditions(
            text_ctx=conditions.text_ctx,
            rate_noise_tok=conditions.rate_noise_tok + source_token.to(conditions.rate_noise_tok),
            codec_ctx=conditions.codec_ctx,
            spatial_anchor=state,
        )
        spatial_map = decoded_spatial_noise_scale_map(state, self.spatial_noise_config)
        conditioned = self.spatial_noise_conditioner(state, spatial_map)
        tokens = (state.shape[2] // pt) * (state.shape[3] // ph) * (state.shape[4] // pw)
        tau = torch.full((batch, tokens), TAU_TABLE[qi], device=state.device, dtype=torch.float32)
        velocity = self.backbone(
            conditioned, tau, conditions, schedule_idx=self.path.qp_indices(qp), restoration_progress=t
        )
        velocity = -velocity
        scale = self.solver_step_calibrator.scale(qi, velocity.device, velocity.dtype)
        velocity = velocity * scale
        return velocity / self.path.sigma_like(qp, velocity).to(velocity)

    def contract(self):
        return dict(
            gop_contract=self.gop_contract,
            tau_table=list(TAU_TABLE),
            solver_steps=1,
            output_bridge="unit",
            noise_alpha=0.0,
            lora_rank=8,
        )
