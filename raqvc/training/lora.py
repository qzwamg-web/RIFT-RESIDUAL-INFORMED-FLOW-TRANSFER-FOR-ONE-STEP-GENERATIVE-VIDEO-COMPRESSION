"""Rank-8 self-attention LoRA for RIFT fine-tuning."""

import math
import torch
from torch import nn
from torch.nn import functional as F


class Stage0LoRALinear(nn.Module):
    def __init__(self, base: nn.Linear, rank: int = 8, alpha: float = 8.0):
        super().__init__()
        if rank < 1 or not math.isfinite(alpha) or alpha <= 0:
            raise ValueError("rank and alpha must be positive")
        self.base = base.requires_grad_(False)
        self.rank, self.alpha = int(rank), float(alpha)
        self.lora_A = nn.Parameter(torch.empty(rank, base.in_features, device=base.weight.device, dtype=torch.float32))
        self.lora_B = nn.Parameter(torch.zeros(base.out_features, rank, device=base.weight.device, dtype=torch.float32))
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))

    @property
    def in_features(self):
        return self.base.in_features

    @property
    def out_features(self):
        return self.base.out_features

    @property
    def weight(self):
        return self.base.weight

    @property
    def bias(self):
        return self.base.bias

    def forward(self, x):
        result = self.base(x)
        # Float master parameters; autocast may choose BF16 GEMM execution.
        residual = F.linear(F.linear(x.to(self.lora_A.dtype), self.lora_A), self.lora_B)
        return result + residual.to(result.dtype) * (self.alpha / self.rank)


def install_stage0(model):
    for block in model.backbone.dit.blocks:
        attention = getattr(block.self_attn, "_original", block.self_attn)
        for name in ("q", "k", "v", "o"):
            projection = getattr(attention, name)
            if not isinstance(projection, nn.Linear):
                raise TypeError("LoRA already installed or projection is not Linear")
            setattr(attention, name, Stage0LoRALinear(projection, 8, 8.0))
