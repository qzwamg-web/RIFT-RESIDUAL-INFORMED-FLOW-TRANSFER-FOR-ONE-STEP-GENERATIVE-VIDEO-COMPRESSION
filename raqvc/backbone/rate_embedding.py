"""Fixed four-QP context embedding used by RIFT."""

import torch
from torch import nn


class RateEmbeddingMLP(nn.Module):
    def __init__(self, hidden_dim=3072):
        super().__init__()
        self.embedding = nn.Embedding(4, hidden_dim)
        self.qp_mlp = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.SiLU(), nn.Linear(hidden_dim, hidden_dim))

    def forward(self, rate=None, delta=None, *, qp_idx):
        embedding = self.embedding(qp_idx.long())
        return self.qp_mlp(embedding).unsqueeze(1)
