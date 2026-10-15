"""Two-hidden-layer MLP baseline for XJTU 17-to-1 forecasting."""
from __future__ import annotations
import torch
import torch.nn as nn


class MLP(nn.Module):
    def __init__(self, seq_len, pred_len, enc_in=17, hidden=(128, 64), dropout=0.05):
        super().__init__()
        dims = [int(seq_len) * int(enc_in), *map(int, hidden), int(pred_len)]
        layers = []
        for a, b in zip(dims[:-2], dims[1:-1]):
            layers.extend([nn.Linear(a, b), nn.GELU(), nn.Dropout(float(dropout))])
        layers.append(nn.Linear(dims[-2], dims[-1]))
        self.net = nn.Sequential(*layers)
        self.seq_len, self.enc_in = int(seq_len), int(enc_in)

    def forward(self, x):
        if x.shape[1:] != (self.seq_len, self.enc_in):
            raise ValueError(f"Expected [B,{self.seq_len},{self.enc_in}], got {tuple(x.shape)}")
        return self.net(x.reshape(x.shape[0], -1)).unsqueeze(-1)


Model = MLP
__all__ = ["MLP", "Model"]
