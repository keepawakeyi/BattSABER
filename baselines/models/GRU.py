"""Two-layer GRU baseline for XJTU 17-to-1 forecasting."""
from __future__ import annotations
import torch
import torch.nn as nn


class GRU(nn.Module):
    def __init__(self, seq_len, pred_len, enc_in=17, hidden=64, layers=2, dropout=0.05):
        super().__init__()
        self.rnn = nn.GRU(int(enc_in), int(hidden), int(layers), batch_first=True,
                          dropout=float(dropout) if int(layers) > 1 else 0.0)
        self.head = nn.Linear(int(hidden), int(pred_len))

    def forward(self, x):
        return self.head(self.rnn(x)[0][:, -1]).unsqueeze(-1)


Model = GRU
__all__ = ["GRU", "Model"]
