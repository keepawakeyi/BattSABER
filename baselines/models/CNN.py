"""Lightweight 1D-CNN baseline for XJTU 17-to-1 forecasting."""
from __future__ import annotations
import torch
import torch.nn as nn


class CNN(nn.Module):
    def __init__(self, seq_len, pred_len, enc_in=17, channels=(32, 64), kernel_size=3, dropout=0.05):
        super().__init__()
        blocks, current = [], int(enc_in)
        for c in channels:
            c = int(c)
            blocks.extend([nn.Conv1d(current, c, kernel_size, padding=kernel_size // 2), nn.BatchNorm1d(c), nn.GELU()])
            current = c
        self.net = nn.Sequential(*blocks)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.dropout = nn.Dropout(float(dropout))
        self.head = nn.Linear(current, int(pred_len))

    def forward(self, x):
        z = self.net(x.transpose(1, 2))
        return self.head(self.dropout(self.pool(z).squeeze(-1))).unsqueeze(-1)


Model = CNN
__all__ = ["CNN", "Model"]
