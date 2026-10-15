"""Multivariate-to-univariate DLinear baseline for XJTU."""
from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F


class DLinear(nn.Module):
    def __init__(self, seq_len, pred_len, enc_in=17, moving_avg=5):
        super().__init__()
        self.seq_len = int(seq_len)
        self.moving_avg = int(moving_avg)
        input_dim = self.seq_len * int(enc_in)
        self.linear_trend = nn.Linear(input_dim, int(pred_len))
        self.linear_seasonal = nn.Linear(input_dim, int(pred_len))

    def forward(self, x):
        pad = self.moving_avg // 2
        xt = F.avg_pool1d(
            F.pad(x.transpose(1, 2), (pad, pad), mode="replicate"),
            self.moving_avg, stride=1,
        ).transpose(1, 2)
        trend = self.linear_trend(xt.reshape(x.shape[0], -1))
        seasonal = self.linear_seasonal((x - xt).reshape(x.shape[0], -1))
        return (trend + seasonal).unsqueeze(-1)


Model = DLinear
__all__ = ["DLinear", "Model"]
