"""ModernTCN adapted for the BattSABER forecasting interface.

The backbone follows the authors' official ModernTCN implementation:
patch/stem projection, large- and small-kernel depthwise temporal convolution,
the two grouped pointwise FFNs, and temporal downsampling.  The official
implementation is multivariate-to-multivariate; this adapter keeps all 37
input variables in the backbone and uses a single capacity head for the
project's multivariate-to-univariate SOH task.

Official source:
https://github.com/luodhhh/ModernTCN
Paper: https://openreview.net/forum?id=vpJMJerXHU
"""
from __future__ import annotations

import torch
import torch.nn as nn


class ModernTCNBlock(nn.Module):
    """A ModernTCN block with temporal and cross-variable grouped mixing."""

    def __init__(
        self,
        nvars: int,
        d_model: int,
        ffn_ratio: int = 1,
        large_kernel: int = 51,
        small_kernel: int = 5,
        dropout: float = 0.3,
    ) -> None:
        super().__init__()
        nvars, d_model, ffn_ratio = int(nvars), int(d_model), int(ffn_ratio)
        if nvars < 1 or d_model < 1 or ffn_ratio < 1:
            raise ValueError("nvars, d_model and ffn_ratio must be positive")
        if large_kernel % 2 == 0 or small_kernel % 2 == 0:
            raise ValueError("large_kernel and small_kernel must be odd")

        channels = nvars * d_model
        d_ff = d_model * ffn_ratio
        self.nvars = nvars
        self.d_model = d_model
        self.temporal_large = nn.Conv1d(
            channels, channels, large_kernel, padding=large_kernel // 2, groups=channels, bias=False
        )
        self.temporal_small = nn.Conv1d(
            channels, channels, small_kernel, padding=small_kernel // 2, groups=channels, bias=False
        )
        self.temporal_norm = nn.BatchNorm1d(d_model)

        # Grouped by variable: channel mixing inside each input variable.
        self.variable_ffn = nn.Sequential(
            nn.Conv1d(channels, nvars * d_ff, kernel_size=1, groups=nvars),
            nn.GELU(),
            nn.Dropout(float(dropout)),
            nn.Conv1d(nvars * d_ff, channels, kernel_size=1, groups=nvars),
            nn.Dropout(float(dropout)),
        )
        # Grouped by embedding channel: cross-variable mixing.
        self.cross_variable_ffn = nn.Sequential(
            nn.Conv1d(channels, nvars * d_ff, kernel_size=1, groups=d_model),
            nn.GELU(),
            nn.Dropout(float(dropout)),
            nn.Conv1d(nvars * d_ff, channels, kernel_size=1, groups=d_model),
            nn.Dropout(float(dropout)),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [batch, variables, embedding_dim, temporal_tokens]
        if x.ndim != 4 or x.shape[1] != self.nvars or x.shape[2] != self.d_model:
            raise RuntimeError(
                f"ModernTCNBlock expected [B,{self.nvars},{self.d_model},N], got {tuple(x.shape)}"
            )
        residual = x
        batch, variables, d_model, tokens = x.shape
        z = x.reshape(batch, variables * d_model, tokens)
        z = self.temporal_large(z) + self.temporal_small(z)
        z = z.reshape(batch * variables, d_model, tokens)
        z = self.temporal_norm(z)
        z = z.reshape(batch, variables * d_model, tokens)
        z = self.variable_ffn(z)
        z = z.reshape(batch, variables, d_model, tokens)
        z = z.permute(0, 2, 1, 3).reshape(batch, d_model * variables, tokens)
        z = self.cross_variable_ffn(z)
        z = z.reshape(batch, d_model, variables, tokens).permute(0, 2, 1, 3)
        return residual + z


class ModernTCN(nn.Module):
    """ModernTCN for [B, seq_len, input_channels] -> [B, pred_len, 1].

    ``revin`` is intentionally not part of this adapter.  The benchmark
    protocol already applies a fold-specific train-only StandardScaler to all
    input channels and the target; adding window-wise RevIN would create a
    second normalization policy that is not shared by the comparison models.
    """

    OFFICIAL_REPOSITORY = "https://github.com/luodhhh/ModernTCN"
    OFFICIAL_PAPER = "https://openreview.net/forum?id=vpJMJerXHU"

    def __init__(
        self,
        seq_len: int,
        pred_len: int,
        enc_in: int = 37,
        patch_size: int = 8,
        patch_stride: int = 4,
        downsample_ratio: int = 2,
        ffn_ratio: int = 1,
        num_blocks: tuple[int, ...] = (1, 1, 1),
        large_kernel: int = 51,
        small_kernel: int = 5,
        dims: tuple[int, ...] = (32, 64, 64),
        dropout: float = 0.3,
        head_dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.seq_len = int(seq_len)
        self.pred_len = int(pred_len)
        self.nvars = int(enc_in)
        self.patch_size = int(patch_size)
        self.patch_stride = int(patch_stride)
        self.downsample_ratio = int(downsample_ratio)
        self.ffn_ratio = int(ffn_ratio)
        self.dims = tuple(int(v) for v in dims)
        self.num_blocks = tuple(int(v) for v in num_blocks)
        if len(self.dims) != len(self.num_blocks) or not self.dims:
            raise ValueError("dims and num_blocks must have the same non-zero length")
        if self.patch_size < self.patch_stride or self.patch_stride < 1:
            raise ValueError("patch_size must be >= patch_stride >= 1")
        if self.downsample_ratio < 2:
            raise ValueError("downsample_ratio must be >= 2")

        self.stem = nn.Sequential(
            nn.Conv1d(1, self.dims[0], kernel_size=self.patch_size, stride=self.patch_stride),
            nn.BatchNorm1d(self.dims[0]),
            nn.GELU(),
        )
        self.downsamples = nn.ModuleList(
            nn.Sequential(
                nn.BatchNorm1d(self.dims[i - 1]),
                nn.Conv1d(
                    self.dims[i - 1],
                    self.dims[i],
                    kernel_size=self.downsample_ratio,
                    stride=self.downsample_ratio,
                ),
                nn.GELU(),
            )
            for i in range(1, len(self.dims))
        )
        self.stages = nn.ModuleList(
            nn.Sequential(
                *(
                    ModernTCNBlock(
                        nvars=self.nvars,
                        d_model=self.dims[i],
                        ffn_ratio=self.ffn_ratio,
                        large_kernel=large_kernel,
                        small_kernel=small_kernel,
                        dropout=dropout,
                    )
                    for _ in range(self.num_blocks[i])
                )
            )
            for i in range(len(self.dims))
        )

        stem_input_len = self.seq_len + self.patch_size - self.patch_stride
        tokens = (stem_input_len - self.patch_size) // self.patch_stride + 1
        for _ in range(1, len(self.dims)):
            tokens = (tokens - self.downsample_ratio) // self.downsample_ratio + 1
        if tokens < 1:
            raise ValueError("seq_len is too short for the configured ModernTCN downsampling")
        self.output_tokens = tokens
        self.head = nn.Sequential(
            nn.Flatten(start_dim=1),
            nn.Dropout(float(head_dropout)),
            nn.Linear(self.nvars * self.dims[-1] * self.output_tokens, self.pred_len),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 3 or x.shape[1] != self.seq_len or x.shape[2] != self.nvars:
            raise RuntimeError(
                f"ModernTCN expected [B,{self.seq_len},{self.nvars}], got {tuple(x.shape)}"
            )
        batch = x.shape[0]
        z = x.transpose(1, 2).reshape(batch * self.nvars, 1, self.seq_len)
        pad_len = self.patch_size - self.patch_stride
        if pad_len:
            z = torch.cat([z, z[:, :, -1:].expand(-1, -1, pad_len)], dim=-1)
        z = self.stem(z)
        tokens = z.shape[-1]
        z = z.reshape(batch, self.nvars, self.dims[0], tokens)
        for i, stage in enumerate(self.stages):
            if i:
                z = self.downsamples[i - 1](z.reshape(batch * self.nvars, self.dims[i - 1], tokens))
                tokens = z.shape[-1]
                z = z.reshape(batch, self.nvars, self.dims[i], tokens)
            z = stage(z)
        y = self.head(z.reshape(batch, self.nvars * self.dims[-1], tokens))
        return y.unsqueeze(-1)


Model = ModernTCN

__all__ = ["ModernTCN", "Model", "ModernTCNBlock"]
