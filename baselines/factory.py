"""Architecture settings matching the saved selected configurations."""
import torch.nn as nn
from .models.MLP import MLP
from .models.CNN import CNN
from .models.GRU import GRU
from .models.DLinear import DLinear
from .models.LDNet import LDNetTemporalSOH
from .models.ModernTCN import ModernTCN
from .models.OmniTIEFormer import OmniTIEFormer


class LastValueDelta(nn.Module):
    def __init__(self, core): super().__init__(); self.core = core
    def forward(self, x):
        delta = self.core(x.transpose(1, 2)).unsqueeze(-1)
        return x[:, -1:, -1:].expand(-1, delta.shape[1], -1) + delta


def make_model(method: str, params: dict | None = None):
    p = params or {}
    seq_len, pred_len, channels = 64, 20, 37
    if method == "MLP":
        return MLP(seq_len, pred_len, enc_in=channels,
                        hidden=tuple(p["hidden"]), dropout=float(p["dropout"]))
    if method == "CNN":
        return CNN(seq_len, pred_len, enc_in=channels,
                        channels=tuple(p["channels"]), kernel_size=int(p["kernel_size"]),
                        dropout=float(p["dropout"]))
    if method == "GRU":
        return GRU(seq_len, pred_len, enc_in=channels,
                        hidden=int(p["hidden"]), layers=int(p["layers"]),
                        dropout=float(p["dropout"]))
    if method == "DLinear":
        return DLinear(seq_len, pred_len, enc_in=channels,
                            moving_avg=int(p["moving_avg"]))
    if method == "LDNet":
        core = LDNetTemporalSOH(
            input_channels=channels, base_channels=int(p["base_channels"]),
            num_blocks=3, channel_schedule=list(p["channel_schedule"]),
            reduction=4, groups=4, group_width=4,
            bottleneck_channels=int(p["bottleneck_channels"]), kernel_size=3,
            dilation=2, dropout=float(p["dropout"]), output_size=pred_len)
        # Match the archived tuned comparison: last capacity + predicted delta.
        return LastValueDelta(core)
    if method == "OmniTIEFormer":
        return OmniTIEFormer(
            patch_len=int(p["patch_len"]), seq_len=seq_len, pred_len=pred_len,
            enc_in=channels, d_model=int(p["d_model"]), n_heads=int(p["n_heads"]),
            e_layers=int(p["e_layers"]), dropout=float(p["dropout"]),
            output_channels=1, tcem_fusion_weight=float(p.get("tcem_fusion_weight", 0.5)))
    if method == "ModernTCN":
        return ModernTCN(
            seq_len=64,
            pred_len=20,
            enc_in=37,
            patch_size=int(p.get("patch_size", 8)),
            patch_stride=int(p.get("patch_stride", 4)),
            downsample_ratio=int(p.get("downsample_ratio", 2)),
            ffn_ratio=int(p.get("ffn_ratio", 1)),
            num_blocks=tuple(int(v) for v in p.get("num_blocks", [1, 1, 1])),
            large_kernel=int(p.get("large_kernel", 51)),
            small_kernel=int(p.get("small_kernel", 5)),
            dims=tuple(int(v) for v in p.get("dims", [32, 64, 64])),
            dropout=float(p.get("dropout", 0.30)),
            head_dropout=float(p.get("head_dropout", 0.0)),
        )
    raise ValueError(method)
