"""
OmniTIEFormer: A tri-branch transformer with cross-scale transfer learning
for multi-scale battery life-cycle forecasting.

Paper: Applied Energy, 2026
Repo: https://github.com/keepawakeyi/OmniTIEFormer

架构:
  Input → Embedding → 三分支 (Local/Regional/Global) → TCEM → 融合 → Transformer Encoder → Head

Portable plain-PyTorch backbone; forecasting wrapper omitted.
"""

from math import sqrt
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from torch.nn import init

# ==================== 基础类 ====================
class Encoder(nn.Module):
    def __init__(self, attn_layers, conv_layers=None, norm_layer=None):
        super(Encoder, self).__init__()
        self.attn_layers = nn.ModuleList(attn_layers)
        self.conv_layers = nn.ModuleList(conv_layers) if conv_layers is not None else None
        self.norm = norm_layer

    def forward(self, x, attn_mask=None, tau=None, delta=None):
        attns = []
        if self.conv_layers is not None:
            for i, (attn_layer, conv_layer) in enumerate(zip(self.attn_layers, self.conv_layers)):
                delta = delta if i == 0 else None
                x, attn = attn_layer(x, attn_mask=attn_mask, tau=tau, delta=delta)
                x = conv_layer(x)
                attns.append(attn)
            x, attn = self.attn_layers[-1](x, tau=tau, delta=None)
            attns.append(attn)
        else:
            for attn_layer in self.attn_layers:
                x, attn = attn_layer(x, attn_mask=attn_mask, tau=tau, delta=delta)
                attns.append(attn)
        if self.norm is not None:
            x = self.norm(x)
        return x, attns


class EncoderLayer(nn.Module):
    def __init__(self, attention, d_model, d_ff=None, dropout=0.1, activation="relu"):
        super(EncoderLayer, self).__init__()
        d_ff = d_ff or 4 * d_model
        self.attention = attention
        self.conv1 = nn.Conv1d(in_channels=d_model, out_channels=d_ff, kernel_size=1)
        self.conv2 = nn.Conv1d(in_channels=d_ff, out_channels=d_model, kernel_size=1)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.activation = F.relu if activation == "relu" else F.gelu

    def forward(self, x, attn_mask=None, tau=None, delta=None):
        new_x, attn = self.attention(x, x, x, attn_mask=attn_mask, tau=tau, delta=delta)
        x = x + self.dropout(new_x)
        y = x = self.norm1(x)
        y = self.dropout(self.activation(self.conv1(y.transpose(-1, 1))))
        y = self.dropout(self.conv2(y).transpose(-1, 1))
        return self.norm2(x + y), attn


class AttentionLayer(nn.Module):
    def __init__(self, attention, d_model, n_heads, d_keys=None, d_values=None):
        super(AttentionLayer, self).__init__()
        d_keys = d_keys or (d_model // n_heads)
        d_values = d_values or (d_model // n_heads)
        self.inner_attention = attention
        self.query_projection = nn.Linear(d_model, d_keys * n_heads)
        self.key_projection = nn.Linear(d_model, d_keys * n_heads)
        self.value_projection = nn.Linear(d_model, d_values * n_heads)
        self.out_projection = nn.Linear(d_values * n_heads, d_model)
        self.n_heads = n_heads

    def forward(self, queries, keys, values, attn_mask, tau=None, delta=None):
        B, L, _ = queries.shape
        _, S, _ = keys.shape
        H = self.n_heads
        queries = self.query_projection(queries).view(B, L, H, -1)
        keys = self.key_projection(keys).view(B, S, H, -1)
        values = self.value_projection(values).view(B, S, H, -1)
        out, attn = self.inner_attention(queries, keys, values, attn_mask, tau=tau, delta=delta)
        out = out.view(B, L, -1)
        return self.out_projection(out), attn


class FullAttention(nn.Module):
    def __init__(self, mask_flag=True, factor=5, scale=None, attention_dropout=0.1, output_attention=False):
        super(FullAttention, self).__init__()
        self.scale = scale
        self.mask_flag = mask_flag
        self.output_attention = output_attention
        self.dropout = nn.Dropout(attention_dropout)

    def forward(self, queries, keys, values, attn_mask, tau=None, delta=None):
        B, L, H, E = queries.shape
        _, S, _, D = values.shape
        scale = self.scale or 1. / sqrt(E)
        scores = torch.einsum("blhe,bshe->bhls", queries, keys)
        if self.mask_flag:
            if attn_mask is None:
                attn_mask = TriangularCausalMask(B, L, device=queries.device)
            scores.masked_fill_(attn_mask.mask, -np.inf)
        A = self.dropout(torch.softmax(scale * scores, dim=-1))
        V = torch.einsum("bhls,bshd->blhd", A, values)
        if self.output_attention:
            return (V.contiguous(), A)
        else:
            return (V.contiguous(), None)


class TriangularCausalMask:
    def __init__(self, B, L, device="cpu"):
        mask_shape = [B, 1, L, L]
        with torch.no_grad():
            self._mask = torch.triu(torch.ones(mask_shape, dtype=torch.bool), diagonal=1).to(device)

    @property
    def mask(self):
        return self._mask


def initialize_weights(net_l, scale=0.1):
    if not isinstance(net_l, list):
        net_l = [net_l]
    for net in net_l:
        for m in net.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight, a=0, mode='fan_in')
                m.weight.data *= scale
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                init.kaiming_normal_(m.weight, a=0, mode='fan_in')
                m.weight.data *= scale
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias.data, 0.0)


class LayerNormFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, weight, bias, eps):
        ctx.eps = eps
        N, C, H, W = x.size()
        mu = x.mean(1, keepdim=True)
        var = (x - mu).pow(2).mean(1, keepdim=True)
        y = (x - mu) / (var + eps).sqrt()
        ctx.save_for_backward(y, var, weight)
        y = weight.view(1, C, 1, 1) * y + bias.view(1, C, 1, 1)
        return y

    @staticmethod
    def backward(ctx, grad_output):
        eps = ctx.eps
        N, C, H, W = grad_output.size()
        y, var, weight = ctx.saved_variables
        g = grad_output * weight.view(1, C, 1, 1)
        mean_g = g.mean(dim=1, keepdim=True)
        mean_gy = (g * y).mean(dim=1, keepdim=True)
        gx = 1. / torch.sqrt(var + eps) * (g - y * mean_gy - mean_g)
        return gx, (grad_output * y).sum(dim=3).sum(dim=2).sum(dim=0), grad_output.sum(dim=3).sum(dim=2).sum(dim=0), None


class LayerNorm2d(nn.Module):
    def __init__(self, channels, eps=1e-6):
        super(LayerNorm2d, self).__init__()
        self.register_parameter('weight', nn.Parameter(torch.ones(channels)))
        self.register_parameter('bias', nn.Parameter(torch.zeros(channels)))
        self.eps = eps

    def forward(self, x):
        return LayerNormFunction.apply(x, self.weight, self.bias, self.eps)


class ResidualBlockNoBN(nn.Module):
    def __init__(self, nf=64):
        super(ResidualBlockNoBN, self).__init__()
        self.conv1 = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)
        self.conv2 = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)
        initialize_weights([self.conv1, self.conv2], 0.1)

    def forward(self, x):
        identity = x
        out = F.relu(self.conv1(x), inplace=True)
        out = self.conv2(out)
        return identity + out


class TCEM(nn.Module):
    """Tri-branch Cross-Exchange Module"""
    def __init__(self, nf=64):
        super(TCEM, self).__init__()
        self.local_enhance = ResidualBlockNoBN(nf)
        self.regional_enhance = self.local_enhance
        self.local_global_fusion = nn.Conv2d(nf * 2, nf, 1, 1, padding=0)
        self.regional_global_fusion = self.local_global_fusion
        self.scale = nf ** -0.5
        self.norm = LayerNorm2d(nf)
        self.query_projection = nn.Conv2d(nf, nf, 1, 1, padding=0)
        self.global_update_projection = nn.Conv2d(nf * 2, nf, 1, stride=1, padding=0)
        self.local_value_projection = nn.Conv2d(nf, nf, 1, stride=1, padding=0)
        self.regional_value_projection = nn.Conv2d(nf, nf, 1, stride=1, padding=0)
        self.local_gate = nn.Sequential(
            nn.Conv2d(nf, nf, 1, stride=1, padding=0), nn.Sigmoid()
        )
        self.regional_gate = nn.Sequential(
            nn.Conv2d(nf, nf, 1, stride=1, padding=0), nn.Sigmoid()
        )
        initialize_weights([
            self.local_global_fusion, self.regional_global_fusion,
            self.query_projection, self.global_update_projection,
            self.local_value_projection, self.regional_value_projection
        ], 0.1)

    def forward(self, x_local, x_regional, x_global):
        b, c, h, w = x_local.shape
        enhanced_local_base = self.local_enhance(x_local)
        enhanced_regional_base = self.regional_enhance(x_regional)

        local_query = self.query_projection(
            self.norm(self.local_global_fusion(torch.cat([x_global, x_local], dim=1)))
        ).view(b, c, -1)
        regional_query = self.query_projection(
            self.norm(self.regional_global_fusion(torch.cat([x_global, x_regional], dim=1)))
        ).view(b, c, -1)

        local_value = self.local_value_projection(x_local).view(b, c, -1).permute(0, 2, 1)
        regional_value = self.regional_value_projection(x_regional).view(b, c, -1).permute(0, 2, 1)

        local_to_regional_attn = torch.bmm(local_query, regional_value) * self.scale
        regional_to_local_attn = torch.bmm(regional_query, local_value) * self.scale

        local_from_regional = torch.bmm(
            torch.softmax(local_to_regional_attn, dim=-1), regional_value.permute(0, 2, 1)
        ).view(b, c, h, w)
        regional_from_local = torch.bmm(
            torch.softmax(regional_to_local_attn, dim=-1), local_value.permute(0, 2, 1)
        ).view(b, c, h, w)

        local_gate = self.local_gate(local_from_regional + enhanced_local_base)
        enhanced_local = local_gate * local_from_regional + (1 - local_gate) * enhanced_local_base
        regional_gate = self.regional_gate(regional_from_local + enhanced_regional_base)
        enhanced_regional = regional_gate * regional_from_local + (1 - regional_gate) * enhanced_regional_base

        updated_global = self.global_update_projection(
            torch.cat([local_query.view(b, c, h, w), regional_query.view(b, c, h, w)], dim=1)
        ) + x_global

        return enhanced_local, enhanced_regional, updated_global


class SEAttention4D(nn.Module):
    """4D SE-style attention block."""
    def __init__(self, in_channels):
        super().__init__()
        squeeze_channels = max(1, in_channels // 2)
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.conv_squeeze = nn.Conv2d(in_channels, squeeze_channels, kernel_size=(1, 1), bias=False)
        self.conv_excitation = nn.Conv2d(squeeze_channels, in_channels, kernel_size=(1, 1), bias=False)
        self.sigmoid = nn.Sigmoid()
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        z = self.avgpool(x)
        z = self.conv_squeeze(z)
        z = self.conv_excitation(z)
        z = self.sigmoid(z)
        z = self.softmax(z)
        return x * z.expand_as(x)


class InputEmbedding(nn.Module):
    def __init__(self, n_var, dim):
        super().__init__()
        self.n_var = n_var
        self.dim = dim
        self.layers = nn.ModuleList([
            nn.Sequential(nn.Linear(1, dim)) for _ in range(n_var)
        ])

    def forward(self, x):
        y = torch.zeros([x.shape[0], x.shape[1], self.n_var, self.dim], device=x.device)
        for i in range(self.n_var):
            y[:, :, i, :] = self.layers[i](x[:, :, i].unsqueeze(-1))
        return y


class ChannelAttention(nn.Module):
    def __init__(self, channel, reduction=16):
        super().__init__()
        self.maxpool = nn.AdaptiveMaxPool2d(1)
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.se = nn.Sequential(
            nn.Conv2d(channel, channel // reduction, 1, bias=False),
            nn.ReLU(),
            nn.Conv2d(channel // reduction, channel, 1, bias=False)
        )

    def forward(self, x):
        max_result = self.maxpool(x)
        avg_result = self.avgpool(x)
        max_out = self.se(max_result)
        avg_out = self.se(avg_result)
        return max_out + avg_out


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super().__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size=kernel_size, padding=kernel_size // 2)

    def forward(self, x):
        max_result, _ = torch.max(x, dim=1, keepdim=True)
        avg_result = torch.mean(x, dim=1, keepdim=True)
        result = torch.cat([max_result, avg_result], 1)
        return self.conv(result)


class LocalHybridFusionAttention(nn.Module):
    """L-HFA: Local Hybrid Fusion Attention (CBAM-like)"""
    def __init__(self, channel=512, reduction=16, kernel_size=49, HW=None):
        super().__init__()
        self.channel_attention = ChannelAttention(channel=channel, reduction=reduction)
        self.spatial_attention = SpatialAttention(kernel_size=kernel_size)
        self.joint_channel = channel + HW
        self.mlp = nn.Sequential(
            nn.Conv2d(self.joint_channel, channel // reduction, 1, bias=False),
            nn.ReLU(),
            nn.Conv2d(channel // reduction, self.joint_channel, 1, bias=False)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        B, C, H, W = x.size()
        channel_descriptor = self.channel_attention(x).reshape(B, C, 1, 1)
        spatial_descriptor = self.spatial_attention(x).reshape(B, H * W, 1, 1)
        joint_descriptor = torch.cat([channel_descriptor, spatial_descriptor], 1)
        fused_descriptor = self.mlp(joint_descriptor)
        channel_descriptor = fused_descriptor[:, :C, :].reshape(B, C, 1, 1)
        spatial_descriptor = fused_descriptor[:, C:, :].reshape(B, 1, H, W)
        channel_weight = self.sigmoid(channel_descriptor)
        spatial_weight = self.sigmoid(spatial_descriptor)
        return x * channel_weight * spatial_weight


class RegionalChannelAttention(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.attention = SEAttention4D(in_channels)

    def forward(self, x):
        return self.attention(x)


class GlobalChannelAttention(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.attention = SEAttention4D(in_channels)

    def forward(self, x):
        return self.attention(x)


class OmniTIEFormer(nn.Module):
    """
    OmniTIEFormer 核心模型

    输入: [B, L, M]  (batch, seq_len, enc_in)
    输出: [B, pred_len]  (标量预测)

    使用示例:
        model = OmniTIEFormer(patch_len=2, seq_len=16, pred_len=1, enc_in=1)
        x = torch.randn(4, 16, 1)
        y = model(x)  # [4, 1]
    """
    def __init__(
        self,
        patch_len,
        seq_len,
        pred_len,
        enc_in,
        d_model=16,
        factor=3,
        dropout=0.1,
        output_attention=False,
        n_heads=8,
        activation='gelu',
        e_layers=2,
        output_channels=1,
        tcem_fusion_weight=0.0,
        lhfa_reduction=16,
        lhfa_kernel_size=3
    ):
        super(OmniTIEFormer, self).__init__()
        self.patch_nums = seq_len // patch_len
        self.patch_size = patch_len
        # The temporal encoder must attend over the L time steps.  After the
        # three branches are concatenated, each time step has 3*d_model
        # channels; keep [B, L, 3*d_model] instead of transposing it into
        # [B, 3*d_model, L].
        self.encoder_d_model = 3 * d_model
        self.d_ff = 2 * self.encoder_d_model
        if self.encoder_d_model % n_heads != 0:
            raise ValueError("3*d_model must be divisible by n_heads")
        self.tcem_fusion_weight = tcem_fusion_weight
        self.output_channels = int(output_channels)
        if self.output_channels <= 0:
            raise ValueError("output_channels must be positive")
        local_hw = enc_in * d_model

        self.input_embedding = InputEmbedding(enc_in, d_model)

        # Local branch
        self.local_anchor_embeddings = nn.Parameter(
            torch.rand(self.patch_nums, 1, 1, enc_in, 16), requires_grad=True
        )
        self.local_embedding_generators = nn.ModuleList([
            nn.Sequential(nn.Linear(16, d_model)) for _ in range(self.patch_nums)
        ])
        self.lhfa = LocalHybridFusionAttention(
            channel=self.patch_size + 1,
            reduction=max(1, (self.patch_size + 1) // lhfa_reduction),
            kernel_size=min(lhfa_kernel_size, self.patch_size + 1),
            HW=local_hw
        )
        self.local_feature_projection = nn.Linear(self.patch_nums, self.patch_nums * self.patch_size)

        # Regional branch
        self.patch_stride = self.patch_size
        self.rca = RegionalChannelAttention(self.patch_nums)

        # Global branch
        self.global_avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.global_feature_expand = nn.Conv2d(d_model, d_model, 1, 1, 0)
        self.gca = GlobalChannelAttention(seq_len)

        # TCEM
        self.tcem = TCEM(nf=d_model)

        # Tri-branch fusion
        self.multi_scale_fusion = SEAttention4D(3 * seq_len)

        # Temporal encoder
        self.encoder = Encoder([
            EncoderLayer(
                AttentionLayer(
                    FullAttention(False, factor, attention_dropout=dropout,
                                  output_attention=output_attention),
                    self.encoder_d_model, n_heads
                ),
                self.encoder_d_model, self.d_ff,
                dropout=dropout, activation=activation
            ) for _ in range(e_layers)
        ], norm_layer=torch.nn.LayerNorm(self.encoder_d_model))

        self.head = nn.Linear(3 * seq_len * d_model, pred_len * self.output_channels)

    def extract_local_features(self, embedded_x):
        batch_size = embedded_x.shape[0]
        local_features = None
        for i in range(self.patch_nums):
            patch_x = embedded_x[:, i * self.patch_size:(i + 1) * self.patch_size, :, :]
            anchor_embedding = self.local_embedding_generators[i](
                self.local_anchor_embeddings[i]
            ).expand(batch_size, -1, -1, -1)
            patch_x = torch.cat([anchor_embedding, patch_x], dim=1)
            patch_x = self.lhfa(patch_x)
            patch_repr = patch_x[:, 0:1, :, :]
            local_features = patch_repr if local_features is None else torch.cat([local_features, patch_repr], dim=1)

        # [B, patch_nums, M, D] → project back to [B, L, M, D]
        local_features = local_features.permute(0, 3, 2, 1)  # [B, D, M, patch_nums]
        local_features = self.local_feature_projection(local_features)  # [B, D, M, patch_nums*patch_size=L]
        local_features = local_features.permute(0, 3, 2, 1)  # [B, L, M, D]
        local_features += embedded_x
        return local_features

    def extract_regional_features(self, embedded_x):
        x = embedded_x.unfold(dimension=1, size=self.patch_size, step=self.patch_stride)
        x = x.permute(0, 2, 1, 3, 4)
        b, nvar, patch_num, dim, patch_len = x.shape
        x = torch.reshape(x, (b * nvar, patch_num, dim, patch_len))
        regional_features = self.rca(x)
        regional_features = torch.reshape(regional_features, (b, nvar, patch_num, dim, patch_len))
        regional_features = regional_features.permute(0, 2, 4, 1, 3).contiguous()
        regional_features = torch.reshape(regional_features, (b, self.patch_size * self.patch_nums, nvar, dim))
        regional_features += embedded_x
        return regional_features

    def extract_global_features(self, embedded_x):
        x_4d = embedded_x.permute(0, 3, 1, 2)  # [B, D, L, M]
        global_context = self.global_avg_pool(x_4d)  # [B, D, 1, 1]
        global_context = self.global_feature_expand(global_context)
        global_features = global_context.expand(-1, -1, x_4d.shape[2], x_4d.shape[3])
        global_features = global_features.permute(0, 2, 3, 1)  # [B, L, M, D]
        global_features = self.gca(global_features)
        global_features += embedded_x
        return global_features

    def forward(self, x):
        batch_size, L, M = x.shape

        embedded_x = self.input_embedding(x)  # [B, L, M, D]

        local_features = self.extract_local_features(embedded_x)
        regional_features = self.extract_regional_features(embedded_x)
        global_features = self.extract_global_features(embedded_x)

        # TCEM
        local_4d = local_features.permute(0, 3, 1, 2)
        regional_4d = regional_features.permute(0, 3, 1, 2)
        global_4d = global_features.permute(0, 3, 1, 2)

        enhanced_local, enhanced_regional, updated_global = self.tcem(
            local_4d, regional_4d, global_4d
        )

        enhanced_local = enhanced_local.permute(0, 2, 3, 1)
        enhanced_regional = enhanced_regional.permute(0, 2, 3, 1)
        updated_global = updated_global.permute(0, 2, 3, 1)

        final_local = self.tcem_fusion_weight * enhanced_local + (1 - self.tcem_fusion_weight) * local_features
        final_regional = self.tcem_fusion_weight * enhanced_regional + (1 - self.tcem_fusion_weight) * regional_features
        final_global = self.tcem_fusion_weight * updated_global + (1 - self.tcem_fusion_weight) * global_features

        # Fusion
        out = torch.concat([final_local, final_regional, final_global], dim=1)
        out = self.multi_scale_fusion(out) + out
        out = torch.sum(out, dim=-2)

        local_out = out[:, :L, :]
        regional_out = out[:, L:2 * L, :]
        global_out = out[:, 2 * L:, :]
        out = torch.concat([local_out, regional_out, global_out], dim=2)

        # Temporal encoder → Head
        out, _ = self.encoder(out)
        z = rearrange(out, 'b d l -> b (d l)')
        z = self.head(z).view(batch_size, -1, self.output_channels)
        return z  # [B, pred_len, output_channels]
