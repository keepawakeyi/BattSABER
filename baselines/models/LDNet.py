from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class DeformableDepthwiseConv2d(nn.Module):
    def __init__(self, channels: int, kernel_size: int = 3, padding: int = 1):
        super().__init__()
        self.channels = int(channels)
        self.kernel_size = int(kernel_size)
        self.padding = int(padding)
        self.num_points = self.kernel_size * self.kernel_size
        self.weight = nn.Parameter(torch.empty(self.channels, self.num_points))
        self.bias = nn.Parameter(torch.zeros(self.channels))
        self.offset_conv = nn.Conv2d(self.channels, 2 * self.num_points, kernel_size=3, padding=1)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.kaiming_normal_(self.weight, mode="fan_out", nonlinearity="relu")
        nn.init.constant_(self.offset_conv.weight, 0.0)
        nn.init.constant_(self.offset_conv.bias, 0.0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, channels, height, width = x.shape
        if channels != self.channels:
            raise ValueError(f"Expected {self.channels} channels, got {channels}.")

        padded = F.pad(x, (self.padding, self.padding, self.padding, self.padding), mode="replicate")
        padded_height, padded_width = padded.shape[-2:]
        offsets = self.offset_conv(x).view(batch_size, self.num_points, 2, height, width)

        y_base = torch.arange(height, device=x.device, dtype=x.dtype) + self.padding
        x_base = torch.arange(width, device=x.device, dtype=x.dtype) + self.padding
        yy, xx = torch.meshgrid(y_base, x_base, indexing="ij")

        samples = []
        index = 0
        radius = self.kernel_size // 2
        for dy in range(-radius, radius + 1):
            for dx in range(-radius, radius + 1):
                sample_y = yy + dy + offsets[:, index, 0]
                sample_x = xx + dx + offsets[:, index, 1]
                grid_y = 2.0 * sample_y / max(padded_height - 1, 1) - 1.0
                grid_x = 2.0 * sample_x / max(padded_width - 1, 1) - 1.0
                grid = torch.stack((grid_x, grid_y), dim=-1)
                samples.append(
                    F.grid_sample(
                        padded,
                        grid,
                        mode="bilinear",
                        padding_mode="border",
                        align_corners=True,
                    )
                )
                index += 1

        sampled = torch.stack(samples, dim=2)
        out = (sampled * self.weight.view(1, channels, self.num_points, 1, 1)).sum(dim=2)
        return out + self.bias.view(1, channels, 1, 1)


class ConvBNReLU(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 1,
        groups: int = 1,
        deformable: bool = False,
    ):
        super().__init__()
        padding = kernel_size // 2
        if deformable:
            conv = DeformableDepthwiseConv2d(in_channels, kernel_size=kernel_size, padding=padding)
            self.proj = nn.Conv2d(in_channels, out_channels, kernel_size=1) if in_channels != out_channels else nn.Identity()
        else:
            conv = nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=kernel_size,
                padding=padding,
                groups=groups,
                bias=True,
            )
            self.proj = nn.Identity()
        self.conv = conv
        self.bn = nn.BatchNorm2d(out_channels)
        self.act = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.proj(self.conv(x))
        return self.act(self.bn(x))


class AdaptiveChannelSelector(nn.Module):
    def __init__(self, channels: int, reduction: int = 4):
        super().__init__()
        hidden = max(1, int(channels) // int(reduction))
        self.fc1 = nn.Linear(channels, hidden)
        self.fc2 = nn.Linear(hidden, channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        weights = x.mean(dim=(2, 3))
        weights = F.relu(self.fc1(weights), inplace=True)
        weights = torch.sigmoid(self.fc2(weights)).view(x.shape[0], x.shape[1], 1, 1)
        return x * weights


class MSARFEBlock(nn.Module):
    def __init__(self, channels: int, reduction: int = 4):
        super().__init__()
        self.global_branch = nn.Sequential(
            ConvBNReLU(channels, channels, kernel_size=1),
            ConvBNReLU(channels, channels, kernel_size=3, groups=channels),
            ConvBNReLU(channels, channels, kernel_size=3, deformable=True),
        )
        self.local_branch = nn.Sequential(
            ConvBNReLU(channels, channels, kernel_size=1),
            ConvBNReLU(channels, channels, kernel_size=3, deformable=True),
        )
        self.fuse = nn.Sequential(
            nn.Conv2d(channels * 2, channels, kernel_size=1),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
        )
        self.selector = AdaptiveChannelSelector(channels, reduction=reduction)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        global_features = self.global_branch(x)
        local_features = self.local_branch(x)
        out = self.fuse(torch.cat([global_features, local_features], dim=1))
        return self.selector(out)


class GroupBottleneckPredictor(nn.Module):
    def __init__(
        self,
        channels: int,
        groups: int = 4,
        group_width: int = 4,
        bottleneck_channels: int = 4,
        dropout: float = 0.0,
        flatten_head: bool = True,
    ):
        super().__init__()
        groups = max(1, min(int(groups), int(channels)))
        while channels % groups != 0 and groups > 1:
            groups -= 1

        self.flatten_head = bool(flatten_head)
        self.group_filter = nn.Sequential(
            nn.Conv2d(channels, groups * int(group_width), kernel_size=1, groups=groups),
            nn.BatchNorm2d(groups * int(group_width)),
            nn.ReLU(inplace=True),
            nn.Conv2d(groups * int(group_width), int(bottleneck_channels), kernel_size=1),
            nn.BatchNorm2d(int(bottleneck_channels)),
            nn.ReLU(inplace=True),
        )
        self.dropout = nn.Dropout(float(dropout))
        if self.flatten_head:
            hidden = max(16, int(bottleneck_channels) * 4)
            self.head = nn.Sequential(
                nn.Flatten(),
                nn.LazyLinear(hidden),
                nn.ReLU(inplace=True),
                nn.Dropout(float(dropout)),
                nn.Linear(hidden, 1),
            )
        else:
            self.head = nn.Linear(int(bottleneck_channels), 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.group_filter(x)
        if self.flatten_head:
            return self.head(self.dropout(x))
        x = x.mean(dim=(2, 3))
        return self.head(self.dropout(x))


class TemporalChannelSelector(nn.Module):
    def __init__(self, channels: int, reduction: int = 4):
        super().__init__()
        hidden = max(1, int(channels) // int(reduction))
        self.fc1 = nn.Linear(int(channels), hidden)
        self.fc2 = nn.Linear(hidden, int(channels))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        weights = x.mean(dim=-1)
        weights = F.relu(self.fc1(weights), inplace=True)
        weights = torch.sigmoid(self.fc2(weights)).unsqueeze(-1)
        return x * weights


class TemporalMSARFEBlock(nn.Module):
    def __init__(self, channels: int, reduction: int = 4, kernel_size: int = 3, dilation: int = 2):
        super().__init__()
        padding = kernel_size // 2
        dilated_padding = dilation * (kernel_size // 2)
        self.local_branch = nn.Sequential(
            nn.Conv1d(channels, channels, kernel_size=kernel_size, padding=padding, groups=channels),
            nn.BatchNorm1d(channels),
            nn.ReLU(inplace=True),
            nn.Conv1d(channels, channels, kernel_size=1),
            nn.BatchNorm1d(channels),
            nn.ReLU(inplace=True),
        )
        self.global_branch = nn.Sequential(
            nn.Conv1d(
                channels,
                channels,
                kernel_size=kernel_size,
                padding=dilated_padding,
                dilation=dilation,
                groups=channels,
            ),
            nn.BatchNorm1d(channels),
            nn.ReLU(inplace=True),
            nn.Conv1d(channels, channels, kernel_size=1),
            nn.BatchNorm1d(channels),
            nn.ReLU(inplace=True),
        )
        self.fuse = nn.Sequential(
            nn.Conv1d(channels * 2, channels, kernel_size=1),
            nn.BatchNorm1d(channels),
            nn.ReLU(inplace=True),
        )
        self.selector = TemporalChannelSelector(channels, reduction=reduction)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        local_features = self.local_branch(x)
        global_features = self.global_branch(x)
        return self.selector(self.fuse(torch.cat([local_features, global_features], dim=1)))


class TemporalBottleneckPredictor(nn.Module):
    def __init__(
        self,
        channels: int,
        groups: int = 4,
        group_width: int = 4,
        bottleneck_channels: int = 4,
        dropout: float = 0.0,
        output_size: int = 1,
    ):
        super().__init__()
        groups = max(1, min(int(groups), int(channels)))
        while int(channels) % groups != 0 and groups > 1:
            groups -= 1
        self.group_filter = nn.Sequential(
            nn.Conv1d(int(channels), groups * int(group_width), kernel_size=1, groups=groups),
            nn.BatchNorm1d(groups * int(group_width)),
            nn.ReLU(inplace=True),
            nn.Conv1d(groups * int(group_width), int(bottleneck_channels), kernel_size=1),
            nn.BatchNorm1d(int(bottleneck_channels)),
            nn.ReLU(inplace=True),
        )
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.dropout = nn.Dropout(float(dropout))
        self.head = nn.Linear(int(bottleneck_channels), int(output_size))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.group_filter(x)
        x = self.pool(x).squeeze(-1)
        return self.head(self.dropout(x))


class LDNetTemporalSOH(nn.Module):
    def __init__(
        self,
        input_channels: int = 1,
        base_channels: int = 8,
        num_blocks: int = 3,
        channel_schedule: list[int] | None = None,
        reduction: int = 4,
        groups: int = 4,
        group_width: int = 4,
        bottleneck_channels: int = 4,
        kernel_size: int = 3,
        dilation: int = 2,
        dropout: float = 0.0,
        output_size: int = 1,
    ):
        super().__init__()
        if channel_schedule is None:
            channel_schedule = [base_channels, base_channels * 2, base_channels * 2]
        channel_schedule = [int(c) for c in channel_schedule[: int(num_blocks)]]
        if len(channel_schedule) < int(num_blocks):
            channel_schedule.extend([channel_schedule[-1]] * (int(num_blocks) - len(channel_schedule)))

        self.input_channels = int(input_channels)
        self.stem = nn.Sequential(
            nn.Conv1d(self.input_channels, channel_schedule[0], kernel_size=1),
            nn.BatchNorm1d(channel_schedule[0]),
            nn.ReLU(inplace=True),
        )
        self.transitions = nn.ModuleList()
        self.blocks = nn.ModuleList()
        self.arc_projections = nn.ModuleList()
        block_channels = []

        current_channels = channel_schedule[0]
        for next_channels in channel_schedule:
            if current_channels != next_channels:
                self.transitions.append(
                    nn.Sequential(
                        nn.Conv1d(current_channels, next_channels, kernel_size=1),
                        nn.BatchNorm1d(next_channels),
                        nn.ReLU(inplace=True),
                    )
                )
                current_channels = next_channels
            else:
                self.transitions.append(nn.Identity())
            self.blocks.append(
                TemporalMSARFEBlock(
                    current_channels,
                    reduction=reduction,
                    kernel_size=kernel_size,
                    dilation=dilation,
                )
            )
            block_channels.append(current_channels)

        self.arc_logits = nn.Parameter(torch.zeros(int(num_blocks)))
        self.input_projection = nn.Conv1d(channel_schedule[0], current_channels, kernel_size=1)
        self.arc_projections.extend(
            nn.Identity() if channels == current_channels else nn.Conv1d(channels, current_channels, kernel_size=1)
            for channels in block_channels
        )
        self.predictor = TemporalBottleneckPredictor(
            current_channels,
            groups=groups,
            group_width=group_width,
            bottleneck_channels=bottleneck_channels,
            dropout=dropout,
            output_size=output_size,
        )
        self.apply(self._init_weights)

    def _init_weights(self, module: nn.Module) -> None:
        if isinstance(module, (nn.Conv1d, nn.Linear)):
            nn.init.kaiming_normal_(module.weight, mode="fan_out", nonlinearity="relu")
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 3:
            raise ValueError(f"Expected input shape (batch, channels, length), got {tuple(x.shape)}.")
        if x.shape[1] != self.input_channels:
            raise ValueError(f"Expected {self.input_channels} input channels, got {x.shape[1]}.")

        stem = self.stem(x)
        out = stem
        block_outputs = []
        for transition, block, projection in zip(self.transitions, self.blocks, self.arc_projections):
            out = transition(out)
            out = block(out)
            block_outputs.append(projection(out))

        weights = torch.sigmoid(self.arc_logits)
        adaptive = sum(weight * feat for weight, feat in zip(weights, block_outputs))
        residual = self.input_projection(stem)
        return self.predictor(residual + adaptive)
