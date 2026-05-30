import torch
import torch.nn as nn
import torch.nn.functional as F
from .ca import *

class Bottleneck(nn.Module):
    def __init__(self, in_channels):
        super(Bottleneck, self).__init__()
        self.bottleneck = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.bottleneck(x)

# 1. Basic Decoder block
class DecoderBlock(nn.Module):
    def __init__(self, in_channels, skip_channels, out_channels):
        super(DecoderBlock, self).__init__()
        self.up = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)
        self.conv = nn.Sequential(
            nn.Conv2d(out_channels + skip_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),  # 添加 Batch Normalization
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),  # 添加 Batch Normalization
            nn.ReLU(inplace=True)
        )

    def forward(self, x, skip):
        x = self.up(x)
        x = torch.cat([x, skip], dim=1)
        return self.conv(x)

# 2. 增加注意力机制的decoder block,
class AttentionGate(nn.Module):
    def __init__(self, g_channels, x_channels, inter_channels):
        super(AttentionGate, self).__init__()
        self.W_g = nn.Sequential(
            nn.Conv2d(g_channels, inter_channels, kernel_size=1),
            nn.BatchNorm2d(inter_channels)
        )
        self.W_x = nn.Sequential(
            nn.Conv2d(x_channels, inter_channels, kernel_size=1),
            nn.BatchNorm2d(inter_channels)
        )
        self.psi = nn.Sequential(
            nn.Conv2d(inter_channels, 1, kernel_size=1),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, g, x):
        """
        g: decoder feature (gate)
        x: encoder skip feature
        return: gated skip feature
        """
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        psi = self.relu(g1 + x1)
        alpha = self.psi(psi)
        return x * alpha

class DecoderBlockWithAttention(nn.Module):
    def __init__(self, in_channels, skip_channels, out_channels):
        super(DecoderBlockWithAttention, self).__init__()
        self.up = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)

        # 使用 gating attention 替代 dot-product attention
        self.attn_gate = AttentionGate(g_channels=out_channels, x_channels=skip_channels, inter_channels=out_channels // 2)

        self.conv = nn.Sequential(
            nn.Conv2d(out_channels + skip_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x, skip):
        x = self.up(x)
        skip_attn = self.attn_gate(x, skip)
        x = torch.cat([x, skip_attn], dim=1)
        return self.conv(x)

class DecoderBlockWithAttentionWithCA(nn.Module):
    def __init__(self, in_channels, skip_channels, out_channels):
        super(DecoderBlockWithAttentionWithCA, self).__init__()
        self.up = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)

        # 使用 gating attention 替代 dot-product attention
        self.attn_gate = AttentionGate(g_channels=out_channels, x_channels=skip_channels, inter_channels=out_channels // 2)

        self.conv = nn.Sequential(
            nn.Conv2d(out_channels + skip_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        self.ca = SEAttention(skip_channels, reduction=2)

    def forward(self, x, skip):
        x = self.up(x)
        skip = self.ca(skip)
        skip_attn = self.attn_gate(x, skip)
        x = torch.cat([x, skip_attn], dim=1)
        return self.conv(x)

# 3. Double Decoder block

# 定义基础卷积块，类似于ResNet的BasicBlock
class BasicBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(BasicBlock, self).__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.block(x)

# 改进版 DecoderBlock：加入 BasicBlock
class DecoderBlockDouble(nn.Module):
    def __init__(self, in_channels, skip_channels, out_channels):
        super(DecoderBlockDouble, self).__init__()
        self.up = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)

        # 卷积 + BN + ReLU（拼接跳跃连接后）
        self.conv = nn.Sequential(
            nn.Conv2d(out_channels + skip_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

        # 增加一个 BasicBlock（类似Encoder）
        self.basic_block = BasicBlock(out_channels, out_channels)

    def forward(self, x, skip):
        x = self.up(x)                 # 上采样
        x = torch.cat([x, skip], dim=1)  # 拼接跳跃连接
        x = self.conv(x)              # 卷积融合
        x = self.basic_block(x)       # 额外的卷积块增强特征
        return x