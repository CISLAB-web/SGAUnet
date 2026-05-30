import torch
import torch.nn as nn
import torch.nn.functional as F
from .ca import SEAttention, SKAttention, ECAAttention
"""
    Class Head Design
"""
"""
    分类头 1. Simple
"""
class SimpleClassificationHead(nn.Module):
    def __init__(self, in_channels, num_classes):
        super(SimpleClassificationHead, self).__init__()
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(in_channels, num_classes)
        )

    def forward(self, x):
        return self.head(x)

"""
    分类头 2. SFP
"""
class SFPClassificationHead(nn.Module):
    def __init__(self, decoder_channels, num_classes, dropout_rate=0.5):
        """
        Args:
            decoder_channels: Decoder 输出的通道数列表
            num_classes: 分类任务类别数
            dropout_rate: Dropout 概率
        """
        super(SFPClassificationHead, self).__init__()
        # 融合卷积延后初始化
        self.fusion_conv = None

        # 全连接层
        self.fc = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),
            nn.Linear(256, num_classes)
        )

    def forward(self, decoder_outputs):
        # print(f"SFP head:")
        target_size = decoder_outputs[-1].shape[2:]
        # print(f"target_size: {target_size}")
        
        # 上采样所有Decoder输出
        upsampled_features = [
            F.interpolate(feature, size=target_size, mode='bilinear', align_corners=False)
            for feature in decoder_outputs
        ]
        

        # 融合多尺度特征
        fused_features = torch.cat(upsampled_features, dim=1)  # 拼接通道
        # print(f" fused features shape:{fused_features.shape}")

        # 动态初始化卷积层（防止通道不匹配）
        if self.fusion_conv is None:
            in_channels = fused_features.shape[1]
            # print(f"in channels :{in_channels}")
            self.fusion_conv = nn.Conv2d(in_channels, 512, kernel_size=3, padding=1).to(fused_features.device)

        fused_features = self.fusion_conv(fused_features)
        # print(f"fused_features : {fused_features.shape}")

        return self.fc(fused_features)
"""
    Regression Head Design
"""
class SFPRegressionHead(nn.Module):
    def __init__(self, decoder_channels, num_outputs, dropout_rate=0.5):
        """
        多维回归预测头（用于连续数值预测）

        Args:
            decoder_channels: Decoder 各阶段输出通道组成的列表
            num_outputs: 要预测的连续值个数（如 4 个皮肤属性）
            dropout_rate: Dropout 比例（防止过拟合）
        """
        super(SFPRegressionHead, self).__init__()
        self.fusion_conv = None

        self.regressor = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),         # [B, C, H, W] → [B, C, 1, 1]
            nn.Flatten(),                    # → [B, C]
            nn.Linear(512, 256),             # 根据 fusion_conv 输出通道数确定
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),
            nn.Linear(256, num_outputs)      # 多输出连续值
        )

    def forward(self, decoder_outputs):
        # 对所有Decoder输出上采样到相同尺寸
        target_size = decoder_outputs[-1].shape[2:]
        upsampled_features = [
            F.interpolate(feature, size=target_size, mode='bilinear', align_corners=False)
            for feature in decoder_outputs
        ]

        # 融合多尺度特征（按通道拼接）
        fused_features = torch.cat(upsampled_features, dim=1)

        # 延迟初始化融合卷积（适配拼接后通道数）
        if self.fusion_conv is None:
            in_channels = fused_features.shape[1]
            self.fusion_conv = nn.Conv2d(in_channels, 512, kernel_size=3, padding=1).to(fused_features.device)

        fused_features = self.fusion_conv(fused_features)

        return self.regressor(fused_features)


"""
    Segmentation Head Design
"""

# 1. Segmentation Head 
"""
    baseline simply design
"""
class SegmentationHead(nn.Module):
    def __init__(self, in_channels, num_classes):
        super(SegmentationHead, self).__init__()
        self.conv = nn.Conv2d(in_channels, num_classes, kernel_size=1)

    def forward(self, x):
        x = self.conv(x)
        return torch.sigmoid(x)  # 将输出映射到 [0, 1] 范围

# 2. FCN Segmentation Head 
"""
    Use FCN design
    输入为 融合特征： 手动提取的pore的特征 + encoder的最后一个块的输出特征
"""
class FCNSegmentationHead(nn.Module):
    def __init__(self, in_channels, pore_channels, num_classes):
        super(FCNSegmentationHead, self).__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels + pore_channels, (in_channels + pore_channels) // 2, kernel_size=3, padding=1),
            nn.BatchNorm2d((in_channels + pore_channels) // 2),
            nn.ReLU(inplace=True),
            nn.Conv2d((in_channels + pore_channels) // 2, num_classes, kernel_size=1)
        )

    def forward(self, features, pore_features):
        # print(f"FCN Head")
        pore_features = F.interpolate(pore_features, size=features.shape[2:], mode='bilinear', align_corners=False)
        fused_features = torch.cat((features, pore_features), dim=1)
        # print(f"fused_features shape:{fused_features.shape}")  #torch.Size([10, 6, 512, 512])

        output = self.block(fused_features)
        # print(f"output shape:{output.shape}")  #torch.Size([10, 3, 512, 512])
        return torch.sigmoid(output)
        
# 3. FCN Segmentation Head and CA Attention
"""
    Use FCN design and use CA Attention
    输入为 融合特征： 手动提取的pore的特征 + encoder的最后一个块的输出特征
"""

class FCN_CA_SegmentationHead(nn.Module):
    def __init__(self, in_channels, pore_channels, num_classes, use_ca=False, ca_type='se'):
        super(FCN_CA_SegmentationHead, self).__init__()

        # print(f"use FCN_CA Segmentation head")
        fused_channels = in_channels + pore_channels
        # print(f"fused_channels :{fused_channels}")
        self.use_ca = use_ca
        self.ca_type = ca_type.lower()

        # 动态加载CA模块
        if self.use_ca:
            if self.ca_type == 'se':
                self.ca_module = SEAttention(fused_channels, reduction=2)
            elif self.ca_type == 'sk':
                self.ca_module = SKAttention(fused_channels)
            elif self.ca_type == 'eca':
                self.ca_module = ECAAttention(fused_channels)
            else:
                raise ValueError(f"Unsupported CA type: {self.ca_type} , only can use 'se', 'sk', 'eca' ")
        else:
            self.ca_module = None

        # 卷积块
        self.block = nn.Sequential(
            nn.Conv2d(fused_channels, fused_channels // 2, kernel_size=3, padding=1),
            nn.BatchNorm2d(fused_channels // 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(fused_channels // 2, num_classes, kernel_size=1)
        )

    def forward(self, decoder_outputs, pore_features):
        
        if isinstance(decoder_outputs, list):
            # 如果传入的是decoder的所有的输出特征
            # 多尺度特征融合：上采样到最大尺寸并相加
            target_size = decoder_outputs[-1].shape[2:]
            upsampled_features = [F.interpolate(feat, size=target_size, mode='bilinear', align_corners=False)
                                  for feat in decoder_outputs]
            features = torch.cat(upsampled_features, dim=1)
        else:
            # 单一层输出，直接使用
            features = decoder_outputs
        # 上采样 pore_features
        pore_features = F.interpolate(pore_features, size=features.shape[2:], mode='bilinear', align_corners=False)
        # 拼接特征
        fused_features = torch.cat((features, pore_features), dim=1)

        # 如果启用CA，应用注意力机制
        if self.use_ca and self.ca_module is not None:
            fused_features = self.ca_module(fused_features)

        # 卷积处理
        output = self.block(fused_features)
        # print(f"output shape:{output.shape}") #torch.Size([10, 3, 512, 512])
        return torch.sigmoid(output)