"""
Baseline Models for Skin Quality Regression.

This script implements baseline regression models based on:
- ResNet-50: He et al., "Deep Residual Learning for Image Recognition," CVPR 2016.
- U-Net: Ronneberger et al., "U-Net: Convolutional Networks for Biomedical Image Segmentation," MICCAI 2015.
- Attention U-Net: Oktay et al., "Attention U-Net: Learning Where to Look for the Pancreas", arXiv 2018.
- DenseNet Encoder (Huang et al., CVPR 2017): densely connected convolutional features for efficient gradient flow.
- SwinTransformer (Liu et al., CVPR 2022): Swin Transformer V2: Scalling Up Capacity and Resolution, arXiv 2022.
- U-KAN bottleneck regression (Li et al., AAAI 2025): adapted from github.com/CUHK-AIM-Group/U-KAN.
- Swin-UNet encoder regression (Cao et al.): SwinTransformerSys.forward_features from github.com/HuCaoFighting/Swin-Unet.
- UTANet regression (Luo et al., AAAI): decoder ``d1`` before ``pred``, from github.com/AshleyLuo001/UTANet.
- MK-UNet-S regression (Rahman & Marculescu, ICCVW 2025): features before ``out4``, github.com/SLDGroup/MK-UNet.

The models are adapted to predict four facial skin quality metrics: Moisture, Glossiness, Sebum, and Elasticity.
Each model follows a unified output interface compatible with the training pipeline:
    (seg_output=None, cls_output=None, reg_output=[B, 4])

"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import segmentation_models_pytorch as smp
import torch.nn as nn
from torchvision.models import swin_t, Swin_T_Weights
from torchvision.models import vit_b_16, ViT_B_16_Weights

from models.ukan.archs import UKANBackbone
from models.swin_unet.swin_transformer_unet_skip_expand_decoder_sys import SwinTransformerSys
from models.utanet.core import UTANet
from models.mkunet.mkunet_network import MK_UNet_S

class ResNet50Regression(nn.Module):
    """
    Regression model based on ResNet-50.
    This baseline extracts global image features using ResNet-50 (excluding avgpool and FC layers),
    and performs multi-output regression via a custom MLP head.

    Returns:
        A tuple (None, None, reg_output) where reg_output is of shape [B, 4].
    """
    def __init__(self, num_outputs=4):
        super(ResNet50Regression, self).__init__()
        from torchvision.models import resnet50
        base_model = resnet50(pretrained=True)
        self.backbone = nn.Sequential(*list(base_model.children())[:-2])  # 去掉 avgpool 和 fc
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(2048, 512),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(512, num_outputs)
        )

    def forward(self, x):
        x = self.backbone(x)
        x = self.pool(x)
        x = self.head(x)
        return None, None, x

class UNetRegression(nn.Module):
    """
    UNet-style regression model for predicting skin quality metrics.
    It uses a simplified UNet encoder and an adaptive pooling + MLP head
    to generate a 4-dimensional regression output.

    Returns:
        A tuple (None, None, reg_output) where reg_output is of shape [B, 4].
    """
    def __init__(self, encoder_name='resnet34', in_channels=3, num_outputs=4, pretrained=True):
        super(UNetRegression, self).__init__()
        self.unet = smp.Unet(
            encoder_name=encoder_name,
            encoder_weights='imagenet' if pretrained else None,
            in_channels=in_channels,
            classes=1,  # 分割输出层，用不到
        )
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.reg_head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(512, 128),  # 注意：这里的输入通道数视具体encoder而定
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, num_outputs)
        )

    def forward(self, x):
        features = self.unet.encoder(x)[-1]  # 最深层的特征图
        pooled = self.pool(features)
        out = self.reg_head(pooled)
        return None, None, out

class DenseNet121Regression(nn.Module):
    """
    Regression model based on DenseNet-121.
    This model uses DenseNet-121 (excluding classifier) as a feature extractor,
    followed by a regression MLP head to output 4-dimensional prediction.

    Returns:
        A tuple (None, None, reg_output) where reg_output is of shape [B, 4].
    """
    def __init__(self, num_outputs=4):
        super(DenseNet121Regression, self).__init__()
        from torchvision.models import densenet121
        base_model = densenet121(pretrained=True)
        self.backbone = base_model.features  # DenseNet feature extractor
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(1024, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, num_outputs)
        )

    def forward(self, x):
        x = self.backbone(x)
        x = self.pool(x)
        x = self.head(x)
        return None, None, x

class AttUNetRegression(nn.Module):
    def __init__(self, num_outputs=4):
        super().__init__()
        from .util.network import AttU_Net
        self.encoder = AttU_Net(img_ch=3, output_ch=1, return_feat=True)  # 需要你改AttU_Net支持返回特征
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(1024, 64),  # 最终特征通道是1024
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, num_outputs)
        )

    def forward(self, x):
        seg_out, deep_feat = self.encoder(x)  # seg_out: [B,1,H,W]  feat: [B,1024,H',W']
        # print(f"deep feat shape:{deep_feat.shape}")
        pooled = self.avgpool(deep_feat)  # [B,C,1,1]
        # print(f"pooled shape:{pooled.shape}")
        reg_out = self.fc(pooled)
        return None, seg_out, reg_out

class SwinTransformerRegression(nn.Module):
    def __init__(self, num_outputs=4):
        super(SwinTransformerRegression, self).__init__()

        # 加载预训练 Swin-T 模型
        base_model = swin_t(weights=Swin_T_Weights.IMAGENET1K_V1)
        self.backbone = base_model.features  # 输出 [B, H, W, C]

        self.pool = nn.AdaptiveAvgPool2d((1, 1))  # 作用在 [B, C, H, W]
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(768, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, num_outputs)
        )

    def forward(self, x):
        x = self.backbone(x)  # 输出 [B, H, W, C]
        # print("x shape out backbone:", x.shape)
        # 转为 [B, C, H, W]，以便池化操作
        x = x.permute(0, 3, 1, 2)  # Swin 输出格式为 NHWC，需要变成 NCHW
        # print("x after permute:", x.shape)
        x = self.pool(x)  # -> [B, C, 1, 1]
        # print("x after pool:", x.shape)
        x = self.head(x)  # -> [B, num_outputs]
        return None, None, x

class ViTRegression(nn.Module):
    def __init__(self, num_outputs=4):
        super().__init__()
        base = vit_b_16(weights=ViT_B_16_Weights.IMAGENET1K_V1)
        self.embed = base.conv_proj
        self.encoder_layers = base.encoder.layers
        self.encoder_ln = base.encoder.ln
        self.class_token = base.class_token
        self.pos_embed = base.encoder.pos_embedding
        self.dropout = base.encoder.dropout

        self.head = nn.Sequential(
            nn.Linear(768, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, num_outputs)
        )

    def interpolate_pos_embed(self, x, pos_embed):
        cls_pos = pos_embed[:, 0:1, :]
        patch_pos = pos_embed[:, 1:, :]
        B, N, C = x.shape
        H = W = int((N - 1) ** 0.5)
        patch_pos = patch_pos.reshape(1, 14, 14, C).permute(0, 3, 1, 2)
        patch_pos = F.interpolate(patch_pos, size=(H, W), mode='bilinear', align_corners=False)
        patch_pos = patch_pos.permute(0, 2, 3, 1).reshape(1, H * W, C)
        return torch.cat((cls_pos, patch_pos), dim=1)

    def forward(self, x):
        x = self.embed(x)                     # [B, 768, 32, 32]
        x = x.flatten(2).transpose(1, 2)      # [B, 1024, 768]
        B, N, _ = x.shape

        cls_token = self.class_token.expand(B, -1, -1)  # [B, 1, 768]
        x = torch.cat((cls_token, x), dim=1)            # [B, 1025, 768]

        pos = self.interpolate_pos_embed(x, self.pos_embed)
        x = x + pos
        x = self.dropout(x)

        for blk in self.encoder_layers:
            x = blk(x)
        x = self.encoder_ln(x)

        x = x[:, 0]  # CLS token
        x = self.head(x)
        return None, None, x


class UKANRegression(nn.Module):
    """
    Regression via U-KAN encoder + deepest tokenized KAN bottleneck, then GAP and two FC layers.
    Implementation derived from Seg_UKAN (https://github.com/CUHK-AIM-Group/U-KAN).

    Returns:
        (None, None, reg_output) with reg_output shape [B, num_outputs].
    """

    def __init__(self, num_outputs=4, img_size=256, embed_dims=None, hidden_dim=128):
        super().__init__()
        embed_dims = embed_dims if embed_dims is not None else [256, 320, 512]
        self.backbone = UKANBackbone(img_size=img_size, embed_dims=embed_dims)
        bottleneck_ch = embed_dims[2]
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(bottleneck_ch, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, num_outputs),
        )

    def forward(self, x):
        feat = self.backbone(x)
        feat = self.pool(feat)
        reg = self.head(feat)
        return None, None, reg


class SwinUNetRegression(nn.Module):
    """
    Uses the Swin-UNet backbone ``SwinTransformerSys`` encoder+bottleneck only (``forward_features``),
    global average over tokens, then two FC layers — aligned with other regression baselines.

    Default ``window_size=8`` matches ``img_size=256`` with ``patch_size=4`` (patch grid 64→…→8).
    Requires: einops, timm (same as upstream Swin-UNet).

    Reference: https://github.com/HuCaoFighting/Swin-Unet
    """

    def __init__(
        self,
        num_outputs=4,
        img_size=256,
        patch_size=4,
        window_size=8,
        embed_dim=96,
        depths=(2, 2, 2, 2),
        depths_decoder=(2, 2, 2, 1),
        num_heads=(3, 6, 12, 24),
        drop_rate=0.0,
        drop_path_rate=0.2,
        hidden_dim=128,
    ):
        super().__init__()
        self.encoder = SwinTransformerSys(
            img_size=img_size,
            patch_size=patch_size,
            in_chans=3,
            num_classes=1,
            embed_dim=embed_dim,
            depths=list(depths),
            depths_decoder=list(depths_decoder),
            num_heads=list(num_heads),
            window_size=window_size,
            drop_rate=drop_rate,
            attn_drop_rate=0.0,
            drop_path_rate=drop_path_rate,
            ape=False,
            patch_norm=True,
            use_checkpoint=False,
            final_upsample="expand_first",
        )
        bottleneck_dim = int(embed_dim * 2 ** (len(depths) - 1))
        self.head = nn.Sequential(
            nn.Linear(bottleneck_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, num_outputs),
        )

    def forward(self, x):
        tokens, _ = self.encoder.forward_features(x)
        pooled = tokens.mean(dim=1)
        reg = self.head(pooled)
        return None, None, reg


class UTANetRegression(nn.Module):
    """
    Scheme B: full UTANet encoder + TA-MoSC + decoder until ``d1`` (before segmentation ``pred``),
    then spatial GAP and two FC layers. Matches other regression baselines; MoE aux loss is not added to train_SKIN_Q loss.

    Reference: https://github.com/AshleyLuo001/UTANet
    """

    def __init__(
        self,
        num_outputs=4,
        img_size=256,
        use_tamosc: bool = True,
        topk: int = 2,
        hidden_dim: int = 128,
    ):
        super().__init__()
        self.backbone = UTANet(
            pretrained=use_tamosc,
            topk=topk,
            n_channels=3,
            n_classes=1,
            img_size=img_size,
        )
        d1_channels = self.backbone.filters_decoder[0]
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(d1_channels, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, num_outputs),
        )

    def forward(self, x):
        d1, _aux = self.backbone.forward_to_d1(x)
        reg = self.head(self.pool(d1))
        return None, None, reg


class MKUNetSRegression(nn.Module):
    """
    MK_UNet_S full decoder features immediately before ``out4``, then GAP + two FC layers.

    Reference: https://github.com/SLDGroup/MK-UNet (ICCVW 2025)
    """

    def __init__(self, num_outputs=4, hidden_dim: int = 128):
        super().__init__()
        self.backbone = MK_UNet_S(num_classes=1, in_channels=3)
        c = self.backbone.reg_feat_channels
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(c, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, num_outputs),
        )

    def forward(self, x):
        feat = self.backbone.forward_to_pre_out4(x)
        reg = self.head(self.pool(feat))
        return None, None, reg

# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# 模拟一张 512x512 的 RGB 图像，batch size = 2
# dummy_input = torch.randn(2, 3, 512, 512).to(device)
# # === 测试 ResNet50Regression ===
# model_resnet = ResNet50Regression(num_outputs=4).to(device)
# model_resnet.eval()
# with torch.no_grad():
#     seg, cls, reg = model_resnet(dummy_input)
# print("ResNet50Regression output shape:", reg.shape)
# # === 测试 UNetRegression ===
# model_unet = UNetRegression().to(device)
# model_unet.eval()
# with torch.no_grad():
#     seg, cls, reg = model_unet(dummy_input)
# print("UNetRegression output shape:", reg.shape)
#
# # === 测试 DenseNet121 ===
# model_densenet121 = DenseNet121Regression(num_outputs=4).to(device)
# model_densenet121.eval()
# with torch.no_grad():
#     seg, cls, reg = model_unet(dummy_input)
# print("model_densenet121 output shape:", reg.shape)
#
#
# === 测试 AttUNetRegression ===
# model_attUnet = AttUNetRegression(num_outputs=4).to(device)
# model_attUnet.eval()
# with torch.no_grad():
#     seg, cls, reg = model_attUnet(dummy_input)
# print("model_attUnet output shape:", reg.shape)


# === 测试 SwinTransformerRegression ===
# dummy_input = torch.randn(2, 3, 512, 512)
# model_attUnet = SwinTransformerRegression(num_outputs=4)
# model_attUnet.eval()
# with torch.no_grad():
#     seg, cls, reg = model_attUnet(dummy_input)
# print("model_attUnet output shape:", reg.shape)

# === 测试 UKANRegression ===
# dummy_input = torch.randn(2, 3, 512, 512)
# model_ukan = UKANRegression(num_outputs=4)
# model_ukan.eval()
# with torch.no_grad():
#     seg, cls, reg = model_ukan(dummy_input)
# print("model_ukan output shape:", reg.shape)