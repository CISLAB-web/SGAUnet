"""
UTANet core — adapted from https://github.com/AshleyLuo001/UTANet (AAAI).
Adds ``forward_to_d1`` for regression on decoder features before ``pred``.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from typing import Optional, Tuple

from .ta_mosc import MoE


class UpBlock(nn.Module):
    def __init__(
        self,
        in_ch: int,
        skip_ch: int,
        out_ch: int,
        img_size: int,
        scale_factor: Optional[Tuple[int, int]] = None,
    ):
        super().__init__()
        self.scale_factor = scale_factor or (img_size // 14, img_size // 14)
        self.up = nn.Sequential(
            nn.ConvTranspose2d(in_ch, in_ch // 2, 2, 2),
            nn.BatchNorm2d(in_ch // 2),
            nn.ReLU(inplace=True),
        )
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch // 2 + skip_ch, out_ch, 3, 1, 1, bias=True),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, decoder_feat: torch.Tensor, skip_feat: torch.Tensor) -> torch.Tensor:
        up_feat = self.up(decoder_feat)
        fused_feat = torch.cat([skip_feat, up_feat], dim=1)
        return self.conv(fused_feat)


class UTANet(nn.Module):
    def __init__(
        self,
        pretrained: bool = True,
        topk: int = 2,
        n_channels: int = 3,
        n_classes: int = 1,
        img_size: int = 224,
    ):
        super().__init__()
        self.n_classes = n_classes
        self.pretrained = pretrained
        self.img_size = img_size

        try:
            from torchvision.models import ResNet34_Weights

            self.resnet = models.resnet34(weights=ResNet34_Weights.IMAGENET1K_V1)
        except Exception:
            self.resnet = models.resnet34(pretrained=True)

        self.filters_resnet = [64, 64, 128, 256, 512]
        self.filters_decoder = [32, 64, 128, 256, 512]

        self.conv1 = nn.Sequential(
            nn.Conv2d(n_channels, self.filters_resnet[0], 3, 1, 1, bias=True),
            nn.BatchNorm2d(self.filters_resnet[0]),
            nn.ReLU(inplace=True),
        )
        self.maxpool = nn.MaxPool2d(2, 2)

        self.conv2 = self.resnet.layer1
        self.conv3 = self.resnet.layer2
        self.conv4 = self.resnet.layer3
        self.conv5 = self.resnet.layer4

        if pretrained:
            self.fuse = nn.Sequential(
                nn.Conv2d(512, 64, 1, 1),
                nn.BatchNorm2d(64),
                nn.ReLU(inplace=True),
            )
            self.moe = MoE(num_experts=4, top=topk, emb_size=64)
            self.docker1 = self._create_docker(64, self.filters_resnet[0])
            self.docker2 = self._create_docker(64, self.filters_resnet[1])
            self.docker3 = self._create_docker(64, self.filters_resnet[2])
            self.docker4 = self._create_docker(64, self.filters_resnet[3])

        self.up5 = UpBlock(self.filters_resnet[4], self.filters_resnet[3], self.filters_decoder[3], 28)
        self.up4 = UpBlock(self.filters_decoder[3], self.filters_resnet[2], self.filters_decoder[2], 56)
        self.up3 = UpBlock(self.filters_decoder[2], self.filters_resnet[1], self.filters_decoder[1], 112)
        self.up2 = UpBlock(self.filters_decoder[1], self.filters_resnet[0], self.filters_decoder[0], 224)

        self.pred = nn.Sequential(
            nn.Conv2d(self.filters_decoder[0], self.filters_decoder[0] // 2, 1),
            nn.BatchNorm2d(self.filters_decoder[0] // 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(self.filters_decoder[0] // 2, n_classes, 1),
        )
        self.sigmoid = nn.Sigmoid() if n_classes == 1 else nn.Identity()

    def _create_docker(self, in_ch: int, out_ch: int) -> nn.Sequential:
        return nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 1, 1, bias=True),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward_to_d1(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Encoder + TA-MoSC (if enabled) + decoder up to ``d1`` (before ``pred``)."""
        e1 = self.conv1(x)
        e1_maxp = self.maxpool(e1)
        e2 = self.conv2(e1_maxp)
        e3 = self.conv3(e2)
        e4 = self.conv4(e3)
        e5 = self.conv5(e4)

        aux_loss = torch.tensor(0.0, device=x.device, dtype=x.dtype)

        if self.pretrained:
            e1_resized = F.interpolate(e1, scale_factor=0.5, mode="bilinear", align_corners=False)
            e3_resized = F.interpolate(e3, scale_factor=2, mode="bilinear", align_corners=False)
            e4_resized = F.interpolate(e4, scale_factor=4, mode="bilinear", align_corners=False)

            fused = torch.cat([e1_resized, e2, e3_resized, e4_resized], dim=1)
            fused = self.fuse(fused)

            o1, o2, o3, o4, loss = self.moe(fused)
            aux_loss = loss

            o1 = self.docker1(o1)
            o2 = self.docker2(o2)
            o3 = self.docker3(o3)
            o4 = self.docker4(o4)

            o4 = F.interpolate(o4, scale_factor=0.25, mode="bilinear", align_corners=False)
            o3 = F.interpolate(o3, scale_factor=0.5, mode="bilinear", align_corners=False)
            o1 = F.interpolate(o1, scale_factor=2, mode="bilinear", align_corners=False)
        else:
            o1, o2, o3, o4 = e1, e2, e3, e4

        d4 = self.up5(e5, o4)
        d3 = self.up4(d4, o3)
        d2 = self.up3(d3, o2)
        d1 = self.up2(d2, o1)
        return d1, aux_loss

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        d1, aux_loss = self.forward_to_d1(x)
        logits = self.pred(d1)
        out = self.sigmoid(logits)
        return out, aux_loss
