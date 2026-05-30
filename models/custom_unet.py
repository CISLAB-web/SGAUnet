import torch
import torch.nn as nn
import torch.nn.functional as F
from models.backbone import get_encoder
from models.neck import Bottleneck, DecoderBlock, DecoderBlockWithAttention, DecoderBlockWithAttentionWithCA
from models.heads import (SimpleClassificationHead,
                          SFPClassificationHead,
                          SegmentationHead,
                          FCNSegmentationHead,
                          FCN_CA_SegmentationHead,
                          SFPRegressionHead)
from segmentation_models_pytorch import Unet  # 使用库中的Unet

# class CustomUnet(nn.Module):
#     def __init__(self, encoder_name, num_classes_seg, num_classes_cls, cls_head_type, pretrained):
#         super(CustomUnet, self).__init__()
#         self.encoder = get_encoder(encoder_name, pretrained)
#         encoder_channels = self.encoder.out_channels
#
#         # Bottleneck
#         self.bottleneck = Bottleneck(encoder_channels[-1])
#
#         # Decoder
#         self.decoder_blocks = nn.ModuleList([
#             DecoderBlock(encoder_channels[i], encoder_channels[i - 1], encoder_channels[i - 1])
#             for i in range(len(encoder_channels) - 1, 1, -1)
#         ])
#
#         # Heads
#         self.segmentation_head = SegmentationHead(encoder_channels[1], num_classes_seg)
#         if cls_head_type == "simple":
#             self.classification_head = SimpleClassificationHead(encoder_channels[-1], num_classes_cls)
#         elif cls_head_type == "sfp":
#             self.classification_head = SFPClassificationHead(encoder_channels[1:-1], num_classes_cls)
#         else:
#             raise ValueError("Invalid cls_head_type")
#
#     def forward(self, x):
#         encoder_features = self.encoder(x)
#         x = self.bottleneck(encoder_features[-1])
#         decoder_outputs = []
#
#         for i, block in enumerate(self.decoder_blocks):
#             skip = encoder_features[-(i + 2)]
#             x = block(x, skip)
#             decoder_outputs.append(x)
#
#         seg_output = self.segmentation_head(x)
#         seg_output = F.interpolate(seg_output, scale_factor=2, mode='bilinear', align_corners=False)
#
#         if isinstance(self.classification_head, SFPClassificationHead):
#             cls_output = self.classification_head(decoder_outputs)
#         else:
#             cls_output = self.classification_head(encoder_features[-1])
#
#         return seg_output, cls_output




# class CustomUnet(nn.Module):
#     def __init__(self, encoder_name, num_classes_seg, num_classes_cls, cls_head_type, pretrained):
#         super(CustomUnet, self).__init__()
#         self.encoder = get_encoder(encoder_name, pretrained)
#         encoder_channels = self.encoder.out_channels
#
#         # Bottleneck
#         self.bottleneck = Bottleneck(encoder_channels[-1])
#
#         # Decoder
#         self.decoder_blocks = nn.ModuleList([
#             DecoderBlock(encoder_channels[i], encoder_channels[i - 1], encoder_channels[i - 1])
#             for i in range(len(encoder_channels) - 1, 1, -1)
#         ])
#
#         # 增加一个额外的 DecoderBlock，将通道数映射到输入通道数（例如 3）
#         self.final_decoder_block = nn.Conv2d(
#             in_channels=encoder_channels[1],
#             out_channels=3,  # 输入通道数
#             kernel_size=3,
#             padding=1
#         )
#
#         # Segmentation 和 Classification 头
#         self.segmentation_head = SegmentationHead(3, num_classes_seg)  # 输入通道修正为 3
#         if cls_head_type == "simple":
#             self.classification_head = SimpleClassificationHead(encoder_channels[-1], num_classes_cls)
#         elif cls_head_type == "sfp":
#             self.classification_head = SFPClassificationHead(encoder_channels[0:-1], num_classes_cls, dropout_rate=0.5)
#         else:
#             raise ValueError("Invalid cls_head_type")
#
#     def forward(self, x):
#         encoder_features = self.encoder(x)
#         x = self.bottleneck(encoder_features[-1])
#         decoder_outputs = []
#
#         # Decoder Blocks
#         for i, block in enumerate(self.decoder_blocks):
#             skip = encoder_features[-(i + 2)]
#             x = block(x, skip)
#             decoder_outputs.append(x)
#
#         # 额外的 DecoderBlock 将通道数转换为输入通道数
#         x = self.final_decoder_block(x)
#         decoder_outputs.append(x)
#
#         # 分割头输出
#         seg_output = self.segmentation_head(x)
#         seg_output = F.interpolate(seg_output, scale_factor=2, mode='bilinear', align_corners=False)
#
#         # 分类头输出
#         if isinstance(self.classification_head, SFPClassificationHead):
#             cls_output = self.classification_head(decoder_outputs)
#         else:
#             cls_output = self.classification_head(encoder_features[-1])
#
#         return seg_output, cls_output


# 基础结构 1. 
class CustomUnet(nn.Module):
    def __init__(self, encoder_name, num_classes_seg, num_classes_cls, cls_head_type, pretrained,
                 use_segmentation=True, use_classification=True,
                 use_regression=True, use_attention=False,
                 use_ca = False,reg_class=4):
        """
        Args:
            encoder_name: 主干网络名称
            num_classes_seg: 分割头类别数
            num_classes_cls: 分类头类别数
            cls_head_type: 分类头类型 ("simple" or "sfp")
            pretrained: 是否加载预训练权重
            use_segmentation: 是否使用分割头
            use_classification: 是否使用分类头
        """
        super(CustomUnet, self).__init__()
        self.use_segmentation = use_segmentation
        self.use_classification = use_classification
        self.use_regression = use_regression

        # 编码器
        self.encoder = get_encoder(encoder_name, pretrained)
        encoder_channels = self.encoder.out_channels

        # Bottleneck
        self.bottleneck = Bottleneck(encoder_channels[-1])

        # Decoder Blocks
        if use_attention and not use_ca:
            print(f"Use DecoderBlockWithAttention...")
            self.decoder_blocks = nn.ModuleList([
                DecoderBlockWithAttention(encoder_channels[i], encoder_channels[i - 1], encoder_channels[i - 1])
                for i in range(len(encoder_channels) - 1, 1, -1)
            ])
        elif use_attention and use_ca:
            print(f"Use DecoderBlockWithAttentionWithCA...")
            self.decoder_blocks = nn.ModuleList([
                DecoderBlockWithAttentionWithCA(encoder_channels[i], encoder_channels[i - 1], encoder_channels[i - 1])
                for i in range(len(encoder_channels) - 1, 1, -1)
            ])

        else:
            self.decoder_blocks = nn.ModuleList([
                DecoderBlock(encoder_channels[i], encoder_channels[i - 1], encoder_channels[i - 1])
                for i in range(len(encoder_channels) - 1, 1, -1)
            ])

        # 最终的 DecoderBlock
        self.final_decoder_block = nn.Conv2d(
            in_channels=encoder_channels[1],
            out_channels=3,  # 输入通道数
            kernel_size=3,
            padding=1
        )

        # 分割头
        if self.use_segmentation:
            self.segmentation_head = SegmentationHead(3, num_classes_seg)

        # 分类头
        if self.use_classification:
            if cls_head_type == "simple":
                self.classification_head = SimpleClassificationHead(encoder_channels[-1], num_classes_cls)
            elif cls_head_type == "sfp":
                self.classification_head = SFPClassificationHead(encoder_channels[0:-1], num_classes_cls, dropout_rate=0.5)
            else:
                raise ValueError("Invalid cls_head_type")
        # 回归头
        if self.use_regression:
            self.regression_head = SFPRegressionHead(encoder_channels[0:-1],num_outputs=reg_class,dropout_rate=0.2)

    def forward(self, x):
        encoder_features = self.encoder(x)
        x = self.bottleneck(encoder_features[-1])
        decoder_outputs = []

        # Decoder Blocks
        for i, block in enumerate(self.decoder_blocks):
            skip = encoder_features[-(i + 2)]
            x = block(x, skip)
            decoder_outputs.append(x)

        # Final DecoderBlock
        x = self.final_decoder_block(x)
        decoder_outputs.append(x)

        # Segmentation output
        seg_output = None
        if self.use_segmentation:
            seg_output = self.segmentation_head(x)
            seg_output = F.interpolate(seg_output, scale_factor=2, mode='bilinear', align_corners=False)

        # Classification head
        cls_output = None
        if self.use_classification:
            if isinstance(self.classification_head, SFPClassificationHead):
                cls_output = self.classification_head(decoder_outputs)
            else:
                cls_output = self.classification_head(encoder_features[-1])
        # SFPRegressionHead head
        reg_output = None
        if self.use_regression:
            reg_output = self.regression_head(decoder_outputs) # use SFP
        return seg_output,cls_output,reg_output

# 基础结构 2 
class CustomUnet_pore(nn.Module):
    """
        使用原始的Unet的网络结果， 增加了手动pore特征的融合
    """
    def __init__(self, encoder_name, num_classes_seg, num_classes_cls, cls_head_type, pretrained,
                 use_segmentation=True, use_classification=True, use_pore_features=True, warmup_epochs=10, pore_channels=3):
        super(CustomUnet_pore, self).__init__()
        self.use_segmentation = use_segmentation
        self.use_classification = use_classification
        self.use_pore_features = use_pore_features
        self.warmup_epochs = warmup_epochs
        self.pore_channel = pore_channels

        # 使用库中的Unet
        self.unet = Unet(
            encoder_name=encoder_name,
            encoder_weights='imagenet' if pretrained else None,
            in_channels=3,
            classes=num_classes_seg
        )

        # 分割头（融合毛孔特征）
        if self.use_segmentation:
            self.segmentation_head = FCN_CA_SegmentationHead(num_classes_seg, pore_channels, num_classes_seg, use_ca=True, ca_type='se')
            # self.segmentation_head = FCNSegmentationHead(num_classes_seg, pore_channels, num_classes_seg)

        # 动态获取编码器输出通道
        encoder_channels = self.unet.encoder.out_channels

        # 分类头
        if self.use_classification:
            if cls_head_type == "simple":
                self.classification_head = SimpleClassificationHead(encoder_channels[-1], num_classes_cls)
            elif cls_head_type == "sfp":
                self.classification_head = SFPClassificationHead(encoder_channels[:-1], num_classes_cls, dropout_rate=0.5)
            else:
                raise ValueError("Invalid cls_head_type")

    def forward(self, x, pore_features=None, epoch=0):
        # 直接获取Unet分割结果
        seg_output = self.unet(x)
        # 使用Unet提取特征
        encoder_outputs = self.unet.encoder(x)

        # 分割头（动态融合毛孔特征）
        if self.use_segmentation:
            if self.use_pore_features and epoch < self.warmup_epochs and pore_features is not None:
                seg_output = self.segmentation_head(seg_output, pore_features)
                # print(f"out seg head : {seg_output.shape}") # torch.Size([10, 3, 512, 512])
            else:
                pore_channel = self.pore_channel
                seg_output = self.segmentation_head(seg_output, torch.zeros_like(seg_output[:, :pore_channel, :, :]).to(x.device))
                # print(f"out seg head : {seg_output.shape}")

            seg_output = F.interpolate(seg_output, size=(x.shape[2], x.shape[3]), mode='bilinear', align_corners=False)
            # print(f"out interpolate : {seg_output.shape}") #torch.Size([10, 3, 512, 512])

        # 分类头输出
        cls_output = None
        if self.use_classification:
            encoder_output = encoder_outputs[-1]
            # print(f"encoder_output shape: {encoder_output.shape}") #torch.Size([10, 512, 16, 16])
            if isinstance(self.classification_head, SFPClassificationHead):
                # for i, feature in enumerate(encoder_outputs):
                #     print(f"Layer {i}: {feature.shape}")
                cls_output = self.classification_head(encoder_outputs)
            else:
                cls_output = self.classification_head(encoder_output)
        if self.use_segmentation and self.use_classification:
            return seg_output, cls_output
        elif self.use_segmentation:
            return seg_output
        elif self.use_classification:
            return cls_output
        else:
            return None