# -*- coding: utf-8 -*-
"""
单张图片分割可视化推理脚本, 测试基于NIA的数据集训练的结果
用法示例：
python infer_single_image.py \
  --config ./config/config_NIA.yaml \
  --image ./demo/face.jpg \
  --checkpoint ./results/CustomUnet/checkpoint.pth \
  --save ./results/CustomUnet/vis_single/face_pred.png
"""

import os
import cv2
import yaml
import torch
import argparse
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from torchvision import transforms

from models.custom_unet import CustomUnet
from models.exp_model import *  # 兼容你的其它备选模型

# ========== 通用工具 ==========
def load_config(config_path: str):
    with open(config_path, "r") as f:
        return yaml.safe_load(f)

def build_model(config, device):
    select_model = config.get("model", "CustomUnet")

    if select_model == "CustomUnet":
        print("Use model CustomUnet with 'gate attention'...")
        model = CustomUnet(
            encoder_name='resnet18',
            num_classes_seg=1,
            num_classes_cls=6,
            cls_head_type='sfp',
            pretrained=True,
            use_segmentation=True,
            use_classification=False,
            use_regression=True,
            use_attention=True,
            reg_class=2,
        ).to(device)
    elif select_model == "CustomUnet_no_attention":
        print("Use model CustomUnet no attention...")
        model = CustomUnet(
            encoder_name='resnet18',
            num_classes_seg=1,
            num_classes_cls=6,
            cls_head_type='sfp',
            pretrained=True,
            use_segmentation=True,
            use_classification=False,
            use_regression=True,
            use_attention=False,
            reg_class=2,
        ).to(device)
    elif select_model == "CustomUnet_att_ca":
        print("Use model CustomUnet with 'ca' , 'gate attention'...")
        model = CustomUnet(
            encoder_name='resnet18',
            num_classes_seg=1,
            num_classes_cls=6,
            cls_head_type='sfp',
            pretrained=True,
            use_segmentation=True,
            use_classification=False,
            use_regression=True,
            use_attention=True,
            use_ca=True,
            reg_class=2,

        ).to(device)
    elif select_model == "ResNet50":
        model = ResNet50Regression(num_outputs=2).to(device)
    elif select_model == "UNet":
        model = UNetRegression(num_outputs=2).to(device)
    elif select_model == "DenseNet121":
        model = DenseNet121Regression(num_outputs=2).to(device)
    elif select_model == "AttUNet":
        model = AttUNetRegression(num_outputs=4).to(device)
    elif select_model == "UKAN":
        model = UKANRegression(num_outputs=2, img_size=256).to(device)
    elif select_model == "SwinUNet":
        model = SwinUNetRegression(num_outputs=2, img_size=256).to(device)
    elif select_model == "UTANet":
        model = UTANetRegression(num_outputs=2, img_size=256, use_tamosc=True).to(device)
    elif select_model == "MK_UNet_S":
        model = MKUNetSRegression(num_outputs=2).to(device)
    else:
        raise ValueError("Model type must be in 'ResNet50, UNet, DenseNet121, AttUNet, CustomUnet, UKAN, SwinUNet, UTANet, MK_UNet_S'")

    return model

def unnormalize(tensor, mean, std):
    for t, m, s in zip(tensor, mean, std):
        t.mul_(s).add_(m)
    return torch.clamp(tensor, 0, 1)

def visualize_pred_only(
    image_tensor: torch.Tensor,
    pred_mask: torch.Tensor,
    save_path: str,
    mean=(0.485, 0.456, 0.406),
    std=(0.229, 0.224, 0.225),
    threshold: float = 0.5,
    alpha: float = 0.4
):
    """
    仅预测图可视化：原图 + 半透明预测叠加 + 概率热力
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    img = image_tensor.detach().cpu().clone()
    img = unnormalize(img, mean, std).permute(1, 2, 0).numpy()
    img = np.clip(img, 0, 1)

    prob = torch.sigmoid(pred_mask.detach().cpu()).squeeze().numpy()  # [H, W]
    binm = (prob > threshold).astype(np.uint8)

    # 叠加彩色mask（红色通道高亮）
    color_mask = np.zeros_like(img)
    color_mask[..., 2] = binm  # R 通道
    overlay = (1 - alpha) * img + alpha * color_mask
    overlay = np.clip(overlay, 0, 1)

    # 画三列：原图 / 叠加 / 概率热力
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    axes[0].imshow(img)
    axes[0].set_title("")
    axes[1].imshow(overlay)
    axes[1].set_title("")
    im = axes[2].imshow(prob, cmap='jet')
    axes[2].set_title("")
    for ax in axes:
        ax.axis('off')
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()

    # 同步导出二值mask
    bin_save = os.path.splitext(save_path)[0] + "_bin.png"
    cv2.imwrite(bin_save, (binm * 255).astype(np.uint8))

def build_preprocess(config):
    """
    与训练一致的预处理。若config中含有图像尺寸就用，没有则默认(256,256)。
    """
    # 常见写法：config["data"]["img_size"] 或 config["img_size"]
    size = None
    if isinstance(config.get("data", {}).get("img_size", None), (list, tuple)):
        size = tuple(config["data"]["img_size"])
    elif isinstance(config.get("img_size", None), (list, tuple)):
        size = tuple(config["img_size"])
    if not size:
        size = (256, 256)

    return transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize(size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])

def load_image_as_tensor(image_path: str, preprocess):
    assert os.path.exists(image_path), f"Image not found: {image_path}"
    bgr = cv2.imread(image_path, cv2.IMREAD_COLOR)
    assert bgr is not None, f"Failed to read image: {image_path}"
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    tensor = preprocess(rgb)  # [C,H,W], float
    return tensor

# ========== 主逻辑 ==========
def main():
    parser = argparse.ArgumentParser(description="Single image inference for skin segmentation head visualization.")
    parser.add_argument("--config", type=str, required=True, help="Path to config yaml.")
    parser.add_argument("--image", type=str, required=True, help="Path to input image.")
    parser.add_argument("--checkpoint", type=str, default="", help="Path to model checkpoint (.pth).")
    parser.add_argument("--save", type=str, default="", help="Output visualization path (png).")
    parser.add_argument("--threshold", type=float, default=0.5, help="Sigmoid threshold for binary mask.")
    args = parser.parse_args()

    config = load_config(args.config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 结果与默认checkpoint路径
    result_dir = config.get("result_path", "./results")
    select_model = config.get("model", "CustomUnet_att_ca")
    model_dir = os.path.join(result_dir, select_model)
    ckpt_path = args.checkpoint if args.checkpoint else os.path.join(model_dir, "best_model.pth")
    assert os.path.exists(ckpt_path), f"Checkpoint not found: {ckpt_path}"

    # 输出可视化目标路径
    save_path = args.save if args.save else os.path.join(model_dir, "vis_single", os.path.basename(args.image).rsplit(".", 1)[0] + "_pred.png")
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    # 构建模型并加载权重
    model = build_model(config, device)
    checkpoint = torch.load(ckpt_path, map_location=device)
    if isinstance(checkpoint, dict) and "model_state" in checkpoint:
        model.load_state_dict(checkpoint["model_state"], strict=False)
    else:
        # 兼容直接torch.save(model)的情况
        try:
            model.load_state_dict(checkpoint, strict=False)
        except Exception:
            model = checkpoint  # 兜底
    model.eval()

    # 预处理与推理
    preprocess = build_preprocess(config)
    img_tensor = load_image_as_tensor(args.image, preprocess).unsqueeze(0).to(device)  # [1,C,H,W]

    with torch.no_grad():
        out = model(img_tensor)
        # CustomUnet在训练中返回 (seg_output, cls_out, reg_out)
        """
        NIA : reg_out has two type , ["Moisture", "Elasticity"]
        and just set model is CusttomUnet.. will have seg_output value.
        """
        print(f"out:{out}")
        if isinstance(out, (list, tuple)) and out[0] != None:
            seg_output = out[0]  # [1,1,H,W]
        else:
            raise "No seg output..."
            seg_output = out

    # 可视化（仅预测）
    visualize_pred_only(
        image_tensor=img_tensor[0].cpu(),
        pred_mask=seg_output[0].cpu(),
        save_path=save_path,
        threshold=args.threshold
    )

    print(f"[OK] Saved visualization to: {save_path}")
    print(f"[OK] Also saved binary mask: {os.path.splitext(save_path)[0]}_bin.png")

if __name__ == "__main__":
    main()
