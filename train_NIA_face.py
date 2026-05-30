import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np
from tqdm import tqdm
import matplotlib
matplotlib.use('Agg')  # 非交互式后端，专门用于服务器、保存图像
import matplotlib.pyplot as plt
import logging
import yaml
import json
from scipy.stats import pearsonr
from torch.utils.tensorboard import SummaryWriter

# from data_load.load_skin_q import SkinDataLoader
from data_load.load_nia_face import NIADataLoader
from utils.pseudo_supervision import segmentation_pseudo_mask
from models.custom_unet import CustomUnet
from models.exp_model import *
from sklearn.metrics import auc
import torchvision.transforms.functional as TF
from PIL import Image

"""
    这个代码用于训练NIA_face数据集
"""

def setup_logger(log_file):
    logger = logging.getLogger("TrainLogger")
    logger.setLevel(logging.INFO)

    if not os.path.exists(os.path.dirname(log_file)):
        os.makedirs(os.path.dirname(log_file))

    file_handler = logging.FileHandler(log_file)
    file_formatter = logging.Formatter('%(asctime)s - %(message)s')
    file_handler.setFormatter(file_formatter)

    console_handler = logging.StreamHandler()
    console_formatter = logging.Formatter('%(asctime)s - %(message)s')
    console_handler.setFormatter(console_formatter)

    if not logger.handlers:
        logger.addHandler(file_handler)
        logger.addHandler(console_handler)

    return logger

def load_config(config_path):
    with open(config_path, "r") as f:
        return yaml.safe_load(f)

def unnormalize(tensor, mean, std):
    for t, m, s in zip(tensor, mean, std):
        t.mul_(s).add_(m)
    return torch.clamp(tensor, 0, 1)

def plot_mae_threshold_accuracy(y_true, y_pred, save_path='./results/mae_threshold_accuracy_curve.png',
                                 thresholds=np.linspace(0, 30, 100),
                                 labels=None):
    """
    绘制 MAE 阈值-准确率曲线，保存高分辨率图片与原始曲线数据 (CSV)
    - 符合SCI期刊图像标准
    - 数据文件便于复现、再绘制
    """
    import matplotlib.pyplot as plt
    from sklearn.metrics import auc
    import numpy as np
    import pandas as pd
    import os

    assert y_true.shape == y_pred.shape, "Shape mismatch"
    num_dims = y_true.shape[1]
    errors = np.abs(y_true - y_pred)

    if labels is None:
        labels = [f'Dim {i+1}' for i in range(num_dims)]

    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    plt.rcParams["font.family"] = "Arial"  # 无衬线字体

    plt.figure(figsize=(6, 4), dpi=300)

    max_threshold = thresholds[-1]

    # 用于存储所有维度的曲线数据
    all_data = {"Threshold": thresholds}

    for i in range(num_dims):
        accs = np.array([(errors[:, i] <= t).mean() for t in thresholds])
        raw_auc = auc(thresholds, accs)
        norm_auc = raw_auc / max_threshold

        plt.plot(thresholds, accs, label=f'{labels[i]} (AUC={norm_auc:.3f})', linewidth=2.0)
        all_data[labels[i]] = accs

    # 字体控制
    plt.xlabel("Absolute Error Threshold (MAE)", fontsize=12)
    plt.ylabel("Cumulative Accuracy", fontsize=12)
    plt.xticks(fontsize=10)
    plt.yticks(fontsize=10)
    plt.grid(True, linestyle='--', alpha=0.5)

    plt.legend(loc='lower right', fontsize=10, frameon=False)
    plt.tight_layout()

    # 保存图片
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

    # 保存数据，文件名与图片一致，只改后缀
    csv_path = os.path.splitext(save_path)[0] + ".csv"
    df = pd.DataFrame(all_data)
    df.to_csv(csv_path, index=False)

    print(f"[完成] 累积准确率曲线已保存至 {save_path}")
    print(f"[完成] 曲线原始数据已保存至 {csv_path}")

def visualize_segmentation_comparison(image, gmdm, gwdm, pred_mask, save_path):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    image = image.squeeze(0).detach().cpu()
    gmdm = gmdm.squeeze().detach().cpu().numpy()
    gwdm = gwdm.squeeze().detach().cpu().numpy()
    pred = torch.sigmoid(pred_mask).squeeze().detach().cpu().numpy()

    fig, axes = plt.subplots(1, 4, figsize=(16, 4))
    axes[0].imshow(image.permute(1, 2, 0))
    axes[0].set_title("Input Image")
    axes[1].imshow(gmdm, cmap='gray')
    axes[1].set_title("Pore Mask (pMDM)")
    axes[2].imshow(gwdm, cmap='gray')
    axes[2].set_title("Wrinkle Mask (pWDM)")
    axes[3].imshow(pred, cmap='gray')
    axes[3].set_title("Predicted Mask")

    for ax in axes:
        ax.axis('off')
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

def visualize_gaussian_pseudo_seg(image, pseudo_mask, pred_mask, save_path):
    """Visualization when ``seg_supervision: gaussian`` (input | random target | pred)."""
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    image = image.squeeze(0).detach().cpu()
    pseudo = pseudo_mask.squeeze().detach().cpu().numpy()
    pred = torch.sigmoid(pred_mask).squeeze().detach().cpu().numpy()

    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    axes[0].imshow(image.permute(1, 2, 0))
    axes[0].set_title("Input Image")
    axes[1].imshow(pseudo, cmap='gray')
    axes[1].set_title("Gaussian pseudo-target (sigmoid of N(0,1))")
    axes[2].imshow(pred, cmap='gray')
    axes[2].set_title("Predicted mask")

    for ax in axes:
        ax.axis('off')
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

def evaluate(model, val_loader, criterion_regression, device , epoch=None ,result_dir="./results"):
    model.eval()
    running_loss = 0.0
    y_true = []
    y_pred = []

    with torch.no_grad():
        for batch in tqdm(val_loader, desc="Validating"):
            images = batch["image"].to(device)
            targets = batch["targets"].to(device)
            targets = targets[:, 2:]  # 取后两列 [moisture, elasticity]

            _, _, reg_output = model(images)

            loss = criterion_regression(reg_output, targets / 100.0)
            running_loss += loss.item() * images.size(0)

            y_true.append(targets.cpu().numpy())
            y_pred.append((reg_output * 100).cpu().numpy())

    y_true = np.concatenate(y_true, axis=0)
    y_pred = np.concatenate(y_pred, axis=0)

    mae_per_channel = np.mean(np.abs(y_true - y_pred), axis=0)
    rmse_per_channel = np.sqrt(np.mean((y_true - y_pred) ** 2, axis=0))
    r2_per_channel = [r2_score(y_true[:, i], y_pred[:, i]) for i in range(y_true.shape[1])]

    mae = float(np.mean(mae_per_channel))
    rmse = float(np.mean(rmse_per_channel))
    r2 = float(np.mean(r2_per_channel))

    save_path = os.path.join(result_dir,"AUC",
                             f'mae_threshold_accuracy_epoch_{epoch:02d}.png') if epoch is not None else './results/mae_threshold_accuracy_curve.png'
    plot_mae_threshold_accuracy(y_true, y_pred, save_path=save_path,
                                labels=["Moisture", "Elasticity"])

    return float(running_loss / len(val_loader.dataset)), mae, rmse, r2, mae_per_channel, rmse_per_channel, r2_per_channel

def train():
    config_path = './config/config_NIA.yaml'
    config = load_config(config_path)
    result_dir = config.get("result_path", "./results")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # data_loader = SkinDataLoader(config_path, augment=True)
    data_loader = NIADataLoader(config_path,augment=False)

    train_loader = data_loader.train_loader
    val_loader = data_loader.test_loader
    select_model = config.get("model","CustomUnet")

    # Resume 参数
    resume = config.get("resume", False)  # 配置文件中设置 resume: True 开启续训
    checkpoint_path = os.path.join(result_dir,select_model, "checkpoint.pth")
    start_epoch = 1
    best_rmse = float('inf')

    if select_model== "CustomUnet": # use gate attention
        print(f"Use model CustomUnet with 'gate attention'... ")
        model = CustomUnet(
            encoder_name='resnet18',
            num_classes_seg=1,
            num_classes_cls=6,
            reg_class=2, # NIA face just regression 2 type
            cls_head_type='sfp',
            pretrained=True,
            use_segmentation=True,
            use_classification=False,
            use_regression=True,
            use_attention=True, # 如果选择True， Unet的连接中就会增加注意力机制
        ).to(device)
    elif select_model=="CustomUnet_no_attention": # no attention
        print(f"Use model CustomUnet no attention... ")
        model = CustomUnet(
            encoder_name='resnet18',
            num_classes_seg=1,
            num_classes_cls=6,
            reg_class=2,
            cls_head_type='sfp',
            pretrained=True,
            use_segmentation=True,
            use_classification=False,
            use_regression=True,
            use_attention=False, # 如果选择True， Unet的连接中就会增加注意力机制
        ).to(device)
    elif select_model=="CustomUnet_att_ca": # use gate attention and ca
        print(f"Use model CustomUnet with 'ca' , 'gate attention'... ")
        model = CustomUnet(
            encoder_name='resnet18',
            num_classes_seg=1,
            num_classes_cls=6,
            reg_class=2,
            cls_head_type='sfp',
            pretrained=True,
            use_segmentation=True,
            use_classification=False,
            use_regression=True,
            use_attention=True,
            use_ca=True
        ).to(device)
    elif select_model=="ResNet50":
        model = ResNet50Regression(num_outputs=2).to(device)
    elif select_model=="UNet":
        model = UNetRegression(num_outputs=2).to(device)
    elif select_model=="DenseNet121":
        model = DenseNet121Regression(num_outputs=2).to(device)
    elif select_model== "AttUNet":
        model = AttUNetRegression(num_outputs=2).to(device)
    elif select_model== "SwinTransformer":
        model = SwinTransformerRegression(num_outputs=2).to(device)
    elif select_model== "ViTRegression":
        model = ViTRegression(num_outputs=2).to(device)
    elif select_model == "UKAN":
        model = UKANRegression(num_outputs=2, img_size=256).to(device)
    elif select_model == "SwinUNet":
        model = SwinUNetRegression(num_outputs=2, img_size=256).to(device)
    elif select_model == "UTANet":
        model = UTANetRegression(num_outputs=2, img_size=256, use_tamosc=True).to(device)
    elif select_model == "MK_UNet_S":
        model = MKUNetSRegression(num_outputs=2).to(device)
    else:
        raise "Model type must in 'ResNet50, UNet, DenseNet121, AttUNet, SwinTransformer, ViTRegression, CustomUnet*, UKAN, SwinUNet, UTANet, MK_UNet_S'"

    result_dir = os.path.join(result_dir,select_model)
    logger = setup_logger(os.path.join(result_dir, "train_log.txt"))

    criterion_segmentation = nn.BCEWithLogitsLoss()
    criterion_regression = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    tb_writer = SummaryWriter(log_dir=os.path.join(result_dir, "runs"))
    best_rmse = float('inf')
    labels = ["Moisture", "Elasticity"]

    seg_supervision = config.get("seg_supervision", "mdm_wdm")
    use_att_ca_pseudo_modes = select_model == "CustomUnet_att_ca"
    if use_att_ca_pseudo_modes:
        logger.info(
            f"[CustomUnet_att_ca] aux seg supervision: "
            f"use_gmdm={data_loader.use_gmdm}, use_gwdm={data_loader.use_gwdm}, "
            f"seg_supervision={seg_supervision}"
        )
        print(
            f"[CustomUnet_att_ca] aux seg supervision: "
            f"use_gmdm={data_loader.use_gmdm}, use_gwdm={data_loader.use_gwdm}, "
            f"seg_supervision={seg_supervision}"
        )
    else:
        logger.info(
            f"model={select_model}: `seg_supervision` applies only "
            f"when model == CustomUnet_att_ca; pure regressors have no auxiliary seg loss; "
            f"CustomUnet / CustomUnet_no_attention use clamp(gmdm+gwdm), ignoring seg_supervision."
        )
        print(
            f"model={select_model}: pseudo-ablation `seg_supervision` only for CustomUnet_att_ca; "
            f"legacy CustomUnet* uses clamp(sum), regressors omit seg branch."
        )
        if seg_supervision not in ("mdm_wdm", None) and seg_supervision:
            logger.warning(
                "seg_supervision=%r has no effect for model %r.", seg_supervision, select_model
            )

    # load checkpoints
    start_epoch = 1
    if resume and os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint["model_state"], strict=False)
        optimizer.load_state_dict(checkpoint["optimizer_state"])
        start_epoch = checkpoint["epoch"] + 1
        best_rmse = checkpoint.get("best_rmse", float('inf'))
        logger.info(f"Resumed from Epoch {start_epoch}, Best RMSE so far: {best_rmse:.2f}")
        print(f"Resumed from Epoch {start_epoch}, Best RMSE so far: {best_rmse:.2f}")

    for epoch in range(start_epoch, 101):
        model.train()
        running_loss = 0.0

        for batch in tqdm(train_loader, desc=f"Training Epoch {epoch}"):
            images = batch["image"].to(device)
            targets = batch["targets"].to(device)  #targets = [pore_grade, pigm_grade, moisture, elasticity]
            targets = targets[:, 2:]  # 取后两列 [moisture, elasticity]

            gmdm = batch.get("gmdm", None)
            gwdm = batch.get("gwdm", None)

            if gmdm is not None:
                gmdm = gmdm.to(device)
            if gwdm is not None:
                gwdm = gwdm.to(device)

            optimizer.zero_grad()
            seg_output, _, reg_output = model(images)

            total_loss = 0.0
            if seg_output is not None:
                if select_model == "CustomUnet_att_ca":
                    pseudo_mask = segmentation_pseudo_mask(
                        seg_output, gmdm, gwdm, seg_supervision
                    )
                    total_loss += criterion_segmentation(seg_output, pseudo_mask)
                elif select_model in ("CustomUnet", "CustomUnet_no_attention"):
                    pseudo_mask = torch.clamp(gmdm + gwdm, 0, 1)
                    total_loss += criterion_segmentation(seg_output, pseudo_mask)

            if reg_output is not None:
                loss_reg = criterion_regression(reg_output, targets / 100.0)  # need scale
                total_loss += loss_reg

            total_loss.backward()
            optimizer.step()
            running_loss += total_loss.item() * images.size(0)

        train_loss = running_loss / len(train_loader.dataset)

        val_loss, mae, rmse, r2, mae_pc, rmse_pc, r2_pc = evaluate(model, val_loader, criterion_regression, device ,epoch, result_dir=result_dir)

        logger.info(f"Epoch {epoch}: Train Loss={train_loss:.4f}, Val Loss={val_loss:.4f}, MAE={mae:.2f}, RMSE={rmse:.2f}, R2={r2:.2f}")
        for i, label in enumerate(labels):
            logger.info(f"{label} - MAE: {mae_pc[i]:.2f}, RMSE: {rmse_pc[i]:.2f}, R2: {r2_pc[i]:.2f}")

        # 每次都保存最新断点，随时可恢复
        torch.save({
            "epoch": epoch,
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "best_rmse": best_rmse,
        }, os.path.join(result_dir, "checkpoint.pth"))

        if rmse < best_rmse:
            best_rmse = rmse
            torch.save({
                "epoch": epoch,
                "model_state": model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "best_rmse": best_rmse,
            }, os.path.join(result_dir, "best_model.pth"))
            logger.info(f"Best model saved at Epoch {epoch} with RMSE={rmse:.2f}") # add logger

            # add to metrics.json
            metrics = {
                "best_epoch": epoch,
                "train_loss": float(train_loss),
                "val_loss": float(val_loss),
                "mae": float(mae),
                "rmse": float(rmse),
                "r2": float(r2),
                "mae_per_channel": [float(x) for x in mae_pc],
                "rmse_per_channel": [float(x) for x in rmse_pc],
                "r2_per_channel": [float(x) for x in r2_pc],
            }
            metrics_path = os.path.join(result_dir, "metrics.json")
            # 如果已有内容就先读取并追加
            if os.path.exists(metrics_path):
                with open(metrics_path, "r") as f:
                    data = json.load(f)
                if isinstance(data, list):
                    data.append(metrics)
                else:
                    data = [data, metrics]
            else:
                data = [metrics]

            # 重写保存
            with open(metrics_path, "w") as f:
                json.dump(data, f, indent=2)

        tb_writer.add_scalar("Loss/Train", train_loss, epoch)
        tb_writer.add_scalar("Loss/Val", val_loss, epoch)
        tb_writer.add_scalar("Metric/MAE", mae, epoch)
        tb_writer.add_scalar("Metric/RMSE", rmse, epoch)
        tb_writer.add_scalar("Metric/R2", r2, epoch)
        for i, label in enumerate(labels):
            tb_writer.add_scalar(f"MAE/{label}", mae_pc[i], epoch)
            tb_writer.add_scalar(f"RMSE/{label}", rmse_pc[i], epoch)
            tb_writer.add_scalar(f"R2/{label}", r2_pc[i], epoch)

        vis_save_path = os.path.join(result_dir, f"seg_epoch_{epoch:02d}.png")
        # if seg_output is not None:
        #     save_segmentation_overlay(images[0:1], seg_output[0:1], vis_save_path)

        if seg_output is not None and gmdm is not None and gwdm is not None:
            vis_save_path = os.path.join(result_dir, f"mask_compare_epoch_{epoch:02d}.png")
            if (
                select_model == "CustomUnet_att_ca"
                and seg_supervision.strip().lower() == "gaussian"
            ):
                pseudo_v = segmentation_pseudo_mask(
                    seg_output[0:1], gmdm[0:1], gwdm[0:1], seg_supervision
                )
                visualize_gaussian_pseudo_seg(
                    images[0:1], pseudo_v, seg_output[0:1], vis_save_path
                )
            elif select_model in (
                "CustomUnet_att_ca",
                "CustomUnet",
                "CustomUnet_no_attention",
            ):
                visualize_segmentation_comparison(
                    images[0:1],
                    gmdm[0:1],
                    gwdm[0:1],
                    seg_output[0:1],
                    vis_save_path,
                )

    tb_writer.close()

if __name__ == "__main__":
    train()
