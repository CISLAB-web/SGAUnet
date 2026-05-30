import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision.models import resnet34
from tqdm import tqdm  # 用于显示进度条
from torchvision import transforms
import csv
import time
import yaml
import random
import json
import numpy as np
# from models.CustomUnetPlusPlus import CustomUnetPlusPlus3
# from models.Resnet import ResNet50Classifier
# from models.CustomUnetPlusPlus import SMPUnetWithClassifier
from models.custom_unet import CustomUnet_pore

from data_load.load_skin_nia_pore import CustomDataset_skin_level
from collections import defaultdict, Counter
from torch.utils.data import WeightedRandomSampler
import os
from torch.nn.functional import softmax
from sklearn.metrics import precision_score, recall_score, f1_score, fbeta_score, confusion_matrix, classification_report, roc_curve, auc
from sklearn.preprocessing import label_binarize
import seaborn as sns
from utils.loss import PoreDiceLoss, CombinedLoss
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
"""
    Skin 等级的训练以及测试，这里使用CustomDataset_skin_level 其中增加了对手动毛孔检测的结果
"""

# 日志记录

def setup_logger(log_file):
    import logging
    import os

    # 如果传入的是文件夹路径，自动追加文件名
    if os.path.isdir(log_file):
        log_file = os.path.join(log_file, "train_log.txt")
    
    logger = logging.getLogger("TrainLogger")
    logger.setLevel(logging.INFO)

    # 创建文件处理器，用于写入日志到文件
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.INFO)
    file_formatter = logging.Formatter('%(asctime)s - %(message)s')
    file_handler.setFormatter(file_formatter)
    
    # 创建控制台处理器，用于在终端打印日志
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_formatter = logging.Formatter('%(asctime)s - %(message)s')
    console_handler.setFormatter(console_formatter)
    
    # 防止多次添加处理器
    if not logger.handlers:
        logger.addHandler(file_handler)
        logger.addHandler(console_handler)

    return logger


# 加载配置文件
def load_config(config_path="./config/config_backbone_val.yml"):
    with open(config_path, "r") as f:
        return yaml.safe_load(f)

def show_class_distribution(dataset):
    """
    打印数据集中每种类型的数量分布
    """
    labels = []
    for idx in range(len(dataset)):
        try:
            _, data = dataset[idx]
            label = data["pore_level"].item()  # 根据你的标签键，提取目标标签
            labels.append(label)
        except Exception as e:
            print(f"Skipping invalid sample at index {idx}: {e}")
            continue

    class_counts = Counter(labels)
    print("Class distribution:")
    for cls, count in class_counts.items():
        print(f"  Class {cls}: {count} samples")
    return class_counts

# 设置随机种子
def set_seed(seed):
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_model(encoder_name, config):
    print(f"Loading SMP U-Net model with encoder: {encoder_name}")
    model = CustomUnet_pore(
        encoder_name=encoder_name,
        num_classes_seg=config['model']['num_classes_seg'],
        num_classes_cls=config['model']['num_classes_cls'],
        cls_head_type = config['model']['cls_head_type'],
        pretrained= config['model']['pretrained'],
        use_segmentation = config['model']['use_segmentation'],
        use_classification = config['model']['use_classification'],
        use_pore_features=True, 
        warmup_epochs=10, #热加载 ,在多少epoch之前增加pore的手动特征到头部
        pore_channels=3
    )
    return model

# 加载数据集
def load_dataset(config):
    transform = {
        "train": transforms.Compose([
            transforms.Resize((1024, 1536)),  # 等比例缩放到接近目标尺寸
            transforms.CenterCrop((512, 512)), # 居中裁剪成正方形
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(10),
            transforms.ColorJitter(
                brightness=0.2,    # 亮度调整范围
                contrast=0.2,      # 对比度调整范围
                saturation=0.2,    # 饱和度调整范围
                hue=0.1            # 色调调整范围 (在[-0.5, 0.5]之间)
        ),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ]),
        "val": transforms.Compose([
            transforms.Resize((1024, 1536)),  # 等比例缩放到接近目标尺寸
            transforms.CenterCrop((512, 512)), # 居中裁剪成正方形
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    }

    dataset_train = CustomDataset_skin_level(
        img_path=config["dataset"]["train_img_path"],
        json_path=config["dataset"]["train_json_path"],
        selected_devices=config["dataset"]["selected_devices"],
        selected_parts=config["dataset"]["selected_parts"],
        transform=transform["train"]
    )
    dataset_val = CustomDataset_skin_level(
        img_path=config["dataset"]["val_img_path"],
        json_path=config["dataset"]["val_json_path"],
        selected_devices=config["dataset"]["selected_devices"],
        selected_parts=config["dataset"]["selected_parts"],
        transform=transform["val"]
    )
    return dataset_train, dataset_val

def get_sampler(dataset):
    # 获取每个样本的类别标签
    labels = [meta["skin_type"] for meta in dataset.data]
    class_counts = Counter(labels)
    class_weights = {cls: 1.0 / count for cls, count in class_counts.items()}
    weights = [class_weights[meta["skin_type"]] for meta in dataset.data]

    sampler = WeightedRandomSampler(weights, num_samples=len(weights), replacement=True)
    return sampler


# def train_one_epoch(model, dataloader, criterion_cls, optimizer, device):
#     model.train()
#     running_loss = 0.0
#     correct = 0
#     total = 0
#     progress_bar = tqdm(dataloader, desc="Training", leave=False)

#     for images, labels in progress_bar:
#         images = images.to(device)
#         cls_labels = labels["skin_type"].to(device)  # 只保留分类标签

#         optimizer.zero_grad()
#         _, classification_output = model(images)  # 只取分类输出
        
#         # 计算分类损失
#         loss_cls = criterion_cls(classification_output, cls_labels)
#         loss_cls.backward()
#         optimizer.step()

#         running_loss += loss_cls.item()
#         _, predicted = classification_output.max(1)
#         total += cls_labels.size(0)
#         correct += predicted.eq(cls_labels).sum().item()

#         progress_bar.set_postfix(loss=loss_cls.item(), acc=100. * correct / total)

#     epoch_loss = running_loss / len(dataloader)
#     epoch_acc = 100. * correct / total
#     return epoch_loss, epoch_acc

def train_one_epoch(model, dataloader, optimizer, criterion, device, epoch=25):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    progress_bar = tqdm(dataloader, desc = "Training", leave = False)

    for images,pore_images, labels in progress_bar:
        images = images.to(device)
        pore_images = pore_images.to(device)
        cls_labels = labels['skin_type'].to(device) # 分类标签
        pore_targets = pore_images[:, :1, :, :]    # 毛孔分割标签( 单通道 )
        # 前向传播（分割 + 分类）
        seg_outputs, cls_outputs = model(images, pore_images, epoch)

        # 计算损失（只关注毛孔分割）
        # seg outputs:torch.Size([30, 3, 512, 512]), pore_targets:torch.Size([30, 1, 512, 512]), cls_outputs:torch.Size([30, 6]), cls_labels:torch.Size([30])
        # print(f"in train: seg outputs:{seg_outputs.shape}, pore_targets:{pore_targets.shape}, cls_outputs:{cls_outputs.shape}, cls_labels:{cls_labels.shape}")
        loss = criterion(seg_outputs, pore_targets, cls_outputs, cls_labels)

        # 反向传播与优化
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        _, predicted = cls_outputs.max(1)
        total += cls_labels.size(0)
        correct += predicted.eq(cls_labels).sum().item()
        progress_bar.set_postfix(loss=loss.item(), acc=100. * correct / total)
        
        running_loss += loss.item() * images.size(0)
    epoch_loss = running_loss / len(dataloader.dataset)
    epoch_acc = 100. * correct /total
    return epoch_loss, epoch_acc


def validate_one_epoch(model, dataloader, criterion_cls, device, n_classes, save_path, config,epoch=None):
    model.eval()
    running_loss = 0.0
    all_preds = []
    all_labels = []
    all_probs = []

    progress_bar = tqdm(dataloader, desc="Validating", leave=False)

    with torch.no_grad():
        for images, _, labels in progress_bar:
            images = images.to(device)
            cls_labels = labels["skin_type"].to(device)

            _, classification_output = model(images)
            probs = softmax(classification_output, dim=1)

            loss_cls = criterion_cls(classification_output, cls_labels)
            running_loss += loss_cls.item()

            _, predicted = classification_output.max(1)

            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(cls_labels.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())

            acc = (predicted == cls_labels).float().mean().item()
            progress_bar.set_postfix(loss=loss_cls.item(), acc=100. * acc)

    epoch_loss = running_loss / len(dataloader)
    epoch_acc = 100. * (np.array(all_preds) == np.array(all_labels)).mean()

    precision = precision_score(all_labels, all_preds, average='macro') * 100
    recall = recall_score(all_labels, all_preds, average='macro') * 100
    f1 = f1_score(all_labels, all_preds, average='macro') * 100
    f2 = fbeta_score(all_labels, all_preds, beta=2, average='macro') * 100

    print("\nClassification Report:\n", classification_report(all_labels, all_preds))
    # 保存ROC曲线
    if config.get("save_plots", {}).get("roc_curve", True):
        plot_multiclass_roc(np.array(all_labels), np.array(all_probs), n_classes, save_path, epoch)

    # 保存混淆矩阵
    if config.get("save_plots", {}).get("confusion_matrix", True):
        cm = confusion_matrix(all_labels, all_preds, labels=np.arange(n_classes))
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=np.arange(n_classes), yticklabels=np.arange(n_classes))
        plt.xlabel("Predicted Label")
        plt.ylabel("True Label")
        plt.title("Confusion Matrix")
        plt.savefig(os.path.join(save_path, f"confusion_matrix_{epoch}.png"))
        plt.close()


    return epoch_loss, epoch_acc, precision, recall, f1 , f2

def plot_multiclass_roc(y_true, y_score, n_classes, save_path, epoch=None):
    y_true_bin = label_binarize(y_true, classes=np.arange(n_classes))
    fpr, tpr, roc_auc = dict(), dict(), dict()

    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(y_true_bin[:, i], y_score[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    plt.figure(figsize=(10, 8))
    for i in range(n_classes):
        plt.plot(fpr[i], tpr[i], label=f'Class {i} ROC (AUC = {roc_auc[i]:.2f})')

    plt.plot([0, 1], [0, 1], 'k--', lw=2)
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Multi-class ROC Curve')
    plt.legend(loc="lower right")
    plt.grid(True)
    plt.savefig(os.path.join(save_path, f"roc_curve_{epoch}.png"))
    plt.close()

def train():
    config = load_config(config_path="./config/config_backbone_val.yml")
    set_seed(config["seed"])
    logger = setup_logger(config["result_path"])

    # 加载数据集
    dataset_train, dataset_val = load_dataset(config)
    train_sampler = get_sampler(dataset_train)
    dataloader_train = DataLoader(dataset_train, 
                                  batch_size=config["train"]["batch_size"],
                                  num_workers=config["train"]["num_workers"], 
                                  sampler=train_sampler)
    dataloader_val = DataLoader(dataset_val, 
                                batch_size=config["train"]["batch_size"], 
                                shuffle=False, 
                                num_workers=config["train"]["num_workers"])

    device = torch.device(f"cuda:{config['device']['gpus'][0]}" if torch.cuda.is_available() else "cpu")
    
    # 遍历主干网络列表
    results = {}
    # 设置保存结果的根目录
    result_root = config["result_path"]
    os.makedirs(result_root, exist_ok=True)
    for encoder_name in config["model"]["encoder_names"]:
        logger.info(f"Testing encoder: {encoder_name}")

        # 为每个主干网络创建单独的文件夹
        encoder_folder = os.path.join(result_root, encoder_name)
        os.makedirs(encoder_folder, exist_ok=True)
        log_path = os.path.join(encoder_folder, "train_log.txt")
        csv_path = os.path.join(encoder_folder, "loss_log.csv")
        json_path = os.path.join(encoder_folder, "metrics.json")
        model_path = os.path.join(encoder_folder, f"best_model_{encoder_name}.pth")

        # 重定向日志到当前主干网络的文件夹
        local_logger = setup_logger(log_path)
        
        # 加载模型
        model = load_model(encoder_name, config)
        if len(config["device"]["gpus"]) > 1:
            model = nn.DataParallel(model, device_ids=config["device"]["gpus"])
        model.to(device)
        
        # 初始化优化器和损失函数
        # 初始化损失函数
        pore_class_idx = 1
        criterion = CombinedLoss(pore_class_idx=pore_class_idx, alpha=1.0, beta=5.0) # 在训练中使用 总损失 = α * 分割损失 + β * 分类损失
        criterion_cls = nn.CrossEntropyLoss() # 在评价中使用
        optimizer = optim.Adam(model.parameters(), lr=config["train"]["learning_rate"], weight_decay=1e-5)

        # # 假设基础学习率为 1e-3
        # base_lr = 1e-3
        # encoder_lr = base_lr * 0.1  # 将encoder的学习率调小
        
        # # 参数分组
        # optimizer = optim.Adam([
        #     {'params': model.unet.encoder.parameters(), 'lr': encoder_lr},  # 编码器（预训练部分）
        #     {'params': model.unet.decoder.parameters(), 'lr': base_lr},     # 解码器（Unet解码部分）
        #     {'params': model.segmentation_head.parameters(), 'lr': base_lr}, # 分割头
        #     {'params': model.classification_head.parameters(), 'lr': base_lr}, # 分类头
        # ], weight_decay=1e-5)
                

        scheduler = None
        if config["train"]["use_lr_scheduler"]:
            scheduler = optim.lr_scheduler.StepLR(optimizer, 
                                                  step_size=config["train"]["lr_scheduler"]["step_size"], 
                                                  gamma=config["train"]["lr_scheduler"]["gamma"])

        # 训练和验证过程
        best_val_acc = 0.0
        metrics = {"train": [], "val": []}
        
        # CSV 文件写入头
        with open(csv_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["Epoch", "Train Loss", "Train Acc", "Val Loss", "Val Acc"])
        
        for epoch in range(config["train"]["num_epochs"]):
            local_logger.info(f"Encoder: {encoder_name} | Epoch {epoch + 1}/{config['train']['num_epochs']}")
            train_loss, train_acc = train_one_epoch(model, dataloader_train, optimizer, criterion, device , epoch)
            
            val_loss, val_acc, test_precision, test_recall, test_f1 , test_f2= validate_one_epoch(model, dataloader_val, criterion_cls, device, config["model"]["num_classes_cls"], encoder_folder, config, epoch)
            print(f"Encoder: {encoder_name} | Epoch {epoch + 1}: Train Loss={train_loss:.4f}, Train Acc={train_acc:.2f}%, Val Loss={val_loss:.4f}, Val Acc={val_acc:.2f}%")

            if scheduler:
                scheduler.step()

            local_logger.info(f"Encoder: {encoder_name} | Epoch {epoch + 1}: Train Loss={train_loss:.4f}, Train Acc={train_acc:.2f}%, Val Loss={val_loss:.4f}, Val Acc={val_acc:.2f}%")

            # 保存日志到 CSV 文件
            with open(csv_path, "a", newline="") as f:
                writer = csv.writer(f)
                writer.writerow([epoch + 1, train_loss, train_acc, val_loss, val_acc])

            if val_acc > best_val_acc:
                best_val_acc = val_acc
                torch.save(model, model_path) # save model
                local_logger.info(f"Best model for {encoder_name} saved with accuracy: {val_acc:.2f}%")

            metrics["train"].append({"epoch": epoch + 1, "train_loss": train_loss, "train_acc": train_acc})
            metrics["val"].append({"epoch": epoch + 1, "val_loss": val_loss, "val_acc": val_acc})

        # 保存 metrics 到 JSON 文件
        with open(json_path, "w") as f:
            json.dump(metrics, f, indent=4)

        results[encoder_name] = {"best_val_acc": best_val_acc}

    # 输出所有主干网络的结果到 XML 文件
    final_xml_path = os.path.join(result_root, "final_results.xml")
    root = ET.Element("Results")
    for encoder, metrics in results.items():
        encoder_elem = ET.SubElement(root, "Encoder", name=encoder)
        ET.SubElement(encoder_elem, "BestValidationAccuracy").text = f"{metrics['best_val_acc']:.2f}"

    tree = ET.ElementTree(root)
    tree.write(final_xml_path, encoding="utf-8", xml_declaration=True)
    logger.info(f"Final results saved to {final_xml_path}")

def val():
    # 加载配置
    config = load_config(config_path="./config/config_backbone_val.yml")
    set_seed(config["seed"])
    logger = setup_logger(config["result_path"])

    # 加载数据集
    _, dataset_val = load_dataset(config)
    dataloader_val = DataLoader(dataset_val,
                                batch_size=config["train"]["batch_size"],
                                shuffle=False,
                                num_workers=config["train"]["num_workers"])

    device = torch.device(f"cuda:{config['device']['gpus'][0]}" if torch.cuda.is_available() else "cpu")

    result_root = config["result_path"]
    os.makedirs(result_root, exist_ok=True)

    for encoder_name in config["model"]["encoder_names"]:
        logger.info(f"Testing encoder: {encoder_name}")

        encoder_folder = os.path.join(result_root, encoder_name)
        model_path = os.path.join(encoder_folder, f"best_model_{encoder_name}.pth")

        if not os.path.exists(model_path):
            logger.error(f"Model file not found for encoder: {encoder_name}. Skipping...")
            continue

        # 加载保存的模型
        model = torch.load(model_path, map_location=device)
        model.eval()
        model.to(device)

        criterion_cls = nn.CrossEntropyLoss()

        logger.info(f"Evaluating model: {model_path}")
        test_loss, test_acc, test_precision, test_recall, test_f1 ,test_f2= validate_one_epoch(model, dataloader_val, criterion_cls, device, config["model"]["num_classes_cls"], encoder_folder, config)

        logger.info(f"Test Results for {encoder_name} - Loss: {test_loss:.4f}, Accuracy: {test_acc:.2f}%, "
                    f"Precision: {test_precision:.2f}%, Recall: {test_recall:.2f}%, F1-Score: {test_f1:.2f}% , F2-Score: {test_f2:.2f}%")

        print(f"Test Results for {encoder_name} - Loss: {test_loss:.4f}, Accuracy: {test_acc:.2f}%, "
              f"Precision: {test_precision:.2f}%, Recall: {test_recall:.2f}%, F1-Score: {test_f1:.2f}% , F2-Score: {test_f2:.2f}%")


if __name__ == "__main__":
    # 从配置文件中读取模式
    config = load_config(config_path="./config/config_backbone_val.yml")
    mode = config.get("mode", "train")  # 默认为 train
    if mode == "train":
        train()
    elif mode == "val":
        val()
    else:
        raise ValueError(f"Unsupported mode: {mode}. Please use 'train' or 'val'.")