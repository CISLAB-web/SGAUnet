import os
import json
import yaml
from PIL import Image
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import matplotlib.pyplot as plt
import math
from collections import defaultdict
import random
import torchvision.transforms.functional as TF

def show_image_with_masks(rgb, gmdm=None, gwdm=None, title=""):
    """
    显示原图和两个结构标签（gmdm, gwdm），每张图片一行三列。
    参数:
        rgb (Tensor): RGB 图像张量 [3, H, W]
        gmdm (Tensor): GMDM 掩码图像张量 [1, H, W] 或 [3, H, W]
        gwdm (Tensor): GWDM 掩码图像张量 [1, H, W] 或 [3, H, W]
    """
    rgb_img = TF.to_pil_image(rgb.cpu())
    gmdm_img = TF.to_pil_image(gmdm.cpu()) if gmdm is not None else None
    gwdm_img = TF.to_pil_image(gwdm.cpu()) if gwdm is not None else None

    plt.figure(figsize=(12, 4))
    plt.suptitle(title, fontsize=14)

    # RGB 原图
    plt.subplot(1, 3, 1)
    plt.imshow(rgb_img)
    plt.title("RGB Image")
    plt.axis('off')

    # GMDM
    plt.subplot(1, 3, 2)
    if gmdm_img:
        plt.imshow(gmdm_img, cmap='gray')
    else:
        plt.text(0.5, 0.5, "No GMDM", ha='center', va='center')
    plt.title("GMDM Mask")
    plt.axis('off')

    # GWDM
    plt.subplot(1, 3, 3)
    if gwdm_img:
        plt.imshow(gwdm_img, cmap='gray')
    else:
        plt.text(0.5, 0.5, "No GWDM", ha='center', va='center')
    plt.title("GWDM Mask")
    plt.axis('off')

    plt.tight_layout()
    plt.show()

class SkinDataset(Dataset):
    def __init__(self, json_folder, image_folder, ids=None, light_conditions=None, transform=None ,use_gmdm=False, use_gwdm=False):
        self.json_folder = json_folder
        self.image_folder = image_folder
        self.ids = ids if ids else []  # 如果没有传入 IDs，默认为空列表
        # self.light_conditions = light_conditions
        self.light_conditions = light_conditions if light_conditions else []  # 如果没有传入光照条件，默认为空列表
        self.transform = transform

        self.use_gmdm = use_gmdm
        self.use_gwdm = use_gwdm

        # 加载伪标签路径
        self.gmdm_folder = None
        self.gwdm_folder = None

        self.mask_transform = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.ToTensor(),  # 转为 [3, H, W] 且归一化为 [0, 1]
            transforms.Lambda(lambda x: x[0:1, :, :])  # 提取第1通道 (R)，变为 [1, H, W]
        ])

        if use_gmdm:
            parent_folder = os.path.dirname(image_folder)
            self.gmdm_folder = os.path.join(parent_folder,"gmdm")
        if use_gwdm:
            parent_folder = os.path.dirname(image_folder)
            self.gwdm_folder = os.path.join(parent_folder,'gwdm')

        self.errors = []
        self.id_counts = defaultdict(int)
        self.light_condition_counts = defaultdict(int)
        self.data = self.load_data()
        self.count_ids_and_conditions()



    def load_data(self):
        data = []
        # os.listdir 顺序在 Linux 上未定义，会导致同一份数据每次构建 Dataset 顺序不同；
        # 排序后顺序稳定，便于可视化/对比时按 group_id 或文件名对齐。
        for filename in sorted(os.listdir(self.json_folder)):
            if filename.endswith('.json'):
                json_file_path = os.path.join(self.json_folder, filename)
                with open(json_file_path, 'r') as file:
                    json_data = json.load(file)
                    # # 获取文件名
                    # image_file_name = os.path.basename(json_data["imagePath"])
                    # image_path = os.path.join(self.image_folder, image_file_name)
                    # image_path = os.path.abspath(image_path.replace('\\', '/'))  # 修正路径分隔符
                    
                    # 这里的linux 版本和window有所不同
                    image_path = os.path.join(self.image_folder, json_data['imagePath'].split('\\')[-1])
                    image_path = os.path.abspath(image_path.replace('\\', '/'))  # 修正路径分隔

                    if self.use_gmdm:
                        gmdm_path = os.path.join(self.gmdm_folder,json_data['imagePath'].split('\\')[-1])
                        gmdm_path = os.path.abspath(gmdm_path.replace('\\', '/'))  # 修正路径分隔
                    else:
                        gmdm_path = None
                    if self.use_gwdm:
                        gwdm_path = os.path.join(self.gwdm_folder,json_data['imagePath'].split('\\')[-1])
                        gwdm_path = os.path.abspath(gwdm_path.replace('\\', '/'))  # 修正路径分隔
                    else:
                        gwdm_path = None
                    
                    for shape in json_data["shapes"]:
                        # if shape["group_id"] in self.ids:
                        # 如果 self.ids 为空，则不过滤 group_id；否则过滤符合条件的 group_id
                        if not self.ids or shape["group_id"] in self.ids:
                            try:
                                # 解析description获取光照值
                                description = shape["description"].split(',')
                                if len(description) != 3:
                                    self.errors.append(f"{json_file_path} description != 3: {shape['description']}")
                                    continue
                                light_condition = int(description[2])
                            except ValueError:
                                self.errors.append(f"{json_file_path} description value error: {shape['description']}")
                                continue

                            # 根据光照值进行分类
                            # if light_condition in self.light_conditions:
                            # 如果 self.light_conditions 为空，则不过滤 light_condition；否则过滤符合条件的 light_condition
                            if not self.light_conditions or light_condition in self.light_conditions:
                                data.append({
                                    "json_path": json_file_path,
                                    "image_path": image_path,
                                    'gwdm_path': gwdm_path,
                                    'gmdm_path': gmdm_path,
                                    "group_id": shape["group_id"],
                                    "points": shape["points"],
                                    "label": shape["label"],
                                    "description": shape["description"]
                                })
        data.sort(key=lambda x: (x["group_id"], x["image_path"], x["json_path"], x["description"]))
        return data

    def count_ids_and_conditions(self):
        for item in self.data:
            self.id_counts[item["group_id"]] += 1
            description = item["description"].split(',')
            if len(description) == 3:
                light_condition = int(description[2])
                self.light_condition_counts[light_condition] += 1

    def get_statistics(self):
        print("ID counts:")
        for group_id, count in self.id_counts.items():
            print(f"ID: {group_id}, Count: {count}")

        print("\nLight condition counts:")
        for light_condition, count in self.light_condition_counts.items():
            print(f"Light Condition: {light_condition}, Count: {count}")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        image = Image.open(item["image_path"]).convert('RGB')
        if self.use_gmdm:
            gmdm = Image.open(item["gmdm_path"]).convert('RGB')
        else:
            gmdm = None
        if self.use_gwdm:
            gwdm = Image.open(item["gwdm_path"]).convert('RGB')
        else:
            gwdm = None

        # ========== 图像和结构标签统一旋转 ==========
        image = np.array(image)
        # image = np.rot90(image, k=-1)  # 逆时针旋转90度
        image = Image.fromarray(image)

        if gmdm is not None:
            gmdm = np.array(gmdm)
            # gmdm = np.rot90(gmdm, k=-1)
            gmdm = Image.fromarray(gmdm)

        if gwdm is not None:
            gwdm = np.array(gwdm)
            # gwdm = np.rot90(gwdm, k=-1)
            gwdm = Image.fromarray(gwdm)

        # 获取裁剪框的左、上、右、下坐标
        points = item["points"]
        left = min(points[0][0], points[1][0])
        top = min(points[0][1], points[1][1])
        right = max(points[0][0], points[1][0])
        bottom = max(points[0][1], points[1][1])
        cropped_image = image.crop((left, top, right, bottom))
        # 可选裁剪结构标签（如果要输入网络）
        gmdm = gmdm.crop((left, top, right, bottom)) if gmdm else None
        gwdm = gwdm.crop((left, top, right, bottom)) if gwdm else None

        # 图像变换
        if self.transform:
            cropped_image = self.transform(cropped_image)
            if gmdm is not None:
                gmdm = self.mask_transform(gmdm)
            if gwdm is not None:
                gwdm = self.mask_transform(gwdm)

        # 解析label 的值获取水分，油分等信息
        label = item["label"].split(',')
        if len(label) != 4:
            self.errors.append(f"{item['json_path']} label != 4: {item['label']}")
            white, water, oil, elastic = 0.0, 0.0, 0.0, 0.0  # 设置默认值或处理错误
        else:
            try:
                white = float(label[0])
                water = float(label[1])
                oil = float(label[2])
                elastic = float(label[3])
            except ValueError:
                self.errors.append(f"{item['json_path']} label value error: {item['label']}")
                white, water, oil, elastic = 0.0, 0.0, 0.0, 0.0  # 设置默认值或处理错误

        targets = torch.tensor([white, water, oil, elastic], dtype=torch.float32)



        # return cropped_image, targets, os.path.basename(item["image_path"]), item["description"]
        # === 返回结构 ===
        dummy_mask = torch.zeros((1, 256, 256))  # 或用 torch.full((1, 256, 256), -1.0)
        return {
            "image": cropped_image,  # 预处理后的图像
            "targets": targets,  # 标签张量
            "filename": os.path.basename(item["image_path"]),
            "description": item["description"],  # 光照描述
            "gmdm": gmdm if gmdm is not None else dummy_mask,  # 结构伪标签（可选）
            "gwdm": gwdm if gwdm is not None else dummy_mask  # 结构伪标签（可选）
        }

class SkinDataLoader:
    def __init__(self, config_path, augment=False):
        with open(config_path, 'r') as file:
            self.config = yaml.safe_load(file)

        # 基础路径
        self.json_folder_path = self.config['paths']['json_folder']
        self.image_folder_path = self.config['paths']['image_folder']
        self.batch_size = self.config['batch_size']
        self.num_workers = self.config.get('num_workers', 4)  # 默认使用4个线程

        # 增加分割模式支持
        self.split_mode = self.config.get('split_mode', 'id')  # 默认使用 ID 划分
        self.split_ratio = self.config.get('split_ratio', 0.8)  # 比例划分时的默认比例

        # 判断是不是加载伪标签
        self.use_gmdm = self.config.get('use_gmdm', False) # 默认不加载
        self.use_gwdm = self.config.get('use_gwdm', False) # 默认不加载

        # 数据增强与预处理
        self.val_transform = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        if not augment:
            self.transform = transforms.Compose([
                transforms.Resize((256, 256)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
        else:
            self.transform = transforms.Compose([
                transforms.Resize((256, 256)),
                # transforms.RandomHorizontalFlip(),
                # transforms.RandomRotation(10),
                transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])

        # 根据分割模式初始化数据集
        if self.split_mode == "id":
            self.train_ids = self.config['ids']['train']
            self.test_ids = self.config['ids']['test']
            self.train_light_conditions = self.config['light_conditions']['train']
            self.test_light_conditions = self.config['light_conditions']['test']

            self.train_dataset = SkinDataset(
                self.json_folder_path,
                self.image_folder_path,
                ids=self.train_ids,
                light_conditions=self.train_light_conditions,
                transform=self.transform,
                use_gmdm= self.use_gmdm,
                use_gwdm= self.use_gwdm
            )
            self.test_dataset = SkinDataset(
                self.json_folder_path,
                self.image_folder_path,
                ids=self.test_ids,
                light_conditions=self.test_light_conditions,
                transform=self.val_transform,
                use_gmdm=self.use_gmdm,
                use_gwdm=self.use_gwdm
            )
        elif self.split_mode == "ratio":
            # 按比例划分数据集
            full_dataset = SkinDataset(
                self.json_folder_path,
                self.image_folder_path,
                transform=self.transform,
                use_gmdm=self.use_gmdm,
                use_gwdm=self.use_gwdm
            )
            self.train_dataset, self.test_dataset = self.split_dataset(full_dataset)
        else:
            raise ValueError("Invalid split_mode! Must be 'id' or 'ratio'.")

        # 数据加载器
        self.train_loader = DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers)
        self.test_loader = DataLoader(self.test_dataset, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers)

    def split_dataset(self, dataset):
        """按比例划分数据集"""
        total_size = len(dataset)
        train_size = int(total_size * self.split_ratio)
        test_size = total_size - train_size

        indices = list(range(total_size))
        random.shuffle(indices)

        train_indices = indices[:train_size]
        test_indices = indices[train_size:]

        train_data = torch.utils.data.Subset(dataset, train_indices)
        test_data = torch.utils.data.Subset(dataset, test_indices)
        

        return train_data, test_data

    def show_batch(self, images, targets, filenames, descriptions, gmdms=None, gwdms=None):
        batch_size = images.size(0)
        grid_size = math.ceil(batch_size ** 0.5)
        fig, axes = plt.subplots(grid_size, grid_size, figsize=(15, 15))

        for i in range(batch_size):
            row = i // grid_size
            col = i % grid_size

            img = images[i].permute(1, 2, 0).cpu()
            axes[row][col].imshow(img)
            axes[row][col].axis('off')

            label_info = f'{filenames[i]}\n{descriptions[i]}\n' \
                         f'White: {targets[i][0]:.2f}, Water: {targets[i][1]:.2f}\n' \
                         f'Oil: {targets[i][2]:.2f}, Elastic: {targets[i][3]:.2f}'

            axes[row][col].set_title(label_info, fontsize=8)

            # 可选可视化结构标签：红色边框为 gmdm，绿色边框为 gwdm
            if gmdms is not None and gmdms[i] is not None:
                g = gmdms[i][0].cpu().numpy()
                axes[row][col].imshow(g, cmap='Reds', alpha=0.2)
            if gwdms is not None and gwdms[i] is not None:
                w = gwdms[i][0].cpu().numpy()
                axes[row][col].imshow(w, cmap='Greens', alpha=0.2)

        # 关闭多余的子图
        for j in range(batch_size, grid_size * grid_size):
            row = j // grid_size
            col = j % grid_size
            axes[row][col].axis('off')

        plt.tight_layout()
        plt.show()

    def display_batches(self, show="train"):
        total_batches = 0
        loader = self.train_loader if show == "train" else self.test_loader

        for batch in loader:
            # 将 batch 字典展开为独立项
            images = batch["image"]
            targets = batch["targets"]
            filenames = batch["filename"]
            descriptions = batch["description"]
            gmdms = batch.get("gmdm", None)
            gwdms = batch.get("gwdm", None)

            print(f"Batch of images shape: {images.shape}")
            print(f"Batch of targets shape: {targets.shape}")
            self.show_batch(images, targets, filenames, descriptions, gmdms, gwdms)
            # for i in range(min(len(images), 4)):  # 每批最多展示 8 个
            #     show_image_with_masks(
            #         rgb=images[i],
            #         gmdm=gmdms[i] if gmdms is not None else None,
            #         gwdm=gwdms[i] if gwdms is not None else None,
            #         title=filenames[i]
            #     )

            total_batches += 1

        print(f"Total number of batches in {show}_loader: {total_batches}")
    def get_statistics(self, dataset):
        """统计数据集的 ID 和光照条件分布"""
        id_counts = defaultdict(int)
        light_condition_counts = defaultdict(int)
        if isinstance(dataset, torch.utils.data.Subset):
            indices = dataset.indices
            dataset = dataset.dataset
        else:
            indices = range(len(dataset))
    
        for idx in indices:
            item = dataset.data[idx]
            id_counts[item["group_id"]] += 1
            description = item["description"].split(',')
            if len(description) == 3:
                light_condition = int(description[2])
                light_condition_counts[light_condition] += 1
    
        print("ID counts:")
        for group_id, count in id_counts.items():
            print(f"ID: {group_id}, Count: {count}")
    
        print("\nLight condition counts:")
        for light_condition, count in light_condition_counts.items():
            print(f"Light Condition: {light_condition}, Count: {count}")
        # 打印统计信息
        print(f"Total samples: {len(indices)}")  # 打印数据总数



# # 使用示例
if __name__ == "__main__":
    data_loader = SkinDataLoader("../config/config_skin_Q.yaml")
    data_loader.display_batches("test")
    print("train===========")
    data_loader.train_dataset.get_statistics()
    print("test============")
    data_loader.test_dataset.get_statistics()
