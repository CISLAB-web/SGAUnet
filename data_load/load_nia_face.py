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

"""
这个类用来加载处理后的NIA的数据， NIA_face， 该数据只保存F角度的两个脸颊位置的数据
"""

class NiaDataset(Dataset):
    def __init__(self, processed_dir, transform=None, use_gmdm=False, use_gwdm=False):
        """
        processed_dir: NIA结构化数据目录，内部需包含 image/gmdm/gwdm/label 文件夹
        """
        self.image_dir = os.path.join(processed_dir, "image")
        self.gmdm_dir = os.path.join(processed_dir, "gmdm") if use_gmdm else None
        self.gwdm_dir = os.path.join(processed_dir, "gwdm") if use_gwdm else None
        self.label_dir = os.path.join(processed_dir, "label")

        self.transform = transform
        self.use_gmdm = use_gmdm
        self.use_gwdm = use_gwdm

        self.mask_transform = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.ToTensor(),
            transforms.Lambda(lambda x: x[0:1, :, :])  # 取R通道
        ])

        self.data = self.load_data()

    def load_data(self):
        data = []
        for file_name in os.listdir(self.label_dir):
            if not file_name.endswith(".json"):
                continue

            label_path = os.path.join(self.label_dir, file_name)
            image_name = file_name.replace(".json", ".png")
            image_path = os.path.join(self.image_dir, image_name)
            gmdm_path = os.path.join(self.gmdm_dir, image_name) if self.use_gmdm else None
            gwdm_path = os.path.join(self.gwdm_dir, image_name) if self.use_gwdm else None

            with open(label_path, "r", encoding="utf-8") as f:
                meta_list = json.load(f)["meta_list"]

            for meta in meta_list:
                bbox = meta.get("images", {}).get("bbox", None)
                facepart = meta.get("images", {}).get("facepart", None)
                ann = meta.get("annotations", {})
                equip = meta.get("equipment", {})

                if not bbox or facepart not in [5, 6]:
                    continue

                if facepart == 5:
                    pore_grade = ann.get("l_cheek_pore", 0)
                    pigm_grade = ann.get("l_cheek_pigmentation", 0)
                    moisture = equip.get("l_cheek_moisture", 0)
                    elasticity = equip.get("l_cheek_elasticity_Q0", 0)
                elif facepart == 6:
                    pore_grade = ann.get("r_cheek_pore", 0)
                    pigm_grade = ann.get("r_cheek_pigmentation", 0)
                    moisture = equip.get("r_cheek_moisture", 0)
                    elasticity = equip.get("r_cheek_elasticity_Q0", 0)

                targets = [pore_grade, pigm_grade, moisture, elasticity]

                data.append({
                    "image_path": image_path,
                    "gmdm_path": gmdm_path,
                    "gwdm_path": gwdm_path,
                    "bbox": bbox,
                    "targets": targets,
                    "facepart": facepart,
                    "filename": image_name
                })
        return data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]

        image = Image.open(item["image_path"]).convert("RGB")
        gmdm = Image.open(item["gmdm_path"]).convert("RGB") if self.use_gmdm else None
        gwdm = Image.open(item["gwdm_path"]).convert("RGB") if self.use_gwdm else None

        x_min, y_min, x_max, y_max = map(int, item["bbox"])
        image = image.crop((x_min, y_min, x_max, y_max))
        if gmdm is not None:
            gmdm = gmdm.crop((x_min, y_min, x_max, y_max))
        if gwdm is not None:
            gwdm = gwdm.crop((x_min, y_min, x_max, y_max))

        if self.transform:
            image = self.transform(image)
        if gmdm is not None:
            gmdm = self.mask_transform(gmdm)
        if gwdm is not None:
            gwdm = self.mask_transform(gwdm)

        dummy_mask = torch.zeros((1, 256, 256))

        return {
            "image": image,
            "targets": torch.tensor(item["targets"], dtype=torch.float32),
            "filename": item["filename"],
            "description": f"facepart:{item['facepart']}",
            "gmdm": gmdm if gmdm is not None else dummy_mask,
            "gwdm": gwdm if gwdm is not None else dummy_mask
        }

class NIADataLoader:
    def __init__(self, config_path, augment=False):
        with open(config_path, 'r') as file:
            self.config = yaml.safe_load(file)

        processed_dir = self.config['paths']['processed_dir']
        self.batch_size = self.config['batch_size']
        self.num_workers = self.config.get('num_workers', 4)

        self.split_ratio = self.config.get('split_ratio', 0.8)
        self.use_gmdm = self.config.get('use_gmdm', False)
        self.use_gwdm = self.config.get('use_gwdm', False)
        self.face_parts = self.config.get('face_parts', [5, 6])  # 5左脸，6右脸，默认全部用

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
                transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])

        full_dataset = NiaDataset(
            processed_dir=processed_dir,
            transform=self.transform,
            use_gmdm=self.use_gmdm,
            use_gwdm=self.use_gwdm
        )

        # 过滤左右脸
        if self.face_parts:
            full_dataset.data = [d for d in full_dataset.data if d["facepart"] in self.face_parts]

        self.train_dataset, self.test_dataset = self.split_dataset(full_dataset)

        self.train_loader = DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers)
        self.test_loader = DataLoader(self.test_dataset, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers)

    def split_dataset(self, dataset):
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
                         f'Pore: {targets[i][0]:.2f}, Pigment: {targets[i][1]:.2f}\n' \
                         f'Moisture: {targets[i][2]:.2f}, Elastic Q0: {targets[i][3]:.2f}'

            axes[row][col].set_title(label_info, fontsize=8)

            if gmdms is not None and gmdms[i] is not None:
                g = gmdms[i][0].cpu().numpy()
                axes[row][col].imshow(g, cmap='Reds', alpha=0.4)
            if gwdms is not None and gwdms[i] is not None:
                w = gwdms[i][0].cpu().numpy()
                axes[row][col].imshow(w, cmap='Greens', alpha=0.4)

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
            images = batch["image"]
            targets = batch["targets"]
            filenames = batch["filename"]
            descriptions = batch["description"]
            gmdms = batch.get("gmdm", None)
            gwdms = batch.get("gwdm", None)

            print(f"Batch of images shape: {images.shape}")
            print(f"Batch of targets shape: {targets.shape}")
            self.show_batch(images, targets, filenames, descriptions, gmdms, gwdms)

            total_batches += 1

        print(f"Total number of batches in {show}_loader: {total_batches}")

# # 使用示例
# if __name__ == "__main__":
#     data_loader = NIADataLoader("../config/config_NIA.yaml")
#     data_loader.display_batches("test")
#     # print("train===========")
#     # data_loader.train_dataset.get_statistics()
#     # print("test============")
#     # data_loader.test_dataset.get_statistics()
