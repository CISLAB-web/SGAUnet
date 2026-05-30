import os
import json
from torchvision import transforms
from torch.utils.data import Dataset
from PIL import Image
import torch
import cv2
from collections import defaultdict, Counter
import numpy as np


class CustomDataset(Dataset):
    def __init__(self, img_path, json_path, selected_devices=None, selected_parts=None, img_size=128, transform=None ):
        """
        初始化数据集
        :param img_path: 图像路径根目录
        :param json_path: 标签路径根目录
        :param selected_devices: 要选择的设备编号列表（如 ['01', '02']），默认为 None，加载所有设备
        :param selected_parts: 要选择的部位编号列表（如 ['1', ~ '9']），默认为 None，加载所有部位
        :param img_size: 图像裁剪后的大小（默认为 128x128）
        """
        self.img_path = img_path
        self.json_path = json_path
        self.selected_devices = selected_devices
        self.selected_parts = selected_parts
        self.img_size = img_size
        self.transform = transform if transform else transforms.ToTensor()
        self.class_distribution_skin_level = Counter()  # 用于统计类别分布
        self.data = []
        self._load_data()  # 加载数据

    def _load_data(self):
        """加载数据集并过滤"""
        for device in os.listdir(self.img_path):  # 遍历设备编号文件夹
            # print(f"Processing device: {device}")
            if self.selected_devices and device not in self.selected_devices:
                print(f"Skipping device: {device}")
                continue
    
            device_img_path = os.path.join(self.img_path, device)
            device_json_path = os.path.join(self.json_path, device)
    
            for person in os.listdir(device_img_path):  # 遍历人物编号文件夹
                # print(f"  Processing person: {person}")
                person_img_path = os.path.join(device_img_path, person)
                person_json_path = os.path.join(device_json_path, person)
    
                for img_name in os.listdir(person_img_path):  # 遍历图像文件
                    if not img_name.endswith(('.jpg', '.png', '.jpeg')):
                        continue
    
                    # 提取 img_id 时移除文件后缀
                    img_id = os.path.splitext("_".join(img_name.split("_")[:3]))[0]  # 提取 0726_01_F
                    for idx_area in range(9):  # 遍历部位编号
                        if self.selected_parts and str(idx_area) not in self.selected_parts:
                            continue
    
                        # 生成对应的 JSON 文件路径
                        json_name = f"{img_id}_{idx_area:02d}.json"
                        json_file = os.path.join(person_json_path, json_name)
    
                        if os.path.exists(json_file):
                            self.data.append({
                                "device": device,
                                "img_path": os.path.join(person_img_path, img_name),
                                "json_path": json_file,
                                "area": idx_area
                            })
                            # print(f"    Added sample: img={img_name}, area={idx_area}, json={json_name}")
                        else:
                            print(f"    Missing JSON: {json_file}")

    def __len__(self):
        """返回数据集的样本总数"""
        return len(self.data)

    def __getitem__(self, idx):
        try:
            sample = self.data[idx]
            img_path = sample["img_path"]
            json_path = sample["json_path"]
    
            # 加载图像
            img = cv2.imread(img_path)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
            with open(json_path, 'r', encoding='utf-8') as f:
                meta = json.load(f)
    
            bbox = meta['images'].get('bbox', None)
            if not bbox or len(bbox) != 4:
                raise ValueError(f"Invalid bbox in {json_path}")
    
            # 裁剪图像
            x_min, y_min, x_max, y_max = map(int, bbox)
            cropped_img = img[y_min:y_max, x_min:x_max]
            cropped_img = cv2.resize(cropped_img, (self.img_size, self.img_size))
            cropped_img = Image.fromarray(cropped_img)
    
            if self.transform:
                cropped_img = self.transform(cropped_img)
    
            # 提取标签
            info = meta.get("info", {})
            skin_type = info.get("skin_type", -1)  # 获取 skin_type，默认值为 -1 表示未知
            skin_type = torch.tensor(skin_type, dtype=torch.long)  # 转为 PyTorch 张量
    
            return cropped_img, skin_type
    
        except Exception as e:
            # print(f"Error processing {self.data[idx]}: {e}")
            print(f"error processing")
            return None



class CustomDataset_pore(Dataset):
    def __init__(self, img_path, json_path, selected_devices=None, selected_parts=None, transform=None ):
        """
        初始化数据集
        :param img_path: 图像路径根目录
        :param json_path: 标签路径根目录
        :param selected_devices: 要选择的设备编号列表（如 ['01', '02', '03']），默认为 None，加载所有设备
        :param selected_parts: 要选择的部位编号列表（如 ['5','6']），这个类用于加载左右脸颊的图片，用于毛孔等级等部分的训练
        :param img_size: 图像裁剪后的大小（默认为256x256）
        :return : 返回的是截图的图片，以及毛孔，色素等级， 弹力，水分，毛孔个数
        """
        self.img_path = img_path
        self.json_path = json_path
        self.selected_devices = selected_devices
        self.selected_parts = selected_parts
        self.img_size = 256
        self.transform = transform if transform else transforms.ToTensor()
        self.data = []
        self.class_distribution = Counter()  # 用于统计类别分布
        self._load_data()  # 加载数据
        

    def _load_data(self):
        """加载数据集并过滤"""
        for device in os.listdir(self.img_path):  # 遍历设备编号文件夹
            # print(f"Processing device: {device}")
            if self.selected_devices and device not in self.selected_devices:
                print(f"Skipping device: {device}")
                continue
    
            device_img_path = os.path.join(self.img_path, device)
            device_json_path = os.path.join(self.json_path, device)
    
            for person in os.listdir(device_img_path):  # 遍历人物编号文件夹
                # print(f"  Processing person: {person}")
                person_img_path = os.path.join(device_img_path, person)
                person_json_path = os.path.join(device_json_path, person)
    
                for img_name in os.listdir(person_img_path):  # 遍历图像文件
                    if not img_name.endswith(('.jpg', '.png', '.jpeg')):
                        continue
    
                    # 提取 img_id 时移除文件后缀
                    img_id = os.path.splitext("_".join(img_name.split("_")[:3]))[0]  # 提取 0726_01_F
                    for idx_area in range(9):  # 遍历部位编号
                        if self.selected_parts and str(idx_area) not in self.selected_parts:
                            continue
    
                        # 生成对应的 JSON 文件路径
                        json_name = f"{img_id}_{idx_area:02d}.json"
                        json_file = os.path.join(person_json_path, json_name)
    
                        if os.path.exists(json_file):
                            try:
                                # 检查是否能成功加载并解析 JSON
                                with open(json_file, 'r', encoding='utf-8') as f:
                                    meta = json.load(f)
                                bbox = meta['images'].get('bbox', None)
                                # 提取毛孔等级标签
                                annotations = meta.get("annotations", {})
                                pore_labels = [v for k, v in annotations.items() if "cheek_pore" in k]
                                if not bbox or len(bbox) != 4:
                                    raise ValueError("Invalid bbox")
                                if not pore_labels:
                                    raise ValueError("Invalid pore")
                                if pore_labels:
                                    pore_label = pore_labels[0]  # 取第一个有效值
                                    self.class_distribution[pore_label] += 1  # 更新类别计数
                                
                                # 如果通过所有检查，保存样本路径和信息
                                self.data.append({
                                    "device": device,
                                    "img_path": os.path.join(person_img_path, img_name),
                                    "json_path": json_file,
                                    "area": idx_area,
                                    "pore_level": pore_labels[0]  # 提前加载毛孔标签
                                })
                            except Exception as e:
                                print(f"Skipping invalid sample: {json_file} due to {e}")

    def __len__(self):
        """返回数据集的样本总数"""
        return len(self.data)

    def __getitem__(self, idx):
        try:
            sample = self.data[idx]
            img_path = sample["img_path"]
            json_path = sample["json_path"]
    
            # 加载图像
            img = cv2.imread(img_path)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
            with open(json_path, 'r', encoding='utf-8') as f:
                meta = json.load(f)
    
            bbox = meta['images'].get('bbox', None)
            if not bbox or len(bbox) != 4:
                raise ValueError(f"Invalid bbox in {json_path}")
    
            # 裁剪图像
            x_min, y_min, x_max, y_max = map(int, bbox)
            cropped_img = img[y_min:y_max, x_min:x_max]
            # cropped_img = cv2.resize(cropped_img, (self.img_size, self.img_size))
            cropped_img = Image.fromarray(cropped_img)
    
            if self.transform:
                cropped_img = self.transform(cropped_img)
    
            # 提取标签 ,每个部位的标签都不一样
            
            info = meta.get("info", {}) # 照片整体信息
            annotations = meta.get("annotations",{}) # 专家标定结果
            equipment = meta.get("equipment" , {}) # 机器检测结果
                        
            skin_type = info.get("skin_type", -1)  # 获取 skin_type，默认值为 -1 表示未知
            skin_type = torch.tensor(skin_type, dtype=torch.long)  # 转为 PyTorch 张量

            # 获取毛孔,色素等级的标签 ,在部位5，6 上
            pore_labels = torch.tensor([v for k, v in annotations.items() if "cheek_pore" in k], dtype=torch.long).squeeze()  # 专家标签毛孔
            pigmentation = torch.tensor([v for k, v in annotations.items() if "cheek_pigmentation" in k], dtype=torch.long).squeeze()  # 专家标签色素
            pore_number = torch.tensor([v for k , v in equipment.items() if "cheek_pore" in k], dtype=torch.long).squeeze() # 机器毛孔个数
            elasticity = torch.tensor([v for k , v in equipment.items() if "cheek_elasticity_R0" in k], dtype=torch.float).squeeze() # 机器弹力值R0
            moisture = torch.tensor([v for k , v in equipment.items() if "cheek_moisture" in k], dtype=torch.float).squeeze() # 机器检测的水分值


            # 构建标签字典
            labels = {
                "skin_type": skin_type,
                "pore_level": pore_labels,  # 毛孔等级
                "pigmentation_level": pigmentation, # 色素等级
                "pore_number": pore_number, # 毛孔个数 机器
                "elasticity": elasticity, # 弹力 机器
                "moisture": moisture, #水分 机器

            }
    
            return cropped_img, labels
    
        except Exception as e:
            print(f"error processing...{idx}")
            # return None
            # 随机选择下一个索引，确保数据加载不中断
            idx = (idx + 1) % len(self.data)

class CustomDataset_pore_pigm(Dataset):
    def __init__(self, img_path, json_path, selected_devices=None, selected_parts=None, transform=None ):
        """
        初始化数据集
        :param img_path: 图像路径根目录
        :param json_path: 标签路径根目录
        :param selected_devices: 要选择的设备编号列表（如 ['01', '02', '03']），默认为 None，加载所有设备
        :param selected_parts: 要选择的部位编号列表（如 ['5','6']），这个类用于加载左右脸颊的图片，用于毛孔等级,色素等级等部分的训练
        :param img_size: 图像裁剪后的大小（默认为256x256）
        :return : 返回的是截图的图片，以及毛孔，色素等级， 弹力，水分，毛孔个数
        """
        self.img_path = img_path
        self.json_path = json_path
        self.selected_devices = selected_devices
        self.selected_parts = selected_parts
        self.img_size = 256
        self.transform = transform if transform else transforms.ToTensor()
        self.data = []
        self.class_distribution = Counter()  # 用于统计类别分布,毛孔
        self.class_distribution_pigmentation = Counter()
        self.class_distribution_skin_type = Counter()
        
        self._load_data()  # 加载数据
        

    def _load_data(self):
        """加载数据集并过滤"""
        for device in os.listdir(self.img_path):  # 遍历设备编号文件夹
            # print(f"Processing device: {device}")
            if self.selected_devices and device not in self.selected_devices:
                print(f"Skipping device: {device}")
                continue
    
            device_img_path = os.path.join(self.img_path, device)
            device_json_path = os.path.join(self.json_path, device)
    
            for person in os.listdir(device_img_path):  # 遍历人物编号文件夹
                # print(f"  Processing person: {person}")
                person_img_path = os.path.join(device_img_path, person)
                person_json_path = os.path.join(device_json_path, person)
    
                for img_name in os.listdir(person_img_path):  # 遍历图像文件
                    if not img_name.endswith(('.jpg', '.png', '.jpeg')):
                        continue
    
                    # 提取 img_id 时移除文件后缀
                    img_id = os.path.splitext("_".join(img_name.split("_")[:3]))[0]  # 提取 0726_01_F
                    for idx_area in range(9):  # 遍历部位编号
                        if self.selected_parts and str(idx_area) not in self.selected_parts:
                            continue
    
                        # 生成对应的 JSON 文件路径
                        json_name = f"{img_id}_{idx_area:02d}.json"
                        json_file = os.path.join(person_json_path, json_name)
    
                        if os.path.exists(json_file):
                            try:
                                # 检查是否能成功加载并解析 JSON
                                with open(json_file, 'r', encoding='utf-8') as f:
                                    meta = json.load(f)
                                bbox = meta['images'].get('bbox', None)
                                # 提取毛孔等级标签
                                annotations = meta.get("annotations", {})
                                info = meta.get("info", {})
                                skin_type = info.get("skin_type", -1)  # 获取 skin_type，默认值为 -1 表示未知
                                pore_labels = [v for k, v in annotations.items() if "cheek_pore" in k]
                                pigme_labels = [v for k, v in annotations.items() if "cheek_pigmentation" in k]
                                if not bbox or len(bbox) != 4:
                                    raise ValueError("Invalid bbox")
                                if not pore_labels:
                                    raise ValueError("Invalid pore")
                                if not pigme_labels:
                                    raise ValueError("Invalid pigmentaion")
                                if not skin_type:
                                    raise ValueError("Invalid skin type")
                                if pore_labels:
                                    pore_label = pore_labels[0]  # 取第一个有效值
                                    self.class_distribution[pore_label] += 1  # 更新类别计数
                                if pigme_labels:
                                    pigme_label = pigme_labels[0] # 取第一个有效值
                                    self.class_distribution_pigmentation[pigme_label]+=1 
                                if skin_type:
                                    self.class_distribution_skin_type[skin_type]+=1
                                    print(f":skine_type:{skin_type}")
                                # 如果通过所有检查，保存样本路径和信息
                                self.data.append({
                                    "device": device,
                                    "img_path": os.path.join(person_img_path, img_name),
                                    "json_path": json_file,
                                    "area": idx_area,
                                    "pore_level": pore_labels[0],  # 提前加载毛孔标签
                                    "pigmentation_level": pigme_labels[0],
                                    'skin_type': skin_type,
                                })
                            except Exception as e:
                                print(f"Skipping invalid sample: {json_file} due to {e}")

    def __len__(self):
        """返回数据集的样本总数"""
        return len(self.data)

    def __getitem__(self, idx):
        try:
            sample = self.data[idx]
            img_path = sample["img_path"]
            json_path = sample["json_path"]
    
            # 加载图像
            img = cv2.imread(img_path)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
            with open(json_path, 'r', encoding='utf-8') as f:
                meta = json.load(f)
    
            bbox = meta['images'].get('bbox', None)
            if not bbox or len(bbox) != 4:
                raise ValueError(f"Invalid bbox in {json_path}")
    
            # 裁剪图像
            x_min, y_min, x_max, y_max = map(int, bbox)
            cropped_img = img[y_min:y_max, x_min:x_max]
            # cropped_img = cv2.resize(cropped_img, (self.img_size, self.img_size))
            cropped_img = Image.fromarray(cropped_img)
    
            if self.transform:
                cropped_img = self.transform(cropped_img)
    
            # 提取标签 ,每个部位的标签都不一样
            
            info = meta.get("info", {}) # 照片整体信息
            annotations = meta.get("annotations",{}) # 专家标定结果
            equipment = meta.get("equipment" , {}) # 机器检测结果
                        
            skin_type = info.get("skin_type", -1)  # 获取 skin_type，默认值为 -1 表示未知
            skin_type = torch.tensor(skin_type, dtype=torch.long)  # 转为 PyTorch 张量

            # 获取毛孔,色素等级的标签 ,在部位5，6 上
            pore_labels = torch.tensor([v for k, v in annotations.items() if "cheek_pore" in k], dtype=torch.long).squeeze()  # 专家标签毛孔
            pigmentation = torch.tensor([v for k, v in annotations.items() if "cheek_pigmentation" in k], dtype=torch.long).squeeze()  # 专家标签色素
            pore_number = torch.tensor([v for k , v in equipment.items() if "cheek_pore" in k], dtype=torch.long).squeeze() # 机器毛孔个数
            elasticity = torch.tensor([v for k , v in equipment.items() if "cheek_elasticity_R0" in k], dtype=torch.float).squeeze() # 机器弹力值R0
            moisture = torch.tensor([v for k , v in equipment.items() if "cheek_moisture" in k], dtype=torch.float).squeeze() # 机器检测的水分值


            # 构建标签字典
            labels = {
                "skin_type": skin_type,
                "pore_level": pore_labels,  # 毛孔等级
                "pigmentation_level": pigmentation, # 色素等级
                "pore_number": pore_number, # 毛孔个数 机器
                "elasticity": elasticity, # 弹力 机器
                "moisture": moisture, #水分 机器

            }
    
            return cropped_img, labels
    
        except Exception as e:
            print(f"error processing...{idx}")
            # return None
            # 随机选择下一个索引，确保数据加载不中断
            idx = (idx + 1) % len(self.data)


class CustomDataset_perocular_wrinkle(Dataset):
    def __init__(self, img_path, json_path, selected_devices=None, selected_parts=None, transform=None ):
        """
        初始化数据集
        :param img_path: 图像路径根目录
        :param json_path: 标签路径根目录
        :param selected_devices: 要选择的设备编号列表（如 ['01', '02', '03']），默认为 None，加载所有设备
        :param selected_parts: 要选择的部位编号列表（如 ['3','4']），这个类用于加载左右眼角部分的皱纹的
        :param img_size: 图像裁剪后的大小（默认为256x256）
        :return : 返回的是截图的图片，以及眼角的皱纹
        """
        self.img_path = img_path
        self.json_path = json_path
        self.selected_devices = selected_devices
        self.selected_parts = selected_parts
        self.img_size = 256
        self.transform = transform if transform else transforms.ToTensor()
        self.data = []
        self.class_distribution_wrinkle= Counter()  # 用于统计眼角皱纹等级的数量
        # self.class_distribution_pigmentation = Counter()
        
        self._load_data()  # 加载数据
        

    def _load_data(self):
        """加载数据集并过滤"""
        for device in os.listdir(self.img_path):  # 遍历设备编号文件夹
            # print(f"Processing device: {device}")
            if self.selected_devices and device not in self.selected_devices:
                print(f"Skipping device: {device}")
                continue
    
            device_img_path = os.path.join(self.img_path, device)
            device_json_path = os.path.join(self.json_path, device)
    
            for person in os.listdir(device_img_path):  # 遍历人物编号文件夹
                # print(f"  Processing person: {person}")
                person_img_path = os.path.join(device_img_path, person)
                person_json_path = os.path.join(device_json_path, person)
    
                for img_name in os.listdir(person_img_path):  # 遍历图像文件
                    if not img_name.endswith(('.jpg', '.png', '.jpeg')):
                        continue
    
                    # 提取 img_id 时移除文件后缀
                    img_id = os.path.splitext("_".join(img_name.split("_")[:3]))[0]  # 提取 0726_01_F
                    for idx_area in range(9):  # 遍历部位编号
                        if self.selected_parts and str(idx_area) not in self.selected_parts:
                            continue
    
                        # 生成对应的 JSON 文件路径
                        json_name = f"{img_id}_{idx_area:02d}.json"
                        json_file = os.path.join(person_json_path, json_name)
    
                        if os.path.exists(json_file):
                            try:
                                # 检查是否能成功加载并解析 JSON
                                with open(json_file, 'r', encoding='utf-8') as f:
                                    meta = json.load(f)
                                bbox = meta['images'].get('bbox', None)
                                # 提取毛孔等级标签
                                annotations = meta.get("annotations", {})
                                # pore_labels = [v for k, v in annotations.items() if "cheek_pore" in k]
                                # pigme_labels = [v for k, v in annotations.items() if "cheek_pigmentation" in k]
                                wrinkle_labels = [v for k, v in annotations.items() if "perocular_wrinkle" in k] #眼周皱纹等级
                                if not bbox or len(bbox) != 4:
                                    raise ValueError("Invalid bbox")
                                if not wrinkle_labels:
                                    raise ValueError("Invalid wrinkle")
                                if wrinkle_labels:
                                    wrinkle_label = wrinkle_labels[0]
                                    self.class_distribution_wrinkle[wrinkle_label] += 1
                                # if not pore_labels:
                                #     raise ValueError("Invalid pore")
                                # if not pigme_labels:
                                #     raise ValueError("Invalid pigmentaion")
                                # if pore_labels:
                                #     pore_label = pore_labels[0]  # 取第一个有效值
                                #     self.class_distribution[pore_label] += 1  # 更新类别计数
                                # if pigme_labels:
                                #     pigme_label = pigme_labels[0] # 取第一个有效值
                                #     self.class_distribution_pigmentation[pigme_label]+=1 
                                # 如果通过所有检查，保存样本路径和信息
                                self.data.append({
                                    "device": device,
                                    "img_path": os.path.join(person_img_path, img_name),
                                    "json_path": json_file,
                                    "area": idx_area,
                                    # "pore_level": pore_labels[0],  # 提前加载毛孔标签
                                    # "pigmentation_level": pigme_labels[0]
                                    "perocular_wrinkle_level": wrinkle_labels[0]
                                })
                            except Exception as e:
                                print(f"Skipping invalid sample: {json_file} due to {e}")

    def __len__(self):
        """返回数据集的样本总数"""
        return len(self.data)

    def __getitem__(self, idx):
        try:
            sample = self.data[idx]
            img_path = sample["img_path"]
            json_path = sample["json_path"]
    
            # 加载图像
            img = cv2.imread(img_path)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
            with open(json_path, 'r', encoding='utf-8') as f:
                meta = json.load(f)
    
            bbox = meta['images'].get('bbox', None)
            if not bbox or len(bbox) != 4:
                raise ValueError(f"Invalid bbox in {json_path}")
    
            # 裁剪图像
            x_min, y_min, x_max, y_max = map(int, bbox)
            cropped_img = img[y_min:y_max, x_min:x_max]
            # cropped_img = cv2.resize(cropped_img, (self.img_size, self.img_size))
            cropped_img = Image.fromarray(cropped_img)
    
            if self.transform:
                cropped_img = self.transform(cropped_img)
    
            # 提取标签 ,每个部位的标签都不一样
            
            info = meta.get("info", {}) # 照片整体信息
            annotations = meta.get("annotations",{}) # 专家标定结果
            equipment = meta.get("equipment" , {}) # 机器检测结果
                        
            skin_type = info.get("skin_type", -1)  # 获取 skin_type，默认值为 -1 表示未知
            skin_type = torch.tensor(skin_type, dtype=torch.long)  # 转为 PyTorch 张量

            # 获取眼周皱纹等级的标签, 在部位 3, 4 上
            wrinkle_labels = torch.tensor([v for k, v in annotations.items() if "perocular_wrinkle" in k], dtype=torch.long).squeeze() 

            # 获取毛孔,色素等级的标签 ,在部位5，6 上
            # pore_labels = torch.tensor([v for k, v in annotations.items() if "cheek_pore" in k], dtype=torch.long).squeeze()  # 专家标签毛孔
            # pigmentation = torch.tensor([v for k, v in annotations.items() if "cheek_pigmentation" in k], dtype=torch.long).squeeze()  # 专家标签色素
            # pore_number = torch.tensor([v for k , v in equipment.items() if "cheek_pore" in k], dtype=torch.long).squeeze() # 机器毛孔个数
            # elasticity = torch.tensor([v for k , v in equipment.items() if "cheek_elasticity_R0" in k], dtype=torch.float).squeeze() # 机器弹力值R0
            # moisture = torch.tensor([v for k , v in equipment.items() if "cheek_moisture" in k], dtype=torch.float).squeeze() # 机器检测的水分值


            # 构建标签字典
            labels = {
                "skin_type": skin_type,
                # "pore_level": pore_labels,  # 毛孔等级
                # "pigmentation_level": pigmentation, # 色素等级
                # "pore_number": pore_number, # 毛孔个数 机器
                # "elasticity": elasticity, # 弹力 机器
                # "moisture": moisture, #水分 机器
                "perocular_wrinkle_level": wrinkle_labels,
            }
    
            return cropped_img, labels
    
        except Exception as e:
            print(f"error processing...{idx}")
            # return None
            # 随机选择下一个索引，确保数据加载不中断
            idx = (idx + 1) % len(self.data)




class CustomDataset_skin_level(Dataset):
    def __init__(self, img_path, json_path, selected_devices=None, selected_parts=None, transform=None ):
        """
        用于骨骼网络的预训练测试，进行骨干模型在数据集上测试用的
        初始化数据集
        :param img_path: 图像路径根目录
        :param json_path: 标签路径根目录
        :param selected_devices: 要选择的设备编号列表（如 ['01', '02', '03']），默认为 None，加载所有设备
        :param selected_parts: 要选择的部位编号列表（如 ['5','6']），这个类用于加载左右脸颊的图片，用于毛孔等级,色素等级等部分的训练
        :param img_size: 图像裁剪后的大小（默认为256x256）
        :return : 返回的是截图的图片，以及毛孔，色素等级， 弹力，水分，毛孔个数
        """
        self.img_path = img_path
        self.json_path = json_path
        self.selected_devices = selected_devices
        self.selected_parts = selected_parts
        self.img_size = 256
        self.transform = transform if transform else transforms.ToTensor()
        self.data = []
        self.class_distribution = Counter()  # 用于统计类别分布,毛孔
        self.class_distribution_pigmentation = Counter()
        self.class_distribution_skin_type = Counter()
        
        self._load_data()  # 加载数据
        

    def _load_data(self):
        """加载数据集并过滤"""
        for device in os.listdir(self.img_path):  # 遍历设备编号文件夹
            # print(f"Processing device: {device}")
            if self.selected_devices and device not in self.selected_devices:
                print(f"Skipping device: {device}")
                continue
    
            device_img_path = os.path.join(self.img_path, device)
            device_json_path = os.path.join(self.json_path, device)
    
            for person in os.listdir(device_img_path):  # 遍历人物编号文件夹
                # print(f"  Processing person: {person}")
                person_img_path = os.path.join(device_img_path, person)
                person_json_path = os.path.join(device_json_path, person)
    
                for img_name in os.listdir(person_img_path):  # 遍历图像文件
                    if not img_name.endswith(('.jpg', '.png', '.jpeg')):
                        continue
    
                    # 提取 img_id 时移除文件后缀
                    img_id = os.path.splitext("_".join(img_name.split("_")[:3]))[0]  # 提取 0726_01_F
                    for idx_area in range(9):  # 遍历部位编号
                        if self.selected_parts and str(idx_area) not in self.selected_parts:
                            continue
    
                        # 生成对应的 JSON 文件路径
                        json_name = f"{img_id}_{idx_area:02d}.json"
                        json_file = os.path.join(person_json_path, json_name)
    
                        if os.path.exists(json_file):
                            try:
                                # 检查是否能成功加载并解析 JSON
                                with open(json_file, 'r', encoding='utf-8') as f:
                                    meta = json.load(f)
                                bbox = meta['images'].get('bbox', None)
                                # 提取毛孔等级标签
                                # annotations = meta.get("annotations", {})
                                info = meta.get("info", {})
                                skin_type = info.get("skin_type", -1)  # 获取 skin_type，默认值为 -1 表示未知
                                # pore_labels = [v for k, v in annotations.items() if "cheek_pore" in k]
                                # pigme_labels = [v for k, v in annotations.items() if "cheek_pigmentation" in k]
                                if not bbox or len(bbox) != 4:
                                    raise ValueError("Invalid bbox")
                                # if not pore_labels:
                                #     raise ValueError("Invalid pore")
                                # if not pigme_labels:
                                #     raise ValueError("Invalid pigmentaion")
                                if not skin_type:
                                    raise ValueError("Invalid skin type")
                                # if pore_labels:
                                #     pore_label = pore_labels[0]  # 取第一个有效值
                                #     self.class_distribution[pore_label] += 1  # 更新类别计数
                                # if pigme_labels:
                                #     pigme_label = pigme_labels[0] # 取第一个有效值
                                #     self.class_distribution_pigmentation[pigme_label]+=1 
                                if skin_type:
                                    self.class_distribution_skin_type[skin_type]+=1
                                    # print(f":skine_type:{skin_type}")
                                # 如果通过所有检查，保存样本路径和信息
                                self.data.append({
                                    "device": device,
                                    "img_path": os.path.join(person_img_path, img_name),
                                    "json_path": json_file,
                                    "area": idx_area,
                                    # "pore_level": pore_labels[0],  # 提前加载毛孔标签
                                    # "pigmentation_level": pigme_labels[0],
                                    'skin_type': skin_type,
                                })
                            except Exception as e:
                                print(f"Skipping invalid sample: {json_file} due to {e}")

    def __len__(self):
        """返回数据集的样本总数"""
        return len(self.data)

    def __getitem__(self, idx):
        try:
            sample = self.data[idx]
            img_path = sample["img_path"]
            json_path = sample["json_path"]
    
            # 加载图像
            img = cv2.imread(img_path)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
            with open(json_path, 'r', encoding='utf-8') as f:
                meta = json.load(f)
    
            bbox = meta['images'].get('bbox', None)
            if not bbox or len(bbox) != 4:
                raise ValueError(f"Invalid bbox in {json_path}")
    
            # 裁剪图像
            x_min, y_min, x_max, y_max = map(int, bbox)
            cropped_img = img[y_min:y_max, x_min:x_max]
            # cropped_img = cv2.resize(cropped_img, (self.img_size, self.img_size))
            # cropped_img = Image.fromarray(cropped_img)
            # print(f"after cropped_img, {cropped_img.shape}")

            # 将裁剪的图像和毛孔检测图都转换为 PIL 格式
            cropped_img_pil = Image.fromarray(cropped_img)

            if self.transform:
                cropped_img_transformed = self.transform(cropped_img_pil)
    
            # 提取标签 ,每个部位的标签都不一样
            info = meta.get("info", {}) # 照片整体信息
            annotations = meta.get("annotations",{}) # 专家标定结果
            equipment = meta.get("equipment" , {}) # 机器检测结果
                        
            skin_type = info.get("skin_type", -1)  # 获取 skin_type，默认值为 -1 表示未知
            skin_type = torch.tensor(skin_type, dtype=torch.long)  # 转为 PyTorch 张量

            # 构建标签字典
            labels = {
                "skin_type": skin_type,

            }
    
            return cropped_img_transformed,  labels
    
        except Exception as e:
            print(f"error processing...{idx}")
            # return None
            # 随机选择下一个索引，确保数据加载不中断
            idx = (idx + 1) % len(self.data)
