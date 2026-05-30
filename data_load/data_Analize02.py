"""

分析专家评判的结果
"""
import os
import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# 定义存放JSON文件的顶层文件夹路径
json_folder = 'E:/SKIN_DATA/NIA korean facial image dataset/label/01'  # 替换为您JSON文件所在的文件夹路径

# 定义要接受的annotations的键
accepted_annotations = [
    "forehead_pigmentation", "forehead_wrinkle", "glabellus_wrinkle",
    "l_perocular_wrinkle", "r_perocular_wrinkle", "l_cheek_pore",
    "l_cheek_pigmentation", "r_cheek_pore", "r_cheek_pigmentation",
    "lip_dryness", "chin_sagging"
]

# 初始化一个空的列表来存储所有的注释信息
all_records = []

# 遍历顶层文件夹中的所有子目录和文件
for root, dirs, files in os.walk(json_folder):
    for filename in files:
        if filename.endswith(".json"):
            file_path = os.path.join(root, filename)

            # 读取并解析JSON文件
            with open(file_path, 'r', encoding='utf-8') as json_file:
                data = json.load(json_file)

                # 提取info和annotations部分
                info = data.get('info', {})
                annotations = data.get('annotations', {})

                # 过滤掉不在接受范围内的 annotations 键
                filtered_annotations = {}
                for key, value in annotations.items():
                    if key in accepted_annotations and isinstance(value, (int, float)):
                        filtered_annotations[key] = value

                # 如果有符合条件的 annotations，才将其合并到记录中
                if filtered_annotations:
                    record = {**info, **filtered_annotations}
                    all_records.append(record)

# 将所有记录转化为DataFrame
annotations_data = pd.DataFrame(all_records)

# 检查数据
print("Collected Annotations Data:")
print(annotations_data.head())

# 获取所有 annotations 列（只处理数值型和字符串型的列）
annotation_columns = [col for col in annotations_data.columns if
                      col not in ['id', 'filename', 'gender', 'age', 'date', 'skin_type', 'sensitive']]

# 统计每种类型的值分布并绘制柱状图
for col in annotation_columns:
    # 过滤掉NaN值并确保列中只包含简单类型数据
    clean_data = annotations_data[col].dropna()

    # 统计该类型值的分布情况
    value_counts = clean_data.value_counts().sort_index()

    # 如果该列的 value_counts 为空，则跳过
    if value_counts.empty:
        print(f"Skipping column '{col}' as it has no data.")
        continue

    # 绘制柱状图
    plt.figure(figsize=(8, 6))
    sns.barplot(x=value_counts.index, y=value_counts.values, palette="viridis")
    plt.title(f'Distribution of {col}')
    plt.xlabel(f'Values of {col}')
    plt.ylabel('Count')
    plt.xticks(rotation=45)
    plt.show()