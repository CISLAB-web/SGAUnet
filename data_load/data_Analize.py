""""
分析隔壁实验室的数据，使用的是机器的数据的分析结果
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import numpy as np

# 读取数据
file_path = 'C:/Users/CHUXIN/Desktop/measurement_data.csv'  # 替换为您的数据文件路径
df = pd.read_csv(file_path,encoding='utf-8')
print(df.head())

# 定义韩文列名与英文列名的映射关系
columns_mapping = {
    'subject_no': 'subject_no',
    '수분_이마': 'moisture_forehead',
    '수분_오른쪽볼': 'moisture_right_cheek',
    '수분_왼쪽볼': 'moisture_left_cheek',
    '수분_턱': 'moisture_chin',
    '탄력_턱_R0': 'elasticity_chin_R0',
    '탄력_턱_R1': 'elasticity_chin_R1',
    '탄력_턱_R2': 'elasticity_chin_R2',
    '탄력_턱_R3': 'elasticity_chin_R3',
    '탄력_턱_R4': 'elasticity_chin_R4',
    '탄력_턱_R5': 'elasticity_chin_R5',
    '탄력_턱_R6': 'elasticity_chin_R6',
    '탄력_턱_R7': 'elasticity_chin_R7',
    '탄력_턱_R8': 'elasticity_chin_R8',
    '탄력_턱_R9': 'elasticity_chin_R9',
    '탄력_턱_Q0': 'elasticity_chin_Q0',
    '탄력_턱_Q1': 'elasticity_chin_Q1',
    '탄력_턱_Q2': 'elasticity_chin_Q2',
    '탄력_턱_Q3': 'elasticity_chin_Q3',
    '탄력_왼쪽볼_R0': 'elasticity_left_cheek_R0',
    '탄력_왼쪽볼_R1': 'elasticity_left_cheek_R1',
    '탄력_왼쪽볼_R2': 'elasticity_left_cheek_R2',
    '탄력_왼쪽볼_R3': 'elasticity_left_cheek_R3',
    '탄력_왼쪽볼_R4': 'elasticity_left_cheek_R4',
    '탄력_왼쪽볼_R5': 'elasticity_left_cheek_R5',
    '탄력_왼쪽볼_R6': 'elasticity_left_cheek_R6',
    '탄력_왼쪽볼_R7': 'elasticity_left_cheek_R7',
    '탄력_왼쪽볼_R8': 'elasticity_left_cheek_R8',
    '탄력_왼쪽볼_R9': 'elasticity_left_cheek_R9',
    '탄력_왼쪽볼_Q0': 'elasticity_left_cheek_Q0',
    '탄력_왼쪽볼_Q1': 'elasticity_left_cheek_Q1',
    '탄력_왼쪽볼_Q2': 'elasticity_left_cheek_Q2',
    '탄력_왼쪽볼_Q3': 'elasticity_left_cheek_Q3',
    '탄력_오른쪽볼_R0': 'elasticity_right_cheek_R0',
    '탄력_오른쪽볼_R1': 'elasticity_right_cheek_R1',
    '탄력_오른쪽볼_R2': 'elasticity_right_cheek_R2',
    '탄력_오른쪽볼_R3': 'elasticity_right_cheek_R3',
    '탄력_오른쪽볼_R4': 'elasticity_right_cheek_R4',
    '탄력_오른쪽볼_R5': 'elasticity_right_cheek_R5',
    '탄력_오른쪽볼_R6': 'elasticity_right_cheek_R6',
    '탄력_오른쪽볼_R7': 'elasticity_right_cheek_R7',
    '탄력_오른쪽볼_R8': 'elasticity_right_cheek_R8',
    '탄력_오른쪽볼_R9': 'elasticity_right_cheek_R9',
    '탄력_오른쪽볼_Q0': 'elasticity_right_cheek_Q0',
    '탄력_오른쪽볼_Q1': 'elasticity_right_cheek_Q1',
    '탄력_오른쪽볼_Q2': 'elasticity_right_cheek_Q2',
    '탄력_오른쪽볼_Q3': 'elasticity_right_cheek_Q3',
    '탄력_이마_R0': 'elasticity_forehead_R0',
    '탄력_이마_R1': 'elasticity_forehead_R1',
    '탄력_이마_R2': 'elasticity_forehead_R2',
    '탄력_이마_R3': 'elasticity_forehead_R3',
    '탄력_이마_R4': 'elasticity_forehead_R4',
    '탄력_이마_R5': 'elasticity_forehead_R5',
    '탄력_이마_R6': 'elasticity_forehead_R6',
    '탄력_이마_R7': 'elasticity_forehead_R7',
    '탄력_이마_R8': 'elasticity_forehead_R8',
    '탄력_이마_R9': 'elasticity_forehead_R9',
    '탄력_이마_Q0': 'elasticity_forehead_Q0',
    '탄력_이마_Q1': 'elasticity_forehead_Q1',
    '탄력_이마_Q2': 'elasticity_forehead_Q2',
    '탄력_이마_Q3': 'elasticity_forehead_Q3',
    '주름_왼쪽눈가_Ra': 'wrinkle_left_eye_Ra',
    '주름_왼쪽눈가_Rq': 'wrinkle_left_eye_Rq',
    '주름_왼쪽눈가_Rmax': 'wrinkle_left_eye_Rmax',
    '주름_왼쪽눈가_R3z': 'wrinkle_left_eye_R3z',
    '주름_왼쪽눈가_Rt': 'wrinkle_left_eye_Rt',
    '주름_왼쪽눈가_Rz=Rtm': 'wrinkle_left_eye_Rz_Rtm',
    '주름_왼쪽눈가_Rp': 'wrinkle_left_eye_Rp',
    '주름_왼쪽눈가_Rv': 'wrinkle_left_eye_Rv',
    '주름_오른쪽눈가_Ra': 'wrinkle_right_eye_Ra',
    '주름_오른쪽눈가_Rq': 'wrinkle_right_eye_Rq',
    '주름_오른쪽눈가_Rmax': 'wrinkle_right_eye_Rmax',
    '주름_오른쪽눈가_R3z': 'wrinkle_right_eye_R3z',
    '주름_오른쪽눈가_Rt': 'wrinkle_right_eye_Rt',
    '주름_오른쪽눈가_Rz=Rtm': 'wrinkle_right_eye_Rz_Rtm',
    '주름_오른쪽눈가_Rp': 'wrinkle_right_eye_Rp',
    '주름_오른쪽눈가_Rv': 'wrinkle_right_eye_Rv',
    '스팟개수_정면': 'spot_count_front',
    '모공개수_오른쪽볼': 'pore_count_right_cheek',
    '모공개수_왼쪽볼': 'pore_count_left_cheek'
}

# 替换列名为英文
df.rename(columns=columns_mapping, inplace=True)

# 确认替换后的列名
print(df.columns)



# 1. 数据基本检查和预处理
print("数据的基本信息：")
print(df.info())
print("\n数据缺失值的统计：")
print(df.isnull().sum())

# 填充缺失值 (可根据需求选择其他方法)
df.fillna(df.mean(), inplace=True)

# 水分相关列
water_columns = ['moisture_forehead', 'moisture_right_cheek', 'moisture_left_cheek', 'moisture_chin']

# 所有弹性相关列
# 定义不同设备检测的名称,这是在检测弹力的时候的不同设备
devices = ['R0', 'R1', 'R2', 'R3', 'R4', 'R5', 'R6', 'R7', 'R8', 'R9', 'Q0', 'Q1', 'Q2', 'Q3']
# 定义部位
areas = ['chin', 'left_cheek', 'right_cheek', 'forehead']

# 仅选择 R0, R2, R5 三种设备对应的弹力列
selected_devices = ['R0', 'R2', 'R5']
elasticity_columns = []
for device in selected_devices:
    for area in areas:
        elasticity_columns.append(f'elasticity_{area}_{device}')

#1. 绘制水分值分布
plt.figure(figsize=(10, 6))
for col in water_columns:
    sns.kdeplot(df[col], label=col)
plt.title('Moisture content distribution')
plt.xlabel('Value')
plt.ylabel('distribution(%)')
plt.legend()
plt.show()
#
#2.  箱线图（查看水分值的分布和离群点）
plt.figure(figsize=(10, 6))
sns.boxplot(data=df[water_columns])
plt.title('Box plot of moisture values')
plt.xticks(range(len(water_columns)), water_columns)
plt.show()


#3.  绘制弹性值分布
# 遍历每个设备检测的结果并绘制图
for device in devices:
    # elasticity_columns = [f'elasticity_{area}_{device}' for area in areas]

    # 绘制每个设备的弹力值分布图
    plt.figure(figsize=(10, 6))
    for col in elasticity_columns:
        sns.kdeplot(df[col], label=col)
    plt.title(f'Elasticity Distribution for {device}')
    plt.xlabel('Value')
    plt.ylabel('Density (%)')
    plt.legend()
    plt.show()



# 4. 相关性分析和热力图
all_columns = water_columns + elasticity_columns
corr_matrix = df[all_columns].corr()

# 绘制相关性热力图
plt.figure(figsize=(12, 8))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt='.2f')
plt.title('Correlation Heatmap of Moisture Value and Elastic Value')
# 使 x 轴和 y 轴的标签倾斜45度
plt.xticks(rotation=45, ha='right')  # 使x轴标签倾斜45度
plt.yticks(rotation=45)  # 使y轴标签也倾斜45度
plt.show()


# 5. 主成分分析（PCA）: 用于减少维度，查看整体数据的变化趋势
scaler = StandardScaler()
scaled_data = scaler.fit_transform(df[water_columns + elasticity_columns])

pca = PCA(n_components=2)
pca_result = pca.fit_transform(scaled_data)

# 绘制主成分分析结果
plt.figure(figsize=(10, 6))
plt.scatter(pca_result[:, 0], pca_result[:, 1], c='blue')
plt.title('PCA result')
plt.xlabel('feature 1')
plt.ylabel('feature2')
plt.grid(True)
plt.show()

# 6. 聚类分析（K-Means）: 用于发现数据的潜在群体结构
from sklearn.cluster import KMeans

kmeans = KMeans(n_clusters=3)  # 假设分成3类，可以调整
kmeans.fit(scaled_data)
cluster_labels = kmeans.labels_

# 绘制聚类结果
plt.figure(figsize=(10, 6))
plt.scatter(pca_result[:, 0], pca_result[:, 1], c=cluster_labels, cmap='viridis')
plt.title('K-Means 聚类结果')
plt.xlabel('主成分1')
plt.ylabel('主成分2')
plt.grid(True)
plt.show()

# 7. 输出描述性统计数据
water_stats = df[water_columns].describe()
elasticity_stats = df[elasticity_columns].describe()

# 输出统计数据
print("水分值的统计数据：")
print(water_stats)

print("\n弹性值的统计数据：")
print(elasticity_stats)

