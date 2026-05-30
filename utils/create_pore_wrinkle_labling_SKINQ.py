import os
import json
import cv2
import numpy as np
from PIL import Image
from matplotlib import pyplot as plt


"""
    这个代码是用于生成结构引导的标签，包括MDM，WDM
"""

# -------- 毛孔检测函数 --------
def detect_pores(image_block):
    gray = cv2.cvtColor(image_block, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=10.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)
    blur = cv2.GaussianBlur(enhanced, (3, 3), 0)
    thresh = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                   cv2.THRESH_BINARY_INV, 21, 5)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    morphed = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
    morphed = cv2.morphologyEx(morphed, cv2.MORPH_CLOSE, kernel)
    contours, _ = cv2.findContours(morphed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours_sel = [cnt for cnt in contours if 20 < cv2.contourArea(cnt) < 500]
    result = np.zeros_like(image_block)
    return cv2.drawContours(result, contours_sel, -1, (255, 255, 255), 1), len(contours_sel)

# -------- 皱纹检测函数 --------
def detect_wrinkles_block(block_bgr):
    lab = cv2.cvtColor(block_bgr, cv2.COLOR_BGR2LAB)
    l = lab[:, :, 0]
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    l_enhanced = clahe.apply(l)

    def adjust_gamma(image, gamma=1.6):
        invGamma = 1.0 / gamma
        table = np.array([(i / 255.0) ** invGamma * 255 for i in np.arange(256)]).astype("uint8")
        return cv2.LUT(image, table)

    gamma_img = adjust_gamma(l_enhanced, gamma=0.5)
    blurred = cv2.GaussianBlur(gamma_img, (5, 5), 0)
    highpass = cv2.addWeighted(gamma_img, 1.5, blurred, -0.5, 0)
    _, binary = cv2.threshold(highpass, 40, 255, cv2.THRESH_BINARY_INV)
    return cv2.cvtColor(binary, cv2.COLOR_GRAY2BGR)

# -------- 主处理函数 --------
def visualize_and_detect(json_file_path, image_folder_path,show=False):
    with open(json_file_path, 'r') as file:
        data = json.load(file)

    image_name = os.path.basename(data["imagePath"])
    image_path = os.path.join(image_folder_path, image_name)

    image = np.array(Image.open(image_path))
    image = np.rot90(image, k=-1)
    h, w, c = image.shape

    pore_overlay = np.zeros((h, w, 3), dtype=np.uint8)
    wrinkle_overlay = np.zeros((h, w, 3), dtype=np.uint8)

    for shape in data["shapes"]:
        points = shape["points"]
        x_min, y_min = int(min(points[0][0], points[1][0])), int(min(points[0][1], points[1][1]))
        x_max, y_max = int(max(points[0][0], points[1][0])), int(max(points[0][1], points[1][1]))

        block = image[y_min:y_max, x_min:x_max]
        if block.size == 0:
            continue

        block_bgr = cv2.cvtColor(block, cv2.COLOR_RGB2BGR)

        # 毛孔检测
        pore_result, _ = detect_pores(block_bgr)
        pore_overlay[y_min:y_max, x_min:x_max] = np.maximum(
            pore_overlay[y_min:y_max, x_min:x_max], pore_result)

        # 皱纹检测
        wrinkle_result = detect_wrinkles_block(block_bgr)
        wrinkle_overlay[y_min:y_max, x_min:x_max] = np.maximum(
            wrinkle_overlay[y_min:y_max, x_min:x_max], wrinkle_result)

    # === 保存mask ===
    pore_out_path = os.path.join(pore_mask_dir, image_name)
    wrinkle_out_path = os.path.join(wrinkle_mask_dir, image_name)
    cv2.imwrite(pore_out_path, cv2.cvtColor(pore_overlay, cv2.COLOR_RGB2BGR))
    cv2.imwrite(wrinkle_out_path, cv2.cvtColor(wrinkle_overlay, cv2.COLOR_RGB2BGR))

    # === 可视化（选用） ===
    if show:
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        axes[0].imshow(image)
        axes[0].set_title("Original Image")
        axes[1].imshow(pore_overlay)
        axes[1].set_title("Pore Detection")
        axes[2].imshow(wrinkle_overlay)
        axes[2].set_title("Wrinkle Detection")
        for ax in axes:
            ax.axis('off')
        plt.tight_layout()
        plt.show()



# 设置路径
json_folder_path = "E:/SKIN_DATA/SKIN_Q/03/label"
image_folder_path = "E:/SKIN_DATA/SKIN_Q/03/image"
base_dir = os.path.dirname(image_folder_path)
pore_mask_dir = os.path.join(base_dir, "gmdm")
wrinkle_mask_dir = os.path.join(base_dir, "gwdm")
os.makedirs(pore_mask_dir, exist_ok=True)
os.makedirs(wrinkle_mask_dir, exist_ok=True)

# -------- 批量处理所有JSON --------
for filename in os.listdir(json_folder_path):
    if filename.endswith('.json'):
        json_file_path = os.path.join(json_folder_path, filename)
        visualize_and_detect(json_file_path, image_folder_path, show = False)
