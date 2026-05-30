import os
import json
import cv2
import numpy as np
from PIL import Image
from tqdm import tqdm
"""
这个代码是处理NIA数据集，只筛选F(正面角度的图片）获取左右脸颊的标签数据，并生成MDM, WDM的标签
"""

# ---------- 结构引导标签函数 ----------
import os
import json
import cv2
import numpy as np
from tqdm import tqdm

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
    return cv2.drawContours(result, contours_sel, -1, (255, 255, 255), -1)

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


def process_nia_skingq_style_fixed(nia_img_root, nia_json_root, output_base, selected_parts=["05", "06"]):
    """
    参数说明：
    - nia_img_root : 图片根目录，例如 /data1/NIA/Validation/img/VS
    - nia_json_root : 标签根目录，例如 /data1/NIA/Validation/label/VL
    - output_base : 输出根目录
    """
    os.makedirs(os.path.join(output_base, "image"), exist_ok=True)
    os.makedirs(os.path.join(output_base, "gmdm"), exist_ok=True)
    os.makedirs(os.path.join(output_base, "gwdm"), exist_ok=True)
    os.makedirs(os.path.join(output_base, "label"), exist_ok=True)

    for device in tqdm(os.listdir(nia_img_root), desc="Devices"):
        device_img_path = os.path.join(nia_img_root, device)
        device_json_path = os.path.join(nia_json_root, device)

        for person in os.listdir(device_img_path):
            person_img_path = os.path.join(device_img_path, person)
            person_json_path = os.path.join(device_json_path, person)

            for img_file in os.listdir(person_img_path):
                if not img_file.lower().endswith(('.jpg', '.png', '.jpeg')):
                    continue

                img_id = os.path.splitext(img_file)[0]  # 例如 0013_03_F
                split_list = img_id.split("_")
                if len(split_list) < 3:
                    continue  # 文件名格式异常
                angle_flag = split_list[2].upper()
                if angle_flag != "F":
                    continue  # 只保留F角度

                full_img_path = os.path.join(person_img_path, img_file)
                img = cv2.imread(full_img_path)
                if img is None:
                    print(f"读取失败: {full_img_path}")
                    continue

                h, w, _ = img.shape
                pore_mask_full = np.zeros((h, w, 3), dtype=np.uint8)
                wrinkle_mask_full = np.zeros((h, w, 3), dtype=np.uint8)

                for part_num in selected_parts:
                    json_file = f"{img_id}_{part_num}.json"
                    json_path = os.path.join(person_json_path, json_file)
                    if not os.path.exists(json_path):
                        continue

                    with open(json_path, "r", encoding="utf-8") as f:
                        meta = json.load(f)
                    bbox = meta["images"].get("bbox", None)
                    if not bbox or len(bbox) != 4:
                        continue

                    x_min, y_min, x_max, y_max = map(int, bbox)
                    cropped_block = img[y_min:y_max, x_min:x_max]

                    pore_mask = detect_pores(cropped_block)
                    wrinkle_mask = detect_wrinkles_block(cropped_block)

                    pore_mask_full[y_min:y_max, x_min:x_max] = np.maximum(pore_mask_full[y_min:y_max, x_min:x_max], pore_mask)
                    wrinkle_mask_full[y_min:y_max, x_min:x_max] = np.maximum(wrinkle_mask_full[y_min:y_max, x_min:x_max], wrinkle_mask)

                save_name = f"{device}_{img_id}.png"  # 删除冗余 person

                cv2.imwrite(os.path.join(output_base, "image", save_name), img)  # 直接保存BGR原图
                cv2.imwrite(os.path.join(output_base, "gmdm", save_name), pore_mask_full)
                cv2.imwrite(os.path.join(output_base, "gwdm", save_name), wrinkle_mask_full)

                label_out = {"meta_list": []}
                for part_num in selected_parts:
                    json_file = f"{img_id}_{part_num}.json"
                    json_path = os.path.join(person_json_path, json_file)
                    if not os.path.exists(json_path):
                        continue
                    with open(json_path, "r", encoding="utf-8") as f:
                        meta = json.load(f)
                    label_out["meta_list"].append(meta)

                label_save_path = os.path.join(output_base, "label", save_name.replace(".png", ".json"))
                with open(label_save_path, "w", encoding="utf-8") as f:
                    json.dump(label_out, f, ensure_ascii=False, indent=2)

    print("数据保存完成，通道与文件名全部修正，严格对齐SKIN-Q")

# make train dataset
# nia_img_root = "/data1/NIA/Training/img/TS"
# nia_json_root = "/data1/NIA/Training/label/TL"
# output_base = "/data1/NIA_face_anonymize_train"

# make val dataset
nia_img_root = "/data1/NIA/Validation/img/VS"
nia_json_root = "/data1/NIA/Validation/label/VL"
output_base = "/data1/NIA_face_anonymize_val"

process_nia_skingq_style_fixed(nia_img_root, nia_json_root, output_base)