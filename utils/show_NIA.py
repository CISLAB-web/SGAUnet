import os
import json
import cv2
import matplotlib.pyplot as plt

"""
可视化处理后的NIA的数据集
"""

def visualize_with_bbox_moisture(processed_dir, num_samples=5):
    """
    processed_dir: 结构化输出目录
    num_samples: 展示多少个样本
    """
    image_dir = os.path.join(processed_dir, "image")
    gmdm_dir = os.path.join(processed_dir, "gmdm")
    gwdm_dir = os.path.join(processed_dir, "gwdm")
    label_dir = os.path.join(processed_dir, "label")

    all_files = [f for f in os.listdir(image_dir) if f.endswith(".png")]
    if len(all_files) == 0:
        print("未找到任何图片，请检查目录")
        return

    num_samples = min(num_samples, len(all_files))
    selected = all_files[:num_samples]

    for file_name in selected:
        img_path = os.path.join(image_dir, file_name)
        gmdm_path = os.path.join(gmdm_dir, file_name)
        gwdm_path = os.path.join(gwdm_dir, file_name)
        label_path = os.path.join(label_dir, file_name.replace(".png", ".json"))

        img = cv2.imread(img_path)
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        gmdm = cv2.imread(gmdm_path)
        gwdm = cv2.imread(gwdm_path)

        with open(label_path, "r", encoding="utf-8") as f:
            meta_all = json.load(f)

        display_img = img_rgb.copy()

        info_texts = []
        for meta in meta_all["meta_list"]:
            info = meta.get("info", {})
            ann = meta.get("annotations", {})
            equip = meta.get("equipment", {})
            bbox = meta.get("images", {}).get("bbox", None)
            part = meta["images"].get("facepart", "")

            gender = info.get("gender", "-")
            age = info.get("age", "-")
            skin_type = info.get("skin_type", "-")

            if part == 5:
                region = "L_cheek"
                pore_grade = ann.get("l_cheek_pore", "-")
                pigm_grade = ann.get("l_cheek_pigmentation", "-")
                moisture = equip.get("l_cheek_moisture", "-")
                elasticity_r0 = equip.get("l_cheek_elasticity_R0", "-")
            elif part == 6:
                region = "R_cheek"
                pore_grade = ann.get("r_cheek_pore", "-")
                pigm_grade = ann.get("r_cheek_pigmentation", "-")
                moisture = equip.get("r_cheek_moisture", "-")
                elasticity_r0 = equip.get("r_cheek_elasticity_R0", "-")
            else:
                region = "Unknown"
                pore_grade = pigm_grade = moisture = elasticity_r0 = "-"

            if bbox and len(bbox) == 4:
                x_min, y_min, x_max, y_max = map(int, bbox)
                cv2.rectangle(display_img, (x_min, y_min), (x_max, y_max), (255, 0, 0), 2)
                label = f"{region} | Pore:{pore_grade} | Pigment:{pigm_grade} | Moisture:{moisture} | R0:{elasticity_r0}"
                cv2.putText(display_img, label, (x_min, y_min - 10), cv2.FONT_HERSHEY_SIMPLEX,
                            0.5, (255, 0, 0), 1, cv2.LINE_AA)

            info_text = f"Part:{part} | Gender:{gender} | Age:{age} | SkinType:{skin_type} | PoreGrade:{pore_grade} | Pigment:{pigm_grade} | Moisture:{moisture} | R0:{elasticity_r0}"
            info_texts.append(info_text)

        # 绘图
        plt.figure(figsize=(18, 6))
        plt.subplot(1, 3, 1)
        plt.imshow(display_img)
        plt.title(f"Original + BBox: {file_name}")
        plt.axis('off')

        plt.subplot(1, 3, 2)
        plt.imshow(gmdm)
        plt.title("Pore Mask (MDM)")
        plt.axis('off')

        plt.subplot(1, 3, 3)
        plt.imshow(gwdm)
        plt.title("Wrinkle Mask (WDM)")
        plt.axis('off')

        plt.suptitle("\n".join(info_texts), fontsize=10)
        plt.tight_layout()
        plt.show()


processed_dir = "/data1/NIA_open/NIA_face_anonymize_train"
visualize_with_bbox_moisture(processed_dir, num_samples=10)