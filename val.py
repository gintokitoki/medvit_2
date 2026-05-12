import os
import cv2
import numpy as np
from tqdm import tqdm
from pathlib import Path


def crop_val_dataset(base_dir, crop_size=512):
    """
    针对 REFUGE1 验证集进行 ROI 裁剪
    base_dir: /home/wyh/data2/REFUGE1/val
    """
    img_dir = os.path.join(base_dir, 'images')
    mask_dir = os.path.join(base_dir, 'mask')
    save_dir = os.path.join(base_dir, 'crop')

    # 自动创建 crop 文件夹
    os.makedirs(save_dir, exist_ok=True)

    # 获取目录下所有 jpg 图片
    img_names = [f for f in os.listdir(img_dir) if f.lower().endswith('.jpg')]

    if not img_names:
        print(f"!!! 错误: 在 {img_dir} 下未找到图片，请检查路径。")
        return

    print(f"--- 启动验证集预处理 | 共 {len(img_names)} 张 ---")

    for name in tqdm(img_names):
        file_stem = Path(name).stem

        img_path = os.path.join(img_dir, name)
        # 验证集的 Mask 通常也是 .bmp 格式
        mask_path = os.path.join(mask_dir, f"{file_stem}.bmp")

        if not os.path.exists(mask_path):
            # 如果找不到 .bmp，尝试寻找 .png (部分数据集 val 集格式可能不同)
            mask_path_alt = os.path.join(mask_dir, f"{file_stem}.png")
            if os.path.exists(mask_path_alt):
                mask_path = mask_path_alt
            else:
                print(f"\n[跳过] 找不到对应的 Mask: {file_stem}")
                continue

        # 读取
        img = cv2.imread(img_path)
        mask = cv2.imread(mask_path, 0)

        if img is None or mask is None:
            continue

        # 定位视盘中心 (REFUGE 标准：背景白，目标深色)
        # 视盘和视杯的像素值通常远小于 200
        binary = (mask < 200).astype(np.uint8)
        coords = np.column_stack(np.where(binary > 0))

        if len(coords) == 0:
            # 如果没找到目标，回退到图像中心
            y_c, x_c = img.shape[0] // 2, img.shape[1] // 2
        else:
            y_c, x_c = coords.mean(axis=0).astype(int)

        # 补齐与裁剪逻辑
        h, w = img.shape[:2]
        r = crop_size // 2

        pad_t = max(0, r - y_c)
        pad_b = max(0, (y_c + r) - h)
        pad_l = max(0, r - x_c)
        pad_r = max(0, (x_c + r) - w)

        img_padded = cv2.copyMakeBorder(img, pad_t, pad_b, pad_l, pad_r, cv2.BORDER_CONSTANT, value=[0, 0, 0])

        new_y_c = y_c + pad_t
        new_x_c = x_c + pad_l
        roi = img_padded[new_y_c - r: new_y_c + r, new_x_c - r: new_x_c + r]

        # 保存
        save_path = os.path.join(save_dir, f"{name}")  # 保持原名保存
        cv2.imwrite(save_path, roi)


if __name__ == "__main__":
    VAL_PATH = "/home/wyh/data2/REFUGE1/val"
    crop_val_dataset(VAL_PATH, crop_size=512)
    print("\n--- 验证集裁剪完成！路径: /home/wyh/data2/REFUGE1/val/crop ---")