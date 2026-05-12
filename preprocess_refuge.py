import os
import cv2
import numpy as np
from tqdm import tqdm
from pathlib import Path


def crop_refuge_roi(base_dir, split, crop_size=512):
    """
    base_dir: /home/wyh/data2/REFUGE1
    split: 'train', 'val', 或 'test'
    Image: .jpg
    Mask: .bmp
    """
    # 1. 严格定义路径
    img_dir = os.path.join(base_dir, split, 'images')
    mask_dir = os.path.join(base_dir, split, 'mask')
    save_dir = os.path.join(base_dir, split, 'crop')

    # 自动创建 crop 文件夹
    os.makedirs(save_dir, exist_ok=True)

    # 获取目录下所有 jpg 图片
    img_names = [f for f in os.listdir(img_dir) if f.lower().endswith('.jpg')]

    if not img_names:
        print(f"!!! 警告: 在 {img_dir} 下未找到 .jpg 图片，请检查路径。")
        return

    print(f"--- 启动 {split} 集处理 | 共 {len(img_names)} 张 | 模式: JPG -> BMP 匹配 ---")

    for name in tqdm(img_names):
        # 获取不带后缀的文件名，例如 'g0001'
        file_stem = Path(name).stem

        img_path = os.path.join(img_dir, name)
        # 强制指定寻找同名的 .bmp 文件
        mask_path = os.path.join(mask_dir, f"{file_stem}.bmp")

        # 检查 Mask 是否存在
        if not os.path.exists(mask_path):
            print(f"\n[错误] 找不到对应的 Mask: {mask_path}")
            continue

        # 读取图片
        img = cv2.imread(img_path)
        mask = cv2.imread(mask_path, 0)  # 灰度读取

        if img is None or mask is None:
            continue

        # 2. 定位视盘中心 (REFUGE Mask 背景为白255，目标为深色)
        # 寻找所有像素值小于 200 的点（视盘+视杯）
        binary = (mask < 200).astype(np.uint8)
        coords = np.column_stack(np.where(binary > 0))

        if len(coords) == 0:
            y_c, x_c = img.shape[0] // 2, img.shape[1] // 2
        else:
            y_c, x_c = coords.mean(axis=0).astype(int)

        # 3. 补齐与裁剪 (Padding Logic)
        h, w = img.shape[:2]
        r = crop_size // 2

        # 计算 Padding 边界
        pad_t = max(0, r - y_c)
        pad_b = max(0, (y_c + r) - h)
        pad_l = max(0, r - x_c)
        pad_r = max(0, (x_c + r) - w)

        # 补黑边
        img_padded = cv2.copyMakeBorder(img, pad_t, pad_b, pad_l, pad_r, cv2.BORDER_CONSTANT, value=[0, 0, 0])

        # 裁剪
        new_y_c = y_c + pad_t
        new_x_c = x_c + pad_l
        roi = img_padded[new_y_c - r: new_y_c + r, new_x_c - r: new_x_c + r]

        # 4. 保存为 jpg (保持与原图格式一致)
        save_path = os.path.join(save_dir, f"{file_stem}.jpg")
        cv2.imwrite(save_path, roi)


if __name__ == "__main__":
    # 使用你提供的绝对路径
    BASE_PATH = "/home/wyh/data2/REFUGE1"

    # 处理三个子集
    splits = ['train', 'val', 'test']
    for s in splits:
        # 检查子目录是否存在
        if os.path.exists(os.path.join(BASE_PATH, s)):
            crop_refuge_roi(BASE_PATH, s, crop_size=512)
        else:
            print(f"跳过不存在的目录: {s}")

    print("\n--- 预处理圆满完成！裁剪后的图片存放在各 split/crop 文件夹中 ---")