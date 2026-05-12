import pandas as pd
import os
import numpy as np
import cv2
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

class MedicalCLAHE:
    """针对医学影像的自适应直方图均衡化，增强视神经盘边缘特征"""
    def __init__(self, clip_limit=2.0, tile_grid_size=(8, 8)):
        self.clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)

    def __call__(self, img):
        img_np = np.array(img)
        lab = cv2.cvtColor(img_np, cv2.COLOR_RGB2LAB)
        l, a, b = cv2.split(lab)
        l = self.clahe.apply(l)
        lab = cv2.merge((l, a, b))
        return Image.fromarray(cv2.cvtColor(lab, cv2.COLOR_LAB2RGB))

class G1020Dataset(Dataset):
    def __init__(self, csv_path, img_dir, transform=None):
        self.csv_path = csv_path
        self.img_dir = img_dir
        self.df = pd.read_csv(self.csv_path)
        self.transform = transform

        # --- 强制适配你提供的 CSV 截图列名 ---
        self.image_col = 'imageID'  # 对应截图第一列
        self.label_col = 'binaryLabels'  # 对应截图第二列




    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        img_name = str(self.df.iloc[idx][self.image_col])
        img_path = os.path.join(self.img_dir, img_name)
        label = int(self.df.iloc[idx][self.label_col])

        try:
            image = Image.open(img_path).convert('RGB')
        except Exception as e:
            print(f"警告: 无法读取 {img_path}, 错误: {e}")
            image = Image.new('RGB', (384, 384), (0, 0, 0))

        if self.transform:
            image = self.transform(image)
        return image, label

def get_g1020_transforms(img_size=384, is_train=True):
    if is_train:
        return transforms.Compose([
            MedicalCLAHE(),
            transforms.Resize((img_size, img_size)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(15),
            transforms.ColorJitter(brightness=0.1, contrast=0.1),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    else:
        return transforms.Compose([
            MedicalCLAHE(),
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])