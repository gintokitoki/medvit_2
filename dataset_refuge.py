import os
import torch
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as transforms
import numpy as np


class RefugeCropDataset(Dataset):
    def __init__(self, root_dir, mode='train', augment_factor=1):
        """
        root_dir: 数据集根目录 (e.g., /home/wyh/data2/REFUGE1/train)
        mode: 'train' 或 'val'
        augment_factor: 扩充倍数，用于人为增加 epoch 内的样本迭代次数
        """
        # 根据你的需求，训练集在 test/crop (假设为实验路径)，验证在 val/crop
        self.img_dir = os.path.join(root_dir, 'crop')

        if not os.path.exists(self.img_dir):
            print(f"警告: 目录不存在 {self.img_dir}")
            self.img_names = []
        else:
            self.img_names = [f for f in os.listdir(self.img_dir) if f.lower().endswith('.jpg')]
        
        self.augment_factor = augment_factor

        # 标签解析：g 开头为 1，n 开头为 0
        self.labels = [1 if n.lower().startswith('g') else 0 for n in self.img_names]

        # 医学图像专用增强：由于视盘是圆形的，旋转和翻转最为有效
        if mode == 'train':
            self.transform = transforms.Compose([
                transforms.Resize((384, 384)),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomVerticalFlip(p=0.5),
                transforms.RandomRotation(degrees=360),  # 360度全旋转
                transforms.ColorJitter(brightness=0.1, contrast=0.1),  # 轻微亮度抖动
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])
        else:
            self.transform = transforms.Compose([
                transforms.Resize((384, 384)),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])

    def __len__(self):
        return len(self.img_names) * self.augment_factor

    def __getitem__(self, idx):
        actual_idx = idx % len(self.img_names)
        img_path = os.path.join(self.img_dir, self.img_names[actual_idx])
        label = self.labels[actual_idx]

        try:
            image = Image.open(img_path).convert('RGB')
        except Exception as e:
            print(f"错误: 无法读取图片 {img_path}, {e}")
            # 返回一个全黑图作为占位，防止训练中断
            image = Image.new('RGB', (384, 384), (0, 0, 0))

        if self.transform:
            image = self.transform(image)

        return image, label

    def get_weights_for_balanced_sampling(self):
        """动态计算采样权重，处理不平衡数据集"""
        if not self.labels:
            return torch.DoubleTensor([])
            
        labels_np = np.array(self.labels)
        class_counts = np.bincount(labels_np) # 自动计算每个类别的数量
        
        # 计算权重：数量越少权重越高
        weights = 1.0 / torch.tensor(class_counts, dtype=torch.float)
        sample_weights = [weights[l] for l in self.labels]
        return torch.DoubleTensor(sample_weights)