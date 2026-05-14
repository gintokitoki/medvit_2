import os
from typing import List, Optional, Sequence, Tuple

import numpy as np
import torch
from PIL import Image
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset
import torchvision.transforms as transforms


def _resolve_crop_image_dir(root_dir):
    """支持 .../train 或已指向 .../train/crop。"""
    root_dir = os.path.normpath(root_dir)
    if os.path.basename(root_dir) == "crop":
        return root_dir
    return os.path.join(root_dir, "crop")


def stratified_train_val_indices(
    root_dir: str, val_size: int = 64, seed: int = 42
) -> Tuple[List[int], List[int]]:
    """
    在排序后的 JPG 列表上做分层划分，返回 (train 文件下标, val 文件下标)。
    val_size 约在 50–80 之间较常见；若样本过少则自动缩小验证集。
    """
    img_dir = _resolve_crop_image_dir(root_dir)
    if not os.path.isdir(img_dir):
        return [], []

    all_jpg = sorted(f for f in os.listdir(img_dir) if f.lower().endswith(".jpg"))
    n = len(all_jpg)
    if n == 0:
        return [], []

    labels = np.array(
        [1 if name.lower().startswith("g") else 0 for name in all_jpg], dtype=np.int64
    )
    indices = np.arange(n)
    # 至少留给训练集 2 张，且验证集不超过 n-1
    val_size = max(1, min(val_size, n - 2))

    try:
        train_idx, val_idx = train_test_split(
            indices, test_size=val_size, stratify=labels, random_state=seed
        )
    except ValueError:
        train_idx, val_idx = train_test_split(
            indices, test_size=val_size, random_state=seed
        )

    return sorted(train_idx.tolist()), sorted(val_idx.tolist())


class RefugeCropDataset(Dataset):
    def __init__(
        self,
        root_dir,
        mode="train",
        augment_factor=1,
        file_indices: Optional[Sequence[int]] = None,
    ):
        """
        root_dir: 含 JPG 的目录，或为其父目录（自动拼接 crop）
        mode: 'train' 或 'val'
        augment_factor: 扩充倍数，用于人为增加 epoch 内的样本迭代次数
        file_indices: 可选；对目录下排序后的 JPG 列表取下标子集（用于 train/val 划分）
        """
        self.img_dir = _resolve_crop_image_dir(root_dir)

        if not os.path.exists(self.img_dir):
            print(f"警告: 目录不存在 {self.img_dir}")
            self.img_names = []
        else:
            all_jpg = sorted(
                f for f in os.listdir(self.img_dir) if f.lower().endswith(".jpg")
            )
            if file_indices is not None:
                self.img_names = [all_jpg[i] for i in file_indices]
            else:
                self.img_names = all_jpg

        self.augment_factor = augment_factor

        # 标签解析：g 开头为 1，n 开头为 0
        self.labels = [1 if n.lower().startswith('g') else 0 for n in self.img_names]

        # 几何增强对两路都友好；弱化颜色抖动，减轻 FFT 分支上低频谱的剧烈抖动
        if mode == 'train':
            self.transform = transforms.Compose([
                transforms.Resize((512, 512)),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomVerticalFlip(p=0.5),
                transforms.RandomRotation(degrees=360),  # 360度全旋转
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])
        else:
            self.transform = transforms.Compose([
                transforms.Resize((512, 512)),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])

    def __len__(self):
        if not self.img_names:
            return 0
        return len(self.img_names) * self.augment_factor

    def __getitem__(self, idx):
        if not self.img_names:
            raise RuntimeError(f"目录下无可用 JPG 图像: {self.img_dir}")
        actual_idx = idx % len(self.img_names)
        img_path = os.path.join(self.img_dir, self.img_names[actual_idx])
        label = self.labels[actual_idx]

        try:
            image = Image.open(img_path).convert('RGB')
        except Exception as e:
            print(f"错误: 无法读取图片 {img_path}, {e}")
            # 返回一个全黑图作为占位，防止训练中断
            image = Image.new('RGB', (512, 512), (0, 0, 0))

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