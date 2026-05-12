import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, WeightedRandomSampler
import numpy as np
from sklearn.metrics import accuracy_score, recall_score, f1_score
import subprocess
import os
import sys

# --- 修复导入路径问题 ---
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.append(os.path.join(current_dir, 'models'))
# -----------------------

# 导入你之前的模型
from models.hybrid_model import HybridMedViT
from dataset_refuge import RefugeCropDataset


def get_idle_gpu():
    """自动选择空闲显存最大的 GPU"""
    try:
        command = "nvidia-smi --query-gpu=memory.free --format=csv,nounits,noheader"
        memory_free = subprocess.check_output(command.split()).decode('utf-8').strip().split('\n')
        memory_free = [int(x) for x in memory_free]
        idx = np.argmax(memory_free)
        print(f"检测到最空闲 GPU ID: {idx}, 剩余显存: {memory_free[idx]} MB")
        return torch.device(f"cuda:{idx}")
    except:
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def train():
    # 1. 硬件与环境配置
    device = get_idle_gpu()
    
    # --- 核心修改：只定义一个数据源 ---
    data_root = "/home/wyh/data2/REFUGE1/train"
    pretrained_weights = "/home/wyh/project/medvit_2/weights/MedViT_small_im1k.pth"

    # 1. 训练 Loader：用于参数更新 (开启增强 + 采样器)
    train_dataset = RefugeCropDataset(data_root, mode='train', augment_factor=3)
    sampler = WeightedRandomSampler(
        train_dataset.get_weights_for_balanced_sampling(),
        num_samples=len(train_dataset),
        replacement=True
    )
    train_loader = DataLoader(train_dataset, batch_size=8, sampler=sampler, num_workers=4)

    # 2. 评估 Loader：用于展示当前模型对训练集的拟合情况 (关闭增强)
    # 我们依然保留这个 loader，但它读取的是和上面一样的图，只是为了客观评估当前 epoch 的水平
    eval_dataset = RefugeCropDataset(data_root, mode='val') # mode='val' 会关闭旋转抖动，看原图拟合度
    eval_loader = DataLoader(eval_dataset, batch_size=8, shuffle=False, num_workers=4)
    # --------------------------------

    # 3. 模型与不平衡损失函数
    class_weights = torch.tensor([1.0, 9.0]).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)

    # 修复路径：使用绝对路径加载权重
    model = HybridMedViT(num_classes=2, pretrained_path=pretrained_weights).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=1e-5, weight_decay=1e-2)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=30)

    # 4. 训练循环
    epochs = 50
    best_f1 = 0.0

    print("--- 开始模型训练 (REFUGE-1) ---")
    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        for imgs, labels in train_loader:
            imgs, labels = imgs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(imgs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        scheduler.step()

        # 验证监控：改为评估训练集拟合情况
        acc, rec, f1 = evaluate(model, eval_loader, device)
        print(f"Epoch [{epoch + 1}/{epochs}] | Loss: {train_loss / len(train_loader):.4f} | "
              f"Fit Acc: {acc:.4f} | Fit Recall: {rec:.4f} | Fit F1: {f1:.4f}")

        if f1 > best_f1:
            best_f1 = f1
            torch.save(model.state_dict(), "best_refuge_model.pth")

    print("\n--- 最终模型拟合报告 ---")
    final_eval(model, "best_refuge_model.pth", eval_loader, device)


def evaluate(model, loader, device):
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for imgs, labels in loader:
            imgs = imgs.to(device)
            outputs = model(imgs)
            preds = torch.argmax(outputs, dim=1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(labels.numpy())

    acc = accuracy_score(all_labels, all_preds)
    rec = recall_score(all_labels, all_preds, zero_division=0)
    f1 = f1_score(all_labels, all_preds, zero_division=0)
    return acc, rec, f1


def final_eval(model, ckpt_path, loader, device):
    model.load_state_dict(torch.load(ckpt_path))
    acc, rec, f1 = evaluate(model, loader, device)
    print("=" * 30)
    print(f"测试集准确率: {acc * 100:.2f}%")
    print(f"测试集召回率: {rec * 100:.2f}%")
    print(f"测试集 F1 分数: {f1 * 100:.2f}%")
    print("=" * 30)


if __name__ == "__main__":
    train()