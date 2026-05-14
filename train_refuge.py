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
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.append(os.path.join(PROJECT_ROOT, "models"))
# -----------------------

# 导入你之前的模型
from models.hybrid_model import HybridMedViT
from dataset_refuge import RefugeCropDataset, stratified_train_val_indices


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
    
    # 数据与权重（Linux 服务器）；JPG 位于 .../train/crop
    data_root = "/home/wyh/data2/REFUGE1/train/crop"
    pretrained_weights = os.path.join(PROJECT_ROOT, "weights", "MedViT_small_im1k.pth")
    ckpt_path = os.path.join(PROJECT_ROOT, "best_refuge_model.pth")

    train_idx, val_idx = stratified_train_val_indices(data_root, val_size=64, seed=42)
    if not train_idx:
        raise RuntimeError(f"训练集为空，请检查路径是否存在 JPG: {data_root}")
    if not val_idx:
        raise RuntimeError("验证集为空（样本过少无法划分），请增加数据或减小 val_size")

    print(f"数据划分: 训练 {len(train_idx)} 张 | 验证 {len(val_idx)} 张（分层抽样）")

    train_dataset = RefugeCropDataset(
        data_root, mode="train", augment_factor=3, file_indices=train_idx
    )
    val_dataset = RefugeCropDataset(
        data_root, mode="val", augment_factor=1, file_indices=val_idx
    )

    sampler = WeightedRandomSampler(
        train_dataset.get_weights_for_balanced_sampling(),
        num_samples=len(train_dataset),
        replacement=True,
    )
    train_loader = DataLoader(train_dataset, batch_size=8, sampler=sampler, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False, num_workers=4)

    # 已有 WeightedRandomSampler 时，CE 类权重不宜再与 1:9 同量级叠加，略抬少数类即可
    class_weights = torch.tensor([1.0, 2.0]).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)

    model = HybridMedViT(
        num_classes=2, pretrained_path=pretrained_weights, input_res=512
    ).to(device)

    spatial_params = list(model.spatial_branch.parameters())
    new_params = (
        list(model.freq_branch.parameters())
        + list(model.gate.parameters())
        + list(model.classifier.parameters())
    )
    optimizer = optim.AdamW(
        [
            {"params": spatial_params, "lr": 1e-5},
            {"params": new_params, "lr": 1e-4},
        ],
        weight_decay=1e-2,
    )
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=30)

    # 4. 训练循环
    epochs = 50
    best_f1 = 0.0

    print("--- 开始模型训练 (REFUGE-1) ---")
    for epoch in range(epochs):
        # 三阶段门控：1–10 轮强制 50/50；11–30 轮 FFT 低保 0.3；31+ 自由竞争
        if epoch < 10:
            strategy = {"force_ratio": 0.5}
            mode_str = "FORCED (0.5)"
        elif epoch < 30:
            strategy = {"min_ratio": 0.3}
            mode_str = "PROTECTED (min 0.3)"
        else:
            strategy = {}
            mode_str = "FREE"

        model.train()
        train_loss = 0.0
        for imgs, labels in train_loader:
            imgs, labels = imgs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(imgs, **strategy)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        scheduler.step()

        acc, rec, f1 = evaluate_with_strategy(model, val_loader, device, **strategy)
        print(
            f"Epoch [{epoch + 1}/{epochs}] Mode: {mode_str} | "
            f"Loss: {train_loss / len(train_loader):.4f} | "
            f"Val Acc: {acc:.4f} | Val Recall: {rec:.4f} | Val F1: {f1:.4f}"
        )

        if epoch % 5 == 0:
            test_img, _ = next(iter(val_loader))
            weights = model.get_gate_stats(test_img.to(device), **strategy)
            w_mean = weights.mean(dim=0).cpu().numpy()
            print(
                f"  Gating mean [spatial, freq]: [{w_mean[0]:.4f}, {w_mean[1]:.4f}] "
                f"(若 freq≈0 则频域支路未参与)"
            )

        if f1 > best_f1:
            best_f1 = f1
            torch.save(model.state_dict(), ckpt_path)

    print("\n--- 验证集上的最佳模型评估（FREE 门控，与部署一致）---")
    final_eval(model, ckpt_path, val_loader, device)


def evaluate_with_strategy(model, loader, device, **strategy):
    """验证时与训练使用相同的门控策略（force_ratio / min_ratio）。"""
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for imgs, labels in loader:
            imgs = imgs.to(device)
            outputs = model(imgs, **strategy)
            preds = torch.argmax(outputs, dim=1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(labels.numpy())

    acc = accuracy_score(all_labels, all_preds)
    rec = recall_score(all_labels, all_preds, zero_division=0)
    f1 = f1_score(all_labels, all_preds, zero_division=0)
    return acc, rec, f1


def final_eval(model, ckpt_path, loader, device):
    model.load_state_dict(torch.load(ckpt_path, map_location=device))
    acc, rec, f1 = evaluate_with_strategy(model, loader, device)
    print("=" * 30)
    print(f"验证集准确率: {acc * 100:.2f}%")
    print(f"验证集召回率: {rec * 100:.2f}%")
    print(f"验证集 F1 分数: {f1 * 100:.2f}%")
    print("=" * 30)


if __name__ == "__main__":
    train()