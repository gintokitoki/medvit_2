import os
import torch
import torch.nn as nn
import subprocess
import numpy as np
import time
from sklearn.metrics import recall_score, f1_score, accuracy_score
from torch.utils.data import DataLoader, Subset, WeightedRandomSampler
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.cuda.amp import autocast, GradScaler

# 导入自定义组件
from models.hybrid_model import HybridMedViT
from dataset import G1020Dataset, get_g1020_transforms

# --- 全局路径配置 ---
CSV_PATH = "/home/wyh/data2/G1020/G1020.csv"
IMG_DIR = "/home/wyh/data2/G1020/Images"
PRETRAINED_WEIGHTS = "weights/MedViT_small_im1k.pth"

# 自动生成独立运行目录，确保互不干扰
TIMESTAMP = time.strftime("%Y%m%d_%H%M%S")
RUN_DIR = f"runs/exp_{TIMESTAMP}"
os.makedirs(RUN_DIR, exist_ok=True)


class FocalLoss(nn.Module):
    def __init__(self, alpha=0.8, gamma=2.0):  # alpha=0.8 保持对漏诊的高压态势，但给了 Accuracy 一定的呼吸空间
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.ce = nn.CrossEntropyLoss(reduction='none', label_smoothing=0.1)

    def forward(self, logits, labels):
        ce_loss = self.ce(logits, labels)
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss
        return focal_loss.mean()


def get_best_gpu():
    """获取显存最空闲的 GPU 并返回其 ID"""
    try:
        res = subprocess.check_output(['nvidia-smi', '--query-gpu=memory.free', '--format=csv,nounits,noheader'],
                                      encoding='utf-8')
        free_mem = [int(x.strip()) for x in res.strip().split('\n')]
        best_gpu = np.argmax(free_mem)
        print(f"[GPU 状态] 各卡空闲显存: {[f'{m}MB' for m in free_mem]} → 选中 CUDA:{best_gpu}")
        return best_gpu
    except:
        return 0


def train():
    # 0. 确定独立 GPU 资源
    gpu_id = get_best_gpu()
    device = torch.device(f"cuda:{gpu_id}")
    torch.cuda.empty_cache()  # 清理残留显存
    print(f"--- 独立训练启动 | 运行目录: {RUN_DIR} | 使用显卡: CUDA:{gpu_id} ---")

    # 1. 实例化模型
    model = HybridMedViT(num_classes=2, pretrained_path=PRETRAINED_WEIGHTS)

    # 2. Stage 1: 冻结策略
    for name, param in model.named_parameters():
        if "spatial_branch" in name:
            param.requires_grad = False

    model.to(device)

    # 3. 数据准备
    ds_train_full = G1020Dataset(CSV_PATH, IMG_DIR, get_g1020_transforms(384, True))
    ds_val_full = G1020Dataset(CSV_PATH, IMG_DIR, get_g1020_transforms(384, False))

    indices = np.arange(len(ds_train_full))
    np.random.seed(int(time.time()))  # 确保每次启动的 shuffle 不同
    np.random.shuffle(indices)
    split = int(0.8 * len(indices))
    train_idx = indices[:split]
    val_idx = indices[split:]

    # --- 修改点 1: 采样权重计算 ---
    # 提取当前训练索引对应的标签
    labels = [int(ds_train_full.df.iloc[i]['binaryLabels']) for i in train_idx]
    # 计算各类别数量 [健康数, 病人数]
    class_counts = np.bincount(labels)
    # 计算权重：使用平方根倒数来平滑权重，防止过拟合少数类噪声
    weights = 1. / np.sqrt(class_counts)
    # 为训练集中的每一个样本分配权重
    sample_weights = np.array([weights[t] for t in labels])
    sample_weights = torch.from_numpy(sample_weights).double()

    # 定义采样器
    sampler = WeightedRandomSampler(sample_weights, len(sample_weights), replacement=True)

    # 修改 DataLoader: 传入 sampler，注意此时 shuffle 必须为 False
    train_loader = DataLoader(
        Subset(ds_train_full, train_idx),
        batch_size=8,
        sampler=sampler, # 关键：使用采样器
        num_workers=4,
        pin_memory=True
    )
    val_loader = DataLoader(Subset(ds_val_full, val_idx), batch_size=8, shuffle=False, num_workers=4)

    # 4. 优化器与损失
    # --- 修改点 2: 适当放权 ---
    # alpha=0.8 依然保持对漏诊的高压态势，但给了 Accuracy 一定的呼吸空间
    criterion = FocalLoss(alpha=0.8, gamma=2.0).to(device)
    optimizer = AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=1e-4)
    scheduler = ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=3)

    # 记录最佳指标
    # --- 修改点 3: 监控变量初始化 ---
    best_acc = 0.0
    best_recall = 0.0
    curr_recall = 0.0 # 初始化当前召回率
    early_stop_counter = 0
    PATIENCE = 10  # 提高耐受力，连续 10 轮 Recall 不涨就停
    MAX_EPOCHS = 40

    # 引入混合精度训练以节省显存
    scaler = GradScaler()

    for epoch in range(MAX_EPOCHS):
        # --- 修改点 4: 提前解冻 ---
        if epoch == 8: # 缩短冻结期，从 10 改为 8
            print(">>> Stage 2: 启动全量微调，极低学习率保护主干...")
            for param in model.parameters():
                param.requires_grad = True
            
            optimizer = AdamW([
                {'params': model.freq_branch.parameters(), 'lr': 1e-4},   # FFT 分支保持高灵敏度
                {'params': model.classifier.parameters(), 'lr': 1e-4},    # 分类头继续收敛
                {'params': model.spatial_branch.parameters(), 'lr': 2e-6} # 主干极低速微调，防止抹除预训练特征
            ])
            # 关键补充：同步更新 scheduler，绑定到新的 optimizer
            scheduler = ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=3)
            
            # --- 专项优化：给微调阶段重置寿命 ---
            early_stop_counter = 0 
            best_recall = curr_recall # 以当前水平作为微调的起点

        model.train()
        train_loss = 0
        for imgs, labels in train_loader:
            imgs, labels = imgs.to(device), labels.to(device)

            # 使用 autocast 开启混合精度
            with autocast():
                preds = model(imgs)
                loss = criterion(preds, labels)

            optimizer.zero_grad()
            # 使用 scaler 缩放梯度
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            train_loss += loss.item()

        # 显存回收
        torch.cuda.empty_cache()

        # 验证
        model.eval()
        all_true, all_pred = [], []
        with torch.no_grad():
            for imgs, labels in val_loader:
                imgs, labels = imgs.to(device), labels.to(device)
                out = model(imgs)
                all_pred.extend(torch.argmax(out, dim=1).cpu().numpy())
                all_true.extend(labels.cpu().numpy())

        # 计算指标
        curr_acc = accuracy_score(all_true, all_pred)
        curr_recall = recall_score(all_true, all_pred)
        curr_f1 = f1_score(all_true, all_pred)
        pred_counts = np.bincount(all_pred, minlength=2)

        print(
            f"Epoch {epoch + 1:02d} | 预测分布: [健康:{pred_counts[0]}, 病人:{pred_counts[1]}] | "
            f"Loss: {train_loss / len(train_loader):.4f} | Acc: {curr_acc:.4f} | Recall: {curr_recall:.4f} | F1: {curr_f1:.4f}")

        # --- 修改点 4: 引入验证集“双重保护” ---
        # scheduler 盯着 F1 看（兼顾 Acc 和 Recall），但保存模型依然以 Recall 为主
        scheduler.step(curr_f1)

        # --- 循环内部末尾的逻辑 ---
        if curr_recall > best_recall:
            best_recall = curr_recall
            early_stop_counter = 0  # 重置计数器
            # 执行保存逻辑...
            torch.save(model.state_dict(), os.path.join(RUN_DIR, "best_recall_model.pth"))
            print(f"*** 召回率提升！已保存最佳 Recall 权重: Recall={curr_recall:.4f} ***")
            
            # 同时检查是否也是最佳 Acc，若是则更新 best_acc
            if curr_acc > best_acc:
                best_acc = curr_acc
        else:
            early_stop_counter += 1
            print(f"--- Recall 未提升计数: {early_stop_counter}/{PATIENCE} ---")

        if early_stop_counter >= PATIENCE:
            print(f"!!! 检测到模型性能退化或停滞，触发早停，当前 Epoch: {epoch + 1} !!!")
            break


if __name__ == "__main__":
    train()