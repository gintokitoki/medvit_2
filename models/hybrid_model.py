import torch
import torch.nn as nn
from .MedViT import MedViT_small
from .fft_branch import FFTBranch


class HybridMedViT(nn.Module):
    def __init__(self, num_classes=2, pretrained_path=None):
        super(HybridMedViT, self).__init__()

        # 1. 空间分支 (MedViT)
        self.spatial_branch = MedViT_small()
        if pretrained_path:
            print(f">>> 正在加载 MedViT 预训练权重: {pretrained_path}")
            state_dict = torch.load(pretrained_path, map_location='cpu')
            self.spatial_branch.load_state_dict(state_dict, strict=False)

        # 2. 频率分支 (FFT)
        self.freq_branch = FFTBranch(output_dim=512)

        # 3. 自主门控机制 (Gating Mechanism)
        # 输入：1024 (MedViT) + 512 (FFT) = 1536
        self.total_dim = 1024 + 512

        self.gate = nn.Sequential(
            nn.Linear(self.total_dim, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 2),  # 输出两个权重比例
            nn.Softmax(dim=1)
        )

        # 4. 最终分类器
        self.classifier = nn.Sequential(
            nn.Linear(self.total_dim, 512),
            nn.BatchNorm1d(512),
            nn.Hardswish(),
            nn.Dropout(0.4),  # 加大 Dropout 应对 400 张小样本
            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        # 提取双路特征
        feat_s = self.spatial_branch(x, return_feat=True)  # [B, 1024]
        feat_f = self.freq_branch(x)  # [B, 512]

        # 原始特征拼接用于门控决策
        combined_raw = torch.cat((feat_s, feat_f), dim=1)  # [B, 1536]

        # 门控：模型自动学习当前图片该多看哪一边
        gate_weights = self.gate(combined_raw)  # [B, 2]

        # 应用权重
        feat_s_weighted = feat_s * gate_weights[:, 0:1]
        feat_f_weighted = feat_f * gate_weights[:, 1:2]

        # 二次拼接作为最终分类特征
        final_feat = torch.cat((feat_s_weighted, feat_f_weighted), dim=1)

        return self.classifier(final_feat)