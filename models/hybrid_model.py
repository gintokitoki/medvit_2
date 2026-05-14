import torch
import torch.nn as nn
from .MedViT import MedViT_small
from .fft_branch import FFTBranch


class HybridMedViT(nn.Module):
    def __init__(self, num_classes=2, pretrained_path=None, input_res=512):
        super(HybridMedViT, self).__init__()

        # 1. 空间分支 (保持 MedViT 不变)
        self.spatial_branch = MedViT_small()
        if pretrained_path:
            print(f">>> 正在加载 MedViT 预训练权重: {pretrained_path}")
            state_dict = torch.load(pretrained_path, map_location='cpu')
            # 过滤掉不匹配的权重（如最后分类层）
            model_dict = self.spatial_branch.state_dict()
            state_dict = {k: v for k, v in state_dict.items() if k in model_dict and v.shape == model_dict[k].shape}
            self.spatial_branch.load_state_dict(state_dict, strict=False)

        # 2. 进化版频率分支
        self.freq_branch = FFTBranch(input_res=input_res, output_dim=512)

        # 3. 特征维度定义
        self.s_dim = 1024  # MedViT_small 输出维度
        self.f_dim = 512
        self.total_dim = self.s_dim + self.f_dim

        # 4. 进化版门控机制 (增加了一层深度以提高决策非线性)
        self.gate = nn.Sequential(
            nn.Linear(self.total_dim, 256),
            nn.LayerNorm(256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(256, 2),
            nn.Softmax(dim=1)
        )

        # 5. 最终分类器 (加入残差连接的思想)
        self.classifier = nn.Sequential(
            nn.Linear(self.total_dim, 1024),
            nn.BatchNorm1d(1024),
            nn.Hardswish(),
            nn.Dropout(0.4),

            nn.Linear(1024, 512),
            nn.BatchNorm1d(512),
            nn.Hardswish(),
            nn.Dropout(0.3),

            nn.Linear(512, num_classes)
        )

    def _effective_branch_weights(self, gate_weights, batch_size, device, dtype, force_ratio=None, min_ratio=None):
        """根据训练阶段得到实际作用于两路特征的权重 [w_s, w_f]（列向量）。"""
        if force_ratio is not None:
            w_f = torch.full((batch_size, 1), float(force_ratio), device=device, dtype=dtype)
            w_s = 1.0 - w_f
        elif min_ratio is not None:
            raw_w_f = gate_weights[:, 1:2]
            m = float(min_ratio)
            w_f = m + (1.0 - m) * raw_w_f
            w_s = 1.0 - w_f
        else:
            w_s = gate_weights[:, 0:1]
            w_f = gate_weights[:, 1:2]
        return w_s, w_f

    def forward(self, x, force_ratio=None, min_ratio=None):
        feat_s = self.spatial_branch(x, return_feat=True)
        feat_f = self.freq_branch(x)

        combined_raw = torch.cat((feat_s, feat_f), dim=1)
        gate_weights = self.gate(combined_raw)  # [B, 2] softmax -> [spatial, freq]

        w_s, w_f = self._effective_branch_weights(
            gate_weights,
            x.size(0),
            x.device,
            gate_weights.dtype,
            force_ratio=force_ratio,
            min_ratio=min_ratio,
        )

        feat_s_weighted = feat_s * w_s
        feat_f_weighted = feat_f * w_f

        final_feat = torch.cat((feat_s_weighted, feat_f_weighted), dim=1)
        logits = self.classifier(final_feat)
        return logits

    def get_gate_stats(self, x, force_ratio=None, min_ratio=None):
        """调试辅助：返回当前策略下实际用于加权的两路比例 [spatial, freq]，形状 [B, 2]。"""
        self.eval()
        with torch.no_grad():
            feat_s = self.spatial_branch(x, return_feat=True)
            feat_f = self.freq_branch(x)
            combined_raw = torch.cat((feat_s, feat_f), dim=1)
            gate_weights = self.gate(combined_raw)
            w_s, w_f = self._effective_branch_weights(
                gate_weights,
                x.size(0),
                x.device,
                gate_weights.dtype,
                force_ratio=force_ratio,
                min_ratio=min_ratio,
            )
            return torch.cat((w_s, w_f), dim=1)