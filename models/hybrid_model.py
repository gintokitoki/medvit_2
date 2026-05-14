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
            state_dict = torch.load(pretrained_path, map_location="cpu")
            model_dict = self.spatial_branch.state_dict()
            state_dict = {
                k: v
                for k, v in state_dict.items()
                if k in model_dict and v.shape == model_dict[k].shape
            }
            self.spatial_branch.load_state_dict(state_dict, strict=False)

        # 2. 频率分支：与空间分支同维 1024，便于加权和融合
        self.freq_branch = FFTBranch(input_res=input_res, output_dim=1024)

        self.s_dim = 1024
        self.f_dim = 1024
        # 门控仍看两路原始拼接，决策 w_s / w_f
        self.gate_in_dim = self.s_dim + self.f_dim

        # 融合前分别 LayerNorm，缓解两路尺度/语义不一致，再在同一维空间做加权和
        self.feat_norm_s = nn.LayerNorm(self.s_dim)
        self.feat_norm_f = nn.LayerNorm(self.f_dim)

        # 3. 门控
        self.gate = nn.Sequential(
            nn.Linear(self.gate_in_dim, 256),
            nn.LayerNorm(256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(256, 2),
            nn.Softmax(dim=1),
        )

        # 4. 分类器：输入为融合后的 1024 维
        self.classifier = nn.Sequential(
            nn.Linear(self.s_dim, 512),
            nn.BatchNorm1d(512),
            nn.Hardswish(),
            nn.Dropout(0.35),
            nn.Linear(512, num_classes),
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

    def forward(self, x, force_ratio=None, min_ratio=None, return_branch_feats=False):
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

        s_n = self.feat_norm_s(feat_s)
        f_n = self.feat_norm_f(feat_f)
        final_feat = s_n * w_s + f_n * w_f
        logits = self.classifier(final_feat)

        if return_branch_feats:
            return logits, feat_s, feat_f
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
