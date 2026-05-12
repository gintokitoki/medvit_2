import torch
import torch.nn as nn
import torch.nn.functional as F


class FFTBranch(nn.Module):
    def __init__(self, output_dim=512):
        super(FFTBranch, self).__init__()

        # 特征提取器：处理 512x512 频谱图
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.Hardswish(inplace=True),

            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.Hardswish(inplace=True),

            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.Hardswish(inplace=True),

            nn.AdaptiveAvgPool2d((1, 1))
        )

        self.fc = nn.Sequential(
            nn.Linear(256, output_dim),
            nn.LayerNorm(output_dim)  # LayerNorm 在小样本频率特征上比 BN 更稳
        )

    def forward(self, x):
        # 1. 2D FFT 变换
        fft_feat = torch.fft.fft2(x, norm='ortho')
        # 2. 将低频移动到中心（与裁剪后的视盘位置物理对齐）
        fft_shift = torch.fft.fftshift(fft_feat)

        # 3. 取振幅谱并对数化缩放
        amp_spectrum = torch.abs(fft_shift)
        amp_spectrum = torch.log1p(amp_spectrum)

        # 4. CNN 提取全局频率特征
        out = self.features(amp_spectrum)
        out = torch.flatten(out, 1)
        out = self.fc(out)

        return out