import torch
import torch.nn as nn
import torch.nn.functional as F


class FFTBranch(nn.Module):
    def __init__(self, input_res=512, output_dim=1024, high_pass_radius=20):
        super(FFTBranch, self).__init__()
        self.res = input_res
        self.high_pass_radius = float(high_pass_radius)

        # 1. 可学习的频域滤波器 (Learnable Spectral Filter)
        # 它可以学习哪些频率分量（如高频噪声或低频结构）对分类更重要
        self.learnable_mask = nn.Parameter(torch.ones(1, 1, input_res, input_res))

        # 固定高通：抑制频谱中心低频，与可学习 mask 相乘；尺寸在 forward 中与 FFT 网格对齐
        yy, xx = torch.meshgrid(
            torch.arange(input_res), torch.arange(input_res), indexing="ij"
        )
        cx = cy = input_res // 2
        dist = torch.sqrt((xx.float() - cx) ** 2 + (yy.float() - cy) ** 2)
        high_pass = (dist >= self.high_pass_radius).float()
        self.register_buffer("_high_pass_base", high_pass.view(1, 1, input_res, input_res))

        # 2. 增强型特征提取器
        # 输入变为 2 通道：振幅谱 + 相位谱
        self.features = nn.Sequential(
            nn.Conv2d(2, 64, kernel_size=7, stride=2, padding=3),  # 较大的感受野
            nn.BatchNorm2d(64),
            nn.Hardswish(inplace=True),
            nn.MaxPool2d(3, stride=2, padding=1),

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
            nn.LayerNorm(output_dim)
        )

    def forward(self, x):
        # 转换到灰度图进行频率分析（医学影像通常单通道频率信息最稳，也可扩展为3通道）
        if x.shape[1] == 3:
            x_gray = 0.299 * x[:, 0:1, :, :] + 0.587 * x[:, 1:2, :, :] + 0.114 * x[:, 2:3, :, :]
        else:
            x_gray = x

        # 1. 2D FFT（float32，避免半精度下 torch.fft 不稳定）
        x_gray = x_gray.float()
        fft_feat = torch.fft.fft2(x_gray, norm='ortho')
        fft_shift = torch.fft.fftshift(fft_feat)

        # 2. 可学习滤波器需与当前 FFT 空间尺寸一致（输入分辨率可与 self.res 不同）
        mask = self.learnable_mask
        if mask.shape[-2:] != fft_shift.shape[-2:]:
            mask = F.interpolate(
                mask, size=fft_shift.shape[-2:], mode="bilinear", align_corners=False
            )

        hp = self._high_pass_base
        if hp.shape[-2:] != fft_shift.shape[-2:]:
            hp = F.interpolate(
                hp, size=fft_shift.shape[-2:], mode="bilinear", align_corners=False
            )

        # 这一步让模型决定加强或削弱哪些频率点（复数 × 实数，广播到 batch）
        filtered_fft = fft_shift * mask * hp

        # 3. 提取 振幅谱 和 相位谱
        amp = torch.abs(filtered_fft)
        amp = torch.log1p(amp)  # 对数缩放

        phase = torch.angle(filtered_fft)  # 相位信息包含结构轮廓

        # 4. 拼接双流信息 [B, 2, H, W]
        freq_info = torch.cat([amp, phase], dim=1)

        # 5. CNN 提取特征
        out = self.features(freq_info)
        out = torch.flatten(out, 1)
        out = self.fc(out)

        return out