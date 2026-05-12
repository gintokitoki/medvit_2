# Glaucoma_Project/env_check_v2.py
import torch
import timm
import transformers


def check_v2():
    print("=== 环境检查 V2.0 (针对频域掩码设计) ===")
    # 1. 检测 GPU
    if not torch.cuda.is_available():
        print("错误: 未检测到 GPU")
        return
    print(f"检测到 {torch.cuda.device_count()} 张 RTX 3090")

    # 2. 检测复数运算支持 (核心测试)
    try:
        x = torch.randn(1, 64, 56, 56).cuda()
        xf = torch.fft.rfft2(x)  # 快速实数傅里叶变换
        mask = torch.nn.Parameter(torch.ones_like(xf))  # 初始化可学习掩码
        out = xf * mask  # 执行点乘
        print("复数张量与可学习掩码点乘测试: 成功")
    except Exception as e:
        print(f"复数运算测试失败: {e}")


if __name__ == "__main__":
    check_v2()