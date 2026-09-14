"""批级别数据增强：MixUp 与 CutMix。

这两个方法与 ``data.py`` 里基于单张图片的变换不同，它们是**在一个 batch 内**
把两张图按比例混合，因此必须在拿到 batch 之后才能做，不能放进
``torchvision.transforms`` 的流水线里。

两者都在训练时以一定概率随机启用，混合后的样本使用软标签计算损失，
所以配套的损失函数是 :func:`soft_target_cross_entropy`。

参考文献:
    MixUp:  Zhang et al., "mixup: Beyond Empirical Risk Minimization" (ICLR 2018)
    CutMix: Yun et al., "CutMix: Regularization Strategy to Train Strong
            Classifiers with Localizable Features" (ICCV 2019)
"""

from __future__ import annotations

import random

import numpy as np
import torch
import torch.nn.functional as F


def mixup(x: torch.Tensor, y: torch.Tensor, alpha: float = 1.0):
    """按 Beta(alpha, alpha) 采样的比例对 batch 做线性插值。

    Returns:
        ``(x_mixed, y_a, y_b, lam)``，其中 lam 是 x 中来自 ``y_a`` 的权重。
    """
    lam = float(np.random.beta(alpha, alpha)) if alpha > 0 else 1.0
    index = torch.randperm(x.size(0), device=x.device)
    x_mixed = lam * x + (1.0 - lam) * x[index]
    return x_mixed, y, y[index], lam


def cutmix(x: torch.Tensor, y: torch.Tensor, alpha: float = 1.0):
    """把 batch 内另一张图的一个随机矩形区域粘贴过来。

    粘贴面积占整图的比例服从 Beta(alpha, alpha)，因此 lam 与 MixUp 的含义一致。
    """
    lam = float(np.random.beta(alpha, alpha)) if alpha > 0 else 1.0
    index = torch.randperm(x.size(0), device=x.device)

    _, _, h, w = x.shape
    # 让裁剪框面积占比等于 1-lam
    ratio = (1.0 - lam) ** 0.5
    cut_h, cut_w = int(h * ratio), int(w * ratio)
    cy, cx = np.random.randint(h), np.random.randint(w)
    y1, y2 = max(cy - cut_h // 2, 0), min(cy + cut_h // 2, h)
    x1, x2 = max(cx - cut_w // 2, 0), min(cx + cut_w // 2, w)

    x_mixed = x.clone()
    x_mixed[:, :, y1:y2, x1:x2] = x[index, :, y1:y2, x1:x2]

    # 实际粘贴面积可能与计划值有偏差（碰到边界被裁），按真实面积重算 lam
    lam = 1.0 - (y2 - y1) * (x2 - x1) / (h * w)
    return x_mixed, y, y[index], lam


def mixup_or_cutmix(x: torch.Tensor, y: torch.Tensor, mixup_alpha: float = 1.0,
                    cutmix_alpha: float = 1.0, prob: float = 0.5,
                    switch: float = 0.5):
    """以 ``prob`` 的概率随机选择 MixUp 或 CutMix，否则原样返回。

    **两种方法各自使用自己的 alpha**。早期版本只接受单个 alpha 并把 MixUp 的
    alpha 也套用到 CutMix 上，导致 ``mixup=0.2, cutmix=1.0`` 时 CutMix 实际用的是
    0.2；这里改为分别传入。``alpha <= 0`` 表示该方法被禁用，两者都禁用时直接返回原样。

    Args:
        mixup_alpha: MixUp 的 Beta 分布参数，<=0 表示禁用 MixUp。
        cutmix_alpha: CutMix 的 Beta 分布参数，<=0 表示禁用 CutMix。
        prob: 启用混合的概率。
        switch: 两者都可用时选择 MixUp 的概率（其余情况选 CutMix）。

    Returns:
        ``(x, y_a, y_b, lam)``；未启用混合时 ``y_a`` 与 ``y_b`` 相同且 ``lam=1``。
    """
    use_mixup, use_cutmix = mixup_alpha > 0, cutmix_alpha > 0
    if not (use_mixup or use_cutmix) or random.random() > prob:
        return x, y, y, 1.0

    if use_mixup and use_cutmix:
        pick_mixup = random.random() < switch
    else:
        pick_mixup = use_mixup

    if pick_mixup:
        return mixup(x, y, mixup_alpha)
    return cutmix(x, y, cutmix_alpha)


def soft_target_cross_entropy(logits: torch.Tensor, y_a: torch.Tensor,
                              y_b: torch.Tensor, lam: float,
                              label_smoothing: float = 0.0) -> torch.Tensor:
    """混合样本的损失：两个目标损失的加权和。"""
    if lam >= 1.0:
        return F.cross_entropy(logits, y_a, label_smoothing=label_smoothing)
    return (lam * F.cross_entropy(logits, y_a, label_smoothing=label_smoothing)
            + (1.0 - lam) * F.cross_entropy(logits, y_b, label_smoothing=label_smoothing))
