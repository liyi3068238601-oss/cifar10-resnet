"""针对 CIFAR-10 的 ResNet 实现。

与 torchvision 中面向 ImageNet (224x224) 的版本不同，这里采用 CIFAR 版本的关键改动：

1. 主干不再使用 7x7 stride-2 卷积 + maxpool，而是 3x3 stride-1 卷积，
   因此输入 32x32 的图片在进入残差层前不会被下采样；
2. 第一个 stage 步长为 1，之后的每个 stage 步长为 2，最终特征图为 4x4；
3. 最后接全局平均池化 + 单层全连接。

各深度的差异只体现在「stage 数量、每个 stage 的块数、基础通道数」三项上，
因此用同一套代码统一表示：ResNet-20 是 3 个 stage / 每层 3 块 / 基础宽度 16，
而 ResNet-18 是 4 个 stage / 每层 2 块 / 基础宽度 64。

参考: He et al., "Deep Residual Learning for Image Recognition" (CVPR 2016)。
"""

from typing import List, Type

import torch
import torch.nn as nn
import torch.nn.functional as F


class BasicBlock(nn.Module):
    """ResNet-18/34 使用的基础残差块（两个 3x3 卷积）。"""

    expansion = 1

    def __init__(self, in_planes: int, planes: int, stride: int = 1) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        # 当空间尺寸或通道数发生变化时，恒等映射需要用 1x1 卷积投影对齐
        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes),
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = F.relu(self.bn1(self.conv1(x)), inplace=True)
        out = self.bn2(self.conv2(out))
        out = out + self.shortcut(x)
        return F.relu(out, inplace=True)


class Bottleneck(nn.Module):
    """ResNet-50/101/152 使用的瓶颈块（1x1 -> 3x3 -> 1x1）。"""

    expansion = 4

    def __init__(self, in_planes: int, planes: int, stride: int = 1) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * self.expansion, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != planes * self.expansion:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, planes * self.expansion, kernel_size=1,
                          stride=stride, bias=False),
                nn.BatchNorm2d(planes * self.expansion),
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = F.relu(self.bn1(self.conv1(x)), inplace=True)
        out = F.relu(self.bn2(self.conv2(out)), inplace=True)
        out = self.bn3(self.conv3(out))
        out = out + self.shortcut(x)
        return F.relu(out, inplace=True)


class ResNetCIFAR(nn.Module):
    """可配置深度的 CIFAR 版 ResNet。

    Args:
        block: 残差块类型，``BasicBlock`` 或 ``Bottleneck``。
        layers: 每个 stage 的块数量，例如 ResNet-18 为 ``[2, 2, 2, 2]``。
        num_classes: 分类数，CIFAR-10 为 10。
        base_planes: 第一个 stage 的通道数（ResNet-18 为 64，ResNet-20 为 16）。
        width: 通道数缩放系数，1.0 表示标准宽度。
    """

    def __init__(self, block: Type[nn.Module], layers: List[int],
                 num_classes: int = 10, base_planes: int = 64,
                 width: float = 1.0) -> None:
        super().__init__()
        self.in_planes = max(1, int(round(base_planes * width)))

        # CIFAR 主干：保持 32x32 分辨率
        self.conv1 = nn.Conv2d(3, self.in_planes, kernel_size=3, stride=1,
                               padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(self.in_planes)

        # 逐个 stage 构建。第一个 stage 不下采样，其余 stride=2，通道数翻倍。
        stages, planes = [], self.in_planes
        for i, num_blocks in enumerate(layers):
            stages.append(self._make_layer(block, planes, num_blocks,
                                           stride=1 if i == 0 else 2))
            planes *= 2
        self.stages = nn.ModuleList(stages)

        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(planes // 2 * block.expansion, num_classes)

        # 权重初始化：残差分支最后一层 BN 置零，使每个 block 初始近似恒等映射，
        # 这样深层网络在训练初期也能稳定传播梯度。
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
        for m in self.modules():
            if isinstance(m, BasicBlock):
                nn.init.zeros_(m.bn2.weight)
            elif isinstance(m, Bottleneck):
                nn.init.zeros_(m.bn3.weight)

    def _make_layer(self, block: Type[nn.Module], planes: int,
                    num_blocks: int, stride: int) -> nn.Sequential:
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for s in strides:
            layers.append(block(self.in_planes, planes, s))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = F.relu(self.bn1(self.conv1(x)), inplace=True)
        for stage in self.stages:
            out = stage(out)
        out = self.avgpool(out)
        out = torch.flatten(out, 1)
        return self.fc(out)


# 架构名 -> (残差块, 每个 stage 的块数, 基础通道数)
_ARCHITECTURES = {
    "resnet20": (BasicBlock, [3, 3, 3], 16),
    "resnet18": (BasicBlock, [2, 2, 2, 2], 64),
    "resnet34": (BasicBlock, [3, 4, 6, 3], 64),
    "resnet50": (Bottleneck, [3, 4, 6, 3], 64),
}


def build_model(name: str = "resnet18", num_classes: int = 10,
                width: float = 1.0) -> ResNetCIFAR:
    """按名称构造模型。

    Args:
        name: 架构名，见 ``_ARCHITECTURES``。
        num_classes: 分类数。
        width: 通道数缩放系数。
    """
    key = name.lower()
    if key not in _ARCHITECTURES:
        raise ValueError(f"未知模型 {name!r}，可选: {sorted(_ARCHITECTURES)}")
    block, layers, base_planes = _ARCHITECTURES[key]
    return ResNetCIFAR(block, layers, num_classes=num_classes,
                       base_planes=base_planes, width=width)


def count_parameters(model: nn.Module) -> int:
    """返回可训练参数总量。"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == "__main__":
    for arch in ("resnet20", "resnet18", "resnet34", "resnet50"):
        m = build_model(arch)
        x = torch.randn(2, 3, 32, 32)
        with torch.no_grad():
            y = m(x)
        print(f"{arch:10s} params={count_parameters(m):>9,}  out={tuple(y.shape)}")
