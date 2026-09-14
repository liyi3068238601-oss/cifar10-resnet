"""CIFAR-10 的网络结构：ResNet 与 WideResNet。

ResNet
------
与 torchvision 中面向 ImageNet (224x224) 的版本不同，这里采用 CIFAR 版本的关键改动：

1. 主干不再使用 7x7 stride-2 卷积 + maxpool，而是 3x3 stride-1 卷积，
   因此输入 32x32 的图片在进入残差层前不会被下采样；
2. 第一个 stage 步长为 1，之后的每个 stage 步长为 2，最终特征图为 4x4；
3. 最后接全局平均池化 + 单层全连接。

各深度的差异只体现在「stage 数量、每个 stage 的块数、基础通道数」三项上，
因此用同一套代码统一表示：ResNet-20 是 3 个 stage / 每层 3 块 / 基础宽度 16，
而 ResNet-18 是 4 个 stage / 每层 2 块 / 基础宽度 64。

WideResNet
----------
WRN 采用 pre-activation 残差块（BN-ReLU-Conv 顺序，即 He et al. 2016 的
"identity mappings" 结构），只有 3 个 stage，用 widen factor 加宽而非加深：

    WRN-depth-widen: 每 stage 块数 n = (depth - 4) / 6，
    通道数依次为 16*widen, 32*widen, 64*widen

深度 28、widen 10 即经典的 WRN-28-10（36.5M 参数）。两个卷积之间可以加 dropout
（WRN 原文用 0.3，是这类宽网络的关键正则）。

Stochastic Depth
----------------
``DropPath`` 以一定概率整块丢弃残差分支、只保留恒等映射。概率在块之间**线性递增**
（浅层丢得少、深层丢得多）。它几乎不增加训练成本，在深层网络上可稳定涨点
（Huang et al., "Deep Networks with Stochastic Depth", ECCV 2016，ResNet-110 上 +1.16）。

参考:
    He et al., "Deep Residual Learning for Image Recognition" (CVPR 2016)
    He et al., "Identity Mappings in Deep Residual Networks" (ECCV 2016)
    Zagoruyko & Komodakis, "Wide Residual Networks" (BMVC 2016)
    Huang et al., "Deep Networks with Stochastic Depth" (ECCV 2016)
"""

from typing import List, Type

import torch
import torch.nn as nn
import torch.nn.functional as F


class DropPath(nn.Module):
    """Stochastic Depth：按样本整块丢弃残差分支。

    训练时以 ``drop_prob`` 的概率把该样本的残差输出置零（保留恒等映射），
    并按 ``1/(1-drop_prob)`` 缩放以保持期望不变。评估时不做任何事。

    这是 ResNet / WRN 里唯一被广泛验证「几乎免费还能涨点」的正则手段。
    """

    def __init__(self, drop_prob: float = 0.0) -> None:
        super().__init__()
        self.drop_prob = float(drop_prob)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.drop_prob <= 0.0 or not self.training:
            return x
        keep = 1.0 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        mask = x.new_empty(shape).bernoulli_(keep)
        return x * mask / keep

    def extra_repr(self) -> str:
        return f"drop_prob={self.drop_prob:g}"


def _drop_path_ramp(n_blocks: int, max_prob: float) -> List[float]:
    """生成线性递增的 drop-path 概率序列：0 -> max_prob。"""
    if n_blocks <= 1 or max_prob <= 0:
        return [0.0] * n_blocks
    return [max_prob * i / (n_blocks - 1) for i in range(n_blocks)]


# --------------------------------------------------------------------- ResNet

class BasicBlock(nn.Module):
    """ResNet-18/34 使用的基础残差块（两个 3x3 卷积，post-activation）。"""

    expansion = 1

    def __init__(self, in_planes: int, planes: int, stride: int = 1,
                 drop_path: float = 0.0) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.drop_path = DropPath(drop_path)

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
        out = self.drop_path(out) + self.shortcut(x)
        return F.relu(out, inplace=True)


class Bottleneck(nn.Module):
    """ResNet-50/101/152 使用的瓶颈块（1x1 -> 3x3 -> 1x1）。"""

    expansion = 4

    def __init__(self, in_planes: int, planes: int, stride: int = 1,
                 drop_path: float = 0.0) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * self.expansion, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)
        self.drop_path = DropPath(drop_path)

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
        out = self.drop_path(out) + self.shortcut(x)
        return F.relu(out, inplace=True)


class ResNetCIFAR(nn.Module):
    """可配置深度的 CIFAR 版 ResNet。

    Args:
        block: 残差块类型，``BasicBlock`` 或 ``Bottleneck``。
        layers: 每个 stage 的块数量，例如 ResNet-18 为 ``[2, 2, 2, 2]``。
        num_classes: 分类数，CIFAR-10 为 10。
        base_planes: 第一个 stage 的通道数（ResNet-18 为 64，ResNet-20 为 16）。
        width: 通道数缩放系数，1.0 表示标准宽度。
        drop_path: 残差分支的最大丢弃概率，在各块之间线性递增（0 表示关闭）。
    """

    def __init__(self, block: Type[nn.Module], layers: List[int],
                 num_classes: int = 10, base_planes: int = 64,
                 width: float = 1.0, drop_path: float = 0.0) -> None:
        super().__init__()
        self.in_planes = max(1, int(round(base_planes * width)))
        n_blocks = sum(layers)
        probs = _drop_path_ramp(n_blocks, drop_path)

        # CIFAR 主干：保持 32x32 分辨率
        self.conv1 = nn.Conv2d(3, self.in_planes, kernel_size=3, stride=1,
                               padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(self.in_planes)

        # 逐个 stage 构建。第一个 stage 不下采样，其余 stage 首块 stride=2。
        # 每个块的输入通道始终是 self.in_planes（上一块的输出）。
        stages, planes, idx = [], self.in_planes, 0
        for i, num_blocks in enumerate(layers):
            blocks = []
            for b in range(num_blocks):
                stride = 1 if (i == 0 or b > 0) else 2
                blocks.append(block(self.in_planes, planes, stride, probs[idx]))
                self.in_planes = planes * block.expansion
                idx += 1
            stages.append(nn.Sequential(*blocks))
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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = F.relu(self.bn1(self.conv1(x)), inplace=True)
        for stage in self.stages:
            out = stage(out)
        out = self.avgpool(out)
        out = torch.flatten(out, 1)
        return self.fc(out)


# ---------------------------------------------------------------- WideResNet

class PreActBlock(nn.Module):
    """WRN 使用的 pre-activation 残差块。

    与 ``BasicBlock`` 的区别在于顺序：先 BN-ReLU 再卷积（identity mappings），
    这样恒等映射路径上没有任何非线性，梯度可以无衰减地回传。
    """

    def __init__(self, in_planes: int, planes: int, stride: int = 1,
                 dropout: float = 0.0, drop_path: float = 0.0) -> None:
        super().__init__()
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1,
                               padding=1, bias=False)
        # WRN 原文在两个卷积之间放 dropout，是宽网络的关键正则
        self.dropout = dropout
        self.drop_path = DropPath(drop_path)

        self.equal = stride == 1 and in_planes == planes
        self.shortcut = None if self.equal else nn.Conv2d(
            in_planes, planes, kernel_size=1, stride=stride, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = F.relu(self.bn1(x), inplace=True)
        skip = x if self.equal else self.shortcut(out)
        out = self.conv1(out)
        out = F.relu(self.bn2(out), inplace=True)
        if self.dropout > 0:
            out = F.dropout(out, self.dropout, self.training)
        out = self.conv2(out)
        return self.drop_path(out) + skip


class WideResNet(nn.Module):
    """WideResNet，3 个 stage + pre-activation。

    Args:
        depth: 总深度，必须满足 ``(depth - 4) % 6 == 0``；每 stage 块数
            ``n = (depth - 4) / 6``（深度 28 -> n=4）。
        widen: widen factor，通道数依次为 ``16*widen, 32*widen, 64*widen``。
        num_classes: 分类数。
        dropout: 两个卷积之间的 dropout 概率（经典配置 0.3）。
        drop_path: 残差分支的最大丢弃概率，线性递增（0 表示关闭）。
    """

    def __init__(self, depth: int = 28, widen: int = 10, num_classes: int = 10,
                 dropout: float = 0.3, drop_path: float = 0.0) -> None:
        super().__init__()
        if (depth - 4) % 6 != 0:
            raise ValueError(f"depth 必须满足 (depth-4)%6==0，收到 {depth}")
        n = (depth - 4) // 6
        widths = [16 * widen, 32 * widen, 64 * widen]
        probs = _drop_path_ramp(3 * n, drop_path)

        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1, bias=False)

        blocks, in_planes, idx = [], 16, 0
        for si, planes in enumerate(widths):
            for bi in range(n):
                stride = 2 if (bi == 0 and si > 0) else 1
                blocks.append(PreActBlock(in_planes, planes, stride,
                                          dropout, probs[idx]))
                in_planes = planes
                idx += 1
        self.blocks = nn.Sequential(*blocks)

        self.bn = nn.BatchNorm2d(in_planes)
        self.fc = nn.Linear(in_planes, num_classes)

        # WRN 不使用「残差分支末层 BN 置零」的初始化：pre-act 结构下残差分支的
        # 输出量级本就受 BN 控制，置零反而会拖慢收敛，原文也未采用。
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.conv1(x)
        out = self.blocks(out)
        out = F.relu(self.bn(out), inplace=True)
        out = F.adaptive_avg_pool2d(out, 1)
        out = torch.flatten(out, 1)
        return self.fc(out)


# ------------------------------------------------------------------- 工厂函数

# 架构名 -> (残差块, 每个 stage 的块数, 基础通道数)
_RESNET_ARCHITECTURES = {
    "resnet20": (BasicBlock, [3, 3, 3], 16),
    "resnet18": (BasicBlock, [2, 2, 2, 2], 64),
    "resnet34": (BasicBlock, [3, 4, 6, 3], 64),
    "resnet50": (Bottleneck, [3, 4, 6, 3], 64),
}

# WRN 架构名 -> (depth, widen)
_WRN_ARCHITECTURES = {
    "wrn16-8": (16, 8),
    "wrn28-4": (28, 4),
    "wrn28-10": (28, 10),
    "wrn40-4": (40, 4),
}

ARCHITECTURES = sorted(list(_RESNET_ARCHITECTURES) + list(_WRN_ARCHITECTURES))


def build_model(name: str = "resnet18", num_classes: int = 10,
                width: float = 1.0, drop_path: float = 0.0,
                dropout: float = 0.0) -> nn.Module:
    """按名称构造模型。

    Args:
        name: 架构名，见 ``ARCHITECTURES``。
        num_classes: 分类数。
        width: 通道数缩放系数。ResNet 缩放基础通道数；WRN 缩放 widen factor。
        drop_path: 残差分支最大丢弃概率（线性递增），0 表示关闭。
        dropout: WRN 内部 dropout 概率，ResNet 忽略此项。
    """
    key = name.lower()

    if key in _WRN_ARCHITECTURES:
        depth, widen = _WRN_ARCHITECTURES[key]
        scaled = max(1, int(round(widen * width)))
        return WideResNet(depth=depth, widen=scaled, num_classes=num_classes,
                          dropout=dropout, drop_path=drop_path)

    if key not in _RESNET_ARCHITECTURES:
        raise ValueError(f"未知模型 {name!r}，可选: {ARCHITECTURES}")
    block, layers, base_planes = _RESNET_ARCHITECTURES[key]
    return ResNetCIFAR(block, layers, num_classes=num_classes,
                       base_planes=base_planes, width=width, drop_path=drop_path)


def count_parameters(model: nn.Module) -> int:
    """返回可训练参数总量。"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == "__main__":
    x = torch.randn(2, 3, 32, 32)
    for arch in ("resnet20", "resnet18", "resnet34", "resnet50",
                 "wrn16-8", "wrn28-4", "wrn28-10", "wrn40-4"):
        m = build_model(arch, dropout=0.3, drop_path=0.1).eval()
        with torch.no_grad():
            y = m(x)
        print(f"{arch:10s} params={count_parameters(m):>10,}  out={tuple(y.shape)}")
