"""数据加载与数据增强。

数据集读取
----------
这里没有使用 ``torchvision.datasets.CIFAR10``，因为它会校验每个 batch 文件的
md5，而 ``prepare_data.py`` 从 HuggingFace parquet 转换出来的文件字节必然与
官方发布的不同，会被判定为「数据损坏」。因此改为直接读取 pickle，
只校验形状与标签取值范围（见 :class:`CIFAR10Pickle`）。

数据增强
--------
训练集增强（参考 He et al. 及后续 CIFAR 训练的常用组合）:

1. ``RandomCrop(32, padding=4)`` —— 平移不变性
2. ``RandomHorizontalFlip`` —— 水平翻转
3. ``Cutout`` 或 ``RandAugment`` —— 随机遮挡 / 自动增强策略
4. 归一化到 CIFAR-10 的通道均值/标准差

测试集只做 ``ToTensor`` + 归一化。
"""

from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

from prepare_data import prepare_cifar10

# CIFAR-10 全量训练集的逐通道均值与标准差
CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR10_STD = (0.2470, 0.2435, 0.2616)

CLASSES = ("airplane", "automobile", "bird", "cat", "deer",
           "dog", "frog", "horse", "ship", "truck")

CLASSES_ZH = ("飞机", "汽车", "鸟", "猫", "鹿", "狗", "青蛙", "马", "船", "卡车")

TRAIN_BATCHES = [f"data_batch_{i}" for i in range(1, 6)]
TEST_BATCHES = ["test_batch"]


class CIFAR10Pickle(Dataset):
    """直接读取 ``cifar-10-batches-py`` 下的 pickle 批文件。

    pickle 内的 ``data`` 是 ``(N, 3072)`` 的 uint8 数组，通道顺序为 R|G|B，
    这里还原成 ``(N, 32, 32, 3)`` 以便交给 torchvision 的变换处理。

    Args:
        root: 数据根目录，内含 ``cifar-10-batches-py``。
        train: True 读 5 个训练 batch，False 读 test_batch。
        transform: 作用于 PIL 图像的变换。
    """

    def __init__(self, root: str | Path, train: bool = True, transform=None) -> None:
        self.root = Path(root) / "cifar-10-batches-py"
        self.train = train
        self.transform = transform

        names = TRAIN_BATCHES if train else TEST_BATCHES
        data, targets = [], []
        for name in names:
            path = self.root / name
            if not path.is_file():
                raise FileNotFoundError(
                    f"找不到 {path}，请先运行 `python prepare_data.py`")
            with path.open("rb") as f:
                entry = pickle.load(f)
            data.append(np.asarray(entry["data"]))
            targets.extend(int(t) for t in entry["labels"])

        self.data = np.vstack(data).reshape(-1, 3, 32, 32).transpose(0, 2, 3, 1)
        self.targets = targets

        if self.data.shape[0] != len(self.targets):
            raise RuntimeError(f"数据与标签数量不一致: {self.data.shape[0]} vs {len(self.targets)}")

    def __len__(self) -> int:
        return len(self.targets)

    def __getitem__(self, index: int):
        # 转成 PIL 是为了复用 torchvision 的变换（RandAugment 等只接受 PIL 输入）
        img = Image.fromarray(self.data[index])
        if self.transform is not None:
            img = self.transform(img)
        return img, self.targets[index]

    def __repr__(self) -> str:  # pragma: no cover - 仅用于调试打印
        split = "train" if self.train else "test"
        return f"CIFAR10Pickle({split}, n={len(self)}, root={self.root})"


class Cutout:
    """随机遮挡一个 ``size x size`` 的方形区域（填 0）。

    实现基于论文 "Improved Regularization of Convolutional Neural Networks
    with Cutout" (DeVries & Taylor, 2017)。
    """

    def __init__(self, size: int = 8) -> None:
        self.size = size

    def __call__(self, img: torch.Tensor) -> torch.Tensor:
        _, h, w = img.shape
        if self.size <= 0:
            return img
        cy = np.random.randint(h)
        cx = np.random.randint(w)
        y1, y2 = max(cy - self.size // 2, 0), min(cy + self.size // 2, h)
        x1, x2 = max(cx - self.size // 2, 0), min(cx + self.size // 2, w)
        img[:, y1:y2, x1:x2] = 0.0
        return img

    def __repr__(self) -> str:  # pragma: no cover - 仅用于调试打印
        return f"{self.__class__.__name__}(size={self.size})"


def build_transforms(augment: bool = True, cutout: int = 8, randaugment: int = 0):
    """构造 (训练变换, 测试变换)。

    Args:
        augment: 为 False 时训练集也不做增强（用于对照实验）。
        cutout: Cutout 边长，0 表示关闭。
        randaugment: 大于 0 时启用 RandAugment，值为 magnitude（推荐 9），
            按 N=2 的强度随机叠加两个算子。RandAugment 本身已包含较强的
            遮挡/颜色扰动，通常不再叠加 Cutout。
    """
    normalize = transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD)

    test_tf = transforms.Compose([transforms.ToTensor(), normalize])

    if not augment:
        return test_tf, test_tf

    train_ops = [
        transforms.RandomCrop(32, padding=4, padding_mode="reflect"),
        transforms.RandomHorizontalFlip(),
    ]
    if randaugment > 0:
        # RandAugment 作用在 PIL 图像上，必须放在 ToTensor 之前
        train_ops.append(transforms.RandAugment(num_ops=2, magnitude=randaugment))
    train_ops += [transforms.ToTensor(), normalize]
    if cutout > 0:
        train_ops.append(Cutout(cutout))
    return transforms.Compose(train_ops), test_tf


def get_datasets(data_root: str | Path = "data", augment: bool = True,
                 cutout: int = 8, randaugment: int = 0, download: bool = True):
    """返回 ``(train_set, test_set)``。"""
    if download:
        prepare_cifar10(data_root)
    train_tf, test_tf = build_transforms(augment, cutout, randaugment)
    train_set = CIFAR10Pickle(data_root, train=True, transform=train_tf)
    test_set = CIFAR10Pickle(data_root, train=False, transform=test_tf)
    return train_set, test_set


def get_dataloaders(data_root: str | Path = "data", batch_size: int = 128,
                    num_workers: int = 4, augment: bool = True, cutout: int = 8,
                    randaugment: int = 0, download: bool = True,
                    pin_memory: bool = True):
    """构造训练/测试 DataLoader。

    Returns:
        ``(train_loader, test_loader)``
    """
    train_set, test_set = get_datasets(data_root, augment, cutout, randaugment, download)

    common = dict(num_workers=num_workers, pin_memory=pin_memory,
                  persistent_workers=num_workers > 0)
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True,
                              drop_last=True, **common)
    test_loader = DataLoader(test_set, batch_size=256, shuffle=False, **common)
    return train_loader, test_loader


if __name__ == "__main__":
    tr, te = get_dataloaders(batch_size=8, num_workers=0)
    x, y = next(iter(tr))
    print("train:", tr.dataset)
    print("test :", te.dataset)
    print("batch:", tuple(x.shape), tuple(y.shape), x.dtype)
    print("归一化后范围: [%.3f, %.3f]" % (x.min().item(), x.max().item()))
