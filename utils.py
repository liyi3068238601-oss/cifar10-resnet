"""训练相关的通用工具：随机种子、指标统计、日志与绘图。"""

from __future__ import annotations

import csv
import json
import random
from pathlib import Path
from typing import Dict, List, Sequence

import numpy as np
import torch

from data import CIFAR10_MEAN, CIFAR10_STD


def set_seed(seed: int = 42, deterministic: bool = False) -> None:
    """固定随机种子，保证结果可复现。"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    else:
        torch.backends.cudnn.benchmark = True


def get_device() -> torch.device:
    """优先使用 CUDA。"""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def describe_device(device: torch.device) -> str:
    if device.type == "cuda":
        idx = device.index or 0
        name = torch.cuda.get_device_name(idx)
        total = torch.cuda.get_device_properties(idx).total_memory / 1e9
        return f"CUDA:{idx} {name} ({total:.1f} GB)"
    return "CPU"


class AverageMeter:
    """滑动统计最近 ``window`` 次的均值。"""

    def __init__(self, window: int = 0) -> None:
        self.window = window
        self.reset()

    def reset(self) -> None:
        self.sum = 0.0
        self.count = 0
        self.history: List[float] = []

    def update(self, value: float, n: int = 1) -> None:
        self.sum += float(value) * n
        self.count += n
        self.history.append(float(value))

    @property
    def avg(self) -> float:
        return self.sum / self.count if self.count else 0.0

    @property
    def recent(self) -> float:
        """最近 ``window`` 次的均值（window<=0 时等同于 avg）。"""
        if self.window <= 0 or len(self.history) < self.window:
            return self.avg
        return float(np.mean(self.history[-self.window:]))


class ModelEMA:
    """模型权重的指数滑动平均（Exponential Moving Average）。

    训练时维护一份权重的滑动平均副本，评估时用它而不是原始权重。
    由于平均掉了单步的噪声，通常能稳定涨 0.3~1 个点，且几乎不增加训练开销。

    参考: Polyak & Juditsky (1992); 在深度学习中的实践见
    "Averaging Weights Leads to Wider Optima and Better Generalization" (2018)。
    """

    def __init__(self, model: torch.nn.Module, decay: float = 0.999) -> None:
        import copy

        self.module = copy.deepcopy(model).eval()
        for p in self.module.parameters():
            p.requires_grad_(False)
        self.decay = decay
        self.updates = 0

    @torch.no_grad()
    def update(self, model: torch.nn.Module) -> None:
        self.updates += 1
        # 训练初期让衰减系数快速爬升，避免前几步就把权重锁死在初始值附近
        d = min(self.decay, (1 + self.updates) / (10 + self.updates))
        msd = model.state_dict()
        for key, value in self.module.state_dict().items():
            if value.dtype.is_floating_point:
                value.mul_(d).add_(msd[key].detach(), alpha=1.0 - d)
            else:
                # BN 的 num_batches_tracked 是整型，直接拷贝
                value.copy_(msd[key])


@torch.no_grad()
def accuracy(logits: torch.Tensor, target: torch.Tensor, topk=(1,)) -> List[float]:
    """返回 top-k 准确率（百分比）。"""
    maxk = max(topk)
    batch_size = target.size(0)
    _, pred = logits.topk(maxk, dim=1, largest=True, sorted=True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    out = []
    for k in topk:
        correct_k = correct[:k].reshape(-1).float().sum().item()
        out.append(100.0 * correct_k / batch_size)
    return out


def save_json(obj, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def load_model_from_checkpoint(ckpt_path: str | Path, device,
                               num_classes: int = 10):
    """按 checkpoint 里记录的元信息重建模型并载入权重。

    Returns:
        ``(model, ckpt)``，model 已 ``to(device)`` 且处于 eval 模式。
    """
    import torch

    from model import build_model

    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    model = build_model(ckpt.get("model", "resnet18"),
                        num_classes=num_classes,
                        width=ckpt.get("width", 1.0),
                        drop_path=ckpt.get("drop_path", 0.0),
                        dropout=ckpt.get("dropout", 0.0)).to(device)
    missing, unexpected = model.load_state_dict(ckpt["state_dict"], strict=False)
    if missing or unexpected:
        raise RuntimeError(f"权重与结构不匹配: 缺失 {missing[:3]} / 多余 {unexpected[:3]}")
    model.eval()
    return model, ckpt


@torch.no_grad()
def recalibrate_bn(model, loader, device, max_batches: int = 0,
                   verbose: bool = True) -> int:
    """只用**训练图像**重新估计 BatchNorm 的 running statistics。

    为什么需要：平均权重（EMA/SWA）与 BN 统计量可能不匹配。EMA 的 running stats
    是对训练过程中各步统计量的滑动平均，而那些统计量是在**增强后**的图上算出来的
    ——强增强、尤其是 MixUp/CutMix 的混合图会明显改变激活的均值与方差，与推理时
    看到的干净图像分布不一致。重新估计一次成本很低（只前向），值得用实验判断。

    实现要点（顺序很重要）：

    1. 先整体 ``eval()``，确保 Dropout / DropPath **关闭**。否则这一步会变成一次
       新的随机前向，而不是统计量校准。
    2. 再把 BN 单独切回 ``train()``，用累计平均（``momentum=None``）从零重新累计，
       避免旧统计量残留。
    3. 只喂训练图像，不得使用测试图像。

    Args:
        max_batches: 只用前若干个 batch（0 表示用完整 loader）。
        verbose: 是否打印进度。

    Returns:
        实际参与累计的 batch 数。
    """
    import torch.nn as nn

    was_training = model.training
    model.eval()

    bns = [m for m in model.modules()
           if isinstance(m, nn.modules.batchnorm._BatchNorm)]
    if not bns:
        if verbose:
            print("[bn] 模型不含 BatchNorm，跳过")
        return 0

    saved_momentum = [bn.momentum for bn in bns]
    for bn in bns:
        bn.reset_running_stats()
        bn.momentum = None          # 累计移动平均，而不是指数滑动平均
        bn.train()                  # 只让 BN 更新统计

    n = 0
    for images, _ in loader:
        if max_batches and n >= max_batches:
            break
        model(images.to(device, non_blocking=True))
        n += 1
        if verbose and n % 50 == 0:
            print(f"[bn]   已累计 {n} 个 batch", flush=True)

    for bn, mom in zip(bns, saved_momentum):
        bn.momentum = mom
    model.train(was_training)

    if verbose:
        seen = n * (loader.batch_size or 0)
        print(f"[bn] 重新估计完成：{n} 个 batch，约 {seen} 张训练图")
    return n


# ------------------------------------------------------------ 配对显著性检验

def load_correctness(npz_path: str | Path) -> np.ndarray:
    """从 evaluate.py 导出的 ``preds*.npz`` 读取逐样本判对与否。"""
    data = np.load(npz_path)
    return (data["preds"] == data["labels"]).astype(np.int8)


def mcnemar_test(correct_a: np.ndarray, correct_b: np.ndarray) -> Dict[str, float]:
    """McNemar 检验：比较两个模型在**同一测试集**上的表现是否有显著差异。

    这里不能用单模型的二项置信区间来判断两个模型谁更好。两个模型面对的是同一批
    图片，它们的错误是**配对**的（同一张难图往往两个都错），差值的不确定性远小于
    把两次评估当成独立样本时的估计。McNemar 只看"不一致"的那部分：

        b = A 错 B 对     c = A 对 B 错

    在"两个模型等价"的原假设下，b 服从 Binomial(b+c, 0.5)。

    Args:
        correct_a: 模型 A 的逐样本判对情况（bool/int 数组）。
        correct_b: 模型 B 的逐样本判对情况，顺序必须与 A 完全一致。

    Returns:
        ``{"n_a_only", "n_b_only", "n_discordant", "acc_a", "acc_b",
           "acc_diff", "p_value", "method"}``；``acc_diff = acc_b - acc_a``（百分点）。
    """
    a = np.asarray(correct_a).astype(bool)
    b_arr = np.asarray(correct_b).astype(bool)
    if a.shape != b_arr.shape:
        raise ValueError(f"两个模型的预测数量不一致: {a.shape} vs {b_arr.shape}")

    n_a_only = int(np.sum(a & ~b_arr))   # A 对 B 错
    n_b_only = int(np.sum(~a & b_arr))   # A 错 B 对
    n = n_a_only + n_b_only

    if n == 0:
        p_value, method = 1.0, "identical"
    elif n < 25:
        # 精确二项检验（双侧）
        from math import comb
        tail = sum(comb(n, k) for k in range(min(n_a_only, n_b_only) + 1))
        p_value = min(1.0, 2.0 * tail * 0.5 ** n)
        method = "exact_binomial"
    else:
        # 连续性校正的卡方检验；自由度 1 时 P(X > x) = erfc(sqrt(x/2))
        from math import erfc, sqrt
        chi2 = (abs(n_a_only - n_b_only) - 1) ** 2 / n
        chi2 = max(chi2, 0.0)
        p_value = erfc(sqrt(chi2 / 2.0))
        method = "chi2_continuity_corrected"

    acc_a = float(a.mean() * 100)
    acc_b = float(b_arr.mean() * 100)
    return {
        "n_a_only": n_a_only,
        "n_b_only": n_b_only,
        "n_discordant": n,
        "acc_a": acc_a,
        "acc_b": acc_b,
        "acc_diff": acc_b - acc_a,
        "p_value": float(p_value),
        "method": method,
    }


def paired_bootstrap_ci(correct_a: np.ndarray, correct_b: np.ndarray,
                        n_boot: int = 10000, alpha: float = 0.05,
                        seed: int = 0) -> tuple[float, float, float]:
    """配对 bootstrap：给准确率差值一个置信区间。

    与 McNemar 互补——McNemar 回答"是否有差异"，这里回答"差异大概多大"。
    对样本（而非对模型）重采样，因此保留了配对结构。

    Returns:
        ``(diff, lo, hi)``，单位为百分点，``diff = acc_b - acc_a``。
    """
    a = np.asarray(correct_a).astype(np.float32)
    b_arr = np.asarray(correct_b).astype(np.float32)
    if a.shape != b_arr.shape:
        raise ValueError(f"两个模型的预测数量不一致: {a.shape} vs {b_arr.shape}")

    diff = b_arr - a                                  # 每个样本的 (B对-A对)，取值 -1/0/+1
    rng = np.random.default_rng(seed)
    n = len(diff)
    idx = rng.integers(0, n, size=(n_boot, n))
    boot = diff[idx].mean(axis=1) * 100
    lo, hi = np.percentile(boot, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return float(diff.mean() * 100), float(lo), float(hi)


class CSVLogger:
    """把每个 epoch 的指标追加写入 CSV。"""

    def __init__(self, path: str | Path, fieldnames: Sequence[str]) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.fieldnames = list(fieldnames)
        with self.path.open("w", newline="", encoding="utf-8") as f:
            csv.DictWriter(f, fieldnames=self.fieldnames).writeheader()

    def log(self, row: Dict[str, object]) -> None:
        with self.path.open("a", newline="", encoding="utf-8") as f:
            csv.DictWriter(f, fieldnames=self.fieldnames).writerow(row)


def plot_history(history: Dict[str, List[float]], path: str | Path,
                 title: str = "CIFAR-10 training curves") -> None:
    """绘制 loss 与 accuracy 曲线（2x1 子图）。

    图内文字使用英文：matplotlib 默认字体不含中日韩字形，用中文会渲染成方框，
    而依赖系统中文字体又会让脚本在其他机器上不可移植。
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    epochs = np.arange(1, len(history["train_loss"]) + 1)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

    axes[0].plot(epochs, history["train_loss"], label="train", linewidth=1.8)
    axes[0].plot(epochs, history["test_loss"], label="test", linewidth=1.8)
    axes[0].set_xlabel("epoch")
    axes[0].set_ylabel("loss")
    axes[0].set_title("Loss")
    axes[0].legend()
    axes[0].grid(alpha=0.3)

    axes[1].plot(epochs, history["train_acc"], label="train", linewidth=1.8)
    axes[1].plot(epochs, history["test_acc"], label="test", linewidth=1.8)
    best_ep = int(np.argmax(history["test_acc"])) + 1
    best_acc = max(history["test_acc"])
    axes[1].scatter([best_ep], [best_acc], color="crimson", zorder=5,
                    label=f"best {best_acc:.2f}% @ epoch {best_ep}")
    axes[1].set_xlabel("epoch")
    axes[1].set_ylabel("accuracy (%)")
    axes[1].set_title("Accuracy")
    axes[1].legend()
    axes[1].grid(alpha=0.3)

    fig.suptitle(title)
    fig.tight_layout()
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_confusion_matrix(cm: np.ndarray, classes: Sequence[str], path: str | Path,
                          normalize: bool = True,
                          title: str = "Confusion matrix") -> None:
    """绘制混淆矩阵热力图。"""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    if normalize:
        cm = cm.astype(np.float64) / np.maximum(cm.sum(axis=1, keepdims=True), 1) * 100

    fig, ax = plt.subplots(figsize=(8.2, 7))
    im = ax.imshow(cm, cmap="Blues", vmin=0, vmax=100 if normalize else None)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04,
                 label="row-normalized (%)" if normalize else "count")

    ax.set_xticks(range(len(classes)))
    ax.set_yticks(range(len(classes)))
    ax.set_xticklabels(classes, rotation=45, ha="right")
    ax.set_yticklabels(classes)
    ax.set_xlabel("predicted")
    ax.set_ylabel("true")
    ax.set_title(title)

    thresh = cm.max() / 2.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            v = cm[i, j]
            if v < (0.5 if normalize else 1):
                continue
            text = f"{v:.0f}" if normalize else f"{int(v)}"
            ax.text(j, i, text, ha="center", va="center", fontsize=8,
                    color="white" if v > thresh else "black")

    fig.tight_layout()
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def denormalize(img: torch.Tensor, mean=CIFAR10_MEAN, std=CIFAR10_STD) -> torch.Tensor:
    """把归一化后的张量还原成可显示范围 [0, 1]。"""
    mean_t = torch.tensor(mean).view(3, 1, 1)
    std_t = torch.tensor(std).view(3, 1, 1)
    return (img * std_t + mean_t).clamp(0, 1)


def plot_predictions(images: torch.Tensor, labels: Sequence[int],
                     preds: Sequence[int], probs: Sequence[float],
                     classes: Sequence[str], path: str | Path,
                     cols: int = 8) -> None:
    """可视化一批预测结果，标题用绿色表示正确、红色表示错误。"""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    n = len(images)
    rows = int(np.ceil(n / cols))
    # 标题用单行并留足行距，否则下一行的标题会压到上一行的图片上
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 1.4, rows * 1.95))
    axes = np.atleast_1d(axes).ravel()

    for ax in axes:
        ax.axis("off")

    for i in range(n):
        ax = axes[i]
        img = denormalize(images[i].cpu()).permute(1, 2, 0).numpy()
        ax.imshow(img)
        ok = int(labels[i]) == int(preds[i])
        color = "#1a7f37" if ok else "#c1121f"
        ax.set_title(f"{classes[int(preds[i])]} {probs[i] * 100:.0f}%",
                     color=color, fontsize=9, pad=5)
        ax.axis("off")

    fig.suptitle("Test set predictions (green = correct, red = wrong)", fontsize=12)
    fig.subplots_adjust(left=0.01, right=0.99, top=0.86, bottom=0.01,
                        hspace=0.42, wspace=0.06)
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
