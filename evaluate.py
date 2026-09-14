"""评估脚本：在 CIFAR-10 测试集上评测已保存的模型。

用法::

    python evaluate.py --ckpt checkpoints/best.pt
    python evaluate.py --ckpt checkpoints/best.pt --tta      # 带水平翻转 TTA

输出:
    results/confusion_matrix.png   行归一化混淆矩阵
    results/predictions.png        预测示例（含分错的样本）
    results/eval.json              总体/每类指标
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm

from data import CLASSES, CLASSES_ZH, get_dataloaders
from model import build_model, count_parameters
from utils import (accuracy, describe_device, get_device, plot_confusion_matrix,
                   plot_predictions, save_json, set_seed, denormalize)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="CIFAR-10 模型评估")
    p.add_argument("--ckpt", default="checkpoints/best.pt", help="权重文件路径")
    p.add_argument("--data-root", default="data")
    p.add_argument("--out-dir", default="results")
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--device", default="auto")
    p.add_argument("--tta", action="store_true",
                   help="测试时增强：原图与水平翻转的预测概率取平均")
    p.add_argument("--n-samples", type=int, default=16, help="预测示例图的数量")
    p.add_argument("--tag", default="", help="输出文件名后缀，便于对比多个模型")
    return p.parse_args()


@torch.no_grad()
def collect_probs(model, loader, device, use_amp, tta: bool = False):
    """收集整个测试集的预测概率与标签。"""
    model.eval()
    all_probs, all_labels = [], []

    desc = "评估" + (" (TTA)" if tta else "")
    for images, targets in tqdm(loader, desc=desc, ncols=100, leave=False):
        images = images.to(device, non_blocking=True)
        with torch.amp.autocast("cuda", enabled=use_amp):
            logits = model(images)
            probs = torch.softmax(logits.float(), dim=1)
            if tta:
                logits_f = model(torch.flip(images, dims=[3]))
                probs = probs + torch.softmax(logits_f.float(), dim=1)
        all_probs.append(probs.cpu())
        all_labels.append(targets)

    return torch.cat(all_probs), torch.cat(all_labels)


def per_class_report(cm: np.ndarray, classes) -> list[dict]:
    """由混淆矩阵计算每类的 precision / recall / f1。"""
    rows = []
    for i, name in enumerate(classes):
        tp = cm[i, i]
        fp = cm[:, i].sum() - tp
        fn = cm[i, :].sum() - tp
        prec = tp / (tp + fp) if tp + fp else 0.0
        rec = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
        rows.append({
            "class": name,
            "name_zh": CLASSES_ZH[i] if i < len(CLASSES_ZH) else name,
            "correct": int(tp),
            "support": int(cm[i, :].sum()),
            "precision": round(float(prec), 4),
            "recall": round(float(rec), 4),
            "f1": round(float(f1), 4),
            "acc": round(float(rec), 4),
        })
    return rows


def main() -> None:
    args = parse_args()
    set_seed(42)

    device = get_device() if args.device == "auto" else torch.device(args.device)
    use_amp = device.type == "cuda"

    # ---- 载入权重 ----
    ckpt_path = Path(args.ckpt)
    if not ckpt_path.is_file():
        raise FileNotFoundError(f"找不到权重文件 {ckpt_path}，请先运行 train.py")
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    arch = ckpt.get("model", "resnet18")
    width = ckpt.get("width", 1.0)
    model = build_model(arch, num_classes=len(CLASSES), width=width).to(device)
    model.load_state_dict(ckpt["state_dict"])

    print(f"设备: {describe_device(device)}")
    print(f"权重: {ckpt_path}  ({arch}, {count_parameters(model):,} 参数)")
    if "best_acc" in ckpt:
        print(f"训练时的最佳测试准确率: {ckpt['best_acc']:.2f}% (epoch {ckpt.get('epoch')})")

    # ---- 数据（评估不使用增强）----
    _, test_loader = get_dataloaders(
        args.data_root, batch_size=args.batch_size, num_workers=args.workers,
        augment=False, download=True,
    )

    probs, labels = collect_probs(model, test_loader, device, use_amp, args.tta)
    preds = probs.argmax(dim=1)

    top1 = (preds == labels).float().mean().item() * 100
    top5 = accuracy(probs, labels, topk=(1, 5))[1]
    test_loss = nn.functional.cross_entropy(
        torch.log(probs.clamp_min(1e-12)), labels).item()

    # ---- 混淆矩阵 ----
    cm = np.zeros((len(CLASSES), len(CLASSES)), dtype=np.int64)
    for t, p in zip(labels.tolist(), preds.tolist()):
        cm[t, p] += 1

    report = per_class_report(cm, CLASSES)
    worst = sorted(report, key=lambda r: r["recall"])[:3]

    print("-" * 62)
    print(f"Top-1 准确率: {top1:.2f}%   Top-5 准确率: {top5:.2f}%   交叉熵: {test_loss:.4f}")
    if args.tta:
        print("（已启用水平翻转 TTA）")
    print("-" * 62)
    print(f"{'类别':<12}{'正确/总数':>12}{'precision':>11}{'recall':>9}{'f1':>8}")
    for r in report:
        print(f"{r['class']:<12}{r['correct']:>6}/{r['support']:<5}"
              f"{r['precision']:>11.4f}{r['recall']:>9.4f}{r['f1']:>8.4f}")
    print("-" * 62)
    print("召回率最低的三类: " + ", ".join(f"{r['class']} ({r['recall']*100:.1f}%)"
                                          for r in worst))

    # ---- 图表 ----
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    # 例如 --tag strong --tta -> "_strong_tta"，避免不同模型的结果互相覆盖
    tag = (f"_{args.tag}" if args.tag else "") + ("_tta" if args.tta else "")
    plot_confusion_matrix(cm, CLASSES, out_dir / f"confusion_matrix{tag}.png",
                          normalize=True,
                          title=f"CIFAR-10 confusion matrix (top-1 {top1:.2f}%)")

    # 预测示例：一半正确、一半错误，便于观察失败模式
    n = args.n_samples
    correct_idx = (preds == labels).nonzero(as_tuple=True)[0]
    wrong_idx = (preds != labels).nonzero(as_tuple=True)[0]
    n_wrong = min(len(wrong_idx), n // 2)
    n_correct = min(len(correct_idx), n - n_wrong)
    idx = torch.cat([correct_idx[:n_correct], wrong_idx[:n_wrong]])
    if len(idx) > 0:
        images = torch.stack([test_loader.dataset[int(i)][0] for i in idx])
        plot_predictions(images, labels[idx].tolist(), preds[idx].tolist(),
                         probs[idx].max(dim=1).values.tolist(), CLASSES,
                         out_dir / f"predictions{tag}.png")

    save_json({
        "checkpoint": str(ckpt_path),
        "model": arch,
        "params": count_parameters(model),
        "tta": args.tta,
        "top1": round(top1, 4),
        "top5": round(top5, 4),
        "test_loss": round(test_loss, 4),
        "per_class": report,
        "confusion_matrix": cm.tolist(),
    }, out_dir / f"eval{tag}.json")
    print(f"结果已写入 {out_dir}/eval{tag}.json、confusion_matrix{tag}.png、predictions{tag}.png")


if __name__ == "__main__":
    main()
