"""CIFAR-10 训练脚本。

用法示例::

    # 基线配方：crop + flip + cutout
    python train.py --recipe baseline --epochs 100

    # 强配方：RandAugment + MixUp/CutMix + EMA
    python train.py --recipe strong --epochs 100 --tag strong

    # 完全手动指定
    python train.py --epochs 100 --mixup 1.0 --cutmix 1.0 --randaugment 9 --ema-decay 0.999

训练策略:
    * 优化器: SGD + Nesterov 动量，weight decay 5e-4（BN 权重与偏置除外）
    * 学习率: 线性 warmup + 余弦退火
    * 混合精度: AMP (仅在 CUDA 上启用)
    * 正则:   见 ``--recipe`` 预设
"""

from __future__ import annotations

import argparse
import math
import time
from pathlib import Path

import torch
import torch.nn as nn
from tqdm import tqdm

from augment import mixup_or_cutmix, soft_target_cross_entropy
from data import CLASSES, get_dataloaders
from model import build_model, count_parameters
from utils import (AverageMeter, CSVLogger, ModelEMA, accuracy, describe_device,
                   get_device, plot_history, save_json, set_seed)

# 两套预设配方。命令行未显式指定的项由预设填充。
RECIPE_PRESETS = {
    "baseline": {
        "cutout": 8, "randaugment": 0,
        "mixup": 0.0, "cutmix": 0.0, "mix_prob": 0.0,
        "ema_decay": 0.0,
    },
    "strong": {
        "cutout": 0, "randaugment": 9,
        "mixup": 1.0, "cutmix": 1.0, "mix_prob": 0.5,
        "ema_decay": 0.999,
    },
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="在 CIFAR-10 上训练 ResNet")
    # 模型
    p.add_argument("--model", default="resnet18",
                   choices=["resnet20", "resnet18", "resnet34", "resnet50"],
                   help="网络结构")
    p.add_argument("--width", type=float, default=1.0, help="通道数缩放系数")
    # 训练
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--lr", type=float, default=0.1, help="初始学习率（也是余弦退火的峰值）")
    p.add_argument("--momentum", type=float, default=0.9)
    p.add_argument("--weight-decay", type=float, default=5e-4)
    p.add_argument("--warmup-epochs", type=float, default=5.0, help="线性预热轮数")
    p.add_argument("--label-smoothing", type=float, default=0.1)
    p.add_argument("--no-augment", action="store_true", help="关闭训练集增强（用于对照实验）")
    # 正则与增强（默认 None 表示由 --recipe 预设决定）
    p.add_argument("--recipe", default="baseline", choices=sorted(RECIPE_PRESETS),
                   help="配方预设；显式指定的单项参数会覆盖预设")
    p.add_argument("--cutout", type=int, default=None, help="Cutout 边长，0 表示关闭")
    p.add_argument("--randaugment", type=int, default=None,
                   help="RandAugment 强度（推荐 9），0 表示关闭")
    p.add_argument("--mixup", type=float, default=None, help="MixUp 的 alpha，0 表示关闭")
    p.add_argument("--cutmix", type=float, default=None, help="CutMix 的 alpha，0 表示关闭")
    p.add_argument("--mix-prob", type=float, default=None, help="每个 batch 启用混合的概率")
    p.add_argument("--ema-decay", type=float, default=None,
                   help="权重的指数滑动平均衰减率，0 表示关闭（推荐 0.999）")
    # 运行
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", default="auto", help="auto/cuda/cpu")
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--amp", action="store_true", help="启用混合精度训练（推荐 CUDA 下开启）")
    p.add_argument("--deterministic", action="store_true", help="使用确定性算法（更慢）")
    # 输出
    p.add_argument("--data-root", default="data")
    p.add_argument("--out-dir", default="results", help="日志与图表输出目录")
    p.add_argument("--ckpt-dir", default="checkpoints", help="权重保存目录")
    p.add_argument("--tag", default="", help="输出文件名后缀，便于区分多次实验")
    # 快速自检
    p.add_argument("--max-steps", type=int, default=0,
                   help="每个 epoch 最多训练多少 step（0 表示不限，用于快速自检）")
    args = p.parse_args()

    # 用配方预设补全未显式指定的项
    preset = RECIPE_PRESETS[args.recipe]
    for key, value in preset.items():
        attr = key.replace("-", "_")
        if getattr(args, attr) is None:
            setattr(args, attr, value)
    return args


def build_optimizer(model: nn.Module, args: argparse.Namespace) -> torch.optim.Optimizer:
    """SGD + Nesterov。BN 与偏置不做 weight decay。"""
    decay, no_decay = [], []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if param.ndim == 1 or name.endswith(".bias"):  # BN 权重、偏置
            no_decay.append(param)
        else:
            decay.append(param)
    groups = [
        {"params": decay, "weight_decay": args.weight_decay},
        {"params": no_decay, "weight_decay": 0.0},
    ]
    return torch.optim.SGD(groups, lr=args.lr, momentum=args.momentum, nesterov=True)


def build_scheduler(optimizer: torch.optim.Optimizer, args: argparse.Namespace,
                    steps_per_epoch: int):
    """线性 warmup + 余弦退火，按 step 更新。"""
    total_steps = max(1, args.epochs * steps_per_epoch)
    warmup_steps = max(1, int(args.warmup_epochs * steps_per_epoch))

    def lr_lambda(step: int) -> float:
        if step < warmup_steps:
            # 从 0.01 线性升到 1.0，避免训练初期梯度爆炸
            return 0.01 + 0.99 * step / warmup_steps
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        progress = min(1.0, progress)
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def train_one_epoch(model, loader, args, optimizer, scheduler, scaler, ema,
                    device, epoch, use_amp) -> tuple[float, float, float]:
    """训练一个 epoch。

    Returns:
        ``(loss, acc, 混合样本占比)``。启用 MixUp/CutMix 时 acc 是相对主导
        标签算出来的，只能作为趋势参考。
    """
    model.train()
    loss_meter, acc_meter = AverageMeter(window=50), AverageMeter(window=50)
    mixed_count = 0

    use_mix = args.mix_prob > 0 and (args.mixup > 0 or args.cutmix > 0)
    desc = f"Epoch {epoch:3d}/{args.epochs}"
    pbar = tqdm(loader, desc=desc, ncols=118, leave=False, dynamic_ncols=True)

    for step, (images, targets) in enumerate(pbar):
        if args.max_steps and step >= args.max_steps:
            break
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        if use_mix:
            # 混合系数按 MixUp/CutMix 各自的比例随机选择
            alpha = args.mixup if args.mixup > 0 else args.cutmix
            switch = 1.0 if args.cutmix <= 0 else (0.0 if args.mixup <= 0 else 0.5)
            images, y_a, y_b, lam = mixup_or_cutmix(
                images, targets, alpha=alpha, prob=args.mix_prob, switch=switch)
            if lam < 1.0:
                mixed_count += 1
        else:
            y_a, y_b, lam = targets, targets, 1.0

        with torch.amp.autocast("cuda", enabled=use_amp):
            logits = model(images)
            loss = soft_target_cross_entropy(logits, y_a, y_b, lam,
                                             args.label_smoothing)

        optimizer.zero_grad(set_to_none=True)
        if scaler is not None:
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()
        scheduler.step()
        if ema is not None:
            ema.update(model)

        bs = images.size(0)
        acc1 = accuracy(logits.detach().float(), y_a, topk=(1,))[0]
        loss_meter.update(loss.item(), bs)
        acc_meter.update(acc1, bs)

        pbar.set_postfix(loss=f"{loss_meter.recent:.4f}", acc=f"{acc_meter.recent:.2f}%",
                         lr=f"{scheduler.get_last_lr()[0]:.4f}")

    mixed_ratio = mixed_count / max(1, min(len(loader), args.max_steps or len(loader)))
    return loss_meter.avg, acc_meter.avg, mixed_ratio


@torch.no_grad()
def evaluate(model, loader, criterion, device, use_amp=False) -> tuple[float, float]:
    """返回 (loss, top1)。"""
    model.eval()
    loss_meter, acc_meter = AverageMeter(), AverageMeter()
    for images, targets in loader:
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        with torch.amp.autocast("cuda", enabled=use_amp):
            logits = model(images)
            loss = criterion(logits, targets)
        bs = images.size(0)
        loss_meter.update(loss.item(), bs)
        acc_meter.update(accuracy(logits.float(), targets, topk=(1,))[0], bs)
    return loss_meter.avg, acc_meter.avg


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir)
    ckpt_dir = Path(args.ckpt_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    suffix = f"_{args.tag}" if args.tag else ""

    set_seed(args.seed, deterministic=args.deterministic)

    # ---- 设备 ----
    device = get_device() if args.device == "auto" else torch.device(args.device)
    use_amp = bool(args.amp and device.type == "cuda")
    print(f"设备: {describe_device(device)}   混合精度: {'开' if use_amp else '关'}")
    print(f"随机种子: {args.seed}   配方: {args.recipe}")

    # ---- 数据 ----
    train_loader, test_loader = get_dataloaders(
        args.data_root, batch_size=args.batch_size, num_workers=args.workers,
        augment=not args.no_augment, cutout=args.cutout,
        randaugment=args.randaugment,
    )
    print(f"训练集 {len(train_loader.dataset)} 张 / 测试集 {len(test_loader.dataset)} 张")

    # ---- 模型 ----
    model = build_model(args.model, num_classes=len(CLASSES), width=args.width).to(device)
    n_params = count_parameters(model)

    criterion = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)
    optimizer = build_optimizer(model, args)
    steps_per_epoch = args.max_steps or len(train_loader)
    scheduler = build_scheduler(optimizer, args, steps_per_epoch)
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp) if use_amp else None
    ema = ModelEMA(model, args.ema_decay) if args.ema_decay > 0 else None

    aug_parts = []
    if args.no_augment:
        aug_parts.append("关闭")
    else:
        aug_parts.append("crop+flip")
        if args.randaugment > 0:
            aug_parts.append(f"RandAugment(N=2,M={args.randaugment})")
        if args.cutout > 0:
            aug_parts.append(f"Cutout({args.cutout})")
        if args.mix_prob > 0:
            aug_parts.append(f"MixUp/CutMix(p={args.mix_prob}, alpha={args.mixup or args.cutmix})")
    aug_desc = " + ".join(aug_parts)

    print(f"模型: {args.model} (width={args.width})   参数量: {n_params:,}")
    print(f"优化器: SGD(lr={args.lr}, momentum={args.momentum}, wd={args.weight_decay}, "
          f"nesterov)   调度: warmup {args.warmup_epochs}ep + cosine")
    print(f"增强: {aug_desc}")
    print(f"标签平滑: {args.label_smoothing}   EMA: "
          f"{f'decay={args.ema_decay}' if ema else '关闭'}")
    print("-" * 96)

    history = {k: [] for k in ("train_loss", "train_acc", "test_loss", "test_acc",
                               "raw_acc", "lr")}
    logger = CSVLogger(out_dir / f"history{suffix}.csv",
                       ["epoch", "lr", "train_loss", "train_acc", "test_loss",
                        "test_acc", "raw_acc", "mixed_ratio", "epoch_time", "best_acc"])
    best_acc, best_epoch = 0.0, 0
    ckpt_path = ckpt_dir / f"best{suffix}.pt"
    total_start = time.time()

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        train_loss, train_acc, mixed_ratio = train_one_epoch(
            model, train_loader, args, optimizer, scheduler, scaler, ema,
            device, epoch, use_amp)

        eval_model = ema.module if ema is not None else model
        test_loss, test_acc = evaluate(eval_model, test_loader, criterion, device, use_amp)
        raw_acc = test_acc
        if ema is not None:
            _, raw_acc = evaluate(model, test_loader, criterion, device, use_amp)

        dt = time.time() - t0
        lr_now = scheduler.get_last_lr()[0]
        for key, val in (("train_loss", train_loss), ("train_acc", train_acc),
                         ("test_loss", test_loss), ("test_acc", test_acc),
                         ("raw_acc", raw_acc), ("lr", lr_now)):
            history[key].append(val)

        improved = test_acc > best_acc
        if improved:
            best_acc, best_epoch = test_acc, epoch
            torch.save({
                "model": args.model,
                "width": args.width,
                "state_dict": eval_model.state_dict(),
                "epoch": epoch,
                "best_acc": best_acc,
                "recipe": args.recipe,
                "ema": ema is not None,
                "args": vars(args),
            }, ckpt_path)

        logger.log({
            "epoch": epoch, "lr": f"{lr_now:.6f}",
            "train_loss": f"{train_loss:.4f}", "train_acc": f"{train_acc:.2f}",
            "test_loss": f"{test_loss:.4f}", "test_acc": f"{test_acc:.2f}",
            "raw_acc": f"{raw_acc:.2f}", "mixed_ratio": f"{mixed_ratio:.3f}",
            "epoch_time": f"{dt:.1f}", "best_acc": f"{best_acc:.2f}",
        })

        flag = "  *" if improved else ""
        ema_note = f" (raw {raw_acc:5.2f}%)" if ema is not None else ""
        print(f"Epoch {epoch:3d}/{args.epochs} | lr {lr_now:.4f} | "
              f"train loss {train_loss:.4f} acc {train_acc:5.2f}% | "
              f"test loss {test_loss:.4f} acc {test_acc:5.2f}%{ema_note} | "
              f"{dt:5.1f}s | best {best_acc:5.2f}%{flag}")

    total_time = time.time() - total_start
    print("-" * 96)
    print(f"训练完成，用时 {total_time / 60:.1f} 分钟 "
          f"({total_time / max(1, args.epochs):.1f}s/epoch)")
    print(f"最佳测试准确率: {best_acc:.2f}% (epoch {best_epoch})  ->  {ckpt_path}")

    plot_history(history, out_dir / f"curves{suffix}.png",
                 title=f"{args.model} on CIFAR-10 ({args.recipe})")
    save_json({
        "model": args.model,
        "width": args.width,
        "params": n_params,
        "recipe": args.recipe,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "lr": args.lr,
        "weight_decay": args.weight_decay,
        "label_smoothing": args.label_smoothing,
        "cutout": args.cutout,
        "randaugment": args.randaugment,
        "mixup": args.mixup,
        "cutmix": args.cutmix,
        "mix_prob": args.mix_prob,
        "ema_decay": args.ema_decay,
        "augment": not args.no_augment,
        "amp": use_amp,
        "seed": args.seed,
        "device": describe_device(device),
        "best_acc": round(best_acc, 4),
        "best_epoch": best_epoch,
        "final_acc": round(history["test_acc"][-1], 4),
        "total_minutes": round(total_time / 60, 2),
        "history": history,
    }, out_dir / f"metrics{suffix}.json")


if __name__ == "__main__":
    main()
