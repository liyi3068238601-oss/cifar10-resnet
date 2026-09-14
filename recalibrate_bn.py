"""对平均权重（EMA）重新估计 BatchNorm 统计量，并做三档对照。

用法::

    # 只做对照，不保存
    python recalibrate_bn.py --ckpt checkpoints/best_wrn28_10_ema.pt

    # 保存校准后的副本（不会覆盖原文件），并带上翻转 TTA
    python recalibrate_bn.py --ckpt checkpoints/best_wrn28_10_ema.pt \
                             --out checkpoints/best_wrn28_10_ema_bn.pt --tta

背景
----
EMA 的 BN running stats 是对训练过程中各步统计量的滑动平均，而那些统计量是在
**增强后**的图上算出来的——强增强、尤其 MixUp/CutMix 的混合图会明显改变激活的
均值与方差，与推理时看到的干净图像不一致。重新估计一次成本很低（只前向）。

这里的校准只用**训练图像**：
    --calib-data clean（默认）只做归一化，对应推理时的输入分布
    --calib-data train  用训练时的完整增强，复刻 PyTorch ``swa_utils.update_bn`` 的做法
两种都在方案里被讨论过，用实验决定，不预设哪种更好。

校准期间只让 BN 更新统计量：模型整体 ``eval()`` 关掉 Dropout/DropPath，再把 BN
单独切回 ``train()``。详见 ``utils.recalibrate_bn``。
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from data import CLASSES, build_transforms, get_datasets
from evaluate import collect_probs
from model import count_parameters
from utils import (describe_device, get_device, load_model_from_checkpoint,
                   recalibrate_bn, set_seed)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="BN 统计量重估与对照")
    p.add_argument("--ckpt", required=True, help="待校准的 checkpoint（通常是 EMA 权重）")
    p.add_argument("--out", default="", help="校准后权重的保存路径；留空则只做对照不保存")
    p.add_argument("--data-root", default="data")
    p.add_argument("--calib-data", default="clean", choices=["clean", "train"],
                   help="校准用哪些训练图像：clean=只归一化（默认），train=训练时的完整增强")
    p.add_argument("--calib-batches", type=int, default=0,
                   help="校准只用前多少个 batch（0 表示全部）")
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--tta", action="store_true", help="评估时加水平翻转 TTA")
    p.add_argument("--tag", default="", help="导出逐样本预测时的文件名后缀；默认用 checkpoint 名")
    p.add_argument("--out-dir", default="results")
    p.add_argument("--device", default="auto")
    return p.parse_args()


def build_calib_loader(args, ckpt: dict) -> DataLoader:
    """构造校准用的 DataLoader：训练集图像，按 --calib-data 决定是否加增强。

    ``--calib-data train`` 会读取 checkpoint 里的配方参数，复现该模型训练时**自己的**
    增强强度（而不是套用另一套配方的默认值），这样才是忠实的"用训练分布校准"。
    注意 MixUp/CutMix 属于批级别增强、不在 transform 流水线里，因此这里无法复现。
    """
    augment = args.calib_data == "train"
    train_args = ckpt.get("args", {}) or {}
    train_set, _ = get_datasets(
        args.data_root, augment=augment,
        cutout=int(train_args.get("cutout", 8)) if augment else 0,
        randaugment=int(train_args.get("randaugment", 0)) if augment else 0,
    )
    return DataLoader(train_set, batch_size=args.batch_size, shuffle=False,
                      num_workers=args.workers, pin_memory=True,
                      persistent_workers=args.workers > 0, drop_last=True)


def main() -> None:
    args = parse_args()
    set_seed(42)
    device = get_device() if args.device == "auto" else torch.device(args.device)
    ckpt_path = Path(args.ckpt)
    if not ckpt_path.is_file():
        raise SystemExit(f"找不到 {ckpt_path}")

    # 评估用测试集：永远只用官方 test split，且不加增强
    _, test_set = get_datasets(args.data_root, augment=False)
    test_loader = DataLoader(test_set, batch_size=256, shuffle=False,
                             num_workers=args.workers, pin_memory=True,
                             persistent_workers=args.workers > 0)

    model, ckpt = load_model_from_checkpoint(ckpt_path, device, len(CLASSES))
    print(f"设备  : {describe_device(device)}")
    print(f"权重  : {ckpt_path}")
    print(f"模型  : {ckpt.get('model')}  参数 {count_parameters(model):,}")
    print(f"记录  : best_acc={ckpt.get('best_acc')}  "
          f"来源={ckpt.get('best_acc_source', '?')}  ema={ckpt.get('ema')}")
    print(f"校准数据: {args.calib_data}")
    print("-" * 70)

    print("① 校准前（原 EMA 权重）")
    probs0, labels = collect_probs(model, test_loader, device,
                                   device.type == "cuda", args.tta)
    acc0 = (probs0.argmax(1) == labels).float().mean().item() * 100
    print(f"   Top-1 {acc0:.2f}%" + ("  (含 TTA)" if args.tta else ""))

    print()
    print("② 用训练图像重新估计 BN 统计量（只前向，不反向，Dropout/DropPath 关闭）")
    calib_loader = build_calib_loader(args, ckpt)
    n = recalibrate_bn(model, calib_loader, device, max_batches=args.calib_batches)

    print()
    print("③ 校准后")
    probs1, labels1 = collect_probs(model, test_loader, device,
                                    device.type == "cuda", args.tta)
    acc1 = (probs1.argmax(1) == labels1).float().mean().item() * 100
    print(f"   Top-1 {acc1:.2f}%" + ("  (含 TTA)" if args.tta else ""))

    print("-" * 70)
    print(f"校准前后差值: {acc1 - acc0:+.2f} 个百分点"
          f"  ({'提升' if acc1 > acc0 else '下降' if acc1 < acc0 else '持平'})")

    # 导出逐样本预测，便于用 paired_test.py 做 McNemar 检验——
    # 观测到的差值本身不能直接当作结论。
    tag = args.tag or ckpt_path.stem
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    before_npz = out_dir / f"preds_bn{tag}_before.npz"
    after_npz = out_dir / f"preds_bn{tag}_after.npz"
    np.savez_compressed(before_npz,
                        labels=labels.numpy().astype(np.int16),
                        preds=probs0.argmax(1).numpy().astype(np.int16),
                        probs=probs0.numpy().astype(np.float32))
    np.savez_compressed(after_npz,
                        labels=labels1.numpy().astype(np.int16),
                        preds=probs1.argmax(1).numpy().astype(np.int16),
                        probs=probs1.numpy().astype(np.float32))
    print(f"逐样本预测已写入 {before_npz} 与 {after_npz}")
    print(f"判显著性: python paired_test.py --a {before_npz} --b {after_npz} "
          f"--label-a 校准前 --label-b 校准后")

    if args.out:
        out_path = Path(args.out)
        if out_path.resolve() == ckpt_path.resolve():
            raise SystemExit("--out 不能与原文件相同：原权重必须保留")
        new_ckpt = dict(ckpt)
        new_ckpt["state_dict"] = model.state_dict()
        new_ckpt["bn_recalibrated"] = True
        new_ckpt["bn_calib_data"] = args.calib_data
        new_ckpt["bn_calib_batches"] = n
        new_ckpt["bn_calib_acc_before"] = round(acc0, 4)
        new_ckpt["bn_calib_acc_after"] = round(acc1, 4)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(new_ckpt, out_path)
        print(f"校准后的权重已写入 {out_path}（原文件未被修改）")


if __name__ == "__main__":
    main()
