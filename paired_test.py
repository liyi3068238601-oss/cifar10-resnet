"""比较两个模型在同一测试集上的表现（配对显著性检验）。

用法::

    python paired_test.py --a results/preds_strong200.npz \
                          --b results/preds_wrn28_10.npz --label-a ResNet-18 --label-b WRN-28-10

为什么需要这个脚本
------------------
不能用"单模型的二项置信区间"来判断两个模型谁更好。两个模型跑的是同一批测试图片，
它们的错误是配对的：同一张难图往往两个都错。因此差值的不确定性远小于把两次评估
当作独立样本时的估计。McNemar 检验只统计"不一致"的样本（A 错 B 对 / A 对 B 错），
在样本量只有 1 万的情况下，它的检验力明显高于朴素的区间比较。

``preds*.npz`` 由 ``evaluate.py`` 生成。
"""

from __future__ import annotations

import argparse
from pathlib import Path

from utils import load_correctness, mcnemar_test, paired_bootstrap_ci


def main() -> None:
    ap = argparse.ArgumentParser(description="两个模型的配对显著性检验")
    ap.add_argument("--a", required=True, help="模型 A 的 preds*.npz")
    ap.add_argument("--b", required=True, help="模型 B 的 preds*.npz")
    ap.add_argument("--label-a", default="A")
    ap.add_argument("--label-b", default="B")
    ap.add_argument("--n-boot", type=int, default=10000)
    args = ap.parse_args()

    for p in (args.a, args.b):
        if not Path(p).is_file():
            raise SystemExit(f"找不到 {p}，请先运行 evaluate.py 生成逐样本预测")

    ca, cb = load_correctness(args.a), load_correctness(args.b)
    if ca.shape != cb.shape:
        raise SystemExit(f"两个文件的样本数不一致: {ca.shape[0]} vs {cb.shape[0]}")
    # 同一份测试集必须顺序一致，否则配对检验没有意义
    la = __import__("numpy").load(args.a)["labels"]
    lb = __import__("numpy").load(args.b)["labels"]
    if not (la == lb).all():
        raise SystemExit("两个文件的标签顺序不一致，无法做配对检验")

    res = mcnemar_test(ca, cb)
    diff, lo, hi = paired_bootstrap_ci(ca, cb, n_boot=args.n_boot)

    print("=" * 74)
    print(f"测试集样本数: {len(ca)}")
    print(f"  {args.label_a:24s} Top-1 {res['acc_a']:.2f}%")
    print(f"  {args.label_b:24s} Top-1 {res['acc_b']:.2f}%")
    print("-" * 74)
    print("配对错误分解（只看两个模型判断不一致的样本）:")
    print(f"  {args.label_a} 对 而 {args.label_b} 错 : {res['n_a_only']:5d} 张")
    print(f"  {args.label_a} 错 而 {args.label_b} 对 : {res['n_b_only']:5d} 张")
    print(f"  不一致样本合计                  : {res['n_discordant']:5d} 张")
    print("-" * 74)
    print(f"准确率差值 ({args.label_b} − {args.label_a}): {res['acc_diff']:+.2f} 个百分点")
    print(f"配对 bootstrap 95% 置信区间      : [{lo:+.2f}, {hi:+.2f}]")
    print(f"McNemar 检验 p 值 ({res['method']}): {res['p_value']:.4g}")
    print("-" * 74)
    if res["p_value"] < 0.05:
        winner = args.label_b if res["acc_diff"] > 0 else args.label_a
        print(f"结论: 在 0.05 水平上差异显著，{winner} 更好。")
    else:
        print("结论: 差异未达到 0.05 显著性水平，现有证据不足以判定谁更好。")
    print("=" * 74)


if __name__ == "__main__":
    main()
