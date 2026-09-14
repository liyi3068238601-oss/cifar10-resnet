"""多模型概率集成（等权平均），并做配对显著性检验。

用法::

    python ensemble.py --inputs results/preds_wrn28_10_tta.npz results/preds_strong200_tta.npz \
                       --labels "WRN-28-10 + TTA" "ResNet-18 + TTA"

设计说明
--------
集成在**概率层面**做等权平均，不是把不同架构的权重平均（那没有意义）：

    p_ensemble(x) = (1/K) * sum_k p_k(x)

输入是 ``evaluate.py`` 导出的 ``preds*.npz``，因此不需要重新跑模型。每个成员如果
要用 TTA，应当**先各自完成原图与翻转图的概率平均**，再在模型之间平均。

必须注意的评估纪律
------------------
集成是否有效取决于成员之间**错误是否互补**，较弱的成员也可能拉低整体。因此：

* 不要在测试集上反复搜索权重或成员组合，直到恰好超过某个目标——那是选择性偏差，
  报告出来的数字会偏乐观。有独立验证集时才用验证集选方案。
* 本脚本会给出等权集成的显著性检验结果，与单模型对照，但**不会**替你搜索最优权重。
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from utils import mcnemar_test, paired_bootstrap_ci


def load_probs(path: str | Path):
    data = np.load(path)
    return (data["probs"].astype(np.float64), data["labels"].astype(np.int64))


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="等权概率集成 + 配对显著性检验")
    p.add_argument("--inputs", nargs="+", required=True, help="两个及以上的 preds*.npz")
    p.add_argument("--labels", nargs="+", default=None, help="每个成员的名字，用于报告")
    p.add_argument("--weights", nargs="+", type=float, default=None,
                   help="可选的权重；默认全部等权。不建议在测试集上调这组数")
    p.add_argument("--out", default="", help="集成结果 preds*.npz 的保存路径")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    if args.labels and len(args.labels) != len(args.inputs):
        raise SystemExit("--labels 的数量必须与 --inputs 一致")
    names = args.labels or [Path(p).stem for p in args.inputs]

    probs_list, labels_ref = [], None
    for path in args.inputs:
        if not Path(path).is_file():
            raise SystemExit(f"找不到 {path}")
        probs, labels = load_probs(path)
        if labels_ref is None:
            labels_ref = labels
        elif not np.array_equal(labels_ref, labels):
            raise SystemExit(f"{path} 的标签顺序与其它成员不一致，无法集成")
        probs_list.append(probs)

    weights = np.asarray(args.weights if args.weights else [1.0] * len(probs_list),
                         dtype=np.float64)
    weights = weights / weights.sum()

    print("=" * 78)
    print(f"测试集样本数: {len(labels_ref)}    成员数: {len(probs_list)}")
    print("-" * 78)
    print("各成员单独表现（单模型口径）:")
    correct_list = []
    for name, probs, w in zip(names, probs_list, weights):
        pred = probs.argmax(1)
        correct = (pred == labels_ref)
        correct_list.append(correct)
        print(f"  {name:28s} Top-1 {(correct.mean() * 100):.2f}%   权重 {w:.3f}")

    ens = np.tensordot(weights, np.stack(probs_list), axes=1)
    ens_pred = ens.argmax(1)
    ens_correct = (ens_pred == labels_ref)
    ens_acc = ens_correct.mean() * 100

    print("-" * 78)
    print(f"{'集成（等权概率平均）':28s} Top-1 {ens_acc:.2f}%")
    print("-" * 78)

    # 与最强成员做配对检验：只有超过噪声才能算真的变好
    best_i = int(np.argmax([c.mean() for c in correct_list]))
    res = mcnemar_test(correct_list[best_i], ens_correct)
    diff, lo, hi = paired_bootstrap_ci(correct_list[best_i], ens_correct)
    print(f"与最强成员对比（{names[best_i]}，单模型 {correct_list[best_i].mean() * 100:.2f}%）:")
    print(f"  不一致样本: {res['n_discordant']} 张 "
          f"(集成对/成员错 {res['n_b_only']}，成员对/集成错 {res['n_a_only']})")
    print(f"  准确率差值: {diff:+.2f} 个百分点   bootstrap 95% CI [{lo:+.2f}, {hi:+.2f}]")
    print(f"  McNemar p 值 ({res['method']}): {res['p_value']:.4g}")
    print("-" * 78)
    if res["p_value"] < 0.05:
        who = "集成" if diff > 0 else f"单模型 {names[best_i]}"
        print(f"结论: 差异显著，{who} 更好。")
    else:
        print("结论: 差异不显著——集成没有带来可判定的提升。")
    print()
    print("注意：以上是**多模型集成**口径，不能与「单模型单次前向」的成绩混写。")

    if args.out:
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(out,
                            labels=labels_ref.astype(np.int16),
                            preds=ens_pred.astype(np.int16),
                            probs=ens.astype(np.float32))
        print(f"集成概率已写入 {out}")


if __name__ == "__main__":
    main()
