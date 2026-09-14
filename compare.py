"""对比多次实验结果。

读取 ``results/`` 下训练产生的 ``metrics_*.json``，把各次实验的测试准确率曲线
画在一起，并给出最佳准确率的柱状对比。

用法::

    python compare.py
    python compare.py --results-dir results --out results/curves_compare.png

图内文字使用英文，原因见 ``utils.plot_history`` 的说明（默认字体不含中日韩字形）。
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

# 配方 -> 颜色，保证同一配方在多次实验里颜色一致
RECIPE_COLOR = {"baseline": "#4878a8", "strong": "#c1663f"}
RECIPE_LABEL = {"baseline": "baseline", "strong": "strong"}


def load_runs(results_dir: str | Path) -> list[tuple[str, dict]]:
    """读取所有 metrics_*.json，按 (配方, 轮数) 排序返回。"""
    runs = []
    for path in sorted(Path(results_dir).glob("metrics_*.json")):
        with path.open(encoding="utf-8") as f:
            metrics = json.load(f)
        tag = path.stem.replace("metrics_", "")
        runs.append((tag, metrics))
    runs.sort(key=lambda r: (r[1].get("recipe", ""), r[1].get("epochs", 0)))
    return runs


def main() -> None:
    parser = argparse.ArgumentParser(description="对比多次实验结果")
    parser.add_argument("--results-dir", default="results")
    parser.add_argument("--out", default="results/curves_compare.png")
    args = parser.parse_args()

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    runs = load_runs(args.results_dir)
    if not runs:
        raise SystemExit(f"{args.results_dir} 下没有找到 metrics_*.json，请先训练")

    fig, axes = plt.subplots(1, 2, figsize=(13.5, 5))

    # 左：测试准确率曲线
    for tag, m in runs:
        acc = m["history"]["test_acc"]
        epochs = np.arange(1, len(acc) + 1)
        recipe = m.get("recipe", "")
        style = "-" if recipe == "baseline" else "--"
        axes[0].plot(epochs, acc, style, linewidth=1.9,
                     color=RECIPE_COLOR.get(recipe),
                     label=f"{recipe} {m['epochs']}ep (best {m['best_acc']:.2f}%)")
    axes[0].set_xlabel("epoch")
    axes[0].set_ylabel("test accuracy (%)")
    axes[0].set_title("Test accuracy")
    axes[0].legend(fontsize=9, loc="lower right")
    axes[0].grid(alpha=0.3)

    # 右：最佳准确率柱状对比
    labels = [f"{m.get('recipe', '')}\n{m['epochs']}ep" for _, m in runs]
    values = [m["best_acc"] for _, m in runs]
    colors = [RECIPE_COLOR.get(m.get("recipe", ""), "#888888") for _, m in runs]
    bars = axes[1].bar(range(len(runs)), values, color=colors, width=0.62)
    axes[1].set_xticks(range(len(runs)))
    axes[1].set_xticklabels(labels, fontsize=9)
    axes[1].set_ylabel("best test accuracy (%)")
    axes[1].set_ylim(min(values) - 1.5, max(values) + 1.0)
    axes[1].set_title("Best accuracy")
    axes[1].grid(alpha=0.3, axis="y")
    for bar, value in zip(bars, values):
        axes[1].text(bar.get_x() + bar.get_width() / 2, value + 0.08,
                     f"{value:.2f}%", ha="center", va="bottom", fontsize=9)

    # 去掉 y 轴起点，避免小差异被视觉放大
    axes[1].set_ylim(min(values) - 1.5, max(values) + 0.9)

    fig.suptitle("ResNet-18 on CIFAR-10: baseline vs strong recipe", fontsize=13)
    fig.tight_layout()
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"对比图已写入 {out}")


if __name__ == "__main__":
    main()
