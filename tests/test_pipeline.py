"""关键正确性的单元测试。

可以直接运行（无需 pytest）::

    python tests/test_pipeline.py

也可以用 pytest::

    pytest tests/ -v

覆盖的是「出错不会报错、只会静默给出错误结果」的那类问题：混合增强的 alpha 是否
真的到达了对应函数、CutMix 的 lam 与真实粘贴面积是否一致、DropPath 的期望是否保持、
TTA 概率是否归一化、配对检验的实现是否正确。
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import augment  # noqa: E402
from augment import cutmix, mixup, mixup_or_cutmix, soft_target_cross_entropy  # noqa: E402
from model import DropPath, build_model, count_parameters  # noqa: E402
from utils import mcnemar_test, paired_bootstrap_ci  # noqa: E402


# ---------------------------------------------------------------- 混合增强

def test_alpha_reaches_correct_function():
    """MixUp 与 CutMix 必须各自使用自己的 alpha（历史 bug：共用 MixUp 的 alpha）。"""
    calls = []
    orig_mixup, orig_cutmix = augment.mixup, augment.cutmix

    def fake_mixup(x, y, alpha=1.0):
        calls.append(("mixup", alpha))
        return x, y, y, 0.5

    def fake_cutmix(x, y, alpha=1.0):
        calls.append(("cutmix", alpha))
        return x, y, y, 0.5

    augment.mixup, augment.cutmix = fake_mixup, fake_cutmix
    try:
        x, y = torch.randn(4, 3, 8, 8), torch.tensor([0, 1, 2, 3])
        # switch=1.0 强制选 MixUp
        mixup_or_cutmix(x, y, mixup_alpha=0.2, cutmix_alpha=1.0, prob=1.0, switch=1.0)
        assert calls[-1] == ("mixup", 0.2), f"MixUp 收到的 alpha 不对: {calls[-1]}"
        # switch=0.0 强制选 CutMix —— 修复前这里会收到 0.2
        mixup_or_cutmix(x, y, mixup_alpha=0.2, cutmix_alpha=1.0, prob=1.0, switch=0.0)
        assert calls[-1] == ("cutmix", 1.0), f"CutMix 收到的 alpha 不对: {calls[-1]}"
    finally:
        augment.mixup, augment.cutmix = orig_mixup, orig_cutmix


def test_only_one_method_enabled():
    """关掉一个方法后，另一个必须被无条件选中，不受 switch 影响。"""
    calls = []
    orig_mixup, orig_cutmix = augment.mixup, augment.cutmix
    augment.mixup = lambda x, y, alpha=1.0: (calls.append("mixup"), (x, y, y, 0.5))[1]
    augment.cutmix = lambda x, y, alpha=1.0: (calls.append("cutmix"), (x, y, y, 0.5))[1]
    try:
        x, y = torch.randn(4, 3, 8, 8), torch.tensor([0, 1, 2, 3])
        for _ in range(20):  # switch=0.5，但 cutmix 已关闭
            mixup_or_cutmix(x, y, mixup_alpha=1.0, cutmix_alpha=0.0, prob=1.0, switch=0.5)
        assert set(calls) == {"mixup"}, f"关闭 CutMix 后仍调用了它: {set(calls)}"

        calls.clear()
        for _ in range(20):  # mixup 已关闭
            mixup_or_cutmix(x, y, mixup_alpha=0.0, cutmix_alpha=1.0, prob=1.0, switch=0.5)
        assert set(calls) == {"cutmix"}, f"关闭 MixUp 后仍调用了它: {set(calls)}"
    finally:
        augment.mixup, augment.cutmix = orig_mixup, orig_cutmix


def test_prob_zero_and_both_disabled_are_identity():
    """prob=0 或两种方法都关闭时，输入必须原样返回。"""
    x, y = torch.randn(4, 3, 8, 8), torch.tensor([0, 1, 2, 3])
    x0 = x.clone()
    xo, ya, yb, lam = mixup_or_cutmix(x, y, mixup_alpha=1.0, cutmix_alpha=1.0, prob=0.0)
    assert lam == 1.0 and torch.equal(x, xo) and torch.equal(ya, yb)

    xo, ya, yb, lam = mixup_or_cutmix(x, y, mixup_alpha=0.0, cutmix_alpha=0.0, prob=1.0)
    assert lam == 1.0 and torch.equal(x, xo), "两者都关闭时不应修改输入"
    assert torch.equal(x, x0)


def test_mixup_lambda_and_interpolation():
    """MixUp 的输出必须等于 lam*x + (1-lam)*x[perm]。"""
    torch.manual_seed(0)
    x = torch.arange(2 * 3 * 4 * 4, dtype=torch.float32).reshape(2, 3, 4, 4)
    y = torch.tensor([0, 1])
    xm, _, _, lam = mixup(x, y, alpha=1.0)
    assert 0.0 <= lam <= 1.0
    # 输出必须是两张图（或自身）的凸组合，值域不会超出输入范围
    assert xm.min() >= x.min() and xm.max() <= x.max()


def test_cutmix_lambda_matches_pasted_area():
    """CutMix 的 lam 必须与真实粘贴面积一致（碰到边界被裁时要重算）。

    约定：返回值 lam 是**原始标签 y_a 的权重**，因此粘贴进来、属于 y_b 的面积
    应当是 ``1 - lam``。这与 ``soft_target_cross_entropy`` 里
    ``lam * CE(y_a) + (1 - lam) * CE(y_b)`` 的用法对应。

    注意：小 batch 下随机排列可能把某张图映射到它自己，该样本等于没有混合
    （此时 y_a 与 y_b 相同，损失退化为普通交叉熵，无害）。这类样本无法从外部
    观测到粘贴面积，因此跳过；真实训练 batch 为 128/256，概率可忽略。
    """
    torch.manual_seed(1)
    checked = 0
    for _ in range(50):
        x = torch.zeros(4, 3, 32, 32)
        for i in range(4):
            x[i] = i + 1.0                     # 每张图填不同常数，便于识别来源
        y = torch.tensor([0, 1, 2, 3])
        xm, _, _, lam = cutmix(x, y, alpha=1.0)
        for i in range(4):
            changed = (xm[i] != x[i]).any(dim=0).float().mean().item()
            if changed == 0:
                continue                       # 抽到了自己，跳过
            assert abs((1.0 - lam) - changed) < 0.01, \
                f"1-lam={1 - lam:.4f} 与实际改变比例 {changed:.4f} 不符"
            checked += 1
    assert checked >= 50, f"有效样本太少，测试没有真正生效: {checked}"


def test_soft_target_loss_matches_ce_when_lambda_one():
    torch.manual_seed(0)
    logits, target = torch.randn(8, 10), torch.randint(0, 10, (8,))
    a = soft_target_cross_entropy(logits, target, target, 1.0, label_smoothing=0.0)
    b = torch.nn.functional.cross_entropy(logits, target, label_smoothing=0.0)
    assert torch.allclose(a, b), f"lam=1 时应退化为普通交叉熵: {a.item()} vs {b.item()}"


# ------------------------------------------------------------------ TTA 概率

def test_tta_probability_normalization():
    """TTA 必须取平均而不是相加，否则概率行和是 2（置信度会显示成 172%）。"""
    torch.manual_seed(0)
    p_orig = torch.softmax(torch.randn(64, 10), dim=1)
    p_flip = torch.softmax(torch.randn(64, 10), dim=1)

    wrong = p_orig + p_flip                       # 修复前的写法
    right = 0.5 * (p_orig + p_flip)               # 修复后的写法

    assert not torch.allclose(wrong.sum(1), torch.ones(64), atol=1e-4)
    assert abs(wrong.sum(1).mean().item() - 2.0) < 1e-5
    assert torch.allclose(right.sum(1), torch.ones(64), atol=1e-5)
    # 关键：两种写法给出的 argmax 相同，所以 Top-1 不受影响（报告的数字没错，只是图错了）
    assert torch.equal(wrong.argmax(1), right.argmax(1))


# ------------------------------------------------------------------ DropPath

def test_droppath_eval_is_noop():
    dp = DropPath(0.5).eval()
    x = torch.ones(1000, 4)
    outs = torch.stack([dp(x) for _ in range(5)])
    assert torch.equal(outs[0], x) and bool((outs == 1.0).all())


def test_droppath_train_is_stochastic_and_unbiased():
    torch.manual_seed(0)
    dp = DropPath(0.5).train()
    x = torch.ones(4000, 4)
    outs = torch.stack([dp(x) for _ in range(200)])
    zero_rate = (outs == 0).float().mean().item()
    mean = outs.mean().item()
    assert 0.45 < zero_rate < 0.55, f"丢弃比例异常: {zero_rate}"
    assert abs(mean - 1.0) < 0.05, f"缩放没有保持期望: {mean}"


def test_droppath_zero_is_deterministic():
    model = build_model("resnet20", drop_path=0.0).train()
    x = torch.randn(4, 3, 32, 32)
    assert torch.equal(model(x).detach(), model(x).detach())


def test_droppath_adds_no_parameters():
    assert count_parameters(build_model("resnet18", drop_path=0.0)) == \
           count_parameters(build_model("resnet18", drop_path=0.1))


# ------------------------------------------------------------ BN 重估

def test_recalibrate_bn_updates_stats_only():
    """BN 重估只改 running stats、不动权重；且过程必须确定（说明 Dropout/DropPath 已关闭）。"""
    from torch.utils.data import DataLoader, TensorDataset

    from utils import recalibrate_bn

    torch.manual_seed(0)
    # 刻意把 drop-path 开到 0.5：如果校准期间它没被关掉，两次结果就不会一致
    model = build_model("resnet20", drop_path=0.5).train()
    for m in model.modules():
        if isinstance(m, torch.nn.BatchNorm2d):
            m.running_mean.fill_(0.5)      # 人为设一个明显不对的统计量
            m.running_var.fill_(2.0)

    def split(sd):
        w = {k: v.clone() for k, v in sd.items()
             if "running" not in k and "num_batches_tracked" not in k}
        s = {k: v.clone() for k, v in sd.items() if "running" in k}
        return w, s

    w_before, s_before = split(model.state_dict())

    ds = TensorDataset(torch.randn(64, 3, 32, 32), torch.zeros(64, dtype=torch.long))
    loader = DataLoader(ds, batch_size=16)

    n = recalibrate_bn(model, loader, "cpu", verbose=False)
    assert n == 4, f"应累计 4 个 batch，实际 {n}"
    w_after, s_after = split(model.state_dict())

    for k in w_before:
        assert torch.equal(w_before[k], w_after[k]), f"可学习权重被改动了: {k}"
    assert any(not torch.allclose(s_before[k], s_after[k]) for k in s_before), \
        "running stats 没有被重新估计"

    # 再跑一次：结果必须逐位相同，否则说明校准过程里混入了随机性
    recalibrate_bn(model, loader, "cpu", verbose=False)
    _, s_again = split(model.state_dict())
    for k in s_after:
        assert torch.allclose(s_after[k], s_again[k]), \
            f"两次重估结果不一致，说明 Dropout/DropPath 没被关闭: {k}"


# ------------------------------------------------------------ 配对显著性检验

def test_mcnemar_symmetric_case_is_not_significant():
    """完全对称的不一致（各错各的同样多）不应判为显著。

    注意这里 n=50 走的是卡方近似分支，对称情形的精确二项 p 值为 1.0，
    而带连续性校正的卡方给出约 0.89 —— 两者都判定"不显著"，这里只要求后者。
    """
    a = np.array([1, 1, 0, 0] * 25, dtype=np.int8)
    b = np.array([1, 0, 1, 0] * 25, dtype=np.int8)
    res = mcnemar_test(a, b)
    assert res["n_a_only"] == res["n_b_only"] == 25
    assert res["p_value"] > 0.5, f"对称情形不应显著，得到 p={res['p_value']}"
    assert res["acc_diff"] == 0.0


def test_mcnemar_exact_branch_on_small_counts():
    """样本量小的时候走精确二项分支，对称情形应给出 p≈1。"""
    a = np.array([1, 1, 0, 0] * 3, dtype=np.int8)     # 12 个样本，不一致 6 个
    b = np.array([1, 0, 1, 0] * 3, dtype=np.int8)
    res = mcnemar_test(a, b)
    assert res["n_discordant"] == 6 and res["method"] == "exact_binomial"
    assert res["p_value"] > 0.9, f"精确分支的对称情形应接近 1，得到 {res['p_value']}"


def test_mcnemar_detects_clear_difference():
    """一方明显更好时应判为显著。"""
    n = 10000
    a = np.ones(n, dtype=np.int8)
    b = np.ones(n, dtype=np.int8)
    a[:300] = 0          # A 错 300 张
    b[:300] = 0
    b[:300] = 1          # B 把这些全修对了
    b[300:350] = 0       # B 另外错 50 张
    res = mcnemar_test(a, b)
    assert res["n_b_only"] == 300 and res["n_a_only"] == 50
    assert res["p_value"] < 1e-4, f"明显差异却未显著: p={res['p_value']}"


def test_mcnemar_identical_models():
    a = np.array([1, 0, 1, 1, 0] * 20, dtype=np.int8)
    res = mcnemar_test(a, a.copy())
    assert res["n_discordant"] == 0 and res["p_value"] == 1.0


def test_mcnemar_rejects_mismatched_length():
    try:
        mcnemar_test(np.ones(10, dtype=np.int8), np.ones(9, dtype=np.int8))
    except ValueError:
        return
    raise AssertionError("长度不一致时应抛出 ValueError")


def test_paired_bootstrap_interval_brackets_diff():
    rng = np.random.default_rng(0)
    n = 2000
    base = (rng.random(n) > 0.05).astype(np.int8)
    better = base.copy()
    flip = rng.random(n) < 0.02
    better[flip] = 1
    diff, lo, hi = paired_bootstrap_ci(base, better, n_boot=2000)
    assert lo <= diff <= hi, f"点估计 {diff} 不在区间 [{lo}, {hi}] 内"
    assert diff > 0


# ------------------------------------------------------------ 旧权重兼容性

def test_legacy_checkpoint_still_loads():
    """model.py 的重构不能破坏已发布的 checkpoint。"""
    ckpt_path = Path(__file__).resolve().parent.parent / "checkpoints" / "best.pt"
    if not ckpt_path.is_file():
        print("    (跳过：checkpoints/best.pt 不存在)")
        return
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    model = build_model(ckpt["model"], width=ckpt.get("width", 1.0),
                        drop_path=ckpt.get("drop_path", 0.0),
                        dropout=ckpt.get("dropout", 0.0))
    missing, unexpected = model.load_state_dict(ckpt["state_dict"], strict=False)
    assert not missing and not unexpected, f"结构不一致: {missing[:3]} / {unexpected[:3]}"


def _run_all() -> int:
    tests = [(n, f) for n, f in sorted(globals().items())
             if n.startswith("test_") and callable(f)]
    failed = 0
    for name, fn in tests:
        try:
            fn()
            print(f"  PASS  {name}")
        except Exception as exc:  # noqa: BLE001 - 测试运行器需要报告所有失败
            failed += 1
            print(f"  FAIL  {name}: {type(exc).__name__}: {exc}")
    print(f"\n{len(tests) - failed}/{len(tests)} 通过")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(_run_all())
