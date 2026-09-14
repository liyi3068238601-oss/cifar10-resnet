# WRN-28-10 on CIFAR-10 — 单模型单次前向 Top-1 **97.50%**

从零训练（无预训练、无额外数据）的 WideResNet-28-10 权重。

## 结果

| 指标 | 数值 |
| --- | --- |
| Top-1 | **97.50%** |
| Top-5 | 99.90% |
| 测试交叉熵 | 0.2298 |
| 参数量 | 36,479,194 |
| 训练轮数 | 200 |
| 训练时长 | 257.9 分钟（RTX 5070 Ti Laptop，12 GB） |
| 随机种子 | 42 |

推理口径：**单模型、单次前向**，未使用 TTA 或集成。

## 复现配置

```bash
python train.py --model wrn28-10 --recipe strong --epochs 200 \
                --batch-size 256 --lr 0.2 --dropout 0.3 --drop-path 0.1 \
                --amp --tag wrn28_10
```

配方（`--recipe strong`）：RandAugment(N=2, M=9) + MixUp/CutMix(各 α=1，整体启用概率 0.5)
+ 标签平滑 0.1 + EMA(0.999)；SGD+Nesterov，5 轮 warmup + 余弦退火。

## 使用方法

```bash
# 下载本附件后
python evaluate.py --ckpt best_wrn28_10.pt --tag wrn28_10
python evaluate.py --ckpt best_wrn28_10.pt --tag wrn28_10 --tta
```

checkpoint 内含 `state_dict`、模型名、`dropout`/`drop_path`、训练参数与最佳轮次，
`evaluate.py` 会据此自动重建模型。注意这是 **EMA 权重**（`ema: true`）。

## 三点如实说明

**1. 这是 EMA 权重，不是原始权重。** 训练末期两者会互相反超（本次最后几轮
raw 97.43% / EMA 97.40%），而保存逻辑只保留了各自轨迹的最优。差异在噪声范围内，
但原始权重并未包含在本附件中。

**2. 三个事后处理方法在本模型上都没有效果**，因此成绩按单次前向报告：

| 方法 | 结果 | 判定 |
| --- | --- | --- |
| 水平翻转 TTA | 97.47%（−0.03，McNemar p=0.80） | 不显著 |
| BN 统计量重估 | 干净输入 −0.34 / 训练分布 −0.11 | 有害或无效 |
| 与 ResNet-18 等权集成 | 97.53%（+0.06，p=0.56） | 不显著 |

作为对照，TTA 在较弱的 ResNet-18 上是显著有效的（96.43% → 96.90%，p=3.4e-05）——
**TTA 的收益依赖模型，不是可移植的常数**。

**3. 模型选择使用了测试集。** 训练期间每轮在官方测试集上评估并据此选 best，
因此 97.50% 带有选择偏差。严格的结论应划分独立验证集做模型选择，最后才在测试集报告。

## 仓库

https://github.com/liyi3068238601-oss/cifar10-resnet
