# CIFAR-10 图像分类：ResNet 从零训练

用 PyTorch 在 CIFAR-10 上从零训练 ResNet，包含一套完整的训练流程：数据准备、
数据增强、混合精度训练、学习率调度、权重平均，以及训练后的评估与可视化。

项目重点不在于堆砌技巧，而在于把每一步的取舍讲清楚——为什么这样改、效果如何、
代价是什么。代码里的注释解释了每个非显然的设计决策。

**最佳结果：ResNet-18（11.2M 参数）测试集 Top-1 准确率 96.43%，加水平翻转 TTA 后 96.90%。**

## 结果

| 配方 | 轮数 | Top-1 | Top-5 | 交叉熵 | TTA Top-1 | 训练时长 |
| --- | --- | --- | --- | --- | --- | --- |
| baseline | 100 | 95.19% | 99.57% | 0.2376 | — | 18.1 min |
| baseline | 200 | 95.49% | 99.61% | 0.2377 | — | 42.8 min |
| strong | 100 | 95.87% | 99.87% | 0.2766 | — | 24.2 min |
| **strong** | **200** | **96.43%** | 99.87% | 0.2594 | **96.90%** | 48.1 min |

全部为 ResNet-18，11,173,962 参数，单张 RTX 5070 Ti Laptop。随机种子固定为 42，
表内数字直接取自 `results/metrics_*.json` 与 `results/eval_*.json`。

![配方对比](results/curves_compare.png)

### 配方对比：把「配方」和「训练预算」分开看

4 次实验构成一个 2×2，可以区分到底是配方起作用还是单纯训练久了：

|  | 100 轮 | 200 轮 | 长训练的收益 |
| --- | --- | --- | --- |
| **baseline** | 95.19% | 95.49% | +0.30 |
| **strong** | 95.87% | 96.43% | +0.55 |
| **配方的收益** | **+0.68** | **+0.94** |  |

两个结论：

1. **强配方在两个训练预算下都更好**，而且轮数翻倍后优势从 +0.68 扩大到 +0.94。
   这符合强正则化的特性：它需要足够的训练步数，才能把额外的正则化转化成泛化收益。
   只跑 100 轮时，MixUp/CutMix 的收益还没有完全释放。

2. **单看交叉熵会得出错误结论**。baseline 的测试交叉熵反而更低（0.2376 对 0.2594），
   因为 strong 用了标签平滑和 MixUp/CutMix 的软标签，模型被显式训练成「不要过度
   自信」，交叉熵天然偏高。**准确率才是这里的决策指标**。

### TTA

对最佳模型加水平翻转 TTA（原图与翻转图的预测概率取平均）：

| | Top-1 | Top-5 |
| --- | --- | --- |
| 无 TTA | 96.43% | 99.87% |
| 水平翻转 TTA | **96.90%** | 99.92% |

**+0.47 个点**，代价只是推理时间翻倍，不增加任何训练成本。

### 每类表现

200 轮模型的每类召回率（%）：

| 类别 | airplane | automobile | bird | cat | deer | dog | frog | horse | ship | truck |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| baseline | 95.5 | 98.4 | 93.6 | 90.8 | 96.2 | 92.9 | 96.6 | 96.9 | 97.7 | 96.3 |
| strong | 96.9 | 98.8 | 95.5 | 91.4 | 96.7 | 94.2 | 98.3 | 97.4 | 97.3 | 97.8 |

**cat 始终是最难的类别**（90.8% → 91.4%），因为它和 dog 在 32×32 的分辨率下确实
难以区分。strong 配方在几乎所有类别上都有提升，提升最大的是 bird（+1.9）和
truck（+1.5）。

![混淆矩阵](results/confusion_matrix_strong200.png)

![预测示例](results/predictions_strong200.png)

### 关于精度差异的可信度

CIFAR-10 测试集只有 10000 张，**0.5% 以内的差异基本属于随机波动**。因此：

- baseline 100 轮与 200 轮的差距（+0.30）落在这个区间内，**不足以单独下结论**；
- 强配方相对基线的 +0.94 则超出了噪声范围；
- 要得到更严格的结论，需要固定多个随机种子重复实验并报告均值±标准差。

## 环境

| 项目 | 版本 |
| --- | --- |
| Python | 3.12 |
| PyTorch | 2.9.1+cu128 |
| torchvision | 0.24.1+cu128 |
| GPU | RTX 5070 Ti Laptop（12 GB，Blackwell sm_120） |
| 系统 | Windows 11 |

CUDA 12.8 是支持 RTX 40/50 系（含 Blackwell）的最低版本，用更早的 CUDA 构建会报
`no kernel image is available for execution`。

## 快速开始

### 1. 安装依赖

```bash
python -m venv .venv
source .venv/Scripts/activate        # Windows
# source .venv/bin/activate          # Linux / macOS

# PyTorch 的 CUDA 版本不在 PyPI 上，需要指定官方源；国内可用阿里云镜像
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
# pip install torch torchvision --index-url https://mirrors.aliyun.com/pytorch-wheels/cu128/

pip install -r requirements.txt
```

### 2. 准备数据

```bash
python prepare_data.py
```

详见下方「数据准备」一节。

### 3. 训练

```bash
# 基线配方：RandomCrop + Flip + Cutout
python train.py --recipe baseline --epochs 100 --amp --tag baseline

# 强配方：RandAugment + MixUp/CutMix + EMA
python train.py --recipe strong --epochs 200 --amp --tag strong200
```

单卡 RTX 5070 Ti Laptop 上的实测速度：baseline 约 12.8 s/轮，strong 约 14.4 s/轮
（200 轮分别是 42.8 和 48.1 分钟）。注意**第一个 epoch 会额外花约 100 秒**，用于
启动 DataLoader worker、反序列化数据集和 cuDNN 自动调优卷积算法。

### 4. 评估

```bash
python evaluate.py --ckpt checkpoints/best_strong200.pt --tag strong200
python evaluate.py --ckpt checkpoints/best_strong200.pt --tag strong200 --tta
```

评估会输出 Top-1 / Top-5 准确率、每类的 precision/recall/F1、混淆矩阵，
以及一张预测示例图（含分错的样本，便于观察失败模式）。

### 5. 对比多次实验

```bash
python compare.py
```

读取 `results/metrics_*.json`，绘制准确率曲线与最佳准确率的柱状对比图。

## 项目结构

| 文件 | 说明 |
| --- | --- |
| `model.py` | CIFAR 版 ResNet，支持 resnet20/18/34/50，可调宽度 |
| `data.py` | 数据集读取、数据增强（含 Cutout） |
| `augment.py` | 批级别增强：MixUp、CutMix，以及软标签损失 |
| `train.py` | 训练主流程，含两套配方预设 |
| `evaluate.py` | 评估、混淆矩阵、每类指标、TTA |
| `compare.py` | 汇总多次实验并绘制对比图 |
| `prepare_data.py` | 数据集下载与格式转换 |
| `utils.py` | 随机种子、指标统计、EMA、绘图 |
| `results/` | 训练曲线、混淆矩阵、指标 JSON、逐轮日志 CSV |
| `checkpoints/best.pt` | 最佳权重（strong 配方 200 轮，测试准确率 96.43%）|

## 实现要点

### 1. 为什么不用 `torchvision.models.resnet18`

torchvision 的 ResNet 是为 ImageNet（224×224）设计的，第一层是 7×7 stride-2
卷积加 3×3 maxpool，会把输入连降 4 倍分辨率。CIFAR-10 的图片只有 32×32，套用
这个结构后进入残差层时特征图只剩 8×8，跑完四个 stage 变成 1×1，精度会明显下降。

`model.py` 里改成了 CIFAR 版本：

- 主干换成 3×3 stride-1 卷积，不做 maxpool，保持 32×32；
- 第一个 stage 不下采样，其余 stage 步长为 2，最终特征图为 4×4；
- 全局平均池化后接单层全连接。

残差分支最后一层 BatchNorm 的权重初始化为 0，使每个残差块在训练初期近似恒等
映射，这样深层网络起步时梯度能稳定传播。

参数量：ResNet-18 为 11,173,962，ResNet-20 为 272,474。

### 2. 数据准备：为什么要自己写 Dataset

官方数据发布在 `www.cs.toronto.edu`，实测速度只有约 100 KB/s，170 MB 需要
二十多分钟，部分网络环境下还完全不可达。`prepare_data.py` 因此支持两个来源：

| 来源 | 说明 |
| --- | --- |
| `hf` | 从 HuggingFace 的 `uoft-cs/cifar10` 下载 parquet 再转换成本地 pickle 格式，实测约 4 MB/s |
| `tar` | 下载官方 `cifar-10-python.tar.gz`，依次尝试 fast.ai 的 S3 镜像和官方站点 |

转换后的目录结构与 torchvision 的标准布局一致（`cifar-10-batches-py/` 下有
5 个训练 batch、1 个测试 batch 和 `batches.meta`）。

但要注意：**转出来的文件字节与官方发布的不同，因此不能直接用
`torchvision.datasets.CIFAR10`**——它会校验每个 batch 文件的 md5，会直接报
"Dataset not found or corrupted"。`data.py` 里因此实现了 `CIFAR10Pickle`，
直接读 pickle 并只校验形状与标签取值范围（50000/10000 条、每类均衡、
标签落在 0-9）。

### 3. 训练策略

| 项目 | 设置 | 理由 |
| --- | --- | --- |
| 优化器 | SGD + Nesterov，momentum 0.9 | CIFAR 上 SGD 泛化性优于 Adam 系列 |
| 学习率 | 线性 warmup 5 轮 + 余弦退火 | 预热避免初期梯度爆炸，余弦让后期平稳收敛 |
| 权重衰减 | 5e-4，**不作用于** BN 权重和偏置 | 对归一化层做衰减会损害其表达能力 |
| 标签平滑 | 0.1 | 抑制过度自信，通常涨 0.2~0.5 点 |
| 混合精度 | AMP | 显存减半、速度提升，精度基本无损 |
| 梯度裁剪 | max_norm 5.0 | 防止个别 batch 的异常梯度破坏训练 |

### 4. 两套配方

`--recipe` 提供两套预设，命令行里显式指定的单项参数会覆盖预设。

**baseline**：`RandomCrop(32, padding=4)` + 水平翻转 + `Cutout(8)`。

**strong**：在强增强基础上加入 RandAugment 与 MixUp/CutMix，并用 EMA 权重做评估。

- **RandAugment**（`num_ops=2, magnitude=9`）：从一组几何与颜色变换里随机采样
  两个叠加，用强度参数控制幅度。相比手工设计增强组合，它把「选哪些增强、多强」
  变成两个超参数。
- **MixUp**：在一个 batch 内按 Beta 分布采样比例做线性插值。
- **CutMix**：把另一张图的一块矩形区域粘贴过来，标签按面积比例混合。
  相比 MixUp，它保留了局部真实像素，对定位能力更友好。
  两者各以 50% 概率启用，整体启用概率 0.5。
- 混合后的样本用软标签计算损失（两个交叉熵的加权和），因此训练时的准确率只能
  作为趋势参考，不是真实精度。
- **EMA**（指数滑动平均）：训练时维护权重的滑动平均副本，评估用这份权重。
  平均掉了单步噪声，通常涨 0.3~1 点，几乎不增加开销。衰减率 0.999，
  且训练初期会自动加速爬升，避免权重被锁死在初始值附近。

EMA 在这个项目里的作用相当明显：训练中期原始权重还在 86~90% 波动时，
EMA 权重已经稳定在 92% 以上；到训练末期两者收敛，差距缩小到 0.4 点左右。

## 复现说明

- 所有随机种子固定为 42（`--seed`），可通过 `--deterministic` 进一步启用
  确定性算法，但会变慢。
- 强配方中的 RandAugment 作用在 PIL 图像上，是 CPU 密集操作。实测 8 个
  DataLoader worker 的稳态吞吐约 31,000 img/s，远高于训练所需，因此数据加载
  不是瓶颈（训练时 GPU 利用率稳定在 90% 左右）。
- 训练时间取决于 GPU。如果第一轮特别慢，是 DataLoader worker 启动和 cuDNN
  自动调优的一次性开销，从第二轮起会稳定下来。

## 许可

MIT，见 [LICENSE](LICENSE)。
