"""CIFAR-10 数据准备：下载、格式转换、完整性校验。

最终产物是 torchvision 的标准目录结构，之后 ``torchvision.datasets.CIFAR10``
可以直接以 ``download=False`` 读取::

    <root>/cifar-10-batches-py/
        data_batch_1 ... data_batch_5     # 训练集，各 10000 张
        test_batch                        # 测试集，10000 张
        batches.meta                      # 类别名

支持两种数据来源，通过 ``source`` 选择：

``hf``
    从 HuggingFace 的 ``uoft-cs/cifar10`` 下载 parquet 再转成本地 pickle 格式。
    实测速度最快（约 4 MB/s），且国内可直连，因此是 ``auto`` 的首选。

``tar``
    下载官方 ``cifar-10-python.tar.gz``。官方站点
    ``www.cs.toronto.edu`` 在部分网络环境下只有约 100 KB/s，
    因此同时准备了 fast.ai 的 S3 镜像作为备选。

``auto``
    先试 ``hf``，失败再试 ``tar``。
"""

from __future__ import annotations

import hashlib
import pickle
import shutil
import tarfile
import urllib.error
import urllib.request
from pathlib import Path
from typing import Iterable, List, Sequence

import numpy as np

# ---------------------------------------------------------------- 数据源配置

_HF_BASE = "https://huggingface.co/datasets/uoft-cs/cifar10/resolve/main/plain_text"
HF_PARQUET = {
    "train": f"{_HF_BASE}/train-00000-of-00001.parquet",
    "test": f"{_HF_BASE}/test-00000-of-00001.parquet",
}

# 按优先级排列的 tar.gz 镜像
TAR_MIRRORS: List[tuple[str, str]] = [
    ("fast.ai (S3 镜像)",
     "https://s3.amazonaws.com/fast-ai-imageclas/cifar10.tgz"),
    ("toronto (官方)",
     "https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz"),
]

# 官方 tar.gz 的 md5，仅作日志参考（镜像可能是重新打包的，md5 会不同）
OFFICIAL_MD5 = "c58f30108f718f92721af3b95e74349a"

LABEL_NAMES = ["airplane", "automobile", "bird", "cat", "deer",
               "dog", "frog", "horse", "ship", "truck"]

BATCH_FILES = ["data_batch_1", "data_batch_2", "data_batch_3",
               "data_batch_4", "data_batch_5", "test_batch", "batches.meta"]

N_TRAIN, N_TEST = 50000, 10000


def _log(msg: str) -> None:
    print(f"[data] {msg}", flush=True)


def _md5(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.md5()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(chunk), b""):
            h.update(block)
    return h.hexdigest()


# ---------------------------------------------------------------- 通用下载

def _download(url: str, dest: Path, timeout: int = 60) -> None:
    """带百分比进度输出的下载。"""
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        total = int(resp.headers.get("Content-Length") or 0)
        done, next_report = 0, 10
        with dest.open("wb") as f:
            while True:
                block = resp.read(1 << 20)
                if not block:
                    break
                f.write(block)
                done += len(block)
                if total:
                    pct = done * 100 / total
                    if pct >= next_report:
                        _log(f"    {pct:5.1f}%  ({done / 1e6:.1f}/{total / 1e6:.1f} MB)")
                        next_report += 10
    _log(f"    完成，{dest.stat().st_size / 1e6:.1f} MB")


# ---------------------------------------------------------------- 来源一: HF parquet

def _read_parquet(path: Path) -> tuple[np.ndarray, np.ndarray]:
    """读取 HF parquet，返回 ``(images, labels)``。

    images: uint8 ``(N, 32, 32, 3)``；labels: int64 ``(N,)``。
    """
    from io import BytesIO

    import pyarrow as pa
    import pyarrow.parquet as pq
    from PIL import Image

    table = pq.read_table(path)
    cols = table.column_names

    # 图像列：可能是 struct{bytes,path}（HF Image 特性）也可能是裸 binary
    img_col = next((c for c in cols if c in ("img", "image", "images")), cols[0])
    field = table[img_col]
    if hasattr(field, "combine_chunks"):
        field = field.combine_chunks()
    if pa.types.is_struct(field.type):
        field = field.field("bytes")
    blobs = field.to_pylist()

    # 标签列
    lab_col = next((c for c in cols if c in ("label", "labels", "fine_label")),
                   next(c for c in cols if c != img_col))
    labels = np.asarray(table[lab_col].to_pylist(), dtype=np.int64)

    images = np.empty((len(blobs), 32, 32, 3), dtype=np.uint8)
    for i, blob in enumerate(blobs):
        # PNG 是无损的，解出来的像素与原始 CIFAR 数据完全一致
        img = Image.open(BytesIO(blob)).convert("RGB")
        if img.size != (32, 32):
            raise ValueError(f"第 {i} 张图片尺寸异常: {img.size}，期望 (32, 32)")
        images[i] = np.asarray(img, dtype=np.uint8)

    return images, labels


def _write_batches(out_dir: Path, train_x: np.ndarray, train_y: np.ndarray,
                   test_x: np.ndarray, test_y: np.ndarray) -> None:
    """把数组写成 torchvision 期望的 pickle 批文件。"""
    out_dir.mkdir(parents=True, exist_ok=True)

    def flatten(x: np.ndarray) -> np.ndarray:
        # (N, 32, 32, 3) -> (N, 3072)，通道顺序 R|G|B，与官方格式一致
        return np.ascontiguousarray(x.transpose(0, 3, 1, 2).reshape(len(x), -1))

    per_batch = len(train_x) // 5
    for i in range(5):
        sl = slice(i * per_batch, (i + 1) * per_batch)
        batch = {
            "batch_label": f"training batch {i + 1} of 5",
            "labels": train_y[sl].tolist(),
            "data": flatten(train_x[sl]),
            "filenames": [f"train_{j}.png" for j in range(sl.start, sl.stop)],
        }
        with (out_dir / f"data_batch_{i + 1}").open("wb") as f:
            pickle.dump(batch, f, protocol=4)

    test = {
        "batch_label": "testing batch 1 of 1",
        "labels": test_y.tolist(),
        "data": flatten(test_x),
        "filenames": [f"test_{j}.png" for j in range(len(test_x))],
    }
    with (out_dir / "test_batch").open("wb") as f:
        pickle.dump(test, f, protocol=4)

    meta = {
        "num_cases_per_batch": per_batch,
        "label_names": LABEL_NAMES,
        "num_vis": 3072,
    }
    with (out_dir / "batches.meta").open("wb") as f:
        pickle.dump(meta, f, protocol=4)


def _prepare_from_hf(root: Path) -> None:
    """下载 HF parquet 并转换成 torchvision 目录结构。"""
    cache = root / "hf"
    cache.mkdir(parents=True, exist_ok=True)
    arrays = {}
    for split, url in HF_PARQUET.items():
        dest = cache / f"{split}.parquet"
        if not dest.is_file():
            _log(f"  下载 {split} parquet…")
            _download(url, dest)
        else:
            _log(f"  复用已下载的 {dest.name}")
        images, labels = _read_parquet(dest)
        _log(f"  {split}: {len(images)} 张")
        arrays[split] = (images, labels)

    train_x, train_y = arrays["train"]
    test_x, test_y = arrays["test"]
    if (len(train_x), len(test_x)) != (N_TRAIN, N_TEST):
        raise RuntimeError(f"样本数异常: train={len(train_x)}, test={len(test_x)}")
    if sorted(np.unique(train_y).tolist()) != list(range(10)):
        raise RuntimeError(f"标签取值异常: {np.unique(train_y).tolist()}")

    _log("  转换为 torchvision 批格式…")
    _write_batches(root / "cifar-10-batches-py", train_x, train_y, test_x, test_y)


# ---------------------------------------------------------------- 来源二: tar.gz

def _extract(tar_path: Path, root: Path) -> None:
    with tarfile.open(tar_path, "r:*") as tf:
        members = []
        for m in tf.getmembers():
            p = Path(m.name)
            if p.is_absolute() or ".." in p.parts:
                raise RuntimeError(f"压缩包内含可疑路径: {m.name}")
            members.append(m)
        tf.extractall(root, members=members)


def _prepare_from_tar(root: Path, mirrors: Iterable[tuple[str, str]]) -> None:
    """从 tar.gz 镜像下载并解压。"""
    tar_path = root / "cifar10.tar.gz"
    errors: List[str] = []
    for name, url in mirrors:
        _log(f"  尝试镜像: {name}")
        try:
            _download(url, tar_path)
            digest = _md5(tar_path)
            note = " (与官方 md5 一致)" if digest == OFFICIAL_MD5 else ""
            _log(f"    md5={digest}{note}")
            _log("    解压中…")
            _extract(tar_path, root)
            return
        except (urllib.error.URLError, OSError, RuntimeError) as exc:
            errors.append(f"{name}: {type(exc).__name__}: {exc}")
            _log(f"    失败: {type(exc).__name__}: {exc}")
            shutil.rmtree(root / "cifar-10-batches-py", ignore_errors=True)
    raise RuntimeError("所有 tar 镜像均失败：\n  " + "\n  ".join(errors))


# ---------------------------------------------------------------- 校验

def _verify(root: Path) -> bool:
    """通过实际读取 pickle 校验数据集完整性。"""
    d = root / "cifar-10-batches-py"
    if not d.is_dir():
        return False
    missing = [f for f in BATCH_FILES if not (d / f).is_file()]
    if missing:
        _log(f"  缺少文件: {missing}")
        return False

    n_train = n_test = 0
    try:
        labels_seen = set()
        for i in range(1, 6):
            with (d / f"data_batch_{i}").open("rb") as f:
                batch = pickle.load(f)
            data = batch["data"]
            if data.shape != (10000, 3072):
                _log(f"  data_batch_{i} 形状异常: {data.shape}")
                return False
            labels_seen.update(np.unique(batch["labels"]).tolist())
            n_train += len(data)
        with (d / "test_batch").open("rb") as f:
            batch = pickle.load(f)
        n_test = len(batch["data"])
    except Exception as exc:  # noqa: BLE001 - 校验失败原因需要打印出来
        _log(f"  校验时读取出错: {type(exc).__name__}: {exc}")
        return False

    if (n_train, n_test) != (N_TRAIN, N_TEST):
        _log(f"  样本数异常: train={n_train}, test={n_test}")
        return False
    if labels_seen != set(range(10)):
        _log(f"  标签取值异常: {sorted(labels_seen)}")
        return False

    _log(f"  校验通过: train={n_train}, test={n_test}, 每张 32x32x3")
    return True


# ---------------------------------------------------------------- 入口

def prepare_cifar10(root: str | Path = "data", source: str = "auto",
                    force: bool = False) -> Path:
    """确保 CIFAR-10 已就绪，返回 ``root``。

    Args:
        root: 数据根目录，最终数据位于 ``<root>/cifar-10-batches-py``。
        source: ``auto`` / ``hf`` / ``tar``。
        force: 为 True 时忽略已有数据重新下载。
    """
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)

    if (root / "cifar-10-batches-py").is_dir() and not force:
        _log(f"检测到已有数据 {root / 'cifar-10-batches-py'}，执行校验…")
        if _verify(root):
            return root
        _log("已有数据不完整，重新下载")
        shutil.rmtree(root / "cifar-10-batches-py", ignore_errors=True)

    order = {"auto": ["hf", "tar"], "hf": ["hf"], "tar": ["tar"]}[source]
    errors: List[str] = []
    for kind in order:
        _log(f"数据来源: {kind}")
        try:
            if kind == "hf":
                _prepare_from_hf(root)
            else:
                _prepare_from_tar(root, TAR_MIRRORS)
            if _verify(root):
                _log(f"数据已就绪: {root / 'cifar-10-batches-py'}")
                return root
            _log("  校验未通过")
            shutil.rmtree(root / "cifar-10-batches-py", ignore_errors=True)
        except Exception as exc:  # noqa: BLE001 - 逐个来源尝试，失败要继续
            errors.append(f"{kind}: {type(exc).__name__}: {exc}")
            _log(f"  来源 {kind} 失败: {type(exc).__name__}: {exc}")

    raise RuntimeError(
        "所有数据来源均失败：\n  " + "\n  ".join(errors) +
        f"\n可手动下载 cifar-10-python.tar.gz 解压到 {root}"
    )


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description="准备 CIFAR-10 数据集")
    ap.add_argument("--root", default="data")
    ap.add_argument("--source", default="auto", choices=["auto", "hf", "tar"])
    ap.add_argument("--force", action="store_true", help="重新下载")
    a = ap.parse_args()
    prepare_cifar10(a.root, a.source, a.force)
