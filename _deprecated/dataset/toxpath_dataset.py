"""ToxPath 数据集：**三分类**语义分割，数据源是单通道标签 PNG。

数据布局
--------
::

    data/toxpath/
    ├── train_patches/{slide}/1.0/{X}_{Y}.jpeg      # H&E 1024×1024 RGB
    └── train_masks_class/{slide}/1.0/{X}_{Y}.png    # 单通道灰度，值域 {0,128,255}

mask 编码 → 模型类别
--------------------
``train_masks_class`` 由 ``scripts/relabel_masks.py`` 生成，
像素值只有 0/128/255 三个（**间隔 128 个灰阶，黑/中灰/白，肉眼可分**；
不是 1/2/3 那种看起来全黑的编码）：

============  ================  ==============================
文件像素值     模型类别索引       含义
============  ================  ==============================
0             0                背景（含纯背景 patch 的整图）
128           1                前景：零星/小灶/中等
255           2                前景：较大块/大块/多灶累积
============  ================  ==============================

读取时按阈值（<64 / 64~191 / ≥192）映射成 ``{0,1,2}``，
交叉熵要求类别索引从 0 开始。模型输出 ``num_classes=3`` 的 logits。

三类的关系
----------
- 第 0 类（背景）在**所有** patch 里都存在；
- 第 1/2 类只出现在有前景的 patch 里，且**同一 patch 内的前景属于同一类**
  （类别是 patch 级别的病灶规模判定，前景像素按该类别取值）；
- 所以"纯背景 patch" ⟺ 整图都是类别 0。

采样权重
--------
基于 patch 的"主导类别"（有前景时取前景所属类别，否则为背景）加权：

- 背景 patch：weight=1.0
- 类别2（小灶，6.1% 最稀缺）：weight=4.0
- 类别3（大块，10.1%）：weight=2.0

数据记录（训练脚本可调用 ``DatasetLog``）
-----------------------------------------
``__getitem__`` 每次调用会向 ``DatasetLog`` 推一条记录，
训练脚本在每个 epoch 末把 log flush 到磁盘：

- ``dataset_log_epoch_{N}.jsonl``：每行一个 sample，含 slide/name/主导类别/各类像素占比
- ``dataset_log_summary.json``：累计统计（每类被采样次数、像素分布）
"""

from __future__ import annotations

import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import numpy as np
from PIL import Image
import torch
from torch import Tensor
from torch.utils.data import Dataset


# --------------------------------------------------------------------------- #
# 常量
# --------------------------------------------------------------------------- #
BIOPTIMUS_MEAN: tuple[float, float, float] = (0.707223, 0.578729, 0.703617)
"""H0-mini 预训练数据集的 RGB 均值（来源：HistoPLUS ``constants.py``）。"""

BIOPTIMUS_STD: tuple[float, float, float] = (0.211883, 0.230117, 0.177517)
"""H0-mini 预训练数据集的 RGB 标准差。"""

NUM_CLASSES: int = 3
"""模型类别数：0=背景, 1=小灶前景, 2=大块前景。"""

CLASS_NAMES: tuple[str, str, str] = ("背景", "零星/小灶/中等", "较大块/大块/多灶累积")
"""模型类别索引 → 中文语义。"""

MASK_VALUE_TO_CLASS: dict[int, int] = {0: 0, 128: 1, 255: 2}
"""mask 像素值 {0,128,255} → 模型类别索引 {0,1,2}。

三个类别用间隔 128 个灰阶的值表示（黑/中灰/白），肉眼可分，
不是 1/2/3 那种"都是纯黑"的编码。
"""

# 离散映射的容错阈值：落到哪一档就归哪一类。
# 正常文件只会有 0/128/255 三个精确值，但 PNG 若被别的工具重新保存过
# （比如经历有损缩放），用阈值比精确匹配更稳。
_CLASS_THRESHOLDS: tuple[int, int] = (64, 192)
"""``< 64`` → 0（背景）；``64~191`` → 1（小灶）；``>= 192`` → 2（大块）。"""


def mask_to_class(mask: np.ndarray) -> np.ndarray:
    """单通道灰度 mask → 模型类别索引数组。

    Parameters
    ----------
    mask:
        ``[H, W]`` 的 uint8 数组，像素值应为 ``{0, 128, 255}``。

    Returns
    -------
    np.ndarray
        ``[H, W]`` 的 int64 数组，取值 ``{0,1,2}``。

    Notes
    -----
    用两个阈值切三档，而不是精确等于 0/128/255：
    正常文件三值精确，阈值法与精确匹配结果完全一致；
    万一 mask 被别的工具重存过（有损压缩、调过对比度），阈值法仍能正确归类，
    不会因为差 1 个灰阶就整片变成背景。
    """
    lo, hi = _CLASS_THRESHOLDS
    arr = mask.astype(np.uint8, copy=False)
    out = np.zeros(arr.shape, dtype=np.int64)
    out[arr >= lo] = 1
    out[arr >= hi] = 2
    return out


# --------------------------------------------------------------------------- #
# HED 颜色抖动（跨中心泛化关键）
# --------------------------------------------------------------------------- #
_RGB_TO_HED: np.ndarray = np.array([
    [0.65, 0.70, 0.29],
    [0.07, 0.99, 0.11],
    [0.27, 0.57, 0.78],
], dtype=np.float32)
_HED_TO_RGB: np.ndarray = np.linalg.inv(_RGB_TO_HED)


# --------------------------------------------------------------------------- #
# 配置
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class DatasetConfig:
    """数据集的不可变配置。"""

    image_size: int = 224
    """训练 tile 边长，必须能被 14 整除（H0-mini patch_size）。"""

    augment: bool = True
    """是否启用训练时增强（翻转/90°/HED 抖动）。"""

    hed_alpha: float = 0.04
    """HED 通道浓度抖动的标准差（HoVer-Net 通行值）。"""

    hed_beta: float = 0.04
    """HED 通道浓度偏移的标准差。"""

    sample_weight_per_label: tuple[float, float, float] = (1.0, 4.0, 2.0)
    """(背景, 小灶, 大块) → 采样权重。

    小灶(类2) 只占 6.1%，最稀缺且最难学 → 4.0；
    大块(类3) 占 10.1% → 2.0；背景 83.7% → 1.0（不再压低，靠 loss 权重平衡）。
    """


# --------------------------------------------------------------------------- #
# 数据记录器
# --------------------------------------------------------------------------- #
class DatasetLog:
    """记录每个 sample 被采样时的元信息，供训练脚本事后分析。

    训练脚本可在每个 epoch 末调 ``flush(epoch)`` 写到磁盘：
    - ``dataset_log_epoch_{N}.jsonl``：每行一个 sample
    - 累计的 ``dataset_log_summary.json``：每类采样次数 + 像素分布
    """

    def __init__(self, out_dir: Path) -> None:
        self.out_dir: Path = Path(out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.records: list[dict[str, Any]] = []
        self.class_sample_count: dict[int, int] = {0: 0, 1: 0, 2: 0}
        # 像素占比的累计值（不乘任何常数，避免小类被截断成 0）
        self.class_pixel_ratio_sum: dict[int, float] = {0: 0.0, 1: 0.0, 2: 0.0}

    def log(self, *, slide: str, name: str, dominant_label: int,
            pixel_ratios: dict[int, float], epoch: int) -> None:
        """记录一个 sample 被采样的信息。

        Parameters
        ----------
        slide, name:
            sample 的位置标识。
        dominant_label:
            该 patch 的主导类别（0=背景, 1=小灶, 2=大块）。
        pixel_ratios:
            各类别像素占比，``{0: 0.95, 1: 0.04, 2: 0.01}`` 这样。
        epoch:
            当前 epoch 编号。
        """
        rec = {
            "epoch": epoch, "slide": slide, "name": name,
            "dominant_label": dominant_label,
            "pixel_ratios": pixel_ratios,
        }
        self.records.append(rec)
        self.class_sample_count[dominant_label] += 1
        for k, v in pixel_ratios.items():
            self.class_pixel_ratio_sum[int(k)] += float(v)

    def flush(self, epoch: int) -> Path:
        """把当前 epoch 的记录写到磁盘，返回 jsonl 路径。"""
        jsonl_path = self.out_dir / f"dataset_log_epoch_{epoch}.jsonl"
        with jsonl_path.open("w", encoding="utf-8") as fh:
            for rec in self.records:
                fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
        # 累计 summary
        summary_path = self.out_dir / "dataset_log_summary.json"
        summary = {
            "epoch": epoch,
            "class_names": {str(i): n for i, n in enumerate(CLASS_NAMES)},
            "class_sample_count": self.class_sample_count,
            "class_pixel_ratio_sum": {k: round(v, 4)
                                      for k, v in self.class_pixel_ratio_sum.items()},
            "total_records": len(self.records),
        }
        summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2),
                                encoding="utf-8")
        # 清空当前 epoch 记录，下个 epoch 重新累积
        self.records.clear()
        return jsonl_path


# --------------------------------------------------------------------------- #
# Dataset
# --------------------------------------------------------------------------- #
class ToxPathDataset(Dataset):
    """三分类语义分割数据集。

    数据源
    ------
    - ``data_root/train_patches/{slide}/1.0/{X}_{Y}.jpeg``   H&E 1024×1024 RGB
    - ``data_root/train_masks_class/{slide}/1.0/{X}_{Y}.png`` 单通道，值域 {0,128,255}

    Parameters
    ----------
    data_root:
        数据根目录。
    slides:
        训练用切片名列表。
    cfg:
        数据集配置。
    mode:
        ``"train"`` / ``"val"``，影响裁剪与增强。
    dataset_log:
        可选的数据记录器。传入后每次 ``__getitem__`` 会推一条记录。
    epoch:
        当前 epoch 编号（用于 dataset_log）。
    seed:
        随机种子。
    max_samples:
        >0 时只取前 N 个样本（含预扫），用于冒烟测试。

    Raises
    ------
    FileNotFoundError
        数据目录不存在。
    ValueError
        patch 与 mask 数量不匹配。
    """

    def __init__(
        self,
        data_root: Path,
        slides: list[str],
        *,
        cfg: DatasetConfig | None = None,
        mode: Literal["train", "val"] = "train",
        dataset_log: DatasetLog | None = None,
        epoch: int = 0,
        seed: int = 42,
        max_samples: int = 0,
    ) -> None:
        super().__init__()
        self.data_root: Path = Path(data_root)
        self.slides: list[str] = list(slides)
        self.cfg: DatasetConfig = cfg or DatasetConfig()
        self.mode: Literal["train", "val"] = mode
        self.dataset_log: DatasetLog | None = dataset_log
        self.epoch: int = epoch
        self.seed: int = seed

        # ---- 1) 枚举 patch 与标签 mask，建立一一对应 ----
        all_samples: list[dict[str, Any]] = []
        for slide in self.slides:
            patch_dir = self.data_root / "train_patches" / slide / "1.0"
            mask_dir = self.data_root / "train_masks_class" / slide / "1.0"
            if not patch_dir.exists():
                raise FileNotFoundError(f"patch 目录不存在: {patch_dir}")
            if not mask_dir.exists():
                raise FileNotFoundError(
                    f"train_masks_class 目录不存在: {mask_dir}\n"
                    f"请先跑 python scripts/relabel_masks.py 生成三类标签 mask"
                )
            patch_files = {p.stem: p for p in sorted(patch_dir.glob("*.jpeg"))}
            mask_files = {m.stem: m for m in sorted(mask_dir.glob("*.png"))}
            common = set(patch_files) & set(mask_files)
            missing = set(patch_files) - set(mask_files)
            if missing:
                raise ValueError(
                    f"切片 {slide} 有 {len(missing)} 个 patch 缺对应标签 mask，"
                    f"如 {sorted(missing)[:3]}"
                )
            for stem in sorted(common):
                all_samples.append({
                    "slide": slide,
                    "name": stem,
                    "patch_path": patch_files[stem],
                    "mask_path": mask_files[stem],
                })

        # 冒烟模式：在预扫之前截断，避免扫全部 mask。
        # 用**等间隔抽样**而不是取前 N 个：样本是按 slide 顺序拼起来的，
        # 直接取前 N 会全部落在第一个切片上（甚至全是背景 patch），
        # 冒烟时就看不到前景类别，容易误判管线有问题。
        if max_samples > 0 and len(all_samples) > max_samples:
            idx = np.linspace(0, len(all_samples) - 1, max_samples).astype(int)
            all_samples = [all_samples[i] for i in idx]
            print(f"[ToxPathDataset] max_samples={max_samples} 等间隔抽样到 "
                  f"{len(all_samples)} 个")

        self.samples: list[dict[str, Any]] = all_samples

        if not self.samples:
            raise ValueError(f"slides={self.slides} 下没有 patch")

        # ---- 2) 预计算每个 sample 的主导类别 + 像素占比 + 采样权重 ----
        n: int = len(self.samples)
        self.sample_weights: np.ndarray = np.zeros(n, dtype=np.float32)
        self.dominant_labels: np.ndarray = np.zeros(n, dtype=np.int8)
        self.pixel_ratios: list[dict[int, float]] = [{} for _ in range(n)]

        print(f"[ToxPathDataset] 预扫 {n} 张标签 mask 计算采样权重...")
        t0 = __import__("time").time()
        for i, s in enumerate(self.samples):
            arr = np.array(Image.open(s["mask_path"]).convert("L"))
            cls = mask_to_class(arr)  # {0,1,2}
            counts = np.bincount(cls.ravel(), minlength=NUM_CLASSES)
            total = int(counts.sum())
            self.pixel_ratios[i] = {k: int(counts[k]) / total
                                    for k in range(NUM_CLASSES)}
            # 主导类别：有前景就取前景类别（同一 patch 前景只有一个类别），否则背景
            fg = counts[1:]
            dom = 0 if int(fg.sum()) == 0 else int(np.argmax(fg)) + 1
            self.dominant_labels[i] = dom
            self.sample_weights[i] = self.cfg.sample_weight_per_label[dom]
            if (i + 1) % 2000 == 0:
                print(f"  [{i+1}/{n}] 耗时 {__import__('time').time()-t0:.1f}s")
        print(f"[ToxPathDataset] 预扫完成，耗时 {__import__('time').time()-t0:.1f}s")

        # ---- 3) 报告 ----
        dist = {k: int((self.dominant_labels == k).sum()) for k in range(NUM_CLASSES)}
        print(f"[ToxPathDataset] mode={mode} slides={self.slides} n={n}")
        print(f"  主导类别分布: 0(背景)={dist[0]} ({100*dist[0]/n:.1f}%)  "
              f"1(小灶)={dist[1]} ({100*dist[1]/n:.1f}%)  "
              f"2(大块)={dist[2]} ({100*dist[2]/n:.1f}%)")

        # ---- 4) 归一化常量 ----
        self._mean = np.array(BIOPTIMUS_MEAN, dtype=np.float32)
        self._std = np.array(BIOPTIMUS_STD, dtype=np.float32)

    # ------------------------------------------------------------------ #
    # 设置当前 epoch（用于 dataset_log）
    # ------------------------------------------------------------------ #
    def set_epoch(self, epoch: int) -> None:
        """更新当前 epoch 编号（dataset_log 用）。"""
        self.epoch = epoch

    # ------------------------------------------------------------------ #
    # Dataset 协议
    # ------------------------------------------------------------------ #
    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict[str, Tensor | int | float]:
        """取一个样本。

        Returns
        -------
        dict
            - ``"image"``: ``[3, H, W]`` float32，Bioptimus 归一化
            - ``"mask"``: ``[1, H, W]`` long，取值 {0,1,2}（3 类 target）
            - ``"dominant_label"``: int，patch 主导类别
            - ``"sample_weight"``: float
        """
        s = self.samples[idx]

        # 1) 读图（mask 是单通道灰度，不转 RGB，直接拿 0/128/255 原值）
        patch_arr = np.asarray(Image.open(s["patch_path"]).convert("RGB"),
                               dtype=np.uint8)
        mask_arr = np.asarray(Image.open(s["mask_path"]).convert("L"),
                              dtype=np.uint8)

        # 2) 同步随机/中心裁剪
        H, W = patch_arr.shape[:2]
        size = self.cfg.image_size

        if self.mode == "train" and self.cfg.augment:
            if H > size:
                y0 = random.randint(0, H - size)
                x0 = random.randint(0, W - size)
                patch_arr = patch_arr[y0:y0+size, x0:x0+size]
                mask_arr = mask_arr[y0:y0+size, x0:x0+size]
            elif H < size:
                patch_arr = self._pad(patch_arr, size)
                mask_arr = self._pad(mask_arr, size)
        else:
            if H > size:
                y0 = (H - size) // 2
                x0 = (W - size) // 2
                patch_arr = patch_arr[y0:y0+size, x0:x0+size]
                mask_arr = mask_arr[y0:y0+size, x0:x0+size]
            elif H < size:
                patch_arr = self._pad(patch_arr, size)
                mask_arr = self._pad(mask_arr, size)

        # 3) 增强（train only）
        if self.mode == "train" and self.cfg.augment:
            patch_arr, mask_arr = self._augment(patch_arr, mask_arr)

        # 4) 像素值 {0,128,255} → 类别索引 {0,1,2}
        mask_cls = mask_to_class(mask_arr)  # [H,W] int64

        # 5) patch 归一化（HED 抖动已在 augment 里做过）
        patch_f = patch_arr.astype(np.float32) / 255.0
        patch_f = (patch_f - self._mean) / self._std

        # 6) HWC → CHW
        image_chw = patch_f.transpose(2, 0, 1)
        mask_chw = mask_cls[None, :, :]

        dom = int(self.dominant_labels[idx])
        weight = float(self.sample_weights[idx])

        # 7) 数据记录
        if self.dataset_log is not None:
            # 重算当前裁剪后的像素占比（更精确）
            counts = np.bincount(mask_cls.ravel(), minlength=NUM_CLASSES)
            total = int(counts.sum())
            ratios = {k: int(counts[k]) / total for k in range(NUM_CLASSES)}
            self.dataset_log.log(
                slide=s["slide"], name=s["name"],
                dominant_label=dom, pixel_ratios=ratios, epoch=self.epoch,
            )

        return {
            "image": torch.from_numpy(image_chw.copy()).float(),
            "mask": torch.from_numpy(mask_chw.copy()).long(),
            "dominant_label": dom,
            "sample_weight": weight,
        }

    # ------------------------------------------------------------------ #
    # 增强工具
    # ------------------------------------------------------------------ #
    @staticmethod
    def _pad(arr: np.ndarray, target: int) -> np.ndarray:
        """中心 pad 到 target×target。"""
        H, W = arr.shape[:2]
        ph, pw = max(0, target - H), max(0, target - W)
        if ph == 0 and pw == 0:
            return arr
        top, bottom = ph // 2, ph - ph // 2
        left, right = pw // 2, pw - pw // 2
        if arr.ndim == 3:
            return np.pad(arr, ((top, bottom), (left, right), (0, 0)), mode="reflect")
        return np.pad(arr, ((top, bottom), (left, right)), mode="reflect")

    def _augment(
        self, patch: np.ndarray, mask: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """对 patch 与 mask 同步做翻转/90° 旋转；patch 额外做 HED 抖动。

        mask 现在是 2D，翻转与 rot90 都是纯几何操作，不改变像素值，
        所以 0/128/255 的类别语义不会被破坏。
        """
        # 1) 随机水平翻转
        if random.random() < 0.5:
            patch = patch[:, ::-1].copy()
            mask = mask[:, ::-1].copy()
        # 2) 随机垂直翻转
        if random.random() < 0.5:
            patch = patch[::-1].copy()
            mask = mask[::-1].copy()
        # 3) 随机 90° 旋转
        k = random.randint(0, 3)
        if k > 0:
            patch = np.rot90(patch, k=k).copy()
            mask = np.rot90(mask, k=k).copy()
        # 4) HED 颜色抖动（仅 patch，mask 不动）
        patch = self._hed_jitter(patch)
        return patch, mask

    def _hed_jitter(self, rgb: np.ndarray) -> np.ndarray:
        """HED 染色浓度抖动。"""
        rgb_f = np.clip(rgb.astype(np.float32), 1, 255) / 255.0
        od = -np.log(rgb_f)
        stains_t = _RGB_TO_HED.T
        c = od @ np.linalg.inv(stains_t)
        alpha = np.random.normal(1.0, self.cfg.hed_alpha, size=3).astype(np.float32)
        beta = np.random.normal(0.0, self.cfg.hed_beta, size=3).astype(np.float32)
        c = c * alpha + beta
        od_new = c @ stains_t
        rgb_new = np.exp(-od_new)
        return np.clip(rgb_new * 255.0, 0, 255).astype(np.uint8)


# --------------------------------------------------------------------------- #
# 便捷构造函数
# --------------------------------------------------------------------------- #
def build_dataset(
    data_root: Path,
    train_slides: list[str],
    val_slides: list[str] | None = None,
    *,
    cfg: DatasetConfig | None = None,
    log_dir: Path | None = None,
    seed: int = 42,
    max_train_samples: int = 0,
    max_val_samples: int = 0,
) -> tuple[ToxPathDataset, ToxPathDataset | None]:
    """构造训练集 + 验证集。

    ``max_train_samples`` / ``max_val_samples`` > 0 时只取前 N 个样本
    （包括预扫），用于冒烟测试。
    """
    train_cfg = cfg or DatasetConfig()
    val_cfg = DatasetConfig(
        image_size=train_cfg.image_size,
        augment=False,
        hed_alpha=train_cfg.hed_alpha,
        hed_beta=train_cfg.hed_beta,
        sample_weight_per_label=train_cfg.sample_weight_per_label,
    )

    train_log = DatasetLog(log_dir / "train") if log_dir else None
    val_log = DatasetLog(log_dir / "val") if log_dir else None

    train_ds = ToxPathDataset(
        data_root, train_slides,
        cfg=train_cfg, mode="train",
        dataset_log=train_log, seed=seed,
        max_samples=max_train_samples,
    )
    val_ds = None
    if val_slides:
        val_ds = ToxPathDataset(
            data_root, val_slides,
            cfg=val_cfg, mode="val",
            dataset_log=val_log, seed=seed,
            max_samples=max_val_samples,
        )
    return train_ds, val_ds


def build_weighted_sampler(
    dataset: ToxPathDataset,
    *,
    replacement: bool = True,
    seed: int = 42,
) -> torch.utils.data.WeightedRandomSampler:
    """从 sample_weights 构造采样器。"""
    g = torch.Generator().manual_seed(seed)
    return torch.utils.data.WeightedRandomSampler(
        weights=torch.as_tensor(dataset.sample_weights, dtype=torch.double),
        num_samples=len(dataset),
        replacement=replacement,
        generator=g,
    )


__all__ = [
    "BIOPTIMUS_MEAN",
    "BIOPTIMUS_STD",
    "CLASS_NAMES",
    "MASK_VALUE_TO_CLASS",
    "NUM_CLASSES",
    "DatasetConfig",
    "DatasetLog",
    "ToxPathDataset",
    "build_dataset",
    "build_weighted_sampler",
    "mask_to_class",
]
