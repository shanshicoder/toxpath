"""ToxPath 三分类数据集：整张 1024×1024 patch + 彩色三分类 mask。

任务
----
逐像素三分类：

======  ==================  =====================
类别    含义                颜色（RGB）
======  ==================  =====================
0       背景 / 非肿瘤       (0, 0, 0) 黑
1       零星肿瘤             (0, 255, 0) 绿
2       大块肿瘤             (255, 0, 0) 红
======  ==================  =====================

类别由整张 patch 的**连通域统计**决定（见 ``scripts/make_3class_masks.py``）：
最大连通域够大、或面积中等但几乎成一整块 → 大块肿瘤；有前景但都是小灶 /
碎片 → 零星肿瘤。前景像素按类别着色、背景纯黑。

数据布局
--------
::

    data/toxpath/
    ├── train_patches/{slide}/1.0/{X}_{Y}.jpeg          # 1024×1024 H&E
    ├── train_masks_classes/{slide}/1.0/{X}_{Y}.png     # 彩色三分类 mask
    └── _patch_labels_3class.csv                        # 标签索引（采样权重用）

与 HistoPLUS 的输入对齐
-----------------------
- 图像 → ``float32 [3, S, S]``，``(x/255 - Bioptimus_mean) / Bioptimus_std``；
- mask → ``int64 [S, S]``，取值 ``{0,1,2}``；
- 默认 ``S = 1024``。1024 不是 14 的倍数（1024 % 14 = 2），
  ``ToxPathSegmentor.forward`` 内部会 reflect-pad 到 1036 再裁回，Dataset 不用管。
"""

from __future__ import annotations

import csv
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset, WeightedRandomSampler

# 颜色与类别定义只有一处（labels.py），避免三处各抄一份导致颜色错位
from labels import CLASS_COLORS, CLASS_NAMES

# --------------------------------------------------------------------------- #
# 常量
# --------------------------------------------------------------------------- #
BIOPTIMUS_MEAN: tuple[float, float, float] = (0.707223, 0.578729, 0.703617)
BIOPTIMUS_STD: tuple[float, float, float] = (0.211883, 0.230117, 0.177517)

_COLOR_TO_CLASS: dict[int, int] = {
    (r << 16) | (g << 8) | b: c for c, (r, g, b) in CLASS_COLORS.items()
}

DEFAULT_MASKS_DIR: str = "train_masks_classes"
DEFAULT_INDEX_NAME: str = "_patch_labels_3class.csv"


# --------------------------------------------------------------------------- #
# 颜色 → 类别
# --------------------------------------------------------------------------- #
def rgb_to_class(rgb: np.ndarray) -> np.ndarray:
    """``[H, W, 3]`` uint8 RGB → ``[H, W]`` int64 类别 ``{0,1,2}``。

    未知颜色（理论上不应出现）归为 0，便于发现脏数据时不会直接崩。
    """
    keys: np.ndarray = (
        (rgb[..., 0].astype(np.int64) << 16)
        | (rgb[..., 1].astype(np.int64) << 8)
        | rgb[..., 2].astype(np.int64)
    )
    src_keys: np.ndarray = np.array(sorted(_COLOR_TO_CLASS), dtype=np.int64)
    src_vals: np.ndarray = np.array([_COLOR_TO_CLASS[int(k)] for k in src_keys], dtype=np.int64)
    idx: np.ndarray = np.searchsorted(src_keys, keys)
    out: np.ndarray = np.zeros(keys.shape, dtype=np.int64)
    valid: np.ndarray = (idx < len(src_keys)) & (
        src_keys[np.clip(idx, 0, len(src_keys) - 1)] == keys
    )
    out[valid] = src_vals[np.clip(idx, 0, len(src_keys) - 1)][valid]
    return out


# --------------------------------------------------------------------------- #
# HED 颜色抖动
# --------------------------------------------------------------------------- #
_RGB_TO_HED: np.ndarray = np.array(
    [
        [0.65, 0.70, 0.29],
        [0.07, 0.99, 0.11],
        [0.27, 0.57, 0.78],
    ],
    dtype=np.float32,
)
_HED_TO_RGB: np.ndarray = np.linalg.inv(_RGB_TO_HED)


# --------------------------------------------------------------------------- #
# 配置
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class Dataset3Config:
    """三分类数据集的不可变配置。"""

    image_size: int = 1024
    """输出 tile 边长。默认整张 1024；调小（如 224）会 resize，仅用于冒烟测试。"""

    augment: bool = True
    """训练时是否做翻转/90° 旋转/HED 抖动。"""

    hed_alpha: float = 0.04
    """HED 浓度抖动幅度。"""

    hed_beta: float = 0.04
    """HED 浓度偏移幅度。"""

    sample_weight_per_class: tuple[float, float, float] = (1.0, 6.0, 2.5)
    """(背景, 零星肿瘤, 大块肿瘤) 的采样权重。

    本地 4953 张的类别数是 4140 / 149 / 664。权重取**逆数量的平方根**再归一到背景：
    ``sqrt(4140/149) = 5.27 → 6.0``，``sqrt(4140/664) = 2.50 → 2.5``。
    用平方根而不是逆数量，是因为逆数量会把 149 张的类别压到近乎全部样本，
    模型只看得到那几张图、泛化会崩；平方根在「补稀有类」和「不丢多样性」之间折中。
    按此权重，一个 epoch 里三类样本大致是 62% / 13% / 25%。可在训练脚本里覆盖。
    """

    masks_dir: str = DEFAULT_MASKS_DIR
    """彩色 mask 相对 data_root 的目录名。"""

    index_name: str = DEFAULT_INDEX_NAME
    """标签索引 csv 文件名（相对 data_root）。"""


# --------------------------------------------------------------------------- #
# Dataset
# --------------------------------------------------------------------------- #
class ToxPath3Dataset(Dataset):
    """整张 1024 patch 的三分类分割数据集。

    Parameters
    ----------
    data_root:
        ``data/toxpath``。
    slides:
        ``["train_002", ...]``。
    cfg:
        数据集配置。
    mode:
        ``"train"`` / ``"val"``；``val`` 强制关闭增强。
    seed:
        随机种子（每个 worker 会再派生）。
    max_samples:
        >0 时只取前 N 个，仅用于冒烟。
    """

    def __init__(
        self,
        data_root: Path,
        slides: list[str],
        *,
        cfg: Dataset3Config | None = None,
        mode: Literal["train", "val"] = "train",
        seed: int = 42,
        max_samples: int = 0,
    ) -> None:
        super().__init__()
        self.data_root: Path = Path(data_root)
        self.slides: list[str] = list(slides)
        self.cfg: Dataset3Config = cfg or Dataset3Config()
        self.mode: Literal["train", "val"] = mode
        self.seed: int = seed

        if self.mode == "val":
            # val 一律不增强
            self.cfg = Dataset3Config(
                image_size=self.cfg.image_size,
                augment=False,
                hed_alpha=self.cfg.hed_alpha,
                hed_beta=self.cfg.hed_beta,
                sample_weight_per_class=self.cfg.sample_weight_per_class,
                masks_dir=self.cfg.masks_dir,
                index_name=self.cfg.index_name,
            )

        # ---- 1) 枚举 patch 与 mask ----
        samples: list[dict[str, Any]] = []
        for slide in self.slides:
            patch_dir: Path = self.data_root / "train_patches" / slide / "1.0"
            mask_dir: Path = self.data_root / self.cfg.masks_dir / slide / "1.0"
            if not patch_dir.exists():
                raise FileNotFoundError(f"patch 目录不存在: {patch_dir}")
            if not mask_dir.exists():
                raise FileNotFoundError(
                    f"三分类 mask 目录不存在: {mask_dir}\n"
                    f"请先运行: python scripts/make_3class_masks.py"
                )
            patch_files: dict[str, Path] = {p.stem: p for p in sorted(patch_dir.glob("*.jpeg"))}
            mask_files: dict[str, Path] = {m.stem: m for m in sorted(mask_dir.glob("*.png"))}
            missing: set[str] = set(patch_files) - set(mask_files)
            if missing:
                raise ValueError(
                    f"{slide} 有 {len(missing)} 个 patch 缺少三分类 mask，"
                    f"例如 {sorted(missing)[:3]}"
                )
            for stem in sorted(patch_files):
                samples.append({
                    "slide": slide,
                    "name": stem,
                    "image_path": patch_files[stem],
                    "mask_path": mask_files[stem],
                })

        if max_samples > 0 and len(samples) > max_samples:
            samples = samples[:max_samples]
            print(f"[ToxPath3Dataset] max_samples={max_samples} 截断到 {len(samples)}")
        if not samples:
            raise ValueError(f"slides={self.slides} 下没有可用 patch")

        # ---- 2) 标签索引（采样权重 + 诊断） ----
        index: dict[tuple[str, str], tuple[int, float]] = self._load_index()
        self.samples: list[dict[str, Any]] = samples
        self.labels: np.ndarray = np.zeros(len(samples), dtype=np.int8)
        self.fg_ratios: np.ndarray = np.zeros(len(samples), dtype=np.float32)
        for i, s in enumerate(samples):
            rec = index.get((s["slide"], s["name"]))
            if rec is None:
                raise KeyError(
                    f"索引里找不到 {s['slide']}/{s['name']}，"
                    f"请重新运行 scripts/make_3class_masks.py"
                )
            self.labels[i], self.fg_ratios[i] = rec
        self.sample_weights: np.ndarray = np.asarray(
            [self.cfg.sample_weight_per_class[int(l)] for l in self.labels],
            dtype=np.float32,
        )

        # ---- 3) 报告 ----
        n: int = len(samples)
        counts = {c: int((self.labels == c).sum()) for c in (0, 1, 2)}
        print(f"[ToxPath3Dataset] mode={mode} slides={self.slides} n={n}")
        print(f"  类别分布: "
              f"0(背景)={counts[0]} ({100*counts[0]/n:.1f}%)  "
              f"1(零星)={counts[1]} ({100*counts[1]/n:.1f}%)  "
              f"2(大块)={counts[2]} ({100*counts[2]/n:.1f}%)")

        # ---- 4) 归一化常量 ----
        self._mean: np.ndarray = np.array(BIOPTIMUS_MEAN, dtype=np.float32)
        self._std: np.ndarray = np.array(BIOPTIMUS_STD, dtype=np.float32)

    # ------------------------------------------------------------------ #
    def _load_index(self) -> dict[tuple[str, str], tuple[int, float]]:
        """读取 ``_patch_labels_3class.csv`` -> {(slide, stem): (label, fg_ratio)}。"""
        path: Path = self.data_root / self.cfg.index_name
        if not path.exists():
            raise FileNotFoundError(
                f"标签索引不存在: {path}\n"
                f"请先运行: python scripts/make_3class_masks.py"
            )
        index: dict[tuple[str, str], tuple[int, float]] = {}
        with path.open("r", encoding="utf-8", newline="") as fh:
            for row in csv.DictReader(fh):
                stem: str = row["name"].rsplit(".", 1)[0]
                index[(row["slide"], stem)] = (int(row["label"]), float(row["fg_ratio"]))
        return index

    # ------------------------------------------------------------------ #
    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        """返回一个样本。

        Returns
        -------
        dict
            - ``image``: ``[3, S, S]`` float32，Bioptimus 归一化
            - ``mask``: ``[S, S]`` int64，取值 {0,1,2}
            - ``patch_label``: int，该 patch 的类别（0/1/2）
            - ``fg_ratio``: float，该 patch 的前景占比
            - ``sample_weight``: float
            - ``slide`` / ``name``: str
        """
        s = self.samples[idx]
        image: Image.Image = Image.open(s["image_path"]).convert("RGB")
        mask: Image.Image = Image.open(s["mask_path"]).convert("RGB")

        size: int = self.cfg.image_size
        if image.size != (size, size):
            image = image.resize((size, size), Image.BILINEAR)
            mask = mask.resize((size, size), Image.NEAREST)

        img: np.ndarray = np.asarray(image, dtype=np.uint8)
        msk: np.ndarray = np.asarray(mask, dtype=np.uint8)

        if self.mode == "train" and self.cfg.augment:
            img, msk = self._augment(img, msk)

        label_map: np.ndarray = rgb_to_class(msk)
        img_f: np.ndarray = img.astype(np.float32) / 255.0
        img_f = (img_f - self._mean) / self._std

        return {
            "image": torch.from_numpy(img_f.transpose(2, 0, 1).copy()).float(),
            "mask": torch.from_numpy(label_map.copy()).long(),
            "patch_label": int(self.labels[idx]),
            "fg_ratio": float(self.fg_ratios[idx]),
            "sample_weight": float(self.sample_weights[idx]),
            "slide": s["slide"],
            "name": s["name"],
        }

    # ------------------------------------------------------------------ #
    def _augment(
        self, img: np.ndarray, msk: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """同步做翻转/90° 旋转；图像额外做 HED 抖动。"""
        if random.random() < 0.5:
            img = img[:, ::-1].copy()
            msk = msk[:, ::-1].copy()
        if random.random() < 0.5:
            img = img[::-1].copy()
            msk = msk[::-1].copy()
        k: int = random.randint(0, 3)
        if k:
            img = np.rot90(img, k=k).copy()
            msk = np.rot90(msk, k=k).copy()
        if self.cfg.hed_alpha > 0 or self.cfg.hed_beta > 0:
            img = self._hed_jitter(img)
        return np.ascontiguousarray(img), np.ascontiguousarray(msk)

    def _hed_jitter(self, rgb: np.ndarray) -> np.ndarray:
        """HED 染色浓度抖动，只改颜色不改几何。"""
        rgb_f: np.ndarray = np.clip(rgb.astype(np.float32), 1, 255) / 255.0
        od: np.ndarray = -np.log(rgb_f)
        stains_t: np.ndarray = _RGB_TO_HED.T
        c: np.ndarray = od @ np.linalg.inv(stains_t)
        alpha: np.ndarray = np.random.normal(1.0, self.cfg.hed_alpha, size=3).astype(np.float32)
        beta: np.ndarray = np.random.normal(0.0, self.cfg.hed_beta, size=3).astype(np.float32)
        od_new: np.ndarray = (c * alpha + beta) @ stains_t
        out: np.ndarray = np.exp(-od_new)
        return np.clip(out * 255.0, 0, 255).astype(np.uint8)


# --------------------------------------------------------------------------- #
# 便捷构造 / 采样器
# --------------------------------------------------------------------------- #
def build_dataset3(
    data_root: Path,
    train_slides: list[str],
    val_slides: list[str] | None = None,
    *,
    cfg: Dataset3Config | None = None,
    seed: int = 42,
    max_train_samples: int = 0,
    max_val_samples: int = 0,
) -> tuple[ToxPath3Dataset, ToxPath3Dataset | None]:
    """构造训练集与验证集。"""
    train_ds = ToxPath3Dataset(
        data_root, train_slides, cfg=cfg, mode="train",
        seed=seed, max_samples=max_train_samples,
    )
    val_ds: ToxPath3Dataset | None = None
    if val_slides:
        cfg_val = Dataset3Config(
            image_size=(cfg or Dataset3Config()).image_size,
            augment=False,
            hed_alpha=0.0,
            hed_beta=0.0,
            sample_weight_per_class=(cfg or Dataset3Config()).sample_weight_per_class,
            masks_dir=(cfg or Dataset3Config()).masks_dir,
            index_name=(cfg or Dataset3Config()).index_name,
        )
        val_ds = ToxPath3Dataset(
            data_root, val_slides, cfg=cfg_val, mode="val",
            seed=seed, max_samples=max_val_samples,
        )
    return train_ds, val_ds


def build_weighted_sampler(
    dataset: ToxPath3Dataset,
    *,
    num_samples: int | None = None,
    replacement: bool = True,
    seed: int = 42,
) -> WeightedRandomSampler:
    """按 patch 类别权重构造 ``WeightedRandomSampler``。

    ``num_samples`` 默认等于数据集大小；训练脚本通常传 ``samples_per_epoch``
    来控制每个 epoch 的采样量（1024 输入下每个样本都很贵）。
    """
    generator: torch.Generator = torch.Generator().manual_seed(seed)
    return WeightedRandomSampler(
        weights=torch.as_tensor(dataset.sample_weights, dtype=torch.double),
        num_samples=num_samples or len(dataset),
        replacement=replacement,
        generator=generator,
    )


def seed_worker(worker_id: int) -> None:
    """DataLoader worker 初始化：让每个 worker 的 random/numpy 种子不同。"""
    worker_seed: int = (torch.initial_seed() + worker_id) % (2**32)
    np.random.seed(worker_seed)
    random.seed(worker_seed)


__all__ = [
    "BIOPTIMUS_MEAN",
    "BIOPTIMUS_STD",
    "CLASS_COLORS",
    "CLASS_NAMES",
    "Dataset3Config",
    "ToxPath3Dataset",
    "build_dataset3",
    "build_weighted_sampler",
    "rgb_to_class",
    "seed_worker",
]
