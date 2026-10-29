"""按切片划分交叉验证折。

原来 ``train_3class.py`` 把 5 个切片名写死在 ``ALL_SLIDES`` 里，换成 73 个 WSI
就必须改代码，而且多出来的切片会被静默丢掉。这里改成：

1. 运行时从 ``<data_root>/train_patches`` 发现所有切片目录；
2. 从 ``_patch_labels_3class.csv`` 统计每个切片的肿瘤负荷；
3. 按负荷排序后蛇形分配，保证每折验证集分布接近，且每个切片恰好验证一次。

划分是确定性的（不依赖随机数），同样的数据每次算出的结果完全相同；
训练开始时还会把划分写成 ``<out_dir>/splits.json``，评测直接读它，
避免"训练和评测用了不同划分"这种隐蔽错误。
"""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path

# --------------------------------------------------------------------------- #
# 常量
# --------------------------------------------------------------------------- #
INDEX_NAME: str = "_patch_labels_3class.csv"
"""patch 索引文件名，由 ``scripts/make_3class_masks.py`` 生成。"""

SPLITS_NAME: str = "splits.json"
"""训练输出目录下保存折划分的文件名。"""


@dataclass(frozen=True)
class SlideStat:
    """单个切片的统计量，用于分层分折。"""

    slide: str
    n_patch: int
    n_cls1: int
    n_cls2: int
    mean_fg: float

    @property
    def tumor_ratio(self) -> float:
        """有肿瘤的 patch 占比（类别 1 + 类别 2）。"""
        if self.n_patch <= 0:
            return 0.0
        return (self.n_cls1 + self.n_cls2) / self.n_patch


# --------------------------------------------------------------------------- #
# 切片发现
# --------------------------------------------------------------------------- #
def discover_slides(data_root: Path) -> list[str]:
    """扫描 ``<data_root>/train_patches`` 下的切片目录。

    Args:
        data_root: 数据根，其下有 ``train_patches/``。

    Returns:
        切片名列表，按名称排序（``train_001``, ``train_002`` …）。

    Raises:
        FileNotFoundError: ``train_patches`` 不存在。
        ValueError: 目录下没有任何切片。
    """
    patches_root: Path = data_root / "train_patches"
    if not patches_root.is_dir():
        raise FileNotFoundError(
            f"找不到切片目录：{patches_root}\n"
            f"       确认 --data-root 指向的是「包含 train_patches/ 的那一层」"
        )

    slides: list[str] = sorted(
        p.name for p in patches_root.iterdir()
        if p.is_dir() and not p.name.startswith(".")
    )
    if not slides:
        raise ValueError(f"{patches_root} 下没有切片子目录")

    return slides


# --------------------------------------------------------------------------- #
# 统计
# --------------------------------------------------------------------------- #
def load_slide_stats(data_root: Path, slides: list[str] | None = None) -> dict[str, SlideStat]:
    """从 patch 索引 CSV 统计每个切片的类别分布。

    Args:
        data_root: 数据根。
        slides: 只统计这些切片；``None`` 表示 CSV 里出现的全部。

    Returns:
        ``{slide: SlideStat}``。

    Raises:
        FileNotFoundError: 索引 CSV 不存在（需要先跑 make_3class_masks.py）。
    """
    csv_path: Path = data_root / INDEX_NAME
    if not csv_path.is_file():
        raise FileNotFoundError(
            f"找不到 patch 索引：{csv_path}\n"
            f"       先跑：python scripts/make_3class_masks.py --data-root \"{data_root}\""
        )

    allowed: set[str] | None = set(slides) if slides is not None else None
    acc: dict[str, list[float]] = {}

    with csv_path.open(newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            slide: str = row["slide"]
            if allowed is not None and slide not in allowed:
                continue
            # [n_patch, n_cls1, n_cls2, sum_fg]
            rec: list[float] = acc.setdefault(slide, [0.0, 0.0, 0.0, 0.0])
            label: int = int(row["label"])
            rec[0] += 1
            if label == 1:
                rec[1] += 1
            elif label == 2:
                rec[2] += 1
            rec[3] += float(row.get("fg_ratio") or 0.0)

    stats: dict[str, SlideStat] = {}
    for slide, (n, n1, n2, sum_fg) in acc.items():
        stats[slide] = SlideStat(
            slide=slide,
            n_patch=int(n),
            n_cls1=int(n1),
            n_cls2=int(n2),
            mean_fg=(sum_fg / n) if n > 0 else 0.0,
        )
    return stats


# --------------------------------------------------------------------------- #
# 分折
# --------------------------------------------------------------------------- #
def build_slide_folds(
    stats: dict[str, SlideStat],
    num_folds: int = 5,
    *,
    verbose: bool = True,
) -> dict[int, list[str]]:
    """按肿瘤负荷分层，把切片分成 ``num_folds`` 折。

    做法：按 ``mean_fg`` 升序排，然后蛇形分配（1→2→3→3→2→1→…）。
    排序保证相邻切片负荷接近，蛇形保证每折拿到的负荷总量接近。

    Args:
        stats: ``load_slide_stats`` 的结果。
        num_folds: 目标折数；切片数不足时自动降到切片数。
        verbose: 是否打印每折摘要。

    Returns:
        ``{fold: [slide, ...]}``，fold 从 1 开始。实际折数以返回值长度为准。
    """
    if not stats:
        raise ValueError("stats 为空，没有可划分的切片")

    actual_folds: int = min(num_folds, len(stats))
    if actual_folds < num_folds and verbose:
        print(f"[split] 只有 {len(stats)} 个切片，折数从 {num_folds} 降到 {actual_folds}")

    # 负荷低的在前；同负荷按 patch 数、再按名字，保证顺序稳定
    order: list[str] = sorted(
        stats.values(),
        key=lambda s: (s.mean_fg, s.n_patch, s.slide),
    )
    order_names: list[str] = [s.slide for s in order]

    folds: dict[int, list[str]] = {k: [] for k in range(1, actual_folds + 1)}
    for i, slide in enumerate(order_names):
        block, pos = divmod(i, actual_folds)
        # 偶数轮正序、奇数轮倒序，即蛇形
        idx: int = pos if block % 2 == 0 else actual_folds - 1 - pos
        folds[idx + 1].append(slide)

    for k in folds:
        folds[k] = sorted(folds[k])

    if verbose:
        print(f"[split] {len(order_names)} 个切片 → {actual_folds} 折（按肿瘤负荷蛇形分层）")
        for k in sorted(folds):
            load: float = sum(stats[s].mean_fg for s in folds[k])
            n_patch: int = sum(stats[s].n_patch for s in folds[k])
            n_tumor: int = sum(stats[s].n_cls1 + stats[s].n_cls2 for s in folds[k])
            print(f"        fold{k}: {len(folds[k]):>3} 切片  {n_patch:>6} patch  "
                  f"肿瘤 patch {n_tumor:>5}  负荷 {load:.4f}")

    return folds


def validate_folds(folds: dict[int, list[str]], all_slides: list[str]) -> None:
    """检查划分是否合法：每折非空、并集等于全集、无重复。

    Raises:
        ValueError: 划分有问题。
    """
    seen: set[str] = set()
    for k, slides in folds.items():
        if not slides:
            raise ValueError(f"fold{k} 是空的，无法作为验证集")
        dup: set[str] = seen & set(slides)
        if dup:
            raise ValueError(f"切片同时出现在多折：{sorted(dup)}")
        seen |= set(slides)

    missing: set[str] = set(all_slides) - seen
    if missing:
        raise ValueError(f"有切片没有分到任何折：{sorted(missing)}")


# --------------------------------------------------------------------------- #
# 落盘 / 读取
# --------------------------------------------------------------------------- #
def save_folds(folds: dict[int, list[str]], out_dir: Path) -> Path:
    """把折划分写到 ``<out_dir>/splits.json``，供评测复用。"""
    out_dir.mkdir(parents=True, exist_ok=True)
    path: Path = out_dir / SPLITS_NAME
    payload: dict[str, dict[str, list[str]]] = {
        "folds": {str(k): v for k, v in sorted(folds.items())},
    }
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return path


def load_folds(path: Path) -> dict[int, list[str]] | None:
    """读取训练时落盘的划分；文件不存在返回 ``None``。"""
    if not path.is_file():
        return None
    raw = json.loads(path.read_text(encoding="utf-8"))
    return {int(k): list(v) for k, v in raw["folds"].items()}
