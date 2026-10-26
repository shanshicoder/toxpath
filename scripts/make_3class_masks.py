"""把 train_masks/ 的二值 mask 转成三分类彩色 mask。

分类规则（按整张 patch 的前景像素占比 r = 前景像素数 / 1024²）::

    r < 0.0001（0.01%）        → 标签 0（背景）
    0.0001 ≤ r < 0.05（5%）    → 标签 1（少量肿瘤）
    r ≥ 0.05（5%）             → 标签 2（较多肿瘤）

输出
----
``data/toxpath/train_masks_classes/{slide}/1.0/{X}_{Y}.png``

- RGB 三通道 PNG，尺寸与输入 mask 完全一致；
- 背景像素 (0,0,0) 黑；标签 1 前景 (0,255,0) 绿；标签 2 前景 (255,0,0) 红；
- 同一张 patch 的所有前景像素着同一个颜色（类别由整张 patch 的前景占比决定）。

同时输出 ``data/toxpath/_patch_labels_3class.csv``，列为
``slide,name,n_fg_px,fg_ratio,label``，供 Dataset 直接查表构造采样权重。

用法
----
.. code-block:: bash

    python scripts/make_3class_masks.py --dry-run     # 只统计不写文件
    python scripts/make_3class_masks.py               # 正式生成
    python scripts/make_3class_masks.py --low 0.0001 --high 0.05
"""

from __future__ import annotations

import argparse
import csv
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import NamedTuple

import numpy as np
from PIL import Image

# --------------------------------------------------------------------------- #
# 常量
# --------------------------------------------------------------------------- #
CLASS_COLORS: dict[int, tuple[int, int, int]] = {
    0: (0, 0, 0),        # 背景：纯黑
    1: (0, 255, 0),      # 标签1：绿
    2: (255, 0, 0),      # 标签2：红
}
"""三个类别对应的 RGB 颜色。"""

CLASS_NAMES: dict[int, str] = {
    0: "背景",
    1: "少量肿瘤（0.01%~5%）",
    2: "较多肿瘤（≥5%）",
}

DEFAULT_DATA_ROOT: Path = Path("E:/project/toxpath/data/toxpath")
DEFAULT_OUT_ROOT: Path = Path("train_masks_classes")
DEFAULT_INDEX_NAME: str = "_patch_labels_3class.csv"


# --------------------------------------------------------------------------- #
# 配置
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class EdgeConfig:
    """分类阈值。"""

    low: float = 0.0001
    """前景占比低于此值视为背景（标签 0）。0.0001 = 0.01%。"""

    high: float = 0.05
    """前景占比高于等于此值视为标签 2。0.05 = 5%。"""

    binary_thresh: int = 127
    """二值化阈值：像素值 > 此值视为前景。"""

    def label_of(self, ratio: float) -> int:
        """按前景占比返回类别。"""
        if ratio < self.low:
            return 0
        if ratio < self.high:
            return 1
        return 2


# --------------------------------------------------------------------------- #
# 单张 mask 的结果
# --------------------------------------------------------------------------- #
class MaskRecord(NamedTuple):
    slide: str
    name: str
    n_fg_px: int
    fg_ratio: float
    label: int


# --------------------------------------------------------------------------- #
# 核心函数
# --------------------------------------------------------------------------- #
def analyze_mask(path: Path, cfg: EdgeConfig) -> tuple[int, float, int]:
    """读取二值 mask，返回 ``(前景像素数, 前景占比, 标签)``。

    Parameters
    ----------
    path:
        ``train_masks`` 下的二值 PNG（0/255）。
    cfg:
        阈值配置。
    """
    arr: np.ndarray = np.asarray(Image.open(path).convert("L"), dtype=np.uint8)
    fg: np.ndarray = arr > cfg.binary_thresh
    total: int = int(fg.size)
    n_fg: int = int(fg.sum())
    ratio: float = n_fg / total
    return n_fg, ratio, cfg.label_of(ratio)


def render_mask(fg: np.ndarray, label: int) -> np.ndarray:
    """按类别给前景着色，背景保持纯黑，返回 ``[H, W, 3]`` uint8。"""
    h, w = fg.shape
    out: np.ndarray = np.zeros((h, w, 3), dtype=np.uint8)
    out[fg] = CLASS_COLORS[label]
    return out


def find_slides(masks_root: Path) -> list[str]:
    """枚举 ``train_masks`` 下的切片目录。"""
    if not masks_root.exists():
        raise FileNotFoundError(f"train_masks 不存在: {masks_root}")
    slides: list[str] = sorted(
        p.name for p in masks_root.iterdir() if p.is_dir() and p.name.startswith("train_")
    )
    if not slides:
        raise FileNotFoundError(f"{masks_root} 下没有 train_* 子目录")
    return slides


# --------------------------------------------------------------------------- #
# 主流程
# --------------------------------------------------------------------------- #
def run(
    data_root: Path,
    out_root: Path,
    cfg: EdgeConfig,
    *,
    dry_run: bool = False,
    verbose: bool = False,
) -> int:
    """生成三分类彩色 mask 与标签索引。"""
    masks_root: Path = data_root / "train_masks"
    out_dir: Path = out_root if out_root.is_absolute() else data_root / out_root
    slides: list[str] = find_slides(masks_root)

    print(f"[make3] 数据根  : {data_root}")
    print(f"[make3] 输入    : {masks_root}")
    print(f"[make3] 输出    : {out_dir}")
    print(f"[make3] 规则    : r<{cfg.low} → 0 ; {cfg.low}≤r<{cfg.high} → 1 ; r≥{cfg.high} → 2")
    print(f"[make3] 颜色    : 0=黑 {CLASS_COLORS[0]}  1=绿 {CLASS_COLORS[1]}  2=红 {CLASS_COLORS[2]}")
    print(f"[make3] dry_run = {dry_run}\n")

    records: list[MaskRecord] = []
    per_slide: dict[str, dict[int, int]] = {}
    t0: float = time.time()

    for slide in slides:
        files: list[Path] = sorted((masks_root / slide / "1.0").glob("*.png"))
        if not files:
            raise FileNotFoundError(f"切片 {slide} 下没有 PNG: {masks_root / slide / '1.0'}")

        counts: dict[int, int] = {0: 0, 1: 0, 2: 0}
        for i, f in enumerate(files):
            n_fg, ratio, label = analyze_mask(f, cfg)
            records.append(MaskRecord(slide=slide, name=f.name, n_fg_px=n_fg,
                                      fg_ratio=ratio, label=label))
            counts[label] += 1

            if not dry_run:
                arr: np.ndarray = np.asarray(Image.open(f).convert("L"), dtype=np.uint8)
                out: np.ndarray = render_mask(arr > cfg.binary_thresh, label)
                out_path: Path = out_dir / slide / "1.0" / f.name
                out_path.parent.mkdir(parents=True, exist_ok=True)
                Image.fromarray(out, mode="RGB").save(out_path)

            if verbose and (i < 3 or i % 500 == 0):
                print(f"  [{slide}] {i+1}/{len(files)} {f.name} "
                      f"fg={100*ratio:.4f}% → 标签{label}")

        per_slide[slide] = counts
        print(f"[{slide}] {len(files)} 张，标签0/1/2 = "
              f"{counts[0]}/{counts[1]}/{counts[2]}，累计 {time.time()-t0:.1f}s")

    total: int = len(records)
    overall: dict[int, int] = {0: 0, 1: 0, 2: 0}
    px: dict[int, int] = {0: 0, 1: 0, 2: 0}
    for r in records:
        overall[r.label] += 1
        px[r.label] += r.n_fg_px

    print("\n=== 总体类别分布 ===")
    for c in (0, 1, 2):
        print(f"  标签{c}（{CLASS_NAMES[c]:<18s}）: {overall[c]:5d} 张 "
              f"({100*overall[c]/total:5.1f}%)，前景像素 {px[c]:>12,}")
    print(f"  合计: {total} 张")

    print("\n=== 每切片 ===")
    print(f"  {'切片':<12} {'总数':>6} {'标签0':>7} {'标签1':>7} {'标签2':>7}")
    for slide in slides:
        c = per_slide[slide]
        print(f"  {slide:<12} {sum(c.values()):>6} {c[0]:>7} {c[1]:>7} {c[2]:>7}")

    # 写索引 csv（供 Dataset 查表）
    if dry_run:
        print("\n[make3] dry-run：未写任何文件")
        return 0

    index_path: Path = data_root / DEFAULT_INDEX_NAME
    with index_path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(["slide", "name", "n_fg_px", "fg_ratio", "label"])
        for r in records:
            writer.writerow([r.slide, r.name, r.n_fg_px, f"{r.fg_ratio:.8f}", r.label])
    print(f"\n[make3] 标签索引: {index_path}")
    print(f"[make3] 已生成 {total} 张彩色 mask → {out_dir}")
    return 0


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p: argparse.ArgumentParser = argparse.ArgumentParser(
        description="把二值 mask 转成三分类彩色 mask（0 背景 / 1 少肿瘤 / 2 多肿瘤）",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT,
                   help="数据根目录（其下有 train_masks/）")
    p.add_argument("--out-root", type=Path, default=DEFAULT_OUT_ROOT,
                   help="输出目录；相对路径时拼到 --data-root 下")
    p.add_argument("--low", type=float, default=0.0001, help="标签 1 下界（前景占比）")
    p.add_argument("--high", type=float, default=0.05, help="标签 2 下界（前景占比）")
    p.add_argument("--binary-thresh", type=int, default=127, help="二值化阈值")
    p.add_argument("--dry-run", action="store_true", help="只统计，不写文件")
    p.add_argument("--verbose", "-v", action="store_true", help="打印每张 patch 的结果")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args: argparse.Namespace = parse_args(argv)
    cfg: EdgeConfig = EdgeConfig(low=args.low, high=args.high,
                                 binary_thresh=args.binary_thresh)
    return run(args.data_root, args.out_root, cfg,
               dry_run=args.dry_run, verbose=args.verbose)


if __name__ == "__main__":
    sys.exit(main())
