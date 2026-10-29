"""把 train_masks/ 的二值 mask 转成三分类彩色 mask（按**连通域面积 + 数量**判定）。

为什么不用前景占比
------------------
旧规则只看 ``前景像素 / 1024²``（<5% 记「少量」，≥5% 记「较多」）。这个量把
「一大块瘤」和「许多小灶加起来很多」混在同一个数里：一张 200 个 300 px 小灶
的 patch 与一张单个 6 万 px 实性瘤块的 patch 被评为同类，但形态完全不同。
新规则直接对前景做连通域分析：**最大连通域面积**衡量「有没有一块实体瘤灶」，
**连通域数量 + 主体占比**衡量「是不是散成碎片」。

判定规则（详解见 ``labels.py``）::

    前景像素 < 128                                 → 0（背景）
    A_max ≥ 50,000 px（≈4.8% patch）               → 2（大块）
    A_max ≥ 20,000 px 且 n_cc ≤ 2 且 dom ≥ 0.8     → 2（中等但成团）
    其余有前景                                      → 1（零星）

阈值依据：
- ``50000 px`` = 整张 patch 的 4.8%，1.0 mpp 下约 0.05 mm²、220 µm 见方，
  是一块肉眼可辨的实体瘤灶；
- ``20000 px`` 单看面积偏小，但若**几乎就是一整块**（域数 ≤2、最大域占前景
  ≥80%），仍算大块——这正是「用数量区分」的部分：同样 2 万像素，一整块是瘤灶，
  几十个碎点只是零星浸润；
- ``128 px``（≈11×11）以下视为标注噪声并入背景。

本地实测分布（4953 张）：标签 0 / 1 / 2 = 4140 / 149 / 664（83.6% / 3.0% / 13.4%）。

输出
----
``data/toxpath/train_masks_classes/{slide}/1.0/{X}_{Y}.png``

- RGB 三通道 PNG，尺寸与输入 mask 完全一致；
- 背景 (0,0,0) 黑；标签 1 前景 (0,255,0) 绿；标签 2 前景 (255,0,0) 红；
- 同一张 patch 的所有前景像素着同一个颜色（类别由该 patch 的连通域统计决定）。

同时输出 ``data/toxpath/_patch_labels_3class.csv``，列为
``slide,name,n_fg_px,fg_ratio,n_cc,max_cc_px,dominance,label``。
``fg_ratio`` 保留是为了兼容 ``splits.py`` 的肿瘤负荷排序与训练端的空真值统计；
新增三列便于事后按面积/域数重新划分类别，不必重算连通域。

用法
----
.. code-block:: bash

    python scripts/make_3class_masks.py --dry-run      # 只统计不写文件
    python scripts/make_3class_masks.py                # 正式生成
    python scripts/make_3class_masks.py --big-area 80000 --mid-area 30000
    python scripts/make_3class_masks.py --mass-max-cc 1    # 更严格：只认单块
"""

from __future__ import annotations

import argparse
import csv
import sys
import time
from pathlib import Path
from typing import NamedTuple

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from labels import CLASS_COLORS, CLASS_NAMES, CcConfig, component_areas  # noqa: E402
from paths import describe_paths, resolve_data_root  # noqa: E402

DEFAULT_OUT_ROOT: Path = Path("train_masks_classes")
DEFAULT_INDEX_NAME: str = "_patch_labels_3class.csv"


# --------------------------------------------------------------------------- #
# 单张 mask 的结果
# --------------------------------------------------------------------------- #
class MaskRecord(NamedTuple):
    slide: str
    name: str
    n_fg_px: int
    fg_ratio: float
    n_cc: int
    max_cc_px: int
    dominance: float
    label: int


# --------------------------------------------------------------------------- #
# 单张处理
# --------------------------------------------------------------------------- #
def analyze_mask(path: Path, cfg: CcConfig) -> MaskRecord:
    """读取一张二值 mask，做连通域分析并定类。"""
    arr: np.ndarray = np.asarray(Image.open(path).convert("L"), dtype=np.uint8)
    fg: np.ndarray = arr > cfg.binary_thresh
    total: int = int(fg.size)
    n_fg: int = int(fg.sum())

    if n_fg == 0:
        n_cc: int = 0
        max_cc: int = 0
    else:
        areas: np.ndarray = component_areas(fg)
        max_cc = int(areas[0])
        n_cc = int((areas >= cfg.cc_min_area).sum())

    return MaskRecord(
        slide=path.parent.parent.name,
        name=path.name,
        n_fg_px=n_fg,
        fg_ratio=n_fg / total,
        n_cc=n_cc,
        max_cc_px=max_cc,
        dominance=(max_cc / n_fg) if n_fg > 0 else 0.0,
        label=cfg.label_of(n_fg, n_cc, max_cc),
    )


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


def _describe_rule(cfg: CcConfig) -> str:
    """把规则压成一行，便于日志里核对。"""
    return (f"fg<{cfg.min_fg_px}px → 0 ; "
            f"A_max≥{cfg.big_area_px}px → 2 ; "
            f"A_max≥{cfg.mid_area_px}px 且 n_cc≤{cfg.mass_max_cc} "
            f"且 dom≥{cfg.mass_min_dominance} → 2 ; 其余有前景 → 1")


# --------------------------------------------------------------------------- #
# 主流程
# --------------------------------------------------------------------------- #
def run(
    data_root: Path,
    out_root: Path,
    cfg: CcConfig,
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
    print(f"[make3] 规则    : {_describe_rule(cfg)}")
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
            rec: MaskRecord = analyze_mask(f, cfg)
            records.append(rec)
            counts[rec.label] += 1

            if not dry_run:
                arr: np.ndarray = np.asarray(Image.open(f).convert("L"), dtype=np.uint8)
                out: np.ndarray = render_mask(arr > cfg.binary_thresh, rec.label)
                out_path: Path = out_dir / slide / "1.0" / f.name
                out_path.parent.mkdir(parents=True, exist_ok=True)
                Image.fromarray(out, mode="RGB").save(out_path)

            if verbose and (i < 3 or i % 200 == 0):
                print(f"  [{slide}] {i+1}/{len(files)} {f.name} "
                      f"fg={rec.n_fg_px:>7,} n_cc={rec.n_cc:>3} "
                      f"A_max={rec.max_cc_px:>7,} dom={rec.dominance:.2f} "
                      f"→ 标签{rec.label}")

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
        print(f"  标签{c}（{CLASS_NAMES[c]:<8s}）: {overall[c]:5d} 张 "
              f"({100*overall[c]/total:5.1f}%)，前景像素 {px[c]:>12,}")
    print(f"  合计: {total} 张")

    # 连通域统计，便于校核阈值是否合理
    fg_recs: list[MaskRecord] = [r for r in records if r.n_fg_px > 0]
    if fg_recs:
        n_cc: np.ndarray = np.array([r.n_cc for r in fg_recs])
        max_cc: np.ndarray = np.array([r.max_cc_px for r in fg_recs])
        dom: np.ndarray = np.array([r.dominance for r in fg_recs])
        qs: tuple[int, ...] = (1, 5, 25, 50, 75, 95, 99)

        def fmt(name: str, v: np.ndarray, pct: bool = False) -> str:
            items = "  ".join(
                f"p{q}={np.percentile(v, q):.2f}" if pct
                else f"p{q}={int(np.percentile(v, q)):,}"
                for q in qs
            )
            return f"  {name:<10s}{items}"

        print(f"\n=== 有前景 patch 的连通域统计（{len(fg_recs)} 张）===")
        print(fmt("连通域数", n_cc))
        print(fmt("最大域面积", max_cc))
        print(fmt("主体占比", dom, pct=True))
        hist: dict[int, int] = {}
        for k in n_cc.tolist():
            key = 6 if k >= 6 else k
            hist[key] = hist.get(key, 0) + 1
        print("  域数直方图: " + "  ".join(
            f"{k if k < 6 else '6+'}:{hist.get(k, 0)}" for k in range(7)))

    print("\n=== 每切片 ===")
    print(f"  {'切片':<12} {'总数':>6} {'标签0':>7} {'标签1':>7} {'标签2':>7}")
    for slide in slides:
        c = per_slide[slide]
        print(f"  {slide:<12} {sum(c.values()):>6} {c[0]:>7} {c[1]:>7} {c[2]:>7}")

    # 写索引 csv（供 Dataset / splits 查表）
    if dry_run:
        print("\n[make3] dry-run：未写任何文件")
        return 0

    index_path: Path = data_root / DEFAULT_INDEX_NAME
    with index_path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(["slide", "name", "n_fg_px", "fg_ratio", "n_cc",
                         "max_cc_px", "dominance", "label"])
        for r in records:
            writer.writerow([r.slide, r.name, r.n_fg_px, f"{r.fg_ratio:.8f}",
                            r.n_cc, r.max_cc_px, f"{r.dominance:.4f}", r.label])
    print(f"\n[make3] 标签索引: {index_path}")
    print(f"[make3] 已生成 {total} 张彩色 mask → {out_dir}")
    print(f"[make3] 耗时 {time.time()-t0:.1f}s")
    return 0


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p: argparse.ArgumentParser = argparse.ArgumentParser(
        description="把二值 mask 转成三分类彩色 mask（0 背景 / 1 零星肿瘤 / 2 大块肿瘤）",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--data-root", type=Path, default=None,
                   help="数据根目录（其下有 train_masks/）；不填则按 paths.py 自动解析")
    p.add_argument("--out-root", type=Path, default=DEFAULT_OUT_ROOT,
                   help="输出目录；相对路径时拼到 --data-root 下")
    p.add_argument("--min-fg-px", type=int, default=128,
                   help="前景少于此像素数 → 标签 0（背景）")
    p.add_argument("--big-area", type=int, default=50_000,
                   help="最大连通域面积 ≥ 此值 → 标签 2（大块）")
    p.add_argument("--mid-area", type=int, default=20_000,
                   help="成团判定的面积下界（配合 --mass-max-cc / --mass-min-dominance）")
    p.add_argument("--mass-max-cc", type=int, default=2,
                   help="成团判定：有效连通域个数上限")
    p.add_argument("--mass-min-dominance", type=float, default=0.8,
                   help="成团判定：最大域占前景像素比例下限")
    p.add_argument("--cc-min-area", type=int, default=64,
                   help="统计连通域个数时忽略的碎片面积上限")
    p.add_argument("--binary-thresh", type=int, default=127, help="二值化阈值")
    p.add_argument("--dry-run", action="store_true", help="只统计，不写文件")
    p.add_argument("--verbose", "-v", action="store_true", help="打印详细过程")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args: argparse.Namespace = parse_args(argv)
    try:
        args.data_root = resolve_data_root(args.data_root)
    except FileNotFoundError as exc:
        print(f"[error] {exc}", file=sys.stderr)
        return 2
    print("[make3] 路径：")
    print(describe_paths(args.data_root))
    cfg: CcConfig = CcConfig(
        min_fg_px=args.min_fg_px,
        big_area_px=args.big_area,
        mid_area_px=args.mid_area,
        mass_max_cc=args.mass_max_cc,
        mass_min_dominance=args.mass_min_dominance,
        cc_min_area=args.cc_min_area,
        binary_thresh=args.binary_thresh,
    )
    return run(args.data_root, args.out_root, cfg,
               dry_run=args.dry_run, verbose=args.verbose)


if __name__ == "__main__":
    sys.exit(main())
