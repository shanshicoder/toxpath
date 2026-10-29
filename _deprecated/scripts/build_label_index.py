"""扫 train_masks/ 生成三分类标签索引（路径 B：预计算 + 运行时查表）。

为什么需要这一步
-----------------
``Dataset.__getitem__`` 每次跑 scipy ``ndimage.label`` 太慢（30-50ms/张，
一个 epoch 多花 2-4 分钟）。本脚本预先扫一遍所有 mask，把每张 patch 的
三分类标签算出来存成 5KB 的 ``.npz`` 索引文件，训练时只查表 O(1)。

复用 ``scripts/relabel_masks.py`` 的核心函数
-----------------------------------------
- ``RelabelConfig``：规则 C 的阈值配置
- ``analyze_mask``：填空洞 + 8-连通标注 + 剔噪点 + 算指标
- ``classify``：规则 C 决策树
- ``find_slides`` / ``collect_mask_files``：枚举切片与文件

输出
----
- ``data/toxpath/_patch_labels.npz``：``{slide_name: np.ndarray[int8]}``，
  每个数组长度等于该切片的 patch 数，顺序与 ``collect_mask_files`` 返回的
  排序一致（文件名字典序）。
- ``data/toxpath/_patch_meta.csv``（``--with-meta`` 时）：每行一张 patch，
  含 slide/name/label/fg_ratio/num_cc/max_cc_ratio/num_dropped/max_dropped。
  可用 pandas 或 Excel 打开做敏感性分析。

用法
----
.. code-block:: bash

    # 默认规则 C 阈值，生成 .npz
    python scripts/build_label_index.py

    # 同时输出诊断 csv（用于调参时看分布）
    python scripts/build_label_index.py --with-meta

    # dry-run：只打印分布，不写文件
    python scripts/build_label_index.py --dry-run

    # 自定义阈值（重新生成）
    python scripts/build_label_index.py --max-cc 0.15 --multi-fg 0.10
"""

from __future__ import annotations

import argparse
import csv
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

# 复用 relabel_masks.py 的核心逻辑（同目录下）
sys.path.insert(0, str(Path(__file__).resolve().parent))
from relabel_masks import (  # noqa: E402  (sys.path 注入后导入)
    RelabelConfig,
    MaskStats,
    analyze_mask,
    classify,
    collect_mask_files,
    find_slides,
)


# --------------------------------------------------------------------------- #
# 默认路径
# --------------------------------------------------------------------------- #
DEFAULT_DATA_ROOT: Path = Path("E:/project/toxpath/data/toxpath")
"""train_masks / train_patches 的共同父目录。"""

DEFAULT_OUT_DIR: Path = DEFAULT_DATA_ROOT
"""索引文件默认写到数据目录下，与数据同根便于一起 rsync/git。"""


# --------------------------------------------------------------------------- #
# 核心构建函数
# --------------------------------------------------------------------------- #
def build_index(
    data_root: Path,
    cfg: RelabelConfig,
    *,
    out_dir: Path,
    with_meta: bool = False,
    dry_run: bool = False,
    verbose: bool = False,
) -> dict[str, Any]:
    """扫描所有 mask，返回 ``{slide: np.ndarray[int8]}`` 索引。

    Parameters
    ----------
    data_root:
        数据根目录，下含 ``train_masks/{slide}/1.0/*.png``。
    cfg:
        规则 C 阈值配置。
    out_dir:
        索引文件输出目录。
    with_meta:
        是否同时输出诊断 csv。
    dry_run:
        为 True 时只打印分布、不写文件。
    verbose:
        为 True 时每 200 张打印一次进度。

    Returns
    -------
    dict
        ``{"labels": {slide: ndarray}, "meta_rows": list, "stats": {...}}``。
    """
    train_masks: Path = data_root / "train_masks"
    slides: list[str] = find_slides(train_masks)

    print(f"[build_index] 数据根目录: {data_root}")
    print(f"[build_index] train_masks: {train_masks}")
    print(f"[build_index] 切片数: {len(slides)} → {slides}")
    print(f"[build_index] 阈值: {cfg}")
    print(f"[build_index] with_meta={with_meta} dry_run={dry_run}")

    labels: dict[str, np.ndarray] = {}
    meta_rows: list[dict[str, Any]] = []
    overall_counts: dict[int, int] = {1: 0, 2: 0, 3: 0}

    t0: float = time.time()
    for slide in slides:
        files: list[Path] = collect_mask_files(train_masks, slide)
        slide_labels: np.ndarray = np.zeros(len(files), dtype=np.int8)
        for i, f in enumerate(files):
            # 复用 relabel_masks.py 的 process_one 逻辑（但不取 cleaned mask）
            arr: np.ndarray = np.array(Image.open(f).convert("L"))
            binary: np.ndarray = arr > cfg.binary_thresh
            raw_fg_ratio: float = float(binary.mean())

            num_cc: int
            max_cc_ratio: float
            fg_ratio_clean: float
            n_dropped: int
            max_dropped: int
            _cleaned: np.ndarray  # 索引构建不需要 cleaned，丢弃
            (num_cc, max_cc_ratio, fg_ratio_clean, n_dropped, max_dropped, _cleaned) = (
                analyze_mask(binary, cfg)
            )

            label: int = classify((fg_ratio_clean, num_cc, max_cc_ratio), cfg)
            slide_labels[i] = label
            overall_counts[label] += 1

            if with_meta:
                meta_rows.append({
                    "slide": slide,
                    "name": f.name,
                    "label": label,
                    "raw_fg_ratio": raw_fg_ratio,
                    "fg_ratio_clean": fg_ratio_clean,
                    "num_cc": num_cc,
                    "max_cc_ratio": max_cc_ratio,
                    "num_dropped_noise": n_dropped,
                    "max_dropped_area": max_dropped,
                })

            if verbose and (i % 200 == 0 or i < 3):
                print(f"  [{slide}] {i+1}/{len(files)}  {f.name}  "
                      f"fg={fg_ratio_clean:.4f} cc={num_cc} max_cc={max_cc_ratio:.4f} → 标签{label}")

        labels[slide] = slide_labels
        elapsed: float = time.time() - t0
        # 每切片内分布
        s_counts = {k: int((slide_labels == k).sum()) for k in (1, 2, 3)}
        print(f"  [{slide}] 完成 {len(files)} 个，累计 {sum(overall_counts.values())}，"
              f"耗时 {elapsed:.1f}s  标签1/2/3 = {s_counts[1]}/{s_counts[2]}/{s_counts[3]}")

    total: int = sum(overall_counts.values())
    print(f"\n[build_index] 全量完成 {total} 个，耗时 {time.time()-t0:.1f}s")
    print(f"  整体分布: 标签1={overall_counts[1]} ({100*overall_counts[1]/total:.1f}%)  "
          f"标签2={overall_counts[2]} ({100*overall_counts[2]/total:.1f}%)  "
          f"标签3={overall_counts[3]} ({100*overall_counts[3]/total:.1f}%)")

    # 写文件
    if not dry_run:
        out_dir.mkdir(parents=True, exist_ok=True)
        npz_path: Path = out_dir / "_patch_labels.npz"
        # np.savez 要求 key 是合法标识符，slide 名 train_001 等都 OK
        np.savez(npz_path, **labels)
        print(f"\n[build_index] 索引写入: {npz_path}  "
              f"({npz_path.stat().st_size//1024} KB)")

        if with_meta and meta_rows:
            csv_path: Path = out_dir / "_patch_meta.csv"
            with csv_path.open("w", newline="", encoding="utf-8") as fh:
                writer = csv.DictWriter(fh, fieldnames=list(meta_rows[0].keys()))
                writer.writeheader()
                writer.writerows(meta_rows)
            print(f"[build_index] 诊断 csv: {csv_path}  "
                  f"({csv_path.stat().st_size//1024} KB)")
    else:
        print("\n[build_index] === dry-run 模式，未写任何文件 ===")

    return {"labels": labels, "meta_rows": meta_rows, "overall": overall_counts}


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """命令行参数解析。"""
    p = argparse.ArgumentParser(
        description="扫 train_masks/ 生成三分类标签索引 (_patch_labels.npz)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--data-root", type=Path, default=DEFAULT_DATA_ROOT,
        help="数据根目录（含 train_masks/ 与 train_patches/）",
    )
    p.add_argument(
        "--out-dir", type=Path, default=DEFAULT_OUT_DIR,
        help="索引文件输出目录",
    )
    p.add_argument("--with-meta", action="store_true",
                   help="同时输出 _patch_meta.csv 诊断表")
    p.add_argument("--dry-run", action="store_true",
                   help="只打印分布，不写文件")
    p.add_argument("--verbose", action="store_true",
                   help="每 200 张打印一次进度")
    # 规则 C 阈值（与 relabel_masks.py 一致）
    p.add_argument("--fg-zero", type=float, default=0.0005,
                   help="fg_ratio < 此值视为纯背景")
    p.add_argument("--max-cc", type=float, default=0.20,
                   help="最大连通域占比阈值")
    p.add_argument("--multi-num-cc", type=int, default=3,
                   help="多灶兜底要求的连通域数量下限")
    p.add_argument("--multi-fg", type=float, default=0.15,
                   help="多灶兜底要求的前景占比下限")
    p.add_argument("--min-cc-area", type=int, default=64,
                   help="面积小于此值的连通域视为噪点")
    p.add_argument("--binary-thresh", type=int, default=127,
                   help="mask 二值化阈值")
    p.add_argument("--no-fill-holes", action="store_true",
                   help="不做 binary_fill_holes（默认做）")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """CLI 入口。"""
    args: argparse.Namespace = parse_args(argv)

    cfg: RelabelConfig = RelabelConfig(
        fg_zero=args.fg_zero,
        max_cc_thresh=args.max_cc,
        multi_num_cc=args.multi_num_cc,
        multi_fg_thresh=args.multi_fg,
        min_cc_area=args.min_cc_area,
        binary_thresh=args.binary_thresh,
        fill_holes=not args.no_fill_holes,
    )

    build_index(
        args.data_root,
        cfg,
        out_dir=args.out_dir,
        with_meta=args.with_meta,
        dry_run=args.dry_run,
        verbose=args.verbose,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
