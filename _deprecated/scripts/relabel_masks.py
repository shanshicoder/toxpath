"""按规则 C 把二值 mask 重新分类为图像级三分类标签（1/2/3）。

规则 C 决策树（按顺序命中即定，详见 ``标签重定义方案.md`` §3）::

    1. fg_ratio < 0.0005              → 标签 1（纯背景）
    2. max_cc_ratio ≥ 0.20           → 标签 3（连续成片的大块）
    3. num_cc ≥ 3 且 fg_ratio ≥ 0.15 → 标签 3（多灶累积型，兜底）
    4. 其他                           → 标签 2（零星/小灶/中等）

输入  : data/toxpath/train_masks/{slide}/1.0/{X}_{Y}.png   (0/255 二值)
输出  : data/toxpath/train_masks_class/{slide}/1.0/{X}_{Y}.png  (单通道灰度, 值域 {0,128,255})

输出编码（**三类，不是四类**）
------------------------------
================  ===========  ====================================
像素值            显示         含义
================  ===========  ====================================
0                 黑           背景（非肿瘤区域）。纯背景 patch 整图都是 0
128               中灰         前景，patch 属于类别2（零星/小灶/中等）
255               白           前景，patch 属于类别3（较大块/大块/多灶累积）
================  ===========  ====================================

为什么要用 0/128/255 而不是 1/2/3
--------------------------------
1、2、3 都是"接近纯黑"的灰度，打开图肉眼完全看不出区别。三个类别间隔
128 个灰阶后，黑 / 中灰 / 白一眼可分，可以直接目视质检；同时全图仍然
只有 3 个值，严格三类。

即"类别"体现在**前景像素的取值**上，背景恒为 0：
纯背景 patch → 全 0；小灶 patch → 背景 0 + 前景 128；
大块 patch → 背景 0 + 前景 255。
"""

from __future__ import annotations

import argparse
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import NamedTuple

import numpy as np
from PIL import Image
from scipy import ndimage


# --------------------------------------------------------------------------- #
# 配置
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class RelabelConfig:
    """重标签规则 C 的所有阈值（frozen，构造后不可变）。

    所有阈值都从 CLI 注入，方便后续做敏感性分析。
    """

    fg_zero: float = 0.0005
    """fg_ratio < 此值视为纯背景（标签 1）。

    实测最小前景 patch 的 fg_ratio ≈ 0.0001，0.0005 留 5× 余量，
    避免把"5 个像素的噪点"误判为有前景。"""

    max_cc_thresh: float = 0.20
    """最大连通域占比 ≥ 此值 → 标签 3（连续成片）。

    与原分桶"5–20% 中等 / 20–50% 较大块"的天然边界 20% 对齐。"""

    multi_num_cc: int = 3
    """多灶兜底规则要求的连通域数量下限。"""

    multi_fg_thresh: float = 0.15
    """多灶兜底规则要求的前景占比下限。"""

    min_cc_area: int = 64
    """面积 < 此值的连通域视为噪点，剔除后再算 num_cc 与 max_cc。

    8×8 = 64 px，对应 20x mpp 下约 4μm×4μm，小于任何有意义的肿瘤灶。"""

    binary_thresh: int = 127
    """二值化阈值：mask 像素 > 此值视为前景。"""

    fill_holes: bool = True
    """是否在判定前做 binary_fill_holes。

    True 可让环形标注变成一个连通域，避免把一个肿瘤误数成 2 个。
    """


# --------------------------------------------------------------------------- #
# 单 mask 分析结果
# --------------------------------------------------------------------------- #
class MaskStats(NamedTuple):
    """单张 mask 的分析结果。"""

    slide: str
    name: str
    fg_ratio: float
    """**剔噪点后**的前景占比（用于决策）。

    注意是剔噪点后的值，不是原始前景占比。这样椒盐噪点 mask
    （如最大连通域仅 2px、前景全是散点）会被正确归为标签1，
    而不是误判为标签2。
    """
    raw_fg_ratio: float
    """原始前景占比（含噪点），仅用于诊断与日志。"""
    num_cc: int
    max_cc_ratio: float
    num_dropped_noise: int
    """被 min_cc_area 过滤掉的噪点连通域数量。"""
    max_dropped_area: int
    """被过滤的噪点中最大那一个的面积（便于人工复核阈值是否合理）。"""
    label: int
    """最终标签 1/2/3。"""


# --------------------------------------------------------------------------- #
# 核心分析函数
# --------------------------------------------------------------------------- #
def clean_binary(binary: np.ndarray, cfg: RelabelConfig) -> np.ndarray:
    """对二值图做填空洞 + 剔噪点，返回**清理后的前景布尔 mask**。

    处理顺序（重要，不能颠倒）:
        1. ``binary_fill_holes`` —— 先填空洞，避免环形标注被切成多个连通域
        2. ``ndimage.label`` —— 8-连通标注
        3. 剔除面积 < ``min_cc_area`` 的噪点连通域
        4. 把保留的连通域重新组合成一张布尔 mask

    这套清理逻辑被 ``analyze_mask``（算指标）与 ``write_relabeled_mask``
    （写新 mask）共用，保证两者对"前景"的定义完全一致——
    决策用的前景 = 写出的前景。

    Parameters
    ----------
    binary:
        形状 ``[H, W]`` 的布尔数组，True = 前景（来自原 mask 二值化）。
    cfg:
        阈值配置。

    Returns
    -------
    np.ndarray
        形状 ``[H, W]`` 的布尔数组，True = 剔噪点后保留的前景。
        若原图无前景或全被当噪点剔了，返回全 False。
    """
    if not binary.any():
        return np.zeros_like(binary)

    # 1) 填空洞
    filled: np.ndarray = ndimage.binary_fill_holes(binary) if cfg.fill_holes else binary

    # 2) 8-连通标注
    labeled: np.ndarray
    n_raw: int
    labeled, n_raw = ndimage.label(filled)
    if n_raw == 0:
        return np.zeros_like(binary)

    # 3) 各连通域面积
    sizes: np.ndarray = np.asarray(
        ndimage.sum(np.ones_like(labeled), labeled, index=range(1, n_raw + 1)),
        dtype=np.int64,
    )

    # 4) 剔噪点：面积 < min_cc_area 的连通域不要
    keep_labels: np.ndarray = np.where(sizes >= cfg.min_cc_area)[0] + 1  # +1 因为 label 从 1 起
    if len(keep_labels) == 0:
        return np.zeros_like(binary)

    # 5) 重建布尔 mask：只保留通过的连通域
    kept: np.ndarray = np.isin(labeled, keep_labels)
    return kept


def analyze_mask(binary: np.ndarray, cfg: RelabelConfig) -> tuple[int, float, float, int, int, np.ndarray]:
    """对二值图做连通域分析，返回指标 + 清理后的前景 mask。

    处理顺序与 ``clean_binary`` 完全一致，只是在清理后额外统计指标。

    Parameters
    ----------
    binary:
        形状 ``[H, W]`` 的布尔数组，True = 前景。
    cfg:
        阈值配置。

    Returns
    -------
    tuple ``(num_cc, max_cc_ratio, fg_ratio_clean, num_dropped_noise, max_dropped_area, cleaned)``
        - ``num_cc``: 剔噪后的连通域数量
        - ``max_cc_ratio``: 最大连通域面积 / 总像素
        - ``fg_ratio_clean``: **剔噪点后**的前景占比（用于决策）
        - ``num_dropped_noise``: 被过滤的噪点连通域数
        - ``max_dropped_area``: 被过滤噪点中的最大面积
        - ``cleaned``: 剔噪点后的前景布尔 mask（与输入同形状）
    """
    total: int = binary.size
    fg: int = int(binary.sum())
    if fg == 0:
        return 0, 0.0, 0.0, 0, 0, np.zeros_like(binary)

    # 1) 填空洞
    filled: np.ndarray = ndimage.binary_fill_holes(binary) if cfg.fill_holes else binary

    # 2) 8-连通标注
    labeled: np.ndarray
    n_raw: int
    labeled, n_raw = ndimage.label(filled)
    if n_raw == 0:
        return 0, 0.0, 0.0, 0, 0, np.zeros_like(binary)

    # 3) 各连通域面积
    sizes: np.ndarray = np.asarray(
        ndimage.sum(np.ones_like(labeled), labeled, index=range(1, n_raw + 1)),
        dtype=np.int64,
    )

    # 4) 剔噪点
    keep_mask: np.ndarray = sizes >= cfg.min_cc_area
    n_keep: int = int(keep_mask.sum())
    n_dropped: int = n_raw - n_keep
    max_dropped: int = int(sizes[~keep_mask].max()) if n_dropped > 0 else 0

    if n_keep == 0:
        # 所有连通域都被当噪点剔了 —— 视为无前景
        return 0, 0.0, 0.0, n_dropped, max_dropped, np.zeros_like(binary)

    kept_sizes: np.ndarray = sizes[keep_mask]
    max_cc_ratio: float = float(kept_sizes.max()) / total
    # 剔噪点后的前景占比（这才是决策用的）
    fg_ratio_clean: float = float(kept_sizes.sum()) / total
    # 重建清理后的布尔 mask
    keep_labels: np.ndarray = np.where(keep_mask)[0] + 1  # label 从 1 起
    cleaned: np.ndarray = np.isin(labeled, keep_labels)
    return n_keep, max_cc_ratio, fg_ratio_clean, n_dropped, max_dropped, cleaned


def classify(stats: tuple[float, int, float], cfg: RelabelConfig) -> int:
    """按规则 C 决策树定标签。

    Parameters
    ----------
    stats:
        ``(fg_ratio, num_cc, max_cc_ratio)``。
    cfg:
        阈值配置。

    Returns
    -------
    int
        标签 1/2/3。
    """
    fg_ratio, num_cc, max_cc_ratio = stats

    # 规则 1：纯背景
    if fg_ratio < cfg.fg_zero:
        return 1
    # 规则 2：连续成片的大块
    if max_cc_ratio >= cfg.max_cc_thresh:
        return 3
    # 规则 3：多灶累积兜底
    if num_cc >= cfg.multi_num_cc and fg_ratio >= cfg.multi_fg_thresh:
        return 3
    # 规则 4：零星/小灶/中等
    return 2


def process_one(mask_path: Path, slide: str, cfg: RelabelConfig) -> tuple[MaskStats, np.ndarray]:
    """处理单张 mask，返回统计结果 + 清理后的前景 mask。

    决策用的是 **剔噪点后** 的 fg_ratio（``MaskStats.fg_ratio`` 字段），
    这样椒盐噪点 mask 会被正确归为标签1。原始前景占比记录在
    ``MaskStats.raw_fg_ratio`` 仅用于诊断。

    返回的 ``cleaned`` 是剔噪点+填空洞后的前景布尔 mask，
    供 ``write_relabeled_mask`` 直接用来生成新 mask——
    这样保证"决策用的前景 = 写出的前景"，不会出现
    "判定为标签1 但写出的 mask 还有噪点像素"的不一致。

    Parameters
    ----------
    mask_path:
        原 mask 路径。
    slide:
        切片名（用于日志）。
    cfg:
        阈值配置。

    Returns
    -------
    tuple ``(stats, cleaned)``
        - ``stats``: ``MaskStats`` 含标签与诊断指标
        - ``cleaned``: 形状 ``[H, W]`` 的布尔数组，True = 剔噪点后保留的前景
    """
    arr: np.ndarray = np.array(Image.open(mask_path).convert("L"))
    binary: np.ndarray = arr > cfg.binary_thresh
    raw_fg_ratio: float = float(binary.mean())

    num_cc: int
    max_cc_ratio: float
    fg_ratio_clean: float
    n_dropped: int
    max_dropped: int
    cleaned: np.ndarray
    num_cc, max_cc_ratio, fg_ratio_clean, n_dropped, max_dropped, cleaned = analyze_mask(binary, cfg)

    # classify 用剔噪点后的 fg_ratio
    label: int = classify((fg_ratio_clean, num_cc, max_cc_ratio), cfg)
    stats: MaskStats = MaskStats(
        slide=slide,
        name=mask_path.name,
        fg_ratio=fg_ratio_clean,
        raw_fg_ratio=raw_fg_ratio,
        num_cc=num_cc,
        max_cc_ratio=max_cc_ratio,
        num_dropped_noise=n_dropped,
        max_dropped_area=max_dropped,
        label=label,
    )
    return stats, cleaned


# --------------------------------------------------------------------------- #
# 主流程
# --------------------------------------------------------------------------- #
def find_slides(data_root: Path) -> list[str]:
    """枚举 train_masks 下的所有切片目录名（如 train_001）。"""
    if not data_root.exists():
        raise FileNotFoundError(f"train_masks 不存在: {data_root}")
    slides: list[str] = sorted(
        [p.name for p in data_root.iterdir() if p.is_dir() and p.name.startswith("train_")]
    )
    if not slides:
        raise FileNotFoundError(f"train_masks 下没有 train_* 子目录: {data_root}")
    return slides


def collect_mask_files(train_masks: Path, slide: str) -> list[Path]:
    """收集某切片下所有 mask 文件，按文件名排序。"""
    slide_dir: Path = train_masks / slide / "1.0"
    if not slide_dir.exists():
        raise FileNotFoundError(f"切片目录不存在: {slide_dir}")
    return sorted(slide_dir.glob("*.png"))


# --------------------------------------------------------------------------- #
# 标签语义与像素值
# --------------------------------------------------------------------------- #
LABEL_MEANING: dict[int, str] = {
    1: "纯背景",
    2: "零星/小灶/中等",
    3: "较大块/大块/多灶累积",
}
"""三类的语义。"""

LABEL_TO_PIXEL: dict[int, int] = {
    1: 0,      # 纯背景 → 黑
    2: 128,    # 小灶   → 中灰
    3: 255,    # 大块   → 纯白
}
"""patch 标签 → **输出 mask 的像素值**（单通道 8-bit 灰度）。

为什么是 0/128/255 而不是 1/2/3
-------------------------------
8-bit 灰度图里 1、2、3 都等于"接近纯黑"，三者在屏幕上肉眼完全分不出来，
没法做目视质检（打开一看就是一张全黑图）。所以把三个类别映射到
**间隔 128 的三个灰度级**：

===========  ========  ==============
像素值       显示      含义
===========  ========  ==============
0            黑        背景（纯背景 patch 整图都是 0）
128          中灰      前景：零星/小灶/中等
255          白        前景：较大块/大块/多灶累积
===========  ========  ==============

全图仍然**只有 3 个值**（0/128/255），没有第 4 类；
相邻两类灰度差 128，肉眼一眼可分，也不受显示器伽马影响。
"""

PIXEL_TO_LABEL: dict[int, int] = {v: k for k, v in LABEL_TO_PIXEL.items()}
"""输出像素值 → patch 标签（0→1, 128→2, 255→3）。"""

BACKGROUND_PIXEL: int = LABEL_TO_PIXEL[1]
"""背景像素值，等于 0。"""

FG_PIXEL_VALUES: tuple[int, int] = (LABEL_TO_PIXEL[2], LABEL_TO_PIXEL[3])
"""前景可能出现的两个像素值 (128, 255)。"""


def write_relabeled_mask(out_path: Path, cleaned: np.ndarray, label: int) -> None:
    """写一张**单通道灰度** mask PNG，像素值 ∈ {0, 128, 255}。

    输出规则::

        背景像素 = 0                              → 黑
        前景像素 = LABEL_TO_PIXEL[label] (128/255) → 中灰 / 纯白

    因此：
    - 类别1（纯背景）patch → ``cleaned`` 全 False → 整图全 0（全黑）
    - 类别2 patch → 背景 0（黑）+ 前景 128（中灰）
    - 类别3 patch → 背景 0（黑）+ 前景 255（白）

    三个值间隔 128，肉眼直接可辨（不像 1/2/3 那样全是黑的）。

    **关键一致性**：``cleaned`` 是 ``process_one`` 算指标时用的同一张
    剔噪点 mask，所以"判定为标签1 的 patch 输出整图 0"是保证的——
    噪点已经被剔掉，不会再以前景色出现。

    Parameters
    ----------
    out_path:
        输出路径，父目录会被自动创建。
    cleaned:
        形状 ``[H, W]`` 的布尔数组，True = 剔噪点后保留的前景。
    label:
        patch 标签 1/2/3。
    """
    if label not in LABEL_MEANING:
        raise ValueError(f"非法标签 {label}，只允许 1/2/3")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    h, w = cleaned.shape
    out: np.ndarray = np.full((h, w), BACKGROUND_PIXEL, dtype=np.uint8)
    if label != 1:
        out[cleaned] = np.uint8(LABEL_TO_PIXEL[label])
    Image.fromarray(out, mode="L").save(out_path)


def run(
    data_root: Path,
    out_root: Path,
    cfg: RelabelConfig,
    dry_run: bool,
    verbose: bool,
) -> int:
    """主流程：遍历所有 mask，分析并（可选）写出新标签。

    Returns
    -------
    int
        0 表示成功，非 0 表示出错。
    """
    train_masks: Path = data_root / "train_masks"
    train_masks_class: Path = out_root if out_root.is_absolute() else data_root / out_root
    slides: list[str] = find_slides(train_masks)

    print(f"[relabel] 数据根: {data_root}")
    print(f"[relabel] 输出根: {train_masks_class}")
    print(f"[relabel] 切片: {slides}")
    print(f"[relabel] 规则 C: max_cc≥{cfg.max_cc_thresh} 或 (num_cc≥{cfg.multi_num_cc} 且 fg≥{cfg.multi_fg_thresh})")
    print(f"[relabel] 噪点阈值: {cfg.min_cc_area}px, 填空洞: {cfg.fill_holes}, dry_run: {dry_run}")
    print()

    all_stats: list[MaskStats] = []
    per_slide: dict[str, dict[int, int]] = {}
    dropped_summary: dict[str, tuple[int, int]] = {}  # slide -> (total_dropped, max_dropped)
    t0: float = time.time()

    for slide in slides:
        files: list[Path] = collect_mask_files(train_masks, slide)
        slide_label_counts: dict[int, int] = {1: 0, 2: 0, 3: 0}
        total_dropped: int = 0
        slide_max_dropped: int = 0

        for i, f in enumerate(files):
            stats: MaskStats
            cleaned: np.ndarray
            stats, cleaned = process_one(f, slide, cfg)
            all_stats.append(stats)
            slide_label_counts[stats.label] += 1
            total_dropped += stats.num_dropped_noise
            if stats.max_dropped_area > slide_max_dropped:
                slide_max_dropped = stats.max_dropped_area

            if not dry_run:
                # 写新 mask PNG：背景=0，前景=128(类别2)/255(类别3)，值域 {0,128,255}
                # 用剔噪点后的 cleaned mask，保证"判定为标签1 → 整图全 0"
                rel: Path = Path(slide) / "1.0" / f.name
                out_path: Path = train_masks_class / rel
                write_relabeled_mask(out_path, cleaned, stats.label)

            if verbose and (i % 200 == 0 or i < 3):
                print(f"  [{slide}] {i+1}/{len(files)}  {f.name}  "
                      f"raw_fg={stats.raw_fg_ratio:.4f} fg={stats.fg_ratio:.4f} "
                      f"cc={stats.num_cc} max_cc={stats.max_cc_ratio:.4f} "
                      f"dropped={stats.num_dropped_noise} → 标签{stats.label}")

        per_slide[slide] = slide_label_counts
        dropped_summary[slide] = (total_dropped, slide_max_dropped)
        elapsed: float = time.time() - t0
        print(f"[{slide}] 完成 {len(files)} 个，累计 {len(all_stats)}，耗时 {elapsed:.1f}s  "
              f"标签1/2/3 = {slide_label_counts[1]}/{slide_label_counts[2]}/{slide_label_counts[3]}")

    # ---- 总体统计 ----
    total: int = len(all_stats)
    overall: dict[int, int] = {1: 0, 2: 0, 3: 0}
    for s in all_stats:
        overall[s.label] += 1
    total_dropped_all: int = sum(d[0] for d in dropped_summary.values())
    max_dropped_all: int = max(d[1] for d in dropped_summary.values()) if dropped_summary else 0

    print("\n=== 总体标签分布 ===")
    for lab in (1, 2, 3):
        print(f"  标签{lab}: {overall[lab]:5d} ({100*overall[lab]/total:.1f}%)")
    print(f"  合计: {total}")

    print("\n=== 每切片标签分布 ===")
    print(f"  {'切片':<12} {'总数':>6} {'标签1':>6} {'标签2':>6} {'标签3':>6} {'剔噪点':>8} {'最大噪点':>8}")
    for slide in slides:
        cnts = per_slide[slide]
        total_s = sum(cnts.values())
        dropped, max_dropped = dropped_summary[slide]
        print(f"  {slide:<12} {total_s:>6} {cnts[1]:>6} {cnts[2]:>6} {cnts[3]:>6} "
              f"{dropped:>8} {max_dropped:>8}")

    print(f"\n=== 噪点过滤 ===")
    print(f"  总剔噪点连通域: {total_dropped_all}")
    print(f"  最大噪点面积: {max_dropped_all}px  (>={cfg.min_cc_area}px 的才保留)")

    # ---- 多灶兜底命中的 patch（标签3 来自规则3而非规则2）----
    multi_hit: list[MaskStats] = [
        s for s in all_stats
        if s.label == 3
        and s.max_cc_ratio < cfg.max_cc_thresh  # 不是靠规则2进来的
        and s.num_cc >= cfg.multi_num_cc
        and s.fg_ratio >= cfg.multi_fg_thresh
    ]
    print(f"\n=== 多灶兜底命中（标签3 来自规则3）: {len(multi_hit)} 个 ===")
    for s in multi_hit[:10]:
        print(f"  {s.slide}/{s.name}: num_cc={s.num_cc} max_cc={s.max_cc_ratio:.4f} "
              f"fg={s.fg_ratio:.4f} (raw_fg={s.raw_fg_ratio:.4f})")

    # ---- 噪点驱动的标签1（raw_fg > 0 但剔噪点后 fg=0）---- #
    noise_driven_bg: list[MaskStats] = [
        s for s in all_stats
        if s.label == 1 and s.raw_fg_ratio > 0 and s.fg_ratio == 0.0
    ]
    print(f"\n=== 噪点驱动归标签1（原mask有前景但全是噪点）: {len(noise_driven_bg)} 个 ===")
    for s in noise_driven_bg[:10]:
        print(f"  {s.slide}/{s.name}: raw_fg={s.raw_fg_ratio:.4f} "
              f"dropped={s.num_dropped_noise} max_dropped={s.max_dropped_area}px")

    if dry_run:
        print("\n[relabel] === dry-run 模式，未写任何 PNG ===")
    else:
        # 写一份像素值 → 类别 的对照表，方便别人（或几个月后的自己）
        # 打开目录时不用翻代码就知道 0/128/255 分别是什么
        import json
        label_map_path: Path = train_masks_class / "_label_map.json"
        label_map_path.write_text(json.dumps({
            "encoding": "单通道 8-bit 灰度",
            "values": LABEL_TO_PIXEL,
            "meaning": LABEL_MEANING,
            "distribution": {str(k): v for k, v in overall.items()},
        }, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"\n[relabel] === 已写出 {total} 张标签 PNG 到 {train_masks_class} ===")
        print(f"[relabel] 像素值对照表: {label_map_path}  "
              f"(0=背景, 128=小灶, 255=大块)")

    return 0


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """解析命令行参数。"""
    p = argparse.ArgumentParser(
        description="按规则 C 把二值 mask 重新分类为图像级三分类标签 (1/2/3)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--data-root", type=Path, default=Path("E:/project/toxpath/data/toxpath"),
                   help="数据根目录（含 train_masks/）")
    p.add_argument("--out-root", type=Path, default=Path("train_masks_class"),
                   help="输出目录；相对路径时拼到 --data-root 下")
    p.add_argument("--dry-run", action="store_true",
                   help="只统计不写文件")
    p.add_argument("--verbose", "-v", action="store_true",
                   help="打印每个 patch 的指标")

    # 规则 C 阈值
    p.add_argument("--fg-zero", type=float, default=0.0005)
    p.add_argument("--max-cc", type=float, default=0.20)
    p.add_argument("--multi-num-cc", type=int, default=3)
    p.add_argument("--multi-fg", type=float, default=0.15)
    p.add_argument("--min-cc-area", type=int, default=64)
    p.add_argument("--binary-thresh", type=int, default=127)
    p.add_argument("--no-fill-holes", action="store_true",
                   help="禁用 binary_fill_holes（默认开启）")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """入口。"""
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
    return run(
        data_root=args.data_root,
        out_root=args.out_root,
        cfg=cfg,
        dry_run=args.dry_run,
        verbose=args.verbose,
    )


if __name__ == "__main__":
    sys.exit(main())
