"""三分类标签的唯一定义：类别颜色、名称、以及「面积 + 数量」定类规则。

三个地方必须用同一套规则，否则会悄悄错位（颜色对不上 → 训练端把前景当背景）：

- ``scripts/make_3class_masks.py``：生成训练用彩色 mask 与索引 CSV；
- ``dataset/toxpath3_dataset.py``：按颜色反解类别、按 CSV 取 patch 级类别；
- ``inference.py`` / ``predict_test.py``：给预测结果打 patch 级类别（仅展示用）。

所以颜色常量与定类函数都放这里，谁都不许自己再抄一份。

定类规则
--------
对前景做 **4 邻接**连通域分析，记 ``A_max`` = 最大连通域面积、``n_cc`` = 面积
≥ ``cc_min_area`` 的连通域个数、``dom`` = ``A_max / 前景像素``::

    前景像素 < min_fg_px                              → 0（背景）
    A_max ≥ big_area_px                               → 2（大块）
    A_max ≥ mid_area_px 且 n_cc ≤ mass_max_cc 且 dom ≥ mass_min_dominance
                                                      → 2（中等但成团）
    其余有前景                                        → 1（零星）
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

# --------------------------------------------------------------------------- #
# 颜色与名称
# --------------------------------------------------------------------------- #
CLASS_COLORS: dict[int, tuple[int, int, int]] = {
    0: (0, 0, 0),        # 背景：纯黑
    1: (0, 255, 0),      # 标签1：绿
    2: (255, 0, 0),      # 标签2：红
}
"""三个类别对应的 RGB 颜色（写进 mask PNG）。"""

CLASS_NAMES: dict[int, str] = {
    0: "背景",
    1: "零星肿瘤",
    2: "大块肿瘤",
}


# --------------------------------------------------------------------------- #
# 定类配置
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class CcConfig:
    """按连通域面积与数量定类的阈值。

    面积单位是 mask 像素数。本地 mask 为 1024×1024，故 1% ≈ 10,486 px。
    """

    min_fg_px: int = 128
    """前景像素少于此值 → 标签 0。默认 128 ≈ 11×11 区域，视为标注噪声。"""

    big_area_px: int = 50_000
    """最大连通域面积达此值 → 标签 2。默认 ≈ 4.8% patch（1 mm² 视野里约 0.05 mm²）。"""

    mid_area_px: int = 20_000
    """「中等」面积下界；再配合成团条件可升为标签 2。默认 ≈ 1.9% patch。"""

    mass_max_cc: int = 2
    """成团条件：有效连通域个数 ≤ 此值。默认 2（1 个主灶 + 至多 1 个碎片）。"""

    mass_min_dominance: float = 0.8
    """成团条件：最大连通域占前景像素的比例 ≥ 此值。默认 0.8。"""

    cc_min_area: int = 64
    """统计连通域个数时忽略面积小于此值的碎片（去噪）。默认 64 ≈ 8×8。"""

    binary_thresh: int = 127
    """二值化阈值：像素值 > 此值视为前景。"""

    def label_of(self, n_fg: int, n_cc: int, max_cc_px: int) -> int:
        """按连通域统计量返回类别 ``{0,1,2}``。

        Parameters
        ----------
        n_fg:
            前景像素总数。
        n_cc:
            面积 ≥ ``cc_min_area`` 的连通域个数。
        max_cc_px:
            最大连通域面积（不设下限，极小灶也能反映出来）。
        """
        if n_fg < self.min_fg_px:
            return 0
        if max_cc_px >= self.big_area_px:
            return 2
        dominance: float = max_cc_px / n_fg if n_fg > 0 else 0.0
        if (max_cc_px >= self.mid_area_px
                and n_cc <= self.mass_max_cc
                and dominance >= self.mass_min_dominance):
            return 2
        return 1


# --------------------------------------------------------------------------- #
# 连通域（纯 numpy：行程编码 + 并查集，4 邻接，不依赖 scipy / skimage）
# --------------------------------------------------------------------------- #
def component_areas(fg: np.ndarray) -> np.ndarray:
    """返回所有连通域的面积（降序），**4 邻接**（对角线不算连通）。

    实现：先按行做行程编码（一行连续的前景像素算一个 run），再用并查集把相邻行
    里区间相交的 run 合并。只对 run 建索引、不逐像素循环，1024×1024 约 10~20 ms。

    选 4 邻接而非 8 邻接：对角线单点相连的散点会被拆成独立小块，
    「散成一片点」就不会被误算成「一块实性灶」。

    Parameters
    ----------
    fg:
        ``[H, W]`` bool，True = 前景。

    Returns
    -------
    np.ndarray
        各连通域面积，降序；全空时返回空数组。
    """
    h, w = fg.shape
    if not fg.any():
        return np.empty(0, dtype=np.int64)

    # 每行两端补假背景后做差分：上升沿 = run 起点，下降沿 = run 终点（半开区间）
    padded: np.ndarray = np.zeros((h, w + 2), dtype=np.int8)
    padded[:, 1:-1] = fg.astype(np.int8)
    diff: np.ndarray = np.diff(padded, axis=1)
    start_rows, start_cols = np.nonzero(diff == 1)      # 已按 (row, col) 排序
    end_rows, end_cols = np.nonzero(diff == -1)
    n_runs: int = int(start_rows.size)

    parent: list[int] = list(range(n_runs))

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]               # 路径压缩（隔代）
            x = parent[x]
        return x

    def union(a: int, b: int) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra

    # 每行 run 的下标区间 [row_start[y], row_start[y+1])
    row_start: np.ndarray = np.searchsorted(start_rows, np.arange(h + 1))

    for y in range(1, h):
        i: int = int(row_start[y - 1])
        i_end: int = int(row_start[y])
        j: int = i_end
        j_end: int = int(row_start[y + 1])
        # 两个按起点有序的区间表求交；半开区间相交 = 上下相邻（4 邻接）
        while i < i_end and j < j_end:
            ps: int = int(start_cols[i])
            pe: int = int(end_cols[i])
            cs: int = int(start_cols[j])
            ce: int = int(end_cols[j])
            if ps < ce and cs < pe:
                union(i, j)
            if pe <= ce:
                i += 1
            else:
                j += 1

    lengths: np.ndarray = (end_cols - start_cols).astype(np.int64)
    area_of_root: dict[int, int] = {}
    for rid in range(n_runs):
        root: int = find(rid)
        area_of_root[root] = area_of_root.get(root, 0) + int(lengths[rid])

    areas: np.ndarray = np.fromiter(area_of_root.values(), dtype=np.int64,
                                    count=len(area_of_root))
    areas.sort()
    return areas[::-1].copy()


def label_mask(fg: np.ndarray, cfg: CcConfig | None = None) -> tuple[int, int, int, int]:
    """对一张 bool 前景图定类，返回 ``(label, n_fg, n_cc, max_cc_px)``。"""
    conf: CcConfig = cfg or CcConfig()
    n_fg: int = int(fg.sum())
    if n_fg == 0:
        return 0, 0, 0, 0
    areas: np.ndarray = component_areas(fg)
    max_cc: int = int(areas[0])
    n_cc: int = int((areas >= conf.cc_min_area).sum())
    return conf.label_of(n_fg, n_cc, max_cc), n_fg, n_cc, max_cc


__all__ = [
    "CLASS_COLORS",
    "CLASS_NAMES",
    "CcConfig",
    "component_areas",
    "label_mask",
]
