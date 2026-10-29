r"""提交包自检：在花钱上传之前，把官方会判 0 分的错误全查出来。

为什么需要
----------
官方明确写了：

- 缺文件、文件名对不上、尺寸不一致 → **该 slide 直接记 0 分**，系统不做容错；
- 多交文件 → 记为异常提交并告警；
- 只接受单通道 8-bit PNG，像素值只有 0 / 255；
- 目录层级必须与 ``test_patches/`` 严格一一对应，**去掉中间的 ``1.0/``**；
- zip 里不能再套一层文件夹。

而且**每队只有一次正式提交机会**，传错了没有第二次。所以上传前跑一遍这个脚本。

检查项
------
1. slide 集合：提交的 slide 与官方 ``test_patches`` 是否完全一致（多/少都报）；
2. 文件名：每个 stem 必须一一对应，只允许 ``.jpeg → .png`` 的扩展名变化；
3. 数量：每个 slide 的 png 数必须等于官方 patch 数；
4. 图像：单通道、8-bit、尺寸与原图**完全一致**、像素值只含 0/255；
5. 目录：提交目录下不允许出现子目录以外的杂项文件；
6. 分布：统计前景占比，全部为空或全部为满这类明显异常的模型行为会告警；
7. zip（可选）：条目必须是 ``{slide}/{name}.png``，不允许 ``submission/`` 这类外层目录。

用法
----
.. code-block:: bash

    python scripts/check_submission.py --test-root D:\data\toxpath_test \
        --submission submission

    # 连 zip 一起查
    python scripts/check_submission.py --test-root D:\data\toxpath_test \
        --submission submission --zip submission.zip

退出码：0 = 可以交；1 = 有阻塞性问题（必须修）。
"""

from __future__ import annotations

import argparse
import sys
import zipfile
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

IMAGE_SUFFIXES: tuple[str, ...] = (".jpeg", ".jpg", ".png", ".tif", ".tiff")

MAX_PRINTED_ERRORS: int = 20
"""同类错误最多打印这么多条，避免刷屏。"""


# --------------------------------------------------------------------------- #
# 报告器
# --------------------------------------------------------------------------- #
class Report:
    """收集检查结果并按级别打印（ASCII 标记，避免 GBK 控制台炸掉）。"""

    def __init__(self) -> None:
        self.rows: list[tuple[str, str, str, str]] = []

    def add(self, item: str, level: str, detail: str = "", fix: str = "") -> None:
        self.rows.append((item, level, detail, fix))

    @property
    def n_fail(self) -> int:
        return sum(1 for r in self.rows if r[1] == "FAIL")

    @property
    def n_warn(self) -> int:
        return sum(1 for r in self.rows if r[1] == "WARN")

    def render(self) -> None:
        print("=" * 78)
        print(f"{'项目':<22}{'级别':<7}说明")
        print("-" * 78)
        for item, level, detail, fix in self.rows:
            print(f"{item:<22}{level:<7}{detail}")
            if fix:
                print(f"{'':<22}{'':<7}-> {fix}")
        print("=" * 78)


# --------------------------------------------------------------------------- #
# 扫描官方 patch（只读尺寸，不全解码）
# --------------------------------------------------------------------------- #
def scan_official(test_root: Path) -> dict[str, dict[str, tuple[int, int]]]:
    """返回 ``{slide: {stem: (宽, 高)}}``。

    只需要尺寸信息，PIL 打开时不触发像素解码，5000+ 张也很快。
    """
    root: Path = test_root / "test_patches"
    if not root.is_dir():
        root = test_root
    if not root.is_dir():
        raise FileNotFoundError(f"找不到 test_patches：{test_root}")

    out: dict[str, dict[str, tuple[int, int]]] = {}
    for slide_dir in sorted(p for p in root.iterdir() if p.is_dir()):
        files: list[Path] = []
        for suffix in IMAGE_SUFFIXES:
            files.extend(slide_dir.rglob(f"*{suffix}"))
        if not files:
            continue
        table: dict[str, tuple[int, int]] = {}
        for f in sorted(set(files)):
            with Image.open(f) as im:
                table[f.stem] = im.size
        out[slide_dir.name] = table
    return out


# --------------------------------------------------------------------------- #
# 检查
# --------------------------------------------------------------------------- #
def check_structure(
    official: dict[str, dict[str, tuple[int, int]]],
    sub_dir: Path,
    rep: Report,
) -> dict[str, list[Path]]:
    """检查 slide 目录、文件名、数量、图像规范。返回 ``{slide: [png]}``。"""
    if not sub_dir.is_dir():
        rep.add("提交目录", "FAIL", f"不存在：{sub_dir}")
        return {}

    # ---- 1) 顶层：不允许杂项文件，不允许嵌套 -------------------------------------------------
    entries: list[Path] = sorted(sub_dir.iterdir())
    strays: list[Path] = [p for p in entries if not p.is_dir()]
    if strays:
        rep.add("顶层杂项文件", "FAIL", f"{len(strays)} 个",
                f"例如 {[p.name for p in strays[:3]]}；提交目录下只应有 {{slide}}/ 子目录")
    else:
        rep.add("顶层结构", "OK", f"{sum(1 for p in entries if p.is_dir())} 个 slide 目录")

    # ---- 2) slide 集合对齐 ---------------------------------------------------------------
    sub_slides: set[str] = {p.name for p in entries if p.is_dir()}
    off_slides: set[str] = set(official)
    missing_slides: list[str] = sorted(off_slides - sub_slides)
    extra_slides: list[str] = sorted(sub_slides - off_slides)
    if missing_slides:
        rep.add("缺失 slide", "FAIL", f"{len(missing_slides)} 个",
                f"{missing_slides[:5]}；缺一个 slide 就等于缺它的全部 patch")
    if extra_slides:
        rep.add("多余 slide", "FAIL", f"{len(extra_slides)} 个",
                f"{extra_slides[:5]}；官方未提供的 slide 会被判异常提交")
    if not missing_slides and not extra_slides:
        rep.add("slide 对齐", "OK", f"{len(off_slides)} 个，与官方一致")

    # ---- 3) 逐 slide 检查文件与图像 -------------------------------------------------------
    result: dict[str, list[Path]] = {}
    total_png: int = 0
    total_missing: int = 0
    total_extra: int = 0
    total_dim: int = 0
    total_fmt: int = 0
    total_val: int = 0
    printed: dict[str, int] = defaultdict(int)
    fg_ratios: list[float] = []
    n_all_zero: int = 0
    n_all_fg: int = 0

    for slide, table in sorted(official.items()):
        slide_dir: Path = sub_dir / slide
        if not slide_dir.is_dir():
            continue
        pngs: dict[str, Path] = {p.stem: p for p in sorted(slide_dir.glob("*.png"))}
        others: list[Path] = [p for p in slide_dir.iterdir() if p.is_file() and p.suffix != ".png"]
        if others:
            rep.add(f"{slide} 非 png", "FAIL", f"{len(others)} 个",
                    f"例如 {[p.name for p in others[:3]]}")
        result[slide] = list(pngs.values())
        total_png += len(pngs)

        miss: list[str] = sorted(set(table) - set(pngs))
        extra: list[str] = sorted(set(pngs) - set(table))
        total_missing += len(miss)
        total_extra += len(extra)
        if miss and printed["miss"] < MAX_PRINTED_ERRORS:
            rep.add(f"{slide} 缺掩膜", "FAIL", f"{len(miss)} 张",
                    f"例如 {miss[:5]}")
            printed["miss"] += 1
        if extra and printed["extra"] < MAX_PRINTED_ERRORS:
            rep.add(f"{slide} 多掩膜", "FAIL", f"{len(extra)} 张",
                    f"例如 {extra[:5]}；不要给官方没提供的 patch 交掩膜")
            printed["extra"] += 1

        # 逐张查图像规范
        for stem, png in pngs.items():
            if stem not in table:
                continue
            exp_w, exp_h = table[stem]
            try:
                with Image.open(png) as im:
                    mode: str = im.mode
                    w, h = im.size
                    arr: np.ndarray = np.asarray(im)
            except Exception as exc:                       # noqa: BLE001
                rep.add(f"{slide}/{stem}", "FAIL", f"打不开：{exc}")
                total_fmt += 1
                continue

            if (w, h) != (exp_w, exp_h):
                total_dim += 1
                if printed["dim"] < MAX_PRINTED_ERRORS:
                    rep.add(f"{slide}/{stem}", "FAIL", f"尺寸 {w}x{h}，原图 {exp_w}x{exp_h}",
                            "系统不会自动 resize，该 patch 会评测失败")
                    printed["dim"] += 1
            if mode not in ("L", "P", "1"):
                total_fmt += 1
                if printed["fmt"] < MAX_PRINTED_ERRORS:
                    rep.add(f"{slide}/{stem}", "WARN", f"模式 {mode}",
                            "建议单通道 L；三通道会被转灰度再按 0/255 判，有歧义")
                    printed["fmt"] += 1
            if arr.dtype != np.uint8:
                total_val += 1
                if printed["val"] < MAX_PRINTED_ERRORS:
                    rep.add(f"{slide}/{stem}", "FAIL", f"dtype {arr.dtype}", "必须是 8-bit")
                    printed["val"] += 1
                continue

            uniq: np.ndarray = np.unique(arr)
            bad: np.ndarray = uniq[(uniq != 0) & (uniq != 255)]
            if bad.size:
                total_val += 1
                if printed["val"] < MAX_PRINTED_ERRORS:
                    rep.add(f"{slide}/{stem}", "FAIL",
                            f"出现非 0/255 像素值 {bad[:5].tolist()}",
                            "概率图/灰度图不被接受，必须二值化")
                    printed["val"] += 1

            ratio: float = float((arr == 255).mean())
            fg_ratios.append(ratio)
            if ratio == 0.0:
                n_all_zero += 1
            elif ratio == 1.0:
                n_all_fg += 1

    # ---- 4) 汇总 -----------------------------------------------------------------------
    if total_missing == 0:
        rep.add("掩膜完整性", "OK", f"{total_png} 张，无缺失")
    if total_extra:
        rep.add("多余掩膜合计", "FAIL", f"{total_extra} 张", "官方会记为异常提交")
    if total_dim:
        rep.add("尺寸错误合计", "FAIL", f"{total_dim} 张",
                "这些 patch 会被判评测失败，可能拖累整个 slide")
    if total_val:
        rep.add("像素值错误合计", "FAIL", f"{total_val} 张", "必须只有 0 和 255")

    # ---- 5) 分布是否合理（能抓出模型没加载对/全黑全白的情况） ---------------------------------
    if fg_ratios:
        arr_r: np.ndarray = np.asarray(fg_ratios, dtype=np.float64)
        rep.add("前景占比", "OK",
                f"均值 {arr_r.mean():.4f}  中位数 {np.median(arr_r):.4f}  "
                f"最大 {arr_r.max():.4f}")
        rep.add("全空掩膜", "OK" if 0.4 <= n_all_zero / len(fg_ratios) <= 0.95 else "WARN",
                f"{n_all_zero} 张（{n_all_zero/len(fg_ratios)*100:.1f}%）",
                "本地训练集约 83.5% 的 patch 全空；比例过低说明假阳偏多，过高说明漏检")
        if n_all_fg > 0:
            rep.add("全满掩膜", "WARN", f"{n_all_fg} 张",
                    "整张判成肿瘤基本不可能正确，检查归一化/阈值")
    else:
        rep.add("前景占比", "FAIL", "没有可用掩膜")

    return result


def check_zip(zip_path: Path, official: dict[str, dict[str, tuple[int, int]]], rep: Report) -> None:
    """检查 zip 内条目结构，官方要求根目录直接是 ``{slide}/{name}.png``。"""
    if not zip_path.is_file():
        rep.add("zip", "FAIL", f"不存在：{zip_path}")
        return
    try:
        with zipfile.ZipFile(zip_path) as zf:
            names: list[str] = [n for n in zf.namelist() if not n.endswith("/")]
            bad_arch: list[str] = [n for n in names
                                   if "/" not in n or n.count("/") != 1]
            if bad_arch:
                rep.add("zip 层级", "FAIL", f"{len(bad_arch)} 个条目层级不对",
                        f"例如 {bad_arch[:3]}；应为 test_001/0_0.png，"
                        f"不能是 submission/test_001/0_0.png")
            else:
                rep.add("zip 层级", "OK", f"{len(names)} 个条目，形如 test_001/0_0.png")

            z_slides: set[str] = {n.split("/")[0] for n in names}
            off_slides: set[str] = set(official)
            if z_slides != off_slides:
                rep.add("zip slide 对齐", "FAIL",
                        f"缺 {sorted(off_slides - z_slides)[:5]}，"
                        f"多 {sorted(z_slides - off_slides)[:5]}")
            else:
                rep.add("zip slide 对齐", "OK", f"{len(z_slides)} 个")

            # 条目数是否等于官方 patch 总数
            expect: int = sum(len(v) for v in official.values())
            if len(names) != expect:
                rep.add("zip 条目数", "FAIL", f"{len(names)}，官方 patch {expect}")
            else:
                rep.add("zip 条目数", "OK", f"{len(names)} 张，与官方一致")

            size_mb: float = zip_path.stat().st_size / 1024 / 1024
            rep.add("zip 体积", "OK", f"{size_mb:.2f} MB",
                    "二值 PNG 压缩率很高；若超过几百 MB 说明掩膜不是二值的")
    except zipfile.BadZipFile as exc:
        rep.add("zip", "FAIL", f"不是合法 zip：{exc}")


# --------------------------------------------------------------------------- #
# main
# --------------------------------------------------------------------------- #
def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="检查 ToxPath 提交包是否符合官方规范",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--test-root", type=Path, required=True,
                   help="含 test_patches/ 的目录")
    p.add_argument("--submission", type=Path, required=True,
                   help="提交目录（含 {slide}/*.png）")
    p.add_argument("--zip", type=Path, default=None, help="顺便检查 zip 结构")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    official: dict[str, dict[str, tuple[int, int]]] = scan_official(args.test_root)
    n_patch: int = sum(len(v) for v in official.values())
    print(f"[scan] 官方 {len(official)} 个 slide，共 {n_patch} 张 patch")
    print(f"[scan] 提交目录 {args.submission}")

    rep = Report()
    check_structure(official, args.submission, rep)
    if args.zip is not None:
        check_zip(args.zip, official, rep)
    rep.render()

    if rep.n_fail:
        print(f"[result] FAIL {rep.n_fail} 项，WARN {rep.n_warn} 项 —— 现在不能交，先修")
        return 1
    print(f"[result] 全部通过（WARN {rep.n_warn} 项），可以上传")
    return 0


if __name__ == "__main__":
    sys.exit(main())
