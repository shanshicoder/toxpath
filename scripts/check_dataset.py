"""训练前的数据契约检查 + 分折预演。

为什么需要它
------------
``make_3class_masks.py`` 只认「能读到的文件」，它不会替你检查下面这些事：

- patch 是 ``.jpeg`` 还是 ``.png``（数据集只 glob ``*.jpeg``，写成 png 会得到
  「没有可用 patch」，而不是明确的扩展名报错）；
- patch 和 mask 的文件名集合是否**完全一致**（少一张，训练启动时才抛
  ``ValueError``，7 万张里翻那 3 个名字很费劲）；
- mask 与 patch **尺寸是否一致**（不一致会在 collate 阶段炸，或者更糟——
  被 resize 悄悄改掉）；
- 三分类 mask 的颜色是否只有黑/绿/红（用了别的调色板，训练会把绿当背景）；
- CSV 行数是否等于 patch 数、有没有重复键；
- 73 个切片分 5 折后，每折到底多少 patch、全空基线是多少（决定这一折的
  ``dice_fg`` 能不能横比）。

这些全部在本地几十秒内能查完，比在 4090 上跑到第 3 分钟才报错划算。

用法
----
.. code-block:: bash

    python scripts/check_dataset.py                              # 查默认数据根
    python scripts/check_dataset.py --data-root "D:/data/toxpath_73"
    python scripts/check_dataset.py --folds 5 --samples 8        # 每切片抽 8 组核对尺寸
    python scripts/check_dataset.py --table                      # 哪怕 73 个切片也打全表

退出码：``0`` 全部通过；``1`` 有问题（每条都会打印「现象 + 怎么修」）；``2``
数据根都解析不到。
"""

from __future__ import annotations

import argparse
import csv
import sys
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import splits  # noqa: E402
from labels import CLASS_COLORS  # noqa: E402
from paths import describe_paths, resolve_data_root  # noqa: E402

PATCH_DIR: str = "train_patches"
"""1024×1024 H&E patch，训练图像来源。"""

BINARY_DIR: str = "train_masks"
"""上游产出的二值 mask，``make_3class_masks.py`` 的输入。"""

CLASS_DIR: str = "train_masks_classes"
"""三分类彩色 mask，训练真值来源。"""

MAG_DIR: str = "1.0"
"""每切片下的倍率目录名，数据集写死为这个值。"""

PATCH_GLOB: str = "*.jpeg"
"""数据集只 glob 这个模式；``.jpg`` / ``.png`` 都会被漏掉。"""

CLASS_COLOR_SET: frozenset[tuple[int, int, int]] = frozenset(CLASS_COLORS.values())
"""允许出现在三分类 mask 里的颜色（黑 / 绿 / 红）。"""


# --------------------------------------------------------------------------- #
# 数据结构
# --------------------------------------------------------------------------- #
@dataclass
class SlideReport:
    """单个切片的检查结果。"""

    slide: str
    n_patch: int = 0
    n_binary: int = 0
    n_class: int = 0
    #: 文件名（不含扩展名）集合差异：[只在 patch 里有, 只在 mask 里有]
    missing_mask: int = 0
    missing_class: int = 0
    problems: list[str] = field(default_factory=list)


# --------------------------------------------------------------------------- #
# 工具
# --------------------------------------------------------------------------- #
def list_slides(root: Path) -> list[str]:
    """列出 ``root`` 下的切片目录（跳过隐藏目录）。"""
    if not root.is_dir():
        return []
    return sorted(p.name for p in root.iterdir() if p.is_dir() and not p.name.startswith("."))


def stems(directory: Path, pattern: str) -> set[str]:
    """返回目录下匹配文件的「不含扩展名」文件名集合。"""
    if not directory.is_dir():
        return set()
    return {p.stem for p in directory.glob(pattern)}


def iter_files(directory: Path, pattern: str) -> list[Path]:
    """返回目录下匹配的文件列表（已排序）。"""
    if not directory.is_dir():
        return []
    return sorted(directory.glob(pattern))


def sample_check(
    patch_dir: Path,
    class_dir: Path,
    sample_stems: list[str],
    problems: list[str],
) -> int:
    """抽样核对 patch 与三分类 mask 的尺寸、通道和取值。

    Args:
        patch_dir: ``train_patches/<slide>/1.0``。
        class_dir: ``train_masks_classes/<slide>/1.0``。
        sample_stems: 要抽的 patch 文件名（无扩展名）。
        problems: 问题收集列表，直接追加。

    Returns:
        实际核对成功的组数。
    """
    ok: int = 0
    for stem in sample_stems:
        p_path: Path = patch_dir / f"{stem}.jpeg"
        m_path: Path = class_dir / f"{stem}.png"
        if not p_path.is_file() or not m_path.is_file():
            continue
        with Image.open(p_path) as im_p, Image.open(m_path) as im_m:
            if im_p.size != im_m.size:
                problems.append(
                    f"{stem} 尺寸不一致：patch {im_p.size} vs mask {im_m.size}"
                    f"（上游切图与 mask 栅格化必须用同一套坐标）"
                )
                continue
            if im_m.mode != "RGB":
                problems.append(
                    f"{stem} 三分类 mask 不是 RGB（是 {im_m.mode}），"
                    f"训练按 3 通道读，颜色会错位"
                )
                continue
            colors = {c for _, c in im_m.getcolors(maxcolors=1 << 20) or []}
        bad: set[tuple[int, int, int]] = colors - CLASS_COLOR_SET
        if bad:
            problems.append(
                f"{stem} 三分类 mask 含非约定颜色 {sorted(bad)[:3]}，"
                f"只允许 {sorted(CLASS_COLOR_SET)}（跑 make_3class_masks.py 重刷）"
            )
            continue
        ok += 1
    return ok


# --------------------------------------------------------------------------- #
# 检查
# --------------------------------------------------------------------------- #
def check_slides(
    data_root: Path,
    *,
    samples: int,
    table: bool,
) -> tuple[list[SlideReport], list[str], int, int]:
    """逐切片核对 patch / 二值 mask / 三分类 mask。

    Returns:
        (每切片报告, 全局问题, 抽样成功组数, 抽样总组数)
    """
    problems: list[str] = []
    patch_slides: list[str] = list_slides(data_root / PATCH_DIR)
    binary_slides: list[str] = list_slides(data_root / BINARY_DIR)
    class_slides: list[str] = list_slides(data_root / CLASS_DIR)

    if not patch_slides:
        problems.append(
            f"{PATCH_DIR}/ 下没有切片目录（数据根：{data_root}）——"
            f"确认 --data-root 指向的是包含 {PATCH_DIR}/ 的那一层"
        )
        return [], problems, 0, 0

    only_patch: set[str] = set(patch_slides) - set(binary_slides)
    if only_patch:
        problems.append(
            f"这些切片有 patch 但没有二值 mask：{sorted(only_patch)[:5]}"
            f"（共 {len(only_patch)} 个）——上游切图只做了一半"
        )
    only_bin: set[str] = set(binary_slides) - set(patch_slides)
    if only_bin:
        problems.append(
            f"这些切片有 mask 但没有 patch：{sorted(only_bin)[:5]}（共 {len(only_bin)} 个）"
        )

    reports: list[SlideReport] = []
    sample_ok: int = 0
    sample_total: int = 0

    for slide in patch_slides:
        rep: SlideReport = SlideReport(slide=slide)
        p_dir: Path = data_root / PATCH_DIR / slide / MAG_DIR
        # 扩展名普查：数据集只认 .jpeg
        all_files: list[Path] = [p for p in (data_root / PATCH_DIR / slide / MAG_DIR).glob("*") if p.is_file()] \
            if (data_root / PATCH_DIR / slide / MAG_DIR).is_dir() else []
        exts: Counter[str] = Counter(p.suffix.lower() for p in all_files)
        p_stems: set[str] = stems(p_dir, PATCH_GLOB)
        rep.n_patch = len(p_stems)

        if not p_dir.is_dir():
            rep.problems.append(f"缺少 {MAG_DIR}/ 目录：{p_dir}")
        elif rep.n_patch == 0:
            rep.problems.append(
                f"{p_dir} 下没有 {PATCH_GLOB}（现有扩展名 {dict(exts)}）——"
                f"数据集只认 .jpeg，.jpg/.png 会被当成 0 张"
            )
        else:
            wrong: dict[str, int] = {e: n for e, n in exts.items() if e != ".jpeg"}
            if wrong:
                problems.append(
                    f"[{slide}] 混入非 .jpeg 文件 {wrong}，这些会被静默忽略"
                )

        # 二值 mask
        b_dir: Path = data_root / BINARY_DIR / slide / MAG_DIR
        b_stems: set[str] = stems(b_dir, "*.png")
        rep.n_binary = len(b_stems)

        # 三分类 mask
        c_dir: Path = data_root / CLASS_DIR / slide / MAG_DIR
        c_stems: set[str] = stems(c_dir, "*.png")
        rep.n_class = len(c_stems)

        if p_stems:
            rep.missing_mask = len(p_stems - b_stems)
            rep.missing_class = len(p_stems - c_stems)
            if rep.missing_mask:
                rep.problems.append(
                    f"二值 mask 缺 {rep.missing_mask} 张（例：{sorted(p_stems - b_stems)[:3]}）"
                    f"——上游切图/栅格化没做全，make_3class_masks.py 帮不了你"
                )
            if rep.missing_class:
                rep.problems.append(
                    f"三分类 mask 缺 {rep.missing_class} 张（例：{sorted(p_stems - c_stems)[:3]}）"
                    f"——跑 python scripts/make_3class_masks.py"
                )
            extra_class: int = len(c_stems - p_stems)
            if extra_class:
                rep.problems.append(
                    f"三分类 mask 多出 {extra_class} 张（旧数据残留？"
                    f"例：{sorted(c_stems - p_stems)[:3]}）——建议清空 {CLASS_DIR}/ 重跑"
                )

        # 抽样尺寸/颜色
        if samples > 0 and c_stems and p_stems:
            step: int = max(1, len(p_stems) // samples)
            pick: list[str] = sorted(p_stems)[::step][:samples]
            sample_total += len(pick)
            sample_ok += sample_check(p_dir, c_dir, pick, rep.problems)

        if rep.problems:
            problems.extend(f"[{slide}] {msg}" for msg in rep.problems)
        reports.append(rep)

    # 全表 or 只打有问题的
    print(f"\n[check] 逐切片（{'全表' if table else '仅异常'}）")
    header: str = f"  {'切片':<20}{'patch':>8}{'二值':>8}{'三分类':>8}  状态"
    print(header)
    shown: int = 0
    for rep in reports:
        if not table and not rep.problems:
            continue
        state: str = "OK" if not rep.problems else f"{len(rep.problems)} 处问题"
        print(f"  {rep.slide:<20}{rep.n_patch:>8}{rep.n_binary:>8}{rep.n_class:>8}  {state}")
        shown += 1
    if shown == 0:
        print("  （全部正常）")

    return reports, problems, sample_ok, sample_total


def check_index(data_root: Path, slide_names: list[str]) -> tuple[dict[str, int], dict[str, int], list[str]]:
    """核对 ``_patch_labels_3class.csv`` 的行数、重复键与切片覆盖。

    Returns:
        (每切片行数, 每切片标签 0 的个数, 问题列表)
    """
    problems: list[str] = []
    path: Path = data_root / splits.INDEX_NAME
    rows_per_slide: dict[str, int] = {}
    zeros_per_slide: dict[str, int] = {}

    if not path.is_file():
        problems.append(
            f"缺少标签索引 {path}——先跑 "
            f"python scripts/make_3class_masks.py --data-root \"{data_root}\""
        )
        return rows_per_slide, zeros_per_slide, problems

    seen: set[tuple[str, str]] = set()
    dup: int = 0
    bad_label: int = 0
    labels: Counter[int] = Counter()
    with path.open(newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        need: set[str] = {"slide", "name", "label", "fg_ratio"}
        if not need.issubset(set(reader.fieldnames or [])):
            problems.append(
                f"{path.name} 列名不对（现有 {reader.fieldnames}），"
                f"至少要有 {sorted(need)}"
            )
            return rows_per_slide, zeros_per_slide, problems
        for row in reader:
            key: tuple[str, str] = (row["slide"], row["name"])
            if key in seen:
                dup += 1
            seen.add(key)
            try:
                label: int = int(row["label"])
            except (TypeError, ValueError):
                bad_label += 1
                continue
            if label not in (0, 1, 2):
                bad_label += 1
                continue
            labels[label] += 1
            rows_per_slide[row["slide"]] = rows_per_slide.get(row["slide"], 0) + 1
            if label == 0:
                zeros_per_slide[row["slide"]] = zeros_per_slide.get(row["slide"], 0) + 1

    if dup:
        problems.append(f"CSV 有 {dup} 个重复 (slide, name) 键——同一张 patch 被写了两次")
    if bad_label:
        problems.append(f"CSV 有 {bad_label} 行 label 不是 0/1/2，无法当类别用")

    not_in_csv: set[str] = set(slide_names) - set(rows_per_slide)
    if not_in_csv:
        problems.append(
            f"这些切片在 CSV 里一条记录都没有：{sorted(not_in_csv)[:5]}"
            f"（共 {len(not_in_csv)} 个）——训练会直接 KeyError"
        )

    total: int = sum(labels.values())
    if total:
        pct: list[float] = [100.0 * labels[c] / total for c in (0, 1, 2)]
        print(f"\n[check] 标签索引 {path.name}：{total} 行，"
              f"类别 0/1/2 = {labels[0]}/{labels[1]}/{labels[2]} "
              f"({pct[0]:.1f}% / {pct[1]:.1f}% / {pct[2]:.1f}%)")
    else:
        print(f"\n[check] 标签索引 {path.name} 为空")
    return rows_per_slide, zeros_per_slide, problems


def preview_folds(data_root: Path, num_folds: int) -> list[str]:
    """用训练端同一套 ``splits.py`` 预演分折，打印每折规模与全空基线。

    Returns:
        问题列表（分折本身非法时才有内容）。
    """
    problems: list[str] = []
    slide_names: list[str] = list_slides(data_root / PATCH_DIR)
    try:
        stats = splits.load_slide_stats(data_root, slide_names)
    except FileNotFoundError as exc:
        return [str(exc)]
    if not stats:
        return ["CSV 里没有任何切片统计，无法预演分折"]

    missing: set[str] = set(slide_names) - set(stats)
    if missing:
        problems.append(
            f"CSV 没有覆盖这些切片：{sorted(missing)[:5]}（共 {len(missing)} 个）"
        )

    folds: dict[int, list[str]] = splits.build_slide_folds(stats, num_folds, verbose=False)
    try:
        splits.validate_folds(folds, sorted(stats))
    except ValueError as exc:
        problems.append(f"分折非法：{exc}")

    print(f"\n[check] 分折预演（{len(stats)} 切片 → {len(folds)} 折，按 mean_fg 蛇形分层）")
    print(f"  {'折':<6}{'切片':>6}{'patch':>9}{'肿瘤 patch':>11}{'全空基线':>10}  验证切片")
    for k in sorted(folds):
        names: list[str] = folds[k]
        n_patch: int = sum(stats[s].n_patch for s in names)
        n_tumor: int = sum(stats[s].n_cls1 + stats[s].n_cls2 for s in names)
        baseline: float = (n_patch - n_tumor) / n_patch if n_patch else 0.0
        head: str = ", ".join(names[:3]) + ("…" if len(names) > 3 else "")
        print(f"  fold{k:<2}{len(names):>6}{n_patch:>9}{n_tumor:>11}{baseline:>10.3f}  {head}")

    print("  ↑ 全空基线 = 该折验证集里真值全空的 patch 占比，也就是「全猜背景」的 dice_fg。"
          "\n    各折基线差得多时，dice_fg 不能横比，只能对着各自基线看。")
    return problems


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p: argparse.ArgumentParser = argparse.ArgumentParser(
        description="训练前检查数据契约（数量/命名/尺寸/颜色/CSV）并预演分折",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--data-root", type=Path, default=None,
                   help="数据根（其下有 train_patches/）；不填按 paths.py 解析")
    p.add_argument("--folds", type=int, default=5, help="分折预演用的折数")
    p.add_argument("--samples", type=int, default=5,
                   help="每个切片抽几张核对尺寸/颜色；0 表示跳过")
    p.add_argument("--table", action="store_true",
                   help="打印全部切片（默认只打有问题的，73 个切片时更清爽）")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args: argparse.Namespace = parse_args(argv)
    try:
        data_root: Path = resolve_data_root(args.data_root)
    except FileNotFoundError as exc:
        print(f"[error] {exc}", file=sys.stderr)
        return 2

    print("[check] 路径：")
    print(describe_paths(data_root))
    print(f"[check] data_root = {data_root}")

    reports, problems, sample_ok, sample_total = check_slides(
        data_root, samples=args.samples, table=args.table
    )
    if not reports:
        print("\n[check] 结论：无法检查，先解决上面的路径问题。")
        return 1

    rows, zeros, idx_problems = check_index(data_root, [r.slide for r in reports])
    problems.extend(idx_problems)

    # 数量对账：patch / 二值 / 三分类 / CSV 行
    n_patch: int = sum(r.n_patch for r in reports)
    n_binary: int = sum(r.n_binary for r in reports)
    n_class: int = sum(r.n_class for r in reports)
    n_csv: int = sum(rows.values())
    print(f"\n[check] 合计：patch {n_patch} / 二值 mask {n_binary} / "
          f"三分类 mask {n_class} / CSV {n_csv} 行")
    if n_class and n_csv and n_class != n_csv:
        problems.append(
            f"三分类 mask 张数（{n_class}）与 CSV 行数（{n_csv}）不一致——"
            f"多半是旧数据残留或生成中断，清空 {CLASS_DIR}/ 后重跑 make_3class_masks.py"
        )
    if sample_total:
        print(f"[check] 抽样核对：{sample_ok}/{sample_total} 组尺寸与颜色合规")

    fold_problems: list[str] = preview_folds(data_root, args.folds)
    problems.extend(fold_problems)

    print()
    if problems:
        print(f"[check] 结论：发现 {len(problems)} 处问题，先修再开训")
        for i, msg in enumerate(problems, 1):
            print(f"  {i:>2}. {msg}")
        return 1

    print("[check] 结论：通过。数量、命名、尺寸、颜色、CSV、分折都一致，可以进 1.7 冒烟。")
    return 0


if __name__ == "__main__":
    sys.exit(main())
