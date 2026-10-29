"""按**官方口径**在本地带标签数据上打分：逐 patch Dice / IoU。

为什么单独写一个
----------------
``evaluate.py`` 的分层口径（整图 3 类 mIoU、组织区内 Dice、patch 分箱准确率）
是给自己做诊断用的，**和排行榜不是一个算法**。官方口径是：

- 逐 patch 算二值 Dice 和 IoU，单张得分 = ``(Dice + IoU) / 2``；
- **真值与预测都全空的 patch**，Dice 和 IoU 都记 **1.0**；
- 最终分是**所有 patch 的算术平均**（每张等权），slide 级只作参考；
- 真值 = GeoJSON 里 ``classification.name == "Lymphoma"`` 的轮廓栅格化。

最后一条本地已核对：``train_masks`` 与该栅格化的 IoU 为 0.994~1.000。

这个口径有个非常重要的后果
--------------------------
本地 4953 张 patch 里有 **4137 张（83.5%）真值全空**。它们只要预测出一个
前景像素，得分就从 1.0 掉到 0。也就是说：

- **全部预测为空**的综合分就有 **≈0.835**（本地实测）；
- 真正的区分度全在剩下 16.5% 有肿瘤的 patch 上；
- 假阳的代价远大于漏检的一部分，**后处理去小斑点几乎是免费收益**。

所以本脚本除了给分数，还给出 ``--sweep-min-area`` / ``--sweep-threshold``
两维扫描，用来在实际数据上选阈值，而不是凭感觉。

用法
----
.. code-block:: bash

    # 单折，官方口径打分
    python eval_binary.py --fold 1 --ckpt checkpoints3/fold1/best.pt

    # 扫描最小连通域面积，看哪个阈值分最高
    python eval_binary.py --fold 1 --ckpt checkpoints3/fold1/best.pt \
        --sweep-min-area 0,64,128,256,512

    # 5 折集成 + TTA
    python eval_binary.py --ckpt-dir checkpoints3 --tta --sweep-threshold 0.4,0.5,0.6
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image

PROJECT_ROOT: Path = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

from paths import describe_paths, resolve_data_root  # noqa: E402
from predict_test import (  # noqa: E402
    load_models,
    remove_small_components,
    resolve_ckpts,
    tumor_probability,
)

# --------------------------------------------------------------------------- #
# 指标（严格照官方定义）
# --------------------------------------------------------------------------- #
def dice_iou(pred: np.ndarray, gt: np.ndarray) -> tuple[float, float]:
    """二值 Dice / IoU，两边全空时按官方规定返回 ``(1.0, 1.0)``。

    Parameters
    ----------
    pred, gt:
        ``[H, W]`` bool。

    Returns
    -------
    tuple
        ``(dice, iou)``。
    """
    p_sum: int = int(pred.sum())
    g_sum: int = int(gt.sum())
    if p_sum == 0 and g_sum == 0:
        return 1.0, 1.0
    inter: int = int(np.logical_and(pred, gt).sum())
    union: int = int(np.logical_or(pred, gt).sum())
    dice: float = 2.0 * inter / (p_sum + g_sum) if (p_sum + g_sum) else 0.0
    iou: float = inter / union if union else 0.0
    return dice, iou


# --------------------------------------------------------------------------- #
# 配置组合（阈值 × 最小连通域）
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class Combo:
    """一组后处理参数。"""

    threshold: float
    min_area: int

    @property
    def label(self) -> str:
        return f"thr={self.threshold:.2f}/area>={self.min_area}"


@dataclass
class Acc:
    """一组参数的累计结果。"""

    scores: list[float] = field(default_factory=list)
    dices: list[float] = field(default_factory=list)
    ious: list[float] = field(default_factory=list)
    n_empty_gt: int = 0
    n_empty_fp: int = 0          # 真值全空但预测出前景 → 该 patch 得 0
    n_tumor: int = 0
    tumor_dice: list[float] = field(default_factory=list)
    by_slide: dict[str, list[float]] = field(default_factory=dict)

    def add(self, dice: float, iou: float, gt_empty: bool, slide: str) -> None:
        score: float = (dice + iou) / 2.0
        self.scores.append(score)
        self.dices.append(dice)
        self.ious.append(iou)
        self.by_slide.setdefault(slide, []).append(score)
        if gt_empty:
            self.n_empty_gt += 1
            if score < 0.5:
                self.n_empty_fp += 1
        else:
            self.n_tumor += 1
            self.tumor_dice.append(dice)

    def summary(self) -> dict[str, Any]:
        n: int = len(self.scores)
        if n == 0:
            return {}
        arr: np.ndarray = np.asarray(self.scores, dtype=np.float64)
        return {
            "n_patch": n,
            "score_mean": float(arr.mean()),
            "score_std": float(arr.std()),
            "dice_mean": float(np.mean(self.dices)),
            "iou_mean": float(np.mean(self.ious)),
            "n_empty_gt": self.n_empty_gt,
            "n_empty_fp": self.n_empty_fp,
            "empty_recall": (1.0 - self.n_empty_fp / self.n_empty_gt) if self.n_empty_gt else 1.0,
            "n_tumor": self.n_tumor,
            "tumor_dice_mean": float(np.mean(self.tumor_dice)) if self.tumor_dice else float("nan"),
            "per_slide": {k: float(np.mean(v)) for k, v in sorted(self.by_slide.items())},
        }


# --------------------------------------------------------------------------- #
# 数据定位
# --------------------------------------------------------------------------- #
def resolve_slides(
    data_root: Path,
    fold: int | None,
    slides_arg: str | None,
) -> list[str]:
    """决定要评哪些切片：``--slides`` 优先，其次 ``--fold``，否则全部。"""
    if slides_arg:
        return [s.strip() for s in slides_arg.split(",") if s.strip()]

    if fold is not None:
        from splits import build_slide_folds, discover_slides, load_folds, load_slide_stats

        all_slides: list[str] = discover_slides(data_root)
        folds_path: Path = PROJECT_ROOT / "checkpoints3" / "splits.json"
        folds: dict[int, list[str]] | None = load_folds(folds_path) if folds_path.is_file() else None
        if folds is None:
            stats = load_slide_stats(data_root, all_slides)
            folds = build_slide_folds(stats, 5)
        if fold not in folds:
            raise SystemExit(f"fold{fold} 不存在，可用：{sorted(folds)}")
        return list(folds[fold])

    from splits import discover_slides

    return discover_slides(data_root)


def collect_pairs(data_root: Path, slides: list[str]) -> list[tuple[str, Path, Path]]:
    """返回 ``[(slide, patch_jpeg, gt_png), ...]``，只收有 GT 的 patch。"""
    pairs: list[tuple[str, Path, Path]] = []
    for slide in slides:
        patch_dir: Path = data_root / "train_patches" / slide / "1.0"
        gt_dir: Path = data_root / "train_masks" / slide / "1.0"
        if not patch_dir.is_dir():
            print(f"[skip] {slide}: 没有 {patch_dir}")
            continue
        for jpg in sorted(patch_dir.glob("*.jpeg")):
            gt: Path = gt_dir / f"{jpg.stem}.png"
            if not gt.is_file():
                continue
            pairs.append((slide, jpg, gt))
    return pairs


# --------------------------------------------------------------------------- #
# 主流程
# --------------------------------------------------------------------------- #
def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="按官方口径（逐 patch Dice/IoU）在本地带标签数据上打分",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--data-root", type=Path, default=None)
    p.add_argument("--ckpt", type=Path, default=None, help="单折 checkpoint")
    p.add_argument("--ckpt-dir", type=Path, default=None,
                   help="含 fold*/best.pt 的目录（自动集成）")
    p.add_argument("--fold", type=int, default=None, help="只评该折的验证切片")
    p.add_argument("--slides", type=str, default=None,
                   help="逗号分隔的切片名，优先级高于 --fold")
    p.add_argument("--image-size", type=int, default=None,
                   help="覆盖输入尺寸；低于 1024 时分数会与提交时不同")
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--tta", action="store_true")
    p.add_argument("--sweep-min-area", type=str, default="0",
                   help="逗号分隔的最小连通域面积，例如 0,64,128,256")
    p.add_argument("--sweep-threshold", type=str, default="0.5",
                   help="逗号分隔的概率阈值，例如 0.4,0.5,0.6")
    p.add_argument("--max-patches", type=int, default=0, help="只评前 N 张（冒烟）")
    p.add_argument("--out", type=Path, default=None, help="结果 JSON 路径")
    return p.parse_args(argv)


def _parse_ints(text: str) -> list[int]:
    return [int(x) for x in text.split(",") if x.strip() != ""]


def _parse_floats(text: str) -> list[float]:
    return [float(x) for x in text.split(",") if x.strip() != ""]


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        data_root: Path = resolve_data_root(args.data_root)
    except FileNotFoundError as exc:
        raise SystemExit(f"[error] {exc}") from exc

    device: torch.device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        print("[warn] CUDA 不可用，退回 CPU")
        device = torch.device("cpu")

    print("[eval] 路径：")
    print(describe_paths(data_root))

    slides: list[str] = resolve_slides(data_root, args.fold, args.slides)
    pairs: list[tuple[str, Path, Path]] = collect_pairs(data_root, slides)
    if args.max_patches:
        pairs = pairs[: args.max_patches]
    if not pairs:
        raise SystemExit("[error] 没有可用样本（检查 train_patches 与 train_masks 是否齐全）")
    print(f"[eval] {len(slides)} 个切片，{len(pairs)} 张 patch")

    ckpts: list[Path] = resolve_ckpts(args.ckpt, args.ckpt_dir)
    models, size = load_models(ckpts, device, args.image_size)
    print(f"[eval] {len(ckpts)} 个模型  device={device}  image_size={size}  tta={args.tta}")
    if size != 1024:
        print(f"[warn] image_size={size}，GT 与预测都被缩放过，分数不能直接当提交成绩看")

    combos: list[Combo] = [
        Combo(thr, area)
        for thr in _parse_floats(args.sweep_threshold)
        for area in _parse_ints(args.sweep_min_area)
    ]
    accs: dict[str, Acc] = {c.label: Acc() for c in combos}

    t0: float = time.time()
    for i, (slide, jpg, gt_path) in enumerate(pairs, start=1):
        with Image.open(jpg) as im:
            rgb_src: Image.Image = im.convert("RGB")
            orig_w, orig_h = rgb_src.size
            if rgb_src.size != (size, size):
                rgb_src = rgb_src.resize((size, size), Image.BILINEAR)
            rgb: np.ndarray = np.asarray(rgb_src, dtype=np.uint8)

        prob: np.ndarray = tumor_probability(models, rgb, device, tta=args.tta)

        with Image.open(gt_path) as g:
            gt_src: Image.Image = g.convert("L")
            if gt_src.size != (orig_w, orig_h):
                gt_src = gt_src.resize((orig_w, orig_h), Image.NEAREST)
            gt: np.ndarray = np.asarray(gt_src, dtype=np.uint8) > 127

        if (orig_w, orig_h) != (size, size):
            prob = np.asarray(
                Image.fromarray(prob, mode="F").resize((orig_w, orig_h), Image.BILINEAR),
                dtype=np.float32,
            )

        gt_empty: bool = not gt.any()
        for combo in combos:
            pred: np.ndarray = prob >= combo.threshold
            if combo.min_area > 1:
                pred = remove_small_components(pred, combo.min_area)
            d, u = dice_iou(pred, gt)
            accs[combo.label].add(d, u, gt_empty, slide)

        if i % 100 == 0 or i == len(pairs):
            el: float = time.time() - t0
            print(f"[prog] {i}/{len(pairs)}  {i/max(el,1e-6):.1f} 张/s  "
                  f"已用 {el/60:.1f} min")

    # ---- 报告 ----
    print("\n" + "=" * 88)
    print("官方口径（逐 patch Dice/IoU，双空记 1.0，每张等权）")
    print("=" * 88)
    header = (f"{'参数':<24}{'综合分':>9}{'Dice':>8}{'IoU':>8}"
              f"{'空真值FP':>10}{'肿瘤Dice':>10}")
    print(header)
    print("-" * 88)
    results: dict[str, Any] = {}
    best_label: str = ""
    best_score: float = -1.0
    for combo in combos:
        s: dict[str, Any] = accs[combo.label].summary()
        results[combo.label] = s
        if not s:
            continue
        print(f"{combo.label:<24}{s['score_mean']:>9.4f}{s['dice_mean']:>8.4f}"
              f"{s['iou_mean']:>8.4f}"
              f"{s['n_empty_fp']:>10d}{s['tumor_dice_mean']:>10.4f}")
        if s["score_mean"] > best_score:
            best_score, best_label = s["score_mean"], combo.label
    print("-" * 88)

    ref: dict[str, Any] = accs[combos[0].label].summary()
    if ref:
        n_empty: int = ref["n_empty_gt"]
        n: int = ref["n_patch"]
        print(f"[参考] 全空预测基线 = {n_empty/n:.4f}（{n_empty}/{n} 张真值全空）")
        print(f"[参考] 最优参数：{best_label}  综合分 {best_score:.4f}")
        print("[解读] 空真值 FP 是最贵的错误：每张从 1.0 掉到 0，"
              "所以宁可保守，也别乱加假阳")
        print("\n[分切片]")
        for slide, v in ref["per_slide"].items():
            n_s: int = len(accs[combos[0].label].by_slide[slide])
            print(f"  {slide:<12} {v:.4f}  ({n_s} 张)")

    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        payload: dict[str, Any] = {
            "data_root": str(data_root),
            "slides": slides,
            "n_patch": len(pairs),
            "image_size": size,
            "tta": bool(args.tta),
            "ckpts": [str(p) for p in ckpts],
            "combos": results,
            "best": {"label": best_label, "score": best_score},
        }
        args.out.write_text(json.dumps(payload, ensure_ascii=False, indent=2),
                            encoding="utf-8")
        print(f"\n[out ] {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
