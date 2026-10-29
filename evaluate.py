"""ToxPath 正式评测：checkpoint → 三层指标 + 逐 patch 明细 + 5 折汇总。

三层指标
--------
1. **像素级（整图口径）**：混淆矩阵 → 每类 Dice / IoU / 像素精度。
   与 ``train_3class.py`` 的验证完全同口径（直接复用其函数）；
2. **像素级（组织区域内口径，主指标）**：整图口径下背景类占 95%+ 像素，
   会把分数抬高。用 Otsu 从 H&E 提取组织掩膜，只在组织内重算——
   这个数字反映真实分割能力，写报告 / 论文用它；
3. **patch 级分级**：模型预测前景占比，按 0.01% / 5% 阈值分箱，
   与真值 patch 标签对比——回答「少量 vs 较多会不会搞混」。

产出（默认 ``eval3/``）
----------------------
::

    eval3/foldN/
    ├── metrics.json          # 本折全部指标
    ├── per_patch.csv         # 每 patch 一行（slide/name/gt/pred/占比）
    └── compare_*.png         # 三联对比图（原图 | GT | 预测），抽样保存
    eval3/summary.csv         # 5 折汇总（mean ± std）
    eval3/summary.json

用法
----
.. code-block:: bash

    # 单折（用本地训练产出的 best.pt）
    python evaluate.py --fold 1 --ckpt checkpoints3/fold1/best.pt

    # 全 5 折 + 汇总
    python evaluate.py --all-folds --ckpt-dir checkpoints3

    # 本地 CPU 冒烟（随机权重 + 224 输入 + 8 张）
    python evaluate.py --fold 1 --ckpt checkpoints3/_smoke/best.pt \
        --image-size 224 --max-samples 8 --device cpu --overlay-every 2
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import Tensor
from torch.utils.data import DataLoader

PROJECT_ROOT: Path = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

from dataset import ToxPath3Dataset  # noqa: E402
from inference import (  # noqa: E402
    bin_fg_ratio,
    denormalize_to_rgb,
    fg_ratio_of,
    load_checkpoint_model,
    make_compare_figure,
    tissue_mask_from_rgb,
)
from paths import describe_paths, resolve_data_root  # noqa: E402
from train_3class import (  # noqa: E402
    ALL_SLIDES,
    FOLD_VAL_MAP,
    _empty_conf,
    metrics_from_conf,
)

# --------------------------------------------------------------------------- #
# 混淆矩阵（numpy 版，供组织掩膜的子集统计用）
# --------------------------------------------------------------------------- #
def _update_conf_np(conf: np.ndarray, pred: np.ndarray, target: np.ndarray) -> None:
    """按像素累加混淆矩阵（numpy 版，行 = GT，列 = 预测）。"""
    idx: np.ndarray = target.reshape(-1).astype(np.int64) * conf.shape[0] \
        + pred.reshape(-1).astype(np.int64)
    b: np.ndarray = np.bincount(idx, minlength=conf.size)
    conf += b.reshape(conf.shape)


def patch_fg_dice(pred: np.ndarray, target: np.ndarray) -> float:
    """单张 patch 的前景（类别 1+2）Dice；两边都无前景时返回 nan。"""
    p: np.ndarray = pred > 0
    t: np.ndarray = target > 0
    denom: int = int(p.sum() + t.sum())
    if denom == 0:
        return float("nan")
    return float(2.0 * int((p & t).sum()) / denom)


# --------------------------------------------------------------------------- #
# 单折评测
# --------------------------------------------------------------------------- #
def evaluate_fold(
    fold: int,
    ckpt_path: Path,
    *,
    data_root: Path,
    device: torch.device,
    image_size: int = 1024,
    max_samples: int = 0,
    overlay_every: int = 40,
    out_dir: Path,
    slide: str | None = None,
    batch_size: int = 1,
    num_workers: int = 0,
) -> dict[str, Any]:
    """评测一折，落盘 metrics.json / per_patch.csv / 对比图，返回指标 dict。"""
    val_slide: str = slide or FOLD_VAL_MAP[fold]
    print(f"\n{'=' * 72}\n[Eval fold {fold}] ckpt={ckpt_path.name} val={val_slide}\n{'=' * 72}")

    model, meta = load_checkpoint_model(ckpt_path, device, image_size=image_size)

    ds = ToxPath3Dataset(
        data_root, [val_slide],
        cfg=None, mode="val", max_samples=max_samples,
    )
    ds.cfg = type(ds.cfg)(
        image_size=image_size, augment=False,
        masks_dir=ds.cfg.masks_dir, index_name=ds.cfg.index_name,
    )
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False,
                        num_workers=num_workers)

    n_cls: int = 3
    conf_full: np.ndarray = _empty_conf(n_cls)
    conf_tissue: np.ndarray = _empty_conf(n_cls)
    conf_patch_cls: np.ndarray = _empty_conf(n_cls)   # 行 = GT patch 标签，列 = 预测分级
    per_slide_full: dict[str, np.ndarray] = {}
    per_slide_tissue: dict[str, np.ndarray] = {}
    rows: list[dict[str, Any]] = []

    fold_out: Path = Path(out_dir) / f"fold{fold}"
    fold_out.mkdir(parents=True, exist_ok=True)

    seen: int = 0
    for batch in loader:
        images: Tensor = batch["image"].to(device, non_blocking=True)
        logits: Tensor = model(images)
        pred_t: Tensor = logits.argmax(dim=1).cpu()

        masks: np.ndarray = batch["mask"].numpy()
        preds: np.ndarray = pred_t.numpy()

        for b in range(preds.shape[0]):
            slide_name: str = batch["slide"][b]
            name: str = batch["name"][b]
            gt: np.ndarray = masks[b]
            pr: np.ndarray = preds[b]

            # ---- 像素级（整图口径）----
            _update_conf_np(conf_full, pr, gt)
            per_slide_full.setdefault(slide_name, _empty_conf(n_cls))
            _update_conf_np(per_slide_full[slide_name], pr, gt)

            # ---- 像素级（组织区域内口径）----
            rgb: np.ndarray = denormalize_to_rgb(images[b])
            tissue: np.ndarray = tissue_mask_from_rgb(rgb)
            if tissue.any():
                _update_conf_np(conf_tissue, pr[tissue], gt[tissue])
                per_slide_tissue.setdefault(slide_name, _empty_conf(n_cls))
                _update_conf_np(per_slide_tissue[slide_name], pr[tissue], gt[tissue])

            # ---- patch 级分级 ----
            pred_ratio: float = fg_ratio_of(pr)
            pred_label: int = bin_fg_ratio(pred_ratio)
            gt_label: int = int(batch["patch_label"][b])
            conf_patch_cls[gt_label, pred_label] += 1

            rows.append({
                "fold": fold,
                "slide": slide_name,
                "name": name,
                "gt_label": gt_label,
                "pred_label": pred_label,
                "gt_fg_ratio": round(float(batch["fg_ratio"][b]), 8),
                "pred_fg_ratio": round(pred_ratio, 8),
                "patch_fg_dice": round(patch_fg_dice(pr, gt), 6),
                "tissue_ratio": round(float(tissue.mean()), 6),
                "correct": int(pred_label == gt_label),
            })

            # ---- 对比图（抽样）----
            if overlay_every > 0 and seen % overlay_every == 0:
                fig: np.ndarray = make_compare_figure(rgb, gt, pr)
                from PIL import Image as _Image
                _Image.fromarray(fig).save(fold_out / f"compare_{slide_name}_{name}.png")

            seen += 1

        if (len(rows) // max(batch_size, 1)) % 20 == 0 or seen >= len(ds):
            print(f"  [eval] {seen}/{len(ds)}")

    # ---- 汇总指标 ----
    m_full: dict[str, float] = metrics_from_conf(conf_full)
    m_tissue: dict[str, float] = metrics_from_conf(conf_tissue)

    # patch 级分级指标
    patch_acc: float = float(np.trace(conf_patch_cls) / max(conf_patch_cls.sum(), 1))
    patch_metrics: dict[str, Any] = {
        "patch_acc": patch_acc,
        "patch_confusion": conf_patch_cls.tolist(),   # 行 = GT(0/1/2)，列 = 预测
        "patch_correct": int(np.trace(conf_patch_cls)),
        "patch_total": int(conf_patch_cls.sum()),
    }

    # 逐切片
    per_slide_metrics: dict[str, Any] = {}
    for slide_name, c in per_slide_full.items():
        m: dict[str, float] = metrics_from_conf(c)
        entry: dict[str, Any] = {
            "dice_c1": m["dice_c1"], "dice_c2": m["dice_c2"], "dice_fg": m["dice_fg"],
            "n_patches": sum(1 for r in rows if r["slide"] == slide_name),
        }
        if slide_name in per_slide_tissue:
            mt: dict[str, float] = metrics_from_conf(per_slide_tissue[slide_name])
            entry["tissue_dice_fg"] = mt["dice_fg"]
            entry["tissue_dice_c2"] = mt["dice_c2"]
        per_slide_metrics[slide_name] = entry

    result: dict[str, Any] = {
        "fold": fold,
        "val_slide": val_slide,
        "ckpt": str(ckpt_path),
        "ckpt_epoch": meta["epoch"],
        "n_patches": len(rows),
        # 整图口径
        **{f"full_{k}": v for k, v in m_full.items() if not k.startswith("pixels")},
        # 组织口径（主指标）
        **{f"tissue_{k}": v for k, v in m_tissue.items() if not k.startswith("pixels")},
        # patch 级
        **patch_metrics,
        # 逐切片
        "per_slide": per_slide_metrics,
    }

    # ---- 落盘 ----
    (fold_out / "metrics.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    with (fold_out / "per_patch.csv").open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    print(f"[Eval fold {fold}] "
          f"full: dice_c2={m_full['dice_c2']:.4f} dice_fg={m_full['dice_fg']:.4f} | "
          f"tissue: dice_c2={m_tissue['dice_c2']:.4f} dice_fg={m_tissue['dice_fg']:.4f} | "
          f"patch_acc={patch_acc:.4f}")
    print(f"  输出: {fold_out}")
    return result


# --------------------------------------------------------------------------- #
# 多折汇总
# --------------------------------------------------------------------------- #
KEY_METRICS: list[str] = [
    "full_dice_c1", "full_dice_c2", "full_dice_fg", "full_iou_fg", "full_pixel_acc",
    "tissue_dice_c1", "tissue_dice_c2", "tissue_dice_fg", "tissue_iou_fg",
    "tissue_pixel_acc", "patch_acc",
]


def summarize(results: list[dict[str, Any]], out_dir: Path) -> None:
    """5 折指标 mean ± std → summary.csv + summary.json。"""
    out_dir = Path(out_dir)
    lines: list[str] = ["metric,mean,std"] + [
        f"{k},{np.nanmean([r[k] for r in results]):.6f},"
        f"{np.nanstd([r[k] for r in results]):.6f}"
        for k in KEY_METRICS if k in results[0]
    ]
    (out_dir / "summary.csv").write_text("\n".join(lines) + "\n", encoding="utf-8")
    (out_dir / "summary.json").write_text(
        json.dumps([{k: v for k, v in r.items() if k != "per_slide"} for r in results],
                   ensure_ascii=False, indent=2),
        encoding="utf-8")
    print(f"\n[summary] {len(results)} 折已写入 {out_dir / 'summary.csv'}")
    for k in ("tissue_dice_c2", "tissue_dice_fg", "patch_acc"):
        vals = [r[k] for r in results]
        print(f"  {k:18s} = {np.nanmean(vals):.4f} ± {np.nanstd(vals):.4f}")


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="ToxPath 三分类正式评测（像素级整图/组织 + patch 级分级）",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--fold", type=int, choices=[1, 2, 3, 4, 5])
    p.add_argument("--all-folds", action="store_true")
    p.add_argument("--ckpt", type=Path, help="单折 checkpoint（--fold 用）")
    p.add_argument("--ckpt-dir", type=Path, default=PROJECT_ROOT / "checkpoints3",
                   help="含 foldN/best.pt 的目录（--all-folds 用）")
    p.add_argument("--slide", type=str, default=None,
                   help="覆盖验证切片（默认按 FOLD_VAL_MAP）")
    p.add_argument("--data-root", type=Path, default=None,
                   help="数据根（含 train_patches/）；不填则按 paths.py 的优先级自动解析")
    p.add_argument("--out-dir", type=Path, default=PROJECT_ROOT / "eval3")
    p.add_argument("--image-size", type=int, default=1024)
    p.add_argument("--max-samples", type=int, default=0, help=">0 每折只测前 N 张（冒烟）")
    p.add_argument("--overlay-every", type=int, default=40,
                   help="每 N 张存一张三联对比图；0 = 不存")
    p.add_argument("--batch-size", type=int, default=1)
    p.add_argument("--num-workers", type=int, default=0)
    p.add_argument("--device", type=str,
                   default="cuda" if torch.cuda.is_available() else "cpu")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    try:
        args.data_root = resolve_data_root(args.data_root)
    except FileNotFoundError as exc:
        raise SystemExit(f"[error] {exc}") from exc

    device: torch.device = torch.device(args.device)
    print("[evaluate] 路径：")
    print(describe_paths(args.data_root))

    if args.all_folds:
        results: list[dict[str, Any]] = []
        for fold in range(1, len(ALL_SLIDES) + 1):
            ckpt: Path = args.ckpt_dir / f"fold{fold}" / "best.pt"
            if not ckpt.is_file():
                print(f"[skip] fold{fold} 没有 {ckpt}")
                continue
            results.append(evaluate_fold(
                fold, ckpt, data_root=args.data_root, device=device,
                image_size=args.image_size, max_samples=args.max_samples,
                overlay_every=args.overlay_every, out_dir=args.out_dir,
                slide=args.slide, batch_size=args.batch_size,
                num_workers=args.num_workers,
            ))
        if results:
            summarize(results, args.out_dir)
        else:
            print("[warn] 没有可评测的 checkpoint")
    else:
        if args.fold is None or args.ckpt is None:
            raise SystemExit("单折评测需要 --fold N --ckpt path/to/best.pt")
        evaluate_fold(
            args.fold, args.ckpt, data_root=args.data_root, device=device,
            image_size=args.image_size, max_samples=args.max_samples,
            overlay_every=args.overlay_every, out_dir=args.out_dir,
            slide=args.slide, batch_size=args.batch_size,
            num_workers=args.num_workers,
        )


if __name__ == "__main__":
    main()
