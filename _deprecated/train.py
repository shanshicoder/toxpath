"""ToxPath 训练脚本：三分类语义分割 + 5 折留一交叉验证 + 数据记录。

任务定义
--------
三分类（**不是四分类**），类别索引与 ``train_masks_class`` 像素值的对应：

============  ============  ==============================
mask 像素值    模型类别       含义
============  ============  ==============================
1             0             背景（纯背景 patch 整图都是 1）
2             1             前景：零星/小灶/中等
3             2             前景：较大块/大块/多灶累积
============  ============  ==============================

主要特性
--------
1. **三分类**：模型 ``num_classes=3``，loss ``DiceCrossEntropyLoss(num_classes=3)``，
   数据源为 ``train_masks_class/``（单通道标签 PNG）。
2. **独立 dataset 包**：``dataset/`` 文件夹。
3. **数据记录**：每个 epoch 末把采样日志（每类被采样多少次、像素分布）写到
   ``logs/foldN/train/dataset_log_epoch_{N}.jsonl`` + ``dataset_log_summary.json``。
4. **训练日志**：每个 epoch 的 loss/Dice/IoU 写到 ``logs/foldN/train_log.json``，
   可直接用 pandas 读。

用法
----
.. code-block:: bash

    # 本地 CPU 冒烟（32 样本 2 epoch 冻结编码器）
    python train.py --fold 1 --epochs 2 --batch-size 4 --freeze-encoder \\
        --device cpu --max-train-samples 32 --max-val-samples 16

    # 4090D 单折 50 epoch
    python train.py --fold 1 --epochs 50 --batch-size 8 --lr 3e-4

    # 全 5 折
    python train.py --all-folds --epochs 50 --batch-size 8
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import Tensor
from torch.utils.data import DataLoader

PROJECT_ROOT: Path = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

from dataset import (  # noqa: E402
    DatasetConfig, DatasetLog, ToxPathDataset,
    build_dataset, build_weighted_sampler,
)
from model import ToxPathConfig, ToxPathSegmentor  # noqa: E402
from model.losses import DiceCrossEntropyLoss  # noqa: E402


# --------------------------------------------------------------------------- #
# 默认配置
# --------------------------------------------------------------------------- #
DEFAULT_DATA_ROOT: Path = Path("E:/project/toxpath/data/toxpath")
DEFAULT_OUT_DIR: Path = PROJECT_ROOT / "checkpoints"
DEFAULT_LOG_DIR: Path = PROJECT_ROOT / "logs"

ALL_SLIDES: list[str] = ["train_001", "train_002", "train_003", "train_004", "train_005"]
FOLD_VAL_MAP: dict[int, str] = {i + 1: s for i, s in enumerate(ALL_SLIDES)}
"""fold → 留作 val 的切片名。fold 1 val=train_001。"""

# 3 类的 class_weights：小灶（最稀缺）最高、背景最低
DEFAULT_CLASS_WEIGHTS: tuple[float, float, float] = (0.25, 0.40, 0.35)
"""(背景, 小灶, 大块) → CE 权重，和为 1。

- 背景 0.25：像素占比最高（全图约 95%+），但已经被 Dice 项和采样器平衡，
  再给大权重会把前景梯度淹掉
- 小灶 0.40：只占 6.1% 的 patch，前景面积也小 → 最该学，权重最高
- 大块 0.35：占 10.1%，容易学一些 → 略低于小灶
"""


# --------------------------------------------------------------------------- #
# 指标
# --------------------------------------------------------------------------- #
def compute_metrics_per_class(
    pred: Tensor, target: Tensor, *, num_classes: int = 3, smooth: float = 1e-6
) -> dict[str, float]:
    """三类的 Dice/IoU（每类独立 + 平均）。

    Parameters
    ----------
    pred:
        ``[B, H, W]`` 的 int 预测（0/1/2）。
    target:
        ``[B, H, W]`` 的 int 真值。

    Returns
    -------
    dict
        ``dice_mean/iou_mean`` 是三类算术平均；
        ``dice_bg/dice_small/dice_large`` 是单类结果。
        ``fg_dice_mean`` 只对前景两类（1、2）取平均——这个指标更贴近
        "肿瘤区域分得准不准"，背景类占比过大会把均值抬得很高。
    """
    p = pred.cpu().numpy().reshape(-1)
    t = target.cpu().numpy().reshape(-1)
    dices: list[float] = []
    ious: list[float] = []
    for c in range(num_classes):
        pc = (p == c)
        tc = (t == c)
        inter = (pc & tc).sum()
        union = (pc | tc).sum()
        if union == 0:
            dices.append(0.0)
            ious.append(0.0)
        else:
            dices.append((2 * inter + smooth) / (pc.sum() + tc.sum() + smooth))
            ious.append((inter + smooth) / (union + smooth))
    return {
        "dice_mean": float(np.mean(dices)),
        "iou_mean": float(np.mean(ious)),
        "fg_dice_mean": float(np.mean(dices[1:])) if num_classes > 1 else 0.0,
        "dice_bg": dices[0], "dice_small": dices[1], "dice_large": dices[2],
        "iou_bg": ious[0], "iou_small": ious[1], "iou_large": ious[2],
    }


# --------------------------------------------------------------------------- #
# 单 epoch 训练
# --------------------------------------------------------------------------- #
def train_one_epoch(
    model: ToxPathSegmentor,
    loader: DataLoader,
    criterion: DiceCrossEntropyLoss,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    *,
    epoch: int,
    log_every: int = 20,
) -> dict[str, float]:
    """跑一个训练 epoch。"""
    model.train()
    losses: list[float] = []
    ces: list[float] = []
    dices: list[float] = []

    n = len(loader)
    t0 = time.time()
    for i, batch in enumerate(loader):
        img = batch["image"].to(device, non_blocking=True)
        msk = batch["mask"].to(device, non_blocking=True).squeeze(1)  # [B,H,W]

        logits = model(img)  # [B,3,H,W]
        out = criterion(logits, msk)
        loss = out["loss"]

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        losses.append(loss.item())
        ces.append(out["ce"].item())
        dices.append(out["dice"].item())

        if (i + 1) % log_every == 0 or i == 0:
            print(f"  [epoch {epoch} {i+1}/{n}] "
                  f"loss={loss.item():.4f} ce={out['ce'].item():.4f} "
                  f"dice={out['dice'].item():.4f}  "
                  f"({(i+1)/(time.time()-t0):.1f} it/s)")

    return {
        "loss": float(np.mean(losses)),
        "ce": float(np.mean(ces)),
        "dice_loss": float(np.mean(dices)),
    }


# --------------------------------------------------------------------------- #
# 验证
# --------------------------------------------------------------------------- #
@torch.no_grad()
def validate(
    model: ToxPathSegmentor,
    loader: DataLoader,
    criterion: DiceCrossEntropyLoss,
    device: torch.device,
) -> dict[str, float]:
    """跑验证，按 patch 算 3 类 Dice/IoU 再取均值。"""
    model.eval()
    losses, ces, dice_losses = [], [], []
    all_metrics: list[dict[str, float]] = []

    for batch in loader:
        img = batch["image"].to(device, non_blocking=True)
        msk = batch["mask"].to(device, non_blocking=True).squeeze(1)
        logits = model(img)
        out = criterion(logits, msk)
        losses.append(out["loss"].item())
        ces.append(out["ce"].item())
        dice_losses.append(out["dice"].item())

        pred = logits.argmax(dim=1)  # [B,H,W]
        for b in range(pred.shape[0]):
            m = compute_metrics_per_class(pred[b], msk[b])
            all_metrics.append(m)

    keys = all_metrics[0].keys() if all_metrics else []
    avg = {k: float(np.mean([m[k] for m in all_metrics])) for k in keys}
    return {
        "loss": float(np.mean(losses)),
        "ce": float(np.mean(ces)),
        "dice_loss": float(np.mean(dice_losses)),
        **avg,
    }


# --------------------------------------------------------------------------- #
# Checkpoint
# --------------------------------------------------------------------------- #
def save_checkpoint(
    path: Path,
    *,
    epoch: int,
    model: ToxPathSegmentor,
    optimizer: torch.optim.Optimizer,
    best_metric: float,
    fold: int,
    cfg: dict[str, Any],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "epoch": epoch,
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "best_metric": best_metric,
        "fold": fold,
        "cfg": cfg,
    }, path)


def load_checkpoint(
    path: Path,
    model: ToxPathSegmentor,
    optimizer: torch.optim.Optimizer | None = None,
    *,
    device: torch.device | None = None,
) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"checkpoint 不存在: {path}")
    ckpt = torch.load(path, map_location=device or "cpu", weights_only=False)
    model.load_state_dict(ckpt["model_state"])
    if optimizer is not None:
        optimizer.load_state_dict(ckpt["optimizer_state"])
    return {"epoch": ckpt["epoch"], "best_metric": ckpt["best_metric"],
            "fold": ckpt["fold"], "cfg": ckpt["cfg"]}


# --------------------------------------------------------------------------- #
# 单折训练
# --------------------------------------------------------------------------- #
def train_fold(
    fold: int,
    *,
    data_root: Path,
    out_dir: Path,
    log_dir: Path,
    epochs: int,
    batch_size: int,
    lr: float,
    weight_decay: float,
    encoder_lr_mult: float,
    freeze_encoder: bool,
    num_workers: int,
    image_size: int,
    class_weights: tuple[float, ...],
    device: torch.device,
    resume: Path | None = None,
    seed: int = 42,
    max_train_samples: int = 0,
    max_val_samples: int = 0,
) -> dict[str, Any]:
    """训练单折。"""
    val_slide = FOLD_VAL_MAP[fold]
    train_slides = [s for s in ALL_SLIDES if s != val_slide]
    print(f"\n{'='*60}\n[Fold {fold}] val={val_slide}  train={train_slides}\n{'='*60}")

    # ---- 1) 数据（含日志记录器 + 冒烟截断） ----
    fold_log_dir: Path = log_dir / f"fold{fold}"
    cfg = DatasetConfig(image_size=image_size, augment=True)
    train_ds, val_ds = build_dataset(
        data_root, train_slides, [val_slide],
        cfg=cfg, log_dir=fold_log_dir, seed=seed,
        max_train_samples=max_train_samples,
        max_val_samples=max_val_samples,
    )

    sampler = build_weighted_sampler(train_ds, seed=seed)
    train_loader = DataLoader(
        train_ds, batch_size=batch_size, sampler=sampler,
        num_workers=num_workers, pin_memory=(device.type == "cuda"),
        drop_last=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=(device.type == "cuda"),
    )

    # ---- 2) 模型（3 类） ----
    model_cfg = ToxPathConfig(image_size=image_size, num_classes=3, pretrained=False)
    model = ToxPathSegmentor(model_cfg).to(device)
    if freeze_encoder:
        model.freeze_encoder(True)
        print(f"[train] 编码器已冻结，可训练参数: "
              f"{sum(p.numel() for p in model.trainable_parameters()):,}")
    else:
        print(f"[train] 全模型可训练参数: "
              f"{sum(p.numel() for p in model.trainable_parameters()):,}")

    # ---- 3) 优化器 + loss ----
    param_groups = model.param_groups(
        lr=lr, encoder_lr_mult=encoder_lr_mult, weight_decay=weight_decay,
    )
    optimizer = torch.optim.AdamW(param_groups)
    criterion = DiceCrossEntropyLoss(
        num_classes=3,
        class_weights=class_weights,
        dice_include_background=True,  # 3 类分割，背景也参与 Dice
    ).to(device)

    # ---- 4) 续训 ----
    start_epoch = 1
    best_metric = 0.0
    ckpt_dir = out_dir / f"fold{fold}"
    if resume is not None:
        info = load_checkpoint(resume, model, optimizer, device=device)
        start_epoch = info["epoch"] + 1
        best_metric = info["best_metric"]
        print(f"[train] 从 {resume} 续训，从 epoch {start_epoch} 开始")

    # ---- 5) 训练循环 ----
    history: list[dict[str, Any]] = []
    train_log_path: Path = fold_log_dir / "train_log.json"
    train_log_path.parent.mkdir(parents=True, exist_ok=True)

    for epoch in range(start_epoch, epochs + 1):
        # 通知 dataset 当前 epoch（用于日志记录）
        train_ds.set_epoch(epoch)
        if val_ds is not None:
            val_ds.set_epoch(epoch)

        # 训练
        train_stats = train_one_epoch(
            model, train_loader, criterion, optimizer, device, epoch=epoch,
        )
        # 验证
        val_stats = validate(model, val_loader, criterion, device)

        print(f"[Fold {fold} Epoch {epoch}/{epochs}] "
              f"train_loss={train_stats['loss']:.4f} "
              f"val_loss={val_stats['loss']:.4f} "
              f"val_dice_mean={val_stats.get('dice_mean', 0):.4f} "
              f"val_fg_dice={val_stats.get('fg_dice_mean', 0):.4f} "
              f"(small={val_stats.get('dice_small', 0):.3f} "
              f"large={val_stats.get('dice_large', 0):.3f}) "
              f"val_iou_mean={val_stats.get('iou_mean', 0):.4f}")

        # 数据记录：flush 当前 epoch 的采样日志到磁盘
        if train_ds.dataset_log is not None:
            log_path = train_ds.dataset_log.flush(epoch)
            print(f"  ✓ 训练数据日志: {log_path}")
        if val_ds is not None and val_ds.dataset_log is not None:
            val_log_path = val_ds.dataset_log.flush(epoch)
            print(f"  ✓ 验证数据日志: {val_log_path}")

        # 更新最佳：用前景平均 Dice 选模型
        # 背景类占 95%+ 像素，dice_mean 从第 1 个 epoch 就 0.9+，几乎不敏感；
        # fg_dice_mean 只看小灶+大块两类，才是真正反映肿瘤分得准不准的指标
        cur_metric = val_stats.get("fg_dice_mean", 0)
        if cur_metric > best_metric:
            best_metric = cur_metric
            save_checkpoint(
                ckpt_dir / "best.pt",
                epoch=epoch, model=model, optimizer=optimizer,
                best_metric=best_metric, fold=fold,
                cfg={"lr": lr, "wd": weight_decay, "bs": batch_size,
                     "image_size": image_size, "freeze_encoder": freeze_encoder,
                     "num_classes": 3},
            )
            print(f"  ★ 新最佳 val_fg_dice={best_metric:.4f}，已存 best.pt")

        # 末尾 checkpoint
        save_checkpoint(
            ckpt_dir / "last.pt",
            epoch=epoch, model=model, optimizer=optimizer,
            best_metric=best_metric, fold=fold,
            cfg={"lr": lr, "wd": weight_decay, "bs": batch_size,
                 "image_size": image_size, "freeze_encoder": freeze_encoder,
                 "num_classes": 3},
        )
        # 训练日志
        rec = {
            "epoch": epoch, "fold": fold,
            "train_loss": train_stats["loss"],
            "train_ce": train_stats["ce"],
            "train_dice_loss": train_stats["dice_loss"],
            "val_loss": val_stats["loss"],
            "val_dice_mean": val_stats.get("dice_mean", 0),
            "val_iou_mean": val_stats.get("iou_mean", 0),
            "val_dice_bg": val_stats.get("dice_bg", 0),
            "val_dice_small": val_stats.get("dice_small", 0),
            "val_dice_large": val_stats.get("dice_large", 0),
            "val_fg_dice_mean": val_stats.get("fg_dice_mean", 0),
            "best_fg_dice_mean": best_metric,
        }
        history.append(rec)
        # 增量写入训练日志（每 epoch 一次，便于中途查看）
        train_log_path.write_text(
            json.dumps(history, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    # 保存该折完整历史
    hist_path = ckpt_dir / "history.json"
    hist_path.parent.mkdir(parents=True, exist_ok=True)
    hist_path.write_text(json.dumps(history, ensure_ascii=False, indent=2),
                         encoding="utf-8")

    return {
        "fold": fold, "val_slide": val_slide,
        "best_fg_dice_mean": best_metric,
        "last_val_dice_mean": history[-1]["val_dice_mean"] if history else 0.0,
        "last_val_fg_dice_mean": (history[-1]["val_fg_dice_mean"] if history else 0.0),
        "last_val_iou_mean": history[-1]["val_iou_mean"] if history else 0.0,
        "history": history,
    }


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="ToxPath 训练脚本（三分类语义分割，5 折留一）",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    p.add_argument("--fold", type=int, default=1, choices=[1, 2, 3, 4, 5])
    p.add_argument("--all-folds", action="store_true")
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--weight-decay", type=float, default=0.01)
    p.add_argument("--encoder-lr-mult", type=float, default=0.1)
    p.add_argument("--freeze-encoder", action="store_true")
    p.add_argument("--class-weights", type=float, nargs=3,
                   default=list(DEFAULT_CLASS_WEIGHTS),
                   help="3 类 CE 权重 [背景, 小灶, 大块]")
    p.add_argument("--image-size", type=int, default=224)
    p.add_argument("--max-train-samples", type=int, default=0,
                   help=">0 时只取前 N 个 train 样本（冒烟）")
    p.add_argument("--max-val-samples", type=int, default=0,
                   help=">0 时只取前 N 个 val 样本（冒烟）")
    p.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    p.add_argument("--log-dir", type=Path, default=DEFAULT_LOG_DIR,
                   help="数据日志与训练日志目录")
    p.add_argument("--resume", type=Path, default=None)
    p.add_argument("--device", type=str, default="auto")
    p.add_argument("--num-workers", type=int, default=2)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    print(f"[main] device = {device}")
    print(f"[main] 数据日志目录: {args.log_dir}")
    print(f"[main] checkpoint 目录: {args.out_dir}")

    folds = list(range(1, 6)) if args.all_folds else [args.fold]
    all_results: list[dict[str, Any]] = []

    for fold in folds:
        result = train_fold(
            fold,
            data_root=args.data_root,
            out_dir=args.out_dir,
            log_dir=args.log_dir,
            epochs=args.epochs,
            batch_size=args.batch_size,
            lr=args.lr,
            weight_decay=args.weight_decay,
            encoder_lr_mult=args.encoder_lr_mult,
            freeze_encoder=args.freeze_encoder,
            num_workers=args.num_workers,
            image_size=args.image_size,
            class_weights=tuple(args.class_weights),
            device=device,
            resume=args.resume if fold == args.fold else None,
            seed=args.seed,
            max_train_samples=args.max_train_samples,
            max_val_samples=args.max_val_samples,
        )
        all_results.append(result)

    # 汇总
    print("\n" + "="*60)
    print("[汇总] 5 折交叉验证结果")
    print("="*60)
    print(f"{'Fold':<6}{'val_slide':<14}{'best_fg_dice':<14}{'last_dice':<12}{'last_iou':<10}")
    for r in all_results:
        print(f"{r['fold']:<6}{r['val_slide']:<14}"
              f"{r['best_fg_dice_mean']:<14.4f}"
              f"{r['last_val_dice_mean']:<12.4f}"
              f"{r['last_val_iou_mean']:<10.4f}")
    if all_results:
        avg_dice = float(np.mean([r["best_fg_dice_mean"] for r in all_results]))
        avg_iou = float(np.mean([r["last_val_iou_mean"] for r in all_results]))
        print(f"\n  平均 best_fg_dice_mean = {avg_dice:.4f}")
        print(f"  平均 last_iou_mean     = {avg_iou:.4f}")

    summary_path = args.out_dir / "summary.json"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(
        json.dumps(all_results, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"\n[main] 汇总写入: {summary_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
