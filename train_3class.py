"""ToxPath 三分类训练脚本（1024×1024 输入，HistoPLUS 架构）。

任务
----
逐像素三分类（类别来自 ``train_masks_classes`` 的彩色 mask）：

======  ==============================  =====================
类别    含义                            颜色
======  ==============================  =====================
0       背景 / 非肿瘤                   (0, 0, 0) 黑
1       前景占比 0.01% ~ 5%             (0, 255, 0) 绿
2       前景占比 ≥ 5%                   (255, 0, 0) 红
======  ==============================  =====================

模型
----
``ToxPathSegmentor(num_classes=3)``：H0-mini 编码器 + HistoPLUS 四路 Neck +
单头语义解码器。网络内部会对 1024 做 reflect-pad（1024 → 1036）再裁回，因此
1024×1024 可以直接输入。

预训练权重
----------
``E:/project/histoplus_cellvit_segmentor_20x.pt`` 里的参数可以分三部分加载：

- ``encoder``：H0-mini（173 个张量，严格匹配）；
- ``neck``：四路 Neck（68 个张量，严格匹配）；
- ``decoder``：``np_branch``（73 个张量，只差最后一层 1×1 的命名/通道数，
  3 分类时最后一层会跳过，其余全部加载）。

用法
----
.. code-block:: bash

    # 本地冒烟（小尺寸、少量样本）
    python train_3class.py --fold 1 --epochs 1 --image-size 224 --batch-size 1 \\
        --max-train-samples 4 --max-val-samples 2 --device cpu --no-amp

    # 4090D 正式训练（1024 输入，梯度累积 8）
    python train_3class.py --fold 1 --epochs 12 --batch-size 1 --grad-accum 8 \\
        --samples-per-epoch 800 --amp --grad-checkpointing

    # 全 5 折
    python train_3class.py --all-folds --epochs 12 --batch-size 1 --grad-accum 8
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import Tensor, nn
from torch.utils.data import DataLoader

PROJECT_ROOT: Path = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

from dataset import (  # noqa: E402
    Dataset3Config,
    build_dataset3,
    build_weighted_sampler,
    seed_worker,
)
from model import ToxPathConfig, ToxPathSegmentor  # noqa: E402
from model.losses import DiceCrossEntropyLoss  # noqa: E402

# --------------------------------------------------------------------------- #
# 默认配置
# --------------------------------------------------------------------------- #
DEFAULT_DATA_ROOT: Path = Path("E:/project/toxpath/data/toxpath")
DEFAULT_HISTOPLUS_CKPT: Path = Path("E:/project/histoplus_cellvit_segmentor_20x.pt")
DEFAULT_OUT_DIR: Path = PROJECT_ROOT / "checkpoints3"
DEFAULT_LOG_DIR: Path = PROJECT_ROOT / "logs3"

ALL_SLIDES: list[str] = ["train_001", "train_002", "train_003", "train_004", "train_005"]
FOLD_VAL_MAP: dict[int, str] = {i + 1: s for i, s in enumerate(ALL_SLIDES)}

DEFAULT_CLASS_WEIGHTS: tuple[float, float, float] = (0.20, 0.50, 0.30)
"""CE 类别权重 (背景, 少量肿瘤, 较多肿瘤)。

类别 1 只有 172 张 patch（3.5%），是三类里最稀缺的，给最高权重。
"""


# --------------------------------------------------------------------------- #
# 随机性
# --------------------------------------------------------------------------- #
def set_seed(seed: int) -> None:
    """固定所有随机源，保证可复现。"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# --------------------------------------------------------------------------- #
# HistoPLUS 权重加载
# --------------------------------------------------------------------------- #
def _extract_state_dict(obj: Any) -> dict[str, Tensor]:
    """从 checkpoint 容器里取出真正的 state_dict。"""
    if isinstance(obj, dict):
        for key in ("state_dict", "model", "model_state_dict"):
            inner = obj.get(key)
            if isinstance(inner, dict) and any(isinstance(v, Tensor) for v in inner.values()):
                return {str(k): v for k, v in inner.items()}
        if any(isinstance(v, Tensor) for v in obj.values()):
            return {str(k): v for k, v in obj.items()}
    raise KeyError("无法从 checkpoint 中解析出 state_dict")


def load_histoplus_weights(
    model: ToxPathSegmentor,
    checkpoint: str | Path,
    *,
    parts: tuple[str, ...] = ("encoder", "neck", "decoder"),
    verbose: bool = True,
) -> dict[str, str]:
    """把 HistoPLUS 权重加载到 ``model`` 的 encoder / neck / decoder。

    返回每部分的加载摘要，便于打印与排查。
    """
    path: Path = Path(checkpoint)
    if not path.is_file():
        raise FileNotFoundError(f"找不到 HistoPLUS 权重: {path}")

    report: dict[str, str] = {}

    if "encoder" in parts:
        r = model.encoder.load_pretrained(path, verbose=verbose)
        report["encoder"] = r.summary()

    if ("neck" in parts) or ("decoder" in parts):
        sd: dict[str, Tensor] = _extract_state_dict(
            torch.load(str(path), map_location="cpu", weights_only=False)
        )

        if "neck" in parts:
            sub: dict[str, Tensor] = {
                k[len("neck."):]: v for k, v in sd.items() if k.startswith("neck.")
            }
            if sub:
                model.neck.load_state_dict(sub, strict=True)
                report["neck"] = f"loaded={len(sub)} (strict)"
            else:
                report["neck"] = "WARNING: checkpoint 里没有 neck.* 参数"

        if "decoder" in parts:
            # HistoPLUS 的 np 解码头与本项目 SemanticDecoder 结构一致，
            # 只有最后一层 1×1 叫 decoder0_header.2（本项目包了一层 FinalConv.proj）。
            sub = {}
            for k, v in sd.items():
                if not k.startswith("np_branch."):
                    continue
                name: str = k[len("np_branch."):]
                if name == "decoder0_header.2.weight":
                    name = "decoder0_header.2.proj.weight"
                elif name == "decoder0_header.2.bias":
                    name = "decoder0_header.2.proj.bias"
                sub[name] = v

            target: dict[str, Tensor] = model.decoder.state_dict()
            matched: dict[str, Tensor] = {
                k: v for k, v in sub.items()
                if k in target and target[k].shape == v.shape
            }
            skipped: list[str] = sorted(set(sub) - set(matched))
            model.decoder.load_state_dict(matched, strict=False)
            report["decoder"] = (
                f"loaded={len(matched)}/{len(sub)}"
                + (f", skipped(shape/name)={skipped}" if skipped else "")
            )

    if verbose:
        print("[weights] HistoPLUS 加载摘要:")
        for k, v in report.items():
            print(f"  {k:8s}: {v}")
    return report


# --------------------------------------------------------------------------- #
# 指标：全局混淆矩阵
# --------------------------------------------------------------------------- #
def _empty_conf(num_classes: int) -> np.ndarray:
    return np.zeros((num_classes, num_classes), dtype=np.int64)


def _update_conf(conf: np.ndarray, pred: Tensor, target: Tensor, num_classes: int) -> None:
    """按像素累加混淆矩阵：行 = GT，列 = 预测。"""
    idx: Tensor = (target.reshape(-1).long() * num_classes + pred.reshape(-1).long())
    b: Tensor = torch.bincount(idx.cpu(), minlength=num_classes * num_classes)
    conf += b.reshape(num_classes, num_classes).cpu().numpy()


def metrics_from_conf(conf: np.ndarray) -> dict[str, float]:
    """由混淆矩阵算每类 Dice/IoU 与前景平均。"""
    num_classes: int = conf.shape[0]
    tp: np.ndarray = np.diag(conf).astype(np.float64)
    fp: np.ndarray = conf.sum(axis=0) - tp
    fn: np.ndarray = conf.sum(axis=1) - tp
    eps: float = 1e-9

    dice: np.ndarray = 2 * tp / (2 * tp + fp + fn + eps)
    iou: np.ndarray = tp / (tp + fp + fn + eps)
    support: np.ndarray = conf.sum(axis=1)

    out: dict[str, float] = {}
    for c in range(num_classes):
        out[f"dice_c{c}"] = float(dice[c]) if support[c] > 0 else float("nan")
        out[f"iou_c{c}"] = float(iou[c]) if support[c] > 0 else float("nan")
        out[f"pixels_c{c}"] = float(support[c])

    # 前景两类（1、2）的 Dice 平均；缺失的类不参与平均
    fg_vals: list[float] = [float(dice[c]) for c in range(1, num_classes) if support[c] > 0]
    out["dice_fg"] = float(np.mean(fg_vals)) if fg_vals else float("nan")
    out["iou_fg"] = float(np.mean(
        [float(iou[c]) for c in range(1, num_classes) if support[c] > 0]
    )) if fg_vals else float("nan")
    out["pixel_acc"] = float(tp.sum() / (conf.sum() + eps))
    return out


# --------------------------------------------------------------------------- #
# 训练 / 验证
# --------------------------------------------------------------------------- #
def _set_bn_eval(model: nn.Module) -> None:
    """把所有 BatchNorm 固定为 eval（batch=1 时稳定 running stats）。"""
    for m in model.modules():
        if isinstance(m, nn.modules.batchnorm._BatchNorm):
            m.eval()


def train_one_epoch(
    model: ToxPathSegmentor,
    loader: DataLoader,
    criterion: DiceCrossEntropyLoss,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    *,
    epoch: int,
    grad_accum: int,
    amp: bool,
    freeze_bn: bool,
    log_every: int = 20,
) -> dict[str, float]:
    """跑一个训练 epoch（支持梯度累积与 AMP）。"""
    model.train()
    if freeze_bn:
        _set_bn_eval(model)

    losses: list[float] = []
    ces: list[float] = []
    dices: list[float] = []
    n_batches: int = len(loader)
    t0: float = time.time()
    optimizer.zero_grad(set_to_none=True)

    for i, batch in enumerate(loader):
        images: Tensor = batch["image"].to(device, non_blocking=True)
        masks: Tensor = batch["mask"].to(device, non_blocking=True)

        with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=amp):
            logits: Tensor = model(images)
            out = criterion(logits, masks)
            loss: Tensor = out["loss"] / grad_accum

        loss.backward()

        reached_end: bool = (i + 1) == n_batches
        if (i + 1) % grad_accum == 0 or reached_end:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)

        losses.append(float(out["loss"].item()))
        ces.append(float(out["ce"].item()))
        dices.append(float(out["dice"].item()))

        if (i + 1) % log_every == 0 or i == 0:
            speed: float = (i + 1) / max(time.time() - t0, 1e-6)
            print(f"  [epoch {epoch} {i+1}/{n_batches}] "
                  f"loss={losses[-1]:.4f} ce={ces[-1]:.4f} dice={dices[-1]:.4f} "
                  f"({speed:.2f} it/s)")

    return {
        "loss": float(np.mean(losses)) if losses else float("nan"),
        "ce": float(np.mean(ces)) if ces else float("nan"),
        "dice_loss": float(np.mean(dices)) if dices else float("nan"),
    }


@torch.no_grad()
def validate(
    model: ToxPathSegmentor,
    loader: DataLoader,
    criterion: DiceCrossEntropyLoss,
    device: torch.device,
    *,
    num_classes: int = 3,
) -> dict[str, float]:
    """整图验证：累加全局混淆矩阵，按切片汇总 Dice。"""
    model.eval()
    conf: np.ndarray = _empty_conf(num_classes)
    losses: list[float] = []
    per_slide: dict[str, np.ndarray] = {}

    for batch in loader:
        images: Tensor = batch["image"].to(device, non_blocking=True)
        masks: Tensor = batch["mask"].to(device, non_blocking=True)
        logits: Tensor = model(images)
        out = criterion(logits, masks)
        losses.append(float(out["loss"].item()))

        pred: Tensor = logits.argmax(dim=1)
        _update_conf(conf, pred, masks, num_classes)

        for b in range(pred.shape[0]):
            slide: str = batch["slide"][b]
            if slide not in per_slide:
                per_slide[slide] = _empty_conf(num_classes)
            _update_conf(per_slide[slide], pred[b], masks[b], num_classes)

    metrics: dict[str, float] = metrics_from_conf(conf)
    metrics["loss"] = float(np.mean(losses)) if losses else float("nan")
    for slide, c in per_slide.items():
        m = metrics_from_conf(c)
        metrics[f"{slide}_dice_c1"] = m["dice_c1"]
        metrics[f"{slide}_dice_c2"] = m["dice_c2"]
        metrics[f"{slide}_dice_fg"] = m["dice_fg"]
    return metrics


# --------------------------------------------------------------------------- #
# checkpoint
# --------------------------------------------------------------------------- #
def save_checkpoint(
    path: Path,
    *,
    model: ToxPathSegmentor,
    epoch: int,
    metrics: dict[str, float],
    cfg: dict[str, Any],
) -> None:
    """只保存模型权重 + 元信息（不存优化器，避免 1.3 GB/epoch 的磁盘开销）。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "model": model.state_dict(),
        "epoch": epoch,
        "metrics": metrics,
        "cfg": cfg,
    }, path)


# --------------------------------------------------------------------------- #
# 单折训练
# --------------------------------------------------------------------------- #
def train_fold(
    fold: int,
    args: argparse.Namespace,
    device: torch.device,
    *,
    amp: bool,
) -> dict[str, Any]:
    """训练并验证一折。"""
    val_slide: str = FOLD_VAL_MAP[fold]
    train_slides: list[str] = [s for s in ALL_SLIDES if s != val_slide]
    print(f"\n{'='*72}\n[Fold {fold}] val={val_slide} train={train_slides}\n{'='*72}")

    # ---- 1) 数据 ----
    cfg_ds = Dataset3Config(
        image_size=args.image_size,
        augment=True,
    )
    train_ds, val_ds = build_dataset3(
        args.data_root, train_slides, [val_slide],
        cfg=cfg_ds, seed=args.seed,
        max_train_samples=args.max_train_samples,
        max_val_samples=args.max_val_samples,
    )
    assert val_ds is not None

    n_samples: int = min(args.samples_per_epoch, len(train_ds)) if args.samples_per_epoch > 0 else len(train_ds)
    sampler = build_weighted_sampler(train_ds, num_samples=n_samples, seed=args.seed)
    loader_gen = torch.Generator().manual_seed(args.seed)

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        sampler=sampler,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
        drop_last=False,
        worker_init_fn=seed_worker if args.num_workers > 0 else None,
        generator=loader_gen,
        persistent_workers=args.num_workers > 0,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.val_batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
        worker_init_fn=seed_worker if args.num_workers > 0 else None,
    )

    # ---- 2) 模型 ----
    cfg_model = ToxPathConfig(
        image_size=args.image_size,
        num_classes=3,
        pretrained=False,
        grad_checkpointing=args.grad_checkpointing,
    )
    model = ToxPathSegmentor(cfg_model)
    if args.init != "none":
        parts: tuple[str, ...] = {
            "encoder": ("encoder",),
            "encoder+neck": ("encoder", "neck"),
            "full": ("encoder", "neck", "decoder"),
        }[args.init]
        load_histoplus_weights(model, args.histoplus_ckpt, parts=parts, verbose=True)
    if args.freeze_encoder:
        model.freeze_encoder(True)
    model = model.to(device)
    print(model.describe())

    # ---- 3) 优化器 + loss ----
    optimizer = torch.optim.AdamW(
        model.param_groups(
            lr=args.lr,
            encoder_lr_mult=args.encoder_lr_mult,
            weight_decay=args.weight_decay,
        )
    )
    criterion = DiceCrossEntropyLoss(
        num_classes=3,
        class_weights=tuple(args.class_weights),
        dice_include_background=False,
    ).to(device)

    # ---- 4) 续训 ----
    start_epoch: int = 1
    best_fg: float = -1.0
    if args.resume is not None:
        ckpt = torch.load(args.resume, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model"])
        start_epoch = int(ckpt.get("epoch", 0)) + 1
        best_fg = float(ckpt.get("metrics", {}).get("dice_fg", -1.0))
        print(f"[resume] 从 {args.resume} 恢复，epoch {start_epoch} 开始（best_fg={best_fg:.4f}）")

    # ---- 5) 训练循环 ----
    fold_out: Path = args.out_dir / f"fold{fold}"
    fold_log: Path = args.log_dir / f"fold{fold}"
    fold_log.mkdir(parents=True, exist_ok=True)
    history: list[dict[str, Any]] = []

    for epoch in range(start_epoch, args.epochs + 1):
        t_epoch: float = time.time()
        train_stats = train_one_epoch(
            model, train_loader, criterion, optimizer, device,
            epoch=epoch, grad_accum=args.grad_accum, amp=amp,
            freeze_bn=args.freeze_bn, log_every=args.log_every,
        )

        do_val: bool = (epoch % args.val_every == 0) or (epoch == args.epochs)
        val_stats: dict[str, float] = (
            validate(model, val_loader, criterion, device, num_classes=3)
            if do_val else {}
        )
        cur_fg: float = val_stats.get("dice_fg", float("nan"))

        line: str = (f"[Fold {fold} Epoch {epoch}/{args.epochs}] "
                     f"train_loss={train_stats['loss']:.4f} "
                     f"train_dice_loss={train_stats['dice_loss']:.4f}")
        if do_val:
            line += (f" val_loss={val_stats['loss']:.4f} "
                     f"dice_c1={val_stats['dice_c1']:.4f} "
                     f"dice_c2={val_stats['dice_c2']:.4f} "
                     f"dice_fg={cur_fg:.4f} acc={val_stats['pixel_acc']:.4f}")
        line += f" ({(time.time()-t_epoch)/60:.1f} min)"
        print(line)

        rec: dict[str, Any] = {"epoch": epoch, "fold": fold, **train_stats, **val_stats}
        history.append(rec)
        (fold_log / "train_log.json").write_text(
            json.dumps(history, ensure_ascii=False, indent=2), encoding="utf-8"
        )

        if not do_val:
            continue

        cfg_save: dict[str, Any] = {
            "fold": fold, "epoch": epoch, "image_size": args.image_size,
            "num_classes": 3, "init": args.init, "lr": args.lr,
            "batch_size": args.batch_size, "grad_accum": args.grad_accum,
        }
        if epoch % args.save_every == 0 or epoch == args.epochs:
            save_checkpoint(fold_out / "last.pt", model=model, epoch=epoch,
                            metrics=val_stats, cfg=cfg_save)
        if not np.isnan(cur_fg) and cur_fg > best_fg:
            best_fg = cur_fg
            save_checkpoint(fold_out / "best.pt", model=model, epoch=epoch,
                            metrics=val_stats, cfg=cfg_save)
            print(f"  ★ 新最佳 dice_fg={best_fg:.4f}，已保存 best.pt")

    (fold_out / "history.json").write_text(
        json.dumps(history, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return {
        "fold": fold,
        "val_slide": val_slide,
        "best_dice_fg": best_fg,
        "last_dice_fg": history[-1].get("dice_fg", float("nan")) if history else float("nan"),
        "last_dice_c1": history[-1].get("dice_c1", float("nan")) if history else float("nan"),
        "last_dice_c2": history[-1].get("dice_c2", float("nan")) if history else float("nan"),
    }


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="ToxPath 三分类训练（1024 输入，HistoPLUS 架构）",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    p.add_argument("--fold", type=int, default=1, choices=[1, 2, 3, 4, 5])
    p.add_argument("--all-folds", action="store_true")

    # 训练规模
    p.add_argument("--epochs", type=int, default=12)
    p.add_argument("--batch-size", type=int, default=1, help="1024 输入下通常只能开 1~2")
    p.add_argument("--val-batch-size", type=int, default=1)
    p.add_argument("--grad-accum", type=int, default=8, help="梯度累积步数")
    p.add_argument("--samples-per-epoch", type=int, default=800,
                   help="每个 epoch 采样多少样本；0 表示用整个训练集")
    p.add_argument("--val-every", type=int, default=1,
                   help="每 N 个 epoch 验证一次（1024 输入下可设 2~3 省时间）")

    # 优化
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--weight-decay", type=float, default=0.01)
    p.add_argument("--encoder-lr-mult", type=float, default=0.1)
    p.add_argument("--class-weights", type=float, nargs=3, default=list(DEFAULT_CLASS_WEIGHTS),
                   help="CE 权重 [背景, 少量, 较多]")
    p.add_argument("--freeze-encoder", action="store_true")
    p.add_argument("--freeze-bn", action="store_true",
                   help="固定 BatchNorm 为 eval（batch=1 时更稳）")
    p.add_argument("--grad-checkpointing", action="store_true",
                   help="ViT 梯度检查点，用时间换显存")

    # 输入 / 权重
    p.add_argument("--image-size", type=int, default=1024)
    p.add_argument("--init", choices=["none", "encoder", "encoder+neck", "full"],
                   default="encoder+neck", help="HistoPLUS 权重加载范围")
    p.add_argument("--histoplus-ckpt", type=Path, default=DEFAULT_HISTOPLUS_CKPT)
    p.add_argument("--amp", action=argparse.BooleanOptionalAction, default=True,
                   help="cuda 上用 bfloat16 自动混合精度")

    # 运行
    p.add_argument("--device", type=str, default="auto")
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--log-every", type=int, default=20)
    p.add_argument("--save-every", type=int, default=2,
                   help="每 N 个 epoch 保存一次 last.pt（best.pt 每次提升都会存）")
    p.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    p.add_argument("--log-dir", type=Path, default=DEFAULT_LOG_DIR)
    p.add_argument("--resume", type=Path, default=None)
    p.add_argument("--max-train-samples", type=int, default=0, help=">0 时只取前 N 个（冒烟）")
    p.add_argument("--max-val-samples", type=int, default=0)
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    set_seed(args.seed)

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    amp: bool = bool(args.amp and device.type == "cuda")
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True

    print(f"[main] device={device}  amp={amp}")
    print(f"[main] data_root={args.data_root}")
    print(f"[main] image_size={args.image_size}  batch={args.batch_size}  accum={args.grad_accum}")

    folds: list[int] = list(range(1, 6)) if args.all_folds else [args.fold]
    if args.all_folds and args.resume is not None:
        print("[warn] --all-folds 模式下忽略 --resume（resume 只对单折有意义）")
        args.resume = None
    results: list[dict[str, Any]] = []
    for fold in folds:
        results.append(train_fold(
            fold, args, device, amp=amp,
        ))

    print("\n" + "=" * 72)
    print("[汇总] 5 折交叉验证")
    print("=" * 72)
    print(f"{'Fold':<6}{'val_slide':<14}{'best_dice_fg':>14}{'dice_c1':>10}{'dice_c2':>10}")
    for r in results:
        print(f"{r['fold']:<6}{r['val_slide']:<14}"
              f"{r['best_dice_fg']:>14.4f}{r['last_dice_c1']:>10.4f}{r['last_dice_c2']:>10.4f}")
    if results:
        vals = [r["best_dice_fg"] for r in results if not np.isnan(r["best_dice_fg"])]
        if vals:
            print(f"\n  dice_fg 均值 = {np.mean(vals):.4f} ± {np.std(vals):.4f}")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    summary_path: Path = args.out_dir / "summary.json"
    summary_path.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n[main] 汇总写入: {summary_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
