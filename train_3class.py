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
``histoplus_cellvit_segmentor_20x.pt`` 里的参数可以分三部分加载（权重路径由
:mod:`paths` 解析，默认在 ``<项目根>/weights/`` 下找）：

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

    # 实验室 4090 正式训练（1024 输入，梯度累积 8）
    python train_3class.py --fold 1 --epochs 12 --batch-size 1 --grad-accum 8 \\
        --samples-per-epoch 800 --amp --grad-checkpointing

    # 全 5 折
    python train_3class.py --all-folds --epochs 12 --batch-size 1 --grad-accum 8
"""

from __future__ import annotations

import argparse
import json
import math
import random
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor, nn
from torch.utils.data import DataLoader

try:  # tensorboard 是可选依赖：没装也能训练，只是不写曲线
    from torch.utils.tensorboard import SummaryWriter
except ImportError:  # pragma: no cover - 仅在缺包时触发
    SummaryWriter = None  # type: ignore[assignment,misc]

PROJECT_ROOT: Path = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

from dataset import (  # noqa: E402
    BIOPTIMUS_MEAN,
    BIOPTIMUS_STD,
    Dataset3Config,
    build_dataset3,
    build_weighted_sampler,
    seed_worker,
)
from model import ToxPathConfig, ToxPathSegmentor  # noqa: E402
from model.losses import DiceCrossEntropyLoss  # noqa: E402
from paths import (  # noqa: E402
    describe_paths,
    resolve_data_root,
    resolve_histoplus_ckpt,
)
from splits import (  # noqa: E402
    build_slide_folds,
    discover_slides,
    load_slide_stats,
    save_folds,
    validate_folds,
)

# --------------------------------------------------------------------------- #
# 默认配置
# --------------------------------------------------------------------------- #
# 数据根与预训练权重不写死盘符：统一由 paths.py 按「命令行 > 环境变量 >
# 项目内相对路径 > 历史绝对路径」的顺序解析，换机器不用改代码。
DEFAULT_OUT_DIR: Path = PROJECT_ROOT / "checkpoints3"
DEFAULT_LOG_DIR: Path = PROJECT_ROOT / "logs3"
DEFAULT_TB_DIR: Path = PROJECT_ROOT / "logs3" / "tb"
"""TensorBoard 事件目录。``tensorboard --logdir logs3/tb`` 就能看到全部折。"""

DEFAULT_NUM_FOLDS: int = 5
"""默认折数。切片数少于它时会在 ``splits.build_slide_folds`` 里自动降低。"""

DEFAULT_CLASS_WEIGHTS: tuple[float, float, float] = (0.20, 0.50, 0.30)
"""CE 类别权重 (背景, 少量肿瘤, 较多肿瘤)。

当前值是按 5 个切片（4953 patch，类别 1 占 3.5%）调的。换成 73 个 WSI 后
类别分布必然变化，跑 ``python scripts/make_3class_masks.py --dry-run``
（只统计不写文件）看到新分布后，按 ``背景:少量:较多 ≈ (1/占比) 归一化``
重新估一遍再改。
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


# --------------------------------------------------------------------------- #
# TensorBoard 记录
# --------------------------------------------------------------------------- #
def create_writer(tb_root: Path, fold: int) -> "SummaryWriter":
    """为当前折建一个 ``SummaryWriter``，默认目录 ``<tb_root>/fold{N}``。

    若该目录里已经有上一次的 event 文件，则自动追加时间戳后缀。原因是
    TensorBoard 不会覆盖同 tag 同 step 的旧值：重跑同一折时若继续写同一目录，
    两张图的点会叠在一起，看起来像曲线在抖。
    """
    run_dir: Path = tb_root / f"fold{fold}"
    if any(run_dir.glob("events.out.tfevents.*")):
        run_dir = tb_root / f"fold{fold}_{time.strftime('%Y%m%d-%H%M%S')}"
    run_dir.mkdir(parents=True, exist_ok=True)
    # flush_secs 默认 120 s，远程看板会以为训练卡死；10 s 兼顾实时性和写盘开销
    writer = SummaryWriter(log_dir=str(run_dir), flush_secs=10)
    print(f"[tensorboard] 曲线写入 {run_dir}")
    return writer


def tb_tag(fold: int, *parts: str) -> str:
    """折内统一的 tag 前缀：``fold1/train/loss``。

    多折各写各的 run（目录不同），再叠一层 fold 前缀，是为了把五折的
    event 拷到一个目录里时也能分清。
    """
    return "/".join((f"fold{fold}", *parts))


def add_finite(writer: "SummaryWriter", tag: str, value: float, step: int) -> None:
    """只写有限值。

    ``dice_c1`` 在小折上可能整折没有该类样本，算出来是 NaN；TensorBoard 会
    把 NaN 画成断点，冒烟时图上满屏缺口反而看不懂。直接跳过更干净。
    """
    if isinstance(value, (int, float)) and math.isfinite(float(value)):
        writer.add_scalar(tag, float(value), step)


def log_epoch(
    writer: "SummaryWriter",
    *,
    fold: int,
    epoch: int,
    train_stats: dict[str, float],
    val_stats: dict[str, float],
    lr: float,
    minutes: float,
    it_per_s: float,
) -> None:
    """把一个 epoch 的标量写进 TensorBoard。

    tag 分三组：

    - ``train/*``：本 epoch 的训练均值（步级曲线在 ``train_step/*``，别混）
    - ``val/*``：验证指标，只在跑了验证的 epoch 才有点
    - ``time/*``：耗时与吞吐，用来判断是不是该换 ``num-workers`` 了
    """
    add_finite(writer, tb_tag(fold, "train", "loss"), train_stats["loss"], epoch)
    add_finite(writer, tb_tag(fold, "train", "ce"), train_stats["ce"], epoch)
    add_finite(writer, tb_tag(fold, "train", "dice_loss"),
               train_stats["dice_loss"], epoch)
    add_finite(writer, tb_tag(fold, "train", "lr"), lr, epoch)
    add_finite(writer, tb_tag(fold, "time", "epoch_min"), minutes, epoch)
    add_finite(writer, tb_tag(fold, "time", "it_per_s"), it_per_s, epoch)

    for key, tag in (
        ("loss", "val/loss"),
        ("dice_c1", "val/dice_c1"),
        ("dice_c2", "val/dice_c2"),
        ("dice_fg", "val/dice_fg"),
        ("iou_fg", "val/iou_fg"),
        ("pixel_acc", "val/pixel_acc"),
    ):
        if key in val_stats:
            add_finite(writer, tb_tag(fold, tag), val_stats[key], epoch)


# 与 ``dataset.CLASS_COLORS`` 保持一致：0 黑 / 1 绿 / 2 红
_CLASS_RGB: Tensor = torch.tensor(
    [[0, 0, 0], [0, 255, 0], [255, 0, 0]], dtype=torch.uint8
)


@torch.no_grad()
def make_preview(
    model: ToxPathSegmentor,
    batch: dict[str, Any],
    device: torch.device,
    *,
    max_items: int = 3,
    tile: int = 256,
) -> Tensor | None:
    """拼一行 ``原图 | 真值 | 预测``，返回 ``[3, H, W]`` uint8，给 ``add_image`` 用。

    看 dice 数字只能知道"错了多少"，看不出是整片漏检还是边界糊。每个验证
    epoch 存一张图，在 TensorBoard 的 IMAGES 页翻一遍比读表快得多。

    任何异常都返回 ``None``：可视化失败绝不能把训练搞挂。
    """
    try:
        images: Tensor = batch["image"].to(device)
        masks: Tensor = batch["mask"].to(device)
        pred: Tensor = model(images).argmax(dim=1)

        mean = torch.tensor(BIOPTIMUS_MEAN, device=device).view(3, 1, 1)
        std = torch.tensor(BIOPTIMUS_STD, device=device).view(3, 1, 1)
        colors = _CLASS_RGB.to(device)

        tiles: list[Tensor] = []
        for b in range(min(max_items, images.shape[0])):
            # 反归一化回 0~1，否则存进去只是一张彩色噪声
            rgb: Tensor = (images[b].float() * std + mean).clamp(0.0, 1.0)
            gt_rgb: Tensor = colors[masks[b].long()].permute(2, 0, 1).float() / 255.0
            pd_rgb: Tensor = colors[pred[b].long()].permute(2, 0, 1).float() / 255.0
            tiles.extend([rgb, gt_rgb, pd_rgb])

        panel: Tensor = torch.cat(
            [F.interpolate(t.unsqueeze(0), size=(tile, tile), mode="nearest")
             for t in tiles],
            dim=3,
        )
        return (panel[0] * 255.0).round().clamp(0, 255).to(torch.uint8)
    except Exception as exc:  # noqa: BLE001 - 可视化是锦上添花，失败就跳过
        print(f"[warn] TensorBoard 预览图生成失败，跳过：{type(exc).__name__}: {exc}")
        return None


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
    grad_accum: int,
    amp: bool,
    freeze_bn: bool,
    log_every: int = 20,
    writer: "SummaryWriter | None" = None,
    fold: int = 0,
) -> dict[str, float]:
    """跑一个训练 epoch（支持梯度累积与 AMP）。

    ``writer`` 不为空时，每 ``log_every`` 步写一次步级曲线（``train_step/*``），
    x 轴是全局步数 ``(epoch-1) * n_batches + i``，续训时也连续。
    """
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
            if writer is not None:
                # 全局步数而非 epoch 内步数：这样跨 epoch、跨续训连成一条线
                gstep: int = (epoch - 1) * n_batches + (i + 1)
                add_finite(writer, tb_tag(fold, "train_step", "loss"),
                           losses[-1], gstep)
                add_finite(writer, tb_tag(fold, "train_step", "ce"),
                           ces[-1], gstep)
                add_finite(writer, tb_tag(fold, "train_step", "dice_loss"),
                           dices[-1], gstep)
                add_finite(writer, tb_tag(fold, "train_step", "it_per_s"),
                           speed, gstep)

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
    train_slides: list[str],
    val_slides: list[str],
) -> dict[str, Any]:
    """训练并验证一折。

    Args:
        fold: 折号（从 1 开始）。
        args: 命令行参数。
        device: 训练设备。
        amp: 是否启用混合精度。
        train_slides: 本折训练切片。
        val_slides: 本折验证切片（切片数 ≥ 6 时可含多个）。
    """
    val_desc: str = (", ".join(val_slides) if len(val_slides) <= 6
                     else f"{len(val_slides)} 个切片")
    print(f"\n{'=' * 72}\n[Fold {fold}] val={val_desc}  "
          f"train={len(train_slides)} 个切片\n{'=' * 72}")

    # ---- 1) 数据 ----
    cfg_ds = Dataset3Config(
        image_size=args.image_size,
        augment=True,
    )
    train_ds, val_ds = build_dataset3(
        args.data_root, train_slides, val_slides,
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

    # ---- TensorBoard ----
    writer = create_writer(args.tb_dir, fold) if args.tensorboard else None
    if writer is not None:
        # TEXT 页留一份配置快照：几天后回头看曲线时，能确认当时是什么参数
        cfg_text: str = "\n".join([
            f"- fold: {fold}",
            f"- val_slides: {', '.join(val_slides)}",
            f"- n_train_slides: {len(train_slides)}",
            f"- epochs: {start_epoch}..{args.epochs}",
            f"- image_size: {args.image_size}",
            f"- batch_size: {args.batch_size}  grad_accum: {args.grad_accum}",
            f"- samples_per_epoch: {args.samples_per_epoch}",
            f"- lr: {args.lr}  encoder_lr_mult: {args.encoder_lr_mult}",
            f"- init: {args.init}  freeze_encoder: {args.freeze_encoder}",
            f"- class_weights: {list(args.class_weights)}",
            f"- device: {device}  amp: {amp}",
        ])
        writer.add_text(tb_tag(fold, "config"), cfg_text, 0)
        # 参考线：真值全空的 patch 占多少，全猜空就能拿到多少 Dice。
        # 曲线贴着这条线走，说明模型还没学到东西。
        fg_ratios: np.ndarray = np.asarray(
            [float(x) for x in getattr(val_loader.dataset, "fg_ratios", [])],
            dtype=np.float64,
        )
        if fg_ratios.size:
            add_finite(writer, tb_tag(fold, "val", "dice_fg_allbg_baseline"),
                       float((fg_ratios <= 0.0).mean()), 0)
        # 固定用验证集第一个 batch 出预览图，跨 epoch 才能看出同图的变化
        try:
            preview_batch: dict[str, Any] | None = next(iter(val_loader))
        except (StopIteration, TypeError):
            preview_batch = None
    else:
        preview_batch = None

    for epoch in range(start_epoch, args.epochs + 1):
        t_epoch: float = time.time()
        train_stats = train_one_epoch(
            model, train_loader, criterion, optimizer, device,
            epoch=epoch, grad_accum=args.grad_accum, amp=amp,
            freeze_bn=args.freeze_bn, log_every=args.log_every,
            writer=writer, fold=fold,
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

        # 逐 epoch 标量 + 一行预览图（原图 | 真值 | 预测），只在跑了验证时写
        if writer is not None:
            elapsed: float = time.time() - t_epoch
            log_epoch(
                writer, fold=fold, epoch=epoch,
                train_stats=train_stats, val_stats=val_stats,
                lr=float(optimizer.param_groups[0]["lr"]),
                minutes=elapsed / 60.0,
                it_per_s=len(train_loader) / max(elapsed, 1e-6),
            )
            if do_val and preview_batch is not None:
                panel = make_preview(model, preview_batch, device)
                if panel is not None:
                    writer.add_image(tb_tag(fold, "preview"), panel, epoch)

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
    if writer is not None:
        add_finite(writer, tb_tag(fold, "val", "best_dice_fg"), best_fg, 0)
        writer.flush()
        writer.close()
        print(f"[tensorboard] fold{fold} 记录完成，运行 "
              f"`tensorboard --logdir {args.tb_dir}` 查看")
    return {
        "fold": fold,
        "val_slides": list(val_slides),
        "n_train_slides": len(train_slides),
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
    p.add_argument("--data-root", type=Path, default=None,
                   help="数据根（含 train_patches/）；不填则按 paths.py 的优先级自动解析")
    p.add_argument("--fold", type=int, default=1,
                   help="训练第几折（1..实际折数）；实际折数由切片数与 --num-folds 决定")
    p.add_argument("--num-folds", type=int, default=DEFAULT_NUM_FOLDS,
                   help="交叉验证折数；切片数少于它时自动降到切片数")
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
    p.add_argument("--histoplus-ckpt", type=Path, default=None,
                   help="HistoPLUS 权重路径；不填则按 paths.py 的优先级自动解析")
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
    p.add_argument("--tb-dir", type=Path, default=DEFAULT_TB_DIR,
                   help="TensorBoard 事件目录，每折一个子目录 fold{N}")
    p.add_argument("--tensorboard", action=argparse.BooleanOptionalAction,
                   default=True, help="写 TensorBoard 曲线（--no-tensorboard 关闭）")
    p.add_argument("--resume", type=Path, default=None)
    p.add_argument("--max-train-samples", type=int, default=0, help=">0 时只取前 N 个（冒烟）")
    p.add_argument("--max-val-samples", type=int, default=0)
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    set_seed(args.seed)

    if args.tensorboard and SummaryWriter is None:
        print("[error] 开了 --tensorboard 但环境里没有 tensorboard，装一下：\n"
              "        pip install tensorboard\n"
              "        或者加 --no-tensorboard 只训练、不记曲线。", file=sys.stderr)
        return 2

    # ---- 0) 路径解析（换机器不用改代码：命令行 > 环境变量 > 项目内 > 历史路径）
    try:
        args.data_root = resolve_data_root(args.data_root)
        args.histoplus_ckpt = (
            resolve_histoplus_ckpt(args.histoplus_ckpt)
            if args.init != "none" else None
        )
    except FileNotFoundError as exc:
        print(f"[error] {exc}", file=sys.stderr)
        return 2

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    amp: bool = bool(args.amp and device.type == "cuda")
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True

    print(f"[main] device={device}  amp={amp}")
    if device.type == "cuda":
        props = torch.cuda.get_device_properties(0)
        print(f"[main] gpu={props.name}  vram={props.total_memory / 1024**3:.1f} GB")
    print("[main] 路径：")
    print(describe_paths(args.data_root, args.histoplus_ckpt))
    print(f"[main] image_size={args.image_size}  batch={args.batch_size}  accum={args.grad_accum}")

    # ---- 折划分：运行时扫描切片，按肿瘤负荷分层 ----
    try:
        slides: list[str] = discover_slides(args.data_root)
        stats = load_slide_stats(args.data_root, slides)
    except (FileNotFoundError, ValueError) as exc:
        print(f"[error] {exc}")
        return 1

    fold_map: dict[int, list[str]] = build_slide_folds(stats, args.num_folds)
    validate_folds(fold_map, slides)
    num_folds: int = len(fold_map)
    splits_path: Path = save_folds(fold_map, args.out_dir)
    print(f"[main] 折划分已写入 {splits_path}（评测会复用它，保证两边一致）")

    if args.all_folds:
        folds: list[int] = list(range(1, num_folds + 1))
    else:
        if not 1 <= args.fold <= num_folds:
            print(f"[error] --fold 必须在 1..{num_folds} 之间，当前给的是 {args.fold}")
            return 1
        folds = [args.fold]

    if args.all_folds and args.resume is not None:
        print("[warn] --all-folds 模式下忽略 --resume（resume 只对单折有意义）")
        args.resume = None
    results: list[dict[str, Any]] = []
    for fold in folds:
        val_slides: list[str] = fold_map[fold]
        val_set: set[str] = set(val_slides)
        train_slides: list[str] = [s for s in slides if s not in val_set]
        results.append(train_fold(
            fold, args, device, amp=amp,
            train_slides=train_slides, val_slides=val_slides,
        ))

    print("\n" + "=" * 72)
    print(f"[汇总] {len(results)} 折交叉验证")
    print("=" * 72)
    print(f"{'Fold':<6}{'n_val':>6}{'n_train':>8}{'best_dice_fg':>14}"
          f"{'dice_c1':>10}{'dice_c2':>10}")
    for r in results:
        print(f"{r['fold']:<6}{len(r['val_slides']):>6}{r['n_train_slides']:>8}"
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
