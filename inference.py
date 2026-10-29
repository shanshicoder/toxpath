"""ToxPath 推理工具：checkpoint → 三分类预测 → 彩色图 / 对比图 / 组织掩膜。

职责
----
1. **加载**：从 ``train_3class.py`` 保存的 checkpoint（``best.pt`` / ``last.pt``）
   恢复 ``ToxPathSegmentor``；
2. **前向**：读 patch JPEG → Bioptimus 归一化 → 模型 → ``argmax`` 类别图；
3. **可视化**：类别图 → 黑/绿/红彩色 PNG（配色与 ``train_masks_classes``
   完全一致，方便并排对比）；原图 + GT + 预测三联对比图；
4. **组织掩膜**：纯 numpy 的 Otsu 阈值，从 H&E 提取组织区域，
   供 ``evaluate.py`` 计算「组织区域内」的指标——白色空背景不算类别 0 的
   正确像素，避免 Dice 虚高。

被谁使用
--------
- ``evaluate.py`` 导入本模块的推理与工具函数；
- 也可以直接命令行对单张切片推理：

.. code-block:: bash

    python inference.py --ckpt checkpoints3/fold1/best.pt --slide train_001 \
        --out-dir runs3/pred_fold1 --image-size 1024 --overlay-every 20

    # 本地 CPU 冒烟（缩到 224 只跑 8 张）
    python inference.py --ckpt checkpoints3/_smoke/best.pt --slide train_001 \
        --image-size 224 --max-samples 8 --device cpu --out-dir /tmp/pred_smoke
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image
from torch import Tensor

PROJECT_ROOT: Path = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

from dataset import (  # noqa: E402
    BIOPTIMUS_MEAN,
    BIOPTIMUS_STD,
    CLASS_COLORS,
    ToxPath3Dataset,
)
from model import ToxPathConfig, ToxPathSegmentor  # noqa: E402
from paths import describe_paths, resolve_data_root  # noqa: E402

# --------------------------------------------------------------------------- #
# 常量
# --------------------------------------------------------------------------- #
# 数据根不写死盘符：见 paths.py，按「命令行 > 环境变量 > 项目内 > 历史路径」解析。

FG_RATIO_LOW: float = 0.0001
"""前景占比低于此值 → patch 级标签 0（背景）。与 make_3class_masks.py 一致。"""

FG_RATIO_HIGH: float = 0.05
"""前景占比 ≥ 此值 → patch 级标签 2（较多肿瘤）。与 make_3class_masks.py 一致。"""

_MEAN: np.ndarray = np.array(BIOPTIMUS_MEAN, dtype=np.float32)
_STD: np.ndarray = np.array(BIOPTIMUS_STD, dtype=np.float32)


# --------------------------------------------------------------------------- #
# 组织掩膜（纯 numpy，无 skimage 依赖）
# --------------------------------------------------------------------------- #
def otsu_threshold(gray: np.ndarray) -> int:
    """对 8-bit 灰度图计算 Otsu 全局阈值。

    Parameters
    ----------
    gray:
        ``[H, W]`` 的 uint8 灰度图。

    Returns
    -------
    int
        使类间方差最大的阈值（0~255）。
    """
    hist: np.ndarray = np.bincount(gray.reshape(-1), minlength=256).astype(np.float64)
    total: float = float(hist.sum())
    levels: np.ndarray = np.arange(256, dtype=np.float64)
    sum_all: float = float((levels * hist).sum())

    w_bg: np.ndarray = np.cumsum(hist)                    # 背景像素数（灰度 ≤ t）
    sum_bg: np.ndarray = np.cumsum(levels * hist)         # 背景灰度和
    w_fg: np.ndarray = total - w_bg
    valid: np.ndarray = (w_bg > 0) & (w_fg > 0)

    var_between: np.ndarray = np.full(256, -np.inf, dtype=np.float64)
    mu_bg: np.ndarray = np.where(valid, sum_bg / np.where(w_bg > 0, w_bg, 1), 0.0)
    mu_fg: np.ndarray = np.where(valid, (sum_all - sum_bg) / np.where(w_fg > 0, w_fg, 1), 0.0)
    var_between[valid] = w_bg[valid] * w_fg[valid] * (mu_bg[valid] - mu_fg[valid]) ** 2
    return int(np.argmax(var_between))


def rgb_to_gray(rgb: np.ndarray) -> np.ndarray:
    """RGB uint8 → 8-bit 灰度（Rec.601 加权）。"""
    return (0.299 * rgb[..., 0] + 0.587 * rgb[..., 1] + 0.114 * rgb[..., 2]).astype(np.uint8)


def tissue_mask_from_rgb(rgb: np.ndarray) -> np.ndarray:
    """从 H&E patch 提取组织掩膜（True = 组织）。

    原理：白色空背景灰度接近 255，组织更暗；Otsu 找到全局阈值后，
    灰度低于阈值即为组织。

    Parameters
    ----------
    rgb:
        ``[H, W, 3]`` uint8。

    Returns
    -------
    np.ndarray
        ``[H, W]`` bool，True = 组织像素。
    """
    gray: np.ndarray = rgb_to_gray(rgb)
    thr: int = otsu_threshold(gray)
    return gray < thr


# --------------------------------------------------------------------------- #
# 模型加载
# --------------------------------------------------------------------------- #
def load_checkpoint_model(
    ckpt_path: str | Path,
    device: torch.device,
    *,
    image_size: int | None = None,
) -> tuple[ToxPathSegmentor, dict[str, Any]]:
    """从 checkpoint 恢复模型并切到 eval 模式。

    Parameters
    ----------
    ckpt_path:
        ``train_3class.py`` 保存的 ``{model, epoch, metrics, cfg}`` 文件。
    device:
        目标设备。
    image_size:
        覆盖 checkpoint 里记录的 image_size（冒烟时把 1024 缩到 224 用）。
        只影响 Dataset 的 resize，不影响权重形状。

    Returns
    -------
    tuple
        ``(model, meta)``；``meta`` 含 epoch / 训练时 metrics / cfg。
    """
    path: Path = Path(ckpt_path)
    if not path.is_file():
        raise FileNotFoundError(f"checkpoint 不存在: {path}")
    ckpt: dict[str, Any] = torch.load(str(path), map_location="cpu", weights_only=False)
    cfg_ckpt: dict[str, Any] = ckpt.get("cfg", {}) or {}

    size: int = int(image_size or cfg_ckpt.get("image_size", 1024))
    num_classes: int = int(cfg_ckpt.get("num_classes", 3))
    model = ToxPathSegmentor(
        ToxPathConfig(image_size=size, num_classes=num_classes,
                      pretrained=False, grad_checkpointing=False)
    )
    model.load_state_dict(ckpt["model"], strict=True)
    model.to(device).eval()

    meta: dict[str, Any] = {
        "ckpt_path": str(path),
        "epoch": ckpt.get("epoch"),
        "train_metrics": ckpt.get("metrics", {}),
        "cfg": cfg_ckpt,
        "image_size": size,
        "num_classes": num_classes,
    }
    print(f"[load] {path.name}: epoch={meta['epoch']} image_size={size} "
          f"num_classes={num_classes}")
    return model, meta


# --------------------------------------------------------------------------- #
# 前向与可视化
# --------------------------------------------------------------------------- #
def preprocess_rgb(rgb: np.ndarray) -> Tensor:
    """``[H, W, 3]`` uint8 → ``[1, 3, H, W]`` float32（Bioptimus 归一化）。"""
    img: np.ndarray = rgb.astype(np.float32) / 255.0
    img = (img - _MEAN) / _STD
    return torch.from_numpy(img.transpose(2, 0, 1).copy()).float().unsqueeze(0)


@torch.no_grad()
def predict_rgb(model: ToxPathSegmentor, rgb: np.ndarray, device: torch.device) -> np.ndarray:
    """对一张 RGB patch 推理，返回 ``[H, W]`` int64 类别图 ``{0,1,2}``。"""
    x: Tensor = preprocess_rgb(rgb).to(device)
    logits: Tensor = model(x)
    return logits.argmax(dim=1)[0].cpu().numpy().astype(np.int64)


def fg_ratio_of(mask: np.ndarray) -> float:
    """类别图的前景（类别 1+2）像素占比。"""
    return float((mask > 0).mean())


def bin_fg_ratio(ratio: float) -> int:
    """把前景占比按 make_3class_masks.py 的阈值分箱 → {0,1,2}。"""
    if ratio < FG_RATIO_LOW:
        return 0
    if ratio < FG_RATIO_HIGH:
        return 1
    return 2


def colorize(mask: np.ndarray) -> np.ndarray:
    """``[H, W]`` 类别图 → ``[H, W, 3]`` uint8 彩色（黑/绿/红）。"""
    out: np.ndarray = np.zeros((*mask.shape, 3), dtype=np.uint8)
    for c, color in CLASS_COLORS.items():
        out[mask == c] = color
    return out


def make_compare_figure(
    rgb: np.ndarray,
    gt_mask: np.ndarray,
    pred_mask: np.ndarray,
) -> np.ndarray:
    """原图 | GT | 预测 三联横向拼接（中间 8px 白条），用于肉眼质检。"""
    h, w = rgb.shape[:2]
    gap: int = 8
    canvas: np.ndarray = np.full((h, w * 3 + gap * 2, 3), 255, dtype=np.uint8)
    canvas[:, :w] = rgb
    canvas[:, w + gap: 2 * w + gap] = colorize(gt_mask)
    canvas[:, 2 * w + gap * 2:] = colorize(pred_mask)
    return canvas


def denormalize_to_rgb(image_tensor: Tensor) -> np.ndarray:
    """Dataset 输出的归一化 ``[3, S, S]`` float → ``[S, S, 3]`` uint8 RGB。"""
    img: np.ndarray = image_tensor.cpu().numpy().transpose(1, 2, 0)
    img = img * _STD[None, None, :] + _MEAN[None, None, :]
    return np.clip(img * 255.0, 0, 255).astype(np.uint8)


# --------------------------------------------------------------------------- #
# 批量推理
# --------------------------------------------------------------------------- #
def run_inference(
    model: ToxPathSegmentor,
    data_root: Path,
    slides: list[str],
    *,
    device: torch.device,
    image_size: int = 1024,
    out_dir: Path | None = None,
    max_samples: int = 0,
    overlay_every: int = 0,
    save_pred_png: bool = False,
) -> list[dict[str, Any]]:
    """对若干切片的所有 patch 推理。

    Parameters
    ----------
    model:
        已加载权重的模型（eval 模式）。
    out_dir:
        给了就落盘：彩色预测 PNG（``save_pred_png``）、对比图（每
        ``overlay_every`` 张一张）、``predictions.jsonl`` 逐 patch 记录。
    max_samples:
        >0 时只推理前 N 个（冒烟用）。
    overlay_every:
        每 N 张存一张三联对比图；0 = 不存。

    Returns
    -------
    list[dict]
        每个 patch 一条记录：slide / name / pred_fg_ratio / pred_label。
    """
    ds = ToxPath3Dataset(
        data_root, slides,
        cfg=None, mode="val", max_samples=max_samples,
    )
    # Dataset 默认 image_size=1024；冒烟时需要小图
    ds.cfg = type(ds.cfg)(
        image_size=image_size, augment=False,
        masks_dir=ds.cfg.masks_dir, index_name=ds.cfg.index_name,
    )

    records: list[dict[str, Any]] = []
    if out_dir is not None:
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        jsonl = (out_dir / "predictions.jsonl").open("w", encoding="utf-8")

    for i in range(len(ds)):
        s: dict[str, Any] = ds.samples[i]
        item: dict[str, Any] = ds[i]
        rgb: np.ndarray = denormalize_to_rgb(item["image"])
        pred: np.ndarray = predict_rgb(model, rgb, device)
        ratio: float = fg_ratio_of(pred)
        rec: dict[str, Any] = {
            "slide": s["slide"], "name": s["name"],
            "pred_fg_ratio": round(ratio, 8),
            "pred_label": bin_fg_ratio(ratio),
        }
        records.append(rec)

        if out_dir is not None:
            if save_pred_png:
                Image.fromarray(colorize(pred)).save(
                    out_dir / f"{s['slide']}_{s['name']}_pred.png")
            if overlay_every > 0 and i % overlay_every == 0:
                gt: np.ndarray = item["mask"].numpy().astype(np.int64)
                Image.fromarray(make_compare_figure(rgb, gt, pred)).save(
                    out_dir / f"{s['slide']}_{s['name']}_compare.png")
            jsonl.write(json.dumps(rec, ensure_ascii=False) + "\n")

        if (i + 1) % 50 == 0 or i == len(ds) - 1:
            print(f"  [inference] {i + 1}/{len(ds)}")

    if out_dir is not None:
        jsonl.close()
    return records


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="ToxPath 三分类推理：checkpoint → 彩色预测图 / 对比图",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--ckpt", type=Path, required=True, help="best.pt / last.pt 路径")
    p.add_argument("--data-root", type=Path, default=None,
                   help="数据根（含 train_patches/）；不填则按 paths.py 的优先级自动解析")
    p.add_argument("--slide", type=str, required=True, help="切片名，如 train_001")
    p.add_argument("--out-dir", type=Path, required=True, help="输出目录")
    p.add_argument("--image-size", type=int, default=1024)
    p.add_argument("--max-samples", type=int, default=0, help=">0 只推理前 N 张（冒烟）")
    p.add_argument("--overlay-every", type=int, default=0,
                   help="每 N 张存一张三联对比图；0 = 不存")
    p.add_argument("--save-pred", action="store_true", help="保存每张的彩色预测 PNG")
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    try:
        args.data_root = resolve_data_root(args.data_root)
    except FileNotFoundError as exc:
        raise SystemExit(f"[error] {exc}") from exc

    device: torch.device = torch.device(args.device)
    print("[inference] 路径：")
    print(describe_paths(args.data_root))
    model, _ = load_checkpoint_model(args.ckpt, device, image_size=args.image_size)
    run_inference(
        model, args.data_root, [args.slide],
        device=device, image_size=args.image_size,
        out_dir=args.out_dir, max_samples=args.max_samples,
        overlay_every=args.overlay_every, save_pred_png=args.save_pred,
    )
    print(f"[done] 输出在 {args.out_dir}")


if __name__ == "__main__":
    main()
