r"""ToxPath 比赛提交生成：``test_patches/`` → 逐 patch 二值掩膜 → 一个 zip。

官方提交规范（ToxPath AI Challenge 2026）
-----------------------------------------
::

    test_patches/test_001/1.0/0_0.jpeg   →   submission/test_001/0_0.png

- 每张官方 patch 输出一张掩膜，**不能多也不能少**；
- PNG、**单通道灰度 8-bit**、像素值**只能是 0 和 255**；
- 尺寸必须与**原图完全一致**，评测系统不会帮你 resize；
- 文件名只把 ``.jpeg`` 换成 ``.png``，**不许加 ``_mask`` / ``_pred`` 后缀**；
- 顶层只保留 ``{slide}/`` 这一层，中间那个 ``1.0/`` 层级目录**要去掉**；
- zip 里直接就是 ``test_001/0_0.png``，**不能多套一层文件夹**。

本脚本做四件事
--------------
1. 扫 ``test_patches/{slide}/**/*.jpeg``（层级目录有或没有都能处理）；
2. 逐张推理 → 三类 logits → 肿瘤概率（``1 - p_background``）；
   支持多折 checkpoint 的**概率平均集成**与 hflip/vflip **TTA**；
3. 可选**最小连通域过滤**（``--min-area``）——按官方指标，一张真值全空的
   patch 只要出现 1 个假阳像素，Dice 就直接从 1.0 掉到 0，
   所以去掉零星小斑点对总分是纯收益（本地实测最小真实病灶 83 px）；
4. 按官方目录结构落盘，可选直接压成 zip（zip 内条目不套外层目录）。

用法
----
.. code-block:: bash

    # 单折
    python predict_test.py --test-root D:\data\toxpath_test \
        --ckpt checkpoints3/fold1/best.pt --out-dir submission --zip

    # 5 折集成 + TTA + 最小连通域 200 px
    python predict_test.py --test-root D:\data\toxpath_test \
        --ckpt-dir checkpoints3 --tta --min-area 200 \
        --out-dir submission --zip submission.zip

    # 本地 CPU 冒烟（缩到 224，只跑 4 张）
    python predict_test.py --test-root /tmp/fake_test --ckpt /tmp/rand.pt \
        --image-size 224 --max-patches 4 --device cpu --out-dir /tmp/sub

注意
----
**不要**从 ``test_images/*.tif`` 重新切图。官方 patch 文件名编码了原图坐标，
自己切的图坐标系不同，会和真值对不上；直接在官方 ``test_patches/`` 上推理。
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import zipfile
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch
from PIL import Image
from torch import Tensor

PROJECT_ROOT: Path = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

from inference import load_checkpoint_model, preprocess_rgb  # noqa: E402

# --------------------------------------------------------------------------- #
# 常量
# --------------------------------------------------------------------------- #
IMAGE_SUFFIXES: tuple[str, ...] = (".jpeg", ".jpg", ".png", ".tif", ".tiff")
"""官方 patch 的扩展名（正常只有 .jpeg，多留几个容错）。"""

FOREGROUND_VALUE: int = 255
"""提交掩膜的前景像素值（官方只接受 0 / 255）。"""

BACKGROUND_VALUE: int = 0


# --------------------------------------------------------------------------- #
# 1. 扫描官方 patch
# --------------------------------------------------------------------------- #
def discover_test_patches(test_root: Path) -> dict[str, list[Path]]:
    """扫描 ``{test_root}/test_patches/{slide}/**/*.jpeg``。

    Parameters
    ----------
    test_root:
        含 ``test_patches/`` 的那一层目录。若直接把 ``test_patches`` 本身
        传进来也能识别。

    Returns
    -------
    dict
        ``{slide 名: [patch 路径, ...]}``，按名字排序保证可复现。

    Raises
    ------
    FileNotFoundError
        找不到 ``test_patches/`` 或里面一个 patch 都没有。
    ValueError
        同一 slide 下出现重名 stem（例如同时存在 ``1.0/0_0.jpeg`` 与
        ``2.0/0_0.jpeg``）。官方提交要求把层级目录去掉，重名会互相覆盖，
        这时必须停下来人工确认，不能静默丢图。
    """
    root: Path = test_root / "test_patches"
    if not root.is_dir():
        # 允许直接传 test_patches 本身
        root = test_root
    if not root.is_dir():
        raise FileNotFoundError(
            f"找不到 test_patches 目录：{test_root}\n"
            f"        期望结构：{test_root}\\test_patches\\test_001\\1.0\\*.jpeg"
        )

    slides: dict[str, list[Path]] = {}
    for slide_dir in sorted(p for p in root.iterdir() if p.is_dir()):
        files: list[Path] = []
        for suffix in IMAGE_SUFFIXES:
            files.extend(slide_dir.rglob(f"*{suffix}"))
        if not files:
            continue
        files = sorted(set(files))

        # 同名 stem 冲突检查：提交目录会拍平，重名会互相覆盖
        seen: dict[str, Path] = {}
        for f in files:
            if f.stem in seen:
                raise ValueError(
                    f"{slide_dir.name} 下出现重名 patch："
                    f"{seen[f.stem]} 与 {f}\n"
                    f"        官方提交要求去掉层级目录，重名会覆盖，"
                    f"请先确认官方给的 test_patches 是否只有一层"
                )
            seen[f.stem] = f
        slides[slide_dir.name] = files

    if not slides:
        raise FileNotFoundError(f"{root} 下没有找到任何 patch 图像")
    return slides


# --------------------------------------------------------------------------- #
# 2. 模型与推理
# --------------------------------------------------------------------------- #
def resolve_ckpts(
    ckpt: Path | None,
    ckpt_dir: Path | None,
) -> list[Path]:
    """整理要用的 checkpoint 列表（1 个 = 单模型，多个 = 概率集成）。

    约定：``--ckpt`` 传单个文件；``--ckpt-dir`` 传目录则自动收集其中
    所有 ``fold*/best.pt`` 做集成。
    """
    if ckpt is not None:
        if not ckpt.is_file():
            raise FileNotFoundError(f"checkpoint 不存在：{ckpt}")
        return [ckpt]

    if ckpt_dir is None:
        raise SystemExit("需要 --ckpt 或 --ckpt-dir 指定模型")

    found: list[Path] = []
    # 按 fold1..foldN 顺序收集，顺序固定保证集成结果可复现
    candidates: list[Path] = sorted(
        ckpt_dir.glob("fold*/best.pt"),
        key=lambda p: int("".join(c for c in p.parent.name if c.isdigit()) or 0),
    )
    found.extend(candidates)
    if not found:
        single: Path = ckpt_dir / "best.pt"
        if single.is_file():
            found.append(single)
    if not found:
        raise FileNotFoundError(f"{ckpt_dir} 下没有 fold*/best.pt 也没有 best.pt")
    return found


def load_models(
    ckpts: list[Path],
    device: torch.device,
    image_size: int | None,
) -> tuple[list[Any], int]:
    """加载若干模型，返回 ``(models, 统一输入尺寸)``。

    所有 checkpoint 必须是同一个 ``image_size``（集成的概率图要逐像素平均），
    不一致时直接报错而不是偷偷 resize。
    """
    models: list[Any] = []
    sizes: list[int] = []
    for p in ckpts:
        model, meta = load_checkpoint_model(p, device, image_size=image_size)
        models.append(model)
        sizes.append(int(meta["image_size"]))
    if len(set(sizes)) > 1:
        raise SystemExit(f"多个 checkpoint 的 image_size 不一致：{sizes}，无法集成")
    return models, sizes[0]


@torch.no_grad()
def tumor_probability(
    models: Iterable[Any],
    rgb: np.ndarray,
    device: torch.device,
    *,
    tta: bool,
) -> np.ndarray:
    """对一张 patch 输出肿瘤概率图 ``[H, W]`` float32（0~1）。

    肿瘤概率 = ``1 - softmax(logits)[背景类]``，等价于 ``p1 + p2``。
    三类模型把肿瘤分成「零星」「大块」两类，但比赛只判肿瘤/非肿瘤，
    二值化时两类合并即可。

    前景一致性核对（本地 4953 张）：4950 张三分类 mask 的前景像素与二值
    真值**逐像素完全一致**；另有 3 张（``train_001/12_66`` 83 px、
    ``train_002/65_16`` 100 px、``train_002/68_21`` 106 px）前景不足
    128 px，按 ``labels.CcConfig.min_fg_px`` 视为标注噪声并入类 0，
    这是标签构造的既定行为，对总分无影响。

    Parameters
    ----------
    models:
        一个或多个模型；多个时对概率图逐像素平均。
    rgb:
        ``[H, W, 3]`` uint8。
    tta:
        是否做 hflip / vflip 两种翻转平均。
    """
    # (变换, 逆变换) 成对写清楚，避免靠顺序推断翻转方向
    views: list[tuple[np.ndarray, int]] = [(rgb, 0)]
    if tta:
        views.append((rgb[:, ::-1], 1))   # 1 = 水平翻转，逆变换仍是水平翻转
        views.append((rgb[::-1, :], 2))   # 2 = 垂直翻转

    acc: np.ndarray | None = None
    for view, flip in views:
        x: Tensor = preprocess_rgb(np.ascontiguousarray(view)).to(device)
        probs: list[np.ndarray] = []
        for model in models:
            logits: Tensor = model(x)
            p: Tensor = torch.softmax(logits.float(), dim=1)
            probs.append((1.0 - p[0, 0]).cpu().numpy().astype(np.float32))
        view_prob: np.ndarray = np.mean(probs, axis=0)

        if flip == 1:
            view_prob = view_prob[:, ::-1]
        elif flip == 2:
            view_prob = view_prob[::-1, :]
        acc = view_prob if acc is None else acc + view_prob

    assert acc is not None
    return (acc / len(views)).astype(np.float32)


# --------------------------------------------------------------------------- #
# 3. 后处理：最小连通域过滤
# --------------------------------------------------------------------------- #
def remove_small_components(mask: np.ndarray, min_area: int) -> np.ndarray:
    """去掉面积小于 ``min_area`` 的连通域（**4 邻接**：对角线不算连通）。

    为什么需要：官方指标里「真值与预测都全空」的 patch 记 Dice=IoU=1.0，
    而真值全空的 patch 只要预测出**一个**前景像素，Dice 立刻变 0。
    本地数据 83.5% 的 patch 是全空背景，所以零星假阳的代价极高。

    这里刻意用 4 邻接而不是 8 邻接：对角线相连的散点会被拆成独立小块，
    更容易被清掉，对抑制椒盐噪声更严格；真实肿瘤是成片的，
    两种邻接结果无差别。

    实现：按行行程编码（run-length）建图 + 并查集，只对行程建索引，
    不逐像素循环，400k 前景像素的 patch 也在毫秒级完成，
    且不依赖 scipy / skimage。

    Parameters
    ----------
    mask:
        ``[H, W]`` bool，True = 前景。
    min_area:
        面积阈值（像素）。``<= 1`` 时原样返回。

    Returns
    -------
    np.ndarray
        过滤后的 bool 掩膜。
    """
    if min_area <= 1 or not mask.any():
        return mask

    h, w = mask.shape
    parent: list[int] = []
    run_y: list[int] = []
    run_s: list[int] = []
    run_e: list[int] = []

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]      # 路径压缩（隔代）
            x = parent[x]
        return x

    def union(a: int, b: int) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra

    prev: list[tuple[int, int, int]] = []       # (start, end, run_id)
    for y in range(h):
        row: np.ndarray = mask[y]
        cur: list[tuple[int, int, int]] = []
        if row.any():
            padded: np.ndarray = np.concatenate(([False], row, [False]))
            diff: np.ndarray = np.diff(padded.astype(np.int8))
            starts: np.ndarray = np.flatnonzero(diff == 1)
            ends: np.ndarray = np.flatnonzero(diff == -1)
            for s, e in zip(starts.tolist(), ends.tolist()):
                rid: int = len(parent)
                parent.append(rid)
                run_y.append(y)
                run_s.append(s)
                run_e.append(e)
                cur.append((s, e, rid))

            # 与上一行的行程求交（半开区间相交 → 4 邻接连通，对角线不算）
            i = j = 0
            while i < len(prev) and j < len(cur):
                ps, pe, pid = prev[i]
                cs, ce, cid = cur[j]
                if ps < ce and cs < pe:
                    union(pid, cid)
                if pe <= ce:
                    i += 1
                else:
                    j += 1
        prev = cur

    # 汇总每个连通域的面积
    area: dict[int, int] = {}
    for rid in range(len(parent)):
        root: int = find(rid)
        area[root] = area.get(root, 0) + (run_e[rid] - run_s[rid])

    out: np.ndarray = np.zeros_like(mask)
    for rid in range(len(parent)):
        if area[find(rid)] >= min_area:
            out[run_y[rid], run_s[rid]:run_e[rid]] = True
    return out


# --------------------------------------------------------------------------- #
# 4. 落盘与打包
# --------------------------------------------------------------------------- #
def save_mask(mask: np.ndarray, path: Path) -> None:
    """写单通道 8-bit PNG，像素值 0 / 255（官方硬性要求）。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    arr: np.ndarray = np.where(mask, FOREGROUND_VALUE, BACKGROUND_VALUE).astype(np.uint8)
    Image.fromarray(arr, mode="L").save(path, format="PNG", optimize=True)


def zip_submission(sub_dir: Path, zip_path: Path) -> tuple[int, int]:
    """把 ``sub_dir`` 下的 ``{slide}/*.png`` 打进 zip，**不套外层目录**。

    Returns
    -------
    tuple
        ``(条目数, zip 字节数)``。
    """
    zip_path.parent.mkdir(parents=True, exist_ok=True)
    if zip_path.exists():
        zip_path.unlink()
    count: int = 0
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED,
                         compresslevel=9) as zf:
        for slide_dir in sorted(p for p in sub_dir.iterdir() if p.is_dir()):
            for png in sorted(slide_dir.glob("*.png")):
                # arcname 用相对路径，根目录就是 {slide}/{name}.png
                zf.write(png, arcname=f"{slide_dir.name}/{png.name}")
                count += 1
    return count, zip_path.stat().st_size


# --------------------------------------------------------------------------- #
# 5. 主流程
# --------------------------------------------------------------------------- #
def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="按官方规范生成 ToxPath 测试集提交掩膜",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--test-root", type=Path, required=True,
                   help="含 test_patches/ 的目录")
    p.add_argument("--ckpt", type=Path, default=None, help="单折 checkpoint")
    p.add_argument("--ckpt-dir", type=Path, default=None,
                   help="含 fold*/best.pt 的目录（自动全部集成）")
    p.add_argument("--out-dir", type=Path, default=PROJECT_ROOT / "submission",
                   help="提交目录（只会写 {slide}/{name}.png）")
    p.add_argument("--zip", type=Path, default=None,
                   help="额外打一个 zip（条目为 {slide}/{name}.png）")
    p.add_argument("--image-size", type=int, default=None,
                   help="覆盖模型输入尺寸（默认取 checkpoint 里记录的）")
    p.add_argument("--device", type=str, default="cuda",
                   help="cuda / cpu")
    p.add_argument("--fg-threshold", type=float, default=0.5,
                   help="肿瘤概率阈值")
    p.add_argument("--min-area", type=int, default=0,
                   help="去掉小于该面积的连通域（像素）；本地真实病灶最小 83 px")
    p.add_argument("--tta", action="store_true", help="hflip/vflip 翻转平均")
    p.add_argument("--max-patches", type=int, default=0,
                   help="只跑前 N 张（冒烟用，0 = 全跑）")
    p.add_argument("--skip-existing", action="store_true",
                   help="跳过已存在的掩膜（断点续跑）")
    p.add_argument("--log-dir", type=Path, default=PROJECT_ROOT / "runs3",
                   help="逐 patch 记录写这里（不进提交目录）")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    device: torch.device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        print("[warn] CUDA 不可用，退回 CPU（会非常慢）")
        device = torch.device("cpu")

    ckpts: list[Path] = resolve_ckpts(args.ckpt, args.ckpt_dir)
    print(f"[ckpt] {len(ckpts)} 个模型：" + ", ".join(p.parent.name + "/" + p.name
                                                    for p in ckpts))
    models, size = load_models(ckpts, device, args.image_size)
    print(f"[model] device={device}  image_size={size}  "
          f"tta={args.tta}  min_area={args.min_area}  thr={args.fg_threshold}")

    slides: dict[str, list[Path]] = discover_test_patches(args.test_root)
    total: int = sum(len(v) for v in slides.values())
    print(f"[data ] {len(slides)} 个 slide，共 {total} 张 patch")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    args.log_dir.mkdir(parents=True, exist_ok=True)
    log_path: Path = args.log_dir / "predict_test.jsonl"

    done: int = 0
    skipped: int = 0
    t0: float = time.time()
    with log_path.open("w", encoding="utf-8") as log_f:
        for slide, files in slides.items():
            for patch_path in files:
                if args.max_patches and done + skipped >= args.max_patches:
                    break
                out_path: Path = args.out_dir / slide / f"{patch_path.stem}.png"
                if args.skip_existing and out_path.is_file():
                    skipped += 1
                    continue

                with Image.open(patch_path) as im:
                    rgb_src: Image.Image = im.convert("RGB")
                    orig_w, orig_h = rgb_src.size
                    if rgb_src.size != (size, size):
                        rgb_src = rgb_src.resize((size, size), Image.BILINEAR)
                    rgb: np.ndarray = np.asarray(rgb_src, dtype=np.uint8)

                prob: np.ndarray = tumor_probability(models, rgb, device, tta=args.tta)

                # 概率图统一回到原图尺寸（官方要求尺寸完全一致）
                if (orig_w, orig_h) != (size, size):
                    prob_img = Image.fromarray(prob, mode="F").resize(
                        (orig_w, orig_h), Image.BILINEAR)
                    prob = np.asarray(prob_img, dtype=np.float32)

                mask: np.ndarray = prob >= args.fg_threshold
                if args.min_area > 1:
                    mask = remove_small_components(mask, args.min_area)

                save_mask(mask, out_path)
                done += 1

                log_f.write(json.dumps({
                    "slide": slide,
                    "name": patch_path.stem,
                    "src": str(patch_path),
                    "fg_px": int(mask.sum()),
                    "fg_ratio": round(float(mask.mean()), 6),
                    "max_prob": round(float(prob.max()), 6),
                }, ensure_ascii=False) + "\n")

                if done % 200 == 0:
                    elapsed: float = time.time() - t0
                    rate: float = done / max(elapsed, 1e-6)
                    remain: float = (total - done) / max(rate, 1e-6)
                    print(f"[prog ] {done}/{total}  {rate:.1f} 张/s  "
                          f"已用 {elapsed/60:.1f} min  预计剩余 {remain/60:.1f} min")
            if args.max_patches and done + skipped >= args.max_patches:
                break

    elapsed = time.time() - t0
    print(f"[done ] {done} 张已写，跳过 {skipped} 张，用时 {elapsed/60:.1f} min")
    print(f"[out  ] {args.out_dir}")
    print(f"[log  ] {log_path}")

    if args.zip is not None:
        n, nbytes = zip_submission(args.out_dir, args.zip)
        print(f"[zip  ] {n} 个条目，{nbytes/1024/1024:.2f} MB → {args.zip}")
        print("[zip  ] 条目形如 test_001/0_0.png（无外层目录），符合官方要求")

    print("\n人工核对（提交前必做）：")
    print(f"  1) {args.out_dir} 下每个 slide 的 PNG 数量 == 官方 test_patches 下该 slide 的图片数")
    print("  2) 文件名只能把 .jpeg 换成 .png，不加后缀")
    print("  3) 每张 PNG 是单通道 L、8-bit、只含 0/255、尺寸与原图一致")
    print("  如需自动自检脚本：git checkout 2ed284a -- scripts/check_submission.py")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
