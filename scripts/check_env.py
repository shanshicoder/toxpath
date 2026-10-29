"""环境体检：一条命令看清这台机器能不能直接开训。

为什么需要它
------------
把项目拷到实验室电脑后，最容易出问题的不是模型代码，而是这些琐碎的东西：

- 装的是 CPU 版 torch（``cuda.is_available()`` 为 False，训练会慢 50 倍）；
- timm 版本不是 1.0.8（ViT block 实现可能变了，权重加载静默失配）；
- ``--data-root`` 指错目录，跑到一半才报 ``FileNotFoundError``；
- HistoPLUS 权重忘了拷（271 MB，最容易漏）；
- 显卡上还挂着别人的进程，剩不到 10 GB 显存，一开训就 OOM。

所以训练前先跑这个脚本，把上面这些都查一遍，并按显存给出推荐参数。

用法
----
.. code-block:: powershell

    # 在项目根目录，激活环境后
    python scripts/check_env.py

    # 顺便检查指定数据根 / 输出盘
    python scripts/check_env.py --data-root "D:/data/toxpath" --out-dir "D:/ckpt"

退出码
------
- ``0``：全部通过（可能有黄色提醒，不影响开训）；
- ``1``：存在阻塞项（CUDA 不可用、数据缺失、权重缺失等），先修再训。

输出用 ``[OK] / [WARN] / [FAIL]`` 三个纯 ASCII 标记，避免 Windows 默认
GBK 控制台打印 ``✓`` 时抛 ``UnicodeEncodeError``。
"""

from __future__ import annotations

import argparse
import csv
import shutil
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path

PROJECT_ROOT: Path = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from paths import (  # noqa: E402
    ENV_DATA_ROOT,
    ENV_HISTOPLUS_CKPT,
    PATCHES_DIR_NAME,
    WEIGHT_FILENAME,
    ckpt_candidates,
    data_root_candidates,
    is_valid_data_root,
    resolve_data_root,
    resolve_histoplus_ckpt,
)

# --------------------------------------------------------------------------- #
# 期望值（与 train_3class.py / requirements.txt 保持一致）
# --------------------------------------------------------------------------- #
EXPECT_MIN_FOLDS: int = 5
"""建议的交叉验证折数。

切片数少于它时 ``splits.build_slide_folds`` 会自动降低折数，
这里只做提示，不作为 FAIL。
"""

EXPECT_TIMM: str = "1.0.8"
"""HistoPLUS 钉死的 timm 版本。"""

MIN_VRAM_GB: float = 10.0
"""低于这个显存，1024 输入 + batch1 都很勉强。"""

WEIGHT_SIZE_MB: float = 271.0
"""HistoPLUS 权重约 271 MB，明显偏小说明没拷完整。"""


# --------------------------------------------------------------------------- #
# 结果收集
# --------------------------------------------------------------------------- #
@dataclass
class Item:
    """一条检查结果。"""

    name: str
    status: str  # "OK" | "WARN" | "FAIL"
    detail: str
    hint: str = ""


@dataclass
class Report:
    """全部检查结果。"""

    items: list[Item] = field(default_factory=list)

    def add(self, name: str, status: str, detail: str, hint: str = "") -> None:
        self.items.append(Item(name, status, detail, hint))

    @property
    def n_fail(self) -> int:
        return sum(1 for i in self.items if i.status == "FAIL")

    @property
    def n_warn(self) -> int:
        return sum(1 for i in self.items if i.status == "WARN")

    def render(self) -> str:
        """渲染成对齐的文本表格。"""
        w_name: int = max(len(i.name) for i in self.items) + 2
        lines: list[str] = []
        for it in self.items:
            head: str = f"[{it.status:<4}] {it.name:<{w_name}}{it.detail}"
            lines.append(head)
            if it.hint:
                lines.append(f"{'':10}{'':<{w_name}}> {it.hint}")
        return "\n".join(lines)


# --------------------------------------------------------------------------- #
# 单项检查
# --------------------------------------------------------------------------- #
def check_python(rep: Report) -> None:
    """Python 版本：3.10 最稳（HistoPLUS 原始环境），3.12+ 部分依赖轮子不全。"""
    v: str = ".".join(str(x) for x in sys.version_info[:3])
    exe: str = sys.executable
    major, minor = sys.version_info[0], sys.version_info[1]
    if major == 3 and 10 <= minor <= 12:
        rep.add("Python", "OK", f"{v}  ({exe})")
    elif major == 3 and minor >= 13:
        rep.add("Python", "WARN", f"{v}  ({exe})",
                "3.13 上 torch/timm 轮子可能不全，建议 conda 建 python=3.10 环境")
    else:
        rep.add("Python", "WARN", f"{v}  ({exe})", "建议 3.10")


def check_packages(rep: Report) -> tuple[str | None, bool]:
    """torch / timm / numpy / Pillow 版本，以及 CUDA 是否可用。

    Returns:
        ``(torch 版本字符串或 None, cuda 是否可用)``。
    """
    try:
        import torch
    except Exception as exc:  # noqa: BLE001 - 任何导入异常都算阻塞
        rep.add("torch", "FAIL", f"导入失败：{exc}",
                "conda activate 你的环境后 pip install torch torchvision（cu124）")
        rep.add("CUDA", "FAIL", "无法检测（torch 没装上）")
        return None, False

    torch_ver: str = torch.__version__
    cu_ver: str | None = torch.version.cuda
    if cu_ver is None:
        rep.add("torch", "FAIL", f"{torch_ver}（CPU 版）",
                "装的是 CPU 版，训练会慢 50 倍以上；重装 cu124 版本")
    else:
        rep.add("torch", "OK", f"{torch_ver}  (cuda {cu_ver})")

    try:
        import torchvision
        rep.add("torchvision", "OK", torchvision.__version__)
    except Exception as exc:  # noqa: BLE001
        rep.add("torchvision", "WARN", f"导入失败：{exc}")

    try:
        import timm
        tv: str = getattr(timm, "__version__", "?")
        if tv == EXPECT_TIMM:
            rep.add("timm", "OK", tv)
        else:
            rep.add("timm", "FAIL", f"{tv}（要求 {EXPECT_TIMM}）",
                    f"pip install timm=={EXPECT_TIMM}，版本不对会导致 ViT 权重静默失配")
    except Exception as exc:  # noqa: BLE001
        rep.add("timm", "FAIL", f"导入失败：{exc}", f"pip install timm=={EXPECT_TIMM}")

    for mod in ("numpy", "PIL"):
        try:
            m = __import__(mod)
            rep.add(mod, "OK", getattr(m, "__version__", "?"))
        except Exception as exc:  # noqa: BLE001
            rep.add(mod, "FAIL", f"导入失败：{exc}")

    cuda_ok: bool = bool(torch.cuda.is_available())
    return torch_ver, cuda_ok


def check_gpu(rep: Report, cuda_ok: bool) -> float:
    """GPU 型号 / 显存 / 当前占用。

    Args:
        rep: 结果容器。
        cuda_ok: ``torch.cuda.is_available()``。

    Returns:
        可用显存（GB）；无法检测时返回 0.0。
    """
    if not cuda_ok:
        rep.add("GPU", "FAIL", "torch 看不到 GPU",
                "① 确认装的是 CUDA 版 torch；② nvidia-smi 能跑；③ 重装驱动后重启")
        return 0.0

    import torch

    props = torch.cuda.get_device_properties(0)
    total_gb: float = props.total_memory / 1024 ** 3
    rep.add("GPU", "OK", f"{props.name}  {total_gb:.1f} GB  "
                         f"SM {props.major}.{props.minor}  {props.multi_processor_count} SM")

    # 当前占用（别人可能正在用这张卡）
    try:
        free_b, _total_b = torch.cuda.mem_get_info(0)
        free_gb: float = free_b / 1024 ** 3
        used_gb: float = total_gb - free_gb
        if free_gb < MIN_VRAM_GB:
            rep.add("显存", "FAIL", f"已用 {used_gb:.1f} GB，仅剩 {free_gb:.1f} GB",
                    "这张卡上有别的进程，nvidia-smi 看 PID 后等它结束或换卡")
        else:
            rep.add("显存", "OK", f"已用 {used_gb:.1f} GB，可用 {free_gb:.1f} GB")
    except Exception as exc:  # noqa: BLE001
        free_gb = total_gb
        rep.add("显存", "WARN", f"查询占用失败：{exc}")

    # 驱动版本（可选，nvidia-smi 不存在就算了）
    if shutil.which("nvidia-smi"):
        try:
            out: str = subprocess.run(
                ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
                capture_output=True, text=True, timeout=10,
            ).stdout.strip().splitlines()[0]
            rep.add("驱动", "OK", out)
        except Exception:  # noqa: BLE001
            pass

    if total_gb < MIN_VRAM_GB:
        rep.add("显存容量", "WARN", f"{total_gb:.1f} GB 偏小",
                "需要 --grad-checkpointing + --freeze-encoder，或降到 --image-size 512")
    return total_gb


def check_data(rep: Report, data_root: Path | None) -> None:
    """数据三件套：patch、三分类 mask、索引 CSV。"""
    if data_root is None:
        cands: str = "\n".join(f"      - {c}" for c in data_root_candidates())
        rep.add("数据根", "FAIL", "未找到", f"试过：\n{cands}\n"
                f"      修复：--data-root 指定，或 setx {ENV_DATA_ROOT} \"D:/data/toxpath\"")
        return

    rep.add("数据根", "OK", str(data_root))

    patches: Path = data_root / PATCHES_DIR_NAME
    masks: Path = data_root / "train_masks_classes"
    csv_path: Path = data_root / "_patch_labels_3class.csv"
    binary_masks: Path = data_root / "train_masks"

    def _count_files(d: Path) -> int:
        """递归统计目录下的图片文件（数据是 train_patches/<slide>/<level>/*.jpeg 三层）。"""
        if not d.is_dir():
            return -1
        return sum(1 for p in d.rglob("*")
                   if p.is_file() and p.suffix.lower() in {".jpg", ".jpeg", ".png", ".tif", ".tiff"})

    n_patch: int = _count_files(patches)
    n_slide: int = (sum(1 for p in patches.iterdir() if p.is_dir())
                    if patches.is_dir() else 0)

    if n_patch < 0:
        rep.add("train_patches", "FAIL", "目录不存在", f"{patches}")
    elif n_patch == 0:
        rep.add("train_patches", "FAIL", "目录是空的", f"{patches}")
    else:
        # 不再和固定数字比：数据集换成 73 个 WSI 后 4953 就不再是基准
        rep.add("train_patches", "OK", f"{n_patch} 张 / {n_slide} 个切片")

    if n_slide > 0:
        if n_slide >= EXPECT_MIN_FOLDS:
            rep.add("切片数", "OK",
                    f"{n_slide} 个（可做 {EXPECT_MIN_FOLDS} 折，每折约 "
                    f"{n_slide // EXPECT_MIN_FOLDS} 个验证切片）")
        else:
            rep.add("切片数", "WARN", f"{n_slide} 个，少于 {EXPECT_MIN_FOLDS}",
                    "折数会自动降到切片数，每折只有 1 个验证切片，指标波动会很大")

    n_mask: int = _count_files(masks)
    if n_mask < 0:
        rep.add("train_masks_classes", "FAIL", "目录不存在",
                "先跑：python scripts/make_3class_masks.py（需要 train_masks/ 原始二值掩膜）")
    elif n_mask == n_patch:
        rep.add("train_masks_classes", "OK", f"{n_mask} 张（与 patch 数一致）")
    else:
        rep.add("train_masks_classes", "FAIL", f"{n_mask} 张，patch 有 {max(n_patch, 0)} 张",
                "mask 与 patch 数量必须一一对应，缺的用 make_3class_masks.py 重新生成")

    if csv_path.is_file():
        n_rows: int = 0
        counts: dict[str, int] = {"0": 0, "1": 0, "2": 0}
        with csv_path.open("r", encoding="utf-8", newline="") as f:
            for row in csv.DictReader(f):
                n_rows += 1
                lab: str | None = row.get("label")
                if lab in counts:
                    counts[lab] += 1

        if n_rows == max(n_patch, 0):
            rep.add("patch 索引 CSV", "OK", f"{n_rows} 行")
        else:
            rep.add("patch 索引 CSV", "WARN", f"{n_rows} 行，patch {max(n_patch, 0)} 张",
                    "索引与数据不一致时采样权重会偏差；建议重跑 make_3class_masks.py")

        if n_rows > 0:
            share = {k: v / n_rows for k, v in counts.items()}
            detail: str = (f"背景 {counts['0']} ({share['0']:.1%})  "
                           f"少量 {counts['1']} ({share['1']:.1%})  "
                           f"较多 {counts['2']} ({share['2']:.1%})")
            # 类别 1 太稀缺会导致该类的 dice 波动大；换数据集后务必看这一行
            if share["1"] < 0.005 or share["2"] < 0.01:
                rep.add("类别分布", "WARN", detail,
                        "某个类别不到 1%，dice 会很不稳；考虑放宽 make_3class_masks 的阈值")
            else:
                rep.add("类别分布", "OK", detail)
            rep.add("CE 类别权重", "INFO",
                    "当前默认 (0.20, 0.50, 0.30)",
                    "换成新数据集后按上面的占比重估：越稀缺的类给越大权重")
    else:
        rep.add("patch 索引 CSV", "FAIL", "文件不存在", str(csv_path))

    if binary_masks.is_dir():
        rep.add("train_masks", "OK", f"{_count_files(binary_masks)} 张二值掩膜（原始标注，训练不读）")
    else:
        rep.add("train_masks", "WARN", "缺失（训练不需要，但重新生成三分类 mask 时需要）")


def check_weights(rep: Report, ckpt: Path | None) -> None:
    """HistoPLUS 预训练权重。"""
    if ckpt is None:
        cands: str = "\n".join(f"      - {c}" for c in ckpt_candidates())
        rep.add("HistoPLUS 权重", "FAIL", "未找到", f"试过：\n{cands}\n"
                f"      修复：拷到 <项目根>/weights/{WEIGHT_FILENAME}，"
                f"或 setx {ENV_HISTOPLUS_CKPT} \"D:/weights/{WEIGHT_FILENAME}\"")
        return

    size_mb: float = ckpt.stat().st_size / 1024 ** 2
    if size_mb < WEIGHT_SIZE_MB * 0.9:
        rep.add("HistoPLUS 权重", "WARN", f"{ckpt}  ({size_mb:.0f} MB)",
                f"正常约 {WEIGHT_SIZE_MB:.0f} MB，偏小可能没拷完")
    else:
        rep.add("HistoPLUS 权重", "OK", f"{size_mb:.0f} MB  {ckpt}")


def check_disk(rep: Report, out_dir: Path) -> None:
    """输出盘剩余空间：每折 best+last 约 0.9 GB，5 折约 4.6 GB。"""
    anchor: Path = out_dir
    while not anchor.exists() and anchor != anchor.parent:
        anchor = anchor.parent
    try:
        usage = shutil.disk_usage(anchor)
        free_gb: float = usage.free / 1024 ** 3
        if free_gb < 8.0:
            rep.add("磁盘", "WARN", f"{anchor.drive or anchor} 剩余 {free_gb:.1f} GB",
                    "5 折约需 4.6 GB checkpoint + 日志，建议留 10 GB 以上")
        else:
            rep.add("磁盘", "OK", f"{anchor.drive or anchor} 剩余 {free_gb:.1f} GB")
    except Exception as exc:  # noqa: BLE001
        rep.add("磁盘", "WARN", f"无法查询：{exc}")


def check_repo(rep: Report) -> None:
    """代码完整性：关键文件在不在。"""
    required: list[Path] = [
        PROJECT_ROOT / "train_3class.py",
        PROJECT_ROOT / "evaluate.py",
        PROJECT_ROOT / "inference.py",
        PROJECT_ROOT / "paths.py",
        PROJECT_ROOT / "model" / "toxpath.py",
        PROJECT_ROOT / "dataset" / "toxpath3_dataset.py",
        PROJECT_ROOT / "tests" / "test_dataset.py",
        PROJECT_ROOT / "tests" / "test_model_forward.py",
    ]
    missing: list[str] = [p.name for p in required if not p.is_file()]
    if missing:
        rep.add("代码文件", "FAIL", f"缺 {len(missing)} 个：{', '.join(missing)}",
                "代码没拷全，重新打包（scripts/make_code_package.ps1）")
    else:
        rep.add("代码文件", "OK", f"{len(required)} 个关键文件齐全")


# --------------------------------------------------------------------------- #
# 参数推荐
# --------------------------------------------------------------------------- #
def recommend(vram_gb: float) -> list[str]:
    """按显存给出推荐训练参数。

    Args:
        vram_gb: 显卡总显存（GB）。

    Returns:
        建议命令行的若干行文本。
    """
    if vram_gb >= 20:
        return [
            "显存 >= 20 GB：1024 输入，batch 1 + 梯度累积 8，开 AMP(bf16)",
            "  --batch-size 1 --grad-accum 8 --samples-per-epoch 800 "
            "--val-every 2 --amp --grad-checkpointing --num-workers 4",
            "  预期峰值显存 12~16 GB，单折 20 epoch 约 2 小时",
        ]
    if vram_gb >= 14:
        return [
            "显存 14~20 GB：必须开梯度检查点，batch 1",
            "  --batch-size 1 --grad-accum 8 --amp --grad-checkpointing --num-workers 4",
            "  OOM 时再降 --image-size 768",
        ]
    return [
        "显存 < 14 GB：1024 全尺寸很勉强，按下面顺序降级",
        "  ① --batch-size 1 --grad-accum 16 --amp --grad-checkpointing --freeze-encoder",
        "  ② 仍 OOM → --image-size 512（精度会掉，仅用于跑通管线）",
    ]


# --------------------------------------------------------------------------- #
# 主流程
# --------------------------------------------------------------------------- #
def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description="ToxPath 训练环境体检（先跑这个，再开训）",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    ap.add_argument("--data-root", type=Path, default=None,
                    help="数据根；不填则按 paths.py 自动解析")
    ap.add_argument("--histoplus-ckpt", type=Path, default=None,
                    help="HistoPLUS 权重；不填则自动解析")
    ap.add_argument("--out-dir", type=Path, default=PROJECT_ROOT / "checkpoints3",
                    help="训练输出目录（用来检查磁盘空间）")
    args = ap.parse_args(argv)

    print("=" * 78)
    print(" ToxPath 环境体检")
    print("=" * 78)

    rep: Report = Report()

    check_repo(rep)
    check_python(rep)
    _torch_ver, cuda_ok = check_packages(rep)
    vram_gb: float = check_gpu(rep, cuda_ok)

    try:
        data_root: Path | None = resolve_data_root(args.data_root)
    except FileNotFoundError:
        data_root = None
    try:
        ckpt: Path | None = resolve_histoplus_ckpt(args.histoplus_ckpt)
    except FileNotFoundError:
        ckpt = None

    check_data(rep, data_root)
    check_weights(rep, ckpt)
    check_disk(rep, args.out_dir)

    print(rep.render())
    print("-" * 78)

    if vram_gb > 0:
        print(" 推荐参数：")
        for line in recommend(vram_gb):
            print(f"   {line}")
        print("-" * 78)

    if rep.n_fail == 0:
        print(f" 结论：可以开训（{rep.n_warn} 条提醒不影响运行）")
        if data_root is not None:
            print(" 下一步：scripts\\train_local.bat -Mode smoke（先冒烟 32 张）")
        return 0

    print(f" 结论：有 {rep.n_fail} 个阻塞项、{rep.n_warn} 条提醒，先按上面的提示修")
    return 1


if __name__ == "__main__":
    sys.exit(main())
