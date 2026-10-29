"""统一的路径解析：让同一份代码在笔记本和实验室电脑上都能直接跑。

背景
----
代码要在至少两台机器上运行，两边的盘符、目录结构都不一样：

==================  ==========================================
机器                数据位置
==================  ==========================================
笔记本（CPU 冒烟）  ``E:\\project\\toxpath\\data\\toxpath``
实验室（RTX 4090）  任意位置，例如 ``D:\\data\\toxpath``
==================  ==========================================

如果把 ``E:/project/...`` 写死在代码里，换台机器就要改源码。所以这里把
“数据在哪、权重在哪”收敛到一处，按下面的优先级依次尝试：

**数据根** ``resolve_data_root``

1. 命令行 ``--data-root``（最高优先级，永远覆盖其它一切）
2. 环境变量 ``TOXPATH_DATA_ROOT``
3. 项目内相对路径 ``<PROJECT_ROOT>/data/toxpath``（推荐布局，拷项目时带上 data 即可）
4. 项目同级 ``<PROJECT_ROOT>/../data/toxpath``
5. 历史绝对路径 ``E:/project/toxpath/data/toxpath``（笔记本上的老位置）

**HistoPLUS 权重** ``resolve_histoplus_ckpt``

1. 命令行 ``--histoplus-ckpt``
2. 环境变量 ``TOXPATH_HISTOPLUS_CKPT``
3. ``<PROJECT_ROOT>/weights/histoplus_cellvit_segmentor_20x.pt``（推荐布局）
4. ``<PROJECT_ROOT>/histoplus_cellvit_segmentor_20x.pt``
5. ``<PROJECT_ROOT>/../histoplus_cellvit_segmentor_20x.pt``（笔记本上的老位置）
6. 历史绝对路径 ``E:/project/histoplus_cellvit_segmentor_20x.pt``

找不到时抛 :class:`FileNotFoundError`，消息里会列出所有试过的候选路径和
修复办法，而不是抛一个没有上下文的报错。

用法
----
.. code-block:: python

    from paths import resolve_data_root, resolve_histoplus_ckpt, describe_paths

    data_root: Path = resolve_data_root()          # 走默认优先级
    ckpt: Path = resolve_histoplus_ckpt("D:/w/histoplus.pt")   # 显式指定

命令行临时覆盖（不设环境变量、不改代码）::

    python train_3class.py --data-root D:/data/toxpath --histoplus-ckpt D:/w/histoplus.pt

长期设置（Windows，一次即可，新开终端生效）::

    setx TOXPATH_DATA_ROOT "D:/data/toxpath"
    setx TOXPATH_HISTOPLUS_CKPT "D:/weights/histoplus_cellvit_segmentor_20x.pt"
"""

from __future__ import annotations

import os
from pathlib import Path

PROJECT_ROOT: Path = Path(__file__).resolve().parent
"""项目根目录（本文件所在目录）。所有相对路径都以它为基准，与当前工作目录无关。"""

ENV_DATA_ROOT: str = "TOXPATH_DATA_ROOT"
"""指定数据根的环境变量名。"""

ENV_HISTOPLUS_CKPT: str = "TOXPATH_HISTOPLUS_CKPT"
"""指定 HistoPLUS 权重的环境变量名。"""

DATA_DIR_NAME: str = "toxpath"
"""数据根在 ``data/`` 下的目录名，即 ``data/toxpath``。"""

WEIGHT_FILENAME: str = "histoplus_cellvit_segmentor_20x.pt"
"""HistoPLUS 预训练权重的固定文件名。"""

PATCHES_DIR_NAME: str = "train_patches"
"""判定“这是一个合法数据根”的标志目录：1024×1024 H&E patch。"""

LEGACY_DATA_ROOT: Path = Path("E:/project/toxpath/data/toxpath")
LEGACY_CKPT: Path = Path("E:/project/histoplus_cellvit_segmentor_20x.pt")
"""笔记本上的历史位置，仅作为最后的兜底候选，不存在就跳过。"""


# --------------------------------------------------------------------------- #
# 候选列表
# --------------------------------------------------------------------------- #
def data_root_candidates() -> list[Path]:
    """返回所有会被尝试的数据根候选（按优先级排序，已去重）。

    顺序：环境变量 → 项目内 → 项目同级 → 历史绝对路径。

    Returns:
        候选路径列表；不保证都存在。
    """
    cands: list[Path] = []

    env_value: str | None = os.environ.get(ENV_DATA_ROOT)
    if env_value:
        cands.append(Path(env_value).expanduser())

    cands.append(PROJECT_ROOT / "data" / DATA_DIR_NAME)
    cands.append(PROJECT_ROOT.parent / "data" / DATA_DIR_NAME)
    cands.append(LEGACY_DATA_ROOT)

    seen: set[str] = set()
    unique: list[Path] = []
    for p in cands:
        key: str = str(p.resolve() if p.exists() else p).lower()
        if key not in seen:
            seen.add(key)
            unique.append(p)
    return unique


def ckpt_candidates() -> list[Path]:
    """返回所有会被尝试的 HistoPLUS 权重候选（按优先级排序，已去重）。

    Returns:
        候选文件路径列表；不保证都存在。
    """
    cands: list[Path] = []

    env_value: str | None = os.environ.get(ENV_HISTOPLUS_CKPT)
    if env_value:
        cands.append(Path(env_value).expanduser())

    cands.append(PROJECT_ROOT / "weights" / WEIGHT_FILENAME)
    cands.append(PROJECT_ROOT / WEIGHT_FILENAME)
    cands.append(PROJECT_ROOT.parent / WEIGHT_FILENAME)
    cands.append(LEGACY_CKPT)

    seen: set[str] = set()
    unique: list[Path] = []
    for p in cands:
        key: str = str(p.resolve() if p.exists() else p).lower()
        if key not in seen:
            seen.add(key)
            unique.append(p)
    return unique


# --------------------------------------------------------------------------- #
# 解析
# --------------------------------------------------------------------------- #
def is_valid_data_root(path: Path) -> bool:
    """判断目录是否是可用数据根：必须含 ``train_patches`` 子目录。

    Args:
        path: 待检查的目录。

    Returns:
        ``True`` 表示该目录可以直接喂给 Dataset。
    """
    return path.is_dir() and (path / PATCHES_DIR_NAME).is_dir()


def resolve_data_root(explicit: str | Path | None = None) -> Path:
    """确定数据根目录。

    Args:
        explicit: 命令行传入的 ``--data-root``；``None`` 表示走默认优先级。

    Returns:
        真实存在且含 ``train_patches`` 的数据根目录。

    Raises:
        FileNotFoundError: 显式路径无效，或所有候选都不存在。
    """
    if explicit is not None:
        p: Path = Path(explicit).expanduser()
        if is_valid_data_root(p):
            return p.resolve()
        raise FileNotFoundError(
            f"--data-root 指向的目录不可用：{p}\n"
            f"  要求：目录存在，且其下有 {PATCHES_DIR_NAME}/ 子目录。\n"
            f"  当前是否存在：{p.is_dir()}\n"
            f"  其下是否有 {PATCHES_DIR_NAME}：{(p / PATCHES_DIR_NAME).is_dir()}"
        )

    tried: list[Path] = data_root_candidates()
    for cand in tried:
        if is_valid_data_root(cand):
            return cand.resolve()

    lines: str = "\n".join(f"  - {c}" for c in tried)
    raise FileNotFoundError(
        "找不到数据根目录（需要包含 train_patches/ 的目录）。已尝试：\n"
        f"{lines}\n"
        "任选一种办法修复：\n"
        "  1) 命令行显式指定：--data-root \"D:/data/toxpath\"\n"
        f"  2) 设环境变量：setx {ENV_DATA_ROOT} \"D:/data/toxpath\"（重开终端生效）\n"
        "  3) 把数据放到 <项目根>/data/toxpath/ 下（最省事，换机器直接拷）"
    )


def resolve_histoplus_ckpt(explicit: str | Path | None = None) -> Path:
    """确定 HistoPLUS 预训练权重文件。

    Args:
        explicit: 命令行传入的 ``--histoplus-ckpt``；``None`` 表示走默认优先级。

    Returns:
        真实存在的权重文件路径。

    Raises:
        FileNotFoundError: 显式路径无效，或所有候选都不存在。
    """
    if explicit is not None:
        p: Path = Path(explicit).expanduser()
        if p.is_file():
            return p.resolve()
        raise FileNotFoundError(
            f"--histoplus-ckpt 指向的文件不存在：{p}\n"
            f"  确认把 {WEIGHT_FILENAME} 拷到了该路径（约 271 MB）。"
        )

    tried: list[Path] = ckpt_candidates()
    for cand in tried:
        if cand.is_file():
            return cand.resolve()

    lines: str = "\n".join(f"  - {c}" for c in tried)
    raise FileNotFoundError(
        f"找不到 HistoPLUS 权重 {WEIGHT_FILENAME}。已尝试：\n"
        f"{lines}\n"
        "任选一种办法修复：\n"
        "  1) 命令行显式指定：--histoplus-ckpt \"D:/weights/"
        f"{WEIGHT_FILENAME}\"\n"
        f"  2) 设环境变量：setx {ENV_HISTOPLUS_CKPT} \"D:/weights/{WEIGHT_FILENAME}\"\n"
        "  3) 放到 <项目根>/weights/ 下\n"
        "  4) 完全不用预训练权重：加 --init none（从零训练，不推荐）"
    )


# --------------------------------------------------------------------------- #
# 打印
# --------------------------------------------------------------------------- #
def describe_paths(
    data_root: str | Path | None = None,
    ckpt: str | Path | None = None,
) -> str:
    """生成一段人类可读的路径说明，用于训练开始时的日志。

    Args:
        data_root: 已解析的数据根；``None`` 则现场解析（失败也不抛错）。
        ckpt: 已解析的权重路径；``None`` 则现场解析（失败也不抛错）。

    Returns:
        多行字符串，每行一个条目。
    """
    def _try_data() -> str:
        try:
            return str(resolve_data_root(data_root))
        except FileNotFoundError:
            return "（未找到）"

    def _try_ckpt() -> str:
        try:
            return str(resolve_histoplus_ckpt(ckpt))
        except FileNotFoundError:
            return "（未找到）"

    return (
        f"  project_root : {PROJECT_ROOT}\n"
        f"  data_root    : {_try_data()}\n"
        f"  histoplus    : {_try_ckpt()}"
    )


if __name__ == "__main__":
    # 手动排查用：直接 python paths.py 看当前解析结果
    print("[paths] 当前解析：")
    print(describe_paths())
    print("\n[paths] 数据根候选：")
    for c in data_root_candidates():
        mark: str = "OK " if is_valid_data_root(c) else "-- "
        print(f"  {mark}{c}")
    print("\n[paths] 权重候选：")
    for c in ckpt_candidates():
        mark = "OK " if c.is_file() else "-- "
        print(f"  {mark}{c}")
