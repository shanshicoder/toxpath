"""ToxPath 模型的超参数配置。

设计原则
--------
1. 本模块**只依赖标准库**，不 import torch / timm，
   因此可以在任何环境下被导入、做参数校验、写进 YAML。
2. 所有配置项都是不可变（frozen）的，避免训练脚本中途意外改坏网络结构，
   导致 ``state_dict`` 与结构不一致。
3. 默认值直接对齐 HistoPLUS（Owkin）的 CellViT 主干设置，
   保证后续可以把 HistoPLUS 官方权重里的 H0-mini 编码器直接搬过来。

任务背景
--------
本项目做的是 **WSI 切片上的肿瘤区域二分类语义分割**：
输入一个 tile ``[B, 3, H, W]``，输出同尺寸的 logits ``[B, 2, H, W]``，
``channel 0 = 背景/非肿瘤``，``channel 1 = 肿瘤``。

与 HistoPLUS 原始任务的差别只有一处：
HistoPLUS 是"细胞实例分割"，有 3 个解码头（np / hv / tp，输出 2 / 2 / 15 通道）；
本项目是"区域语义分割"，只保留 **1 个解码头**，输出 2 通道。
编码器（H0-mini）与 Neck（四路 FPN）原封不动复用。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, TypeAlias

# --------------------------------------------------------------------------- #
# 类型别名
# --------------------------------------------------------------------------- #

NormLayerName: TypeAlias = Literal["bn", "gn"]
"""归一化层类型。

- ``"bn"``: BatchNorm2d，与 HistoPLUS 原实现完全一致（默认）。
- ``"gn"``: GroupNorm，batch_size 很小（如本地 2GB 显存只能跑 B=1）时更稳定。
"""


# --------------------------------------------------------------------------- #
# 配置
# --------------------------------------------------------------------------- #


@dataclass(frozen=True, slots=True)
class ToxPathConfig:
    """ToxPath 分割模型的完整结构配置。

    Attributes
    ----------
    num_classes:
        输出类别数。二分类区域分割为 ``2``；本项目当前的三分类任务为 ``3``。
    in_channels:
        输入图像通道数，H&E 染色 RGB 为 ``3``。
    image_size:
        训练 / 推理时 tile 的边长（正方形假设），**不要求**是 ``patch_size`` 的整数倍：
        不是整数倍时 ``ToxPathSegmentor.forward`` 会 reflect-pad 到最近的倍数再裁回。
        HistoPLUS 的推理 tile 是 ``784``（即 656 + 2×64 的 overlap 补边），
        训练 tile 依倍率取 224（20x）或 448（40x）；本项目当前用整张 ``1024``。
        **注意**：本参数只用于文档与 Self-Check，真正的空间尺寸由 forward 时
        实际输入的 ``H, W`` 决定（模型是全卷积 + 动态位置编码，支持变尺寸）。
    encoder_name:
        timm 模型名。``vit_base_patch14_reg4_dinov2.lvd142m`` 是 H0-mini
        的骨架定义（ViT-B/14 + 4 个 register token + DINOv2 预训练）。
    embed_dim:
        ViT 的 token 维度，ViT-B 为 ``768``。
    patch_size:
        ViT 的 patch 边长，H0-mini 为 ``14``。
    out_indices:
        从 ViT 的哪几个 block 抽特征，**timm 的 0-based 索引**。
        ``(3, 5, 7, 11)`` 表示取第 4、6、8、12 个 block 的输出。
        ⚠️ 易错点：HistoPLUS 里这些层被命名为 ``encoder_layer_3/5/7/11``
        （1-based 的**命名**），但传给 timm 的 ``out_indices`` 是 **0-based**，
        两处数字相同、含义不同。本配置沿用 HistoPLUS 的实际行为。
    neck_channels:
        Neck 四路输出通道，顺序为 ``(n0, n1, n2, n3)``，分别对应
        原图 / 1 次上采样 / 2 次 / 3 次上采样后的特征。
        与 HistoPLUS 一致：``(64, 128, 256, 512)``。
    bottleneck_dim:
        Neck 最深层与 decoder 瓶颈的通道数，HistoPLUS 取 ``512``。
    skip_dim_1 / skip_dim_2:
        Neck 中逐级降维的两个中间通道数，HistoPLUS 取 ``512`` / ``256``。
    dropout:
        Decoder / Neck 中每个卷积块后的 dropout 概率。
    norm:
        归一化层类型，见 :data:`NormLayerName`。
    pretrained:
        是否让 timm 自动下载 DINOv2 预训练权重。
        离线环境务必保持 ``False``，改用 :meth:`H0MiniEncoder.load_pretrained`
        加载 HistoPLUS 权重里的 H0-mini。
    freeze_encoder:
        是否冻结 H0-mini 编码器。低资源微调常用（只训 Neck + Decoder）。
    grad_checkpointing:
        是否对 ViT block 开启梯度检查点，用时间换显存（显存不足时打开）。
    """

    # ---- 任务相关 ----
    num_classes: int = 2
    in_channels: int = 3
    image_size: int = 784

    # ---- 编码器（H0-mini / ViT-B/14）----
    encoder_name: str = "vit_base_patch14_reg4_dinov2.lvd142m"
    embed_dim: int = 768
    patch_size: int = 14
    out_indices: tuple[int, ...] = (3, 5, 7, 11)

    # ---- Neck ----
    neck_channels: tuple[int, int, int, int] = (64, 128, 256, 512)
    bottleneck_dim: int = 512
    skip_dim_1: int = 512
    skip_dim_2: int = 256

    # ---- 正则化 / 训练策略 ----
    dropout: float = 0.0
    norm: NormLayerName = "bn"
    pretrained: bool = False
    freeze_encoder: bool = False
    grad_checkpointing: bool = False

    # ------------------------------------------------------------------ #
    # 构造时校验
    # ------------------------------------------------------------------ #
    def __post_init__(self) -> None:
        """在对象创建时立刻校验非法组合，避免训练到一半才崩。"""
        if self.num_classes < 2:
            raise ValueError(f"num_classes 至少为 2，当前为 {self.num_classes}")
        if self.in_channels != 3:
            raise ValueError(
                "H0-mini 的 patch_embed 权重是按 RGB 3 通道预训练的，"
                f"in_channels 必须为 3，当前为 {self.in_channels}"
            )
        if self.image_size < self.patch_size:
            raise ValueError(
                f"image_size({self.image_size}) 至少为 patch_size({self.patch_size})"
            )
        if len(self.out_indices) != 4:
            raise ValueError(
                f"out_indices 必须恰好 4 个（Neck 是四路），当前为 {self.out_indices}"
            )
        if list(self.out_indices) != sorted(self.out_indices):
            raise ValueError(f"out_indices 必须按从浅到深升序排列，当前为 {self.out_indices}")
        if len(self.neck_channels) != 4:
            raise ValueError(
                f"neck_channels 必须是 4 元组 (n0, n1, n2, n3)，当前为 {self.neck_channels}"
            )

    # ------------------------------------------------------------------ #
    # 便捷属性
    # ------------------------------------------------------------------ #
    @property
    def num_stages(self) -> int:
        """上采样 / 下采样级数。

        ViT 的 patch 化相当于一次 ``/patch_size`` 的下采样，
        decoder 再从 ``H/patch_size`` 逐级放大 ``2**4 = 16`` 倍，
        最终得到 ``16 * H/patch_size``。
        对 patch=14、H=784 即 ``16 * 56 = 896``，再插值回 ``784``。
        """
        return 4

    def to_dict(self) -> dict[str, object]:
        """导出为可序列化字典（写 YAML / 日志用）。"""
        from dataclasses import asdict

        return asdict(self)
