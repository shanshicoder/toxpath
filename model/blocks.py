"""基础卷积算子。

与 HistoPLUS 源码的对应关系
---------------------------
====================  ============================================================
本项目                 HistoPLUS
====================  ============================================================
``ConvNormAct``       ``histoplus.helpers.nn.cellvit.decoder.Conv2DBlock``
``DeconvNormAct``     ``histoplus.helpers.nn.cellvit.decoder.Deconv2DBlock``
``interpolate_to``    ``histoplus.helpers.nn.cellvit.decoder.downscale_map``
====================  ============================================================

唯一的有意改动：归一化层可以换成 GroupNorm。
原因是组里的 MX350（2GB 显存）训练时 batch 只能开到 1~2，
BatchNorm 的滑动统计会很不稳；换成 GroupNorm 后小 batch 也能正常收敛。
默认仍是 ``bn``，与 HistoPLUS 权重、论文设置保持一致。
"""

from __future__ import annotations

from torch import Tensor, nn
from torch.nn import functional as F

from .config import NormLayerName


# --------------------------------------------------------------------------- #
# 归一化层工厂
# --------------------------------------------------------------------------- #
def build_norm(name: NormLayerName, num_channels: int) -> nn.Module:
    """按名称构造归一化层。

    Parameters
    ----------
    name:
        ``"bn"`` → :class:`torch.nn.BatchNorm2d`；``"gn"`` → :class:`torch.nn.GroupNorm`。
    num_channels:
        通道数。GroupNorm 要求通道数能被分组数整除，这里按 32 组自适应。

    Returns
    -------
    nn.Module
        未初始化的归一化层。
    """
    if name == "bn":
        return nn.BatchNorm2d(num_channels)
    if name == "gn":
        # 32 是常用的分组数；通道数小于 32 时退化为逐通道分组，保证整除。
        num_groups = 32 if num_channels % 32 == 0 else (8 if num_channels % 8 == 0 else 1)
        return nn.GroupNorm(num_groups=num_groups, num_channels=num_channels)
    raise ValueError(f"不支持的归一化类型: {name!r}，可选 'bn' / 'gn'")


# --------------------------------------------------------------------------- #
# 尺寸对齐
# --------------------------------------------------------------------------- #
def interpolate_to(x: Tensor, size: tuple[int, int]) -> Tensor:
    """把特征图双线性插值到目标空间尺寸（尺寸已一致时直接返回，零开销）。

    对应 HistoPLUS 的 ``downscale_map``：ViT patch=14 时 decoder 末端会放大到
    ``16 * H/14``，与输入边长不相等（例：784 → 896），必须缩回 ``H × W``
    才能与 ``n0``（原图分支）逐像素 concat。

    相比原实现的两个改进：

    1. 不写死 ``256 → 224``，而是直接对齐到调用方传入的尺寸，
       因此支持任意能被 14 整除的输入边长。
    2. 使用 ``antialias=True`` 做下采样，避免 896→784 产生摩尔纹，
       对 1 像素宽的组织边界更友好。

    Parameters
    ----------
    x:
        输入特征图，形状 ``[B, C, H_in, W_in]``。
    size:
        目标 ``(H_out, W_out)``。

    Returns
    -------
    Tensor
        形状 ``[B, C, H_out, W_out]``。
    """
    if tuple(x.shape[-2:]) == tuple(size):
        return x

    # antialias 只在下采样时有意义；torch >= 2.0 支持。
    is_downsample = x.shape[-2] > size[0] or x.shape[-1] > size[1]
    return F.interpolate(
        x,
        size=size,
        mode="bilinear",
        align_corners=False,
        antialias=is_downsample,
    )


# --------------------------------------------------------------------------- #
# 卷积块
# --------------------------------------------------------------------------- #
class ConvNormAct(nn.Module):
    """``Conv2d(stride=1) → Norm → ReLU → Dropout``。

    等价于 HistoPLUS 的 ``Conv2DBlock``：保持空间分辨率、只改变通道数。

    Parameters
    ----------
    in_channels:
        输入通道数。
    out_channels:
        输出通道数。
    kernel_size:
        卷积核边长，默认 3（same padding）。
    norm:
        归一化类型，默认 ``"bn"``。
    dropout:
        丢弃概率，默认 0。
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        norm: NormLayerName = "bn",
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        padding = (kernel_size - 1) // 2
        self.block = nn.Sequential(
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=kernel_size,
                stride=1,
                padding=padding,
                bias=True,
            ),
            build_norm(norm, out_channels),
            nn.ReLU(inplace=True),
            nn.Dropout2d(dropout) if dropout > 0 else nn.Identity(),
        )

    def forward(self, x: Tensor) -> Tensor:
        """前向。

        Parameters
        ----------
        x: ``[B, C_in, H, W]``

        Returns
        -------
        ``[B, C_out, H, W]``
        """
        return self.block(x)


class DeconvNormAct(nn.Module):
    """``ConvTranspose2d(k=2, s=2) → Conv3x3 → Norm → ReLU → Dropout``。

    等价于 HistoPLUS 的 ``Deconv2DBlock``：空间分辨率 ×2、通道数可任意改变。
    反卷积之后接一个 3×3 卷积是为了抑制 stride=2 反卷积典型的棋盘格伪影。

    Parameters
    ----------
    in_channels:
        输入通道数。
    out_channels:
        输出通道数。
    norm:
        归一化类型，默认 ``"bn"``。
    dropout:
        丢弃概率，默认 0。
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        norm: NormLayerName = "bn",
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.ConvTranspose2d(
                in_channels,
                out_channels,
                kernel_size=2,
                stride=2,
                padding=0,
                output_padding=0,
                bias=True,
            ),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=True),
            build_norm(norm, out_channels),
            nn.ReLU(inplace=True),
            nn.Dropout2d(dropout) if dropout > 0 else nn.Identity(),
        )

    def forward(self, x: Tensor) -> Tensor:
        """前向。

        Parameters
        ----------
        x: ``[B, C_in, H, W]``

        Returns
        -------
        ``[B, C_out, 2H, 2W]``
        """
        return self.block(x)


class FinalConv(nn.Module):
    """1×1 卷积输出每个像素的类别 logits。

    Parameters
    ----------
    in_channels:
        输入通道数。
    num_classes:
        输出类别数（本任务为 2）。
    """

    def __init__(self, in_channels: int, num_classes: int) -> None:
        super().__init__()
        self.proj = nn.Conv2d(in_channels, num_classes, kernel_size=1, stride=1, padding=0)

    def forward(self, x: Tensor) -> Tensor:
        """``[B, C_in, H, W]`` → ``[B, num_classes, H, W]``。"""
        return self.proj(x)


# --------------------------------------------------------------------------- #
# H0-mini 的 FFN：DINOv2 风格的 SwiGLU
# --------------------------------------------------------------------------- #
class SwiGLUFFNFused(nn.Module):
    """DINOv2 / H0-mini 使用的 SwiGLU 前馈网络。

    为什么不能用 timm 自带的 ``SwiGLU``
    ---------------------------------
    两个原因，任何一条都会让"加载 H0-mini 权重"失败或行为错误：

    1. **参数名不同**。DINOv2 的实现用 ``w12`` / ``w3``；
       timm 的 ``SwiGLU`` 用 ``fc1`` / ``fc2``。名字对不上 → 权重加载不上。
    2. **门控顺序不同**。DINOv2 是 ``SiLU(前半) * 后半``；
       timm 是 ``SiLU(后半) * 前半``。形状完全一致，加载时不会报错，
       但语义相反 —— 这是最典型的"静默错误"。

    ``hidden_features`` 按 DINOv2 的约定做 2/3 缩放并向上取 8 的倍数::

        hidden = (int(hidden * 2 / 3) + 7) // 8 * 8

    对齐依据：``histoplus/helpers/nn/ops.py::SwiGLUFFNFused``
    （那里继承 ``xformers.ops.SwiGLU``，数学上完全等价，
    本项目用纯 PyTorch 实现以摆脱 xformers 在 Windows 上的安装难题）。

    Parameters
    ----------
    in_features:
        输入维度。
    hidden_features:
        未缩放的隐藏维度；为 ``None`` 时取 ``in_features``。
    out_features:
        输出维度；为 ``None`` 时取 ``in_features``。
    act_layer:
        timm 的 Block 会把这个参数传进来，DINOv2 固定用 SiLU，因此忽略。
    drop:
        dropout 概率。
    bias:
        线性层是否带 bias。
    """

    def __init__(
        self,
        in_features: int,
        hidden_features: int | None = None,
        out_features: int | None = None,
        act_layer: type[nn.Module] | None = None,
        drop: float = 0.0,
        bias: bool = True,
        **_unused: object,
    ) -> None:
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        # DINOv2 的 2/3 缩放 + 8 对齐
        hidden_features = (int(hidden_features * 2 / 3) + 7) // 8 * 8

        self.w12 = nn.Linear(in_features, 2 * hidden_features, bias=bias)
        self.w3 = nn.Linear(hidden_features, out_features, bias=bias)
        self.drop = nn.Dropout(drop) if drop > 0 else nn.Identity()

    def forward(self, x: Tensor) -> Tensor:
        """``[..., in_features]`` → ``[..., out_features]``。"""
        x1, x2 = self.w12(x).chunk(2, dim=-1)
        return self.drop(self.w3(F.silu(x1) * x2))


__all__ = [
    "ConvNormAct",
    "SwiGLUFFNFused",
    "DeconvNormAct",
    "FinalConv",
    "build_norm",
    "interpolate_to",
]
