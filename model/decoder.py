"""单头语义解码器：从 HistoPLUS 的三分支解码头改造而来。

原始 HistoPLUS 与改造后的差异
---------------------------
HistoPLUS 是**细胞实例分割**，共享同一个 Neck，但并联 3 个完全独立的
``DecoderBranch``（论文里称为 "isolated upsampling pathways"）：

==========  ===========  ==========================  ==========================
分支         输出通道      监督目标                     本项目是否需要
==========  ===========  ==========================  ==========================
``np``       2           细胞核像素二分类               否
``hv``       2           像素到细胞核中心的水平/垂直距离  否
``tp``       15          细胞类型（15 类）              否
==========  ===========  ==========================  ==========================

本项目只做**区域级二分类语义分割**（背景 / 肿瘤），因此只保留 1 条通路，
把最后的 1×1 卷积从 ``15`` 通道改成 ``2`` 通道。
其余结构（每级 ``conv-conv-deconv`` 的通道变化 512→256→128→64、
以及和 Neck 的 concat 顺序）完全保持 HistoPLUS 原样。

形状账本（以 784 tile、patch=14 为例，``h = 784/14 = 56``）
---------------------------------------------------------
::

    z4              [B, 768, 56, 56]      ViT 最深特征
      ↓ ConvT(k2,s2)
    b4              [B, 512, 112, 112]
      ↓ cat n3(512) → 1024 → 3×(conv) → ConvT
    b3              [B, 256, 224, 224]
      ↓ cat n2(256) → 512  → 2×(conv) → ConvT
    b2              [B, 128, 448, 448]
      ↓ cat n1(128) → 256  → 2×(conv) → ConvT
    b1              [B, 64, 896, 896]
      ↓ 插值对齐到输入尺寸（896 → 784）
    b1              [B, 64, 784, 784]
      ↓ cat n0(64)  → 128  → 2×(conv)
    logits          [B, 2, 784, 784]

注意 ``896 ≠ 784`` 是这个主干最反直觉的地方：ViT 的 patch=14 导致
"16 倍上采样"落到 ``16 × 56 = 896``，而输入是 ``784``，必须缩回去。
HistoPLUS 用 ``downscale_map(256, 224)`` 处理（那是 224 tile 的情形），
本项目改成"对齐到实际输入尺寸"，因此 224 / 448 / 784 都能跑。
"""

from __future__ import annotations

from typing import Sequence

import torch
from torch import Tensor, nn

from .blocks import ConvNormAct, FinalConv, interpolate_to
from .config import ToxPathConfig


class SemanticDecoder(nn.Module):
    """把 Neck 特征逐级融合上采样成逐像素类别 logits。

    Parameters
    ----------
    cfg:
        模型配置，通道数来自 ``cfg.embed_dim`` / ``cfg.bottleneck_dim`` /
        ``cfg.neck_channels``。

    Notes
    -----
    融合顺序与 HistoPLUS 一致：**先 concat 再卷积**（而非相加），
    因为 ViT 特征与 Neck 特征的通道语义不同，concat 让卷积自己学权重，
    在小数据微调时比相加更稳。
    """

    def __init__(self, cfg: ToxPathConfig) -> None:
        super().__init__()
        self.cfg = cfg

        c0, c1, c2, _c3 = cfg.neck_channels   # (64, 128, 256, 512)
        bd = cfg.bottleneck_dim               # 512
        embed = cfg.embed_dim                 # 768
        dropout = cfg.dropout
        norm = cfg.norm

        # ---- 1) ViT 最深特征 768 → 512，同时 ×2 上采样 ----
        self.bottleneck_upsampler: nn.ConvTranspose2d = nn.ConvTranspose2d(
            in_channels=embed,
            out_channels=bd,
            kernel_size=2,
            stride=2,
            padding=0,
            output_padding=0,
        )

        # ---- 2) 融合 n3（512）→ 256，×2 ----
        self.decoder3_upsampler: nn.Sequential = nn.Sequential(
            ConvNormAct(bd * 2, bd, norm=norm, dropout=dropout),
            ConvNormAct(bd, bd, norm=norm, dropout=dropout),
            ConvNormAct(bd, bd, norm=norm, dropout=dropout),
            nn.ConvTranspose2d(bd, c2, kernel_size=2, stride=2),
        )

        # ---- 3) 融合 n2（256）→ 128，×2 ----
        self.decoder2_upsampler: nn.Sequential = nn.Sequential(
            ConvNormAct(c2 * 2, c2, norm=norm, dropout=dropout),
            ConvNormAct(c2, c2, norm=norm, dropout=dropout),
            nn.ConvTranspose2d(c2, c1, kernel_size=2, stride=2),
        )

        # ---- 4) 融合 n1（128）→ 64，×2 ----
        self.decoder1_upsampler: nn.Sequential = nn.Sequential(
            ConvNormAct(c1 * 2, c1, norm=norm, dropout=dropout),
            ConvNormAct(c1, c1, norm=norm, dropout=dropout),
            nn.ConvTranspose2d(c1, c0, kernel_size=2, stride=2),
        )

        # ---- 5) 融合 n0（原始分辨率）→ 类别 logits ----
        self.decoder0_header: nn.Sequential = nn.Sequential(
            ConvNormAct(c0 * 2, c0, norm=norm, dropout=dropout),
            ConvNormAct(c0, c0, norm=norm, dropout=dropout),
            FinalConv(c0, cfg.num_classes),
        )

    # ------------------------------------------------------------------ #
    def forward(
        self,
        z4: Tensor,
        neck_features: Sequence[Tensor],
        out_size: tuple[int, int] | None = None,
    ) -> Tensor:
        """前向。

        Parameters
        ----------
        z4:
            ViT 最深一层特征 ``[B, 768, h, w]``。
        neck_features:
            Neck 输出 ``[n0, n1, n2, n3]``，
            形状 ``[B,64,4h,4w] / [B,128,2h,2w] / [B,256,·] / [B,512,·]`` 之一组
            （具体见 :class:`model.neck.HistoPlusNeck`）。
        out_size:
            输出空间尺寸 ``(H, W)``。默认为 ``(16h, 16w)``，即不做最后的尺寸对齐。

        Returns
        -------
        Tensor
            ``[B, num_classes, H, W]`` 的 logits（**未过 softmax**，
            训练时直接喂给 ``CrossEntropyLoss``）。

        Raises
        ------
        ValueError
            Neck 特征层数不为 4 时。
        """
        if len(neck_features) != 4:
            raise ValueError(f"Decoder 需要 4 层 Neck 特征，当前收到 {len(neck_features)} 层")

        n0, n1, n2, n3 = neck_features

        # 1) bottleneck：768 → 512，×2
        b4 = self.bottleneck_upsampler(z4)

        # 2) 与 n3 融合：1024 → 512 → 256，×2
        b3 = self.decoder3_upsampler(torch.cat([n3, b4], dim=1))

        # 3) 与 n2 融合：512 → 256 → 128，×2
        b2 = self.decoder2_upsampler(torch.cat([n2, b3], dim=1))

        # 4) 与 n1 融合：256 → 128 → 64，×2
        b1 = self.decoder1_upsampler(torch.cat([n1, b2], dim=1))

        # 5) 尺寸对齐（patch=14 时 16h ≠ H），再与 n0（原图分支）融合
        if out_size is not None:
            b1 = interpolate_to(b1, out_size)
        logits = self.decoder0_header(torch.cat([n0, b1], dim=1))

        return logits


__all__ = ["SemanticDecoder"]
