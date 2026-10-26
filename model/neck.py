"""HistoPLUS / CellViT 的四路并行 Neck（FPN 式特征金字塔）。

为什么要"四路并行"而不是"串行 FPN"
---------------------------------
ViT 的 4 层特征**分辨率完全相同**（都是 ``H/14 × W/14``，因为 block 之间不下采样），
所以没法像 CNN 那样"由深到浅逐级上采样再相加"。
HistoPLUS 的做法是：每一层特征各自走**独立的一条反卷积链**，
把分辨率分别放大 2× / 4× / 8×，人为制造出多尺度：

===================  ==========  ===========================  ======================
分支                 输入        操作                          输出
===================  ==========  ===========================  ======================
``neck0``            原图        2 个 3×3 Conv（不下采样）      ``[B, 64, H, W]``
``neck1``            f0（最浅）  3 次反卷积（×8）               ``[B, 128, H/2, W/2]``
``neck2``            f1          2 次反卷积（×4）               ``[B, 256, H/4, W/4]``
``neck3``            f2          1 次反卷积（×2）               ``[B, 512, H/8, W/8]``
===================  ==========  ===========================  ======================

注：``neck0`` 吃的是**原始 RGB 图像**而不是 ViT 特征，
这是为了把高频的细胞核边界信息直接送到解码器末端（类似 U-Net 的第一层 skip）。
最深的一层 ``f3`` **不经过 Neck**，直接进 decoder 的 bottleneck 上采样路径。

上表的 "H/2、H/4、H/8" 是 HistoPLUS 文档里按 patch=16 写的说法；
换成 patch=14 的 H0-mini 后，实际输出边长是 ``8h / 4h / 2h``（``h = H/14``）。
以 784 tile 为例，``h = 56``，于是 ``n1 / n2 / n3 = 448 / 224 / 112``。
（Decoder 的第 5 步会再把 ``n1`` 上采样到 ``16h = 896``，最后插值回 784。）
"""

from __future__ import annotations

from typing import Sequence

from torch import Tensor, nn

from .blocks import ConvNormAct, DeconvNormAct
from .config import ToxPathConfig


class HistoPlusNeck(nn.Module):
    """四路并行 Neck，输出 ``n = [n0, n1, n2, n3]``。

    Parameters
    ----------
    cfg:
        模型配置。通道数取自 ``cfg.neck_channels``、``cfg.skip_dim_1``、
        ``cfg.skip_dim_2``、``cfg.bottleneck_dim``。

    Notes
    -----
    输入特征必须是**从浅到深**排列的 4 个张量，且空间尺寸相同。
    这个顺序由 :class:`model.encoder.H0MiniEncoder` 的 ``out_indices`` 保证
    （timm 的 ``features_only`` 按 ``out_indices`` 升序返回）。
    """

    def __init__(self, cfg: ToxPathConfig) -> None:
        super().__init__()
        self.cfg = cfg

        c0, c1, c2, _c3 = cfg.neck_channels
        embed = cfg.embed_dim
        s1, s2 = cfg.skip_dim_1, cfg.skip_dim_2
        bottleneck = cfg.bottleneck_dim
        dropout = cfg.dropout
        norm = cfg.norm

        # ---- neck0：原始图像 → 高分辨率浅层特征（保持 H×W） ----
        neck0 = nn.Sequential(
            ConvNormAct(cfg.in_channels, 32, kernel_size=3, norm=norm, dropout=dropout),
            ConvNormAct(32, c0, kernel_size=3, norm=norm, dropout=dropout),
        )

        # ---- neck1：最浅 ViT 层 → ×8 上采样 ----
        neck1 = nn.Sequential(
            DeconvNormAct(embed, s1, norm=norm, dropout=dropout),
            DeconvNormAct(s1, s2, norm=norm, dropout=dropout),
            DeconvNormAct(s2, c1, norm=norm, dropout=dropout),
        )

        # ---- neck2：第二浅 ViT 层 → ×4 上采样 ----
        neck2 = nn.Sequential(
            DeconvNormAct(embed, s1, norm=norm, dropout=dropout),
            DeconvNormAct(s1, c2, norm=norm, dropout=dropout),
        )

        # ---- neck3：第三浅 ViT 层 → ×2 上采样（最深一层不走这里）----
        neck3 = nn.Sequential(
            DeconvNormAct(embed, bottleneck, norm=norm, dropout=dropout),
        )

        # 用 ModuleList 而不是 4 个属性，方便按索引循环、与 HistoPLUS 的
        # ``self.necks = nn.ModuleList([neck0, neck1, neck2, neck3])`` 对应。
        self.necks: nn.ModuleList = nn.ModuleList([neck0, neck1, neck2, neck3])

    # ------------------------------------------------------------------ #
    def forward(self, image: Tensor, features: Sequence[Tensor]) -> list[Tensor]:
        """前向。

        Parameters
        ----------
        image:
            原始输入 ``[B, 3, H, W]``（是否归一化不影响形状，但必须与训练一致）。
        features:
            编码器输出的 4 层特征，``[f0, f1, f2, f3]``，每层 ``[B, 768, h, w]``，
            ``h = H // 14``。

        Returns
        -------
        list[Tensor]
            ``[n0, n1, n2, n3]``，形状依次为

            - ``n0: [B, 64, H, W]``          ← 原图分支，不下采样
            - ``n1: [B, 128, 8h, 8w]``       ← 最浅层 ×8
            - ``n2: [B, 256, 4h, 4w]``       ← 第二浅 ×4
            - ``n3: [B, 512, 2h, 2w]``       ← 第三浅 ×2

        Raises
        ------
        ValueError
            特征层数不等于 4，或 4 层特征的空间尺寸不一致时。
        """
        if len(features) != 4:
            raise ValueError(f"Neck 需要恰好 4 层特征，当前收到 {len(features)} 层")

        spatial = {tuple(int(s) for s in f.shape[-2:]) for f in features}
        if len(spatial) != 1:
            raise ValueError(f"4 层 ViT 特征的空间尺寸应保持一致，当前为 {sorted(spatial)}")

        # 与 HistoPLUS 完全一致：z = [image, *features]，逐层送进对应分支。
        # features 有 4 层、image 占 z[0]，所以这里只用到 z[0..3]；
        # 原实现在调用侧写 self.neck(z[:-1]) 丢掉最深的 z[4]，效果相同。
        z: list[Tensor] = [image, *features]

        outputs: list[Tensor] = []
        for branch_index in range(4):
            outputs.append(self.necks[branch_index](z[branch_index]))
        return outputs


__all__ = ["HistoPlusNeck"]
