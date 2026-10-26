"""分割损失函数。

为什么用 CE + Dice 的组合
------------------------
WSI 里的肿瘤区域极度不平衡：一张 784×784 的 tile 里可能 95% 都是背景/间质，
纯交叉熵会被背景主导，模型倾向于"全预测背景"也能拿到很低 loss。
Dice loss 直接优化前景的 overlap，对稀疏前景更敏感；
两者相加（``loss = ce_weight * CE + dice_weight * Dice``）是病理分割的常用配方。

CE 侧支持 ``class_weight``（如 ``[0.2, 0.8]``）和 ``ignore_index``
（把切片外的白边 / 非组织区域标成 -100 后不参与梯度）。
"""

from __future__ import annotations

from typing import Sequence

import torch
from torch import Tensor, nn
from torch.nn import functional as F


class DiceCrossEntropyLoss(nn.Module):
    """交叉熵 + soft Dice 的联合损失。

    Parameters
    ----------
    num_classes:
        类别数，本任务为 2。
    ce_weight:
        交叉熵项的权重。
    dice_weight:
        Dice 项的权重。
    class_weights:
        交叉熵的类别权重，长度必须等于 ``num_classes``；
        ``None`` 表示等权。小数据微调时常用 ``[1.0, 2.0]`` 提升前景召回。
    ignore_index:
        被忽略的标签值，默认 ``-100``（PyTorch 的惯例）。
    smooth:
        Dice 的平滑项，避免分母为 0；对空 mask 的切片很重要。
    dice_include_background:
        Dice 是否把背景类也算进去。默认 ``False``，只优化前景，
        否则背景的 Dice 通常接近 1，会把前景的贡献稀释掉。

    Examples
    --------
    >>> import torch
    >>> from model.losses import DiceCrossEntropyLoss
    >>> crit = DiceCrossEntropyLoss(num_classes=2)
    >>> logits = torch.randn(1, 2, 32, 32)
    >>> target = torch.randint(0, 2, (1, 32, 32))
    >>> out = crit(logits, target)
    >>> sorted(out)
    ['ce', 'dice', 'loss']
    """

    def __init__(
        self,
        num_classes: int = 2,
        ce_weight: float = 1.0,
        dice_weight: float = 1.0,
        class_weights: Sequence[float] | None = None,
        ignore_index: int = -100,
        smooth: float = 1.0,
        dice_include_background: bool = False,
    ) -> None:
        super().__init__()
        if num_classes < 2:
            raise ValueError(f"num_classes 至少为 2，当前为 {num_classes}")
        if class_weights is not None and len(class_weights) != num_classes:
            raise ValueError(
                f"class_weights 长度应为 {num_classes}，当前为 {len(class_weights)}"
            )

        self.num_classes = num_classes
        self.ce_weight = ce_weight
        self.dice_weight = dice_weight
        self.ignore_index = ignore_index
        self.smooth = smooth
        self.dice_include_background = dice_include_background

        if class_weights is None:
            self.register_buffer("class_weights", torch.ones(num_classes))
        else:
            self.register_buffer(
                "class_weights", torch.as_tensor(class_weights, dtype=torch.float32)
            )

    # ------------------------------------------------------------------ #
    def forward(self, logits: Tensor, target: Tensor) -> dict[str, Tensor]:
        """计算损失。

        Parameters
        ----------
        logits:
            ``[B, num_classes, H, W]`` 的原始输出（未过 softmax）。
        target:
            ``[B, H, W]`` 的 ``int64`` 标签，取值 ``0..num_classes-1`` 或 ``ignore_index``。

        Returns
        -------
        dict[str, Tensor]
            ``{"loss": 总损失, "ce": 交叉熵, "dice": dice loss}``，
            三个都是标量张量，可直接 ``backward()``（用 ``loss`` 那个）。

        Raises
        ------
        ValueError
            形状不匹配或 target 不是整数类型时。
        """
        if logits.ndim != 4:
            raise ValueError(f"logits 必须是 [B, C, H, W]，当前为 {tuple(logits.shape)}")
        if target.ndim != 3:
            raise ValueError(f"target 必须是 [B, H, W]，当前为 {tuple(target.shape)}")
        if logits.shape[0] != target.shape[0] or logits.shape[-2:] != target.shape[-2:]:
            raise ValueError(
                f"logits {tuple(logits.shape)} 与 target {tuple(target.shape)} 的 B/H/W 不一致"
            )
        if target.dtype not in (torch.int64, torch.int32, torch.uint8):
            raise ValueError(f"target 应为整数类型，当前为 {target.dtype}")

        ce = F.cross_entropy(
            logits,
            target.long(),
            weight=self.class_weights.to(logits.device),
            ignore_index=self.ignore_index,
        )
        dice = self._soft_dice(logits, target)
        total = self.ce_weight * ce + self.dice_weight * dice
        return {"loss": total, "ce": ce, "dice": dice}

    # ------------------------------------------------------------------ #
    def _soft_dice(self, logits: Tensor, target: Tensor) -> Tensor:
        """按类别计算 soft Dice，再对选中的类别取平均。

        soft Dice 用概率而非硬标签计算，因此处处可导：
        ``dice_c = (2·Σ p_c·t_c + smooth) / (Σ p_c + Σ t_c + smooth)``。
        """
        probs = F.softmax(logits, dim=1)

        # ignore_index 位置先置零，避免污染分子分母
        valid = (target != self.ignore_index).unsqueeze(1).to(probs.dtype)
        safe_target = target.clone()
        safe_target[safe_target == self.ignore_index] = 0

        one_hot = F.one_hot(safe_target.long(), num_classes=self.num_classes)
        one_hot = one_hot.permute(0, 3, 1, 2).to(probs.dtype) * valid
        probs = probs * valid

        # 在 (B, H, W) 上求和，得到每个类别的 dice
        dims = (0, 2, 3)
        intersection = (probs * one_hot).sum(dim=dims)
        cardinality = probs.sum(dim=dims) + one_hot.sum(dim=dims)
        dice_per_class = (2.0 * intersection + self.smooth) / (cardinality + self.smooth)

        if self.dice_include_background:
            dice = dice_per_class.mean()
        else:
            dice = dice_per_class[1:].mean()
        return 1.0 - dice


__all__ = ["DiceCrossEntropyLoss"]
