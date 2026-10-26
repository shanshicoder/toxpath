"""ToxPath 主模型：H0-mini 编码器 + HistoPLUS Neck + 单头语义解码器。

整体数据流（以 784×784 输入、二分类为例）
--------------------------------------
::

    输入 RGB tile                     [B, 3,   784, 784]
      │
      ├─ H0MiniEncoder（ViT-B/14） ──→  f0..f3  每个 [B, 768, 56, 56]
      │                                   │
      │        ┌──────────────────────────┴───────────────────────────┐
      │        │ HistoPlusNeck（4 条独立反卷积链）                      │
      │        │   n0 ← 原图          [B,  64, 784, 784]              │
      │        │   n1 ← f0 (×8)       [B, 128, 448, 448]              │
      │        │   n2 ← f1 (×4)       [B, 256, 224, 224]              │
      │        │   n3 ← f2 (×2)       [B, 512, 112, 112]              │
      │        └──────────────────────────┬───────────────────────────┘
      │                                   │
      └─ z4 = f3 [B, 768, 56, 56] ──→ SemanticDecoder（单头，2 类）
                                          ↓
                                    logits [B, 2, 784, 784]

对 HistoPLUS 的复用与改动
------------------------
- **复用**：编码器（H0-mini ViT-B/14）、Neck 的 4 条链、解码器的逐级融合结构。
- **改动**：3 个 ``DecoderBranch``（np/hv/tp）→ 1 个 ``SemanticDecoder``。
  参数量从 142.2M 降到约 114M，显存占用约为原来的 1/3。
"""

from __future__ import annotations

from typing import Iterator

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from .config import ToxPathConfig
from .decoder import SemanticDecoder
from .encoder import H0MiniEncoder, LoadReport
from .neck import HistoPlusNeck


class ToxPathSegmentor(nn.Module):
    """面向 WSI 肿瘤区域二分类的语义分割模型。

    Parameters
    ----------
    cfg:
        模型配置；为 ``None`` 时使用默认配置（H0-mini + 784 tile + 2 类）。

    Examples
    --------
    >>> import torch
    >>> from model import ToxPathConfig, ToxPathSegmentor
    >>> model = ToxPathSegmentor(ToxPathConfig(pretrained=False)).eval()
    >>> x = torch.randn(1, 3, 224, 224)
    >>> with torch.no_grad():
    ...     y = model(x)
    >>> tuple(y.shape)
    (1, 2, 224, 224)
    """

    def __init__(self, cfg: ToxPathConfig | None = None) -> None:
        super().__init__()
        self.cfg: ToxPathConfig = cfg or ToxPathConfig()

        self.encoder: H0MiniEncoder = H0MiniEncoder(self.cfg)
        self.neck: HistoPlusNeck = HistoPlusNeck(self.cfg)
        self.decoder: SemanticDecoder = SemanticDecoder(self.cfg)

    # ------------------------------------------------------------------ #
    # 前向
    # ------------------------------------------------------------------ #
    def forward(self, x: Tensor) -> Tensor:
        """端到端前向。

        Parameters
        ----------
        x:
            ``[B, 3, H, W]``。
            归一化使用 Bioptimus 的统计量（``BIOPTIMUS_MEAN/STD``，见 README）。

        Returns
        -------
        Tensor
            ``[B, num_classes, H, W]`` 的原始 logits，空间尺寸与输入**完全一致**。

        Notes
        -----
        ViT 的 patch_size=14 要求边长是 14 的倍数，但实际数据常见 1024
        （1024 % 14 = 2）。这里在右侧/下侧做 reflect padding 到最近的 14 倍数
        （1024 → 1036 = 14×74），跑完网络后再裁回 ``H×W``。
        因此 224 / 448 / 1024 / 784 等任意尺寸都能直接输入，调用方不需要自己 pad。

        Raises
        ------
        ValueError
            输入维度不为 4。
        """
        if x.ndim != 4:
            raise ValueError(f"输入必须是 [B, C, H, W]，当前为 {tuple(x.shape)}")

        h, w = int(x.shape[-2]), int(x.shape[-1])
        patch = self.cfg.patch_size
        pad_h, pad_w = (-h) % patch, (-w) % patch
        x_in: Tensor = x
        if pad_h or pad_w:
            # reflect 比 constant 更不引入明显的人工边界；pad 最大只有 13 px。
            x_in = F.pad(x, (0, pad_w, 0, pad_h), mode="reflect")

        features: list[Tensor] = self.encoder(x_in)
        neck_features: list[Tensor] = self.neck(x_in, features)
        z4: Tensor = features[-1]                       # 最深一层，绕过 Neck

        logits: Tensor = self.decoder(
            z4, neck_features, out_size=(h + pad_h, w + pad_w)
        )
        if pad_h or pad_w:
            logits = logits[..., :h, :w]
        return logits

    # ------------------------------------------------------------------ #
    # 推理便捷方法
    # ------------------------------------------------------------------ #
    @torch.no_grad()
    def predict_proba(self, x: Tensor) -> Tensor:
        """推理用：返回 softmax 后的概率图。

        Parameters
        ----------
        x: ``[B, 3, H, W]``

        Returns
        -------
        Tensor
            ``[B, num_classes, H, W]``，各通道在通道维上求和为 1。
        """
        was_training = self.training
        self.eval()
        logits = self.forward(x)
        if was_training:
            self.train(True)
        return F.softmax(logits, dim=1)

    @torch.no_grad()
    def predict_mask(self, x: Tensor, threshold: float = 0.5) -> Tensor:
        """推理用：返回掩膜。

        二分类（``num_classes=2``）时返回 ``{0,1}`` 的二值前景；
        多分类（如本项目三分类）时返回 ``argmax`` 的类别索引 ``{0..C-1}``。

        Parameters
        ----------
        x: ``[B, 3, H, W]``
        threshold:
            仅二分类时生效：前景概率阈值，默认 0.5。

        Returns
        -------
        Tensor
            ``[B, H, W]`` 的 ``uint8`` 张量。
        """
        proba: Tensor = self.predict_proba(x)
        if self.cfg.num_classes == 2:
            return (proba[:, 1] >= threshold).to(torch.uint8)
        return proba.argmax(dim=1).to(torch.uint8)

    # ------------------------------------------------------------------ #
    # 冻结 / 参数管理
    # ------------------------------------------------------------------ #
    def freeze_encoder(self, freeze: bool = True) -> None:
        """冻结或解冻 H0-mini 编码器。"""
        self.encoder.freeze(freeze)

    def train(self, mode: bool = True) -> "ToxPathSegmentor":
        """切换训练/评估模式，并保护被冻结的编码器。

        覆盖 ``nn.Module.train`` 的原因：父类的 ``train(True)`` 会把**所有**子模块
        的 ``training`` 标志置为 True，包括被冻结的编码器。虽然 ViT 只有 LayerNorm
        （不受 training 标志影响），但保持语义一致、也避免将来换成带 Dropout 的
        主干时出现意外。
        """
        super().train(mode)
        if self.encoder.is_frozen:
            self.encoder.backbone.eval()
        return self

    def trainable_parameters(self) -> Iterator[nn.Parameter]:
        """只产出 ``requires_grad=True`` 的参数。"""
        return (p for p in self.parameters() if p.requires_grad)

    def param_groups(
        self,
        lr: float,
        *,
        encoder_lr_mult: float = 0.1,
        weight_decay: float = 0.01,
    ) -> list[dict[str, object]]:
        """构造 AdamW 参数组：编码器用较小学习率，且 norm/bias 不做权重衰减。

        Parameters
        ----------
        lr:
            解码头（Neck + Decoder）的基础学习率。
        encoder_lr_mult:
            编码器学习率相对 ``lr`` 的倍率。微调预训练 ViT 时通常取 0.05~0.1。
        weight_decay:
            非 norm/bias 参数的权重衰减系数。

        Returns
        -------
        list[dict]
            可直接传给 ``torch.optim.AdamW(params=...)``。

        Notes
        -----
        ``BatchNorm``/``GroupNorm`` 的权重与所有 bias 都不做衰减，
        这是分割任务里的通行做法（否则小数据下容易把归一化层压死）。
        """
        no_decay_keys = ("bias",)

        def _split(module: nn.Module, group_lr: float) -> list[dict[str, object]]:
            decay: list[nn.Parameter] = []
            no_decay: list[nn.Parameter] = []
            for name, param in module.named_parameters():
                if not param.requires_grad:
                    continue
                if any(k in name for k in no_decay_keys) or param.ndim <= 1:
                    no_decay.append(param)
                else:
                    decay.append(param)
            groups: list[dict[str, object]] = []
            if decay:
                groups.append({"params": decay, "lr": group_lr, "weight_decay": weight_decay})
            if no_decay:
                groups.append({"params": no_decay, "lr": group_lr, "weight_decay": 0.0})
            return groups

        head_modules = [self.neck, self.decoder]
        groups: list[dict[str, object]] = []
        if not self.encoder.is_frozen:
            groups.extend(_split(self.encoder, lr * encoder_lr_mult))
        for module in head_modules:
            groups.extend(_split(module, lr))
        return groups

    # ------------------------------------------------------------------ #
    # 统计信息
    # ------------------------------------------------------------------ #
    @property
    def num_parameters(self) -> int:
        """模型总参数量。"""
        return sum(p.numel() for p in self.parameters())

    @property
    def num_trainable_parameters(self) -> int:
        """可训练参数量（冻结编码器后会显著变小）。"""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def describe(self) -> str:
        """返回各子模块的参数量摘要（训练前打印一次，确认结构没搭错）。"""
        def _count(module: nn.Module) -> int:
            return sum(p.numel() for p in module.parameters())

        lines = [
            "ToxPathSegmentor 结构摘要",
            "-" * 46,
            f"  encoder (H0-mini) : {_count(self.encoder):>12,}",
            f"  neck              : {_count(self.neck):>12,}",
            f"  decoder ({self.cfg.num_classes}-class) : {_count(self.decoder):>12,}",
            "-" * 46,
            f"  total             : {self.num_parameters:>12,}",
            f"  trainable         : {self.num_trainable_parameters:>12,}",
            f"  patch_size        : {self.cfg.patch_size}",
            f"  num_classes       : {self.cfg.num_classes}",
        ]
        return "\n".join(lines)

    # ------------------------------------------------------------------ #
    # 权重
    # ------------------------------------------------------------------ #
    def load_histoplus_backbone(self, checkpoint: str, *, verbose: bool = True) -> LoadReport:
        """从 HistoPLUS 官方 checkpoint 初始化 H0-mini 编码器。

        只会加载编码器部分；checkpoint 里的 Neck 与 3 个细胞解码头会被忽略
        （它们的通道数与本项目的单头结构不兼容）。

        Parameters
        ----------
        checkpoint:
            ``histoplus_cellvit_segmentor_20x.pt`` 之类的路径。
        verbose:
            是否打印加载摘要。

        Returns
        -------
        LoadReport
            加载统计。
        """
        return self.encoder.load_pretrained(checkpoint, verbose=verbose)


# --------------------------------------------------------------------------- #
# 便捷工厂
# --------------------------------------------------------------------------- #
def build_toxpath(**overrides: object) -> ToxPathSegmentor:
    """用关键字参数快速构造模型。

    Parameters
    ----------
    **overrides:
        覆盖 :class:`ToxPathConfig` 的任意字段，例如
        ``build_toxpath(num_classes=2, image_size=448, freeze_encoder=True)``。

    Returns
    -------
    ToxPathSegmentor
        已按配置实例化的模型。

    Examples
    --------
    >>> model = build_toxpath(image_size=448, dropout=0.1)
    >>> model.cfg.image_size
    448
    """
    cfg = ToxPathConfig(**overrides)  # type: ignore[arg-type]
    return ToxPathSegmentor(cfg)


__all__ = [
    "ToxPathSegmentor",
    "build_toxpath",
    "ToxPathConfig",
    "H0MiniEncoder",
    "HistoPlusNeck",
    "SemanticDecoder",
]
