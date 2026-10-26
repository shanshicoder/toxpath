"""H0-mini 编码器（ViT-B/14，DINOv2 + registers）的特征提取封装。

为什么是 H0-mini
----------------
H0-mini 是 Bioptimus / Owkin 在 MICCAI 2025 发布的病理基础模型：
ViT-B/14（86M 参数），由 H0-Optimus-0（ViT-g/14，1.1B）蒸馏而来，
蒸馏跑了 105k 次迭代（HistoPLUS 代码里那个 ``aquavit_105k`` 的后缀即来源于此）。

它不是一个"能直接分割"的模型，而是**特征前端**：
输入 ``[B, 3, H, W]``（H、W 必须能被 14 整除），
输出 4 个中间层特征，每个形状都是 ``[B, 768, H/14, W/14]``。
注意 ViT 的各 block 之间**不做空间下采样**，所以 4 层特征的分辨率相同、
只有语义深度不同 —— 这一点和 CNN 主干（ResNet 的 1/4、1/8、1/16…）完全不同，
也是 HistoPLUS 的 Neck 必须用"反卷积次数 3/2/1"来制造多尺度的原因。

与 HistoPLUS 的严格对齐点（见 ``histoplus/helpers/nn/vit.py``）
--------------------------------------------------------------
.. code-block:: python

    timm.create_model(
        model_name="vit_base_patch14_reg4_dinov2.lvd142m",
        img_size=224, patch_size=14, init_values=1e-5, num_classes=0,
        dynamic_img_size=True, mlp_ratio=4, mlp_layer=SwiGLUFFNFused,
        features_only=True, out_indices=out_indices,
    )

其中 ``mlp_layer=SwiGLUFFNFused`` 是最容易被漏掉的一项：
H0-mini 的 FFN 用的是 SwiGLU（隐藏维按 2/3·4d 再向上取 8 的倍数），
如果直接用 timm 默认的 GELU-MLP，权重形状对不上、加载会大面积 mismatch。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Final

import torch
from torch import Tensor, nn

from .blocks import SwiGLUFFNFused
from .config import ToxPathConfig

# checkpoint 中可能出现的、需要剥离的前缀（按长度从长到短尝试）。
# 注意这里不是穷举，真正的对齐靠下面的"按 anchor 后缀匹配"逻辑。
_KNOWN_PREFIXES: Final[tuple[str, ...]] = (
    "module.",
    "model.",
    "backbone.",
    "feature_extractor.",
)

# 用来在 state_dict key 里定位"有效后缀"的锚点。
# 只要 key 里出现这些子串，就认为从该处开始是真正的参数路径。
_ANCHORS: Final[tuple[str, ...]] = (
    "blocks.",
    "patch_embed.",
    "pos_embed",
    "reg_token",
    "cls_token",
    "norm.",
    "head.",
)


@dataclass(slots=True)
class LoadReport:
    """权重加载结果摘要（便于排查"为什么没加载上"）。

    Attributes
    ----------
    loaded:
        成功匹配并载入的张量个数。
    missing:
        模型有、checkpoint 没有的参数名（前 20 条）。
    unexpected:
        checkpoint 有、模型没有的参数名（前 20 条）。
    shape_mismatch:
        名字对上但形状不一致的参数名（前 20 条）。
    """

    loaded: int = 0
    missing: list[str] = field(default_factory=list)
    unexpected: list[str] = field(default_factory=list)
    shape_mismatch: list[str] = field(default_factory=list)

    def summary(self) -> str:
        """返回一行人类可读的摘要。"""
        return (
            f"loaded={self.loaded}, missing={len(self.missing)}, "
            f"unexpected={len(self.unexpected)}, shape_mismatch={len(self.shape_mismatch)}"
        )


class H0MiniEncoder(nn.Module):
    """把 H0-mini 包装成"返回 4 层 NCHW 特征图"的编码器。

    Parameters
    ----------
    cfg:
        模型配置，见 :class:`model.config.ToxPathConfig`。

    Notes
    -----
    输入约束：``H % 14 == 0`` 且 ``W % 14 == 0``。
    这是因为 ViT 先把图像切成 14×14 的 patch，尺寸不整除时 patch 化会丢边。
    推荐取值：224（ViT 原生）、336、448、784（HistoPLUS 推理尺寸）。
    """

    def __init__(self, cfg: ToxPathConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.patch_size: int = cfg.patch_size
        self.embed_dim: int = cfg.embed_dim
        self.out_indices: tuple[int, ...] = tuple(cfg.out_indices)

        self.backbone: nn.Module = self._build_backbone()

        if cfg.grad_checkpointing:
            self.set_grad_checkpointing(True)
        if cfg.freeze_encoder:
            self.freeze(True)

    # ------------------------------------------------------------------ #
    # 构造
    # ------------------------------------------------------------------ #
    def _build_backbone(self) -> nn.Module:
        """用 timm 构造 ViT-B/14 特征提取器（参数与 HistoPLUS 完全一致）。

        Returns
        -------
        nn.Module
            timm 的 ``FeatureListNet``，forward 返回 list[Tensor]。
        """
        try:
            import timm
        except ImportError as exc:  # pragma: no cover - 环境问题
            raise ImportError(
                "需要 timm 才能构造 H0-mini 编码器，请先在 toxpath 环境执行:\n"
                "    pip install timm==1.0.8"
            ) from exc

        backbone = timm.create_model(
            self.cfg.encoder_name,
            pretrained=self.cfg.pretrained,
            img_size=224,               # pos_embed 的原生网格是 16×16（224/14）
            patch_size=self.cfg.patch_size,
            in_chans=self.cfg.in_channels,
            init_values=1e-5,           # LayerScale 初值，DINOv2 家族固定为 1e-5
            num_classes=0,              # 不要分类头
            dynamic_img_size=True,      # 允许任意输入尺寸，timm 自动插值 pos_embed
            mlp_ratio=4,
            mlp_layer=SwiGLUFFNFused,   # ⚠️ 关键：H0-mini 的 FFN 是 SwiGLU
            features_only=True,         # forward 返回各层特征而非单个 pool 向量
            out_indices=list(self.out_indices),
        )
        return backbone

    # ------------------------------------------------------------------ #
    # 前向
    # ------------------------------------------------------------------ #
    def forward(self, x: Tensor) -> list[Tensor]:
        """提取 4 层特征。

        Parameters
        ----------
        x:
            ``[B, 3, H, W]``，其中 ``H % patch_size == 0``、``W % patch_size == 0``。
            归一化请使用 Bioptimus 的 mean/std：
            ``(0.707223, 0.578729, 0.703617)`` / ``(0.211883, 0.230117, 0.177517)``。

        Returns
        -------
        list[Tensor]
            长度为 4，按**从浅到深**排列，每个形状 ``[B, 768, H//14, W//14]``。

        Raises
        ------
        ValueError
            当输入不是 4 维、或空间尺寸不能被 patch_size 整除时。
        """
        if x.ndim != 4:
            raise ValueError(f"输入必须是 [B, C, H, W] 4 维张量，当前为 {tuple(x.shape)}")
        if x.shape[1] != self.cfg.in_channels:
            raise ValueError(
                f"输入通道数应为 {self.cfg.in_channels}，当前为 {x.shape[1]}"
            )
        h, w = int(x.shape[-2]), int(x.shape[-1])
        if h % self.patch_size != 0 or w % self.patch_size != 0:
            raise ValueError(
                f"输入空间尺寸 ({h}, {w}) 必须能被 patch_size={self.patch_size} 整除，"
                f"建议改用 {self._nearest_valid_size(h)}×{self._nearest_valid_size(w)}"
            )

        raw: Any = self.backbone(x)
        return self._to_list(raw)

    @staticmethod
    def _to_list(raw: Any) -> list[Tensor]:
        """把 timm 可能的返回值（list / tuple / dict / 单个 Tensor）统一成 list。"""
        if isinstance(raw, Tensor):
            return [raw]
        if isinstance(raw, dict):
            return [t for t in raw.values()]
        if isinstance(raw, (list, tuple)):
            return [t for t in raw if isinstance(t, Tensor)]
        raise TypeError(f"无法解析编码器输出类型: {type(raw)!r}")

    def _nearest_valid_size(self, size: int) -> int:
        """把边长向上取到最近的 ``patch_size`` 整数倍（仅用于报错提示）。"""
        return ((size + self.patch_size - 1) // self.patch_size) * self.patch_size

    # ------------------------------------------------------------------ #
    # 冻结 / 梯度检查点
    # ------------------------------------------------------------------ #
    def freeze(self, freeze: bool = True) -> None:
        """冻结或解冻整个编码器。

        Parameters
        ----------
        freeze:
            ``True`` 冻结（``requires_grad=False`` 且切到 eval 模式），
            ``False`` 恢复训练。

        Notes
        -----
        冻结后请不要再整体调用 ``model.train()``，否则会把子模块的
        ``training`` 标志翻回来（本文件在 :class:`model.toxpath.ToxPathSegmentor`
        里做了保护性处理）。
        """
        for param in self.backbone.parameters():
            param.requires_grad = not freeze
        self.backbone.train(not freeze)
        self._frozen = freeze

    @property
    def is_frozen(self) -> bool:
        """编码器当前是否处于冻结状态。"""
        return getattr(self, "_frozen", False)

    def set_grad_checkpointing(self, enable: bool = True) -> None:
        """开关 ViT 的梯度检查点（用 ~20% 速度换 ~40% 激活显存）。

        timm 在 ``features_only=True`` 时返回的是 ``FeatureGetterNet`` 包装器，
        真正的 ``set_grad_checkpointing`` 在它内部的 ``VisionTransformer`` 上，
        包装器本身没有这个方法。这里向下找到第一个提供该方法的子模块再调用，
        避免 ``--grad-checkpointing`` 在构造模型时直接报 AttributeError。
        """
        for module in self.backbone.modules():
            setter = getattr(module, "set_grad_checkpointing", None)
            if callable(setter):
                setter(enable)
                return
        raise AttributeError(
            f"{type(self.backbone).__name__} 及其子模块都不支持梯度检查点，"
            "请检查 timm 版本"
        )

    # ------------------------------------------------------------------ #
    # 权重加载
    # ------------------------------------------------------------------ #
    def load_pretrained(
        self,
        checkpoint: str | Path,
        *,
        verbose: bool = True,
    ) -> LoadReport:
        """从 HistoPLUS 官方 checkpoint（或任意 H0-mini 权重）加载编码器。

        典型用法：把 ``histoplus_cellvit_segmentor_20x.pt`` 里的
        H0-mini 编码器搬过来，作为本项目分割模型的初始化。
        checkpoint 里那三个细胞解码头（np / hv / tp）会被自动忽略。

        Parameters
        ----------
        checkpoint:
            ``.pt`` / ``.pth`` 文件路径。
        verbose:
            是否打印加载摘要。

        Returns
        -------
        LoadReport
            加载统计，用于确认是否真的加载成功。

        Raises
        ------
        FileNotFoundError
            checkpoint 不存在。
        KeyError
            文件里找不到任何形如 ``blocks.*`` 的编码器参数（多半是路径写错、
            或加载成了 decoder-only 的权重）。
        """
        path = Path(checkpoint)
        if not path.is_file():
            raise FileNotFoundError(f"找不到权重文件: {path}")

        raw_obj: Any = torch.load(str(path), map_location="cpu", weights_only=False)
        state_dict = self._extract_state_dict(raw_obj)

        target = self.backbone.state_dict()
        suffix_index = self._build_suffix_index(target)

        report = LoadReport()
        merged: dict[str, Tensor] = {}

        for key, value in state_dict.items():
            if not isinstance(value, Tensor):
                continue
            suffix = self._strip_to_suffix(key)
            if suffix is None:
                report.unexpected.append(key)
                continue
            target_key = suffix_index.get(suffix)
            if target_key is None:
                report.unexpected.append(key)
                continue

            expected = target[target_key]
            if expected.shape != value.shape:
                resized = self._try_resize_pos_embed(value, expected)
                if resized is None:
                    report.shape_mismatch.append(f"{key} {tuple(value.shape)} -> {tuple(expected.shape)}")
                    continue
                value = resized
            merged[target_key] = value
            report.loaded += 1

        incompatible = self.backbone.load_state_dict(merged, strict=False)
        report.missing = list(incompatible.missing_keys)[:20]
        report.unexpected = (report.unexpected + list(incompatible.unexpected_keys))[:20]

        if report.loaded == 0:
            raise KeyError(
                f"{path} 中没有找到任何可匹配的编码器参数。"
                f"文件顶层 keys 示例: {list(state_dict)[:5]}"
            )

        if verbose:
            print(f"[H0MiniEncoder] 从 {path.name} 加载 -> {report.summary()}")
            if report.missing:
                print(f"  未加载(missing, 前 20): {report.missing}")
            if report.shape_mismatch:
                print(f"  形状不匹配: {report.shape_mismatch}")

        return report

    # -- 加载辅助 ------------------------------------------------------- #
    @staticmethod
    def _extract_state_dict(obj: Any) -> dict[str, Tensor]:
        """从各种 checkpoint 容器里取出真正的 state_dict。

        依次尝试 ``obj["state_dict"]`` → ``obj["model"]`` → 直接当 dict 用。
        Lightning 保存的 ckpt 通常是 ``{"state_dict": {...}, "hyper_parameters": ...}``。
        """
        if isinstance(obj, dict):
            for key in ("state_dict", "model", "model_state_dict"):
                inner = obj.get(key)
                if isinstance(inner, dict) and any(isinstance(v, Tensor) for v in inner.values()):
                    return {str(k): v for k, v in inner.items()}
            if any(isinstance(v, Tensor) for v in obj.values()):
                return {str(k): v for k, v in obj.items()}
        raise KeyError("无法从 checkpoint 中解析出 state_dict")

    @staticmethod
    def _strip_to_suffix(key: str) -> str | None:
        """把 ``backbone.feature_extractor.model.blocks.0...`` 削成 ``blocks.0...``。

        做法是找第一个出现的 anchor 子串并截断，这样不依赖 checkpoints 里
        到底套了几层前缀（``model.`` / ``module.`` / ``backbone.`` …）。
        """
        # 先按已知前缀剥离，兼容 anchor 出现在奇怪位置的情况
        for prefix in _KNOWN_PREFIXES:
            if key.startswith(prefix):
                key = key[len(prefix) :]

        positions = [key.find(anchor) for anchor in _ANCHORS]
        positions = [p for p in positions if p >= 0]
        if not positions:
            return None
        return key[min(positions) :]

    @staticmethod
    def _build_suffix_index(target: dict[str, Tensor]) -> dict[str, str]:
        """构建 ``有效后缀 -> 完整参数名`` 的索引。"""
        index: dict[str, str] = {}
        for key in target:
            suffix = H0MiniEncoder._strip_to_suffix(key)
            if suffix is not None:
                index[suffix] = key
        return index

    @staticmethod
    def _try_resize_pos_embed(src: Tensor, dst: Tensor) -> Tensor | None:
        """位置编码尺寸不一致时做双三次插值，其余情况返回 ``None``。

        H0-mini 的 ``pos_embed`` 形状是 ``[1, 256, 768]``（16×16 grid，不含 cls/reg）。
        若 checkpoint 的网格与当前 ``img_size`` 不同就需要重采样。
        """
        if src.ndim != 3 or dst.ndim != 3:
            return None
        if src.shape[0] != dst.shape[0] or src.shape[2] != dst.shape[2]:
            return None
        if src.shape[-2] == dst.shape[-2]:
            return None

        side_src = int(round(src.shape[-2] ** 0.5))
        side_dst = int(round(dst.shape[-2] ** 0.5))
        if side_src * side_src != src.shape[-2] or side_dst * side_dst != dst.shape[-2]:
            return None

        resized = src.reshape(1, side_src, side_src, -1).permute(0, 3, 1, 2)
        resized = torch.nn.functional.interpolate(
            resized, size=(side_dst, side_dst), mode="bicubic", align_corners=False
        )
        return resized.permute(0, 2, 3, 1).reshape(1, side_dst * side_dst, -1)


__all__ = ["H0MiniEncoder", "LoadReport"]
