"""ToxPath —— 基于 HistoPLUS 架构的 WSI 肿瘤区域语义分割模型。

目录结构
--------
::

    toxpath/
    ├── model/
    │   ├── config.py     # 超参数（纯 dataclass，无 torch 依赖）
    │   ├── blocks.py     # Conv / Deconv 基础块
    │   ├── encoder.py    # H0-mini（ViT-B/14）特征提取器
    │   ├── neck.py       # HistoPLUS 四路并行 Neck
    │   ├── decoder.py    # 单头语义解码器（2 类）
    │   ├── toxpath.py    # 组装 + 前向 + 参数组
    │   └── losses.py     # CE + Dice 联合损失
    └── tests/
        └── test_model_forward.py

快速开始
--------
.. code-block:: python

    import torch
    from model import ToxPathConfig, ToxPathSegmentor

    model = ToxPathSegmentor(ToxPathConfig(image_size=224, pretrained=False)).eval()
    with torch.no_grad():
        logits = model(torch.randn(1, 3, 224, 224))
    print(logits.shape)   # torch.Size([1, 2, 224, 224])
"""

from __future__ import annotations

from .blocks import ConvNormAct, DeconvNormAct, FinalConv, interpolate_to
from .config import NormLayerName, ToxPathConfig
from .decoder import SemanticDecoder
from .encoder import H0MiniEncoder, LoadReport
from .losses import DiceCrossEntropyLoss
from .neck import HistoPlusNeck
from .toxpath import ToxPathSegmentor, build_toxpath

__version__ = "0.1.0"

__all__ = [
    # 配置
    "ToxPathConfig",
    "NormLayerName",
    # 组件
    "H0MiniEncoder",
    "HistoPlusNeck",
    "SemanticDecoder",
    "ConvNormAct",
    "DeconvNormAct",
    "FinalConv",
    "interpolate_to",
    # 模型
    "ToxPathSegmentor",
    "build_toxpath",
    "LoadReport",
    # 损失
    "DiceCrossEntropyLoss",
    # 版本
    "__version__",
]
