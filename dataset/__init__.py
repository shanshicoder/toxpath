"""ToxPath 数据集包：三分类语义分割（整张 1024×1024 输入）。

数据源
------
- ``data/toxpath/train_patches/{slide}/1.0/{X}_{Y}.jpeg``      H&E 1024×1024 RGB
- ``data/toxpath/train_masks_classes/{slide}/1.0/{X}_{Y}.png`` 彩色三分类 mask
- ``data/toxpath/_patch_labels_3class.csv``                    标签索引（采样权重用）

类别与颜色
----------
======  ==========================  =====================
类别    含义                        颜色（RGB）
======  ==========================  =====================
0       背景 / 非肿瘤               (0, 0, 0) 黑
1       零星肿瘤（小灶 / 碎片）     (0, 255, 0) 绿
2       大块肿瘤（实体瘤灶）        (255, 0, 0) 红
======  ==========================  =====================

彩色 mask 由 ``scripts/make_3class_masks.py`` 生成；类别由整张 patch 的**连通域
统计**决定（最大连通域面积 + 连通域数量 + 主体占比），规则与颜色集中定义在
项目根的 ``labels.py``。同一张 patch 的所有前景像素着同一颜色，背景纯黑。

与 HistoPLUS 的输入对齐
-----------------------
``ToxPath3Dataset.__getitem__`` 返回：

- ``image``: ``[3, 1024, 1024]`` float32，``(x/255 - Bioptimus_mean) / Bioptimus_std``
- ``mask``:  ``[1024, 1024]`` int64，取值 ``{0,1,2}``

1024 不是 14 的倍数（1024 % 14 = 2），由 ``ToxPathSegmentor.forward``
内部 reflect-pad 到 1036、跑完再裁回，Dataset 不需要自己 pad。

快速开始
--------
.. code-block:: python

    from pathlib import Path
    from dataset import Dataset3Config, ToxPath3Dataset, build_weighted_sampler

    ds = ToxPath3Dataset(
        Path("data/toxpath"), ["train_002", "train_003"],
        cfg=Dataset3Config(image_size=1024, augment=True),
        mode="train",
    )
    sample = ds[0]
    print(sample["image"].shape, sample["mask"].shape, sample["patch_label"])
"""

from .toxpath3_dataset import (
    BIOPTIMUS_MEAN,
    BIOPTIMUS_STD,
    CLASS_COLORS,
    CLASS_NAMES,
    Dataset3Config,
    ToxPath3Dataset,
    build_dataset3,
    build_weighted_sampler,
    rgb_to_class,
    seed_worker,
)

__all__ = [
    "BIOPTIMUS_MEAN",
    "BIOPTIMUS_STD",
    "CLASS_COLORS",
    "CLASS_NAMES",
    "Dataset3Config",
    "ToxPath3Dataset",
    "build_dataset3",
    "build_weighted_sampler",
    "rgb_to_class",
    "seed_worker",
]
