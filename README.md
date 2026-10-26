# ToxPath —— 基于 HistoPLUS 架构的 WSI 肿瘤三分类语义分割

把 Owkin 的 **HistoPLUS**（细胞实例分割）改造成**区域级三分类语义分割**：

```
输入  1024×1024 H&E patch            [B, 3, 1024, 1024]
输出  逐像素 3 类 logits             [B, 3, 1024, 1024]
类别  0 = 背景 / 非肿瘤
      1 = 前景占比 0.01% ~ 5%（少量肿瘤）
      2 = 前景占比 ≥ 5%（较多肿瘤）
```

编码器（H0-mini ViT-B/14）与四路 Neck 完全沿用 HistoPLUS，只把原来的
np/hv/tp 三个解码头换成一个 3 类语义头；总参数 114.1 M。

## 环境

```bash
conda activate toxpath          # python 3.10 + torch + timm==1.0.8
# 若 conda activate 不可用：
source "D:/Program Files/anaconda3/etc/profile.d/conda.sh"
```

本地是 CPU 版 torch，只做形状/管线冒烟；正式训练在 AutoDL 4090D 上用 CUDA 版。

## 数据

```
data/toxpath/
├── train_images/      5 张金字塔 TIFF（L0 8.6万~10万 px）
├── train_labels/      5 个 GeoJSON（QuPath 导出的 Lymphoma / Region* / Other）
├── train_patches/     4953 张 1024×1024 H&E JPEG
├── train_masks/       4953 张二值 mask（0/255）        ← 原始标注
├── train_masks_classes/ 4953 张彩色三分类 mask（黑/绿/红） ← 训练读取
└── _patch_labels_3class.csv  patch 类别与前景占比索引
```

类别分布（实测）：

| 类别 | 数量 | 占比 |
|---|---:|---:|
| 0 背景 | 4139 | 83.6% |
| 1 少量（0.01–5%） | 172 | 3.5% |
| 2 较多（≥5%） | 642 | 13.0% |

生成三分类 mask（已执行，改阈值后可重跑）：

```bash
python scripts/make_3class_masks.py --dry-run   # 先看分布
python scripts/make_3class_masks.py             # 写 train_masks_classes/
```

## 训练

```bash
# 单折（1024 输入，4090D）
python train_3class.py --fold 1 --epochs 12 --batch-size 1 --grad-accum 8 \
    --samples-per-epoch 800 --amp --grad-checkpointing

# 全 5 折（按切片留一）
python train_3class.py --all-folds --epochs 12 --batch-size 1 --grad-accum 8 --amp

# 本地 CPU 冒烟
python train_3class.py --fold 1 --epochs 1 --image-size 224 --max-train-samples 4 \
    --max-val-samples 2 --device cpu --no-amp --num-workers 0
```

产物：`checkpoints3/foldN/{best.pt,last.pt,history.json}`、
`logs3/foldN/train_log.json`、`checkpoints3/summary.json`。

预训练权重默认从 `E:/project/histoplus_cellvit_segmentor_20x.pt` 加载：

| `--init` | 加载内容 |
|---|---|
| `encoder` | 仅 H0-mini（173 个张量） |
| `encoder+neck` | 编码器 + 四路 Neck（68 个张量，默认） |
| `full` | 再加 np 解码头（71/73，最后一层 1×1 因 3 类而跳过） |

## 目录

```
toxpath/
├── model/
│   ├── config.py     # ToxPathConfig（num_classes/in_channels/patch_size/...）
│   ├── blocks.py     # ConvNormAct / DeconvNormAct / SwiGLUFFNFused
│   ├── encoder.py    # H0-mini → 4 层 [B,768,H/14,W/14]
│   ├── neck.py       # HistoPLUS 四路 Neck（64/128/256/512）
│   ├── decoder.py    # 单头语义解码器（3 类）
│   ├── toxpath.py    # 组装 + forward（含 1024→1036 的 pad/crop）
│   └── losses.py     # CE + soft Dice
├── dataset/
│   └── toxpath3_dataset.py   # 1024 输入、彩色三分类 mask、采样权重
├── scripts/
│   └── make_3class_masks.py  # 二值 mask → 彩色三分类 mask
├── tests/
│   └── test_model_forward.py # 模型形状/反传/冻结/权重往返 8 项测试
└── train_3class.py           # 5 折训练脚本（AMP + 梯度累积）
```

## 必须记住的约束

1. **1024 不是 14 的倍数**：`ToxPathSegmentor.forward` 会 reflect-pad 到
   1036（14×74）再裁回 1024；Dataset 不需要自己 pad。
2. **归一化用 Bioptimus 统计量**：`mean=(0.707223,0.578729,0.703617)`，
   `std=(0.211883,0.230117,0.177517)`，不要用 ImageNet。
3. **类别是 patch 级的**：同一张 patch 的前景只有一种颜色，类别由整图前景
   占比决定；模型实际学的是「前景分割 + 整图肿瘤负荷分级」。
4. **不要换 `SwiGLUFFNFused`**：参数名（`w12/w3`）与门控顺序都与 timm 不同。

