# ToxPath —— 基于 HistoPLUS 架构的肿瘤三分类分割（模型 + 训练）

把 Owkin 的 **HistoPLUS**（细胞实例分割）改造成**区域级三分类语义分割**：

```
输入  1024×1024 H&E patch            [B, 3, 1024, 1024]
输出  逐像素 3 类 logits             [B, 3, 1024, 1024]
类别  0 = 背景 / 非肿瘤
      1 = 前景占比 0.01% ~ 5%（少量肿瘤）
      2 = 前景占比 ≥ 5%（较多肿瘤）
```

编码器（H0-mini ViT-B/14）与四路 Neck 沿用 HistoPLUS，只把原来的 np/hv/tp
三个解码头换成一个 3 类语义头：encoder 85.7 M + neck 14.3 M + decoder 14.0 M
= **114.1 M** 参数。

> 项目当前**只保留模型定义与训练代码**。推理、评测、比赛提交、打包脚本已从工作区
> 移除（仍完整保存在 git 历史里），需要时见文末「恢复旧文件」。

## 目录结构

```
toxpath/
├── model/                    # 模型定义（唯一需要读的模型代码）
│   ├── config.py             #   ToxPathConfig：尺寸/类别数/初始化策略
│   ├── encoder.py            #   H0-mini ViT-B/14（timm 构造，支持 out_indices 四层特征）
│   ├── neck.py               #   HistoPLUS 四路 Neck（多尺度特征聚合）
│   ├── decoder.py            #   3 类语义解码头
│   ├── blocks.py             #   FinalConv 等基础块
│   ├── losses.py             #   DiceCrossEntropyLoss（CE + soft Dice）
│   └── toxpath.py            #   ToxPathSegmentor：encoder → neck → decoder
├── dataset/
│   └── toxpath3_dataset.py   # 读 patch + 三分类掩膜，按类别加权采样
├── train_3class.py           # 训练入口（单折 / 多折 / 续训）
├── inference.py              # checkpoint 加载、前向、Otsu 组织掩膜
├── predict_test.py           # 三分类 → 二值肿瘤掩膜，按官方规范出提交 zip
├── splits.py                 # 按切片做蛇形分折（切片数自适应）
├── paths.py                  # 数据根 / 权重路径解析，不写死盘符
├── scripts/
│   └── make_3class_masks.py  # 二值 mask → 三分类彩色 mask + 标签索引（训练前置）
├── requirements.txt
└── data/toxpath/             # 数据（不进仓库）
```

`scripts/make_3class_masks.py` 是唯一保留的脚本：`dataset` 与 `splits` 的报错信息
都指向它，删掉后换数据就没法重新生成三分类标签。其余脚本已移除。

## 环境

`timm` **必须锁 1.0.8**，换版本会改变 ViT block 实现或 `out_indices` 语义。

```powershell
conda create -n toxpath python=3.10 -y
conda activate toxpath
pip install --index-url https://download.pytorch.org/whl/cu124 torch torchvision   # 4090
pip install timm==1.0.8 numpy Pillow
```

CPU 机器（只验形状）把第一行换成 `.../whl/cpu`。

## 数据

```
data/toxpath/
├── train_patches/{slide}/1.0/{X}_{Y}.jpeg        # 1024×1024 patch
├── train_masks_classes/{slide}/1.0/{X}_{Y}.png   # 三分类彩色掩膜（黑/绿/红）
└── _patch_labels_3class.csv                      # slide,name,n_fg_px,fg_ratio,label
```

路径不写死，`paths.py` 按 **命令行 `--data-root` > 环境变量 `TOXPATH_DATA_ROOT` >
`<项目>/data/toxpath` > 同级目录 > 历史绝对路径** 的顺序找，找不到会列出所有候选。
权重同理：`--histoplus-ckpt` > `TOXPATH_HISTOPLUS_CKPT` > `<项目>/weights/`。

三分类标签由二值 mask 派生（类别按整张 patch 的前景占比划分：<0.01% 记背景，
0.01%~5% 记 1，≥5% 记 2）：

```powershell
python scripts/make_3class_masks.py --dry-run   # 只统计分布，不写文件
python scripts/make_3class_masks.py             # 生成 train_masks_classes/ 与 CSV
```

## 训练

```powershell
# 冒烟：1 epoch、8 个样本、224 输入，确认数据与显存（CPU 也能跑，约 1 分钟）
python train_3class.py --fold 1 --epochs 1 --image-size 224 `
  --max-train-samples 8 --max-val-samples 4 --samples-per-epoch 8 `
  --num-workers 0 --device cpu --init none `
  --out-dir _smoke/ckpt --log-dir _smoke/log

# 单折（先跑 1 折看曲线）
python train_3class.py --fold 1 --epochs 12

# 五折（按切片蛇形分折，结果写 checkpoints3/splits.json）
python train_3class.py --all-folds --epochs 12

# 续训（从 last.pt 接着跑）
python train_3class.py --fold 1 --epochs 24 --resume checkpoints3/fold1/last.pt
```

主要默认值（都可用命令行覆盖）：

| 参数 | 默认 | 说明 |
|---|---|---|
| `--image-size` | 1024 | 1024 输入下 `--batch-size` 通常只能 1~2 |
| `--grad-accum` | 8 | 等效 batch = batch_size × grad_accum |
| `--samples-per-epoch` | 800 | 按类别权重采样，不是全量遍历 |
| `--lr` / `--encoder-lr-mult` | 3e-4 / 0.1 | 编码器用 1/10 学习率 |
| `--class-weights` | 0.20 0.50 0.30 | 背景 / 少量 / 较多 |
| `--init` | encoder+neck | 可选 `none`/`encoder`/`encoder+neck`/`full` |
| `--freeze-encoder` | 关 | 显存吃紧时开 |
| `--amp` | 开 | CUDA 上 bfloat16 |

产物：

```
checkpoints3/
├── splits.json            # 折划分（切片名 → fold），复现与对账用
├── fold1/best.pt          # dice_fg 最好的那个 epoch
├── fold1/last.pt          # 续训用
└── summary.json           # 多折汇总（全折模式）
logs3/fold1.jsonl          # 逐 epoch 的 loss / dice / acc
```

## 三分类 → 二值肿瘤掩膜

模型仍是 3 类头，二值化只是导出时多一步。三分类 softmax 互斥，所以：

```
p_肿瘤 = 1 - p_bg = p_1 + p_2        # 两种写法数值完全等价
mask   = (p_肿瘤 >= --fg-threshold)
```

`argmax > 0` 与 `p_肿瘤 > 0.5` 也等价，因此 `dice_fg`（把 1、2 合并）就是二值 Dice。

```powershell
python predict_test.py --test-root D:\data\toxpath_test `
  --ckpt-dir checkpoints3 --tta --min-area 128 `
  --out-dir submission --zip submission.zip
```

产物：`submission\{slide}\{name}.png`，单通道 L、8-bit、只含 0/255、尺寸同原图，
zip 根目录直挂 `{slide}/`（中间那层 `1.0/` 已去掉）。

**不要**从 `test_images\*.tif` 重新切图——官方 patch 文件名编码了原图坐标，
自己切的坐标系不同会对不上真值。

阈值与最小连通域别用默认值硬跑：本地 83.5% 的 patch 真值全空，空图上 1 个假阳
就让该 patch 从 1.0 变 0。要在**验证折**上扫（扫参数的脚本见「恢复旧文件」）。

## 恢复旧文件

被裁掉的推理 / 评测 / 提交链路在 git 历史里（提交 `2ed284a` 及之前），按需取回：

```powershell
# 看当时有哪些文件
git ls-tree -r --name-only 2ed284a

# 取回单个文件或整个目录
git checkout 2ed284a -- predict_test.py
git checkout 2ed284a -- scripts docs tests
```

`2ed284a` 里的 `docs/本地训练SOP.md` 是完整 SOP（环境、权重下载、断电续训、
比赛提交规范与实测数据），需要时取回即可。
