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

## 快速开始（实验室 RTX 4090）

代码走 Git，权重走 GitHub Release（271 MB 超过单文件 100 MB 限制，进不了仓库）。

```powershell
# 0) 实验室机器：拉代码（第一次）+ 下权重
git clone git@github.com:shanshicoder/toxpath.git D:\work\toxpath
cd D:\work\toxpath
scripts\fetch_weights.bat                    # 271 MB → weights\，自动校验 SHA256

# 1) 建环境（一次）
conda create -n toxpath python=3.10 -y
conda activate toxpath
pip install --index-url https://download.pytorch.org/whl/cu124 torch torchvision
pip install timm==1.0.8 numpy Pillow

# 2) 体检：GPU / torch / timm / 数据 / 权重 / 磁盘
python scripts\check_env.py --data-root "D:\data\toxpath"

# 3) 冒烟 5 分钟 → 单折 → 5 折
.\scripts\train_local.bat -Mode smoke
.\scripts\train_local.bat -Mode fold -Fold 1 -Epochs 20
.\scripts\train_local.bat -Mode all -Epochs 20
```

之后每次改完代码，笔记本上 `scripts\sync_github.bat`，实验室机器 `git pull` 即可。
细节（首次配 SSH key、权重下载失败重试、电源设置、远程桌面断开、续训、评测、故障排查）
见 `docs/本地训练SOP.md`。

## 环境

代码要跑在两台机器上：笔记本（CPU，只做形状冒烟）和实验室工作站（RTX 4090 24 GB，正式训练）。
**数据根与权重路径不写死**，统一由 `paths.py` 按「命令行 > 环境变量 > 项目内相对路径 > 历史绝对路径」解析，
换机器不用改代码。

```powershell
# 实验室机器（Windows + 4090）
conda create -n toxpath python=3.10 -y
conda activate toxpath
pip install --index-url https://download.pytorch.org/whl/cu124 torch torchvision
pip install timm==1.0.8 numpy Pillow

# 一条命令体检（GPU / torch / timm / 数据 / 权重 / 磁盘）
python scripts\check_env.py --data-root "D:\data\toxpath"
```

完整的本地训练流程见 **[docs/本地训练SOP.md](docs/本地训练SOP.md)**（环境搭建、电源设置、冒烟、5 折、续训、评测、故障排查）。


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

一键脚本（推荐，后台运行 + 日志重定向，远程桌面断开不影响）：

```powershell
.\scripts\train_local.bat -Mode smoke                 # 冒烟：1 epoch / 32 样本，约 5 分钟
.\scripts\train_local.bat -Mode fold -Fold 1 -Epochs 20   # 单折
.\scripts\train_local.bat -Mode all -Epochs 20            # 5 折，约 8~10 小时
.\scripts\train_local.bat -Mode resume -Fold 1            # 从 last.pt 续训
.\scripts\train_local.bat -Mode all -DryRun               # 只打印命令，不启动
```

直接调 Python（等价写法）：

```bash
# 单折（1024 输入，4090）
python train_3class.py --fold 1 --epochs 20 --batch-size 1 --grad-accum 8 \
    --samples-per-epoch 800 --val-every 2 --amp --grad-checkpointing

# 全 5 折（按切片留一）
python train_3class.py --all-folds --epochs 20 --batch-size 1 --grad-accum 8 --amp

# 本地 CPU 冒烟
python train_3class.py --fold 1 --epochs 1 --image-size 224 --max-train-samples 4 \
    --max-val-samples 2 --device cpu --no-amp --num-workers 0
```

产物：`checkpoints3/foldN/{best.pt,last.pt,history.json}`、
`logs3/foldN/train_log.json`、`checkpoints3/summary.json`。
实际单次运行日志在 `logs3/_runs/<模式>_<时间戳>.log`。

预训练权重按以下顺序查找（不再写死 E 盘）：

1. `--histoplus-ckpt`
2. 环境变量 `TOXPATH_HISTOPLUS_CKPT`
3. `<项目根>/weights/histoplus_cellvit_segmentor_20x.pt` ← 推荐
4. `<项目根>/histoplus_cellvit_segmentor_20x.pt`
5. `<项目根>/../histoplus_cellvit_segmentor_20x.pt`
6. `E:/project/histoplus_cellvit_segmentor_20x.pt`（笔记本旧位置）

> 权重不进 Git（271 MB > 100 MB 单文件限制），由 `scripts/fetch_weights.bat`
> 从 GitHub Release 下到 `<项目根>/weights/`，第 3 条直接命中，不用配环境变量。

| `--init` | 加载内容 |
|---|---|
| `encoder` | 仅 H0-mini（173 个张量） |
| `encoder+neck` | 编码器 + 四路 Neck（68 个张量，默认） |
| `full` | 再加 np 解码头（71/73，最后一层 1×1 因 3 类而跳过） |

## 评测与推理（训练完成后）

```powershell
# 单折评测
python evaluate.py --fold 1 --ckpt checkpoints3\fold1\best.pt

# 全 5 折 + 汇总（mean ± std → eval3\summary.csv）
python evaluate.py --all-folds --ckpt-dir checkpoints3

# 单张切片推理出彩色预测图 / 三联对比图
python inference.py --ckpt checkpoints3\fold1\best.pt --slide train_001 `
    --out-dir runs3\pred_fold1 --save-pred --overlay-every 20

# 本地 CPU 冒烟（随机权重 + 224 输入）
python evaluate.py --fold 1 --ckpt <任意.pt> --image-size 224 `
    --max-samples 8 --device cpu --overlay-every 2
```

三层指标（`eval3/foldN/metrics.json` + `per_patch.csv` + `summary.csv`）：

| 口径 | 指标 | 用途 |
|---|---|---|
| `full_*` 整图 | 每类 Dice/IoU + 像素精度 | 与训练日志同口径 |
| `tissue_*` 组织区域内 | 同上（Otsu 剔除白色空背景） | **主指标**，写报告用 |
| `patch_acc` | 预测前景占比按 0.01%/5% 分箱 vs 真值 | 「少量 vs 较多」分级准确率 |

## 目录

```
toxpath/
├── weights/                  # HistoPLUS 权重 271 MB（gitignore，随 -IncludeWeights 包分发）
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
│   ├── check_env.py          # 环境体检：GPU/torch/timm/数据/权重/磁盘 + 参数推荐
│   ├── train_local.ps1       # Windows 一键训练（smoke/fold/all/resume，后台运行）
│   ├── train_local.bat       # 上面脚本的启动器（绕过 .ps1 执行策略限制，推荐用这个）
│   ├── sync_github.ps1       # 提交并推送到 GitHub（含大文件护栏）
│   ├── sync_github.bat       # 上面脚本的启动器
│   ├── fetch_weights.ps1     # 从 GitHub Release 下载 271 MB 权重 + SHA256 校验
│   ├── fetch_weights.bat     # 上面脚本的启动器
│   ├── make_code_package.ps1 # 离线备用：打包代码（/带权重）走 U 盘
│   ├── make_code_package.bat # 上面脚本的启动器
│   └── make_3class_masks.py  # 二值 mask → 彩色三分类 mask
├── tests/
│   ├── test_dataset.py       # Dataset 35 项测试
│   └── test_model_forward.py # 模型形状/反传/冻结/权重往返 8 项测试
├── docs/
│   └── 本地训练SOP.md        # Windows + 4090 全流程（电源设置、续训、故障排查）
├── paths.py                  # 数据根/权重统一解析（命令行>环境变量>项目内>历史）
├── inference.py              # 推理：彩色预测图 / 三联对比图 / Otsu 组织掩膜
├── evaluate.py               # 正式评测：整图 + 组织区域 + patch 分级，5 折汇总
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

