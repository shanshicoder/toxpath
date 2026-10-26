# AutoDL 4090D 训练 SOP

> 适用：ToxPath 三分类（1024 输入）+ HistoPLUS 架构
> 本地项目：`E:\project\toxpath`
> 服务器目录：`/root/autodl-tmp/toxpath`

---

## 0. 上传清单（先看这张表）

### 必须上传（约 1.49 GB）

| 本地路径 | 大小 | 用途 |
|---|---:|---|
| `toxpath\model\` | 0.1 MB | 模型代码（8 个 .py） |
| `toxpath\dataset\` | 0.05 MB | Dataset 代码 |
| `toxpath\scripts\` | 0.02 MB | 生成 mask 的脚本（服务器上一般不用跑） |
| `toxpath\tests\` | 0.03 MB | 自检用 |
| `toxpath\train_3class.py` | 24 KB | 训练入口 |
| `toxpath\requirements.txt` | 1 KB | 依赖说明 |
| `toxpath\README.md` | 4 KB | 说明 |
| `toxpath\data\toxpath\train_patches\` | **1,221 MB** | 4953 张 JPEG（训练输入） |
| `toxpath\data\toxpath\train_masks_classes\` | **16.9 MB** | 4953 张三分类彩色 mask |
| `toxpath\data\toxpath\_patch_labels_3class.csv` | 0.2 MB | patch 类别索引（采样权重） |
| `histoplus_cellvit_segmentor_20x.pt` | **271 MB** | HistoPLUS 预训练权重 |

### 可选上传

| 路径 | 大小 | 说明 |
|---|---:|---|
| `toxpath\data\toxpath\train_masks\` | 14.4 MB | 原始二值 mask，只有需要重新生成三分类 mask 时才要 |

### 不要上传

| 路径 | 原因 |
|---|---|
| `toxpath\data\toxpath\train_images\` | 2.5 GB 金字塔 TIFF，训练不读 |
| `toxpath\data\toxpath\train_labels\` | GeoJSON 原始标注，训练不读 |
| `toxpath\_deprecated\` | 旧方案存档 |
| `toxpath\.idea\`、`mask_visual_check.png`、`__pycache__` | IDE/临时文件 |

---

## 1. 本地打包

```powershell
cd E:\project\toxpath
powershell -ExecutionPolicy Bypass -File scripts\make_upload_package.ps1
```

脚本会：

- 按白名单打包成 `E:\project\toxpath_upload.tar`（约 1.5 GB，**不压缩**，因为 JPEG 已压缩）；
- 打印每个条目的体积、总大小、SHA256；
- 自动排除 `__pycache__` / `*.pyc`。

记下脚本输出的 **SHA256**，服务器上要核对。

> 如果 `tar` 报错，可用 7-Zip 手动打包同样的目录，生成 `toxpath_upload.7z` 或 `.zip`。

---

## 2. 上传到 AutoDL

**建议先用「无卡模式」开机上传**，上传完再关机切「有卡模式」，省钱。

### 方式 A：scp（命令行）

AutoDL 控制台 → 实例 → 登录指令，形如：

```bash
ssh -p 12345 root@region-1.autodl.com
```

本地 PowerShell：

```powershell
scp -P 12345 "E:\project\toxpath_upload.tar" root@region-1.autodl.com:/root/autodl-tmp/
```

> `scp` 不支持断点续传；1.5 GB 一般几分钟。网络不稳时用 WinSCP 或 `rsync`（见方式 B）。

### 方式 B：WinSCP / FileZilla（推荐，可续传）

- 协议 SFTP，主机、端口、用户名 `root`、密码同 SSH；
- 把 `toxpath_upload.tar` 拖到 `/root/autodl-tmp/`；
- 断线后重新连接可继续。

### 方式 C：AutoDL JupyterLab 上传

打开 JupyterLab → 左侧文件树 → 进入 `/root/autodl-tmp` → 上传。
**不推荐 1.5 GB 大文件**（浏览器容易断），小文件可以。

---

## 3. 服务器上解压与校验

```bash
cd /root/autodl-tmp
sha256sum toxpath_upload.tar          # 与本地脚本输出对比，必须一致
tar -xf toxpath_upload.tar
ls -lh
```

解压后应有：

```
/root/autodl-tmp/
├── toxpath/
│   ├── model/  dataset/  scripts/  tests/
│   ├── train_3class.py  requirements.txt  README.md
│   └── data/toxpath/{train_patches,train_masks_classes,_patch_labels_3class.csv}
└── histoplus_cellvit_segmentor_20x.pt
```

**数量校验：**

```bash
cd /root/autodl-tmp/toxpath/data/toxpath
find train_patches -type f | wc -l          # 期望 4953
find train_masks_classes -type f | wc -l    # 期望 4953
wc -l _patch_labels_3class.csv              # 期望 4954（含表头）
```

---

## 4. 环境准备

```bash
nvidia-smi                                   # 确认 4090D 与显存
python -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0))"
python -c "import timm; print(timm.__version__)"
```

要求：

- `torch.cuda.is_available()` 为 `True`；
- `timm == 1.0.8`（AutoDL 镜像自带的版本可能不同，必须锁版本）。

如果 timm 不是 1.0.8：

```bash
pip install timm==1.0.8
```

如果 torch 是 CPU 版（极少见）：

```bash
pip install --index-url https://download.pytorch.org/whl/cu124 torch torchvision
```

---

## 5. 冒烟验证（必做，约 5 分钟）

### 5.1 代码自检

```bash
cd /root/autodl-tmp/toxpath
python tests/test_dataset.py         # 期望 35/35 OK
python tests/test_model_forward.py   # 期望 8/8 通过
```

### 5.2 训练冒烟（32 样本 1 epoch）

```bash
nohup python train_3class.py \
  --data-root /root/autodl-tmp/toxpath/data/toxpath \
  --histoplus-ckpt /root/autodl-tmp/histoplus_cellvit_segmentor_20x.pt \
  --fold 1 --epochs 1 --batch-size 1 --grad-accum 4 \
  --max-train-samples 32 --max-val-samples 8 \
  --amp --grad-checkpointing --num-workers 4 \
  --out-dir /root/autodl-tmp/ckpt3_smoke --log-dir /root/autodl-tmp/logs3_smoke \
  > smoke.log 2>&1 &
```

观察：

```bash
tail -f smoke.log
watch -n 1 nvidia-smi
```

通过标准：

- 日志出现 `amp=True`、权重加载摘要（encoder 173 / neck 68）；
- 显存峰值 **< 22 GB**；
- 单步耗时记录一下（后面估总时长用）；
- `train_loss` 是有限值，不是 `nan`。

---

## 6. 正式训练

### 6.1 单折（先跑 1 折看曲线）

```bash
cd /root/autodl-tmp/toxpath
nohup python train_3class.py \
  --data-root /root/autodl-tmp/toxpath/data/toxpath \
  --histoplus-ckpt /root/autodl-tmp/histoplus_cellvit_segmentor_20x.pt \
  --fold 1 --epochs 20 --batch-size 1 --grad-accum 8 \
  --samples-per-epoch 800 --val-every 2 \
  --amp --grad-checkpointing --num-workers 4 \
  --out-dir /root/autodl-tmp/ckpt3 --log-dir /root/autodl-tmp/logs3 \
  --save-every 5 \
  > logs3_fold1.log 2>&1 &
```

### 6.2 5 折

把 `--fold 1` 换成 `--all-folds`（同一个进程顺序跑 5 折）：

```bash
nohup python train_3class.py \
  --data-root /root/autodl-tmp/toxpath/data/toxpath \
  --histoplus-ckpt /root/autodl-tmp/histoplus_cellvit_segmentor_20x.pt \
  --all-folds --epochs 20 --batch-size 1 --grad-accum 8 \
  --samples-per-epoch 800 --val-every 2 \
  --amp --grad-checkpointing --num-workers 4 \
  --out-dir /root/autodl-tmp/ckpt3 --log-dir /root/autodl-tmp/logs3 \
  --save-every 5 \
  > logs3_all.log 2>&1 &
```

**参数含义：**

| 参数 | 说明 |
|---|---|
| `--batch-size 1 --grad-accum 8` | 1024 输入显存有限，用累积等效 batch 8 |
| `--samples-per-epoch 800` | 每 epoch 采样 800 个 patch（不是全量 4000） |
| `--val-every 2` | 每 2 个 epoch 验证一次，省时间 |
| `--amp` | 4090D 上用 bfloat16 |
| `--grad-checkpointing` | 省显存（已修复 timm 包装器问题） |
| `--save-every 5` | 每 5 epoch 存一次 last.pt |

**预估：** 单折 20 epoch 约 2 小时，5 折约 **8–10 小时**；`ckpt3` 约 4.6 GB（每折 best + last 各 456 MB）。

---

## 7. 训练中监控

```bash
tail -f logs3_all.log                 # 训练日志
nvidia-smi                            # 显存 / 利用率
df -h /root/autodl-tmp                # 磁盘
ls -lh ckpt3/fold1/                   # checkpoint
```

每个 epoch 关注：

- `train_loss` 是否下降；
- `dice_c2`（较多肿瘤）与 `dice_fg`；
- `dice_c1` 可能是 `nan` 或 0（类别 1 太少），属正常，不要据此判断崩溃。

---

## 8. 中断与续训

训练被中断（实例关机、进程被杀）后：

```bash
nohup python train_3class.py \
  --data-root /root/autodl-tmp/toxpath/data/toxpath \
  --histoplus-ckpt /root/autodl-tmp/histoplus_cellvit_segmentor_20x.pt \
  --fold 1 --epochs 20 --batch-size 1 --grad-accum 8 \
  --samples-per-epoch 800 --val-every 2 \
  --amp --grad-checkpointing --num-workers 4 \
  --out-dir /root/autodl-tmp/ckpt3 --log-dir /root/autodl-tmp/logs3 \
  --save-every 5 \
  --resume /root/autodl-tmp/ckpt3/fold1/last.pt \
  > logs3_fold1_resume.log 2>&1 &
```

> `--resume` 只恢复模型权重与 epoch，**不恢复优化器状态**（为了把 checkpoint 从 1.3 GB 降到 456 MB）。中途续训会有轻微抖动，属预期。

---

## 9. 下载结果

```bash
cd /root/autodl-tmp
# 只打包每折 best + 曲线 + 汇总（约 2.3 GB）
tar -czf results3.tar.gz \
  ckpt3/fold*/best.pt ckpt3/fold*/history.json ckpt3/summary.json logs3

ls -lh results3.tar.gz
```

在本地 PowerShell：

```powershell
scp -P 12345 root@region-1.autodl.com:/root/autodl-tmp/results3.tar.gz E:\project\
```

> **实例释放会清空数据盘**，下载完再释放；仅关机则数据保留。

---

## 10. 常见问题

| 现象 | 原因 / 处理 |
|---|---|
| `FileNotFoundError: 三分类 mask 目录不存在` | `--data-root` 写错，或 tar 没解压到 `/root/autodl-tmp` |
| `找不到 HistoPLUS 权重` | `--histoplus-ckpt` 路径不对；确认 `.pt` 在数据盘根目录 |
| `AttributeError: FeatureGetterNet 不支持梯度检查点` | 用的是旧代码，拉最新 `model/encoder.py` |
| `CUDA out of memory` | 确认 `--grad-checkpointing`；还不行加 `--freeze-encoder`；再不行 `--image-size 512` |
| 训练很慢 | 确认日志里 `amp=True`；`nvidia-smi` 看 GPU 利用率是否 < 50% |
| DataLoader 卡住 / BrokenPipe | `--num-workers 2`，仍不行改 `0` |
| 磁盘满 | `--out-dir` 指到 `/root/autodl-tmp`；删掉 `ckpt3_smoke`、`logs3_smoke` |
| 指标 `nan` | 某折验证集没有类别 1（如 fold 5 只有 4 张），看 `dice_c2` 即可 |
| 断点后 loss 突然变大 | 正常，优化器状态没恢复；跑几个 epoch 会回来 |

---

## 11. 一次跑通的检查清单

- [ ] 本地 `sha256sum` 与服务器一致
- [ ] `train_patches` = 4953、`train_masks_classes` = 4953、csv = 4954 行
- [ ] `torch.cuda.is_available() == True`，`timm == 1.0.8`
- [ ] `tests/test_dataset.py` 35/35、`tests/test_model_forward.py` 8/8
- [ ] 32 样本冒烟显存 < 22 GB、loss 有限
- [ ] 单折 20 epoch 曲线正常（`dice_c2` 上升）
- [ ] 5 折跑完，`ckpt3/summary.json` 有 5 条记录
- [ ] `results3.tar.gz` 已下载，再释放实例
