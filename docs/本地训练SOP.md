# ToxPath 实验室本地训练 SOP

> 适用机器：实验室 Windows 工作站，单卡 **RTX 4090 24 GB**，通过远程桌面 / 向日葵操作
> 适用代码：main 分支（连通域三分类标签、TensorBoard 监控、默认开梯度检查点、`dice_binary` 指标、`-ExtraArgs` 参数透传）
> 版本自检：`Select-String -Path scripts\run_train.ps1 -Pattern "ExtraArgs"` 没输出就说明代码旧了，先 `git pull`。
> 本文是**照着敲就能跑**的操作手册；原理和取舍看 `README.md`，这里只讲怎么做、怎么判断正常、出问题查哪。

---

## 0. 一页速查

| 我要做什么 | 命令（在仓库根目录、已 `conda activate toxpath`） |
|---|---|
| 体检：这台机器能不能开训 | `nvidia-smi` + 第 1 节逐项核对 |
| 拉最新代码 | `git pull` |
| **数据在项目外 / 换了盘** | `python paths.py` 确认解析结果；临时指定用 `-DataRoot "D:\data\toxpath"`（见 1.5） |
| 数据换过 / 第一次跑 | `python scripts\make_3class_masks.py --dry-run` 看分布 → 去掉 `--dry-run` 生成 |
| 冒烟（约 1 分钟，**没显卡也能跑**） | `.\scripts\run_train.ps1 -Mode smoke`；验权重加载加 `-WithWeights`（见 1.7） |
| 单折训练 | `.\scripts\run_train.ps1 -Mode fold -Fold 1 -Epochs 12` |
| 临时改某个训练参数 | `.\scripts\run_train.ps1 -Mode fold -ExtraArgs "--samples-per-epoch","3500"`（见 3.4） |
| 五折训练（过夜） | `.\scripts\run_train.ps1 -Mode all -Epochs 12` |
| 断点续训 | `.\scripts\run_train.ps1 -Mode resume -Fold 1 -Epochs 24` |
| 起监控看板 | `.\scripts\run_train.ps1 -Mode tb` → 浏览器开 `http://127.0.0.1:6006` |
| 盯日志 | `Get-Content logs3\train_fold_*.out.log -Wait -Tail 20` |
| 停训练 | `Stop-Process -Id <启动时打印的 PID>` |
| 笔记本改完代码推上来 | （笔记本上）`scripts\sync_github.bat -Message "..."` |

**三条铁律：**

1. **开训前必跑一次冒烟**，1 分钟能省掉 2 小时的错。
2. **关远程桌面安全，电脑睡眠致命**——挂机前先做第 2 节。
3. **看 `dice_fg`，不看 `acc`**。背景占 83.6%，全猜背景就有 0.836 的像素精度。

---

## 1. 一次性准备（换机器才做，约 30 分钟）

### 1.1 显卡与占用

```powershell
nvidia-smi
nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv
```

要看到 `RTX 4090`、`24564MiB`、驱动 550 以上。第二条命令**有输出说明卡被别人占着**，等它跑完再开训（两个人同时训会双输）。

### 1.2 Python 环境

```powershell
conda create -n toxpath python=3.10 -y
conda activate toxpath
pip install --index-url https://download.pytorch.org/whl/cu124 torch torchvision
pip install timm==1.0.8 numpy Pillow tensorboard
```

**`timm` 必须锁 1.0.8**，别的版本会改 ViT block 实现或 `out_indices` 语义，权重加载会静默错位。

验证：

```powershell
python -c "import torch,timm;print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0));print('timm', timm.__version__)"
# 期望：2.5.x+cu124 True NVIDIA GeForce RTX 4090 / timm 1.0.8
```

`conda activate` 报「无法识别」→ 先 `conda init powershell` 再重开窗口。

### 1.3 代码

```powershell
git clone git@github.com:shanshicoder/toxpath.git D:\work\toxpath
cd D:\work\toxpath
```

私有仓库要先配 SSH key（`ssh-keygen -t ed25519 -C "lab-4090"`，公钥贴到 GitHub → Settings → SSH keys），`ssh -T git@github.com` 通了再 clone。22 端口被封就改走 443，见 `%USERPROFILE%\.ssh\config`：

```
Host github.com
    HostName ssh.github.com
    Port 443
    User git
```

### 1.4 权重（271 MB，不进仓库）

放 `weights\histoplus_cellvit_segmentor_20x.pt`。`paths.py` 会按「命令行 → 环境变量 → `weights\` → 项目根 → 上一级 → 历史绝对路径」自动找，找不到会列出全部候选。

从 GitHub Release `weights-v1` 下载，或 U 盘拷。**拷完必须核对哈希**（U 盘只拷一半也会"看起来成功"）：

```powershell
Get-FileHash .\weights\histoplus_cellvit_segmentor_20x.pt -Algorithm SHA256
# 必须是：AA19DC31C109C0A4FCF7BD773559C45480CEE792A331AE0A9FFE42EB08267BFC
# 字节数：284507558
```

### 1.5 数据（放在项目外也可以）

**规则只有一条：`--data-root` 指的是「包含 `train_patches/` 的那一层」，不是 `train_patches` 本身。** 目录名不必叫 `toxpath`，位置不必在项目里。

```
D:\data\toxpath\                                  ← 这就是 data-root
├── train_patches\{slide}\1.0\{X}_{Y}.jpeg         # 1024×1024，训练读这个
├── train_masks_classes\{slide}\1.0\{X}_{Y}.png    # 彩色三分类 mask（黑/绿/红）
├── train_masks\...                                # 原始二值 mask，重造标签时用
├── _patch_labels_3class.csv                       # 标签索引
├── train_images\ train_labels\                    # 金字塔 TIFF + GeoJSON，训练不读
```

三种指定方式，按优先级从高到低：

| 方式 | 怎么写 | 适合 |
|---|---|---|
| 命令行 | `--data-root "D:\data\toxpath"`<br>`run_train.ps1` 用 `-DataRoot "D:\data\toxpath"` | 临时用一次、或同一台机器跑多份数据 |
| 环境变量 | `setx TOXPATH_DATA_ROOT "D:\data\toxpath"`（**重开终端才生效**） | 这台机器长期只用这一份数据 |
| 放项目内 | `<项目根>\data\toxpath\` | 换机器直接拷整个项目，最省事 |

命令行永远覆盖环境变量。还有两个兜底候选（`<项目根>\../data\toxpath`、笔记本的历史路径），只在前三个都不存在时才会命中。

**先自检，别等训练跑了 3 分钟才报错：**

```powershell
python paths.py
```

输出会列出当前解析结果和**全部候选**，命中的标 `OK`、不存在的标 `--`：

```
[paths] 当前解析：
  project_root : E:\project\toxpath
  data_root    : D:\data\toxpath          ← 确认这里是你要的那份
  histoplus    : E:\project\toxpath\weights\histoplus_cellvit_segmentor_20x.pt
```

**产物目录与数据位置无关。** `checkpoints3/`、`logs3/`、`logs3/tb` 的默认值是**项目根下的绝对路径**（写死在 `train_3class.py:101-103`），数据在 D 盘也照样落在项目里。想让产物跟着数据走，必须显式传：

```powershell
.\scripts\run_train.ps1 -Mode fold -Fold 1 -Epochs 12 -DataRoot "D:\data\toxpath_73" `
  -ExtraArgs @("--out-dir","D:\runs\toxpath_73\ckpt",
               "--log-dir","D:\runs\toxpath_73\log",
               "--tb-dir","D:\runs\toxpath_73\tb")
```

**相对路径的坑（已实测）**：显式传 `--out-dir relckpt` 这样的相对路径时，它相对的是**当前工作目录**，不是项目根。`cd E:\tmp` 后跑，产物就落在 `E:\tmp\relckpt`。所以跨目录启动时，**输出目录一律写绝对路径**；用 `run_train.ps1` 则无此问题（它内部 `Set-Location` 到项目根）。

**用外部盘时的三条经验：**

1. **网络盘 / 移动硬盘要把 `--num-workers` 降下来**（`-NumWorkers 2` 或 `3`）。4 个进程排队读同一块慢盘，GPU 会空转等 I/O，比单进程还慢。
2. **路径含空格没问题**：`-DataRoot "E:\my data\toxpath"` 实测引号被正确剥离，Python 收到的是干净路径。
3. **别把 `--data-root` 指到 `train_patches` 那一层**，会报 `其下是否有 train_patches：False`——这句报错就是在提示高了一层。

### 1.6 生成三分类标签（换数据后必做）

二值 mask 不会自动变成三分类，必须先跑一次标签脚本。规则（4 邻接连通域，定义在 `labels.py`，训练/导出/可视化共用同一份）：

| 条件 | 类别 |
|---|---|
| 前景 < 128 px（≈11×11，标注噪声） | 0 背景 |
| 最大连通域 ≥ 50,000 px（≈4.8% patch） | 2 大块 |
| 最大连通域 ≥ 20,000 px **且** 域数 ≤ 2 **且** 主体占比 ≥ 80% | 2 大块 |
| 其余有前景 | 1 零星 |

```powershell
python scripts\make_3class_masks.py --dry-run     # 只统计分布，约 45 秒（5 切片）
python scripts\make_3class_masks.py               # 确认后写文件，约 5 分钟（5 切片）
```

数据在项目外时照常加 `--data-root`，**输出会写回那个数据根**（`<data-root>\train_masks_classes\` 和 `<data-root>\_patch_labels_3class.csv`），不会跑到项目目录里。已实测。

**看 `--dry-run` 的分布再决定阈值**。本地 4953 张的结果是 **4140 / 149 / 664**（83.6% / 3.0% / 13.4%）。上 73 张 WSI、约 7 万 patch 时，如果分辨率或标注风格变了导致类别 1 塌到接近 0，就调阈值：

```powershell
python scripts\make_3class_masks.py --big-area 80000 --mid-area 30000   # 更严格
```

### 1.7 冒烟（每次开训前都跑）

```powershell
.\scripts\run_train.ps1 -Mode smoke
```

固定用 CPU + 224 输入 + 8 个样本，只验「数据能读、标签能对、loss 能算、checkpoint 能存」，约 1 分钟。**通过标准：** 无报错、日志末行有 `[Fold 1 Epoch 1/1]` 汇总、`_smoke\ckpt\last.pt` 已生成（`best.pt` 只在验证集存在前景时才写，没有属正常）。

**这个冒烟在没显卡的笔记本上一样能跑**（固定 `--device cpu`），本机实测 47 秒。两个常用变体：

```powershell
# 顺带验 HistoPLUS 权重加载（271 MB，最易在新机器上出问题的一环）—— 实测 44 秒
.\scripts\run_train.ps1 -Mode smoke -WithWeights

# 验一份放在别处的数据能不能喂进去（不传就用 paths.py 的默认解析，注意别验错了数据）
.\scripts\run_train.ps1 -Mode smoke -DataRoot "D:\data\toxpath_73"
```

`-WithWeights` 的通过标准是日志里出现这两行，`missing=0` 与 `shape_mismatch=0` 是关键：

```
encoder : loaded=173, missing=0, unexpected=20, shape_mismatch=0
neck    : loaded=68 (strict)
```

**本机能验到哪一步、验不到什么**（这是笔记本上唯一值得做的事）：

| ✅ 能验 | ❌ 验不了 |
|---|---|
| 数据读取、标签反解、切片分折 | 真实训练速度 |
| 模型构建 + 权重加载 | 显存够不够（2 GB 卡与 24 GB 卡完全不同） |
| 前向 / 反向 / loss 下降 / checkpoint 落盘 | CUDA、bf16、梯度累积的显存行为 |
| TensorBoard 事件文件写入 | 加了梯度检查点后的峰值 |

**别指望在本机把模型训出来。** 实测 CPU 上 0.30 it/s（≈3.3 s/步，224 输入），1024 输入是 107 s/步；单折 12 轮 × 800 样本 = 9600 步，1024 下要 **11.9 天**，而 4090 上同口径约 3.8 小时。

**笔记本的独显也帮不上忙**：常见环境装的是 `torch==2.x+cpu`（`torch.__version__` 末尾带 `+cpu`），`torch.cuda.is_available()` 直接是 `False`；即便换成 CUDA 版，MX350 只有 2 GB 显存，而 1024 输入峰值约 22 GB。不值得在这上面折腾，验 CUDA 请到 4090 那台机器上跑。

> 冒烟产物约 **436 MB**（`last.pt` 含 114 M 参数 + 优化器状态）。跑完随手清：`Remove-Item _smoke -Recurse -Force`。

要额外验 CUDA 与显存，直接跑一次 GPU 小样本：

```powershell
python train_3class.py --fold 1 --epochs 1 --image-size 512 `
  --max-train-samples 8 --max-val-samples 4 --samples-per-epoch 8 `
  --num-workers 0 --init none `
  --out-dir _smoke\ckpt_gpu --log-dir _smoke\log_gpu --tb-dir _smoke\tb_gpu
```

看开头两行 `device=cuda` / `gpu=NVIDIA GeForce RTX 4090 vram=24.0 GB`，以及 `nvidia-smi` 里的显存峰值。

---

## 2. 挂机前必做：别让机器睡过去（管理员 PowerShell）

| 你做的事 | 训练进程 |
|---|---|
| 关远程桌面窗口 / 向日葵断连 | **继续跑**（会话锁定，进程活着） |
| 电脑睡眠 / 休眠 | **死** |
| Windows 自动重启更新 | **死** |

```powershell
powercfg /change standby-timeout-ac 0      # 插电不睡眠
powercfg /change hibernate-timeout-ac 0    # 不休眠
powercfg /change monitor-timeout-ac 15     # 只关显示器
powercfg /query SCHEME_CURRENT SUB_SLEEP   # 复核：AC 那几行应为 0
```

再确认两件事：Windows 更新已「暂停 7 天」；机箱电源不会被同事顺手拔掉。

`run_train.ps1` 用 `Start-Process` 把训练**脱离当前窗口**启动，所以关窗口、断 RDP 都不影响；真正会杀掉它的是睡眠和重启。

---

## 3. 训练

### 3.1 单折（第一次正式训练只跑这一条）

```powershell
cd D:\work\toxpath
conda activate toxpath
.\scripts\run_train.ps1 -Mode fold -Fold 1 -Epochs 12
```

启动后会打印 PID、stdout/stderr 日志路径和四条提示。记住 PID，后面停训练要用。

`-PythonExe` 默认是 `python`（吃当前 PATH）。**不想依赖 activate 就写全路径**：

```powershell
.\scripts\run_train.ps1 -Mode fold -Fold 1 -Epochs 12 `
  -PythonExe "C:\Users\你的用户名\miniconda3\envs\toxpath\python.exe"
```

### 3.2 五折（按切片留一交叉验证）

```powershell
.\scripts\run_train.ps1 -Mode all -Epochs 12
```

同一个进程顺序跑 5 折，前面折的产物不会因为后面折失败而丢。写完 `checkpoints3\summary.json` 并打印 `dice_fg` 均值 ± 标准差。建议下班前挂上，第二天收。

### 3.3 续训

```powershell
.\scripts\run_train.ps1 -Mode resume -Fold 1 -Epochs 24
```

从 `checkpoints3\fold1\last.pt` 接着跑。**注意：`--resume` 只恢复模型权重和 epoch 计数，不恢复优化器状态**（checkpoint 因此从 1.3 GB 降到约 456 MB）。所以续训后头 1–2 轮 loss 会抖一下，属预期。

`last.pt` 每 `--save-every` 轮（默认 2）落一次，最坏丢 2 轮。

### 3.4 参数与预期资源

默认值如下。注意**启动器只自动传 `--fold / --epochs / --num-workers` 和 `--data-root / --grad-checkpointing`**，表里其他参数要自己用 `-ExtraArgs` 透传，否则改了不生效：

```powershell
# 逗号分隔或逐个写都行，两种等价
.\scripts\run_train.ps1 -Mode fold -Fold 1 -Epochs 12 -ExtraArgs "--samples-per-epoch","3500","--val-every","3"
.\scripts\run_train.ps1 -Mode fold -Fold 1 -Epochs 12 -ExtraArgs "--samples-per-epoch,3500,--val-every,3"
```

`-ExtraArgs` 追加在最后，所以同名的参数会覆盖启动器的默认值（`--grad-checkpointing` 除外，它是开关，要关用 `-NoGradCkpt`）。冒烟模式不吃这个参数，它用的是固定的一小串。

| 参数 | 默认 | 说明 |
|---|---|---|
| `--image-size` | 1024 | patch 原生分辨率，不缩放 |
| `--batch-size` | 1 | 1024 输入下单卡只塞得下 1 张 |
| `--grad-accum` | 8 | 等效 batch = 8 |
| `--samples-per-epoch` | 800 | 按类别加权采样，**不是**全量遍历 |
| `--lr` / `--encoder-lr-mult` | 3e-4 / 0.1 | 编码器只用 1/10 学习率 |
| `--class-weights` | 0.15 0.55 0.30 | CE 类权重（背景/零星/大块） |
| 采样权重 | 1.0 / 6.0 / 2.5 | 写死在 `dataset/toxpath3_dataset.py`，≈逆频率平方根 |
| `--amp` | 开 | 4090 上 bfloat16 |
| `--val-every` | 1 | 每轮都验证 |
| `--save-every` | 2 | 每 2 轮存 `last.pt` |
| `--init` | encoder+neck | 从 HistoPLUS 权重初始化 |

跑 5 折的粗糙估算（**第一次以日志实测为准**，下面只是让你知道大概多久）：

| 项目 | 单折 12 epoch | 五折 12 epoch |
|---|---|---|
| 每轮 | 8–15 min（800 训练 + 约 1000 验证） | 同左 |
| 合计 | 约 2 小时 | 8–12 小时 |
| 显存峰值 | 1024 + batch 1：不开 `--grad-checkpointing` 会顶到 22 GB 左右，4090 正处在临界；开了约省一半激活。量级估计，以 `nvidia-smi` 为准 | 同左 |
| 单折磁盘 | best + last ≈ 0.9 GB | 约 4.6 GB |

**显存不够、或卡上还有别人的任务**时（启动器已默认带 `--grad-checkpointing`，48 GB 卡想提速才加 `-NoGradCkpt`；要再挤就得直接跑 python 传这些参数）：

```powershell
python train_3class.py --fold 1 --epochs 12 --grad-checkpointing `
  --num-workers 2 --out-dir checkpoints3 --log-dir logs3
```

真的挤不下再上 `--freeze-encoder`（只训 neck + 解码头），但那已经不是完整微调了，结果要单独标注。

### 3.5 换成 73 张 WSI 时要改什么

数据从 5 个切片扩到 73 个（约 7 万 patch）后，**默认参数会跑不完**——不是报错，是五折要跑六天。三件事必须跟着调。

**① 时长会从"过夜"变成"一周"。** 按 4953 张的实测口径外推（训练 ~1 s/样本、验证 ~0.35 s/张）：

| | 现在（4953 张） | 73 张（约 7 万 patch） |
|---|---:|---:|
| 每折验证 patch | ~990 | ~14,000 |
| 每轮训练（800 样本） | 13 min | — |
| 每轮训练（3500 样本） | — | 58 min |
| 每轮验证 | 6 min | 82 min |
| **每轮合计** | ~19 min | **~2.3 h** |
| **单折 12 轮** | ~3.8 h | **~28 h** |
| **五折** | ~19 h | **~6 天** |

**② `--samples-per-epoch` 必须提到 3500 左右。** 默认 800 在 7 万张里只覆盖 1.1%，12 轮 × 800 = 9600 次采样，对有放回采样来说等于一遍都没看完。

**③ 验证太贵，改成每 3 轮验一次**，验证成本直接降到 1/3。加上这两条，单折降到约 17 h、五折约 3.5 天。

完整命令（单行 `@()` 包起来，避免 PowerShell 跨行解析问题）：

```powershell
.\scripts\run_train.ps1 -Mode fold -Fold 1 -Epochs 12 -DataRoot "D:\data\toxpath_73" -NumWorkers 6 `
  -ExtraArgs @("--samples-per-epoch","3500","--val-every","3","--val-batch-size","2",
               "--out-dir","checkpoints3_73","--log-dir","logs3_73","--tb-dir","logs3_73/tb")
```

`-NumWorkers` 是启动器自己的参数，别塞进 `-ExtraArgs`（塞了也能跑，但同一行会打印两次 `--num-workers`，容易看错）。

**输出目录必须换。** `save_folds` 每次启动都会写 `<out-dir>\splits.json`，沿用 `checkpoints3` 的话，5 切片时代的划分会被 73 张的新划分直接覆盖，旧 checkpoint 的评测就再也对不上划分了。所以上面显式指定了 `checkpoints3_73` / `logs3_73`。

换目录后看板也要跟着换，否则打开是空的：`.\scripts\run_train.ps1 -Mode tb -TbDir "logs3_73\tb"`。这几条相对路径能安全使用，是因为 `run_train.ps1` 会先 `Set-Location` 到项目根；**直接跑 `python train_3class.py` 时相对路径跟着当前目录走，一律写绝对路径**（见 1.5）。

**第一次务必先跑 1 个 epoch 量真实时长**，别拿上表当预算：

```powershell
.\scripts\run_train.ps1 -Mode fold -Fold 1 -Epochs 1 -DataRoot "D:\data\toxpath_73" -NumWorkers 6 `
  -ExtraArgs @("--samples-per-epoch","3500","--val-every","1",
               "--out-dir","checkpoints3_73","--log-dir","logs3_73","--tb-dir","logs3_73/tb")
```

日志里的分钟数 × 12 才是单折真实耗时。7 万张时 I/O 行为会变（更多 JPEG 解码、缓存命中率下降），上表的系数只是外推。

**两个需要提前想清楚的问题：**

- **分折只按 `mean_fg` 排序，不按中心分层。** `splits.py` 现在按肿瘤负荷升序排再蛇形分配，保证每折负荷接近，但**不保证每折的中心构成接近**。73 张如果来自多个中心，可能某一折恰好全是 A 中心的切片——那一折的 `dice_fg` 实际测的是「跨中心性能」，和其他折不可比。你的课题正是跨中心泛化，上全量数据前要么给 `splits.py` 加中心分层，要么在论文里把这一点讲明。
- **各折 patch 数可能很不均衡。** 排序键是 `(mean_fg, n_patch, slide)`，蛇形只平衡负荷。跑前看 `[split]` 打印的 `n_patch` 列，最大/最小差超过 50% 就值得调整排序策略。

---

## 4. 监控：TensorBoard

### 4.1 起看板（另开一个窗口）

```powershell
cd D:\work\toxpath
.\scripts\run_train.ps1 -Mode tb            # 默认 6006；-Port 6010 换端口
```

**默认只看 `logs3\tb`。** 如果你训练时用了别的 `--tb-dir`，看板要跟着指过去，否则页面能打开但一条曲线都没有：

```powershell
.\scripts\run_train.ps1 -Mode tb -TbDir "D:\runs\toxpath_73\tb"
```

浏览器开 `http://127.0.0.1:6006`。局域网里别的机器开 `http://<实验室机器IP>:6006`。

拿本机 IP：

```powershell
(Get-NetIPAddress -AddressFamily IPv4 | Where-Object { $_.IPAddress -notlike "127.*" }).IPAddress
```

**端口被占**（上次没关干净）：`Get-NetTCPConnection -LocalPort 6006 | Select OwningProcess` 找到 PID，`Stop-Process -Id <PID>`。

**局域网打不开** = Windows 防火墙挡了，管理员 PowerShell：

```powershell
netsh advfirewall firewall add rule name=TensorBoard dir=in action=allow protocol=TCP localport=6006
```

跨网段/在校外用：`ssh -L 6006:127.0.0.1:6006 用户名@实验室IP`，然后本地开 `http://127.0.0.1:6006`。

> TensorBoard **无鉴权**，只当内网工具用，别往公网映射。只自己看就把启动参数的 `--host 0.0.0.0` 换成 `--host 127.0.0.1`。

训练和看板完全解耦：先训后开、关掉再开，曲线都在，服务会自动发现新的事件文件。

### 4.2 页面上的三个页签

| 页签 | 内容 |
|---|---|
| SCALARS | `fold1/train_step/loss`（每 `--log-every` 步）、`fold1/train/*` 与 `fold1/val/*`（每 epoch）、`time/it_per_s`、`time/epoch_min`、`train/lr` |
| IMAGES | `fold1/preview`：每轮一张 `原图 \| 真值 \| 预测` 三联图，用 step 滑块回看 |
| TEXT | `fold1/config`：这次运行的尺寸、lr、类别权重、验证切片等快照 |

`val/*` 里有 `loss / dice_c1 / dice_c2 / dice_fg / dice_binary / iou_fg / pixel_acc`，还有一个 `val/dice_fg_allbg_baseline`——**验证集里真值全空的 patch 占比**，等价于「全猜背景」的 dice_fg。曲线贴着这条基线走，说明模型还没学到东西。

**`dice_fg` 和 `dice_binary` 不是一回事，别混用：**

| 指标 | 算法 | 用在哪 |
|---|---|---|
| `dice_fg` | 类 1、类 2 各自 Dice 的**算术平均**（哪类在该批中不存在就不参与平均） | **模型选择**：`best.pt` 按它刷新 |
| `dice_binary` | 把类 1、2 合并成一类前景后重算的**二值 Dice** | 与导出掩膜（只有 0/255）口径一致，**论文报这个** |

差多少取决于类 1 的占比。举个实测例子：某个混淆矩阵下 `dice_c1=0.61`、`dice_c2=0.83`，宏平均是 **0.72**，合并二值却是 **0.86**——相差 0.14。类 1 像素越少、预测越差，这个差距越大。因为你验证集里类 1 本来就只有几十张，宏平均会被它拽下来。

五折各是一个独立 run（`fold1`…`fold5`），左侧勾多个就能把同一指标叠一张图比。**同一折重跑会自动换目录**（`fold1_20261029-203015`），不会和上一次的曲线叠在一起。

### 4.3 读图顺序（每折都这么看）

1. **`train/loss` 有没有在降**，前 20 步别下结论；
2. **`val/dice_fg` 有没有站到 `dice_fg_allbg_baseline` 之上**——这是「到底学没学到」的唯一硬标准；
3. **`val/dice_c2` 跟不跟着涨**（大块肿瘤，最重要的业务指标）；
4. **`val/dice_c1` 只作参考**：类别 1 总量只有 149 张，某些折验证集里只有 5–6 张，这条曲线抖动是数据量决定的，不要据此判断模型崩了；
5. **`time/epoch_min` 突然翻倍** → I/O 卡了或散热降频，去 `nvidia-smi` 看温度和占用；
6. **IMAGES 里的预测图**最直观：如果预测全是黑，回去看 `train/lr` 和 `class-weights` 是否把背景压得太狠。

### 4.4 没装 TensorBoard 也能看

`tensorboard` 只是可选依赖，不装照样训练（`--no-tensorboard` 亦可）。纯文本记录一直在：

```powershell
Get-Content logs3\fold1\train_log.json     # 逐 epoch 的 loss / dice / acc
Get-Content logs3\fold1\history.json       # 该折完整历史
```

---

## 5. 日志、产物、判读

### 5.1 日志在哪

```powershell
Get-Content logs3\train_fold_*.out.log -Wait -Tail 20   # 实时刷屏，Ctrl+C 只退出 tail，不杀训练
Get-Content logs3\train_fold_*.err.log -Tail 40         # 报错看这个
Get-Process -Id <PID>                                    # 还活着吗
```

日志中文不会乱码——启动器强制 `PYTHONUTF8=1` 并去掉了大小写重复的环境变量，这两点在别的机器上是已知坑。

### 5.2 一个正常 epoch 长这样（示意值，别和你的结果比对）

```
[Fold 1 Epoch 3/12] train_loss=0.4123 train_dice_loss=0.2871 val_loss=0.3902 dice_c1=0.0312 dice_c2=0.6188 dice_fg=0.6021 acc=0.9412 (11.4 min)
  ★ 新最佳 dice_fg=0.6021，已保存 best.pt
```

| 字段 | 含义 | 期望 |
|---|---|---|
| `train_loss` | CE + Dice 总损失 | 稳定下降 |
| `val_loss` | 验证损失 | 跟随下降，不持续反弹 |
| `dice_c2` | 大块肿瘤 Dice | 主看，应上升 |
| `dice_fg` | 类 1、类 2 Dice 的算术平均（**不是**二值 Dice，见 4.2） | **模型选择指标**，`best.pt` 按它刷新 |
| `dice_c1` | 零星肿瘤 Dice | 可能接近 0，别慌 |
| `acc` | 像素精度 | 参考值，背景多时天然偏高 |
| 括号分钟 | 单轮耗时 | 突然翻倍 = I/O 或降频 |

这一行**不含** `dice_binary`（控制台行保持短）。要看二值口径去 TensorBoard 的 `foldN/val/dice_binary`，或读 `checkpoints3\foldN\history.json`——每轮的完整指标都在里面。

### 5.3 产物

```
checkpoints3\
├── splits.json              # 折划分（切片 → fold），评测/复现都靠它
├── fold1\best.pt            # dice_fg 最好的那个 epoch
├── fold1\last.pt            # 续训用
├── fold1\history.json       # 该折逐 epoch 指标
└── summary.json             # 五折汇总（--all-folds 才有）
logs3\
├── tb\fold1\                # TensorBoard 事件（曲线 + 预览图）
├── fold1\train_log.json     # 逐 epoch 纯文本备份
└── train_*.out.log / *.err.log
```

### 5.4 本机五折的验证集构成（判读时对照）

| 折 | 验证切片 | 验证 patch | c0 / c1 / c2 | 全空基线 |
|---|---|---:|---|---:|
| fold1 | train_005 | 952 | 921 / 6 / 25 | 0.967 |
| fold2 | train_003 | 842 | 799 / 5 / 38 | 0.949 |
| fold3 | train_002 | 1264 | 1137 / 33 / 94 | 0.900 |
| fold4 | train_004 | 1021 | 852 / 57 / 112 | 0.834 |
| fold5 | train_001 | 874 | 431 / 48 / 395 | 0.493 |

看这张表就知道两件事：

- **各折的 `dice_fg` 不能直接横向比**——fold1 的全空基线是 0.967，fold5 只有 0.493，fold5 的模型比 fold1 难得多。比较时要对着各自基线看。
- **fold1/fold2 的 `dice_c1` 基本没有统计意义**（验证集里只有 5–6 张），汇报时只说 `dice_c2` 和 `dice_fg`。

---

## 6. 中断与恢复

| 场景 | 现象 | 处理 |
|---|---|---|
| 主动 `Stop-Process` | 进程消失 | 从 `last.pt` 续训 |
| 睡眠 / 断电 | 日志停住、GPU 占用归零 | 唤醒后 `-Mode resume` |
| Windows 重启 | 进程没了，日志停在某轮 | 同上，先 `git status` 确认工作区没坏 |
| RDP 断开 | 无影响 | 什么都不用做 |

五折跑到一半挂了：已完成的折结果都在 `checkpoints3\foldN\` 里。**不要直接重跑 `-Mode all`**（会从头覆盖），用 `-Mode fold` 逐折补：

```powershell
.\scripts\run_train.ps1 -Mode fold -Fold 3 -Epochs 12
.\scripts\run_train.ps1 -Mode fold -Fold 4 -Epochs 12
.\scripts\run_train.ps1 -Mode fold -Fold 5 -Epochs 12
```

续训之前先确认 `last.pt` 的时间戳是新的，别对着三天前的旧文件续：

```powershell
Get-ChildItem checkpoints3\fold1\*.pt | Select Name, Length, LastWriteTime
```

---

## 7. 训完怎么算收工

1. `checkpoints3\summary.json` 有内容，5 折齐全；
2. 打开 TensorBoard，5 条 `val/dice_fg` 都在各自 `dice_fg_allbg_baseline` 之上且后段趋平；
3. `checkpoints3\fold*\best.pt` 都在，大小一致（约 456 MB）；
4. 把 `summary.json` 里的均值 ± 标准差抄进实验记录——**这是写论文要报的数**。里面有两套口径别混：`last_dice_fg` 是宏平均，`last_dice_binary` 是二值 Dice。论文写「二值 Dice」就用后者；
5. 磁盘别塞满：`Get-PSDrive D` 看剩余，5 折加日志约 6 GB。

---

## 8. 导出二值肿瘤掩膜

模型是三分类，二值化只是导出时多做一步。三分类 softmax 互斥，所以 `p_肿瘤 = 1 - p_bg = p1 + p2`，`argmax > 0` 与 `p_肿瘤 > 0.5` 等价，**不需要为出二值掩膜再训一个二分类模型**。

但这条等价链只说明「推理时能直接得到二值概率」，**不表示 `dice_fg` 就是二值 Dice**。`dice_fg` 是两类 Dice 的宏平均，二值口径是 `dice_binary`（见 4.2）。论文里报二值 Dice 用后者，别拿 `dice_fg` 顶替——两者能差 0.1 以上。

```powershell
python predict_test.py --test-root D:\data\toxpath_test `
  --ckpt-dir checkpoints3 --tta --min-area 128 `
  --out-dir submission --zip submission.zip
```

- `--ckpt-dir` 把 `fold*\best.pt` 全部拿来做**概率平均集成**；单模型用 `--ckpt`；
- `--tta` 做 hflip/vflip 平均，慢约 3 倍；
- `--skip-existing` 断点续跑；
- 产物：`submission\{slide}\{name}.png`，单通道 L、8-bit、只有 0/255、尺寸同原图，zip 根目录直接挂 `{slide}\`（中间那层 `1.0\` 已去掉）。

**阈值和最小连通域别用默认值硬跑。** 本地 83.5% 的 patch 真值全空，空图上多 1 个假阳像素，该 patch 就从 1.0 掉到 0。要在**验证折**上扫 `--fg-threshold`（0.4/0.5/0.6）和 `--min-area`（0/64/128/256），看综合分和「空真值却预测出前景」的张数两个一起定。

**不要**从 `test_images\*.tif` 重新切图——官方 patch 文件名编码了原图坐标，自己切坐标系不同会对不上真值。

---

## 9. 代码同步

**实验室机器**（拿别人的改动）：

```powershell
cd D:\work\toxpath
git pull
```

**笔记本**（改完推上去）：

```powershell
cd E:\project\toxpath
scripts\sync_github.bat                                      # 提交信息按时间自动生成
scripts\sync_github.bat -Message "fix: 修正 fold5 采样权重"   # 自己写
scripts\sync_github.bat -DryRun                              # 只看它打算做什么
scripts\sync_github.bat -NoCommit                            # 已手动 commit，只推
```

脚本会：校验仓库与身份 → 扫待提交文件拦住 `.pt`/`data\`/大文件 → `add` + `commit` → `push`，最后打印实验室机器该敲的命令。

已知的两个环境坑由脚本内部处理，不用手动干预：

- **死代理**：git 全局配置曾把 `http.proxy` 写死 `127.0.0.1:7890`，而那个端口没监听，连 443 都失败。走 HTTPS 时脚本会 `-c http.proxy= -c https.proxy=` 清空直连。
- **upstream 被改坏**：用 HTTPS URL 推送时不带 `-u`，推完 `fetch` 更新 `origin/main` 再写回跟踪关系，否则 `git status` 会一直显示 ahead。

协议自动选择（`-Via auto`）：先用 `BatchMode=yes` 探 SSH 免密是否可用，私钥带 passphrase 又没装 ssh-agent 时自动转 HTTPS（凭据从 Windows 凭据管理器取）。也可强制 `-Via ssh` / `-Via https`。

> 真需要清掉全局死代理（一次性）：
> `git config --global --unset http.proxy; git config --global --unset https.proxy`

---

## 10. 故障速查

| 现象 | 原因 | 处理 |
|---|---|---|
| `Start-Process : 已添加了具有相同键的项` | 环境变量大小写重复（`HTTPS_PROXY` + `https_proxy`） | 用 `run_train.ps1`（已内置去重，只删真正成对的，不会误删 `SystemRoot`） |
| 日志中文乱码 | 控制台 codepage 与文件编码不一致 | 用启动器（已设 `PYTHONUTF8=1`）；`-Foreground` 时已切 UTF-8 输出 |
| `-Foreground` 跑一半被 FutureWarning 掐断 | PS 5.1 把 stderr 当终止错误 | 启动器已临时设 `$ErrorActionPreference='Continue'`，靠退出码判断 |
| `tensorboard.exe` 找不到 | 不在 PATH | 用 `-Mode tb`，它调 `python -m tensorboard.main` |
| `tensorboard is not installed` | 当前解释器没装 | `python -m pip install "tensorboard>=2.15"`（确认 `python` 就是 `-PythonExe` 那个） |
| `CUDA driver version is insufficient` | 驱动太老 | 更新 NVIDIA 驱动 |
| `CUDA out of memory` | 1024 输入显存不够 | 先确认 `--grad-checkpointing` 开着（启动器默认已带，别用 `-NoGradCkpt`）；再加 `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` 缓解碎片；还不够 `--freeze-encoder`；别只降 `--batch-size`（本来就是 1） |
| DataLoader 卡住不动 | Windows spawn + 大 `num-workers` | `--num-workers 2` 或 `0` |
| `dice_c1` 是 `nan` 或 0 | 该折验证集几乎没有类 1 | 正常，看 `dice_c2` / `dice_fg` |
| `dice_binary` 是 `nan` | 该折验证集一个前景像素都没有 | 正常，说明这折没法学到前景 |
| 改了参数但训练没变化 | 参数直接写在启动器后面（它不认），或用了 `-Mode smoke` | 一律放进 `-ExtraArgs`；加 `-Foreground` 会把完整命令行打出来核对 |
| 找不到数据 / 权重 | `paths.py` 候选都没命中 | 报错会列出全部试过的路径，按提示设 `TOXPATH_DATA_ROOT` / `TOXPATH_HISTOPLUS_CKPT` |
| `--data-root 指向的目录不可用` / `其下是否有 train_patches：False` | 指到了 `train_patches` 那一层，或指到了父目录 | 指向**包含** `train_patches/` 的那一层；`python paths.py` 先自检 |
| 训练跑起来了，但结果明显不是这份数据 | 环境变量指向了另一份数据，命令行没覆盖它 | 看日志头 `data_root` 那一行是否是你以为的路径；临时用 `-DataRoot` 覆盖 |
| 产物落在奇怪的地方（比如 `E:\tmp\checkpoints3`） | 显式传了相对的 `--out-dir`，而当时的工作目录不在项目根 | 一律写绝对路径；或改用 `run_train.ps1`（内部会切到项目根） |
| 看板能打开，但一条曲线都没有 | 训练时用了别的 `--tb-dir`，看板还在看 `logs3\tb` | `-Mode tb -TbDir "<训练时的 tb 目录>"`，启动日志里的 `[logdir ]` 会回显实际目录 |
| 训练日志停在某轮、GPU 占用 0 | 睡眠或重启 | 第 2 节设电源策略，`-Mode resume` 续上 |
| 同折曲线上有两条重叠 | 用了旧事件目录 | 新代码重跑会自动带时间戳后缀；清 `logs3\tb\foldN_*` 里不要的 |
| 局域网打不开 6006 | 防火墙 | 第 4.1 节的 `netsh` 命令 |
| 推 GitHub 报 `Failed to connect ... port 443` | 本机到 GitHub 的 443 被挡，或 DNS 落到不可达的 IP（实测 `20.205.243.166` 不通、`140.82.112.3` 通） | 先 `Test-NetConnection github.com -Port 22`：22 通就 `sync_github.bat -Via ssh`；只是某个 IP 不通，管理员 PowerShell 临时往 `C:\Windows\System32\drivers\etc\hosts` 加一行 `140.82.112.3 github.com`，推完删掉；都行不通就开代理。**commit 已在本地，网络恢复后 `sync_github.bat -NoCommit` 即可** |

---

## 11. 收工清单（打印贴墙）

**开训前**

- [ ] `nvidia-smi` 有卡、没人占
- [ ] `conda activate toxpath`（或 `-PythonExe` 全路径）
- [ ] `git pull` 拿的是最新代码
- [ ] `data\toxpath` 齐、`_patch_labels_3class.csv` 与三分类 mask 是同一次生成的
- [ ] 换过数据就跑过 `make_3class_masks.py --dry-run` 并看过分布
- [ ] `-Mode smoke` 通过
- [ ] 电源策略已设（`powercfg` 三条）
- [ ] Windows 更新已暂停

**训练中**

- [ ] `val/dice_fg` 站在 `dice_fg_allbg_baseline` 之上
- [ ] `time/epoch_min` 稳定
- [ ] 磁盘剩余 > 10 GB

**收工**

- [ ] `summary.json` 齐全，均值 ± 标准差已记录
- [ ] 报「二值 Dice」用的是 `last_dice_binary`，没拿 `dice_fg` 顶替
- [ ] `best.pt` 都在
- [ ] `logs3\` 与 `checkpoints3\` 已备份（或确认不需要）
- [ ] 代码改动已 `sync_github.bat` 推走

---

## 附录：本次实测事实（写论文可引用）

- 本地 5 个切片、**4953 张** 1024×1024 patch；二分类前景与三分类前景（1+2）逐像素一致的有 **4951 张**，3 张前景 < 128 px 被并入背景（83 / 100 / 110 px）。
- 类别分布 **4140 / 149 / 664**（83.6% / 3.0% / 13.4%）；**4137 张（83.5%）真值全空**，所以「全猜背景」的 dice_fg 基线 ≈ **0.835**。
- 816 张有前景的 patch 中，75% 只有 1 个连通域（域数 p50=1 / p95=3 / p99=7），最大连通域面积 p25=65k / p50=354k。
- 连通域统计（纯 numpy 4 邻接行程编码 + 并查集）与 `scipy.ndimage.label`(4 邻接) 在 60 张随机 + 20 张真实 mask 上结果完全一致。
- 分类规则可复现：200 个碎点共 2 万 px → 类 1；2.5 万 px 整块 → 类 2；2.1 万 px 切成 3 块 → 类 1。
- **`dice_fg` ≠ 二值 Dice**：前者是类 1、类 2 Dice 的算术平均，后者是合并 1、2 后按 2×2 混淆矩阵算的 `dice_binary`。在一个实测混淆矩阵上两者为 **0.7197 / 0.8571**，差 0.1374；类 1 越稀少、越难分，差距越大。
- CPU（8 逻辑核 / 15.8 GB，4 线程）参考：224 输入 3.6 s/样本、512 输入 16.4 s / 3.5 GB、1024 输入 107 s / 10.4 GB。**CPU 只配跑 224 冒烟**，正式训练必须上 4090。
- HistoPLUS 权重 `histoplus_cellvit_segmentor_20x.pt`：284,507,558 字节，SHA256 `AA19DC31C109C0A4FCF7BD773559C45480CEE792A331AE0A9FFE42EB08267BFC`。
- 模型规模：encoder 85.7 M + neck 14.3 M + decoder 14.0 M = **114.1 M** 参数。
