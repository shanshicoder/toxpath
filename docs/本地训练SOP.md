# ToxPath 实验室本地训练 SOP

> 适用机器：实验室 Windows 工作站，单卡 **RTX 4090 24 GB**，通过远程桌面 / 向日葵操作
> 适用代码：`7b8df82` 及之后（三分类标签已改为连通域规则，训练监控走 TensorBoard）
> 本文是**照着敲就能跑**的操作手册；原理和取舍看 `README.md`，这里只讲怎么做、怎么判断正常、出问题查哪。

---

## 0. 一页速查

| 我要做什么 | 命令（在仓库根目录、已 `conda activate toxpath`） |
|---|---|
| 体检：这台机器能不能开训 | `nvidia-smi` + 第 1 节逐项核对 |
| 拉最新代码 | `git pull` |
| 数据换过 / 第一次跑 | `python scripts\make_3class_masks.py --dry-run` 看分布 → 去掉 `--dry-run` 生成 |
| 冒烟（约 1 分钟） | `.\scripts\run_train.ps1 -Mode smoke` |
| 单折训练 | `.\scripts\run_train.ps1 -Mode fold -Fold 1 -Epochs 12` |
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

### 1.5 数据

```
D:\data\toxpath\
├── train_patches\{slide}\1.0\{X}_{Y}.jpeg        # 1024×1024，训练读这个
├── train_masks_classes\{slide}\1.0\{X}_{Y}.png   # 彩色三分类 mask（黑/绿/红）
├── train_masks\...                               # 原始二值 mask，重造标签时用
├── _patch_labels_3class.csv                      # 标签索引
├── train_images\ train_labels\                   # 金字塔 TIFF + GeoJSON，训练不读
```

数据放哪都行，别改代码。推荐设一次环境变量（重开窗口生效）：

```powershell
setx TOXPATH_DATA_ROOT "D:\data\toxpath"
```

只想临时指定就每条命令加 `--data-root "D:\data\toxpath"`，`run_train.ps1` 对应参数是 `-DataRoot`。

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

**看 `--dry-run` 的分布再决定阈值**。本地 4953 张的结果是 **4140 / 149 / 664**（83.6% / 3.0% / 13.4%）。上 73 张 WSI、约 7 万 patch 时，如果分辨率或标注风格变了导致类别 1 塌到接近 0，就调阈值：

```powershell
python scripts\make_3class_masks.py --big-area 80000 --mid-area 30000   # 更严格
```

### 1.7 冒烟（每次开训前都跑）

```powershell
.\scripts\run_train.ps1 -Mode smoke
```

固定用 CPU + 224 输入 + 8 个样本，只验「数据能读、标签能对、loss 能算、checkpoint 能存」，约 1 分钟。**通过标准：** 无报错、`_smoke\ckpt\` 里有 `best.pt`、日志末行是正常的 epoch 汇总。

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

默认值（都能用命令行覆盖）：

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
| 显存峰值 | 10–14 GB；加 `--grad-checkpointing` 降到 6–8 GB | 同左 |
| 单折磁盘 | best + last ≈ 0.9 GB | 约 4.6 GB |

**显存不够、或卡上还有别人的任务**时（`run_train.ps1` 没透传这些参数，直接跑 python）：

```powershell
python train_3class.py --fold 1 --epochs 12 --grad-checkpointing `
  --num-workers 2 --out-dir checkpoints3 --log-dir logs3
```

真的挤不下再上 `--freeze-encoder`（只训 neck + 解码头），但那已经不是完整微调了，结果要单独标注。

---

## 4. 监控：TensorBoard

### 4.1 起看板（另开一个窗口）

```powershell
cd D:\work\toxpath
.\scripts\run_train.ps1 -Mode tb            # 默认 6006；-Port 6010 换端口
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

`val/*` 里有 `loss / dice_c1 / dice_c2 / dice_fg / iou_fg / pixel_acc`，还有一个 `val/dice_fg_allbg_baseline`——**验证集里真值全空的 patch 占比**，等价于「全猜背景」的 dice_fg。曲线贴着这条基线走，说明模型还没学到东西。

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

### 5.2 一个正常 epoch 长这样

```
[Fold 1 Epoch 3/12] train_loss=0.4123 train_dice_loss=0.2871 val_loss=0.3902 dice_c1=0.0312 dice_c2=0.6188 dice_fg=0.6021 acc=0.9412 (11.4 min)
  ★ 新最佳 dice_fg=0.6021，已保存 best.pt
```

| 字段 | 含义 | 期望 |
|---|---|---|
| `train_loss` | CE + Dice 总损失 | 稳定下降 |
| `val_loss` | 验证损失 | 跟随下降，不持续反弹 |
| `dice_c2` | 大块肿瘤 Dice | 主看，应上升 |
| `dice_fg` | 前景合并（1+2）Dice = 二值 Dice | **模型选择指标**，`best.pt` 按它刷新 |
| `dice_c1` | 零星肿瘤 Dice | 可能接近 0，别慌 |
| `acc` | 像素精度 | 参考值，背景多时天然偏高 |
| 括号分钟 | 单轮耗时 | 突然翻倍 = I/O 或降频 |

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
4. 把 `summary.json` 里的均值 ± 标准差抄进实验记录——**这是写论文要报的数**；
5. 磁盘别塞满：`Get-PSDrive D` 看剩余，5 折加日志约 6 GB。

---

## 8. 导出二值肿瘤掩膜

模型是三分类，二值化只是导出时多做一步。三分类 softmax 互斥，所以 `p_肿瘤 = 1 - p_bg = p1 + p2`，`argmax > 0` 与 `p_肿瘤 > 0.5` 等价，**`dice_fg` 本身就是二值 Dice**，不需要重训。

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
| `CUDA out of memory` | 1024 输入显存不够 | 加 `--grad-checkpointing`，再不够 `--freeze-encoder`；别只降 `--batch-size`（本来就是 1） |
| DataLoader 卡住不动 | Windows spawn + 大 `num-workers` | `--num-workers 2` 或 `0` |
| `dice_c1` 是 `nan` 或 0 | 该折验证集几乎没有类 1 | 正常，看 `dice_c2` / `dice_fg` |
| 找不到数据 / 权重 | `paths.py` 候选都没命中 | 报错会列出全部试过的路径，按提示设 `TOXPATH_DATA_ROOT` / `TOXPATH_HISTOPLUS_CKPT` |
| 训练日志停在某轮、GPU 占用 0 | 睡眠或重启 | 第 2 节设电源策略，`-Mode resume` 续上 |
| 同折曲线上有两条重叠 | 用了旧事件目录 | 新代码重跑会自动带时间戳后缀；清 `logs3\tb\foldN_*` 里不要的 |
| 局域网打不开 6006 | 防火墙 | 第 4.1 节的 `netsh` 命令 |

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
- CPU（8 逻辑核 / 15.8 GB，4 线程）参考：224 输入 3.6 s/样本、512 输入 16.4 s / 3.5 GB、1024 输入 107 s / 10.4 GB。**CPU 只配跑 224 冒烟**，正式训练必须上 4090。
- HistoPLUS 权重 `histoplus_cellvit_segmentor_20x.pt`：284,507,558 字节，SHA256 `AA19DC31C109C0A4FCF7BD773559C45480CEE792A331AE0A9FFE42EB08267BFC`。
- 模型规模：encoder 85.7 M + neck 14.3 M + decoder 14.0 M = **114.1 M** 参数。
