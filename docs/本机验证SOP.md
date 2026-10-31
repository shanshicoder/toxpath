# ToxPath 本机（笔记本 / 无显卡）验证 SOP

> 适用机器：**自己的 Windows 笔记本**，NVIDIA MX350 2 GB、8 逻辑核、15.8 GB 内存、`torch==2.14.1+cpu`
> 适用目的：**在不上 4090 的情况下，把除「显存 / CUDA / 真实速度」以外的所有环节验一遍**
> 配套文档：`docs\本地训练SOP.md`（实验室 4090 上的正式训练）；本文只管"本机能验到哪一步"
> 版本自检：`Select-String -Path scripts\run_train.ps1 -Pattern "WithWeights"` 没输出就 `git pull`

---

## 0. 一页速查

全部命令在仓库根目录执行，**统一用 `-PythonExe` 指定解释器**（原因见 2.1，这是本机第一大坑）：

```powershell
cd E:\project\toxpath
$PY = "D:\Program Files\anaconda3\envs\toxpath\python.exe"
```

| 我要做什么 | 命令 | 耗时 |
|---|---|---|
| 看数据根解析到哪 | `& $PY paths.py` | < 1 s |
| 数据契约 + 分折预演 | `& $PY scripts\check_dataset.py` | ~2 s |
| 看三分类标签分布 | `& $PY scripts\make_3class_masks.py --dry-run` | ~43 s |
| **端到端冒烟（主力验证）** | `.\scripts\run_train.ps1 -Mode smoke -Foreground -PythonExe $PY` | ~58 s |
| 顺带验权重加载 | 同上加 `-WithWeights` | ~59 s |
| 验一份别处的数据 | 同上加 `-DataRoot "D:\data\toxpath_73"` | ~58 s |
| 验续训逻辑 | 见 4.1（不能直接用 `-Mode resume`） | ~56 s |
| 看冒烟曲线 | `.\scripts\run_train.ps1 -Mode tb -TbDir "_smoke\tb" -PythonExe $PY` | 起服务约 10 s |
| 清理冒烟产物 | `Remove-Item _smoke -Recurse -Force` | — |

**一句话结论：本机能验「代码 + 数据 + 标签 + checkpoint 读写」是否正确，验不了「跑不跑得动」。**

---

## 1. 本机能验什么、不能验什么

| ✅ 能验（而且只有本机方便验） | ❌ 验不了 | 为什么 |
|---|---|---|
| 数据读取、标签反解、切片分折 | 真实训练速度 | CPU 与 4090 差 ~75 倍 |
| 模型构建 + HistoPLUS 权重加载 | 显存够不够 | 2 GB 卡 vs 24 GB 卡，量级完全不同 |
| 前向 / 反向 / loss 是否在算 | CUDA / bf16 / AMP 行为 | `torch.version.cuda is None`，CUDA 都没编译 |
| 梯度检查点开关不报错 | 加检查点后的显存峰值 | 同上 |
| checkpoint 落盘 / 恢复 | 五折真实耗时 | 单折 12 轮在 CPU 上要 11.9 天 |
| TensorBoard 事件写入与读图 | 收敛效果 | 8 个样本 1 轮，什么也证明不了 |
| 路径解析、外部数据根 | — | — |
| 参数拼装（`-ExtraArgs` 转发） | — | `-Foreground` 会把完整命令行打出来 |

**冒烟通过 ≠ 能训出来。** 它只证明"管线通"，不证明"结果好"。别用本机结果写论文。

---

## 2. 环境自检（换机器先做这三条）

```powershell
# ① 解释器
& $PY -c "import sys, torch; print(sys.version.split()[0], torch.__version__, torch.cuda.is_available(), torch.get_num_threads())"
# 本机期望：3.10.22 2.14.1+cpu False 4

# ② 显卡（有卡但用不上，确认一下型号和驱动就行）
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv
# 本机期望：NVIDIA GeForce MX350, 2048 MiB, 582.28

# ③ 磁盘与内存
Get-PSDrive C,D,E | Select-Object Name,@{n='FreeGB';e={[math]::Round($_.Free/1GB,1)}}
(Get-CimInstance Win32_OperatingSystem | Select-Object @{n='总内存GB';e={[math]::Round($_.TotalVisibleMemorySize/1MB,1)}},@{n='可用GB';e={[math]::Round($_.FreePhysicalMemory/1MB,1)}})
# 本机实测：内存 15.8 GB / 当前可用 4.7 GB；C 85.9、D 98.3、E 39.5 GB 可用
```

### 2.1 第一大坑：PATH 上的 `python` 不是 toxpath 环境

本机实测：

```powershell
python -c "import sys; print(sys.executable)"
# D:\Program Files\Python314\python.exe   ← 3.14.2，没装 torch
```

所以下面这条**看起来最正常**的命令，会在 **1.6 秒**后失败：

```powershell
.\scripts\run_train.ps1 -Mode smoke
# ModuleNotFoundError: No module named 'torch'
```

两种修法，**推荐第一种**（每次显式指定，不依赖终端状态）：

```powershell
# ① 显式指定解释器（本文全部命令都用这个）
.\scripts\run_train.ps1 -Mode smoke -Foreground -PythonExe "D:\Program Files\anaconda3\envs\toxpath\python.exe"

# ② 先激活环境（要求 conda init powershell 做过；临时开窗口时容易忘）
conda activate toxpath
.\scripts\run_train.ps1 -Mode smoke -Foreground
```

把 `$PY` 存成变量能少敲很多字，但它**只在当前窗口有效**，新开窗口要重设。

### 2.2 显卡有，但等于没有

`nvidia-smi` 能看到 MX350 2 GB，但 `torch.cuda.is_available()` 是 `False`——本机装的是 CPU 版 torch（版本号带 `+cpu`，`torch.version.cuda` 为 `None`）。就算换成 CUDA 版也没用：1024 输入单步峰值内存约 10.4 GB，2 GB 显存塞不下。**别在这上面花时间。**

---

## 3. 五分钟跑通整条链（照抄）

### ① 路径自检（< 1 s）

```powershell
& $PY paths.py
```

期望能在候选列表里看到 `OK` 的那一行，并且 `data_root` 是你想验的那份数据：

```
[paths] 当前解析：
  project_root : E:\project\toxpath
  data_root    : E:\project\toxpath\data\toxpath
  histoplus    : E:\project\histoplus_cellvit_segmentor_20x.pt
```

`data_root` 不对就先解决它（改 `TOXPATH_DATA_ROOT`、或后面用 `-DataRoot` 临时覆盖），否则后面每一步都在验错的数据。

### ② 数据契约 + 分折预演（~2 s，只读）

```powershell
& $PY scripts\check_dataset.py
```

**通过标准：退出码 0，最后一行是「通过」。** 本机在 4953 张上的实测：

```
[check] 合计：patch 4953 / 二值 mask 4953 / 三分类 mask 4953 / CSV 4953 行
[check] 抽样核对：25/25 组尺寸与颜色合规
[check] 分折预演 → fold1 952 张 基线 0.967 … fold5 874 张 基线 0.493
[check] 结论：通过。数量、命名、尺寸、颜色、CSV、分折都一致，可以进 1.7 冒烟。
```

这一步**不需要显卡、不需要 torch 的 CUDA**，是最便宜的排错环节。有问题它会逐条给出「现象 + 怎么修」。

### ③ 三分类标签分布（~43 s，不写文件）

```powershell
& $PY scripts\make_3class_masks.py --dry-run
```

**通过标准：** 4953 张、类别 `4140 / 149 / 664`（83.6% / 3.0% / 13.4%）。换过数据/改过阈值时数字会变，只要类别 1 不是 0 就正常；类别 1 塌了看 `本地训练SOP.md` 的阈值调整表。

### ④ 端到端冒烟（主力，~58 s）

```powershell
.\scripts\run_train.ps1 -Mode smoke -Foreground -PythonExe $PY
```

**为什么加 `-Foreground`**：不加的话脚本用 `Start-Process` 把训练丢到后台，你只拿到 PID 和日志路径，退出码要自己去翻；验证场景就该盯着它跑完。

本机实测 **57.7 秒、退出码 0**，末尾应该看到：

```
[Fold 1 Epoch 1/1] train_loss=2.0514 train_dice_loss=0.9327 val_loss=2.0337 dice_c1=nan dice_c2=nan dice_fg=nan acc=0.8080 (0.7 min)
```

**通过标准（四条全中才算过）：**

1. 退出码 `0`，没有 Traceback；
2. 有 `[Fold 1 Epoch 1/1]` 汇总行；
3. `_smoke\ckpt\fold1\last.pt` 已生成（本机 435.4 MB）；
4. `_smoke\tb\fold1\events.out.tfevents.*` 已生成（约 0.1 MB）。

**`dice_c1/dice_c2/dice_fg` 是 `nan`、`acc` 0.7~0.9 都属正常**：冒烟验证集只有 1 张、8 个训练样本、`--init none`，模型什么也没学到。**没有 `best.pt` 也正常**——`best.pt` 只在 `dice_fg` 有数值且刷新时写。

### ⑤ 顺带验权重加载（~59 s）

权重加载是最容易在新机器上炸的一环（`timm` 版本、key 名、`out_indices` 语义），**必须单独验一次**：

```powershell
.\scripts\run_train.ps1 -Mode smoke -Foreground -WithWeights -PythonExe $PY
```

本机实测 **58.5 秒、退出码 0**。**通过标准是这三行数字，`missing=0` 与 `shape_mismatch=0` 是关键：**

```
[H0MiniEncoder] 从 histoplus_cellvit_segmentor_20x.pt 加载 -> loaded=173, missing=0, unexpected=20, shape_mismatch=0
  encoder : loaded=173, missing=0, unexpected=20, shape_mismatch=0
  neck    : loaded=68 (strict)
```

`unexpected=20` 是正常的（权重里有 HistoPLUS 原来的 np/hv/tp 解码头，本项目换成了 3 类语义头，用不上）。**`missing` 一旦非 0，说明 timm 版本不对**——`pip install timm==1.0.8`。

> 加载 271 MB 权重在本机 SSD 上只多花约 1 秒（57.7 → 58.5 s），所以没有理由跳过这一步。

### ⑥ 验一份放在别处的数据（换数据时必做）

```powershell
.\scripts\run_train.ps1 -Mode smoke -Foreground -PythonExe $PY -DataRoot "D:\data\toxpath_73"
```

冒烟有自己的固定参数表，`-DataRoot` 是单独拼进去的（历史 bug：不加这行它会静默去读默认数据）。**通过标准同上**，但要多看一眼数据根是不是你指定的那份——`-Foreground` 时开头就会打印：

```
  project_root : E:\project\toxpath
  data_root    : D:\data\toxpath_73        ← 必须是你要验的那份
  histoplus    : E:\project\histoplus_cellvit_segmentor_20x.pt
```

想留档就去掉 `-Foreground`（改成后台跑），然后翻日志：

```powershell
Get-Content (Get-ChildItem logs3\train_smoke_*.out.log | Sort-Object LastWriteTime | Select-Object -Last 1) -TotalCount 20
```

前提：那个数据根**必须先跑过 `make_3class_masks.py`**（要有 `train_masks_classes\` 和 `_patch_labels_3class.csv`），否则冒烟会报「三分类 mask 目录不存在 / 请先运行 scripts/make_3class_masks.py」。想只验上游 patch 与二值 mask 是否齐，用 `check_dataset.py`，它不要求三分类已经生成。

---

## 4. 可选：再深一点的验证

### 4.1 续训逻辑（~56 s）

**`-Mode resume` 在本机用不了**：它写死找 `checkpoints3\fold1\last.pt`（4090 上的产物路径），本机没有、也不该伪造。直接用 python 跑，`--resume` 指到冒烟产出的那个 checkpoint：

```powershell
$env:PYTHONUTF8 = "1"     # 不设的话中文日志会乱码（见 4.3）
& $PY train_3class.py --fold 1 --epochs 2 --image-size 224 `
  --max-train-samples 8 --max-val-samples 4 --samples-per-epoch 8 `
  --num-workers 0 --device cpu --init none `
  --resume _smoke\ckpt\fold1\last.pt `
  --out-dir _smoke\ckpt --log-dir _smoke\log --tb-dir _smoke\tb
```

本机实测 **56 秒、退出码 0**，要看到：

```
[resume] 从 _smoke\ckpt\fold1\last.pt 恢复，epoch 2 开始（best_fg=nan）
[Fold 1 Epoch 2/2] train_loss=1.9295 ... (0.7 min)
```

**只跑第 2 轮就是通过**——说明 epoch 计数恢复了。注意 `--resume` **不恢复优化器状态**（checkpoint 因此从 1.3 GB 降到 456 MB），续训后头 1–2 轮 loss 抖一下属预期。

### 4.2 TensorBoard 起看板（本机能看）

```powershell
.\scripts\run_train.ps1 -Mode tb -TbDir "_smoke\tb" -PythonExe $PY
# 浏览器开 http://127.0.0.1:6006
```

**通过标准：SCALARS 里能看到 `fold1/train/*`，IMAGES 里有 `fold1/preview` 三联图。** 这一步证明「事件写入了 + 能读出来」，与显卡无关。看完 `Stop-Process -Id <PID>`。

`-TbDir` 必须指向训练时实际的 `--tb-dir`（冒烟是 `_smoke\tb`），否则页面能打开但一条曲线都没有。

### 4.3 直接跑 python 的两个注意

| 坑 | 现象 | 做法 |
|---|---|---|
| 中文乱码 | 日志里出现「浠?_smoke...鎭㈠」 | 先 `$env:PYTHONUTF8 = "1"`；用 `run_train.ps1` 则已内置 |
| 相对路径跟当前目录走 | 产物落到奇怪的地方 | 一律用绝对路径，或先 `cd` 到仓库根 |

### 4.4 单步计时（想知道 CPU 到底多慢）

```powershell
$env:PYTHONUTF8 = "1"
& $PY train_3class.py --fold 1 --epochs 1 --image-size 224 `
  --max-train-samples 8 --max-val-samples 2 --samples-per-epoch 8 `
  --num-workers 0 --device cpu --init none --val-every 99 `
  --out-dir _smoke\ckpt_speed --log-dir _smoke\log_speed --tb-dir _smoke\tb_speed
```

日志里 `(0.2x it/s)` 就是吞吐。本机实测 **224 输入约 0.2 it/s ≈ 5 s/步**。

---

## 5. 哪些事**不要**在本机做

| 别做 | 原因 | 本机实测依据 |
|---|---|---|
| 真的跑一次训练 | 单折要 11.9 天 | 1024 输入约 107 s/步，单折 12 轮 × 800 样本 = 9600 步 |
| 试 1024 输入的显存/内存 | 单步约需 **10.4 GB**，本机当前可用内存只有 **4.7 GB** | 会吃满内存、开始换页，机器卡死而不是"慢" |
| 指望 MX350 帮忙 | 2 GB 显存 vs 22 GB 峰值；且当前 torch 根本没编 CUDA | `torch.version.cuda is None` |
| 用 `-NoGradCkpt` 提速 | 那是 48 GB 卡才用的开关，本机走 CPU 路径，与它无关 | — |
| 用 `-Mode all` 跑五折 | 会往 `checkpoints3\` 写 4.6 GB 垃圾，且几天跑不完 | — |
| 拿冒烟结果做任何结论 | 8 个样本 1 轮，`dice` 是 `nan` | — |

**本机的定位：把 4090 上会踩的坑提前踩掉。** 真正必须上 4090 才能答的问题只有三个——显存够不够、bf16 稳不稳、一轮要多久。

---

## 6. 换 73 张数据时，本机能做什么

数据侧（**全部不需要显卡**，建议在本机做完再传 4090）：

```powershell
$DR = "D:\data\toxpath_73"
& $PY scripts\check_dataset.py --data-root $DR --table          # 先查上游产物齐不齐
& $PY scripts\make_3class_masks.py --data-root $DR --dry-run    # 看分布，定阈值
& $PY scripts\make_3class_masks.py --data-root $DR              # 生成（7 万张约 70 分钟）
& $PY scripts\check_dataset.py --data-root $DR                  # 再验收 + 预演分折
.\scripts\run_train.ps1 -Mode smoke -Foreground -PythonExe $PY -DataRoot $DR
```

顺序要点：**先 `check_dataset` 再 `make_3class_masks`**。前者会先把「patch 扩展名不对、缺二值 mask、尺寸不一致」这类上游问题报出来，省得生成一小时后才发现。

跑完 `check_dataset` 的**分折预演表要抄进实验记录**：73 个切片分 5 折，每折的 patch 数和全空基线决定后面各折 `dice_fg` 能不能横比。这张表在 4090 上也是同一套 `splits.py` 算出来的，不会变。

> 7 万张时 `check_dataset` 约 20 秒（主要是列 7 万个文件名），`make_3class_masks --dry-run` 约 10 分钟，别以为是卡住了。

---

## 7. 故障速查（本机特有）

| 现象 | 原因 | 处理 |
|---|---|---|
| `ModuleNotFoundError: No module named 'torch'`（1.6 s 就退） | PATH 上的 `python` 是 3.14，不是 toxpath 环境 | 加 `-PythonExe "D:\Program Files\anaconda3\envs\toxpath\python.exe"` |
| `conda activate toxpath` 报「无法识别」 | 没做 `conda init powershell` | 用 `-PythonExe` 全路径，或 `conda init powershell` 后重开窗口 |
| 冒烟跑完 `dice_fg=nan`、没有 `best.pt` | 验证集只有 1 张、样本太少 | 正常，不是失败 |
| 冒烟 `acc` 只有 0.7~0.9 且波动 | 8 个样本 1 轮 | 正常 |
| `-Mode smoke -DataRoot` 后仍读老数据 | 代码旧（smoke 没拼 `--data-root`） | `git pull`；新版会在命令行里看到 `--data-root` |
| `-Mode resume` 报 `resume target not found` | 它只认 `checkpoints3\foldN\last.pt` | 本机按 4.1 直接跑 python |
| 直接跑 python 中文变「浠?_smoke」 | 没设 `PYTHONUTF8` | `$env:PYTHONUTF8 = "1"` |
| 看板打开是空的 | `-TbDir` 与训练时 `--tb-dir` 不一致 | 冒烟要写 `-TbDir "_smoke\tb"` |
| `_smoke` 占了 400+ MB | `last.pt` 435 MB（含 114 M 参数 + 优化器状态） | `Remove-Item _smoke -Recurse -Force` |
| 机器整体卡死、风扇狂转 | 试了 1024 输入（要 10.4 GB 内存） | 别试；本机只跑 224 |
| `Start-Process : 已添加了具有相同键的项` | 环境变量大小写重复（本机确有 `http_proxy`/`https_proxy` 成对） | 用 `run_train.ps1`（已内置去重，本机实测每次都会打印 `dropped 2 duplicated env vars`，属正常） |
| 满屏红字 `NativeCommandError` + `FutureWarning: torch.jit.interface is deprecated`，但退出码 0 | PS 5.1 把 Python 的 stderr 当错误显示，torch 每次启动都会打这条 | **正常**，只看退出码和最后有没有 `[Fold 1 Epoch 1/1]`；这不是失败 |
| TensorBoard 6006 端口被占 | 上次没关干净 | `Get-NetTCPConnection -LocalPort 6006` 找 PID，`Stop-Process` |

---

## 8. 收工

```powershell
Remove-Item _smoke -Recurse -Force                  # 回收 436 MB
Get-ChildItem logs3\train_smoke_*.log | Remove-Item # 冒烟日志，不需要留
```

**本机验证完毕的标准动作**：把下面三样带上 4090，别重新发明：

1. `check_dataset.py` 的输出（分折预演那张表）；
2. `make_3class_masks.py --dry-run` 的类别分布；
3. 冒烟里 `encoder loaded=173, missing=0, shape_mismatch=0` 这一行（证明权重与 timm 版本是配的）。

到了 4090 上只需要再跑：`-Mode smoke -WithWeights`（验 CUDA 与显存）→ `-Mode fold -Fold 1 -Epochs 1`（量真实速度）→ 正式训练。

---

## 附录：本机实测数字（2026-10-31，写论文引用时注明「CPU 参考」）

| 项 | 实测值 |
|---|---|
| Python / torch | 3.10.22 / **2.14.1+cpu**，`torch.version.cuda=None`，`cuda.is_available()=False`，`get_num_threads()=4` |
| 硬件 | 8 逻辑核、15.8 GB 内存（验证时可用 4.7 GB）、MX350 2048 MiB、驱动 582.28 |
| 磁盘可用 | C 85.9 GB / D 98.3 GB / E 39.5 GB |
| PATH 上的 python | `D:\Program Files\Python314\python.exe`（3.14.2，**无 torch**） |
| 冒烟（`--init none`） | **57.7 s**，退出码 0，224 输入 8 步约 40 s（≈0.2 it/s ≈ 5 s/步） |
| 冒烟（`-WithWeights`） | **58.5 s**，退出码 0，`encoder loaded=173 missing=0 unexpected=20 shape_mismatch=0`，`neck loaded=68 (strict)` |
| 权重加载开销 | 约 1 s（本地 SSD；271 MB 文件） |
| 续训（`--resume` 到冒烟 ckpt） | **56 s**，退出码 0，从 epoch 2 开始 |
| `check_dataset.py` | 4953 张 **1.5 s**，抽样 25/25 合规 |
| `make_3class_masks.py --dry-run` | 4953 张 **43.3 s** |
| 冒烟产物 | 合计 **436 MB**，其中 `last.pt` 435.4 MB、事件文件 0.1 MB |
| 1024 输入单步内存 | 约 10.4 GB（历史实测）→ 本机可用内存不足，**不要试** |
| CPU 外推 | 1024 输入 107 s/步 → 单折 9600 步 ≈ 11.9 天（4090 同口径约 3.8 h，差约 75 倍） |
