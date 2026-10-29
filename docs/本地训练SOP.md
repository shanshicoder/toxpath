# 本地训练 SOP（Windows + RTX 4090 24GB）

> 适用：ToxPath 三分类（1024 输入）+ HistoPLUS 架构
> 机器：实验室 Windows 工作站，单卡 RTX 4090 24 GB，远程桌面 / 向日葵操作
> 本文替代原 `AutoDL训练SOP.md`（该文档已移到 `_deprecated/docs/` 存档）

---

## 0. 和云端方案的区别（先看这张表）

| 环节 | 云端（旧） | 本地（现在） |
|---|---|---|
| 代码到位 | tar 打包 → scp 上传 1.5 GB | **`git clone`**，仓库几百 KB，秒完成 |
| 数据到位 | 跟包一起传 | 机器上已有，不用动，`--data-root` 指过去 |
| 权重到位 | 一起传 | **GitHub Release 附件**（271 MB，超单文件限制），脚本下载 + 校验 |
| 代码更新 | 重新打包上传 | `git pull`，只传差异 |
| 路径 | 写死 `/root/autodl-tmp/...` | `paths.py` 自动解析，换机器不改代码 |
| 环境 | 镜像自带 | 自己建 conda 环境（一次） |
| 进程存活 | `nohup ... &` | `train_local.bat` 后台进程 + 日志重定向 |
| 监控 | `tail -f` | `Get-Content -Wait` |
| 结果回收 | scp 下载 tar.gz | 直接在本机，拷 checkpoint 即可 |

**关键前提：** 这台机器不能睡眠。远程桌面断开是安全的，睡眠会中断训练（见第 2 节）。

---

## 1. 一次性准备（只做一遍）

### 1.1 确认显卡与驱动

```powershell
nvidia-smi
```

应该看到 `RTX 4090`、`24564MiB`（24 GB）、驱动版本 550 以上。
驱动太老时 CUDA 版 torch 会报 `CUDA driver version is insufficient`，去 NVIDIA 官网更新。

顺手确认这台机器上没人在跑任务：

```powershell
nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv
```

有输出说明显卡被占用，等它结束再开训。

### 1.2 建 Python 环境（约 10 分钟）

需要 **Python 3.10**（HistoPLUS 原始环境就是 3.10，timm 1.0.8 在该版本上最稳）。

```powershell
conda create -n toxpath python=3.10 -y
conda activate toxpath

# torch：按显卡驱动选 cu124；驱动 >= 550 装 cu124 最稳
pip install --index-url https://download.pytorch.org/whl/cu124 torch torchvision

# 其余依赖（timm 必须 1.0.8）
pip install timm==1.0.8 numpy Pillow
```

验证三件事：

```powershell
python -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0))"
# 期望类似：2.5.1+cu124 True NVIDIA GeForce RTX 4090

python -c "import timm; print(timm.__version__)"
# 期望：1.0.8
```

> PowerShell 里 `conda activate` 报「无法识别」时，先执行一次 `conda init powershell`，然后重开窗口。
> 想开机自动进环境：`conda config --set auto_activate_base false` 之后手动 activate 即可。

### 1.3 从 GitHub 拉代码

代码走 Git，不是拷压缩包。仓库：`git@github.com:shanshicoder/toxpath.git`。

**第一次（实验室机器上）**

```powershell
git clone git@github.com:shanshicoder/toxpath.git D:\work\toxpath
cd D:\work\toxpath
dir
# 应看到 train_3class.py、evaluate.py、inference.py、paths.py、model\、dataset\、docs\ ...
```

**以后同步**（笔记本推上去之后）

```powershell
cd D:\work\toxpath
git pull
```

**笔记本这边：改完代码怎么推**

```powershell
cd E:\project\toxpath
scripts\sync_github.bat -Message "fix: 修正 fold5 采样权重"
```

不给 `-Message` 也行，会自动写 `sync: <时间> 自动同步`。脚本会先扫描待提交文件，
拦住误入的 `.pt` / `.jpg` / `data\` / checkpoint（权重 271 MB 推上去会被 GitHub
直接拒），再 add + commit + push。`-DryRun` 可以只看它打算做什么。

**实验室机器要先能连 GitHub**

私有仓库不能匿名 clone，得有一把属于这台机器的 SSH key：

```powershell
ssh-keygen -t ed25519 -C "lab-4090"        # 一路回车，密码可留空
type $env:USERPROFILE\.ssh\id_ed25519.pub  # 复制整段输出
```

粘到 GitHub → Settings → SSH and GPG keys → New SSH key → 验证：

```powershell
ssh -T git@github.com
# 出现 Hi shanshicoder! You've successfully authenticated 即通
```

> **连不上时先判断封的是哪个端口**。国内网络常见的是 **443 被封、22 反而通**
> （本机实测就是这样），所以别想当然换 HTTPS：
>
> ```powershell
> Test-NetConnection github.com -Port 22    # TcpTestSucceeded
> ```
>
> - **22 通** → 直接用 SSH，别折腾 HTTPS
> - **22 封、443 通** → 让 SSH 改走 443。在 `%USERPROFILE%\.ssh\config` 里加：
>
>   ```
>   Host github.com
>       HostName ssh.github.com
>       Port 443
>       User git
>   ```
>
>   然后 `ssh -T git@github.com` 验证
> - **两个都封** → 只能挂代理，或走第 1.3 节末尾的 U 盘兜底

**仓库里有什么、没什么**

| 进仓库 | 不进仓库（`.gitignore` 挡住） |
|---|---|
| 全部 `.py` 源码、`docs\`、`scripts\`、`tests\` | `data\`（1.2 GB patch + mask） |
| `requirements.txt`、`README.md`、本 SOP | `weights\`（271 MB 权重，走 Release） |
| `.gitattributes`（统一换行符） | `checkpoints3\`、`logs3\`、`eval3\`（训练产物） |

所以 clone 下来只有几百 KB，几秒钟完成。

> **完全连不上 GitHub 时的兜底**：笔记本上跑 `scripts\make_code_package.bat -IncludeWeights`，
> 打一个含权重的 zip（约 250 MB），U 盘拷到实验室机器解压即用。
> 这是应急路径，日常同步还是走 `git pull`——增量只传改动，比每次搬 250 MB 强。

### 1.4 权重：从 GitHub Release 下载

271 MB 的 `histoplus_cellvit_segmentor_20x.pt` 超过 GitHub 单文件 100 MB 硬限制，
**不能**放进仓库，所以作为 Release 附件单独分发。

> **这个 Release 只需建一次**（仓库所有者操作）：
> GitHub 仓库页 → Releases → Draft a new release → tag 填 `weights-v1` →
> 把 `histoplus_cellvit_segmentor_20x.pt` 拖进附件框 → Publish release。
> 私有仓库的附件同样要登录/带 token 才能下。之后所有机器都用下面的方式拉。

**方式 A：脚本下载（推荐）**

```powershell
cd D:\work\toxpath
scripts\fetch_weights.bat
```

它会下到 `weights\`，支持断点续传，下完自动核对字节数与 SHA256：

```
[size] 284507558 字节
[hash] 正在算 SHA256...
[done] 校验通过，权重就位：D:\work\toxpath\weights\histoplus_cellvit_segmentor_20x.pt
```

私有仓库要带 token（GitHub → Settings → Developer settings → Personal access
tokens → 勾 `repo` 权限）：

```powershell
setx GITHUB_TOKEN "ghp_xxxxxxxx"     # 重开窗口生效
scripts\fetch_weights.bat
```

> GitHub 的附件域名（`objects.githubusercontent.com`）在国内经常传到一半被重置。
> 报 curl 退出码 56 时**直接重跑**，会从断点接着下，一般 2–3 次就能完成。

**方式 B：浏览器手动下载**

打开 `https://github.com/shanshicoder/toxpath/releases/tag/weights-v1`，
下载 `.pt` 放到 `D:\work\toxpath\weights\`（目录不存在就新建）。

**方式 C：U 盘 / 网盘**

网络实在不通时才用。拷完务必核对哈希，见下。

**核对完整性**（U 盘拷贝最容易只拷一半，只拷 200 MB 也会"看起来成功"）：

```powershell
Get-FileHash .\weights\histoplus_cellvit_segmentor_20x.pt -Algorithm SHA256
# 基准值，必须完全一致：
# AA19DC31C109C0A4FCF7BD773559C45480CEE792A331AE0A9FFE42EB08267BFC
```

`paths.py` 会自动在这三个位置找权重，命中第一个就用：

```
① <项目根>\weights\histoplus_cellvit_segmentor_20x.pt   ← 推荐，脚本就放这里
② <项目根>\histoplus_cellvit_segmentor_20x.pt
③ <项目根>\..\histoplus_cellvit_segmentor_20x.pt
```

都不中时 `check_env.py` 会把试过的路径全列出来；也可以显式指定：
`setx TOXPATH_HISTOPLUS_CKPT "D:\w\histoplus_cellvit_segmentor_20x.pt"`，
或每次加 `--histoplus-ckpt`。

### 1.5 数据目录

数据**不进仓库**，实验室机器上放哪都行，只要 `--data-root` 指对。目录结构：

```
D:\data\toxpath\
├── train_patches\           4953 张 1024×1024 JPEG，按 train_001..005\<层级>\ 三层嵌套
├── train_masks_classes\     4953 张彩色三分类 mask（黑/绿/红）
├── train_masks\             4953 张原始二值 mask（训练不读，重生成三分类时要用）
├── _patch_labels_3class.csv 4953 行 patch 索引
├── train_images\            金字塔 TIFF，训练不读
└── train_labels\            GeoJSON 原始标注，训练不读
```

类别分布（实测，用于判断采样是否合理）：

| 类别 | 含义 | 数量 | 占比 | train_001 | 002 | 003 | 004 | 005 |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| 0 | 背景 | 4139 | 83.6% | 431 | 1136 | 799 | 852 | 921 |
| 1 | 少量（0.01%–5%） | 172 | 3.5% | 63 | 36 | 9 | 60 | **4** |
| 2 | 较多（≥5%） | 642 | 13.0% | 380 | 92 | 34 | 109 | 27 |

> 注意最后一列：**fold5 的验证集（train_005）只有 4 张类别 1**，那一折的 `dice_c1` 可能是 `nan` 或 0，属正常，看 `dice_c2` / `dice_fg` 即可。

数据路径**不用改代码**，按优先级任选一种：

| 方式 | 命令 | 适用 |
|---|---|---|
| A. 命令行参数 | `--data-root "D:\data\toxpath"` | 临时、偶尔换数据 |
| B. 环境变量 | `setx TOXPATH_DATA_ROOT "D:\data\toxpath"`（重开窗口生效） | 长期固定，一次搞定 |
| C. 放进项目 | 把 `data\toxpath\` 放到 `D:\work\toxpath\data\toxpath\` | 想让项目和数据绑在一起 |

推荐 B，设一次以后所有命令都不用带 `--data-root`。

### 1.6 一条命令体检（必做）

```powershell
cd D:\work\toxpath
conda activate toxpath
python scripts\check_env.py --data-root "D:\data\toxpath"
```

它会逐项检查并给出结论：

```
[OK  ] 代码文件                 8 个关键文件齐全
[OK  ] Python               3.10.x
[OK  ] torch                2.5.1+cu124  (cuda 12.4)
[OK  ] timm                 1.0.8
[OK  ] GPU                  NVIDIA GeForce RTX 4090  24.0 GB  SM 8.9  128 SM
[OK  ] 显存                  已用 0.5 GB，可用 23.5 GB
[OK  ] 数据根                 D:\data\toxpath
[OK  ] train_patches        4953 张
[OK  ] train_masks_classes  4953 张（与 patch 数一致）
[OK  ] patch 索引 CSV        4953 行
[OK  ] HistoPLUS 权重        271 MB  D:\work\toxpath\weights\...
[OK  ] 磁盘                  D: 剩余 xxx GB
```

出现 `[FAIL]` 就按提示修，退出码非 0 表示还没到能开训的状态。
`[WARN]` 一般不影响，但会告诉你风险（比如 Python 3.13、显存只剩 8 GB）。

---

## 2. 让训练熬过远程桌面断开

这是本地训练最容易踩的坑，单独一节。

### 2.1 三种情况分别会怎样

| 操作 | 训练进程 | 说明 |
|---|---|---|
| 关掉远程桌面窗口 / 向日葵断连 | **继续跑** | 会话只是锁定，进程还活着 |
| 电脑睡眠 / 休眠 | **中断** | GPU 掉电，进程冻死 |
| Windows 自动重启更新 | **中断** | 半夜重启，训练全废 |

所以只要治住后两条，就可以放心下班挂机。

### 2.2 关掉睡眠（必须做，管理员 PowerShell）

```powershell
# 插电时永不睡眠、永不休眠；只关显示器
powercfg /change standby-timeout-ac 0
powercfg /change hibernate-timeout-ac 0
powercfg /change monitor-timeout-ac 15

# 再确认一遍（输出里 AC 那几行应为 0）
powercfg /query SCHEME_CURRENT SUB_SLEEP
```

注意：以上只对「插电（AC）」生效，笔记本合盖仍可能睡眠；工作站请确认机箱电源不会被误断。

### 2.3 暂停 Windows 更新自动重启

设置 → Windows 更新 → 暂停更新 7 天（长跑 5 折时建议这么做）。
或者至少确认「活动时间」覆盖训练时段，避免它在你离开时重启。

### 2.4 后台启动，不依赖窗口

`scripts\train_local.bat` 默认用独立进程 + 日志重定向（`.bat` 只是包了一层
`powershell -ExecutionPolicy Bypass`，免得被「禁止运行脚本」拦下；习惯用 PowerShell 时
直接 `.\scripts\train_local.ps1 ...` 也一样，前提是执行策略允许）：

- 日志写到 `logs3\_runs\<模式>_<时间戳>.log`；
- 关闭 PowerShell 窗口、断开远程桌面都不影响；
- 想知道还活着：`Get-Process -Id <PID>`；
- 想停：`Stop-Process -Id <PID>`（会丢掉当前 epoch，但有 `last.pt` 可续训）。

---

## 3. 冒烟验证（约 5 分钟）

```powershell
cd D:\work\toxpath
conda activate toxpath
.\scripts\train_local.bat -Mode smoke
```

等价于：

```powershell
python train_3class.py --fold 1 --epochs 1 --batch-size 1 --grad-accum 4 `
  --max-train-samples 32 --max-val-samples 8 --num-workers 2 `
  --amp --grad-checkpointing `
  --out-dir checkpoints3_smoke --log-dir logs3_smoke
```

看日志（另开一个窗口）：

```powershell
Get-Content .\logs3_smoke\_runs\smoke_*.log -Wait -Tail 30
```

**通过标准：**

- 开头打印 `device=cuda`、`amp=True`、`gpu=NVIDIA GeForce RTX 4090  vram=24.0 GB`；
- 打印路径三行：`project_root / data_root / histoplus` 都指向真实位置；
- 权重加载摘要出现 `encoder 173 / neck 68`；
- 显存峰值 **< 20 GB**（4090 上通常 12–16 GB）；
- `train_loss` 是有限值，不是 `nan`；
- 记下单步耗时，用来估正式训练的时长。

同时开一个窗口看显存：

```powershell
nvidia-smi --query-gpu=utilization.gpu,memory.used,memory.total,temperature.gpu --format=csv -l 2
```

冒烟产物在 `checkpoints3_smoke\`、`logs3_smoke\`，跑完可以直接删。

---

## 4. 正式训练

### 4.1 单折（先跑 1 折看曲线）

```powershell
.\scripts\train_local.bat -Mode fold -Fold 1 -Epochs 20
```

等价于：

```powershell
python train_3class.py --fold 1 --epochs 20 --batch-size 1 --grad-accum 8 `
  --samples-per-epoch 800 --val-every 2 --save-every 5 --num-workers 4 `
  --amp --grad-checkpointing
```

### 4.2 五折（按切片留一交叉验证）

```powershell
.\scripts\train_local.bat -Mode all -Epochs 20
```

同一个进程顺序跑 5 折，任何一折失败都不影响前面已存的结果。建议下班前挂上。

### 4.3 参数为什么这么配

| 参数 | 值 | 原因 |
|---|---|---|
| `--image-size` | 1024 | 全尺寸输入，patch 原生分辨率，不缩放 |
| `--batch-size` | 1 | 1024×1024 走 ViT-B/14，单卡一次只能塞一张（24 GB 下也是这样） |
| `--grad-accum` | 8 | 等效 batch 8，弥补小 batch 的梯度噪声 |
| `--samples-per-epoch` | 800 | 每轮从 4000 张训练样本里采样 800，控制单轮时间 |
| `--val-every` | 2 | 每 2 轮验证一次，验证集一次要跑 800~1200 张，省时间 |
| `--amp` | 开 | bf16 自动混合精度，4090（Ada）原生支持，显存和速度双收益 |
| `--grad-checkpointing` | 开 | 用计算换显存，激活值不再全存 |
| `--num-workers` | 4 | Windows 下 DataLoader 用 spawn，4 个进程读 JPEG 足够；卡住就降到 2 或 0 |
| `--save-every` | 5 | 每 5 轮存 `last.pt`（`best.pt` 每次刷新指标都会存） |

**预期资源占用（4090 24 GB 实测口径）：**

| 项目 | 数值 |
|---|---|
| 显存峰值 | 12–16 GB（开 checkpointing + AMP） |
| 单轮耗时 | 12–18 分钟（800 样本 + 每 2 轮验证） |
| 单折 20 轮 | 约 1.5–2 小时 |
| 5 折 | **约 8–10 小时**，建议过夜 |
| checkpoint 体积 | 每折 best + last ≈ 0.9 GB，5 折约 4.6 GB |

> 首次跑请以日志里实际单轮耗时为基准重新估算，上面只是参考值。
> 420 W 满载 8 小时，注意机箱散热与电费；温度持续 > 85 ℃ 时考虑 `nvidia-smi -pl 350` 限功耗。

### 4.4 常用变体

```powershell
# 显存吃紧 / 别人还在用这张卡：冻结编码器，只训 neck + 解码头
.\scripts\train_local.bat -Mode fold -Fold 1 -FreezeEncoder

# 想先看参数拼得对不对，不真的跑
.\scripts\train_local.bat -Mode all -Epochs 20 -DryRun

# 数据/权重在别的地方
.\scripts\train_local.bat -Mode fold -Fold 1 `
  -DataRoot "E:\data\toxpath" -Ckpt "E:\weights\histoplus_cellvit_segmentor_20x.pt"
```

---

## 5. 训练中监控

```powershell
cd D:\work\toxpath

# 1) 追日志（Ctrl+C 只退出 tail，不杀训练）
Get-Content .\logs3\_runs\all5_*.log -Wait -Tail 30

# 2) 看进程还在不在（PID 启动时打印过）
Get-Process python

# 3) 看 GPU
nvidia-smi --query-gpu=utilization.gpu,memory.used,temperature.gpu --format=csv -l 2

# 4) 看产物
dir .\checkpoints3\fold1\, .\logs3\fold1\
```

每个 epoch 关注这几项：

| 指标 | 含义 | 期望 |
|---|---|---|
| `train_loss` | 训练总损失（CE + Dice） | 稳定下降 |
| `val_loss` | 验证损失 | 跟随下降，不反弹 |
| `dice_c2` | 类别 2（较多肿瘤）Dice | 主看这个，应上升 |
| `dice_fg` | 前景合并（类别 1+2）Dice | **模型选择指标**，`best.pt` 按它刷新 |
| `dice_c1` | 类别 1（少量）Dice | 可能是 `nan` 或 0，类别极少，别据此判断崩了 |
| `acc` | 像素精度 | 参考，背景占 83% 时会偏高 |
| 括号里的分钟数 | 单轮耗时 | 突然翻倍说明 I/O 或散热有问题 |

---

## 6. 中断与续训

### 6.1 三种中断场景

| 场景 | 现象 | 处理 |
|---|---|---|
| 主动 `Stop-Process` | 进程消失 | 从 `last.pt` 续训 |
| 电脑睡眠/断电 | 日志停住、GPU 掉 0% | 唤醒后按 6.2 续训 |
| Windows 重启 | 进程没了 | 同上 |

`last.pt` 每 `--save-every` 轮（默认 5）落一次，最坏丢 5 轮。

### 6.2 续训

```powershell
.\scripts\train_local.bat -Mode resume -Fold 1 -Epochs 20
```

等价于加 `--resume checkpoints3\fold1\last.pt`。

> `--resume` 只恢复模型权重与 epoch 计数，**不恢复优化器状态**（这样 checkpoint 从 1.3 GB 降到 456 MB）。
> 所以续训后头 1~2 轮的 loss 会抖一下，属预期，别慌。

五折跑到一半中断：已完成的折结果都在 `checkpoints3\foldN\` 和 `summary.json` 里，
重新用 `-Mode all` 会**从头覆盖**。想接着跑剩下几折，用 `-Mode fold` 逐折补：

```powershell
.\scripts\train_local.bat -Mode fold -Fold 3
.\scripts\train_local.bat -Mode fold -Fold 4
.\scripts\train_local.bat -Mode fold -Fold 5
```

---

## 7. 评测与推理

训练完先看 `checkpoints3\summary.json`（5 折 `dice_fg` 均值），再跑正式评测：

```powershell
# 单折评测
python evaluate.py --fold 1 --ckpt checkpoints3\fold1\best.pt

# 全 5 折 + 汇总（mean ± std → eval3\summary.csv）
python evaluate.py --all-folds --ckpt-dir checkpoints3

# 某张切片出彩色预测图 + 三联对比图
python inference.py --ckpt checkpoints3\fold1\best.pt --slide train_001 `
  --out-dir runs3\pred_fold1 --save-pred --overlay-every 20
```

三层指标（`eval3\foldN\metrics.json`、`per_patch.csv`、`eval3\summary.csv`）：

| 口径 | 指标 | 用途 |
|---|---|---|
| `full_*` 整图 | 每类 Dice / IoU / 像素精度 | 与训练日志同口径 |
| `tissue_*` 组织区域内 | 同上，Otsu 剔除白色空背景 | **主指标**，写报告用 |
| `patch_acc` | 预测前景占比按 0.01%/5% 分箱 vs 真值 | 「少量 vs 较多」分级准确率 |

---

## 8. 结果归档

训练完把这几样拷到移动硬盘 / 课题组存储：

```powershell
# 只打 checkpoint + 曲线 + 汇总（约 2.3 GB，不含 last.pt）
Compress-Archive -Path .\checkpoints3\fold*\best.pt, .\checkpoints3\fold*\history.json,
  .\checkpoints3\summary.json -DestinationPath D:\backup\toxpath_results3.zip

# 日志与评测也建议一起留档
Compress-Archive -Path .\logs3, .\eval3 -DestinationPath D:\backup\toxpath_logs3.zip
```

机器是公用的，`checkpoints3\` 里每折 0.9 GB，跑几轮实验就会顶满盘，**归档后及时清理旧实验**。

---

## 9. 常见问题（Windows 特有）

| 现象 | 原因 / 处理 |
|---|---|
| `无法加载文件 ...因为在此系统上禁止运行脚本` | 用 `.bat` 启动器（已内置 `-ExecutionPolicy Bypass`）；或 `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned` |
| `conda activate` 无法识别 | 先 `conda init powershell`，重开窗口 |
| `torch.cuda.is_available()` 为 False | 装成 CPU 版；或驱动太老；跑 `check_env.py` 看具体哪项 FAIL |
| `CUDA out of memory` | 确认 `--grad-checkpointing` 已开；再加 `-FreezeEncoder`；仍不行 `--image-size 512` |
| 显存够但提示别人占卡 | `nvidia-smi` 看 PID，等别人结束；不要硬抢 |
| `timm` 不是 1.0.8 | `pip install timm==1.0.8`，版本不对会导致 ViT 权重**静默**失配（不报错但学不好） |
| DataLoader 卡住 / `BrokenPipeError` | `--num-workers 2`；仍卡就 `0`（慢但稳） |
| 路径里有中文或空格导致报错 | 项目和数据的上级目录尽量用纯英文；真要带空格，用引号包住整个路径 |
| 日志里 `data_root` 指向错的地方 | 优先级是 命令行 > 环境变量 > 项目内 `data\toxpath`；显式加 `--data-root` 最保险 |
| `找不到 HistoPLUS 权重` | 跑 `scripts\fetch_weights.bat` 自动下载；或手动把 `.pt` 放到 `D:\work\toxpath\weights\`；或 `setx TOXPATH_HISTOPLUS_CKPT "..."` |
| 权重只有几十 MB / `check_env` 提示偏小 | 下载或拷贝没完成，重跑 `fetch_weights.bat`（会续传），再核对 SHA256（见 1.4） |
| `fetch_weights` 报 curl 退出码 56 / 28 | GitHub 附件域名被重置或超时，**直接重跑**，脚本从断点续传；实在不行用浏览器下 |
| `ssh -T git@github.com` 超时 / `connect to host github.com port 22` | 22 被封：在 `%USERPROFILE%\.ssh\config` 配 `HostName ssh.github.com` + `Port 443`（见 1.3） |
| 浏览器也打不开 github.com | 443 被封，只能挂代理；代码可用代理克隆，权重用 U 盘拷（见 1.4 方式 C） |
| `git push` 报 `Permission denied (publickey)` | 公钥没加到 GitHub，或私钥带 passphrase 而当前环境不能输入；`ssh -T git@github.com` 先验证 |
| `git pull` 冲突 | 实验室机器上别改代码，只拉不改；已改就 `git stash` 再 `git pull`，或 `git checkout -- <文件>` 丢弃 |
| `FileNotFoundError: 三分类 mask 目录` | 这台机器只有原始 mask：`python scripts\make_3class_masks.py --dry-run` 先看分布，再去掉 `--dry-run` 生成 |
| 训练突然变慢 / 单轮耗时翻倍 | 散热降频（看温度）、或别人在抢 GPU、或磁盘被日志写满 |
| 远程桌面断了回来发现训练没了 | 用了 `-Foreground`；改后台模式，并按第 2 节关掉睡眠 |
| `dice_c1` 是 `nan` | 那折验证集类别 1 太少（fold5 只有 4 张），正常 |
| 续训后 loss 变大 | 优化器状态没恢复，跑 1~2 轮会回来 |

---

## 10. 一次跑通的检查清单

- [ ] `nvidia-smi` 看到 RTX 4090、24 GB、驱动 ≥ 550，且没有别人占卡
- [ ] `python -c "import torch; print(torch.cuda.is_available())"` → `True`
- [ ] `timm == 1.0.8`
- [ ] `git clone` 成功，`git log --oneline -3` 能看到最新提交
- [ ] `python scripts\check_env.py` 全 `[OK]`（无 `[FAIL]`）
- [ ] 数据：patch 4953、三分类 mask 4953、CSV 4953 行
- [ ] 权重 271 MB 在 `weights\` 下，SHA256 = `AA19DC31...08267BFC`
- [ ] 电源设置：插电永不睡眠（`powercfg /change standby-timeout-ac 0`）
- [ ] 冒烟通过：显存 < 20 GB、loss 有限、路径打印正确
- [ ] 单折 20 轮曲线正常（`dice_c2` / `dice_fg` 上升）
- [ ] 5 折跑完，`checkpoints3\summary.json` 有 5 条记录
- [ ] 跑过 `python evaluate.py --all-folds`，拿到 `eval3\summary.csv`
- [ ] 结果已归档到移动硬盘，旧实验目录已清理
