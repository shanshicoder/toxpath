# 实验室训练 SOP（Windows + RTX PRO 5000 Blackwell 48 GB）

首次使用：先做第 0 节建环境。之后每个新开的 PowerShell 窗口，先执行第 1 节，再执行后面的任意一节。

## 0 首次创建虚拟环境

### 0.1 找到 conda.exe

```powershell
@(
  "C:\ProgramData\anaconda3","C:\ProgramData\miniconda3",
  "$env:USERPROFILE\anaconda3","$env:USERPROFILE\miniconda3",
  "$env:LOCALAPPDATA\anaconda3","$env:LOCALAPPDATA\miniconda3",
  "D:\Program Files\anaconda3","D:\anaconda3"
) | ForEach-Object { if (Test-Path "$_\Scripts\conda.exe") { "$_\Scripts\conda.exe" } }
```

上面没打印任何路径时，再执行这一条（较慢，约 1 分钟）：

```powershell
Get-ChildItem C:\,D:\ -Filter conda.exe -Recurse -Depth 4 -ErrorAction SilentlyContinue |
  Select-Object -First 3 -ExpandProperty FullName
```

### 0.2 让 conda 进入 PowerShell

```powershell
$CONDA = "C:\ProgramData\anaconda3\Scripts\conda.exe"   # 换成 0.1 打印出的真实路径

& $CONDA init powershell
Set-ExecutionPolicy -Scope CurrentUser -ExecutionPolicy RemoteSigned -Force
```

**关掉这个 PowerShell 窗口，重新开一个**，然后：

```powershell
conda --version
```

能打印版本号即成功，之后 `conda` 可直接使用。

不想改 PATH、只想在 cmd 里临时用（不需要重开窗口）：

```cmd
call "C:\ProgramData\anaconda3\Scripts\activate.bat"
conda --version
```

开始菜单里的 `Anaconda Prompt` 是同样的效果，打开它 `conda` 直接可用。

### 0.3 建环境并装依赖

在 `Anaconda Prompt` 里执行。若在普通 cmd 窗口，先执行 0.2 的 `call ...activate.bat`。

```cmd
conda env list
```

列表里已有 `toxpath` 就跳过下面这条 `conda create`。

```cmd
conda create -n toxpath python=3.10 -y
conda activate toxpath
python -m pip install --upgrade pip
```

```cmd
cd /d "D:\work\toxpath"                                # 换成项目实际路径

pip install --index-url https://download.pytorch.org/whl/cu128 torch torchvision
pip install -r requirements.txt
```

**必须是 cu128。** 这块卡是 Blackwell（compute capability 12.0 / sm_120），`cu121`、`cu124` 的 wheel 里没有 sm_120 内核，装上后 `torch.cuda.is_available()` 依旧返回 `True`，张量也能 `.cuda()`，但一算就报 `no kernel image is available`。驱动要求 **R570 以上**（`nvidia-smi` 看到的 `CUDA Version` 需 ≥ 12.8）。

若 cu128 稳定版装不上（Windows 上曾有打包延迟），改用 nightly：

```cmd
pip uninstall -y torch torchvision
pip install --pre torch torchvision --index-url https://download.pytorch.org/whl/nightly/cu128 --no-cache-dir
```

### 0.4 验证并取到第 1 节要用的 $PY

```cmd
conda activate toxpath

python -c "import sys, torch, timm, tensorboard, numpy, PIL; print(sys.version.split()[0], torch.__version__, timm.__version__, torch.version.cuda, torch.cuda.is_available(), torch.cuda.get_device_name(0))"

python -c "import torch; print(torch.cuda.get_device_capability(0)); print(torch.tensor([2.0], device='cuda') ** 2)"

where python
```

预期输出形如：

```
3.10.x 2.x.x 1.0.8 12.8 True NVIDIA RTX PRO 5000 Blackwell
(12, 0)
tensor([4.], device='cuda:0')
```

第二行必须是 `(12, 0)`，第三行必须真的算出 `tensor([4.])`。**只有 `is_available()` 为 `True` 不代表能用** —— sm_120 装错版本时它同样返回 `True`，但一算就崩，所以第三条是真正的判据。

`where python` 第一行就是第 1 节 `$PY` 的值。

想直接确认 wheel 里有没有 sm_120 内核：

```cmd
python -c "import torch; print(torch.cuda.get_arch_list())"
```

输出里必须包含 `'sm_120'`，形如 `['sm_70', 'sm_75', 'sm_80', 'sm_86', 'sm_89', 'sm_90', 'sm_100', 'sm_120']`。没有就是装错了 wheel。

## 1 设置路径与环境变量

```powershell
$PROJ = "K:\ToxPath AI Challenge\toxpath"
$PY   = "C:\Users\pc\.conda\envs\toxpath\python.exe"
$DATA = "D:\data\toxpath"
$CKPT = "D:\data\histoplus_cellvit_segmentor_20x.pt"

$env:TOXPATH_DATA_ROOT      = $DATA
$env:TOXPATH_HISTOPLUS_CKPT = $CKPT
```

`$PROJ` 与 `$PY` 已是实验室实测值，只需改 `$DATA`、`$CKPT`。

项目路径**含空格**，所有命令行里凡是用到 `$PROJ` 的地方都必须带双引号（SOP 里已全部带上）。

## 2 环境自检

```powershell
& $PY -c "import torch, timm, tensorboard; print(torch.__version__, timm.__version__, torch.cuda.get_device_name(0), round(torch.cuda.get_device_properties(0).total_memory/1024**3,1), 'GB')"
```

预期输出形如：`2.x.x 1.0.8 NVIDIA RTX PRO 5000 Blackwell 47.x GB`

## 3 数据与权重自检

```powershell
cd $PROJ
& $PY paths.py
```

`data_root` 与 `histoplus` 两行都必须是 `OK`，否则回到第 1 节改路径。

## 4 冒烟：确认 1024 输入不爆显存

```powershell
& $PY train_3class.py --fold 1 --epochs 1 --image-size 1024 `
  --batch-size 4 --grad-accum 2 --samples-per-epoch 8 `
  --max-train-samples 8 --max-val-samples 4 --val-every 1 `
  --num-workers 2 --device cuda --init encoder+neck `
  --out-dir "$PROJ\_lab\smoke\ckpt" --log-dir "$PROJ\_lab\smoke\log" --tb-dir "$PROJ\_lab\smoke\tb"
```

出现 `[Fold 1 Epoch 1/1]` 且退出码为 0 即为通过。这一步只验证显存与速度，验证集只有 4 张、`dice_fg` 会是 `nan`，属正常。

训练跑起来时另开一个窗口看显存余量，决定后面能不能加大 batch：

```powershell
nvidia-smi --query-gpu=memory.used,memory.total,utilization.gpu --format=csv -l 2
```

显存占用稳定在 30 GB 以下就用第 5 节的 `--batch-size 8`；超过 44 GB 就退回 `--batch-size 2`。冒烟就用正式 batch，OOM 在这里爆比在第 5 节爆划算。

## 5 单折训练（先跑折 1 看曲线）

```powershell
& $PY train_3class.py --fold 1 --epochs 12 --image-size 1024 `
  --batch-size 4 --grad-accum 2 --samples-per-epoch 800 --val-every 3 `
  --num-workers 6 --device cuda --init encoder+neck `
  --out-dir "$PROJ\checkpoints3" --log-dir "$PROJ\logs3" --tb-dir "$PROJ\logs3\tb"
```

## 6 五折训练

```powershell
& $PY train_3class.py --all-folds --epochs 12 --image-size 1024 `
  --batch-size 4 --grad-accum 2 --samples-per-epoch 800 --val-every 3 `
  --num-workers 6 --device cuda --init encoder+neck `
  --out-dir "$PROJ\checkpoints3" --log-dir "$PROJ\logs3" --tb-dir "$PROJ\logs3\tb"
```

## 7 续训

```powershell
& $PY train_3class.py --fold 1 --epochs 24 --image-size 1024 `
  --batch-size 4 --grad-accum 2 --samples-per-epoch 800 --val-every 3 `
  --num-workers 6 --device cuda --init none `
  --resume "$PROJ\checkpoints3\fold1\last.pt" `
  --out-dir "$PROJ\checkpoints3" --log-dir "$PROJ\logs3" --tb-dir "$PROJ\logs3\tb"
```

## 8 后台运行（断开远程桌面 / SSH 不中断）

`-WorkingDirectory` 已设为 `$PROJ`，所以参数里一律用**相对路径**。项目路径含空格，若把 `$PROJ` 写进 `-ArgumentList`，PowerShell 5.1 会按空格拆成两个参数，训练直接启动失败。

```powershell
New-Item -ItemType Directory -Force "$PROJ\logs3" | Out-Null

Start-Process -FilePath $PY -WorkingDirectory $PROJ -WindowStyle Hidden `
  -RedirectStandardOutput "$PROJ\logs3\train_fold1.out.log" `
  -RedirectStandardError  "$PROJ\logs3\train_fold1.err.log" `
  -ArgumentList @(
    "train_3class.py","--fold","1","--epochs","12","--image-size","1024",
    "--batch-size","4","--grad-accum","2","--samples-per-epoch","800",
    "--val-every","3","--num-workers","6","--device","cuda",
    "--init","encoder+neck",
    "--out-dir","checkpoints3","--log-dir","logs3","--tb-dir","logs3\tb"
  )

Get-Process python | Select-Object Id, StartTime, CPU
```

## 9 TensorBoard

```powershell
& $PY -m tensorboard.main --logdir "$PROJ\logs3\tb" --port 6006 --host 0.0.0.0
```

本机浏览器 `http://127.0.0.1:6006`，局域网 `http://<实验室机器IP>:6006`。

## 10 实时看日志

```powershell
# 后台运行时（第 8 节）
Get-Content "$PROJ\logs3\train_fold1.out.log" -Wait -Tail 20

# 逐 epoch 指标
Get-Content "$PROJ\logs3\fold1\train_log.json" -Tail 40
```

```powershell
# 查 PID（取启动时间最新的那个）
$p = Get-Process python | Sort-Object StartTime -Descending | Select-Object -First 1
$p.Id, $p.StartTime, [math]::Round($p.CPU/60,1)
```

```powershell
# 停止
Stop-Process -Id $p.Id
```

## 11 产物

```
checkpoints3\splits.json              折划分（切片名 → fold）
checkpoints3\fold1\best.pt            按 val dice_fg 选出的最佳权重
checkpoints3\fold1\last.pt            续训用
checkpoints3\summary.json             五折汇总
logs3\fold1\train_log.json            逐 epoch 指标
logs3\tb\fold1                        TensorBoard 事件
logs3\train_fold1.out.log             后台运行的控制台输出
```

## 12 显存不够时的调整顺序

```powershell
# 第一步：batch 减半，等效 batch 仍为 8
--batch-size 2 --grad-accum 4

# 第二步：开梯度检查点，省 30~40% 显存，慢 20~30%
--grad-checkpointing

# 第三步：编码器冻结，只训 Neck + Decoder
--freeze-encoder

# 第四步：降低采样量，先把训练跑起来
--samples-per-epoch 400
```

## 13 数据换到别处 / 换 73 张 WSI

```powershell
# 只改第 1 节的 $DATA，例如：
$DATA = "D:\data\toxpath_73"
$env:TOXPATH_DATA_ROOT = $DATA
& $PY paths.py
```

```powershell
# 采样量与验证频率按切片数放大
& $PY train_3class.py --all-folds --epochs 12 --image-size 1024 `
  --batch-size 4 --grad-accum 2 --samples-per-epoch 3500 --val-every 3 `
  --num-workers 6 --device cuda --init encoder+neck `
  --out-dir "$PROJ\checkpoints3_73" --log-dir "$PROJ\logs3_73" --tb-dir "$PROJ\logs3_73\tb"
```
