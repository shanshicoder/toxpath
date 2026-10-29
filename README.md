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
├── train_3class.py           # 训练入口（单折 / 多折 / 续训），曲线写 TensorBoard
├── inference.py              # checkpoint 加载、前向、Otsu 组织掩膜
├── predict_test.py           # 三分类 → 二值肿瘤掩膜，按官方规范出提交 zip
├── splits.py                 # 按切片做蛇形分折（切片数自适应）
├── paths.py                  # 数据根 / 权重路径解析，不写死盘符
├── scripts/
│   ├── make_3class_masks.py  # 二值 mask → 三分类彩色 mask + 标签索引（训练前置）
│   ├── run_train.ps1         # 实验室启动器：后台训练 + TensorBoard 服务 + 断远程桌面不中断
│   └── sync_github.ps1/.bat  # 提交 + 推送（大文件护栏，自动选 SSH/HTTPS）
├── requirements.txt
└── data/toxpath/             # 数据（不进仓库）
```

`scripts/make_3class_masks.py` 是数据准备脚本：`dataset` 与 `splits` 的报错信息
都指向它，删掉后换数据就没法重新生成三分类标签。`run_train.ps1` 是实验室
机器的启动入口（见下节）。

## 环境

`timm` **必须锁 1.0.8**，换版本会改变 ViT block 实现或 `out_indices` 语义。

```powershell
conda create -n toxpath python=3.10 -y
conda activate toxpath
pip install --index-url https://download.pytorch.org/whl/cu124 torch torchvision   # 4090
pip install timm==1.0.8 numpy Pillow tensorboard
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
| `--tb-dir` | `logs3/tb` | TensorBoard 事件根目录，每折一个 `fold{N}` 子目录 |
| `--tensorboard` / `--no-tensorboard` | 开 | 关掉就只训练、不写曲线 |

产物：

```
checkpoints3/
├── splits.json                 # 折划分（切片名 → fold），复现与对账用
├── fold1/best.pt               # dice_fg 最好的那个 epoch
├── fold1/last.pt               # 续训用
└── summary.json                # 多折汇总（全折模式）
logs3/tb/fold1/                 # TensorBoard 事件（曲线 + 预览图），见「TensorBoard」一节
logs3/fold1/train_log.json      # 逐 epoch 的 loss / dice / acc（纯文本备份）
logs3/train_*.out.log           # 启动器重定向的控制台输出
```

## 推送改动（笔记本 → GitHub）

改完代码跑一条命令，实验室机器 `git pull` 就能拿到：

```powershell
scripts\sync_github.bat                                     # 提交信息按时间自动生成
scripts\sync_github.bat -Message "fix: 修正 fold5 采样权重"  # 自己写 message
scripts\sync_github.bat -DryRun                             # 只打印会做什么
scripts\sync_github.bat -NoCommit                           # 已手动 commit 过，只推送
```

脚本做四件事：校验仓库与身份 → 扫待提交文件拦住权重/数据/大文件 → `add` + `commit` → `push`，
最后打印实验室机器该执行的命令。

**协议自动选择**（`-Via auto`，默认）：先用 `BatchMode=yes` 探一次 SSH 能否免密认证。
私钥带 passphrase 又没装 ssh-agent 时会自动改用 HTTPS——凭据从 Windows 凭据管理器取，
不用输密码。也可以强制 `-Via ssh` 或 `-Via https`。

两个踩过的环境坑，脚本里都处理了：

- **代理**：git 全局配置曾把 `http.proxy` 写死到 `127.0.0.1:7890`，而那个端口没在监听，
  连 443 都会失败（报 `Failed to connect ... over proxy`）。走 HTTPS 时脚本显式
  `-c http.proxy= -c https.proxy=` 清空后直连。
- **upstream**：用 HTTPS URL 推送时不带 `-u`，推完再 `fetch` 更新 `origin/main`
  并把跟踪关系指回 `origin`，避免 `git status` 一直显示 ahead、下次又重复推送。

失败时按退出码给排查清单：passphrase 输错、publickey 未授权、non-fast-forward、代理不通。

## 实验室机器：跑起来 + 看曲线（4090）

### 一次性准备

```powershell
git clone git@github.com:shanshicoder/toxpath.git D:\work\toxpath   # 或 git pull
conda create -n toxpath python=3.10 -y ; conda activate toxpath
pip install --index-url https://download.pytorch.org/whl/cu124 torch torchvision
pip install timm==1.0.8 numpy Pillow tensorboard
```

- **权重（271 MB）**：`git checkout 2ed284a -- scripts/fetch_weights.ps1` 取回下载
  脚本（自动校验 SHA256），或手动从 GitHub Release 下、U 盘拷到 `weights\`；
- **数据**：放 `data\toxpath\`，或放别处后设 `TOXPATH_DATA_ROOT`；
- **冒烟**（约 1 分钟，必须先跑）：
  `.\scripts\run_train.ps1 -Mode smoke -PythonExe (Get-Command python).Source`

### 正式训练

```powershell
# 单折（先跑 1 折看曲线）；-PythonExe 填 conda 环境里的 python.exe 全路径
.\scripts\run_train.ps1 -Mode fold -Fold 1 -Epochs 12 -PythonExe "C:\...\envs\toxpath\python.exe"

# 五折
.\scripts\run_train.ps1 -Mode all -Epochs 12

# 续训
.\scripts\run_train.ps1 -Mode resume -Fold 1 -Epochs 20
```

脚本会把训练**脱离当前窗口**启动（关远程桌面、断 SSH 都不影响），并打印
PID 与日志路径：

```powershell
Get-Content logs3\train_fold_*.out.log -Wait -Tail 20   # 实时刷屏
Get-Process -Id <PID>                                   # 还活着吗
Stop-Process -Id <PID>                                  # 停止
```

### TensorBoard 看曲线

训练时每个 epoch 自动写 `logs3/tb/fold{N}/`，另开一个窗口把服务起来：

```powershell
.\scripts\run_train.ps1 -Mode tb              # 默认 6006，加 -Port 6010 换端口
# 等价于：python -m tensorboard.main --logdir logs3/tb --port 6006 --host 0.0.0.0
```

浏览器打开 `http://127.0.0.1:6006`（本机）或 `http://<实验室机器IP>:6006`（局域网）。
训练与看板互不依赖：先开训练后开看板、看板关掉再开，曲线都在，服务会自己
发现新事件文件。

页面上能看到的：

| 页签 | 内容 |
|---|---|
| SCALARS | `fold1/train_step/loss`（每 `--log-every` 步）、`fold1/train/*` 与 `fold1/val/*`（每 epoch）、`time/it_per_s`、`time/epoch_min`、`train/lr` |
| IMAGES | `fold1/preview`：每 epoch 一张 `原图 \| 真值 \| 预测` 三联图，按 step 滑块回看 |
| TEXT | `fold1/config`：这次运行的尺寸、lr、类别权重、验证切片等配置快照 |

几条使用要点：

- **五折各是一个独立的 run**（`fold1`…`fold5`），左侧勾多个就能把同一指标叠在一张图里对比；
- 想把 `val/dice_fg` 和"全猜空"比：同理勾上各折，注意 `fold1/val/dice_fg_allbg_baseline` 记录的
  是验证集里真值全空的 patch 占比——曲线贴着它走说明模型还没学到东西；
- **重跑同一折会自动换目录**（`fold1_20261029-203015`），不会和上次的曲线叠在一起；
- 局域网访问不通：管理员 PowerShell 执行
  `netsh advfirewall firewall add rule name=TensorBoard dir=in action=allow protocol=TCP localport=6006`；
  跨网段用 `ssh -L 6006:127.0.0.1:6006 用户名@实验室IP` 转发；
- **无鉴权**，只当内网工具，别暴露公网；只本机看就加 `--host 127.0.0.1`；
- 不想写曲线加 `--no-tensorboard`（`tensorboard` 只是可选依赖，没装也照常训练）。

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
