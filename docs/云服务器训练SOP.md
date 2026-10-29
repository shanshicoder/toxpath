# ToxPath 云服务器训练 SOP（AutoDL / 恒源云，Linux + 单卡）

> 适用机器：云上按量计费实例（RTX 4090 24 GB 或 A40 / A6000 48 GB），Ubuntu 镜像 + root
> 适用代码：`7b8df82` 及之后（连通域三分类标签 + TensorBoard 监控 + `scripts/make_upload_package.ps1`）
> 姊妹文档：`docs/本地训练SOP.md`（实验室 Windows 机器）；本文只讲**云上**特有的部分

---

## 0. 和本地 SOP 的差别（先看这张表）

| 环节 | 实验室本地 | 云服务器 |
|---|---|---|
| 开机方式 | 一直开着 | **按量计费，跑完就关机**，忘记关就是烧钱 |
| 代码到位 | `git pull` | `git clone`（或 scp 一个几百 KB 的代码包） |
| 数据到位 | 机器上已有 | **本地打包 → rsync 上传 → 解压校验** |
| 权重 | 已放好 | 本地下载好再 scp（云上拉 GitHub Release 常断） |
| 启动器 | `run_train.ps1`（PowerShell） | **不可用**，用 `nohup` + `tmux`（第 6 节给了 `run.sh`） |
| 断线 | 关远程桌面没事 | SSH 断开**会杀掉前台进程**，必须 `nohup` / `tmux` |
| 监控 | `127.0.0.1:6006` | SSH 端口转发，或平台自带的「自定义服务」 |
| 突发风险 | 睡眠 / 断电 | **关机忘了下载结果**、实例被释放、余额不足被停 |
| 计费 | 电费 | 约 2 元/h（4090）、3.5 元/h（A40） |

**三条铁律：**

1. **先把「无卡模式」用满**：装环境、传数据、解压、校验都不需要 GPU，别在有卡模式里做。
2. **训练一律 `nohup` 或 `tmux`**，SSH 一断前台进程就没了。
3. **跑完立刻下载结果再释放实例**——释放会清空数据盘，且不可恢复。

---

## 1. 费用预估（先算清再租）

按 2026 年 10 月行情：4090 约 2.0–2.3 元/h，A40 48G 约 3.5 元/h，按秒计费、关机不收 GPU 费。

| 场景 | 单折耗时 | 5 折耗时 | 4090 费用 | A40 费用 |
|---|---:|---:|---:|---:|
| 4953 张（5 切片），12 epoch | 约 2 h | 约 10 h | **约 20 元** | 约 40 元 |
| 7.2 万张（73 WSI），12 epoch、`--val-every 3` | 约 17 h | 约 85 h | **约 170 元** | 约 300 元 |
| 7.2 万张，先把 `--samples-per-epoch` 降到 1500 试水 | 约 10 h | 约 50 h | 约 100 元 | 约 175 元 |

> 这些是按「训练 ~1 s/样本、验证 ~0.35 s/张」外推的，**必须用第 7 节的实测算一次再决定跑几折**。
> 省钱要点：分折跑、跑完就关机；不要一口气开机三天，中途可以停机改代码。

---

## 2. 选实例

| 项 | 建议 | 原因 |
|---|---|---|
| GPU | 先租 4090 24G 跑通，正式跑 73 张时换 A40 48G | 24G 只能 batch 1；48G 能 batch 2–3，验证吞吐翻倍 |
| 内存 | **≥ 64 GB** | 数据加载 + 解压很吃内存，32 GB 在解压 17 GB 数据时会吃紧 |
| CPU | 8–16 核 | `num-workers` 越多，GPU 等数据越少 |
| 系统盘 | 50–100 GB | 放代码和环境 |
| 数据盘 | 5 切片 20 GB 够；73 张给 **80 GB** | 17 GB patch 解压后 + 每折 1 GB checkpoint × 5 + 日志 |
| 计费 | **按量 + 关机不计费** | 别买包月，负载是间歇性的 |

镜像选 **PyTorch 2.x + CUDA 12.x + Ubuntu 22.04**，自带 torch 和 conda，省 20 分钟。

---

## 3. 无卡模式：上传代码、权重、数据

以下都在 `ssh -p <端口> root@<host>` 里执行。

### 3.0 本地一次打包（推荐）

在本地 PowerShell 里跑一次，得到「代码包 + 数据包 + 校验清单」三样东西：

```powershell
cd E:\project\toxpath
powershell -ExecutionPolicy Bypass -File scripts\make_upload_package.ps1
```

产物落在 `dist\`（已在 `.gitignore` 里，不会被提交）：

| 文件 | 大小 | 条目 | 内容 |
|---|---:|---:|---|
| `toxpath_code.tar.gz` | 约 100 KB | 约 30 | `git archive HEAD`，归档根就是仓库根 |
| `toxpath_data.tar` | 1,306,798,592 B（1.22 GiB） | 9,929（9,907 文件 + 22 目录） | `train_patches/` + `train_masks_classes/` + CSV |
| `SHA256SUMS` | 约 200 B | — | 服务器上 `sha256sum -c` 直接校验（**LF 行尾**） |
| `CHECKSUMS.txt` | 约 700 B | — | 人读清单：commit、大小、文件数、解压命令 |

> 代码包大小随 commit 变化（当前约 100 KB），精确值以脚本输出和 `CHECKSUMS.txt` 为准；数据包大小只在换数据时变。

脚本内置五项检查，**任何一项不过就中止并报红，不会产出可上传的包**：

1. `patches in tar` = `masks in tar` = 磁盘上的文件数（本次 4953 / 4953 / 4953）
2. `CSV 行数` = patch 数 + 1（本次 4954，含表头）
3. 归档内路径为正斜杠（Windows 自带 `tar.exe` 是 bsdtar，已实测不会写出反斜杠文件名）
4. `SHA256SUMS` 必须是 LF 行尾 —— PowerShell 的 `Set-Content` 默认写 CRLF，GNU `sha256sum` 会把 `\r` 当成文件名的一部分，于是报 `No such file or directory`。这个坑真踩过，所以写进脚本自检
5. 生成 SHA256 并写入 `SHA256SUMS`

| 参数 | 用途 |
|---|---|
| `-CodeOnly` | 只重打代码包（改了 `.py` 时用，跳过 1.3 GB 数据）。`dist\` 里若已有数据包，会复用它补全 `SHA256SUMS`（多花约 10 秒算哈希），清单不会缺行 |
| `-WithMasks` | 额外带上 `train_masks/`（只在云上要重造标签时才需要） |
| `-OutDir D:\dist` | 换输出目录 |

> **不要拿哈希当跨机器的固定值。** 同一份数据用 GNU tar 和 Windows 自带的 bsdtar 打，字节数差 9,728，哈希完全不同。完整性判据是「三个计数相等」+ 上传后用**本次生成**的 `SHA256SUMS` 校验，不是某个写死的哈希。

### 3.1 目录约定

```bash
mkdir -p /root/autodl-tmp/work
cd /root/autodl-tmp/work
```

**代码和数据都放 `/root/autodl-tmp`（数据盘）**，不要放 `/root`（系统盘）——换机器、重置镜像时数据盘才是保命的。

### 3.2 代码（几百 KB，二选一）

```bash
# 方式 A：能连 GitHub（推荐，后续可 git pull）
git clone https://github.com/shanshicoder/toxpath.git
cd toxpath
git log --oneline -1        # 记下 commit，出问题好对账
```

私有仓库需要 token：`git clone https://<你的PAT>@github.com/shanshicoder/toxpath.git`。

```powershell
# 方式 B：云上连不上 GitHub，用 3.0 打好的包（本地 PowerShell 执行）
cd E:\project\toxpath
scp -P 12345 dist\toxpath_code.tar.gz root@<host>:/root/autodl-tmp/work/

# 改了代码？先重打（1 秒，不带数据）
powershell -ExecutionPolicy Bypass -File scripts\make_upload_package.ps1 -CodeOnly
# 等价手动命令：git archive --format=tar.gz -o dist\toxpath_code.tar.gz HEAD
```

```bash
# 服务器上解压（归档内容是仓库根目录，直接铺到 toxpath/）
cd /root/autodl-tmp/work
mkdir -p toxpath && tar -xzf toxpath_code.tar.gz -C toxpath
ls toxpath/train_3class.py                 # 确认解压正确
ls toxpath/labels.py toxpath/scripts/      # 连通域标签与启动器都在
```

### 3.3 权重（271 MB）

云上直接下 GitHub Release 经常断在半路，**本地先下好再传**最稳：

```powershell
# 本地
scp -P 12345 "E:\project\histoplus_cellvit_segmentor_20x.pt" root@<host>:/root/autodl-tmp/
```

```bash
# 服务器校验，必须完全一致
cd /root/autodl-tmp
ls -l histoplus_cellvit_segmentor_20x.pt      # 284507558 字节
sha256sum histoplus_cellvit_segmentor_20x.pt
# 期望：aa19dc31c109c0a4fcf7bd773559c45480cee792a331ae0a9ffe42eb08267bfc
```

### 3.4 数据（关键：一定先打包，别逐个文件传）

| 目录 | 5 切片 | 73 张 | 要传吗 |
|---|---:|---:|---|
| `train_patches/` | 1,221 MB | 约 17.7 GB | **要** |
| `train_masks_classes/` | 16.9 MB | 约 245 MB | **要** |
| `_patch_labels_3class.csv` | 0.2 MB | 约 3 MB | **要** |
| `train_masks/` | 14.4 MB | 约 210 MB | 只在云上要重造标签时才传 |
| `train_images/`（金字塔 TIFF） | 2.5 GB | 数十 GB | **不要**，训练不读 |
| `train_labels/`（GeoJSON） | — | — | **不要**，训练不读 |

```powershell
# 本地：用 3.0 打好的包（dist\toxpath_data.tar，1.22 GiB）
# 等价手动命令（JPEG 已压缩，用 tar 不压缩；小文件多，逐个 scp 会慢十倍）：
#   cd E:\project\toxpath\data\toxpath
#   tar -cf ..\..\dist\toxpath_data.tar train_patches train_masks_classes _patch_labels_3class.csv
# 73 张时这个包约 18 GB，打包本身要十几分钟
```

**上传用 rsync 而不是 scp**（断了能续，18 GB 一次传完不现实）：

```powershell
# 本地（Windows 10+ 自带 rsync 的话；没有就用 WSL，或 WinSCP 拖文件）
cd E:\project\toxpath
rsync -avP -e "ssh -p 12345" dist\toxpath_data.tar dist\SHA256SUMS root@<host>:/root/autodl-tmp/work/
```

没有 rsync 就用 WinSCP（SFTP，支持断点续传），把 `toxpath_data.tar` 和 `SHA256SUMS` 拖到 `/root/autodl-tmp/work/`。

```bash
# 服务器：先验哈希，再解压，再数文件
cd /root/autodl-tmp/work
sha256sum -c SHA256SUMS                   # 两行都要 OK（code 包也在里面，先传齐再跑）
mkdir -p toxpath/data/toxpath
tar -xf toxpath_data.tar -C toxpath/data/toxpath
rm toxpath_data.tar                       # 解压完就删，省磁盘

cd toxpath/data/toxpath
find train_patches -type f | wc -l        # 4953（或 72000）
find train_masks_classes -type f | wc -l  # 必须与上一行相等
wc -l _patch_labels_3class.csv            # 4954（或 72001，含表头）
```

**这三个数字必须对得上**。patch 数与 mask 数不等、或 CSV 行数少了，说明传输截断，重传。

> 只传了数据包、没传代码包时，`sha256sum -c` 会报 `toxpath_code.tar.gz: No such file` —— 这不是损坏，是缺文件。要么补传，要么只用它校验数据包：`sha256sum -c SHA256SUMS --ignore-missing`。

---

## 4. 环境（5 分钟）

```bash
nvidia-smi                                  # 有卡模式下才看得到
python -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0))"
python -c "import timm; print(timm.__version__)"
```

镜像自带 torch 的话不用装。**只需要补三样**：

```bash
pip install timm==1.0.8 tensorboard numpy Pillow
```

> `timm` **必须 1.0.8**。镜像里如果是别的版本，直接覆盖安装，否则 ViT block 实现不同，权重加载会静默错位——loss 照降，指标不对。
> 不要新建 conda 环境：镜像的 base 已经配好 CUDA，新建一个再装 torch 要重下 2 GB，纯浪费时间。

验证：

```bash
python -c "import timm; print(timm.__version__)"        # 1.0.8
python -c "import tensorboard; print('tb ok')"
```

---

## 5. 冒烟（1 分钟，前台跑，不用后台）

```bash
cd /root/autodl-tmp/work/toxpath

python train_3class.py \
  --data-root /root/autodl-tmp/work/toxpath/data/toxpath \
  --fold 1 --epochs 1 --image-size 512 \
  --max-train-samples 8 --max-val-samples 4 --samples-per-epoch 8 \
  --num-workers 0 --init none \
  --out-dir _smoke/ckpt --log-dir _smoke/log --tb-dir _smoke/tb
```

**通过标准：**

- 开头 `device=cuda  amp=True`、`gpu=... vram=... GB`；
- 路径三行（`project_root` / `data_root` / `histoplus`）都指向真实位置；
- 出现 `[ToxPath3Dataset] mode=train ... 类别分布: 0=... 1=... 2=...`；
- `[Fold 1 Epoch 1/1]` 汇总行、`train_loss` 是有限值；
- `_smoke/ckpt/last.pt` 已生成。

顺手记下这一轮的 `it/s`，用来外推正式训练时长。跑完删掉：

```bash
rm -rf _smoke
```

**冒烟里特意用 `--init none` 是为了不依赖权重**，所以这一步只验数据链路。接下来单折训练那一步会加载权重，权重路径错会在那时报出来——所以第 3.3 节的哈希校验别省。

---

## 6. 正式训练

### 6.1 先建一个启动脚本（一次，之后都用它）

Linux 上 `run_train.ps1` 用不了。复制下面整段粘到服务器，生成 `run.sh`：

```bash
cat > /root/autodl-tmp/work/toxpath/run.sh <<'EOF'
#!/usr/bin/env bash
# ToxPath 云上训练启动器：后台运行 + 日志落盘 + 可选训练完自动关机
# 用法: bash run.sh <fold|all|resume> <fold号> <epochs> <shutdown: yes|no>
set -uo pipefail

ROOT=/root/autodl-tmp/work/toxpath
DATA=$ROOT/data/toxpath
CKPT=/root/autodl-tmp/histoplus_cellvit_segmentor_20x.pt
MODE=${1:-fold}; FOLD=${2:-1}; EPOCHS=${3:-12}; SHUTDOWN=${4:-no}
STAMP=$(date +%Y%m%d_%H%M%S)
mkdir -p "$ROOT/logs3"
LOG=$ROOT/logs3/train_${MODE}_${STAMP}.log

case "$MODE" in
  fold)   ARGS="--fold $FOLD --epochs $EPOCHS --num-workers 8 --val-every 1" ;;
  all)    ARGS="--all-folds --epochs $EPOCHS --num-workers 8 --val-every 1" ;;
  resume) ARGS="--fold $FOLD --epochs $EPOCHS --num-workers 8 --resume checkpoints3/fold$FOLD/last.pt" ;;
  *) echo "mode 必须是 fold / all / resume"; exit 2 ;;
esac

cd "$ROOT"
nohup bash -c "python train_3class.py --data-root '$DATA' --histoplus-ckpt '$CKPT' $ARGS >> '$LOG' 2>&1; rc=\$?; echo \"[done] exit=\$rc\" >> '$LOG'; if [ \$rc -eq 0 ] && [ '$SHUTDOWN' = yes ]; then sleep 30; shutdown; fi" >/dev/null 2>&1 &

echo "pid    : $!"
echo "log    : $LOG"
echo "tail   : tail -f $LOG"
echo "stop   : kill \$(pgrep -f train_3class.py)"
[ "$SHUTDOWN" = yes ] && echo "注意   : 训练成功结束后会自动关机（不下载结果！先确认下面的下载流程）"
EOF
chmod +x /root/autodl-tmp/work/toxpath/run.sh
```

自动关机只在**退出码 0**（训练正常结束）时触发；报错退出会留在开机状态，方便你排查——这条是有意的，别改成无条件 `shutdown`，否则一个参数写错就白等一次开机。

### 6.2 单折（先跑这一条看曲线）

```bash
cd /root/autodl-tmp/work/toxpath
bash run.sh fold 1 12 no
```

### 6.3 五折（过夜，跑完自动关机）

```bash
bash run.sh all 12 yes
```

五折时前面折的结果不会因为后面折失败而丢。`summary.json` 最后一次性写。

### 6.4 续训

```bash
bash run.sh resume 1 24 no
```

只恢复权重和 epoch，**不恢复优化器状态**，所以头 1–2 轮 loss 会抖一下。`last.pt` 每 `--save-every`（默认 2）轮落一次。

### 6.5 参数怎么调（分卡型）

| 参数 | 4090 24G | A40 / A6000 48G | 说明 |
|---|---|---|---|
| `--batch-size` | 1 | 2（显存够可试 3） | 1024 输入下这是硬约束 |
| `--grad-accum` | 8 | 4 | 等效 batch 保持 8 |
| `--grad-checkpointing` | 建议开 | 不用开 | 24G 上省显存；48G 上白损失速度 |
| `--num-workers` | 6–8 | 8–12 | 卡 CPU 核数，别超过核数 |
| `--val-every` | 1 | 1 | 73 张数据时改成 3（见下） |
| `--samples-per-epoch` | 800（4953 张） | 同左 | 73 张时 **3500** |

48G 卡的单折命令（`run.sh` 没透传这些，直接跑）：

```bash
nohup python train_3class.py \
  --data-root /root/autodl-tmp/work/toxpath/data/toxpath \
  --histoplus-ckpt /root/autodl-tmp/histoplus_cellvit_segmentor_20x.pt \
  --fold 1 --epochs 12 --batch-size 2 --grad-accum 4 --num-workers 12 \
  --val-every 1 \
  > logs3/fold1_a40.log 2>&1 &
echo "pid=$!"
```

**73 张 WSI（7.2 万 patch）必须改这三个参数**，否则 5 折算不完：

```bash
--samples-per-epoch 3500      # 默认 800 只覆盖 1.1%
--val-every 3                 # 每折验证集约 1.4 万张，每轮都验会拖死
--val-batch-size 2            # 显存够就开，验证吞吐接近翻倍
```

并且**给新数据单独开输出目录**（`--out-dir checkpoints3_73 --log-dir logs3_73 --tb-dir logs3_73/tb`），否则会覆盖 5 切片时代的 `splits.json`，旧 checkpoint 就再也对不上划分了。

---

## 7. 训练中监控

```bash
cd /root/autodl-tmp/work/toxpath

tail -f logs3/train_all_*.log                                  # 日志（Ctrl+C 只退出 tail）
nvidia-smi                                                     # 显存 / 利用率 / 温度
nvidia-smi --query-gpu=utilization.gpu,memory.used,temperature.gpu --format=csv -l 5
df -h /root/autodl-tmp                                          # 磁盘
ls -lh checkpoints3/fold1/                                      # checkpoint
pgrep -af train_3class.py                                       # 进程还在吗
```

**关键判断：**

- `nvidia-smi` 里 GPU 利用率长期 < 50% → 卡在数据加载，把 `--num-workers` 调大；
- 日志里 `(xx.x min)` 突然翻倍 → I/O 或降频，看 `df -h` 是否快满；
- `dice_fg` 贴着 `val/dice_fg_allbg_baseline` 不动 → 模型还没学到东西，先别加 epoch。

### TensorBoard 看曲线

```bash
cd /root/autodl-tmp/work/toxpath
nohup python -m tensorboard.main --logdir logs3/tb --port 6006 --host 0.0.0.0 \
  > logs3/tb.out.log 2>&1 &
```

**访问方式（二选一）：**

```powershell
# 方式 A：SSH 隧道（通用，任何平台都能用）
ssh -p 12345 -L 6006:127.0.0.1:6006 root@<host>
# 保持这个窗口不关，本地浏览器开 http://127.0.0.1:6006
```

方式 B：AutoDL 控制台 → 实例 → 「自定义服务」→ 端口填 `6006` → 生成访问链接。
（要求 tensorboard 监听 `0.0.0.0`，上面命令已经是。）

`--host 0.0.0.0` + 公网端口是**无鉴权**的，只在隧道/平台映射下用，别自己开安全组直连。

---

## 8. 训练完：下载结果再关机（最容易出事的一步）

### 8.1 先确认真的跑完了

```bash
cd /root/autodl-tmp/work/toxpath
tail -3 logs3/train_all_*.log          # 应看到 [done] exit=0 和 [汇总] 表
cat checkpoints3/summary.json          # 5 条记录
grep -h "dice_fg 均值" logs3/train_all_*.log
ls -lh checkpoints3/fold*/best.pt      # 5 个文件，每个约 456 MB
```

### 8.2 打包（只打要的，别把 17 GB 数据再传回来）

```bash
cd /root/autodl-tmp/work/toxpath
tar -czf results3.tar.gz \
  checkpoints3/splits.json \
  checkpoints3/summary.json \
  checkpoints3/fold*/best.pt \
  checkpoints3/fold*/history.json \
  logs3/fold*/train_log.json \
  logs3/train_*.log \
  logs3/tb

ls -lh results3.tar.gz                 # 5 折约 2.3 GB
```

```powershell
# 本地下载
scp -P 12345 root@<host>:/root/autodl-tmp/work/toxpath/results3.tar.gz E:\project\
```

想省流量就只下 `summary.json` + 每折 `best.pt`（曲线可以事后用 `history.json` 重画，或者干脆把 `logs3/tb` 留在云上，下次开机再接 TensorBoard 看）。

### 8.3 关机 vs 释放

| 操作 | GPU 计费 | 数据盘 | 什么时候用 |
|---|---|---|---|
| **关机** | 停 | **保留** | 明天还要接着跑、只是过夜 |
| **释放** | 停 | **清空** | 这个实验彻底结束，且结果已下载 |

释放前再确认一遍 `results3.tar.gz` 已经落到本地，`ls -lh` 看过大小不是 0。这一步没有回收站。

---

## 9. 故障速查

| 现象 | 原因 | 处理 |
|---|---|---|
| 打包脚本报 `verification failed` | patch 数与 mask 数不等，或 CSV 行数和 patch 数对不上 | 先跑 `make_3class_masks.py` 重建标签；再确认 `data/toxpath` 没缺切片 |
| 服务器 `sha256sum -c` 报 `FAILED` | 传输截断（rsync 被中断 / 磁盘满） | 重传该包，`rsync -avP` 会续传；**不要解压损坏的包** |
| `sha256sum -c` 报 `No such file` | 只传了一个包 | 补传另一个，或加 `--ignore-missing` 只校验已有的 |
| `找不到 patch 索引：_patch_labels_3class.csv` | 数据没解压到 `--data-root` 指定层 | 确认 `--data-root` 指向**含 `train_patches/` 的那一层** |
| `找不到 HistoPLUS 权重` | 路径不对 / 只传了一半 | 重新 `sha256sum` 校验，284507558 字节 |
| `timm` 版本不是 1.0.8 | 镜像自带别的版本 | `pip install timm==1.0.8`，装完再验一次 |
| `CUDA out of memory` | batch 太大 / 没开 checkpointing | 4090 上 `--batch-size 1 --grad-checkpointing`；A40 上降 `--batch-size 1` |
| GPU 利用率长期 < 50% | DataLoader 喂不上 | `--num-workers` 加到核数附近；数据盘 I/O 慢就把数据放本地盘 |
| 训练进程莫名消失 | SSH 断了，前台进程被杀 | 一律用 `run.sh`（`nohup`）或 `tmux new -s t` |
| `DataLoader worker (pid xxx) is killed` | 内存不足 | 降 `--num-workers`；确认实例内存 ≥ 64 GB |
| 日志里中文乱码 | Linux 默认 locale | `export LANG=C.UTF-8 PYTHONUTF8=1` 后再启动 |
| 磁盘满 | 数据 tar 没删 / checkpoint 堆积 | `rm toxpath_data.tar`；删掉 `_smoke`、旧折的 `last.pt` |
| TensorBoard 打不开 | 没监听 0.0.0.0 / 隧道没建 | 重起加 `--host 0.0.0.0`；确认 `ssh -L` 窗口还开着 |
| 训练跑到一半实例停了 | 余额不足 / 平台回收 | 充值后开机，`bash run.sh resume <折号> <总epochs> no` |
| `dice_c1` 是 `nan` 或 0 | 该折验证集几乎没有类 1 | 正常，看 `dice_c2` / `dice_fg` |
| 续训后 loss 突然变大 | 优化器状态没恢复（设计如此） | 跑 1–2 轮会回来，不用管 |

---

## 10. 收工清单

**开机后（有卡之前）**

- [ ] 无卡模式：`git clone`（或传代码包）、传权重、传数据 tar
- [ ] `sha256sum` 权重 = `aa19dc31…7bfc`
- [ ] patch 数 = mask 数 = CSV 行数 − 1
- [ ] `pip install timm==1.0.8 tensorboard`

**开训前**

- [ ] `nvidia-smi` 有卡、显存未被占
- [ ] 冒烟通过（`_smoke/ckpt/last.pt` 存在）
- [ ] 73 张数据：`--samples-per-epoch 3500 --val-every 3`、独立 `--out-dir`
- [ ] 启动命令是 `run.sh`（后台），不是裸 `python`
- [ ] 自动关机参数确认（`no` 表示不自动关，适合第一次跑）

**收工**

- [ ] `summary.json` 5 条、`dice_fg 均值 ± 标准差`已记下
- [ ] `results3.tar.gz` 已下载到本地，`ls -lh` 大小正常
- [ ] 先**关机**（保留数据），确认不再需要后才**释放**
- [ ] 本地的 `splits.json` 与 `summary.json` 归档到实验记录

---

## 附录 A：与本地 SOP 的命令对照

| 目的 | 本地（PowerShell） | 云上（bash） |
|---|---|---|
| 冒烟 | `.\scripts\run_train.ps1 -Mode smoke` | 第 5 节的前台命令 |
| 单折 | `.\scripts\run_train.ps1 -Mode fold -Fold 1 -Epochs 12` | `bash run.sh fold 1 12 no` |
| 五折 | `.\scripts\run_train.ps1 -Mode all -Epochs 12` | `bash run.sh all 12 yes` |
| 续训 | `.\scripts\run_train.ps1 -Mode resume -Fold 1 -Epochs 24` | `bash run.sh resume 1 24 no` |
| 看板 | `.\scripts\run_train.ps1 -Mode tb` | `python -m tensorboard.main --logdir logs3/tb --port 6006 --host 0.0.0.0` |
| 日志 | `Get-Content logs3\train_*.out.log -Wait` | `tail -f logs3/train_*.log` |
| 停训练 | `Stop-Process -Id <PID>` | `pkill -f train_3class.py` |

## 附录 B：73 张 WSI 的云上预算（按 7.2 万 patch 估）

| 项目 | 大小 / 耗时 | 备注 |
|---|---:|---|
| 上传数据 | 约 18 GB | rsync 断点续传，按上行 5 MB/s 算约 1 小时 |
| 磁盘占用 | 约 25 GB | 数据 18 GB + checkpoint 4.6 GB + 日志 |
| 单折 12 epoch | 约 17 h | `--samples-per-epoch 3500 --val-every 3` |
| 5 折 | 约 85 h | 建议分折跑，每折跑完关机 |
| A40 费用 | 约 300 元 | 3.5 元/h |
| 4090 费用 | 约 170 元 | 2.0 元/h，但只能 batch 1，单折更慢 |

**先花 10 元验证管线**：租 4090 一小时，只跑 1 折 1 epoch，看日志里的真实分钟数，再乘出总时长和预算。这 10 元能挡住 300 元的错误。
