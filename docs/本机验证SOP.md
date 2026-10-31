# 本机训练 SOP（Windows / CPU）

每开一个新终端，先执行第 1 节。

## 1 定位项目与解释器

```powershell
cd E:\project\toxpath
$PY = "D:\Program Files\anaconda3\envs\toxpath\python.exe"
```

## 2 环境自检

```powershell
& $PY -c "import torch, timm, tensorboard; print(torch.__version__, timm.__version__, 'cuda=', torch.cuda.is_available())"
```

## 3 数据自检

```powershell
& $PY paths.py
```

## 4 权重自检

```powershell
& $PY -c "from model import ToxPathConfig, ToxPathSegmentor; print(ToxPathSegmentor(ToxPathConfig(image_size=224, num_classes=3)).load_histoplus_backbone(r'E:\project\histoplus_cellvit_segmentor_20x.pt').summary())"
```

## 5 冒烟（约 1.5 分钟）

```powershell
& $PY train_3class.py --fold 5 --epochs 1 --image-size 224 `
  --samples-per-epoch 8 --max-val-samples 16 --num-workers 0 `
  --device cpu --init none `
  --out-dir _local\smoke\ckpt --log-dir _local\smoke\log --tb-dir _local\smoke\tb
```

## 6 本机训练（约 12 分钟）

```powershell
# 必须用 fold 5：fold 1 的验证切片前 468 张全空，dice 恒为 nan、不出 best.pt
& $PY train_3class.py --fold 5 --epochs 3 --image-size 224 `
  --samples-per-epoch 48 --max-val-samples 32 --num-workers 0 `
  --device cpu --init encoder+neck `
  --histoplus-ckpt "E:\project\histoplus_cellvit_segmentor_20x.pt" `
  --out-dir _local\ckpt --log-dir _local\log --tb-dir _local\tb
```

## 7 续训（约 13 分钟）

```powershell
& $PY train_3class.py --fold 5 --epochs 6 --image-size 224 `
  --samples-per-epoch 48 --max-val-samples 32 --num-workers 0 `
  --device cpu --init none --resume _local\ckpt\fold5\last.pt `
  --out-dir _local\ckpt --log-dir _local\log --tb-dir _local\tb
```

## 8 TensorBoard

```powershell
& $PY -m tensorboard.main --logdir _local\tb --port 6006 --host 127.0.0.1
```

浏览器打开 `http://127.0.0.1:6006`

## 9 产物

```
_local\ckpt\fold5\best.pt          按 val dice_fg 选出的最佳权重
_local\ckpt\fold5\last.pt          续训用的最后一轮
_local\ckpt\fold5\history.json     逐 epoch 指标
_local\ckpt\splits.json            折划分
_local\log\fold5\train_log.json    与 history 同内容
_local\tb\fold5                    TensorBoard 事件
```

## 10 停止

`Ctrl+C`

## 11 调参

```powershell
# 增大输入（时间约 4 倍）
--image-size 448

# 加长训练
--epochs 6 --samples-per-epoch 96

# 换折（2/3/4 的验证切片前 194/103/7 张才出现首个正样本）
--fold 4 --max-val-samples 64

# 只要权重不写曲线
--no-tensorboard
```
