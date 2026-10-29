# _deprecated —— 旧方案存档（不要再使用）

本目录保存的是 **已被废弃的旧方案**，当前项目只使用：

| 用途 | 当前文件 |
|---|---|
| 生成三分类 mask | `scripts/make_3class_masks.py` |
| 数据集（1024 输入） | `dataset/toxpath3_dataset.py` |
| 训练脚本 | `train_3class.py` |
| mask 目录 | `data/toxpath/train_masks_classes/` |

## 这里为什么会有这些文件

项目在一天内迭代过三套标签方案，旧文件被移到这里以免误用：

1. **二分类（0/255）**：`docs/数据集分析与训练方案.md` —— 最初把 HistoPLUS
   改成二分类区域分割，后来演化出区域级三分类标签。
2. **彩色三分类（蓝/绿/红，标签 1/2/3）**：`scripts/relabel_masks.py`
   （旧版）与 `dataset/toxpath_dataset.py`（旧版）—— 用连通域规则给 patch 分
   1/2/3 并涂色，训练时按 4 类/3 类像素分割。该方案的类别依附于
   `train_masks_class/`，目录已被删除。
3. **灰度三分类（0/128/255）+ 5 折训练**：`train.py`（旧版）与
   `docs/标签重定义方案.md`、`docs/运行时三分类映射方案.md` ——
   把三分类编码成黑/中灰/白，训练脚本按 224 输入训练。

当前方案与它们的区别：类别阈值改为 **前景占比 0.01% ~ 5% → 类 1，≥5% → 类 2**，
颜色为 **黑 / 绿 / 红**，目录为 `train_masks_classes/`（注意是复数），
训练输入为**整张 1024×1024**。

## 如何彻底删除

确认不再需要后，直接删除整个目录即可：

```powershell
Remove-Item -LiteralPath E:\project\toxpath\_deprecated -Recurse -Force
```

项目没有 git，删除后不可恢复，所以这里先做「移出活动目录」而不是直接删。
