"""ToxPath 三分类 Dataset 的测试（配置、颜色映射、样本契约、增强、采样）。

运行
----
.. code-block:: bash

    # 直接跑（推荐，输出更详细）
    python tests/test_dataset.py

    # 或者交给 pytest / unittest
    pytest tests/test_dataset.py -v
    python -m unittest tests.test_dataset -v

说明
----
- **配置类测试**（``TestDataset3Config`` / ``TestColorMapping``）不依赖数据，任何环境都能跑；
- **数据类测试**（其余）需要 ``data/toxpath/`` 与 ``train_masks_classes/``，
  缺数据时会自动 skip，不会报错；
- 为了速度，数据类测试默认用 ``image_size=224``；只有少数用例用默认的 1024
  验证“整张输入”的形状契约。
"""

from __future__ import annotations

import csv
import dataclasses
import random
import sys
import unittest
from pathlib import Path

import numpy as np
from PIL import Image
import torch

PROJECT_ROOT: Path = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from dataset import (  # noqa: E402
    BIOPTIMUS_MEAN,
    BIOPTIMUS_STD,
    CLASS_COLORS,
    Dataset3Config,
    ToxPath3Dataset,
    build_dataset3,
    build_weighted_sampler,
    rgb_to_class,
    seed_worker,
)

DATA_ROOT: Path = PROJECT_ROOT / "data" / "toxpath"
MASK_ROOT: Path = DATA_ROOT / "train_masks_classes"
INDEX_PATH: Path = DATA_ROOT / "_patch_labels_3class.csv"
HAS_DATA: bool = (
    (MASK_ROOT / "train_001" / "1.0").is_dir()
    and (DATA_ROOT / "train_patches" / "train_001" / "1.0").is_dir()
    and INDEX_PATH.is_file()
)

requires_data = unittest.skipUnless(HAS_DATA, "需要 data/toxpath 与 train_masks_classes/")


# --------------------------------------------------------------------------- #
# 工具
# --------------------------------------------------------------------------- #
def _read_index() -> dict[tuple[str, str], tuple[int, float]]:
    """读标签索引：{(slide, patch_stem): (label, fg_ratio)}。"""
    out: dict[tuple[str, str], tuple[int, float]] = {}
    with INDEX_PATH.open("r", encoding="utf-8", newline="") as fh:
        for row in csv.DictReader(fh):
            stem: str = row["name"].rsplit(".", 1)[0]
            out[(row["slide"], stem)] = (int(row["label"]), float(row["fg_ratio"]))
    return out


def _find_patch_by_label(label: int) -> tuple[str, str]:
    """从索引里找第一张指定类别的 patch，返回 (slide, stem)。"""
    for (slide, stem), (lab, _ratio) in _read_index().items():
        if lab == label:
            return slide, stem
    raise unittest.SkipTest(f"索引里没有 label={label} 的 patch")


# --------------------------------------------------------------------------- #
# 1. 配置
# --------------------------------------------------------------------------- #
class TestDataset3Config(unittest.TestCase):
    """``Dataset3Config`` 的默认值、不可变性与自定义值。"""

    def test_defaults(self) -> None:
        cfg = Dataset3Config()
        self.assertEqual(cfg.image_size, 1024)
        self.assertTrue(cfg.augment)
        self.assertAlmostEqual(cfg.hed_alpha, 0.04)
        self.assertAlmostEqual(cfg.hed_beta, 0.04)
        self.assertEqual(cfg.sample_weight_per_class, (1.0, 6.0, 2.5))
        self.assertEqual(cfg.masks_dir, "train_masks_classes")
        self.assertEqual(cfg.index_name, "_patch_labels_3class.csv")

    def test_frozen(self) -> None:
        """配置是 frozen dataclass，训练中不允许被意外修改。"""
        cfg = Dataset3Config()
        with self.assertRaises(dataclasses.FrozenInstanceError):
            cfg.image_size = 512  # type: ignore[misc]

    def test_custom_values(self) -> None:
        cfg = Dataset3Config(
            image_size=224,
            augment=False,
            hed_alpha=0.0,
            hed_beta=0.0,
            sample_weight_per_class=(1.0, 2.0, 3.0),
        )
        self.assertEqual(cfg.image_size, 224)
        self.assertFalse(cfg.augment)
        self.assertEqual(cfg.sample_weight_per_class, (1.0, 2.0, 3.0))

    def test_frozen_hashable(self) -> None:
        """frozen dataclass 应当可哈希（便于缓存/去重）。"""
        self.assertEqual(hash(Dataset3Config()), hash(Dataset3Config()))


# --------------------------------------------------------------------------- #
# 2. 颜色 → 类别
# --------------------------------------------------------------------------- #
class TestColorMapping(unittest.TestCase):
    """``rgb_to_class`` 与 ``CLASS_COLORS`` 的一致性。"""

    def test_known_colors(self) -> None:
        rgb = np.array(
            [
                [[0, 0, 0], [0, 255, 0], [255, 0, 0]],
            ],
            dtype=np.uint8,
        )
        out = rgb_to_class(rgb)
        np.testing.assert_array_equal(out, np.array([[0, 1, 2]], dtype=np.int64))
        self.assertEqual(out.dtype, np.int64)

    def test_unknown_color_falls_back_to_background(self) -> None:
        rgb = np.array([[[1, 2, 3], [255, 255, 255], [0, 0, 254]]], dtype=np.uint8)
        out = rgb_to_class(rgb)
        np.testing.assert_array_equal(out, np.zeros((1, 3), dtype=np.int64))

    def test_class_colors_are_distinct(self) -> None:
        colors = list(CLASS_COLORS.values())
        self.assertEqual(len(colors), len(set(colors)), "三个类别的颜色不能重复")
        self.assertEqual(CLASS_COLORS[0], (0, 0, 0), "背景必须是纯黑")

    def test_large_random_shapes(self) -> None:
        """随机 64×64 输入：形状不变，输出只会落在 {0,1,2}。"""
        rng = np.random.default_rng(0)
        rgb = rng.choice(np.array([0, 255], dtype=np.uint8), size=(64, 64, 3))
        out = rgb_to_class(rgb)
        self.assertEqual(out.shape, (64, 64))
        self.assertTrue(set(np.unique(out).tolist()) <= {0, 1, 2})


# --------------------------------------------------------------------------- #
# 3. Dataset 样本契约（需要数据）
# --------------------------------------------------------------------------- #
@requires_data
class TestToxPath3Dataset(unittest.TestCase):
    """数据集长度、返回字段、归一化、mask 与索引的一致性。"""

    @classmethod
    def setUpClass(cls) -> None:
        cls.cfg = Dataset3Config(image_size=224, augment=False)
        cls.ds = ToxPath3Dataset(
            DATA_ROOT, ["train_001"], cfg=cls.cfg, mode="train", max_samples=8
        )
        cls.index = _read_index()

    def test_len(self) -> None:
        self.assertEqual(len(self.ds), 8)

    def test_sample_keys(self) -> None:
        s = self.ds[0]
        expected = {
            "image", "mask", "patch_label", "fg_ratio",
            "sample_weight", "slide", "name",
        }
        self.assertEqual(set(s.keys()), expected)

    def test_sample_shapes_and_dtypes(self) -> None:
        s = self.ds[0]
        self.assertEqual(tuple(s["image"].shape), (3, 224, 224))
        self.assertEqual(s["image"].dtype, torch.float32)
        self.assertTrue(torch.isfinite(s["image"]).all())

        self.assertEqual(tuple(s["mask"].shape), (224, 224))
        self.assertEqual(s["mask"].dtype, torch.int64)

        self.assertIsInstance(s["patch_label"], int)
        self.assertIsInstance(s["fg_ratio"], float)

    def test_mask_values_in_range(self) -> None:
        for i in range(len(self.ds)):
            uniq = set(self.ds[i]["mask"].unique().tolist())
            self.assertTrue(
                uniq <= {0, 1, 2}, f"第 {i} 个样本 mask 出现非法类别: {uniq}"
            )

    def test_sample_weight_matches_config(self) -> None:
        table = self.cfg.sample_weight_per_class
        for i in range(len(self.ds)):
            s = self.ds[i]
            self.assertAlmostEqual(s["sample_weight"], table[s["patch_label"]])

    def test_label_matches_index(self) -> None:
        for i in range(len(self.ds)):
            s = self.ds[i]
            label, ratio = self.index[(s["slide"], s["name"])]
            self.assertEqual(s["patch_label"], label)
            self.assertAlmostEqual(s["fg_ratio"], ratio, places=8)

    def test_mask_matches_stored_png(self) -> None:
        """Dataset 输出的 mask 应与彩色 PNG 用 NEAREST 缩放后完全一致。"""
        s = self.ds[0]
        mask_path = DATA_ROOT / "train_masks_classes" / s["slide"] / "1.0" / f"{s['name']}.png"
        raw = Image.open(mask_path).convert("RGB").resize((224, 224), Image.NEAREST)
        expected = rgb_to_class(np.asarray(raw, dtype=np.uint8))
        np.testing.assert_array_equal(s["mask"].numpy(), expected)

    def test_normalization_matches_manual(self) -> None:
        """image 必须等于 (x/255 - mean)/std，且顺序是 CHW。"""
        s = self.ds[0]
        img_path = DATA_ROOT / "train_patches" / s["slide"] / "1.0" / f"{s['name']}.jpeg"
        raw = Image.open(img_path).convert("RGB").resize((224, 224), Image.BILINEAR)
        arr = np.asarray(raw, dtype=np.float32) / 255.0
        expected = (arr - np.array(BIOPTIMUS_MEAN, np.float32)) / np.array(BIOPTIMUS_STD, np.float32)
        expected = expected.transpose(2, 0, 1)
        np.testing.assert_allclose(s["image"].numpy(), expected, rtol=1e-5, atol=1e-5)

    def test_patch_label_consistent_with_mask(self) -> None:
        """有前景的 patch：mask 里非零类别必须等于 patch_label；背景则全 0。

        用 1024 原始尺寸，避免极小的前景（类 1 可能只有 100 多像素）
        在缩小到 224 时被完全丢掉而误判。
        """
        ds = ToxPath3Dataset(
            DATA_ROOT, ["train_001"],
            cfg=Dataset3Config(image_size=1024, augment=False),
            max_samples=8,
        )
        for i in range(len(ds)):
            s = ds[i]
            uniq = set(s["mask"].unique().tolist())
            if s["patch_label"] == 0:
                self.assertEqual(uniq, {0})
            else:
                self.assertEqual(uniq, {0, s["patch_label"]})

    def test_default_image_size_is_1024(self) -> None:
        ds = ToxPath3Dataset(
            DATA_ROOT, ["train_001"],
            cfg=Dataset3Config(augment=False),  # 默认 image_size=1024
            mode="train", max_samples=1,
        )
        s = ds[0]
        self.assertEqual(tuple(s["image"].shape), (3, 1024, 1024))
        self.assertEqual(tuple(s["mask"].shape), (1024, 1024))

    def test_missing_masks_dir_raises(self) -> None:
        with self.assertRaises(FileNotFoundError) as ctx:
            ToxPath3Dataset(
                DATA_ROOT, ["train_001"],
                cfg=Dataset3Config(masks_dir="no_such_masks_dir", image_size=224),
                max_samples=1,
            )
        self.assertIn("make_3class_masks", str(ctx.exception))

    def test_missing_index_raises(self) -> None:
        with self.assertRaises(FileNotFoundError) as ctx:
            ToxPath3Dataset(
                DATA_ROOT, ["train_001"],
                cfg=Dataset3Config(index_name="no_such_index.csv", image_size=224),
                max_samples=1,
            )
        self.assertIn("make_3class_masks", str(ctx.exception))

    def test_nonexistent_patch_dir_raises(self) -> None:
        with self.assertRaises(FileNotFoundError):
            ToxPath3Dataset(
                DATA_ROOT, ["train_999"], cfg=self.cfg, max_samples=1
            )

    def test_val_mode_disables_augment(self) -> None:
        train_cfg = Dataset3Config(image_size=64, augment=True)
        ds = ToxPath3Dataset(
            DATA_ROOT, ["train_001"], cfg=train_cfg, mode="val", max_samples=1
        )
        self.assertFalse(ds.cfg.augment, "val 模式必须强制关闭增强")
        self.assertEqual(ds.mode, "val")

    def test_label_index_covers_every_sample(self) -> None:
        """整张切片的每个 patch 都能在索引里查到（防止索引与数据错位）。"""
        ds = ToxPath3Dataset(
            DATA_ROOT, ["train_001"], cfg=Dataset3Config(image_size=64, augment=False)
        )
        missing = [
            (s["slide"], s["name"]) for s in ds.samples
            if (s["slide"], s["name"]) not in self.index
        ]
        self.assertEqual(missing, [], f"索引缺失 {len(missing)} 个 patch")


# --------------------------------------------------------------------------- #
# 4. 三类 mask 的实际颜色
# --------------------------------------------------------------------------- #
@requires_data
class TestClassSpecificMasks(unittest.TestCase):
    """每个类别各抽一张，确认颜色与类别严格对应。"""

    def _load_one(self, label: int) -> np.ndarray:
        slide, stem = _find_patch_by_label(label)
        ds = ToxPath3Dataset(
            DATA_ROOT, [slide],
            cfg=Dataset3Config(image_size=1024, augment=False),
            mode="val",
        )
        idx = next(i for i, s in enumerate(ds.samples) if s["name"] == stem)
        return ds[idx]["mask"].numpy()

    def test_background_mask_is_all_zero(self) -> None:
        mask = self._load_one(0)
        self.assertEqual(set(np.unique(mask).tolist()), {0})

    def test_few_tumor_mask(self) -> None:
        mask = self._load_one(1)
        self.assertEqual(set(np.unique(mask).tolist()), {0, 1})

    def test_many_tumor_mask(self) -> None:
        mask = self._load_one(2)
        self.assertEqual(set(np.unique(mask).tolist()), {0, 2})


# --------------------------------------------------------------------------- #
# 5. 增强
# --------------------------------------------------------------------------- #
@requires_data
class TestAugmentation(unittest.TestCase):
    """几何增强必须对 image 与 mask 完全同步。"""

    @classmethod
    def setUpClass(cls) -> None:
        cls.ds = ToxPath3Dataset(
            DATA_ROOT, ["train_001"],
            cfg=Dataset3Config(image_size=64, augment=True, hed_alpha=0.0, hed_beta=0.0),
            max_samples=1,
        )

    def test_geometry_is_synchronized(self) -> None:
        """用合成图验证：mask 前景位置与 image 标记位置始终一致。"""
        img = np.zeros((64, 64, 3), dtype=np.uint8)
        msk = np.zeros((64, 64, 3), dtype=np.uint8)
        img[10:30, 20:40] = [255, 0, 0]
        msk[10:30, 20:40] = [255, 0, 0]

        random.seed(123)
        np.random.seed(123)
        img2, msk2 = self.ds._augment(img, msk)

        self.assertEqual(img2.shape, img.shape)
        self.assertEqual(msk2.shape, msk.shape)
        img_fg = (img2[..., 0] == 255) & (img2[..., 1] == 0) & (img2[..., 2] == 0)
        msk_fg = (msk2[..., 0] == 255) & (msk2[..., 1] == 0) & (msk2[..., 2] == 0)
        np.testing.assert_array_equal(img_fg, msk_fg)
        self.assertGreater(int(msk_fg.sum()), 0)

    def test_augment_keeps_mask_on_class_colors(self) -> None:
        s = self.ds[0]
        self.assertTrue(set(s["mask"].unique().tolist()) <= {0, 1, 2})

    def test_hed_jitter_dtype_and_shape(self) -> None:
        rgb = np.random.default_rng(0).integers(1, 256, size=(32, 48, 3), dtype=np.uint8)
        out = self.ds._hed_jitter(rgb)
        self.assertEqual(out.shape, rgb.shape)
        self.assertEqual(out.dtype, np.uint8)

    def test_same_seed_same_augmentation(self) -> None:
        img = np.random.default_rng(1).integers(0, 256, size=(64, 64, 3), dtype=np.uint8)
        msk = np.zeros((64, 64, 3), dtype=np.uint8)
        msk[20:40, 20:40] = [0, 255, 0]

        random.seed(7)
        np.random.seed(7)
        a1, b1 = self.ds._augment(img, msk)
        random.seed(7)
        np.random.seed(7)
        a2, b2 = self.ds._augment(img, msk)

        np.testing.assert_array_equal(a1, a2)
        np.testing.assert_array_equal(b1, b2)


# --------------------------------------------------------------------------- #
# 6. 采样器与 build_dataset3
# --------------------------------------------------------------------------- #
@requires_data
class TestSamplerAndBuilder(unittest.TestCase):
    """采样权重、``WeightedRandomSampler`` 与 train/val 构造。"""

    @classmethod
    def setUpClass(cls) -> None:
        cls.ds = ToxPath3Dataset(
            DATA_ROOT, ["train_001"],
            cfg=Dataset3Config(image_size=224, augment=False), max_samples=8,
        )

    def test_sampler_weights(self) -> None:
        sampler = build_weighted_sampler(self.ds, num_samples=16, seed=0)
        self.assertEqual(sampler.num_samples, 16)
        weights = sampler.weights.numpy()
        np.testing.assert_allclose(weights, self.ds.sample_weights)

    def test_sampler_without_num_samples_uses_dataset_size(self) -> None:
        sampler = build_weighted_sampler(self.ds, seed=0)
        self.assertEqual(sampler.num_samples, len(self.ds))

    def test_build_dataset3(self) -> None:
        train_ds, val_ds = build_dataset3(
            DATA_ROOT, ["train_002"], ["train_001"],
            cfg=Dataset3Config(image_size=224),
            max_train_samples=2, max_val_samples=1,
        )
        self.assertIsNotNone(val_ds)
        self.assertTrue(train_ds.cfg.augment)
        self.assertFalse(val_ds.cfg.augment)
        self.assertEqual(len(train_ds), 2)
        self.assertEqual(len(val_ds), 1)
        self.assertEqual(train_ds.slides, ["train_002"])
        self.assertEqual(val_ds.slides, ["train_001"])

    def test_build_dataset3_without_val(self) -> None:
        train_ds, val_ds = build_dataset3(
            DATA_ROOT, ["train_002"], None,
            cfg=Dataset3Config(image_size=64), max_train_samples=1,
        )
        self.assertIsNone(val_ds)
        self.assertEqual(len(train_ds), 1)


# --------------------------------------------------------------------------- #
# 7. worker 种子
# --------------------------------------------------------------------------- #
class TestSeedWorker(unittest.TestCase):
    """``seed_worker`` 应把 numpy / random 种子设置成 worker 相关值。"""

    def test_seed_worker_changes_state(self) -> None:
        random.seed(0)
        np.random.seed(0)
        before = (random.random(), float(np.random.rand()))

        random.seed(999)
        np.random.seed(999)
        after = (random.random(), float(np.random.rand()))
        self.assertNotEqual(before, after)

        # seed_worker 只依赖 torch.initial_seed()，这里验证它不会抛异常且可重复
        seed_worker(0)
        a = (random.random(), float(np.random.rand()))
        seed_worker(0)
        b = (random.random(), float(np.random.rand()))
        self.assertEqual(a, b)


if __name__ == "__main__":
    unittest.main(verbosity=2)
