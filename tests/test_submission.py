"""提交链路测试：官方指标、连通域过滤、掩膜格式、zip 结构、自检脚本。

跑法
----
.. code-block:: bash

    python tests/test_submission.py

只依赖 numpy / Pillow，**不需要 torch 与 timm**，在没装 GPU 环境的机器上
也能跑（提交格式问题不该等到实验室才发现）。

覆盖
----
1. ``dice_iou``：双空记 1.0、漏检 0、完全吻合 1.0、半重叠手算值；
2. ``remove_small_components``：小块删除、大块保留、4 邻接语义；
3. ``save_mask``：单通道 L、8-bit、像素值只有 0/255、尺寸不变；
4. ``zip_submission``：条目为 ``{slide}/{name}.png``，无外层目录；
5. ``check_submission``：正常包退出码 0，五类致命错误全部被拦下（退出码 1）；
6. ``discover_test_patches``：有/无层级目录都能扫；同名 stem 冲突要报错。
"""

from __future__ import annotations

import shutil
import subprocess
import sys
import tempfile
import unittest
import zipfile
from pathlib import Path

import numpy as np
from PIL import Image

PROJECT_ROOT: Path = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from eval_binary import dice_iou  # noqa: E402
from predict_test import (  # noqa: E402
    discover_test_patches,
    remove_small_components,
    save_mask,
    zip_submission,
)


class TestOfficialMetric(unittest.TestCase):
    """官方指标口径：逐 patch Dice/IoU，双空记 1.0。"""

    def test_both_empty_is_one(self) -> None:
        z: np.ndarray = np.zeros((8, 8), dtype=bool)
        self.assertEqual(dice_iou(z, z), (1.0, 1.0))

    def test_missed_tumor_is_zero(self) -> None:
        z: np.ndarray = np.zeros((8, 8), dtype=bool)
        half: np.ndarray = np.zeros((8, 8), dtype=bool)
        half[:4] = True
        self.assertEqual(dice_iou(z, half), (0.0, 0.0))

    def test_perfect_match_is_one(self) -> None:
        f: np.ndarray = np.ones((8, 8), dtype=bool)
        self.assertEqual(dice_iou(f, f), (1.0, 1.0))

    def test_half_overlap(self) -> None:
        pred: np.ndarray = np.zeros((8, 8), dtype=bool)
        pred[:4] = True                     # 32 px
        gt: np.ndarray = np.ones((8, 8), dtype=bool)   # 64 px
        d, i = dice_iou(pred, gt)
        self.assertAlmostEqual(d, 2 * 32 / (32 + 64), places=9)
        self.assertAlmostEqual(i, 32 / 64, places=9)

    def test_false_positive_on_empty_gt_costs_everything(self) -> None:
        """真值全空 + 只多 1 个像素 → 该 patch 直接 0 分。"""
        gt: np.ndarray = np.zeros((8, 8), dtype=bool)
        pred: np.ndarray = np.zeros((8, 8), dtype=bool)
        pred[3, 3] = True
        self.assertEqual(dice_iou(pred, gt), (0.0, 0.0))


class TestMinArea(unittest.TestCase):
    """最小连通域过滤（4 邻接）。"""

    def test_small_blob_removed_big_kept(self) -> None:
        m: np.ndarray = np.zeros((20, 20), dtype=bool)
        m[2:4, 2:4] = True          # 4 px
        m[10:16, 10:16] = True      # 36 px
        out: np.ndarray = remove_small_components(m, 10)
        self.assertEqual(int(out[2:4, 2:4].sum()), 0)
        self.assertEqual(int(out[10:16, 10:16].sum()), 36)

    def test_threshold_one_is_noop(self) -> None:
        m: np.ndarray = np.zeros((20, 20), dtype=bool)
        m[2:4, 2:4] = True
        m[10:16, 10:16] = True
        self.assertEqual(int(remove_small_components(m, 1).sum()), 40)

    def test_diagonal_is_not_connected(self) -> None:
        """4 邻接：对角线两像素各自独立，都会被当成 1 px 小块删掉。"""
        m: np.ndarray = np.zeros((5, 5), dtype=bool)
        m[1, 1] = True
        m[2, 2] = True
        self.assertEqual(int(remove_small_components(m, 3).sum()), 0)

    def test_vertical_and_horizontal_merge(self) -> None:
        v: np.ndarray = np.zeros((6, 6), dtype=bool)
        v[2:5, 2] = True
        self.assertEqual(int(remove_small_components(v, 3).sum()), 3)
        h: np.ndarray = np.zeros((6, 6), dtype=bool)
        h[2, 2:5] = True
        self.assertEqual(int(remove_small_components(h, 3).sum()), 3)

    def test_empty_mask(self) -> None:
        z: np.ndarray = np.zeros((10, 10), dtype=bool)
        self.assertEqual(int(remove_small_components(z, 100).sum()), 0)


class TestSubmissionIO(unittest.TestCase):
    """落盘格式与 zip 结构。"""

    def setUp(self) -> None:
        self.tmp: Path = Path(tempfile.mkdtemp(prefix="tox_sub_"))

    def tearDown(self) -> None:
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_save_mask_format(self) -> None:
        mask: np.ndarray = np.zeros((64, 48), dtype=bool)
        mask[:10, :10] = True
        p: Path = self.tmp / "s" / "x.png"
        save_mask(mask, p)
        with Image.open(p) as im:
            self.assertEqual(im.mode, "L")            # 单通道
            self.assertEqual(im.size, (48, 64))       # 尺寸不变
            arr: np.ndarray = np.asarray(im)
        self.assertEqual(arr.dtype, np.uint8)
        self.assertEqual(set(np.unique(arr).tolist()), {0, 255})
        self.assertEqual(int((arr == 255).sum()), 100)

    def test_zip_has_no_outer_folder(self) -> None:
        sub: Path = self.tmp / "submission"
        save_mask(np.zeros((8, 8), bool), sub / "test_001" / "0_0.png")
        save_mask(np.zeros((8, 8), bool), sub / "test_001" / "1024_0.png")
        save_mask(np.zeros((8, 8), bool), sub / "test_002" / "0_0.png")
        z: Path = self.tmp / "out.zip"
        n, nbytes = zip_submission(sub, z)
        self.assertEqual(n, 3)
        self.assertGreater(nbytes, 0)
        with zipfile.ZipFile(z) as zf:
            names: list[str] = sorted(zf.namelist())
        self.assertEqual(names, ["test_001/0_0.png", "test_001/1024_0.png",
                                 "test_002/0_0.png"])
        for name in names:
            self.assertEqual(name.count("/"), 1, "zip 里不能再套外层目录")


class TestDiscover(unittest.TestCase):
    """官方 patch 目录扫描。"""

    def setUp(self) -> None:
        self.tmp: Path = Path(tempfile.mkdtemp(prefix="tox_disc_"))

    def tearDown(self) -> None:
        shutil.rmtree(self.tmp, ignore_errors=True)

    def _touch(self, rel: str) -> Path:
        p: Path = self.tmp / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        Image.fromarray(np.zeros((8, 8, 3), np.uint8)).save(p)
        return p

    def test_with_level_dir(self) -> None:
        self._touch("test_patches/test_001/1.0/0_0.jpeg")
        self._touch("test_patches/test_001/1.0/1024_0.jpeg")
        self._touch("test_patches/test_002/1.0/0_0.jpeg")
        slides = discover_test_patches(self.tmp)
        self.assertEqual(sorted(slides), ["test_001", "test_002"])
        self.assertEqual(len(slides["test_001"]), 2)

    def test_without_level_dir(self) -> None:
        """官方若直接平铺 patch（无 1.0/）也要能扫。"""
        self._touch("test_patches/test_001/0_0.jpeg")
        slides = discover_test_patches(self.tmp)
        self.assertEqual(len(slides["test_001"]), 1)

    def test_duplicate_stem_raises(self) -> None:
        """两个层级目录出现同名 stem，提交会互相覆盖 → 必须报错。"""
        self._touch("test_patches/test_001/1.0/0_0.jpeg")
        self._touch("test_patches/test_001/2.0/0_0.jpeg")
        with self.assertRaises(ValueError):
            discover_test_patches(self.tmp)


class TestChecker(unittest.TestCase):
    """自检脚本：正常包放行，致命错误必须退出码 1。"""

    def setUp(self) -> None:
        self.tmp: Path = Path(tempfile.mkdtemp(prefix="tox_chk_"))
        self.test_root: Path = self.tmp / "test_root"
        self.sub: Path = self.tmp / "submission"
        # 官方 patch：两张，均 32×32
        for slide, stem in (("test_001", "0_0"), ("test_001", "1024_0"),
                            ("test_002", "0_0")):
            p: Path = self.test_root / "test_patches" / slide / "1.0" / f"{stem}.jpeg"
            p.parent.mkdir(parents=True, exist_ok=True)
            Image.fromarray(np.zeros((32, 32, 3), np.uint8)).save(p)
        # 合法提交
        for slide, stem in (("test_001", "0_0"), ("test_001", "1024_0"),
                            ("test_002", "0_0")):
            save_mask(np.zeros((32, 32), bool), self.sub / slide / f"{stem}.png")

    def tearDown(self) -> None:
        shutil.rmtree(self.tmp, ignore_errors=True)

    def _run(self) -> int:
        script: Path = PROJECT_ROOT / "scripts" / "check_submission.py"
        proc = subprocess.run(
            [sys.executable, str(script),
             "--test-root", str(self.test_root),
             "--submission", str(self.sub)],
            capture_output=True, text=True, encoding="utf-8", errors="replace",
        )
        return proc.returncode

    def test_valid_passes(self) -> None:
        self.assertEqual(self._run(), 0)

    def test_missing_mask_fails(self) -> None:
        (self.sub / "test_001" / "0_0.png").unlink()
        self.assertEqual(self._run(), 1)

    def test_extra_mask_fails(self) -> None:
        save_mask(np.zeros((32, 32), bool), self.sub / "test_001" / "9_9.png")
        self.assertEqual(self._run(), 1)

    def test_wrong_dimension_fails(self) -> None:
        save_mask(np.zeros((16, 16), bool), self.sub / "test_001" / "0_0.png")
        self.assertEqual(self._run(), 1)

    def test_non_binary_values_fail(self) -> None:
        arr: np.ndarray = np.full((32, 32), 128, np.uint8)
        Image.fromarray(arr, mode="L").save(self.sub / "test_001" / "0_0.png")
        self.assertEqual(self._run(), 1)

    def test_missing_slide_fails(self) -> None:
        shutil.rmtree(self.sub / "test_002")
        self.assertEqual(self._run(), 1)

    def test_stray_file_fails(self) -> None:
        (self.sub / "readme.txt").write_text("x", encoding="utf-8")
        self.assertEqual(self._run(), 1)


if __name__ == "__main__":
    unittest.main(verbosity=2)
