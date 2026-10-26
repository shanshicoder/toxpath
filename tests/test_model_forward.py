"""ToxPath 模型端到端冒烟测试。

目的
----
在**不训练、不读数据**的前提下，验证整条计算链路是否通：

1. 配置校验能否拦住非法参数（尺寸不整除 patch、类别数 < 2 ……）；
2. H0-mini 编码器是否输出 4 层 ``[B, 768, H/14, W/14]`` 特征；
3. Neck 四路输出是否符合 ``64/128/256/512 @ H, H/2, H/4, H/8`` 的账本；
4. 解码器 + 整体前向是否输出 ``[B, 2, H, W]``，且**空间尺寸与输入完全一致**；
5. 换一个非默认边长（350）能否照样跑通（验证动态尺寸 + 尺寸对齐分支）；
6. 损失能否反传、梯度是否真的流到了编码器和解码头；
7. 冻结编码器是否生效（参数量、``requires_grad``、``train()`` 不会被翻回来）；
8. ``state_dict`` 存取是否严格一致（strict=True 不报错）。

运行方式
--------
在项目根目录 ``E:/project/toxpath`` 下::

    conda activate toxpath
    python tests/test_model_forward.py

也可以交给 pytest::

    pytest tests/test_model_forward.py -v

说明
----
本测试**故意**用 ``pretrained=False``，因此完全离线：
timm 只会按结构随机初始化，不会去 HuggingFace 下载 H0-mini 权重。
想验证"官方权重能否加载"，请另外跑 ``load_histoplus_backbone`` 的用例
（需要 ``histoplus_cellvit_segmentor_20x.pt``，见 README）。
"""

from __future__ import annotations

import sys
import traceback
from pathlib import Path
from typing import Callable

import torch
from torch import Tensor

# 允许直接 `python tests/test_model_forward.py`（把项目根目录加进 sys.path）
PROJECT_ROOT: Path = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from model import (  # noqa: E402  (必须在 sys.path 调整之后导入)
    DiceCrossEntropyLoss,
    H0MiniEncoder,
    HistoPlusNeck,
    SemanticDecoder,
    ToxPathConfig,
    ToxPathSegmentor,
)

# --------------------------------------------------------------------------- #
# 公共测试配置
# --------------------------------------------------------------------------- #
# 用 224 而不是 784：224/14 = 16，是最小的"干净尺寸"，
# CPU 上单次前向约 1 秒，足够验证形状而不用等太久。
BASE_SIZE: int = 224
BATCH_SIZE: int = 2


def _make_cfg(**overrides: object) -> ToxPathConfig:
    """构造测试用配置（默认离线、小尺寸）。"""
    defaults: dict[str, object] = {
        "image_size": BASE_SIZE,
        "pretrained": False,
        "dropout": 0.0,
    }
    defaults.update(overrides)
    return ToxPathConfig(**defaults)  # type: ignore[arg-type]


def _assert_shape(actual: tuple[int, ...], expected: tuple[int, ...], what: str) -> None:
    """断言形状，失败时打印可读信息。"""
    if tuple(actual) != tuple(expected):
        raise AssertionError(f"{what} 形状不符: 期望 {expected}，实际 {tuple(actual)}")


# --------------------------------------------------------------------------- #
# 1. 配置校验
# --------------------------------------------------------------------------- #
def test_config_validation() -> None:
    """非法配置必须在构造时就被拦住，而不是训练到一半才崩。"""
    cfg = ToxPathConfig()
    assert cfg.num_classes == 2
    assert cfg.patch_size == 14
    assert len(cfg.out_indices) == 4

    invalid_cases: list[dict[str, object]] = [
        {"image_size": 7},            # 比 patch_size 还小
        {"num_classes": 1},           # 二分类至少 2 类
        {"in_channels": 1},           # H0-mini 的 patch_embed 是 3 通道
        {"out_indices": (3, 5, 7)},   # Neck 必须 4 路
        {"out_indices": (11, 7, 5, 3)},  # 必须升序
    ]
    for bad in invalid_cases:
        try:
            ToxPathConfig(**bad)  # type: ignore[arg-type]
        except ValueError:
            continue
        raise AssertionError(f"非法配置 {bad} 没有被拒绝")

    print("    [1/8] 配置校验：5 种非法配置全部被拦截")


# --------------------------------------------------------------------------- #
# 2. 编码器输出形状
# --------------------------------------------------------------------------- #
def test_encoder_shapes() -> None:
    """H0-mini 应输出 4 层、同分辨率、768 维的特征图。"""
    cfg = _make_cfg()
    encoder = H0MiniEncoder(cfg).eval()
    x = torch.randn(1, 3, BASE_SIZE, BASE_SIZE)

    with torch.no_grad():
        feats = encoder(x)

    assert isinstance(feats, list), f"编码器应返回 list，实际 {type(feats)}"
    assert len(feats) == 4, f"应输出 4 层特征，实际 {len(feats)}"
    side = BASE_SIZE // cfg.patch_size
    for i, feat in enumerate(feats):
        _assert_shape(tuple(feat.shape), (1, cfg.embed_dim, side, side), f"特征 f{i}")

    # 尺寸非法时必须给出明确报错，而不是在 patch_embed 里抛看不懂的 reshape 错误
    try:
        encoder(torch.randn(1, 3, 100, 100))
    except ValueError:
        pass
    else:
        raise AssertionError("非法输入尺寸没有被拒绝")

    del encoder, x, feats
    print(f"    [2/8] 编码器：4 × [1, {cfg.embed_dim}, {side}, {side}]")


# --------------------------------------------------------------------------- #
# 3. Neck 输出形状
# --------------------------------------------------------------------------- #
def test_neck_shapes() -> None:
    """Neck 四路：64@H、128@2h、256@4h、512@8h。"""
    cfg = _make_cfg()
    encoder = H0MiniEncoder(cfg).eval()
    neck = HistoPlusNeck(cfg).eval()
    x = torch.randn(1, 3, BASE_SIZE, BASE_SIZE)
    h = BASE_SIZE // cfg.patch_size

    with torch.no_grad():
        feats = encoder(x)
        necks = neck(x, feats)

    c0, c1, c2, c3 = cfg.neck_channels
    # n1/n2/n3 的上采样次数分别是 3/2/1，也就是 8h/4h/2h（注意顺序是反的）
    expected: list[tuple[int, int, int, int]] = [
        (1, c0, BASE_SIZE, BASE_SIZE),   # n0：原图分辨率
        (1, c1, 8 * h, 8 * h),           # n1：×8
        (1, c2, 4 * h, 4 * h),           # n2：×4
        (1, c3, 2 * h, 2 * h),           # n3：×2
    ]
    for i, (feat, exp) in enumerate(zip(necks, expected, strict=True)):
        _assert_shape(tuple(feat.shape), exp, f"neck 输出 n{i}")

    del encoder, neck, x, feats, necks
    shapes = " / ".join(f"n{i}={exp[1]}x{exp[2]}x{exp[3]}" for i, exp in enumerate(expected))
    print(f"    [3/8] Neck 四路：{shapes}")


# --------------------------------------------------------------------------- #
# 4. 整体前向
# --------------------------------------------------------------------------- #
def test_full_forward() -> None:
    """端到端：``[B, 3, H, W]`` → ``[B, 2, H, W]``，尺寸必须严格一致。"""
    cfg = _make_cfg()
    model = ToxPathSegmentor(cfg).eval()
    x = torch.randn(BATCH_SIZE, 3, BASE_SIZE, BASE_SIZE)

    with torch.no_grad():
        logits = model(x)

    _assert_shape(tuple(logits.shape), (BATCH_SIZE, cfg.num_classes, BASE_SIZE, BASE_SIZE), "logits")
    assert torch.isfinite(logits).all(), "输出里出现了 NaN / Inf"

    # 概率应在 [0, 1] 且和为 1
    proba = model.predict_proba(x)
    _assert_shape(tuple(proba.shape), (BATCH_SIZE, 2, BASE_SIZE, BASE_SIZE), "proba")
    assert torch.allclose(proba.sum(dim=1), torch.ones(BATCH_SIZE, BASE_SIZE, BASE_SIZE), atol=1e-5)

    # 二值掩膜
    mask = model.predict_mask(x, threshold=0.5)
    _assert_shape(tuple(mask.shape), (BATCH_SIZE, BASE_SIZE, BASE_SIZE), "mask")
    assert mask.dtype == torch.uint8
    assert set(mask.unique().tolist()) <= {0, 1}

    print(f"    [4/8] 前向：{tuple(x.shape)} -> {tuple(logits.shape)}，proba/mask 均正常")
    del model, x, logits, proba, mask


# --------------------------------------------------------------------------- #
# 5. 动态尺寸
# --------------------------------------------------------------------------- #
def test_dynamic_size() -> None:
    """非 224 的边长也要能跑（验证 16h ≠ H 时的插值对齐分支）。"""
    cfg = _make_cfg()
    model = ToxPathSegmentor(cfg).eval()

    # 350 = 14 × 25：h=25，解码器末端 16h = 400 ≠ 350，必然走到 interpolate_to
    side = 350
    x = torch.randn(1, 3, side, side)
    with torch.no_grad():
        logits = model(x)
    _assert_shape(tuple(logits.shape), (1, 2, side, side), f"{side}×{side} 的 logits")

    # 448 = 14 × 32：HistoPLUS 训练时用的 40x tile 尺寸
    side = 448
    x = torch.randn(1, 3, side, side)
    with torch.no_grad():
        logits = model(x)
    _assert_shape(tuple(logits.shape), (1, 2, side, side), f"{side}×{side} 的 logits")

    print("    [5/8] 动态尺寸：350×350 与 448×448 均输出同尺寸 logits")
    del model, x, logits


# --------------------------------------------------------------------------- #
# 6. 损失与反向传播
# --------------------------------------------------------------------------- #
def test_backward() -> None:
    """损失能算、梯度能回传，且编码器与解码头都拿到了非零梯度。"""
    cfg = _make_cfg()
    model = ToxPathSegmentor(cfg).train()
    criterion = DiceCrossEntropyLoss(num_classes=2, class_weights=(1.0, 2.0))

    # 用 112 而不是 224：112 = 14 × 8，CPU 上反传更快，形状逻辑完全一样
    side = 112
    x = torch.randn(BATCH_SIZE, 3, side, side)
    target = torch.randint(0, 2, (BATCH_SIZE, side, side), dtype=torch.int64)

    logits = model(x)
    losses = criterion(logits, target)

    for key in ("loss", "ce", "dice"):
        assert key in losses, f"损失字典缺少 {key}"
        assert torch.isfinite(losses[key]), f"{key} 不是有限值"

    losses["loss"].backward()

    def _grad_ok(module: torch.nn.Module, name: str) -> None:
        grads = [p.grad for p in module.parameters() if p.requires_grad]
        assert grads, f"{name} 没有可训练参数"
        assert any(g is not None and torch.isfinite(g).all() for g in grads), f"{name} 的梯度全为空或含 NaN"

    _grad_ok(model.encoder, "encoder")
    _grad_ok(model.neck, "neck")
    _grad_ok(model.decoder, "decoder")

    print(
        f"    [6/8] 反传：loss={losses['loss'].item():.4f} "
        f"(ce={losses['ce'].item():.4f}, dice={losses['dice'].item():.4f})，三部分梯度正常"
    )
    del model, x, target, logits, losses


# --------------------------------------------------------------------------- #
# 7. 冻结编码器
# --------------------------------------------------------------------------- #
def test_freeze_encoder() -> None:
    """冻结后：参数量下降、梯度为 None、``train()`` 不会把 backbone 翻回训练态。"""
    cfg = _make_cfg(freeze_encoder=True)
    model = ToxPathSegmentor(cfg)
    assert model.encoder.is_frozen, "构造后应处于冻结状态"
    assert all(not p.requires_grad for p in model.encoder.parameters()), "编码器仍可训练"
    assert model.num_parameters > model.num_trainable_parameters, "冻结后可训练参数应更少"

    model.train(True)
    assert not model.encoder.backbone.training, "调用 train() 后 backbone 被错误地切回训练态"

    x = torch.randn(1, 3, 112, 112)
    target = torch.randint(0, 2, (1, 112, 112), dtype=torch.int64)
    criterion = DiceCrossEntropyLoss(num_classes=2)
    losses = criterion(model(x), target)
    losses["loss"].backward()

    assert all(p.grad is None for p in model.encoder.parameters()), "冻结参数不应有梯度"
    assert any(
        p.grad is not None for p in model.decoder.parameters()
    ), "解码头应仍有梯度"

    print(
        f"    [7/8] 冻结：total={model.num_parameters:,}，"
        f"trainable={model.num_trainable_parameters:,}"
    )
    del model, x, target, losses


# --------------------------------------------------------------------------- #
# 8. state_dict 往返
# --------------------------------------------------------------------------- #
def test_state_dict_roundtrip() -> None:
    """保存再加载必须严格一致（strict=True 不报错且权重逐位相等）。"""
    cfg = _make_cfg()
    model_a = ToxPathSegmentor(cfg)
    state = model_a.state_dict()

    model_b = ToxPathSegmentor(cfg)
    missing, unexpected = model_b.load_state_dict(state, strict=True)
    assert not missing and not unexpected, "strict 加载不应有缺失/多余 key"

    key = next(iter(state))
    assert torch.equal(state[key], model_b.state_dict()[key]), "加载后权重不一致"

    # 参数组：编码器 lr 应为头部的 encoder_lr_mult 倍
    groups = model_a.param_groups(lr=1e-3, encoder_lr_mult=0.1)
    lrs = sorted({float(g["lr"]) for g in groups})
    assert any(abs(lr - 1e-4) < 1e-12 for lr in lrs), f"未找到较小的编码器学习率，实际 {lrs}"
    assert any(abs(lr - 1e-3) < 1e-12 for lr in lrs), f"未找到头部学习率，实际 {lrs}"

    print(f"    [8/8] 权重往返：{len(state)} 个张量严格一致；参数组 lr={lrs}")
    del model_a, model_b, state


# --------------------------------------------------------------------------- #
# runner
# --------------------------------------------------------------------------- #
def main() -> int:
    """顺序执行全部测试，返回进程退出码（0 = 全通过）。"""
    torch.manual_seed(0)
    # 限制线程数，避免小尺寸测试在小机器上反而被线程调度拖慢
    torch.set_num_threads(min(4, torch.get_num_threads()))

    tests: list[tuple[str, Callable[[], None]]] = [
        ("配置校验", test_config_validation),
        ("编码器形状", test_encoder_shapes),
        ("Neck 形状", test_neck_shapes),
        ("整体前向", test_full_forward),
        ("动态尺寸", test_dynamic_size),
        ("损失反传", test_backward),
        ("冻结编码器", test_freeze_encoder),
        ("权重往返", test_state_dict_roundtrip),
    ]

    print("=" * 68)
    print("ToxPath 模型前向 / 反向冒烟测试")
    print(f"  torch  : {torch.__version__}")
    print(f"  device : cpu（本测试不要求 GPU）")
    print("=" * 68)

    failures: list[tuple[str, str]] = []
    for name, fn in tests:
        try:
            fn()
        except Exception:  # noqa: BLE001 - 测试 runner 需要捕获一切
            failures.append((name, traceback.format_exc()))
            print(f"    [FAIL] {name}")
            print(traceback.format_exc())

    print("-" * 68)
    if failures:
        print(f"结果：{len(failures)}/{len(tests)} 个测试失败 -> {[n for n, _ in failures]}")
    else:
        print(f"结果：全部 {len(tests)} 个测试通过，模型输入输出链路正常")
    print("=" * 68)

    # 打印一次结构摘要，方便肉眼确认通道数没有搭错
    try:
        demo = ToxPathSegmentor(_make_cfg())
        print()
        print(demo.describe())
        del demo
    except Exception:  # noqa: BLE001
        pass

    return 0 if not failures else 1


if __name__ == "__main__":
    raise SystemExit(main())
