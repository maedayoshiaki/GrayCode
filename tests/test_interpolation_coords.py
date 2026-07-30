"""座標系・GT保存性の回帰テスト (HIGH 修正 + pixel-is-point 統一)。

内部は単一規約 pixel-is-point (整数=画素中心、カメラ・プロジェクタ共通)。
詳細は COORDINATES.md / coords.py を参照。
"""

from __future__ import annotations

import numpy as np
import torch

from graycode import gen_graycode
from graycode.interpolate_p2c import (
    _aggregate_duplicate_points,
    interpolate_p2c_delaunay,
)
from graycode.warp_image import (
    PixelMapWarperTorch,
    SplatMethod,
    AggregationMethod,
    InpaintMethod,
)


def _curved_cam(px: np.ndarray, py: np.ndarray):
    """非アフィンな proj->cam 場 (アフィンだと格子整合の効果が隠れるため)。"""
    cx = 0.9 * px + 3.0 + 4.0 * np.sin(px / 7.0)
    cy = 1.1 * py + 2.0 + 3.0 * np.cos(py / 9.0)
    return cx, cy


# ── P2C 補間が GT を保存し、出力 proj 列が整数中心になる (pixel-is-point) ──


def test_p2c_preserves_gt_on_integer_grid() -> None:
    H = W = 16
    rng = np.random.default_rng(0)
    rows = []
    gt: dict[tuple[int, int], tuple[float, float]] = {}
    for j in range(H):
        for i in range(W):
            if rng.random() < 0.8:  # 部分的な復号被覆を模擬
                px, py = float(i), float(j)  # pixel-is-point (step=1, block_center=g)
                cx, cy = _curved_cam(np.array(px), np.array(py))
                rows.append([px, py, float(cx), float(cy)])
                gt[(i, j)] = (float(cx), float(cy))
    arr = np.array(rows, dtype=np.float32)

    out = interpolate_p2c_delaunay(H, W, arr).reshape(H, W, 4)

    # 出力 proj 座標列は整数 (pixel-is-point の画素中心)
    assert np.allclose(out[5, 7, 0], 7.0)
    assert np.allclose(out[5, 7, 1], 5.0)

    # 復号画素 (GT) は厳密に再現される (クエリが既知点に一致するため)
    errs = [
        abs(out[j, i, 2] - cx) + abs(out[j, i, 3] - cy)
        for (i, j), (cx, cy) in gt.items()
    ]
    assert max(errs) < 1e-3, f"GT not preserved: max abs err = {max(errs)}"


# ── 1対多の重複点を中央値で集約 (規約非依存) ────────────────────────


def test_aggregate_duplicates_is_robust_median() -> None:
    pts = np.array([[1, 1], [1, 1], [1, 1], [2, 2]], dtype=np.float32)
    vals = np.array([[10, 10], [12, 12], [100, 100], [5, 5]], dtype=np.float32)

    up, uv = _aggregate_duplicate_points(pts, vals, stat="median")

    assert len(up) == 2
    row = uv[np.where((up == [1, 1]).all(axis=1))[0][0]]
    assert np.allclose(row, [12, 12])  # 中央値(12), 平均(40.7)でない


def test_p2c_duplicates_not_silently_dropped() -> None:
    # proj 画素(1,1) を 2 観測 (pixel-is-point: 中心=整数 1,1)
    rows = [
        [1.0, 1.0, 10.0, 10.0],
        [1.0, 1.0, 20.0, 20.0],
        [1.0, 5.0, 10.0, 50.0],
        [5.0, 1.0, 50.0, 10.0],
        [5.0, 5.0, 50.0, 50.0],
    ]
    arr = np.array(rows, dtype=np.float32)
    out = interpolate_p2c_delaunay(8, 8, arr).reshape(8, 8, 4)
    # proj 画素(1,1) のカメラ値は 2 観測の中央値 = 15
    assert np.allclose(out[1, 1, 2], 15.0)
    assert np.allclose(out[1, 1, 3], 15.0)


# ── forward_warp の bilinear が画素中心の点を単一画素に載せる ─────────


def test_forward_bilinear_centers_on_single_pixel() -> None:
    # cam(2,2) -> proj(3,3) (pixel-is-point: 整数 3 が画素 3 の中心)
    pmap = np.array([[2.0, 2.0, 3.0, 3.0]], dtype=np.float32)
    w = PixelMapWarperTorch(pmap, device="cpu")
    src = torch.zeros((1, 6, 6))
    src[0, 2, 2] = 100.0

    dst = w.forward_warp(
        src,
        dst_size=(6, 6),
        splat_method=SplatMethod.BILINEAR,
        aggregation=AggregationMethod.MEAN,
        inpaint=InpaintMethod.NONE,
    ).numpy()[0]

    nz = np.argwhere(dst > 1e-6).tolist()
    assert nz == [[3, 3]], f"expected single pixel (3,3), got {nz}"
    assert abs(dst[3, 3] - 100.0) < 1e-4


def test_forward_conv_inpaint_propagates_without_dimming() -> None:
    """Sparse splats are filled outwards while preserving a constant value."""
    pmap = np.array([[2.0, 2.0, 3.0, 3.0]], dtype=np.float32)
    w = PixelMapWarperTorch(pmap, device="cpu")
    src = torch.zeros((1, 6, 6))
    src[0, 2, 2] = 160.0

    dst = w.forward_warp(
        src,
        dst_size=(7, 7),
        splat_method=SplatMethod.BILINEAR,
        aggregation=AggregationMethod.MEAN,
        inpaint=InpaintMethod.CONV,
        inpaint_iter=2,
    ).numpy()[0]

    expected = np.zeros((7, 7), dtype=np.float32)
    expected[1:6, 1:6] = 160.0
    assert np.allclose(dst, expected, atol=1e-4)


def test_forward_backward_roundtrip_identity() -> None:
    H = W = 12
    ys, xs = np.mgrid[0:H, 0:W]
    # 恒等幾何 (pixel-is-point): cam(x,y) -> proj(x,y)
    rows = np.stack(
        [
            xs.ravel().astype(np.float32),
            ys.ravel().astype(np.float32),
            xs.ravel().astype(np.float32),
            ys.ravel().astype(np.float32),
        ],
        axis=1,
    ).astype(np.float32)
    w = PixelMapWarperTorch(rows, device="cpu")

    src = torch.rand((1, H, W))
    uv = w.forward_warp(
        src,
        dst_size=(W, H),
        splat_method=SplatMethod.BILINEAR,
        aggregation=AggregationMethod.MEAN,
        inpaint=InpaintMethod.NONE,
    )
    back = w.backward_warp(
        uv, dst_size=(W, H), mode="bilinear", inpaint=InpaintMethod.NONE
    )

    s = src.numpy()[0]
    b = back.numpy()[0]
    inner = (slice(1, H - 1), slice(1, W - 1))
    assert np.abs(s[inner] - b[inner]).mean() < 1e-3


# ── gen の step 引数順が decode と一致 (height_step, width_step) ──────


def test_gen_cli_step_arg_order(monkeypatch) -> None:
    captured: dict[str, int] = {}

    def fake_gen(height, width, height_step, width_step):
        captured["height_step"] = height_step
        captured["width_step"] = width_step
        return [np.zeros((height, width), np.uint8)]

    monkeypatch.setattr(gen_graycode, "generate_expanded_patterns", fake_gen)
    monkeypatch.setattr(gen_graycode, "save_patterns", lambda *a, **k: None)

    gen_graycode.main(["gen_graycode.py", "16", "32", "2", "4"])

    assert captured["height_step"] == 2
    assert captured["width_step"] == 4
