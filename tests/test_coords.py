"""coords.py の正準定義に対する単体テスト。

ここが座標規約の単一の真実源。内部は単一規約 pixel-is-point (整数=画素中心) を
用い、texel 規約 (中心=i+0.5) は grid_sample 境界の point_to_normalized のみ。
numpy/torch 双方の backend を固定する。詳細は COORDINATES.md を参照。
"""

from __future__ import annotations

import numpy as np
import torch

from graycode import coords


# ── 画素中心 = 整数 (pixel-is-point) ─────────────────────────────────


def test_pixel_centers() -> None:
    assert np.array_equal(coords.pixel_centers(4), [0.0, 1.0, 2.0, 3.0])


def test_to_pixel_is_round_half_up() -> None:
    c = np.array([0.0, 2.0, 2.4, 2.5, -0.6])
    # floor(c+0.5): 0, 2, 2, 3, -1
    assert np.array_equal(coords.to_pixel(c), [0.0, 2.0, 2.0, 3.0, -1.0])


def test_pixel_centers_roundtrip() -> None:
    n = 8
    centers = coords.pixel_centers(n)
    assert np.array_equal(coords.to_pixel(centers), np.arange(n))


# ── grid_sample 境界変換 (唯一の texel 規約) ─────────────────────────


def test_point_to_normalized_matches_grid_sample_align_corners_false() -> None:
    # align_corners=False: array_pixel_center = ((g+1)*size - 1) / 2
    size = 4
    for i in range(size):  # pixel-is-point 座標 i (= 画素 i の中心)
        g = coords.point_to_normalized(i, size)
        pixel = ((g + 1.0) * size - 1.0) / 2.0
        assert abs(pixel - i) < 1e-9  # 配列画素 i の中心に戻る


# ── GrayCode の step (ブロック / 縮小解像度) ─────────────────────────


def test_reduced_size() -> None:
    assert coords.reduced_size(1920, 1) == 1920
    assert coords.reduced_size(1920, 4) == 480
    assert coords.reduced_size(10, 3) == 4


def test_block_of() -> None:
    assert coords.block_of(7, 3) == 2
    assert np.array_equal(coords.block_of(np.array([0, 2, 3, 5]), 3), [0, 0, 1, 1])


def test_block_center_step1_is_pixel_center() -> None:
    g = np.arange(5)
    assert np.array_equal(coords.block_center(g, 1), coords.pixel_centers(5))


def test_block_center_scaling_and_parity() -> None:
    # ブロック中心 = step*g + (step-1)/2
    assert coords.block_center(0, 4) == 1.5  # 画素[0..3]の中心
    assert coords.block_center(2, 4) == 9.5  # 画素[8..11]の中心
    assert coords.block_center(2, 3) == 7.0  # 奇数 step は整数 (画素[6..8]の中心=7)


def test_block_center_is_old_uv_center_minus_half() -> None:
    # 移行関係: pixel-is-point 中心 = 旧 UV 中心 step*(g+0.5) - 0.5 (step 不問)
    for step in (1, 2, 3, 4):
        for g in range(4):
            assert coords.block_center(g, step) == step * (g + 0.5) - 0.5


# ── backend: torch テンソルでも同じ規約 ─────────────────────────────


def test_torch_backend_matches_numpy() -> None:
    cs = [0.0, 2.4, 2.5, -0.6]
    assert torch.equal(
        coords.to_pixel(torch.tensor(cs)), torch.tensor([0.0, 2.0, 3.0, -1.0])
    )
    # point_to_normalized も torch で動く
    g = coords.point_to_normalized(torch.tensor([0.0, 1.0]), 4)
    assert torch.allclose(g, torch.tensor([2.0 * 0.5 / 4 - 1, 2.0 * 1.5 / 4 - 1]))
