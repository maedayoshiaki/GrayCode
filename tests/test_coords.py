"""coords.py の正準定義に対する単体テスト。

ここが座標規約の単一の真実源。各規約 (カメラ XY=整数中心 / プロジェクタ UV=i+0.5中心
/ GrayCode の step ブロック) と、numpy/torch 双方の backend を固定する。
詳細は COORDINATES.md を参照。
"""

from __future__ import annotations

import numpy as np
import torch

from graycode import coords


# ── カメラ (XY): 画素中心 = 整数 ──────────────────────────────────────


def test_xy_pixel_centers() -> None:
    assert np.array_equal(coords.xy_pixel_centers(4), [0.0, 1.0, 2.0, 3.0])


def test_xy_to_pixel_is_round_half_up() -> None:
    x = np.array([0.0, 2.0, 2.4, 2.5, -0.6])
    # floor(x+0.5): 0, 2, 2, 3, -1
    assert np.array_equal(coords.xy_to_pixel(x), [0.0, 2.0, 2.0, 3.0, -1.0])


def test_xy_centers_roundtrip() -> None:
    # 画素中心を変換すると自分自身の画素インデックスに戻る
    n = 8
    centers = coords.xy_pixel_centers(n)
    assert np.array_equal(coords.xy_to_pixel(centers), np.arange(n))


# ── プロジェクタ (UV): 画素中心 = 整数 + 0.5 ─────────────────────────


def test_uv_pixel_centers() -> None:
    assert np.array_equal(coords.uv_pixel_centers(4), [0.5, 1.5, 2.5, 3.5])


def test_uv_to_pixel_is_floor() -> None:
    u = np.array([0.0, 0.5, 2.0, 2.9, 2.5])
    assert np.array_equal(coords.uv_to_pixel(u), [0.0, 0.0, 2.0, 2.0, 2.0])


def test_uv_centers_roundtrip() -> None:
    n = 8
    centers = coords.uv_pixel_centers(n)
    assert np.array_equal(coords.uv_to_pixel(centers), np.arange(n))


def test_uv_array_roundtrip() -> None:
    u = np.array([0.5, 1.5, 7.5])
    assert np.allclose(coords.array_to_uv(coords.uv_to_array(u)), u)
    # UV 中心 i+0.5 は配列インデックス座標で整数 i になる
    assert np.allclose(coords.uv_to_array(coords.uv_pixel_centers(5)), np.arange(5))


def test_uv_to_normalized_matches_grid_sample_align_corners_false() -> None:
    # align_corners=False: pixel = ((g+1)*size - 1) / 2
    size = 4
    for i in range(size):
        u = i + 0.5  # UV 画素 i の中心
        g = coords.uv_to_normalized(u, size)
        pixel = ((g + 1.0) * size - 1.0) / 2.0
        assert abs(pixel - i) < 1e-9  # 画素 i の配列中心に戻る


# ── GrayCode の step (ブロック / 縮小解像度) ─────────────────────────


def test_reduced_size() -> None:
    assert coords.reduced_size(1920, 1) == 1920
    assert coords.reduced_size(1920, 4) == 480
    assert coords.reduced_size(10, 3) == 4  # 端数切り上げ的挙動


def test_block_of() -> None:
    assert coords.block_of(7, 3) == 2
    assert np.array_equal(coords.block_of(np.array([0, 2, 3, 5]), 3), [0, 0, 1, 1])


def test_block_center_uv_step1_is_uv_center() -> None:
    g = np.arange(5)
    # step=1 ではブロック中心 = 画素中心 i+0.5
    assert np.array_equal(coords.block_center_uv(g, 1), coords.uv_pixel_centers(5))


def test_block_center_uv_step_scaling() -> None:
    assert coords.block_center_uv(0, 4) == 2.0  # ブロック[0,4)の中心
    assert coords.block_center_uv(2, 4) == 10.0  # ブロック[8,12)の中心


def test_block_center_parity_note() -> None:
    # 奇数 step: ブロック中心は単一画素中心に一致 (i = step*g + (step-1)/2)
    step, g = 3, 2
    c = coords.block_center_uv(g, step)  # 3*2.5 = 7.5
    i = step * g + (step - 1) // 2  # 7
    assert c == i + 0.5
    assert coords.uv_to_pixel(c) == i


# ── backend: torch テンソルでも同じ規約が成り立つ ────────────────────


def test_torch_backend_matches_numpy() -> None:
    xs = [0.0, 2.4, 2.5, -0.6]
    us = [0.0, 0.5, 2.9]
    assert torch.equal(
        coords.xy_to_pixel(torch.tensor(xs)), torch.tensor([0.0, 2.0, 3.0, -1.0])
    )
    assert torch.equal(
        coords.uv_to_pixel(torch.tensor(us)), torch.tensor([0.0, 0.0, 2.0])
    )
    # uv_to_array も torch で動く
    assert torch.allclose(
        coords.uv_to_array(torch.tensor([0.5, 1.5])), torch.tensor([0.0, 1.0])
    )
