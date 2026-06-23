"""graycode.evaluation.patterns の検証テスト (A3 既知パターン生成)。"""
from __future__ import annotations

import numpy as np
import pytest

from graycode.evaluation import patterns


def test_checkerboard_true_coords_count_and_determinism() -> None:
    img, tc = patterns.generate_checkerboard_pattern(
        1920, 1080, squares_x=13, squares_y=9, square_px=80
    )
    assert img.shape == (1080, 1920)
    assert img.dtype == np.uint8
    # 内側コーナー数 = (squares_x-1)*(squares_y-1)
    assert len(tc) == (13 - 1) * (9 - 1)
    # 決定的
    _, tc2 = patterns.generate_checkerboard_pattern(1920, 1080, 13, 9, 80)
    assert tc == tc2


def test_checkerboard_corner_spacing() -> None:
    img, tc = patterns.generate_checkerboard_pattern(
        1280, 800, squares_x=9, squares_y=7, square_px=100
    )
    # id 0 と id 1 (同じ行の隣) は square_px だけ離れる
    inner_x = 9 - 1
    c0 = np.array(tc[0])
    c1 = np.array(tc[1])
    assert abs((c1[0] - c0[0]) - 100.0) < 1e-9
    # 次の行 (id = inner_x) は y が square_px ずれる
    c_row = np.array(tc[inner_x])
    assert abs((c_row[1] - c0[1]) - 100.0) < 1e-9


def test_charuco_generation_optional() -> None:
    if not patterns._HAS_ARUCO:
        pytest.skip("cv2.aruco not available")
    try:
        img, tc, board = patterns.generate_charuco_pattern(
            1280, 800, squares_x=8, squares_y=6, square_px=120
        )
    except Exception as e:  # cv2.aruco API 差異に対して頑健にスキップ
        pytest.skip(f"charuco API mismatch: {e}")
    assert img.shape == (800, 1280)
    assert len(tc) > 0
    # 各真座標がプロジェクタ範囲内
    for x, y in tc.values():
        assert 0 <= x <= 1280 and 0 <= y <= 800
