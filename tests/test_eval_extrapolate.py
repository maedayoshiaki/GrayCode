"""interpolate_*_delaunay の extrapolate フラグ (外挿オン/オフ) の検証。

extrapolate=False: 凸包外 (投影領域マスク外) を NaN のまま残す。
extrapolate=True (既定): 凸包外を NearestND で埋める (後方互換)。
"""
from __future__ import annotations

import numpy as np

from graycode.interpolate_c2p import interpolate_c2p_delaunay
from graycode.interpolate_p2c import interpolate_p2c_delaunay

# 4 点 (2,2),(5,2),(2,5),(5,5) → 凸包は正方形 [2..5]×[2..5]。16×16 格子なので
# 角 (0,0) は凸包外、(3,3) は凸包内。列順は各関数の規約に合わせて [a,b,va,vb]。
_PTS = np.array([
    [2.0, 2.0, 10.0, 10.0],
    [5.0, 2.0, 20.0, 11.0],
    [2.0, 5.0, 11.0, 20.0],
    [5.0, 5.0, 21.0, 21.0],
], dtype=np.float32)


def test_p2c_extrapolate_false_leaves_nan_outside_hull() -> None:
    out = interpolate_p2c_delaunay(16, 16, _PTS, extrapolate=False).reshape(16, 16, 4)
    assert np.isnan(out[0, 0, 2]) and np.isnan(out[0, 0, 3])  # 凸包外 → NaN
    assert np.isfinite(out[3, 3, 2]) and np.isfinite(out[3, 3, 3])  # 凸包内 → 有限


def test_p2c_extrapolate_true_fills_outside_hull() -> None:
    out = interpolate_p2c_delaunay(16, 16, _PTS, extrapolate=True).reshape(16, 16, 4)
    assert np.isfinite(out[0, 0, 2]) and np.isfinite(out[0, 0, 3])  # Nearest 埋め


def test_p2c_extrapolate_default_is_true() -> None:
    out = interpolate_p2c_delaunay(16, 16, _PTS).reshape(16, 16, 4)  # 既定 = 後方互換
    assert np.isfinite(out[0, 0, 2])


def test_c2p_extrapolate_false_leaves_nan_outside_hull() -> None:
    out = interpolate_c2p_delaunay(16, 16, _PTS, extrapolate=False).reshape(16, 16, 4)
    assert np.isnan(out[0, 0, 2]) and np.isnan(out[0, 0, 3])
    assert np.isfinite(out[3, 3, 2]) and np.isfinite(out[3, 3, 3])


def test_c2p_extrapolate_default_is_true() -> None:
    out = interpolate_c2p_delaunay(16, 16, _PTS).reshape(16, 16, 4)
    assert np.isfinite(out[0, 0, 2])
