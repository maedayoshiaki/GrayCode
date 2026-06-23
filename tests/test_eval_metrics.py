"""graycode.evaluation.metrics の検証テスト (A1-A4)。

合成データで「恒等→0」「既知オフセット→厳密一致」「既知 F→Sampson 0」「外れ値検出」を確認。
座標は pixel-is-point (整数=画素中心)。
"""
from __future__ import annotations

import cv2
import numpy as np

from graycode.evaluation import metrics


# ── 格子ビルダ ────────────────────────────────────────────────────────


def _grid_map(height: int, width: int, fx, fy) -> np.ndarray:
    """cam/proj 格子上の (N,4) マップ。列 [a,b,fx(a,b),fy(a,b)] (y外, x内 行優先)。"""
    ys, xs = np.mgrid[0:height, 0:width]
    a = xs.ravel().astype(np.float64)
    b = ys.ravel().astype(np.float64)
    return np.column_stack([a, b, fx(a, b), fy(a, b)])


# ── A1: 往復整合性 ────────────────────────────────────────────────────


def test_a1_identity_is_zero() -> None:
    H = W = 8
    c2p = _grid_map(H, W, lambda x, y: x, lambda x, y: y)
    p2c = _grid_map(H, W, lambda x, y: x, lambda x, y: y)
    r = metrics.cycle_consistency(c2p, p2c, (H, W), (H, W))
    assert r.stats.rmse < 1e-9


def test_a1_known_shift() -> None:
    H = W = 8
    c2p = _grid_map(H, W, lambda x, y: x, lambda x, y: y)
    # p2c: proj(x,y) -> cam(x+1, y) → 往復残差 = (1,0)
    p2c = _grid_map(H, W, lambda x, y: x + 1.0, lambda x, y: y)
    r = metrics.cycle_consistency(c2p, p2c, (H, W), (H, W))
    assert abs(r.stats.median - 1.0) < 1e-6
    assert abs(r.stats.rmse - 1.0) < 1e-6


# ── A2: 補間ホールドアウト ───────────────────────────────────────────


def _scatter(height: int, width: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    ys, xs = np.mgrid[0:height, 0:width]
    pts = np.column_stack([xs.ravel(), ys.ravel()]).astype(np.float64)
    pts += rng.normal(0.0, 0.1, pts.shape)
    return pts


def test_a2_affine_field_near_zero() -> None:
    pts = _scatter(20, 20, seed=1)
    A = np.array([[1.2, 0.1], [-0.05, 0.9]])
    b = np.array([3.0, -2.0])
    proj = pts @ A.T + b
    raw = np.column_stack([pts, proj])
    r = metrics.holdout_interpolation_residual(raw, test_frac=0.2, seed=2)
    assert r.stats.n > 0
    # 線形補間はアフィン場を厳密復元 → 凸包内の残差はほぼ 0
    assert r.stats.rmse < 1e-6


def test_a2_nonaffine_field_positive() -> None:
    pts = _scatter(24, 24, seed=3)
    proj_x = pts[:, 0] + 5.0 * np.sin(pts[:, 1] / 3.0)
    proj_y = pts[:, 1] + 4.0 * np.cos(pts[:, 0] / 4.0)
    raw = np.column_stack([pts, proj_x, proj_y])
    r = metrics.holdout_interpolation_residual(raw, test_frac=0.2, seed=4)
    assert r.stats.n > 0
    assert r.stats.rmse > 1e-3  # 非線形場では補間誤差が出る


# ── A3: 既知パターン絶対誤差 ─────────────────────────────────────────


def test_a3_recovers_injected_offset() -> None:
    H = W = 40
    dense = _grid_map(H, W, lambda x, y: x, lambda x, y: y)  # 恒等 dense C2P
    rng = np.random.default_rng(3)
    delta = np.array([2.0, 3.0])
    feats = []
    true = {}
    for k in range(25):
        cx = rng.uniform(5.0, W - 5.0)
        cy = rng.uniform(5.0, H - 5.0)
        feats.append([cx, cy, k])
        true[k] = (cx - delta[0], cy - delta[1])  # 真値を delta だけずらして埋め込む
    res = metrics.known_pattern_error(np.array(feats), true, dense, (H, W))
    assert res.residual.shape[0] == 25
    assert np.allclose(res.residual, delta, atol=1e-6)
    assert abs(res.stats.rmse - np.hypot(*delta)) < 1e-6


# ── A4: エピポーラ Sampson ───────────────────────────────────────────


def _projective_corr(n: int, seed: int):
    rng = np.random.default_rng(seed)
    K = np.array([[800.0, 0, 320.0], [0, 800.0, 240.0], [0, 0, 1.0]])
    rvec = np.array([0.02, 0.10, 0.01])
    R, _ = cv2.Rodrigues(rvec)
    t = np.array([[-100.0], [5.0], [20.0]])
    P1 = K @ np.hstack([np.eye(3), np.zeros((3, 1))])
    P2 = K @ np.hstack([R, t])
    X = np.column_stack([
        rng.uniform(-200, 200, n),
        rng.uniform(-200, 200, n),
        rng.uniform(400, 900, n),
        np.ones(n),
    ])
    x1 = (P1 @ X.T).T
    x1 = x1[:, :2] / x1[:, 2:3]
    x2 = (P2 @ X.T).T
    x2 = x2[:, :2] / x2[:, 2:3]
    return np.column_stack([x1, x2]), P1, P2


def _F_from_P(P1: np.ndarray, P2: np.ndarray) -> np.ndarray:
    C1 = np.array([0.0, 0.0, 0.0, 1.0])  # camera-1 center (P1 = K[I|0])
    e2 = (P2 @ C1).ravel()
    skew = np.array([
        [0, -e2[2], e2[1]],
        [e2[2], 0, -e2[0]],
        [-e2[1], e2[0], 0],
    ])
    return skew @ P2 @ np.linalg.pinv(P1)


def test_sampson_zero_on_exact_epipolar() -> None:
    corr, P1, P2 = _projective_corr(60, seed=7)
    F = _F_from_P(P1, P2)
    d = metrics.sampson_distance(corr, F)
    assert np.nanmax(d) < 1e-6  # x2^T F x1 = 0 を満たすので 0
    # 規約を独立に固定: cam<->proj 列入替や F 転置では大きくずれる
    # (自己整合的な入替バグが「0」で通り抜けるのを防ぐ)
    swapped = corr[:, [2, 3, 0, 1]]
    assert np.nanmedian(metrics.sampson_distance(swapped, F)) > 1.0
    assert np.nanmedian(metrics.sampson_distance(corr, F.T)) > 1.0


def test_a4_consistent_correspondences_small_sampson() -> None:
    corr, _, _ = _projective_corr(300, seed=4)
    r = metrics.epipolar_sampson(corr, ransac_thresh=1.0, seed=0)
    assert r.inlier_ratio > 0.9
    assert r.stats_all.median < 1.0  # ノイズなし → ほぼ 0
    assert r.stats_all.p99 < 1.0  # 裾も小さい (規約が正しく揃っている証拠)


def test_a4_flags_outliers() -> None:
    from graycode.evaluation import stats as _st

    corr, _, _ = _projective_corr(300, seed=5)
    rng = np.random.default_rng(6)
    bad = corr.copy()
    idx = rng.choice(len(bad), 30, replace=False)
    bad[idx, 2:4] = rng.uniform(0.0, 640.0, (30, 2))  # gross outliers
    r = metrics.epipolar_sampson(bad, ransac_thresh=1.0, seed=0)
    # 外れ値は大きな Sampson 距離を持つ
    assert np.nanmedian(r.sampson[idx]) > 1.0
    # ロバスト内れ値統計は真の内れ値 (~0) に近い (全体統計に引きずられない)
    assert r.stats_inliers.median < 0.1
    # MAD ロバストマスクが注入外れ値の大半を除外する
    inmask = _st.robust_inlier_mask(r.sampson, k=3.0)
    assert inmask[idx].mean() < 0.2
