"""graycode.evaluation.io の検証テスト (2dsr-prc ProjectorCameraMap .npz 連携)。"""
from __future__ import annotations

import numpy as np

from graycode.evaluation import io, metrics


def _write_pcmap_npz(path, p2c, proj_size, convention="pixel-is-point"):
    """2dsr-prc ProjectorCameraMap.save と同形式の .npz を書く。"""
    np.savez(
        path,
        p2c=np.asarray(p2c, dtype=np.float32),
        proj_size=np.asarray(proj_size, dtype=np.int64),
        coord_convention=np.asarray(convention),
    )


def test_load_npz_reorders_columns(tmp_path) -> None:
    # p2c rows: [proj_x, proj_y, cam_x, cam_y]
    p2c = np.array([
        [10.0, 20.0, 1.0, 2.0],
        [11.0, 21.0, 3.0, 4.0],
    ], dtype=np.float32)
    path = tmp_path / "p2c.npz"
    _write_pcmap_npz(path, p2c, (1080, 1920))

    corr, proj_size, conv = io.load_projector_camera_map_npz(path)
    # corr は [cam_x, cam_y, proj_x, proj_y] に並べ替えられる
    assert np.allclose(corr[0], [1.0, 2.0, 10.0, 20.0])
    assert np.allclose(corr[1], [3.0, 4.0, 11.0, 21.0])
    assert proj_size == (1080, 1920)
    assert conv == "pixel-is-point"


def test_p2c_grid_roundtrip(tmp_path) -> None:
    p2c = np.array([[10.0, 20.0, 1.0, 2.0]], dtype=np.float32)
    path = tmp_path / "p2c.npz"
    _write_pcmap_npz(path, p2c, (800, 1280))
    corr, _, _ = io.load_projector_camera_map_npz(path)
    # 列順を戻すと元の p2c [proj,proj,cam,cam] に一致 (A1 の dense_p2c 用)
    back = io.p2c_grid_from_correspondences(corr)
    assert np.allclose(back, p2c)


def test_load_npz_missing_p2c_raises(tmp_path) -> None:
    path = tmp_path / "bad.npz"
    np.savez(path, foo=np.zeros(3))
    try:
        io.load_projector_camera_map_npz(path)
        assert False, "should raise"
    except ValueError as e:
        assert "p2c" in str(e)


def test_npz_correspondences_feed_a4(tmp_path) -> None:
    """npz 由来の対応がそのまま A4 (epipolar Sampson) に流せる。"""
    import cv2

    rng = np.random.default_rng(0)
    K = np.array([[800.0, 0, 320.0], [0, 800.0, 240.0], [0, 0, 1.0]])
    R, _ = cv2.Rodrigues(np.array([0.02, 0.10, 0.01]))
    t = np.array([[-100.0], [5.0], [20.0]])
    P1 = K @ np.hstack([np.eye(3), np.zeros((3, 1))])
    P2 = K @ np.hstack([R, t])
    X = np.column_stack([rng.uniform(-200, 200, 200), rng.uniform(-200, 200, 200),
                         rng.uniform(400, 900, 200), np.ones(200)])
    x1 = (P1 @ X.T).T; x1 = x1[:, :2] / x1[:, 2:3]  # cam
    x2 = (P2 @ X.T).T; x2 = x2[:, :2] / x2[:, 2:3]  # proj
    # ProjectorCameraMap stores p2c = [proj_x,proj_y,cam_x,cam_y]
    p2c = np.column_stack([x2, x1]).astype(np.float32)
    path = tmp_path / "p2c.npz"
    _write_pcmap_npz(path, p2c, (480, 640))

    corr, _, _ = io.load_projector_camera_map_npz(path)
    r = metrics.epipolar_sampson(corr, ransac_thresh=1.0, seed=0)
    assert r.inlier_ratio > 0.9
    assert r.stats_all.median < 1.0  # 整合した対応 → 小さい Sampson
