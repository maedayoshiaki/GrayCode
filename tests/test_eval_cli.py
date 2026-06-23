# coding: utf-8
"""cli.py の密マップ構築 (_get_dense_c2p/_get_dense_p2c) の列順回帰テスト。

``--no-extrapolate`` + npz 経路 (scan-eval が踏む) で密 P2C を Delaunay 再構築する際、
``corr`` の列順 ``[cam_x,cam_y,proj_x,proj_y]`` を ``interpolate_p2c_delaunay`` の期待
``[proj_x,proj_y,cam_x,cam_y]`` に戻し忘れると points/values が入れ替わり、座標系ごと
壊れた密 P2C になって A1 cycle が ~100px に化ける (実機 1280x800 で 105px を観測)。

恒等写像では入替を検出できないため、**非対称アフィン** ``proj = 0.5*cam + 5``
(逆 ``cam = 2*proj - 10``) で往復が ≈0 になることを保証して再発を防ぐ。
"""
from __future__ import annotations

from types import SimpleNamespace

import numpy as np

from graycode.evaluation import cli, metrics


def _make_npz(tmp_path):
    """ProjectorCameraMap 互換 .npz を合成。p2c 列順 = [proj_x,proj_y,cam_x,cam_y]。"""
    xs, ys = np.meshgrid(
        np.arange(2, 18, dtype=np.float64), np.arange(2, 18, dtype=np.float64)
    )
    cam = np.column_stack([xs.ravel(), ys.ravel()])
    proj = 0.5 * cam + 5.0  # 非対称: A!=I, b!=0 → 入替バグを検出できる
    p2c = np.column_stack([proj, cam]).astype(np.float32)  # [proj_x,proj_y,cam_x,cam_y]
    path = tmp_path / "p2c.npz"
    np.savez(
        path, p2c=p2c, proj_size=np.array([20, 20]), coord_convention="pixel-is-point"
    )
    return path


def _args(npz):
    return SimpleNamespace(
        p2c_npz=str(npz), raw_c2p=None, dense_c2p=None, dense_p2c=None,
        cam_h=20, cam_w=20, no_extrapolate=True,
    )


def test_dense_p2c_uses_correct_column_order(tmp_path) -> None:
    """密 P2C の値列 (cam) が proj 画素で cam=2*proj-10 を満たす (列順が正しい証拠)。"""
    npz = _make_npz(tmp_path)
    args = _args(npz)
    corr, _ps, _conv = cli._get_correspondences(args, tmp_path)
    dp2c = cli._get_dense_p2c(args, tmp_path, corr, 20, 20).reshape(20, 20, 4)
    # proj=(8,12) は凸包内かつアフィンの不動点 (cam==proj) を避ける: cam=2*proj-10=(6,14)。
    # 列順を取り違えると points=cam/values=proj になり cam=(9,11) を返して検出できる。
    val = dp2c[12, 8]  # reshape は [y,x] 行優先 → proj_x=8, proj_y=12
    assert np.isfinite(val[2]) and np.isfinite(val[3])
    assert abs(val[2] - (2 * 8 - 10)) < 1e-2  # cam_x = 6
    assert abs(val[3] - (2 * 12 - 10)) < 1e-2  # cam_y = 14


def test_a1_cycle_near_zero_via_cli_dense_maps(tmp_path) -> None:
    """cli が組む密 C2P/P2C で往復 ≈0 (列順バグ時は ~100px)。"""
    npz = _make_npz(tmp_path)
    args = _args(npz)
    corr, _ps, _conv = cli._get_correspondences(args, tmp_path)
    dc2p = cli._get_dense_c2p(args, tmp_path, corr)
    dp2c = cli._get_dense_p2c(args, tmp_path, corr, 20, 20)
    r = metrics.cycle_consistency(dc2p, dp2c, (20, 20), (20, 20))
    assert r.stats.n > 0
    assert r.stats.rmse < 1e-2
