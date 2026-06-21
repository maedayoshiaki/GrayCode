"""M2 (外挿マスク) と M6 (出力ディレクトリ設定) の回帰テスト。"""

from __future__ import annotations

import numpy as np

from graycode import interpolate_p2c
from graycode.config import AppConfig, PathsConfig, reload_config, resolve_output_path
from graycode.interpolate_c2p import interpolate_c2p_delaunay
from graycode.interpolate_p2c import interpolate_p2c_delaunay


# ── M6: 出力パス解決 ─────────────────────────────────────────────────


def test_resolve_output_path_joins_and_creates_dir(tmp_path) -> None:
    cfg = AppConfig(paths=PathsConfig(output_dir=str(tmp_path / "out")))
    p = resolve_output_path("result_p2c.npy", config=cfg)
    assert p == (tmp_path / "out" / "result_p2c.npy")
    assert (tmp_path / "out").is_dir()  # 自動作成される


def test_resolve_output_path_default_is_cwd() -> None:
    cfg = AppConfig()  # 既定 output_dir = "."
    p = resolve_output_path("x.npy", config=cfg)
    assert str(p) == "x.npy"


# ── M2: 凸包外 (Nearest 外挿) マスク ─────────────────────────────────


def _inner_block_rows(proj_first: bool):
    """内側 [2..5]^2 のみ既知点を作る (端は凸包外になる)。"""
    rows = []
    for j in range(2, 6):
        for i in range(2, 6):
            if proj_first:  # p2c: [proj_x, proj_y, cam_x, cam_y]
                rows.append([i, j, i * 1.0, j * 1.0])
            else:  # c2p: [cam_x, cam_y, proj_x, proj_y]
                rows.append([i, j, i * 2.0, j * 3.0])
    return np.array(rows, dtype=np.float32)


def test_p2c_delaunay_return_mask_flags_exterior() -> None:
    H = W = 8
    out, mask = interpolate_p2c_delaunay(
        H, W, _inner_block_rows(proj_first=True), return_mask=True
    )
    assert mask.shape == (H * W,) and mask.dtype == bool
    assert bool(mask[0 * W + 0]) is True  # 角 (0,0) は凸包外 -> 外挿
    assert bool(mask[3 * W + 3]) is False  # 内側 (3,3) は補間


def test_c2p_delaunay_return_mask_flags_exterior() -> None:
    H = W = 8
    out, mask = interpolate_c2p_delaunay(
        H, W, _inner_block_rows(proj_first=False), return_mask=True
    )
    assert mask.shape == (H * W,) and mask.dtype == bool
    assert bool(mask[0 * W + 0]) is True
    assert bool(mask[3 * W + 3]) is False


def test_delaunay_default_return_is_array_only() -> None:
    # return_mask=False (既定) では (N,4) 配列のみを返す (後方互換)
    out = interpolate_p2c_delaunay(8, 8, _inner_block_rows(proj_first=True))
    assert isinstance(out, np.ndarray) and out.shape == (64, 4)


# ── M2 + M6 結合: main が output_dir にマスク等を書き出す ─────────────


def test_interpolate_p2c_main_writes_to_output_dir(tmp_path) -> None:
    cfg_toml = tmp_path / "cfg.toml"
    cfg_toml.write_text(
        f'[paths]\noutput_dir = "{(tmp_path / "out").as_posix()}"\n', encoding="utf-8"
    )
    inp = tmp_path / "result_p2c.npy"
    np.save(inp, _inner_block_rows(proj_first=True))

    try:
        interpolate_p2c.main(
            ["interpolate_p2c.py", str(inp), "8", "8", "--config", str(cfg_toml)]
        )
        out_dir = tmp_path / "out"
        assert (out_dir / "result_p2c_compensated_delaunay.npy").exists()
        assert (out_dir / "result_p2c_compensated_delaunay.csv").exists()
        assert (out_dir / "result_p2c_compensated_delaunay_extrapolated.png").exists()
    finally:
        reload_config()  # グローバル設定を既定へ戻す (テスト隔離)
