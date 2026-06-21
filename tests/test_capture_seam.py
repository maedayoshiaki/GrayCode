"""Fake-camera test for the injectable capture seam (no hardware, no display).

Drives ``run_calibration`` with a synthetic camera that sees the projected
pattern 1:1 (identity projector↔camera mapping) and a no-op display, then
asserts the decoded p2c map recovers that identity. This exercises the
gen → display+capture → decode → p2c path without a Canon EDSDK or a projector.
"""

from __future__ import annotations

import numpy as np

from graycode import run_calibration


def test_run_calibration_recovers_identity_mapping() -> None:
    proj_h = proj_w = 32

    # The "screen": display_fn records the projected pattern; the fake camera
    # returns exactly that frame (identity projector→camera, 1:1, same size).
    screen: dict[str, np.ndarray] = {}

    def display_fn(index: int, pattern: np.ndarray) -> None:
        screen["current"] = pattern

    def capture_fn() -> np.ndarray:
        return screen["current"].copy()

    p2c = run_calibration(
        capture_fn=capture_fn,
        display_fn=display_fn,
        proj_height=proj_h,
        proj_width=proj_w,
        wait_ms=0,
        black_threshold=5,
        white_threshold=20,
    )

    assert p2c.ndim == 2 and p2c.shape[1] == 4
    assert len(p2c) > 0.5 * proj_h * proj_w  # most pixels decode under identity

    proj_xy = p2c[:, 0:2]
    cam_xy = p2c[:, 2:4]
    # pixel-is-point 統一後: 恒等幾何 (step=1) では proj == cam (block_center=g)。
    assert np.median(np.abs(proj_xy - cam_xy)) < 1.0


def test_run_calibration_debug_dir_saves_artifacts(tmp_path) -> None:
    proj_h = proj_w = 16

    screen: dict[str, np.ndarray] = {}

    def display_fn(index: int, pattern: np.ndarray) -> None:
        screen["current"] = pattern

    def capture_fn() -> np.ndarray:
        return screen["current"].copy()

    dbg = tmp_path / "gc_debug"
    run_calibration(
        capture_fn=capture_fn,
        display_fn=display_fn,
        proj_height=proj_h,
        proj_width=proj_w,
        wait_ms=0,
        black_threshold=5,
        white_threshold=20,
        debug_dir=str(dbg),
    )

    captures = sorted(dbg.glob("capture_*.png"))
    assert len(captures) > 0  # gray-code bit captures saved
    assert (dbg / "white.png").exists()
    assert (dbg / "black.png").exists()
    assert (dbg / "valid_mask.png").exists()
    assert (dbg / "result_c2p.npy").exists()  # raw (un-densified) c2p for visualization
    import cv2

    vm = cv2.imread(str(dbg / "valid_mask.png"), cv2.IMREAD_GRAYSCALE)
    assert vm.shape == (proj_h, proj_w)  # identity: camera == projector size
    assert (vm > 0).mean() > 0.9  # identity → almost all pixels valid
