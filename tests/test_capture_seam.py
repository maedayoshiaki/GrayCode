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
    # Identity mapping up to the +0.5 block-center offset (step=1).
    assert np.median(np.abs(proj_xy - (cam_xy + 0.5))) < 1.0
