import glob
import sys
from pathlib import Path
from typing import List

import cv2
import numpy as np

from src.python.camera import create_camera_backend
from src.python.config import CameraConfig as SharedCameraConfig

from .config import get_config, reload_config, split_cli_config_path


def _normalize_camera_backend_name(backend: str) -> str:
    backend_name = backend.strip().lower()
    if backend_name == "edsdk":
        return "canon_edsdk"
    return backend_name


def _build_shared_camera_config() -> SharedCameraConfig:
    cam_cfg = get_config().camera
    return SharedCameraConfig(
        backend=_normalize_camera_backend_name(cam_cfg.backend),
        av=cam_cfg.av,
        tv=cam_cfg.tv,
        iso=cam_cfg.iso,
        image_quality=cam_cfg.image_quality,
        device_index=cam_cfg.device_index,
        wait_key_ms=cam_cfg.wait_key_ms,
    )


def _linear_to_srgb(linear_rgb: np.ndarray) -> np.ndarray:
    linear_rgb = np.clip(np.asarray(linear_rgb, dtype=np.float32), 0.0, 1.0)
    return np.where(
        linear_rgb <= 0.0031308,
        12.92 * linear_rgb,
        1.055 * np.power(linear_rgb, 1.0 / 2.4) - 0.055,
    )


def capture() -> np.ndarray:
    camera = create_camera_backend(_build_shared_camera_config())
    linear_rgb = camera.capture_linear_rgb()
    srgb = _linear_to_srgb(linear_rgb)
    return np.clip(srgb * 255.0, 0.0, 255.0).astype(np.uint8)


def print_usage() -> None:
    print(
        "Usage : python cap_graycode.py <window position x> <window position y> "
        "[--config <config.toml>]"
    )
    print()


def main(argv: list[str] | None = None) -> None:
    if argv is None:
        argv = sys.argv
    try:
        argv, config_path = split_cli_config_path(argv)
    except ValueError as e:
        print(e)
        print_usage()
        return

    if config_path is not None:
        reload_config(config_path)

    if len(argv) != 3:
        print_usage()
        return

    try:
        window_pos_x = int(argv[1])
        window_pos_y = int(argv[2])
    except ValueError:
        print("window_pos_x, window_pos_y は整数で指定してください。")
        print_usage()
        return
    cfg = get_config()
    target_dir = Path(cfg.paths.pattern_dir)
    capture_dir = Path(cfg.paths.captured_dir)
    wait_ms = cfg.camera.wait_key_ms

    graycode_imgs: List[np.ndarray] = []
    # グレイコードをファイルから参照
    for fname in sorted(glob.glob(str(target_dir / "pattern_*.png"))):
        print(f"Loading pattern image: {fname}")
        pat_img = cv2.imread(fname, cv2.IMREAD_GRAYSCALE)
        if pat_img is None:
            raise RuntimeError(f"Failed to load pattern image: {fname}")
        graycode_imgs.append(pat_img)

    cv2.namedWindow("Pattern", cv2.WINDOW_NORMAL)
    cv2.setWindowProperty("Pattern", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
    cv2.moveWindow("Pattern", window_pos_x, window_pos_y)

    # キャプチャディレクトリ作成
    capture_dir.mkdir(parents=True, exist_ok=True)

    try:
        for i, pat in enumerate(graycode_imgs):
            print(f"Displaying pattern image {i:02d}...")
            cv2.imshow("Pattern", pat)
            cv2.waitKey(wait_ms)
            captured_img = capture()
            if captured_img.ndim == 2:
                captured_img_gray = captured_img
            else:
                captured_img_gray = cv2.cvtColor(captured_img, cv2.COLOR_RGB2GRAY)
            cv2.imwrite(f"{capture_dir}/capture_{i:02d}.png", captured_img_gray)
            print(f"Captured and saved image: capture_{i:02d}.png")
    finally:
        cv2.destroyAllWindows()

    print("All patterns have been captured and saved.")

    print()
    print("=== Next step ===")
    print(
        "Run 'python decode.py <projector image height> <projector image width>' to decode the captured images."
    )
    print()


if __name__ == "__main__":
    main()
