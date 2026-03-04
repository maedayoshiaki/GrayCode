import cv2
import numpy as np
import glob
import sys
from pathlib import Path
from typing import List

from .config import get_config, reload_config, split_cli_config_path


def _capture_with_edsdk() -> np.ndarray:
    try:
        from edsdk.camera_controller import CameraController
    except ImportError as e:
        raise ImportError(
            "The 'edsdk' module is required for capture. "
            "Install EDSDK bindings or skip capture in pipeline."
        ) from e

    cam_cfg = get_config().camera
    with CameraController(register_property_events=False) as camera:
        camera.set_properties(
            av=cam_cfg.av,
            tv=cam_cfg.tv,
            iso=cam_cfg.iso,
            image_quality=cam_cfg.image_quality,
        )
        imgs = camera.capture_numpy()
        if not imgs:
            raise RuntimeError("No image returned from EDSDK camera.")
        img = imgs[0]
    return img


def _capture_with_opencv(device_index: int) -> np.ndarray:
    cap = cv2.VideoCapture(int(device_index))
    if not cap.isOpened():
        raise RuntimeError(
            f"Failed to open camera device_index={device_index} with OpenCV."
        )
    try:
        ok, frame_bgr = cap.read()
    finally:
        cap.release()

    if not ok or frame_bgr is None:
        raise RuntimeError("Failed to capture frame with OpenCV camera.")

    return cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)


def capture() -> np.ndarray:
    cam_cfg = get_config().camera
    backend = str(cam_cfg.backend).strip().lower()

    if backend in ("edsdk", "canon_edsdk"):
        return _capture_with_edsdk()
    if backend == "opencv":
        return _capture_with_opencv(cam_cfg.device_index)

    raise ValueError(
        f"Unknown camera backend: '{cam_cfg.backend}'. "
        "Use 'edsdk', 'canon_edsdk', or 'opencv'."
    )


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
