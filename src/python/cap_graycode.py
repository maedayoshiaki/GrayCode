import cv2
import numpy as np
import glob
import sys
from pathlib import Path
from edsdk.camera_controller import CameraController
from typing import List

TARGETDIR = Path("data/graycode_pattern")


def open_cam() -> None:
    pass


def close_cam() -> None:
    pass


def capture() -> np.ndarray:
    with CameraController(register_property_events=False) as camera:
        camera.set_properties(av=5, tv=1 / 15, iso=100, image_quality="LJF")
        imgs = camera.capture_numpy()
        img = imgs[0]
    return img


def print_usage() -> None:
    print("Usage : python cap_graycode.py")
    print()


def main(argv: List[str] | None) -> None:
    if argv is None:
        argv = sys.argv

    if len(argv) != 3:
        print_usage()
        return

    try:
        proj_x = int(argv[1])
        proj_y = int(argv[2])
    except ValueError:
        print("Error: Arguments must be integers.")
        print_usage()
        return

    print(f"Arguments: {argv}")

    graycode_imgs = []
    # グレイコードをファイルから参照
    for idx, fname in enumerate(sorted(glob.glob(str(TARGETDIR / "pattern_*.png")))):
        print(f"Loading pattern image: {fname}")
        pat_img = cv2.imread(fname, cv2.IMREAD_GRAYSCALE)
        graycode_imgs.append(pat_img)

    cv2.namedWindow("Pattern", cv2.WINDOW_NORMAL)
    cv2.setWindowProperty("Pattern", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
    cv2.resizeWindow("Pattern", proj_x, proj_y)

    for i, pat in enumerate(graycode_imgs):
        print(f"Displaying pattern image {i:02d}...")
        cv2.imshow("Pattern", pat)
        cv2.waitKey(500)  # 0.5秒待機してからキャプチャ
        captured_img = capture()
        cv2.imwrite(f"./captured/capture_{i:02d}.png", captured_img)
