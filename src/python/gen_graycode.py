# coding: UTF-8

import os
import sys
from pathlib import Path
from typing import List

import cv2
from cv2 import structured_light
import numpy as np

TARGETDIR = Path("data/graycode_pattern")
CAPTUREDDIR = Path("./captured")


def generate_expanded_patterns(height: int, width: int, step: int) -> List[np.ndarray]:
    """GrayCodeパターンを生成し、指定サイズに拡大した画像配列を返す。"""
    gc_height = (height - 1) // step + 1
    gc_width = (width - 1) // step + 1

    graycode = structured_light.GrayCodePattern.create(gc_width, gc_height)
    _, patterns = graycode.generate()

    expanded: List[np.ndarray] = []
    for pat in patterns:
        img = np.zeros((height, width), np.uint8)
        for y in range(height):
            src_y = y // step
            for x in range(width):
                src_x = x // step
                img[y, x] = pat[src_y, src_x]
        expanded.append(img)

    expanded.append(255 * np.ones((height, width), np.uint8))  # white
    expanded.append(np.zeros((height, width), np.uint8))  # black

    return expanded


def save_patterns(patterns: List[np.ndarray], target_dir: Path) -> None:
    """パターン画像を PNG で保存する。"""
    target_dir.mkdir(parents=True, exist_ok=True)
    for i, pat in enumerate(patterns):
        filename = target_dir / f"pattern_{i:02d}.png"
        cv2.imwrite(str(filename), pat)


def print_usage() -> None:
    print(
        "Usage : python gen_graycode.py "
        "<projector image height> <projector image width> "
        "[graycode step(default=1)]"
    )
    print()


def main(argv: list[str] | None = None) -> None:
    if argv is None:
        argv = sys.argv

    if len(argv) not in (3, 4):
        print_usage()
        return

    try:
        height = int(argv[1])
        width = int(argv[2])
        step = int(argv[3]) if len(argv) == 4 else 1
    except ValueError:
        print("height, width, step は整数で指定してください。")
        print_usage()
        return

    patterns = generate_expanded_patterns(height, width, step)
    save_patterns(patterns, TARGETDIR)

    print("=== Result ===")
    print(
        f"'{TARGETDIR}/pattern_00.png ~ "
        f"pattern_{len(patterns) - 1:02d}.png' were generated"
    )
    print()
    print("=== Next step ===")
    print(f"Project patterns and save captured images as '{CAPTUREDDIR}/capture_*.png'")
    print()


if __name__ == "__main__":
    main()
