# coding: utf-8
"""グレイコード画素対応の可視化ツール。

生成する図:

1. 対応グラデーション (correspondence gradient) — 各 view × {raw, filled}
   - **projector-view**: プロジェクタ画素を 対応カメラ座標 (cam_x/cam_y) で濃淡着色。
   - **camera-view**:    カメラ画素を 対応プロジェクタ座標 (proj_x/proj_y) で濃淡着色。
   - **raw**:    生のデコード結果を散布 (穴埋めなし)。デコードできなかった画素は黒(穴)。
   - **filled**: ドロネー線形補間 + 凸包外最近傍で穴埋め (滑らかなグラデーション)。
   各 view×mode で grayscale(濃淡, 大きい座標ほど明るい) / JET / 合成RGB(R=X,G=Y) を出力。
2. 投影範囲マスク (valid mask): 白投影 - 黒投影 > 閾値。--white/--black または --debug-dir。
3. 撮影グレイコードのモンタージュ: --captures または --debug-dir の capture_*.png をタイル表示。

入力対応ファイル:
  --c2p <file>  : graycode 生 c2p .npy (object (N,2,2))。raw/filled の両方を作れる。
  --p2c <file>  : .npz(2dsr-prc) または graycode p2c .npy。densify 済みなら filled のみ。
  --debug-dir <dir> : run_calibration(debug_dir=...) が保存した
                      result_c2p.npy / capture_*.png / white.png / black.png / valid_mask.png
                      をまとめて読む(個別指定より簡単)。

使用例:
  uv run python scripts/visualize_graycode.py \
      --debug-dir ../2dsr-prc/output/m7_gc_vis/gc_debug \
      --p2c ../2dsr-prc/output/m7_gc_vis/p2c.npz --out output/gc_vis
"""

from __future__ import annotations

import argparse
import glob
import os
import re
import sys
from pathlib import Path

import cv2
import numpy as np


# ── 読み込み(すべて (N,4) に正規化) ───────────────────────────────────


def load_p2c(path: str) -> tuple[np.ndarray, tuple[int, int] | None]:
    """p2c を (N,4)[proj_x, proj_y, cam_x, cam_y] と (proj_h, proj_w) で返す。"""
    p = Path(path)
    if p.suffix == ".npz":
        with np.load(path, allow_pickle=True) as d:
            arr = np.asarray(d["p2c"], dtype=np.float64).reshape(-1, 4)
            proj_size = (
                tuple(int(v) for v in d["proj_size"]) if "proj_size" in d.files else None
            )
        return arr, proj_size
    from graycode.interpolate_p2c import load_p2c_numpy_array

    return np.asarray(load_p2c_numpy_array(path), dtype=np.float64).reshape(-1, 4), None


def load_c2p(path: str) -> np.ndarray:
    """c2p を (N,4)[cam_x, cam_y, proj_x, proj_y] で返す。"""
    from graycode.interpolate_c2p import load_c2p_numpy_array

    return np.asarray(load_c2p_numpy_array(path), dtype=np.float64).reshape(-1, 4)


# ── 描画プリミティブ ──────────────────────────────────────────────────


def _norm_u8(values: np.ndarray, vmin: float, vmax: float) -> np.ndarray:
    if vmax - vmin < 1e-9:
        return np.zeros_like(values, dtype=np.uint8)
    return np.clip((values - vmin) / (vmax - vmin) * 255.0, 0, 255).astype(np.uint8)


def _colorize(gray_u8: np.ndarray, covered: np.ndarray) -> np.ndarray:
    col = cv2.applyColorMap(gray_u8, cv2.COLORMAP_JET)
    col[~covered] = (0, 0, 0)
    return col


def _write_channel_set(
    img_x_u8: np.ndarray,
    img_y_u8: np.ndarray,
    covered: np.ndarray,
    out_dir: Path,
    prefix: str,
    xl: str,
    yl: str,
    written: list[Path],
) -> None:
    rgb = np.zeros((*covered.shape, 3), dtype=np.uint8)
    rgb[..., 2] = img_x_u8  # BGR: R=X
    rgb[..., 1] = img_y_u8  # G=Y
    rgb[..., 0] = np.where(covered, 128, 0).astype(np.uint8)
    items = {
        f"{prefix}_{xl}_gray.png": img_x_u8,
        f"{prefix}_{yl}_gray.png": img_y_u8,
        f"{prefix}_{xl}_jet.png": _colorize(img_x_u8, covered),
        f"{prefix}_{yl}_jet.png": _colorize(img_y_u8, covered),
        f"{prefix}_rgb.png": rgb,
        f"{prefix}_coverage.png": (covered * 255).astype(np.uint8),
    }
    for name, im in items.items():
        path = out_dir / name
        cv2.imwrite(str(path), im)
        written.append(path)


def render_raw(
    grid_h: int,
    grid_w: int,
    pix_x: np.ndarray,
    pix_y: np.ndarray,
    val_x: np.ndarray,
    val_y: np.ndarray,
    out_dir: Path,
    prefix: str,
    xl: str,
    yl: str,
    written: list[Path],
) -> None:
    """生(散布)。デコードされた画素のみ着色、残りは黒(穴)。"""
    ix = np.rint(pix_x).astype(np.int64)
    iy = np.rint(pix_y).astype(np.int64)
    inside = (ix >= 0) & (ix < grid_w) & (iy >= 0) & (iy < grid_h)
    vx_u8 = _norm_u8(val_x, float(val_x.min()), float(val_x.max()))
    vy_u8 = _norm_u8(val_y, float(val_y.min()), float(val_y.max()))
    img_x = np.zeros((grid_h, grid_w), np.uint8)
    img_y = np.zeros((grid_h, grid_w), np.uint8)
    cov = np.zeros((grid_h, grid_w), bool)
    img_x[iy[inside], ix[inside]] = vx_u8[inside]
    img_y[iy[inside], ix[inside]] = vy_u8[inside]
    cov[iy[inside], ix[inside]] = True
    _write_channel_set(img_x, img_y, cov, out_dir, prefix, xl, yl, written)


def render_filled(
    grid_h: int,
    grid_w: int,
    points: np.ndarray,
    values: np.ndarray,
    out_dir: Path,
    prefix: str,
    xl: str,
    yl: str,
    written: list[Path],
) -> None:
    """穴埋め(ドロネー線形 + 凸包外最近傍)で全画素を補間して着色。"""
    from scipy.interpolate import LinearNDInterpolator, NearestNDInterpolator

    gy, gx = np.mgrid[0:grid_h, 0:grid_w]
    q = np.stack([gx.ravel(), gy.ravel()], axis=1).astype(float)
    out = LinearNDInterpolator(points, values)(q)
    nan = np.isnan(out[:, 0])
    if nan.any():
        out[nan] = NearestNDInterpolator(points, values)(q[nan])
    vx, vy = values[:, 0], values[:, 1]
    img_x = _norm_u8(out[:, 0], float(vx.min()), float(vx.max())).reshape(grid_h, grid_w)
    img_y = _norm_u8(out[:, 1], float(vy.min()), float(vy.max())).reshape(grid_h, grid_w)
    cov = np.ones((grid_h, grid_w), bool)
    _write_channel_set(img_x, img_y, cov, out_dir, prefix, xl, yl, written)


# ── valid マスク・撮影モンタージュ ──────────────────────────────────


def render_valid_mask(white_path: str, black_path: str, threshold: int, out_path: Path) -> None:
    white = cv2.imread(white_path, cv2.IMREAD_GRAYSCALE).astype(np.int16)
    black = cv2.imread(black_path, cv2.IMREAD_GRAYSCALE).astype(np.int16)
    mask = ((white - black) > threshold).astype(np.uint8) * 255
    cv2.imwrite(str(out_path), mask)


def render_captures_montage(captures_dir: str, out_path: Path, cols: int = 8) -> bool:
    re_num = re.compile(r"(\d+)")
    files = sorted(
        glob.glob(os.path.join(captures_dir, "capture_*.png")),
        key=lambda t: int(re_num.findall(t)[-1]) if re_num.findall(t) else 0,
    )
    thumbs = []
    th, tw = 160, 240
    for f in files:
        im = cv2.imread(f, cv2.IMREAD_GRAYSCALE)
        if im is not None:
            thumbs.append(cv2.resize(im, (tw, th)))
    if not thumbs:
        return False
    rows = (len(thumbs) + cols - 1) // cols
    sheet = np.zeros((rows * th, cols * tw), dtype=np.uint8)
    for i, t in enumerate(thumbs):
        r, c = divmod(i, cols)
        sheet[r * th : (r + 1) * th, c * tw : (c + 1) * tw] = t
    cv2.imwrite(str(out_path), sheet)
    return True


# ── CLI ─────────────────────────────────────────────────────────────


def _cam_size(c2p_or_cam: np.ndarray, valid_mask_path: str | None, override) -> tuple[int, int]:
    if override:
        return override[0], override[1]
    if valid_mask_path and os.path.exists(valid_mask_path):
        m = cv2.imread(valid_mask_path, cv2.IMREAD_GRAYSCALE)
        if m is not None:
            return m.shape[0], m.shape[1]
    cam = c2p_or_cam
    return int(np.ceil(cam[:, 1].max())) + 1, int(np.ceil(cam[:, 0].max())) + 1


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="グレイコード画素対応の可視化")
    ap.add_argument("--c2p", type=str, default=None, help="生 c2p .npy")
    ap.add_argument("--p2c", type=str, default=None, help="p2c .npz / .npy")
    ap.add_argument("--debug-dir", type=str, default=None, help="run_calibration debug_dir")
    ap.add_argument("--out", type=str, default="output/gc_vis", help="出力ディレクトリ")
    ap.add_argument("--proj-size", type=int, nargs=2, default=None, metavar=("H", "W"))
    ap.add_argument("--cam-size", type=int, nargs=2, default=None, metavar=("H", "W"))
    ap.add_argument("--white", type=str, default=None)
    ap.add_argument("--black", type=str, default=None)
    ap.add_argument("--black-threshold", type=int, default=50)
    ap.add_argument("--captures", type=str, default=None, help="capture_*.png のディレクトリ")
    args = ap.parse_args(argv)

    # --debug-dir からデフォルトを補完
    dbg = args.debug_dir
    if dbg:
        if args.c2p is None and os.path.exists(os.path.join(dbg, "result_c2p.npy")):
            args.c2p = os.path.join(dbg, "result_c2p.npy")
        if args.white is None and os.path.exists(os.path.join(dbg, "white.png")):
            args.white = os.path.join(dbg, "white.png")
        if args.black is None and os.path.exists(os.path.join(dbg, "black.png")):
            args.black = os.path.join(dbg, "black.png")
        if args.captures is None:
            args.captures = dbg

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    valid_mask_path = os.path.join(dbg, "valid_mask.png") if dbg else None

    # 生 c2p から: camera-view(raw + filled), projector-view(raw)
    if args.c2p:
        c2p = load_c2p(args.c2p)
        cam, proj = c2p[:, 0:2], c2p[:, 2:4]
        ch, cw = _cam_size(c2p, valid_mask_path, args.cam_size)
        print(f"[c2p] N={len(c2p)}  cam_grid={ch}x{cw}")
        # camera-view: 生(穴あり) と 穴埋め
        render_raw(ch, cw, cam[:, 0], cam[:, 1], proj[:, 0], proj[:, 1],
                   out_dir, "cam_raw", "projX", "projY", written)
        render_filled(ch, cw, cam, proj, out_dir, "cam_filled", "projX", "projY", written)
        # projector-view: 生(穴あり) — どのプロジェクタ画素が実際にデコードされたか
        if args.proj_size:
            ph, pw = args.proj_size
        else:
            ph, pw = int(np.ceil(proj[:, 1].max())) + 1, int(np.ceil(proj[:, 0].max())) + 1
        print(f"[c2p] proj_grid={ph}x{pw}")
        render_raw(ph, pw, proj[:, 0], proj[:, 1], cam[:, 0], cam[:, 1],
                   out_dir, "proj_raw", "camX", "camY", written)

    # p2c(densify 済みなら filled)から: projector-view(filled), camera-view(filled)
    if args.p2c:
        p2c, proj_size = load_p2c(args.p2c)
        proj, cam = p2c[:, 0:2], p2c[:, 2:4]
        if args.proj_size:
            proj_size = (args.proj_size[0], args.proj_size[1])
        if proj_size is None:
            proj_size = (int(proj[:, 1].max()) + 1, int(proj[:, 0].max()) + 1)
        ph, pw = proj_size
        print(f"[p2c] N={len(p2c)}  proj_grid={ph}x{pw}")
        # projector-view filled(dense p2c をそのまま散布=穴なし)
        render_raw(ph, pw, proj[:, 0], proj[:, 1], cam[:, 0], cam[:, 1],
                   out_dir, "proj_filled", "camX", "camY", written)
        # camera-view filled(p2c の cam->proj を densify。c2p が無い場合の代替)
        if not args.c2p:
            ch, cw = _cam_size(cam, valid_mask_path, args.cam_size)
            print(f"[p2c] cam_grid(filled)={ch}x{cw}")
            render_filled(ch, cw, cam, proj, out_dir, "cam_filled", "projX", "projY", written)

    # valid マスク
    if args.white and args.black:
        render_valid_mask(args.white, args.black, args.black_threshold, out_dir / "valid_mask.png")
        written.append(out_dir / "valid_mask.png")
        print("[mask] valid_mask.png")
    elif valid_mask_path and os.path.exists(valid_mask_path):
        cv2.imwrite(str(out_dir / "valid_mask.png"), cv2.imread(valid_mask_path, cv2.IMREAD_GRAYSCALE))
        written.append(out_dir / "valid_mask.png")
        print("[mask] valid_mask.png (copied from debug-dir)")

    # 撮影モンタージュ
    if args.captures and render_captures_montage(args.captures, out_dir / "captures_montage.png"):
        written.append(out_dir / "captures_montage.png")
        print("[captures] captures_montage.png")

    if not written:
        print("入力がありません(--c2p / --p2c / --debug-dir / --white+--black / --captures)。")
        return 2
    print(f"\n{len(written)} 枚を {out_dir} に出力。")
    return 0


if __name__ == "__main__":
    sys.exit(main())
