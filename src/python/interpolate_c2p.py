# coding: utf-8

import os
import sys

import cv2
import numpy as np
from scipy.sparse.linalg import LinearOperator, cg


def load_c2p_numpy(
    map_file_path: str,
) -> np.ndarray:
    """c2pのnpyを読み込んで (N,4) float32 配列で返す。

    互換入力:
      - decode.py 互換: dtype=object の [[x,y],[u,v]] リスト
      - 本スクリプト互換: (N,4) 数値配列 [cam_x,cam_y,proj_x,proj_y]
      - 参考: (H,W,2) 数値配列 [proj_x,proj_y]
    """
    map_data = np.load(map_file_path, allow_pickle=True)

    # 既に (N,4) の数値配列
    if isinstance(map_data, np.ndarray) and map_data.dtype != object:
        if map_data.ndim == 2 and map_data.shape[1] == 4:
            return map_data.astype(np.float32, copy=False)

        # (H,W,2) の密マップ
        if map_data.ndim == 3 and map_data.shape[2] == 2:
            h, w, _ = map_data.shape
            out = np.empty((h * w, 4), dtype=np.float32)
            x_coords = np.arange(w, dtype=np.float32)
            for y in range(h):
                row = slice(y * w, (y + 1) * w)
                out[row, 0] = x_coords
                out[row, 1] = np.float32(y)
                out[row, 2] = map_data[y, :, 0].astype(np.float32, copy=False)
                out[row, 3] = map_data[y, :, 1].astype(np.float32, copy=False)
            return out

    # decode.py 互換: dtype=object の配列
    if not (
        isinstance(map_data, np.ndarray)
        and map_data.dtype == object
        and map_data.ndim >= 1
    ):
        raise TypeError("Unsupported c2p numpy format")

    n = int(map_data.shape[0])
    out = np.empty((n, 4), dtype=np.float32)

    # 先頭だけチェック
    for i in range(min(10, n)):
        item = map_data[i]
        if (
            not isinstance(item, (list, tuple))
            or len(item) != 2
            or not isinstance(item[0], (list, tuple))
            or len(item[0]) != 2
            or not isinstance(item[1], (list, tuple))
            or len(item[1]) != 2
        ):
            raise TypeError(f"map_list[{i}] must be [[x,y],[u,v]]")

    for i in range(n):
        cam_xy, proj_uv = map_data[i]
        out[i, 0] = float(cam_xy[0])
        out[i, 1] = float(cam_xy[1])
        out[i, 2] = float(proj_uv[0])
        out[i, 3] = float(proj_uv[1])
    return out


def interpolate_c2p_list(
    cam_height: int,
    cam_width: int,
    c2p_list: np.ndarray,
) -> np.ndarray:
    """
    Fill missing correspondences using Laplacian interpolation.
    Matrix-free iterative solve (CG) to reduce memory usage.
    """
    # Initialize maps with NaN
    proj_x_map = np.full((cam_height, cam_width), np.nan, dtype=np.float64)
    proj_y_map = np.full((cam_height, cam_width), np.nan, dtype=np.float64)

    # Fill known correspondences (vectorized)
    if not (
        isinstance(c2p_list, np.ndarray)
        and c2p_list.ndim == 2
        and c2p_list.shape[1] == 4
    ):
        raise TypeError("c2p_list must be a NumPy array with shape (N,4)")

    cam_x = c2p_list[:, 0]
    cam_y = c2p_list[:, 1]
    proj_x = c2p_list[:, 2]
    proj_y = c2p_list[:, 3]

    ix = cam_x.astype(np.int32)
    iy = cam_y.astype(np.int32)
    valid = (
        (0 <= ix)
        & (ix < cam_width)
        & (0 <= iy)
        & (iy < cam_height)
        & ~np.isnan(proj_x)
        & ~np.isnan(proj_y)
    )

    ix_v = ix[valid]
    iy_v = iy[valid]
    proj_x_map[iy_v, ix_v] = proj_x[valid].astype(np.float64, copy=False)
    proj_y_map[iy_v, ix_v] = proj_y[valid].astype(np.float64, copy=False)

    # Solve for each channel
    proj_x_filled = _laplacian_fill(proj_x_map)
    proj_y_filled = _laplacian_fill(proj_y_map)

    # Build output as (N,4) float32 to avoid huge Python object overhead
    out = np.empty((cam_height * cam_width, 4), dtype=np.float32)
    x_coords = np.arange(cam_width, dtype=np.float32)
    for y in range(cam_height):
        row = slice(y * cam_width, (y + 1) * cam_width)
        out[row, 0] = x_coords
        out[row, 1] = np.float32(y)
        out[row, 2] = proj_x_filled[y, :].astype(np.float32, copy=False)
        out[row, 3] = proj_y_filled[y, :].astype(np.float32, copy=False)

    return out


def _laplacian_fill(data: np.ndarray) -> np.ndarray:
    """
    Fill NaN regions by solving Laplace equation.
    Known pixels act as Dirichlet boundary conditions.
    """
    h, w = data.shape
    n_pixels = h * w

    mask = np.isnan(data)
    if not np.any(mask):
        return data.copy()

    known_mask = ~mask
    known_flat = known_mask.ravel()

    # RHS (known: value, unknown: 0)
    rhs = np.zeros(n_pixels, dtype=np.float64)
    rhs[known_flat] = data.ravel()[known_flat]

    # Initial guess
    x0 = np.zeros(n_pixels, dtype=np.float64)
    x0[known_flat] = rhs[known_flat]

    # Degree (number of 4-neighbors): 2/3/4
    deg = np.full((h, w), 4.0, dtype=np.float32)
    deg[0, :] -= 1.0
    deg[-1, :] -= 1.0
    deg[:, 0] -= 1.0
    deg[:, -1] -= 1.0

    class _DirichletLaplacian(LinearOperator):
        def __init__(self, known: np.ndarray, degree: np.ndarray):
            self._known = known
            self._deg = degree.astype(np.float64, copy=False)
            self._h, self._w = known.shape
            n = self._h * self._w

            # SciPy のバージョン/スタブ差分に備えて両方試す
            try:
                super().__init__(dtype=np.float64, shape=(n, n))
            except TypeError:
                super().__init__((n, n), np.float64)

        def _matvec(self, x: np.ndarray) -> np.ndarray:
            x2 = x.reshape(self._h, self._w)
            y = self._deg * x2
            y[:-1, :] -= x2[1:, :]
            y[1:, :] -= x2[:-1, :]
            y[:, :-1] -= x2[:, 1:]
            y[:, 1:] -= x2[:, :-1]
            y[self._known] = x2[self._known]
            return y.ravel()

    Aop = _DirichletLaplacian(known_mask, deg)
    sol, info = cg(Aop, rhs, x0=x0, rtol=1e-6, atol=0.0, maxiter=2000)
    if info != 0:
        print(f"[warn] Laplacian CG did not fully converge (info={info}).")

    result = sol.reshape(h, w)
    result[known_mask] = data[known_mask]
    return result


def create_vis_image(
    cam_height: int,
    cam_width: int,
    c2p_list_interp: np.ndarray,
    dtype: np.dtype = np.dtype(np.uint8),
) -> np.ndarray:
    if not (
        isinstance(c2p_list_interp, np.ndarray)
        and c2p_list_interp.ndim == 2
        and c2p_list_interp.shape[1] == 4
    ):
        raise TypeError("c2p_list_interp must be a NumPy array with shape (N,4)")

    arr = c2p_list_interp.astype(np.float32, copy=False)

    cam_x = arr[:, 0]
    cam_y = arr[:, 1]
    proj_x = arr[:, 2]
    proj_y = arr[:, 3]

    ix = np.rint(cam_x).astype(np.int32)  # round
    iy = np.rint(cam_y).astype(np.int32)

    # 有効な点だけをマスク
    valid = (
        (0 <= ix)
        & (ix < cam_width)
        & (0 <= iy)
        & (iy < cam_height)
        & ~np.isnan(proj_x)
        & ~np.isnan(proj_y)
    )

    ix_v = ix[valid]
    iy_v = iy[valid]
    proj_x_v = proj_x[valid]
    proj_y_v = proj_y[valid]

    vis_image = np.zeros((cam_height, cam_width, 3), dtype=dtype)

    # 色をベクトル化して計算
    r = (proj_x_v).astype(np.int32) % (np.iinfo(dtype).max + 1)
    g = (proj_y_v).astype(np.int32) % (np.iinfo(dtype).max + 1)
    b = 128 * np.ones_like(r, dtype=np.int32)

    vis_image[iy_v, ix_v, 0] = r.astype(dtype)
    vis_image[iy_v, ix_v, 1] = g.astype(dtype)
    vis_image[iy_v, ix_v, 2] = b.astype(dtype)

    return vis_image


def main(argv: list[str] | None = None) -> None:
    if argv is None:
        argv = sys.argv

    if len(argv) != 4:
        print(
            "Usage : python interpolate_c2p.py <c2p_numpy_filename> <cam_height> <cam_width>"
        )
        print()
        return

    try:
        cam_height = int(argv[2])
        cam_width = int(argv[3])
        c2p_numpy_filename = str(argv[1])
    except ValueError:
        print("cam_height, cam_width は整数で指定してください。")
        print(
            "Usage : python interpolate_c2p.py <c2p_numpy_filename> <cam_height> <cam_width>"
        )
        print()
        return

    try:
        c2p_list = load_c2p_numpy(c2p_numpy_filename)
    except Exception as e:
        print(f"Error loading c2p numpy file: {e}")
        return

    print(
        f"Loaded {len(c2p_list)} camera-to-projector correspondences from '{c2p_numpy_filename}'"
    )
    c2p_list_interp = interpolate_c2p_list(cam_height, cam_width, c2p_list)

    # create image for visualization
    vis_image = create_vis_image(
        cam_height, cam_width, c2p_list_interp, dtype=np.dtype(np.uint8)
    )
    vis_filename = os.path.splitext(c2p_numpy_filename)[0] + "_compensated_vis.png"
    cv2.imwrite(vis_filename, vis_image)
    print(f"Saved visualization image to '{vis_filename}'")

    out_filename = os.path.splitext(c2p_numpy_filename)[0] + "_compensated.npy"
    np.save(out_filename, c2p_list_interp.astype(np.float32, copy=False))
    print(f"Saved compensated correspondences to '{out_filename}' (float32 Nx4)")

    with open("result_c2p_compensated.csv", "w", encoding="utf-8") as f:
        f.write("cam_x, cam_y, proj_x, proj_y\n")
        for cam_x, cam_y, proj_x, proj_y in c2p_list_interp:
            f.write(
                f"{float(cam_x)}, {float(cam_y)}, {float(proj_x)}, {float(proj_y)}\n"
            )
    print("output : './result_c2p_compensated.csv'")
    print()


if __name__ == "__main__":
    main()
