# coding: utf-8
"""ピクセル座標変換の単一の真実源 (single source of truth)。

このプロジェクトは **2 つの異なるピクセル中心規約** を併用する。両者を取り違える
と半画素 (0.5px) のズレが連鎖的に混入するため、座標↔画素インデックスの変換は
すべてこのモジュールの関数を経由すること。各関数の docstring が「正準定義」であり、
他モジュール (decode / interpolate_c2p / interpolate_p2c / warp_image) はこれを
再実装せず必ず呼び出す。規約の全体像と数式は ``COORDINATES.md`` を参照。

────────────────────────────────────────────────────────────────────
規約のまとめ
────────────────────────────────────────────────────────────────────
カメラ (XY, ソース) : 画素 i の **中心は整数 i**。画素 i は [i-0.5, i+0.5)。
    画素インデックス = round(x) = floor(x + 0.5)。
    (numpy/OpenCV の配列標本化と同じ。``np.where`` が返す整数がそのまま中心。)

プロジェクタ (UV, デスティネーション) : 画素 i の **中心は i + 0.5**。
    画素 i は [i, i+1)。画素インデックス = floor(u)。
    (GPU テクスチャ / grid_sample と同じ。decode の +0.5 オフセットもこれ。)

GrayCode の step (ブロック) : step=s のとき、縮小グレイコード座標 g 1 つが
    s×s の全解像度画素ブロックを表す。代表座標はブロック中心 = s*(g+0.5) (UV)。
    s が大きいほど中心の間隔と基準オフセットが s に比例して変わる点に注意。
"""

from __future__ import annotations

import numpy as np

__all__ = [
    "xy_pixel_centers",
    "xy_to_pixel",
    "uv_pixel_centers",
    "uv_to_pixel",
    "uv_to_array",
    "array_to_uv",
    "uv_to_normalized",
    "reduced_size",
    "block_of",
    "block_center_uv",
]


def _floor(a):
    """backend 非依存の floor。numpy 配列/スカラ と torch テンソルの双方に対応。"""
    floor = getattr(a, "floor", None)
    if callable(floor):  # torch.Tensor は .floor() を持つ
        return a.floor()
    return np.floor(a)


# ── カメラ (XY): 画素中心 = 整数 ──────────────────────────────────────


def xy_pixel_centers(n: int) -> np.ndarray:
    """全 ``n`` 画素の中心 XY 座標 (= 0, 1, ..., n-1)。

    カメラでは画素 i の中心が整数 i なので、画素中心の座標列は arange そのもの。
    補間のクエリ格子 (カメラ全画素) を作るのに使う。
    """
    return np.arange(n, dtype=np.float64)


def xy_to_pixel(x):
    """連続 XY 座標 → 画素インデックス (float, 未キャスト)。

    画素中心 = 整数なので最近傍画素は round(x) = floor(x + 0.5)。
    返り値は float (numpy なら ``.astype(np.int32)``、torch なら ``.long()`` で
    呼び出し側がキャストする)。入力 dtype/backend をそのまま保つ。
    """
    return _floor(x + 0.5)


# ── プロジェクタ (UV): 画素中心 = 整数 + 0.5 ─────────────────────────


def uv_pixel_centers(n: int) -> np.ndarray:
    """全 ``n`` 画素の中心 UV 座標 (= 0.5, 1.5, ..., n-0.5)。

    プロジェクタでは画素 i の中心が i + 0.5。補間のクエリ格子 (プロジェクタ
    全画素) を画素中心で作るのに使う。
    """
    return np.arange(n, dtype=np.float64) + 0.5


def uv_to_pixel(u):
    """連続 UV 座標 → 画素インデックス (float, 未キャスト)。

    画素 i が [i, i+1) を占めるので、座標 u を含む画素は floor(u)。
    """
    return _floor(u)


def uv_to_array(u):
    """UV 座標 → 「配列インデックス座標」(= u - 0.5)。

    配列インデックス座標とは「整数 = 画素中心」となる座標系 (XY と同じ規約)。
    UV では中心が i+0.5 なので 0.5 引くと配列インデックス上の位置になる。
    双線形スプラッティングや中心基準の重み計算で使う
    (例: forward warp の bilinear splat)。
    """
    return u - 0.5


def array_to_uv(a):
    """「配列インデックス座標」→ UV 座標 (= a + 0.5)。:func:`uv_to_array` の逆。"""
    return a + 0.5


def uv_to_normalized(u, size: int):
    """UV 座標 → ``F.grid_sample(align_corners=False)`` 用の正規化座標 [-1, 1]。

    align_corners=False では正規化座標 g と画素中心座標 p が
        p = ((g + 1) * size - 1) / 2
    で対応する。UV 座標 u (中心 = i+0.5) は配列上の画素中心座標 u-0.5 に当たるので
        g = 2*u/size - 1
    が UV を直接正規化する式になる (導出は COORDINATES.md)。
    """
    return 2.0 * u / size - 1.0


# ── GrayCode の step (ブロック / 縮小解像度) ─────────────────────────


def reduced_size(full: int, step: int) -> int:
    """全解像度 ``full`` を step で縮小したグレイコード解像度。

    ``(full - 1) // step + 1``。step 画素ごとに 1 つのグレイコード値を割り当てる
    ときに必要なブロック数。gen_graycode と decode が同じ値を使う必要がある。
    """
    return (full - 1) // step + 1


def block_of(pixel, step: int):
    """全解像度の画素インデックス → 所属するグレイコードブロック (= pixel // step)。

    パターン生成時のブロック展開 ``img[y, x] = pat[y // hs, x // ws]`` に対応。
    """
    return pixel // step


def block_center_uv(g, step: int):
    """縮小グレイコード座標 ``g`` → ブロック中心の UV 座標 (= step*(g+0.5))。

    decode が記録するプロジェクタ座標 (GT) の定義。ブロック g は UV 範囲
    [step*g, step*g+step) を占め、その中心が step*(g+0.5)。

    注意 (step によるシフト): step=1 では g+0.5 で画素 g の中心に一致するが、
    step>1 では「画素」ではなく step 幅の **ブロック** の中心であり、隣接ブロック
    中心の間隔は step、基準オフセットは step/2 になる。ブロック中心が単一画素の
    中心 (i+0.5) に一致するのは step が奇数のとき (i = step*g + (step-1)/2) のみ。
    """
    return step * (g + 0.5)
