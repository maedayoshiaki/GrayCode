# 座標系とピクセルの扱い

このプロジェクトは内部で **単一のピクセル規約 "pixel-is-point"** を用いる：

$$\boxed{\text{整数 } i = \text{画素 } i \text{ の中心}}$$

カメラ・プロジェクタの**両方**でこの規約に統一されている。連続座標と整数ピクセル
インデックスの対応はすべて単一の真実源 [`src/graycode/coords.py`](src/graycode/coords.py)
の関数を経由する（各所で再実装しない）。

唯一の例外は GPU サンプリング (`torch.nn.functional.grid_sample`) との境界で、そこ
だけ「テクセル規約（中心 = $i+0.5$）」への半画素変換が必要になる。その変換は
[`coords.point_to_normalized`](src/graycode/coords.py) 1 か所に閉じ込めてある。

> **2026-06 に旧版（プロジェクタのみ中心 $i+0.5$ の二重規約）から本規約へ統一した。**
> 旧データの移行は §8 を参照。背景の業界調査は §7。

---

## 0. 記号

- プロジェクタ全解像度: 高さ $H$、幅 $W$。カメラ: $H_c$、$W_c$。
- 連続座標: $(x, y)$（カメラ）, $(u, v)$（プロジェクタ）。画素インデックス: 整数 $i, j$。
- GrayCode のステップ（ブロック幅）: $s$、縮小グレイコード座標: $g$。

---

## 1. 規約（pixel-is-point）

$$\text{画素 } i \text{ の中心} = i, \qquad \text{画素 } i \text{ の範囲} = [\,i-\tfrac12,\; i+\tfrac12\,)$$

$$\text{画素インデックス} = \operatorname{round}(c) = \big\lfloor c + \tfrac12 \big\rfloor \qquad (\texttt{coords.to\_pixel})$$

$$\text{全 } n \text{ 画素の中心座標} = \{0, 1, \dots, n-1\} = \texttt{arange}(n) \qquad (\texttt{coords.pixel\_centers})$$

```
 index :      0       1       2       3
 center:      0       1       2       3
 coord : --|--o--|--o--|--o--|--o--|--
        -0.5  0 0.5  1 1.5  2 2.5  3 3.5
```

カメラ・プロジェクタとも同一。`np.where` が返す整数、`cv2.imread` の配列インデックス、
matplotlib/scipy の標本点—すべてこの規約と一致する。

---

## 2. GrayCode の step（ブロック）とブロック中心

ステップ $s$ では、1 つの縮小グレイコード座標 $g$ が $s$ 画素の **ブロック** を表す。

$$\text{縮小解像度} = \Big\lfloor \tfrac{N-1}{s} \Big\rfloor + 1 \qquad (\texttt{coords.reduced\_size})$$

$$\text{パターン拡大: } \text{img}[y,x] = \text{pat}\big[\lfloor y/s_h\rfloor,\ \lfloor x/s_w\rfloor\big] \qquad (\texttt{coords.block\_of})$$

decode が記録するプロジェクタ座標（GT）は **ブロック中心（pixel-is-point）**：

$$p = s\,g + \tfrac{s-1}{2} \qquad (\texttt{coords.block\_center})$$

ブロック $g$ は全解像度画素 $[s g,\; s g+s-1]$ を覆い、その中心が $s g + \tfrac{s-1}{2}$。

| $s$ | ブロック中心 $p$ | 値 | 備考 |
|---|---|---|---|
| 1 | $g$ | 整数 | 画素 $g$ の中心（デコード値そのもの） |
| 2 | $2g + 0.5$ | 半整数 | 隣り合う2画素中心の中間 |
| 3 | $3g + 1$ | 整数 | ブロック中央画素の中心 |
| $s$ | $s g + \tfrac{s-1}{2}$ | $s$ 奇数で整数 | 一様照射ブロックの幾何中心（正しい点推定） |

> **step 注意**: $s>1$ では $p$ は単一画素ではなく $s$ 幅ブロックの中心であり、隣接中心
> の間隔は $s$。半整数になっても pixel-is-point 座標として正しい（サブ画素位置）。
> 補間（§4）は全画素中心 $\{0,1,\dots\}$ でクエリし、ブロック中心の間を線形補間する。
> $s=1$（既定）では復号画素で GT が**厳密に再現**される。

---

## 3. grid_sample 境界（唯一のテクセル規約）

`grid_sample(align_corners=False)` は内部でテクセル規約（画素 $i$ の中心 $= i+0.5$）を
用い、正規化座標 $g\in[-1,1]$ と配列画素中心座標 $q$ を

$$q = \frac{(g+1)\,\text{size} - 1}{2}$$

で対応づける。pixel-is-point 座標 $p$ はテクセル座標で $p+0.5$ に当たるので、$q = p+0.5$
を解いて

$$g = \frac{2(p + 0.5)}{\text{size}} - 1 \qquad (\texttt{coords.point\_to\_normalized})$$

**この $+0.5$ が pixel-is-point ↔ テクセル規約の唯一の境界変換**であり、`warp_image` の
backward sampling でのみ現れる。

---

## 4. モジュール・成果物ごとの扱い（すべて pixel-is-point）

| 場所 | カメラ列 | プロジェクタ列 | 画素化 |
|---|---|---|---|
| `decode` `result_c2p` / `result_p2c` | `cam_x, cam_y`（整数中心） | `proj` = `block_center`（$s{=}1$ で整数） | — |
| `interpolate_c2p` | クエリ格子 = `pixel_centers` | 出力 `proj` = 補間値 | `to_pixel` |
| `interpolate_p2c` | 出力 `cam` = 補間値 | クエリ格子 = `pixel_centers`（整数） | `to_pixel` |
| `warp` forward | src 標本: `to_pixel` | dst splat: nearest=`to_pixel`, bilinear=`floor(u)`(整数中心) | — |
| `warp` backward | dst 配置: `to_pixel` | サンプリング: `point_to_normalized`（唯一の +0.5） | — |

`PixelMapWarperTorch` のマップは `(N,4) = [src_x, src_y, dst_x, dst_y]`。src=カメラ、
dst=プロジェクタで、**両方とも pixel-is-point**（"XY"/"UV" は空間ラベルであり中心規約の
違いではない）。`interpolate_c2p` の出力 `[cam, cam, proj, proj]` はそのまま warp マップ
として使える。`interpolate_p2c` の出力 `[proj, proj, cam, cam]` は列順が逆。

---

## 5. `coords.py` API（正準定義）

| 関数 | 式 | 用途 |
|---|---|---|
| `pixel_centers(n)` | $\{0,\dots,n-1\}$ | 画素中心の座標列（補間クエリ格子） |
| `to_pixel(c)` | $\lfloor c+0.5\rfloor$ | 連続座標→画素インデックス（round） |
| `point_to_normalized(p, size)` | $2(p+0.5)/\text{size}-1$ | grid_sample 境界（唯一の +0.5） |
| `reduced_size(full, step)` | $\lfloor(full-1)/step\rfloor+1$ | 縮小グレイコード解像度 |
| `block_of(pixel, step)` | $\lfloor pixel/step\rfloor$ | 画素→所属ブロック |
| `block_center(g, step)` | $step\cdot g+(step-1)/2$ | ブロック中心（decode GT） |

`to_pixel` / `point_to_normalized` は numpy 配列・スカラと torch テンソルの双方で動作する
（backend は入力から自動判別）。

---

## 6. データファイルの列定義

| ファイル | 列 |
|---|---|
| `result_c2p.csv` | `cam_x, cam_y, proj_x, proj_y` |
| `result_p2c.csv` | `proj_x, proj_y, cam_x, cam_y` |
| `result_c2p_compensated_{method}.npy` | `(N,2,2)` object: `[(cam_x,cam_y),(proj_x,proj_y)]` |
| `result_p2c_compensated_delaunay.npy` | `(N,4)` float32: `[proj_x,proj_y,cam_x,cam_y]` |

すべて pixel-is-point（整数=画素中心）。proj は $s=1$ で整数、$s>1$ ではブロック中心。

---

## 7. 業界調査（なぜ pixel-is-point に統一したか）

ピクセル中心の置き方には業界に **2 大規約** が実在する（いずれも一次情報で確認）。

| 規約 | 標準呼称 | 中心 | 主な分野 |
|---|---|---|---|
| **pixel-is-point** | integer-center / GeoTIFF "PixelIsPoint" / grid_sample `align_corners=True` | $i$ | コンピュータビジョン、配列標本化、科学画像 |
| **pixel-is-area** | texel-center / GeoTIFF "PixelIsArea"（規格既定） / `align_corners=False` | $i+0.5$ | レンダリング、テクスチャ、SfM |

- **integer-center**: OpenCV（remap/warpAffine/getRectSubPix, [issue #10130](https://github.com/opencv/opencv/issues/10130)「0,0=左上画素の中心」）、matplotlib imshow、scipy.ndimage、scikit-image。
- **half-integer-center**: OpenGL/Vulkan/Direct3D 10+（"Texel # = floor(U)"）、PBRT（"c = d + 1/2"）、COLMAP（"upper-left pixel center = (0.5,0.5)"）、`grid_sample` 既定。
- **構造化光校正の成熟系は integer-center 単一規約**: OpenCV `structured_light`（`getProjPixel` は整数 Point を返す）、Moreno & Taubin 2012（「projectors are modeled as inverse cameras」）、[kamino410/procam-calibration](https://github.com/kamino410/procam-calibration)（`gc_step * proj_pix`、**+0.5 を付けない**）。

本リポジトリは OpenCV ベースの**校正**ツールであり、ドメインの常道（integer-center、
decode に +0.5 を付けない）に合わせて **pixel-is-point に統一**した。テクセル規約は
本来 grid_sample に渡す瞬間だけ必要なので §3 の 1 か所に局所化した。これは
GDAL/matplotlib と同じ「単一規約 + 境界変換」パターンであり、同一パイプライン内で 2
規約を混在させた歴史的失敗例（Direct3D 9 の "half-texel offset"）を避ける。

---

## 8. 旧データの移行

旧版（プロジェクタのみ中心 $i+0.5$）の保存ファイルは、新規約では **プロジェクタ座標が
0.5 ずれる**（カメラ座標は不変）。差は step によらず常に 0.5：

$$\underbrace{s\,(g+\tfrac12)}_{\text{旧 UV 中心}} - \underbrace{\big(s\,g+\tfrac{s-1}{2}\big)}_{\text{新 pixel-is-point 中心}} = \tfrac12$$

**推奨は再生成**（decode / interpolate を再実行）。既存結果を使い続ける場合は移行
スクリプトで proj 座標から 0.5 を引く（非破壊・`<stem>_pixelpoint` を出力）：

```bash
uv run python -m graycode.migrate result_c2p.npy result_p2c.npy \
    result_p2c_compensated_delaunay.npy result_c2p.csv result_p2c.csv
```

対応形式: c2p `(N,2,2)` object / p2c dict / p2c_compensated `(N,4)` / 各 CSV（ヘッダの
`proj_x`,`proj_y` 列を自動検出）。実装は [`src/graycode/migrate.py`](src/graycode/migrate.py)。

---

## 9. よくある間違い（アンチパターン）

| 誤 | 正 |
|---|---|
| 座標を `int()`（0 方向切り捨て）で画素化 | `to_pixel`（= `floor(c+0.5)` = round） |
| forward bilinear や補間で勝手に ±0.5 を足す | `coords` の関数のみを使う（+0.5 は grid_sample 境界だけ） |
| プロジェクタだけ別規約で扱う | カメラ・プロジェクタとも pixel-is-point |
| gen と decode で step 引数順を変える | 両者 `(height_step, width_step)` |
| grid_sample に pixel-is-point 座標を直接渡す | `point_to_normalized` で変換してから渡す |
