# 座標系とピクセルの扱い

このプロジェクトは **2 つの異なる「ピクセル中心」規約** を併用する。両者を取り違える
と半画素 (0.5px) のズレが連鎖的に混入する。本書はその規約を数式で定義し、コード上の
単一の真実源 [`src/graycode/coords.py`](src/graycode/coords.py) と対応づける。
座標↔画素インデックスの変換は **必ず `coords.py` の関数を経由する**（各所で再実装しない）。

---

## 0. 記号

- 全解像度のプロジェクタ: 高さ $H$、幅 $W$。カメラ: 高さ $H_c$、幅 $W_c$。
- 連続座標: カメラは $(x, y)$、プロジェクタは $(u, v)$。
- 画素インデックス: 整数 $i, j$。
- GrayCode のステップ（ブロック幅）: $s$（`width_step` / `height_step`）。縮小グレイコード座標: $g$。

---

## 1. 2 つのピクセル中心規約

### 1.1 カメラ（XY, ソース）— 画素中心 = 整数

$$\text{画素 } i \text{ の中心} = i, \qquad \text{画素 } i \text{ の範囲} = [\,i-\tfrac12,\; i+\tfrac12\,)$$

$$\text{画素インデックス} = \operatorname{round}(x) = \big\lfloor x + \tfrac12 \big\rfloor$$

$$\text{全 } n \text{ 画素の中心座標} = \{0, 1, 2, \dots, n-1\} = \texttt{arange}(n)$$

```
 index :      0       1       2       3
 center:      0       1       2       3
 coord : --|--o--|--o--|--o--|--o--|--
        -0.5  0 0.5  1 1.5  2 2.5  3 3.5
```

**理由**: カメラはセンサによる *標本化*。`np.where` が返す整数がそのまま標本点（画素中心）。
numpy / OpenCV の配列インデックスと一致する。

対応する `coords.py`: [`xy_to_pixel(x)`](src/graycode/coords.py) / [`xy_pixel_centers(n)`](src/graycode/coords.py)

### 1.2 プロジェクタ（UV, デスティネーション）— 画素中心 = 整数 + 0.5

$$\text{画素 } i \text{ の中心} = i + \tfrac12, \qquad \text{画素 } i \text{ の範囲} = [\,i,\; i+1\,)$$

$$\text{画素インデックス} = \lfloor u \rfloor$$

$$\text{全 } n \text{ 画素の中心座標} = \{0.5, 1.5, \dots, n-0.5\} = \texttt{arange}(n) + 0.5$$

```
 index :    0       1       2       3
 center:   0.5     1.5     2.5     3.5
 coord : |--o--|--o--|--o--|--o--|
         0 0.5  1 1.5  2 2.5  3 3.5 4
```

**理由**: プロジェクタは画像の *描画 / テクスチャ*。画素は面積を持つセルで、その中心が
$i+0.5$。GPU の `torch.nn.functional.grid_sample`（後述の逆ワープ）もこの規約。
decode が付ける $+0.5$ オフセットもこれに由来する。

対応する `coords.py`: [`uv_to_pixel(u)`](src/graycode/coords.py) / [`uv_pixel_centers(n)`](src/graycode/coords.py) /
[`uv_to_array(u)`](src/graycode/coords.py) / [`uv_to_normalized(u, size)`](src/graycode/coords.py)

### 1.3 2 規約の差は厳密に 0.5

同じ物理点でも、カメラ規約の整数中心とプロジェクタ規約の整数中心は **0.5 ずれる**。
恒等幾何（プロジェクタ画素とカメラ画素が 1:1）のとき、decode は

$$u = x + 0.5 \quad (\text{step}=1)$$

を返す（`tests/test_capture_seam.py` が実測で確認）。この $+0.5$ は誤差ではなく、
**XY→UV の規約変換そのもの** である。

---

## 2. なぜ非対称のままにするのか

「カメラもプロジェクタも整数=中心」に統一することは可能だが、その場合：

- `grid_sample(align_corners=False)` は UV（中心 $i+0.5$）が自然なため、逆ワープの正規化が
  かえって煩雑になる。
- decode 出力（GT）や保存済み `.npy` / `.csv` の意味が変わり、後方互換が壊れる。

カメラ＝**標本化**、プロジェクタ＝**描画** という役割の違いに即した非対称は graphics では
原理的に妥当である。したがって本リポジトリは **規約は 2 つのまま保持し、扱いを `coords.py`
に一元化して明示・強制する** 方針を採る。

---

## 3. GrayCode の step（ブロック）とピクセル中心のシフト

ステップ $s$ を使うと、1 つの縮小グレイコード座標 $g$ が $s$ 画素分の **ブロック** を表す。

### 3.1 縮小解像度とブロック展開

$$\text{縮小解像度} = \Big\lfloor \tfrac{N-1}{s} \Big\rfloor + 1 \qquad (\texttt{coords.reduced\_size})$$

$$\text{パターン拡大: } \text{img}[y, x] = \text{pat}\big[\,\lfloor y/s_h\rfloor,\; \lfloor x/s_w\rfloor\,\big] \qquad (\texttt{coords.block\_of})$$

gen_graycode と decode は **同じ縮小解像度** を使う必要がある（ずれるとデコードが破綻）。
このため step 引数の順序は両者で `(height_step, width_step)` に統一されている。

### 3.2 ブロック中心（= decode が記録する GT 座標, UV）

$$p = s\,(g + \tfrac12) \qquad (\texttt{coords.block\_center\_uv})$$

ブロック $g$ は UV 範囲 $[\,s g,\; s g + s\,)$ を占め、その中心が $s(g+\tfrac12)$。

| $s$ | ブロック $g$ の中心 $p$ | 隣接中心の間隔 | 単一画素中心 $i+0.5$ に一致するか |
|---|---|---|---|
| 1 | $g + 0.5$ | 1 | 常に一致（画素 $g$ の中心） |
| 2 | $2g + 1$ | 2 | しない（画素中心の中間） |
| 3 | $3g + 1.5$ | 3 | する（$i = 3g + 1$） |
| $s$ | $s g + s/2$ | $s$ | $s$ が奇数のときのみ（$i = s g + \tfrac{s-1}{2}$） |

> **注意（step によるシフト）**: $s>1$ では $p$ は「画素」ではなく **ブロック** の中心であり、
> 隣接中心の間隔は $s$、基準オフセットは $s/2$ に比例して変わる。GT 座標がどの画素中心に
> 乗るか（あるいは乗らないか）は $s$ の偶奇に依存する。補間時はこの点に注意する。

### 3.3 補間との関係

P2C 補間は全プロジェクタ画素の **中心** $i+0.5$（`coords.uv_pixel_centers`）でクエリする。

- $s=1$: クエリ点 = ブロック中心 = GT 点。よって復号済み画素では GT が **厳密に再現** される。
- $s>1$: クエリ点（画素中心、間隔 1）はブロック中心（間隔 $s$）の **間** に来るため、
  ブロック中心の間を線形補間した値になる（GT が格子上に逐語では乗らないのは正常）。

---

## 4. モジュール・成果物ごとの規約

| 場所 | カメラ列 (XY) | プロジェクタ列 (UV) | 画素化に使う関数 |
|---|---|---|---|
| `decode` 出力 `result_c2p` | `cam_x, cam_y`（整数中心） | `proj_x, proj_y` = `block_center_uv`（半整数） | — |
| `decode` 出力 `result_p2c` | 値 `cam_x, cam_y`（整数中心） | キー `proj_x, proj_y`（UV ブロック中心） | — |
| `interpolate_c2p` | クエリ格子 = `xy_pixel_centers`（整数） | 出力 `proj` = 補間値（UV） | 既知点配置 `xy_to_pixel` |
| `interpolate_p2c` | 出力 `cam` = 補間値（XY） | クエリ格子 = `uv_pixel_centers`（$i+0.5$） | 可視化 `uv_to_pixel` |
| `warp` forward | src 標本: `xy_to_pixel` | dst: nearest=`uv_to_pixel`, bilinear=`uv_to_array` | — |
| `warp` backward | dst 配置: `xy_to_pixel` | サンプリング: `uv_to_normalized` | — |

`PixelMapWarperTorch` の対応マップは `(N, 4) = [src_x, src_y, dst_x, dst_y] = [XY, XY, UV, UV]`。

- `interpolate_c2p` の出力 `[cam_x, cam_y, proj_x, proj_y] = [XY, XY, UV, UV]` は
  そのまま warp マップ（src=カメラ, dst=プロジェクタ）として使える。
- `interpolate_p2c` の出力 `[proj_x, proj_y, cam_x, cam_y]` は列順が逆（UV が先頭）なので、
  warp マップとして使う場合は列の並べ替えが必要。

---

## 5. `coords.py` API（正準定義一覧）

| 関数 | 式 | 規約 |
|---|---|---|
| `xy_pixel_centers(n)` | $\{0,\dots,n-1\}$ | カメラ画素中心の座標列 |
| `xy_to_pixel(x)` | $\lfloor x + 0.5\rfloor$ | カメラ 連続座標→画素インデックス |
| `uv_pixel_centers(n)` | $\{0.5,\dots,n-0.5\}$ | プロジェクタ画素中心の座標列 |
| `uv_to_pixel(u)` | $\lfloor u\rfloor$ | プロジェクタ 連続座標→画素インデックス |
| `uv_to_array(u)` | $u - 0.5$ | UV→「整数=中心」配列座標（bilinear 用） |
| `array_to_uv(a)` | $a + 0.5$ | 上の逆 |
| `uv_to_normalized(u, size)` | $2u/\text{size} - 1$ | UV→`grid_sample(align_corners=False)` 正規化 |
| `reduced_size(full, step)` | $\lfloor (full-1)/step\rfloor + 1$ | 縮小グレイコード解像度 |
| `block_of(pixel, step)` | $\lfloor pixel/step\rfloor$ | 画素→所属ブロック |
| `block_center_uv(g, step)` | $step\,(g+0.5)$ | ブロック中心の UV 座標（decode GT） |

`xy_to_pixel` / `uv_to_pixel` / `uv_to_array` / `uv_to_normalized` は numpy 配列・スカラと
torch テンソルの双方で動作する（backend は入力から自動判別）。

### 補足: `uv_to_normalized` の導出

`grid_sample(align_corners=False)` は正規化座標 $g\in[-1,1]$ と配列画素中心座標 $p$ を

$$p = \frac{(g+1)\,\text{size} - 1}{2}$$

で対応づける。UV 座標 $u$（中心 $i+0.5$）は配列画素中心座標で $u-0.5$ に当たるので、
$p = u-0.5$ を解くと

$$g = \frac{2u}{\text{size}} - 1$$

を得る。これが UV を直接正規化する式である。

---

## 6. データファイルの列定義

| ファイル | 列 |
|---|---|
| `result_c2p.csv` | `cam_x, cam_y, proj_x, proj_y` |
| `result_p2c.csv` | `proj_x, proj_y, cam_x, cam_y` |
| `result_c2p_compensated_{method}.npy` | `(N, 2, 2)` object: `[(cam_x, cam_y), (proj_x, proj_y)]`（互換形式） |
| `result_p2c_compensated_delaunay.npy` | `(H*W, 4)` float32: `[proj_x, proj_y, cam_x, cam_y]` |

カメラ列は XY（整数中心）、プロジェクタ列は UV（中心 $i+0.5$、step>1 ではブロック中心）。

---

## 7. よくある間違い（アンチパターン）

| 誤 | 正 |
|---|---|
| UV 座標を `round` / `np.rint` で画素化 | `uv_to_pixel`（= `floor`） |
| P2C 補間を整数格子でクエリ | `uv_pixel_centers`（= $i+0.5$） |
| forward の bilinear で `floor(u)` を中心扱い | `uv_to_array`（= $u-0.5$）してから重み計算 |
| gen と decode で step 引数順を変える | 両者 `(height_step, width_step)` |
| カメラ座標を `int()`（0 方向切り捨て）で画素化 | `xy_to_pixel`（= `floor(x+0.5)`） |
