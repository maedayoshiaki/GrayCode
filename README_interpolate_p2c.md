# interpolate_p2c.py - P2C マップのドロネー補間

## 概要

`decode.py` で生成された P2C（Projector-to-Camera）対応マップに対し、
**ドロネー三角形分割に基づく線形補間**を適用して、
プロジェクタ画像の全画素にカメラ座標を割り当てる。

## 使い方

```bash
python interpolate_p2c.py <p2c_numpy_filename> <proj_height> <proj_width>
```

**例:**

```bash
python interpolate_p2c.py result_p2c.npy 768 1024
```

## 入出力

| 種別 | ファイル | 形式 |
|------|----------|------|
| 入力 | `result_p2c.npy` | decode.py が出力した P2C 辞書 (pickle) |
| 出力 | `result_p2c_compensated_delaunay.npy` | (H\*W, 4) float32 配列 `[proj_x, proj_y, cam_x, cam_y]` |
| 出力 | `result_p2c_compensated_delaunay.csv` | 同内容の CSV |
| 出力 | `result_p2c_compensated_delaunay_vis.png` | 可視化画像 |

## 数学的な挙動

### 1. 問題設定

グレイコードデコードにより、一部のプロジェクタ座標 $\mathbf{p}_i = (p_{x,i},\; p_{y,i})$ に対して
対応するカメラ座標 $\mathbf{c}_i = (c_{x,i},\; c_{y,i})$ が既知である。
目標は、プロジェクタ画像の **全画素** $(p_x,\; p_y)$ に対してカメラ座標 $(c_x,\; c_y)$ を求めることである。

$$f: \mathbb{R}^2 \to \mathbb{R}^2, \quad (p_x, p_y) \mapsto (c_x, c_y)$$

### 2. ドロネー三角形分割 (Delaunay Triangulation)

既知の対応点集合 $\{\mathbf{p}_1, \mathbf{p}_2, \ldots, \mathbf{p}_N\}$ に対して、
プロジェクタ座標平面上でドロネー三角形分割を構築する。

ドロネー三角形分割は以下の性質を持つ：

- **空円性 (Empty Circumcircle Property)**:
  どの三角形の外接円の内部にも、他の入力点が含まれない。
  これにより、極端に細長い三角形が避けられ、補間の数値的安定性が高まる。

- **凸包 (Convex Hull)**:
  三角形分割は入力点集合の凸包を覆う。凸包の外側には三角形が存在しない。

### 3. 三角形内の線形補間 (Barycentric Interpolation)

あるクエリ点 $\mathbf{q} = (q_x, q_y)$ が三角形 $\triangle(\mathbf{p}_a, \mathbf{p}_b, \mathbf{p}_c)$ の
内部にある場合、**重心座標 (Barycentric Coordinates)** $(\lambda_a, \lambda_b, \lambda_c)$ を計算する：

$$\mathbf{q} = \lambda_a \mathbf{p}_a + \lambda_b \mathbf{p}_b + \lambda_c \mathbf{p}_c, \quad \lambda_a + \lambda_b + \lambda_c = 1, \quad \lambda_i \geq 0$$

対応するカメラ座標は同じ重心座標で補間される：

$$f(\mathbf{q}) = \lambda_a \mathbf{c}_a + \lambda_b \mathbf{c}_b + \lambda_c \mathbf{c}_c$$

これは三角形内で **アフィン変換** に相当し、
三角形の各頂点では既知の対応値を正確に再現する（**補間** であり **近似** ではない）。

### 4. 凸包外部の処理 (Nearest Neighbor Extrapolation)

凸包の外側に位置するクエリ点に対しては、ドロネー補間は値を定義できない。
本プログラムでは **最近傍補間 (Nearest Neighbor Interpolation)** を適用する：

$$f(\mathbf{q}) = \mathbf{c}_{k}, \quad k = \arg\min_i \|\mathbf{q} - \mathbf{p}_i\|_2$$

最も近い既知点のカメラ座標をそのまま割り当てる。
これにより全プロジェクタ画素が埋まるが、凸包外部では不連続な値の変化が生じうる。

### 5. 重複点の扱い

同一プロジェクタ座標に複数のカメラ座標が対応する場合（1つのプロジェクタ画素を複数のカメラ画素が観測する場合）、
全ての対応点をそのままドロネー三角形分割の入力とする。

`LinearNDInterpolator` は重複入力点を持つ場合でも動作し、
重複点の値は三角形分割の過程で暗黙的に平均化される。

### 6. C2P 補間との対比

| | interpolate_c2p | interpolate_p2c |
|---|---|---|
| 三角形分割の定義域 | カメラ座標平面 | プロジェクタ座標平面 |
| 補間する値 | プロジェクタ座標 $(p_x, p_y)$ | カメラ座標 $(c_x, c_y)$ |
| 出力グリッド | カメラ画像の全画素 | プロジェクタ画像の全画素 |
| 1対多の扱い | 発生しない（カメラ1画素→プロジェクタ1点） | 複数カメラ画素→同一プロジェクタ座標を全て保持 |
