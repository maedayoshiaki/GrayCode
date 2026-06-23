# A群 再投影/整合性誤差 評価手法 — 実装ドキュメント

作成日: 2026-06-23
対象モジュール: `src/graycode/evaluation/`
関連: 文献調査レポート [`reprojection_error_survey.md`](reprojection_error_survey.md)、座標規約 [`../COORDINATES.md`](../COORDINATES.md)

> 本ドキュメントは、内部・外部パラメータ(カメラ/プロジェクタの内部行列・歪み・相対姿勢)を
> **推定していない**構造化光 (GrayCode) システムで、得られた**画素対応マップだけ**から
> 「何 px ずれているか」を統計解析する手法群 (A群) と、その実装・引用文献を説明する。

---

## 0. 前提と全体像

較正パラメータが無いため、古典的な再投影誤差(3D→画像)は計算できない。A群が測るのは
**画素単位の整合性/幾何残差**である。「何 px ずれているか」は出せるが、**何に対するずれか**を
手法ごとに区別する必要がある。

| # | 手法 | 実装 | 何を測るか | **測れないもの** | 必要入力 |
|---|---|---|---|---|---|
| **A1** | 往復(サイクル)整合性 | `cycle_consistency` | 2つの密マップ(補間ドメインが異なる)の相互整合性 | decode 自体の絶対精度(両マップが同一 decode 由来) | 密C2P + 密P2C |
| **A2** | 補間ホールドアウト残差 | `holdout_interpolation_residual` | 補間/穴埋めの品質(未知点での誤差) | decode の絶対精度 | 生C2P(補間前) |
| **A3** | 既知パターン絶対誤差 | `known_pattern_error` | **真値からの絶対ずれ**(proj平面) | — (上限はカメラ検出精度) | 既知パターン撮影 + 密C2P |
| **A4** | エピポーラ Sampson | `epipolar_sampson` | 幾何(エピポーラ)整合性。外れ値検出兼 | 絶対スケール(F は up-to-scale) | 対応集合 |

座標はすべて **pixel-is-point** 規約(整数 = 画素中心、`graycode.coords`)。サンプリングも
`map_coordinates` のインデックス = 画素中心なので半画素オフセット不要。

**文献の最重要教訓(全手法に通底):** 再投影誤差は精度と単調でない(少枚数で過適合し小さく
見える)[Lang & Schlegl 2016; Moreno & Taubin 2012]。よって本実装は単一 RMSE で結論せず、
**複数指標の併記**(`report.py` が A1–A4 を 1 つの表に)と**ロバスト統計**を既定とする。

---

## 1. 各手法の定義・数式・引用

### A1. 往復(サイクル)整合性 — `metrics.cycle_consistency`

各カメラ画素 `c=(x,y)` を 密C2P で proj 座標 `p=C2P(c)` に写し、密P2C で
`c'=P2C(p)`(双線形)に戻す。残差 `c'−c`(カメラ px)を集計する。

```
residual(x,y) = P2C( C2P(x,y) ) − (x,y)
```

- **意味:** 2 つの密マップは同じ生 decode 由来だが、**補間ドメインが異なる**
  (C2P=カメラ格子で補間、P2C=プロジェクタ格子で集約+補間)。その**相互整合性**を px 化し、
  穴埋め・1対多集約・反転で生じる不整合を検出する。
- **限界:** 両マップが同一 decode 由来のため、**decode 自体の精度は検証できない**
  (補間が一致すれば 0 でも decode は誤りうる)。サニティチェックに留める。
- **引用:** forward–backward / cycle consistency は対応評価の一般的手法。幾何誤差の枠組みは
  Hartley & Zisserman『MVG』第4章。三角測量の往復性は Hartley & Sturm。
- **検証:** 恒等マップ → RMSE `< 1e-9`、既知1pxシフトの逆マップ → 中央値 `≈1.0`
  (`tests/test_eval_metrics.py::test_a1_*`)。

### A2. 補間ホールドアウト残差 — `metrics.holdout_interpolation_residual`

生 decode 点を train/test に分割し、train から `LinearNDInterpolator` を構築、test 点の
proj 座標を予測して真の decode 値との差(proj px)を測る。

- **意味:** **補間/穴埋めの品質**を未知点で評価する交差検証。本プロジェクトの inpaint 監査
  ([[inpaint-audit-findings-2026-06]]) と同じ問題意識。
- **なぜホールドアウトか:** Delaunay 補間は既知点を**厳密保存**する
  (`tests/test_interpolation_coords.py::test_p2c_preserves_gt_on_integer_grid`)。よって
  「密マップ vs 生 decode」は構造上ほぼ 0 になり無意味。未知点での誤差を測る必要がある。
- **限界:** decode の絶対精度ではなく補間の質。凸包外(外挿)点は別カウントし除外。
- **引用:** 線形補間/サブピクセル位置決めの精度依存は Salvi, Pagès & Batlle 2004。
  交差検証は標準的な汎化誤差推定。
- **検証:** アフィン場 → RMSE `< 1e-6`(線形補間が厳密復元)、非線形場 → RMSE `> 1e-3`
  (`test_a2_affine_field_near_zero` / `test_a2_nonaffine_field_positive`)。

### A3. 既知パターン絶対誤差 — `metrics.known_pattern_error`

自作した既知パターン(ChArUco/市松)の各特徴は、**真のプロジェクタ座標が設計値として既知**。
カメラ画像でサブピクセル検出した特徴位置 `c_feat` で密C2P を参照し、得られた proj 座標
`p_graycode` と真値 `p_true` の差を測る。

```
residual = C2P(c_feat) − p_true          # proj 平面、px
```

- **意味:** **較正なしで到達できる唯一の「絶対」ずれ**。GrayCode とは独立の参照(設計値 +
  サブピクセル検出)を持ち込むことで、自己整合性(A1)より強い「真値からのずれ」を測る。
- **限界:** 絶対値の上限は**カメラ側特徴検出精度**(良い検出器で ~0.1–0.2px)。
- **真座標の取得:** `patterns.generate_*_pattern` は生成したクリーン画像に同じ検出器をかけて
  真座標テーブルを作る。これでカメラ側との検出器差をキャンセルし、差分が純粋に対応誤差になる。
  **ID を持つ ChArUco を推奨**(向き/並びの曖昧さなし)。市松は順序依存。
- **引用:** Moreno & Taubin 2012 が*較正*で使う「特徴のカメラ座標→局所ホモグラフィ→
  プロジェクタ座標」を*評価*に転用したもの。サブピクセルコーナーは Zhang 2000 / OpenCV、
  マーカー ID は ArUco [Garrido-Jurado et al. 2014]。
- **検証:** 既知オフセット δ=(2,3) を埋め込んだ恒等マップ → 残差が δ に厳密一致
  (`test_a3_recovers_injected_offset`)。

### A4. エピポーラ整合性 (基礎行列 + Sampson 距離) — `metrics.epipolar_sampson`

対応集合から基礎行列 F を RANSAC 推定し(`proj^T F cam = 0`)、各対応の **Sampson 距離**(px)を
求める。Sampson 距離は真の再投影誤差の1次近似 [Hartley & Zisserman §11.4.3]:

```
d_Sampson = |x2ᵀ F x1| / sqrt( (Fx1)_x² + (Fx1)_y² + (Fᵀx2)_x² + (Fᵀx2)_y² )
   x1 = [cam_x, cam_y, 1]ᵀ,  x2 = [proj_x, proj_y, 1]ᵀ
```

- **意味:** **較正なしで最も学術標準に近い「px 再投影誤差」**。エピポーラ幾何に矛盾する対応
  (デコード誤り等)が大きな Sampson 距離として現れ、**外れ値検出**を兼ねる。
- **限界:** F は up-to-scale なので**絶対スケールは測れない**(相対的な幾何整合性)。
- **実装詳細:** `cv2.findFundamentalMat(cam, proj, FM_RANSAC, thresh, conf)`。点数が多い場合は
  `max_fit` 点をランダム抽出して F を推定し、Sampson は全点で評価。RANSAC inlier 率と、
  MAD ベースのロバスト内れ値統計(`stats_inliers`)を併記。
- **引用:** Sampson 距離・基礎行列・幾何誤差は Hartley & Zisserman『MVG』§11.4.3 / §12。
  RANSAC は Fischler & Bolles 1981。最適三角測量との関係は Hartley & Sturm。
- **検証:** 2 視点投影で生成した厳密対応 → `sampson_distance` が `< 1e-6`
  (既知 F で `x2ᵀ F x1=0`)、RANSAC 推定でも median `< 1.0px`・inlier率 `>0.9`。
  外れ値注入 → 当該点の Sampson が大、ロバスト内れ値統計は小
  (`test_sampson_zero_on_exact_epipolar` / `test_a4_consistent_*` / `test_a4_flags_outliers`)。

---

## 2. 統計層 — `evaluation/stats.py`

全手法の誤差大きさ(px)を `summarize()` で要約: **n / RMSE / 平均 / 標準偏差 / 中央値 /
p90 / p95 / p99 / 最大 / MAD / MAD_std**(NaN/inf は除外)。

ロバスト統計:
- `mad`, `mad_std` (= 1.4826·MAD): 外れ値に頑健な散らばり。裾の重い誤差分布で標準偏差の代替。
- `robust_inlier_mask(k=3)`: `|x − median| ≤ k·MAD_std` で外れ値除去(RANSAC を使わない既定手段)。
- `huber_mean`: Huber M 推定の頑健な位置(IRLS、既定 δ=1.345·MAD_std で正規分布 95% 効率)。

引用: ロバスト統計 [Huber 1964; Rousseeuw & Croux (MAD)]、RANSAC [Fischler & Bolles 1981]、
Huber ノルムによるバンドル調整の頑健化は survey §2.3 参照。

---

## 3. 可視化 — `evaluation/viz.py`

- `plot_histogram`: 誤差大きさのヒストグラム(平均・中央値線付き)。Lang & Schlegl は平面フィット
  距離を 256 ビンで図示。
- `plot_heatmap`: 画像平面上の誤差大きさ(turbo カラー)。レンズ歪み残差・視野端劣化の検出。
- `plot_quiver`: **誤差ベクトル場**。A群は方向付き残差なので、系統的歪み(放射状)や
  **一様シフト**(座標規約の取り違え=±0.5px)が一目で分かる。

**matplotlib は任意依存**(`pyproject.toml` の dev グループ)。不在時はヒートマップを cv2 の
JET カラーマップで代替し、ヒストグラム/quiver はスキップ。いずれの場合も**生の誤差配列
(`*_magnitude.npy` / `*_residual.npy`)と統計サマリは常に保存**するので後から任意ツールで作図可。

---

## 4. 使い方 (CLI) — `python -m graycode.evaluation`

### 4.1 標準ワークフロー

```bash
# A1/A2/A4 を保存済みマップから評価 (output_dir に result_*.npy がある前提)
uv run python -m graycode.evaluation eval \
    --metrics a1,a2,a4 \
    --output-dir <experiment_dir> \
    --cam-h <camH> --cam-w <camW> --proj-h <projH> --proj-w <projW>
```

既定の入力ファイル名(`paths.output_dir` 配下、`--raw-c2p`/`--dense-c2p`/`--dense-p2c` で上書き可):
- 生C2P: `result_c2p.npy`(A2/A4)
- 密C2P: `result_c2p_compensated_delaunay.npy`(A1/A3)
- 密P2C: `result_p2c_compensated_delaunay.npy`(A1)

出力: `eval_report.json` / `eval_report.csv`(指標横断サマリ)、`eval_figures/`(図 + .npy)。

### 4.2 A3(既知パターン絶対誤差)の手順

投影ウィンドウの作成は **projector-controller**(`projector_controller`、`C:/py_scripts/projector-controller`)を
用いる(`evaluation/project.py`)。`gen-pattern --project` で生成と投影を一度に行える。

```bash
# 1) 既知パターン + 真座標 JSON を生成し、そのまま projector-controller で投影
uv run python -m graycode.evaluation gen-pattern \
    --pattern-type charuco --output-dir <dir> --proj-h <H> --proj-w <W> \
    --project --display 1            # display 1 = プロジェクタ。--duration 秒 で自動クローズ

# (任意) 既存の任意画像を投影だけする
uv run python -m graycode.evaluation project --image <pattern.png> --display 1

# 2) 投影中に GrayCode と一緒に撮影 (同一配置)。pipeline で密C2P を作る。

# 3) パターン撮影画像で A3 を評価
uv run python -m graycode.evaluation eval --metrics a3 \
    --output-dir <dir> --cam-h <camH> --cam-w <camW> \
    --pattern-image <camera_capture.png> \
    --true-coords <dir>/eval_pattern_charuco_true_coords.json \
    --pattern-type charuco
```

projector-controller は任意依存(`evaluation/project.py` で lazy import)。2dsr-prc には editable
依存として既に入っている。パターンはプロジェクタ解像度で生成し、`fit_mode="native"`(等倍)で
フルスクリーン投影すると プロジェクタ画素 = パターン画素 の 1:1 投影になる。

### 4.3 ライブラリとして

```python
from graycode.evaluation import (
    cycle_consistency, holdout_interpolation_residual,
    known_pattern_error, epipolar_sampson,
)
res = epipolar_sampson(corr)         # corr: (N,4)=[cam_x,cam_y,proj_x,proj_y]
print(res.stats_all.rmse, res.stats_inliers.median, res.inlier_ratio)
```

### 4.4 2dsr-prc 連携(GrayCode 撮影後に向こう側で評価)

2dsr-prc は GrayCode 結果を `ProjectorCameraMap` の `.npz`(キー `p2c`(N,4)=`[proj_x,proj_y,cam_x,cam_y]`、
`proj_size`、`coord_convention`)で `output/<run>/p2c.npz` に保存する。graycode は 2dsr-prc に editable
依存として入っているので、**2dsr-prc のディレクトリから直接評価を実行できる**:

```bash
# 2dsr-prc のルートで (graycode は hardware extra で editable 導入済み)
uv run python -m graycode.evaluation eval \
    --p2c-npz output/<run>/p2c.npz \
    --metrics a4,a2 \
    --output-dir output/<run>/eval
# A3 も使うなら密C2Pが要るので --cam-h/--cam-w を足す (npz の対応から Delaunay で密C2P構築):
#   --metrics a4,a3 --cam-h <camH> --cam-w <camW> --pattern-image ... --true-coords ...
```

`io.load_projector_camera_map_npz` が `.npz` を読み、列を `[cam_x,cam_y,proj_x,proj_y]` に並べ替える
(`coord_convention` が `pixel-is-point` でなければ警告)。`--p2c-npz` 指定時は `proj_size` を npz から
自動取得。

**2dsr-prc 側に便利スクリプト** `scripts/eval_graycode.py` を配置済み(`graycode.evaluation` への薄い
ラッパー)。`--run <dir>` で `p2c.npz` と出力先(`<dir>/eval`)を解決し、A1/A3 のカメラサイズを対応から
**自動推定**する:

```bash
# 2dsr-prc のルートで
uv run python scripts/eval_graycode.py eval --run output/<run> --metrics a1,a2,a4

# A3 自動撮影 (ベンチ: SR-5100 + プロジェクタ要)。パターン生成→投影→撮影→A3 を一気通貫:
uv run python scripts/eval_graycode.py capture-pattern --run output/<run> \
    --recipe path/to/recipe.xml --display 1 --eval

# (手動の場合) パターン生成→投影→自前撮影→画像を渡して A3
uv run python scripts/eval_graycode.py gen-pattern --run output/<run> --project --display 1
uv run python scripts/eval_graycode.py eval --run output/<run> --metrics a3 \
    --pattern-image cap.png --true-coords eval_pattern/eval_pattern_charuco_true_coords.json
```

**A3 自動撮影 (`capture-pattern`, ベンチ専用)** は `run_prc_live.py` の取得シーケンス
(`sr5100.session`→`open_device`→recipe/`mr_full`→`device.live_capture`→`_crop_to_mr`、
`ProjectorControllerWindow`)を忠実に再現する。投影は projector-controller、撮影は SR-5100 の
**ライブ・グレースケールストリーム**を 1 枚、`measurement_range` にクロップして保存する
(= GrayCode 撮影と同じカメラ格子 → p2c と座標系が一致)。**GrayCode 撮影で使ったのと同じ
`--recipe`(または `--measurement-roi`/`--integration-ms`)を渡すこと** — クロップ ROI が一致しないと
A3 の特徴座標が p2c の格子からずれる。CI では実行しない(実機が要る)。

- **A4 / A2**: npz の対応からそのまま計算可(追加情報不要)。**2dsr の p2c はプロジェクタ格子上で
  既に密**なので、A4 の RANSAC inlier 率が低めに出る場合は**凸包外の外挿画素がエピポーラ不整合**として
  拾われている(`stats_inliers` のロバスト中央値が整合する大多数の実誤差)。実データ例:
  median(robust) ≈ 0.21px、inlier率 ≈ 0.72。
- **A1 / A3**: 密C2P が要る。`--cam-h/--cam-w` を与えると npz の対応から `interpolate_c2p_delaunay` で
  密C2P を構築する(2dsr の densify と同規模)。dense_p2c は npz の `p2c` をそのまま使う。
- **A2 の注意**: 2dsr の p2c は既に密(補間済)なので、A2 ホールドアウトは「補間済み場の汎化」を測る
  (生 decode に対する補間品質ではない)。生の疎 decode を評価したい場合は graycode の `result_c2p.npy`
  (補間前)を使う。

---

## 5. 推奨運用(文献の教訓を反映)

1. **A4(F+Sampson)を主指標**に — 較正なしで幾何的に何 px かを最も正当に出せ、外れ値検出も兼ねる。
2. **A3** が可能なら併用 — 唯一の絶対誤差。ChArUco 推奨。
3. **A2** で補間品質を、**A1** はサニティチェックとして補助。
4. 単発 RMSE で結論せず、**複数姿勢/複数回**で 平均±SD・分位点・正規化SD を報告(Lang & Schlegl)。
5. **着手時は合成ゼロ検証**(誤差ゼロの理想対応で各指標が厳密に 0)で座標規約の取り違えを切り分け
   — 本実装のテスト群 (`test_a1_identity_is_zero`, `test_a3_recovers_injected_offset` 等) が
   そのリグレッションガードになっている。

---

## 6. 引用文献(本実装が依拠する一次文献)

調査レポート [`reprojection_error_survey.md`](reprojection_error_survey.md) の文献に加え、本実装で
新たに参照:

1. **Hartley & Zisserman**, *Multiple View Geometry in Computer Vision*, 2nd ed. — Sampson 距離
   (§11.4.3)、幾何誤差・三角測量(§12)、RANSAC。**A1/A4** の理論基盤。
2. **Hartley & Sturm (1997)** "Triangulation", CVIU — 最適三角測量、エピポーラ制約下の再投影。**A4**。
3. **Moreno & Taubin (2012)** "Simple, Accurate, and Robust Projector-Camera Calibration", 3DIMPVT —
   局所ホモグラフィによる特徴のプロジェクタ座標化。**A3** の発想元。再投影誤差の非単調性の警告。
4. **Zhang (2000)** "A Flexible New Technique for Camera Calibration", IEEE TPAMI — サブピクセル
   コーナー検出と較正残差。**A3**。
5. **Salvi, Pagès & Batlle (2004)** "Pattern codification strategies in structured light systems",
   Pattern Recognition — サブピクセル位置決めの精度依存。**A2** の文脈。
6. **Lang & Schlegl (2016)** "Camera-Projector Calibration — Methods, Influencing Factors and
   Evaluation...", ICIRA, LNAI 9835 — 再投影誤差を単独指標にしない、σn・復元誤差・ヒストグラム
   可視化。**統計層・運用方針**の根拠。
7. **Fischler & Bolles (1981)** "Random Sample Consensus", Comm. ACM — RANSAC。**A4・外れ値除去**。
8. **Huber (1964)** "Robust Estimation of a Location Parameter", Ann. Math. Statist. — Huber M 推定。
   **stats.huber_mean**。
9. **Garrido-Jurado et al. (2014)** "Automatic generation and detection of highly reliable fiducial
   markers under occlusion", Pattern Recognition — ArUco/ChArUco。**A3 patterns**。

---

## 7. 実装ファイル一覧

| ファイル | 役割 |
|---|---|
| `src/graycode/evaluation/stats.py` | 統計集約・ロバスト統計 |
| `src/graycode/evaluation/metrics.py` | A1–A4 指標 |
| `src/graycode/evaluation/patterns.py` | A3 既知パターン生成・検出(ChArUco/市松) |
| `src/graycode/evaluation/project.py` | projector-controller による投影(A3 パターン投影) |
| `src/graycode/evaluation/io.py` | 2dsr-prc ProjectorCameraMap `.npz` 読み込み |
| `src/graycode/evaluation/viz.py` | ヒストグラム/ヒートマップ/誤差ベクトル場 |
| `src/graycode/evaluation/report.py` | JSON/CSV サマリ出力 |
| `src/graycode/evaluation/cli.py` / `__main__.py` | CLI(`eval` / `gen-pattern` / `project`) |
| `tests/test_eval_{stats,metrics,patterns,io}.py` | 検証テスト(23件) |
