"""graycode.evaluation.stats の単体テスト (統計集約・ロバスト統計)。"""
from __future__ import annotations

import numpy as np

from graycode.evaluation import stats


def test_summarize_known_values() -> None:
    x = np.array([3.0, 4.0])
    s = stats.summarize(x)
    assert s.n == 2
    assert abs(s.rmse - np.sqrt(12.5)) < 1e-12
    assert abs(s.mean - 3.5) < 1e-12
    assert abs(s.median - 3.5) < 1e-12
    assert s.unit == "px"


def test_summarize_drops_nonfinite() -> None:
    x = np.array([1.0, np.nan, np.inf, 3.0])
    s = stats.summarize(x)
    assert s.n == 2  # nan/inf 除外


def test_summarize_empty_is_nan() -> None:
    s = stats.summarize(np.array([]))
    assert s.n == 0
    assert np.isnan(s.rmse)


def test_magnitudes_vectors() -> None:
    v = np.array([[3.0, 4.0], [0.0, 0.0]])
    assert np.allclose(stats.magnitudes(v), [5.0, 0.0])


def test_mad_robust_to_outlier() -> None:
    x = np.array([1.0, 1.0, 1.0, 1.0, 1.0, 100.0])
    assert stats.mad(x) == 0.0  # median=1, 偏差中央値=0


def test_robust_inlier_mask_excludes_outlier() -> None:
    rng = np.random.default_rng(0)
    x = np.abs(rng.normal(0.0, 1.0, 1000))
    x[0] = 50.0
    mask = stats.robust_inlier_mask(x, k=3.0)
    assert not mask[0]
    assert mask[1:].mean() > 0.9


def test_robust_inlier_all_when_zero_spread() -> None:
    x = np.array([2.0, 2.0, 2.0, 2.0])
    mask = stats.robust_inlier_mask(x)
    assert mask.all()


def test_huber_mean_is_robust() -> None:
    x = np.array([1.0, 2.0, 3.0, 4.0, 100.0])
    h = stats.huber_mean(x)
    assert np.isfinite(h)
    assert h < float(np.mean(x))  # 外れ値の影響を抑える
    assert h > 1.0
