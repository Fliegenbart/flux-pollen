"""Unit tests for the scoring functions."""

import math

import numpy as np
import pytest

from app.services.ml.metrics import (
    coverage,
    interval_score,
    mae,
    pinball_loss,
    rmse,
    weighted_interval_score,
)


def test_mae_rmse_match_textbook_definitions():
    y = [1.0, 2.0, 3.0]
    p = [1.0, 2.5, 3.5]
    assert mae(y, p) == pytest.approx((0.0 + 0.5 + 0.5) / 3.0)
    assert rmse(y, p) == pytest.approx(math.sqrt((0 + 0.25 + 0.25) / 3.0))


def test_pinball_at_median_is_half_mae():
    rng = np.random.default_rng(42)
    y = rng.normal(0, 1, size=500)
    p = rng.normal(0, 1, size=500)
    # Textbook identity: pinball(q=0.5) == 0.5 * mean(|y − p|)
    assert pinball_loss(y, p, 0.5) == pytest.approx(0.5 * mae(y, p), rel=1e-9)


def test_pinball_rejects_quantile_outside_unit_interval():
    with pytest.raises(ValueError):
        pinball_loss([1.0], [0.0], 0.0)
    with pytest.raises(ValueError):
        pinball_loss([1.0], [0.0], 1.0)


def test_interval_score_is_width_when_observation_inside():
    y = [5.0, 5.0]
    lower = [4.0, 4.0]
    upper = [6.0, 6.0]
    # Inside → score equals the interval width (= 2.0).
    assert interval_score(y, lower, upper, alpha=0.2) == pytest.approx(2.0)


def test_interval_score_penalises_miss_from_below():
    # y is below the lower bound — penalty scales like (2/alpha)·(lo − y).
    y = [0.0]
    lower = [4.0]
    upper = [6.0]
    alpha = 0.2
    expected = (6.0 - 4.0) + (2.0 / alpha) * (4.0 - 0.0)
    assert interval_score(y, lower, upper, alpha=alpha) == pytest.approx(expected)


def test_wis_reduces_to_half_mae_when_intervals_collapse_to_the_median():
    # If the forecast degenerates to a point — L = U = median — every interval
    # score collapses to its miss-penalty term, and WIS simplifies. In the
    # limit of one level this becomes proportional to |y − m|.
    y = [2.0, 4.0, 6.0]
    median = [1.0, 5.0, 5.0]
    intervals = {0.2: (median, median)}
    # With one level k=1: WIS = 1/(1+0.5) · [0.5 |y−m| + (0.2/2)·IS]
    # IS on collapsed interval at level α = (2/α)·(|y − m|) → cancels alpha.
    # → WIS = (0.5 |y−m| + |y − m|) / 1.5 = |y − m|
    expected = np.mean(np.abs(np.asarray(y) - np.asarray(median)))
    assert weighted_interval_score(y, median, intervals) == pytest.approx(expected)


def test_coverage_counts_inclusive_hits():
    y = [1.0, 5.0, 10.0]
    lower = [0.0, 5.0, 11.0]
    upper = [1.0, 4.9, 12.0]
    # 1.0 hits (inclusive), 5.0 just misses (5 < 4.9 is false but 5.0 < 5 is false), 10.0 out
    # Row 0: y=1, in [0,1] → hit
    # Row 1: y=5, in [5,4.9] → lower<=upper fails, but code does y>=lo & y<=hi → y=5>=5 ✓, y=5<=4.9 ✗
    # Row 2: y=10, [11,12] → ✗
    assert coverage(y, lower, upper) == pytest.approx(1.0 / 3.0)
