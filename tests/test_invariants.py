"""Properties that must hold across many shapes, not just the hand-checked cases.

The hand-computed examples in ``test_purge_and_embargo`` and
``test_purge_embargo_toy`` pin exact indices for two specific datasets. These
tests take the opposite approach: they assert the *guarantee* over a grid of
sample counts, fold counts, embargo widths and randomly varying label horizons,
where nobody has worked out the answer in advance.

Neither style substitutes for the other. Exact examples catch a boundary that
moved by one; property checks catch a boundary that only misbehaves in a shape
nobody thought to write down.
"""

from __future__ import annotations

import numpy as np
import pytest

from purged_cv import PurgedKFold

X20 = np.zeros((20, 2))
ZERO_HORIZON = np.arange(20)


@pytest.mark.parametrize("n_samples", [23, 50, 97])
@pytest.mark.parametrize("n_splits", [2, 3, 5])
@pytest.mark.parametrize("embargo_pct", [0.0, 0.02, 0.10])
def test_no_overlap_survives_for_random_horizons(
    n_samples: int, n_splits: int, embargo_pct: float
) -> None:
    """The guarantee itself, over randomly varying label horizons.

    For every fold, no surviving training row's closed label window may
    intersect the test span. This is the property the package sells.
    """
    rng = np.random.default_rng(n_samples * n_splits)
    starts = np.arange(n_samples)
    ends = starts + rng.integers(0, 12, size=n_samples)
    X = np.zeros((n_samples, 2))

    cv = PurgedKFold(n_splits=n_splits, label_end_times=ends, embargo_pct=embargo_pct)
    try:
        folds = list(cv.split(X))
    except ValueError as exc:
        # Some combinations are genuinely infeasible: with few splits and long
        # random horizons a fold's span can swallow the dataset. Refusing is the
        # documented behaviour and satisfies the invariant trivially -- but only
        # that specific refusal is an acceptable outcome here.
        assert "no training samples left" in str(exc)
        return

    for train, test in folds:
        span_start = starts[test].min()
        span_end = ends[test].max()
        overlapping = (ends[train] >= span_start) & (starts[train] <= span_end)
        assert not overlapping.any()
        assert set(train).isdisjoint(test)


@pytest.mark.parametrize("n_samples", [30, 61])
@pytest.mark.parametrize("n_splits", [3, 4])
def test_test_folds_always_tile_the_data_whatever_the_purging(
    n_samples: int, n_splits: int
) -> None:
    """Purging removes training rows only; every row is still tested exactly once."""
    rng = np.random.default_rng(n_samples)
    ends = np.arange(n_samples) + rng.integers(0, 6, size=n_samples)
    cv = PurgedKFold(n_splits=n_splits, label_end_times=ends, embargo_pct=0.05)

    tested = np.concatenate([test for _, test in cv.split(np.zeros((n_samples, 2)))])
    np.testing.assert_array_equal(np.sort(tested), np.arange(n_samples))


def test_a_large_embargo_clears_the_whole_right_hand_side() -> None:
    """With nothing to purge, the embargo's reach is exactly ceil(pct * n)."""
    cv = PurgedKFold(n_splits=4, label_end_times=ZERO_HORIZON, embargo_pct=0.25)
    train, test = list(cv.split(X20))[0]
    assert test.tolist() == [0, 1, 2, 3, 4]
    assert train.tolist() == [10, 11, 12, 13, 14, 15, 16, 17, 18, 19]


def test_irregular_datetime_index_is_supported() -> None:
    """Gaps in the calendar are fine; only row order has to be chronological.

    Financial data has weekends and holidays, so an evenly spaced index is the
    exception rather than the rule.
    """
    pd = pytest.importorskip("pandas")
    index = pd.to_datetime(
        [f"2024-01-{d:02d}" for d in (1, 2, 5, 6, 7, 12, 13, 14, 20, 21, 22, 25)]
    )
    starts = index.to_numpy()
    ends = (index + pd.Timedelta(days=4)).to_numpy()
    cv = PurgedKFold(n_splits=3, label_start_times=starts, label_end_times=ends)

    for train, test in cv.split(np.zeros((len(index), 2))):
        span_start = starts[test].min()
        span_end = ends[test].max()
        assert not ((ends[train] >= span_start) & (starts[train] <= span_end)).any()
