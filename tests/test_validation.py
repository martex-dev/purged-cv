"""Input validation.

A splitter that silently accepts nonsense produces a number the user will trust.
Every rejection here exists because the alternative is a plausible-looking score
computed on a leaky fold.
"""

from __future__ import annotations

import numpy as np
import pytest

from purged_cv import PurgedKFold

X10 = np.zeros((10, 2))


# --------------------------------------------------------------------------
# embargo_pct
# --------------------------------------------------------------------------


@pytest.mark.parametrize("bad", [-0.01, 1.0, 1.5, 100])
def test_embargo_pct_outside_unit_interval_is_rejected(bad: float) -> None:
    with pytest.raises(ValueError, match=r"embargo_pct must lie in \[0, 1\)"):
        PurgedKFold(n_splits=2, label_end_times=np.arange(10), embargo_pct=bad)


@pytest.mark.parametrize("bad", ["0.1", None, True, [0.1]])
def test_non_numeric_embargo_pct_is_rejected(bad: object) -> None:
    with pytest.raises(ValueError, match="embargo_pct must be a real number"):
        PurgedKFold(n_splits=2, label_end_times=np.arange(10), embargo_pct=bad)


def test_n_splits_validation_is_inherited() -> None:
    with pytest.raises(ValueError, match="at least one train/test split"):
        PurgedKFold(n_splits=1, label_end_times=np.arange(10))


# --------------------------------------------------------------------------
# Label time shape and length
# --------------------------------------------------------------------------


def test_label_end_times_length_must_match_x() -> None:
    cv = PurgedKFold(n_splits=2, label_end_times=np.arange(7))
    with pytest.raises(ValueError, match="has 7 entries but X has 10 samples"):
        list(cv.split(X10))


def test_label_start_times_length_must_match_x() -> None:
    cv = PurgedKFold(
        n_splits=2, label_end_times=np.arange(10), label_start_times=np.arange(4)
    )
    with pytest.raises(ValueError, match="label_start_times has 4 entries"):
        list(cv.split(X10))


def test_two_dimensional_label_times_are_rejected() -> None:
    cv = PurgedKFold(n_splits=2, label_end_times=np.zeros((10, 2)))
    with pytest.raises(ValueError, match="must be one-dimensional"):
        list(cv.split(X10))


@pytest.mark.parametrize(
    "end",
    [
        np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, np.nan]),
        np.array(["2024-01-01", "NaT"] + ["2024-01-03"] * 8, dtype="datetime64[ns]"),
    ],
    ids=["nan", "nat"],
)
def test_missing_label_times_are_rejected(end: np.ndarray) -> None:
    cv = PurgedKFold(n_splits=2, label_end_times=end)
    with pytest.raises(ValueError, match="must not contain missing values"):
        list(cv.split(X10))


# --------------------------------------------------------------------------
# Ordering constraints
# --------------------------------------------------------------------------


def test_unsorted_start_times_are_rejected_not_silently_sorted() -> None:
    """Sorting internally would break the caller's row correspondence with y."""
    start = np.array([0, 1, 2, 3, 4, 5, 6, 9, 8, 7])
    cv = PurgedKFold(
        n_splits=2, label_end_times=start + 1, label_start_times=start
    )
    with pytest.raises(ValueError, match="must be sorted by label start time"):
        list(cv.split(X10))


def test_label_ending_before_it_starts_is_rejected() -> None:
    end = np.arange(10).copy()
    end[5] = 2  # closes three rows before it opens
    cv = PurgedKFold(n_splits=2, label_end_times=end)
    with pytest.raises(ValueError, match="must end at or after it starts"):
        list(cv.split(X10))


def test_ties_in_start_times_are_allowed() -> None:
    """Several observations may share a timestamp; only decreasing order is wrong."""
    start = np.array([0, 0, 1, 1, 2, 2, 3, 3, 4, 4])
    cv = PurgedKFold(
        n_splits=2, label_end_times=start + 1, label_start_times=start, embargo_pct=0.0
    )
    folds = list(cv.split(X10))
    assert len(folds) == 2


def test_incomparable_label_time_dtypes_are_rejected() -> None:
    """Datetime end times against positional integer start times cannot be compared."""
    end = np.array(["2024-01-%02d" % (d + 1) for d in range(10)], dtype="datetime64[ns]")
    cv = PurgedKFold(n_splits=2, label_end_times=end)  # start defaults to 0..9
    with pytest.raises(ValueError, match="must be mutually comparable"):
        list(cv.split(X10))


# --------------------------------------------------------------------------
# Degenerate results
# --------------------------------------------------------------------------


def test_a_fold_with_no_training_rows_left_raises() -> None:
    """Better a named error than an opaque shape failure inside the estimator."""
    end = np.full(10, 9)  # every label runs to the end of the dataset
    cv = PurgedKFold(n_splits=2, label_end_times=end, embargo_pct=0.0)
    with pytest.raises(ValueError, match="Fold 0 has no training samples left"):
        list(cv.split(X10))


def test_x_is_required() -> None:
    cv = PurgedKFold(n_splits=2, label_end_times=np.arange(10))
    with pytest.raises(ValueError, match="should not be None"):
        list(cv.split())


def test_more_splits_than_samples_is_reported_before_label_length() -> None:
    """Whichever the caller got wrong, the n_splits error is the actionable one."""
    cv = PurgedKFold(n_splits=50, label_end_times=np.arange(3))
    with pytest.raises(ValueError, match="greater than the number of samples"):
        list(cv.split(X10))
