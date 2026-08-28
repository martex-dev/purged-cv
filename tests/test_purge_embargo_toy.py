"""The hand-verifiable toy example. Written before the implementation.

Twenty samples, integer index, worked out by hand. Every expected index set
below was derived on paper from the four conventions recorded in
``docs/boundary-semantics.md`` and reproduced here, *not* read back out of the
implementation. If a change to the splitter makes one of these fail, the
splitter is wrong until proven otherwise -- these numbers are the spec.

The dataset
-----------
Sample ``i`` is observed at ``t0 = i`` and its label closes at ``t1[i]``::

    i  :  0  1   2  3  4  5  6  7  8  9 10 11 12 13 14 15 16 17 18 19
    t1 :  0  2  13  4  5  8  9 10 11 12 13 14 15 16 17 18 19 20 21 21

Four values do specific work:

``t1[4] = 5``
    A training sample whose label closes *exactly* when fold 2's test window
    opens at ``t0 = 5``. Under the closed-interval convention it is purged.
    This single value is what distinguishes our convention from a half-open
    one, so it is the most load-bearing number in the file.
``t1[2] = 13``
    Starts before fold 2 and ends after it: the label window *envelops* the
    test window. This is the third of the three overlap cases in Snippet 7.1
    and the one hand-rolled implementations most often drop.
``t1[2] = 13`` alongside ``t1[3] = 4``
    Label end times are deliberately **non-monotonic**. The book's Snippet 7.3
    locates the purge boundary with ``searchsorted``, which is only valid for a
    sorted ``t1`` -- i.e. only for a constant label horizon. This fixture is the
    permanent regression guard against reintroducing that bug.
``t1[0] = 0`` and ``t1[18] = t1[19] = 21``
    A zero-length label window, and labels that run off the end of the data as
    the most recent labels always do in practice.

Conventions applied
-------------------
(a) Label windows are **closed**: ``[t0, t1]``. Two windows overlap if they
    intersect inclusively, so touching endpoints count as overlap and are
    purged. Stricter than a half-open convention by exactly one sample here.
(b) The test window for a fold is a **single span**
    ``[min(t0 over test), max(t1 over test)]``. Test folds are contiguous
    blocks, so this is identical to the union of per-sample windows.
(c) The embargo is **one-sided (forward only)** and anchored **after the purge
    boundary**: training resumes at the first row that survives purging on the
    right of the test fold, and the embargo then removes the next ``mbrg`` rows
    from there. Purging already handles the backward direction.
(d) ``mbrg = ceil(embargo_pct * n_samples)``. The book floors; we round up, on
    the CLAUDE.md rule that rounding must never leave a boundary sample in.
    With ``embargo_pct=0.10`` and 20 samples this is 2 either way, so the
    floor/ceil question is isolated in its own test rather than tangled in
    here.
"""

from __future__ import annotations

import numpy as np
import pytest

from purged_cv import PurgedKFold

N_SAMPLES = 20
N_SPLITS = 4

# t0 is implicit: sample i is observed at time i (the default sample_times).
LABEL_END_TIMES = np.array(
    [0, 2, 13, 4, 5, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 21]
)

X = np.zeros((N_SAMPLES, 2))

# Fixed by Step 1: four contiguous folds of five.
EXPECTED_TEST_FOLDS = [
    [0, 1, 2, 3, 4],
    [5, 6, 7, 8, 9],
    [10, 11, 12, 13, 14],
    [15, 16, 17, 18, 19],
]

# ---------------------------------------------------------------------------
# Purging only (embargo_pct = 0.0)
# ---------------------------------------------------------------------------
#
# Fold 1  test {0..4}   span [0, max(0,2,13,4,5)] = [0, 13]
#         Only a right side exists. A row is purged when t0 <= 13, so 5..13 go
#         and 14..19 stay.
#
# Fold 2  test {5..9}   span [5, max(8,9,10,11,12)] = [5, 12]
#         Left:  keep unless t1 >= 5.  i=0 (t1=0) keep, i=1 (t1=2) keep,
#                i=2 (t1=13) PURGE - envelops the span,
#                i=3 (t1=4) keep, i=4 (t1=5) PURGE - closed-interval boundary.
#         Right: purge while t0 <= 12, so 10,11,12 go and 13..19 stay.
#
# Fold 3  test {10..14} span [10, max(13,14,15,16,17)] = [10, 17]
#         Left:  purge when t1 >= 10 -> i=2 (13), i=7 (10), i=8 (11), i=9 (12).
#                Note i=7 is purged on a touching endpoint, same rule as i=4.
#         Right: purge while t0 <= 17 -> 15,16,17 go; 18,19 stay.
#
# Fold 4  test {15..19} span [15, max(18,19,20,21,21)] = [15, 21]
#         No right side. Left: purge when t1 >= 15 -> i=12 (15), i=13 (16),
#         i=14 (17). Everything from 0..11 survives.
EXPECTED_TRAIN_PURGE_ONLY = [
    [14, 15, 16, 17, 18, 19],
    [0, 1, 3, 13, 14, 15, 16, 17, 18, 19],
    [0, 1, 3, 4, 5, 6, 18, 19],
    [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11],
]

# ---------------------------------------------------------------------------
# Purging + embargo (embargo_pct = 0.10 -> mbrg = 2 rows)
# ---------------------------------------------------------------------------
#
# The embargo touches only the right side, and only the rows that survived
# purging there.
#
# Fold 1  right side resumes at 14 -> embargo removes 14, 15 -> 16..19 remain.
# Fold 2  right side resumes at 13 -> embargo removes 13, 14 -> 15..19 remain.
#         The left side is untouched: {0, 1, 3}.
# Fold 3  right side resumes at 18 -> embargo removes 18, 19 -> nothing remains
#         on the right at all. This fold is the reason the implementation has
#         to cope with an empty right-hand side.
# Fold 4  has no right side, so the embargo is inert and the train set is
#         exactly the purge-only one.
#
# Fold 2 is also what discriminates convention (c) from the casual reading: an
# embargo anchored on the last *test* index would drop rows 10 and 11, which
# purging has already removed, and would therefore do nothing whatsoever.
EXPECTED_TRAIN_WITH_EMBARGO = [
    [16, 17, 18, 19],
    [0, 1, 3, 15, 16, 17, 18, 19],
    [0, 1, 3, 4, 5, 6],
    [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11],
]


def test_test_folds_are_unchanged_by_purging() -> None:
    """Purging removes training rows only. Test folds must still tile the data."""
    cv = PurgedKFold(n_splits=N_SPLITS, label_end_times=LABEL_END_TIMES, embargo_pct=0.10)
    folds = [test.tolist() for _, test in cv.split(X)]
    assert folds == EXPECTED_TEST_FOLDS


@pytest.mark.parametrize("fold", range(N_SPLITS))
def test_purge_only_matches_hand_computation(fold: int) -> None:
    cv = PurgedKFold(n_splits=N_SPLITS, label_end_times=LABEL_END_TIMES, embargo_pct=0.0)
    train, _ = list(cv.split(X))[fold]
    assert train.tolist() == EXPECTED_TRAIN_PURGE_ONLY[fold]


@pytest.mark.parametrize("fold", range(N_SPLITS))
def test_purge_plus_embargo_matches_hand_computation(fold: int) -> None:
    cv = PurgedKFold(n_splits=N_SPLITS, label_end_times=LABEL_END_TIMES, embargo_pct=0.10)
    train, _ = list(cv.split(X))[fold]
    assert train.tolist() == EXPECTED_TRAIN_WITH_EMBARGO[fold]


def test_the_touching_boundary_sample_is_purged() -> None:
    """Sample 4's label closes exactly as fold 2's test window opens.

    Called out on its own because it is the one sample that a half-open
    convention would keep, and keeping it is precisely the quiet leak this
    package exists to prevent.
    """
    cv = PurgedKFold(n_splits=N_SPLITS, label_end_times=LABEL_END_TIMES, embargo_pct=0.0)
    train, test = list(cv.split(X))[1]
    assert test[0] == 5
    assert LABEL_END_TIMES[4] == 5
    assert 4 not in train


def test_the_enveloping_sample_is_purged() -> None:
    """Sample 2 starts before fold 2 and ends after it -- Snippet 7.1's third case."""
    cv = PurgedKFold(n_splits=N_SPLITS, label_end_times=LABEL_END_TIMES, embargo_pct=0.0)
    train, test = list(cv.split(X))[1]
    assert 2 < test[0] and LABEL_END_TIMES[2] > LABEL_END_TIMES[test].max()
    assert 2 not in train


def test_non_monotonic_label_end_times_are_handled() -> None:
    """The fixture is non-monotonic; a searchsorted-based purge would be wrong here."""
    assert LABEL_END_TIMES[2] > LABEL_END_TIMES[3], "fixture must stay non-monotonic"
    cv = PurgedKFold(n_splits=N_SPLITS, label_end_times=LABEL_END_TIMES, embargo_pct=0.0)
    for train, test in cv.split(X):
        span_start = test.min()
        span_end = LABEL_END_TIMES[test].max()
        overlaps = (LABEL_END_TIMES[train] >= span_start) & (train <= span_end)
        assert not overlaps.any()


def test_zero_embargo_is_the_same_as_no_embargo() -> None:
    a = PurgedKFold(n_splits=N_SPLITS, label_end_times=LABEL_END_TIMES, embargo_pct=0.0)
    b = PurgedKFold(n_splits=N_SPLITS, label_end_times=LABEL_END_TIMES)
    for (tr_a, _), (tr_b, _) in zip(a.split(X), b.split(X)):
        np.testing.assert_array_equal(tr_a, tr_b)


def test_embargo_only_removes_rows_after_the_test_fold() -> None:
    """The embargo is forward-only; purging is what protects the backward side."""
    no_emb = PurgedKFold(n_splits=N_SPLITS, label_end_times=LABEL_END_TIMES, embargo_pct=0.0)
    with_emb = PurgedKFold(n_splits=N_SPLITS, label_end_times=LABEL_END_TIMES, embargo_pct=0.10)
    for (tr_a, test), (tr_b, _) in zip(no_emb.split(X), with_emb.split(X)):
        removed = set(tr_a.tolist()) - set(tr_b.tolist())
        assert all(i > test.max() for i in removed)
