"""Boundary arithmetic: what exactly gets purged, and what exactly gets embargoed.

This is the package's product. Every test here pins a specific boundary, and
several are written against a toy dataset small enough to verify by hand rather
than against the implementation's own output at other parameter values. The
expected arrays in :func:`test_hand_computed_twenty_sample_example` were derived
on paper from ``docs/boundary-semantics.md`` before the implementation existed.
"""

from __future__ import annotations

import numpy as np
import pytest
from sklearn.model_selection import KFold

from purged_cv import PurgedKFold
from purged_cv._split import _BaseContiguousKFold

# --------------------------------------------------------------------------
# The hand-computed reference case
# --------------------------------------------------------------------------

# Twenty rows. Row i opens at i and its label closes at i + 2 (clipped at the
# last row), so every label overlaps the two rows after it. Four folds of five.
# Embargo of ceil(20 * 0.10) = 2 rows.
HAND_N = 20
HAND_END = np.minimum(np.arange(HAND_N) + 2, HAND_N - 1)

# fold -> (test, purged, embargoed, train), all worked out by hand.
HAND_EXPECTED = [
    ([0, 1, 2, 3, 4], [5, 6], [7, 8], [9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19]),
    ([5, 6, 7, 8, 9], [3, 4, 10, 11], [12, 13], [0, 1, 2, 14, 15, 16, 17, 18, 19]),
    ([10, 11, 12, 13, 14], [8, 9, 15, 16], [17, 18], [0, 1, 2, 3, 4, 5, 6, 7, 19]),
    ([15, 16, 17, 18, 19], [13, 14], [], [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]),
]


@pytest.fixture
def hand_cv() -> PurgedKFold:
    return PurgedKFold(n_splits=4, label_end_times=HAND_END, embargo_pct=0.10)


@pytest.mark.parametrize("fold", range(4))
def test_hand_computed_twenty_sample_example(hand_cv: PurgedKFold, fold: int) -> None:
    """Every index in every category, checked against arithmetic done on paper."""
    X = np.zeros((HAND_N, 1))
    detail = list(hand_cv.split_detail(X))[fold]
    expected_test, expected_purged, expected_embargoed, expected_train = HAND_EXPECTED[fold]

    assert detail.test.tolist() == expected_test
    assert detail.purged.tolist() == expected_purged
    assert detail.embargoed.tolist() == expected_embargoed
    assert detail.train.tolist() == expected_train


def test_hand_computed_example_via_split(hand_cv: PurgedKFold) -> None:
    """``split`` must agree with the same hand-computed answer."""
    X = np.zeros((HAND_N, 1))
    for (train, test), (exp_test, _, _, exp_train) in zip(hand_cv.split(X), HAND_EXPECTED):
        assert test.tolist() == exp_test
        assert train.tolist() == exp_train


# --------------------------------------------------------------------------
# Purge boundary: touching counts as overlapping
# --------------------------------------------------------------------------

# Ten rows opening at 0..9. The end times are chosen so that each fold has one
# row that touches the test span at exactly one endpoint, and one row that
# misses it by exactly one.
TOUCH_END = np.array([0, 1, 2, 4, 5, 5, 6, 7, 8, 9])


def test_label_ending_exactly_at_test_start_is_purged() -> None:
    """Row 4's label closes at 5, the instant fold 1's test span opens.

    Sharing a boundary instant means sharing information, so it is purged.
    Row 3 closes at 4, strictly before, and survives.
    """
    cv = PurgedKFold(n_splits=2, label_end_times=TOUCH_END, embargo_pct=0.0)
    detail = list(cv.split_detail(np.zeros((10, 1))))[1]

    assert detail.test.tolist() == [5, 6, 7, 8, 9]
    assert 4 in detail.purged.tolist(), "a label ending exactly at test_start must be purged"
    assert 3 in detail.train.tolist(), "a label ending strictly before test_start must be kept"
    assert detail.train.tolist() == [0, 1, 2, 3]


def test_label_starting_exactly_at_test_end_is_purged() -> None:
    """Row 5 opens at 5, exactly when fold 0's test span closes.

    Fold 0's test span closes at ``max(end)`` over rows 0-4, which is 5 --
    *not* at ``max(start)``, which would be 4. A splitter that used the start
    times to bound the span would keep row 5 here, which is the single most
    damaging off-by-one available.
    """
    cv = PurgedKFold(n_splits=2, label_end_times=TOUCH_END, embargo_pct=0.0)
    detail = list(cv.split_detail(np.zeros((10, 1))))[0]

    assert detail.test.tolist() == [0, 1, 2, 3, 4]
    assert detail.purged.tolist() == [5]
    assert detail.train.tolist() == [6, 7, 8, 9]


def test_test_span_upper_edge_follows_the_slowest_label() -> None:
    """One long label in the test fold extends the purged region for everyone."""
    n = 12
    end = np.arange(n).astype(float)
    end[4] = 9.0  # row 4 is in fold 1 and its label runs far into the future
    cv = PurgedKFold(n_splits=3, label_end_times=end, embargo_pct=0.0)
    detail = list(cv.split_detail(np.zeros((n, 1))))[1]

    assert detail.test.tolist() == [4, 5, 6, 7]
    # Rows 8 and 9 open at or before 9, so they overlap row 4's label window.
    assert detail.purged.tolist() == [8, 9]
    assert detail.train.tolist() == [0, 1, 2, 3, 10, 11]


# --------------------------------------------------------------------------
# Embargo arithmetic
# --------------------------------------------------------------------------


def test_embargo_pct_rounds_up_not_down() -> None:
    """150 samples at 1% is 1.5 rows, which must embargo 2 -- not 1.

    The reference implementation truncates here. Truncating leaves a row in
    training that the parameter asked to exclude, so this package rounds up.
    See docs/boundary-semantics.md section 4.3.
    """
    n = 150
    instantaneous = np.arange(n)  # labels close the instant they open
    cv = PurgedKFold(n_splits=3, label_end_times=instantaneous, embargo_pct=0.01)
    detail = list(cv.split_detail(np.zeros((n, 1))))[0]

    assert detail.purged.tolist() == [], "instantaneous labels cannot overlap"
    assert detail.embargoed.tolist() == [50, 51]


@pytest.mark.parametrize(
    "n, embargo_pct, expected",
    [
        (100, 0.0, 0),  # disabled
        (100, 0.05, 5),  # exact
        (100, 0.051, 6),  # 5.1 -> 6
        (150, 0.01, 2),  # 1.5 -> 2
        (100, 0.001, 1),  # 0.1 -> 1: any non-zero embargo covers at least one row
    ],
)
def test_embargo_width(n: int, embargo_pct: float, expected: int) -> None:
    cv = PurgedKFold(n_splits=4, label_end_times=np.arange(n), embargo_pct=embargo_pct)
    first_fold = next(cv.split_detail(np.zeros((n, 1))))
    assert len(first_fold.embargoed) == expected


def test_embargo_starts_after_the_purged_region_not_after_the_test_block() -> None:
    """With overlapping labels the two regions must abut, never overlap.

    Fold 0 tests rows 0-4 whose labels run to row 6, so purging takes rows 5
    and 6. The embargo must then start at row 7 -- if it were anchored to the
    end of the test block it would start at row 5, land inside the purged
    region, and remove nothing that purging had not already removed.
    """
    cv = PurgedKFold(n_splits=4, label_end_times=HAND_END, embargo_pct=0.10)
    detail = list(cv.split_detail(np.zeros((HAND_N, 1))))[0]

    assert detail.purged.tolist() == [5, 6]
    assert detail.embargoed.tolist() == [7, 8]
    assert max(detail.purged) < min(detail.embargoed)


def test_embargo_is_forward_only() -> None:
    """Nothing is embargoed before the test span; purging already covers that side."""
    n = 40
    cv = PurgedKFold(n_splits=4, label_end_times=np.arange(n), embargo_pct=0.10)
    for detail in cv.split_detail(np.zeros((n, 1))):
        if len(detail.embargoed):
            assert min(detail.embargoed) > max(detail.test)


def test_embargo_truncated_at_the_end_of_the_data() -> None:
    """The final fold has nothing after it to embargo."""
    n = 40
    cv = PurgedKFold(n_splits=4, label_end_times=np.arange(n), embargo_pct=0.10)
    assert list(cv.split_detail(np.zeros((n, 1))))[-1].embargoed.tolist() == []


# --------------------------------------------------------------------------
# Relationship to the unpurged splitter
# --------------------------------------------------------------------------


@pytest.mark.parametrize("n_samples, n_splits", [(10, 5), (13, 5), (100, 7), (97, 3)])
def test_instantaneous_labels_and_no_embargo_reduce_to_plain_kfold(
    n_samples: int, n_splits: int
) -> None:
    """The degenerate case must be exactly ``KFold(shuffle=False)``.

    If every label closes the instant it opens there is no overlap to purge,
    so with the embargo disabled purged-cv must return sklearn's own splits
    untouched. This is the bridge between the unpurged base class and the
    purging subclass: it proves purging adds nothing when nothing is at stake.
    """
    X = np.zeros((n_samples, 2))
    cv = PurgedKFold(
        n_splits=n_splits, label_end_times=np.arange(n_samples), embargo_pct=0.0
    )
    for (ours_train, ours_test), (theirs_train, theirs_test) in zip(
        cv.split(X), KFold(n_splits=n_splits, shuffle=False).split(X)
    ):
        np.testing.assert_array_equal(ours_test, theirs_test)
        np.testing.assert_array_equal(ours_train, theirs_train)


@pytest.mark.parametrize("embargo_pct", [0.0, 0.02, 0.1])
def test_purging_only_ever_removes_training_rows(embargo_pct: float) -> None:
    """Test folds are untouched, and training is always a subset of the unpurged one."""
    n = 60
    X = np.zeros((n, 2))
    end = np.minimum(np.arange(n) + 4, n - 1)
    purged = PurgedKFold(n_splits=5, label_end_times=end, embargo_pct=embargo_pct)
    base = _BaseContiguousKFold(n_splits=5)

    for (p_train, p_test), (b_train, b_test) in zip(purged.split(X), base.split(X)):
        np.testing.assert_array_equal(p_test, b_test)
        assert set(p_train) <= set(b_train)


def test_longer_label_windows_purge_at_least_as_much() -> None:
    """Monotonicity: a wider label window can never shrink the purged set."""
    n = 60
    X = np.zeros((n, 2))
    previous: set[int] = set()
    for horizon in [0, 1, 3, 8, 15]:
        end = np.minimum(np.arange(n) + horizon, n - 1)
        cv = PurgedKFold(n_splits=5, label_end_times=end, embargo_pct=0.0)
        purged = {int(i) for detail in cv.split_detail(X) for i in detail.purged}
        assert previous <= purged, f"horizon {horizon} purged less than the one before"
        previous = purged
