"""Accounting invariants for ``split_detail``.

``split()`` drops purged and embargoed rows indistinguishably -- both are simply
absent from ``train``. ``split_detail`` attributes them, which is what lets a
downstream consumer (cv-visualizer) render the two as separate bands without
re-deriving this package's boundary arithmetic.

The invariants asserted here are the ones ``API_NOTES.md`` names as the
downstream contract.
"""

from __future__ import annotations

import numpy as np
import pytest

from purged_cv import FoldDetail, PurgedKFold

PARAMS = [
    (60, 5, 0, 0.0),
    (60, 5, 4, 0.0),
    (60, 5, 4, 0.05),
    (60, 3, 10, 0.10),
    (97, 7, 6, 0.02),
    (20, 4, 2, 0.10),
]


def _cv(n: int, n_splits: int, horizon: int, embargo_pct: float) -> PurgedKFold:
    end = np.minimum(np.arange(n) + horizon, n - 1)
    return PurgedKFold(n_splits=n_splits, label_end_times=end, embargo_pct=embargo_pct)


@pytest.mark.parametrize("n, n_splits, horizon, embargo_pct", PARAMS)
def test_categories_are_pairwise_disjoint(
    n: int, n_splits: int, horizon: int, embargo_pct: float
) -> None:
    for detail in _cv(n, n_splits, horizon, embargo_pct).split_detail(np.zeros((n, 1))):
        sets = [set(a.tolist()) for a in detail]
        for i, left in enumerate(sets):
            for right in sets[i + 1 :]:
                assert left.isdisjoint(right)


@pytest.mark.parametrize("n, n_splits, horizon, embargo_pct", PARAMS)
def test_categories_account_for_every_row(
    n: int, n_splits: int, horizon: int, embargo_pct: float
) -> None:
    """Nothing is unattributed: PurgedKFold has no 'not yet reached' category.

    Unlike ``TimeSeriesSplit``, every row is either trained on, tested on,
    purged or embargoed in every fold. ``API_NOTES.md`` leaves room for a fifth
    'unused' category; this asserts that it stays empty here, so cv-visualizer
    computing it as a set difference will correctly find nothing.
    """
    for detail in _cv(n, n_splits, horizon, embargo_pct).split_detail(np.zeros((n, 1))):
        union = set().union(*(set(a.tolist()) for a in detail))
        assert union == set(range(n))


@pytest.mark.parametrize("n, n_splits, horizon, embargo_pct", PARAMS)
def test_detail_train_and_test_match_split_exactly(
    n: int, n_splits: int, horizon: int, embargo_pct: float
) -> None:
    """The important one: the detail view must describe the folds people train on.

    Both methods are driven by a single computation, so this is structurally
    guaranteed rather than coincidental -- but it is the guarantee cv-visualizer
    relies on to claim it shows what ``split`` really did, so it is asserted.
    """
    cv = _cv(n, n_splits, horizon, embargo_pct)
    X = np.zeros((n, 1))
    for (train, test), detail in zip(cv.split(X), cv.split_detail(X)):
        np.testing.assert_array_equal(train, detail.train)
        np.testing.assert_array_equal(test, detail.test)


@pytest.mark.parametrize("n, n_splits, horizon, embargo_pct", PARAMS)
def test_purged_and_embargoed_never_overlap(
    n: int, n_splits: int, horizon: int, embargo_pct: float
) -> None:
    """The precedence question in API_NOTES.md never has to be answered.

    Because the embargo is anchored to the first row that survived purging
    rather than to the end of the test block, a row can never be both. The
    documented 'purge wins' rule stands in case that anchor ever changes; this
    test is what would catch the change.
    """
    for detail in _cv(n, n_splits, horizon, embargo_pct).split_detail(np.zeros((n, 1))):
        assert set(detail.purged.tolist()).isdisjoint(detail.embargoed.tolist())


def test_fold_detail_is_a_named_tuple() -> None:
    """Downstream code may unpack it positionally or by name."""
    cv = _cv(20, 4, 2, 0.10)
    detail = next(cv.split_detail(np.zeros((20, 1))))
    assert isinstance(detail, FoldDetail)
    train, test, purged, embargoed = detail
    np.testing.assert_array_equal(train, detail.train)
    np.testing.assert_array_equal(test, detail.test)
    np.testing.assert_array_equal(purged, detail.purged)
    np.testing.assert_array_equal(embargoed, detail.embargoed)


def test_split_detail_is_repeatable() -> None:
    cv = _cv(40, 4, 3, 0.05)
    X = np.zeros((40, 1))
    first = [FoldDetail(*(a.copy() for a in d)) for d in cv.split_detail(X)]
    for before, after in zip(first, cv.split_detail(X)):
        for a, b in zip(before, after):
            np.testing.assert_array_equal(a, b)


def test_duck_typing_contract_used_by_cv_visualizer() -> None:
    """cv-visualizer probes for the method rather than the type."""
    cv = _cv(20, 4, 2, 0.10)
    assert callable(getattr(cv, "split_detail", None))
