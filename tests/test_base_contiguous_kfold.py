"""Step 1: prove the fold layout and the scikit-learn integration.

These tests deliberately contain no purging. Their only job is to establish
that the splitter is genuinely drop-in for scikit-learn *before* any
leakage-prevention logic is layered on, so that a later failure can be
attributed to the purge/embargo arithmetic rather than to the plumbing.
"""

from __future__ import annotations

import numpy as np
import pytest
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import BaseCrossValidator, GridSearchCV, KFold, cross_val_score
from sklearn.model_selection._split import check_cv
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from purged_cv._split import _BaseContiguousKFold, _n_samples


@pytest.fixture
def data() -> tuple[np.ndarray, np.ndarray]:
    return make_classification(n_samples=100, n_features=5, random_state=0)


# --------------------------------------------------------------------------
# Fold layout
# --------------------------------------------------------------------------


@pytest.mark.parametrize("n_samples", [10, 11, 12, 13, 14, 97, 100])
@pytest.mark.parametrize("n_splits", [2, 3, 5, 7])
def test_folds_match_sklearn_kfold_exactly(n_samples: int, n_splits: int) -> None:
    """Without purging the splits must be identical to KFold(shuffle=False).

    This is the anchor for the whole package: any later difference from these
    indices is caused by purging or embargo and by nothing else.
    """
    X = np.zeros((n_samples, 3))
    ours = list(_BaseContiguousKFold(n_splits=n_splits).split(X))
    theirs = list(KFold(n_splits=n_splits, shuffle=False).split(X))

    assert len(ours) == len(theirs)
    for (our_train, our_test), (their_train, their_test) in zip(ours, theirs):
        np.testing.assert_array_equal(our_test, their_test)
        np.testing.assert_array_equal(our_train, their_train)


@pytest.mark.parametrize("n_samples, n_splits", [(10, 5), (13, 5), (100, 7)])
def test_test_folds_partition_the_data_in_time_order(n_samples: int, n_splits: int) -> None:
    """Every sample is tested exactly once, and folds advance through time."""
    X = np.zeros((n_samples, 2))
    tests = [test for _, test in _BaseContiguousKFold(n_splits=n_splits).split(X)]

    np.testing.assert_array_equal(np.sort(np.concatenate(tests)), np.arange(n_samples))
    for test in tests:
        # contiguous block, ascending
        np.testing.assert_array_equal(test, np.arange(test[0], test[-1] + 1))
    for earlier, later in zip(tests, tests[1:]):
        assert earlier[-1] < later[0], "test folds must move forward through the index"


@pytest.mark.parametrize("n_samples, n_splits", [(10, 5), (13, 5), (11, 3)])
def test_fold_sizes_differ_by_at_most_one(n_samples: int, n_splits: int) -> None:
    X = np.zeros((n_samples, 2))
    sizes = [len(test) for _, test in _BaseContiguousKFold(n_splits=n_splits).split(X)]
    assert sum(sizes) == n_samples
    assert max(sizes) - min(sizes) <= 1


def test_train_is_the_complement_of_test() -> None:
    """No purging yet: train must be exactly everything that is not test."""
    X = np.zeros((20, 2))
    for train, test in _BaseContiguousKFold(n_splits=4).split(X):
        assert set(train).isdisjoint(test)
        assert set(train) | set(test) == set(range(20))


def test_split_is_repeatable() -> None:
    X = np.zeros((23, 2))
    cv = _BaseContiguousKFold(n_splits=4)
    first = [(tr.copy(), te.copy()) for tr, te in cv.split(X)]
    second = list(cv.split(X))
    for (tr1, te1), (tr2, te2) in zip(first, second):
        np.testing.assert_array_equal(tr1, tr2)
        np.testing.assert_array_equal(te1, te2)


# --------------------------------------------------------------------------
# scikit-learn interface conformance
# --------------------------------------------------------------------------


def test_is_a_sklearn_cross_validator() -> None:
    cv = _BaseContiguousKFold(n_splits=3)
    assert isinstance(cv, BaseCrossValidator)
    assert check_cv(cv) is cv


def test_get_n_splits_accepts_the_sklearn_signature(data) -> None:
    X, y = data
    cv = _BaseContiguousKFold(n_splits=6)
    assert cv.get_n_splits() == 6
    assert cv.get_n_splits(X) == 6
    assert cv.get_n_splits(X, y) == 6
    assert cv.get_n_splits(X, y, None) == 6  # groups passed positionally


def test_split_accepts_y_and_groups_positionally(data) -> None:
    X, y = data
    groups = np.zeros(len(y))
    splits = list(_BaseContiguousKFold(n_splits=3).split(X, y, groups))
    assert len(splits) == 3


def test_repr_works() -> None:
    """sklearn's ``_build_repr`` reflects over ``__init__`` and reads attributes back."""
    assert repr(_BaseContiguousKFold(n_splits=4)) == "_BaseContiguousKFold(n_splits=4)"


def test_index_type_does_not_matter() -> None:
    """A DataFrame with a DatetimeIndex must split identically to a bare array."""
    pd = pytest.importorskip("pandas")
    n = 30
    array = np.arange(n * 2, dtype=float).reshape(n, 2)
    frame = pd.DataFrame(array, index=pd.date_range("2024-01-01", periods=n, freq="D"))

    cv = _BaseContiguousKFold(n_splits=5)
    for (tr_a, te_a), (tr_b, te_b) in zip(cv.split(array), cv.split(frame)):
        np.testing.assert_array_equal(tr_a, tr_b)
        np.testing.assert_array_equal(te_a, te_b)


# --------------------------------------------------------------------------
# Drop-in behaviour inside scikit-learn's own machinery
# --------------------------------------------------------------------------


def test_works_inside_cross_val_score(data) -> None:
    X, y = data
    scores = cross_val_score(LogisticRegression(), X, y, cv=_BaseContiguousKFold(n_splits=5))
    assert scores.shape == (5,)
    assert np.all(np.isfinite(scores))


def test_scores_identical_to_kfold_shuffle_false(data) -> None:
    """The strongest drop-in check: same estimator, same data, same numbers."""
    X, y = data
    est = make_pipeline(StandardScaler(), LogisticRegression())
    ours = cross_val_score(est, X, y, cv=_BaseContiguousKFold(n_splits=5))
    theirs = cross_val_score(est, X, y, cv=KFold(n_splits=5, shuffle=False))
    np.testing.assert_allclose(ours, theirs)


def test_works_inside_grid_search(data) -> None:
    X, y = data
    search = GridSearchCV(
        LogisticRegression(),
        {"C": [0.1, 1.0]},
        cv=_BaseContiguousKFold(n_splits=3),
    )
    search.fit(X, y)
    assert search.n_splits_ == 3
    assert search.best_params_["C"] in (0.1, 1.0)


# --------------------------------------------------------------------------
# Validation
# --------------------------------------------------------------------------


@pytest.mark.parametrize("bad", [1, 0, -3])
def test_n_splits_below_two_is_rejected(bad: int) -> None:
    with pytest.raises(ValueError, match="at least one train/test split"):
        _BaseContiguousKFold(n_splits=bad)


@pytest.mark.parametrize("bad", [2.0, "5", None, True])
def test_non_integer_n_splits_is_rejected(bad: object) -> None:
    with pytest.raises(ValueError, match="Integral type"):
        _BaseContiguousKFold(n_splits=bad)  # type: ignore[arg-type]


def test_more_splits_than_samples_is_rejected() -> None:
    X = np.zeros((3, 2))
    with pytest.raises(ValueError, match="greater than the number of samples"):
        list(_BaseContiguousKFold(n_splits=5).split(X))


def test_n_samples_helper_matches_sklearn() -> None:
    from sklearn.utils.validation import _num_samples

    for obj in [np.zeros((7, 3)), [[1], [2], [3]], np.arange(4)]:
        assert _n_samples(obj) == _num_samples(obj)


def test_zero_dimensional_input_is_rejected() -> None:
    with pytest.raises(TypeError, match="Singleton array"):
        _n_samples(np.array(5))


def test_iter_test_indices_requires_X() -> None:
    """Reachable only by calling the hook directly; ``split`` guards X earlier."""
    with pytest.raises(ValueError, match="should not be None"):
        list(_BaseContiguousKFold(n_splits=3)._iter_test_indices(None))
