"""Drop-in compatibility with scikit-learn, and index-type independence.

"sklearn-compatible" is the value proposition, so it is tested against sklearn's
own machinery rather than against an assumption about what that machinery does.
"""

from __future__ import annotations

import numpy as np
import pytest
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import (
    BaseCrossValidator,
    GridSearchCV,
    cross_val_score,
    cross_validate,
)
from sklearn.model_selection._split import check_cv
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from purged_cv import PurgedKFold

N = 120


@pytest.fixture
def data() -> tuple[np.ndarray, np.ndarray]:
    return make_classification(n_samples=N, n_features=5, random_state=0)


@pytest.fixture
def cv() -> PurgedKFold:
    return PurgedKFold(
        n_splits=5,
        label_end_times=np.minimum(np.arange(N) + 5, N - 1),
        embargo_pct=0.02,
    )


# --------------------------------------------------------------------------
# Interface conformance
# --------------------------------------------------------------------------


def test_is_a_sklearn_cross_validator(cv: PurgedKFold) -> None:
    assert isinstance(cv, BaseCrossValidator)
    assert check_cv(cv) is cv


def test_get_n_splits_accepts_the_sklearn_signature(
    cv: PurgedKFold, data: tuple[np.ndarray, np.ndarray]
) -> None:
    X, y = data
    assert cv.get_n_splits() == 5
    assert cv.get_n_splits(X, y, None) == 5


def test_split_accepts_y_and_groups_positionally(
    cv: PurgedKFold, data: tuple[np.ndarray, np.ndarray]
) -> None:
    X, y = data
    assert len(list(cv.split(X, y, np.zeros(len(y))))) == 5


def test_repr_does_not_raise(cv: PurgedKFold) -> None:
    """sklearn's ``_build_repr`` reflects over ``__init__``, including keyword-only args."""
    text = repr(cv)
    assert text.startswith("PurgedKFold(")
    assert "embargo_pct=0.02" in text


def test_number_of_yielded_folds_matches_get_n_splits(
    cv: PurgedKFold, data: tuple[np.ndarray, np.ndarray]
) -> None:
    X, _ = data
    assert len(list(cv.split(X))) == cv.get_n_splits(X)


# --------------------------------------------------------------------------
# Inside sklearn's own machinery
# --------------------------------------------------------------------------


def test_works_inside_cross_val_score(
    cv: PurgedKFold, data: tuple[np.ndarray, np.ndarray]
) -> None:
    X, y = data
    scores = cross_val_score(LogisticRegression(), X, y, cv=cv)
    assert scores.shape == (5,)
    assert np.all(np.isfinite(scores))


def test_works_inside_cross_validate(
    cv: PurgedKFold, data: tuple[np.ndarray, np.ndarray]
) -> None:
    X, y = data
    result = cross_validate(LogisticRegression(), X, y, cv=cv, return_train_score=True)
    assert len(result["test_score"]) == 5
    assert np.all(np.isfinite(result["train_score"]))


def test_works_inside_grid_search(
    cv: PurgedKFold, data: tuple[np.ndarray, np.ndarray]
) -> None:
    X, y = data
    search = GridSearchCV(LogisticRegression(), {"C": [0.1, 1.0]}, cv=cv)
    search.fit(X, y)
    assert search.n_splits_ == 5
    assert search.best_params_["C"] in (0.1, 1.0)


def test_works_with_a_pipeline(cv: PurgedKFold, data: tuple[np.ndarray, np.ndarray]) -> None:
    X, y = data
    est = make_pipeline(StandardScaler(), LogisticRegression())
    assert np.all(np.isfinite(cross_val_score(est, X, y, cv=cv)))


def test_training_folds_are_smaller_than_unpurged_ones(
    cv: PurgedKFold, data: tuple[np.ndarray, np.ndarray]
) -> None:
    """Guards against the splitter being wired in but silently purging nothing."""
    X, y = data
    for train, test in cv.split(X, y):
        assert len(train) < N - len(test)


# --------------------------------------------------------------------------
# Index types
# --------------------------------------------------------------------------


def test_datetime_index_via_pandas_series() -> None:
    """A Series of end times indexed by start times -- AFML's ``t1`` layout."""
    pd = pytest.importorskip("pandas")
    n = 60
    idx = pd.date_range("2024-01-01", periods=n, freq="D")
    end = pd.Series(idx + pd.Timedelta(days=6), index=idx)

    cv = PurgedKFold(n_splits=5, label_end_times=end, embargo_pct=0.05)
    details = list(cv.split_detail(np.zeros((n, 1))))

    assert len(details) == 5
    assert any(len(d.purged) for d in details), "a 6-day label window must purge something"
    assert any(len(d.embargoed) for d in details)


def test_datetime_and_positional_indices_agree_on_the_same_geometry() -> None:
    """Daily timestamps with a 6-day window must split like integers with a 6-row window."""
    pd = pytest.importorskip("pandas")
    n = 60
    idx = pd.date_range("2024-01-01", periods=n, freq="D")

    by_time = PurgedKFold(
        n_splits=5,
        label_end_times=pd.Series(idx + pd.Timedelta(days=6), index=idx),
        embargo_pct=0.05,
    )
    by_position = PurgedKFold(
        n_splits=5, label_end_times=np.arange(n) + 6, embargo_pct=0.05
    )
    X = np.zeros((n, 1))
    for left, right in zip(by_time.split_detail(X), by_position.split_detail(X)):
        for a, b in zip(left, right):
            np.testing.assert_array_equal(a, b)


def test_explicit_start_times_override_a_series_index() -> None:
    pd = pytest.importorskip("pandas")
    n = 40
    start = np.arange(n)
    series = pd.Series(start + 3, index=pd.RangeIndex(1000, 1000 + n))  # misleading index

    explicit = PurgedKFold(
        n_splits=4, label_end_times=series, label_start_times=start, embargo_pct=0.0
    )
    plain = PurgedKFold(n_splits=4, label_end_times=start + 3, embargo_pct=0.0)
    X = np.zeros((n, 1))
    for left, right in zip(explicit.split(X), plain.split(X)):
        np.testing.assert_array_equal(left[0], right[0])
        np.testing.assert_array_equal(left[1], right[1])


def test_dataframe_input_splits_like_an_array() -> None:
    """Only the row count of X is used; its own index is never consulted."""
    pd = pytest.importorskip("pandas")
    n = 40
    array = np.arange(n * 2, dtype=float).reshape(n, 2)
    frame = pd.DataFrame(array, index=pd.date_range("2020-06-01", periods=n, freq="D"))

    cv = PurgedKFold(n_splits=4, label_end_times=np.arange(n) + 2, embargo_pct=0.05)
    for left, right in zip(cv.split(array), cv.split(frame)):
        np.testing.assert_array_equal(left[0], right[0])
        np.testing.assert_array_equal(left[1], right[1])
