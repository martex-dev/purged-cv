"""The proof the package does what it claims.

This is the single most important test in the repository. Everything else checks
that the arithmetic is what was intended; this checks that the intended
arithmetic actually removes leakage.

The construction
----------------
Returns are i.i.d. Gaussian noise -- there is no signal to find, and the honest
score for any model is 0.5. Each label is the sign of the cumulative return over
the following ``HORIZON`` steps, so labels computed at adjacent times share
almost all of their inputs and are therefore highly correlated with one another.

The feature is a single slowly-varying column (here, elapsed time). It carries
no information about the future whatsoever. Its only role is to make
"temporally nearby" mean "nearby in feature space", so that a 1-nearest-
neighbour model, asked to predict a test row, will reach for the training row
closest in time. That is the leakage channel this package exists to close:
if the nearest training row's label window overlaps the test fold, the model
recovers the test answer without learning anything.

Three splitters are compared on identical data:

* ``KFold(shuffle=True)`` -- what a practitioner reaches for by default. Test
  rows are surrounded by training rows whose labels overlap theirs almost
  completely. Score collapses towards 1.0.
* ``_BaseContiguousKFold`` -- the same contiguous fold layout this package
  uses, with no purging. Only rows near each fold boundary leak, so the
  inflation is smaller but still large.
* ``PurgedKFold`` -- identical fold layout, plus purging and embargo. The
  overlapping neighbours are removed and the score returns to chance.

The second and third differ in exactly one respect, which is what makes the
comparison attributable to purging rather than to fold geometry.
"""

from __future__ import annotations

import numpy as np
import pytest
from sklearn.model_selection import KFold, cross_val_score
from sklearn.neighbors import KNeighborsClassifier

from purged_cv import PurgedKFold
from purged_cv._split import _BaseContiguousKFold

N_SAMPLES = 1200
HORIZON = 50
N_SPLITS = 24
EMBARGO_PCT = 0.01
N_SEEDS = 12

CHANCE = 0.5


def make_no_signal_dataset(seed: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Pure noise with overlapping labels. Returns ``(X, y, label_end_times)``."""
    rng = np.random.default_rng(seed)
    returns = rng.normal(size=N_SAMPLES + HORIZON + 1)
    cumulative = np.concatenate([[0.0], np.cumsum(returns)])

    # y[i] is the sign of the return over (i, i + HORIZON]: consecutive labels
    # share HORIZON - 1 of their HORIZON terms.
    forward = cumulative[1 + HORIZON : N_SAMPLES + 1 + HORIZON] - cumulative[1 : N_SAMPLES + 1]
    y = (forward > 0).astype(int)

    X = np.arange(N_SAMPLES, dtype=float).reshape(-1, 1)
    label_end_times = np.minimum(np.arange(N_SAMPLES) + HORIZON, N_SAMPLES - 1)
    return X, y, label_end_times


def _score(model, X, y, cv) -> float:
    return float(cross_val_score(model, X, y, cv=cv).mean())


@pytest.fixture(scope="module")
def scores() -> dict[str, np.ndarray]:
    """Mean CV accuracy per seed under each splitter."""
    collected: dict[str, list[float]] = {"shuffled": [], "contiguous": [], "purged": []}
    for seed in range(N_SEEDS):
        X, y, end = make_no_signal_dataset(seed)
        model = KNeighborsClassifier(n_neighbors=1)
        collected["shuffled"].append(
            _score(model, X, y, KFold(N_SPLITS, shuffle=True, random_state=seed))
        )
        collected["contiguous"].append(_score(model, X, y, _BaseContiguousKFold(N_SPLITS)))
        collected["purged"].append(
            _score(
                model,
                X,
                y,
                PurgedKFold(
                    n_splits=N_SPLITS, label_end_times=end, embargo_pct=EMBARGO_PCT
                ),
            )
        )
    return {name: np.array(values) for name, values in collected.items()}


def test_the_data_really_has_no_signal() -> None:
    """Labels must be near balanced and unpredictable from the feature by construction.

    If this fails the rest of the module proves nothing, because a genuine
    signal would let every splitter score above chance honestly.
    """
    balances = [make_no_signal_dataset(seed)[1].mean() for seed in range(N_SEEDS)]
    assert 0.35 < float(np.mean(balances)) < 0.65
    # The feature is elapsed time and the labels derive from independent noise,
    # so no monotone relationship exists between them.
    X, y, _ = make_no_signal_dataset(0)
    correlation = np.corrcoef(X.ravel(), y)[0, 1]
    assert abs(correlation) < 0.2


def test_naive_shuffled_kfold_looks_highly_skillful_on_pure_noise(
    scores: dict[str, np.ndarray],
) -> None:
    """The failure mode this package exists to prevent, at full strength."""
    assert scores["shuffled"].mean() > 0.85


def test_unpurged_contiguous_folds_still_look_skillful(
    scores: dict[str, np.ndarray],
) -> None:
    """Respecting time order is not enough on its own.

    Contiguous folds remove the interleaving that makes shuffled CV so
    catastrophic, and a practitioner may reasonably believe that is sufficient.
    It is not: the rows on either side of each test block still carry labels
    that reach into it.
    """
    assert scores["contiguous"].mean() > 0.70


def test_purged_kfold_returns_the_score_to_chance(scores: dict[str, np.ndarray]) -> None:
    """The claim: with overlapping labels purged, no skill is found because none exists."""
    assert CHANCE - 0.05 < scores["purged"].mean() < CHANCE + 0.05


def test_purging_is_what_removes_the_inflation(scores: dict[str, np.ndarray]) -> None:
    """Same fold layout, same data, same model -- only purging differs.

    This is the attributable comparison. ``_BaseContiguousKFold`` and
    ``PurgedKFold`` produce byte-identical test folds; the only difference
    between them is which training rows survive.
    """
    gap = scores["contiguous"].mean() - scores["purged"].mean()
    assert gap > 0.20


def test_purged_beats_unpurged_on_every_individual_seed(
    scores: dict[str, np.ndarray],
) -> None:
    """Not an artefact of averaging: the effect holds seed by seed."""
    per_seed = scores["contiguous"] - scores["purged"]
    assert np.all(per_seed > 0.05), f"weakest seed gap was {per_seed.min():.3f}"


def test_embargo_alone_does_not_account_for_the_effect() -> None:
    """Purging carries the result; the embargo is a smaller additional margin.

    Worth pinning: if a refactor accidentally disabled purging and left only the
    embargo, the headline numbers would still move in the right direction and
    could be mistaken for success.
    """
    X, y, end = make_no_signal_dataset(0)
    model = KNeighborsClassifier(n_neighbors=1)
    instantaneous = np.arange(N_SAMPLES)  # no label overlap -> nothing to purge

    embargo_only = _score(
        model,
        X,
        y,
        PurgedKFold(
            n_splits=N_SPLITS, label_end_times=instantaneous, embargo_pct=EMBARGO_PCT
        ),
    )
    fully_purged = _score(
        model,
        X,
        y,
        PurgedKFold(n_splits=N_SPLITS, label_end_times=end, embargo_pct=EMBARGO_PCT),
    )
    assert embargo_only > fully_purged + 0.10
