"""Fold layout and scikit-learn plumbing shared by the purged splitters.

Nothing in this module purges or embargoes anything. It exists so that the
scikit-learn integration (correct ``split`` / ``get_n_splits`` signatures,
``repr``, behaviour inside ``cross_val_score`` and ``GridSearchCV``) can be
built and proven correct *before* any leakage-prevention logic is layered on
top. Keeping it private means no half-guaranteed splitter is ever exposed to
a user: a public class only appears once it actually purges.
"""

from __future__ import annotations

from collections.abc import Iterator
from numbers import Integral
from typing import Any

import numpy as np
from numpy.typing import NDArray
from sklearn.model_selection import BaseCrossValidator

__all__ = ["_BaseContiguousKFold"]


def _n_samples(X: Any) -> int:
    """Return the number of samples in ``X``.

    Mirrors ``sklearn.utils.validation._num_samples`` for the cases a splitter
    can encounter, without importing a private scikit-learn symbol -- this
    package is small on purpose and should not break on an internal rename.

    ``X`` has already been through ``sklearn.utils.indexable`` by the time the
    base class calls this, so it is an array, a sparse matrix, a DataFrame or a
    sequence.

    Examples:
        >>> import numpy as np
        >>> _n_samples(np.zeros((7, 3)))
        7
        >>> _n_samples([[1], [2], [3]])
        3
    """
    if hasattr(X, "shape"):
        shape = X.shape
        if len(shape) == 0:
            raise TypeError(f"Singleton array {X!r} cannot be considered a valid collection.")
        return int(shape[0])
    return len(X)


class _BaseContiguousKFold(BaseCrossValidator):
    """K-fold splitter whose test folds are contiguous blocks in index order.

    Fold ``k`` is the block of positional indices
    ``[start_k, start_k + size_k)``, taken in the order the samples were
    supplied -- the data is never shuffled, because for time-series data the
    row order *is* the time order and shuffling destroys it. The training set
    is the plain complement of the test fold.

    Fold sizes match :class:`sklearn.model_selection.KFold` exactly: the first
    ``n_samples % n_splits`` folds get ``n_samples // n_splits + 1`` samples and
    the rest get ``n_samples // n_splits``, so every sample is tested exactly
    once. Matching sklearn here is deliberate -- it means the difference between
    this splitter's output and ``KFold(shuffle=False)`` is *only* the purging
    and embargo added by subclasses, which is what the package's central claim
    rests on.

    Args:
        n_splits: Number of folds. Must be an integer >= 2.

    Raises:
        ValueError: If ``n_splits`` is not an integer, or is less than 2.

    Examples:
        >>> import numpy as np
        >>> X = np.zeros((10, 2))
        >>> for train, test in _BaseContiguousKFold(n_splits=5).split(X):
        ...     print(train, test)
        [2 3 4 5 6 7 8 9] [0 1]
        [0 1 4 5 6 7 8 9] [2 3]
        [0 1 2 3 6 7 8 9] [4 5]
        [0 1 2 3 4 5 8 9] [6 7]
        [0 1 2 3 4 5 6 7] [8 9]
    """

    def __init__(self, n_splits: int = 5) -> None:
        if not isinstance(n_splits, Integral) or isinstance(n_splits, bool):
            raise ValueError(
                f"The number of folds must be of Integral type. "
                f"{n_splits} of type {type(n_splits)} was passed."
            )
        if n_splits < 2:
            raise ValueError(
                "k-fold cross-validation requires at least one train/test split "
                f"by setting n_splits=2 or more, got n_splits={n_splits}."
            )
        # Stored unmodified under the parameter's own name: sklearn's
        # ``_build_repr`` reflects over ``__init__`` and reads these back.
        self.n_splits = int(n_splits)

    def get_n_splits(self, X: Any = None, y: Any = None, groups: Any = None) -> int:
        """Return the number of splitting iterations.

        The arguments are ignored; they exist because scikit-learn calls this
        method with them positionally.
        """
        return self.n_splits

    def _iter_test_indices(
        self, X: Any = None, y: Any = None, groups: Any = None
    ) -> Iterator[NDArray[np.intp]]:
        """Yield the positional indices of each contiguous test fold."""
        if X is None:
            raise ValueError("The 'X' parameter should not be None.")
        n_samples = _n_samples(X)
        if self.n_splits > n_samples:
            raise ValueError(
                f"Cannot have number of splits n_splits={self.n_splits} greater than "
                f"the number of samples: n_samples={n_samples}."
            )
        fold_size, remainder = divmod(n_samples, self.n_splits)
        start = 0
        for k in range(self.n_splits):
            stop = start + fold_size + (1 if k < remainder else 0)
            yield np.arange(start, stop, dtype=np.intp)
            start = stop
