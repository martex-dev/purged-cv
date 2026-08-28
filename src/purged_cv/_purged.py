"""Purging and embargo -- the leakage-prevention layer.

The boundary decisions implemented here are argued in
``docs/boundary-semantics.md``. Where this module diverges from López de
Prado's reference code (*Advances in Financial Machine Learning*, ch. 7) the
divergence is deliberate, always in the conservative direction, and called
out both there and in the docstrings below.
"""

from __future__ import annotations

from collections.abc import Iterator
from math import ceil
from numbers import Real
from typing import Any, NamedTuple

import numpy as np
from numpy.typing import NDArray
from sklearn.utils import indexable

from purged_cv._split import _BaseContiguousKFold, _n_samples

__all__ = ["FoldDetail", "PurgedKFold"]


class FoldDetail(NamedTuple):
    """One fold, with every dropped row attributed to the reason it was dropped.

    ``split()`` yields only ``train`` and ``test``; from outside the splitter a
    purged row and an embargoed row are then indistinguishable, since both are
    simply absent from ``train``. This type exists so downstream consumers --
    cv-visualizer in particular -- can tell them apart without re-deriving the
    boundary arithmetic, which is this package's product and would drift if
    copied.

    The four arrays are pairwise disjoint and their union is exactly
    ``range(n_samples)``: every row is accounted for by exactly one category.

    Attributes:
        train: Rows used for fitting.
        test: Rows used for scoring.
        purged: Rows dropped because their label window overlaps the test span.
        embargoed: Rows dropped because they fall in the serial-correlation
            margin immediately after the purge boundary.
    """

    train: NDArray[np.intp]
    test: NDArray[np.intp]
    purged: NDArray[np.intp]
    embargoed: NDArray[np.intp]


def _as_1d(values: Any, name: str) -> NDArray[Any]:
    """Coerce ``values`` to a 1-D array, accepting pandas objects without importing pandas."""
    array = np.asarray(values.to_numpy() if hasattr(values, "to_numpy") else values)
    if array.ndim != 1:
        raise ValueError(f"{name} must be one-dimensional, got shape {array.shape}.")
    return array


class PurgedKFold(_BaseContiguousKFold):
    """K-fold cross-validation that purges overlapping labels and embargoes their tail.

    Test folds are contiguous blocks in index order, identical to
    :class:`sklearn.model_selection.KFold` with ``shuffle=False``. The training
    set is *not* the plain complement: rows whose label window overlaps the test
    fold's are removed (**purged**), and a further margin of rows immediately
    after that overlap is removed (**embargoed**).

    What is purged, precisely. For each fold the test span is
    ``[min(start) over test, max(end) over test]`` -- note the upper edge uses
    the label *end* times, so it reaches as far forward as the slowest label in
    the fold. A training row is kept only if its own ``[start, end]`` interval
    is disjoint from that span, with strict inequalities: an interval that
    merely touches the span at an endpoint is purged.

    What is embargoed, precisely. ``ceil(n_samples * embargo_pct)`` rows,
    beginning at the first row that survived purging on the right-hand side.
    The count rounds **up**, unlike the reference implementation's truncation,
    so a non-zero ``embargo_pct`` always embargoes at least one row. The
    embargo is applied forward only; rows before the test fold are already
    handled by purging, because label windows extend forwards.

    What is *not* handled. Rows must already be sorted by label start time --
    this is checked, not fixed. Alignment between ``X`` and the label times is
    positional, exactly as for ``y``. Combinatorial purged CV (CPCV) and
    uniqueness weighting are out of scope.

    Args:
        n_splits: Number of folds. Must be an integer >= 2.
        label_end_times: When each label's window closes, one entry per row of
            ``X``, in the same row order. Required: it cannot be inferred from
            the index, and guessing it defeats the purpose of the splitter. If
            this is a pandas ``Series`` and ``label_start_times`` is omitted,
            its index is used as the start times.
        label_start_times: When each label's window opens. Defaults to the
            index of ``label_end_times`` when that is a pandas object, and to
            positional integers ``0..n-1`` otherwise.
        embargo_pct: Fraction of the dataset embargoed after each test fold, in
            ``[0, 1)``. ``0.0`` disables the embargo; purging still applies.

    Raises:
        ValueError: If ``n_splits`` is not an integer >= 2, if ``embargo_pct``
            is not a real number in ``[0, 1)``, if the label times are the wrong
            length or shape, contain missing values, are not mutually
            comparable, are not sorted by start time, or describe a label that
            ends before it starts. Also if a fold's training set is emptied
            entirely by purging and embargo.

    Examples:
        Twenty rows whose labels each close two rows later, four folds, and an
        embargo of ``ceil(20 * 0.10) = 2`` rows:

        >>> import numpy as np
        >>> from purged_cv import PurgedKFold
        >>> n = 20
        >>> end = np.minimum(np.arange(n) + 2, n - 1)
        >>> cv = PurgedKFold(n_splits=4, label_end_times=end, embargo_pct=0.10)
        >>> for train, test in cv.split(np.zeros((n, 1))):
        ...     print(f"test {test[0]:>2}-{test[-1]:<2} train {train.tolist()}")
        test  0-4  train [9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19]
        test  5-9  train [0, 1, 2, 14, 15, 16, 17, 18, 19]
        test 10-14 train [0, 1, 2, 3, 4, 5, 6, 7, 19]
        test 15-19 train [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]

        The same fold, with the dropped rows attributed:

        >>> fold = list(cv.split_detail(np.zeros((n, 1))))[1]
        >>> fold.test.tolist(), fold.purged.tolist(), fold.embargoed.tolist()
        ([5, 6, 7, 8, 9], [3, 4, 10, 11], [12, 13])
    """

    def __init__(
        self,
        n_splits: int = 5,
        *,
        label_end_times: Any,
        label_start_times: Any = None,
        embargo_pct: float = 0.0,
    ) -> None:
        super().__init__(n_splits=n_splits)
        if not isinstance(embargo_pct, Real) or isinstance(embargo_pct, bool):
            raise ValueError(
                f"embargo_pct must be a real number, got {embargo_pct!r} of "
                f"type {type(embargo_pct)}."
            )
        if not 0.0 <= float(embargo_pct) < 1.0:
            raise ValueError(f"embargo_pct must lie in [0, 1), got {embargo_pct!r}.")
        # Stored unmodified under their own parameter names: sklearn's
        # ``_build_repr`` reflects over ``__init__`` and reads these back.
        self.label_end_times = label_end_times
        self.label_start_times = label_start_times
        self.embargo_pct = embargo_pct

    def _resolve_times(self, n_samples: int) -> tuple[NDArray[Any], NDArray[Any]]:
        """Return validated ``(start, end)`` arrays aligned positionally with ``X``."""
        end_raw = self.label_end_times
        start_raw = self.label_start_times

        if start_raw is None and hasattr(end_raw, "index") and hasattr(end_raw, "to_numpy"):
            # A pandas Series: its index carries the start times, as in AFML's ``t1``.
            start = _as_1d(end_raw.index, "label_end_times.index")
            end = _as_1d(end_raw, "label_end_times")
        else:
            end = _as_1d(end_raw, "label_end_times")
            start = (
                np.arange(len(end))
                if start_raw is None
                else _as_1d(start_raw, "label_start_times")
            )

        if len(end) != n_samples:
            raise ValueError(
                f"label_end_times has {len(end)} entries but X has {n_samples} samples. "
                f"They are aligned positionally, one label time per row of X."
            )
        if len(start) != n_samples:
            raise ValueError(
                f"label_start_times has {len(start)} entries but X has {n_samples} samples."
            )
        if bool(np.any(start != start)) or bool(np.any(end != end)):
            raise ValueError("label times must not contain missing values (NaN/NaT).")

        try:
            ends_after_start = bool(np.all(end >= start))
            is_sorted = bool(np.all(start[1:] >= start[:-1]))
        except TypeError as exc:  # e.g. datetime64 end times against integer start times
            raise ValueError(
                f"label_start_times (dtype {start.dtype}) and label_end_times "
                f"(dtype {end.dtype}) must be mutually comparable."
            ) from exc

        if not is_sorted:
            raise ValueError(
                "Samples must be sorted by label start time, ascending. Test folds are "
                "contiguous blocks of positions and the embargo is located by binary "
                "search, so both are meaningless on unsorted input. Sort X, y and the "
                "label times together before constructing the splitter -- this is not "
                "done for you, because reordering rows on your behalf would silently "
                "break their correspondence with y."
            )
        if not ends_after_start:
            raise ValueError("Every label must end at or after it starts (end >= start).")
        return start, end

    def _iter_fold_details(
        self, X: Any = None, y: Any = None, groups: Any = None
    ) -> Iterator[FoldDetail]:
        """Compute every fold once, so ``split`` and ``split_detail`` cannot disagree."""
        if X is None:
            raise ValueError("The 'X' parameter should not be None.")
        # Materialised before resolving label times so that an n_splits error is
        # reported ahead of a label-length error, whichever the caller got wrong.
        test_folds = list(self._iter_test_indices(X, y, groups))

        n_samples = _n_samples(X)
        start, end = self._resolve_times(n_samples)
        embargo = ceil(n_samples * float(self.embargo_pct))
        indices = np.arange(n_samples, dtype=np.intp)

        for fold, test_idx in enumerate(test_folds):
            test_mask = np.zeros(n_samples, dtype=bool)
            test_mask[test_idx] = True

            test_start = start[test_idx].min()
            test_end = end[test_idx].max()

            # A label interval [start_i, end_i] is disjoint from the test span
            # [test_start, test_end] only if it closes strictly before the span
            # opens, or opens strictly after the span closes. Anything else
            # overlaps and is purged -- touching at an endpoint counts as
            # overlapping. See docs/boundary-semantics.md section 3.
            disjoint = (end < test_start) | (start > test_end)
            purged_mask = ~disjoint & ~test_mask

            # The embargo starts at the first row that survived purging on the
            # right, i.e. the first row opening strictly after the test span
            # closes. Anchoring it there rather than at the end of the test
            # block is what keeps it from landing inside the purged region --
            # so purged and embargoed never overlap. Section 4.2.
            embargo_mask = np.zeros(n_samples, dtype=bool)
            if embargo:
                first_kept = int(np.searchsorted(start, test_end, side="right"))
                embargo_mask[first_kept : first_kept + embargo] = True

            train_mask = ~(test_mask | purged_mask | embargo_mask)
            if not train_mask.any():
                raise ValueError(
                    f"Fold {fold} has no training samples left: purging removed "
                    f"{int(purged_mask.sum())} rows and the embargo removed "
                    f"{int(embargo_mask.sum())} more, out of {n_samples}. Use fewer "
                    f"splits, a smaller embargo_pct, shorter label windows, or more data."
                )
            yield FoldDetail(
                train=indices[train_mask],
                test=test_idx,
                purged=indices[purged_mask],
                embargoed=indices[embargo_mask],
            )

    def split(
        self, X: Any = None, y: Any = None, groups: Any = None
    ) -> Iterator[tuple[NDArray[np.intp], NDArray[np.intp]]]:
        """Yield ``(train, test)`` positional indices for each fold.

        Args:
            X: Data to split. Only its number of samples is used.
            y: Ignored; present for scikit-learn interface compatibility.
            groups: Ignored; label times are supplied to ``__init__`` instead,
                because ``groups`` cannot express an interval per sample.

        Yields:
            ``(train, test)`` arrays of positional indices. ``train`` is the
            complement of ``test`` *minus* the purged and embargoed rows, so
            the two do not cover every row.
        """
        X, y, groups = indexable(X, y, groups)
        for detail in self._iter_fold_details(X, y, groups):
            yield detail.train, detail.test

    def split_detail(
        self, X: Any = None, y: Any = None, groups: Any = None
    ) -> Iterator[FoldDetail]:
        """Yield the same folds as :meth:`split`, with dropped rows attributed.

        ``train`` and ``test`` are identical to what :meth:`split` yields for
        the same fold -- both methods are driven by one computation, so they
        cannot drift apart. ``purged`` and ``embargoed`` account for every row
        that :meth:`split` silently dropped.

        Args:
            X: Data to split. Only its number of samples is used.
            y: Ignored; present for scikit-learn interface compatibility.
            groups: Ignored; see :meth:`split`.

        Yields:
            :class:`FoldDetail` per fold.
        """
        X, y, groups = indexable(X, y, groups)
        yield from self._iter_fold_details(X, y, groups)
