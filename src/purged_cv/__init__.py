"""Purged k-fold cross-validation with an embargo, compatible with scikit-learn.

Standard cross-validation assumes samples are independent. Time-series labels
are not: a label observed at time ``t`` is often determined by data up to
``t + k``, so a training sample whose label window overlaps a test fold has
already seen that fold's outcome. The model then looks skillful for reasons
that will not survive contact with live data.

:class:`PurgedKFold` removes those samples (*purging*) and additionally drops a
margin of samples immediately afterwards to break serial correlation
(*embargo*). It is a drop-in scikit-learn splitter::

    from sklearn.model_selection import cross_val_score
    from purged_cv import PurgedKFold

    cv = PurgedKFold(n_splits=5, label_end_times=label_end, embargo_pct=0.01)
    scores = cross_val_score(model, X, y, cv=cv)

The precise boundary rules -- and the two places this package deliberately
departs from López de Prado's reference code, both in the conservative
direction -- are documented in ``docs/boundary-semantics.md``.
"""

from purged_cv._purged import FoldDetail, PurgedKFold

__all__ = ["FoldDetail", "PurgedKFold", "__version__"]

__version__ = "0.1.0.dev0"
