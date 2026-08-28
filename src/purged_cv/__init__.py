"""Minimal, sklearn-compatible purged k-fold cross-validation with an embargo.

No public splitter is exported yet: the fold layout and scikit-learn plumbing
land first (see ``purged_cv._split``), and ``PurgedKFold`` appears only once it
genuinely purges. Exporting a splitter that silently fails to purge would be
worse than exporting nothing.
"""

__all__: list[str] = ["__version__"]

__version__ = "0.1.0.dev0"
