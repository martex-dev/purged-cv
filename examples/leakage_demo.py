"""Before/after: the same no-signal dataset scored under three splitters.

Run with ``python examples/leakage_demo.py``. Requires only the package's own
dependencies.

The data contains no signal at all -- returns are i.i.d. Gaussian noise, so the
honest accuracy for any model is 0.50. The labels overlap in time, which is the
only thing that lets a model appear skillful.
"""

from __future__ import annotations

import numpy as np
from sklearn.model_selection import KFold, cross_val_score
from sklearn.neighbors import KNeighborsClassifier

from purged_cv import PurgedKFold

N_SAMPLES = 1200
HORIZON = 50
N_SPLITS = 24
EMBARGO_PCT = 0.01
N_SEEDS = 12


def make_no_signal_dataset(seed: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Pure noise with overlapping labels. Returns ``(X, y, label_end_times)``."""
    rng = np.random.default_rng(seed)
    returns = rng.normal(size=N_SAMPLES + HORIZON + 1)
    cumulative = np.concatenate([[0.0], np.cumsum(returns)])

    # Label i is the sign of the return over (i, i + HORIZON]. Consecutive
    # labels therefore share HORIZON - 1 of their HORIZON terms.
    forward = cumulative[1 + HORIZON : N_SAMPLES + 1 + HORIZON] - cumulative[1 : N_SAMPLES + 1]
    y = (forward > 0).astype(int)

    # A single slowly-varying feature carrying no information about the future.
    # Its only job is to make "close in time" mean "close in feature space".
    X = np.arange(N_SAMPLES, dtype=float).reshape(-1, 1)

    label_end_times = np.minimum(np.arange(N_SAMPLES) + HORIZON, N_SAMPLES - 1)
    return X, y, label_end_times


def main() -> None:
    model = KNeighborsClassifier(n_neighbors=1)
    results: dict[str, list[float]] = {"shuffled": [], "contiguous": [], "purged": []}

    for seed in range(N_SEEDS):
        X, y, label_end_times = make_no_signal_dataset(seed)
        splitters = {
            "shuffled": KFold(N_SPLITS, shuffle=True, random_state=seed),
            "contiguous": KFold(N_SPLITS, shuffle=False),
            "purged": PurgedKFold(
                n_splits=N_SPLITS,
                label_end_times=label_end_times,
                embargo_pct=EMBARGO_PCT,
            ),
        }
        for name, cv in splitters.items():
            results[name].append(float(cross_val_score(model, X, y, cv=cv).mean()))

    labels = {
        "shuffled": "KFold(shuffle=True)",
        "contiguous": "KFold(shuffle=False)",
        "purged": f"PurgedKFold(embargo_pct={EMBARGO_PCT})",
    }
    print(
        f"{N_SAMPLES} samples of pure noise, {HORIZON}-step overlapping labels, "
        f"{N_SPLITS} folds, mean of {N_SEEDS} seeds.\nTrue accuracy is 0.500 -- "
        f"there is no signal in this data.\n"
    )
    print(f"{'splitter':<34}{'accuracy':>10}{'inflation':>12}")
    print("-" * 56)
    for name, values in results.items():
        mean = float(np.mean(values))
        print(f"{labels[name]:<34}{mean:>10.3f}{mean - 0.5:>+12.3f}")


if __name__ == "__main__":
    main()
