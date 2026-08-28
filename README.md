# purged-cv

A small, dependency-light, scikit-learn-compatible **purged k-fold cross-validator
with an embargo**, for time-series and financial machine learning.

Standard cross-validation assumes samples are independent. Time-series labels are
not: a label observed at time `t` is often determined by data up to `t + k`. So a
training row whose label window overlaps a test fold has already seen that fold's
outcome, and the model scores well for a reason that will not survive contact with
live data.

## The problem, measured

`examples/leakage_demo.py` builds a dataset with **no signal in it at all** —
returns are i.i.d. Gaussian noise, so the honest accuracy for any model is 0.500.
The labels overlap in time (each is the sign of the next 50 steps' return), and
the feature is just elapsed time, so "close in time" means "close in feature
space". A 1-nearest-neighbour model is then scored under three splitters:

| splitter | accuracy | inflation |
|---|---:|---:|
| `KFold(shuffle=True)` | 0.935 | +0.435 |
| `KFold(shuffle=False)` | 0.781 | +0.281 |
| `PurgedKFold(embargo_pct=0.01)` | **0.503** | +0.003 |

*1200 samples, 50-step overlapping labels, 24 folds, mean of 12 seeds.*

```bash
python examples/leakage_demo.py
```

Every point of accuracy above 0.500 is leakage. Two things are worth noticing:

- Shuffling is catastrophic, which is widely known.
- **Not shuffling is not enough.** Contiguous, time-ordered folds still score
  0.781 on data containing nothing to learn, because the rows on either side of
  each test block carry labels that reach into it. This is the part people miss,
  and it is why `TimeSeriesSplit` alone does not solve the problem.

`PurgedKFold` and `KFold(shuffle=False)` produce **byte-identical test folds** —
the only difference between those two rows is which training rows survive. That
makes the gap attributable to purging rather than to fold geometry.

## Install

```bash
pip install purged-cv
```

Requires Python 3.10+, `numpy` and `scikit-learn`. No pandas dependency (pandas
objects are accepted if you have it).

## Use

The one thing you must supply is `label_end_times`: when each label's window
closes. It cannot be inferred from the index, and guessing it defeats the point,
so it is required.

```python
import numpy as np
from sklearn.model_selection import cross_val_score
from purged_cv import PurgedKFold

# Row i's label is not resolved until 10 rows later.
label_end_times = np.arange(len(X)) + 10

cv = PurgedKFold(n_splits=5, label_end_times=label_end_times, embargo_pct=0.01)
scores = cross_val_score(model, X, y, cv=cv)
```

It is a drop-in splitter: `cross_val_score`, `cross_validate`, `GridSearchCV` and
`Pipeline` all work unmodified.

**Wall-clock times** work as well as row offsets — pass `label_start_times`, or a
pandas `Series` whose index carries the start times (the shape López de Prado's
`t1` uses):

```python
cv = PurgedKFold(n_splits=5, label_end_times=pd.Series(end_times, index=df.index))
```

Both index styles are supported deliberately, and are tested against each other:
an integer/positional index is a first-class case, not an afterthought.

## What is purged, precisely

Row `i` occupies the **closed** interval `[start_i, end_i]`. For each fold the
test span is `[min(start) over test, max(end) over test]` — the upper edge uses
label *end* times, so it reaches as far forward as the slowest label in the fold.

A training row is kept only if its interval is **disjoint** from that span.
Touching at an endpoint counts as overlapping and is purged: a training label
that closes at the exact instant the test span opens is removed. That is stricter
than a half-open convention by one row at each boundary, and is deliberate — an
unnecessary purge costs a little training data, a missed one silently invalidates
the backtest.

## What is embargoed, precisely

After purging, `ceil(n_samples * embargo_pct)` further rows are dropped, starting
at the first row that survived purging on the right of the fold. The embargo is
**forward-only**; purging already covers the backward direction.

Anchoring on the purge boundary rather than on the last test row matters: on
typical data the latter would only remove rows purging had already taken, and so
would do nothing at all.

## What this package does not do

- **No combinatorial purged CV (CPCV).** One backtest path, not many.
- **No sample-weighting or uniqueness adjustment.** A different feature with its
  own correctness surface.
- **No plotting.** See [cv-visualizer](https://github.com/martex-dev/cv-visualizer),
  which renders purged and embargoed bands using `split_detail()` (below).
- **Purging cannot rescue a label that is simply wrong.** If `label_end_times`
  understates when a label truly closes, this splitter will report a clean split
  that is not clean. The array is your claim about your data, and the guarantee
  is only as good as that claim.

Rows must already be sorted by label start time. This is checked and raises,
rather than being silently fixed — sorting internally would permute your rows
away from their `y`.

## Attributing dropped rows

`split()` yields `(train, test)` like any sklearn splitter, so a purged row and
an embargoed row are indistinguishable from outside — both are simply absent.
`split_detail()` yields the same folds with the reason attached:

```python
for fold in cv.split_detail(X):
    fold.train, fold.test, fold.purged, fold.embargoed
```

The four arrays are pairwise disjoint and together account for every row.
`fold.train` and `fold.test` are identical to what `split()` yields, driven by one
computation so they cannot drift apart.

## Correctness

- 199 tests, 100% line coverage on the splitter modules.
- Two independent **hand-computed** 20-row examples, derived on paper from
  `docs/boundary-semantics.md` before the implementation existed, pinning every
  index in every category.
- A randomised invariant check across sample counts, fold counts, embargo widths
  and varying label horizons.
- The leakage proof above, asserted across 12 seeds — not just illustrated.

Two places this package **deliberately diverges** from the reference code in
*Advances in Financial Machine Learning*, both in the conservative direction:

1. **The embargo width rounds up**, not down. Snippet 7.2 truncates with `int()`;
   truncating can leave a row inside the window the embargo was asked to clear.
2. **The purge boundary is found by comparison, not `searchsorted` on `t1`.**
   Snippet 7.3's binary search over label end times is only valid when those
   times are sorted — i.e. only for a constant label horizon. Variable horizons
   are ordinary (a triple-barrier label resolves whenever a barrier is touched),
   and the test suite keeps a non-monotonic fixture as a permanent guard.

Both are argued in [`docs/boundary-semantics.md`](docs/boundary-semantics.md).

## Prior art

[`purgedcv`](https://pypi.org/project/purgedcv/) is a good, actively maintained,
MIT-licensed package covering the same ground and more — `PurgedKFold`,
`PurgedGroupKFold`, combinatorial purged CV, deflated Sharpe ratios. **If you want
the full toolkit, use it.**

purged-cv exists as the deliberately minimal alternative: one splitter, ~120
statements, no pandas dependency, readable end to end in a sitting, and accepting
a plain integer/positional index, which `purgedcv` rejects. If you would rather
audit your cross-validator than trust it, that is the trade this package makes.

## Reference

López de Prado, M. (2018). *Advances in Financial Machine Learning*, chapter 7:
Cross-Validation in Finance. Wiley.

The algorithms are his. What this package adds is a small, tested, installable
implementation with its boundary conventions written down.

## Licence

MIT.
