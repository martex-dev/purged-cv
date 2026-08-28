# API note: `split_detail()` — requested by cv-visualizer

Status: **proposal, not implemented.** Written 2026-08-28 while scoping
cv-visualizer, before purged-cv itself was built. Read this before
finalising the public API, so it doesn't need retrofitting later.

## The problem

sklearn's splitter protocol yields `(train_idx, test_idx)` and nothing
else. Any row that was purged and any row that was embargoed are both
simply *absent from `train_idx`* — from outside the splitter they are
indistinguishable.

This is fine for `cross_val_score`, which only needs the two arrays. It is
not enough for cv-visualizer, whose entire job is to render purged and
embargoed regions as **visually distinct bands**. Without a way to
attribute dropped rows, the diagram either omits the distinction (which
deletes the reason the picture exists) or guesses at it (which teaches the
wrong thing — worse than no diagram).

The alternative — re-deriving the purge/embargo boundary arithmetic inside
cv-visualizer — was rejected. That logic is *the product* of this package;
a second copy in a downstream repo would drift, and a drifted copy
mislabels bands silently. One source of truth.

## Proposed addition

Additive only. `split()` keeps its exact current signature and behaviour,
so sklearn compatibility — the whole value proposition — is untouched.

```python
class FoldDetail(NamedTuple):
    train: np.ndarray
    test: np.ndarray
    purged: np.ndarray
    embargoed: np.ndarray


def split_detail(self, X, y=None, groups=None) -> Iterator[FoldDetail]:
    """Yield the same folds as `split()`, with dropped rows attributed.

    `train` and `test` are identical to what `split()` yields for the
    same fold. `purged` and `embargoed` together account for every row
    that `split()` silently dropped from training.
    """
```

## Invariants worth testing inside purged-cv

These are cheap to assert and they become cv-visualizer's ground truth:

1. `train`, `test`, `purged`, `embargoed` are **pairwise disjoint**.
2. Their union is exactly `range(n_samples)` — nothing is unaccounted for.
3. `split_detail(...)[i].train` is **array-identical** to
   `split(...)[i][0]`, and likewise for `test`. The detail method must
   never disagree with the method people actually train against.

Invariant 3 is the important one. It is what lets a downstream diagram
claim to show what `split()` really did, rather than a plausible
reconstruction of it.

## Two design calls this forces

### 1. Purged and embargoed overlap — pick a precedence

A row can legitimately sit *both* inside a test fold's label-overlap
window *and* inside the embargo window following that fold. The sets must
still be returned disjoint, because a diagram cannot paint one row two
colours, and because invariant 1 above is what makes the accounting
checkable.

Suggested rule: **purge wins.** It is the stronger, label-driven guarantee;
embargo is the belt-and-braces serial-correlation margin on top. Whichever
way it goes, the docstring must state it explicitly — this is exactly the
kind of quiet boundary decision purged-cv's CLAUDE.md says to write down
rather than let emerge from the implementation.

### 2. There is a fifth category, and it is not purged-cv's

Some splitters leave rows **unused** in a given fold without purging them.
`TimeSeriesSplit(n_splits=3)` on 20 rows produces train/test sizes
`(5,5), (10,5), (15,5)` — in fold 0, rows 10–19 are in neither train nor
test. They were not purged; they simply have not been reached yet.

`PurgedKFold` may not produce this category at all, in which case
`split_detail` need not model it. It is recorded here so the distinction is
deliberate: cv-visualizer computes `unused` itself as
`all − train − test − purged − embargoed`, and renders it as absence
rather than as a purge. Colouring a not-yet-reached row as "purged" would
misteach the concept.

## Downstream contract (already implemented in cv-visualizer)

cv-visualizer duck-types this and degrades honestly:

```python
if callable(getattr(splitter, "split_detail", None)):
    ...  # render train / test / purged / embargoed / unused
else:
    ...  # render train / test / unused only
```

So plain `KFold` and `TimeSeriesSplit` render correctly today with two
band types — which is accurate for them, not a missing feature. Implement
`split_detail` and the purge/embargo bands light up with no change on the
visualiser side.
