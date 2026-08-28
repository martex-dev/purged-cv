# Purge and embargo boundary semantics

This file records *why* each boundary is drawn where it is, before the code
that draws it. Every decision below is a place where an off-by-one silently
reintroduces the leakage this package exists to prevent, so none of them are
left implicit in the implementation.

Reference: López de Prado, *Advances in Financial Machine Learning* (2018),
chapter 7. Cited below as AFML.

---

## 1. The data model

Every observation `i` has two times:

- **`start_i`** — when the observation's feature window closes. This is the
  time you would have made the decision.
- **`end_i`** — when the observation's label is finally determined
  (AFML calls this `t1`). For a label like "did price move 5% within 10
  days", `end_i` is up to 10 days after `start_i`.

The label of observation `i` therefore depends on information spanning the
closed interval `[start_i, end_i]`.

`end_i` **cannot be inferred** from the index. A splitter that guesses it is
worse than no splitter, because it produces a number the user will trust.
So `label_end_times` is a required argument with no default.

**Alignment is positional, not by index label.** Element `k` of
`label_end_times` describes row `k` of `X`. If you pass a `Series` whose
index is unrelated to `X`'s, no error is raised and the result is wrong —
the same rule sklearn applies to `y`.

## 2. Samples must be sorted by `start`

Test folds are contiguous blocks of *positions*, and the embargo is located
with `searchsorted`, which requires sorted input. If `start` is not
non-decreasing, both are meaningless.

Rather than sort internally — which would silently permute the caller's rows
relative to their `y` — this is a hard `ValueError`. Sorting is the caller's
job and they can do it correctly; guessing on their behalf cannot.

## 3. The purge rule

For a test fold, the **test span** is

```
test_start = min(start_i for i in test)
test_end   = max(end_i   for i in test)
```

Note `test_end` uses `end`, not `start`. The test fold's information reaches
as far forward as its slowest label, not as far as its last feature window.
Using `start` here is the single most damaging off-by-one available, because
it leaves exactly the overlapping rows that motivate the package.

A training row `i` is **kept** only if its label interval is disjoint from
the test span:

```
keep_i  =  (end_i < test_start)  or  (start_i > test_end)
```

Everything else is **purged**. The inequalities are strict, so an interval
that merely *touches* the test span at an endpoint is purged. Touching means
the two labels share a boundary instant of information; per the project rule
"if in doubt, purge", that is dropped.

### 3.1 AFML contradicts itself here, and we take the stricter branch

AFML gives two implementations that disagree at exactly this boundary:

- `getTrainTimes` drops any training row whose interval intersects the test
  span, endpoints included. That is the rule above.
- `PurgedKFold.split` keeps left-side rows where `t1 <= t0` (with `t0` the
  test start) — so it **retains** a row whose label ends exactly when the
  test fold begins.

These differ by one row at each boundary. We implement `getTrainTimes`
semantics, the stricter of the two. The discrepancy is noted rather than
quietly resolved, per the project's working-style rule about the book's
definition. Anyone who needs the looser behaviour is, in practice, asking
for a boundary sample that the purge exists to remove.

### 3.2 The overlap test is a comparison, not a binary search

AFML's `PurgedKFold.split` locates the right-hand purge boundary with
`t1.index.searchsorted(t1[test].max())`. A binary search is only valid on
sorted input, and `t1` — the label *end* times — is sorted only when every
label has the same horizon.

Variable horizons are ordinary, not exotic: a triple-barrier label closes
whenever a barrier is first touched, so `end` routinely jumps around while
`start` marches forward. On such data the book's binary search silently
returns the wrong boundary and under-purges.

This package therefore evaluates the overlap rule of section 3 as a direct
elementwise comparison over all rows, which needs no ordering assumption on
`end` at all. Binary search is used in exactly one place — locating the
embargo's start within `start`, which *is* validated as sorted (section 2).

The test suite keeps a deliberately non-monotonic `end` fixture as a
permanent guard against this being "optimised" back into a `searchsorted`.

## 4. The embargo

Purging removes label *overlap*. The embargo removes what purging cannot
see: **serial correlation**. Two non-overlapping windows that sit close
together in time are still statistically dependent when the underlying
series is autocorrelated, which is the normal case for financial data.

### 4.1 It applies forward only

Embargo is applied only *after* the test span, never before. This is not
an asymmetry in the statistics, it is a consequence of how labels are
shaped: label intervals extend forward from `start`, so a training row that
sits shortly *before* the test fold already has its label reaching into it,
and purging removes it. A training row shortly *after* the test span has no
such overlap, so nothing has removed it — and it is precisely the row most
correlated with the test fold's outcome. That gap is what the embargo
covers.

### 4.2 It is anchored to the purge boundary, not to the test block

The embargo covers the first `m` rows that **survived purging** on the right
side — that is, it begins at the first position whose `start > test_end`.

Anchoring it to the end of the test *block* instead would be wrong: the
purged region already extends past the block, and the embargo would then
overlap it, doing nothing. This matches AFML's `indices[maxT1Idx + mbrg:]`.

**A consequence worth stating**: purged and embargoed rows are disjoint by
construction. The embargo begins exactly where purging stops. `split_detail`
therefore never has to arbitrate between the two categories, and the "purge
wins" precedence proposed in `API_NOTES.md` never has to fire. It remains
the documented rule should the anchor ever change, and a test asserts the
disjointness rather than assuming it.

### 4.3 `embargo_pct` rounds **up**

```
m = ceil(n_samples * embargo_pct)
```

AFML uses `int(...)`, which truncates. Truncation is the wrong direction:
with 150 samples and `embargo_pct=0.01`, the requested 1.5 rows becomes 1,
leaving in training a row the parameter asked to exclude. Rounding up
embargoes 2.

This is a deliberate divergence from the book's code, in the conservative
direction, and it is the one place where "match the book exactly" and "never
let convenience weaken the guarantee" genuinely conflict. The project's rule
is *if in doubt, purge*, so it rounds up. Consequence: any `embargo_pct > 0`
embargoes at least one row, and `embargo_pct=0.0` embargoes none — the only
way to disable it.

## 5. Test rows are never purged

A test row's own interval trivially intersects the test span, so the purge
rule flags it. It is reported as `test`, not as `purged` — a row cannot be
in two categories, and `split_detail`'s accounting invariant depends on the
four sets partitioning `range(n_samples)` exactly.

## 6. An empty training fold is an error, not a result

Aggressive `embargo_pct`, long label windows or too many splits can purge a
fold's entire training set. Yielding an empty array would surface deep
inside the estimator as a confusing error about array shapes. `PurgedKFold`
raises instead, naming the fold and the parameters responsible.
