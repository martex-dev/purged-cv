# purged-cv

An sklearn-compatible cross-validation splitter for time-series / financial
data: PurgedKFold with an embargo period. Prevents the single most common
and most damaging leakage pattern in quant/ML backtests — training on data
whose label window overlaps with the test window.

Target: a real, citable, pip-installable package. Not a tutorial repo, not
a Kaggle notebook. Something a stranger could `pip install` and trust with
real backtests.
Budget ceiling: $0 (no paid infra needed for v1).

## Why this exists (read before making design calls)

Every standard sklearn CV splitter (KFold, TimeSeriesSplit, StratifiedKFold)
assumes samples are independent. Financial/time-series labels are not — a
label at time t often depends on data up to t+k (e.g. "did the price move
5% in the next 10 days"). Standard CV silently leaks: a training fold can
contain samples whose label window overlaps a test-fold sample's feature
window, and the model looks skillful purely because it saw the future.

This is a known problem (López de Prado's "Advances in Financial Machine
Learning" is the standard citation) but there is no clean, well-tested,
sklearn-compatible pip package for it as of today. Existing versions are
either buried inside larger paid platforms, copy-pasted from blog posts
with no tests, or subtly wrong at the fold boundaries. The value here is
doing the boring, careful version properly — not the idea, which is already
public.

## v1 scope — what ships

- `PurgedKFold(n_splits, embargo_pct)` — sklearn-compatible splitter
  (implements `split()`, `get_n_splits()`, works inside
  `cross_val_score` / `GridSearchCV` unmodified)
- Purging: drop training samples whose label window overlaps any test
  sample's window
- Embargo: additionally drop training samples in a fixed window *after*
  each test fold, to prevent leakage through serial correlation
- Accepts an explicit `label_end_times` array (the user must supply when
  each label's window closes — this cannot be inferred from index alone,
  and guessing it wrong defeats the whole point)
- Works on both a plain DatetimeIndex and an explicit integer/positional
  index — don't assume the index type
- Full test suite proving leakage is actually removed: a test that
  demonstrates a model looks skillful under naive KFold on synthetic data
  with no real signal, and does not look skillful under PurgedKFold on the
  same data. This is the single most important test in the repo — it's
  the proof the package does what it claims.
- README with a runnable before/after example: naive CV score vs. purged
  CV score on the same synthetic no-signal dataset, so the leakage is
  visible in one screenshot, not just asserted in prose

## Explicitly NOT in v1 (don't build unless asked)

- Combinatorial purged CV (CPCV) — a real López de Prado extension, but
  meaningfully more complex and not needed to make the core point
  correctly
- Any plotting/visualization — that's a separate project (cv-visualizer)
  and should stay separate so each package has one job
- Sample-weight / uniqueness weighting (also from the same book) — a
  different feature with its own correctness surface, don't fold it in
- Support for non-sklearn frameworks (PyTorch loaders, etc.)
- PyPI publishing automation / CI release pipeline — get the package
  correct first, publish manually once it's stable

## Correctness discipline — this is the entire point of the package

- The purge/embargo boundary logic is the product. Every off-by-one here
  quietly reintroduces the exact leakage this package exists to prevent.
  Treat fold-boundary arithmetic with the same discipline as the statutory
  interest date math on late-payment-chaser — write the boundary test
  before the implementation, not after.
- Never let convenience quietly weaken the guarantee — e.g. don't round
  or approximate `embargo_pct` in a way that could leave a boundary sample
  in when it should be purged. If in doubt, purge.
- Docstrings must state precisely what is and isn't purged. A user who
  misunderstands the guarantee and ships a leaky backtest anyway is a
  worse outcome than the package not existing.
- Compare output against a hand-computed toy example (small enough to
  verify by hand, e.g. 20 samples) in the test suite, not just against
  itself at different parameter values.

## Conventions

- Tabs for indentation, single quotes (per project owner's standing
  preference) — Python convention is PEP 8 with 4 spaces, so flag this
  conflict once and default to PEP 8 (Black-formatted) unless told
  otherwise, since this is a public package other people will read
- Type hints throughout; this is a library, not a script
- `pytest` for tests, aim for near-100% coverage on the splitter logic
  specifically (coverage on README examples/docs matters less)
- Package structure: `src/purged_cv/`, tests in `tests/`, so it installs
  cleanly and doesn't pollute the import namespace

## Working style for this project

- This is a learning + portfolio project — prefer explaining *why* a
  boundary case is handled a certain way before implementing it, not just
  shipping it
- Flag any place where "make the test pass" and "match the book's actual
  definition" might diverge — go with the book's definition and explain
  the discrepancy rather than silently adjusting the test
- If sklearn's splitter interface has changed or has an edge case not
  covered by prior knowledge (e.g. `groups` parameter handling), check
  current sklearn docs rather than assuming — API compatibility is the
  whole value proposition of "sklearn-compatible"
