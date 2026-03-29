# autoresearch: Bitcoin Price Formula Discovery

## Objective

Find the functional form `log10(price) = f(day_index)` that minimizes
walk-forward multi-horizon RMSE when predicting Bitcoin's price over time.

The baseline is the **power law**: `log10(price) = a * log10(day) + b`,
which achieves R² ~ 0.95 in-sample. Can you beat it out-of-sample?

## Setup

1. Agree on a run tag (e.g. `mar22`). Create branch `autoresearch/<tag>`.
2. Read these files:
   - `README.md` (if it exists)
   - `prepare.py` — **READ-ONLY**. Evaluation harness. Do not modify.
   - `fit.py` — **MUTABLE**. Contains the model. This is what you change.
3. Verify data exists: `btc_daily_prices.csv`
4. Run the baseline: `uv run fit.py` — note the `mean_rmse` value.
5. Initialize `results.tsv` with header:
   `commit\tmean_rmse\tstatus\tdescription`
6. Confirm baseline result with user, then begin the loop.

## The Loop

```
LOOP FOREVER:
  1. Think of an idea to improve the formula in fit.py
  2. Edit fit.py with the change
  3. git commit the change with a descriptive message
  4. Run: uv run fit.py > run.log 2>&1
  5. Extract: grep "^mean_rmse:" run.log
  6. If grep is empty → crash. Read run.log, fix or revert.
  7. Log to results.tsv (do NOT commit results.tsv)
  8. If mean_rmse IMPROVED (lower) → KEEP the commit → git push origin <branch>
  9. If mean_rmse EQUAL or WORSE → git reset to prior commit (REVERT)
```

## Rules

1. **Only `fit.py` may be modified.** `prepare.py` is sacred.
2. **No new dependencies.** Only numpy, scipy, matplotlib (in pyproject.toml).
3. **Metric: `mean_rmse`** (lower is better). This is the mean RMSE of
   log10(price) across 9 walk-forward splits × 7 forward horizons
   (1, 3, 6, 12, 18, 24, 36 months).
4. **No future information.** The model receives only training data.
   It must predict test days using only what it learned from training.
5. **Simplicity matters.** If two formulas achieve similar RMSE, prefer the
   simpler one. But a complex formula that clearly wins is fine — the
   out-of-sample evaluation prevents overfitting.
6. **NEVER STOP EXPERIMENTING.** The human may be asleep. Run continuously.
   Do not ask "should I continue?" — just keep going.

## What to Try

The model in fit.py has three parts you can modify:

1. **`formula(days, ...)`** — the functional form itself. Ideas:
   - Log-periodic power law: `a * log10(d) + b + c * cos(w * log(d) + phi)`
   - Stretched exponential: `a * (1 - exp(-b * d^c))`
   - Piecewise power laws (different slopes for different epochs)
   - Power law with saturation: `a * log10(d) / (1 + d/d0) + b`
   - Polynomial in log-space: `a * log10(d)^2 + b * log10(d) + c`
   - S-curve / logistic on log scale
   - Combinations and hybrids

2. **`P0` and `BOUNDS`** — initial guesses and parameter bounds for curve_fit.
   More parameters need better initial guesses.

3. **`model_fn`** — the fitting procedure itself. Ideas:
   - Weighted least squares (more weight on recent data)
   - Robust fitting (e.g. Huber loss via scipy.optimize.minimize)
   - Ensemble of formulas
   - Fitting in different spaces (log-log, log-linear, etc.)

## Output Format

fit.py must print a line exactly matching:
```
mean_rmse: <value>
```
The evaluation harness in prepare.py handles this automatically.

## Tips

- Each run takes seconds (curve fitting, not ML training), so iterate fast.
- Start with small modifications to the power law before trying radically
  different forms.
- Watch the per-horizon breakdown: a formula might win at short horizons
  but lose at long ones (or vice versa). The metric averages across all.
- If curve_fit fails to converge, try better P0 values or add bounds.
- The power law is surprisingly hard to beat. That's the point — if you
  beat it, you've found something interesting.
