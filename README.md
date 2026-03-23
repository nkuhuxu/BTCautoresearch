# BTCautoresearch

Autonomous formula discovery for Bitcoin price prediction, inspired by
[Karpathy's autoresearch](https://github.com/karpathy/autoresearch) pattern.

An AI agent (Claude) autonomously ran **328 experiments** in a single session,
searching for the best time-based formula to predict Bitcoin's price. Each
experiment modifies a single file (`fit.py`), evaluates via walk-forward
out-of-sample RMSE, and keeps or reverts the change — building a
monotonically improving commit history.

## Results

**50.5% improvement** over the standard power law baseline, achieved through
328 autonomous experiments (114 kept).

### Prediction RMSE vs Forward Horizon (up to 5 years)

![Horizon comparison](fig_horizon.png)

The best model (dark blue) maintains a clear advantage across all horizons,
from 1 month to 5 years out-of-sample.

### Experiment-by-Experiment Progress

![Experiment progress](fig_experiments.png)

The agent discovers key improvements in phases: mean-reversion correction,
local linear extrapolation, Huber-robust fitting, ensemble decay, and
shifted power law — then fine-tunes hyperparameters.

### Bootstrap Statistical Comparison

![Bootstrap](fig_bootstrap.png)

Split-level bootstrap (10,000 resamples) comparing three models:

| Model | mean RMSE | 95% CI | vs baseline |
|---|---|---|---|
| Power law (baseline) | 0.267 | [0.190, 0.355] | — |
| Simple last-day decay | 0.171 | [0.131, 0.219] | **-36%** |
| Model T (fully tuned) | 0.132 | [0.105, 0.162] | **-51%** |

Both models significantly beat the power law. The improvement of Model T
over the simple model is not statistically significant (CI includes zero),
suggesting the core insight — mean-reversion to the power law — captures
most of the achievable gain.

### Residual Analysis

![Residuals](fig_residuals.png)

Out-of-sample residuals show strong autocorrelation (~250 days), positive
skew (bubble underestimation), and halving-cycle structure — indicating
exploitable signal remains beyond what calendar time can capture.

## The Core Finding

The power law can be significantly improved with a simple modification:

```python
# Fit power law on training data
a, b = polyfit(log10(train_days), train_log_prices, 1)

# Last training day's deviation from trend
r0 = train_log_prices[-1] - (a * log10(train_days[-1]) + b)

# Predict: power law + decaying correction
dt = test_days - train_days[-1]
prediction = a * log10(test_days) + b + r0 * exp(-log(2) * dt / 180)
```

**Interpretation**: The power law sets the long-term trend. The correction
acknowledges where price is right now relative to trend and decays it toward
zero over ~6 months (half-life 180 days). This is mean-reversion to the
power law.

## Project Structure

```
├── fit.py          # The mutable model file (agent modifies this)
├── prepare.py      # Read-only evaluation harness (walk-forward, multi-horizon)
├── program.md      # Agent instructions (the autoresearch protocol)
├── btc_daily_prices.csv  # Daily BTC price data (2009–2026)
├── results.tsv     # Full experiment log (328 experiments)
└── pyproject.toml  # Dependencies: numpy, scipy, matplotlib
```

## How to Use

### Prerequisites

- Python 3.10+
- [uv](https://docs.astral.sh/uv/) package manager
- An AI coding agent (Claude Code, Cursor, Codex, etc.)

### Setup

```bash
git clone https://github.com/CBaquero/BTCautoresearch.git
cd BTCautoresearch
uv sync
```

### Run the Current Best Model

```bash
uv run fit.py
```

### Start a New Experiment Loop

Point your AI agent at the repo and say:

> Read program.md and start a new experiment loop. Tag: \<your-tag\>

The agent will:
1. Create a branch `autoresearch/<tag>`
2. Read `prepare.py` (read-only rules) and `fit.py` (what to modify)
3. Run the baseline: `uv run fit.py`
4. Loop: edit `fit.py` → commit → run → keep if improved, revert if not
5. Log results to `results.tsv`

**Tips:**
- Use `--dangerously-skip-permissions` for unattended runs
- Sonnet-class models work well for this task (good at code editing, fast)
- Each experiment takes ~1-2 seconds (curve fitting, not ML training)
- Expect ~20-30 experiments/hour, limited by agent thinking time

### Evaluation Protocol

`prepare.py` implements strict walk-forward validation:
- **9 splits**: 2016–2024 (expanding training window)
- **7 horizons**: 1, 3, 6, 12, 18, 24, 36 months forward
- **Metric**: mean RMSE of log10(price) across all 63 evaluations
- **No future information**: model receives only training data at each split

## License

MIT
