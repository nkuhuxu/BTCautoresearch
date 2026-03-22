#!/usr/bin/env python3
"""
fit.py — MUTABLE. The agent modifies this file.

Goal: find the functional form that best predicts Bitcoin's price as a
function of time (day index since genesis), measured by walk-forward
multi-horizon RMSE on log10(price).

The model_fn receives:
  - train_days:       np.array of day indices (e.g. 365, 366, ..., 4000)
  - train_log_prices: np.array of log10(price_usd) for training period
  - test_days:        np.array of day indices to predict

It must return:
  - np.array of predicted log10(price_usd) for each test day

Current model: POWER LAW + LAST-DAY DEVIATION DECAY
  Fit power law on all training data, use the last training day's residual
  as the deviation, then predict with it decaying toward zero.
  prediction = a*log10(d) + b + last_residual * exp(-log(2) * dt / half_life)
  where half_life = 180 days (6 months)
"""

import numpy as np
from scipy.optimize import curve_fit


# ============================================================
# MODEL DEFINITION — modify this section
# ============================================================

def formula(days, a, b):
    """The functional form to fit. Returns log10(price)."""
    return a * np.log10(days) + b


# Initial parameter guesses for curve_fit
P0 = [5.0, -15.0]

# Parameter bounds (use None for unbounded)
BOUNDS = (-np.inf, np.inf)


# ============================================================
# MODEL FUNCTION — the agent may also modify this if needed
# (e.g. to add preprocessing, custom fitting, etc.)
# ============================================================

def model_fn(train_days, train_log_prices, test_days):
    """
    Power law + last-day deviation decay.
    Fit the long-term trend, use the last training day's residual as the
    deviation, then decay it back to zero (mean reversion).
    Half-life of 180 days.
    """
    try:
        popt, _ = curve_fit(
            formula,
            train_days,
            train_log_prices,
            p0=P0,
            bounds=BOUNDS,
            maxfev=10000,
        )
    except RuntimeError:
        popt = np.polyfit(np.log10(train_days), train_log_prices, 1)
        return popt[0] * np.log10(test_days) + popt[1]

    a, b = popt

    # Use the last training day's residual as deviation from trend
    deviation = train_log_prices[-1] - formula(train_days[-1], a, b)

    # Decay the deviation toward zero with half-life of 180 days
    last_day = train_days[-1]
    half_life = 180.0
    decay = np.exp(-np.log(2) * (test_days - last_day) / half_life)

    return formula(test_days, a, b) + deviation * decay


# ============================================================
# MAIN — do not remove this block
# ============================================================
if __name__ == "__main__":
    from prepare import evaluate_model, print_results

    mean_rmse, details = evaluate_model(model_fn)
    print_results(mean_rmse, details)
