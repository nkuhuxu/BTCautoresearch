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

Current model: POWER LAW + LOCAL LINEAR RESIDUAL EXTRAPOLATION
  Fit power law on all training data. Fit a linear trend to the last 30 days
  of residuals to get level + velocity. Extrapolate with 180-day decay.
  r_forecast(dt) = (r0 + slope * dt) * exp(-log(2)*dt/half_life)
  where r0 = last residual, slope from OLS on last 30d residuals
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
    Power law + local linear residual extrapolation.
    Fit the long-term trend, then fit a local linear model to the last 30 days
    of residuals to capture both the current level and trend velocity.
    The extrapolated residual decays toward zero with 180-day half-life.
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

    # Fit linear trend to last 30 days of residuals
    n_local = min(30, len(train_days))
    local_days = train_days[-n_local:]
    local_resid = train_log_prices[-n_local:] - formula(local_days, a, b)

    # OLS: resid ≈ r0 + slope * (d - last_day)
    last_day = train_days[-1]
    t = local_days - last_day  # relative time (0 is the last day)
    slope = np.polyfit(t, local_resid, 1)[0]
    r0 = local_resid[-1]  # value at last_day

    # Extrapolate residual with decay
    half_life = 120.0
    dt = test_days - last_day
    decay = np.exp(-np.log(2) * dt / half_life)

    return formula(test_days, a, b) + (r0 + slope * dt) * decay


# ============================================================
# MAIN — do not remove this block
# ============================================================
if __name__ == "__main__":
    from prepare import evaluate_model, print_results

    mean_rmse, details = evaluate_model(model_fn)
    print_results(mean_rmse, details)
