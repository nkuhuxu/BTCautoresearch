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

Current model: POWER LAW + MEAN-REVERSION DECAY (EWMA)
  Fit power law on all training data, compute recent deviation from trend
  using an exponentially weighted mean (EWMA) over the last 180 days,
  then predict with deviation decaying back to zero.
  prediction = a*log10(d) + b + ewma_deviation * exp(-log(2) * dt / half_life)
  where half_life = 180 days (6 months), EWMA span = 60 days
"""

import numpy as np
from scipy.optimize import curve_fit


# ============================================================
# MODEL DEFINITION — modify this section
# ============================================================

def formula(days, a, b, c, d):
    """Quadratic in log-space: a * log10(days+c)^2 + b * log10(days+c) + d."""
    log_d = np.log10(days + c)
    return a * log_d**2 + b * log_d + d


# Initial parameter guesses for curve_fit
P0 = [-1.0, 7.0, -5.0, 500.0]

# Parameter bounds (use None for unbounded)
BOUNDS = ([-np.inf, -np.inf, -np.inf, 1.0], [np.inf, np.inf, np.inf, 2000.0])


# ============================================================
# MODEL FUNCTION — the agent may also modify this if needed
# (e.g. to add preprocessing, custom fitting, etc.)
# ============================================================

def model_fn(train_days, train_log_prices, test_days):
    """
    Power law + mean-reversion decay with EWMA deviation.
    Fit the long-term trend, measure recent deviation using exponentially
    weighted mean (more weight on recent days), then add a decaying correction.
    Half-life of 180 days, EWMA span of 60 days.
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
        popt = np.polyfit(np.log10(train_days), train_log_prices, 2)
        log_test = np.log10(test_days)
        return popt[0] * log_test**2 + popt[1] * log_test + popt[2]

    a, b, c, d = popt

    # Measure deviation using last 365 days
    recent_n = min(365, len(train_days))
    recent_days = train_days[-recent_n:]
    recent_prices = train_log_prices[-recent_n:]
    residuals = recent_prices - formula(recent_days, a, b, c, d)

    # Simple last-day deviation (no smoothing)
    deviation = residuals[-1] if len(residuals) > 0 else 0.0

    # Blend formula with linear extrapolation of recent residuals
    last_day = train_days[-1]
    dt = test_days - last_day
    half_life_dev = 140.0
    half_life_trend = 127.0
    decay = np.exp(-np.log(2) * dt / half_life_dev)

    # Weighted linear trend from last 30 days (emphasize recent)
    n_trend = min(33, len(residuals))
    if n_trend > 1:
        x = np.arange(n_trend)
        w = np.exp(1.16 * x / n_trend)  # More weight on recent points
        slope, _ = np.polyfit(x, residuals[-n_trend:], 1, w=w)
        trend = slope * dt / 2.95  # Scale by days
    else:
        trend = 0.0

    # Blend: deviation decays, trend continues with light damping
    trend_damping = np.exp(-0.50 * dt / half_life_trend)  # Very light damping
    return formula(test_days, a, b, c, d) + 1.065 * deviation * decay + 1.63 * trend * trend_damping


# ============================================================
# MAIN — do not remove this block
# ============================================================
if __name__ == "__main__":
    from prepare import evaluate_model, print_results

    mean_rmse, details = evaluate_model(model_fn)
    print_results(mean_rmse, details)