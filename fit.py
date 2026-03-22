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

Current model: RECENCY-WEIGHTED HUBER POWER LAW + LINEAR RESIDUAL EXTRAPOLATION
  Fit power law with recency-weighted Huber loss (recent data gets 7x more weight).
  Fit linear trend to the last 30 days of residuals. Extrapolate with 120-day decay.
  r_forecast(dt) = (r0 + slope * dt) * exp(-log(2)*dt/120)
"""

import numpy as np
from scipy.optimize import curve_fit, minimize


# ============================================================
# MODEL DEFINITION — modify this section
# ============================================================

def formula(days, a, b):
    """Shifted power law: log10(d + 200) models earlier effective genesis."""
    return a * np.log10(days + 175.0) + b


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
    Huber-robust power law + local linear residual extrapolation.
    Fit power law with Huber loss to reduce influence of price spikes.
    Then fit a local linear model to the last 30 days of residuals.
    The extrapolated residual decays toward zero with 120-day half-life.
    """
    # Huber-robust fitting of power law
    log10_days = np.log10(train_days + 175.0)

    # Recency weights: more weight to recent data
    span = len(train_days)
    raw_weights = np.exp(1.65 * np.arange(span) / span)
    weights = raw_weights / raw_weights.sum() * span

    def huber_loss(params):
        a, b = params
        pred = a * log10_days + b
        residuals = train_log_prices - pred
        delta = 0.5
        mask = np.abs(residuals) <= delta
        loss = np.where(mask,
                        0.5 * residuals**2,
                        delta * (np.abs(residuals) - 0.5 * delta))
        return (loss * weights).sum()

    try:
        # Start from OLS solution (using shifted feature, consistent with huber_loss)
        ols = np.polyfit(log10_days, train_log_prices, 1)
        result = minimize(huber_loss, x0=ols, method='Nelder-Mead',
                          options={'maxiter': 10000, 'xatol': 1e-8, 'fatol': 1e-8})
        a, b = result.x
    except Exception:
        a, b = np.polyfit(log10_days, train_log_prices, 1)

    # Fit linear trend to last 30 days of residuals
    n_local = min(30, len(train_days))
    local_days = train_days[-n_local:]
    local_resid = train_log_prices[-n_local:] - formula(local_days, a, b)

    # OLS: resid ≈ r0 + slope * (d - last_day)
    last_day = train_days[-1]
    t = local_days - last_day  # relative time (0 is the last day)
    slope = np.polyfit(t, local_resid, 1)[0]
    r0 = local_resid[-1]  # value at last_day

    # Ensemble of 5 half-life values (model averaging over time scales)
    dt = test_days - last_day
    trend_pred = formula(test_days, a, b)
    corrections = []
    for hl in [60.0, 90.0, 120.0, 150.0, 180.0]:
        decay = np.exp(-np.log(2) * dt / hl)
        corrections.append((r0 + slope * dt) * decay)
    ensemble_correction = np.mean(corrections, axis=0)

    return trend_pred + ensemble_correction


# ============================================================
# MAIN — do not remove this block
# ============================================================
if __name__ == "__main__":
    from prepare import evaluate_model, print_results

    mean_rmse, details = evaluate_model(model_fn)
    print_results(mean_rmse, details)
