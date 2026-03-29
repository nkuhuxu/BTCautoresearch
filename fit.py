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

Current model: HORIZON-ADAPTIVE TREND BLEND + LOCAL RESIDUAL
  Two trends: past-weighted (gamma=-0.8) and recency-weighted (gamma=+0.8).
  At short horizons, blend toward recency trend; at long horizons, use past trend.
  blend(dt) = past_trend + exp(-dt/tau) * (recency_trend - past_trend)
  + local residual correction with 5-ensemble.
"""

import numpy as np
from scipy.optimize import minimize


# ============================================================
# MODEL DEFINITION — modify this section
# ============================================================

def formula(days, a, b):
    """Shifted power law: a * log10(d+325) + b."""
    return a * np.log10(days + 330.0) + b


# Initial parameter guesses for curve_fit
P0 = [5.0, -15.0]

# Parameter bounds (use None for unbounded)
BOUNDS = (-np.inf, np.inf)


# ============================================================
# MODEL FUNCTION — the agent may also modify this if needed
# ============================================================

def _fit_trend(log10_days, train_log_prices, gamma):
    """Fit Huber power law trend with given recency weight gamma."""
    span = len(train_log_prices)
    raw_weights = np.exp(gamma * np.arange(span) / span)
    weights = raw_weights / raw_weights.sum() * span

    def huber_loss(params):
        a, b = params
        pred = a * log10_days + b
        residuals = train_log_prices - pred
        delta = 0.4
        mask = np.abs(residuals) <= delta
        loss = np.where(mask,
                        0.5 * residuals**2,
                        delta * (np.abs(residuals) - 0.5 * delta))
        return (loss * weights).sum()

    try:
        ols = np.polyfit(log10_days, train_log_prices, 1)
        result = minimize(huber_loss, x0=ols, method='Nelder-Mead',
                          options={'maxiter': 10000, 'xatol': 1e-8, 'fatol': 1e-8})
        return result.x
    except Exception:
        return np.polyfit(log10_days, train_log_prices, 1)


def model_fn(train_days, train_log_prices, test_days):
    """
    Two Huber trends (past gamma=-0.8 and recency gamma=+0.8) blended adaptively.
    At short horizons, blend toward recency; at long horizons, use past trend.
    Plus local 30-day linear residual with 5-ensemble decay.
    """
    log10_days = np.log10(train_days + 330.0)

    # Fit past-weighted trend (long-term stable)
    a_past, b_past = _fit_trend(log10_days, train_log_prices, gamma=-3.2)
    # Fit recency-weighted trend (adapts to recent prices)
    a_rec, b_rec = _fit_trend(log10_days, train_log_prices, gamma=0.0)

    # Use past trend for the residual correction baseline
    local_a, local_b = a_past, b_past
    n_local = min(30, len(train_days))
    local_days = train_days[-n_local:]
    local_resid = train_log_prices[-n_local:] - formula(local_days, local_a, local_b)

    last_day = train_days[-1]
    t = local_days - last_day
    # Recency-weighted slope (upweight recent days)
    n = len(t)
    sw = np.exp(1.20 * np.arange(n) / n)
    sw /= sw.sum()
    cov = np.cov(t, local_resid, aweights=sw)
    slope = (cov[0, 1] / cov[0, 0] if cov[0, 0] > 0 else 0.0) * 0.90
    r0 = np.median(local_resid[-3:])

    dt = test_days - last_day

    # Horizon-adaptive trend blend: at dt=0, use recency; at large dt, use past
    tau_blend = 150.0  # blend half-life
    w_rec = np.exp(-(dt / tau_blend) ** 12.0)  # 12th-power blend
    blend_trend = w_rec * formula(test_days, a_rec, b_rec) + (1 - w_rec) * formula(test_days, a_past, b_past)

    # Correction relative to past trend (same convention)
    corrections = []
    for hl in [112.0, 127.0, 142.0]:
        decay = np.exp(-np.log(2) * dt / hl)
        corrections.append((r0 + slope * dt) * decay)
    ensemble_correction = np.mean(corrections, axis=0)

    return blend_trend + ensemble_correction


# ============================================================
# MAIN — do not remove this block
# ============================================================
if __name__ == "__main__":
    from prepare import evaluate_model, print_results

    mean_rmse, details = evaluate_model(model_fn)
    print_results(mean_rmse, details)
