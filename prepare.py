#!/usr/bin/env python3
"""
prepare.py — READ-ONLY. Do not modify.

Loads Bitcoin daily price data and provides the evaluation harness.
The agent modifies fit.py only. This file defines the rules of the game.

Evaluation protocol (walk-forward, multi-horizon):
  For each cutoff in WALK_SPLITS:
    1. Train: fit the model on data up to cutoff
    2. Test: predict forward at horizons 1,3,6,12,18,24,36 months
    3. Compute RMSE(log10 price) at each horizon
  Final metric: mean RMSE across all splits and horizons.
"""

import csv
import json
import os
import sys

import numpy as np

# --- Constants ---
DATA_FILE = os.path.join(os.path.dirname(__file__), "btc_daily_prices.csv")
START_DAY = 365  # Exclude first year (no meaningful price data)

WALK_SPLITS = [
    "2016-01-01", "2017-01-01", "2018-01-01", "2019-01-01",
    "2020-01-01", "2021-01-01", "2022-01-01", "2023-01-01",
    "2024-01-01",
]
MIN_TRAIN_DAYS = 4 * 365

HORIZON_MONTHS = [1, 3, 6, 12, 18, 24, 36]


def load_data():
    """Load price data. Returns (dates, day_indices, log10_prices) filtered."""
    dates = []
    prices = []
    with open(DATA_FILE) as f:
        for row in csv.DictReader(f):
            p = float(row["price_usd"])
            if p > 0:
                dates.append(row["date"])
                prices.append(p)

    prices = np.array(prices)
    n = len(dates)
    day_indices = np.arange(1, n + 1, dtype=float)

    # Filter out first year
    mask = day_indices >= START_DAY
    dates = [dates[i] for i in range(n) if mask[i]]
    day_indices = day_indices[mask]
    log_prices = np.log10(prices[mask])

    return dates, day_indices, log_prices


def evaluate_model(model_fn):
    """
    Evaluate a model function using walk-forward multi-horizon protocol.

    model_fn(train_days, train_log_prices, test_days) -> test_predictions

    Args:
        model_fn: callable that takes (train_days, train_log_prices, test_days)
                  and returns predicted log10(price) for test_days.

    Returns:
        mean_rmse: the single scalar metric (lower is better)
        details: dict with per-split, per-horizon breakdown
    """
    dates, day_indices, log_prices = load_data()
    n = len(dates)

    all_horizon_rmses = {h: [] for h in HORIZON_MONTHS}
    split_details = []

    for split_date in WALK_SPLITS:
        si = next((i for i, d in enumerate(dates) if d >= split_date), None)
        if si is None or si < MIN_TRAIN_DAYS or si >= n - 30:
            continue

        train_days = day_indices[:si]
        train_prices = log_prices[:si]

        horizon_results = {}
        for h_months in HORIZON_MONTHS:
            h_days = h_months * 30
            test_end = min(si + h_days, n)
            if test_end <= si:
                continue

            test_days = day_indices[si:test_end]
            test_prices = log_prices[si:test_end]

            try:
                predictions = model_fn(train_days, train_prices, test_days)
                predictions = np.asarray(predictions, dtype=float)
                if predictions.shape != test_prices.shape:
                    raise ValueError(
                        f"Shape mismatch: predictions {predictions.shape} "
                        f"vs actual {test_prices.shape}"
                    )
                if np.any(np.isnan(predictions)) or np.any(np.isinf(predictions)):
                    raise ValueError("NaN or Inf in predictions")

                rmse = float(np.sqrt(np.mean((test_prices - predictions) ** 2)))
            except Exception as e:
                rmse = 999.0  # Penalty for crashes
                print(f"  WARNING: model crashed at split={split_date} "
                      f"horizon={h_months}m: {e}", file=sys.stderr)

            all_horizon_rmses[h_months].append(rmse)
            horizon_results[h_months] = rmse

        split_details.append({
            "split_date": split_date,
            "horizons": horizon_results,
        })

    # Compute mean RMSE across all splits and horizons
    all_rmses = []
    horizon_means = {}
    for h in HORIZON_MONTHS:
        if all_horizon_rmses[h]:
            mean_h = float(np.mean(all_horizon_rmses[h]))
            horizon_means[h] = mean_h
            all_rmses.extend(all_horizon_rmses[h])

    mean_rmse = float(np.mean(all_rmses)) if all_rmses else 999.0

    return mean_rmse, {
        "horizon_means": horizon_means,
        "splits": split_details,
        "n_evaluations": len(all_rmses),
    }


def print_results(mean_rmse, details):
    """Print results in the standard autoresearch output format."""
    print("=" * 70)
    print("EVALUATION RESULTS")
    print("=" * 70)

    print(f"\n  Per-horizon mean RMSE (log10 price):")
    print(f"  {'Horizon':<12} {'RMSE':<10}")
    print(f"  {'-' * 22}")
    for h, rmse in sorted(details["horizon_means"].items(), key=lambda x: x[0]):
        print(f"  {h:>3} months   {rmse:.6f}")

    print(f"\n  Total evaluations: {details['n_evaluations']}")
    print(f"\n{'=' * 70}")
    print(f"mean_rmse: {mean_rmse:.6f}")
    print(f"{'=' * 70}")
