#!/usr/bin/env python3
"""
Study Curves - Hold Period Optimization

For each signal, track returns at every day from 1 to 90 after entry.
Finds the optimal hold period (peak return day) per signal.
One run per signal instead of 29 runs per signal x exit.

Output: per signal, the return curve + peak day + returns at key intervals.
"""

import warnings
warnings.filterwarnings("ignore")

import json
import time
import numpy as np
import pandas as pd
import ta
from pathlib import Path

import config
import data_fetcher
from studies import SIGNALS, _rolling_sortino, _rolling_omega

MAX_HOLD = 90
RESULTS_DIR = Path(__file__).parent / ".data" / "studies"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


def run_signal_curve(sig_key, all_data):
    """
    For a single signal, compute the average return at each day 1-90 after entry.
    Returns the full curve + peak day + key stats.
    """
    if sig_key not in SIGNALS:
        return None

    _, sig_fn = SIGNALS[sig_key]
    sig_name = SIGNALS[sig_key][0]

    etf_to_sector = {v: k for k, v in config.SECTOR_ETFS.items()}

    # Collect returns at each hold day across all trades
    day_returns = {d: [] for d in range(1, MAX_HOLD + 1)}
    total_trades = 0

    for ticker, df in all_data.items():
        if ticker == config.BENCHMARK or len(df) < MAX_HOLD + 20:
            continue

        try:
            signals = sig_fn(df).fillna(False)
        except Exception:
            continue

        entry_dates = signals[signals].index.tolist()
        close = df["Close"].values
        n = len(close)

        for entry_date in entry_dates:
            idx = df.index.get_loc(entry_date)
            entry_price = close[idx]
            if entry_price <= 0:
                continue

            total_trades += 1
            max_exit = min(idx + MAX_HOLD, n - 1)

            for hold_day in range(1, MAX_HOLD + 1):
                exit_idx = idx + hold_day
                if exit_idx > max_exit:
                    break
                ret = (close[exit_idx] - entry_price) / entry_price * 100
                day_returns[hold_day].append(ret)

    if total_trades == 0:
        return None

    # Compute curve stats
    curve = []
    for day in range(1, MAX_HOLD + 1):
        rets = day_returns[day]
        if not rets:
            curve.append({"day": day, "avg": 0, "wr": 0, "trades": 0})
            continue
        avg = np.mean(rets)
        wr = sum(1 for r in rets if r > 0) / len(rets) * 100
        curve.append({
            "day": day,
            "avg": round(avg, 4),
            "median": round(np.median(rets), 4),
            "wr": round(wr, 1),
            "trades": len(rets),
            "p10": round(np.percentile(rets, 10), 3),
            "p90": round(np.percentile(rets, 90), 3),
        })

    # Find peak day (highest average return) — only over horizons that retain
    # a representative sample. Long holds keep only the oldest entries (those
    # with MAX_HOLD forward bars), so an unfloored argmax can pick a tiny,
    # non-independent tail cohort. Floor at >=20 trades AND >=50% of the
    # day-1 sample so peak_day reflects the same population as the short holds.
    day1_trades = curve[0]["trades"] if curve else 0
    min_peak_trades = max(20, int(0.5 * day1_trades))
    eligible = [c for c in curve if c["trades"] >= min_peak_trades]
    if not eligible:
        eligible = curve
    peak_day = max(eligible, key=lambda x: x["avg"])
    # Find best win rate day (same sample floor)
    best_wr_day = max(eligible, key=lambda x: x["wr"])

    # Key intervals
    key_days = {
        "1d": curve[0] if len(curve) > 0 else None,
        "3d": curve[2] if len(curve) > 2 else None,
        "1w": curve[4] if len(curve) > 4 else None,
        "2w": curve[9] if len(curve) > 9 else None,
        "4w": curve[19] if len(curve) > 19 else None,
        "8w": curve[39] if len(curve) > 39 else None,
        "12w": curve[59] if len(curve) > 59 else None,
    }

    return {
        "signal": sig_key,
        "signal_name": sig_name,
        "category": _categorize_sig(sig_key),
        "total_trades": total_trades,
        "peak_day": peak_day["day"],
        "peak_avg": peak_day["avg"],
        "peak_wr": peak_day["wr"],
        "best_wr_day": best_wr_day["day"],
        "best_wr": best_wr_day["wr"],
        "key_days": key_days,
        "curve": curve,
    }


def _categorize_sig(sig_key):
    from studies import _categorize
    return _categorize(sig_key)


def run_all_curves():
    """Run curve analysis for all signals."""
    print("Loading data...")
    all_data = data_fetcher.fetch_all()
    print(f"Loaded {len(all_data)} tickers")

    # Pre-compute indicators
    for ticker, df in all_data.items():
        if len(df) < 20:
            continue
        df["_sortino"] = _rolling_sortino(df)
        df["_omega"] = _rolling_omega(df)
        df["_rsi"] = ta.momentum.rsi(df["Close"], window=10)
        df["_rsi_sma"] = df["_rsi"].rolling(10).mean()

    signals = list(SIGNALS.keys())
    print(f"Running {len(signals)} signal curves (max hold {MAX_HOLD} days)...")

    results = []
    start = time.time()

    for i, sig_key in enumerate(signals):
        r = run_signal_curve(sig_key, all_data)
        if r and r["total_trades"] > 0:
            results.append(r)
        if (i + 1) % 20 == 0:
            elapsed = time.time() - start
            print(f"  [{i+1}/{len(signals)}] {elapsed:.1f}s", flush=True)

    elapsed = time.time() - start
    print(f"\n{len(results)} signals computed in {elapsed:.1f}s")

    # Sort by peak avg return
    results.sort(key=lambda x: x["peak_avg"], reverse=True)

    # Save
    output = {
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "max_hold": MAX_HOLD,
        "total_signals": len(results),
        "signals": results,
    }

    path = RESULTS_DIR / "curves.json"
    with open(path, "w") as f:
        json.dump(output, f)
    print(f"Saved to {path}")

    # Print top 20
    print(f"\nTop 20 signals by peak return:")
    print(f"{'Signal':45s} {'Peak':>5s} {'PeakRet':>9s} {'PeakWR':>7s} {'1w':>8s} {'2w':>8s} {'4w':>8s} {'8w':>8s} {'Trades':>7s}")
    print("-" * 110)
    for r in results[:20]:
        kd = r["key_days"]
        print(f"{r['signal_name'][:44]:44s} d{r['peak_day']:3d} {r['peak_avg']:+8.3f}% {r['peak_wr']:6.1f}% "
              f"{kd.get('1w',{}).get('avg',0):+7.3f}% "
              f"{kd.get('2w',{}).get('avg',0):+7.3f}% "
              f"{kd.get('4w',{}).get('avg',0):+7.3f}% "
              f"{kd.get('8w',{}).get('avg',0):+7.3f}% "
              f"{r['total_trades']:6d}")


if __name__ == "__main__":
    run_all_curves()
