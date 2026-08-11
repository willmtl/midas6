"""
Interest Rate Data Module

Downloads and provides US interest rate data for regime tagging.
Uses ^TNX (10Y Treasury) and ^IRX (13W T-Bill) as proxies.

Rate regimes:
  - LOW: short rate < 2%
  - MEDIUM: 2% <= short rate < 4%
  - HIGH: short rate >= 4%

Yield curve:
  - NORMAL: 10Y > short rate (positive slope)
  - INVERTED: 10Y < short rate (negative slope)
"""

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import yfinance as yf
from pathlib import Path

CACHE_DIR = Path(__file__).parent / ".data"


def get_rates(period="5y"):
    """Download and return daily interest rate data."""
    cache_file = CACHE_DIR / "rates.pkl"

    # Try cache first
    try:
        if cache_file.exists():
            df = pd.read_pickle(cache_file)
            if len(df) > 100:
                return df
    except Exception:
        pass

    # Download
    tnx = yf.Ticker("^TNX")  # 10Y Treasury Yield
    irx = yf.Ticker("^IRX")  # 13W T-Bill (short-term rate proxy)

    df_10y = tnx.history(period=period)[["Close"]].rename(columns={"Close": "rate_10y"})
    df_3m = irx.history(period=period)[["Close"]].rename(columns={"Close": "rate_3m"})

    # Merge on date
    df = df_10y.join(df_3m, how="outer").sort_index()
    df = df.ffill().dropna()

    # Remove timezone if present
    if df.index.tz is not None:
        df.index = df.index.tz_localize(None)

    # Compute regime
    df["regime"] = pd.cut(
        df["rate_3m"],
        bins=[-float("inf"), 2.0, 4.0, float("inf")],
        labels=["LOW", "MEDIUM", "HIGH"],
    )

    # Yield curve
    df["spread_10y_3m"] = df["rate_10y"] - df["rate_3m"]
    df["curve"] = df["spread_10y_3m"].apply(lambda x: "NORMAL" if x > 0 else "INVERTED")

    # Rate trend (is the short rate rising or falling over last 20 days?)
    df["rate_sma20"] = df["rate_3m"].rolling(20).mean()
    # Emit NaN during the 20-bar SMA warmup instead of forcing "FALLING"
    # (rate_3m > NaN → False → bogus FALLING on the first 19 rows). Build the string labels
    # first, then NaN the warmup on an object array — np.nan (float) and str can't share one np.where.
    _rt_lab = np.where(df["rate_3m"] > df["rate_sma20"], "RISING", "FALLING").astype(object)
    _rt_lab[df["rate_sma20"].isna().to_numpy()] = np.nan
    df["rate_trend"] = _rt_lab

    # Save cache
    try:
        CACHE_DIR.mkdir(exist_ok=True)
        df.to_pickle(cache_file)
    except Exception:
        pass

    return df


def get_regime_on_date(rates_df, date):
    """Get the rate regime for a specific date."""
    if rates_df is None:
        return {"regime": "UNKNOWN", "curve": "UNKNOWN", "rate_trend": "UNKNOWN",
                "rate_3m": None, "rate_10y": None}

    # Find closest date
    date = pd.Timestamp(date)
    if date.tz is not None:
        date = date.tz_localize(None)

    idx = rates_df.index.get_indexer([date], method="ffill")[0]
    if idx < 0:
        return {"regime": "UNKNOWN", "curve": "UNKNOWN", "rate_trend": "UNKNOWN",
                "rate_3m": None, "rate_10y": None}

    row = rates_df.iloc[idx]
    return {
        "regime": str(row.get("regime", "UNKNOWN")),
        "curve": str(row.get("curve", "UNKNOWN")),
        "rate_trend": str(row.get("rate_trend", "UNKNOWN")),
        "rate_3m": round(float(row.get("rate_3m", 0)), 2),
        "rate_10y": round(float(row.get("rate_10y", 0)), 2),
    }


def tag_trades_with_regime(trades, rates_df):
    """Add regime info to each trade dict."""
    for trade in trades:
        entry_date = trade.get("entry_date")
        if entry_date:
            info = get_regime_on_date(rates_df, entry_date)
            trade.update(info)
    return trades


def split_by_regime(trades):
    """Split trades into regime buckets and compute stats per regime."""
    buckets = {}
    for t in trades:
        regime = t.get("regime", "UNKNOWN")
        if regime not in buckets:
            buckets[regime] = []
        buckets[regime].append(t)

    results = {}
    for regime, regime_trades in buckets.items():
        returns = [t.get("return", t.get("return_pct", 0)) for t in regime_trades]
        wins = sum(1 for r in returns if r > 0)
        total = len(returns)
        avg = sum(returns) / total if total else 0
        results[regime] = {
            "trades": total,
            "avg_return": round(avg, 3),
            "win_rate": round(wins / total * 100, 1) if total else 0,
            "total_return": round(sum(returns), 2),
        }
    return results


def split_by_curve(trades):
    """Split trades by yield curve shape."""
    buckets = {}
    for t in trades:
        curve = t.get("curve", "UNKNOWN")
        if curve not in buckets:
            buckets[curve] = []
        buckets[curve].append(t)

    results = {}
    for curve, curve_trades in buckets.items():
        returns = [t.get("return", t.get("return_pct", 0)) for t in curve_trades]
        wins = sum(1 for r in returns if r > 0)
        total = len(returns)
        avg = sum(returns) / total if total else 0
        results[curve] = {
            "trades": total,
            "avg_return": round(avg, 3),
            "win_rate": round(wins / total * 100, 1) if total else 0,
        }
    return results


if __name__ == "__main__":
    df = get_rates()
    print(f"Rate data: {len(df)} days")
    print(f"Date range: {df.index[0].date()} to {df.index[-1].date()}")
    print(f"\nCurrent rates:")
    last = df.iloc[-1]
    print(f"  3M T-Bill: {last['rate_3m']:.2f}%")
    print(f"  10Y Treasury: {last['rate_10y']:.2f}%")
    print(f"  Spread: {last['spread_10y_3m']:.2f}%")
    print(f"  Regime: {last['regime']}")
    print(f"  Curve: {last['curve']}")
    print(f"  Trend: {last['rate_trend']}")

    print(f"\nRegime distribution:")
    print(df["regime"].value_counts().to_string())
    print(f"\nCurve distribution:")
    print(df["curve"].value_counts().to_string())
