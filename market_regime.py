"""
Market Regime Data Module

Downloads and provides market regime data for study filters:
- VIX level (fear gauge)
- SPY trend (above/below 200-SMA)
- Dollar Index (UUP)
- Gold/Silver ratio
- Seasonality

All data cached and aligned to trading days.
"""

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import yfinance as yf
from pathlib import Path

CACHE_DIR = Path(__file__).parent / ".data"


def get_market_data(period="5y"):
    """Download VIX, SPY, UUP, GLD, SLV, TLT, BTC for regime tagging."""
    cache_file = CACHE_DIR / "market_regime.pkl"

    try:
        if cache_file.exists():
            df = pd.read_pickle(cache_file)
            if len(df) > 100:
                return df
    except Exception:
        pass

    tickers = {
        "^VIX": "vix",
        "SPY": "spy_close",
        "UUP": "dollar",
        "GLD": "gold",
        "SLV": "silver",
        "TLT": "tlt",
    }

    frames = {}
    for ticker, col in tickers.items():
        try:
            t = yf.Ticker(ticker)
            h = t.history(period=period)
            if len(h) > 50:
                s = h["Close"].copy()
                if s.index.tz is not None:
                    s.index = s.index.tz_localize(None)
                frames[col] = s
        except Exception:
            pass

    if not frames:
        return pd.DataFrame()

    df = pd.DataFrame(frames).ffill().dropna()

    # VIX regime
    df["vix_regime"] = pd.cut(df["vix"], bins=[0, 15, 25, 100], labels=["LOW_VIX", "MED_VIX", "HIGH_VIX"])

    # SPY trend
    df["spy_sma200"] = df["spy_close"].rolling(200).mean()
    df["spy_trend"] = np.where(df["spy_close"] > df["spy_sma200"], "BULL", "BEAR")

    # SPY SMA50
    df["spy_sma50"] = df["spy_close"].rolling(50).mean()
    df["spy_above_50"] = df["spy_close"] > df["spy_sma50"]

    # Dollar trend
    if "dollar" in df.columns:
        df["dollar_sma50"] = df["dollar"].rolling(50).mean()
        df["dollar_trend"] = np.where(df["dollar"] > df["dollar_sma50"], "STRONG_USD", "WEAK_USD")

    # Gold/Silver ratio
    if "gold" in df.columns and "silver" in df.columns:
        df["gold_silver_ratio"] = df["gold"] / df["silver"]
        df["gs_regime"] = pd.cut(df["gold_silver_ratio"], bins=[0, 70, 85, 200], labels=["LOW_GS", "MED_GS", "HIGH_GS"])

    # TLT trend (bonds)
    if "tlt" in df.columns:
        df["tlt_sma50"] = df["tlt"].rolling(50).mean()
        df["bond_trend"] = np.where(df["tlt"] > df["tlt_sma50"], "BONDS_UP", "BONDS_DOWN")

    # Seasonality
    df["month"] = df.index.month
    df["day_of_month"] = df.index.day
    df["day_of_week"] = df.index.dayofweek
    df["week_of_month"] = (df.index.day - 1) // 7 + 1
    df["sell_in_may"] = df["month"].isin([11, 12, 1, 2, 3, 4]).map({True: "NOV_APR", False: "MAY_OCT"})
    df["quarter_end"] = df["month"].isin([3, 6, 9, 12]) & (df["day_of_month"] >= 25)
    df["january"] = df["month"] == 1

    # Save cache
    try:
        CACHE_DIR.mkdir(exist_ok=True)
        df.to_pickle(cache_file)
    except Exception:
        pass

    return df


def get_regime_on_date(market_df, date):
    """Get all market regime info for a date."""
    if market_df is None or market_df.empty:
        return {}

    date = pd.Timestamp(date)
    if date.tz is not None:
        date = date.tz_localize(None)

    idx = market_df.index.get_indexer([date], method="ffill")[0]
    if idx < 0:
        return {}

    row = market_df.iloc[idx]
    result = {}
    for col in ["vix_regime", "spy_trend", "dollar_trend", "gs_regime", "bond_trend", "sell_in_may"]:
        if col in row:
            result[col] = str(row[col])
    if "vix" in row:
        result["vix"] = round(float(row["vix"]), 1)
    if "quarter_end" in row:
        result["quarter_end"] = bool(row["quarter_end"])
    if "january" in row:
        result["january"] = bool(row["january"])
    return result


if __name__ == "__main__":
    df = get_market_data()
    print(f"Market data: {len(df)} days")
    print(f"Date range: {df.index[0].date()} to {df.index[-1].date()}")
    last = df.iloc[-1]
    print(f"\nCurrent regime:")
    print(f"  VIX: {last.get('vix', 'N/A'):.1f} ({last.get('vix_regime', 'N/A')})")
    print(f"  SPY: {last.get('spy_trend', 'N/A')} (vs 200-SMA)")
    print(f"  Dollar: {last.get('dollar_trend', 'N/A')}")
    print(f"  Gold/Silver: {last.get('gs_regime', 'N/A')}")
    print(f"  Bonds: {last.get('bond_trend', 'N/A')}")
    print(f"  Season: {last.get('sell_in_may', 'N/A')}")
    print(f"\nRegime distribution:")
    for col in ["vix_regime", "spy_trend", "sell_in_may"]:
        if col in df.columns:
            print(f"  {col}:")
            print(df[col].value_counts().to_string().replace('\n', '\n    '))
