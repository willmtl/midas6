"""
Stock Market Trend Bot - JSON Data Store

Saves and loads candle data (OHLCV) as JSON files.
One file per ticker per period: .data/{period}/{TICKER}.json

Tracks last_date and complete flag so incremental updates
only fetch new data from where we left off.
"""

import json
from pathlib import Path
from datetime import datetime, date

import pandas as pd

DATA_DIR = Path(__file__).parent / ".data"


def _ticker_path(ticker: str, period: str) -> Path:
    safe_ticker = ticker.replace("/", "_").replace("\\", "_")
    folder = DATA_DIR / period
    folder.mkdir(parents=True, exist_ok=True)
    return folder / f"{safe_ticker}.json"


def save_ticker(ticker: str, df: pd.DataFrame, period: str):
    """Save a ticker's OHLCV DataFrame as JSON with metadata."""
    path = _ticker_path(ticker, period)
    records = []
    for dt, row in df.iterrows():
        rec = {"date": str(dt)[:10]}
        for col in df.columns:
            val = row[col]
            if pd.notna(val):
                rec[col.lower()] = round(float(val), 6) if col != "Volume" else int(val)
        records.append(rec)

    last_date = records[-1]["date"] if records else None
    today = date.today().isoformat()
    # Data is complete if the last candle is today (market may still be open)
    # or if today is a weekend/holiday and last candle is the most recent trading day
    complete = last_date == today

    data = {
        "ticker": ticker,
        "period": period,
        "saved_at": datetime.now().isoformat(),
        "last_date": last_date,
        "complete": complete,
        "bars": len(records),
        "candles": records,
    }

    with open(path, "w") as f:
        json.dump(data, f)


def load_ticker(ticker: str, period: str) -> pd.DataFrame | None:
    """Load a ticker's OHLCV from JSON."""
    path = _ticker_path(ticker, period)
    if not path.exists():
        return None

    with open(path, "r") as f:
        data = json.load(f)

    records = data.get("candles", [])
    if not records:
        return None

    df = pd.DataFrame(records)
    df["date"] = pd.to_datetime(df["date"])
    df = df.set_index("date")
    df.columns = [c.title() for c in df.columns]
    return df


def get_metadata(ticker: str, period: str) -> dict | None:
    """Get saved metadata without loading all candles."""
    path = _ticker_path(ticker, period)
    if not path.exists():
        return None

    with open(path, "r") as f:
        data = json.load(f)

    return {
        "ticker": data.get("ticker"),
        "last_date": data.get("last_date"),
        "complete": data.get("complete", False),
        "bars": data.get("bars", 0),
        "saved_at": data.get("saved_at"),
    }


def append_candles(ticker: str, new_df: pd.DataFrame, period: str):
    """Append new candles to existing saved data."""
    path = _ticker_path(ticker, period)
    existing_df = load_ticker(ticker, period)

    if existing_df is not None and not existing_df.empty:
        # Combine, drop duplicates by date index, sort
        combined = pd.concat([existing_df, new_df])
        combined = combined[~combined.index.duplicated(keep='last')]
        combined = combined.sort_index()
        save_ticker(ticker, combined, period)
    else:
        save_ticker(ticker, new_df, period)


def save_all(ticker_data: dict[str, pd.DataFrame], period: str):
    for ticker, df in ticker_data.items():
        save_ticker(ticker, df, period)


def load_all(tickers: list[str], period: str) -> dict[str, pd.DataFrame]:
    result = {}
    for ticker in tickers:
        df = load_ticker(ticker, period)
        if df is not None:
            result[ticker] = df
    return result


def has_ticker(ticker: str, period: str) -> bool:
    return _ticker_path(ticker, period).exists()


def list_saved(period: str) -> list[str]:
    folder = DATA_DIR / period
    if not folder.exists():
        return []
    return [p.stem for p in folder.glob("*.json")]
