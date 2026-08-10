"""
Stock Market Trend Bot - Data Fetcher

Downloads historical data. Uses incremental updates:
  1. If no saved data → full download (5y)
  2. If saved but last_date is not today → fetch only from last_date to now
  3. If saved and last_date is today → use cached data as-is
"""

from datetime import datetime, timedelta, date

import pandas as pd
import yfinance as yf

import config
import data_store


def fetch_all(
    period: str = config.DEFAULT_PERIOD,
    interval: str = config.DEFAULT_INTERVAL,
) -> dict[str, pd.DataFrame]:
    tickers = list(config.SECTOR_ETFS.values()) + [config.BENCHMARK]
    return _fetch(tickers, period, interval)


def fetch_tickers(
    tickers: list[str],
    period: str = config.DEFAULT_PERIOD,
    interval: str = config.DEFAULT_INTERVAL,
) -> dict[str, pd.DataFrame]:
    return _fetch(tickers, period, interval)


def _fetch(
    tickers: list[str],
    period: str,
    interval: str,
) -> dict[str, pd.DataFrame]:
    result = {}
    need_full = []
    need_incremental = {}  # ticker -> last_date

    today = date.today().isoformat()

    for ticker in tickers:
        meta = data_store.get_metadata(ticker, period)

        if meta is None:
            # No saved data → full download
            need_full.append(ticker)
            continue

        last_date = meta.get("last_date")

        if last_date == today:
            # Already up to date
            df = data_store.load_ticker(ticker, period)
            if df is not None and not df.empty and len(df) > 20:
                result[ticker] = df
                continue
            else:
                need_full.append(ticker)
                continue

        if last_date:
            # Load existing data first (always available as fallback)
            df = data_store.load_ticker(ticker, period)
            if df is not None and not df.empty and len(df) > 20:
                result[ticker] = df
            # Then try incremental update
            need_incremental[ticker] = last_date
        else:
            need_full.append(ticker)

    # Full downloads
    if need_full:
        _batch_download(need_full, period, interval, result)

    # Incremental downloads
    if need_incremental:
        _incremental_download(need_incremental, period, interval, result)

    return result


def _batch_download(
    tickers: list[str],
    period: str,
    interval: str,
    result: dict,
):
    """Full download for tickers with no saved data."""
    data = yf.download(
        tickers=tickers,
        period=period,
        interval=interval,
        group_by="ticker",
        auto_adjust=True,
        threads=True,
        progress=False,
    )

    if data is not None and not data.empty:
        for ticker in tickers:
            try:
                if len(tickers) == 1:
                    df = data.copy()
                else:
                    df = data[ticker].copy()
                df = _clean_df(df)
                if df is not None:
                    result[ticker] = df
                    data_store.save_ticker(ticker, df, period)
            except (KeyError, Exception):
                pass


def _incremental_download(
    ticker_dates: dict[str, str],
    period: str,
    interval: str,
    result: dict,
):
    """Fetch only new data from last_date onwards, append to existing."""
    tickers = list(ticker_dates.keys())

    # Find the earliest last_date to use as start
    earliest = min(ticker_dates.values())
    # Start from the day before to ensure overlap for dedup
    start_dt = datetime.strptime(earliest, "%Y-%m-%d") - timedelta(days=1)
    start_str = start_dt.strftime("%Y-%m-%d")

    data = yf.download(
        tickers=tickers,
        start=start_str,
        interval=interval,
        group_by="ticker",
        auto_adjust=True,
        threads=True,
        progress=False,
    )

    if data is not None and not data.empty:
        for ticker in tickers:
            try:
                if len(tickers) == 1:
                    new_df = data.copy()
                else:
                    new_df = data[ticker].copy()
                new_df = _clean_df(new_df)
                if new_df is not None:
                    data_store.append_candles(ticker, new_df, period)
                    # Reload full dataset
                    full_df = data_store.load_ticker(ticker, period)
                    if full_df is not None and not full_df.empty:
                        result[ticker] = full_df
            except (KeyError, Exception):
                # Fallback: load existing
                df = data_store.load_ticker(ticker, period)
                if df is not None and not df.empty:
                    result[ticker] = df


def _clean_df(df: pd.DataFrame) -> pd.DataFrame | None:
    """Flatten MultiIndex columns, drop NaN rows."""
    if isinstance(df.columns, pd.MultiIndex):
        level0 = [c[0] for c in df.columns]
        level1 = [c[1] if len(c) > 1 else c[0] for c in df.columns]
        if len(set(level1)) == 1:
            df.columns = level0
        elif len(set(level0)) == 1:
            df.columns = level1
        else:
            df.columns = level0
    df = df.dropna(how="all")
    if df.empty or len(df) < 5:
        return None
    return df
