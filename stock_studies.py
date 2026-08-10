#!/usr/bin/env python3
"""
Stock-Level Studies

When a sector ETF triggers a signal, drill into its top 10 holdings:
1. Find stocks that also have RSI crossed above SMA
2. Pick the one with highest beta vs SPY
3. Buy that stock, hold per exit condition

Compares: buying the ETF vs buying the highest-beta stock within it.
"""

import warnings
warnings.filterwarnings("ignore")

import json
import time
import numpy as np
import pandas as pd
import ta
from pathlib import Path
from datetime import datetime

import config
import data_fetcher
import sector_holdings
import indicators

RESULTS_DIR = Path(__file__).parent / ".data" / "studies"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


def _get_stock_data(sector_name, period="5y"):
    """Fetch data for top 10 US stocks in a sector."""
    holdings = sector_holdings.get_holdings(sector_name)
    if not holdings:
        return {}
    us = [t for t in holdings if "." not in t][:10]
    return data_fetcher.fetch_tickers(us, period)


def _compute_beta(stock_ret, spy_ret, window=10):
    """Rolling beta of stock vs SPY."""
    aligned = pd.DataFrame({"s": stock_ret, "b": spy_ret}).dropna()
    if len(aligned) < window + 5:
        return pd.Series(np.nan, index=stock_ret.index)
    cov = aligned["s"].rolling(window).cov(aligned["b"])
    var = aligned["b"].rolling(window).var()
    beta = (cov / var).replace([np.inf, -np.inf], np.nan)
    return beta.reindex(stock_ret.index)


def _stock_rsi_crossed(df, at_idx):
    """Check if stock's RSI crossed above its SMA at or within 1 day of given index."""
    rsi = ta.momentum.rsi(df["Close"], window=10)
    sma = rsi.rolling(10).mean()
    for offset in range(0, 3):  # check today and last 2 days
        i = at_idx - offset
        if i < 1 or i >= len(rsi):
            continue
        if rsi.iloc[i] > sma.iloc[i] and rsi.iloc[i-1] <= sma.iloc[i-1]:
            return True
    # Also accept if RSI is above SMA (already crossed recently)
    if at_idx < len(rsi) and rsi.iloc[at_idx] > sma.iloc[at_idx]:
        return True
    return False


def run_stock_study(signal_key, exit_key, period="5y"):
    """
    For each sector ETF signal trigger:
    1. Find stocks in that sector with RSI crossed
    2. Pick highest beta stock
    3. Buy that stock
    4. Compare vs buying the ETF directly
    """
    from studies import SIGNALS, EXITS, _rolling_sortino, _rolling_omega

    if signal_key not in SIGNALS or exit_key not in EXITS:
        return None

    _, sig_fn = SIGNALS[signal_key]
    _, exit_fn = EXITS[exit_key]

    # Load ETF data
    all_etf_data = data_fetcher.fetch_all()
    spy_df = all_etf_data.get(config.BENCHMARK)
    if spy_df is None:
        return None
    spy_ret = indicators.daily_returns(spy_df)

    etf_to_sector = {v: k for k, v in config.SECTOR_ETFS.items()}

    etf_trades = []
    stock_trades = []
    stock_picks = []

    for etf_ticker, sector_name in etf_to_sector.items():
        etf_df = all_etf_data.get(etf_ticker)
        if etf_df is None or len(etf_df) < 60:
            continue

        # Pre-compute for signal
        etf_df["_sortino"] = _rolling_sortino(etf_df)
        etf_df["_omega"] = _rolling_omega(etf_df)
        etf_df["_rsi"] = ta.momentum.rsi(etf_df["Close"], window=10)
        etf_df["_rsi_sma"] = etf_df["_rsi"].rolling(10).mean()

        try:
            signals = sig_fn(etf_df).fillna(False)
        except Exception:
            continue

        entry_dates = signals[signals].index.tolist()
        if not entry_dates:
            continue

        # Load stock data for this sector
        stock_data = _get_stock_data(sector_name, period)
        if not stock_data:
            continue

        # Pre-compute stock betas and RSI
        stock_info = {}
        for sticker, sdf in stock_data.items():
            if len(sdf) < 30:
                continue
            sret = indicators.daily_returns(sdf)
            beta = _compute_beta(sret, spy_ret)
            stock_info[sticker] = {
                "df": sdf,
                "beta": beta,
            }

        for entry_date in entry_dates:
            etf_idx = etf_df.index.get_loc(entry_date)
            exit_idx = exit_fn(etf_df, etf_idx)
            if exit_idx is None or exit_idx <= etf_idx:
                continue

            # ETF trade
            etf_entry = float(etf_df["Close"].iloc[etf_idx])
            etf_exit = float(etf_df["Close"].iloc[exit_idx])
            etf_ret = (etf_exit - etf_entry) / etf_entry * 100
            etf_trades.append({
                "sector": sector_name, "ticker": etf_ticker,
                "entry_date": str(entry_date)[:10],
                "return": round(etf_ret, 3),
            })

            # Find highest beta stock with RSI crossed
            best_stock = None
            best_beta = -999

            for sticker, sinfo in stock_info.items():
                sdf = sinfo["df"]
                # Find closest date in stock data
                if entry_date not in sdf.index:
                    # Try to find nearest
                    sidx = sdf.index.get_indexer([entry_date], method="ffill")[0]
                    if sidx < 0 or sidx >= len(sdf) - 1:
                        continue
                else:
                    sidx = sdf.index.get_loc(entry_date)

                # Check RSI crossed
                if not _stock_rsi_crossed(sdf, sidx):
                    continue

                # Get beta
                beta_val = sinfo["beta"].iloc[sidx] if sidx < len(sinfo["beta"]) else np.nan
                if np.isnan(beta_val):
                    continue

                if beta_val > best_beta:
                    best_beta = beta_val
                    best_stock = (sticker, sdf, sidx)

            if best_stock:
                sticker, sdf, sidx = best_stock
                # Hold same number of days as ETF
                hold_days = exit_idx - etf_idx
                s_exit_idx = min(sidx + hold_days, len(sdf) - 1)
                if s_exit_idx <= sidx:
                    continue

                s_entry = float(sdf["Close"].iloc[sidx])
                s_exit = float(sdf["Close"].iloc[s_exit_idx])
                s_ret = (s_exit - s_entry) / s_entry * 100

                stock_trades.append({
                    "sector": sector_name, "etf": etf_ticker,
                    "stock": sticker, "beta": round(best_beta, 2),
                    "entry_date": str(entry_date)[:10],
                    "return": round(s_ret, 3),
                    "etf_return": round(etf_ret, 3),
                })

    # Aggregate
    etf_avg = sum(t["return"] for t in etf_trades) / len(etf_trades) if etf_trades else 0
    etf_wr = sum(1 for t in etf_trades if t["return"] > 0) / len(etf_trades) * 100 if etf_trades else 0

    stock_avg = sum(t["return"] for t in stock_trades) / len(stock_trades) if stock_trades else 0
    stock_wr = sum(1 for t in stock_trades if t["return"] > 0) / len(stock_trades) * 100 if stock_trades else 0

    return {
        "signal": signal_key,
        "exit": exit_key,
        "etf_trades": len(etf_trades),
        "etf_avg_return": round(etf_avg, 3),
        "etf_win_rate": round(etf_wr, 1),
        "stock_trades": len(stock_trades),
        "stock_avg_return": round(stock_avg, 3),
        "stock_win_rate": round(stock_wr, 1),
        "edge": round(stock_avg - etf_avg, 3),
        "all_trades": stock_trades,
        "all_etf_trades": etf_trades,
    }


if __name__ == "__main__":
    # Test with the best performing signals
    test_combos = [
        ("higher_low_rsi_x", "4w"),
        ("higher_low_rsi_x_omega", "4w"),
        ("rsi_x_sma_below50", "4w"),
        ("rsi_x_pos_omega", "4w"),
        ("rsi_x_triple", "4w"),
        ("seq_rs_rsi_10d", "4w"),
        ("seq_rs_hl_rsi_10d", "4w"),
        ("higher_low_rsi_x", "trail_10"),
        ("higher_low_rsi_x_omega", "trail_10"),
        ("rsi_x_sma_below50", "trail_10"),
        ("gap_down_large", "4w"),
        ("new_52low", "4w"),
        ("rsi_oversold30", "4w"),
        ("higher_low_rsi_x", "tp_10"),
        ("rsi_x_pos_omega", "tp_10"),
    ]

    print("Running stock-level studies (highest beta + RSI crossed)...")
    print(f"{'Signal':35s} {'Exit':15s} {'ETF Avg':>8s} {'ETF WR':>7s} {'Stk Avg':>8s} {'Stk WR':>7s} {'Edge':>7s} {'ETF#':>5s} {'Stk#':>5s}")
    print("-" * 110)

    for sig, exit_k in test_combos:
        t0 = time.time()
        r = run_stock_study(sig, exit_k)
        elapsed = time.time() - t0
        if r and r["stock_trades"] > 0:
            print(f"{sig:35s} {exit_k:15s} {r['etf_avg_return']:+7.3f}% {r['etf_win_rate']:6.1f}% {r['stock_avg_return']:+7.3f}% {r['stock_win_rate']:6.1f}% {r['edge']:+6.3f}% {r['etf_trades']:5d} {r['stock_trades']:5d}  ({elapsed:.0f}s)")
        elif r:
            print(f"{sig:35s} {exit_k:15s} {r['etf_avg_return']:+7.3f}% {r['etf_win_rate']:6.1f}%  no stock picks  ({elapsed:.0f}s)")
