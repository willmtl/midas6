#!/usr/bin/env python3
"""
Stock-Level Studies V2

When sector ETF triggers, buy ALL stocks in top 10 that crossed RSI.
Adds stop loss as exit criteria. Tests with and without fundamentals filters.
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
import sector_holdings
import indicators
from studies import SIGNALS, EXITS, _rolling_sortino, _rolling_omega
from fundamentals import load_fundamentals

RESULTS_DIR = Path(__file__).parent / ".data" / "studies"


def run_all_stocks_study(signal_key, exit_key, stop_loss_pct=None, fundamental_filter=None):

    if signal_key not in SIGNALS or exit_key not in EXITS:
        return None

    _, sig_fn = SIGNALS[signal_key]
    _, exit_fn = EXITS[exit_key]

    all_etf_data = data_fetcher.fetch_all()
    spy_df = all_etf_data.get(config.BENCHMARK)
    if spy_df is None:
        return None
    spy_ret = indicators.daily_returns(spy_df)

    # Load fundamentals if needed
    fund_data = {}
    if fundamental_filter:
        try:
            from fundamentals import load_fundamentals
            fund_data = load_fundamentals()
        except Exception:
            pass

    etf_to_sector = {v: k for k, v in config.SECTOR_ETFS.items()}

    all_trades = []
    etf_trades = []

    for etf_ticker, sector_name in etf_to_sector.items():
        etf_df = all_etf_data.get(etf_ticker)
        if etf_df is None or len(etf_df) < 60:
            continue

        # Pre-compute
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

        # Load stock data
        holdings = sector_holdings.get_holdings(sector_name)
        us_tickers = [t for t in holdings if "." not in t][:10]
        stock_data = data_fetcher.fetch_tickers(us_tickers)

        # Pre-compute stock RSI + beta
        stock_info = {}
        for stk, sdf in stock_data.items():
            if len(sdf) < 30:
                continue
            sret = indicators.daily_returns(sdf)
            aligned = pd.DataFrame({"s": sret, "b": spy_ret}).dropna()
            if len(aligned) < 15:
                continue
            cov = aligned["s"].rolling(10).cov(aligned["b"])
            var = aligned["b"].rolling(10).var()
            beta = (cov / var).replace([np.inf, -np.inf], np.nan).reindex(sdf.index)

            rsi = ta.momentum.rsi(sdf["Close"], window=10)
            rsi_sma = rsi.rolling(10).mean()

            stock_info[stk] = {"df": sdf, "beta": beta, "rsi": rsi, "rsi_sma": rsi_sma}

        for entry_date in entry_dates:
            etf_idx = etf_df.index.get_loc(entry_date)
            exit_idx = exit_fn(etf_df, etf_idx)
            if exit_idx is None or exit_idx <= etf_idx:
                continue

            hold_days = exit_idx - etf_idx

            # ETF trade
            etf_entry = float(etf_df["Close"].iloc[etf_idx])
            etf_exit = float(etf_df["Close"].iloc[exit_idx])
            etf_ret = (etf_exit - etf_entry) / etf_entry * 100
            etf_trades.append(etf_ret)

            # Find ALL stocks with RSI crossed
            for stk, sinfo in stock_info.items():
                sdf = sinfo["df"]
                if entry_date not in sdf.index:
                    sidx = sdf.index.get_indexer([entry_date], method="ffill")[0]
                    if sidx < 1 or sidx >= len(sdf) - 1:
                        continue
                else:
                    sidx = sdf.index.get_loc(entry_date)

                # Check RSI crossed (within last 3 days)
                rsi = sinfo["rsi"]
                rsi_sma = sinfo["rsi_sma"]
                crossed = False
                for offset in range(0, 3):
                    i = sidx - offset
                    if i < 1:
                        continue
                    if rsi.iloc[i] > rsi_sma.iloc[i] and rsi.iloc[i-1] <= rsi_sma.iloc[i-1]:
                        crossed = True
                        break
                if not crossed and rsi.iloc[sidx] > rsi_sma.iloc[sidx]:
                    crossed = True  # already above

                if not crossed:
                    continue

                # Apply fundamental filter
                if fundamental_filter and fund_data:
                    fund = fund_data.get(stk, {})
                    if fundamental_filter == "has_dividend":
                        div_y = fund.get("dividend_yield") or fund.get("trailing_div_yield")
                        if not div_y or div_y <= 0:
                            continue
                    elif fundamental_filter == "analyst_buy":
                        rating = fund.get("analyst_rating", "")
                        if rating not in ("buy", "strong_buy"):
                            continue
                    elif fundamental_filter == "high_div":
                        div_y = fund.get("dividend_yield") or fund.get("trailing_div_yield")
                        if not div_y or div_y < 0.02:
                            continue
                    elif fundamental_filter == "revenue_growth":
                        rg = fund.get("revenue_growth")
                        if not rg or rg <= 0:
                            continue

                # Compute stock return
                s_exit_idx = min(sidx + hold_days, len(sdf) - 1)
                if s_exit_idx <= sidx:
                    continue

                s_entry = float(sdf["Close"].iloc[sidx])

                # Apply stop loss during hold period
                if stop_loss_pct:
                    stopped = False
                    for di in range(sidx + 1, s_exit_idx + 1):
                        if di >= len(sdf):
                            break
                        price = float(sdf["Close"].iloc[di])
                        pct = (price - s_entry) / s_entry * 100
                        if pct <= -stop_loss_pct:
                            s_exit = s_entry * (1 - stop_loss_pct / 100)
                            s_ret = -stop_loss_pct
                            stopped = True
                            break
                    if not stopped:
                        s_exit = float(sdf["Close"].iloc[s_exit_idx])
                        s_ret = (s_exit - s_entry) / s_entry * 100
                else:
                    s_exit = float(sdf["Close"].iloc[s_exit_idx])
                    s_ret = (s_exit - s_entry) / s_entry * 100

                beta_val = sinfo["beta"].iloc[sidx] if sidx < len(sinfo["beta"]) else np.nan

                all_trades.append({
                    "sector": sector_name, "stock": stk,
                    "beta": round(float(beta_val), 2) if not np.isnan(beta_val) else None,
                    "entry_date": str(entry_date)[:10],
                    "return": round(s_ret, 3),
                    "etf_return": round(etf_ret, 3),
                })

    if not all_trades:
        return None

    stk_rets = [t["return"] for t in all_trades]
    wins = [x for x in stk_rets if x > 0]
    losses = [x for x in stk_rets if x <= 0]

    etf_avg = np.mean(etf_trades) if etf_trades else 0
    etf_wr = sum(1 for x in etf_trades if x > 0) / len(etf_trades) * 100 if etf_trades else 0

    return {
        "signal": signal_key,
        "exit": exit_key,
        "stop_loss": stop_loss_pct,
        "filter": fundamental_filter,
        "etf_trades": len(etf_trades),
        "etf_avg": round(etf_avg, 3),
        "etf_wr": round(etf_wr, 1),
        "stock_trades": len(all_trades),
        "stock_avg": round(np.mean(stk_rets), 3),
        "stock_wr": round(len(wins) / len(stk_rets) * 100, 1),
        "avg_win": round(np.mean(wins), 3) if wins else 0,
        "avg_loss": round(np.mean(losses), 3) if losses else 0,
        "max_loss": round(min(stk_rets), 3),
        "max_win": round(max(stk_rets), 3),
        "wl_ratio": round(abs(np.mean(wins) / np.mean(losses)), 2) if losses and np.mean(losses) != 0 else 0,
        "edge": round(np.mean(stk_rets) - etf_avg, 3),
        "stocks_per_signal": round(len(all_trades) / len(etf_trades), 1) if etf_trades else 0,
    }


if __name__ == "__main__":
    signals = [
        "higher_low_rsi_x_omega",
        "higher_low_rsi_x",
        "rsi_x_sma_below50",
        "rsi_x_triple",
        "rsi_x_pos_omega",
        "seq_rs_rsi_10d",
        "gap_down_large",
        "new_52low",
        "rsi_oversold30",
    ]

    exits = ["4w", "8w", "trail_10", "tp_10", "tp_15"]
    stop_losses = [None, 10, 8, 5]
    filters = [None, "has_dividend", "analyst_buy", "revenue_growth"]

    combos = []
    for sig in signals:
        for ex in exits:
            for sl in stop_losses:
                for filt in filters:
                    combos.append((sig, ex, sl, filt))

    print(f"Running {len(combos)} stock-level studies...")
    print(f"{'Signal':30s} {'Exit':10s} {'SL':>4s} {'Filter':>15s} {'ETF':>8s} {'Stock':>8s} {'Edge':>7s} {'WR':>6s} {'W/L':>5s} {'MaxL':>7s} {'#':>5s}")
    print("-" * 120)

    results = []
    for i, (sig, ex, sl, filt) in enumerate(combos):
        r = run_all_stocks_study(sig, ex, sl, filt)
        if r and r["stock_trades"] > 5:
            sl_str = f"{sl}%" if sl else "None"
            filt_str = filt or "None"
            print(f"{sig:30s} {ex:10s} {sl_str:>4s} {filt_str:>15s} {r['etf_avg']:+7.2f}% {r['stock_avg']:+7.2f}% {r['edge']:+6.2f}% {r['stock_wr']:5.1f}% {r['wl_ratio']:4.1f}x {r['max_loss']:+6.1f}% {r['stock_trades']:5d}")
            results.append(r)

        if (i + 1) % 20 == 0:
            print(f"  [{i+1}/{len(combos)}]", flush=True)

    # Save
    output = {"generated_at": str(time.time()), "count": len(results), "studies": results}
    with open(RESULTS_DIR / "stock_studies.json", "w") as f:
        json.dump(output, f)

    print(f"\n{len(results)} studies saved")
    top5 = sorted(results, key=lambda x: x["stock_avg"], reverse=True)[:5]
    print("\nTop 5:")
    for r in top5:
        print(f"  {r['signal']:30s} {r['exit']:10s} SL={r['stop_loss']}  avg={r['stock_avg']:+.2f}%  wr={r['stock_wr']:.0f}%  edge={r['edge']:+.2f}%")
