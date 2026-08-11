#!/usr/bin/env python3
"""
Stock Drilldown Engine

For top indicator studies: when a sector ETF signal fires,
buy the highest-beta stock in that sector's top 20 holdings.
Compare stock-level returns vs ETF-level returns.
"""

import warnings
warnings.filterwarnings("ignore")

import time
import numpy as np
import pandas as pd
import ta
from pathlib import Path
from collections import defaultdict

import config
import data_fetcher
import sector_holdings
import studies as studies_mod


def load_all_stock_data():
    """Load price data for all sector holdings."""
    tickers = set()
    for name, data in sector_holdings.HOLDINGS.items():
        for h in data.get("holdings", []):
            tickers.add(h)

    print(f"Loading {len(tickers)} stock tickers...")
    all_data = {}

    # Load in batches to avoid rate limits
    ticker_list = sorted(tickers)
    batch_size = 50
    for i in range(0, len(ticker_list), batch_size):
        batch = ticker_list[i:i+batch_size]
        try:
            fetched = data_fetcher.fetch_tickers(batch, config.DEFAULT_PERIOD, config.DEFAULT_INTERVAL)
            all_data.update(fetched)
        except Exception as e:
            print(f"  Batch {i}-{i+batch_size} error: {e}")

    print(f"Loaded {len(all_data)} stocks")
    return all_data


def compute_betas(stock_data, spy_data, window=60):
    """Pre-compute rolling beta for each stock vs SPY."""
    spy_ret = spy_data["Close"].pct_change().dropna()
    betas = {}

    for ticker, df in stock_data.items():
        try:
            stock_ret = df["Close"].pct_change().dropna()
            # Align
            aligned = pd.DataFrame({"stock": stock_ret, "spy": spy_ret}).dropna()
            if len(aligned) < window:
                continue
            # Rolling beta
            cov = aligned["stock"].rolling(window).cov(aligned["spy"])
            var = aligned["spy"].rolling(window).var()
            beta = (cov / var).replace([np.inf, -np.inf], np.nan).reindex(df.index)
            betas[ticker] = beta
        except Exception:
            continue

    return betas


def get_sector_holdings_map():
    """Build ETF -> list of stock tickers."""
    etf_to_holdings = {}
    for name, data in sector_holdings.HOLDINGS.items():
        etf = data.get("etf")
        if etf and data.get("holdings"):
            etf_to_holdings[etf] = data["holdings"]
    return etf_to_holdings


def run_stock_drilldown(study_obj, etf_data, stock_data, betas, etf_to_holdings):
    """
    For a given study, replay each sector signal but buy the highest-beta stock.
    Returns stock-level stats and comparison.
    """
    sig_key = study_obj.signal_key
    exit_key = study_obj.exit_key

    if sig_key not in studies_mod.SIGNALS or exit_key not in studies_mod.EXITS:
        return None

    _, sig_fn = studies_mod.SIGNALS[sig_key]
    _, exit_fn = studies_mod.EXITS[exit_key]

    etf_to_sector = {v: k for k, v in config.SECTOR_ETFS.items()}
    stock_returns = []
    etf_returns = []
    stock_stats = defaultdict(lambda: {"trades": 0, "total_ret": 0, "wins": 0})

    for etf, sector_name in etf_to_sector.items():
        df = etf_data.get(etf)
        if df is None or len(df) < 60:
            continue

        holdings = etf_to_holdings.get(etf, [])
        if not holdings:
            continue

        try:
            signals = sig_fn(df).fillna(False)
        except Exception:
            continue

        entry_dates = signals[signals].index.tolist()

        for entry_date in entry_dates:
            idx = df.index.get_loc(entry_date)
            exit_idx = exit_fn(df, idx)
            if exit_idx is None or exit_idx <= idx or exit_idx >= len(df):
                continue

            # ETF return
            ep = float(df["Close"].iloc[idx])
            xp = float(df["Close"].iloc[exit_idx])
            if ep <= 0:
                continue
            etf_ret = (xp - ep) / ep * 100
            etf_returns.append(etf_ret)

            # Find highest beta stock in holdings
            best_stock = None
            best_beta = -999
            entry_str = str(entry_date)[:10]

            for ticker in holdings:
                if ticker not in betas or ticker not in stock_data:
                    continue
                beta_series = betas[ticker]
                # Find beta at entry date
                try:
                    b_val = beta_series.asof(entry_date)
                    if pd.notna(b_val) and b_val > best_beta:
                        best_beta = b_val
                        best_stock = ticker
                except Exception:
                    continue

            if best_stock is None:
                continue

            # Simulate stock trade
            sdf = stock_data[best_stock]
            # Find matching dates
            try:
                # Entry must use the last stock bar ON/BEFORE the signal date (ffill) — "nearest"
                # could snap FORWARD to the next session when the exact bar is missing (halt/late
                # listing), buying at a price that didn't exist at signal time (1-bar lookahead).
                s_entry_idx = sdf.index.get_indexer([entry_date], method="ffill")[0]
                exit_date = df.index[exit_idx]
                s_exit_idx = sdf.index.get_indexer([exit_date], method="nearest")[0]
            except Exception:
                continue

            if s_entry_idx < 0 or s_exit_idx < 0 or s_exit_idx <= s_entry_idx \
                    or s_entry_idx >= len(sdf) or s_exit_idx >= len(sdf):
                continue

            s_ep = float(sdf["Close"].iloc[s_entry_idx])
            s_xp = float(sdf["Close"].iloc[s_exit_idx])
            if s_ep <= 0:
                continue
            s_ret = (s_xp - s_ep) / s_ep * 100
            stock_returns.append(s_ret)

            # Track per-stock stats
            stock_stats[best_stock]["trades"] += 1
            stock_stats[best_stock]["total_ret"] += s_ret
            if s_ret > 0:
                stock_stats[best_stock]["wins"] += 1

    if not stock_returns:
        return None

    # Aggregate
    stock_avg = sum(stock_returns) / len(stock_returns)
    etf_avg = sum(etf_returns) / len(etf_returns) if etf_returns else 0
    stock_wins = sum(1 for r in stock_returns if r > 0)

    # Max drawdown (simple)
    max_dd = min(stock_returns) if stock_returns else 0

    # Best/worst stocks
    stock_perf = []
    for ticker, stats in stock_stats.items():
        if stats["trades"] > 0:
            stock_perf.append({
                "ticker": ticker,
                "trades": stats["trades"],
                "avg_return": round(stats["total_ret"] / stats["trades"], 2),
                "win_rate": round(stats["wins"] / stats["trades"] * 100, 1),
            })
    stock_perf.sort(key=lambda x: x["avg_return"], reverse=True)

    return {
        "stock_trades": len(stock_returns),
        "stock_avg_return": round(stock_avg, 3),
        "stock_win_rate": round(stock_wins / len(stock_returns) * 100, 1),
        "stock_avg_hold": study_obj.avg_hold,
        "stock_max_drawdown": round(max_dd, 2),
        "etf_avg_return": round(etf_avg, 3),
        "alpha_vs_etf": round(stock_avg - etf_avg, 3),
        "best_stocks": stock_perf[:5],
        "worst_stocks": stock_perf[-5:] if len(stock_perf) > 5 else [],
    }


def run_all():
    """Run stock drilldown for top 10% of indicator studies."""
    from core.models import Study, StockDrilldown
    from django.utils import timezone

    # Get top 10%
    total = Study.objects.filter(is_computed=True).count()
    cutoff = int(total * 0.1)
    top_studies = list(Study.objects.filter(is_computed=True).order_by('-avg_return')[:cutoff])
    print(f"Top 10%: {len(top_studies)} studies (avg_return >= {top_studies[-1].avg_return:+.3f}%)")

    # Skip already computed
    existing = set(StockDrilldown.objects.values_list('study_id', flat=True))
    to_compute = [s for s in top_studies if s.id not in existing]
    print(f"Already computed: {len(existing)}, New: {len(to_compute)}")

    if not to_compute:
        print("All done!")
        return

    # Load data
    print("Loading ETF data...")
    etf_data = data_fetcher.fetch_all()
    print(f"ETFs: {len(etf_data)}")

    print("Loading stock data...")
    stock_data = load_all_stock_data()

    # SPY for beta
    if "SPY" not in etf_data:
        spy = data_fetcher.fetch_tickers(["SPY"], config.DEFAULT_PERIOD, config.DEFAULT_INTERVAL)
        etf_data.update(spy)

    print("Computing betas...")
    betas = compute_betas(stock_data, etf_data["SPY"])
    print(f"Betas computed for {len(betas)} stocks")

    etf_to_holdings = get_sector_holdings_map()

    # Pre-compute indicators on ETF data (same as studies.py)
    for ticker, df in etf_data.items():
        if len(df) < 20:
            continue
        df["_sortino"] = studies_mod._rolling_sortino(df)
        df["_omega"] = studies_mod._rolling_omega(df)
        df["_rsi"] = ta.momentum.rsi(df["Close"], window=10)
        df["_rsi_sma"] = df["_rsi"].rolling(10).mean()
        df["_rsi_sort"] = studies_mod._rsi_of_sortino(df)
        df["_rsi_sort_sma"] = df["_rsi_sort"].rolling(10).mean()

    now = timezone.now()
    done = 0
    t0 = time.time()

    for i, study in enumerate(to_compute):
        result = run_stock_drilldown(study, etf_data, stock_data, betas, etf_to_holdings)
        if result:
            StockDrilldown.objects.update_or_create(
                study=study,
                defaults={**result, "computed_at": now},
            )
            done += 1

        if (i + 1) % 50 == 0:
            elapsed = time.time() - t0
            print(f"  [{i+1}/{len(to_compute)}] {elapsed:.1f}s  ({done} with trades)", flush=True)

    elapsed = time.time() - t0
    print(f"\nDone: {done} drilldowns computed in {elapsed:.1f}s")

    # Print top results
    top = StockDrilldown.objects.order_by('-stock_avg_return')[:10]
    print("\nTop 10 stock drilldowns:")
    for d in top:
        print(f"  {d.study.name[:50]:50s}  stock={d.stock_avg_return:+.2f}%  etf={d.etf_avg_return:+.2f}%  alpha={d.alpha_vs_etf:+.2f}%  trades={d.stock_trades}")


if __name__ == "__main__":
    import os, sys, django
    os.environ.setdefault("DJANGO_SETTINGS_MODULE", "rotation.settings")
    for p in [Path(__file__).parent / "backend", Path("/app")]:
        if p.exists() and str(p) not in sys.path:
            sys.path.insert(0, str(p))
    django.setup()
    run_all()
