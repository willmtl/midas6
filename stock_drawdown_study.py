#!/usr/bin/env python3
"""Backtest: buy stocks >=30% below their running ATH when RSI(10) crosses above its SMA.
Applies the `dd30_rsi_reversal` signal to each stock in the sector-holdings universe,
across all EXITS. Saves .data/studies/stock_drawdown.json.  Run: python stock_drawdown_study.py
"""
import warnings
warnings.filterwarnings("ignore")

import json
from pathlib import Path

import config
import data_fetcher
import sector_holdings
from studies import SIGNALS, EXITS

STUDIES_DIR = Path(__file__).parent / ".data" / "studies"
STUDIES_DIR.mkdir(parents=True, exist_ok=True)

_SIG_NAME, _SIG_FN = SIGNALS["dd30_rsi_reversal"]
MIN_BARS = 60
MIN_STOCK_TRADES = 3  # min trades for a ticker to appear in top_stocks


def build_universe():
    """Distinct US holdings; each ticker -> first sector that lists it."""
    tickers, t2s = [], {}
    for sector in config.SECTOR_ETFS:
        for t in (sector_holdings.get_holdings(sector) or []):
            if "." in t or t in t2s:
                continue
            t2s[t] = sector
            tickers.append(t)
    return tickers, t2s


def load_from_db(tickers, interval="1d"):
    """Load OHLCV from the PostgreSQL Candle table (no network). Requires Django set up.
    Returns {ticker: df} only for tickers that have candles — delisted/missing are skipped.
    Uses one bulk values_list query + a single pandas groupby (fast on ~1M rows)."""
    import pandas as pd
    from core.models import Candle

    qs = (Candle.objects.filter(ticker__in=list(tickers), interval=interval)
          .values_list("ticker", "date", "open", "high", "low", "close", "volume"))
    big = pd.DataFrame.from_records(
        list(qs), columns=["ticker", "date", "Open", "High", "Low", "Close", "Volume"])
    if big.empty:
        return {}
    big["date"] = pd.to_datetime(big["date"])

    data = {}
    for tk, g in big.groupby("ticker", sort=False):
        g = g.sort_values("date").set_index("date")[["Open", "High", "Low", "Close", "Volume"]]
        data[tk] = g
    return data


def _entries(sdf):
    """Entry integer-indices for a single stock df (empty if none / too short)."""
    if sdf is None or len(sdf) < MIN_BARS:
        return []
    try:
        sig = _SIG_FN(sdf).fillna(False)
    except Exception:
        return []
    return [sdf.index.get_loc(d) for d in sig[sig].index]


def run_one_exit(exit_key, stock_data, ticker_to_sector):
    exit_name, exit_fn = EXITS[exit_key]
    rets, holds = [], []
    sectors_hit = set()
    per_sector = {}   # sector -> list[ret]
    per_stock = {}    # ticker -> list[ret]

    for ticker, sdf in stock_data.items():
        close = sdf["Close"].values
        n = len(close)
        for idx in _entries(sdf):
            exit_idx = exit_fn(sdf, idx)
            if exit_idx is None or exit_idx <= idx or exit_idx >= n:
                continue
            entry_p = float(close[idx])
            if entry_p <= 0:
                continue
            ret = (float(close[exit_idx]) - entry_p) / entry_p * 100
            rets.append(ret)
            holds.append(exit_idx - idx)
            sec = ticker_to_sector.get(ticker, "?")
            sectors_hit.add(sec)
            per_sector.setdefault(sec, []).append(ret)
            per_stock.setdefault(ticker, []).append(ret)

    def _agg(name, lst):
        return {"name": name, "trades": len(lst),
                "win_rate": round(sum(1 for r in lst if r > 0) / len(lst) * 100, 1),
                "avg_return": round(sum(lst) / len(lst), 3)}

    sector_aggs = [_agg(s, l) for s, l in per_sector.items()]
    sector_aggs.sort(key=lambda a: a["avg_return"], reverse=True)
    stock_aggs = [_agg(t, l) for t, l in per_stock.items() if len(l) >= MIN_STOCK_TRADES]
    stock_aggs.sort(key=lambda a: a["avg_return"], reverse=True)

    total = len(rets)
    return {
        "exit_key": exit_key,
        "exit_name": exit_name,
        "total_trades": total,
        "avg_return": round(sum(rets) / total, 3) if total else 0.0,
        "win_rate": round(sum(1 for r in rets if r > 0) / total * 100, 1) if total else 0.0,
        "avg_hold": round(sum(holds) / len(holds), 1) if holds else 0.0,
        "sector_count": len(sectors_hit),
        "best_sectors": sector_aggs[:5],
        "worst_sectors": sector_aggs[-5:][::-1],
        "top_stocks": stock_aggs[:10],
    }


def run_all(loader=None, limit=None):
    """loader(tickers) -> {ticker: df}. Defaults to local/yfinance fetch; pass
    load_from_db to use the PostgreSQL candles we already have (no downloads)."""
    loader = loader or data_fetcher.fetch_tickers
    tickers, t2s = build_universe()
    if limit:
        tickers = tickers[:limit]
    print(f"Universe: {len(tickers)} tickers. Loading data via {loader.__name__}...")
    stock_data = loader(tickers)
    print(f"Loaded {len(stock_data)} tickers. Running {len(EXITS)} exits...")

    results = []
    for i, exit_key in enumerate(EXITS, 1):
        results.append(run_one_exit(exit_key, stock_data, t2s))
        if i % 10 == 0:
            print(f"  [{i}/{len(EXITS)}]")

    out = STUDIES_DIR / "stock_drawdown.json"
    out.write_text(json.dumps(results, indent=2))
    print(f"\nSaved {len(results)} exit-results to {out}")

    ranked = sorted(results, key=lambda r: r["avg_return"], reverse=True)
    print("\nTop 8 exits by avg return:")
    for r in ranked[:8]:
        print(f"  {r['exit_key']:12s} avg={r['avg_return']:+.3f}%  wr={r['win_rate']:.0f}%  "
              f"hold={r['avg_hold']:.0f}d  trades={r['total_trades']}")
    return results


if __name__ == "__main__":
    import sys
    if "--db" in sys.argv:
        import os, django
        os.environ.setdefault("DJANGO_SETTINGS_MODULE", "rotation.settings")
        django.setup()
        run_all(loader=load_from_db)
    else:
        run_all()
