#!/usr/bin/env python3
"""
Trend Studies Engine - Sector Momentum Rotation Backtester

Tests: which sectors to buy based on trailing N-month returns,
hold for M months, then rebalance. Finds optimal lookback, hold period,
and number of top sectors to maximize returns over 5 years.

Excludes crypto sectors.
"""

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import time
from pathlib import Path

import config
import data_fetcher

CRYPTO_TICKERS = {"IBIT", "ETHA", "BTC-USD", "ETH-USD", "BLOK"}


def load_sector_data():
    """Load all sector price data, exclude crypto. Also loads SPY."""
    all_data = data_fetcher.fetch_all()
    # Make sure SPY is loaded
    if "SPY" not in all_data:
        spy = data_fetcher.fetch_tickers(["SPY"], config.DEFAULT_PERIOD, config.DEFAULT_INTERVAL)
        all_data.update(spy)
    sectors = {}
    for name, etf in config.SECTOR_ETFS.items():
        if etf in CRYPTO_TICKERS:
            continue
        df = all_data.get(etf)
        if df is not None and len(df) > 200:
            sectors[etf] = {"name": name, "df": df}
    # Add SPY for benchmark
    if "SPY" in all_data and len(all_data["SPY"]) > 200:
        sectors["SPY"] = {"name": "S&P 500", "df": all_data["SPY"]}
    return sectors


def compute_monthly_returns(sectors):
    """Build a DataFrame of monthly returns per sector."""
    # Resample all sectors to monthly close
    monthly = {}
    for etf, data in sectors.items():
        close = data["df"]["Close"].resample("ME").last()
        monthly[etf] = close

    prices = pd.DataFrame(monthly).dropna(how="all")
    returns = prices.pct_change()
    return prices, returns


def run_momentum_strategy(prices, returns, lookback_months, hold_months, top_n, rebalance_freq="monthly"):
    """
    Momentum rotation strategy:
    - Every rebalance period, rank sectors by trailing lookback_months return
    - Buy top_n sectors equally weighted
    - Hold for hold_months (or until next rebalance)
    - Track cumulative return
    """
    if len(prices) < lookback_months + hold_months + 1:
        return None

    # Compute trailing returns for ranking
    trailing_ret = prices.pct_change(lookback_months)

    equity = 1.0
    equity_curve = []
    trades = []
    spy_equity = 1.0
    spy_curve = []

    spy_col = "SPY" if "SPY" in prices.columns else None

    # Rebalance every hold_months
    rebalance_dates = list(trailing_ret.index[lookback_months::hold_months])

    for i, date in enumerate(rebalance_dates):
        # Rank sectors by trailing return
        ranks = trailing_ret.loc[date].dropna().sort_values(ascending=False)

        # Exclude SPY from ranking (it's the benchmark)
        ranks = ranks.drop("SPY", errors="ignore")

        if len(ranks) < top_n:
            continue

        # Pick top N
        top_sectors = ranks.head(top_n).index.tolist()

        # Determine hold period end
        if i + 1 < len(rebalance_dates):
            end_date = rebalance_dates[i + 1]
        else:
            end_date = prices.index[-1]

        # Get returns for hold period
        hold_prices = prices.loc[date:end_date]
        if len(hold_prices) < 2:
            continue

        # Equal weight portfolio return
        port_ret = 0
        for etf in top_sectors:
            if etf in hold_prices.columns:
                start_p = hold_prices[etf].iloc[0]
                end_p = hold_prices[etf].iloc[-1]
                if start_p > 0:
                    port_ret += (end_p / start_p - 1) / top_n

        equity *= (1 + port_ret)
        equity_curve.append({"date": str(date)[:10], "equity": round(equity, 4), "return": round(port_ret * 100, 2)})

        # SPY benchmark
        if spy_col and spy_col in hold_prices.columns:
            spy_start = hold_prices[spy_col].iloc[0]
            spy_end = hold_prices[spy_col].iloc[-1]
            if spy_start > 0:
                spy_ret = spy_end / spy_start - 1
                spy_equity *= (1 + spy_ret)
        spy_curve.append({"date": str(date)[:10], "spy_equity": round(spy_equity, 4)})

        trades.append({
            "date": str(date)[:10],
            "end_date": str(end_date)[:10],
            "sectors": top_sectors,
            "sector_names": [sectors_map.get(e, e) for e in top_sectors],
            "return_pct": round(port_ret * 100, 2),
        })

    if not equity_curve:
        return None

    # Compute stats
    total_ret = (equity - 1) * 100
    n_years = len(equity_curve) * hold_months / 12
    annual_ret = ((equity ** (1 / max(n_years, 0.1))) - 1) * 100 if n_years > 0 else 0

    spy_total = (spy_equity - 1) * 100
    alpha = total_ret - spy_total

    # Max drawdown
    peak = 1.0
    max_dd = 0
    for pt in equity_curve:
        if pt["equity"] > peak:
            peak = pt["equity"]
        dd = (pt["equity"] - peak) / peak * 100
        if dd < max_dd:
            max_dd = dd

    # Win rate
    wins = sum(1 for t in trades if t["return_pct"] > 0)
    wr = wins / len(trades) * 100 if trades else 0

    return {
        "lookback_months": lookback_months,
        "hold_months": hold_months,
        "top_n": top_n,
        "total_return": round(total_ret, 2),
        "annual_return": round(annual_ret, 2),
        "spy_total": round(spy_total, 2),
        "alpha": round(alpha, 2),
        "max_drawdown": round(max_dd, 2),
        "trades": len(trades),
        "win_rate": round(wr, 1),
        "equity_curve": equity_curve,
        "spy_curve": spy_curve,
        "trade_log": trades,
    }


# Module-level for use inside strategy
sectors_map = {}


def run_all_strategies():
    """Test all parameter combinations and find the best ones."""
    print("Loading sector data...")
    sectors = load_sector_data()
    print(f"Loaded {len(sectors)} sectors")

    global sectors_map
    sectors_map = {etf: data["name"] for etf, data in sectors.items()}

    print("Computing monthly returns...")
    prices, returns = compute_monthly_returns(sectors)
    print(f"Monthly data: {len(prices)} months, {len(prices.columns)} sectors")

    # Parameter grid
    lookbacks = [1, 2, 3, 4, 5, 6, 9, 12]
    holds = [1, 2, 3, 4, 6]
    top_ns = [1, 2, 3, 5, 7, 10, 15, 20]

    results = []
    total = len(lookbacks) * len(holds) * len(top_ns)
    print(f"Testing {total} parameter combinations...")

    t0 = time.time()
    for lb in lookbacks:
        for h in holds:
            for n in top_ns:
                r = run_momentum_strategy(prices, returns, lb, h, n)
                if r:
                    results.append(r)

    elapsed = time.time() - t0
    print(f"Computed {len(results)} strategies in {elapsed:.1f}s")

    results.sort(key=lambda x: x["total_return"], reverse=True)

    print(f"\nTop 10 strategies:")
    for r in results[:10]:
        print(f"  Look={r['lookback_months']}m Hold={r['hold_months']}m Top={r['top_n']}  "
              f"Ret={r['total_return']:+.1f}%  Ann={r['annual_return']:+.1f}%  "
              f"SPY={r['spy_total']:+.1f}%  Alpha={r['alpha']:+.1f}%  "
              f"DD={r['max_drawdown']:.1f}%  WR={r['win_rate']:.0f}%")

    return results


def save_to_db(results):
    """Save trend study results to Django DB."""
    from core.models import TrendStudy
    from django.utils import timezone

    now = timezone.now()
    saved = 0
    for r in results:
        TrendStudy.objects.update_or_create(
            lookback_months=r["lookback_months"],
            hold_months=r["hold_months"],
            top_n=r["top_n"],
            defaults={
                "total_return": r["total_return"],
                "annual_return": r["annual_return"],
                "spy_total": r["spy_total"],
                "alpha": r["alpha"],
                "max_drawdown": r["max_drawdown"],
                "num_trades": r["trades"],
                "win_rate": r["win_rate"],
                "equity_curve": r["equity_curve"],
                "spy_curve": r["spy_curve"],
                "trade_log": r["trade_log"],
                "computed_at": now,
            }
        )
        saved += 1
    print(f"Saved {saved} trend studies to DB")


if __name__ == "__main__":
    import os, sys, django
    os.environ.setdefault("DJANGO_SETTINGS_MODULE", "rotation.settings")
    for p in [Path(__file__).parent / "backend", Path("/app")]:
        if p.exists() and str(p) not in sys.path:
            sys.path.insert(0, str(p))
    django.setup()

    results = run_all_strategies()
    save_to_db(results)
