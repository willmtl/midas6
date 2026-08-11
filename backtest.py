#!/usr/bin/env python3
"""
Stock Market Trend Bot - Backtester

Simulates the sector rotation strategy historically:
  - Each day, compute 14-day rolling Sortino for all sector ETFs and SPY
  - Compute 14-day RSI and its 14-period SMA for each sector ETF
  - Go long (equal-weight) sectors where BOTH:
      1. Sector Sortino > SPY Sortino
      2. RSI > 14-SMA of RSI
  - If no sectors qualify, hold cash (0% return that day)
  - Rebalance daily

Compares strategy vs SPY buy-and-hold.
"""

import warnings
warnings.filterwarnings("ignore")

import argparse
import time

import numpy as np
import pandas as pd
import ta

import config
import data_fetcher

from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.text import Text
from rich import box

import os
try:
    _width = os.get_terminal_size().columns
except (ValueError, OSError):
    _width = 130
console = Console(width=max(_width, 130))


# ── Indicator helpers (same logic as indicators.py but vectorized for full history) ──

def calc_daily_returns(close: pd.Series) -> pd.Series:
    return np.log(close / close.shift(1))


def calc_rolling_sortino(returns: pd.Series, window: int) -> pd.Series:
    daily_rf = config.RISK_FREE_RATE / config.TRADING_DAYS

    def _sortino(r):
        excess = r - daily_rf
        mean_excess = excess.mean()
        downside = np.minimum(excess, 0)
        dd = np.sqrt(np.mean(downside ** 2))
        if dd < 1e-10:
            return np.nan
        return mean_excess / dd

    return returns.rolling(window=window).apply(_sortino, raw=False)


def calc_rsi(close: pd.Series, period: int = 14) -> pd.Series:
    return ta.momentum.rsi(close, window=period)


def calc_rsi_sma(rsi: pd.Series, period: int = 14) -> pd.Series:
    return rsi.rolling(window=period).mean()


# ── Backtester ──

def run_backtest(
    period: str = "2y",
    window: int = config.SORTINO_WINDOW,
) -> dict:
    """
    Run the full backtest.
    Returns a dict with equity curves, trades, and metrics.
    """
    console.print(f"[bold bright_blue]Fetching data ({period})...[/]")
    all_data = data_fetcher.fetch_all(period=period, interval="1d")

    spy_df = all_data.get(config.BENCHMARK)
    if spy_df is None:
        raise RuntimeError("Could not fetch SPY data")

    spy_close = spy_df["Close"].copy()
    spy_returns = calc_daily_returns(spy_close)

    # Build signal matrix for each sector ETF
    etf_to_sector = {v: k for k, v in config.SECTOR_ETFS.items()}
    sector_returns = {}
    sector_signals = {}

    for etf_ticker, sector_name in etf_to_sector.items():
        df = all_data.get(etf_ticker)
        if df is None:
            continue

        close = df["Close"].copy()
        ret = calc_daily_returns(close)
        rsi = calc_rsi(close, 14)
        rsi_sma = calc_rsi_sma(rsi, 14)

        # Align to SPY index
        ret = ret.reindex(spy_returns.index)
        rsi = rsi.reindex(spy_returns.index)
        rsi_sma = rsi_sma.reindex(spy_returns.index)

        # Signal: RSI > RSI SMA
        signal = rsi > rsi_sma

        sector_returns[sector_name] = ret
        sector_signals[sector_name] = signal

    returns_df = pd.DataFrame(sector_returns)
    signals_df = pd.DataFrame(sector_signals)

    # Drop rows where we don't have enough data for indicators
    valid_start = signals_df.dropna(how="all").index[0]
    returns_df = returns_df.loc[valid_start:]
    signals_df = signals_df.loc[valid_start:].fillna(False)
    spy_ret_aligned = spy_returns.reindex(returns_df.index).fillna(0)

    # ── Simulate daily strategy ──
    strategy_daily = []
    holdings_log = []
    num_sectors_held = []

    for date in returns_df.index:
        # Use previous day's signal to trade today (avoid look-ahead bias)
        prev_idx = returns_df.index.get_loc(date)
        if prev_idx == 0:
            strategy_daily.append(0.0)
            num_sectors_held.append(0)
            continue

        prev_date = returns_df.index[prev_idx - 1]
        active = signals_df.loc[prev_date]
        selected = active[active == True].index.tolist()

        n = len(selected)
        num_sectors_held.append(n)

        if n == 0:
            # No sectors qualify → hold cash
            strategy_daily.append(0.0)
        else:
            # Equal-weight across selected sectors
            day_returns = returns_df.loc[date, selected]
            avg_return = day_returns.mean()
            strategy_daily.append(avg_return if not np.isnan(avg_return) else 0.0)
            holdings_log.append((date, selected))

    strategy_series = pd.Series(strategy_daily, index=returns_df.index).fillna(0)

    # ── Equity curves ──
    strategy_equity = (1 + strategy_series).cumprod()
    spy_equity = (1 + spy_ret_aligned).cumprod()

    # ── Metrics ──
    days = len(strategy_series)
    years = days / config.TRADING_DAYS

    strat_total = strategy_equity.iloc[-1] - 1
    spy_total = spy_equity.iloc[-1] - 1

    strat_cagr = (strategy_equity.iloc[-1] ** (1 / years) - 1) if years > 0 else 0
    spy_cagr = (spy_equity.iloc[-1] ** (1 / years) - 1) if years > 0 else 0

    strat_dd = _max_drawdown(strategy_equity)
    spy_dd = _max_drawdown(spy_equity)

    strat_sharpe = _sharpe(strategy_series)
    spy_sharpe = _sharpe(spy_ret_aligned)

    strat_sortino_val = _sortino_ratio(strategy_series)
    spy_sortino_val = _sortino_ratio(spy_ret_aligned)

    strat_vol = strategy_series.std() * np.sqrt(config.TRADING_DAYS)
    spy_vol = spy_ret_aligned.std() * np.sqrt(config.TRADING_DAYS)

    # Win rate (days with positive return)
    trading_days_active = strategy_series[strategy_series != 0]
    win_rate = (trading_days_active > 0).sum() / len(trading_days_active) * 100 if len(trading_days_active) > 0 else 0

    avg_sectors = np.mean(num_sectors_held)
    cash_pct = (np.array(num_sectors_held) == 0).sum() / len(num_sectors_held) * 100

    # Monthly returns
    monthly_strat = strategy_series.resample("ME").sum()
    monthly_spy = spy_ret_aligned.resample("ME").sum()

    # Count how often we held each sector
    sector_counts = {}
    for _, sectors in holdings_log:
        for s in sectors:
            sector_counts[s] = sector_counts.get(s, 0) + 1
    total_holding_days = sum(sector_counts.values())

    return {
        "strategy_equity": strategy_equity,
        "spy_equity": spy_equity,
        "strategy_returns": strategy_series,
        "spy_returns": spy_ret_aligned,
        "monthly_strat": monthly_strat,
        "monthly_spy": monthly_spy,
        "metrics": {
            "period": period,
            "days": days,
            "years": round(years, 2),
            "strat_total": strat_total,
            "spy_total": spy_total,
            "strat_cagr": strat_cagr,
            "spy_cagr": spy_cagr,
            "strat_dd": strat_dd,
            "spy_dd": spy_dd,
            "strat_sharpe": strat_sharpe,
            "spy_sharpe": spy_sharpe,
            "strat_sortino": strat_sortino_val,
            "spy_sortino": spy_sortino_val,
            "strat_vol": strat_vol,
            "spy_vol": spy_vol,
            "win_rate": win_rate,
            "avg_sectors": avg_sectors,
            "cash_pct": cash_pct,
        },
        "sector_counts": sector_counts,
        "total_holding_days": total_holding_days,
    }


def _max_drawdown(equity: pd.Series) -> float:
    peak = equity.cummax()
    dd = (equity - peak) / peak
    return float(dd.min())


def _sharpe(returns: pd.Series) -> float:
    daily_rf = config.RISK_FREE_RATE / config.TRADING_DAYS
    excess = returns - daily_rf
    if excess.std() == 0:
        return 0.0
    return float(excess.mean() / excess.std() * np.sqrt(config.TRADING_DAYS))


def _sortino_ratio(returns: pd.Series) -> float:
    daily_rf = config.RISK_FREE_RATE / config.TRADING_DAYS
    excess = returns - daily_rf
    if len(excess) < 2:
        return 0.0
    # Target downside deviation = RMS of the shortfalls below MAR=0 over the
    # FULL window, not the sample std (ddof=1) of only the negative days centered
    # on their own mean (which understates the denominator → inflates Sortino,
    # and returns 0 when all losses are equal). Matches indicators.rolling_sortino
    # and calc_rolling_sortino above.
    downside_dev = float(np.sqrt(np.mean(np.minimum(excess, 0) ** 2)))
    if downside_dev == 0:
        return 0.0
    return float(excess.mean() / downside_dev * np.sqrt(config.TRADING_DAYS))


# ── Report ──

def print_results(result: dict):
    m = result["metrics"]

    console.print()
    console.print(Panel(
        Text(
            f"BACKTEST RESULTS  —  Sector Rotation Strategy vs SPY  ({m['period']} / {m['days']} trading days)",
            style="bold white",
            justify="center",
        ),
        border_style="bright_blue",
        box=box.DOUBLE,
        padding=(0, 2),
    ))
    console.print()

    # ── Performance table ──
    table = Table(
        box=box.ROUNDED,
        show_header=True,
        header_style="bold cyan",
        border_style="dim",
        title="Performance Metrics",
    )
    table.add_column("Metric", min_width=22)
    table.add_column("Strategy", justify="right", width=14)
    table.add_column("SPY (B&H)", justify="right", width=14)
    table.add_column("Diff", justify="right", width=14)

    def _add_pct_row(label, strat_val, spy_val):
        diff = strat_val - spy_val
        s_style = "bright_green" if strat_val > 0 else "red"
        b_style = "bright_green" if spy_val > 0 else "red"
        d_style = "bright_green" if diff > 0 else "red"
        table.add_row(
            label,
            Text(f"{strat_val * 100:+.2f}%", style=s_style),
            Text(f"{spy_val * 100:+.2f}%", style=b_style),
            Text(f"{diff * 100:+.2f}%", style=d_style),
        )

    def _add_num_row(label, strat_val, spy_val, higher_better=True):
        diff = strat_val - spy_val
        better = diff > 0 if higher_better else diff < 0
        d_style = "bright_green" if better else "red"
        table.add_row(
            label,
            f"{strat_val:.3f}",
            f"{spy_val:.3f}",
            Text(f"{diff:+.3f}", style=d_style),
        )

    _add_pct_row("Total Return", m["strat_total"], m["spy_total"])
    _add_pct_row("CAGR", m["strat_cagr"], m["spy_cagr"])
    _add_pct_row("Max Drawdown", m["strat_dd"], m["spy_dd"])
    _add_pct_row("Annualized Volatility", m["strat_vol"], m["spy_vol"])
    _add_num_row("Sharpe Ratio", m["strat_sharpe"], m["spy_sharpe"])
    _add_num_row("Sortino Ratio", m["strat_sortino"], m["spy_sortino"])

    console.print(table)
    console.print()

    # ── Strategy stats ──
    console.print(f"  Win rate (active days):  [bold]{m['win_rate']:.1f}%[/]")
    console.print(f"  Avg sectors held/day:    [bold]{m['avg_sectors']:.1f}[/]")
    console.print(f"  Days in cash:            [bold]{m['cash_pct']:.1f}%[/]")
    console.print()

    # ── Sector allocation frequency ──
    sc = result["sector_counts"]
    total = result["total_holding_days"]
    if sc:
        alloc_table = Table(
            box=box.SIMPLE,
            show_header=True,
            header_style="bold",
            border_style="dim",
            title="Sector Allocation Frequency",
        )
        alloc_table.add_column("Sector", min_width=24)
        alloc_table.add_column("Days Held", justify="right", width=10)
        alloc_table.add_column("% of Time", justify="right", width=10)

        sorted_sectors = sorted(sc.items(), key=lambda x: x[1], reverse=True)
        for sector, count in sorted_sectors:
            pct = count / result["metrics"]["days"] * 100
            alloc_table.add_row(sector, str(count), f"{pct:.1f}%")

        console.print(alloc_table)
        console.print()

    # ── Monthly returns ──
    monthly_strat = result["monthly_strat"]
    monthly_spy = result["monthly_spy"]

    month_table = Table(
        box=box.SIMPLE,
        show_header=True,
        header_style="bold",
        border_style="dim",
        title="Monthly Returns",
    )
    month_table.add_column("Month", width=10)
    month_table.add_column("Strategy", justify="right", width=10)
    month_table.add_column("SPY", justify="right", width=10)
    month_table.add_column("Excess", justify="right", width=10)

    for date in monthly_strat.index:
        s = monthly_strat.loc[date]
        b = monthly_spy.get(date, 0)
        diff = s - b
        s_style = "green" if s > 0 else "red"
        b_style = "green" if b > 0 else "red"
        d_style = "bright_green" if diff > 0 else "red"
        month_table.add_row(
            date.strftime("%Y-%m"),
            Text(f"{s * 100:+.1f}%", style=s_style),
            Text(f"{b * 100:+.1f}%", style=b_style),
            Text(f"{diff * 100:+.1f}%", style=d_style),
        )

    console.print(month_table)

    # ── Equity curve (ASCII sparkline) ──
    _print_equity_chart(result["strategy_equity"], result["spy_equity"])


def _print_equity_chart(strat: pd.Series, spy: pd.Series):
    """Simple ASCII equity curve comparison."""
    console.print()
    console.print(Panel(
        Text("EQUITY CURVE", style="bold white", justify="center"),
        border_style="bright_blue",
        box=box.HEAVY,
        padding=(0, 2),
    ))

    # Resample to weekly for cleaner chart
    strat_w = strat.resample("W").last().dropna()
    spy_w = spy.resample("W").last().dropna()

    # Align
    idx = strat_w.index.intersection(spy_w.index)
    strat_w = strat_w.loc[idx]
    spy_w = spy_w.loc[idx]

    if len(idx) < 4:
        console.print("[dim]Not enough data for chart[/]")
        return

    all_vals = pd.concat([strat_w, spy_w])
    v_min = all_vals.min()
    v_max = all_vals.max()

    chart_height = 20
    chart_width = min(len(idx), 80)

    # Downsample if too many points
    if len(idx) > chart_width:
        step = len(idx) // chart_width
        strat_w = strat_w.iloc[::step]
        spy_w = spy_w.iloc[::step]
        idx = strat_w.index

    def _scale(v):
        if v_max == v_min:
            return chart_height // 2
        return int((v - v_min) / (v_max - v_min) * (chart_height - 1))

    # Build chart grid
    grid = [[" " for _ in range(len(idx))] for _ in range(chart_height)]

    for col, date in enumerate(idx):
        s_row = chart_height - 1 - _scale(strat_w.loc[date])
        b_row = chart_height - 1 - _scale(spy_w.loc[date])

        if s_row == b_row:
            grid[s_row][col] = "X"
        else:
            grid[s_row][col] = "S"
            grid[b_row][col] = "B"

    # Print
    console.print()
    for row in grid:
        console.print("  " + "".join(row))

    console.print()
    console.print(f"  [bright_green]S[/] = Strategy ({strat_w.iloc[-1]:.3f})    [cyan]B[/] = SPY ({spy_w.iloc[-1]:.3f})")
    console.print(f"  {idx[0].strftime('%Y-%m')} {'-' * max(0, len(idx) - 14)} {idx[-1].strftime('%Y-%m')}")
    console.print()


def main():
    parser = argparse.ArgumentParser(description="Backtest sector rotation strategy")
    parser.add_argument(
        "--period", default="2y",
        help="Backtest period: 1y, 2y, 5y, max (default: 2y)",
    )
    parser.add_argument(
        "--window", type=int, default=config.SORTINO_WINDOW,
        help="Rolling Sortino window (default: %(default)s)",
    )
    args = parser.parse_args()

    start = time.time()
    result = run_backtest(period=args.period, window=args.window)
    elapsed = time.time() - start

    console.print(f"[dim]Backtest completed in {elapsed:.1f}s[/]")
    print_results(result)


if __name__ == "__main__":
    main()
