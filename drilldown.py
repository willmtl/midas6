#!/usr/bin/env python3
"""
Stock Market Trend Bot - Drill-Down Scanner

Two-level strategy using RSI(14) vs SMA(14):
  1. Find bullish sectors (RSI > its 14-SMA)
  2. For each bullish sector, apply the same filter to its top 20 holdings

Final output: individual stocks where RSI > RSI SMA, within bullish sectors.
"""

import warnings
warnings.filterwarnings("ignore")

import argparse
import time
import os

import numpy as np
import pandas as pd

from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.text import Text
from rich import box

import config
import data_fetcher
import indicators
import trend_analyzer
import sector_holdings

try:
    _width = os.get_terminal_size().columns
except (ValueError, OSError):
    _width = 130
console = Console(width=max(_width, 130))


def scan_stocks_in_sector(
    sector_name: str,
    period: str = config.DEFAULT_PERIOD,
) -> list[dict]:
    """
    Apply RSI > RSI SMA filter to top 20 holdings in a sector.
    Returns list of stock results sorted by RSI spread.
    """
    holdings = sector_holdings.get_holdings(sector_name)
    if not holdings:
        return []

    # Filter to US-listed tickers (skip foreign exchange tickers like .T, .HK)
    us_tickers = [t for t in holdings if "." not in t]
    if not us_tickers:
        us_tickers = holdings[:20]

    stock_data = data_fetcher.fetch_tickers(us_tickers[:20], period)

    results = []
    for ticker in us_tickers[:20]:
        df = stock_data.get(ticker)
        if df is None or df.empty or len(df) < 30:
            continue

        try:
            rsi_data = indicators.compute_rsi_crossover(df)
            if rsi_data.get("rsi") is None:
                continue

            rsi = rsi_data["rsi"]
            rsi_sma = rsi_data["rsi_sma"] or 0
            rsi_spread = round(rsi - rsi_sma, 2)
            rsi_above = rsi_data.get("rsi_above_sma", False)
            crossover = rsi_data.get("rsi_crossover", False)

            close = float(df["Close"].iloc[-1])
            ret_1w = float((df["Close"].iloc[-1] / df["Close"].iloc[-6] - 1) * 100) if len(df) > 6 else 0
            ret_1m = float((df["Close"].iloc[-1] / df["Close"].iloc[-22] - 1) * 100) if len(df) > 22 else 0

            if rsi_above:
                signal = "ROTATE IN" if crossover else "BULLISH"
            else:
                signal = "BEARISH"

            results.append({
                "ticker": ticker,
                "price": round(close, 2),
                "return_1w": round(ret_1w, 1),
                "return_1m": round(ret_1m, 1),
                "rsi": rsi,
                "rsi_sma": rsi_sma,
                "rsi_spread": rsi_spread,
                "rsi_above_sma": rsi_above,
                "rsi_crossover": crossover,
                "signal": signal,
            })
        except Exception:
            continue

    results.sort(key=lambda x: x["rsi_spread"], reverse=True)
    return results


def run_drilldown(period: str = config.DEFAULT_PERIOD) -> dict:
    """
    Full two-level scan:
      1. Find bullish sectors (RSI > RSI SMA)
      2. Drill into each bullish sector's holdings
    """
    # Level 1
    console.print("[bold bright_blue]Level 1: Scanning sectors...[/]")
    sector_results = trend_analyzer.analyze(period=period)

    winners = [s for s in sector_results if s["bullish"]]

    console.print(
        f"  Found [bold bright_green]{len(winners)}[/] bullish sectors "
        f"out of {len(sector_results)}"
    )

    # Level 2
    console.print("[bold bright_blue]Level 2: Scanning stocks in bullish sectors...[/]")

    stock_results = {}
    all_picks = []

    for sector in winners:
        name = sector["sector"]
        holdings = sector_holdings.get_holdings(name)
        if not holdings:
            continue

        console.print(f"  Scanning [cyan]{name}[/] ({len(holdings)} stocks)...")
        stocks = scan_stocks_in_sector(name, period)
        stock_results[name] = stocks

        for s in stocks:
            if s["rsi_above_sma"]:
                s["sector"] = name
                all_picks.append(s)

    all_picks.sort(key=lambda x: x["rsi_spread"], reverse=True)

    # Deduplicate
    seen = set()
    unique_picks = []
    for p in all_picks:
        if p["ticker"] not in seen:
            seen.add(p["ticker"])
            unique_picks.append(p)

    return {
        "sector_results": sector_results,
        "winners": winners,
        "stock_results": stock_results,
        "picks": unique_picks,
    }


# ── Report ──

def print_drilldown(result: dict):
    winners = result["winners"]
    stock_results = result["stock_results"]
    picks = result["picks"]

    # Level 1 summary
    console.print()
    console.print(Panel(
        Text(
            "LEVEL 1: BULLISH SECTORS  (RSI(14) > SMA(14))",
            style="bold white",
            justify="center",
        ),
        border_style="bright_green",
        box=box.DOUBLE,
        padding=(0, 2),
    ))
    console.print()

    if not winners:
        console.print("[yellow]No sectors currently pass the filter.[/]")
        return

    table = Table(box=box.ROUNDED, header_style="bold cyan", border_style="dim")
    table.add_column("#", style="dim", width=3, justify="right")
    table.add_column("Sector", min_width=22)
    table.add_column("ETF", width=5, justify="center")
    table.add_column("RSI", justify="right", width=7)
    table.add_column("SMA", justify="right", width=7)
    table.add_column("Spread", justify="right", width=8)
    table.add_column("Cross", justify="center", width=7)
    table.add_column("Stocks Pass", justify="center", width=12)

    for i, s in enumerate(winners, 1):
        name = s["sector"]
        stocks = stock_results.get(name, [])
        passing = sum(1 for st in stocks if st["rsi_above_sma"])
        total = len(stocks)

        cross = Text("BUY", style="bold bright_green") if s["rsi_crossover"] else Text("-", style="dim")
        spread = s["rsi_spread"]
        sp_style = "bright_green" if spread > 5 else "green"

        table.add_row(
            str(i),
            name,
            s["etf"],
            Text(f"{s['rsi']:.1f}", style="bright_green"),
            Text(f"{s['rsi_sma']:.1f}", style="dim"),
            Text(f"{spread:+.1f}", style=sp_style),
            cross,
            Text(f"{passing}/{total}", style="bright_green" if passing > 0 else "dim"),
        )

    console.print(table)
    console.print()

    # Per-sector stock breakdown
    for sector in winners:
        name = sector["sector"]
        stocks = stock_results.get(name, [])
        if not stocks:
            continue

        console.print(Panel(
            Text(
                f"LEVEL 2: {name.upper()} ({sector['etf']}) — Stock Drill-Down",
                style="bold white",
                justify="center",
            ),
            border_style="cyan",
            box=box.HEAVY,
            padding=(0, 2),
        ))
        console.print()

        table = Table(box=box.SIMPLE, header_style="bold", border_style="dim", padding=(0, 1))
        table.add_column("Ticker", width=7, no_wrap=True)
        table.add_column("Price", justify="right", width=9, no_wrap=True)
        table.add_column("1W", justify="right", width=7, no_wrap=True)
        table.add_column("1M", justify="right", width=7, no_wrap=True)
        table.add_column("RSI", justify="right", width=7)
        table.add_column("SMA", justify="right", width=7)
        table.add_column("Spread", justify="right", width=8)
        table.add_column("Cross", justify="center", width=7)
        table.add_column("Signal", justify="center", width=14, no_wrap=True)

        for st in stocks:
            rsi = st["rsi"]
            rsi_sma = st["rsi_sma"]
            spread = st["rsi_spread"]
            rsi_style = "bright_green" if rsi > rsi_sma else "red"
            spread_style = "bright_green" if spread > 5 else "green" if spread > 0 else "red"
            cross = Text("BUY", style="bold bright_green") if st["rsi_crossover"] else Text("-", style="dim")

            sig = st["signal"]
            sig_style = "bold bright_green on dark_green" if sig == "ROTATE IN" else "bold bright_green" if sig == "BULLISH" else "red"

            ret_1w = st["return_1w"]
            ret_1m = st["return_1m"]

            table.add_row(
                Text(st["ticker"], style="bold"),
                f"${st['price']:,.2f}",
                Text(f"{ret_1w:+.1f}%", style="green" if ret_1w > 0 else "red"),
                Text(f"{ret_1m:+.1f}%", style="green" if ret_1m > 0 else "red"),
                Text(f"{rsi:.1f}", style=rsi_style),
                Text(f"{rsi_sma:.1f}", style="dim"),
                Text(f"{spread:+.1f}", style=spread_style),
                cross,
                Text(sig, style=sig_style),
            )

        console.print(table)
        console.print()

    # Final picks
    console.print(Panel(
        Text("FINAL PICKS — Stocks Passing Both Levels", style="bold white", justify="center"),
        border_style="bright_green",
        box=box.DOUBLE,
        padding=(0, 2),
    ))
    console.print()

    if not picks:
        console.print("[yellow]No stocks pass both levels.[/]")
        console.print()
        return

    table = Table(box=box.ROUNDED, header_style="bold cyan", border_style="dim")
    table.add_column("#", style="dim", width=3, justify="right")
    table.add_column("Ticker", width=7, no_wrap=True)
    table.add_column("Sector", min_width=20)
    table.add_column("Price", justify="right", width=9, no_wrap=True)
    table.add_column("1W", justify="right", width=7, no_wrap=True)
    table.add_column("1M", justify="right", width=7, no_wrap=True)
    table.add_column("RSI", justify="right", width=7)
    table.add_column("SMA", justify="right", width=7)
    table.add_column("Spread", justify="right", width=8)
    table.add_column("Cross", justify="center", width=7)

    for i, st in enumerate(picks, 1):
        rsi = st["rsi"]
        rsi_sma = st["rsi_sma"]
        spread = st["rsi_spread"]
        rsi_style = "bright_green"
        spread_style = "bright_green" if spread > 5 else "green"
        cross = Text("BUY", style="bold bright_green") if st["rsi_crossover"] else Text("-", style="dim")

        table.add_row(
            str(i),
            Text(st["ticker"], style="bold"),
            st.get("sector", ""),
            f"${st['price']:,.2f}",
            Text(f"{st['return_1w']:+.1f}%", style="green" if st["return_1w"] > 0 else "red"),
            Text(f"{st['return_1m']:+.1f}%", style="green" if st["return_1m"] > 0 else "red"),
            Text(f"{rsi:.1f}", style=rsi_style),
            Text(f"{rsi_sma:.1f}", style="dim"),
            Text(f"{spread:+.1f}", style=spread_style),
            cross,
        )

    console.print(table)

    cross_picks = [p["ticker"] for p in picks if p["rsi_crossover"]]
    console.print()
    console.print(f"  [bold bright_green]{len(picks)} unique stocks[/] pass both levels")
    if cross_picks:
        console.print(f"  [bold bright_green on dark_green] Fresh BUY crossovers: {', '.join(dict.fromkeys(cross_picks))} [/]")
    console.print()


def main():
    parser = argparse.ArgumentParser(
        description="Two-level drill-down: bullish sectors -> bullish stocks (RSI > SMA)",
    )
    parser.add_argument("--period", default=config.DEFAULT_PERIOD, help="Data period (default: %(default)s)")
    args = parser.parse_args()

    start = time.time()
    result = run_drilldown(period=args.period)
    elapsed = time.time() - start
    console.print(f"[dim]Total scan time: {elapsed:.1f}s[/]")

    print_drilldown(result)


if __name__ == "__main__":
    main()
