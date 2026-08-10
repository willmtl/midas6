#!/usr/bin/env python3
"""
Stock Market Trend Bot - Holdings Viewer

Fetches and displays top holdings for all sector ETFs.
"""

import warnings
warnings.filterwarnings("ignore")

import os
import time
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed

import yfinance as yf
import pandas as pd

from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.text import Text
from rich import box

import config

try:
    _width = os.get_terminal_size().columns
except (ValueError, OSError):
    _width = 130
console = Console(width=max(_width, 130))


def fetch_holdings(etf_ticker: str) -> list[dict] | None:
    """Fetch top holdings for an ETF from yfinance."""
    try:
        etf = yf.Ticker(etf_ticker)
        fd = etf.funds_data
        h = fd.top_holdings
        if h is None or h.empty:
            return None

        holdings = []
        for symbol, row in h.iterrows():
            holdings.append({
                "symbol": str(symbol),
                "name": str(row.get("Name", "")),
                "weight": float(row.get("Holding Percent", 0)) * 100,
            })
        return holdings
    except Exception:
        return None


def fetch_all_holdings() -> dict[str, list[dict]]:
    """Fetch holdings for all sector ETFs in parallel."""
    results = {}

    def _fetch(sector_name, etf_ticker):
        holdings = fetch_holdings(etf_ticker)
        return sector_name, etf_ticker, holdings

    with ThreadPoolExecutor(max_workers=8) as pool:
        futures = {
            pool.submit(_fetch, name, ticker): name
            for name, ticker in config.SECTOR_ETFS.items()
        }

        for future in as_completed(futures):
            sector_name, etf_ticker, holdings = future.result()
            if holdings:
                results[sector_name] = {
                    "etf": etf_ticker,
                    "holdings": holdings,
                }

    return results


def print_holdings(all_holdings: dict, filter_sector: str | None = None):
    """Print holdings tables for all or filtered sectors."""
    console.print()
    console.print(Panel(
        Text("SECTOR ETF HOLDINGS", style="bold white", justify="center"),
        border_style="bright_blue",
        box=box.DOUBLE,
        padding=(0, 2),
    ))
    console.print()

    # Sort by config order
    sector_order = list(config.SECTOR_ETFS.keys())

    for sector_name in sector_order:
        if sector_name not in all_holdings:
            continue
        if filter_sector and filter_sector.lower() not in sector_name.lower():
            continue

        data = all_holdings[sector_name]
        etf = data["etf"]
        holdings = data["holdings"]

        table = Table(
            box=box.SIMPLE,
            show_header=True,
            header_style="bold",
            border_style="dim",
            title=f"[bold cyan]{sector_name}[/] ([dim]{etf}[/])",
            title_justify="left",
            padding=(0, 1),
        )
        table.add_column("#", style="dim", width=3, justify="right")
        table.add_column("Ticker", width=8, no_wrap=True)
        table.add_column("Name", min_width=35)
        table.add_column("Weight", justify="right", width=8)

        for i, h in enumerate(holdings, 1):
            w = h["weight"]
            w_style = "bold bright_green" if w > 8 else "green" if w > 4 else "white"
            table.add_row(
                str(i),
                Text(h["symbol"], style="bold"),
                h["name"],
                Text(f"{w:.1f}%", style=w_style),
            )

        top_weight = sum(h["weight"] for h in holdings)
        table.add_row("", "", Text("Top holdings total", style="dim"), Text(f"{top_weight:.1f}%", style="dim bold"))

        console.print(table)
        console.print()


def print_all_sectors_summary(all_holdings: dict):
    """Print a compact summary: one row per sector with top 5 tickers."""
    console.print()
    console.print(Panel(
        Text("ALL SECTORS — TOP HOLDINGS SUMMARY", style="bold white", justify="center"),
        border_style="bright_blue",
        box=box.DOUBLE,
        padding=(0, 2),
    ))
    console.print()

    table = Table(
        box=box.ROUNDED,
        show_header=True,
        header_style="bold cyan",
        border_style="dim",
    )
    table.add_column("#", style="dim", width=3, justify="right")
    table.add_column("Sector", min_width=22)
    table.add_column("ETF", width=5, justify="center")
    table.add_column("Top Holdings", min_width=60)
    table.add_column("Conc.", justify="right", width=7)

    sector_order = list(config.SECTOR_ETFS.keys())
    i = 0
    for sector_name in sector_order:
        if sector_name not in all_holdings:
            continue
        i += 1
        data = all_holdings[sector_name]
        holdings = data["holdings"]
        top5 = holdings[:5]
        top5_str = ", ".join(
            f"[bold]{h['symbol']}[/] ({h['weight']:.1f}%)" for h in top5
        )
        concentration = sum(h["weight"] for h in holdings)
        conc_style = "bright_red" if concentration > 60 else "yellow" if concentration > 40 else "green"

        table.add_row(
            str(i),
            sector_name,
            data["etf"],
            top5_str,
            Text(f"{concentration:.0f}%", style=conc_style),
        )

    console.print(table)
    console.print()


def main():
    import argparse
    parser = argparse.ArgumentParser(description="View ETF holdings for all sectors")
    parser.add_argument("sector", nargs="?", default=None, help="Filter by sector name (partial match)")
    parser.add_argument("--summary", action="store_true", help="Show compact summary only")
    args = parser.parse_args()

    console.print("[bold bright_blue]Fetching holdings for all sector ETFs...[/]")
    start = time.time()
    all_holdings = fetch_all_holdings()
    elapsed = time.time() - start
    console.print(f"[dim]Fetched {len(all_holdings)}/{len(config.SECTOR_ETFS)} ETFs in {elapsed:.1f}s[/]")

    if args.summary:
        print_all_sectors_summary(all_holdings)
    else:
        print_holdings(all_holdings, filter_sector=args.sector)

    if not args.sector and not args.summary:
        print_all_sectors_summary(all_holdings)


if __name__ == "__main__":
    main()
