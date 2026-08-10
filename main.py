#!/usr/bin/env python3
"""
Stock Market Trend Bot

Scans sector ETFs for RSI(14) crossing above its 14-period SMA.

Usage:
    python main.py              # Scan all sectors
    python main.py --period 2y  # Use 2 years of data
"""

import warnings
warnings.filterwarnings("ignore")

import argparse
import time

import config
import trend_analyzer
import report
from report import console


def main():
    parser = argparse.ArgumentParser(
        description="Sector rotation: RSI(14) vs SMA(14) crossover",
    )
    parser.add_argument(
        "--period", default=config.DEFAULT_PERIOD,
        help="Data lookback: 6mo, 1y, 2y (default: %(default)s)",
    )
    parser.add_argument(
        "--interval", default=config.DEFAULT_INTERVAL,
        help="Data interval: 1d, 1wk (default: %(default)s)",
    )

    args = parser.parse_args()

    console.print(f"[bold bright_blue]Scanning sector ETFs ({args.period})...[/]")
    start = time.time()

    results = trend_analyzer.analyze(period=args.period, interval=args.interval)

    elapsed = time.time() - start
    console.print(f"[dim]Done in {elapsed:.1f}s[/]")

    report.print_results(results)


if __name__ == "__main__":
    main()
