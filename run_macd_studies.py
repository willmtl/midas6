#!/usr/bin/env python3
"""
Compile the NEW MACD-based studies in-memory (no DB required).

Runs the three MACD signals added to the studies engine against every exit
condition, ranks the combos, and writes results to .data/studies/macd_studies.json.

Uses a process pool for true parallelism (data is cached, so each worker loads
its own copy in well under a second).
"""
import warnings
warnings.filterwarnings("ignore")

import json
import time
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed

NEW_SIGNALS = ["macd_great", "rsi_x_macd_great", "rsi_omega_macd_great"]

# Per-process worker state
_WD = None


def _init_worker():
    """Each process loads + pre-computes its own copy of the (cached) data."""
    global _WD
    import ta
    import data_fetcher
    import studies
    d = data_fetcher.fetch_all()
    for ticker, df in d.items():
        if len(df) < 20:
            continue
        df["_sortino"] = studies._rolling_sortino(df)
        df["_omega"] = studies._rolling_omega(df)
        df["_rsi"] = ta.momentum.rsi(df["Close"], window=10)
        df["_rsi_sma"] = df["_rsi"].rolling(10).mean()
        df["_rsi_sort"] = studies._rsi_of_sortino(df)
        df["_rsi_sort_sma"] = df["_rsi_sort"].rolling(10).mean()
    _WD = d


def _run(study):
    import studies
    return studies.run_study(study, _WD)


def main():
    import studies  # in parent only for building the combo list
    t0 = time.time()

    combos = []
    for sk in NEW_SIGNALS:
        sig_name = studies.SIGNALS[sk][0]
        for ek in studies.EXITS:
            exit_name = studies.EXITS[ek][0]
            combos.append({
                "signal": sk, "signal_name": sig_name,
                "exit": ek, "exit_name": exit_name,
                "name": f"{sig_name} -> {exit_name}",
                "category": studies._categorize(sk),
            })
    print(f"Running {len(combos)} studies "
          f"({len(NEW_SIGNALS)} signals x {len(studies.EXITS)} exits)...", flush=True)

    import os
    workers = min(os.cpu_count() or 4, 12)
    results = []
    done = 0
    with ProcessPoolExecutor(max_workers=workers, initializer=_init_worker) as ex:
        futs = {ex.submit(_run, s): s for s in combos}
        for f in as_completed(futs):
            results.append(f.result())
            done += 1
            if done % 20 == 0 or done == len(combos):
                print(f"  [{done}/{len(combos)}] {time.time() - t0:.1f}s", flush=True)

    out_dir = Path(__file__).parent / ".data" / "studies"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "macd_studies.json"
    with open(out_path, "w") as fh:
        json.dump(results, fh, indent=2, default=str)
    print(f"\nDone in {time.time() - t0:.1f}s. Saved {len(results)} studies -> {out_path}\n", flush=True)

    MIN_TRADES = 30
    valid = [r for r in results if r.get("total_trades", 0) >= MIN_TRADES]
    valid.sort(key=lambda r: r.get("avg_return", 0), reverse=True)

    def line(r):
        return (f"{r['avg_return']:+7.3f}%  wr={r['win_rate']:5.1f}%  "
                f"hold={r['avg_hold']:5.1f}d  n={r['total_trades']:5d}  "
                f"peak={str(r.get('peak_avg')):>6}%@{str(r.get('peak_day')):>4}d  {r['name']}")

    print(f"=== TOP 15 MACD studies (>= {MIN_TRADES} trades), by avg return ===")
    for r in valid[:15]:
        print("  " + line(r))

    print(f"\n=== BOTTOM 5 (worst) ===")
    for r in valid[-5:]:
        print("  " + line(r))

    print("\n=== Best exit for each MACD signal ===")
    for sk in NEW_SIGNALS:
        sub = [r for r in valid if r["signal"] == sk]
        if sub:
            print(f"  {studies.SIGNALS[sk][0]:34s}: {line(sub[0])}")
        else:
            print(f"  {studies.SIGNALS[sk][0]:34s}: (no combo reached {MIN_TRADES} trades)")

    profitable = sum(1 for r in results if r.get("avg_return", 0) > 0)
    print(f"\nProfitable combos: {profitable}/{len(results)}  |  reached {MIN_TRADES}+ trades: {len(valid)}")


if __name__ == "__main__":
    main()
