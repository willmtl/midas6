#!/usr/bin/env python3
"""Backfill FULL-history SHORT INTEREST (Polygon /stocks/v1/short-interest, bi-monthly, ~2017-12+) for the
universe -> .data/short_interest.jsonl. Fields per row: ticker, settlement_date, short_interest (shares),
avg_daily_volume, days_to_cover. Idempotent overwrite. Genuinely-new dataset (multi-regime, unlike the
2022+ options data). PIT publication lag is applied at SCREEN time (FINRA disseminates ~8-10 bd after
settlement), NOT here. Run: MSYS_NO_PATHCONV=1 docker exec rotation-backend-1 python -u /app/si_fetch.py"""
import os, sys, json, time
sys.path.insert(0, "/app")
os.environ.setdefault("DJANGO_SETTINGS_MODULE", "rotation.settings")
import django; django.setup()
from pathlib import Path
from api.tasks import _polygon_paginate
from signal_discovery import _universe

OUT = Path("/app/.data/short_interest.jsonl")


def main():
    universe, _ = _universe()
    names = sorted(set(universe))
    print(f"fetching short interest for {len(names)} names...", flush=True)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    n_rows = 0; n_have = 0
    with OUT.open("w") as f:
        for i, tk in enumerate(names):
            rows = _polygon_paginate("/stocks/v1/short-interest", cap=2000, ticker=tk,
                                     limit=1000, order="asc", sort="settlement_date")
            if rows:
                n_have += 1
                for r in rows:
                    rec = {"ticker": tk, "settlement_date": r.get("settlement_date"),
                           "short_interest": r.get("short_interest"),
                           "avg_daily_volume": r.get("avg_daily_volume"),
                           "days_to_cover": r.get("days_to_cover")}
                    f.write(json.dumps(rec) + "\n"); n_rows += 1
            if (i + 1) % 100 == 0:
                print(f"  {i+1}/{len(names)}  names_with_data={n_have}  rows={n_rows}", flush=True)
            time.sleep(0.03)
    print(f"\nDONE: {n_rows} rows over {n_have} names -> {OUT}", flush=True)


if __name__ == "__main__":
    main()
