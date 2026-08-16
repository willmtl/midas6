#!/usr/bin/env python3
"""FRED macro / liquidity fetcher -> core.MacroSeries. Uses the fredgraph CSV endpoint (NO API key,
egress to fred.stlouisfed.org confirmed working; yfinance/EODHD can't give these). Full history,
idempotent (upsert by series+date). Series pulled:
  M2SL        M2 money supply (monthly)          -> liquidity regime (YoY growth)
  WALCL       Fed total assets (weekly)          -> net-liquidity component
  RRPONTSYD   overnight reverse repo (daily, $B) -> net-liquidity DRAIN
  WTREGEN     Treasury General Account (wk, $B)  -> net-liquidity DRAIN
  DTWEXBGS    broad trade-weighted USD (daily)   -> dollar headwind
  BAMLH0A0HYM2 HY OAS credit spread (daily, %)   -> LEADING risk-off signal
  T10Y2Y      10y-2y Treasury curve (daily, %)   -> recession/curve signal
CLI: python -u fetch_fred.py [--probe] [--run]
Run: MSYS_NO_PATHCONV=1 docker exec rotation-backend-1 python -u /app/fetch_fred.py --run
"""
import os, argparse, urllib.request, io, csv
os.environ.setdefault("DJANGO_SETTINGS_MODULE", "rotation.settings")
import django  # noqa: E402
django.setup()

from core.models import MacroSeries  # noqa: E402

SERIES = ["M2SL", "WALCL", "RRPONTSYD", "WTREGEN", "DTWEXBGS", "BAMLH0A0HYM2", "T10Y2Y"]


def fetch(series_id):
    url = f"https://fred.stlouisfed.org/graph/fredgraph.csv?id={series_id}&cosd=1990-01-01"
    raw = urllib.request.urlopen(url, timeout=40).read().decode()
    rows = list(csv.reader(io.StringIO(raw)))
    if not rows or len(rows) < 2:
        return []
    out = []
    for r in rows[1:]:
        if len(r) < 2:
            continue
        d, v = r[0].strip(), r[1].strip()
        if not d or v in ("", "."):
            continue
        try:
            out.append((d, float(v)))
        except ValueError:
            continue
    return out


def backfill(save=True):
    total = 0
    for sid in SERIES:
        obs = fetch(sid)
        if not obs:
            print(f"  {sid}: no data", flush=True)
            continue
        if save:
            MacroSeries.objects.filter(series=sid).delete()
            MacroSeries.objects.bulk_create(
                [MacroSeries(series=sid, date=d, value=v) for d, v in obs], batch_size=5000)
        total += len(obs)
        print(f"  {sid}: {len(obs)} obs  {obs[0][0]} -> {obs[-1][0]}  (last {obs[-1][1]})", flush=True)
    print(f"FRED backfill: {total} observations across {len(SERIES)} series", flush=True)
    return total


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--probe", action="store_true")
    ap.add_argument("--run", action="store_true")
    a = ap.parse_args()
    if not (a.run or a.probe):
        a.probe = True
    backfill(save=a.run)


if __name__ == "__main__":
    main()
