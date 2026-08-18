# -*- coding: utf-8 -*-
"""BACKFILL HISTORICAL SHORT INTEREST (Polygon /stocks/v1/short-interest = FINRA bi-monthly, back to ~2017).
Unlike Fundamental.short_pct_float (current snapshot only), this is DATED history -> point-in-time testable
at each purchase. Per settlement date: short_interest (shares), avg_daily_volume, days_to_cover.

One-time archive dump -> .data/short_interest.jsonl (one JSON line per (ticker, settlement_date)). Idempotent
per ticker (a ticker already present is skipped; --refresh re-pulls all).
Run detached in the celery worker (Polygon egress):
  MSYS_NO_PATHCONV=1 docker exec rotation-celery-worker-1 sh -c \
    'setsid nohup python -u /app/backfill_short_interest.py --workers 10 \
     > /app/.data/short_interest_backfill.log 2>&1 < /dev/null &'
"""
import os, sys, json, time
os.environ.setdefault("DJANGO_SETTINGS_MODULE", "rotation.settings")
import django; django.setup()
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
from api.tasks import _polygon_paginate

OUT = Path("/app/.data/short_interest.jsonl")
KEEP = ("settlement_date", "short_interest", "avg_daily_volume", "days_to_cover")


def analyst_universe():
    from core.models import Fundamental, Sector
    etfs = set(Sector.objects.values_list("etf", flat=True)) | {"SPY", "QQQ"}
    return sorted(set(Fundamental.objects.values_list("ticker", flat=True)) - etfs)


def done_tickers():
    done = set()
    if OUT.exists():
        for line in OUT.read_text(encoding="utf-8").splitlines():
            try:
                done.add(json.loads(line)["ticker"])
            except Exception:
                pass
    return done


def main(workers=8, refresh=False):
    tickers = sorted(set(analyst_universe()))
    done = set() if refresh else done_tickers()
    todo = [t for t in tickers if t not in done]
    print(f"universe {len(tickers)} | already dumped: {len(done)} | to fetch: {len(todo)}", flush=True)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fh = OUT.open("w" if refresh else "a", encoding="utf-8")
    n_tk = n_rows = n_cov = 0
    t0 = time.time()

    def work(tk):
        sym = tk.split(".")[0].upper()
        res = _polygon_paginate("/stocks/v1/short-interest", cap=4000, ticker=sym, limit=1000)
        rows = []
        for r in res or []:
            if not isinstance(r, dict) or not r.get("settlement_date"):
                continue
            rec = {"ticker": tk}
            for k in KEEP:
                if r.get(k) is not None:
                    rec[k] = r[k]
            rows.append(rec)
        return tk, rows

    with ThreadPoolExecutor(max_workers=workers) as ex:
        for tk, rows in ex.map(work, todo):
            n_tk += 1
            if rows:
                n_cov += 1
                for rec in rows:
                    fh.write(json.dumps(rec) + "\n")
                n_rows += len(rows)
            fh.flush()
            if n_tk % 100 == 0:
                print(f"  {n_tk}/{len(todo)}  {n_tk/(time.time()-t0):.1f}tk/s  covered={n_cov} rows={n_rows}", flush=True)
    fh.close()
    print(f"DONE {n_tk} tickers in {(time.time()-t0)/60:.1f}m | with-SI: {n_cov} | rows: {n_rows}", flush=True)
    print("DONE_SIBACKFILL", flush=True)


if __name__ == "__main__":
    a = sys.argv
    workers = int(a[a.index("--workers") + 1]) if "--workers" in a else 8
    main(workers, "--refresh" in a)
