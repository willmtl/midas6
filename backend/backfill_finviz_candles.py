"""
Backfill DAILY CANDLES (EODHD) for the Finviz-universe US/CA names that have none — the ACTUAL
bottleneck for the Finviz industry engine. We had price history for only ~1,177 tickers (the ETF
universe + holdings); the 149 Finviz industries hold ~10k names, so without their candles the engine
can rank/pick from only ~656. This fills the gap so the industry breadth becomes real.

Reuses the non-destructive EODHD re-importer (fetch_candles_eodhd.backfill): a ticker's rows are only
replaced after a NON-EMPTY fetch, and brand-new tickers are simply inserted. Idempotent.

  MSYS_NO_PATHCONV=1 docker exec -d rotation-celery-worker-1 \
    sh -c 'cd /app && setsid nohup python -u backfill_finviz_candles.py --jobs 8 \
           > /app/.data/finviz_candles_backfill.log 2>&1 < /dev/null &'
"""
import os
import sys
import json
import django

os.environ.setdefault("DJANGO_SETTINGS_MODULE", "rotation.settings")
django.setup()

from core.models import Candle

FV = "/app/.data/finviz_universe.json"


def missing_tickers():
    fv = json.load(open(FV))
    usca = [t for t, d in fv["by_ticker"].items() if d["country"] in ("USA", "Canada")]
    from django.db import connection
    with connection.cursor() as cur:                       # DISTINCT over the hypertable needs this (see memory)
        cur.execute("SET max_parallel_workers_per_gather = 0")
    have = set(Candle.objects.filter(interval="1d").values_list("ticker", flat=True).distinct())
    miss = [t for t in usca if t not in have]
    print(f"finviz US/CA={len(usca)} | have candles={len(set(usca) & have)} | MISSING={len(miss)}", flush=True)
    return miss


if __name__ == "__main__":
    jobs = int(sys.argv[sys.argv.index("--jobs") + 1]) if "--jobs" in sys.argv else 8
    limit = int(sys.argv[sys.argv.index("--limit") + 1]) if "--limit" in sys.argv else None
    if not os.environ.get("EODHD_API_KEY"):
        print("EODHD_API_KEY not set — aborting", flush=True); sys.exit(1)
    miss = missing_tickers()
    if limit:
        miss = miss[:limit]
    from fetch_candles_eodhd import backfill
    print(f"EODHD candle backfill launching for {len(miss)} names, jobs={jobs}", flush=True)
    backfill(miss, jobs=jobs)
    print("CANDLES BACKFILL COMPLETE", flush=True)
