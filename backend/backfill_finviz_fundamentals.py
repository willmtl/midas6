"""
Backfill quarterly fundamentals (SEC EDGAR companyfacts) for the Finviz-universe US names that lack
them — so the Finviz industry engine can use its full breadth instead of the ~645 it can see today.

Computes: (Finviz US tickers) − (tickers already in FinancialReport), then reuses the proven
fetch_financial_history EDGAR pipeline (free, full history, true filing dates). Skips dividends
(the rotation engine doesn't need them; and yfinance has no container egress). Idempotent.

  MSYS_NO_PATHCONV=1 docker exec -d rotation-celery-worker-1 \
    sh -c 'cd /app && setsid nohup python -u backfill_finviz_fundamentals.py --jobs 5 \
           > /app/.data/finviz_fund_backfill.log 2>&1 < /dev/null &'
"""
import os
import sys
import json
import django

os.environ.setdefault("DJANGO_SETTINGS_MODULE", "rotation.settings")
django.setup()

from core.models import FinancialReport

FV = "/app/.data/finviz_universe.json"


def missing_us_tickers():
    fv = json.load(open(FV))
    us = [t for t, d in fv["by_ticker"].items() if d["country"] == "USA"]
    have = set(FinancialReport.objects.values_list("ticker", flat=True).distinct())
    miss = [t for t in us if t not in have]
    print(f"finviz USA={len(us)} | have={len(set(us) & have)} | MISSING={len(miss)}", flush=True)
    return miss


if __name__ == "__main__":
    jobs = int(sys.argv[sys.argv.index("--jobs") + 1]) if "--jobs" in sys.argv else 5
    limit = int(sys.argv[sys.argv.index("--limit") + 1]) if "--limit" in sys.argv else None
    miss = missing_us_tickers()
    if limit:
        miss = miss[:limit]
    from fetch_financial_history import run
    print(f"EDGAR backfill launching for {len(miss)} names, jobs={jobs}", flush=True)
    run(jobs=jobs, tickers=miss, skip_div=True)
    print("BACKFILL COMPLETE", flush=True)
