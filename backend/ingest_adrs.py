#!/usr/bin/env python3
"""Ingest the screened foreign small-cap-value ADRs (.data/adr_candidates.json) into the backtest universe:
(1) daily candles (fetch_candles_eodhd.backfill) and (2) PIT financial reports (fetch_delisted_fundamentals._fetch
-> core.FinancialReport, filing-dated). Sector mapping is already in adr_candidates.json (ticker->sector-ETF),
merged by the flagship engine via its ADR hook. Non-destructive per ticker.
Run: docker exec -w /app rotation-backend-1 python -u ingest_adrs.py"""
import os, sys, json
sys.path.insert(0, "/app")
os.environ.setdefault("DJANGO_SETTINGS_MODULE", "rotation.settings")
import django
django.setup()
from core.models import FinancialReport, Candle

TKS = sorted(json.load(open("/app/.data/adr_candidates.json")))
print(f"=== INGEST {len(TKS)} ADRs ===", flush=True)

# 1) daily candles (11y history, non-destructive)
from fetch_candles_eodhd import backfill
backfill(TKS, jobs=8)

# 2) PIT financial reports
from fetch_delisted_fundamentals import _fetch
saved = names = 0
for tk in TKS:
    try:
        r = _fetch(tk, f"{tk}.US")
    except Exception as e:
        print(f"  {tk}: fundamentals ERR {e}", flush=True); continue
    if not r or not r.get("rows"):
        continue
    good = [row for row in r["rows"] if row.get("total_equity") is not None or row.get("net_income") is not None]
    if good:
        FinancialReport.objects.filter(ticker=tk).delete()
        FinancialReport.objects.bulk_create([FinancialReport(**row) for row in good],
                                            ignore_conflicts=True, batch_size=2000)
        saved += len(good); names += 1

# report coverage
c_have = sum(1 for tk in TKS if Candle.objects.filter(ticker=tk, interval="1d").exists())
f_have = sum(1 for tk in TKS if FinancialReport.objects.filter(ticker=tk).exists())
print(f"\nDONE: {names} names got FinancialReport ({saved} rows). Coverage of {len(TKS)}: "
      f"candles {c_have}, financials {f_have}", flush=True)
missing = [tk for tk in TKS if not Candle.objects.filter(ticker=tk, interval='1d').exists()
           or not FinancialReport.objects.filter(ticker=tk).exists()]
print(f"incomplete ({len(missing)}): {missing}", flush=True)
