#!/usr/bin/env python3
"""Ingest PIT fundamentals (core.FinancialReport) for clean US-listed metal-miner equities so the flagship's value
gate can rank them (user: "get the fundamentals and add stocks to Platinum + all metals"). Only USD-reporting,
US/CA-listed, positive-equity names — EXCLUDES SBSW (statements are ZAR while price is USD ADR => fabricated-cheap
P/B bug) and GENM (negative equity). Candles already present for all. Non-destructive per ticker.
Run: MSYS_NO_PATHCONV=1 docker exec rotation-backend-1 python -u /app/ingest_metals.py"""
import os
os.environ.setdefault("DJANGO_SETTINGS_MODULE", "rotation.settings")
import django; django.setup()
from core.models import FinancialReport, Candle
from fetch_delisted_fundamentals import _fetch

# plain ticker -> EODHD symbol (all .US, USD statements verified)
TKS = {
    "PLG": "PLG.US",     # Platinum Group Metals (platinum)
    "TGB": "TGB.US",     # Taseko Mines (copper)
    "WRN": "WRN.US",     # Western Copper & Gold (copper, dev)
    "ERO": "ERO.US",     # Ero Copper
    "HBM": "HBM.US",     # Hudbay Minerals (copper)
    "SCCO": "SCCO.US",   # Southern Copper
    "FCX": "FCX.US",     # Freeport-McMoRan
    "CENX": "CENX.US",   # Century Aluminum
    "KALU": "KALU.US",   # Kaiser Aluminum
    "SXC": "SXC.US",     # SunCoke (met coke / steel input)
}

print(f"=== INGEST FinancialReport for {len(TKS)} metal miners ===", flush=True)
saved = names = 0
for tk, sym in TKS.items():
    try:
        r = _fetch(tk, sym)
    except Exception as e:
        print(f"  {tk}: ERR {e}", flush=True); continue
    if not r or not r.get("rows"):
        print(f"  {tk}: no rows", flush=True); continue
    good = [row for row in r["rows"] if row.get("total_equity") is not None or row.get("net_income") is not None]
    if good:
        FinancialReport.objects.filter(ticker=tk).delete()
        FinancialReport.objects.bulk_create([FinancialReport(**row) for row in good],
                                            ignore_conflicts=True, batch_size=2000)
        saved += len(good); names += 1
    nc = Candle.objects.filter(ticker=tk, interval="1d").count()
    eq = next((row.get("total_equity") for row in reversed(good) if row.get("total_equity")), None)
    print(f"  {tk:5} rows={len(good):3} candles={nc:5} latest_equity={eq}", flush=True)

print(f"\nDONE: {names}/{len(TKS)} names got FinancialReport ({saved} rows)", flush=True)
