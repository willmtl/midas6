"""Targeted EODHD candle backfill for a handful of ETF tickers (the finviz-gap sector adds).

Reuses fetch_candles_eodhd._rows (adjusted EOD, non-destructive per-ticker replace) so we don't
re-fetch the whole ~1200-ticker universe just to add 7 ETFs. Run in the celery worker (EODHD egress):
  docker exec rotation-celery-worker-1 python -u /app/backfill_etfs.py GDX GDXJ XOP OIH IAI KBE IHF
"""
import os, sys
os.environ.setdefault("DJANGO_SETTINGS_MODULE", "rotation.settings")
import django; django.setup()
from core.models import Candle
from fetch_candles_eodhd import _rows, FROM

def main(tickers):
    print(f"backfill {len(tickers)} tickers from {FROM}: {tickers}", flush=True)
    for tk in tickers:
        objs = _rows(tk)
        if not objs:
            print(f"  {tk:6} NO DATA (skipped, existing rows untouched)", flush=True)
            continue
        Candle.objects.filter(ticker=tk, interval="1d").delete()
        Candle.objects.bulk_create(objs, ignore_conflicts=True, batch_size=5000)
        print(f"  {tk:6} {len(objs)} bars  {objs[0].date} -> {objs[-1].date}  (last {objs[-1].close:.2f})", flush=True)
    print("DONE", flush=True)

if __name__ == "__main__":
    tks = sys.argv[1:] or ["GDX", "GDXJ", "XOP", "OIH", "IAI", "KBE", "IHF"]
    main(tks)
