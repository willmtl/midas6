#!/usr/bin/env python3
"""Expand the candle roster toward comprehensive: fetch daily history for the major-exchange US COMMON STOCKS
we're missing (from EODHD's authoritative listing), filtering out warrants/preferreds/units/rights (the junk).
Liquidity is handled at TEST time ($5M/day + >$5), so we only exclude structurally non-common instruments here.
Run DETACHED in the egress container:
  MSYS_NO_PATHCONV=1 docker exec rotation-celery-worker-1 sh -c 'cd /app && setsid nohup python -u fetch_universe_expand.py > /app/.data/_universe_expand.log 2>&1 &'"""
import os, json, re, urllib.request
os.environ.setdefault("DJANGO_SETTINGS_MODULE", "rotation.settings")
import django; django.setup()
from fetch_candles_eodhd import backfill
from django.db import connection

KEY = os.environ.get("EODHD_API_KEY") or os.environ.get("EODHD_TOKEN") or os.environ.get("EODHD_API_TOKEN")
MAJOR = {"NYSE", "NASDAQ", "NYSE ARCA", "NYSE MKT", "BATS", "AMEX", "NMS", "NYQ", "NGM"}

data = json.loads(urllib.request.urlopen(
    f"https://eodhd.com/api/exchange-symbol-list/US?api_token={KEY}&fmt=json", timeout=90).read().decode())
common = [r for r in data if r.get("Type") == "Common Stock" and r.get("Exchange") in MAJOR]

# junk filter: drop warrants/units/rights/preferreds/when-issued (contain '-' or trailing W/U/R warrant codes)
def is_clean(code):
    if "-" in code or "." in code:
        return False
    if not re.fullmatch(r"[A-Z]{1,5}", code):
        return False
    if len(code) == 5 and code[-1] in {"W", "U", "R"}:   # 5th-letter warrant/unit/rights class
        return False
    return True

codes = sorted({r["Code"] for r in common if is_clean(r["Code"])})
with connection.cursor() as c:
    c.execute("SET max_parallel_workers_per_gather=0")
    c.execute("SELECT ticker FROM core_candle GROUP BY ticker")
    have = set(r[0] for r in c.fetchall())
missing = sorted(set(codes) - have)
print(f"major-exch common stocks: {len(common)} | clean: {len(codes)} | already have: {len(codes)-len(missing)} "
      f"| FETCHING {len(missing)} missing", flush=True)
backfill(missing, jobs=8)
print(f"DONE expand: attempted {len(missing)} tickers", flush=True)
