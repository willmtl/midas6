#!/usr/bin/env python3
"""Expand 4h intraday coverage: fetch EODHD 1h->4h for every liquid analyst-covered US name lacking a cached
parquet, so the H4 discovery arm covers the full liquid universe (not just the 562 H4 candidates). Idempotent —
get_4h() skips names already cached. I/O-bound, so threaded. ⚠️ Still survivor-biased: no intraday for delisted.
Run DETACHED in the egress container:
  MSYS_NO_PATHCONV=1 docker exec rotation-celery-worker-1 sh -c 'cd /app && setsid nohup python -u fetch_intraday_expand.py > /app/.data/_intraday_expand.log 2>&1 &'"""
import os, json, glob, threading
os.environ.setdefault("DJANGO_SETTINGS_MODULE", "rotation.settings")
import django; django.setup()
import concurrent.futures as cf
from pathlib import Path
from core.models import Sector
from intraday_data import get_4h, MIN_BARS

etfs = set(Sector.objects.values_list("etf", flat=True)) | {"SPY", "QQQ"}
covered = set()
for line in Path("/app/.data/analyst_ratings.jsonl").read_text(encoding="utf-8").splitlines():
    if not line.strip():
        continue
    try:
        tk = json.loads(line).get("ticker")
    except Exception:
        tk = None
    if tk and "." not in tk and tk not in etfs:
        covered.add(tk)
have = set(f.split("/")[-1][:-8] for f in glob.glob("/app/.data/intraday/4h/*.parquet"))
todo = sorted(covered - have)
print(f"liquid covered {len(covered)} | have 4h {len(have)} | FETCHING {len(todo)}", flush=True)

lock = threading.Lock()
state = {"ok": 0, "empty": 0, "err": 0, "done": 0}


def _one(tk):
    try:
        df = get_4h(tk, years=5, allow_fetch=True)
        r = "ok" if (df is not None and len(df) >= MIN_BARS) else "empty"
    except Exception:
        r = "err"
    with lock:
        state[r] += 1; state["done"] += 1
        if state["done"] % 100 == 0:
            print(f"  {state['done']}/{len(todo)}  ok={state['ok']} empty={state['empty']} err={state['err']}", flush=True)


with cf.ThreadPoolExecutor(max_workers=6) as ex:
    list(ex.map(_one, todo))
print(f"DONE intraday expand: ok={state['ok']} empty={state['empty']} err={state['err']} of {len(todo)}", flush=True)
