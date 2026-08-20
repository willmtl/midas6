#!/usr/bin/env python3
"""Closes the B-group gap: B_plus vs B_climax on the ACTUAL B entry (gap-up confirmation), full exit ladder.
B's deployed trade = capitulation window -> WAIT for a gap-up -> enter. So the fair test of B_climax (volume-
climax window, 21x more frequent) is the same gap-up entry, not raw mean-reversion. Also prints momentum family.
Run: MSYS_NO_PATHCONV=1 docker exec rotation-backend-1 python -u /app/b_climax_check.py [--fetch]"""
import os, sys, warnings
warnings.filterwarnings("ignore")
os.environ.setdefault("DJANGO_SETTINGS_MODULE", "rotation.settings")
import django
django.setup()
import numpy as np
from django.db import connection
with connection.cursor() as cur:
    cur.execute("SET max_parallel_workers_per_gather = 0")
import h4_on_signals_study as S
import h4_study as H
from intraday_data import get_4h

FETCH = "--fetch" in sys.argv
GROUP = ["B_plus", "B_climax"]
EXITS = [k for (k, _b, _) in H.EXITS]
SHOW = [k for k in ["1b", "2b", "3b", "4b", "6b", "8b"] if k in EXITS] or EXITS
MOM = [s for s in H.SIGNALS if H.SIGNALS[s]["family"] in ("momentum", "structure")]


def _t(a):
    a = np.asarray(a, float); a = a[np.isfinite(a)]
    if len(a) < 3:
        return None
    sd = a.std(ddof=1)
    return round(float(a.mean() / (sd / np.sqrt(len(a)))), 2) if sd > 0 else None


def cell(v):
    a = np.asarray(v, float); a = a[np.isfinite(a)]
    return f"{len(a)}/{a.mean():+.2f}/{(a>0).mean()*100:.0f}/{_t(a)}" if len(a) else "—"


def pooled(sel):
    allowed, meta = S.candidate_windows(sel)
    per = {s: {} for s in H.SIGNALS}
    got = 0
    for tk in sorted(allowed):
        df = get_4h(tk, 5, FETCH)
        if df is None or len(df) < 120:
            continue
        got += 1
        res = S.backtest_ticker_masked(df, allowed[tk])
        for s in H.SIGNALS:
            for k, v in res[s]["flat"].items():
                per[s].setdefault(k, []).extend(v)
    return meta, got, per


data = {}
for sel in GROUP:
    meta, got, per = pooled(sel)
    data[sel] = (meta, got, per)
    print(f"[{sel:9}] windows={meta.get('n_windows')} names={meta.get('n_names')} names_with_4h={got}", flush=True)

print(f"\n=== gap-up ENTRY (B's real trade) across exits  (n/avg%/win%/t) ===", flush=True)
print(f"  {'exit':6}" + "".join(f"{sel:>24}" for sel in GROUP), flush=True)
for k in SHOW:
    print(f"  {k:6}" + "".join(f"{cell(data[sel][2]['mo_gap_up'].get(k, [])):>24}" for sel in GROUP), flush=True)

print(f"\n=== all momentum/structure entries at 3b ===", flush=True)
print(f"  {'signal':16}" + "".join(f"{sel:>24}" for sel in GROUP), flush=True)
for s in MOM:
    print(f"  {s:16}" + "".join(f"{cell(data[sel][2][s].get('3b', [])):>24}" for sel in GROUP), flush=True)
