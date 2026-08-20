#!/usr/bin/env python3
"""How often does a flagship pick land in the LIQUID + LOW-IV intersection where a deep-ITM call is viable?
Join flagship_history.json picks to OptionSnapshot (829 liquid US names; atm_iv). Coverage 2022-09+, so
restrict to pick-months from then. 'optioned' (in OptionSnapshot) is a liquidity proxy; IV from nearest
prior snapshot. Prints the share of pick-months that qualify.
Run: MSYS_NO_PATHCONV=1 docker exec rotation-backend-1 python -u /app/opt_intersection.py"""
import os, sys, json, warnings
warnings.filterwarnings("ignore")
os.environ.setdefault("DJANGO_SETTINGS_MODULE", "rotation.settings")
import django
django.setup()
import datetime as dt
from collections import defaultdict
from core.models import OptionSnapshot

H = json.load(open("/app/.data/studies/flagship_history.json"))
months = H.get("months") or H.get("history") or []
print(f"trace months={len(months)}; sample keys={list(months[0].keys()) if months else None}", flush=True)

# gather (date, ticker, weight) pick rows
rows = []
for m in months:
    d = m.get("month") or m.get("date") or m.get("as_of")
    for p in (m.get("picks") or []):
        t = p.get("ticker")
        if t and p.get("weight") and d:
            rows.append((str(d)[:10], t, float(p["weight"])))
print(f"total pick-rows={len(rows)}", flush=True)

# option-universe + per-ticker IV series (date-sorted) for nearest-prior lookup
opt_tickers = set(OptionSnapshot.objects.values_list("ticker", flat=True).distinct())
iv_by_t = defaultdict(list)
for r in OptionSnapshot.objects.filter(atm_iv__isnull=False).values_list("ticker", "date", "atm_iv"):
    iv_by_t[r[0]].append((r[1], float(r[2])))
for t in iv_by_t:
    iv_by_t[t].sort()


def iv_asof(t, d):
    ser = iv_by_t.get(t)
    if not ser:
        return None
    dd = dt.date.fromisoformat(d)
    prior = [iv for (dte, iv) in ser if dte <= dd]
    return prior[-1] if prior else (ser[0][1] if ser else None)


OPT_START = dt.date(2022, 9, 1)
elig = [r for r in rows if dt.date.fromisoformat(r[0]) >= OPT_START]
n = len(elig)
optioned = [r for r in elig if r[1] in opt_tickers]
with_iv = [(r, iv_asof(r[1], r[0])) for r in optioned]
with_iv = [(r, iv) for (r, iv) in with_iv if iv is not None]
lo50 = [(r, iv) for (r, iv) in with_iv if iv < 50]
lo40 = [(r, iv) for (r, iv) in with_iv if iv < 40]

print(f"\n=== pick-months 2022-09+ (option-data era): {n} ===", flush=True)
print(f"  optioned (in 829 liquid-name set): {len(optioned)} ({len(optioned)/n*100:.0f}%)", flush=True)
print(f"  optioned + IV known:               {len(with_iv)} ({len(with_iv)/n*100:.0f}%)", flush=True)
print(f"  optioned + IV<50 (viable-ish):     {len(lo50)} ({len(lo50)/n*100:.0f}%)", flush=True)
print(f"  optioned + IV<40 (cheap deep-ITM): {len(lo40)} ({len(lo40)/n*100:.0f}%)", flush=True)
# weight share (how much of the BOOK could be levered, not just name count)
wtot = sum(r[2] for r in elig) or 1
w50 = sum(r[2] for (r, iv) in lo50)
print(f"  ...as share of BOOK WEIGHT: IV<50 covers {w50/wtot*100:.0f}% of deployed weight", flush=True)
print("\n  sample optioned+low-IV picks:", flush=True)
for (r, iv) in sorted(lo50, key=lambda x: x[1])[:12]:
    print(f"    {r[0]}  {r[1]:6} IV {iv:.0f}%  w={r[2]:.1f}", flush=True)
