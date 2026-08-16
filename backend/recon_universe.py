#!/usr/bin/env python3
"""RECON: how binding is our top-20-holdings cap? For a few strong EQUITY-ETF sectors, pull the FULL
current constituent list from EODHD (ETF_Data.Holdings), fetch P/B for every name, and compare the
cheapest-P/B name in the FULL universe vs the cheapest among OUR hardcoded top-20. Counts how many
cheaper-P/B value names we're currently blind to. Read-only (no DB writes).
Run: MSYS_NO_PATHCONV=1 docker exec rotation-backend-1 python -u /app/recon_universe.py
"""
import os, warnings
warnings.filterwarnings("ignore")
os.environ.setdefault("DJANGO_SETTINGS_MODULE", "rotation.settings")
import django
django.setup()

import sector_holdings
from api.tasks import _eodhd_get, _eodhd_sym

SECTORS = [("Semiconductors", "SMH"), ("Cybersecurity", "CIBR"), ("Cloud Computing", "SKYY"),
           ("Technology", "XLK"), ("Energy", "XLE")]
CAP = 60          # cap constituents per ETF (by weight) to bound API calls


def pb_of(sym):
    d = _eodhd_get(f"fundamentals/{sym}")
    if not isinstance(d, dict):
        return None
    v = (d.get("Valuation") or {}).get("PriceBookMRQ")
    try:
        v = float(v)
        return v if v > 0 else None
    except (TypeError, ValueError):
        return None


for name, etf in SECTORS:
    data = _eodhd_get(f"fundamentals/{etf}.US")
    holds = {}
    if isinstance(data, dict) and isinstance(data.get("ETF_Data"), dict):
        hd = data["ETF_Data"].get("Holdings") or {}
        for k, v in hd.items():
            try:
                w = float(v.get("Assets_%", 0) or 0)
            except (TypeError, ValueError):
                w = 0.0
            holds[k] = (v.get("Code") or k.split(".")[0], w)
    full = sorted(holds.items(), key=lambda kv: -kv[1][1])[:CAP]   # top CAP by weight
    ours = [t for t in sector_holdings.get_holdings(name) if t]
    ours_codes = {t.replace("-", ".").upper() for t in ours}       # BRK-B -> BRK.B for matching

    print(f"\n=== {name} ({etf}) === EODHD holdings returned: {len(holds)} (using top {len(full)}) | ours: {len(ours)}", flush=True)
    if not full:
        print("  (no holdings from EODHD — ETF not covered / futures-based)", flush=True)
        continue

    # P/B for full list + our list
    full_pb = {}
    for k, (code, w) in full:
        p = pb_of(k)
        if p is not None:
            full_pb[code.upper()] = (p, w, k)
    ours_pb = {}
    for t in ours:
        p = pb_of(_eodhd_sym(t))
        if p is not None:
            ours_pb[t.replace("-", ".").upper()] = p

    if not full_pb:
        print("  (no P/B available for constituents)", flush=True)
        continue
    cheapest_ours = min(ours_pb.values()) if ours_pb else None
    ranked_full = sorted(full_pb.items(), key=lambda kv: kv[1][0])
    cheapest_full = ranked_full[0]
    # names in FULL cheaper than our cheapest, that are NOT in our top-20
    missing_cheaper = [(c, pv[0], pv[1]) for c, pv in ranked_full
                       if (cheapest_ours is None or pv[0] < cheapest_ours) and c not in ours_codes]

    print(f"  cheapest P/B  ours: {cheapest_ours}  |  full: {cheapest_full[1][0]} ({cheapest_full[0]})", flush=True)
    print(f"  # full-universe names cheaper than our cheapest AND not in our list: {len(missing_cheaper)}", flush=True)
    for c, p, w in missing_cheaper[:8]:
        print(f"     MISSED  {c:8} P/B {p:>6.2f}  (ETF wt {w:.2f}%)", flush=True)
print("\ndone", flush=True)
