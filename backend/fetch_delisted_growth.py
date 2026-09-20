#!/usr/bin/env python3
"""Fetch candle history for DELISTED growth-GICS names (Info-Tech / Health Care / Comm Services) so the growth-
momentum book can be rebuilt SURVIVORSHIP-CLEAN (user: "fix that by getting past names"). Uses DelistedCompany
.eodhd_symbol -> EODHD eod endpoint, inserts adjusted candles (same convention as fetch_candles_eodhd). Non-
destructive: only writes after a non-empty fetch. Run: MSYS_NO_PATHCONV=1 docker exec rotation-backend-1 python -u /app/fetch_delisted_growth.py"""
import os, json
from concurrent.futures import ThreadPoolExecutor, as_completed
os.environ.setdefault("DJANGO_SETTINGS_MODULE", "rotation.settings")
import django; django.setup()
from api.tasks import _eodhd_get
from core.models import Candle, DelistedCompany

GROWTH_GICS = {"Information Technology", "Technology", "Health Care", "Healthcare", "Communication Services"}


def rows(sym, tk):
    resp = _eodhd_get(f"eod/{sym}")
    if not isinstance(resp, list) or not resp:
        return []
    out = []
    for r in resp:
        d = r.get("date"); cl = r.get("close"); adj = r.get("adjusted_close", cl)
        if not d or cl in (None, "") or adj in (None, ""):
            continue
        try:
            cl = float(cl); adj = float(adj); fac = adj / cl if cl else 1.0
            o = float(r.get("open") or cl) * fac; h = float(r.get("high") or cl) * fac
            lo = float(r.get("low") or cl) * fac; vol = int(float(r.get("volume") or 0))
        except (TypeError, ValueError):
            continue
        if adj <= 0:
            continue
        out.append(Candle(ticker=tk, date=d, interval="1d", open=o, high=h, low=lo, close=adj, volume=vol))
    return out


def main():
    gic = json.load(open("/app/.data/delisted_gic.json"))
    names = [t for t, g in gic.items() if g in GROWTH_GICS]
    sym_map = {d.ticker: (d.eodhd_symbol or (d.ticker + ".US")) for d in DelistedCompany.objects.filter(ticker__in=names)}
    have = set(Candle.objects.filter(ticker__in=names, interval="1d").values_list("ticker", flat=True).distinct())
    todo = [t for t in names if t in sym_map and t not in have]
    print(f"delisted growth-GICS: {len(names)} names, {len(have)} already have candles, fetching {len(todo)}", flush=True)
    results = {}
    with ThreadPoolExecutor(max_workers=3) as ex:
        futs = {ex.submit(rows, sym_map[t], t): t for t in todo}
        for f in as_completed(futs):
            t = futs[f]
            try:
                results[t] = f.result()
            except Exception:
                results[t] = []
    ins = skip = 0
    for t, objs in results.items():
        if not objs:
            skip += 1; continue
        Candle.objects.filter(ticker=t, interval="1d").delete()
        Candle.objects.bulk_create(objs, ignore_conflicts=True, batch_size=5000)
        ins += 1
        if ins % 50 == 0:
            print(f"  ...{ins} inserted / {skip} empty", flush=True)
    print(f"DONE: {ins} names inserted, {skip} empty/unavailable of {len(todo)}", flush=True)


if __name__ == "__main__":
    main()
