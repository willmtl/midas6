#!/usr/bin/env python3
"""How much of the flagship's book is FOREIGN (ADR / foreign-domiciled) vs domestic US/CA? We store no domicile,
so fetch EODHD General::AddressData::Country (the HQ country — the real domicile signal; CountryISO is just the
listing exchange = 'US' for every ADR) + ISIN prefix for each unique pick, classify, and weight by each name's
portfolio contribution. Reads .data/studies/flagship_history.json. Cache: .data/domicile_cache.json.
Run: docker exec -w /app rotation-backend-1 python -u flagship_domicile.py"""
import os, json, urllib.request, urllib.parse
from collections import defaultdict
import numpy as np
from pathlib import Path

KEY = os.environ.get("EODHD_API_KEY") or os.environ.get("EODHD_TOKEN") or os.environ.get("EODHD_API_TOKEN")
CACHE = Path("/app/.data/domicile_cache.json")


def _sym(t):
    if "." in t:
        return t
    return t + ".US"


def fetch_country(t, cache):
    if t in cache:
        return cache[t]
    url = (f"https://eodhd.com/api/fundamentals/{urllib.parse.quote(_sym(t))}"
           f"?api_token={KEY}&filter=General::AddressData::Country,General::ISIN,General::CountryISO&fmt=json")
    hq = iso = None
    try:
        with urllib.request.urlopen(url, timeout=25) as r:
            j = json.load(r)
        hq = j.get("General::AddressData::Country")
        _isin = j.get("General::ISIN") or ""
        iso = _isin[:2] if len(_isin) >= 2 else None
    except Exception:
        pass
    cache[t] = {"hq": hq, "isin_cc": iso}
    return cache[t]


def main():
    d = json.load(open("/app/.data/studies/flagship_history.json"))
    contrib = defaultdict(float); nmo = defaultdict(int); co = {}
    for m in d["months"]:
        ps = [p for p in m.get("picks", []) if p.get("ret") is not None and p.get("weight")]
        W = sum(p["weight"] for p in ps) or 1
        for p in ps:
            if p.get("ticker"):
                t = p["ticker"]; contrib[t] += p["weight"] / W * p["ret"]; nmo[t] += 1; co[t] = p.get("company")
    tickers = sorted(contrib)
    cache = json.load(open(CACHE)) if CACHE.exists() else {}
    for i, t in enumerate(tickers):
        fetch_country(t, cache)
        if i % 40 == 0:
            CACHE.write_text(json.dumps(cache))
    CACHE.write_text(json.dumps(cache))

    DOMESTIC = {"United States", "USA", "Canada", None}   # US+CA HQ = domestic (the flagship universe)
    dom = defaultdict(lambda: {"n": 0, "mo": 0, "contrib": 0.0, "names": []})
    for t in tickers:
        hq = cache[t].get("hq"); iso = cache[t].get("isin_cc")
        # foreign if HQ outside US/CA. Marshall Islands/Bermuda/Cayman = offshore-incorporated US operators -> treat
        # by HQ (AddressData.Country), which is the economic domicile.
        foreign = hq not in DOMESTIC
        key = "FOREIGN (ADR)" if foreign else "domestic US/CA"
        g = dom[key]; g["n"] += 1; g["mo"] += nmo[t]; g["contrib"] += contrib[t]
        if foreign:
            g["names"].append((t, hq, iso, nmo[t], contrib[t] * 100, co.get(t)))
    tot_mo = sum(nmo.values()); tot_ctr = sum(contrib.values())
    print(f"\n=== FLAGSHIP DOMICILE: foreign (ADR) vs domestic exposure ({len(tickers)} unique picks) ===", flush=True)
    for k in ("domestic US/CA", "FOREIGN (ADR)"):
        g = dom[k]
        print(f"  {k:18} {g['n']:>4} names | {g['mo']:>4} pick-months ({100*g['mo']/tot_mo:.0f}%) | "
              f"contrib {g['contrib']*100:>+7.1f}pp ({100*g['contrib']/tot_ctr:+.0f}% of total)", flush=True)
    print("\n  FOREIGN (ADR) picks — HQ / ISIN-cc / #mo / contribution:", flush=True)
    for t, hq, iso, n, c, cn in sorted(dom["FOREIGN (ADR)"]["names"], key=lambda x: -x[4]):
        print(f"    {t:8} {str(hq)[:18]:18} {str(iso):3} {n:>2}mo {c:>+6.1f}pp  {(cn or '')[:24]}", flush=True)


if __name__ == "__main__":
    main()
