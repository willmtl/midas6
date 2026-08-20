#!/usr/bin/env python3
"""Screen US-listed FOREIGN ADRs for the flagship's profile (small-cap value on a major exchange) to surface
SPECIFIC names worth adding to the candidate universe. Reads EODHD's US symbol list, keeps foreign-incorporated
common stocks (non-US ISIN, drops OTC/SPAC/fund/Canada), fetches mktcap/P/B/sector/exchange/HQ, filters to
$300M-$2B + P/B 0.1-3 on NYSE/NASDAQ, ranks by size (liquidity proxy). Cache: .data/adr_screen_cache.json.
NOTE: this floor misses true ADRs carrying US ISINs (ENIA/GDS/WB) — a separate clean set we already partly hold.
Run: docker exec -w /app rotation-backend-1 python -u flagship_adr_screen.py"""
import os, sys, json, urllib.request, urllib.parse
sys.path.insert(0, "/app")
os.environ.setdefault("DJANGO_SETTINGS_MODULE", "rotation.settings")
import django
django.setup()
from pathlib import Path

KEY = os.environ.get("EODHD_API_KEY") or os.environ.get("EODHD_TOKEN") or os.environ.get("EODHD_API_TOKEN")
CACHE = Path("/app/.data/adr_screen_cache.json")


def _fetch(sym):
    url = (f"https://eodhd.com/api/fundamentals/{urllib.parse.quote(sym)}?api_token={KEY}"
           "&filter=General::Exchange,General::Sector,General::AddressData::Country,"
           "Highlights::MarketCapitalization,Valuation::PriceBookMRQ&fmt=json")
    try:
        with urllib.request.urlopen(url, timeout=20) as r:
            return json.load(r)
    except Exception:
        return None


def main():
    from core.models import Fundamental
    have = set(Fundamental.objects.values_list("ticker", flat=True).distinct())
    with urllib.request.urlopen(f"https://eodhd.com/api/exchange-symbol-list/US?api_token={KEY}&fmt=json", timeout=60) as r:
        syms = json.load(r)
    common = [s for s in syms if s.get("Type") == "Common Stock" and s.get("Isin")]

    def junk(s):
        nm = (s.get("Name") or "").lower(); code = s["Code"]
        return (len(code) == 5 and code.endswith(("F", "Y"))) or code.endswith(("W", "R")) or \
               any(x in nm for x in ("acquisition", "spac", "blank check", "trust", " fund", "warrant", "depositary"))
    cands = [s for s in common if s["Isin"][:2] not in ("US", "CA") and not junk(s)]
    print(f"foreign-incorporated real-set to screen: {len(cands)}", flush=True)

    cache = json.load(open(CACHE)) if CACHE.exists() else {}
    hits = []
    for i, s in enumerate(cands):
        code = s["Code"]
        if code not in cache:
            cache[code] = _fetch(f"{code}.US") or {}
            if i % 50 == 0:
                CACHE.write_text(json.dumps(cache)); print(f"  fetched {i}/{len(cands)}", flush=True)
        f = cache[code]
        mc = f.get("Highlights::MarketCapitalization"); pb = f.get("Valuation::PriceBookMRQ")
        exch = f.get("General::Exchange"); sec = f.get("General::Sector"); hq = f.get("General::AddressData::Country")
        if not (mc and pb):
            continue
        if exch not in ("NYSE", "NASDAQ"):                      # major exchange only (drop OTC/PINK)
            continue
        if not (3e8 <= mc <= 2e9):                              # flagship small-cap band
            continue
        if not (0.1 < pb < 3.0):                                # value band
            continue
        hits.append((code, s.get("Name", "")[:26], s["Isin"][:2], hq, sec, mc, pb, code in have))
    CACHE.write_text(json.dumps(cache))

    hits.sort(key=lambda x: -x[5])                              # by mktcap desc (liquidity/popularity proxy)
    new = [h for h in hits if not h[7]]
    # ── write the candidate file the flagship engine will merge into its sector pools (ticker -> sector-ETF) ──
    GIC_TO_ETF = {"Technology": "XLK", "Information Technology": "XLK", "Financials": "XLF",
                  "Financial Services": "XLF", "Health Care": "XLV", "Healthcare": "XLV", "Energy": "XLE",
                  "Consumer Cyclical": "XLY", "Consumer Discretionary": "XLY", "Consumer Defensive": "XLP",
                  "Consumer Staples": "XLP", "Industrials": "XLI", "Basic Materials": "XLB", "Materials": "XLB",
                  "Real Estate": "XLRE", "Communication Services": "XLC", "Utilities": "XLU"}
    adr_map = {}
    for code, nm, cc, hq, sec, mc, pb, held in hits:
        if held:
            continue
        etf = GIC_TO_ETF.get(sec)
        if etf:
            adr_map[code] = etf
    Path("/app/.data/adr_candidates.json").write_text(json.dumps(adr_map, indent=1))
    print(f"\nwrote /app/.data/adr_candidates.json: {len(adr_map)} NEW ADRs mapped to sector-ETFs "
          f"({sorted(set(adr_map.values()))})", flush=True)
    print(f"\n=== FOREIGN small-cap VALUE ADRs on NYSE/NASDAQ ($300M-$2B, P/B 0.1-3): {len(hits)} hits, {len(new)} NOT already held ===", flush=True)
    print(f"  {'code':7}{'name':27}{'cc':4}{'HQ':14}{'sector':22}{'mktcap$M':>9}{'P/B':>6}  have?", flush=True)
    for code, nm, cc, hq, sec, mc, pb, h in hits[:40]:
        print(f"  {code:7}{nm:27}{cc:4}{str(hq)[:13]:14}{str(sec)[:21]:22}{mc/1e6:>8.0f} {pb:>5.2f}  {'HELD' if h else 'NEW'}", flush=True)


if __name__ == "__main__":
    main()
