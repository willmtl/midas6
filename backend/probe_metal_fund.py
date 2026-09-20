#!/usr/bin/env python3
"""PROBE: which metal-miner equities have EODHD PIT fundamentals + are US/CA-listed (flagship country gate accepts
US [no dot] and .TO/.V only). For each candidate try a few EODHD symbol variants; report GicSector, latest
total_equity/shares (=> P/B rankable), and whether we already have candles. No writes.
Run: MSYS_NO_PATHCONV=1 docker exec rotation-backend-1 python -u /app/probe_metal_fund.py"""
import os
os.environ.setdefault("DJANGO_SETTINGS_MODULE", "rotation.settings")
import django; django.setup()
from api.tasks import _eodhd_get
from core.models import Candle
from fetch_delisted_fundamentals import _fetch

# metal -> candidate plain tickers (we'll try .US, .TO, .V variants)
CAND = {
    "Platinum": ["SBSW", "PLG", "PTM", "GENM", "GENMF", "NILSY"],
    "Copper":   ["FCX", "SCCO", "ERO", "TGB", "HBM", "CS", "WRN", "IE", "NGD", "CPPMF", "COP", "COPR", "ATEX", "FIL"],
    "Aluminum": ["AA", "CENX", "KALU"],
    "Diversified/Metals": ["TECK", "TKO", "HL", "CDE", "MTAL", "USAP", "SXC", "CMC"],
}


def usca(sym):
    base = sym.split(".")[0]
    ex = sym.split(".")[1] if "." in sym else "US"
    return ex in ("US", "TO", "V")


def main():
    print(f"{'ticker':>8} {'eodhd_sym':>12} {'GicSector':>24} {'eq(latest)':>14} {'sh(latest)':>14} {'rows':>5} {'candles':>7}", flush=True)
    for metal, tks in CAND.items():
        print(f"--- {metal} ---", flush=True)
        for tk in tks:
            found = False
            for ex in ("US", "TO", "V"):
                sym = f"{tk}.{ex}"
                try:
                    r = _fetch(tk, sym)
                except Exception:
                    r = None
                if r and r.get("rows"):
                    rows = [x for x in r["rows"] if x.get("total_equity") is not None]
                    latest = rows[-1] if rows else (r["rows"][-1] if r["rows"] else {})
                    eq = latest.get("total_equity"); sh = latest.get("shares_outstanding")
                    nc = Candle.objects.filter(ticker=tk, interval="1d").count()
                    nc2 = Candle.objects.filter(ticker=sym, interval="1d").count()
                    print(f"{tk:>8} {sym:>12} {str(r.get('gic'))[:24]:>24} {str(eq):>14} {str(sh):>14} "
                          f"{len(r['rows']):>5} {(nc or nc2):>7}", flush=True)
                    found = True
                    break
            if not found:
                print(f"{tk:>8} {'(none)':>12} {'-- no EODHD fundamentals --':>24}", flush=True)


if __name__ == "__main__":
    main()
