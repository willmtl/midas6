#!/usr/bin/env python3
"""Detect the REPORTING currency of each US-listed candidate (Financials.Balance_Sheet.currency_symbol from EODHD —
NOT General.CurrencyCode, which is the misleading TRADING ccy). US-listed foreign ADRs report their statements in a
foreign ccy (GDS/TCOM=CNY, GGAL=ARS) while trading in USD -> pb=mktcap(USD)/eq(foreign) is fake-cheap. Writes
/app/.data/reporting_ccy.json = {ticker: reporting_ccy} ONLY for no-dot tickers whose reporting ccy != USD, so the
flagship can convert their equity to USD before P/B. Idempotent.
Run: MSYS_NO_PATHCONV=1 docker exec rotation-backend-1 python -u /app/fetch_reporting_ccy.py"""
import os, json
os.environ.setdefault("DJANGO_SETTINGS_MODULE", "rotation.settings")
import django; django.setup()
import sector_holdings as sh, config
from core.models import FinancialReport
from api.tasks import _eodhd_get

OUT = "/app/.data/reporting_ccy.json"


def main():
    tks = set()
    for name in config.SECTOR_ETFS:
        for t in sh.get_holdings(name):
            tks.add(t)
    nodot = sorted(t for t in tks if "." not in t)
    have = set(FinancialReport.objects.filter(ticker__in=nodot).values_list("ticker", flat=True).distinct())
    cand = [t for t in nodot if t in have]
    print(f"scanning reporting ccy for {len(cand)} US-listed candidates...", flush=True)

    foreign = {}
    for i, tk in enumerate(cand, 1):
        try:
            f = _eodhd_get(f"fundamentals/{tk}.US")
            bs = (f or {}).get("Financials", {}).get("Balance_Sheet", {}) if isinstance(f, dict) else {}
            ccy = bs.get("currency_symbol")
        except Exception:
            ccy = None
        if ccy and ccy not in ("USD", "", None):
            foreign[tk] = ccy
            print(f"  [{i}] {tk}: reporting {ccy}", flush=True)
        if i % 100 == 0:
            print(f"  ...{i}/{len(cand)} scanned, {len(foreign)} foreign so far", flush=True)
    json.dump(foreign, open(OUT, "w"), indent=1)
    print(f"\nWROTE {OUT}: {len(foreign)} US-listed foreign-reporting ADRs -> {foreign}", flush=True)


if __name__ == "__main__":
    main()
