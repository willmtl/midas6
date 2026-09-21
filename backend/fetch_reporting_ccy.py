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
from survivorship_smallcap_study import SUF_CCY   # exchange-suffix -> QUOTE currency

OUT = "/app/.data/reporting_ccy.json"


def quote_ccy(tk):
    return "USD" if "." not in tk else SUF_CCY.get(tk.rsplit(".", 1)[1], "USD")


def eodhd_sym(tk):
    return f"{tk}.US" if "." not in tk else tk


def main():
    tks = set()
    for name in config.SECTOR_ETFS:
        for t in sh.get_holdings(name):
            tks.add(t)
    have = set(FinancialReport.objects.filter(ticker__in=list(tks)).values_list("ticker", flat=True).distinct())
    cand = sorted(t for t in tks if t in have)      # ALL candidates (dotted + no-dot)
    print(f"scanning reporting ccy for {len(cand)} candidates (store where reporting != QUOTE ccy)...", flush=True)

    mism = {}
    for i, tk in enumerate(cand, 1):
        q = quote_ccy(tk)
        try:
            f = _eodhd_get(f"fundamentals/{eodhd_sym(tk)}")
            bs = (f or {}).get("Financials", {}).get("Balance_Sheet", {}) if isinstance(f, dict) else {}
            rep = bs.get("currency_symbol")
        except Exception:
            rep = None
        if rep and rep not in ("", None) and rep != q:      # reporting ccy differs from the quote ccy -> P/B mis-scaled
            mism[tk] = rep
            print(f"  [{i}] {tk}: reporting {rep} vs quote {q}", flush=True)
        if i % 150 == 0:
            print(f"  ...{i}/{len(cand)} scanned, {len(mism)} mismatched so far", flush=True)
    json.dump(mism, open(OUT, "w"), indent=1)
    print(f"\nWROTE {OUT}: {len(mism)} reporting!=quote names -> {mism}", flush=True)


if __name__ == "__main__":
    main()
