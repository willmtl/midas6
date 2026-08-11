#!/usr/bin/env python
"""Standalone EODHD Delisted-Companies importer.

Pulls the EODHD exchange-symbol-list with delisted=1 for the given exchange(s)
and upserts rows into core.models.DelistedCompany.

Usage (inside rotation-backend-1 container, CWD=/app):
    python -u /app/fetch_delisted.py --probe   # fetch US list, print counts+samples, save nothing (default)
    python -u /app/fetch_delisted.py --run      # do the import
"""
import os
import argparse

os.environ.setdefault("DJANGO_SETTINGS_MODULE", "rotation.settings")
import django  # noqa: E402
django.setup()

from api.tasks import _eodhd_get  # noqa: E402
from core.models import DelistedCompany  # noqa: E402

# Types we keep (equities / ETFs). Everything else (funds, bonds, preferred, notes...) is skipped.
KEEP_TYPES = {"common stock", "etf", "preferred stock", "common"}


def _row_get(row, *keys, default=""):
    """Case-tolerant fetch of a field from an EODHD row."""
    for k in keys:
        if k in row and row[k] is not None:
            return row[k]
    lower = {str(k).lower(): v for k, v in row.items()}
    for k in keys:
        v = lower.get(str(k).lower())
        if v is not None:
            return v
    return default


def _keep(row):
    t = str(_row_get(row, "Type", default="")).strip().lower()
    if not t:
        return True  # keep if unknown; don't over-filter
    return t in KEEP_TYPES


def import_delisted(exchanges=("US",)):
    """Idempotent import of delisted symbols for each exchange.

    Returns {"saved": total_upserted, "exchanges": {ex: {"fetched": n, "kept": n, "saved": n}}}.
    """
    total_saved = 0
    per_ex = {}
    for ex in exchanges:
        rows = _eodhd_get(f"exchange-symbol-list/{ex}", delisted=1)
        if not isinstance(rows, list):
            per_ex[ex] = {"fetched": 0, "kept": 0, "saved": 0, "error": "no data"}
            continue
        fetched = len(rows)
        saved = 0
        kept = 0
        for row in rows:
            if not _keep(row):
                continue
            kept += 1
            code = str(_row_get(row, "Code", default="")).strip()
            if not code:
                continue
            exchange = str(_row_get(row, "Exchange", default=ex)).strip() or ex
            defaults = {
                "eodhd_symbol": f"{code}.{ex}",
                "name": str(_row_get(row, "Name", default=""))[:255],
                "country": str(_row_get(row, "Country", default=""))[:100],
                "currency": str(_row_get(row, "Currency", default=""))[:20],
                "type": str(_row_get(row, "Type", default=""))[:60],
                "isin": str(_row_get(row, "Isin", "ISIN", default=""))[:40],
                "delisted_date": None,  # not present on this endpoint
            }
            DelistedCompany.objects.update_or_create(
                ticker=code, exchange=exchange, defaults=defaults,
            )
            saved += 1
        per_ex[ex] = {"fetched": fetched, "kept": kept, "saved": saved}
        total_saved += saved
    return {"saved": total_saved, "exchanges": per_ex}


def probe(exchange="US"):
    rows = _eodhd_get(f"exchange-symbol-list/{exchange}", delisted=1)
    if not isinstance(rows, list):
        print(f"[probe] {exchange}: no list returned -> {type(rows).__name__}: {rows!r}")
        return
    print(f"[probe] {exchange} delisted rows: {len(rows)}")
    if rows:
        print(f"[probe] field keys: {list(rows[0].keys())}")
        print("[probe] 5 sample rows:")
        for r in rows[:5]:
            print("   ", r)


def main():
    ap = argparse.ArgumentParser(description="EODHD delisted-companies importer")
    ap.add_argument("--probe", action="store_true", help="fetch US delisted list, print counts+samples, save nothing (default)")
    ap.add_argument("--run", action="store_true", help="perform the import")
    ap.add_argument("--exchanges", default="US", help="comma-separated exchange codes (default: US)")
    args = ap.parse_args()

    exchanges = tuple(x.strip() for x in args.exchanges.split(",") if x.strip())

    if args.run:
        result = import_delisted(exchanges=exchanges)
        print(f"[run] result: {result}")
    else:
        probe(exchanges[0] if exchanges else "US")


if __name__ == "__main__":
    main()
