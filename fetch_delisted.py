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

import datetime as _dt  # noqa: E402
import sys  # noqa: E402
from django.db.models import Max  # noqa: E402
from api.tasks import _eodhd_get  # noqa: E402
from core.models import DelistedCompany, Candle  # noqa: E402

STALE_DAYS = 21      # a ticker >3 weeks behind the live market has effectively stopped trading
MARKET_REF = "SPY"   # liquid always-on proxy for "latest market date"

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
                # NOTE: delisted_date is deliberately NOT in defaults — this endpoint ships no date, and
                # including it here would clobber the candle-derived date on every weekly run. It is set
                # separately by date_from_candles() below.
            }
            DelistedCompany.objects.update_or_create(
                ticker=code, exchange=exchange, defaults=defaults,
            )
            saved += 1
        per_ex[ex] = {"fetched": fetched, "kept": kept, "saved": saved}
        total_saved += saved
    return {"saved": total_saved, "exchanges": per_ex}


def date_from_candles(commit=True):
    """Populate DelistedCompany.delisted_date DERIVED from our own candle history — the only reliable
    "is this ticker dead" signal we own (EODHD's delisted feed carries no date). A ticker whose newest
    candle is >STALE_DAYS behind the market's latest date has stopped trading; its last candle date IS
    its last-trade / delisting date. Also CREATES a row for any stale ticker we hold candles for but have
    no delisted record of, so the survivorship reference is complete. Idempotent.
    """
    mkt = Candle.objects.filter(ticker=MARKET_REF).aggregate(m=Max("date"))["m"]
    if not mkt:
        print(f"[date] no {MARKET_REF} candles — cannot establish market date; skip"); return {}
    last = {r["ticker"]: r["last"] for r in Candle.objects.values("ticker").annotate(last=Max("date"))}
    stale = {t: d for t, d in last.items() if (mkt - d).days > STALE_DAYS}

    existing = {}
    for rec in DelistedCompany.objects.all().only("id", "ticker", "exchange", "delisted_date"):
        existing.setdefault(rec.ticker, rec)

    to_update, to_create, updated, created = [], [], 0, 0
    for t, d in stale.items():
        rec = existing.get(t)
        if rec:
            if rec.delisted_date != d:
                rec.delisted_date = d; to_update.append(rec); updated += 1
        else:
            to_create.append(DelistedCompany(ticker=t, exchange="", name=t, delisted_date=d)); created += 1

    if commit:
        if to_update:
            DelistedCompany.objects.bulk_update(to_update, ["delisted_date"], batch_size=1000)
        if to_create:
            DelistedCompany.objects.bulk_create(to_create, batch_size=1000, ignore_conflicts=True)
    print(f"[date] market({MARKET_REF})={mkt} stale(>{STALE_DAYS}d)={len(stale)} "
          f"{'dated' if commit else 'WOULD date'} {updated} existing, "
          f"{'created' if commit else 'WOULD create'} {created} new")

    # visibility: dead sleeves in the ACTIVE rotation universe (config cruft surfaces here)
    try:
        sys.path.insert(0, "/app")
        from config import SECTOR_ETFS, DEACTIVATED_ETFS
        uni = set(SECTOR_ETFS.values()) | set(DEACTIVATED_ETFS.values())
        dead = sorted((t, str(last[t])) for t in uni if t in stale)
        nodata = sorted(t for t in uni if t not in last)
        print(f"[date] ACTIVE UNIVERSE dead/stale sleeves ({len(dead)}): {dead}")
        print(f"[date] ACTIVE UNIVERSE no-candles sleeves ({len(nodata)}): {nodata}")
    except Exception as e:
        print(f"[date] universe report skipped: {e}")
    return {"stale": len(stale), "updated": updated, "created": created}


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
    ap.add_argument("--run", action="store_true", help="import names from EODHD, then date them from our candles")
    ap.add_argument("--date-only", action="store_true", help="skip the EODHD pull; only (re)date from candles")
    ap.add_argument("--dry-run", action="store_true", help="with --date-only: show what would change, commit nothing")
    ap.add_argument("--exchanges", default="US", help="comma-separated exchange codes (default: US)")
    args = ap.parse_args()

    exchanges = tuple(x.strip() for x in args.exchanges.split(",") if x.strip())

    if args.date_only:
        date_from_candles(commit=not args.dry_run)
    elif args.run:
        result = import_delisted(exchanges=exchanges)
        print(f"[run] import result: {result}")
        date_from_candles(commit=True)   # names have no date on the feed — date them from our candles
    else:
        probe(exchanges[0] if exchanges else "US")


if __name__ == "__main__":
    main()
