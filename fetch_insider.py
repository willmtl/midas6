#!/usr/bin/env python3
"""Backfill insider OPEN-MARKET transactions from SEC's bulk Form 345 datasets into the
InsiderBuy table. One quarterly ZIP per quarter (~14MB, ~24 requests total) — vastly
cheaper than per-filing Form 4 XML. Buys = TRANS_CODE 'P' (open-market purchase), the
signal-rich trades; sells = 'S'. Grants/exercises/gifts are ignored (noise).

Run: docker compose run --rm backend python -u fetch_insider.py --db
Flags: --start-year YYYY (default 2020)  --tickers AAPL,MSFT
`filed_date` (public disclosure) is the point-in-time key.
"""
import warnings; warnings.filterwarnings("ignore")
import os, sys, io, csv, zipfile, urllib.request
from datetime import datetime

UA = {"User-Agent": "webisoft-research william@webisoft.com"}
BASE = "https://www.sec.gov/files/structureddata/data/insider-transactions-data-sets"


def _quarters(start_year):
    # Up to a safe upper bound; missing future quarters just 404 and are skipped.
    for y in range(start_year, 2027):
        for q in range(1, 5):
            yield f"{y}q{q}"


def _parse_date(s):
    s = (s or "").strip()
    for fmt in ("%d-%b-%Y", "%Y-%m-%d", "%m/%d/%Y"):
        try:
            return datetime.strptime(s, fmt).date()
        except ValueError:
            continue
    return None


def _num(s):
    try:
        return float(s)
    except (TypeError, ValueError):
        return None


def _process_quarter(qtr, universe):
    """Return {(ticker, filed_date): [buy_value, sell_value, buy_count]} for one quarter."""
    url = f"{BASE}/{qtr}_form345.zip"
    try:
        req = urllib.request.Request(url, headers=UA)
        raw = urllib.request.urlopen(req, timeout=90).read()
    except Exception as e:
        print(f"  {qtr}: skip ({type(e).__name__})")
        return {}
    z = zipfile.ZipFile(io.BytesIO(raw))
    names = {n.upper(): n for n in z.namelist()}
    sub_n = names.get("SUBMISSION.TSV")
    trans_n = names.get("NONDERIV_TRANS.TSV")
    if not sub_n or not trans_n:
        print(f"  {qtr}: missing TSVs {z.namelist()}")
        return {}

    # accession -> (ticker, filed_date)
    sub = {}
    with z.open(sub_n) as f:
        rd = csv.DictReader(io.TextIOWrapper(f, "latin-1"), delimiter="\t")
        for row in rd:
            sym = (row.get("ISSUERTRADINGSYMBOL") or "").strip().upper()
            if sym not in universe:
                continue
            fd = _parse_date(row.get("FILING_DATE"))
            if fd:
                sub[row["ACCESSION_NUMBER"]] = (sym, fd)

    agg = {}
    with z.open(trans_n) as f:
        rd = csv.DictReader(io.TextIOWrapper(f, "latin-1"), delimiter="\t")
        for row in rd:
            acc = row.get("ACCESSION_NUMBER")
            meta = sub.get(acc)
            if not meta:
                continue
            code = (row.get("TRANS_CODE") or "").strip().upper()
            if code not in ("P", "S"):
                continue
            shares = _num(row.get("TRANS_SHARES"))
            price = _num(row.get("TRANS_PRICEPERSHARE"))
            if not shares or price is None:
                continue
            value = int(abs(shares) * price)
            key = meta
            a = agg.setdefault(key, [0, 0, 0])
            if code == "P":
                a[0] += value; a[2] += 1
            else:
                a[1] += value
    print(f"  {qtr}: {len(agg)} (ticker,day) aggregates")
    return agg


def run(start_year=2020, tickers=None):
    from seq_fundamental_study import build_universe
    from core.models import InsiderBuy
    universe = set(t.upper() for t in (tickers or build_universe()))
    print(f"Insider backfill | universe {len(universe)} tickers | from {start_year}")
    merged = {}
    for qtr in _quarters(start_year):
        for k, v in _process_quarter(qtr, universe).items():
            m = merged.setdefault(k, [0, 0, 0])
            m[0] += v[0]; m[1] += v[1]; m[2] += v[2]
    print(f"Upserting {len(merged)} InsiderBuy rows...")
    for (tk, fd), (bv, sv, bc) in merged.items():
        InsiderBuy.objects.update_or_create(
            ticker=tk, filed_date=fd,
            defaults={"buy_value": bv, "sell_value": sv, "buy_count": bc})
    print(f"Done: {len(merged)} rows.")
    return len(merged)


if __name__ == "__main__":
    argv = sys.argv
    sy = int(argv[argv.index("--start-year") + 1]) if "--start-year" in argv else 2020
    tks = argv[argv.index("--tickers") + 1].split(",") if "--tickers" in argv else None
    if "--db" in argv:
        import django
        os.environ.setdefault("DJANGO_SETTINGS_MODULE", "rotation.settings")
        django.setup()
    run(start_year=sy, tickers=tks)
