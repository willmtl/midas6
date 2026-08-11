#!/usr/bin/env python
"""Standalone EODHD US-Treasury-Rates importer.

Pulls the three UST rate series from EODHD and explodes them into
core.models.TreasuryRate rows (one per date+series+tenor). 5-year window.

Endpoints (base path `ust/{sub}`):
    ust/yield-rates       -> series="yield"  (nominal par yield curve)
    ust/bill-rates        -> series="bill"   (T-bill rates)
    ust/real-yield-rates  -> series="real"   (TIPS real yields)

Usage (inside rotation-backend-1, CWD=/app):
    python -u /app/fetch_ust_rates.py --probe   # inspect raw keys, save nothing
    python -u /app/fetch_ust_rates.py --run      # import 5 years
"""
import os
import re
import argparse
from datetime import date

os.environ.setdefault("DJANGO_SETTINGS_MODULE", "rotation.settings")
import django  # noqa: E402
django.setup()

from api.tasks import _eodhd_get  # noqa: E402
from core.models import TreasuryRate  # noqa: E402

# sub-path -> series label
SERIES = {
    "yield": "ust/yield-rates",
    "bill": "ust/bill-rates",
    "real": "ust/real-yield-rates",
}

# T-bill quotes are given in weeks; map to the calendar-tenor equivalents.
_WEEK_MAP = {4: "1M", 8: "2M", 13: "3M", 17: "4M", 26: "6M", 52: "1Y"}


def _norm_tenor(raw):
    """Normalize an API tenor/column key to a canonical string.

    Handles: already-normalized ("1M","10Y","6M"), Treasury.gov wide-column
    names ("1 Mo","10 Yr","1.5 Month"), and bill weeks ("4WK","13 Wk","52 Week").
    Returns canonical "{n}M"/"{n}Y" or None if it isn't a tenor column.
    """
    s = str(raw).strip().upper()
    # weeks -> month/year equivalent
    m = re.fullmatch(r"(\d+)\s*(?:WK|WKS|WEEK|WEEKS|W)", s)
    if m:
        return _WEEK_MAP.get(int(m.group(1)))
    # months
    m = re.fullmatch(r"(\d+(?:\.\d+)?)\s*(?:MO|MOS|MONTH|MONTHS|M)", s)
    if m:
        n = float(m.group(1))
        return f"{int(n)}M" if n.is_integer() else f"{n:g}M"
    # years
    m = re.fullmatch(r"(\d+(?:\.\d+)?)\s*(?:YR|YRS|YEAR|YEARS|Y)", s)
    if m:
        n = float(m.group(1))
        return f"{int(n)}Y" if n.is_integer() else f"{n:g}Y"
    return None


def _to_float(v):
    if v is None:
        return None
    s = str(v).strip().replace("%", "").replace(",", "")
    if s == "" or s.upper() in ("N/A", "NA", "NULL", "NONE", "-"):
        return None
    try:
        return float(s)
    except (TypeError, ValueError):
        return None


# non-tenor, non-rate keys that may appear in a row
_META_KEYS = {
    "date", "cusip", "maturity_date", "security", "security_type",
    "security_term", "index_date",
}
# in long/tidy rows, the value carrying the rate (in priority order).
# For bills, "coupon" is the coupon-equivalent yield (comparable to par yields).
_RATE_VALUE_KEYS = ("rate", "coupon", "coupon_equivalent", "avg_coupon",
                    "discount", "avg_discount", "value", "yield")


def _rows_from_payload(payload):
    """EODHD may return a bare list or {"data": [...], "meta": {...}}."""
    if payload is None:
        return []
    if isinstance(payload, dict):
        if isinstance(payload.get("data"), list):
            return payload["data"]
        return []
    if isinstance(payload, list):
        return payload
    return []


def _explode_row(row):
    """Yield (date_str, tenor, rate_float) from one API row.

    Supports BOTH shapes:
      long/tidy:  {"date","tenor","rate"|"coupon"|"discount",...}
      wide:       {"date", "1 Mo":.., "10 Yr":.., ...}  (column-per-tenor)
    """
    if not isinstance(row, dict):
        return
    d = row.get("date") or row.get("Date") or row.get("record_date")
    if not d:
        return
    d = str(d)[:10]

    if "tenor" in row or "Tenor" in row:  # long/tidy format
        tenor = _norm_tenor(row.get("tenor") or row.get("Tenor"))
        if not tenor:
            return
        rate = None
        for k in _RATE_VALUE_KEYS:
            if k in row:
                rate = _to_float(row[k])
                if rate is not None:
                    break
        if rate is not None:
            yield d, tenor, rate
        return

    # wide format: every non-meta key is a tenor column
    for k, v in row.items():
        if k in _META_KEYS or str(k).lower() == "date":
            continue
        tenor = _norm_tenor(k)
        if not tenor:
            continue
        rate = _to_float(v)
        if rate is not None:
            yield d, tenor, rate


def _fetch_series_year(sub, year):
    return _eodhd_get(sub, **{"filter[year]": year})


def import_ust_rates(years=5):
    """Import the last `years` calendar years for all three series.
    Idempotent via bulk_create(ignore_conflicts=True) on unique (date,series,tenor).
    """
    this_year = date.today().year
    year_list = list(range(this_year - years + 1, this_year + 1))
    total_saved = 0
    by_series = {}

    for series, sub in SERIES.items():
        objs = {}  # (date,tenor) -> TreasuryRate  (dedupe within pull)
        for yr in year_list:
            payload = _fetch_series_year(sub, yr)
            for row in _rows_from_payload(payload):
                for d, tenor, rate in _explode_row(row):
                    objs[(d, tenor)] = TreasuryRate(
                        date=d, series=series, tenor=tenor, rate=rate
                    )
        before = TreasuryRate.objects.filter(series=series).count()
        if objs:
            TreasuryRate.objects.bulk_create(
                list(objs.values()), ignore_conflicts=True, batch_size=2000
            )
        after = TreasuryRate.objects.filter(series=series).count()
        saved = after - before
        by_series[series] = {"fetched_rows": len(objs), "net_new": saved}
        total_saved += saved

    return {"saved": total_saved, "by_series": by_series}


def probe():
    this_year = date.today().year
    for series, sub in SERIES.items():
        print("=" * 70)
        print(f"SERIES {series!r}  ->  {sub}  (year={this_year})")
        payload = _fetch_series_year(sub, this_year)
        rows = _rows_from_payload(payload)
        if isinstance(payload, dict):
            print(f"  payload type: dict, top keys: {list(payload.keys())}")
            if "meta" in payload:
                print(f"  meta: {payload.get('meta')}")
        else:
            print(f"  payload type: {type(payload).__name__}")
        print(f"  row count: {len(rows)}")
        if rows:
            print(f"  RAW first-row keys: {list(rows[0].keys()) if isinstance(rows[0], dict) else rows[0]}")
            print(f"  RAW first row: {rows[0]}")
            parsed = []
            for row in rows:
                for triple in _explode_row(row):
                    parsed.append(triple)
                    if len(parsed) >= 3:
                        break
                if len(parsed) >= 3:
                    break
            print(f"  sample parsed (date,tenor,rate): {parsed}")
            tenors = set()
            for row in rows:
                for _, tenor, _r in _explode_row(row):
                    tenors.add(tenor)
            print(f"  distinct normalized tenors this year: {sorted(tenors)}")
        else:
            print("  (no rows returned)")


def main():
    ap = argparse.ArgumentParser(description="EODHD US Treasury rates importer")
    ap.add_argument("--probe", action="store_true", help="inspect raw response, save nothing")
    ap.add_argument("--run", action="store_true", help="import 5 years")
    ap.add_argument("--years", type=int, default=5)
    args = ap.parse_args()

    if args.run:
        result = import_ust_rates(years=args.years)
        print("IMPORT RESULT:", result)
    else:
        probe()


if __name__ == "__main__":
    main()
