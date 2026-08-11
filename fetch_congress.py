#!/usr/bin/env python
"""Standalone EODHD Congressional-Trades importer.

Pulls disclosed U.S. legislator (Senate/House) trades from the EODHD
`congressional-trades` (beta) endpoint into core.models.CongressTrade.

PIT note: `report_date` = the PUBLIC disclosure date (key PIT studies on this),
which lags the actual transaction_date by `days_to_disclose`.

Usage:
    python fetch_congress.py --probe   # recent slice, print raw keys + samples, save nothing
    python fetch_congress.py --run     # import the 5-year window (HARD RULE)
Default = probe.
"""
import os
import hashlib
import argparse
from datetime import date, timedelta

os.environ.setdefault("DJANGO_SETTINGS_MODULE", "rotation.settings")
import django  # noqa: E402
django.setup()

from api.tasks import _eodhd_get  # noqa: E402
from core.models import CongressTrade  # noqa: E402

PATH = "congressional-trades"
PAGE_LIMIT = 100  # endpoint max


def _clean_ticker(sym):
    """AAPL.US -> AAPL; strip whitespace; return '' if falsy."""
    if not sym:
        return ""
    return str(sym).strip().split(".")[0].upper()


def _norm_type(t):
    """purchase/buy -> buy, sale/sell -> sell, else exchange."""
    t = (t or "").strip().lower()
    if t in ("purchase", "buy", "p"):
        return "buy"
    if t in ("sale", "sell", "s", "sale_full", "sale_partial"):
        return "sell"
    return "exchange"


def _parse_amount_range(s):
    """'$1,001 - $15,000' -> (1001, 15000). Best-effort; returns (0, 0) on failure."""
    if not s:
        return 0, 0
    import re
    nums = re.findall(r"[\d,]+", str(s))
    ints = []
    for n in nums:
        n = n.replace(",", "").strip()
        if n.isdigit():
            ints.append(int(n))
    if not ints:
        return 0, 0
    if len(ints) == 1:
        return ints[0], ints[0]
    return ints[0], ints[1]


def _to_int(v):
    try:
        if v is None or v == "":
            return None
        return int(float(str(v).replace(",", "").replace("$", "").strip()))
    except Exception:
        return None


def _parse_row(rec):
    """Map one nested API record -> dict of CongressTrade field values.
    Returns None if there is no ticker."""
    member = rec.get("member") or {}
    asset = rec.get("asset") or {}
    txn = rec.get("transaction") or {}
    chamber = rec.get("chamber")

    ticker = _clean_ticker(asset.get("symbol"))
    if not ticker:
        return None

    # member name: prefer full_name, else first+last
    name = (member.get("full_name") or "").strip()
    if not name:
        name = " ".join(x for x in [member.get("first_name"), member.get("last_name")] if x).strip()

    # amount: prefer explicit low/high, fall back to parsing the range string
    amt_min = _to_int(txn.get("amount_low"))
    amt_max = _to_int(txn.get("amount_high"))
    if amt_min is None or amt_max is None:
        r_min, r_max = _parse_amount_range(txn.get("amount_range"))
        if amt_min is None:
            amt_min = r_min
        if amt_max is None:
            amt_max = r_max

    return {
        "ticker": ticker,
        "member": name[:255],
        "chamber": (str(chamber).strip().capitalize() if chamber else "")[:32],
        "party": (member.get("party") or "")[:64],
        "state": (member.get("state") or "")[:64],
        "transaction_type": _norm_type(txn.get("type")),
        "transaction_date": (txn.get("transaction_date") or None),
        "report_date": (txn.get("disclosure_date") or txn.get("report_date") or None),
        "amount_min": amt_min or 0,
        "amount_max": amt_max or 0,
        "owner": (txn.get("owner") or "")[:64],
        "asset_type": (asset.get("asset_type") or "")[:64],
    }


def _make_uid(r):
    key = "{member}|{ticker}|{transaction_date}|{transaction_type}|{amount_min}|{amount_max}".format(**r)
    return hashlib.md5(key.encode()).hexdigest()


def _fetch_window(frm, to=None, max_pages=2000):
    """Page through the endpoint for the [frm, to] disclosure window.
    Yields parsed row dicts (ticker-only). Handles both a bare-list response
    and the documented {'data': [...], 'links': {'next': ...}} envelope."""
    offset = 0
    pages = 0
    while pages < max_pages:
        params = {
            "disclosure_date_from": frm,
            "page[limit]": PAGE_LIMIT,
            "page[offset]": offset,
        }
        if to:
            params["disclosure_date_to"] = to
        resp = _eodhd_get(PATH, **params)
        if resp is None:
            break
        if isinstance(resp, dict):
            records = resp.get("data") or []
        elif isinstance(resp, list):
            records = resp
        else:
            records = []
        if not records:
            break
        for rec in records:
            yield rec
        pages += 1
        if len(records) < PAGE_LIMIT:
            break
        offset += PAGE_LIMIT


def import_congress(years=5):
    """Import the trailing `years`-year disclosure window (default 5y = HARD RULE).
    Idempotent via bulk_create(ignore_conflicts=True) on the unique `uid`."""
    frm = (date.today() - timedelta(days=int(round(years * 365.25)))).isoformat()
    to = date.today().isoformat()

    seen = set()
    objs = []
    for rec in _fetch_window(frm, to):
        r = _parse_row(rec)
        if not r:
            continue
        uid = _make_uid(r)
        if uid in seen:
            continue
        seen.add(uid)
        objs.append(CongressTrade(uid=uid, **r))

    saved = 0
    if objs:
        before = CongressTrade.objects.count()
        CongressTrade.objects.bulk_create(objs, ignore_conflicts=True, batch_size=1000)
        saved = CongressTrade.objects.count() - before
    return {"saved": saved, "fetched": len(objs)}


def probe():
    """Fetch a recent 30-day slice, print raw field keys + 5 sample parsed rows. Save nothing."""
    frm = (date.today() - timedelta(days=30)).isoformat()
    to = date.today().isoformat()
    raw = []
    for rec in _fetch_window(frm, to, max_pages=1):
        raw.append(rec)
        if len(raw) >= 20:
            break

    print("=== PROBE: disclosure window {} .. {} ===".format(frm, to))
    print("raw records fetched (first page):", len(raw))
    if not raw:
        print("NO DATA RETURNED. Endpoint may be unavailable on this account (empty/403).")
        return

    first = raw[0]
    print("\n--- top-level keys:", sorted(first.keys()))
    for k in ("member", "asset", "transaction", "source"):
        v = first.get(k)
        if isinstance(v, dict):
            print("  {}: keys = {}".format(k, sorted(v.keys())))
    print("\n--- raw first record:")
    import json
    print(json.dumps(first, indent=2, default=str)[:2000])

    print("\n--- 5 sample PARSED rows:")
    n = 0
    for rec in raw:
        r = _parse_row(rec)
        if not r:
            continue
        r["uid"] = _make_uid(r)
        print(json.dumps(r, indent=2, default=str))
        n += 1
        if n >= 5:
            break
    if n == 0:
        print("(no records had a ticker/symbol)")


def main():
    ap = argparse.ArgumentParser(description="EODHD Congressional-Trades importer")
    ap.add_argument("--probe", action="store_true", help="fetch recent slice, print fields+samples, save nothing")
    ap.add_argument("--run", action="store_true", help="import the 5-year window")
    args = ap.parse_args()

    if args.run:
        print(import_congress(years=5))
    else:
        probe()


if __name__ == "__main__":
    main()
