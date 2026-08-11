#!/usr/bin/env python
"""EODHD Historical Market Cap + ID-mapping (CUSIP/CIK/ISIN) + Screener importer.

Standalone root module (mounted at /app in rotation-backend-1). Reuses the EODHD helpers
`_eodhd_get` / `_eodhd_sym` from api.tasks and the study universe from seq_fundamental_study.

Three independent parts:
  * import_market_cap  -> core.MarketCapHistory  (5-YEAR window, HARD RULE; idempotent)
  * import_id_mapping  -> updates latest core.Fundamental row's cusip/cik/isin per ticker
  * screen             -> raw EODHD screener list (discovery utility, no persistence)

CLI:
  python -u fetch_market_cap.py --probe            # AAPL + SHOP.TO sample, save nothing (default)
  python -u fetch_market_cap.py --mcap --limit 15
  python -u fetch_market_cap.py --ids  --limit 15
  python -u fetch_market_cap.py --screen
"""
import os
import argparse
import datetime as dt

os.environ.setdefault("DJANGO_SETTINGS_MODULE", "rotation.settings")
import django  # noqa: E402
django.setup()

from api.tasks import _eodhd_get, _eodhd_sym  # noqa: E402
from core.models import MarketCapHistory, Fundamental  # noqa: E402
from seq_fundamental_study import build_universe  # noqa: E402


# ---------------------------------------------------------------------------
# response parsing helpers
# ---------------------------------------------------------------------------
def _iter_rows(resp):
    """EODHD historical-market-cap returns a dict keyed by numeric string indices
    ({"0": {"date":..,"value":..}, ...}); tolerate a plain list or a date-keyed dict too."""
    if not resp:
        return []
    if isinstance(resp, list):
        return [r for r in resp if isinstance(r, dict)]
    if isinstance(resp, dict):
        out = []
        for k, v in resp.items():
            if isinstance(v, dict):
                # date-keyed dict: value may omit its own "date"
                v = dict(v)
                v.setdefault("date", k)
                out.append(v)
        return out
    return []


def _mcap_value(row):
    """Market cap field: EODHD uses `value`; docs elsewhere say `MarketCapitalization`."""
    for k in ("value", "MarketCapitalization", "marketCapitalization", "market_cap"):
        if row.get(k) not in (None, "", 0, "0"):
            return row[k]
    # last resort: 0/None-ish present under value
    for k in ("value", "MarketCapitalization"):
        if k in row:
            return row[k]
    return None


# ---------------------------------------------------------------------------
# 1) historical market cap
# ---------------------------------------------------------------------------
def import_market_cap(tickers=None, years=5):
    """Import 5y of weekly market-cap history -> MarketCapHistory. Idempotent."""
    tickers = tickers or build_universe()
    frm = (dt.date.today() - dt.timedelta(days=int(years * 365.25))).isoformat()
    total_rows = 0
    done = 0
    for tk in tickers:
        sym = _eodhd_sym(tk)
        if sym is None:
            continue
        resp = _eodhd_get(f"historical-market-cap/{sym}", **{"from": frm})
        rows = _iter_rows(resp)
        objs = []
        for r in rows:
            d = r.get("date")
            mc = _mcap_value(r)
            if not d or mc in (None, ""):
                continue
            try:
                mc_int = int(round(float(mc)))
            except (TypeError, ValueError):
                continue
            if mc_int <= 0:
                continue
            objs.append(MarketCapHistory(ticker=tk, date=d, market_cap=mc_int))
        if objs:
            MarketCapHistory.objects.bulk_create(objs, ignore_conflicts=True)
            total_rows += len(objs)
        done += 1
        if done % 100 == 0:
            print(f"  ...market-cap {done}/{len(tickers)} tickers, {total_rows} rows so far",
                  flush=True)
    print(f"import_market_cap: {done} tickers processed, {total_rows} rows upserted", flush=True)
    return total_rows


# ---------------------------------------------------------------------------
# 2) id-mapping (CUSIP / CIK / ISIN) onto latest Fundamental row
# ---------------------------------------------------------------------------
def _general_block(sym):
    g = _eodhd_get(f"fundamentals/{sym}", filter="General")
    if isinstance(g, dict) and (g.get("CUSIP") or g.get("CIK") or g.get("ISIN")):
        return g
    # fallback: full payload, dig out General
    full = _eodhd_get(f"fundamentals/{sym}")
    if isinstance(full, dict):
        blk = full.get("General")
        if isinstance(blk, dict):
            return blk
        return full  # some feeds inline the fields
    return g if isinstance(g, dict) else None


def import_id_mapping(tickers=None):
    """Update the most-recent Fundamental row per ticker with cusip/cik/isin (only when present)."""
    tickers = tickers or build_universe()
    updated = 0
    done = 0
    for tk in tickers:
        sym = _eodhd_sym(tk)
        if sym is None:
            continue
        row = Fundamental.objects.filter(ticker=tk).order_by("-date").first()
        if row is None:
            continue
        g = _general_block(sym)
        if not isinstance(g, dict):
            done += 1
            continue
        changed = False
        for src, field, cap in (("CUSIP", "cusip", 12), ("CIK", "cik", 12), ("ISIN", "isin", 16)):
            val = g.get(src)
            if val in (None, "", "null"):
                continue
            val = str(val).strip()[:cap]
            if val and getattr(row, field) != val:
                setattr(row, field, val)
                changed = True
        if changed:
            row.save(update_fields=["cusip", "cik", "isin"])
            updated += 1
        done += 1
        if done % 100 == 0:
            print(f"  ...id-mapping {done}/{len(tickers)} tickers, {updated} updated so far",
                  flush=True)
    print(f"import_id_mapping: {done} tickers processed, {updated} Fundamental rows updated",
          flush=True)
    return updated


# ---------------------------------------------------------------------------
# 3) screener (discovery utility, no persistence)
# ---------------------------------------------------------------------------
def screen(filters=None, sort="market_capitalization.desc", limit=50):
    """Return the raw EODHD screener list. `filters` is a python list-of-lists, JSON-encoded."""
    import json
    if filters is None:
        filters = [["market_capitalization", ">", 1000000000]]
    params = {"sort": sort, "limit": int(limit)}
    if filters is not None:
        params["filters"] = json.dumps(filters)
    resp = _eodhd_get("screener", **params)
    if isinstance(resp, dict):
        return resp.get("data", resp)
    return resp or []


# ---------------------------------------------------------------------------
# probe
# ---------------------------------------------------------------------------
def _probe():
    frm = (dt.date.today() - dt.timedelta(days=int(5 * 365.25))).isoformat()
    for tk in ("AAPL", "SHOP.TO"):
        sym = _eodhd_sym(tk)
        print(f"\n=== {tk}  (EODHD sym={sym}) ===", flush=True)
        resp = _eodhd_get(f"historical-market-cap/{sym}", **{"from": frm})
        rows = _iter_rows(resp)
        print(f"  market-cap rows: {len(rows)}", flush=True)
        for r in rows[:3]:
            print(f"    sample: date={r.get('date')!r} value={_mcap_value(r)!r} keys={list(r.keys())}",
                  flush=True)
        if rows:
            last = rows[-1]
            print(f"    latest: date={last.get('date')!r} value={_mcap_value(last)!r}", flush=True)
        g = _general_block(sym)
        if isinstance(g, dict):
            print(f"  General.CUSIP={g.get('CUSIP')!r}  General.CIK={g.get('CIK')!r}  "
                  f"General.ISIN={g.get('ISIN')!r}  PrimaryTicker={g.get('PrimaryTicker')!r}",
                  flush=True)
        else:
            print("  General block: not available", flush=True)

    print("\n=== screener sample (market_cap > $1B, top 5 by mcap desc) ===", flush=True)
    data = screen(limit=5)
    print(f"  rows: {len(data) if hasattr(data, '__len__') else 'n/a'}", flush=True)
    if isinstance(data, list):
        for row in data[:5]:
            if isinstance(row, dict):
                print(f"    {row.get('code')!r} mcap={row.get('market_capitalization')!r} "
                      f"name={row.get('name')!r}", flush=True)
            else:
                print(f"    {row!r}", flush=True)
    else:
        print(f"    raw: {str(data)[:300]}", flush=True)


def main():
    ap = argparse.ArgumentParser(description="EODHD market-cap + id-mapping + screener importer")
    ap.add_argument("--probe", action="store_true", help="AAPL + SHOP.TO sample; save nothing")
    ap.add_argument("--mcap", action="store_true", help="import historical market cap")
    ap.add_argument("--ids", action="store_true", help="import CUSIP/CIK/ISIN onto Fundamentals")
    ap.add_argument("--screen", action="store_true", help="print a screener sample")
    ap.add_argument("--limit", type=int, default=None, help="cap number of tickers processed")
    args = ap.parse_args()

    if not (args.mcap or args.ids or args.screen or args.probe):
        args.probe = True

    tickers = None
    if args.limit and (args.mcap or args.ids):
        tickers = build_universe()[: args.limit]
        print(f"limiting to {len(tickers)} tickers", flush=True)

    if args.probe:
        _probe()
    if args.mcap:
        import_market_cap(tickers=tickers)
    if args.ids:
        import_id_mapping(tickers=tickers)
    if args.screen:
        data = screen(limit=args.limit or 20)
        print(f"screener returned {len(data) if hasattr(data, '__len__') else 'n/a'} rows", flush=True)
        for row in (data or [])[:20]:
            print(f"  {row}", flush=True)


if __name__ == "__main__":
    main()
