#!/usr/bin/env python3
"""Standalone EODHD Corporate Actions importer -> core.models.CorporateAction.

Pulls SPLITS (/api/splits/{SYMBOL}) and DIVIDENDS (/api/div/{SYMBOL}) from EODHD
over a HARD 5-year window and upserts them idempotently. Reuses the project's
authenticated EODHD helpers (_eodhd_get / _eodhd_sym) from api.tasks.

Runs with CWD=/app inside rotation-backend-1 (backend/ is the app root on sys.path).

Run:
  docker exec rotation-backend-1 python -u /app/fetch_corp_actions.py --probe
  docker exec rotation-backend-1 python -u /app/fetch_corp_actions.py --limit 20
"""
import warnings; warnings.filterwarnings("ignore")
import os, sys, json, argparse
from datetime import date, timedelta

os.environ.setdefault("DJANGO_SETTINGS_MODULE", "rotation.settings")
import django
django.setup()

from api.tasks import _eodhd_get, _eodhd_sym


def _from_date(years=5):
    return (date.today() - timedelta(days=365 * years)).isoformat()


def _first(d, *keys):
    """Return the first present, non-empty value among keys (handles field-name variants)."""
    for k in keys:
        if k in d and d[k] not in (None, "", "0000-00-00"):
            return d[k]
    return None


def _num(v):
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def parse_split_row(row):
    """{"date","split":"4.000000/1.000000"} -> parsed defaults dict or None."""
    ex_date = _first(row, "date")
    raw = _first(row, "split")
    if not ex_date or not raw or "/" not in str(raw):
        return None
    try:
        to_s, from_s = str(raw).split("/", 1)
        split_to = float(to_s)
        split_from = float(from_s)
    except (ValueError, TypeError):
        return None
    if not split_from:
        return None
    return {
        "ex_date": ex_date,
        "defaults": {
            "split_to": split_to,
            "split_from": split_from,
            "split_ratio": split_to / split_from,
            "dividend": None,
            "currency": "",
            "record_date": None,
            "pay_date": None,
            "declaration_date": None,
            "period": "",
        },
    }


def parse_div_row(row):
    """EODHD div row -> parsed defaults dict or None. Handles field-name variants."""
    ex_date = _first(row, "date")
    val = _num(_first(row, "value", "dividend"))
    if not ex_date or val is None:
        return None
    return {
        "ex_date": ex_date,
        "defaults": {
            "dividend": val,
            "currency": _first(row, "currency") or "",
            "record_date": _first(row, "recordDate", "record_date"),
            "pay_date": _first(row, "paymentDate", "payment_date", "pay_date"),
            "declaration_date": _first(row, "declarationDate", "declaration_date"),
            "period": _first(row, "period") or "",
            "split_ratio": None,
            "split_from": None,
            "split_to": None,
        },
    }


def import_corp_actions(tickers=None, years=5):
    from core.models import CorporateAction
    from seq_fundamental_study import build_universe

    universe = tickers if tickers is not None else build_universe()
    frm = _from_date(years)
    print(f"Corp actions import | {len(universe)} tickers | from {frm}")

    n_new = 0
    n_splits = 0
    n_divs = 0
    for i, tk in enumerate(universe, 1):
        sym = _eodhd_sym(tk)
        if sym is None:
            continue

        splits = _eodhd_get(f"splits/{sym}", **{"from": frm}) or []
        for row in splits if isinstance(splits, list) else []:
            parsed = parse_split_row(row)
            if not parsed:
                continue
            _, created = CorporateAction.objects.update_or_create(
                ticker=tk, action_type="split", ex_date=parsed["ex_date"],
                defaults=parsed["defaults"])
            n_splits += 1
            n_new += int(created)

        divs = _eodhd_get(f"div/{sym}", **{"from": frm}) or []
        for row in divs if isinstance(divs, list) else []:
            parsed = parse_div_row(row)
            if not parsed:
                continue
            _, created = CorporateAction.objects.update_or_create(
                ticker=tk, action_type="dividend", ex_date=parsed["ex_date"],
                defaults=parsed["defaults"])
            n_divs += 1
            n_new += int(created)

        if i % 100 == 0:
            print(f"  ...{i}/{len(universe)} tickers | new={n_new} splits={n_splits} divs={n_divs}")

    print(f"Done: saved(new)={n_new} splits={n_splits} dividends={n_divs}")
    return {"saved": n_new, "splits": n_splits, "dividends": n_divs}


def probe():
    frm = _from_date(5)
    for tk in ("AAPL", "MSFT", "SHOP.TO"):
        sym = _eodhd_sym(tk)
        print(f"\n===== {tk}  ->  sym={sym}  (from={frm}) =====")
        if sym is None:
            print("  _eodhd_sym returned None -> skipped")
            continue

        splits = _eodhd_get(f"splits/{sym}", **{"from": frm}) or []
        print(f"\n  -- SPLITS raw (n={len(splits) if isinstance(splits, list) else '?'}) --")
        print("  ", json.dumps(splits[:3] if isinstance(splits, list) else splits, indent=2, default=str))
        print("  -- SPLITS parsed --")
        for row in (splits[:3] if isinstance(splits, list) else []):
            print("  ", parse_split_row(row))

        divs = _eodhd_get(f"div/{sym}", **{"from": frm}) or []
        print(f"\n  -- DIVIDENDS raw (n={len(divs) if isinstance(divs, list) else '?'}) --")
        print("  ", json.dumps(divs[:3] if isinstance(divs, list) else divs, indent=2, default=str))
        print("  -- DIVIDENDS parsed --")
        for row in (divs[:3] if isinstance(divs, list) else []):
            print("  ", parse_div_row(row))


def main():
    ap = argparse.ArgumentParser(description="EODHD corporate actions importer")
    ap.add_argument("--limit", type=int, default=None, help="import only first N tickers")
    ap.add_argument("--probe", action="store_true",
                    help="fetch AAPL+MSFT+SHOP.TO, print raw+parsed shape, save nothing")
    args = ap.parse_args()

    if args.probe:
        probe()
        return

    from seq_fundamental_study import build_universe
    universe = build_universe()
    if args.limit is not None:
        universe = universe[:args.limit]
    result = import_corp_actions(tickers=universe)
    print(json.dumps(result))


if __name__ == "__main__":
    main()
