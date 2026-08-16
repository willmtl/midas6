#!/usr/bin/env python3
"""DELISTED CANDLE FETCHER — survivorship-bias repair (step 1 of the de-bias). Our stock universe = present-day
ETF holdings, so companies that died aren't candidates — small-cap backtests are optimistically biased. This
pulls EODHD daily history for major-exchange US common-stock DELISTED names (DelistedCompany) and stores the ones
that actually TRADED IN OUR WINDOW (>= MIN_BARS bars on/after 2020-01-01), so later steps can include them in the
candidate universe during their alive-window (candle span; last real bar ≈ delisting since delisted_date is NULL).

Non-destructive (new tickers only). Threaded fetch, resume via --offset/--limit. Adjusted like the live candles
(close=adjusted_close, OHLC scaled). -> core.Candle.
CLI: python -u fetch_delisted_candles.py [--run] [--limit N] [--offset N] [--jobs 16]
Run: MSYS_NO_PATHCONV=1 docker exec rotation-backend-1 python -u /app/fetch_delisted_candles.py --run --jobs 16
"""
import os, argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
os.environ.setdefault("DJANGO_SETTINGS_MODULE", "rotation.settings")
import django  # noqa: E402
django.setup()

import datetime as dt  # noqa: E402
from api.tasks import _eodhd_get  # noqa: E402
from core.models import DelistedCompany, Candle  # noqa: E402

FROM = "2019-06-01"
MIN_BARS_INWINDOW = 40                      # >= 40 daily bars on/after 2020-01-01 to be relevant
MAJOR = ["NYSE", "NASDAQ", "NYSE MKT", "NYSE ARCA", "AMEX", "BATS"]


def _rows(tk, sym):
    resp = _eodhd_get(f"eod/{sym}", **{"from": FROM, "period": "d"})
    if not isinstance(resp, list) or not resp:
        return []
    out = []
    for r in resp:
        d = r.get("date"); cl = r.get("close"); adj = r.get("adjusted_close", cl)
        if not d or cl in (None, "") or adj in (None, ""):
            continue
        try:
            cl = float(cl); adj = float(adj); fac = adj / cl if cl else 1.0
            o = float(r.get("open") or cl) * fac; h = float(r.get("high") or cl) * fac
            lo = float(r.get("low") or cl) * fac; vol = int(float(r.get("volume") or 0))
        except (TypeError, ValueError):
            continue
        if adj <= 0:
            continue
        out.append(Candle(ticker=tk, date=d, interval="1d", open=o, high=h, low=lo, close=adj, volume=vol))
    return out


def targets(offset, limit):
    have = set(Candle.objects.filter(interval="1d").values_list("ticker", flat=True).distinct())
    qs = (DelistedCompany.objects.filter(country="USA", type="Common Stock", exchange__in=MAJOR)
          .order_by("ticker").values_list("ticker", "eodhd_symbol"))
    rows = [(t, s) for t, s in qs if t not in have]
    if offset:
        rows = rows[offset:]
    if limit:
        rows = rows[:limit]
    return rows


def run(offset=0, limit=None, jobs=16):
    tks = targets(offset, limit)
    print(f"delisted fetch: {len(tks)} names (offset {offset}, jobs {jobs})", flush=True)
    kept = tried = empty = thin = 0
    results = {}
    with ThreadPoolExecutor(max_workers=jobs) as ex:
        futs = {ex.submit(_rows, t, s): t for t, s in tks}
        for f in as_completed(futs):
            t = futs[f]
            try:
                results[t] = f.result()
            except Exception:
                results[t] = []
    for t, _ in tks:
        tried += 1
        objs = results.get(t) or []
        if not objs:
            empty += 1; continue
        inwin = [o for o in objs if o.date >= "2020-01-01"]
        if len(inwin) < MIN_BARS_INWINDOW:
            thin += 1; continue
        Candle.objects.filter(ticker=t, interval="1d").delete()
        Candle.objects.bulk_create(objs, ignore_conflicts=True, batch_size=5000)
        kept += 1
        if kept % 200 == 0:
            print(f"  ...{kept} kept / {tried} tried", flush=True)
    print(f"DONE: kept {kept} (traded in-window), skipped {empty} no-data + {thin} thin/pre-window, {tried} tried", flush=True)
    return kept


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", action="store_true")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--offset", type=int, default=0)
    ap.add_argument("--jobs", type=int, default=16)
    a = ap.parse_args()
    if not os.environ.get("EODHD_API_KEY"):
        print("EODHD_API_KEY not set", flush=True); return
    if not a.run:
        tks = targets(0, 5)
        print("probe targets:", tks, flush=True)
        for t, s in tks[:3]:
            r = _rows(t, s); iw = [o for o in r if o.date >= "2020-01-01"]
            print(f"  {t} ({s}): {len(r)} bars, {len(iw)} in-window" + (f", last {r[-1].date}" if r else ""), flush=True)
        return
    run(offset=a.offset, limit=a.limit, jobs=a.jobs)


if __name__ == "__main__":
    main()
