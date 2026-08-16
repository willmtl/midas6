#!/usr/bin/env python3
"""ETF FUND FLOWS from Polygon (the only provider that has it — EODHD outstandingShares='NA', yfinance
get_shares_full empty for ETFs). Polygon's dated ticker-details endpoint returns
`share_class_shares_outstanding` that updates DAILY with creations/redemptions, so:

  flow_shares_t = shares_out_t - shares_out_{prev trading day}   (creation units, +in / -out)
  flow_usd_t    = flow_shares_t * close_t                        (dollar flow; close = DB adj close)
  flow_pct_t    = flow_shares_t / shares_out_{prev}              (price-independent, the clean signal)

Backfills to core.ETFFlow, HARD RULE full 5y history, idempotent (recomputes diffs from the stored share
series each run). Trading calendar = the DB SPY candle dates. Robust getter (retry+backoff) because Polygon
resets long sequential connections.

CLI:
  python -u fetch_etf_flows.py --probe                 # 4 ETFs x ~20 days, save nothing (default)
  python -u fetch_etf_flows.py --run --jobs 8          # full 5y, all 91 sector ETFs
  python -u fetch_etf_flows.py --run --limit 10 --days 260
Run: MSYS_NO_PATHCONV=1 docker exec rotation-backend-1 python -u /app/fetch_etf_flows.py --run --jobs 8
"""
import os, time, json, argparse, urllib.request, urllib.parse
from concurrent.futures import ThreadPoolExecutor, as_completed

os.environ.setdefault("DJANGO_SETTINGS_MODULE", "rotation.settings")
import django  # noqa: E402
django.setup()

import datetime as dt  # noqa: E402
import config  # noqa: E402
from trend_stock_studies import CRYPTO  # noqa: E402
from seq_fundamental_study import load_candles  # noqa: E402
from core.models import ETFFlow  # noqa: E402

KEY = os.environ.get("POLYGON_API_KEY")
BASE = "https://api.polygon.io/v3/reference/tickers/"


def _shares(tk, date, tries=5):
    """share_class_shares_outstanding for `tk` as of `date` (YYYY-MM-DD). None if unavailable."""
    if not KEY:
        return None
    url = f"{BASE}{tk}?" + urllib.parse.urlencode({"date": date, "apiKey": KEY})
    for i in range(tries):
        try:
            with urllib.request.urlopen(url, timeout=30) as r:
                j = json.loads(r.read())
            res = j.get("results") or {}
            return res.get("share_class_shares_outstanding")
        except Exception:
            time.sleep(0.4 * (i + 1))
    return None


def trading_dates(years=5):
    """Business days we actually have SPY prices for (our price calendar)."""
    spy = load_candles(["SPY"]).get("SPY")
    if spy is None or spy.empty:
        # fallback: pandas business days
        import pandas as pd
        end = dt.date.today()
        return [d.date() for d in pd.bdate_range(end - dt.timedelta(days=int(years * 365.25)), end)]
    frm = spy.index.max().normalize() - __import__("pandas").Timedelta(days=int(years * 365.25))
    return [d.date() for d in spy.index if d >= frm]


def month_end_dates(years=5):
    """Last trading day of each month over the window (the study aggregates flow to month-end anyway,
    so this is ~60 dates/ticker instead of ~1255 — 20x fewer Polygon calls, same monthly signal)."""
    import pandas as pd
    ds = trading_dates(years)
    s = pd.Series(1, index=pd.DatetimeIndex(ds))
    return [ts.date() for ts in s.groupby([s.index.year, s.index.month]).apply(lambda g: g.index.max())]


def etf_universe():
    return [e for e in config.SECTOR_ETFS.values() if e not in CRYPTO and "-" not in e]


def _close_maps(tickers):
    """ticker -> {date: adj close} from DB candles (for flow_usd)."""
    cds = load_candles(tickers)
    out = {}
    for t in tickers:
        d = cds.get(t)
        if d is None or "Close" not in d:
            out[t] = {}
            continue
        out[t] = {ix.date(): float(c) for ix, c in d["Close"].items() if c and c == c}
    return out


def backfill(tickers=None, dates=None, jobs=8, save=True, skip_min=None, full_replace=True):
    tickers = tickers or etf_universe()
    dates = dates or trading_dates()
    dstr = [d.isoformat() for d in dates]
    closes = _close_maps(tickers) if save else {}
    if skip_min:
        have = {r["ticker"]: r["n"] for r in
                ETFFlow.objects.filter(ticker__in=tickers).values("ticker").annotate(n=__import__("django").db.models.Count("*"))}
        done0 = [t for t in tickers if have.get(t, 0) >= skip_min]
        tickers = [t for t in tickers if have.get(t, 0) < skip_min]
        print(f"resume: skipping {len(done0)} already-populated tickers, fetching {len(tickers)}", flush=True)
    print(f"ETF flows: {len(tickers)} ETFs x {len(dstr)} days = {len(tickers) * len(dstr):,} Polygon calls "
          f"(jobs={jobs})", flush=True)
    grand = 0
    for ti, tk in enumerate(tickers, 1):
        shares = {}
        with ThreadPoolExecutor(max_workers=jobs) as ex:
            futs = {ex.submit(_shares, tk, d): d for d in dstr}
            for f in as_completed(futs):
                d = futs[f]
                s = f.result()
                if s:
                    shares[d] = int(s)
        if not shares:
            print(f"  [{ti}/{len(tickers)}] {tk}: no shares data (skipped)", flush=True)
            continue
        # order by date, compute diffs vs prior STORED trading day
        ser = sorted(shares.items())          # [(date_str, shares), ...]
        cmap = closes.get(tk, {})
        objs = []
        prev = None
        for dstr_i, sh in ser:
            dd = dt.date.fromisoformat(dstr_i)
            fsh = fusd = fpct = None
            if prev is not None:
                fsh = sh - prev
                if prev:
                    fpct = fsh / prev
                cl = cmap.get(dd)
                if cl is not None:
                    fusd = fsh * cl
            objs.append(ETFFlow(ticker=tk, date=dd, shares_out=sh, close=cmap.get(dd),
                                flow_shares=fsh, flow_usd=fusd, flow_pct=fpct))
            prev = sh
        if save:
            if full_replace:
                ETFFlow.objects.filter(ticker=tk).delete()               # full-series refresh
            else:
                ETFFlow.objects.filter(ticker=tk, date__in=[o.date for o in objs]).delete()  # --days window
            ETFFlow.objects.bulk_create(objs, ignore_conflicts=True, batch_size=2000)
        grand += len(objs)
        avg_pct = sum(abs(o.flow_pct) for o in objs if o.flow_pct is not None) / max(
            sum(1 for o in objs if o.flow_pct is not None), 1)
        print(f"  [{ti}/{len(tickers)}] {tk}: {len(objs)} days, avg |flow| {avg_pct * 100:.2f}%/day, "
              f"latest shares {ser[-1][1]:,}", flush=True)
    print(f"ETF flows backfill: {grand:,} rows across {len(tickers)} ETFs", flush=True)
    return grand


def _probe():
    dates = trading_dates()[-20:]
    print(f"probe: 4 ETFs x last {len(dates)} trading days (XLK, USO, GLD, SMH); saving nothing\n", flush=True)
    for tk in ("XLK", "USO", "GLD", "SMH"):
        vals = []
        for d in dates:
            s = _shares(tk, d.isoformat())
            vals.append((d, s))
            time.sleep(0.15)
        got = [v for _, v in vals if v]
        print(f"=== {tk}: {len(got)}/{len(dates)} days returned shares ===", flush=True)
        prev = None
        for d, s in vals:
            chg = "" if (prev is None or s is None) else f"  Δ={s - prev:+,} ({(s - prev) / prev * 100:+.2f}%)"
            print(f"  {d}: {s}{chg}", flush=True)
            if s:
                prev = s


def main():
    ap = argparse.ArgumentParser(description="Polygon ETF fund-flow backfill -> core.ETFFlow")
    ap.add_argument("--probe", action="store_true", help="sample a few ETFs, save nothing (default)")
    ap.add_argument("--run", action="store_true", help="full backfill -> ETFFlow")
    ap.add_argument("--jobs", type=int, default=8, help="thread pool per ticker (Polygon calls)")
    ap.add_argument("--limit", type=int, default=None, help="cap number of ETFs")
    ap.add_argument("--days", type=int, default=None, help="only the most recent N trading days")
    ap.add_argument("--monthly", action="store_true", help="month-end dates only (~20x fewer calls; same monthly signal)")
    args = ap.parse_args()
    if not (args.run or args.probe):
        args.probe = True
    if not KEY:
        print("POLYGON_API_KEY not set — aborting", flush=True)
        return
    if args.probe:
        _probe()
    if args.run:
        tks = etf_universe()
        if args.limit:
            tks = tks[: args.limit]
        ds = month_end_dates() if args.monthly else trading_dates()
        if args.days:
            ds = ds[-args.days:]
        backfill(tickers=tks, dates=ds, jobs=args.jobs, save=True,
                 skip_min=(len(ds) - 2 if args.monthly else None), full_replace=(not args.days))


if __name__ == "__main__":
    main()
