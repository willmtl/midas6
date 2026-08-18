# -*- coding: utf-8 -*-
"""BACKFILL BENZINGA ANALYST RATINGS (via Polygon) -> dated JSONL archive.

The paid Benzinga Analyst Ratings feed (newly enabled on our Polygon key) returns a STRUCTURED,
DATED, HISTORICAL record per rating action — unlike EODHD's snapshot-only estimates. Fields per row:
  date, firm, analyst, rating / previous_rating, rating_action (upgrades/downgrades/initiates/maintains),
  price_target / previous_price_target (+ adjusted), price_target_action, importance.
Archive depth observed: back to ~2022-08 (covers most of the flagship window; the free headline-scraped
targets in news_price_target.py fill 2020-08 -> 2022).

This is a ONE-TIME archive dump — pages /benzinga/v1/ratings per universe ticker (following next_url)
and appends every record to .data/analyst_ratings.jsonl. Idempotent per ticker: a ticker already present
in the file is skipped (pass --refresh to re-pull all). No Date.now used anywhere.

Run detached in the celery worker (Polygon egress):
  MSYS_NO_PATHCONV=1 docker exec rotation-celery-worker-1 sh -c \
    'setsid nohup python -u /app/backfill_analyst_ratings.py --workers 10 \
     > /app/.data/analyst_ratings_backfill.log 2>&1 < /dev/null &'
"""
import os, sys, json, time
os.environ.setdefault("DJANGO_SETTINGS_MODULE", "rotation.settings")
import django; django.setup()
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
from api.tasks import _polygon_paginate


def analyst_universe():
    """Stocks that have fundamentals, minus ETFs/benchmarks, UNION the Finviz US/CA universe (so the Finviz
    industry engine's analyst-upside blend has coverage too). Avoids build_universe()'s DISTINCT over the
    Candle hypertable (which hangs on the worker without max_parallel_workers_per_gather=0)."""
    from core.models import Fundamental, Sector
    etfs = set(Sector.objects.values_list("etf", flat=True)) | {"SPY", "QQQ"}
    have_fund = set(Fundamental.objects.values_list("ticker", flat=True))
    fv = set()
    try:                                             # add Finviz US/CA names (Benzinga is US-listed)
        import json as _j
        _d = _j.load(open("/app/.data/finviz_universe.json"))
        fv = {t for t, m in _d["by_ticker"].items() if m.get("country") in ("USA", "Canada")}
    except Exception:
        pass
    return sorted((have_fund | fv) - etfs)

OUT = Path("/app/.data/analyst_ratings.jsonl")
KEEP = ("date", "time", "firm", "analyst", "rating", "previous_rating", "rating_action",
        "price_target", "previous_price_target", "adjusted_price_target",
        "previous_adjusted_price_target", "price_target_action", "importance", "company_name")


def _bz_sym(tk):
    """Benzinga is US-listed; map our ticker to a plain symbol. Foreign suffixes (.TO/.V/...) are
    dropped — coverage will simply be empty for names Benzinga doesn't carry (we keep what we get)."""
    return tk.split(".")[0].upper() if tk else None


def done_tickers():
    done = set()
    if OUT.exists():
        for line in OUT.read_text(encoding="utf-8").splitlines():
            try:
                done.add(json.loads(line)["ticker"])
            except Exception:
                pass
    return done


def main(workers=8, refresh=False):
    tickers = sorted(set(analyst_universe()))
    done = set() if refresh else done_tickers()
    todo = [t for t in tickers if t not in done]
    print(f"universe {len(tickers)} | already dumped: {len(done)} | to fetch: {len(todo)}", flush=True)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fh = OUT.open("w" if refresh else "a", encoding="utf-8")
    n_tk = n_rows = n_cov = 0
    t0 = time.time()

    def work(tk):
        sym = _bz_sym(tk)
        if not sym:
            return tk, []
        # page the full archive for this ticker (follow next_url); ratings are ~few-hundred max per name
        res = _polygon_paginate("/benzinga/v1/ratings", cap=4000, ticker=sym, limit=1000)
        rows = []
        for r in res or []:
            if not isinstance(r, dict) or not r.get("date"):
                continue
            rec = {"ticker": tk, "sym": sym}
            for k in KEEP:
                if r.get(k) is not None:
                    rec[k] = r[k]
            rows.append(rec)
        return tk, rows

    with ThreadPoolExecutor(max_workers=workers) as ex:
        for tk, rows in ex.map(work, todo):
            n_tk += 1
            if rows:
                n_cov += 1
                for rec in rows:
                    fh.write(json.dumps(rec) + "\n")
                n_rows += len(rows)
            fh.flush()
            if n_tk % 100 == 0:
                rate = n_tk / (time.time() - t0)
                print(f"  {n_tk}/{len(todo)}  {rate:.1f} tk/s  covered={n_cov} ({100*n_cov/n_tk:.0f}%)  "
                      f"rows={n_rows}", flush=True)
    fh.close()
    print(f"DONE {n_tk} tickers in {(time.time()-t0)/60:.1f}m | with-ratings: {n_cov} "
          f"({100*n_cov/max(n_tk,1):.0f}%) | total rows: {n_rows}", flush=True)
    print(f"  -> {OUT}", flush=True)
    print("DONE_BZBACKFILL", flush=True)


if __name__ == "__main__":
    a = sys.argv
    workers = int(a[a.index("--workers") + 1]) if "--workers" in a else 8
    refresh = "--refresh" in a
    main(workers, refresh)
