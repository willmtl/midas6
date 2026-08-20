#!/usr/bin/env python3
"""EXPAND UNIVERSE step 2b: make newly-added ETF constituents actually TRADEABLE by the live flagship.

`expand_holdings.py` (step 2a, weekly) refreshes the current ETF membership into
.data/expanded_holdings.json, but it is read-only vs the DB — it does NOT backfill data. And
`build_universe()` only returns tickers that ALREADY have (candles ∩ fundamentals), so a brand-new
constituent with no data yet is invisible to the candle/fundamental updaters and gets silently GATED
OUT of the scanner. Result: new names never become tradeable. This step closes that loop.

For every current constituent (expanded_holdings.json full_union ∪ sector_holdings.get_holdings()),
diff against what has daily candles AND a point-in-time FinancialReport, and backfill whatever is
missing — full history (candles via fetch_candles_eodhd.backfill; PIT financials via
fetch_delisted_fundamentals._fetch → core.FinancialReport). Idempotent / fetch-only-missing, so after
the first run it is fast. Honors [[always-want-history]] (full backfill, never forward-only).

Run:  MSYS_NO_PATHCONV=1 docker exec rotation-backend-1 python -u /app/ensure_universe_data.py [--dry-run] [--limit N]
"""
import os, sys, json, re, warnings
warnings.filterwarnings("ignore")
os.environ.setdefault("DJANGO_SETTINGS_MODULE", "rotation.settings")
import django
django.setup()

from pathlib import Path
from django.db import connection
import sector_holdings
from core.models import Candle, FinancialReport

# DISTINCT/GROUP BY over the Candle hypertable spawns parallel workers -> DiskFull on the 64MB /dev/shm
# ([[postgres-shm-limit]]). Disable parallel gather for this connection and use point-lookups below.
with connection.cursor() as _cur:
    _cur.execute("SET max_parallel_workers_per_gather = 0")


def _have_candles(universe):
    """Tickers in `universe` that have >=1 daily candle — one indexed point-lookup per ticker (no hypertable
    DISTINCT). Fast: (ticker, date) is the hypertable's leading index."""
    return {t for t in universe if Candle.objects.filter(ticker=t, interval="1d").exists()}

EXPANDED = Path("/app/.data/expanded_holdings.json")
DRY = "--dry-run" in sys.argv
LIMIT = int(sys.argv[sys.argv.index("--limit") + 1]) if "--limit" in sys.argv else None


# A backfill candidate must be a plausible US/US-ADR ticker fetchable as `{ticker}.US`: 1-5 uppercase letters
# with an optional single-letter share class (BRK-B). This drops the noise that leaks into ETF holdings —
# foreign listings (7203.T, 005935, ADANIENT.NS), futures (CLF4/CLX3), and name-string/collateral placeholders
# ("DERIVATIVES COLLATERAL …"). The flagship trades US/CA only, so these are never tradeable regardless.
_US_TICKER = re.compile(r"^[A-Z]{1,5}(-[A-Z])?$")


def _is_us_ticker(t):
    return bool(_US_TICKER.match(t))


def _current_universe():
    """Union of all current equity constituents: expanded_holdings.json full_union + the engine's own
    get_holdings() pool. ETFs/benchmarks excluded (they update via the sector path); non-US-format noise
    dropped (see _is_us_ticker)."""
    tks = set()
    if EXPANDED.exists():
        exp = json.load(open(EXPANDED))
        for sec in exp.values():
            for t in sec.get("full_union", []):
                if t:
                    tks.add(str(t).upper())
    try:
        for _name, meta in sector_holdings.HOLDINGS.items():
            for t in (meta.get("holdings") or []):
                if t:
                    tks.add(str(t).upper())
    except Exception as e:
        print(f"  (get_holdings union skipped: {e})", flush=True)
    etfs = {str(v).upper() for v in getattr(sector_holdings, "ETF_SET", set())}
    return sorted(t for t in (tks - etfs) if _is_us_ticker(t))


def _fund_rows_for(tk):
    """PIT financials for one ticker via the delisted-fundamentals fetcher (filing-dated rows)."""
    from fetch_delisted_fundamentals import _fetch
    try:
        r = _fetch(tk, f"{tk}.US")
    except Exception as e:
        print(f"  {tk}: fundamentals ERR {e}", flush=True)
        return []
    if not r or not r.get("rows"):
        return []
    return [row for row in r["rows"]
            if row.get("total_equity") is not None or row.get("net_income") is not None]


def main():
    universe = _current_universe()
    if LIMIT:
        universe = universe[:LIMIT]
    have_candles = _have_candles(universe)
    have_fund = set(FinancialReport.objects.filter(ticker__in=universe)
                    .values_list("ticker", flat=True).distinct())   # small non-hypertable table — DISTINCT ok
    miss_candles = sorted(set(universe) - have_candles)
    miss_fund = sorted(set(universe) - have_fund)
    print(f"=== ensure_universe_data: {len(universe)} current constituents | "
          f"have candles {len(have_candles)} / fund {len(have_fund)} | "
          f"MISSING candles {len(miss_candles)} / fund {len(miss_fund)} ===", flush=True)
    if miss_candles:
        print(f"  new (no candles): {miss_candles[:40]}{' …' if len(miss_candles) > 40 else ''}", flush=True)
    if miss_fund:
        print(f"  new (no financials): {miss_fund[:40]}{' …' if len(miss_fund) > 40 else ''}", flush=True)
    if DRY:
        print("  --dry-run: no fetch performed.", flush=True)
        return

    # 1) candles — full history, non-destructive (fetch_candles_eodhd handles brand-new tickers)
    if miss_candles:
        from fetch_candles_eodhd import backfill
        print(f"  backfilling candles for {len(miss_candles)} names…", flush=True)
        backfill(miss_candles, jobs=8)

    # 2) PIT financials -> FinancialReport
    saved = names = 0
    for tk in miss_fund:
        good = _fund_rows_for(tk)
        if good:
            FinancialReport.objects.filter(ticker=tk).delete()
            FinancialReport.objects.bulk_create([FinancialReport(**row) for row in good],
                                                ignore_conflicts=True, batch_size=2000)
            saved += len(good); names += 1

    # report residual coverage (point-lookups; no hypertable DISTINCT)
    have_c2 = _have_candles(universe)
    have_f2 = set(FinancialReport.objects.filter(ticker__in=universe).values_list("ticker", flat=True).distinct())
    still = [t for t in universe if t not in have_c2 or t not in have_f2]
    print(f"DONE: financials backfilled for {names} names ({saved} rows). "
          f"Coverage of {len(universe)}: candles {len(have_c2)}, financials {len(have_f2)}. "
          f"Still incomplete: {len(still)}", flush=True)


if __name__ == "__main__":
    main()
