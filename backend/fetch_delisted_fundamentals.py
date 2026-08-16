#!/usr/bin/env python3
"""DELISTED FUNDAMENTALS (survivorship de-bias, step 2) — fetch EODHD quarterly financials for the delisted
names we have candles for, into core.FinancialReport, so they can be eligible P/B picks. Also record each name's
GicSector (from General block) to /app/.data/delisted_gic.json for the GICS-sector re-run. filing_date = PIT
avail_date (fallback period_end + 45d). Non-destructive per (ticker, period_end).
CLI: python -u fetch_delisted_fundamentals.py [--run] [--limit N] [--offset N] [--jobs 8]
Run: MSYS_NO_PATHCONV=1 docker exec rotation-backend-1 python -u /app/fetch_delisted_fundamentals.py --run --jobs 8
"""
import os, json, argparse, datetime as dt
from concurrent.futures import ThreadPoolExecutor, as_completed
os.environ.setdefault("DJANGO_SETTINGS_MODULE", "rotation.settings")
import django  # noqa: E402
django.setup()

from api.tasks import _eodhd_get  # noqa: E402
from core.models import DelistedCompany, Candle, FinancialReport  # noqa: E402

GIC_OUT = "/app/.data/delisted_gic.json"
LAG = dt.timedelta(days=45)


def _num(v):
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def _fetch(tk, sym):
    f = _eodhd_get(f"fundamentals/{sym}")
    if not isinstance(f, dict):
        return None
    gic = (f.get("General") or {}).get("GicSector") or (f.get("General") or {}).get("Sector")
    fin = f.get("Financials") or {}
    bs = (fin.get("Balance_Sheet") or {}).get("quarterly") or {}
    inc = (fin.get("Income_Statement") or {}).get("quarterly") or {}
    cf = (fin.get("Cash_Flow") or {}).get("quarterly") or {}
    rows = []
    for qd in sorted(set(bs) | set(inc)):
        b = bs.get(qd, {}); i = inc.get(qd, {}); c = cf.get(qd, {})
        try:
            pe = dt.date.fromisoformat(qd[:10])
        except ValueError:
            continue
        fd = b.get("filing_date") or i.get("filing_date")
        try:
            av = dt.date.fromisoformat(fd[:10]) if fd else pe + LAG
        except (ValueError, TypeError):
            av = pe + LAG
        rows.append(dict(
            ticker=tk, period_end=pe, avail_date=av,
            total_equity=_num(b.get("totalStockholderEquity")),
            shares_outstanding=_num(b.get("commonStockSharesOutstanding")),
            total_debt=_num(b.get("shortLongTermDebtTotal")) or _num(b.get("netDebt")),
            total_assets=_num(b.get("totalAssets")),
            current_assets=_num(b.get("totalCurrentAssets")),
            current_liabilities=_num(b.get("totalCurrentLiabilities")),
            cash_and_equivalents=_num(b.get("cash")),
            inventory=_num(b.get("inventory")),
            net_income=_num(i.get("netIncome")),
            revenue=_num(i.get("totalRevenue")),
            operating_income=_num(i.get("ebit")) or _num(i.get("operatingIncome")),
            gross_profit=_num(i.get("grossProfit")),
            eps_diluted=_num(i.get("dilutedEPS")),
            free_cash_flow=_num(c.get("freeCashFlow")),
            operating_cash_flow=_num(c.get("totalCashFromOperatingActivities")),
        ))
    return {"gic": gic, "rows": rows}


def targets(offset, limit):
    # Postgres /dev/shm is only 64MB (Docker default); a parallel DISTINCT over the ~1.5M-row Candle hypertable
    # needs ~324MB of shared memory -> DiskFull. Disable parallel query for this connection so it uses private
    # work_mem (spills to disk temp) instead of /dev/shm. Also restrict the distinct to the delisted universe.
    from django.db import connection
    with connection.cursor() as cur:
        cur.execute("SET max_parallel_workers_per_gather = 0")
    dl = list(DelistedCompany.objects.filter(country="USA", type="Common Stock")
              .order_by("ticker").values_list("ticker", "eodhd_symbol"))
    dl_tk = [t for t, _ in dl]
    have_c = set(Candle.objects.filter(interval="1d", ticker__in=dl_tk)
                 .values_list("ticker", flat=True).distinct())
    have_f = set(FinancialReport.objects.filter(ticker__in=dl_tk)
                 .values_list("ticker", flat=True).distinct())
    rows = [(t, s) for t, s in dl if t in have_c and t not in have_f]   # have candles, need fundamentals
    if offset:
        rows = rows[offset:]
    if limit:
        rows = rows[:limit]
    return rows


def run(offset=0, limit=None, jobs=8):
    tks = targets(offset, limit)
    print(f"delisted fundamentals: {len(tks)} names (offset {offset}, jobs {jobs})", flush=True)
    try:
        gic_map = json.load(open(GIC_OUT))
    except Exception:
        gic_map = {}
    os.makedirs(os.path.dirname(GIC_OUT), exist_ok=True)
    saved = names = 0
    # STREAM: write each ticker as its fetch completes (reap-resilient; resume via targets() skip)
    with ThreadPoolExecutor(max_workers=jobs) as ex:
        futs = {ex.submit(_fetch, t, s): t for t, s in tks}
        for fu in as_completed(futs):
            t = futs[fu]
            try:
                r = fu.result()
            except Exception:
                r = None
            if not r:
                continue
            if r.get("gic"):
                gic_map[t] = r["gic"]
            good = [row for row in r["rows"] if row["total_equity"] is not None or row["net_income"] is not None]
            if not good:
                continue
            FinancialReport.objects.filter(ticker=t).delete()
            FinancialReport.objects.bulk_create([FinancialReport(**row) for row in good],
                                                ignore_conflicts=True, batch_size=2000)
            saved += len(good); names += 1
            if names % 100 == 0:
                json.dump(gic_map, open(GIC_OUT, "w"))
                print(f"  ...{names} names, {saved} rows (gic {len(gic_map)})", flush=True)
    json.dump(gic_map, open(GIC_OUT, "w"))
    print(f"DONE: {names} names, {saved} FinancialReport rows; gic_map {len(gic_map)} tickers -> {GIC_OUT}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", action="store_true"); ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--offset", type=int, default=0); ap.add_argument("--jobs", type=int, default=8)
    a = ap.parse_args()
    if not os.environ.get("EODHD_API_KEY"):
        print("no EODHD key", flush=True); return
    if not a.run:
        tks = targets(0, 3)
        print("probe:", tks, flush=True)
        for t, s in tks:
            r = _fetch(t, s)
            print(f"  {t}: gic={r and r.get('gic')}, {len(r['rows']) if r else 0} quarters" if r else f"  {t}: none", flush=True)
        return
    run(offset=a.offset, limit=a.limit, jobs=a.jobs)


if __name__ == "__main__":
    main()
