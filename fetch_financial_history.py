#!/usr/bin/env python3
"""Backfill quarterly financials into FinancialReport from the SEC EDGAR `companyfacts`
XBRL API (free, 10+ years of filings, with TRUE filing dates). Dividends come from
yfinance (deep history there is fine) unless --skip-dividends.

Why EDGAR over yfinance for financials: yfinance's quarterly_* endpoints return only
~5 recent quarters, so point-in-time buckets were ~94% NA over a 5-year backtest. EDGAR
carries the full filing history and stamps each fact with its `filed` date, so
`avail_date` is the ACTUAL date the number became public (no +45-day approximation).

Run in the backend container:
  docker compose run --rm backend python -u fetch_financial_history.py --db
Flags: --jobs N  --limit N  --tickers AAPL,MSFT  --skip-dividends
⚠️ `run --rm` detaches — check `docker ps | grep rotation-backend-run` for orphans after.
SEC requires a descriptive User-Agent and rate-limits to ~10 req/s; companyfacts is ONE
request per ticker, and workers sleep between requests to stay polite.
"""
import warnings; warnings.filterwarnings("ignore")
import os, sys, json, time, urllib.request, urllib.error
from datetime import date
from concurrent.futures import ProcessPoolExecutor

DEFAULT_JOBS = 6                    # network + SEC rate limit; keep modest
UA = {"User-Agent": "webisoft-research william@webisoft.com"}
REQ_SLEEP = 0.15                    # polite pause between SEC requests, per worker

# XBRL us-gaap concept aliases. Flow = summed over a period (quarterly); instant = point.
_FLOW = {
    "revenue": ["RevenueFromContractWithCustomerExcludingAssessedTax", "Revenues",
                "SalesRevenueNet", "RevenueFromContractWithCustomerIncludingAssessedTax"],
    "net_income": ["NetIncomeLoss", "ProfitLoss"],
    "operating_income": ["OperatingIncomeLoss"],
    "eps_diluted": ["EarningsPerShareDiluted", "EarningsPerShareBasicAndDiluted"],
    "ocf": ["NetCashProvidedByUsedInOperatingActivities",
            "NetCashProvidedByUsedInOperatingActivitiesContinuingOperations"],
    "capex": ["PaymentsToAcquirePropertyPlantAndEquipment", "PaymentsToAcquireProductiveAssets"],
    "gross_profit": ["GrossProfit"],
    "cost_of_revenue": ["CostOfRevenue", "CostOfGoodsAndServicesSold", "CostOfGoodsSold"],
    "rd_expense": ["ResearchAndDevelopmentExpense"],
}
_INSTANT = {
    "total_equity": ["StockholdersEquity", "StockholdersEquityIncludingPortionAttributableToNoncontrollingInterest"],
    "current_assets": ["AssetsCurrent"],
    "current_liabilities": ["LiabilitiesCurrent"],
    "shares_outstanding": ["CommonStockSharesOutstanding"],
    "total_assets": ["Assets"],
    "inventory": ["InventoryNet", "InventoryFinishedGoodsNetOfReserves"],
    "cash_and_equivalents": ["CashAndCashEquivalentsAtCarryingValue",
                             "CashCashEquivalentsRestrictedCashAndRestrictedCashEquivalents"],
}
_DEBT_LT = ["LongTermDebtNoncurrent", "LongTermDebt"]
_DEBT_CUR = ["LongTermDebtCurrent", "DebtCurrent", "ShortTermBorrowings"]


def _d(s):
    return date.fromisoformat(s)


def _get(url):
    req = urllib.request.Request(url, headers=UA)
    with urllib.request.urlopen(req, timeout=45) as r:
        return json.load(r)


def _cik_map():
    m = _get("https://www.sec.gov/files/company_tickers.json")
    return {v["ticker"].upper(): int(v["cik_str"]) for v in m.values()}


# ── XBRL extraction ─────────────────────────────────────────────────────────
def _flow_facts(node):
    """All (end, start, val, filed, fy, fp) for a us-gaap flow node, any unit."""
    out = []
    if not node:
        return out
    for facts in node.get("units", {}).values():
        for f in facts:
            if f.get("val") is None or not f.get("start"):
                continue
            out.append((f["end"], f["start"], f["val"], f.get("filed"),
                        f.get("fy"), f.get("fp")))
    return out


def _collect_flow(g, aliases):
    """Quarterly flow value per period_end, deriving Q4 = annual - (Q1+Q2+Q3) when a
    standalone Q4 isn't tagged. Returns {end_date: (val, filed_date)} (earliest filed)."""
    quarterly = {}   # end -> (val, filed)   period ~90d
    annual = {}      # fiscal-year-end -> (val, filed)  period ~365d
    for name in aliases:
        for end, start, val, filed, fy, fp in _flow_facts(g.get(name)):
            days = (_d(end) - _d(start)).days
            if 80 <= days <= 100:
                prev = quarterly.get(end)
                if prev is None or (filed and filed < prev[1]):
                    quarterly[end] = (val, filed)
            elif 350 <= days <= 380:
                prev = annual.get(end)
                if prev is None or (filed and filed < prev[1]):
                    annual[end] = (val, filed)
    # Derive Q4 for fiscal-year ends that have an annual but no standalone quarter.
    q_ends = sorted(quarterly)
    for ann_end, (ann_val, ann_filed) in annual.items():
        if ann_end in quarterly:
            continue
        ae = _d(ann_end)
        # The three quarters within the ~year preceding the fiscal-year end.
        prior = [e for e in q_ends if 0 < (ae - _d(e)).days <= 300]
        if len(prior) >= 3:
            three = sorted(prior, key=lambda e: _d(e))[-3:]
            s = sum(quarterly[e][0] for e in three)
            quarterly[ann_end] = (ann_val - s, ann_filed)
    return quarterly


def _collect_flow_cumulative(g, aliases):
    """De-cumulate YTD-reported flows (cash-flow statement: OCF, capex) into single
    quarters. Facts within one fiscal year share a `start` (fiscal-year start) and end at
    successive quarter ends (3/6/9/12-month cumulative); Q_n = YTD_n - YTD_(n-1).
    Returns {end_date: (quarter_val, filed_date)}."""
    by_start = {}  # start -> {end: (val, filed)}
    for name in aliases:
        node = g.get(name)
        if not node:
            continue
        for facts in node.get("units", {}).values():
            for f in facts:
                if f.get("val") is None or not f.get("start"):
                    continue
                d = by_start.setdefault(f["start"], {})
                prev = d.get(f["end"])
                if prev is None or (f.get("filed") and f["filed"] < prev[1]):
                    d[f["end"]] = (f["val"], f.get("filed"))
    out = {}
    for start, ends in by_start.items():
        seq = sorted(ends.items(), key=lambda kv: _d(kv[0]))
        prev_val = 0.0
        for end, (val, filed) in seq:
            days = (_d(end) - _d(start)).days
            q = val - prev_val
            prev_val = val
            if not (80 <= days <= 400):   # skip anything not a plausible cumulative step
                continue
            prevout = out.get(end)
            if prevout is None or (filed and filed < prevout[1]):
                out[end] = (q, filed)
    return out


def _collect_instant(g, aliases):
    """Instant value per date. Returns {end_date: (val, filed_date)} (earliest filed)."""
    out = {}
    for name in aliases:
        node = g.get(name)
        if not node:
            continue
        for facts in node.get("units", {}).values():
            for f in facts:
                if f.get("val") is None:
                    continue
                end, filed = f["end"], f.get("filed")
                prev = out.get(end)
                if prev is None or (filed and filed < prev[1]):
                    out[end] = (f["val"], filed)
    return out


def _nearest(instant_map, target_end, tol=8):
    """Instant value whose date is within `tol` days of target_end (balance-sheet dates
    line up with the quarter end but can differ by a few days)."""
    if not instant_map:
        return None
    te = _d(target_end)
    best, bestd = None, tol + 1
    for e, (val, _f) in instant_map.items():
        dd = abs((_d(e) - te).days)
        if dd < bestd:
            best, bestd = val, dd
    return best


def _extract_reports(cf):
    """companyfacts JSON -> list of FinancialReport-shaped dicts (one per quarter)."""
    g = cf.get("facts", {}).get("us-gaap", {})
    dei = cf.get("facts", {}).get("dei", {})
    if not g:
        return []
    rev = _collect_flow(g, _FLOW["revenue"])
    ni = _collect_flow(g, _FLOW["net_income"])
    oi = _collect_flow(g, _FLOW["operating_income"])
    eps = _collect_flow(g, _FLOW["eps_diluted"])
    ocf = _collect_flow_cumulative(g, _FLOW["ocf"])      # cash-flow stmt = YTD cumulative
    capex = _collect_flow_cumulative(g, _FLOW["capex"])
    gross = _collect_flow(g, _FLOW["gross_profit"])
    cogs = _collect_flow(g, _FLOW["cost_of_revenue"])
    rd = _collect_flow(g, _FLOW["rd_expense"])
    equity = _collect_instant(g, _INSTANT["total_equity"])
    cur_a = _collect_instant(g, _INSTANT["current_assets"])
    cur_l = _collect_instant(g, _INSTANT["current_liabilities"])
    debt_lt = _collect_instant(g, _DEBT_LT)
    debt_cur = _collect_instant(g, _DEBT_CUR)
    shares = _collect_instant(g, _INSTANT["shares_outstanding"])
    assets = _collect_instant(g, _INSTANT["total_assets"])
    inv = _collect_instant(g, _INSTANT["inventory"])
    cash = _collect_instant(g, _INSTANT["cash_and_equivalents"])
    if not shares and dei.get("EntityCommonStockSharesOutstanding"):
        shares = _collect_instant({"x": dei["EntityCommonStockSharesOutstanding"]}, ["x"])

    reports = []
    # Anchor on quarters that have revenue OR net income.
    ends = sorted(set(rev) | set(ni))
    for end in ends:
        r_val, r_filed = rev.get(end, (None, None))
        n_val, n_filed = ni.get(end, (None, None))
        filed = min([f for f in (r_filed, n_filed) if f], default=None)
        if not filed:
            continue
        lt = _nearest(debt_lt, end)
        cur = _nearest(debt_cur, end)
        total_debt = None
        if lt is not None or cur is not None:
            total_debt = int((lt or 0) + (cur or 0))
        ocf_v = ocf.get(end, (None, None))[0]
        capex_v = capex.get(end, (None, None))[0]
        fcf = int(ocf_v - capex_v) if (ocf_v is not None and capex_v is not None) else None
        sh = _nearest(shares, end, tol=45)

        def _i(m, tol=8):  # nearest instant as int, or None
            v = _nearest(m, end, tol)
            return int(v) if v is not None else None

        reports.append(dict(
            period_end=_d(end),
            avail_date=_d(filed),
            revenue=int(r_val) if r_val is not None else None,
            net_income=int(n_val) if n_val is not None else None,
            eps_diluted=eps.get(end, (None, None))[0],
            operating_income=int(oi[end][0]) if end in oi else None,
            total_equity=_i(equity),
            total_debt=total_debt,
            current_assets=_i(cur_a),
            current_liabilities=_i(cur_l),
            free_cash_flow=fcf,
            shares_outstanding=int(sh) if sh is not None else None,
            operating_cash_flow=int(ocf_v) if ocf_v is not None else None,
            total_assets=_i(assets),
            gross_profit=int(gross[end][0]) if end in gross else None,
            cost_of_revenue=int(cogs[end][0]) if end in cogs else None,
            rd_expense=int(rd[end][0]) if end in rd else None,
            inventory=_i(inv, tol=45),
            cash_and_equivalents=_i(cash, tol=45),
        ))
    return reports


# ── Worker ────────────────────────────────────────────────────────────────
def _worker(payload):
    tickers, cikmap, skip_div = payload
    import django
    os.environ.setdefault("DJANGO_SETTINGS_MODULE", "rotation.settings")
    try:
        django.setup()
    except Exception:
        pass
    from core.models import FinancialReport, DividendHistory
    saved = 0
    for tk in tickers:
        cik = cikmap.get(tk.upper())
        if cik:
            try:
                cf = _get(f"https://data.sec.gov/api/xbrl/companyfacts/CIK{cik:010d}.json")
                for fr in _extract_reports(cf):
                    pe = fr.pop("period_end")
                    FinancialReport.objects.update_or_create(
                        ticker=tk, period_end=pe, defaults=fr)
                    saved += 1
            except urllib.error.HTTPError as e:
                if e.code != 404:
                    print(f"  {tk} (CIK {cik}): HTTP {e.code}")
            except Exception as e:
                print(f"  {tk}: ERROR {e}")
            time.sleep(REQ_SLEEP)
        if not skip_div:
            try:
                import yfinance as yf
                divs = yf.Ticker(tk).dividends
                if divs is not None and len(divs) > 0:
                    for ex, amt in divs.items():
                        ed = ex.date() if hasattr(ex, "date") else ex
                        DividendHistory.objects.update_or_create(
                            ticker=tk, ex_date=ed, defaults={"amount": float(amt)})
            except Exception:
                pass
    return saved


def _chunk(seq, n):
    n = max(1, n); k, r = divmod(len(seq), n); out = []; i = 0
    for j in range(n):
        sz = k + (1 if j < r else 0)
        if sz:
            out.append(seq[i:i + sz]); i += sz
    return out


def run(jobs=DEFAULT_JOBS, limit=None, tickers=None, skip_div=False):
    from seq_fundamental_study import build_universe
    from django.db import connections
    uni = tickers or build_universe()
    if limit:
        uni = uni[:limit]
    print(f"EDGAR backfill for {len(uni)} tickers | jobs: {jobs} | dividends: {not skip_div}")
    cikmap = _cik_map()
    print(f"CIK map: {len(cikmap)} tickers")
    connections.close_all()
    chunks = _chunk(uni, jobs * 2)
    payloads = [(c, cikmap, skip_div) for c in chunks]
    total = 0
    import multiprocessing as mp
    ctx = mp.get_context("spawn")
    with ProcessPoolExecutor(max_workers=jobs, mp_context=ctx) as ex:
        for got in ex.map(_worker, payloads):
            total += got
    print(f"Done: {total} FinancialReport rows upserted.")
    return total


if __name__ == "__main__":
    argv = sys.argv
    jobs = int(argv[argv.index("--jobs") + 1]) if "--jobs" in argv else DEFAULT_JOBS
    limit = int(argv[argv.index("--limit") + 1]) if "--limit" in argv else None
    tks = argv[argv.index("--tickers") + 1].split(",") if "--tickers" in argv else None
    skip_div = "--skip-dividends" in argv
    if "--db" in argv:
        import django
        os.environ.setdefault("DJANGO_SETTINGS_MODULE", "rotation.settings")
        django.setup()
    run(jobs=jobs, limit=limit, tickers=tks, skip_div=skip_div)
