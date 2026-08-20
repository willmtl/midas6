#!/usr/bin/env python3
"""#1 C-SELECTION screen: does 'cheap AND IMPROVING' beat plain cheapest-P/B? Reconstructs C's value cohort
(cheapest-P/B pick per top-accel sector, monthly — same selection as _windows_A minus the dip filter), tags each
pick by its most-recent earnings GUIDANCE (EarningsEvent.guidance_eps_pct = fwd-EPS-est revision at the report,
the only backtestable estimate-revision signal — forward-EPS/EstimateRevision are snapshot-only), and compares
forward 1mo/2mo returns: guided-UP vs guided-DOWN vs none. If guided-up value picks outperform -> wire a guidance
tilt into the flagship. Screen-before-wire. Run: MSYS_NO_PATHCONV=1 docker exec rotation-backend-1 python -u /app/c_guidance_screen.py"""
import os, bisect, warnings, datetime as dt
warnings.filterwarnings("ignore")
os.environ.setdefault("DJANGO_SETTINGS_MODULE", "rotation.settings")
import django
django.setup()
import numpy as np, pandas as pd, ta
from django.db import connection
with connection.cursor() as cur:
    cur.execute("SET max_parallel_workers_per_gather = 0")
import config, price_basis, sector_holdings
from seq_fundamental_study import load_candles, load_financial_reports
from trend_stock_studies import _pit_monthly_panel, _available_at, CRYPTO
from backtest_lowpb import _monthly_close, BENCH
from core.models import EarningsEvent

LOOKBACK, TOP_N = 6, 10

# ---- build C-style value cohort (cheapest-P/B pick per top-accel sector, monthly) ----
etfs = {name: etf for name, etf in config.SECTOR_ETFS.items() if etf not in CRYPTO}
sector_map, all_holds = {}, set()
for name, etf in etfs.items():
    h = [t for t in sector_holdings.get_holdings(name) if t not in (etf, BENCH) and t not in CRYPTO]
    sector_map[etf] = h; all_holds.update(h)
all_holds = sorted(all_holds)
etf_daily = load_candles(sorted(set(etfs.values()) | {BENCH}))
etf_monthly = _monthly_close({t: d for t, d in etf_daily.items() if t in etfs.values()})
midx = etf_monthly.index
stock_daily = load_candles(all_holds)
stock_monthly = _monthly_close(stock_daily).reindex(midx)
reports = load_financial_reports(all_holds)
shares_p = _pit_monthly_panel(reports, "shares_outstanding", midx)
equity_p = _pit_monthly_panel(reports, "total_equity", midx)
common = stock_monthly.columns.intersection(shares_p.columns).intersection(equity_p.columns)
pb = (price_basis.as_traded_close(stock_monthly[common]) * shares_p[common]) / equity_p[common].where(equity_p[common] != 0)
etf_trail = etf_monthly.pct_change(LOOKBACK)

# ---- guidance store: {ticker: (sorted report_dates, guidance_eps_pct)} ----
gd = {}
for tk, rd, g in EarningsEvent.objects.filter(guidance_eps_pct__isnull=False).order_by("ticker", "report_date").values_list("ticker", "report_date", "guidance_eps_pct"):
    rec = gd.setdefault(tk, ([], [])); rec[0].append(rd); rec[1].append(g)


def guide_asof(tk, d, within_days=100):
    rec = gd.get(tk)
    if not rec:
        return None
    i = bisect.bisect_right(rec[0], d) - 1
    if i < 0 or (d - rec[0][i]).days > within_days:
        return None
    return rec[1][i]


# ---- collect picks + forward returns + guidance tag ----
rows = []   # (fwd1, fwd2, guide)
for i in range(LOOKBACK, len(midx) - 2):
    date = midx[i]
    if date not in etf_trail.index:
        continue
    ranks = etf_trail.loc[date].dropna().sort_values(ascending=False).head(TOP_N)
    for etf in ranks.index:
        holds = sector_map.get(etf, [])
        cands = [h for h in holds if h in stock_monthly.columns and _available_at(stock_monthly[h], date)]
        if not cands or date not in pb.index:
            continue
        row = pb.loc[date, [c for c in cands if c in pb.columns]].dropna()
        row = row[row > 0]
        if not len(row):
            continue
        pick = row.idxmin()
        p0 = stock_monthly.loc[date, pick]
        p1 = stock_monthly.iloc[i + 1][pick] if pick in stock_monthly.columns else np.nan
        p2 = stock_monthly.iloc[i + 2][pick] if pick in stock_monthly.columns else np.nan
        if not (p0 and p0 > 0):
            continue
        fwd1 = (p1 / p0 - 1) * 100 if p1 and p1 > 0 else np.nan
        fwd2 = (p2 / p0 - 1) * 100 if p2 and p2 > 0 else np.nan
        rows.append((fwd1, fwd2, guide_asof(pick, date.date())))

arr = rows
print(f"C-value picks: {len(arr)} pick-months\n", flush=True)


def _t(a):
    a = np.asarray(a, float); a = a[np.isfinite(a)]
    if len(a) < 10:
        return None
    return round(float(a.mean() / (a.std(ddof=1) / np.sqrt(len(a)))), 2)


def show(label, sub):
    for j, h in ((0, "1mo"), (1, "2mo")):
        a = np.array([r[j] for r in sub], float); a = a[np.isfinite(a)]
        if len(a):
            print(f"  {label:22} {h}  n={len(a):>5}  avg {a.mean():+.2f}%  med {np.median(a):+.2f}%  win {(a>0).mean()*100:.0f}%  t={_t(a)}", flush=True)


show("ALL value picks", arr)
print()
show("guided UP (>0)", [r for r in arr if r[2] is not None and r[2] > 0])
show("guided DOWN (<0)", [r for r in arr if r[2] is not None and r[2] < 0])
show("guided FLAT (=0)", [r for r in arr if r[2] is not None and r[2] == 0])
show("NO recent guidance", [r for r in arr if r[2] is None])
print()
# magnitude tail among guided-up
up = [r for r in arr if r[2] is not None and r[2] > 0]
if up:
    qs = np.nanpercentile([r[2] for r in up], [50])
    show("guided UP strong(>med)", [r for r in up if r[2] > qs[0]])
    show("guided UP mild(<=med)", [r for r in up if r[2] <= qs[0]])
