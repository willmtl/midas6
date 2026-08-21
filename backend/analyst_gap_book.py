#!/usr/bin/env python3
"""DIVERSIFIED analyst-gap LONG book — is the isolated price-vs-target-gap signal (Q5-Q1 +1.96%/mo, t6.69
from analyst_gap_study.py) a viable STANDALONE strategy, separate from the concentrated flagship?

Long-only, EQUAL-WEIGHT, monthly rebalance, EXECUTABLE ($5M/day 20d-avg dollar-volume floor + >$1 price).
Buys the top-quintile (and top-decile) names trading furthest BELOW their analyst target. The liquidity
floor is the key honesty test: if the edge is microcap noise it dies here; if it survives, it's a real
diversified book. Reports total/CAGR/Sharpe/maxDD vs SPY buy-hold, 2016-2026.
Run: MSYS_NO_PATHCONV=1 docker exec rotation-backend-1 python -u /app/analyst_gap_book.py"""
import os, json
os.environ.setdefault("DJANGO_SETTINGS_MODULE", "rotation.settings")
import django; django.setup()
import numpy as np, pandas as pd
from pathlib import Path
from collections import defaultdict
from core.models import Candle, Sector
import sys
sys.path.insert(0, "/app")
from seq_fundamental_study import load_financial_reports

DVOL_FLOOR = 5e6
etfs = set(Sector.objects.values_list("etf", flat=True)) | {"SPY", "QQQ"}
tgts = defaultdict(list)
for line in Path("/app/.data/analyst_ratings.jsonl").read_text(encoding="utf-8").splitlines():
    if not line.strip():
        continue
    try:
        r = json.loads(line)
    except Exception:
        continue
    tk, pt, d = r.get("ticker"), r.get("price_target"), r.get("date")
    if tk and pt and d and "." not in tk and tk not in etfs:
        tgts[tk].append((pd.Timestamp(d).value, float(pt)))
uni = sorted(tgts)
print(f"covered US universe: {len(uni)}", flush=True)

rows = list(Candle.objects.filter(ticker__in=uni + ["SPY"], date__gte="2015-12-01")
            .values_list("ticker", "date", "close", "volume"))
df = pd.DataFrame(rows, columns=["ticker", "date", "close", "volume"])
df["date"] = pd.to_datetime(df["date"]); df["close"] = df["close"].astype(float); df["volume"] = df["volume"].astype(float)
df["dvol"] = df["close"] * df["volume"]
close_d = df.pivot_table(index="date", columns="ticker", values="close")
dvol20 = df.pivot_table(index="date", columns="ticker", values="dvol").rolling(20, min_periods=5).mean()
mclose = close_d.resample("ME").last()
mdvol = dvol20.resample("ME").last()
spy_m = mclose["SPY"]
midx = mclose.index
print(f"months {midx[0].date()}..{midx[-1].date()}  tickers {mclose.shape[1]}", flush=True)

# implied-upside panel
midx_i = np.array([t.value for t in midx], dtype="int64")
stale = 180 * 86400 * 10**9
ups = pd.DataFrame(np.nan, index=midx, columns=mclose.columns)
for tk in mclose.columns:
    pts = tgts.get(tk)
    if not pts:
        continue
    arr = np.array(sorted(pts)); di, tv = arr[:, 0], arr[:, 1]
    col = np.full(len(midx), np.nan)
    for j, d in enumerate(midx_i):
        a = np.searchsorted(di, d - stale, side="right"); b = np.searchsorted(di, d, side="right")
        if b > a:
            col[j] = np.median(tv[a:b])
    ups[tk] = col / mclose[tk].values - 1.0

# ── QUALITY-GATE panels (the flagship's "free lunch"): PIT TTM net income (>0 = profitable) + D/E ──
reports = load_financial_reports(uni)
print(f"fundamentals loaded for {len(reports)} of {len(uni)} covered names", flush=True)


def _pit_panel(field, flow):
    """PIT monthly panel. flow=True -> rolling-4Q TTM sum; flow=False -> latest balance-sheet value.
    Forward-filled by avail_date onto the month-end index (no look-ahead: only uses filings available by then)."""
    out = {}
    for tk, r in reports.items():
        if field not in r.columns:
            continue
        d = r[["period_end", "avail_date", field]].dropna(subset=[field]).copy()
        if d.empty:
            continue
        d = d.sort_values("period_end")
        v = d[field].rolling(4).sum() if flow else d[field]
        s = pd.Series(v.values, index=pd.to_datetime(d["avail_date"])).dropna()
        if s.empty:
            continue
        s = s[~s.index.duplicated(keep="last")].sort_index()
        out[tk] = s.reindex(s.index.union(midx)).ffill().reindex(midx)
    return pd.DataFrame(out).reindex(columns=mclose.columns)


ttm_ni = _pit_panel("net_income", True)          # TTM net income (profitability guard: >0)
equity = _pit_panel("total_equity", False)       # latest book equity
debt = _pit_panel("total_debt", False)           # latest total debt
de_ratio = debt / equity.where(equity > 0)       # debt-to-equity (low = healthier balance sheet)

fwd = mclose.shift(-1) / mclose - 1.0
spy_ret = spy_m.shift(-1) / spy_m - 1.0

# ── Strategy-G DD/win-rate levers ──
spy_ma10 = spy_m.rolling(10).mean()                # ~200-day trend filter on the benchmark
risk_on = (spy_m > spy_ma10)                       # market-regime gate: True = SPY above its 10-mo MA
mret = mclose.pct_change()                          # trailing 1-month return per name (turn signal)
_dret = close_d.pct_change()
mvol = _dret.rolling(60, min_periods=20).std().resample("ME").last().reindex(midx)   # trailing 60d daily-vol, monthly


def _profit_ok(d, t):
    v = ttm_ni.get(t)
    return v is not None and pd.notna(ttm_ni.loc[d, t]) and ttm_ni.loc[d, t] > 0


def _lowdebt_ok(d, t):
    v = de_ratio.get(t)
    return v is not None and pd.notna(de_ratio.loc[d, t]) and de_ratio.loc[d, t] < 1.0


def _turning_ok(d, t):
    """Entry-turn gate (flagship lesson: buy the oversold name that has STOPPED falling, not the falling knife)."""
    v = mret.get(t)
    return v is not None and pd.notna(mret.loc[d, t]) and mret.loc[d, t] > 0.0


def sim(top_frac, gate=None, regime=None, weight=None, volgate=False, trim=0.0, cost_bps=0.0):
    """gate: fn(date,ticker)->bool applied pre-rank. regime: None|'cash'|'half' de-risk below SPY 10-mo MA.
    weight: None (equal) | 'invvol' (1/trailing-vol). volgate: drop the top-vol tercile of candidates.
    trim: drop this fraction of the MOST-extreme gaps before taking top_frac (falling-knife guard).
    cost_bps: one-way per-trade cost (bps) charged on monthly turnover (both sides).
    Returns (monthly-return series, avg #held, per-position win rate, avg one-way turnover)."""
    pr, held_n, pos_win, pos_tot = [], [], 0, 0
    prev_w = {}; turns = []
    for d in midx[:-1]:
        u = ups.loc[d].dropna()
        f = fwd.loc[d]; dv = mdvol.loc[d]; cl = mclose.loc[d]; vol = mvol.loc[d]
        cand = [t for t in u.index if t != "SPY" and pd.notna(f.get(t)) and np.isfinite(f[t])
                and pd.notna(dv.get(t)) and dv[t] >= DVOL_FLOOR and pd.notna(cl.get(t)) and cl[t] > 1.0]
        if gate is not None:
            cand = [t for t in cand if gate(d, t)]
        if volgate:
            vv = [(t, vol.get(t)) for t in cand if pd.notna(vol.get(t))]
            if len(vv) >= 6:
                thr = np.quantile([v for _, v in vv], 2 / 3.0)        # keep the lower two vol terciles
                keep = {t for t, v in vv if v <= thr}
                cand = [t for t in cand if t in keep]
        if len(cand) < 10:
            pr.append(np.nan); held_n.append(0); continue
        ranked = u[cand].sort_values(ascending=False)                 # furthest-below-target first
        if trim > 0:
            ranked = ranked.iloc[max(1, int(len(ranked) * trim)):]    # skip the most-extreme (distressed) gaps
        k = max(1, int(len(ranked) * top_frac))
        hold = list(ranked.index[:k])
        rr = f[hold]
        if weight == "invvol":
            w = np.array([1.0 / vol[t] if pd.notna(vol.get(t)) and vol[t] > 0 else np.nan for t in hold])
            if np.isfinite(w).sum() >= 2:
                w = np.where(np.isfinite(w), w, np.nanmedian(w)); w = w / w.sum()
                m = float(np.dot(rr.values, w))
            else:
                m = float(rr.mean())
        else:
            m = float(rr.mean())
        pos_win += int((rr > 0).sum()); pos_tot += len(rr)            # selection win rate (regime-independent)
        # book weights this month (for turnover): equal or inverse-vol, matching the return calc above
        if weight == "invvol":
            wv = np.array([1.0 / vol[t] if pd.notna(vol.get(t)) and vol[t] > 0 else np.nan for t in hold])
            wv = np.where(np.isfinite(wv), wv, np.nanmedian(wv)) if np.isfinite(wv).sum() >= 2 else np.ones(len(hold))
            wv = wv / wv.sum()
        else:
            wv = np.full(len(hold), 1.0 / len(hold))
        cur_w = {t: wv[i] for i, t in enumerate(hold)}
        keys = set(cur_w) | set(prev_w)
        turn = 0.5 * sum(abs(cur_w.get(t, 0.0) - prev_w.get(t, 0.0)) for t in keys)   # one-way turnover
        turns.append(turn)
        m -= turn * 2.0 * (cost_bps / 1e4)                            # charge both sides (sell old + buy new)
        prev_w = cur_w
        if regime is not None and not bool(risk_on.get(d, True)):
            m = 0.0 if regime == "cash" else 0.5 * m                  # DD overlay: de-risk below the 200d trend
        pr.append(m); held_n.append(k)
    posw = pos_win / pos_tot if pos_tot else float("nan")
    return (pd.Series(pr, index=midx[:-1]), float(np.mean([h for h in held_n if h])),
            posw, float(np.mean(turns)) if turns else float("nan"))


def perf(pr, label, posw=None):
    pr = pr.dropna()
    eq = np.cumprod(1 + pr.values); tot = (eq[-1] - 1) * 100
    yrs = len(pr) / 12.0
    cagr = (eq[-1] ** (1 / yrs) - 1) * 100
    shrp = pr.mean() / pr.std() * np.sqrt(12)
    dd = (eq / np.maximum.accumulate(eq) - 1).min() * 100
    hit = (pr.values > 0).mean() * 100                                # monthly hit rate (fraction of up months)
    sr = spy_ret.reindex(pr.index).values
    beat = np.nanmean(pr.values > sr) * 100                           # % of months beating SPY
    pw = f"{posw*100:4.1f}%" if posw is not None else "  -  "
    print(f"  {label:40} TOT {tot:+9.1f}%  CAGR {cagr:+6.1f}%  Sh {shrp:4.2f}  DD {dd:6.1f}%  hit {hit:4.1f}%  >SPY {beat:4.1f}%  posW {pw}", flush=True)
    return eq


prof = _profit_ok
prof_turn = lambda d, t: _profit_ok(d, t) and _turning_ok(d, t)

if os.environ.get("G_SWEEP"):     # full DD/win-rate lever sweep (set G_SWEEP=1); default run is the cost validation
    print(f"\n=== reference books (equal-weight, monthly, >=${DVOL_FLOOR/1e6:.0f}M/day, >$1) ===", flush=True)
    for frac, tag in [(0.20, "top-quintile"), (0.10, "top-decile"), (0.05, "top-5%")]:
        s, n, w, _ = sim(frac); perf(s, f"{tag} (~{n:.0f} names)", w)
    perf(spy_ret, "SPY (buy & hold)")
    print(f"\n=== STRATEGY G lever sweep (base = top-5% + profit guard; goal: lower DD, higher win rate) ===", flush=True)
    b, nb, wb, _ = sim(0.05, prof); perf(b, f"base: top-5% profit-guard (~{nb:.0f})", wb)
    print("  -- lever A: market-regime overlay (de-risk below SPY 10-mo MA) --", flush=True)
    c, nc, wc, _ = sim(0.05, prof, regime="cash"); perf(c, f"  + regime CASH-gate (~{nc:.0f})", wc)
    h, nh, wh, _ = sim(0.05, prof, regime="half"); perf(h, f"  + regime HALF-exposure (~{nh:.0f})", wh)
    print("  -- lever B: entry-turn gate (buy only gap names that stopped falling: 1-mo return>0) --", flush=True)
    t5, nt, wt, _ = sim(0.05, prof_turn); perf(t5, f"  + turn gate (~{nt:.0f})", wt)
    th, nth, wth, _ = sim(0.05, prof_turn, regime="half"); perf(th, f"  + turn gate + regime HALF (~{nth:.0f})", wth)
    print("  -- lever C: wider diversification (more names = smoother), profit-guarded --", flush=True)
    for frac, tag in [(0.10, "decile"), (0.20, "quintile")]:
        s, n, w, _ = sim(frac, prof); perf(s, f"  {tag} profit-guard (~{n:.0f})", w)
        s2, n2, w2, _ = sim(frac, prof_turn, regime="half"); perf(s2, f"  {tag} profit+turn+regimeHALF (~{n2:.0f})", w2)
    print("  -- lever D: inverse-vol weighting / vol gate / trim extreme gaps (base = top-5% profit-guard) --", flush=True)
    iv, niv, wiv, _ = sim(0.05, prof, weight="invvol"); perf(iv, f"  + inverse-vol weight (~{niv:.0f})", wiv)
    vg, nvg, wvg, _ = sim(0.05, prof, volgate=True); perf(vg, f"  + vol gate (drop top-vol third, ~{nvg:.0f})", wvg)
    tr, ntr, wtr, _ = sim(0.05, prof, trim=0.2); perf(tr, f"  + trim most-extreme 20% (~{ntr:.0f})", wtr)

# ── COST VALIDATION: does G-core's 31% CAGR survive realistic turnover + transaction costs? ──
print(f"\n=== STRATEGY G-core COST VALIDATION (top-5% profit-guarded, equal-weight) ===", flush=True)
_, _, _, turn0 = sim(0.05, prof, cost_bps=0.0)
print(f"  avg one-way monthly turnover: {turn0*100:.0f}%   (=> ~{turn0*2*12*100:.0f}% annualized two-way traded)", flush=True)
for cb in [0, 10, 25, 50, 100]:
    s, n, w, tu = sim(0.05, prof, cost_bps=cb)
    perf(s, f"G-core @ {cb:>3d} bps/side cost (~{n:.0f})", w)
perf(spy_ret, "SPY (buy & hold, no cost)")
