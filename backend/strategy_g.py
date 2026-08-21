#!/usr/bin/env python3
"""Strategy G — analyst-gap diversified value book. See docs/superpowers/specs/2026-08-21-strategy-g-analyst-gap-design.md.

Long-only, equal-weight (or inverse-vol), monthly month-end rebalance. Buys the top-5% of analyst-covered US
names trading FURTHEST BELOW their analyst target, gated to TTM-profitable (net income>0). $5M/day dvol floor,
price>$1. High-capacity, low-turnover complement to the flagship. PIT, survivorship-aware.
Run: MSYS_NO_PATHCONV=1 docker exec rotation-backend-1 python -u /app/strategy_g.py [--db] [--variant core|smooth] [--cost-bps N] [--sweep]"""
import os, json
os.environ.setdefault("DJANGO_SETTINGS_MODULE", "rotation.settings")
import django; django.setup()
import numpy as np, pandas as pd
from pathlib import Path
from collections import defaultdict
from core.models import Candle, Sector

DVOL_FLOOR = 5e6
PRICE_FLOOR = 1.0
TOP_FRAC = 0.05
STALE_DAYS = 180
RATINGS = Path("/app/.data/analyst_ratings.jsonl")
_reports = {}   # module-global {ticker: quarterly-report DataFrame}, set by build_panels (used by tests/scanner)


def build_universe():
    """(sorted list of covered US non-ETF tickers, {ticker: [(ts_ns, target), ...]})."""
    etfs = set(Sector.objects.values_list("etf", flat=True)) | {"SPY", "QQQ"}
    tgts = defaultdict(list)
    for line in RATINGS.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            r = json.loads(line)
        except Exception:
            continue
        tk, pt, d = r.get("ticker"), r.get("price_target"), r.get("date")
        if tk and pt and d and "." not in tk and tk not in etfs:   # US-listed only; drop ETFs
            tgts[tk].append((pd.Timestamp(d).value, float(pt)))
    return sorted(tgts), tgts


def build_panels(uni, tgts):
    """Monthly PIT panels. Returns dict(mclose, mdvol, ups, ttm_ni, mvol, fwd, spy_ret, midx)."""
    from seq_fundamental_study import load_financial_reports
    global _reports
    rows = list(Candle.objects.filter(ticker__in=uni + ["SPY"], date__gte="2015-12-01")
                .values_list("ticker", "date", "close", "volume"))
    df = pd.DataFrame(rows, columns=["ticker", "date", "close", "volume"])
    df["date"] = pd.to_datetime(df["date"]); df["close"] = df["close"].astype(float); df["volume"] = df["volume"].astype(float)
    df["dvol"] = df["close"] * df["volume"]
    close_d = df.pivot_table(index="date", columns="ticker", values="close")
    dvol20 = df.pivot_table(index="date", columns="ticker", values="dvol").rolling(20, min_periods=5).mean()
    mclose = close_d.resample("ME").last()
    mdvol = dvol20.resample("ME").last()
    midx = mclose.index

    # implied-upside panel: median target within trailing STALE_DAYS, / month-end close - 1
    midx_i = np.array([t.value for t in midx], dtype="int64")
    stale = STALE_DAYS * 86400 * 10**9
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

    _reports = load_financial_reports(uni)

    def _pit_panel(field, flow):
        """PIT monthly panel; flow=True -> rolling-4Q TTM sum, else latest balance-sheet value; ffill by avail_date."""
        out = {}
        for tk, r in _reports.items():
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

    ttm_ni = _pit_panel("net_income", True)
    mvol = close_d.pct_change().rolling(60, min_periods=20).std().resample("ME").last().reindex(midx)
    fwd = mclose.shift(-1) / mclose - 1.0
    spy_ret = mclose["SPY"].shift(-1) / mclose["SPY"] - 1.0
    return dict(mclose=mclose, mdvol=mdvol, ups=ups, ttm_ni=ttm_ni, mvol=mvol, fwd=fwd, spy_ret=spy_ret, midx=midx)


def profit_ok(P, d, t):
    """TTM net income present and > 0 at month d (the quality guard)."""
    v = P["ttm_ni"].get(t)
    return v is not None and pd.notna(P["ttm_ni"].loc[d, t]) and P["ttm_ni"].loc[d, t] > 0


def sim(P, top_frac=TOP_FRAC, profit_gate=True, weight="equal", cost_bps=0.0):
    """Monthly long-only book. weight: 'equal' | 'invvol'. cost_bps: one-way per-trade cost charged on both
    sides of turnover. Returns dict(ret, avg_n, pos_win, turnover, holdings)."""
    mclose, mdvol, ups, mvol, fwd = P["mclose"], P["mdvol"], P["ups"], P["mvol"], P["fwd"]
    midx = P["midx"]
    pr, held_n, pos_win, pos_tot = [], [], 0, 0
    prev_w, turns, holdings = {}, [], {}
    for d in midx[:-1]:
        u = ups.loc[d].dropna(); f = fwd.loc[d]; dv = mdvol.loc[d]; cl = mclose.loc[d]; vol = mvol.loc[d]
        cand = [t for t in u.index if t != "SPY" and pd.notna(f.get(t)) and np.isfinite(f[t])
                and pd.notna(dv.get(t)) and dv[t] >= DVOL_FLOOR and pd.notna(cl.get(t)) and cl[t] > PRICE_FLOOR]
        if profit_gate:
            cand = [t for t in cand if profit_ok(P, d, t)]
        if len(cand) < 10:
            pr.append(np.nan); held_n.append(0); continue
        ranked = u[cand].sort_values(ascending=False)                 # furthest-below-target first
        k = max(1, int(len(ranked) * top_frac))
        hold = list(ranked.index[:k]); rr = f[hold]
        if weight == "invvol":
            wv = np.array([1.0 / vol[t] if pd.notna(vol.get(t)) and vol[t] > 0 else np.nan for t in hold])
            wv = np.where(np.isfinite(wv), wv, np.nanmedian(wv)) if np.isfinite(wv).sum() >= 2 else np.ones(len(hold))
            wv = wv / wv.sum(); m = float(np.dot(rr.values, wv))
        else:
            wv = np.full(len(hold), 1.0 / len(hold)); m = float(rr.mean())
        pos_win += int((rr > 0).sum()); pos_tot += len(rr)
        cur_w = {t: wv[i] for i, t in enumerate(hold)}
        keys = set(cur_w) | set(prev_w)
        turn = 0.5 * sum(abs(cur_w.get(t, 0.0) - prev_w.get(t, 0.0)) for t in keys)   # one-way turnover
        turns.append(turn); m -= turn * 2.0 * (cost_bps / 1e4); prev_w = cur_w        # charge both sides
        holdings[d.date().isoformat()] = hold
        pr.append(m); held_n.append(k)
    return dict(ret=pd.Series(pr, index=midx[:-1]), avg_n=float(np.mean([h for h in held_n if h])),
                pos_win=pos_win / pos_tot if pos_tot else float("nan"),
                turnover=float(np.mean(turns)) if turns else float("nan"), holdings=holdings)


def latest_picks(P, top_frac=TOP_FRAC, profit_gate=True):
    """Actionable holdings for the MOST RECENT month-end (no forward return needed — for the live scanner).
    Same candidate/gate/rank logic as sim(). Returns (date_iso, [{ticker, upside, close}, ...])."""
    mclose, mdvol, ups = P["mclose"], P["mdvol"], P["ups"]
    d = P["midx"][-1]
    u = ups.loc[d].dropna(); dv = mdvol.loc[d]; cl = mclose.loc[d]
    cand = [t for t in u.index if t != "SPY" and pd.notna(dv.get(t)) and dv[t] >= DVOL_FLOOR
            and pd.notna(cl.get(t)) and cl[t] > PRICE_FLOOR]
    if profit_gate:
        cand = [t for t in cand if profit_ok(P, d, t)]
    ranked = u[cand].sort_values(ascending=False)
    k = max(1, int(len(ranked) * top_frac))
    hold = list(ranked.index[:k])
    return d.date().isoformat(), [dict(ticker=t, upside=float(u[t]), close=float(cl[t])) for t in hold]


def perf(ret, spy_ret, posw=None):
    """Summary stats. Percentages as numbers (e.g. 31.1). Returns dict(total,cagr,sharpe,maxdd,hit,beat_spy,n_mo,pos_win)."""
    pr = ret.dropna(); eq = np.cumprod(1 + pr.values)
    yrs = len(pr) / 12.0; sr = spy_ret.reindex(pr.index).values
    return dict(total=(eq[-1] - 1) * 100, cagr=(eq[-1] ** (1 / yrs) - 1) * 100,
                sharpe=float(pr.mean() / pr.std() * np.sqrt(12)),
                maxdd=float((eq / np.maximum.accumulate(eq) - 1).min() * 100),
                hit=float((pr.values > 0).mean() * 100), beat_spy=float(np.nanmean(pr.values > sr) * 100),
                n_mo=int(len(pr)), pos_win=None if posw is None else float(posw * 100))


_WINDOWS = [("2016-2018", "2016-01-01", "2018-12-31"), ("2019-2021", "2019-01-01", "2021-12-31"),
            ("2022-2023", "2022-01-01", "2023-12-31"), ("2024-2026", "2024-01-01", "2026-12-31"),
            ("covid-2020H1", "2020-01-01", "2020-06-30")]


def walk_forward(P, cost_bps=25.0):
    """Non-overlapping subperiods + a COVID-crash window. verdict='robust' if G-core beats SPY CAGR in >=3 of
    the 4 multi-year windows, else 'fragile'. Returns dict(subperiods=[...], verdict)."""
    r = sim(P, cost_bps=cost_bps)["ret"]; spy = P["spy_ret"]
    subs = []
    for label, a, b in _WINDOWS:
        seg = r[(r.index >= a) & (r.index <= b)].dropna()
        sseg = spy.reindex(seg.index)
        if len(seg) < 3:
            continue
        m = perf(seg, sseg); sm = perf(sseg, sseg)
        subs.append(dict(label=label, start=a, end=b, cagr=m["cagr"], maxdd=m["maxdd"],
                         hit=m["hit"], beat_spy=m["beat_spy"], spy_cagr=sm["cagr"]))
    multi = [s for s in subs if not s["label"].startswith("covid")]
    beats = sum(1 for s in multi if s["cagr"] > s["spy_cagr"])
    return dict(subperiods=subs, verdict="robust" if beats >= 3 else "fragile")


COST_GRID = [0, 10, 25, 50, 100]
OUT = Path("/app/.data/strategy_g.json")


def build_payload(P):
    """Full persisted payload: cost grid for both variants, turnover, walk-forward, latest holdings, curve, caveat."""
    def grid(weight):
        out = []
        for cb in COST_GRID:
            r = sim(P, weight=weight, cost_bps=cb); m = perf(r["ret"], P["spy_ret"], r["pos_win"])
            out.append(dict(cost_bps=cb, weight=weight, avg_n=r["avg_n"], turnover=r["turnover"], **m))
        return out
    core_grid, smooth_grid = grid("equal"), grid("invvol")
    r25 = sim(P, cost_bps=25.0)
    last_d = sorted(r25["holdings"])[-1]
    spy = perf(P["spy_ret"], P["spy_ret"])
    return dict(
        computed_at=pd.Timestamp.utcnow().isoformat(),
        config=dict(top_frac=TOP_FRAC, profit_gate=True, weight="equal", dvol_floor=DVOL_FLOOR,
                    price_floor=PRICE_FLOOR, stale_days=STALE_DAYS, rebalance="monthly"),
        cost_grid=core_grid + smooth_grid, spy=spy,
        turnover_oneway_monthly=r25["turnover"],
        walk_forward=walk_forward(P, cost_bps=25.0),
        latest_holdings=dict(date=last_d, tickers=r25["holdings"][last_d]),
        curve={d.date().isoformat(): float(v) for d, v in r25["ret"].dropna().items()},
        caveat="Long-only EW monthly. PIT (targets<=month-end 180d; TTM ni ffill by avail_date). Survivorship-aware "
               "(delisted-with-candles incl.). US-listed only. In-sample 2016-2026; see walk_forward for subperiods. "
               "Costs modeled as flat bps on turnover (no market-impact curve).")


def _print_sweep(P):
    def row(label, **kw):
        r = sim(P, **kw); m = perf(r["ret"], P["spy_ret"], r["pos_win"])
        print(f"  {label:36} CAGR {m['cagr']:+6.1f}%  DD {m['maxdd']:6.1f}%  Sh {m['sharpe']:4.2f}  "
              f"hit {m['hit']:4.1f}%  posW {m['pos_win']:.1f}%  turn {r['turnover']*100:.0f}%", flush=True)
    print("=== G lever sweep (record) ===", flush=True)
    row("G-core (top-5% profit EW)")
    row("G-smooth (inverse-vol)", weight="invvol")
    for frac, tag in [(0.10, "decile"), (0.20, "quintile")]:
        row(f"{tag} profit EW", top_frac=frac)


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", action="store_true"); ap.add_argument("--variant", default="core", choices=["core", "smooth"])
    ap.add_argument("--cost-bps", type=float, default=25.0); ap.add_argument("--sweep", action="store_true")
    a = ap.parse_args()
    uni, tgts = build_universe(); print(f"covered universe: {len(uni)}", flush=True)
    P = build_panels(uni, tgts)
    w = "invvol" if a.variant == "smooth" else "equal"
    r = sim(P, weight=w, cost_bps=a.cost_bps); m = perf(r["ret"], P["spy_ret"], r["pos_win"])
    print(f"G-{a.variant} @ {a.cost_bps:.0f}bps: CAGR {m['cagr']:+.1f}%  DD {m['maxdd']:.1f}%  "
          f"Sharpe {m['sharpe']:.2f}  hit {m['hit']:.1f}%  >SPY {m['beat_spy']:.1f}%  turn {r['turnover']*100:.0f}%", flush=True)
    if a.sweep:
        _print_sweep(P)
    p = build_payload(P)
    OUT.parent.mkdir(parents=True, exist_ok=True); OUT.write_text(json.dumps(p, indent=2, default=str))
    print(f"verdict: {p['walk_forward']['verdict']}", flush=True)
    if a.db:
        try:
            from core.models import BacktestResult
            from django.utils import timezone
            BacktestResult.objects.update_or_create(
                kind="strategy_g", defaults={"payload": json.loads(json.dumps(p, default=str)), "computed_at": timezone.now()})
            print("Saved BacktestResult[strategy_g]", flush=True)
        except Exception as e:
            print("DB save failed:", e, flush=True)


if __name__ == "__main__":
    main()
