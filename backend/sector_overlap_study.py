#!/usr/bin/env python3
"""SECTOR-OVERLAP PRUNING — several 'sectors' aren't sectors: Mag 7 (MAGS), Nasdaq-100 (QQQ), Growth (VUG),
Momentum (MTUM), Value (VTV), Dividend (SCHD), Low-Vol (SPLV), Small/Micro Cap (IWM/IWC) are broad-index /
factor / size slices whose holdings already live inside the 11 GICS sectors + themes. They (a) duplicate
exposure and (b) create artifacts like the NVDA-via-MAGS pick (cheapest of an all-expensive megacap basket).
MTUM is also CIRCULAR -- we rank sectors BY momentum, then include a momentum-factor 'sector'.

Two parts:
 (A) OVERLAP evidence: for each suspect ETF, max holdings-Jaccard and max monthly-return correlation vs the
     TRUE sectors (GICS + themes + geos + commodities). High overlap => redundant.
 (B) BACKTEST the flagship (accel top-10 -> cheapest as-traded-P/B guard low-debt $5M div_2x, monthly) with
     progressively more of the redundant ETFs REMOVED from the ranking universe. If pruning lifts return/
     Sharpe/DD, the overlap sectors were crowding the top-10 with duplicate megacap-tech exposure.
-> BacktestResult[sector_overlap] + JSON.
Run: MSYS_NO_PATHCONV=1 docker exec rotation-backend-1 python -u /app/sector_overlap_study.py
"""
import os, json, warnings
warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
os.environ.setdefault("DJANGO_SETTINGS_MODULE", "rotation.settings")
import django
django.setup()

from pathlib import Path
import config, sector_holdings, price_basis
from seq_fundamental_study import load_candles, load_financial_reports
from trend_stock_studies import _pit_monthly_panel, _available_at, _ret_delist, CRYPTO
from backtest_lowpb import _monthly_close, _tstat_from_returns, BENCH

TOP_N = 10; CONV = 2.0; MIN_DVOL = 5e6
OUT = Path(__file__).resolve().parent / ".data" / "studies" / "sector_overlap.json"

# suspect broad/factor/size ETFs (the "not really a sector" set), grouped so we can peel them in layers
BROADCAP = ["MAGS", "QQQ"]                         # megacap-tech index baskets
STYLE = ["VUG", "MTUM", "VTV", "SCHD", "SPLV"]     # growth/momentum/value/dividend/low-vol factors
SIZE = ["IWM", "IWC"]                              # broad small/micro size
ACTIVE = ["ARKK"]                                  # active megacap-growth fund
REMOVAL_SETS = {
    "baseline": [],
    "no_mag7": ["MAGS"],
    "no_broadcap": BROADCAP,
    "no_broad+style": BROADCAP + STYLE,
    "no_broad+style+size": BROADCAP + STYLE + SIZE,
    "pure_sectors": BROADCAP + STYLE + SIZE + ACTIVE,
}
SUSPECTS = BROADCAP + STYLE + SIZE + ACTIVE


def _perf(r, spy):
    r = np.asarray(r, float); n = len(r)
    tot = float(np.prod(1 + r) - 1) * 100
    sp = float(np.prod(1 + np.asarray(spy)) - 1) * 100
    sh = float(r.mean() / r.std() * np.sqrt(12)) if r.std() > 1e-9 else 0.0
    eqc = np.cumprod(1 + r); dd = float(((eqc / np.maximum.accumulate(eqc)) - 1).min() * 100)
    t = _tstat_from_returns(list(r))
    return dict(total=round(tot, 1), vs_spy=round(tot - sp, 1), sharpe=round(sh, 2), dd=round(dd, 1),
                t_stat=round(t, 2) if t is not None else None, months=n)


def build():
    etfs = {n: e for n, e in config.SECTOR_ETFS.items() if e not in CRYPTO}
    sector_map, all_holds, name_of = {}, set(), {}
    for n, e in etfs.items():
        h = [t for t in sector_holdings.get_holdings(n) if t not in (e, BENCH) and t not in CRYPTO]
        sector_map[e] = (n, set(h)); all_holds.update(h); name_of[e] = n
    all_holds = sorted(all_holds)
    etf_tk = list(etfs.values())
    print(f"Loading {len(etf_tk)} ETFs + {len(all_holds)} stocks + {BENCH}...", flush=True)
    etf_daily = load_candles(etf_tk + [BENCH])
    etf_m = _monthly_close({t: d for t, d in etf_daily.items() if t in etf_tk})
    midx = etf_m.index
    accel = etf_m.pct_change(3) - etf_m.pct_change(3).shift(3)
    etf_ret = etf_m.pct_change()
    spy_m = etf_daily[BENCH]["Close"].resample("ME").last().reindex(midx)
    stock_daily = load_candles(all_holds)
    stock_m = _monthly_close(stock_daily).reindex(midx)
    reps = load_financial_reports(all_holds)
    sh, eq, ni, dt = (_pit_monthly_panel(reps, f, midx) for f in
                      ("shares_outstanding", "total_equity", "net_income", "total_debt"))
    common = stock_m.columns.intersection(sh.columns).intersection(eq.columns)
    R = lambda p: p.reindex(index=midx, columns=common)
    px = stock_m[common]; sh, eq, ni, dt = R(sh), R(eq), R(ni), R(dt)
    pb = (price_basis.as_traded_close(px) * sh) / eq.where(eq != 0)
    trap = (ni < 0) & (~(eq >= eq.shift(12))) & (~(ni > ni.shift(4)))
    low = (dt / eq.where(eq != 0)) < 1.0
    dvol, adl_m = {}, {}
    for t in common:
        d = stock_daily.get(t)
        if d is None or "Volume" not in d or len(d) < 90:
            continue
        v = d["Volume"]
        dvol[t] = (d["Close"] * v).rolling(20).mean().resample("ME").last().reindex(midx)
        if {"High", "Low", "Close"}.issubset(d.columns):
            rng = (d["High"] - d["Low"]).replace(0, np.nan)
            mfm = ((d["Close"] - d["Low"]) - (d["High"] - d["Close"])) / rng
            adl_m[t] = (mfm.fillna(0) * v).cumsum().resample("ME").last().reindex(midx)
    dvol = pd.DataFrame(dvol).reindex(index=midx, columns=common)
    adl = pd.DataFrame(adl_m).reindex(index=midx, columns=common)
    ad_slope3 = adl - adl.shift(3); px_ret3 = px.pct_change(3)

    # ---- (A) overlap evidence ----
    def jaccard(a, b):
        A, B = sector_map[a][1], sector_map[b][1]
        return len(A & B) / len(A | B) if (A or B) else 0.0
    others = [e for e in etf_tk if e not in SUSPECTS]
    overlap = {}
    print("\n=== (A) OVERLAP of each suspect ETF vs the true sectors ===", flush=True)
    print(f"  {'suspect':<22}{'maxJaccard(with)':<26}{'maxReturnCorr(with)':<26}", flush=True)
    for e in SUSPECTS:
        js = [(jaccard(e, o), o) for o in others if sector_map[o][1]]
        jbest = max(js, key=lambda x: x[0]) if js else (0.0, None)
        cs = []
        for o in others:
            a, b = etf_ret[e], etf_ret[o]
            m = a.notna() & b.notna()
            if m.sum() > 24:
                c = float(np.corrcoef(a[m], b[m])[0, 1])
                if np.isfinite(c):
                    cs.append((c, o))
        cbest = max(cs, key=lambda x: x[0]) if cs else (0.0, None)
        overlap[e] = {"jaccard": round(jbest[0], 2), "jaccard_with": name_of.get(jbest[1]),
                      "ret_corr": round(cbest[0], 2), "ret_corr_with": name_of.get(cbest[1])}
        print(f"  {name_of[e]+' ('+e+')':<22}{f'{jbest[0]:.2f} {name_of.get(jbest[1])}':<26}"
              f"{f'{cbest[0]:.2f} {name_of.get(cbest[1])}':<26}", flush=True)

    # ---- (B) flagship backtest, parametrised by excluded ETFs ----
    def pick(etf, date, held):
        _, holds = sector_map.get(etf, (etf, set()))
        c = [h for h in holds if h in px.columns and h not in held and _available_at(px[h], date)
             and pd.notna(pb.loc[date, h]) and pb.loc[date, h] > 0 and not bool(trap.loc[date, h])
             and pd.notna(dvol.loc[date, h]) and dvol.loc[date, h] >= MIN_DVOL]
        g = [x for x in c if bool(low.loc[date, x])] or c
        return min(g, key=lambda h: pb.loc[date, h]) if g else None

    def wt(name, date):
        a, p = ad_slope3.loc[date].get(name), px_ret3.loc[date].get(name)
        return CONV if (pd.notna(a) and pd.notna(p) and a > 0 and p < 0) else 1.0

    def run(exclude):
        keep = [e for e in etf_tk if e not in set(exclude)]
        ac = accel[keep]
        rets, spies = [], []
        for i in range(9, len(midx) - 1):
            date, ndate = midx[i], midx[i + 1]
            sp = spy_m.iloc[i + 1] / spy_m.iloc[i] - 1
            if not np.isfinite(sp):
                continue
            top = ac.loc[date].dropna().sort_values(ascending=False).head(TOP_N).index
            held = set(); wsum = rr = 0.0
            for etf in top:
                p = pick(etf, date, held)
                if not p:
                    continue
                held.add(p)
                r = _ret_delist(px[p], date, ndate)
                if r is None or not np.isfinite(r):
                    continue
                w = wt(p, date); wsum += w; rr += w * float(r)
            if wsum <= 0:
                continue
            rets.append(rr / wsum); spies.append(float(sp))
        return _perf(rets, spies)

    results = {k: run(v) for k, v in REMOVAL_SETS.items()}
    print(f"\n=== (B) FLAGSHIP with overlap ETFs removed (total / vsSPY / Sharpe / DD / t) ===", flush=True)
    print(f"  {'variant':<22}{'removed':>3}{'total':>10}{'vsSPY':>9}{'Sharpe':>8}{'DD':>9}{'t':>6}", flush=True)
    for k, v in REMOVAL_SETS.items():
        r = results[k]
        print(f"  {k:<22}{len(v):>3}{r['total']:>9}%{r['vs_spy']:>9}{r['sharpe']:>8}{r['dd']:>8}%{str(r['t_stat']):>6}",
              flush=True)

    base = results["baseline"]
    best = max(REMOVAL_SETS, key=lambda k: results[k]["total"])
    best_sh = max(REMOVAL_SETS, key=lambda k: results[k]["sharpe"])
    b, bs = results[best], results[best_sh]
    verdict = (
        f"Baseline {base['total']}%/Sh{base['sharpe']}/DD{base['dd']}%. "
        f"Best total = {best} ({b['total']}%, {b['vs_spy']:+}pp vs SPY, Sh{b['sharpe']}, DD{b['dd']}%); "
        f"best Sharpe = {best_sh} (Sh{bs['sharpe']}, {bs['total']}%). "
        + ("Pruning the overlap 'sectors' IMPROVES the strategy -- they were crowding the top-10 with duplicate "
           "megacap/factor exposure, so removing them is a free upgrade."
           if b["total"] > base["total"] + 10 or bs["sharpe"] > base["sharpe"] + 0.1 else
           "Pruning the overlap 'sectors' does NOT materially help -- the accel ranking already tends to pick the "
           "purer thematic/geo/commodity leaders over the broad baskets, so the overlap is mostly harmless.")
    )
    print("\n" + verdict, flush=True)
    return {
        "computed_at": pd.Timestamp.utcnow().isoformat(),
        "params": {"top_n": TOP_N, "min_dvol": MIN_DVOL, "conviction_mult": CONV, "benchmark": BENCH,
                   "months": int(len(midx)), "pb_basis": "as-traded", "weight": "div_2x",
                   "removal_sets": REMOVAL_SETS},
        "overlap_evidence": overlap, "results": results,
        "best_total": best, "best_sharpe": best_sh, "verdict": verdict,
        "caveat": "PIT, no fees, present-day-holdings survivorship. Holdings-Jaccard uses current constituents "
                  "(sector_holdings), not PIT membership. Removing an ETF only removes it from the RANKING universe; "
                  "its underlying stocks can still be picked via another (non-removed) sector they belong to.",
    }


def main():
    p = build()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(p, indent=2, default=str))
    try:
        from core.models import BacktestResult
        from django.utils import timezone
        BacktestResult.objects.update_or_create(
            kind="sector_overlap", defaults={"payload": json.loads(json.dumps(p, default=str)),
                                             "computed_at": timezone.now()})
        print("Saved BacktestResult[sector_overlap]", flush=True)
    except Exception as e:
        print("DB save failed:", e, flush=True)


if __name__ == "__main__":
    main()
