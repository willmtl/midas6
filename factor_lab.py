#!/usr/bin/env python3
"""FACTOR LAB — sweep many filters / tilts / combos on the value-pick baseline to find the best return.

Baseline = arm3_lowpb: top trailing-6mo-momentum sectors -> cheapest positive-P/B pick (monthly,
equal-weight, PIT). We then test candidate factors, each as either a candidate FILTER (restrict the
set, still pick cheapest P/B), a RANKING change (pick by a different value metric), or a WEIGHTING
change. All PIT from FinancialReport (net_income/total_equity/revenue/free_cash_flow/total_debt/
gross_profit/shares) + candles (A/D accumulation divergence, momentum, vol, market cap).

Reports vs-SPY / t / Sharpe / drawdown for every variant (no-fallback = pure picks; fallback = ETF
for empty sectors), ranks them, and flags the best return + best risk-adjusted + best stacked combo.
-> BacktestResult[factor_lab] + JSON.
Run: MSYS_NO_PATHCONV=1 docker exec rotation-backend-1 python -u /app/factor_lab.py  (--limit 200 quick)
"""
import os, sys, json, warnings
warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
os.environ.setdefault("DJANGO_SETTINGS_MODULE", "rotation.settings")
import django
django.setup()

from pathlib import Path
import config, sector_holdings
from studies import _tstat_from_returns
from seq_fundamental_study import load_candles, load_financial_reports
from trend_stock_studies import _pit_monthly_panel, _available_at, _ret_delist, CRYPTO
from backtest_lowpb import _monthly_close, BENCH

OUT = Path(__file__).resolve().parent / ".data" / "studies" / "factor_lab.json"
LOOKBACK, TOP_N = 6, 10


def _stats(rets, spy):
    r = np.array(rets, float); n = len(r)
    if n == 0:
        return {"total_return": 0, "vs_spy": 0, "sharpe": 0, "max_drawdown": 0, "t_stat": None, "periods": 0}
    tot = float(np.prod(1 + r) - 1) * 100
    sp = float(np.prod(1 + np.array(spy)) - 1) * 100
    sh = float(r.mean() / r.std() * np.sqrt(12)) if r.std() > 1e-9 else 0.0
    eq = np.cumprod(1 + r); dd = float(((eq / np.maximum.accumulate(eq)) - 1).min() * 100)
    t = _tstat_from_returns(list(r))
    return {"total_return": round(tot, 1), "vs_spy": round(tot - sp, 1), "sharpe": round(sh, 2),
            "max_drawdown": round(dd, 1), "t_stat": round(t, 2) if t is not None else None, "periods": n}


def build():
    etfs = {n: e for n, e in config.SECTOR_ETFS.items() if e not in CRYPTO}
    limit = int(sys.argv[sys.argv.index("--limit") + 1]) if "--limit" in sys.argv else None
    sector_map, all_holds = {}, set()
    for n, e in etfs.items():
        h = [t for t in sector_holdings.get_holdings(n) if t not in (e, BENCH) and t not in CRYPTO]
        sector_map[e] = (n, h); all_holds.update(h)
    all_holds = sorted(all_holds)
    if limit:
        all_holds = all_holds[:limit]; hs = set(all_holds)
        sector_map = {e: (n, [h for h in hh if h in hs]) for e, (n, hh) in sector_map.items()}

    etf_tk = list(etfs.values())
    print(f"Loading {len(etf_tk)} ETFs + {len(all_holds)} stocks + {BENCH}...", flush=True)
    etf_daily = load_candles(etf_tk + [BENCH])
    etf_m = _monthly_close({t: d for t, d in etf_daily.items() if t in etf_tk})
    midx = etf_m.index
    spy_m = etf_daily[BENCH]["Close"].resample("ME").last().reindex(midx)
    etf_trail = etf_m.pct_change(LOOKBACK)
    stock_daily = load_candles(all_holds)
    stock_m = _monthly_close(stock_daily).reindex(midx)

    reps = load_financial_reports(all_holds)
    shares = _pit_monthly_panel(reps, "shares_outstanding", midx)
    equity = _pit_monthly_panel(reps, "total_equity", midx)
    ni = _pit_monthly_panel(reps, "net_income", midx)
    rev = _pit_monthly_panel(reps, "revenue", midx)
    fcf = _pit_monthly_panel(reps, "free_cash_flow", midx)
    debt = _pit_monthly_panel(reps, "total_debt", midx)
    gross = _pit_monthly_panel(reps, "gross_profit", midx)
    common = stock_m.columns.intersection(shares.columns).intersection(equity.columns)

    def R(p):
        return p.reindex(index=midx, columns=common)
    px = stock_m[common]
    shares, equity, ni, rev, fcf, debt, gross = map(R, (shares, equity, ni, rev, fcf, debt, gross))
    pb = (px * shares) / equity.where(equity != 0)
    mktcap = px * shares

    # ---- candle-derived panels (A/D accumulation, momentum, vol) ----
    adl_m, vol_m = {}, {}
    for t in common:
        d = stock_daily.get(t)
        if d is None or len(d) < 60:
            continue
        hi, lo, cl, vol = d["High"], d["Low"], d["Close"], d["Volume"]
        rng = (hi - lo).replace(0, np.nan)
        mfm = ((cl - lo) - (hi - cl)) / rng
        adl = (mfm.fillna(0) * vol.fillna(0)).cumsum()
        adl_m[t] = adl.resample("ME").last().reindex(midx)
        vol_m[t] = cl.pct_change().rolling(60).std().resample("ME").last().reindex(midx)
    adl = pd.DataFrame(adl_m).reindex(index=midx, columns=common)
    volp = pd.DataFrame(vol_m).reindex(index=midx, columns=common)
    px_chg3 = px.pct_change(3)

    # ---- factor masks (True = candidate is eligible) ----
    book_stable = equity >= equity.shift(12)
    ni_improving = ni > ni.shift(4)
    trap = (ni < 0) & (~book_stable) & (~ni_improving)
    M = {
        "profit_guard": ~trap,                                   # ex_trap_turn (our known win)
        "fcf_positive": fcf > 0,
        "low_debt": (debt / equity.where(equity != 0)) < 1.0,
        "gross_margin": (gross / rev.where(rev != 0)) >= 0.30,
        "rev_growth": rev > rev.shift(12),                       # revenue YoY > 0
        "ad_rising": (adl > adl.shift(3)) & (px_chg3 <= 0),      # A/D up while price flat/down (divergence)
        "small_cap": mktcap < 2e9,
        "micro_cap": mktcap < 5e8,
        "mom_pos": px.pct_change(6) > 0,                         # stock in its own uptrend
        "earn_yield": ((ni * 4) / mktcap.where(mktcap != 0)) > 0.05,  # annualized earnings yield > 5%
    }
    # value-ranking panels (lower = pick)
    fcf_yield = (fcf / mktcap.where(mktcap != 0))
    de = (debt / equity.where(equity != 0))

    def _z(p):
        return (p.sub(p.mean(axis=1), axis=0)).div(p.std(axis=1).replace(0, np.nan), axis=0)
    composite_rank = _z(pb) - _z(fcf_yield) + 0.5 * _z(de)      # cheap P/B + high FCF-yield + low debt

    print(f"months {len(midx)} | stocks {len(common)} | factors {len(M)}", flush=True)
    warmup = max(LOOKBACK, 1)

    def _run(masks=(), rankpanel=None, invvol=False):
        def keep(date, t):
            for m in masks:
                col = M[m]
                if t not in col.columns or not bool(col.loc[date, t]):
                    return False
            return True
        rp = rankpanel if rankpanel is not None else pb
        rets_nf, spies_nf, rets_fb, spies_fb, nnames = [], [], [], [], []
        for i in range(warmup, len(midx) - 1):
            date, ndate = midx[i], midx[i + 1]
            sp = spy_m.iloc[i + 1] / spy_m.iloc[i] - 1
            if not np.isfinite(sp):
                continue
            ranks = etf_trail.loc[date].dropna().sort_values(ascending=False).head(TOP_N).index
            slot_nf, slot_fb, w_nf = [], [], []
            for etf in ranks:
                _, holds = sector_map.get(etf, (etf, []))
                cands = [h for h in holds if h in px.columns and _available_at(px[h], date)
                         and h in pb.columns and pd.notna(pb.loc[date, h]) and pb.loc[date, h] > 0]
                cands = [c for c in cands if keep(date, c)]
                pick = None
                if cands:
                    row = rp.loc[date, [c for c in cands if pd.notna(rp.loc[date, c])]]
                    if len(row):
                        pick = row.idxmin()
                if pick is not None:
                    r = _ret_delist(px[pick], date, ndate)
                    if r is not None and np.isfinite(r):
                        w = (1.0 / volp.loc[date, pick]) if (invvol and pd.notna(volp.loc[date, pick])
                                                             and volp.loc[date, pick] > 0) else 1.0
                        slot_nf.append(float(r)); w_nf.append(float(w)); slot_fb.append(float(r)); continue
                er = _ret_delist(etf_m[etf], date, ndate) if etf in etf_m.columns else None
                if er is not None and np.isfinite(er):
                    slot_fb.append(float(er))
            if slot_nf:
                w = np.array(w_nf); w = w / w.sum()
                rets_nf.append(float(np.dot(w, slot_nf))); spies_nf.append(float(sp)); nnames.append(len(slot_nf))
            if slot_fb:
                rets_fb.append(float(np.mean(slot_fb))); spies_fb.append(float(sp))
        nf = _stats(rets_nf, spies_nf); fb = _stats(rets_fb, spies_fb)
        nf["avg_names"] = round(float(np.mean(nnames)), 1) if nnames else 0
        return {"no_fallback": nf, "fallback": fb}

    # ---- variants: baseline, each single factor, ranking, weighting, and stacked combos ----
    variants = {"baseline": _run()}
    for m in M:
        variants[m] = _run(masks=(m,))
    variants["composite_value"] = _run(rankpanel=composite_rank)
    variants["invvol_weight"] = _run(invvol=True)
    combos = {
        "guard+lowdebt": ("profit_guard", "low_debt"),
        "guard+ad": ("profit_guard", "ad_rising"),
        "guard+fcf": ("profit_guard", "fcf_positive"),
        "guard+small": ("profit_guard", "small_cap"),
        "guard+mom": ("profit_guard", "mom_pos"),
        "guard+revgrow": ("profit_guard", "rev_growth"),
        "guard+lowdebt+revgrow": ("profit_guard", "low_debt", "rev_growth"),
        "quality(guard+fcf+lowdebt)": ("profit_guard", "fcf_positive", "low_debt"),
    }
    for name, ms in combos.items():
        variants[name] = _run(masks=ms)
    # weighting stacks on the two best quality factors
    variants["guard+lowdebt+invvol"] = _run(masks=("profit_guard", "low_debt"), invvol=True)
    variants["lowdebt+invvol"] = _run(masks=("low_debt",), invvol=True)

    base = variants["baseline"]["no_fallback"]["vs_spy"]
    for v in variants.values():
        v["vs_baseline"] = round(v["no_fallback"]["vs_spy"] - base, 1)
    ranked = sorted([k for k in variants if k != "baseline"],
                    key=lambda k: variants[k]["no_fallback"]["vs_spy"], reverse=True)
    best_return = ranked[0]
    best_riskadj = max([k for k in variants if k != "baseline"],
                       key=lambda k: variants[k]["no_fallback"]["sharpe"])

    payload = {
        "computed_at": pd.Timestamp.utcnow().isoformat(),
        "params": {"lookback_months": LOOKBACK, "top_n_sectors": TOP_N, "benchmark": BENCH,
                   "months": int(len(midx)), "limit": limit,
                   "selection": "arm3_lowpb baseline; each variant adds a filter/tilt/weighting"},
        "baseline": "baseline", "best_return": best_return, "best_risk_adjusted": best_riskadj,
        "ranking": ranked, "variants": variants,
        "factor_legend": {
            "profit_guard": "exclude value traps (unprofitable+eroding book+not improving), keep turnarounds",
            "fcf_positive": "free cash flow > 0", "low_debt": "debt/equity < 1",
            "gross_margin": "gross margin >= 30%", "rev_growth": "revenue YoY > 0",
            "ad_rising": "A/D accumulation line up while price flat/down (smart-money divergence)",
            "small_cap": "market cap < $2B", "micro_cap": "market cap < $500M",
            "mom_pos": "stock's own 6mo return > 0", "earn_yield": "annualized earnings yield > 5%",
            "composite_value": "rank by cheap P/B + high FCF-yield + low debt (not just P/B)",
            "invvol_weight": "inverse-volatility weighting instead of equal-weight"},
        "caveat": ("PIT from FinancialReport + candles; directional/no-fees; ~5y single regime. MANY variants "
                   "tested -> multiple-comparisons hazard: trust a variant only if it beats baseline on vs-SPY "
                   "AND t/Sharpe AND its factor family is coherent. A/D/rev-growth/book use ~12mo/2q proxies."),
    }
    return payload


def _line(k, v):
    nf = v["no_fallback"]
    return (f"  {k:26} vsSPY {nf['vs_spy']:>7}%  t={str(nf['t_stat']):>5}  Sh {nf['sharpe']:>5}  "
            f"DD {nf['max_drawdown']:>6}%  names {nf['avg_names']:>4}  | fb {v['fallback']['vs_spy']:>7}%")


def main():
    payload = build()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(payload, indent=2, default=str))
    try:
        from core.models import BacktestResult
        from django.utils import timezone
        BacktestResult.objects.update_or_create(
            kind="factor_lab",
            defaults={"payload": json.loads(json.dumps(payload, default=str)), "computed_at": timezone.now()})
        print("Saved BacktestResult[factor_lab]", flush=True)
    except Exception as e:
        print("DB save failed:", e, flush=True)
    print(f"\n=== FACTOR LAB (baseline vsSPY {payload['variants']['baseline']['no_fallback']['vs_spy']}%) ===", flush=True)
    print(f"BEST RETURN = {payload['best_return']} | BEST RISK-ADJ = {payload['best_risk_adjusted']}\n", flush=True)
    print(_line("baseline", payload["variants"]["baseline"]), flush=True)
    for k in payload["ranking"]:
        print(_line(k, payload["variants"][k]), flush=True)


if __name__ == "__main__":
    main()
