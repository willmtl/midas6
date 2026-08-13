#!/usr/bin/env python3
"""PORTFOLIO BLENDER — mix the three strategies to raise return-for-risk.

Two sleeves as monthly return streams:
  CORE (C+B): the value engine — top trailing-6mo-momentum sectors -> cheapest positive-P/B pick that
              passes the profit guard AND prefers low debt (the Factor Lab winner). Compounds in trends.
  CAP  (A):   capitulation reversals — each month, buy the DEEP-OVERSOLD names (RSI(10) < 30) that are
              also being ACCUMULATED (A/D line up while price fell = smart-money divergence), equal-weight,
              hold 1 month. Crisis alpha: fires in selloffs, when CORE is drawing down.

The thesis: CORE and CAP are uncorrelated and CAP is anti-correlated in the tail (pays when CORE bleeds),
so a blend has a higher Sharpe -> levered to CORE's volatility it BEATS CORE outright. We measure the
correlation, the crisis-alpha (CAP's return in CORE's down months), sweep static allocations, vol-match
the best blend, and test a regime-switched allocation (more CAP when SPY < 200d MA).
-> BacktestResult[portfolio_blender] + JSON.
Run: MSYS_NO_PATHCONV=1 docker exec rotation-backend-1 python -u /app/portfolio_blender.py  (--limit 250)
"""
import os, sys, json, warnings
warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
os.environ.setdefault("DJANGO_SETTINGS_MODULE", "rotation.settings")
import django
django.setup()

from pathlib import Path
import config, sector_holdings, ta
from studies import _tstat_from_returns
from seq_fundamental_study import load_candles, load_financial_reports
from trend_stock_studies import _pit_monthly_panel, _available_at, _ret_delist, CRYPTO
from backtest_lowpb import _monthly_close, BENCH

OUT = Path(__file__).resolve().parent / ".data" / "studies" / "portfolio_blender.json"
LOOKBACK, TOP_N = 6, 10


def _stats(r, spy):
    r = np.asarray(r, float); n = len(r)
    if n == 0:
        return {"total_return": 0, "vs_spy": 0, "annual_return": 0, "sharpe": 0, "vol": 0,
                "max_drawdown": 0, "t_stat": None, "periods": 0}
    tot = float(np.prod(1 + r) - 1) * 100
    sp = float(np.prod(1 + np.asarray(spy)) - 1) * 100
    ann = (float(np.prod(1 + r)) ** (12.0 / n) - 1) * 100
    vol = float(r.std() * np.sqrt(12)) * 100
    sh = float(r.mean() / r.std() * np.sqrt(12)) if r.std() > 1e-9 else 0.0
    eq = np.cumprod(1 + r); dd = float(((eq / np.maximum.accumulate(eq)) - 1).min() * 100)
    t = _tstat_from_returns(list(r))
    return {"total_return": round(tot, 1), "vs_spy": round(tot - sp, 1), "annual_return": round(ann, 1),
            "sharpe": round(sh, 2), "vol": round(vol, 1), "max_drawdown": round(dd, 1),
            "t_stat": round(t, 2) if t is not None else None, "periods": n}


def _curve(r, idx):
    eq = np.cumprod(1 + np.asarray(r)) if len(r) else []
    return [{"date": str(pd.Timestamp(d).date()), "eq": round(float(v), 4)} for d, v in zip(idx, eq)]


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
    spy_daily = etf_daily[BENCH]
    spy_m = spy_daily["Close"].resample("ME").last().reindex(midx)
    etf_trail = etf_m.pct_change(LOOKBACK)
    stock_daily = load_candles(all_holds)
    stock_m = _monthly_close(stock_daily).reindex(midx)

    reps = load_financial_reports(all_holds)
    shares = _pit_monthly_panel(reps, "shares_outstanding", midx)
    equity = _pit_monthly_panel(reps, "total_equity", midx)
    ni = _pit_monthly_panel(reps, "net_income", midx)
    debt = _pit_monthly_panel(reps, "total_debt", midx)
    common = stock_m.columns.intersection(shares.columns).intersection(equity.columns)

    def R(p):
        return p.reindex(index=midx, columns=common)
    px = stock_m[common]
    shares, equity, ni, debt = map(R, (shares, equity, ni, debt))
    pb = (px * shares) / equity.where(equity != 0)
    book_stable = equity >= equity.shift(12)
    ni_improving = ni > ni.shift(4)
    trap = (ni < 0) & (~book_stable) & (~ni_improving)
    low_debt = (debt / equity.where(equity != 0)) < 1.0

    # RSI + A/D monthly for the capitulation sleeve
    rsi_m, adl_m = {}, {}
    for t in common:
        d = stock_daily.get(t)
        if d is None or len(d) < 60:
            continue
        cl, hi, lo, vol = d["Close"], d["High"], d["Low"], d["Volume"]
        rsi_m[t] = ta.momentum.rsi(cl, window=10).resample("ME").last().reindex(midx)
        rng = (hi - lo).replace(0, np.nan)
        mfm = ((cl - lo) - (hi - cl)) / rng
        adl_m[t] = (mfm.fillna(0) * vol.fillna(0)).cumsum().resample("ME").last().reindex(midx)
    rsi = pd.DataFrame(rsi_m).reindex(index=midx, columns=common)
    adl = pd.DataFrame(adl_m).reindex(index=midx, columns=common)
    ad_rising = (adl > adl.shift(1)) & (px.pct_change(1) < 0)      # accumulation while price fell (1mo)
    oversold = rsi < 30
    cap_setup = oversold & ad_rising
    CAP_HOLD_D = 10        # capitulation bounce is short-horizon: ~2 trading weeks, not a full month

    def _fwd_days(t, date, ndays=CAP_HOLD_D):
        d = stock_daily.get(t)
        if d is None:
            return None
        sub = d["Close"][d["Close"].index >= date]
        if len(sub) < ndays + 1 or sub.iloc[0] == 0:
            return None
        v = float(sub.iloc[ndays] / sub.iloc[0] - 1)
        return v if np.isfinite(v) else None

    warmup = max(LOOKBACK, 1)
    idx_used = []
    core_r, cap_r, spy_r = [], [], []
    cap_names = []

    for i in range(warmup, len(midx) - 1):
        date, ndate = midx[i], midx[i + 1]
        sp = spy_m.iloc[i + 1] / spy_m.iloc[i] - 1
        if not np.isfinite(sp):
            continue
        # CORE (C+B): guard + low_debt value pick per top-momentum sector
        ranks = etf_trail.loc[date].dropna().sort_values(ascending=False).head(TOP_N).index
        cslot = []
        for etf in ranks:
            _, holds = sector_map.get(etf, (etf, []))
            cands = [h for h in holds if h in px.columns and _available_at(px[h], date)
                     and h in pb.columns and pd.notna(pb.loc[date, h]) and pb.loc[date, h] > 0]
            guarded = [c for c in cands if c in trap.columns and not bool(trap.loc[date, c])]
            ld = [c for c in guarded if c in low_debt.columns and bool(low_debt.loc[date, c])]
            use = ld or guarded or cands
            if not use:
                continue
            pick = pb.loc[date, use].idxmin()
            r = _ret_delist(px[pick], date, ndate)
            if r is not None and np.isfinite(r):
                cslot.append(float(r))
        if not cslot:
            continue
        # CAP (A): all capitulation-with-accumulation names this month, equal-weight
        setup_row = cap_setup.loc[date]
        aset = [t for t in setup_row[setup_row].index if _available_at(px[t], date)]
        aret = []
        for t in aset:
            r = _fwd_days(t, date)          # ~10-trading-day bounce, then cash
            if r is not None:
                aret.append(float(r))
        idx_used.append(ndate)
        core_r.append(float(np.mean(cslot)))
        cap_r.append(float(np.mean(aret)) if aret else 0.0)     # no setup -> cash that month
        spy_r.append(float(sp))
        cap_names.append(len(aret))

    core = np.array(core_r); cap = np.array(cap_r); spy = np.array(spy_r)
    corr = float(np.corrcoef(core, cap)[0, 1]) if len(core) > 2 else None
    # crisis alpha: CAP behaviour in CORE's down months vs up months
    dn = core < 0
    crisis = {"core_down_months": int(dn.sum()),
              "cap_mean_when_core_down_pct": round(float(cap[dn].mean()) * 100, 2) if dn.any() else None,
              "cap_mean_when_core_up_pct": round(float(cap[~dn].mean()) * 100, 2) if (~dn).any() else None,
              "core_mean_when_core_down_pct": round(float(core[dn].mean()) * 100, 2) if dn.any() else None}

    # static allocation sweep
    sweep = []
    for w in [round(x, 2) for x in np.arange(0, 1.01, 0.1)]:
        b = (1 - w) * core + w * cap
        s = _stats(b, spy); s["wA"] = w
        sweep.append(s)
    best_ret = max(sweep, key=lambda s: s["total_return"])
    best_sh = max(sweep, key=lambda s: s["sharpe"])
    core_only = sweep[0]

    # vol-match: lever the best-Sharpe blend to CORE's volatility -> same-risk return comparison
    bw = best_sh["wA"]
    blend = (1 - bw) * core + bw * cap
    lev = (core.std() / blend.std()) if blend.std() > 1e-9 else 1.0
    levered = blend * lev
    levered_stats = _stats(levered, spy); levered_stats["lever"] = round(float(lev), 2); levered_stats["wA"] = bw

    # regime-switched allocation: more CAP when SPY < 200d MA (capitulation regime)
    bull = (spy_daily["Close"] > spy_daily["Close"].rolling(200).mean()).reindex(midx, method="ffill")
    bull_used = np.array([bool(bull.loc[d]) if d in bull.index else True for d in idx_used])
    wA_dyn = np.where(bull_used, 0.15, 0.40)
    regime_blend = (1 - wA_dyn) * core + wA_dyn * cap
    regime_stats = _stats(regime_blend, spy)
    regime_lev = (core.std() / regime_blend.std()) if regime_blend.std() > 1e-9 else 1.0
    regime_levered = _stats(regime_blend * regime_lev, spy); regime_levered["lever"] = round(float(regime_lev), 2)

    payload = {
        "computed_at": pd.Timestamp.utcnow().isoformat(),
        "params": {"lookback_months": LOOKBACK, "top_n_sectors": TOP_N, "benchmark": BENCH,
                   "months": int(len(idx_used)), "limit": limit,
                   "core": "C+B: guard+low_debt cheapest-P/B value pick in top-momentum sectors",
                   "cap": "A: RSI(10)<30 AND A/D-rising (accumulation) names, equal-weight, 1mo hold"},
        "sleeves": {"core": {**_stats(core, spy), "curve": _curve(core, idx_used)},
                    "cap": {**_stats(cap, spy), "curve": _curve(cap, idx_used),
                            "avg_names": round(float(np.mean(cap_names)), 1) if cap_names else 0},
                    "spy": {"curve": _curve(spy, idx_used)}},
        "correlation_core_cap": round(corr, 3) if corr is not None else None,
        "crisis_alpha": crisis,
        "allocation_sweep": sweep,
        "best_return_blend": best_ret, "best_sharpe_blend": best_sh, "core_only": core_only,
        "vol_matched_best": levered_stats,
        "regime_switched": {"static": regime_stats, "vol_matched": regime_levered,
                            "rule": "wA=0.15 risk-on (SPY>200d), wA=0.40 risk-off"},
        "blend_curve_best_sharpe": _curve(blend, idx_used),
        "verdict": None,
        "caveat": ("CORE + CAP as monthly return streams (CAP is naturally higher-frequency -> monthly "
                   "understates it). PIT selection; directional/no-fees; ~5y single regime. Vol-matched "
                   "'levered' = scale blend to CORE's vol to compare same-risk return (leverage assumed "
                   "free/available). CAP 1mo A/D proxy, not the strict daily entry."),
    }
    # verdict
    beat = levered_stats["total_return"] > core_only["total_return"]
    payload["verdict"] = (
        f"CORE alone {core_only['total_return']}% (Sharpe {core_only['sharpe']}); CORE+CAP corr {payload['correlation_core_cap']}; "
        f"CAP returns {crisis['cap_mean_when_core_down_pct']}%/mo when CORE is DOWN vs "
        f"{crisis['cap_mean_when_core_up_pct']}%/mo when up. Best-Sharpe blend wA={best_sh['wA']} "
        f"(Sharpe {best_sh['sharpe']} vs {core_only['sharpe']}); vol-matched to CORE it returns "
        f"{levered_stats['total_return']}% ({'BEATS' if beat else 'does NOT beat'} CORE's {core_only['total_return']}%). "
        f"Regime-switched vol-matched: {regime_levered['total_return']}%.")
    return payload


def main():
    payload = build()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(payload, indent=2, default=str))
    try:
        from core.models import BacktestResult
        from django.utils import timezone
        BacktestResult.objects.update_or_create(
            kind="portfolio_blender",
            defaults={"payload": json.loads(json.dumps(payload, default=str)), "computed_at": timezone.now()})
        print("Saved BacktestResult[portfolio_blender]", flush=True)
    except Exception as e:
        print("DB save failed:", e, flush=True)
    s = payload["sleeves"]
    print(f"\n=== PORTFOLIO BLENDER ({payload['params']['months']} months) ===", flush=True)
    print(f"CORE  total {s['core']['total_return']}%  vsSPY {s['core']['vs_spy']}%  Sh {s['core']['sharpe']}  DD {s['core']['max_drawdown']}%  vol {s['core']['vol']}%", flush=True)
    print(f"CAP   total {s['cap']['total_return']}%  vsSPY {s['cap']['vs_spy']}%  Sh {s['cap']['sharpe']}  DD {s['cap']['max_drawdown']}%  vol {s['cap']['vol']}%  names {s['cap']['avg_names']}", flush=True)
    print(f"corr(CORE,CAP) = {payload['correlation_core_cap']}", flush=True)
    c = payload["crisis_alpha"]
    print(f"CRISIS ALPHA: CAP {c['cap_mean_when_core_down_pct']}%/mo when CORE down ({c['core_down_months']}mo) vs {c['cap_mean_when_core_up_pct']}%/mo when up", flush=True)
    print("\nallocation sweep (wA = capitulation weight):", flush=True)
    for st in payload["allocation_sweep"]:
        print(f"  wA {st['wA']:.1f}  total {st['total_return']:>7}%  Sh {st['sharpe']:>5}  DD {st['max_drawdown']:>6}%  vol {st['vol']:>5}%", flush=True)
    print("\n" + payload["verdict"], flush=True)


if __name__ == "__main__":
    main()
