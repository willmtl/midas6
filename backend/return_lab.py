#!/usr/bin/env python3
"""RETURN LAB — push the validated engine (rotation + guard + low_debt + cheapest-P/B) for MORE RETURN
across all 4 levers, honestly, on one data load. Base = +313% total / +229% vs SPY / DD -16.5%.

  L1 CONCENTRATION+WEIGHTING  top-N sectors {3,5,7,10} x weight {equal,momentum,cheapness,inv_vol}
  L2 LEVERAGE                 base x {1.0,1.5,2.0,3.0} (minus crude financing on the borrowed part)
  L3 ORTHOGONAL EDGE          core vs momentum-pick vs breadth sleeves; correlation + best blend
  L4 DRAWDOWN / REGIME        SPY-above-200d-MA and SPY-above-10mo-MA risk-off overlays (hard + soft)
  COMBO                       best concentration + regime overlay stacked

Everything monthly, PIT, equal unless noted, vs SPY. Each variant reports total_return / vs_spy / t /
Sharpe / max_drawdown so the return-vs-risk frontier is explicit (concentration & leverage buy return
WITH drawdown; only orthogonality or drawdown-avoidance can raise RISK-ADJUSTED return).
-> BacktestResult[return_lab] + JSON.
Run: MSYS_NO_PATHCONV=1 docker exec rotation-backend-1 python -u /app/return_lab.py
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
import price_basis

OUT = Path(__file__).resolve().parent / ".data" / "studies" / "return_lab.json"
LOOKBACK, TOP_N_MAX = 6, 10
RF_M = 0.0033          # ~4%/yr monthly risk-free, charged on the borrowed part for leverage


def _stats(r, spy):
    r = np.asarray(r, float); n = len(r)
    if n == 0:
        return dict(total_return=0, vs_spy=0, sharpe=0, max_drawdown=0, t_stat=None, periods=0, ann=0)
    tot = float(np.prod(1 + r) - 1) * 100
    sp = float(np.prod(1 + np.asarray(spy)) - 1) * 100
    ann = (float(np.prod(1 + r)) ** (12.0 / n) - 1) * 100
    sh = float(r.mean() / r.std() * np.sqrt(12)) if r.std() > 1e-9 else 0.0
    eq = np.cumprod(1 + r); dd = float(((eq / np.maximum.accumulate(eq)) - 1).min() * 100)
    t = _tstat_from_returns(list(r))
    return dict(total_return=round(tot, 1), vs_spy=round(tot - sp, 1), ann=round(ann, 1),
                sharpe=round(sh, 2), max_drawdown=round(dd, 1),
                t_stat=round(t, 2) if t is not None else None, periods=n)


def build():
    etfs = {n: e for n, e in config.SECTOR_ETFS.items() if e not in CRYPTO}
    sector_map, all_holds = {}, set()
    for n, e in etfs.items():
        h = [t for t in sector_holdings.get_holdings(n) if t not in (e, BENCH) and t not in CRYPTO]
        sector_map[e] = (n, h); all_holds.update(h)
    all_holds = sorted(all_holds)

    etf_tk = list(etfs.values())
    print(f"Loading {len(etf_tk)} ETFs + {len(all_holds)} stocks + {BENCH}...", flush=True)
    etf_daily = load_candles(etf_tk + [BENCH])
    etf_m = _monthly_close({t: d for t, d in etf_daily.items() if t in etf_tk})
    midx = etf_m.index
    spy_close = etf_daily[BENCH]["Close"]
    spy_m = spy_close.resample("ME").last().reindex(midx)
    etf_trail = etf_m.pct_change(LOOKBACK)
    stock_daily = load_candles(all_holds)
    stock_m = _monthly_close(stock_daily).reindex(midx)

    reps = load_financial_reports(all_holds)
    P = lambda f: _pit_monthly_panel(reps, f, midx)
    shares, equity, ni, debt = P("shares_outstanding"), P("total_equity"), P("net_income"), P("total_debt")
    common = stock_m.columns.intersection(shares.columns).intersection(equity.columns)

    def R(p):
        return p.reindex(index=midx, columns=common)
    px = stock_m[common]
    shares, equity, ni, debt = map(R, (shares, equity, ni, debt))
    pb = (price_basis.as_traded_close(px) * shares) / equity.where(equity != 0)
    trap = (ni < 0) & (~(equity >= equity.shift(12))) & (~(ni > ni.shift(4)))
    low_debt = (debt / equity.where(equity != 0)) < 1.0
    stock_mom = px.pct_change(LOOKBACK)

    # 60d realized vol per stock, monthly (for inverse-vol weighting)
    vol_m = {}
    for t in common:
        d = stock_daily.get(t)
        if d is None or len(d) < 70:
            continue
        vol_m[t] = d["Close"].pct_change().rolling(60).std().resample("ME").last().reindex(midx)
    volp = pd.DataFrame(vol_m).reindex(index=midx, columns=common)

    # regime filters: SPY above its 200d MA (daily->monthly) and above its 10-month MA
    spy_ma200 = (spy_close > spy_close.rolling(200).mean()).resample("ME").last().reindex(midx).fillna(False)
    spy_ma10 = (spy_m > spy_m.rolling(10).mean())

    print(f"months {len(midx)} | stocks {len(common)}", flush=True)
    warmup = max(LOOKBACK, 1)

    # ---- per-month pick table for the top-10 momentum sectors -----------------
    # rows[i] = list of dicts {etf, mom, core_pick, core_ret, core_pb, core_vol, mom_pick, mom_ret,
    #                          breadth_ret} ordered by sector momentum desc
    rows, spies = [], []
    for i in range(warmup, len(midx) - 1):
        date, ndate = midx[i], midx[i + 1]
        sp = spy_m.iloc[i + 1] / spy_m.iloc[i] - 1
        if not np.isfinite(sp):
            rows.append(None); spies.append(np.nan); continue
        ranks = etf_trail.loc[date].dropna().sort_values(ascending=False).head(TOP_N_MAX).index
        month = []
        for etf in ranks:
            _, holds = sector_map.get(etf, (etf, []))
            cands = [h for h in holds if h in px.columns and _available_at(px[h], date)
                     and pd.notna(pb.loc[date, h]) and pb.loc[date, h] > 0 and not bool(trap.loc[date, h])]
            ld = [c for c in cands if bool(low_debt.loc[date, c])]
            use = ld or cands
            if not use:
                continue
            core = pb.loc[date, use].idxmin()
            momc = [c for c in use if pd.notna(stock_mom.loc[date, c])]
            mpick = stock_mom.loc[date, momc].idxmax() if momc else core

            def rr(t):
                v = _ret_delist(px[t], date, ndate)
                return float(v) if (v is not None and np.isfinite(v)) else None
            core_ret, mom_ret = rr(core), rr(mpick)
            brt = [rr(c) for c in use]; brt = [x for x in brt if x is not None]
            month.append(dict(
                etf=etf, mom=float(etf_trail.loc[date, etf]),
                core_pick=core, core_ret=core_ret,
                core_pb=float(pb.loc[date, core]),
                core_vol=(float(volp.loc[date, core]) if pd.notna(volp.loc[date, core]) else None),
                mom_pick=mpick, mom_ret=mom_ret,
                breadth_ret=(float(np.mean(brt)) if brt else None)))
        rows.append(month); spies.append(float(sp))

    valid = [j for j in range(len(rows)) if rows[j] and np.isfinite(spies[j])]
    spy_series = np.array([spies[j] for j in valid])
    spy_ma200_v = [bool(spy_ma200.iloc[warmup + j]) for j in range(len(rows))]
    spy_ma10_v = [bool(spy_ma10.iloc[warmup + j]) if pd.notna(spy_ma10.iloc[warmup + j]) else True
                  for j in range(len(rows))]

    def weights(month, n, scheme):
        m = month[:n]
        if not m:
            return [], []
        picks = [d for d in m if d["core_ret"] is not None]
        if not picks:
            return [], []
        if scheme == "equal":
            w = np.ones(len(picks))
        elif scheme == "momentum":
            w = np.array([max(d["mom"], 0.0) for d in picks]); w = w if w.sum() > 0 else np.ones(len(picks))
        elif scheme == "cheapness":
            w = np.array([1.0 / d["core_pb"] if d["core_pb"] > 0 else 0.0 for d in picks])
            w = w if w.sum() > 0 else np.ones(len(picks))
        elif scheme == "inv_vol":
            w = np.array([1.0 / d["core_vol"] if d["core_vol"] and d["core_vol"] > 0 else np.nan for d in picks])
            if np.isnan(w).all():
                w = np.ones(len(picks))
            else:
                w = np.where(np.isnan(w), np.nanmedian(w), w)
        w = w / w.sum()
        return [d["core_ret"] for d in picks], w

    def series_conc(n, scheme):
        out = []
        for j in valid:
            r, w = weights(rows[j], n, scheme)
            out.append(float(np.dot(w, r)) if len(r) else np.nan)
        return np.array(out)

    def base_series(key="core_ret", n=TOP_N_MAX):
        out = []
        for j in valid:
            vals = [d[key] for d in rows[j][:n] if d[key] is not None]
            out.append(float(np.mean(vals)) if vals else np.nan)
        return np.array(out)

    def clean(a):
        m = np.isfinite(a) & np.isfinite(spy_series)
        return a[m], spy_series[m]

    core10 = base_series("core_ret", TOP_N_MAX)

    # ===== L1 CONCENTRATION + WEIGHTING =====
    L1 = {}
    for n in (3, 5, 7, 10):
        for scheme in ("equal", "momentum", "cheapness", "inv_vol"):
            a, s = clean(series_conc(n, scheme))
            L1[f"top{n}_{scheme}"] = _stats(a, s)

    # ===== L2 LEVERAGE (on top-10 equal base) =====
    L2 = {}
    a0, s0 = clean(core10)
    for L in (1.0, 1.5, 2.0, 3.0):
        lev = a0 * L - (L - 1.0) * RF_M          # borrow the extra exposure at RF
        L2[f"{L:.1f}x"] = _stats(lev, s0)

    # ===== L3 ORTHOGONAL EDGE (core vs momentum-pick vs breadth) =====
    mom10 = base_series("mom_ret", TOP_N_MAX)
    brd10 = base_series("breadth_ret", TOP_N_MAX)
    ac, sc = clean(core10); am, _ = clean(mom10 if len(mom10) == len(core10) else mom10)
    # align all three on common finite mask
    mask = np.isfinite(core10) & np.isfinite(mom10) & np.isfinite(brd10) & np.isfinite(spy_series)
    C, M, Bd, S = core10[mask], mom10[mask], brd10[mask], spy_series[mask]
    corr = lambda x, y: float(np.corrcoef(x, y)[0, 1]) if len(x) > 2 else None
    L3 = {"sleeves": {"core": _stats(C, S), "momentum_pick": _stats(M, S), "breadth": _stats(Bd, S)},
          "corr": {"core_vs_momentum": round(corr(C, M), 2), "core_vs_breadth": round(corr(C, Bd), 2),
                   "momentum_vs_breadth": round(corr(M, Bd), 2)},
          "blends_core_momentum": {}, "blends_core_breadth": {}}
    for w in (0.75, 0.5, 0.25):
        L3["blends_core_momentum"][f"{int(w*100)}c/{int((1-w)*100)}m"] = _stats(w * C + (1 - w) * M, S)
        L3["blends_core_breadth"][f"{int(w*100)}c/{int((1-w)*100)}b"] = _stats(w * C + (1 - w) * Bd, S)

    # ===== L4 DRAWDOWN / REGIME OVERLAY (on top-10 equal base) =====
    def overlay(a, flags, floor):
        exp = np.array([1.0 if flags[valid_idx] else floor for valid_idx in range(len(a))])
        # flags indexed by position in `valid`
        return a * exp + (1 - exp) * RF_M
    flags200 = [spy_ma200_v[j] for j in valid]
    flags10 = [spy_ma10_v[j] for j in valid]
    L4 = {}
    for name, flags in (("spy_200d", flags200), ("spy_10mo", flags10)):
        for floor, tag in ((0.0, "cash"), (0.5, "half")):
            exp = np.array([1.0 if f else floor for f in flags])
            lev = core10 * exp + (1 - exp) * RF_M
            a, s = clean(lev)
            L4[f"{name}_{tag}"] = _stats(a, s)
            L4[f"{name}_{tag}"]["months_risk_off"] = int(sum(1 for f in flags if not f))

    # ===== COMBO: best concentration + best regime overlay =====
    best_conc = max(L1, key=lambda k: L1[k]["total_return"])
    n_b = int(best_conc.split("_")[0][3:]); sch_b = best_conc.split("_", 1)[1]
    conc_series = series_conc(n_b, sch_b)
    exp = np.array([1.0 if spy_ma200_v[j] else 0.0 for j in valid])
    combo = conc_series * exp + (1 - exp) * RF_M
    a, s = clean(combo)
    COMBO = {"recipe": f"{best_conc} + spy_200d_cash overlay", "stats": _stats(a, s)}

    base = _stats(*clean(core10))

    # --- SELF-CHECK: re-run the EXACT value_ranking pb logic; base must match (~313%/+229%) ---
    def _vr_pb():
        rr, ss = [], []
        for i in range(warmup, len(midx) - 1):
            date, ndate = midx[i], midx[i + 1]
            sp = spy_m.iloc[i + 1] / spy_m.iloc[i] - 1
            if not np.isfinite(sp):
                continue
            ranks = etf_trail.loc[date].dropna().sort_values(ascending=False).head(TOP_N_MAX).index
            slot = []
            for etf in ranks:
                _, holds = sector_map.get(etf, (etf, []))
                cands = [h for h in holds if h in px.columns and _available_at(px[h], date)
                         and pd.notna(pb.loc[date, h]) and pb.loc[date, h] > 0 and not bool(trap.loc[date, h])]
                ld = [c for c in cands if bool(low_debt.loc[date, c])]
                use = ld or cands
                if not use:
                    continue
                pick = pb.loc[date, use].idxmin()
                r = _ret_delist(px[pick], date, ndate)
                if r is not None and np.isfinite(r):
                    slot.append(float(r))
            if slot:
                rr.append(float(np.mean(slot))); ss.append(float(sp))
        return _stats(rr, ss)
    vr = _vr_pb()
    print(f"[SELFCHECK] value_ranking-pb: tot {vr['total_return']}% vsSPY {vr['vs_spy']}% n{vr['periods']}"
          f"  ||  lab-baseline: tot {base['total_return']}% vsSPY {base['vs_spy']}% n{base['periods']}", flush=True)

    payload = {
        "computed_at": pd.Timestamp.utcnow().isoformat(),
        "params": {"lookback_months": LOOKBACK, "benchmark": BENCH, "months": int(len(midx)),
                   "rf_monthly": RF_M, "base": "rotation top-10 + guard + low_debt + cheapest-P/B, equal-weight"},
        "baseline": base,
        "L1_concentration": L1, "L2_leverage": L2, "L3_orthogonal": L3, "L4_regime": L4, "combo": COMBO,
        "notes": ("Concentration & leverage raise return WITH drawdown (not new alpha). Orthogonal helps "
                  "only if a sleeve is low-correlation AND high-return. Regime overlay raises risk-adjusted "
                  "return by losing less. In-sample, no fees, ~5y single regime — validate before trusting."),
    }
    return payload


def _line(tag, s):
    return (f"  {tag:22} tot {s['total_return']:>7}%  vsSPY {s['vs_spy']:>7}%  ann {s.get('ann',0):>6}%  "
            f"t={str(s['t_stat']):>5}  Sh {s['sharpe']:>5}  DD {s['max_drawdown']:>7}%")


def main():
    p = build()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(p, indent=2, default=str))
    try:
        from core.models import BacktestResult
        from django.utils import timezone
        BacktestResult.objects.update_or_create(
            kind="return_lab", defaults={"payload": json.loads(json.dumps(p, default=str)),
                                         "computed_at": timezone.now()})
        print("Saved BacktestResult[return_lab]", flush=True)
    except Exception as e:
        print("DB save failed:", e, flush=True)
    print("\n=== RETURN LAB ===", flush=True)
    print(_line("BASELINE (top10 eq)", p["baseline"]), flush=True)
    print("\n-- L1 concentration+weighting --", flush=True)
    for k in sorted(p["L1_concentration"], key=lambda k: -p["L1_concentration"][k]["total_return"]):
        print(_line(k, p["L1_concentration"][k]), flush=True)
    print("\n-- L2 leverage --", flush=True)
    for k in p["L2_leverage"]:
        print(_line(k, p["L2_leverage"][k]), flush=True)
    print("\n-- L3 orthogonal sleeves + blends --", flush=True)
    for k, v in p["L3_orthogonal"]["sleeves"].items():
        print(_line("sleeve:" + k, v), flush=True)
    print("   corr:", p["L3_orthogonal"]["corr"], flush=True)
    for grp in ("blends_core_momentum", "blends_core_breadth"):
        for k, v in p["L3_orthogonal"][grp].items():
            print(_line(grp.split("_")[-1] + ":" + k, v), flush=True)
    print("\n-- L4 regime overlay --", flush=True)
    for k, v in p["L4_regime"].items():
        print(_line(k, v) + f"  off={v.get('months_risk_off')}", flush=True)
    print("\n-- COMBO --", flush=True)
    print("  recipe:", p["combo"]["recipe"], flush=True)
    print(_line("combo", p["combo"]["stats"]), flush=True)


if __name__ == "__main__":
    main()
