#!/usr/bin/env python3
"""V2 STACK — combine the individually-validated levers and see if they're ADDITIVE or REDUNDANT, then
walk-forward the full stack. Cumulative decomposition (each row adds ONE lever, all div_2x weighted):
  A top10                     (current live baseline)
  B top5                      (+concentrate — sizing_sweep's return lever)
  C top5 + bear gate          (+slow 12mo-abs-mom SPY<0 -> cash)
  D top5 + bear + 15% distress(+distressed satellite = neg-book unprofitable eq-wt sleeve)  == V2 FULL
  E top10 + inv_vol + bear + distress  (the Sharpe-max path from the sweep)
Then WALK-FORWARD D vs A (both halves + per-year vs SPY): does top5 concentration survive OOS or overfit?
-> BacktestResult[v2_stack]. Run: MSYS_NO_PATHCONV=1 docker exec rotation-backend-1 python -u /app/v2_stack_study.py
"""
import os, json, warnings
warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
os.environ.setdefault("DJANGO_SETTINGS_MODULE", "rotation.settings")
import django
django.setup()

import config, sector_holdings
import price_basis
from seq_fundamental_study import load_candles, load_financial_reports
from trend_stock_studies import _pit_monthly_panel, _available_at, _ret_delist, CRYPTO
from backtest_lowpb import _monthly_close, BENCH

W_DISTRESS = 0.15


def _stats(r, spy):
    r = np.asarray(r, float); n = len(r)
    if n == 0:
        return dict(total=0, vs_spy=0, sharpe=0, dd=0, win=0)
    tot = float(np.prod(1 + r) - 1) * 100; sp = float(np.prod(1 + np.asarray(spy)) - 1) * 100
    sh = float(r.mean() / r.std() * np.sqrt(12)) if r.std() > 1e-9 else 0.0
    eqc = np.cumprod(1 + r); dd = float(((eqc / np.maximum.accumulate(eqc)) - 1).min() * 100)
    return dict(total=round(tot, 1), vs_spy=round(tot - sp, 1), sharpe=round(sh, 2), dd=round(dd, 1),
                win=round((r > 0).mean() * 100, 1))


def _tot(r):
    return float(np.prod(1 + np.asarray(r, float)) - 1) * 100 if len(r) else 0.0


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
    spy_m = etf_daily[BENCH]["Close"].resample("ME").last().reindex(midx)
    spy_abs12 = spy_m.pct_change(12)                      # slow absolute momentum for the bear gate
    accel = etf_m.pct_change(3) - etf_m.pct_change(3).shift(3)
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
    mret = px.pct_change(); vol6 = mret.rolling(6).std()
    adl_m, dvol = {}, {}
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
    print(f"months {len(midx)} | stocks {len(common)}", flush=True)

    def wsum(picks, weights):
        w = np.asarray(weights, float); r = np.asarray([p[1] for p in picks], float)
        return float(np.sum(w * r) / np.sum(w)) if w.sum() else 0.0

    A, B, C, D, E, spies, dts = [], [], [], [], [], [], []
    for i in range(9, len(midx) - 1):
        date, ndate = midx[i], midx[i + 1]
        sp = spy_m.iloc[i + 1] / spy_m.iloc[i] - 1
        if not np.isfinite(sp):
            continue
        ranked = accel.loc[date].dropna().sort_values(ascending=False)
        adsl, pr3 = ad_slope3.loc[date], px_ret3.loc[date]
        allpicks = []       # (pick, ret, is_div, inv_vol_w)
        for etf in ranked.head(10).index:
            _, holds = sector_map.get(etf, (etf, []))
            c = [h for h in holds if h in px.columns and _available_at(px[h], date) and pd.notna(pb.loc[date, h])
                 and pb.loc[date, h] > 0 and not bool(trap.loc[date, h])
                 and pd.notna(dvol.loc[date, h]) and dvol.loc[date, h] >= 5e6]
            g = [x for x in c if bool(low.loc[date, x])] or c
            if not g:
                continue
            pick = min(g, key=lambda h: pb.loc[date, h])
            r = _ret_delist(px[pick], date, ndate)
            if r is None or not np.isfinite(r):
                continue
            is_div = bool(pd.notna(adsl.get(pick)) and pd.notna(pr3.get(pick)) and adsl.get(pick) > 0 and pr3.get(pick) < 0)
            vv = vol6.loc[date, pick]
            ivw = 1.0 / max(float(vv), 0.02) if pd.notna(vv) and vv > 0 else 1.0
            allpicks.append((pick, float(r), is_div, ivw))
        if not allpicks:
            continue
        # distressed sleeve: neg-book AND unprofitable, $5M floor, equal-weight next-month return
        drs = []
        negbook = eq.loc[date] < 0
        for t in common[negbook.values]:
            if not (_available_at(px[t], date) and bool(ni.loc[date, t] < 0)
                    and pd.notna(dvol.loc[date, t]) and dvol.loc[date, t] >= 5e6):
                continue
            dr = _ret_delist(px[t], date, ndate)
            if dr is not None and np.isfinite(dr):
                drs.append(float(dr))
        distress_ret = float(np.mean(drs)) if drs else 0.0

        top10 = allpicks; top5 = allpicks[:5]
        div_w = lambda subset: [2.0 if p[2] else 1.0 for p in subset]
        a = wsum(top10, div_w(top10))                                   # A: top10 div_2x
        b = wsum(top5, div_w(top5))                                     # B: top5 div_2x
        bear = bool(pd.notna(spy_abs12.iloc[i]) and spy_abs12.iloc[i] < 0)
        c_ = 0.0 if bear else b                                         # C: top5 + bear gate
        core_d = 0.0 if bear else b
        d_ = (1 - W_DISTRESS) * core_d + W_DISTRESS * distress_ret      # D: + 15% distressed (gated core)
        e_top10_iv = wsum(top10, [p[3] * (2.0 if p[2] else 1.0) for p in top10])   # top10 inv_vol*div_2x
        core_e = 0.0 if bear else e_top10_iv
        e_ = (1 - W_DISTRESS) * core_e + W_DISTRESS * distress_ret      # E: Sharpe path + bear + distress
        A.append(a); B.append(b); C.append(c_); D.append(d_); E.append(e_)
        spies.append(float(sp)); dts.append(ndate)

    variants = {"A_top10_div2x": A, "B_top5_div2x": B, "C_top5_bear": C,
                "D_top5_bear_distress(V2)": D, "E_top10_invvol_bear_distress": E}
    print(f"\n=== V2 STACK — cumulative levers ({len(spies)} months) ===", flush=True)
    print(f"  {'variant':30} {'total':>8} {'vsSPY':>8} {'Sh':>5} {'DD':>8} {'win':>6}", flush=True)
    res = {}
    for k, series in variants.items():
        st = _stats(series[:len(spies)], spies); res[k] = st
        print(f"  {k:30} {st['total']:>7}% {st['vs_spy']:>8} {st['sharpe']:>5} {st['dd']:>7}% {st['win']:>5}%", flush=True)
    spy_st = _stats([0] * 0, spies); spy_tot = round(float(np.prod(1 + np.asarray(spies)) - 1) * 100, 1)
    print(f"  {'SPY':30} {spy_tot:>7}%", flush=True)

    # WALK-FORWARD D (V2) vs A (baseline)
    df = pd.DataFrame({"date": pd.to_datetime(dts), "A": A, "D": D, "spy": spies})
    df["yr"] = df["date"].dt.year
    h = len(df) // 2
    wf = {"halves": {"A_1st": round(_tot(df['A'][:h]), 1), "A_2nd": round(_tot(df['A'][h:]), 1),
                     "D_1st": round(_tot(df['D'][:h]), 1), "D_2nd": round(_tot(df['D'][h:]), 1),
                     "spy_1st": round(_tot(df['spy'][:h]), 1), "spy_2nd": round(_tot(df['spy'][h:]), 1)},
          "per_year": {}}
    d_beats = a_beats = 0
    print("  walk-forward per-year (V2=D vs baseline=A vs SPY):", flush=True)
    for yr, g in df.groupby("yr"):
        a_, d_, s_ = _tot(g["A"]), _tot(g["D"]), _tot(g["spy"])
        wf["per_year"][int(yr)] = {"A": round(a_, 1), "D": round(d_, 1), "spy": round(s_, 1)}
        d_beats += d_ > s_; a_beats += a_ > s_
        print(f"     {yr}: A {a_:>+7.1f}%  V2 {d_:>+7.1f}%  SPY {s_:>+7.1f}%", flush=True)

    ny = len(wf["per_year"])
    v2_robust = (wf["halves"]["D_1st"] > wf["halves"]["spy_1st"] and wf["halves"]["D_2nd"] > wf["halves"]["spy_2nd"]
                 and d_beats >= ny - 1)
    v2, base = res["D_top5_bear_distress(V2)"], res["A_top10_div2x"]
    payload = {
        "computed_at": pd.Timestamp.utcnow().isoformat(),
        "params": {"w_distress": W_DISTRESS, "benchmark": BENCH, "months": int(len(spies)), "spy_total": spy_tot},
        "results": res, "walk_forward": wf, "v2_beats_spy_years": f"{d_beats}/{ny}", "v2_robust": bool(v2_robust),
        "verdict": (
            f"V2 (top5 + bear gate + 15% distressed, div_2x) = {v2['total']}%/vsSPY{v2['vs_spy']}/Sh{v2['sharpe']}/DD{v2['dd']}% "
            f"vs baseline top10 {base['total']}%/Sh{base['sharpe']}/DD{base['dd']}%. " + (
            f"LEVERS STACK — V2 beats baseline on return and holds up in walk-forward ({d_beats}/{ny} yrs beat SPY, both halves +). "
            "The top5 concentration is the return driver; bear gate + distressed shape risk. Candidate v2."
            if v2_robust else
            f"MIXED — V2 lifts return but the top5 concentration does NOT cleanly survive walk-forward ({d_beats}/{ny} yrs), "
            "meaning some of the concentration gain is in-sample luck about which sectors won. Keep top10 as the robust core; "
            "treat top5 as a higher-variance return tilt, not a validated upgrade.")),
        "caveat": "In-sample ~5y, no fees, 1 bear episode (gate barely tested), distressed pro-cyclical/crisis-untested. "
                  "top5 = higher single-sector concentration risk. Walk-forward is subperiod split, not true holdout.",
    }
    return payload


def main():
    p = build()
    from pathlib import Path
    Path("/app/.data/studies/v2_stack.json").write_text(json.dumps(p, indent=2, default=str))
    try:
        from core.models import BacktestResult
        from django.utils import timezone
        BacktestResult.objects.update_or_create(
            kind="v2_stack", defaults={"payload": json.loads(json.dumps(p, default=str)), "computed_at": timezone.now()})
        print("Saved BacktestResult[v2_stack]", flush=True)
    except Exception as e:
        print("DB save failed:", e, flush=True)
    print("\n" + p["verdict"], flush=True)


if __name__ == "__main__":
    main()
