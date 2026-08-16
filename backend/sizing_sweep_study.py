#!/usr/bin/env python3
"""CONCENTRATION x WEIGHTING SWEEP — map the risk-adjusted frontier of levers we already own. Hold the
validated selection FIXED (accel top-N sectors -> cheapest positive-P/B guard low-debt pick, $5M floor) and
jointly sweep: TOP_N sectors {3,5,8,10} x weighting {equal, inv_vol (1/trailing-6mo vol), cheapness (1/pb),
div_2x (A/D-divergence 2x), inv_vol_div_2x (combine)}. Which #sectors and weighting gives the best Sharpe /
return / DD? Feeds the v2 stack. -> BacktestResult[sizing_sweep].
Run: MSYS_NO_PATHCONV=1 docker exec rotation-backend-1 python -u /app/sizing_sweep_study.py
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

TOP_NS = [3, 5, 8, 10]
SCHEMES = ["equal", "inv_vol", "cheapness", "div_2x", "inv_vol_div_2x"]


def _stats(r, spy):
    r = np.asarray(r, float); n = len(r)
    if n == 0:
        return dict(total=0, vs_spy=0, sharpe=0, dd=0, win=0)
    tot = float(np.prod(1 + r) - 1) * 100; sp = float(np.prod(1 + np.asarray(spy)) - 1) * 100
    sh = float(r.mean() / r.std() * np.sqrt(12)) if r.std() > 1e-9 else 0.0
    eqc = np.cumprod(1 + r); dd = float(((eqc / np.maximum.accumulate(eqc)) - 1).min() * 100)
    return dict(total=round(tot, 1), vs_spy=round(tot - sp, 1), sharpe=round(sh, 2), dd=round(dd, 1),
                win=round((r > 0).mean() * 100, 1))


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
    mret = px.pct_change()
    vol6 = mret.rolling(6).std()                        # trailing ~6mo monthly-return vol (for inv-vol)
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

    def weight(scheme, pick, date, is_div):
        if scheme == "equal":
            return 1.0
        if scheme == "inv_vol":
            v = vol6.loc[date, pick]
            return 1.0 / max(float(v), 0.02) if pd.notna(v) and v > 0 else 1.0
        if scheme == "cheapness":
            p = pb.loc[date, pick]
            return 1.0 / max(float(p), 0.1) if pd.notna(p) and p > 0 else 1.0
        if scheme == "div_2x":
            return 2.0 if is_div else 1.0
        if scheme == "inv_vol_div_2x":
            v = vol6.loc[date, pick]
            base = 1.0 / max(float(v), 0.02) if pd.notna(v) and v > 0 else 1.0
            return base * (2.0 if is_div else 1.0)
        return 1.0

    # port[topn][scheme] = list of monthly returns
    port = {n: {s: [] for s in SCHEMES} for n in TOP_NS}
    spies = []
    for i in range(9, len(midx) - 1):
        date, ndate = midx[i], midx[i + 1]
        sp = spy_m.iloc[i + 1] / spy_m.iloc[i] - 1
        if not np.isfinite(sp):
            continue
        ranked = accel.loc[date].dropna().sort_values(ascending=False)
        adsl, pr3 = ad_slope3.loc[date], px_ret3.loc[date]
        # build the full top-10 pick list once (with div flag), then slice per TOP_N
        allpicks = []
        for etf in ranked.head(max(TOP_NS)).index:
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
            allpicks.append((pick, float(r), is_div))
        if not allpicks:
            continue
        for n in TOP_NS:
            sub = allpicks[:n]
            if not sub:
                continue
            for s in SCHEMES:
                ws = np.array([weight(s, pk, date, dv) for pk, _, dv in sub])
                rs = np.array([rr for _, rr, _ in sub])
                port[n][s].append(float(np.sum(ws * rs) / np.sum(ws)))
        spies.append(float(sp))

    print(f"\n=== CONCENTRATION x WEIGHTING SWEEP ({len(spies)} months) ===", flush=True)
    print(f"  {'topN/scheme':22} {'total':>8} {'vsSPY':>8} {'Sh':>5} {'DD':>8} {'win':>6}", flush=True)
    res = {}
    for n in TOP_NS:
        for s in SCHEMES:
            st = _stats(port[n][s][:len(spies)], spies); res[f"top{n}_{s}"] = st
            print(f"  top{n:<2} {s:16} {st['total']:>7}% {st['vs_spy']:>8} {st['sharpe']:>5} {st['dd']:>7}% {st['win']:>5}%", flush=True)

    best_sh = max(res, key=lambda k: res[k]["sharpe"])
    best_ret = max(res, key=lambda k: res[k]["total"])
    base = res["top10_equal"]
    payload = {
        "computed_at": pd.Timestamp.utcnow().isoformat(),
        "params": {"top_ns": TOP_NS, "schemes": SCHEMES, "benchmark": BENCH, "months": int(len(spies))},
        "results": res, "best_sharpe": best_sh, "best_return": best_ret, "baseline_top10_equal": base,
        "verdict": (f"Best Sharpe = {best_sh} (Sh {res[best_sh]['sharpe']}, total {res[best_sh]['total']}%, DD {res[best_sh]['dd']}%); "
                    f"best return = {best_ret} ({res[best_ret]['total']}%). Baseline top10_equal Sh {base['sharpe']}/{base['total']}%. "
                    f"vs live div_2x top10 = {res.get('top10_div_2x',{}).get('total')}%/Sh{res.get('top10_div_2x',{}).get('sharpe')}."),
        "caveat": "In-sample ~5y, no fees. inv_vol=1/trailing-6mo monthly-return vol; cheapness=1/pb; div_2x=A/D-divergence 2x.",
    }
    return payload


def main():
    p = build()
    from pathlib import Path
    Path("/app/.data/studies/sizing_sweep.json").write_text(json.dumps(p, indent=2, default=str))
    try:
        from core.models import BacktestResult
        from django.utils import timezone
        BacktestResult.objects.update_or_create(
            kind="sizing_sweep", defaults={"payload": json.loads(json.dumps(p, default=str)), "computed_at": timezone.now()})
        print("Saved BacktestResult[sizing_sweep]", flush=True)
    except Exception as e:
        print("DB save failed:", e, flush=True)
    print("\n" + p["verdict"], flush=True)


if __name__ == "__main__":
    main()
