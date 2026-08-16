#!/usr/bin/env python3
"""VOLUME CONVICTION WEIGHT — the CORRECT harvest of the A/D-divergence / volume-surge edge. volume_overlay
proved the signal is real on the flagship pick (A/D-divergence +7.78%/66%win vs +2.28%; vol-surge +5.32%) but
crude tilt/filter mis-harvest it (tilt swaps off cheapest-P/B = −262pp; filter over-concentrates = Sharpe
collapse). Right way: KEEP the cheapest-P/B pick in every sector (breadth intact), just OVERWEIGHT the ones
showing accumulation. Test conviction multipliers on divergence / surge / either, vs equal-weight flagship.
-> BacktestResult[volume_conviction]. Run: MSYS_NO_PATHCONV=1 docker exec rotation-backend-1 python -u /app/volume_conviction_study.py
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

TOP_N = 10


def _stats(r, spy):
    r = np.asarray(r, float); n = len(r)
    if n == 0:
        return dict(total=0, vs_spy=0, sharpe=0, dd=0, win=0)
    tot = float(np.prod(1 + r) - 1) * 100; sp = float(np.prod(1 + np.asarray(spy)) - 1) * 100
    sh = float(r.mean() / r.std() * np.sqrt(12)) if r.std() > 1e-9 else 0.0
    eq = np.cumprod(1 + r); dd = float(((eq / np.maximum.accumulate(eq)) - 1).min() * 100)
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
    print("building volume/accumulation panels...", flush=True)
    relvol, adl_m, dvol = {}, {}, {}
    for t in common:
        d = stock_daily.get(t)
        if d is None or "Volume" not in d or len(d) < 90:
            continue
        v = d["Volume"]
        relvol[t] = (v.rolling(20).mean() / v.rolling(90).mean()).resample("ME").last().reindex(midx)
        dvol[t] = (d["Close"] * v).rolling(20).mean().resample("ME").last().reindex(midx)
        if {"High", "Low", "Close"}.issubset(d.columns):
            rng = (d["High"] - d["Low"]).replace(0, np.nan)
            mfm = ((d["Close"] - d["Low"]) - (d["High"] - d["Close"])) / rng
            adl_m[t] = (mfm.fillna(0) * v).cumsum().resample("ME").last().reindex(midx)
    relvol = pd.DataFrame(relvol).reindex(index=midx, columns=common)
    dvol = pd.DataFrame(dvol).reindex(index=midx, columns=common)
    adl = pd.DataFrame(adl_m).reindex(index=midx, columns=common)
    ad_slope3 = adl - adl.shift(3)
    px_ret3 = px.pct_change(3)
    print(f"months {len(midx)} | stocks {len(common)}", flush=True)

    # weight schemes: name -> weight given (is_div, is_surge)
    schemes = {
        "equal": lambda d, s: 1.0,
        "div_1.5x": lambda d, s: 1.5 if d else 1.0,
        "div_2x": lambda d, s: 2.0 if d else 1.0,
        "div_3x": lambda d, s: 3.0 if d else 1.0,
        "surge_2x": lambda d, s: 2.0 if s else 1.0,
        "either_2x": lambda d, s: 2.0 if (d or s) else 1.0,
        "div_2x_surge_1.5x": lambda d, s: 2.0 if d else (1.5 if s else 1.0),
    }
    port = {k: [] for k in schemes}; spies = []
    for i in range(9, len(midx) - 1):
        date, ndate = midx[i], midx[i + 1]
        sp = spy_m.iloc[i + 1] / spy_m.iloc[i] - 1
        if not np.isfinite(sp):
            continue
        top = accel.loc[date].dropna().sort_values(ascending=False).head(TOP_N).index
        adsl, pr3, rv = ad_slope3.loc[date], px_ret3.loc[date], relvol.loc[date]
        picks = []      # (ret, is_div, is_surge)
        for etf in top:
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
            is_surge = bool(pd.notna(rv.get(pick)) and rv.get(pick) > 1.15)
            picks.append((float(r), is_div, is_surge))
        if not picks:
            continue
        for k, wf in schemes.items():
            ws = np.array([wf(d, s) for _, d, s in picks]); rs = np.array([r for r, _, _ in picks])
            port[k].append(float(np.sum(ws * rs) / np.sum(ws)))
        spies.append(float(sp))

    print(f"\n=== VOLUME CONVICTION WEIGHT (keep cheapest-P/B pick, overweight accumulation) ===", flush=True)
    res = {}
    for k in schemes:
        s = _stats(port[k][:len(spies)], spies); res[k] = s
        print(f"  {k:20} total {s['total']:>7}%  vsSPY {s['vs_spy']:>7}  Sh {s['sharpe']:>5}  DD {s['dd']:>7}%  win {s['win']}%", flush=True)

    base = res["equal"]
    best = max(schemes, key=lambda k: res[k]["sharpe"])
    helps = res[best]["sharpe"] > base["sharpe"] + 0.03 and res[best]["vs_spy"] > base["vs_spy"]
    payload = {
        "computed_at": pd.Timestamp.utcnow().isoformat(),
        "params": {"top_n": TOP_N, "benchmark": BENCH, "months": int(len(spies))},
        "results": res, "best": best,
        "verdict": (f"Best conviction weight = {best} (Sh {res[best]['sharpe']} vs equal {base['sharpe']}, vsSPY "
                    f"{res[best]['vs_spy']} vs {base['vs_spy']}). " + (
                    f"VOLUME CONVICTION ADDS — overweighting accumulation lifts return AND Sharpe over equal-weight "
                    f"WITHOUT losing breadth. First orthogonal overlay to genuinely improve the flagship. Wire it."
                    if helps else
                    "Even as a conviction WEIGHT, volume does not cleanly beat equal-weight risk-adjusted — the "
                    "divergence edge is real but too rare/lumpy to move the equal-weight portfolio. Keep as a live "
                    "conviction FLAG, not a sizing rule.")),
        "caveat": "A/D-divergence = ADL(cumsum MFM*Vol) 3mo-slope>0 while price 3mo<0; surge = 20d/90d vol>1.15. PIT. "
                  "Weights renormalized per month, breadth unchanged. In-sample ~5y, no fees.",
    }
    return payload


def main():
    p = build()
    from pathlib import Path
    Path("/app/.data/studies/volume_conviction.json").write_text(json.dumps(p, indent=2, default=str))
    try:
        from core.models import BacktestResult
        from django.utils import timezone
        BacktestResult.objects.update_or_create(
            kind="volume_conviction", defaults={"payload": json.loads(json.dumps(p, default=str)), "computed_at": timezone.now()})
        print("Saved BacktestResult[volume_conviction]", flush=True)
    except Exception as e:
        print("DB save failed:", e, flush=True)
    print("\n" + p["verdict"], flush=True)


if __name__ == "__main__":
    main()
