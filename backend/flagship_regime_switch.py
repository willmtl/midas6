#!/usr/bin/env python3
"""FLAGSHIP MACRO-REGIME SWITCH (user: the ONE persistent macro finding = INFLATION DIRECTION drives value-vs-growth
for 6-12mo; flagship is structurally small-cap VALUE, so cooling-inflation regimes are its STYLE HEADWIND. Test: step
aside to QQQ (or cash) ONLY when a slow macro regime says growth should lead — distinct from the REFUTED price-momentum
QQQ overlay (that used QQQ>SPY and just held QQQ ~70% of months). Regimes here are SLOW MACRO, PIT, publication-lagged:
  cpi_cool   : CPI YoY falling over last 3 prints (realized inflation cooling)
  cpi_cool_l : CPI YoY below its 12-mo average (cooling vs trend)
  m2_accel   : M2 YoY growth RISING (liquidity accelerating)   [user: "also try m2 growth rate"]
  m2_decel   : M2 YoY growth FALLING (liquidity draining)
  m2_high    : M2 YoY growth above its full-sample median (loose-money regime)
Vehicles when regime ON: QQQ (growth leads) and CASH (0%, just sit out). Flagship otherwise. Modes FULL / BLEND50.
Honesty: baseline flagship untouched; per-year (does it fix 2023-24?); both-halves; %months-in-regime. Absolute return
is the objective (leverage/DD off-table). Saves BacktestResult[flagship_regime_switch].
Run: MSYS_NO_PATHCONV=1 docker exec rotation-backend-1 python -u /app/flagship_regime_switch.py"""
import os, json, math
os.environ.setdefault("DJANGO_SETTINGS_MODULE", "rotation.settings")
import django; django.setup()
import numpy as np, pandas as pd
from seq_fundamental_study import load_candles
from core.models import MacroSeries

SPLIT = pd.Timestamp("2021-01-01")


def stats(r):
    r = r.dropna()
    if len(r) < 12:
        return {"n": int(len(r))}
    eq = (1 + r).prod(); yrs = len(r) / 12.0
    return {"n": int(len(r)), "total_pct": round(float((eq - 1) * 100)),
            "cagr_pct": round(float((eq ** (1 / yrs) - 1) * 100), 1) if eq > 0 else -100.0,
            "sharpe": round(float(r.mean() / r.std() * math.sqrt(12)) if r.std() > 0 else 0.0, 2),
            "maxdd_pct": round(float(((1 + r).cumprod() / (1 + r).cumprod().cummax() - 1).min() * 100), 1)}


def peryear(r):
    return {int(y): round(((1 + r[r.index.year == y].dropna()).prod() - 1) * 100, 1) for y in sorted(set(r.index.year))}


def main():
    fj = json.load(open("/app/.data/studies/flagship_history.json"))
    flag = pd.Series({pd.Timestamp(d): float(v) for d, v in (fj.get("monthly_net") or [])}).sort_index()
    idx = flag.index                                             # decision dates d; flag[d] = return over [d, d+1]
    ed = load_candles(["QQQ"])
    qm = ed["QQQ"]["Close"].resample("ME").last()
    qqq_fwd = (qm.shift(-1) / qm - 1.0).reindex(idx)             # QQQ return over [d, d+1] — same convention as flag

    def macro(series):
        q = MacroSeries.objects.filter(series=series).order_by("date").values_list("date", "value")
        return pd.Series({pd.Timestamp(d): (float(v) if v is not None else np.nan) for d, v in q}).sort_index()

    # publication-lagged as-of: at decision date d only ref-months strictly before d's month are public
    def asof_lagged(monthly_series):
        out = pd.Series(index=idx, dtype=float)
        for d in idx:
            avail = monthly_series[monthly_series.index < d.replace(day=1)]
            out.loc[d] = avail.iloc[-1] if len(avail) else np.nan
        return out

    cpi = macro("CPIAUCSL"); cpi_yoy = cpi / cpi.shift(12) - 1.0
    cpi_yoy_a = asof_lagged(cpi_yoy)
    m2 = macro("M2SL"); m2_yoy = m2 / m2.shift(12) - 1.0
    m2_yoy_a = asof_lagged(m2_yoy)

    regimes = {
        "cpi_cool":   cpi_yoy_a.diff(3) <= -0.003,
        "cpi_cool_l": cpi_yoy_a < cpi_yoy_a.rolling(12).mean(),
        "m2_accel":   m2_yoy_a.diff(1) > 0,
        "m2_decel":   m2_yoy_a.diff(1) < 0,
        "m2_high":    m2_yoy_a > float(m2_yoy_a.median()),
    }

    base = flag.dropna()
    res = {"window": f"{str(idx.min().date())}..{str(idx.max().date())}",
           "baseline": stats(base), "qqq_buyhold": stats(qqq_fwd.reindex(base.index).dropna()),
           "baseline_yearly": peryear(base), "arms": {}}
    b = res["baseline"]; q = res["qqq_buyhold"]
    print(f"=== FLAGSHIP MACRO-REGIME SWITCH ({res['window']}) ===", flush=True)
    print(f"  baseline flagship: {b['total_pct']:,}%  CAGR {b['cagr_pct']}%  Sh {b['sharpe']}  DD {b['maxdd_pct']}%", flush=True)
    print(f"  QQQ buy&hold     : {q['total_pct']:,}%  CAGR {q['cagr_pct']}%  Sh {q['sharpe']}", flush=True)
    print(f"\n{'arm':>22} {'%on':>4} {'total':>12} {'CAGR':>6} {'Sh':>5} {'DD':>7} {'dCAGR':>7}  {'H1CAGR/H2CAGR':>14}", flush=True)

    for rname, reg in regimes.items():
        reg = reg.reindex(idx).fillna(False)
        for veh, vfwd in [("QQQ", qqq_fwd), ("CASH", pd.Series(0.0, index=idx))]:
            for mode, w in [("FULL", 1.0), ("BLEND50", 0.5)]:
                ov = flag.copy()
                for d in idx:
                    if bool(reg.get(d, False)) and pd.notna(flag.get(d)) and pd.notna(vfwd.get(d)):
                        ov.loc[d] = w * float(vfwd.get(d)) + (1 - w) * float(flag.get(d))
                ov = ov.dropna(); st = stats(ov)
                h1 = ov[ov.index < SPLIT]; h2 = ov[ov.index >= SPLIT]
                st["pct_on"] = round(float(reg.reindex(ov.index).mean()) * 100, 0)
                st["d_cagr"] = round(st["cagr_pct"] - b["cagr_pct"], 1)
                st["h1_cagr"] = stats(h1).get("cagr_pct"); st["h2_cagr"] = stats(h2).get("cagr_pct")
                st["yearly"] = peryear(ov)
                key = f"{rname}_{veh}_{mode}"
                res["arms"][key] = st
                print(f"{key:>22} {st['pct_on']:>3.0f}% {st['total_pct']:>12,} {st['cagr_pct']:>5.1f}% "
                      f"{st['sharpe']:>5.2f} {st['maxdd_pct']:>6.1f}% {st['d_cagr']:>+6.1f}  "
                      f"{str(st['h1_cagr']):>6}/{str(st['h2_cagr']):>6}", flush=True)

    best = max(res["arms"], key=lambda k: res["arms"][k]["total_pct"])
    beats = res["arms"][best]["total_pct"] > b["total_pct"]
    res["best_arm"] = best; res["beats_baseline"] = bool(beats)
    print(f"\nbest-total arm: {best} = {res['arms'][best]['total_pct']:,}% "
          f"({'BEATS' if beats else 'LOSES vs'} baseline {b['total_pct']:,}%)", flush=True)
    print(f"\nper-year % (baseline vs {best}) — does the regime switch fix the flagship's weak years?", flush=True)
    byb = res["baseline_yearly"]; byo = res["arms"][best]["yearly"]; byq = peryear(qqq_fwd.reindex(base.index).dropna())
    print(f"  {'year':>6} {'flagship':>10} {'switch':>10} {'QQQ':>8}", flush=True)
    for y in sorted(byb):
        print(f"  {y:>6} {byb[y]:>+9.1f}% {byo.get(y,0):>+9.1f}% {byq.get(y,0):>+7.1f}%", flush=True)

    open("/app/.data/studies/flagship_regime_switch.json", "w").write(json.dumps(res, indent=2, default=float))
    try:
        from core.models import BacktestResult
        from django.utils import timezone
        BacktestResult.objects.update_or_create(kind="flagship_regime_switch", defaults={"payload": res, "computed_at": timezone.now()})
        print("\nSaved BacktestResult[flagship_regime_switch]", flush=True)
    except Exception as e:
        print("save skipped:", e, flush=True)


if __name__ == "__main__":
    main()
