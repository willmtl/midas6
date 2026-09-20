#!/usr/bin/env python3
"""REGIME-CONDITIONAL QQQ OVERLAY on the flagship (user: "large caps sometimes overperform" — catch the mega-cap-
growth regimes the small-cap value flagship misses, 2023-24). We established: blanket all-cap = -12x, and a growth
stock-picking momentum book = just QQQ. So the vehicle is QQQ ITSELF, held ONLY when a PIT mega-cap-growth regime
fires; flagship value otherwise. Post-hoc on the deployed flagship NET series (flagship_history.json monthly_net) +
QQQ returns — no engine change.

Regime signals (PIT, known at month-end d, applied to the NEXT month's return):
  mega12/6/3 : QQQ trailing-N-mo return > SPY trailing-N-mo return (mega-cap-growth leadership)
  mega_bull  : mega12 AND SPY above its 200d MA (only chase QQQ in an uptrend)
Overlay modes: FULL (100% QQQ that month) and BLEND50 (50% QQQ / 50% flagship). Reports total/CAGR/Sharpe/DD, %months
in-regime, and per-year (does it fix 2023?). Baseline = flagship untouched. Saves BacktestResult[growth_regime_overlay].
Run: MSYS_NO_PATHCONV=1 docker exec rotation-backend-1 python -u /app/growth_regime_overlay.py"""
import os, json, math
os.environ.setdefault("DJANGO_SETTINGS_MODULE", "rotation.settings")
import django; django.setup()
import numpy as np, pandas as pd
from seq_fundamental_study import load_candles


def stats(r):
    r = r.dropna()
    if len(r) < 12:
        return {"n": int(len(r))}
    eq = (1 + r).prod(); yrs = len(r) / 12.0
    return {"n": int(len(r)), "total_pct": round(float((eq - 1) * 100)), "cagr_pct": round(float((eq ** (1 / yrs) - 1) * 100), 1),
            "sharpe": round(float(r.mean() / r.std() * math.sqrt(12)) if r.std() > 0 else 0.0, 2),
            "maxdd_pct": round(float(((1 + r).cumprod() / (1 + r).cumprod().cummax() - 1).min() * 100), 1)}


def main():
    fj = json.load(open("/app/.data/studies/flagship_history.json"))
    flag = pd.Series({pd.Timestamp(d): float(v) for d, v in (fj.get("monthly_net") or [])}).sort_index()
    ed = load_candles(["QQQ", "SPY"])
    qm = ed["QQQ"]["Close"].resample("ME").last(); sm = ed["SPY"]["Close"].resample("ME").last()
    idx = flag.index
    # forward (next-month) returns aligned to the flagship's holding-month convention (date d -> return over d..d+1)
    qqq_fwd = (qm.shift(-1) / qm - 1.0).reindex(idx)
    spy_fwd = (sm.shift(-1) / sm - 1.0).reindex(idx)
    sm200 = ed["SPY"]["Close"].rolling(200).mean().resample("ME").last().reindex(idx)

    def trail(s, n):
        return (s / s.shift(n) - 1.0).reindex(idx)
    regimes = {
        "mega12": (trail(qm, 12) > trail(sm, 12)),
        "mega6": (trail(qm, 6) > trail(sm, 6)),
        "mega3": (trail(qm, 3) > trail(sm, 3)),
        "mega_bull": (trail(qm, 12) > trail(sm, 12)) & (sm.reindex(idx) >= sm200),
    }

    base = flag.dropna()
    res = {"baseline_flagship": stats(base), "qqq_buyhold": stats(qqq_fwd.reindex(base.index)), "arms": {}}
    print(f"=== REGIME-CONDITIONAL QQQ OVERLAY on flagship ({base.index.min().date()}..{base.index.max().date()}) ===", flush=True)
    b = res["baseline_flagship"]; q = res["qqq_buyhold"]
    print(f"  baseline flagship: total {b['total_pct']:,}% CAGR {b['cagr_pct']}% Sh {b['sharpe']} DD {b['maxdd_pct']}%", flush=True)
    print(f"  QQQ buy&hold same window: total {q['total_pct']:,}% CAGR {q['cagr_pct']}% Sh {q['sharpe']}", flush=True)
    print(f"\n{'arm':>22} {'%inRegime':>9} {'total':>12} {'CAGR':>6} {'Sh':>5} {'DD':>7}  {'vs base CAGR':>12}", flush=True)

    def peryear(r):
        return {int(y): round(((1 + r[r.index.year == y].dropna()).prod() - 1) * 100, 1) for y in sorted(set(r.index.year))}

    for rname, reg in regimes.items():
        reg = reg.reindex(idx).fillna(False)
        for mode, w in [("FULL", 1.0), ("BLEND50", 0.5)]:
            ov = flag.copy()
            for d in idx:
                if bool(reg.get(d, False)) and pd.notna(qqq_fwd.get(d)) and pd.notna(flag.get(d)):
                    ov.loc[d] = w * float(qqq_fwd.get(d)) + (1 - w) * float(flag.get(d))
            ov = ov.dropna(); st = stats(ov); st["pct_in_regime"] = round(float(reg.reindex(ov.index).mean()) * 100, 0)
            st["yearly_pct"] = peryear(ov)
            res["arms"][f"{rname}_{mode}"] = st
            print(f"{rname+'_'+mode:>22} {st['pct_in_regime']:>8.0f}% {st['total_pct']:>12,} {st['cagr_pct']:>5.1f}% {st['sharpe']:>5.2f} {st['maxdd_pct']:>6.1f}%  {st['cagr_pct']-b['cagr_pct']:>+11.1f}", flush=True)

    # per-year comparison for the best-return arm + baseline (does it fix 2023?)
    best = max(res["arms"], key=lambda k: res["arms"][k]["total_pct"])
    print(f"\nper-year % (baseline vs best-return arm {best}) — does the overlay fix the flagship's weak years?", flush=True)
    by_base = peryear(base); by_best = res["arms"][best]["yearly_pct"]
    print(f"  {'year':>6} {'flagship':>10} {best:>16} {'QQQ':>8}", flush=True)
    qpy = peryear(qqq_fwd.reindex(base.index))
    for y in sorted(by_base):
        print(f"  {y:>6} {by_base[y]:>+9.1f}% {by_best.get(y,0):>+15.1f}% {qpy.get(y,0):>+7.1f}%", flush=True)

    open("/app/.data/studies/growth_regime_overlay.json", "w").write(json.dumps(res, indent=2, default=float))
    try:
        from core.models import BacktestResult
        from django.utils import timezone
        BacktestResult.objects.update_or_create(kind="growth_regime_overlay", defaults={"payload": res, "computed_at": timezone.now()})
        print("\nSaved BacktestResult[growth_regime_overlay]", flush=True)
    except Exception as e:
        print("save skipped:", e, flush=True)


if __name__ == "__main__":
    main()
