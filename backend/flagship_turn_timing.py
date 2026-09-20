#!/usr/bin/env python3
"""FLAGSHIP TURN-TIMING (user: "find a way to influence it SHORT TERM when it TURNS"). The full regime-switch died on
DWELL TIME + vehicle (parked in 21%-CAGR QQQ ~32% of months, dragging every up-year). Refinement: intervene BRIEFLY,
only in the K months right after the inflation regime TURNS to cooling (the flagship's style headwind), then revert to
full flagship. Viable ONLY if the flagship's losses actually CLUSTER at cool-turns — so this script DIAGNOSES that
first, then tests short interventions.

Regime (PIT, publication-lagged): infl_mom = CPI-YoY 3-print change; cooling = infl_mom<0. cool-TURN = first month it
flips negative. hot-TURN = first month it flips positive (flagship tailwind, context only — leverage off-table).
DIAGNOSTIC: mean flagship return in the K months after each cool-turn vs elsewhere; list every turn + following returns.
INTERVENTION: for K in {1,2,3} months after each cool-turn, replace flagship with QQQ (FULL / BLEND50) or CASH; revert.
Report total/CAGR/Sharpe/DD + per-year + both-halves vs untouched baseline. Absolute return is the objective.
Saves BacktestResult[flagship_turn_timing].
Run: MSYS_NO_PATHCONV=1 docker exec rotation-backend-1 python -u /app/flagship_turn_timing.py"""
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
    idx = flag.index
    ed = load_candles(["QQQ"])
    qm = ed["QQQ"]["Close"].resample("ME").last()
    qqq_fwd = (qm.shift(-1) / qm - 1.0).reindex(idx)

    def macro(series):
        q = MacroSeries.objects.filter(series=series).order_by("date").values_list("date", "value")
        return pd.Series({pd.Timestamp(d): (float(v) if v is not None else np.nan) for d, v in q}).sort_index()

    def asof_lagged(monthly):
        out = pd.Series(index=idx, dtype=float)
        for d in idx:
            avail = monthly[monthly.index < d.replace(day=1)]
            out.loc[d] = avail.iloc[-1] if len(avail) else np.nan
        return out

    cpi = macro("CPIAUCSL"); cpi_yoy = cpi / cpi.shift(12) - 1.0
    infl_mom = asof_lagged(cpi_yoy).diff(3)                      # 3-print change in YoY, PIT
    cooling = (infl_mom < 0)
    cool_turn = cooling & (~cooling.shift(1).fillna(False))      # first month it flips to cooling
    hot_turn = (~cooling) & (cooling.shift(1).fillna(False))
    turns = [d for d in idx if bool(cool_turn.get(d, False))]
    hot_turns = [d for d in idx if bool(hot_turn.get(d, False))]

    # DIAGNOSTIC: do flagship losses cluster in the K months after a cool-turn?
    print("=== DIAGNOSTIC: flagship return around inflation cool-turns (PIT) ===", flush=True)
    print(f"  cool-turns (n={len(turns)}): {[str(d.date()) for d in turns]}", flush=True)
    pos = {d: i for i, d in enumerate(idx)}
    for K in [1, 2, 3, 4, 6]:
        inwin = pd.Series(False, index=idx)
        for t in turns:
            i0 = pos[t]
            for j in range(i0, min(i0 + K, len(idx))):
                inwin.iloc[j] = True
        rin = flag[inwin]; rout = flag[~inwin]
        print(f"  K={K}: in-window mean {rin.mean()*100:>+6.2f}% (n={len(rin)}) | "
              f"elsewhere {rout.mean()*100:>+6.2f}% (n={len(rout)}) | "
              f"cum-in {( (1+rin).prod()-1)*100:>+7.1f}% cum-out {((1+rout).prod()-1)*100:>+8.1f}%", flush=True)
    print("\n  per cool-turn: flagship vs QQQ over the following 3 months", flush=True)
    for t in turns:
        i0 = pos[t]; sl = idx[i0:i0 + 3]
        fr = (1 + flag.reindex(sl)).prod() - 1; qr = (1 + qqq_fwd.reindex(sl)).prod() - 1
        print(f"    {t.date()}  flag {fr*100:>+7.1f}%  QQQ {qr*100:>+7.1f}%  (edge {(qr-fr)*100:>+6.1f})", flush=True)

    # INTERVENTION: short switch for K months after each cool-turn
    base = flag.dropna(); b = stats(base)
    res = {"window": f"{str(idx.min().date())}..{str(idx.max().date())}", "baseline": b,
           "baseline_yearly": peryear(base), "cool_turns": [str(d.date()) for d in turns],
           "hot_turns": [str(d.date()) for d in hot_turns], "arms": {}}
    print(f"\n=== INTERVENTION vs baseline {b['total_pct']:,}% (CAGR {b['cagr_pct']} Sh {b['sharpe']} DD {b['maxdd_pct']}) ===", flush=True)
    print(f"{'arm':>20} {'%chg':>5} {'total':>12} {'CAGR':>6} {'Sh':>5} {'DD':>7} {'dCAGR':>7} {'H1/H2 CAGR':>13}", flush=True)
    for K in [1, 2, 3]:
        winmask = pd.Series(False, index=idx)
        for t in turns:
            i0 = pos[t]
            for j in range(i0, min(i0 + K, len(idx))):
                winmask.iloc[j] = True
        for veh, vf in [("QQQ", qqq_fwd), ("CASH", pd.Series(0.0, index=idx))]:
            for mode, w in [("FULL", 1.0), ("BLEND50", 0.5)]:
                ov = flag.copy()
                for d in idx:
                    if bool(winmask.get(d, False)) and pd.notna(flag.get(d)) and pd.notna(vf.get(d)):
                        ov.loc[d] = w * float(vf.get(d)) + (1 - w) * float(flag.get(d))
                ov = ov.dropna(); st = stats(ov)
                h1 = stats(ov[ov.index < SPLIT]); h2 = stats(ov[ov.index >= SPLIT])
                st["pct_changed"] = round(float(winmask.reindex(ov.index).mean()) * 100, 0)
                st["d_cagr"] = round(st["cagr_pct"] - b["cagr_pct"], 1)
                st["yearly"] = peryear(ov)
                key = f"K{K}_{veh}_{mode}"
                res["arms"][key] = st
                print(f"{key:>20} {st['pct_changed']:>4.0f}% {st['total_pct']:>12,} {st['cagr_pct']:>5.1f}% "
                      f"{st['sharpe']:>5.2f} {st['maxdd_pct']:>6.1f}% {st['d_cagr']:>+6.1f} "
                      f"{str(h1.get('cagr_pct')):>6}/{str(h2.get('cagr_pct')):>6}", flush=True)

    best = max(res["arms"], key=lambda k: res["arms"][k]["total_pct"])
    beats = res["arms"][best]["total_pct"] > b["total_pct"]
    res["best_arm"] = best; res["beats_baseline"] = bool(beats)
    print(f"\nbest-total arm: {best} = {res['arms'][best]['total_pct']:,}% "
          f"({'BEATS' if beats else 'LOSES vs'} baseline {b['total_pct']:,}%)", flush=True)
    byb = res["baseline_yearly"]; byo = res["arms"][best]["yearly"]
    print(f"  {'year':>6} {'flagship':>10} {best:>14}", flush=True)
    for y in sorted(byb):
        print(f"  {y:>6} {byb[y]:>+9.1f}% {byo.get(y,0):>+13.1f}%", flush=True)

    open("/app/.data/studies/flagship_turn_timing.json", "w").write(json.dumps(res, indent=2, default=float))
    try:
        from core.models import BacktestResult
        from django.utils import timezone
        BacktestResult.objects.update_or_create(kind="flagship_turn_timing", defaults={"payload": res, "computed_at": timezone.now()})
        print("\nSaved BacktestResult[flagship_turn_timing]", flush=True)
    except Exception as e:
        print("save skipped:", e, flush=True)


if __name__ == "__main__":
    main()
