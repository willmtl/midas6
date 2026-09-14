#!/usr/bin/env python3
"""Try a CREDIT-STRESS regime on the flagship (user: "do we use any of those, try if not").
The flagship already uses sector-momentum / value / quality / A-D-divergence; it does NOT use
credit stress. Our overlay work found the oversold-reversal edge concentrates in wide-HY-spread
regimes — but the FRED HY series is only 2023+. So build a FULL-HISTORY (2015-2026) proxy from
bond ETFs we DO have: HYG/LQD ratio (high-yield vs investment-grade) — falling = spreads widening
= credit stress. Duration-neutral, covers 2015-16 / 2018Q4 / 2020 / 2022.

Reuses the exact deployed engine (volatility_regime_study.flagship_monthly). Reports, per return-
priority (ABSOLUTE return, not Sharpe/DD):
 (A) DIAGNOSTIC: flagship mean monthly return AND mean vs-SPY EXCESS, split by stress ON/OFF
     (regime as-of PRIOR month-end -> no look-ahead).
 (B) RETURN attribution: total return if invested ONLY in stress months vs ONLY in calm months
     vs baseline (where does the flagship's return actually live?).
-> BacktestResult[flagship_credit_stress] + JSON.
Run: MSYS_NO_PATHCONV=1 docker exec rotation-celery-worker-1 python -u /app/flagship_credit_stress.py
"""
import os, json, warnings
warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
os.environ.setdefault("DJANGO_SETTINGS_MODULE", "rotation.settings")
import django
django.setup()
from pathlib import Path
from backtest_lowpb import _tstat_from_returns
from volatility_regime_study import flagship_monthly
from seq_fundamental_study import load_candles

OUT = Path(__file__).resolve().parent / ".data" / "studies" / "flagship_credit_stress.json"


def robust(fn, tries=6):
    from django.db import connection
    import time
    for a in range(tries):
        try:
            return fn()
        except Exception:
            connection.close(); time.sleep(4 * (a + 1))
    return None


def _perf(r, spy):
    r = np.asarray(r, float)
    tot = float(np.prod(1 + r) - 1) * 100
    sp = float(np.prod(1 + np.asarray(spy)) - 1) * 100
    sh = float(r.mean() / r.std() * np.sqrt(12)) if r.std() > 1e-9 else 0.0
    eqc = np.cumprod(1 + r); dd = float(((eqc / np.maximum.accumulate(eqc)) - 1).min() * 100)
    t = _tstat_from_returns(list(r))
    return dict(total=round(tot, 1), vs_spy=round(tot - sp, 1), sharpe=round(sh, 2), dd=round(dd, 1),
                t_stat=round(t, 2) if t is not None else None, months=len(r))


def build():
    dates, rets, spies, _ = flagship_monthly()
    rets = np.asarray(rets); spies = np.asarray(spies)
    midx = pd.DatetimeIndex(dates)
    print(f"flagship months {len(midx)} ({midx[0].date()}..{midx[-1].date()})", flush=True)

    bonds = robust(lambda: load_candles(["HYG", "LQD", "JNK", "TLT"])) or {}
    hyg = bonds["HYG"]["Close"]; lqd = bonds["LQD"]["Close"]
    ratio = (hyg / lqd)                                   # HY vs IG; falling = spreads widening = stress
    ratio_m = ratio.resample("ME").last().reindex(midx).ffill()
    # credit-stress regimes (as-of month-end; we shift(1) before use)
    stress_med = (ratio_m < ratio_m.rolling(12, min_periods=6).median())     # HY under its 1y norm
    stress_fall = (ratio_m.diff(1) < 0)                                       # HY/IG falling this month
    hyg_dd = (hyg.resample("ME").last().reindex(midx).ffill()
              / hyg.resample("ME").last().reindex(midx).ffill().cummax() - 1.0)
    stress_dd = (hyg_dd < -0.03)                                             # HYG >3% off its high

    regimes = {"stress_ratio_below_med": stress_med, "stress_ratio_falling": stress_fall,
               "stress_hyg_dd3": stress_dd}

    base = _perf(rets, spies)
    print(f"\nBASELINE flagship: {base['total']}% total | vsSPY {base['vs_spy']} | "
          f"Sh {base['sharpe']} | DD {base['dd']}% | t {base['t_stat']} | n{base['months']}", flush=True)

    out = {"baseline": base, "regimes": {}}
    for name, reg in regimes.items():
        prior = reg.shift(1)                              # decide from prior month-end
        on_mask = np.array([prior.iloc[i] == True for i in range(len(midx))])   # noqa: E712
        off_mask = np.array([prior.iloc[i] == False for i in range(len(midx))]) # noqa: E712
        # (A) diagnostic: mean monthly ret + mean vs-SPY excess by regime
        def stat(mask):
            if mask.sum() == 0:
                return {"n": 0}
            r = rets[mask]; ex = (rets[mask] - spies[mask])
            t = _tstat_from_returns(list(ex))
            return {"n": int(mask.sum()), "mean_ret_%": round(float(r.mean()) * 100, 2),
                    "mean_excess_%": round(float(ex.mean()) * 100, 2),
                    "excess_t": round(t, 2) if t is not None else None}
        diag = {"stress_ON": stat(on_mask), "calm_OFF": stat(off_mask)}
        # (B) return attribution: invest ONLY in stress vs ONLY in calm (cash=0 otherwise)
        only_stress = _perf([rets[i] if on_mask[i] else 0.0 for i in range(len(midx))], spies)
        only_calm = _perf([rets[i] if off_mask[i] else 0.0 for i in range(len(midx))], spies)
        out["regimes"][name] = {"diagnostic": diag, "only_stress": only_stress, "only_calm": only_calm}

        print(f"\n=== {name} ===", flush=True)
        print(f"  (A) stress ON : n{diag['stress_ON'].get('n')}  mean ret {diag['stress_ON'].get('mean_ret_%')}%/mo  "
              f"excess {diag['stress_ON'].get('mean_excess_%')}%/mo (t{diag['stress_ON'].get('excess_t')})", flush=True)
        print(f"      calm  OFF : n{diag['calm_OFF'].get('n')}  mean ret {diag['calm_OFF'].get('mean_ret_%')}%/mo  "
              f"excess {diag['calm_OFF'].get('mean_excess_%')}%/mo (t{diag['calm_OFF'].get('excess_t')})", flush=True)
        print(f"  (B) invest ONLY in stress: {only_stress['total']}% (vsSPY {only_stress['vs_spy']})  |  "
              f"ONLY in calm: {only_calm['total']}% (vsSPY {only_calm['vs_spy']})", flush=True)

    # verdict (return-priority framing)
    r0 = out["regimes"]["stress_ratio_below_med"]["diagnostic"]
    on_ex = r0["stress_ON"].get("mean_excess_%"); off_ex = r0["calm_OFF"].get("mean_excess_%")
    verdict = (f"Flagship baseline {base['total']}% (vsSPY {base['vs_spy']}). "
               f"By credit stress (HY/IG ratio below 1y median): stress-month excess {on_ex}%/mo vs "
               f"calm-month excess {off_ex}%/mo. "
               + ("Flagship alpha is LARGER in credit-stress months -> credit regime is descriptive of WHEN the "
                  "edge lands, but acting on it needs leverage-into-stress (off table) or a selection tilt; a "
                  "cash-in-calm gate would cut return (declined). "
                  if (on_ex is not None and off_ex is not None and on_ex > off_ex) else
                  "Flagship alpha is NOT larger in stress -> credit-stress adds nothing the rotation doesn't already capture. ")
               + "Full-history HYG/LQD proxy (2015-2026) since FRED HY OAS is 2023+ only.")
    print("\n" + verdict, flush=True)
    out["verdict"] = verdict
    out["computed_at"] = pd.Timestamp.utcnow().isoformat()
    out["caveat"] = ("Credit stress proxied by HYG/LQD (HY vs IG bond ETFs), full 2015-2026 history "
                     "(FRED BAMLH0A0HYM2 truncated to 2023+). Whole-book regime overlay on the exact deployed "
                     "flagship (PIT, same survivorship as base). Regime as-of prior month-end (no look-ahead). "
                     "Return-priority framing: exposure cuts/leverage are off-table, so (A) is the actionable read.")
    return out


def main():
    p = build()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(p, indent=2, default=str))
    try:
        from core.models import BacktestResult
        from django.utils import timezone
        BacktestResult.objects.update_or_create(
            kind="flagship_credit_stress",
            defaults={"payload": json.loads(json.dumps(p, default=str)), "computed_at": timezone.now()})
        print("Saved BacktestResult[flagship_credit_stress]", flush=True)
    except Exception as e:
        print("DB save failed:", e, flush=True)


if __name__ == "__main__":
    main()
