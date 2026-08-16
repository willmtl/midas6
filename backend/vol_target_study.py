#!/usr/bin/env python3
"""PORTFOLIO VOL-TARGETING — scale gross exposure to hit a constant volatility (lever up when the strategy
is calm, cut when it's stormy). A Sharpe/drawdown play, not a raw-return play. Operates on the LIVE flagship
monthly series (top10 div_2x, from BacktestResult[rotation_history].months[].port_ret) so no data reload.
exposure_t = clip(target_monthly_vol / trailing_vol_{t-1}, 0, cap); scaled_r = exposure*r - financing on the
levered part. Sweep target vol x leverage cap. -> BacktestResult[vol_target].
Run: MSYS_NO_PATHCONV=1 docker exec rotation-backend-1 python -u /app/vol_target_study.py
"""
import os, json, warnings
warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
os.environ.setdefault("DJANGO_SETTINGS_MODULE", "rotation.settings")
import django
django.setup()

from core.models import BacktestResult

LOOKBACK = 6            # trailing months for realized-vol estimate
FIN_ANNUAL = 0.05      # financing cost on leverage>1 (5%/yr on the borrowed fraction)
CAPS = [1.0, 1.5, 2.0]              # max gross exposure
TARGETS_ANN = [0.10, 0.12, 0.15]   # annualized vol targets


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
    row = BacktestResult.objects.filter(kind="rotation_history").first()
    if not row:
        raise SystemExit("rotation_history not computed — run rotation_history_scan.py first")
    months = row.payload["months"]
    r = pd.Series([m["port_ret"] / 100 for m in months], dtype=float)   # live top10 div_2x monthly returns
    spy = pd.Series([m["spy_ret"] / 100 for m in months], dtype=float)
    n = len(r)
    print(f"vol-targeting {n} months of the live flagship (top10 div_2x)", flush=True)

    base = _stats(r.values, spy.values)
    # trailing realized MONTHLY vol, shifted so only past info sets this month's exposure
    tvol = r.rolling(LOOKBACK).std().shift(1)
    fin_m = FIN_ANNUAL / 12.0

    res = {"base_unscaled": base}
    for tg in TARGETS_ANN:
        tg_m = tg / np.sqrt(12)                         # monthly target vol
        for cap in CAPS:
            exp = (tg_m / tvol).clip(lower=0, upper=cap)
            exp = exp.fillna(1.0)                        # warmup months: full exposure
            lev_excess = (exp - 1.0).clip(lower=0)      # borrowed fraction
            scaled = exp * r - lev_excess * fin_m
            st = _stats(scaled.values, spy.values)
            st["avg_exposure"] = round(float(exp.mean()), 2)
            res[f"tgt{int(tg*100)}_cap{cap}"] = st

    best = max((k for k in res if k != "base_unscaled"), key=lambda k: res[k]["sharpe"])
    helps = res[best]["sharpe"] > base["sharpe"] + 0.03
    print(f"\n=== VOL-TARGETING the flagship (base Sh {base['sharpe']}, {base['total']}%, DD {base['dd']}%) ===", flush=True)
    print(f"  {'variant':16} {'total':>8} {'Sh':>5} {'DD':>8} {'win':>6} {'avgExp':>7}", flush=True)
    print(f"  {'base_unscaled':16} {base['total']:>7}% {base['sharpe']:>5} {base['dd']:>7}% {base['win']:>5}%    1.00", flush=True)
    for k in res:
        if k == "base_unscaled":
            continue
        st = res[k]
        print(f"  {k:16} {st['total']:>7}% {st['sharpe']:>5} {st['dd']:>7}% {st['win']:>5}% {st['avg_exposure']:>7}", flush=True)

    payload = {
        "computed_at": pd.Timestamp.utcnow().isoformat(),
        "params": {"lookback_months": LOOKBACK, "financing_annual": FIN_ANNUAL, "caps": CAPS,
                   "targets_annual": TARGETS_ANN, "months": int(n), "source": "top10 div_2x (rotation_history)"},
        "results": res, "best": best,
        "verdict": (f"Best vol-target = {best} (Sh {res[best]['sharpe']} vs base {base['sharpe']}, total {res[best]['total']}% "
                    f"vs {base['total']}%, DD {res[best]['dd']}% vs {base['dd']}%). " + (
                    "VOL-TARGETING HELPS — constant-vol scaling lifts risk-adjusted return; lever up in calm regimes, "
                    "cut in stormy ones. A Sharpe/DD improvement (return effect depends on cap)."
                    if helps else
                    "Vol-targeting does NOT clearly beat static full exposure risk-adjusted — the flagship's vol is "
                    "already fairly stable and the monthly signal is coarse; scaling mostly trades return for a marginally "
                    "smoother ride. Keep static sizing; use the cap only as a risk preference.")),
        "caveat": "Monthly data, 6mo trailing vol (coarse), single ~5y sample, 5%/yr financing on leverage. "
                  "In-sample; vol-target is a real-time estimate but tested on one regime.",
    }
    return payload


def main():
    p = build()
    from pathlib import Path
    Path("/app/.data/studies/vol_target.json").write_text(json.dumps(p, indent=2, default=str))
    try:
        from django.utils import timezone
        BacktestResult.objects.update_or_create(
            kind="vol_target", defaults={"payload": json.loads(json.dumps(p, default=str)), "computed_at": timezone.now()})
        print("Saved BacktestResult[vol_target]", flush=True)
    except Exception as e:
        print("DB save failed:", e, flush=True)
    print("\n" + p["verdict"], flush=True)


if __name__ == "__main__":
    main()
