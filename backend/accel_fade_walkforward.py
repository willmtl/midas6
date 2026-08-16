#!/usr/bin/env python3
"""WALK-FORWARD the two-stage 'top-5 cheapest raw P/B -> pick most-fading (accel_fade)' pick.

The two-stage battery (survivorship_smallcap_study.py) found accel_fade + pe beat the cheapest_pb control in BOTH
halves. The real overfitting risk is METHOD SELECTION (13 stage-2 methods tried, 2 passed in-sample). This is the
skip_wide-style honest test: choose the method on PAST data only, test on UNTOUCHED future.
  1. split-half: pick the best stage-2 method on the train half -> does it (and accel_fade) beat control OOS?
  2. expanding-window: each month re-pick the best-trailing method (>=12mo warmup), trade next month, accumulate,
     vs always-control and always-accel_fade. (Does live method-selection beat the flagship?)
  3. per-year: accel_fade vs control (edge distributed or one lucky year?)
Reads BacktestResult[survivorship_smallcap].results[t5_*].monthly (no re-run). -> BacktestResult[accel_fade_wf].
Run: MSYS_NO_PATHCONV=1 docker exec rotation-backend-1 python -u /app/accel_fade_walkforward.py
"""
import os, json, warnings
warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
os.environ.setdefault("DJANGO_SETTINGS_MODULE", "rotation.settings")
import django
django.setup()
from pathlib import Path

OUT = Path(__file__).resolve().parent / ".data" / "studies" / "accel_fade_wf.json"


def _cum(vals):
    return float(np.prod([1 + v for v in vals]) - 1) * 100


def main():
    from core.models import BacktestResult
    res = BacktestResult.objects.get(kind="survivorship_smallcap").payload["results"]
    methods = [k[3:] for k in res if k.startswith("t5_")]
    # align all methods on a common month index
    series = {}
    for m in methods:
        mo = res[f"t5_{m}"].get("monthly") or []
        series[m] = pd.Series({pd.Timestamp(d): float(v) for d, v in mo}).sort_index()
    idx = sorted(set().union(*[set(s.index) for s in series.values()]))
    df = pd.DataFrame({m: series[m].reindex(idx) for m in methods}).dropna(how="all")
    n = len(df)
    ctrl = "cheapest_pb"
    print(f"methods={len(methods)}  months={n}  {df.index[0].date()}..{df.index[-1].date()}", flush=True)

    # ---- 1. SPLIT-HALF method-selection walk-forward ----
    mid = n // 2
    tr, te = df.iloc[:mid], df.iloc[mid:]
    def totals(frame):
        return {m: _cum(frame[m].dropna().values) for m in methods}
    tr_tot, te_tot = totals(tr), totals(te)
    best_tr = max((m for m in methods if m != ctrl), key=lambda m: tr_tot[m])
    best_te = max((m for m in methods if m != ctrl), key=lambda m: te_tot[m])
    print("\n=== SPLIT-HALF method-selection walk-forward ===", flush=True)
    print(f"  train half winner: {best_tr} (train {tr_tot[best_tr]:.1f}%, ctrl {tr_tot[ctrl]:.1f}%)", flush=True)
    print(f"  -> OOS (test half): {best_tr} {te_tot[best_tr]:.1f}% vs ctrl {te_tot[ctrl]:.1f}%  "
          f"({'BEATS' if te_tot[best_tr] > te_tot[ctrl] else 'LOSES to'} control OOS)", flush=True)
    print(f"  accel_fade OOS test half: {te_tot['accel_fade']:.1f}% vs ctrl {te_tot[ctrl]:.1f}%  "
          f"({'BEATS' if te_tot['accel_fade'] > te_tot[ctrl] else 'LOSES'})", flush=True)
    print(f"  (test half winner would have been {best_te}; train winner held up = {best_tr == best_te})", flush=True)
    # reverse split (train on 2nd half, test 1st) for symmetry
    best_tr2 = max((m for m in methods if m != ctrl), key=lambda m: te_tot[m])
    print(f"  reverse: train=2nd half winner {best_tr2} -> OOS 1st half {tr_tot[best_tr2]:.1f}% vs ctrl {tr_tot[ctrl]:.1f}% "
          f"({'BEATS' if tr_tot[best_tr2] > tr_tot[ctrl] else 'LOSES'})", flush=True)

    # ---- 2. EXPANDING-WINDOW live method selection ----
    WARM = 12
    picks, exp_rets = [], []
    for i in range(WARM, n):
        past = df.iloc[:i]
        cand = {m: _cum(past[m].dropna().values) for m in methods if m != ctrl}
        pick = max(cand, key=cand.get)
        nxt = df.iloc[i]
        if pd.notna(nxt[pick]):
            picks.append(pick); exp_rets.append(float(nxt[pick]))
    exp_tot = _cum(exp_rets)
    ctrl_after = _cum(df[ctrl].iloc[WARM:].dropna().values)
    af_after = _cum(df["accel_fade"].iloc[WARM:].dropna().values)
    from collections import Counter
    pc = Counter(picks)
    print("\n=== EXPANDING-WINDOW live method selection (>=12mo warmup) ===", flush=True)
    print(f"  adaptive (re-pick best-trailing each month): {exp_tot:.1f}%", flush=True)
    print(f"  always accel_fade:                           {af_after:.1f}%", flush=True)
    print(f"  always control (cheapest_pb):                {ctrl_after:.1f}%", flush=True)
    print(f"  method chosen by the adaptive rule: {dict(pc.most_common())}", flush=True)

    # ---- 3. PER-YEAR accel_fade vs control ----
    print("\n=== PER-YEAR: accel_fade vs control ===", flush=True)
    yrs = sorted({d.year for d in df.index})
    beat = 0; peryr = {}
    for y in yrs:
        sub = df[[c for c in [ "accel_fade", ctrl]]].loc[[d for d in df.index if d.year == y]]
        af = _cum(sub["accel_fade"].dropna().values); cc = _cum(sub[ctrl].dropna().values)
        w = af > cc; beat += int(w)
        peryr[y] = {"accel_fade": round(af, 1), "control": round(cc, 1), "win": w}
        print(f"  {y}: accel_fade {af:+7.1f}%   control {cc:+7.1f}%   {'WIN' if w else 'lose'}", flush=True)

    af_robust = (te_tot["accel_fade"] > te_tot[ctrl] and tr_tot["accel_fade"] > tr_tot[ctrl]
                 and af_after > ctrl_after and beat >= len(yrs) - 1)
    verdict = (
        f"accel_fade OOS: both halves ({tr_tot['accel_fade']:.0f}%/{te_tot['accel_fade']:.0f}% vs ctrl "
        f"{tr_tot[ctrl]:.0f}%/{te_tot[ctrl]:.0f}%), adaptive-selection {exp_tot:.0f}% vs ctrl {ctrl_after:.0f}%, "
        f"beats control {beat}/{len(yrs)} yrs. "
        + ("ROBUST -> the top-5-then-fade pick survives method-selection walk-forward; safe to wire (still gross/"
           "GICS-universe caveated). Confirms the documented stock-accel-FADE duality."
           if af_robust else
           "NOT fully robust -> treat as promising but do NOT wire yet; the in-sample both-halves win is partly "
           "method-selection luck.")
    )
    print("\n" + verdict, flush=True)

    payload = {"computed_at": pd.Timestamp.utcnow().isoformat(), "months": n, "control": ctrl,
               "split_half": {"train_winner": best_tr, "test_winner": best_te,
                              "train_totals": {k: round(v, 1) for k, v in tr_tot.items()},
                              "test_totals": {k: round(v, 1) for k, v in te_tot.items()}},
               "expanding": {"adaptive": round(exp_tot, 1), "always_accel_fade": round(af_after, 1),
                             "always_control": round(ctrl_after, 1), "picks": dict(pc.most_common())},
               "per_year": peryr, "accel_fade_robust": bool(af_robust), "verdict": verdict}
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(payload, indent=2, default=str))
    BacktestResult.objects.update_or_create(kind="accel_fade_wf",
        defaults={"payload": json.loads(json.dumps(payload, default=str)), "computed_at": django.utils.timezone.now()})
    print("Saved BacktestResult[accel_fade_wf]", flush=True)


if __name__ == "__main__":
    import django.utils.timezone
    main()
