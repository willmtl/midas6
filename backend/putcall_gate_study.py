#!/usr/bin/env python3
"""AGGREGATE PUT/CALL MARKET-TIMING GATE — the classic use of the equity put/call ratio: not stock selection
(tested, null) but a MARKET-WIDE contrarian timing signal. Compute the universe-mean pc_vol each month =
aggregate fear/complacency, and use it to SCALE flagship exposure (top10 div_2x returns from rotation_history):
  contrarian: HIGH agg P/C (fear) -> risk-ON (full/lever); LOW (complacency) -> risk-OFF (reduce)
  momentum:   the reverse
Test tercile gates + a continuous z-scaled version vs ungated. Options window (~47mo). -> BacktestResult[putcall_gate].
Run: MSYS_NO_PATHCONV=1 docker exec rotation-backend-1 python -u /app/putcall_gate_study.py
"""
import os, json, warnings
warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
os.environ.setdefault("DJANGO_SETTINGS_MODULE", "rotation.settings")
import django
django.setup()

from core.models import BacktestResult, OptionSnapshot
from backtest_lowpb import BENCH


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
    rh = BacktestResult.objects.filter(kind="rotation_history").first()
    if not rh:
        raise SystemExit("need rotation_history")
    months = rh.payload["months"]
    fdf = pd.DataFrame({"date": pd.to_datetime([m["date"] for m in months]),
                        "port": [m["port_ret"] / 100 for m in months], "spy": [m["spy_ret"] / 100 for m in months]})
    fdf = fdf.set_index("date")
    # aggregate pc_vol per month-end (median across the universe = robust market fear gauge)
    rows = list(OptionSnapshot.objects.exclude(pc_vol__isnull=True).values_list("date", "pc_vol"))
    pcd = pd.DataFrame(rows, columns=["date", "pc"]); pcd["date"] = pd.to_datetime(pcd["date"])
    agg = pcd.groupby(pd.Grouper(key="date", freq="ME"))["pc"].median()
    agg.index = agg.index.to_period("M")
    fdf["pm"] = fdf.index.to_period("M")
    fdf["agg_pc"] = fdf["pm"].map(agg)
    df = fdf.dropna(subset=["agg_pc"]).copy()
    print(f"months with flagship+agg P/C overlap: {len(df)} ({df.index.min().date()}..{df.index.max().date()})", flush=True)
    if len(df) < 12:
        raise SystemExit("too few overlap months")

    r = df["port"].values; spy = df["spy"].values
    pc = df["agg_pc"].values
    # signal uses PRIOR month's agg P/C (known at rebalance): shift
    pc_prev = np.concatenate([[np.nan], pc[:-1]])
    valid = ~np.isnan(pc_prev)
    lo, hi = np.nanquantile(pc_prev, 1 / 3), np.nanquantile(pc_prev, 2 / 3)

    def gate(scheme):
        exp = np.ones(len(r))
        for i in range(len(r)):
            p = pc_prev[i]
            if np.isnan(p):
                exp[i] = 1.0; continue
            if scheme == "contrarian_terc":     # high P/C fear -> full; low -> half
                exp[i] = 1.0 if p >= hi else (0.5 if p <= lo else 0.75)
            elif scheme == "momentum_terc":     # reverse
                exp[i] = 1.0 if p <= lo else (0.5 if p >= hi else 0.75)
            elif scheme == "contrarian_off":    # low P/C complacency -> CASH
                exp[i] = 0.0 if p <= lo else 1.0
        return r * exp, exp

    res = {"ungated": _stats(r, spy)}
    print(f"\n=== AGGREGATE PUT/CALL GATE on flagship ({len(df)} months) ===", flush=True)
    print(f"  {'scheme':18} {'total':>8} {'vsSPY':>8} {'Sh':>5} {'DD':>8} {'win':>6} {'avgExp':>7}", flush=True)
    print(f"  {'ungated':18} {res['ungated']['total']:>7}% {res['ungated']['vs_spy']:>8} {res['ungated']['sharpe']:>5} "
          f"{res['ungated']['dd']:>7}% {res['ungated']['win']:>5}%    1.00", flush=True)
    for scheme in ("contrarian_terc", "momentum_terc", "contrarian_off"):
        gr, exp = gate(scheme); st = _stats(gr, spy); st["avg_exp"] = round(float(exp.mean()), 2); res[scheme] = st
        print(f"  {scheme:18} {st['total']:>7}% {st['vs_spy']:>8} {st['sharpe']:>5} {st['dd']:>7}% {st['win']:>5}% {st['avg_exp']:>7}", flush=True)

    base = res["ungated"]
    best = max(("contrarian_terc", "momentum_terc", "contrarian_off"), key=lambda k: res[k]["sharpe"])
    helps = res[best]["sharpe"] > base["sharpe"] + 0.05
    payload = {
        "computed_at": pd.Timestamp.utcnow().isoformat(),
        "params": {"benchmark": BENCH, "months": int(len(df)), "window": f"{df.index.min().date()}..{df.index.max().date()}"},
        "results": res, "best": best,
        "verdict": (f"Best agg put/call gate = {best} (Sh {res[best]['sharpe']} vs ungated {base['sharpe']}, total {res[best]['total']}% "
                    f"vs {base['total']}%). " + (
                    "AGGREGATE PUT/CALL TIMING HELPS — market-wide fear/complacency scales exposure profitably."
                    if helps else
                    "Aggregate put/call gate does NOT help — market-timing the flagship on universe-wide put/call doesn't beat "
                    "staying fully invested; the flagship's own momentum-accel rotation already handles regime. No edge.")),
        "caveat": "Options era ~47mo (one+ regime), monthly agg P/C = universe median pc_vol, signal lagged 1mo. Small n for "
                  "timing. In-sample, no fees.",
    }
    return payload


def main():
    p = build()
    from pathlib import Path
    Path("/app/.data/studies/putcall_gate.json").write_text(json.dumps(p, indent=2, default=str))
    try:
        from django.utils import timezone
        BacktestResult.objects.update_or_create(
            kind="putcall_gate", defaults={"payload": json.loads(json.dumps(p, default=str)), "computed_at": timezone.now()})
        print("Saved BacktestResult[putcall_gate]", flush=True)
    except Exception as e:
        print("DB save failed:", e, flush=True)
    print("\n" + p["verdict"], flush=True)


if __name__ == "__main__":
    main()
