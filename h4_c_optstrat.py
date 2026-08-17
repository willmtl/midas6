#!/usr/bin/env python3
"""The C dip-buy WITH all the options data tried — full universe (no liquidity-floor change).

h4_c_options found real option signals: IV SKEW amplifies (fear-priced dips bounce, monotone t3+), elevated
atm_iv helps, and EXTREME put/call volume is a veto (falling knife). This wires them into the strategy and
backtests each + the combination on the C dip-buy (steep_4x + QQQ hedge). All variants run on the SAME
option-covered entry set (options history 2022-09+) for a fair comparison. Gross + net@20bps (full-universe
cost). -> BacktestResult[h4_c_optstrat]. Run: MSYS_NO_PATHCONV=1 docker exec rotation-backend-1 python -u /app/h4_c_optstrat.py
"""
import os, json, bisect, warnings
warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
import h4_study as H
import h4_c_enhance as E
from h4_c_upside import load_targets, upside_asof, bucket_upside

SIGS = ["mr_rsi_os", "mr_newlow60", "mr_ndown"]
OPTF = ["atm_iv", "pc_vol", "iv_skew"]
# iv_skew bucket edges from h4_c_options (Q4+ = skew>=-2.6 was best; <-15.4 weak/negative)
SKEW_HI, SKEW_LO = -2.6, -15.4
PCVOL_PANIC = 1.2      # extreme put/call volume -> falling-knife veto (h4_c_options Q5 was negative)
IV_ELEVATED = 43.3     # atm_iv >= this (Q3+) = fear priced in


def collect(allowedC, store, sectors):
    import intraday_data as ID
    from core.models import OptionSnapshot
    names = sorted(allowedC)
    opt = {}
    for r in (OptionSnapshot.objects.filter(ticker__in=names).values("ticker", "date", *OPTF).order_by("ticker", "date")):
        rec = opt.setdefault(r["ticker"], ([], {f: [] for f in OPTF}))
        rec[0].append(r["date"])
        for f in OPTF:
            rec[1][f].append(r[f])

    def asof(tk, d):
        rec = opt.get(tk)
        if not rec:
            return None
        i = bisect.bisect_right(rec[0], d) - 1
        return {f: rec[1][f][i] for f in OPTF} if i >= 0 else None

    trades = []
    for tk in names:
        df = ID.get_4h(tk, 5, False)
        if df is None or len(df) < 120:
            continue
        c = df["Close"].values
        ts = df.index
        n = len(c)
        ad = allowedC[tk]
        fire = np.zeros(n, dtype=bool)
        for s in SIGS:
            e, _ = H.SIGNALS[s]["fn"](df); fire |= np.asarray(e, dtype=bool)
        for i in sorted(H._episode_starts([j for j in range(n) if fire[j]], gap=H.GAP)):
            if i + 1 >= n or c[i] <= 0 or ts[i].date() not in ad:
                continue
            o = asof(tk, ts[i].date())
            if o is None or o.get("iv_skew") is None:
                continue
            up = upside_asof(store, tk, ts[i].date(), float(c[i]))
            sched = [(ts[i + b], float(c[i + b] / c[i + b - 1] - 1)) for b in range(1, E.MAXHOLD + 1) if i + b < n]
            if not sched:
                continue
            trades.append({"entry_ts": ts[i], "bucket": bucket_upside(up), "upside": up,
                           "sector": sectors.get(tk, tk), "sched": sched, "vol": 60.0,
                           "iv_skew": float(o["iv_skew"]),
                           "atm_iv": float(o["atm_iv"]) if o.get("atm_iv") is not None else None,
                           "pc_vol": float(o["pc_vol"]) if o.get("pc_vol") is not None else None})
    return trades


def _skew_mult(sk):
    if sk >= SKEW_HI:
        return 2.0
    if sk >= SKEW_LO:
        return 1.0
    return 0.3


def prep(trades, veto_pcvol=False, skew_filter=False, iv_filter=False, skew_amplify=False):
    out = []
    for t in trades:
        if veto_pcvol and t.get("pc_vol") is not None and t["pc_vol"] > PCVOL_PANIC:
            continue
        if skew_filter and t["iv_skew"] < SKEW_HI:
            continue
        if iv_filter and (t.get("atm_iv") is None or t["atm_iv"] < IV_ELEVATED):
            continue
        t2 = dict(t)
        t2["opt_mult"] = _skew_mult(t["iv_skew"]) if skew_amplify else 1.0
        out.append(t2)
    return out


def main():
    os.environ.setdefault("DJANGO_SETTINGS_MODULE", "rotation.settings")
    import django; django.setup()
    from pathlib import Path
    from h4_on_signals_study import candidate_windows
    from h4_vol_ddfix import _qqq
    allowedC, _ = candidate_windows("C")
    store = E.load_targets(); sectors = E._sector_map()
    daily, _spy = E._spy(5)
    qqq_bar, _ = _qqq()
    trades = collect(allowedC, store, sectors)
    print(f"option-covered C dips {len(trades)} (2022-09+)", flush=True)
    cfg = {"weight": "steep_4x", "hold": E.HOLD_FIXED, "gate": False, "hedge_frac": 0.5}

    VARIANTS = {
        "baseline (no options)":        dict(),
        "pc_vol veto (drop panic)":     dict(veto_pcvol=True),
        "iv_skew filter (fear only)":   dict(skew_filter=True),
        "iv_skew amplify (size by skew)": dict(skew_amplify=True),
        "atm_iv elevated filter":       dict(iv_filter=True),
        "ALL: skew-amplify + pc veto":  dict(skew_amplify=True, veto_pcvol=True),
        "ALL: skew-filter + pc veto + iv": dict(skew_filter=True, veto_pcvol=True, iv_filter=True),
    }
    rows = []
    for name, kw in VARIANTS.items():
        trs = prep(trades, **kw)
        g = E.simulate(trs, daily, qqq_bar, {**cfg, "cost_bps": 0})
        net = E.simulate(trs, daily, qqq_bar, {**cfg, "cost_bps": 10})   # 20bps round-trip (full-universe cost)
        rows.append({"variant": name, "n_pool": len(trs), "gross_pct": g["total_return_pct"],
                     "gross_dd": g["max_dd_pct"], "gross_sharpe": g["sharpe"],
                     "net20_pct": net["total_return_pct"], "net20_sharpe": net["sharpe"]})
        print(f"  {name:34} n{len(trs):>6}  gross {g['total_return_pct']:>8}%  DD {g['max_dd_pct']:>7}  "
              f"Sh {g['sharpe']:>5}  | net@20bps {net['total_return_pct']:>8}%  Sh {net['sharpe']:>5}", flush=True)

    payload = {"computed_at": pd.Timestamp.utcnow().isoformat(), "rows": rows, "n_trades": len(trades),
               "note": ("C dip-buy (steep_4x + QQQ hedge, FULL universe, no liquidity floor) with all option "
                        "signals: iv_skew amplify/filter (fear-priced dips), pc_vol veto (drop panic put-flow), "
                        "atm_iv elevated filter. Same option-covered entry set (2022-09+) for fair comparison. "
                        "gross + net@20bps round-trip (full-universe cost). Current-membership; PIT.")}
    Path("/app/.data/studies").mkdir(parents=True, exist_ok=True)
    Path("/app/.data/studies/h4_c_optstrat.json").write_text(json.dumps(payload, indent=2, default=str))
    try:
        from core.models import BacktestResult
        from django.utils import timezone
        BacktestResult.objects.update_or_create(kind="h4_c_optstrat",
            defaults={"payload": json.loads(json.dumps(payload, default=str)), "computed_at": timezone.now()})
        print("saved BacktestResult[h4_c_optstrat]", flush=True)
    except Exception as e:
        print("DB save failed:", e, flush=True)


if __name__ == "__main__":
    print("=== C DIP-BUY WITH ALL OPTIONS DATA (full universe) ===", flush=True)
    main()
