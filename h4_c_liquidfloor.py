#!/usr/bin/env python3
"""Raise the liquidity floor — the deployable, fee-honest version of the C dip-buy.

Fees are the binding constraint (h4_c_fees: breakeven ~19-20bps round-trip) and the only fix is trading
LIQUID names (tight spreads). So tag each C dip trade with its name's DAILY dollar-volume, sweep the floor,
and apply the REALISTIC per-side spread cost for each liquidity tier (liquid -> cheap). Net return by floor
answers: does raising the floor give a viable strategy? -> BacktestResult[h4_c_liquidfloor].
Run: MSYS_NO_PATHCONV=1 docker exec rotation-backend-1 python -u /app/h4_c_liquidfloor.py
"""
import os, json, warnings
warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
import h4_study as H
import h4_c_enhance as E
from h4_c_upside import load_targets, upside_asof, bucket_upside

SIGS = ["mr_rsi_os", "mr_newlow60", "mr_ndown"]
# floor ($ daily dvol) -> realistic per-SIDE spread/slippage (bps). Liquid names trade far cheaper.
TIERS = [(5e6, 12.0), (20e6, 6.0), (50e6, 4.0), (100e6, 3.0), (200e6, 2.0)]


def collect(allowedC, store, sectors):
    import intraday_data as ID
    from seq_fundamental_study import load_candles
    daily = load_candles(sorted(allowedC))
    dvol_ser = {}
    for tk, d in daily.items():
        if d is not None and "Volume" in d and len(d) > 25:
            dvol_ser[tk] = (d["Close"] * d["Volume"]).rolling(20).mean()
    trades = []
    for tk in sorted(allowedC):
        df = ID.get_4h(tk, 5, False)
        if df is None or len(df) < 120:
            continue
        c = df["Close"].values
        ts = df.index
        n = len(c)
        ad = allowedC[tk]
        dser = dvol_ser.get(tk)
        fire = np.zeros(n, dtype=bool)
        for s in SIGS:
            e, _ = H.SIGNALS[s]["fn"](df); fire |= np.asarray(e, dtype=bool)
        idxs = sorted(H._episode_starts([i for i in range(n) if fire[i]], gap=H.GAP))
        for i in idxs:
            if i + 1 >= n or c[i] <= 0 or ts[i].date() not in ad:
                continue
            dv = float(dser.asof(pd.Timestamp(ts[i].date()))) if dser is not None else 0.0
            if not np.isfinite(dv):
                dv = 0.0
            up = upside_asof(store, tk, ts[i].date(), float(c[i]))
            sched = [(ts[i + b], float(c[i + b] / c[i + b - 1] - 1)) for b in range(1, E.MAXHOLD + 1) if i + b < n]
            if not sched:
                continue
            trades.append({"entry_ts": ts[i], "bucket": bucket_upside(up), "upside": up,
                           "sector": sectors.get(tk, tk), "sched": sched, "vol": 60.0, "dvol": dv})
    return trades


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
    print(f"C dips {len(trades)} (with dvol tags)", flush=True)
    base = {"weight": "steep_4x", "hold": E.HOLD_FIXED, "gate": False, "hedge_frac": 0.5}

    rows = []
    for floor, side_bps in TIERS:
        sub = [t for t in trades if t["dvol"] >= floor]
        gross = E.simulate(sub, daily, qqq_bar, {**base, "cost_bps": 0})
        net = E.simulate(sub, daily, qqq_bar, {**base, "cost_bps": side_bps})
        rows.append({"floor_usd": floor, "side_bps": side_bps, "round_trip_bps": side_bps * 2,
                     "n_trades": len(sub), "gross_pct": gross["total_return_pct"],
                     "net_pct": net["total_return_pct"], "net_dd_pct": net["max_dd_pct"],
                     "net_sharpe": net["sharpe"], "net_taken": net["n_taken"]})
        print(f"  floor ${floor/1e6:>4.0f}M  n{len(sub):>6}  gross {gross['total_return_pct']:>8}%  "
              f"-> NET @{side_bps:>4}bps/side ({side_bps*2:.0f}bps rt) {net['total_return_pct']:>8}%  "
              f"DD {net['max_dd_pct']:>7}  Sh {net['net_sharpe'] if False else net['sharpe']:>5}", flush=True)

    payload = {"computed_at": pd.Timestamp.utcnow().isoformat(), "rows": rows, "n_trades": len(trades),
               "note": ("Liquidity-floor sweep on the final C config (steep_4x + QQQ hedge). Each floor uses a "
                        "REALISTIC per-side spread cost for that liquidity tier ($5M->12bps ... $200M->2bps). "
                        "Net = what you'd actually capture trading only names above the floor. Higher floor = "
                        "fewer trades but far cheaper execution. Current-membership survivorship; PIT.")}
    Path("/app/.data/studies").mkdir(parents=True, exist_ok=True)
    Path("/app/.data/studies/h4_c_liquidfloor.json").write_text(json.dumps(payload, indent=2, default=str))
    try:
        from core.models import BacktestResult
        from django.utils import timezone
        BacktestResult.objects.update_or_create(kind="h4_c_liquidfloor",
            defaults={"payload": json.loads(json.dumps(payload, default=str)), "computed_at": timezone.now()})
        print("saved BacktestResult[h4_c_liquidfloor]", flush=True)
    except Exception as e:
        print("DB save failed:", e, flush=True)


if __name__ == "__main__":
    print("=== LIQUIDITY-FLOOR SWEEP (net of realistic fees) ===", flush=True)
    main()
