#!/usr/bin/env python3
"""THE final deployable config: liquid-floor C dip-buy + steep_4x conviction + QQQ hedge + IV-SKEW filter,
NET of tier-realistic fees. Combines every survivor: value universe (C), oversold H4 dip, conviction sizing
(analyst upside), QQQ stress-hedge, liquidity floor (fees are the wall), and the IV-skew edge (fear-priced
dips) as a filter/amplifier. -> BacktestResult[h4_c_final].
Run: MSYS_NO_PATHCONV=1 docker exec rotation-backend-1 python -u /app/h4_c_final.py
"""
import os, json, bisect, warnings
warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
import h4_study as H
import h4_c_enhance as E
from h4_c_upside import load_targets, upside_asof, bucket_upside

SIGS = ["mr_rsi_os", "mr_newlow60", "mr_ndown"]
SKEW_HI, SKEW_LO = -2.6, -15.4       # iv_skew: Q4+ (fear-priced) best; <Q2 weak (h4_c_options)


def _skew_mult(sk):
    if sk is None:
        return 1.0
    return 2.0 if sk >= SKEW_HI else (1.0 if sk >= SKEW_LO else 0.3)


def main():
    os.environ.setdefault("DJANGO_SETTINGS_MODULE", "rotation.settings")
    import django; django.setup()
    from pathlib import Path
    import intraday_data as ID
    from h4_on_signals_study import candidate_windows
    from seq_fundamental_study import load_candles
    from core.models import OptionSnapshot
    from h4_vol_ddfix import _qqq
    allowedC, _ = candidate_windows("C")
    names = sorted(allowedC)
    store = load_targets(); sectors = E._sector_map()
    daily_env, _sp = E._spy(5)
    qqq_bar, _ = _qqq()
    dcand = load_candles(names)
    dvol_ser = {t: (d["Close"] * d["Volume"]).rolling(20).mean() for t, d in dcand.items() if d is not None and "Volume" in d}
    # iv_skew as-of
    sk = {}
    for r in OptionSnapshot.objects.filter(ticker__in=names).exclude(iv_skew=None).values("ticker", "date", "iv_skew").order_by("ticker", "date"):
        rec = sk.setdefault(r["ticker"], ([], []))
        rec[0].append(r["date"]); rec[1].append(r["iv_skew"])

    def skew_asof(tk, d):
        rec = sk.get(tk)
        if not rec:
            return None
        i = bisect.bisect_right(rec[0], d) - 1
        return float(rec[1][i]) if i >= 0 else None

    trades = []
    for tk in names:
        df = ID.get_4h(tk, 5, False)
        if df is None or len(df) < 120:
            continue
        c = df["Close"].values; ts = df.index; n = len(c)
        ad = allowedC[tk]; dser = dvol_ser.get(tk)
        fire = np.zeros(n, dtype=bool)
        for s in SIGS:
            e, _ = H.SIGNALS[s]["fn"](df); fire |= np.asarray(e, dtype=bool)
        for i in sorted(H._episode_starts([j for j in range(n) if fire[j]], gap=H.GAP)):
            if i + 1 >= n or c[i] <= 0 or ts[i].date() not in ad:
                continue
            dv = float(dser.asof(pd.Timestamp(ts[i].date()))) if dser is not None else 0.0
            if not np.isfinite(dv):
                dv = 0.0
            up = upside_asof(store, tk, ts[i].date(), float(c[i]))
            sched = [(ts[i + b], float(c[i + b] / c[i + b - 1] - 1)) for b in range(1, E.MAXHOLD + 1) if i + b < n]
            if not sched:
                continue
            trades.append({"entry_ts": ts[i], "bucket": bucket_upside(up), "upside": up, "sector": sectors.get(tk, tk),
                           "sched": sched, "vol": 60.0, "dvol": dv, "iv_skew": skew_asof(tk, ts[i].date())})

    base = {"weight": "steep_4x", "hold": E.HOLD_FIXED, "gate": False, "hedge_frac": 0.5}
    # (floor $, per-side bps) tiers
    def run(sub, cost):
        return E.simulate(sub, daily_env, qqq_bar, {**base, "cost_bps": cost})

    rows = []
    def add(label, sub, cost):
        m = run(sub, cost)
        rows.append({"config": label, "n": len(sub), "cost_side_bps": cost, **m})
        print(f"  {label:44} n{len(sub):>5}  net {m['total_return_pct']:>8}%  DD {m['max_dd_pct']:>7}  Sh {m['sharpe']:>5}", flush=True)

    liq100 = [t for t in trades if t["dvol"] >= 100e6]
    liq50 = [t for t in trades if t["dvol"] >= 50e6]
    skew100 = [t for t in liq100 if t["iv_skew"] is not None and t["iv_skew"] >= SKEW_HI]     # fear-priced only
    amp100 = [dict(t, opt_mult=_skew_mult(t["iv_skew"])) for t in liq100]                     # skew-amplified
    print(f"trades {len(trades)} | liq$100M {len(liq100)} | liq$50M {len(liq50)} | skew-filter@100M {len(skew100)}", flush=True)
    add("$100M liquid (gross)", liq100, 0)
    add("$100M liquid + fees(3bps/side)", liq100, 3)
    add("$100M + SKEW-FILTER + fees", skew100, 3)
    add("$100M + SKEW-AMPLIFY + fees", amp100, 3)
    add("$50M liquid + fees(4bps/side)", liq50, 4)
    add("$50M + SKEW-AMPLIFY + fees", [dict(t, opt_mult=_skew_mult(t["iv_skew"])) for t in liq50], 4)

    payload = {"computed_at": pd.Timestamp.utcnow().isoformat(), "rows": rows, "n_trades": len(trades),
               "note": ("FINAL deployable C dip-buy: steep_4x conviction + QQQ stress-hedge + liquidity floor "
                        "(tier-realistic per-side fees) + IV-skew filter/amplify (fear-priced dips). Skew from "
                        "OptionSnapshot as-of (2022-09+ where available; None=neutral weight). Net of fees.")}
    Path("/app/.data/studies").mkdir(parents=True, exist_ok=True)
    Path("/app/.data/studies/h4_c_final.json").write_text(json.dumps(payload, indent=2, default=str))
    try:
        from core.models import BacktestResult
        from django.utils import timezone
        BacktestResult.objects.update_or_create(kind="h4_c_final",
            defaults={"payload": json.loads(json.dumps(payload, default=str)), "computed_at": timezone.now()})
        print("saved BacktestResult[h4_c_final]", flush=True)
    except Exception as e:
        print("DB save failed:", e, flush=True)


if __name__ == "__main__":
    print("=== FINAL DEPLOYABLE CONFIG (net of fees) ===", flush=True)
    main()
