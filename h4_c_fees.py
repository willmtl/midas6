#!/usr/bin/env python3
"""How much of the final C config survives REAL transaction costs? — the one unmodeled risk.

Final config = C value-pick + H4 oversold dip + steep_4x conviction + QQQ stress-hedge (+1823% gross).
It's a high-churn 0-3 day strategy, so fees matter. Sweeps a per-side cost (spread+slippage, commissions
~0 on modern brokers) charged on every dip-buy entry+exit and on hedge on/off. -> BacktestResult[h4_c_fees].
Run: MSYS_NO_PATHCONV=1 docker exec rotation-backend-1 python -u /app/h4_c_fees.py
"""
import os, json, warnings
warnings.filterwarnings("ignore")
import pandas as pd
import h4_c_enhance as E
import h4_vol_dipbuy as V
from h4_vol_ddfix import _qqq


def main():
    os.environ.setdefault("DJANGO_SETTINGS_MODULE", "rotation.settings")
    import django; django.setup()
    from pathlib import Path
    from h4_on_signals_study import candidate_windows
    allowedC, _ = candidate_windows("C")
    store = E.load_targets(); sectors = E._sector_map()
    daily, _spy = E._spy(5)
    qqq_bar, _ = _qqq()
    trades = V.collect("C_only", allowedC, store, sectors)
    base = {"weight": "steep_4x", "hold": E.HOLD_FIXED, "gate": False, "hedge_frac": 0.5}

    # fee-robust variant: only the big-edge dips (50%+ analyst upside) -> fewer trades, bigger per-trade edge
    hi = [t for t in trades if t["bucket"] in ("50-100%", ">100%")]
    print(f"all C dips {len(trades)}  |  high-conviction (50%+ upside) {len(hi)}", flush=True)

    rows = []
    for label, trs in [("ALL dips (steep_4x)", trades), ("HIGH-conv only (50%+ upside)", hi)]:
        print(f"-- {label} --", flush=True)
        for per_side in [0, 2.5, 5, 10, 15, 25]:             # bps per side
            cfg = {**base, "cost_bps": per_side}
            m = E.simulate(trs, daily, qqq_bar, cfg)
            rows.append({"variant": label, "per_side_bps": per_side, "round_trip_bps": per_side * 2, **m})
            print(f"  round-trip {per_side*2:>3.0f}bps  total {m['total_return_pct']:>9}%  "
                  f"DD {m['max_dd_pct']:>7}  Sh {m['sharpe']:>5}  taken {m['n_taken']}", flush=True)

    payload = {"computed_at": pd.Timestamp.utcnow().isoformat(), "rows": rows, "n_trades": len(trades),
               "n_high_conv": len(hi),
               "note": ("Transaction-cost sweep on the FINAL C config (steep_4x + QQQ stress-hedge). Per-side "
                        "cost = spread/slippage (commissions ~0). Charged on each dip-buy entry+exit and hedge "
                        "on/off. ~1 round-trip per 0-3 day trade. Everything else point-in-time; C current-"
                        "membership survivorship.")}
    Path("/app/.data/studies").mkdir(parents=True, exist_ok=True)
    Path("/app/.data/studies/h4_c_fees.json").write_text(json.dumps(payload, indent=2, default=str))
    try:
        from core.models import BacktestResult
        from django.utils import timezone
        BacktestResult.objects.update_or_create(kind="h4_c_fees",
            defaults={"payload": json.loads(json.dumps(payload, default=str)), "computed_at": timezone.now()})
        print("saved BacktestResult[h4_c_fees]", flush=True)
    except Exception as e:
        print("DB save failed:", e, flush=True)


if __name__ == "__main__":
    print("=== FINAL C CONFIG — NET OF FEES ===", flush=True)
    main()
