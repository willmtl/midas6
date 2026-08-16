#!/usr/bin/env python3
"""LIVE SECTOR ACCELERATION signal — rank every sector by momentum ACCELERATION (the walk-forward-validated
sector signal that replaced 6mo momentum: +422% vs SPY / Sharpe 1.81 / beats SPY 5/5 yrs).

acceleration = (last-3mo return) minus (the 3mo before that) = is momentum speeding up (gas) or fading
(brake)? Positive = inflecting UP (early in a move); negative = rolling over (late). 'hidden_turn' flags
sectors with NEGATIVE 6mo momentum but POSITIVE acceleration — turning up from lows, which plain momentum
misses entirely. Top-10 by acceleration = the rotation pick-zone.
-> BacktestResult[sector_acceleration] + JSON.
Run: docker exec rotation-backend-1 python -u /app/sector_acceleration_scan.py
"""
import os, json, warnings
warnings.filterwarnings("ignore")
os.environ.setdefault("DJANGO_SETTINGS_MODULE", "rotation.settings")
import django
django.setup()
import pandas as pd
from seq_fundamental_study import load_candles

LOOKBACK_D = 126          # ~6 trading months
HALF = LOOKBACK_D // 2    # ~3 trading months
TOP_N = 10


def build():
    from core.models import Sector
    sectors = [(n, e) for n, e in Sector.objects.values_list("name", "etf") if e]
    daily = load_candles([e for _, e in sectors])
    rows = []
    for name, etf in sectors:
        df = daily.get(etf)
        if df is None or len(df) < LOOKBACK_D + 1:
            continue
        c = df["Close"]
        mom6 = float((c.iloc[-1] / c.iloc[-1 - LOOKBACK_D] - 1) * 100)
        m3n = float((c.iloc[-1] / c.iloc[-1 - HALF] - 1) * 100)
        m3p = float((c.iloc[-1 - HALF] / c.iloc[-1 - 2 * HALF] - 1) * 100)
        acc = m3n - m3p
        rows.append({"sector": name, "etf": etf, "momentum_6m": round(mom6, 1),
                     "momentum_3m": round(m3n, 1), "momentum_3m_prior": round(m3p, 1),
                     "acceleration": round(acc, 1), "accelerating": bool(acc > 0),
                     "hidden_turn": bool(acc > 0 and mom6 < 0)})
    rows.sort(key=lambda r: -r["acceleration"])
    for i, r in enumerate(rows, 1):
        r["rank"] = i
        r["in_pick_zone"] = i <= TOP_N          # top-N by acceleration = where the rotation picks come from
    payload = {
        "computed_at": pd.Timestamp.utcnow().isoformat(),
        "params": {"lookback_days": LOOKBACK_D, "half_days": HALF, "top_n": TOP_N,
                   "signal": "acceleration = 3mo-now minus 3mo-3ago (2nd derivative of price)",
                   "backtest": "walk-forward +422% vs SPY / Sharpe 1.81 / beats SPY 5/5 yrs — replaced 6mo momentum"},
        "sectors": rows, "n": len(rows),
        "note": ("Acceleration = is momentum speeding up (gas) or fading (brake). Top-10 = the rotation "
                 "pick-zone. 'hidden turn' = negative 6mo momentum but turning up now — the early inflections "
                 "plain momentum misses. Directional; the sector-selection signal feeding the value pick."),
    }
    return payload


def main():
    from pathlib import Path
    OUT = Path(__file__).resolve().parent / ".data" / "studies" / "sector_acceleration.json"
    payload = build()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(payload, indent=2, default=str))
    try:
        from core.models import BacktestResult
        from django.utils import timezone
        BacktestResult.objects.update_or_create(
            kind="sector_acceleration",
            defaults={"payload": json.loads(json.dumps(payload, default=str)), "computed_at": timezone.now()})
        print("Saved BacktestResult[sector_acceleration]", flush=True)
    except Exception as e:
        print("DB save failed:", e, flush=True)
    print("\n=== SECTOR ACCELERATION (top by gas pedal) ===", flush=True)
    for r in payload["sectors"][:14]:
        ht = "  ⟵ hidden turn (neg 6mo, turning up)" if r["hidden_turn"] else ""
        z = "*" if r["in_pick_zone"] else " "
        print(f" {z}#{r['rank']:>2} {r['sector']:24} accel {r['acceleration']:>+6.1f}  "
              f"(3mo {r['momentum_3m']:>+6.1f} vs prior {r['momentum_3m_prior']:>+6.1f}; 6mo {r['momentum_6m']:>+6.1f}){ht}", flush=True)


if __name__ == "__main__":
    main()
