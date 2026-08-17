#!/usr/bin/env python3
"""H4 dip-buy on a HIGH-VOLATILITY liquid universe — the pivot from the vol-vs-value finding.

h4_vol_universe proved the H4 dip-buy edge is a VOLATILITY effect (top vol quintile +0.457%/3b vs ~0 in
Q1-Q4), and C was only a proxy for it. So select on volatility directly. Universes compared:
  - highvol      : any LIQUID name, on bars where trailing vol is high (PIT threshold + liquidity/vol-cap)
  - highvol_in_C : the intersection — high-vol AND in a C value window (does value+vol beat vol alone?)
  - C_only       : the current C basket (reference)
Each run through the cash-aware portfolio engine (reuses h4_c_enhance) with (a) the baseline config and
(b) the winning steep_4x + SPY-hedge config. -> BacktestResult[h4_vol_dipbuy].
Run: MSYS_NO_PATHCONV=1 docker exec rotation-backend-1 python -u /app/h4_vol_dipbuy.py
"""
import os, json, warnings
warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
import h4_study as H
import h4_c_enhance as E
from h4_c_upside import load_targets, upside_asof, bucket_upside

SIGS = ["mr_rsi_os", "mr_newlow60", "mr_ndown"]
VOL_LO, VOL_HI = 50.0, 300.0    # trailing annualized vol %: high-vol band (Q5 started ~50%); cap drops glitches
MIN_DVOL = 5e6                  # $5M/day tradeability floor (matches the rotation flagship floor)
VOLWIN = 30


def _tag_series(df):
    """Per-bar point-in-time trailing vol (% annualized) and 20-bar dollar volume."""
    close = df["Close"].values
    n = len(close)
    ret = np.zeros(n); ret[1:] = close[1:] / close[:-1] - 1
    vol = pd.Series(ret).rolling(VOLWIN).std().values * np.sqrt(2 * 252) * 100
    if "Volume" in df:
        dv = (df["Close"] * df["Volume"]).rolling(20).mean().values
    else:
        dv = np.full(n, np.inf)
    return vol, dv


def collect(mode, allowedC, store, sectors):
    """mode: 'highvol' | 'highvol_in_C' | 'C_only'. Builds trades for the chosen universe."""
    import intraday_data as ID
    d4 = ID.DATA / "4h"
    trades = []
    for p in sorted(d4.glob("*.parquet")):
        tk = p.stem
        df = ID.get_4h(tk, 5, False)
        if df is None or len(df) < 120:
            continue
        close = df["Close"].values
        ts = df.index
        n = len(close)
        vol, dv = _tag_series(df)
        cwin = allowedC.get(tk, set())
        fire = np.zeros(n, dtype=bool)
        for s in SIGS:
            e, _ = H.SIGNALS[s]["fn"](df); fire |= np.asarray(e, dtype=bool)
        idxs = sorted(H._episode_starts([i for i in range(n) if fire[i]], gap=H.GAP))
        for i in idxs:
            if i + 1 >= n or i < VOLWIN or close[i] <= 0:
                continue
            d = ts[i].date()
            in_c = d in cwin
            hv = np.isfinite(vol[i]) and VOL_LO <= vol[i] <= VOL_HI and (dv[i] or 0) >= MIN_DVOL
            if mode == "highvol" and not hv:
                continue
            if mode == "highvol_in_C" and not (hv and in_c):
                continue
            if mode == "C_only" and not in_c:
                continue
            up = upside_asof(store, tk, d, float(close[i]))
            sched = [(ts[i + b], float(close[i + b] / close[i + b - 1] - 1))
                     for b in range(1, E.MAXHOLD + 1) if i + b < n]
            if not sched:
                continue
            trades.append({"entry_ts": ts[i], "bucket": bucket_upside(up), "upside": up,
                           "sector": sectors.get(tk, tk), "sched": sched})
    return trades


def main():
    os.environ.setdefault("DJANGO_SETTINGS_MODULE", "rotation.settings")
    import django; django.setup()
    from pathlib import Path
    from h4_on_signals_study import candidate_windows
    allowedC, _ = candidate_windows("C")
    store = load_targets()
    sectors = E._sector_map()
    daily, spybar = E._spy(5)

    BASE = {"weight": "steep_2x", "hold": E.HOLD_FIXED, "gate": True}
    BEST = {"weight": "steep_4x", "hold": E.HOLD_FIXED, "gate": False, "hedge_frac": 0.5}
    rows = []
    for uni in ["C_only", "highvol_in_C", "highvol"]:
        trades = collect(uni, allowedC, store, sectors)
        for cfgname, cfg in [("baseline", BASE), ("steep4x+hedge50", BEST)]:
            m = E.simulate(trades, daily, spybar, cfg)
            rows.append({"universe": uni, "config": cfgname, "n_trades_pool": len(trades), **m})
            print(f"  {uni:14} {cfgname:16} n{len(trades):>6}  total {m['total_return_pct']:>8}%  "
                  f"DD {m['max_dd_pct']:>7}  Sh {m['sharpe']:>5}  taken {m['n_taken']}", flush=True)

    payload = {"computed_at": pd.Timestamp.utcnow().isoformat(), "rows": rows,
               "params": {"vol_band_pct": [VOL_LO, VOL_HI], "min_dvol": MIN_DVOL, "signals": SIGS},
               "note": ("H4 dip-buy on a high-volatility liquid universe (PIT trailing vol in [50,300]%, $5M "
                        "dvol floor) vs high-vol∩C vs C-only. Cash-aware engine; baseline (steep_2x+gate) and "
                        "the winning steep_4x+SPY-hedge config. Gross of fees; cached-4h universe.")}
    Path("/app/.data/studies").mkdir(parents=True, exist_ok=True)
    Path("/app/.data/studies/h4_vol_dipbuy.json").write_text(json.dumps(payload, indent=2, default=str))
    try:
        from core.models import BacktestResult
        from django.utils import timezone
        BacktestResult.objects.update_or_create(kind="h4_vol_dipbuy",
            defaults={"payload": json.loads(json.dumps(payload, default=str)), "computed_at": timezone.now()})
        print("saved BacktestResult[h4_vol_dipbuy]", flush=True)
    except Exception as e:
        print("DB save failed:", e, flush=True)


if __name__ == "__main__":
    print("=== H4 DIP-BUY: high-vol universe vs C ===", flush=True)
    main()
