#!/usr/bin/env python3
"""TAKE-PROFIT-AT-100%, TESTED CORRECTLY (in the flagship's hold/rotation context). The broad test showed
doubled stocks keep running +16.5%/63d — but the flagship never captures that: a doubled name gets
expensive + its sector cools, so it's rotated out. The RIGHT question (user): among flagship winners that
double, is TAKING PROFIT at +100% better than RIDING until the sector falls out of the top-10 ranking
(which — per rank_band_exit — happens AFTER the peak, exiting deep)?

Reconstructs top-10 sector membership per month from the flagship trace. Each pick = a position held from
its buy month until its sector (ETF) leaves the top-10 (capped 18mo). Loads the daily path. Compares, per
position and on the doubled subset:
  ride  = exit at sector-fallout (close-to-close)
  tp100 = exit at +100% if the path hits it before fallout, else ride
-> .data/studies/takeprofit_hold.json + BacktestResult[takeprofit_hold].
Run: MSYS_NO_PATHCONV=1 docker exec rotation-backend-1 python -u /app/takeprofit_hold_study.py
"""
import os, json, warnings
warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
os.environ.setdefault("DJANGO_SETTINGS_MODULE", "rotation.settings")
import django
django.setup()
from pathlib import Path
from django.db import connection
from studies import _tstat_from_returns

TRACE = Path("/app/.data/studies/flagship_history.json")
OUT = Path("/app/.data/studies/takeprofit_hold.json")
CAP_MONTHS = 18


def main():
    with connection.cursor() as c:
        c.execute("SET max_parallel_workers_per_gather=0")
    from core.models import Candle, BacktestResult
    from django.utils import timezone

    d = json.load(open(TRACE))
    months = d["months"]
    mdates = [pd.Timestamp(m["date"]) for m in months]
    # top-10 ETF membership per month
    topset = [set(s["etf"] for s in m["top_sectors"]) for m in months]
    midx_of = {dt: i for i, dt in enumerate(mdates)}

    # positions: first buy of each (ticker, etf) run; hold until the etf leaves top-10 (or +CAP_MONTHS / data end)
    positions = []
    seen = set()
    for i, m in enumerate(months):
        for p in m["picks"]:
            if p.get("ret") is None:
                continue
            key = (p["ticker"], p["etf"])
            if key in seen:
                continue
            seen.add(key)
            # fallout month: first j>i where etf not in top-10
            j = i
            while j + 1 < len(months) and (j + 1 - i) <= CAP_MONTHS and p["etf"] in topset[j + 1]:
                j += 1
            positions.append({"tk": p["ticker"], "etf": p["etf"], "entry": mdates[i], "exit_m": mdates[min(j + 1, len(months) - 1)]})
    tickers = sorted({p["tk"] for p in positions})
    print(f"positions (first-entry episodes): {len(positions)} | tickers {len(tickers)} | months {len(months)}", flush=True)

    cand = {}
    for tk in tickers:
        q = list(Candle.objects.filter(ticker=tk, interval="1d").order_by("date").values_list("date", "close"))
        if len(q) > 60:
            cand[tk] = pd.Series({pd.Timestamp(dt): float(c) for dt, c in q}).sort_index()

    def trail_exit(prices, e, drop, activate=None):
        """Return the exit cum-return under a trailing stop: exit when price falls `drop` from the running
        peak (optionally only ARMED after cum first reaches `activate`). Else the final (ride) return."""
        peak = e
        armed = activate is None
        for px in prices:
            cum = px / e - 1
            if activate is not None and cum >= activate:
                armed = True
            if px > peak:
                peak = px
            if armed and px <= peak * (1 - drop):
                return px / e - 1
        return prices[-1] / e - 1

    rec = []
    for p in positions:
        s = cand.get(p["tk"])
        if s is None:
            continue
        entry_s = s[s.index <= p["entry"]]
        path = s[(s.index > p["entry"]) & (s.index <= p["exit_m"])]
        if len(entry_s) < 1 or len(path) < 2:
            continue
        e = float(entry_s.iloc[-1])
        if e <= 0:
            continue
        pv = path.values
        cum = path / e - 1
        ride = float(cum.iloc[-1])                          # exit at sector-fallout
        doubled = bool((cum >= 1.0).any())
        rec.append({"tk": p["tk"], "hold_m": round((p["exit_m"] - p["entry"]).days / 30.4, 1),
                    "ride": ride, "tp": (1.0 if doubled else ride), "peak": float(cum.max()), "doubled": doubled,
                    "trail_15": trail_exit(pv, e, 0.15),
                    "trail_20": trail_exit(pv, e, 0.20),
                    "trail_30": trail_exit(pv, e, 0.30),
                    "act30_tr20": trail_exit(pv, e, 0.20, activate=0.30)})
    df = pd.DataFrame(rec)
    print(f"positions with path: {len(df)} | avg hold {df['hold_m'].mean():.1f}mo | "
          f"ever doubled: {df['doubled'].sum()} ({100*df['doubled'].mean():.0f}%)", flush=True)

    def comp(col):
        r = df[col].values
        return round(float(np.mean(r)) * 100, 2), round(float(np.median(r)) * 100, 2)

    rm, rmed = comp("ride"); tm, tmed = comp("tp")
    out = {"computed_at": pd.Timestamp.utcnow().isoformat(), "n_positions": int(len(df)),
           "avg_hold_months": round(float(df["hold_m"].mean()), 1),
           "pct_doubled": round(float(df["doubled"].mean()) * 100, 1),
           "all": {"ride_mean": rm, "tp_mean": tm, "ride_med": rmed, "tp_med": tmed}}
    print(f"\n  ALL positions (hold-to-sector-fallout): ride mean {rm:+.1f}% / med {rmed:+.1f}%  |  "
          f"take-profit@100% mean {tm:+.1f}% / med {tmed:+.1f}%  -> TP {'BETTER' if tm > rm else 'WORSE'} by {tm-rm:+.1f}pp", flush=True)
    print(f"\n  TRAILING-STOP variants (per-position mean / median return vs ride {rm:+.1f}%/{rmed:+.1f}%):", flush=True)
    out["trailing"] = {}
    for col, lab in [("trail_15", "trail 15% from peak"), ("trail_20", "trail 20% from peak"),
                     ("trail_30", "trail 30% from peak"), ("act30_tr20", "arm at +30%, then trail 20%")]:
        cm, cmed = comp(col)
        out["trailing"][col] = {"mean": cm, "median": cmed, "vs_ride": round(cm - rm, 2)}
        print(f"    {lab:32} mean {cm:>+6.1f}%  med {cmed:>+6.1f}%  -> {cm-rm:>+5.1f}pp vs ride", flush=True)

    dd = df[df["doubled"]]
    if len(dd):
        rmd = float(dd["ride"].mean()) * 100; tmd = float(dd["tp"].mean()) * 100
        peak = float(dd["peak"].mean()) * 100
        out["doubled_subset"] = {"n": int(len(dd)), "ride_mean": round(rmd, 1), "tp_mean": round(tmd, 1),
                                 "avg_peak": round(peak, 1),
                                 "gaveback_after_double_to_fallout": round(tmd - rmd, 1)}
        print(f"\n  >>> POSITIONS THAT DOUBLED (n={len(dd)}): avg PEAK {peak:+.0f}%; "
              f"ride-to-fallout ends at {rmd:+.0f}%  vs  take-profit@100% locks {tmd:+.0f}%  -> "
              f"{'TAKE-PROFIT WINS (they give back after the double)' if tmd > rmd else 'RIDING WINS (they keep running past the double)'} "
              f"by {tmd-rmd:+.0f}pp", flush=True)
        out["verdict"] = (f"Of {len(df)} flagship positions held to sector-fallout, {len(dd)} doubled (avg peak "
                          f"{peak:+.0f}%). Riding to fallout ends at {rmd:+.0f}% vs take-profit@100% {tmd:+.0f}% -> "
                          + ("take-profit at the double BEATS riding to fallout (winners give back after doubling once the sector cools)."
                             if tmd > rmd else "riding past the double still wins (they keep running even to sector-fallout)."))
        print("\nVERDICT:", out["verdict"], flush=True)

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(out, indent=2, default=str))
    try:
        BacktestResult.objects.update_or_create(kind="takeprofit_hold",
            defaults={"payload": json.loads(json.dumps(out, default=str)), "computed_at": timezone.now()})
        print("Saved BacktestResult[takeprofit_hold]", flush=True)
    except Exception as e:
        print("DB save failed:", e, flush=True)
    print("DONE_TPHOLD", flush=True)


if __name__ == "__main__":
    main()
