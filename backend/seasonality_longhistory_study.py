#!/usr/bin/env python3
"""LONG-HISTORY SECTOR SEASONALITY — the 5y test was too short (2-4 samples/month). Pull FULL monthly history
(EODHD, from 1999/inception) for every sector ETF WITHOUT touching the working Candle table, and measure
per-sector calendar-month seasonality with 15-25y of data (~15-25 samples/month). For each (sector, month):
mean return, t-stat, % positive. Flag significant months (|t|>=2). ROBUSTNESS: split history in half and
correlate the 12 month-means (first vs second half) — real seasonality PERSISTS, noise doesn't.
-> BacktestResult[seasonality_longhistory] + JSON.
Run: MSYS_NO_PATHCONV=1 docker exec rotation-backend-1 python -u /app/seasonality_longhistory_study.py
"""
import os, json, warnings
warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
os.environ.setdefault("DJANGO_SETTINGS_MODULE", "rotation.settings")
import django
django.setup()

from pathlib import Path
import config
from api.tasks import _eodhd_get, _eodhd_sym
from trend_stock_studies import CRYPTO
from scipy.stats import ttest_1samp

MIN_YEARS = 12
OUT = Path(__file__).resolve().parent / ".data" / "studies" / "seasonality_longhistory.json"


def monthly_returns(sym):
    r = _eodhd_get(f"eod/{sym}", **{"from": "1999-01-01", "period": "m"})
    if not isinstance(r, list) or len(r) < 24:
        return None
    s = {}
    for row in r:
        d = row.get("date"); a = row.get("adjusted_close", row.get("close"))
        if d and a not in (None, ""):
            try:
                s[pd.Timestamp(d)] = float(a)
            except (TypeError, ValueError):
                pass
    if len(s) < 24:
        return None
    px = pd.Series(s).sort_index()
    return px.pct_change().dropna()


def build():
    etfs = {n: e for n, e in config.SECTOR_ETFS.items() if e not in CRYPTO}
    name_of = {e: n for n, e in etfs.items()}
    print(f"fetching long monthly history for {len(etfs)} sector ETFs from EODHD...", flush=True)
    rets = {}
    for n, e in etfs.items():
        sym = _eodhd_sym(e)
        if sym is None:
            continue
        r = monthly_returns(sym)
        if r is not None and len(r) >= MIN_YEARS * 12 * 0.8:
            rets[e] = r
    print(f"usable (>= ~{MIN_YEARS}y): {len(rets)} sectors", flush=True)

    per_sector = {}
    persist = {}
    sig_rows = []
    for e, r in rets.items():
        m = r.groupby(r.index.month)
        table = {}
        for mo in range(1, 13):
            v = r[r.index.month == mo]
            if len(v) < 5:
                continue
            mean = float(v.mean()) * 100
            t = float(ttest_1samp(v, 0.0)[0]) if len(v) > 2 else None
            table[mo] = {"mean_pct": round(mean, 2), "t": round(t, 2) if t is not None else None,
                         "n": len(v), "pos_pct": round(float((v > 0).mean()) * 100, 0)}
            if t is not None and abs(t) >= 2:
                sig_rows.append((name_of[e], mo, round(mean, 2), round(t, 2), len(v)))
        per_sector[name_of[e]] = {"years": round(len(r) / 12, 1), "by_month": table}
        # split-half persistence: corr of 12 month-means, first vs second half
        half = len(r) // 2
        a = r.iloc[:half].groupby(r.iloc[:half].index.month).mean()
        b = r.iloc[half:].groupby(r.iloc[half:].index.month).mean()
        j = a.index.intersection(b.index)
        if len(j) >= 10:
            persist[name_of[e]] = round(float(np.corrcoef(a[j], b[j])[0, 1]), 2)

    # aggregate market month pattern (equal-weight across sectors with full history)
    allr = pd.concat(rets.values())
    agg = {int(mo): round(float(allr[allr.index.month == mo].mean()) * 100, 2) for mo in range(1, 13)}
    persist_vals = [v for v in persist.values() if np.isfinite(v)]
    med_persist = round(float(np.median(persist_vals)), 2) if persist_vals else None
    sig_rows.sort(key=lambda x: -abs(x[3]))

    print(f"\n=== aggregate month-of-year return (all sectors, %) ===", flush=True)
    print("  " + "  ".join(f"{mo}:{agg[mo]}" for mo in range(1, 13)), flush=True)
    print(f"  best {max(agg, key=agg.get)} ({agg[max(agg, key=agg.get)]}%)  worst {min(agg, key=agg.get)} ({agg[min(agg, key=agg.get)]}%)", flush=True)
    print(f"\n=== significant sector-months (|t|>=2), top 18 ===", flush=True)
    for nm, mo, mean, t, n in sig_rows[:18]:
        print(f"  {nm:<24} month {mo:>2}: {mean:>6}%  t {t:>5}  (n {n})", flush=True)
    print(f"\n=== SPLIT-HALF PERSISTENCE (corr of month-means 1st vs 2nd half; >0 = seasonality repeats) ===", flush=True)
    print(f"  median across sectors: {med_persist}   (most persistent: "
          + ", ".join(f"{k}({v})" for k, v in sorted(persist.items(), key=lambda x: -x[1])[:6]) + ")", flush=True)

    verdict = (
        f"Full history ({len(rets)} sectors, up to ~{max(v['years'] for v in per_sector.values())}y). "
        f"Aggregate: best month {max(agg, key=agg.get)} ({agg[max(agg, key=agg.get)]}%), worst {min(agg, key=agg.get)} "
        f"({agg[min(agg, key=agg.get)]}%). {len(sig_rows)} significant sector-months at |t|>=2. "
        f"SPLIT-HALF persistence median {med_persist}: "
        + ("seasonality REPEATS across halves -> real, usable as a tilt with long-history estimation."
           if med_persist is not None and med_persist > 0.3 else
           "seasonality does NOT persist across halves (median corr ~0) -> even 20y of monthly data, sector "
           "seasonality is mostly noise; the 'significant' months are largely in-sample artifacts.")
    )
    print("\n" + verdict, flush=True)
    return {"computed_at": pd.Timestamp.utcnow().isoformat(),
            "params": {"min_years": MIN_YEARS, "n_sectors": len(rets), "source": "EODHD monthly from 1999"},
            "aggregate_by_month_pct": agg, "significant_sector_months": sig_rows[:40],
            "split_half_persistence": persist, "median_persistence": med_persist,
            "per_sector": per_sector, "verdict": verdict,
            "caveat": "EODHD monthly adjusted close from 1999/inception (not stored in Candle table). Only sectors "
                      "with >=~12y counted (thematics like Cannabis/Quantum too young). t-tests on overlapping "
                      "calendar months; split-half is the honest robustness check. In-sample means overstate."}


def main():
    p = build()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(p, indent=2, default=str))
    try:
        from core.models import BacktestResult
        from django.utils import timezone
        BacktestResult.objects.update_or_create(
            kind="seasonality_longhistory", defaults={"payload": json.loads(json.dumps(p, default=str)), "computed_at": timezone.now()})
        print("Saved BacktestResult[seasonality_longhistory]", flush=True)
    except Exception as e:
        print("DB save failed:", e, flush=True)


if __name__ == "__main__":
    main()
