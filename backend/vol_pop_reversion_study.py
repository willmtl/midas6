#!/usr/bin/env python3
"""DOES AN EXTREME POP IN A LOW-VOL NAME REVERT? — broad-universe event study (investigate the flagship
pop-exit finding at scale: the flagship only had ~12 events; here N is in the thousands).

For every stock: daily close-to-close returns, trailing-60d daily vol (shifted, so it's PRE-pop). An EVENT
= a single-day close-to-close jump >= threshold. Bucket events by the name's trailing vol (LOW < 2%/day,
MID 2-4%, HIGH > 4%) and by threshold (10/15/20/25/30%). Measure FORWARD return from the pop close at
1/5/10/21 trading days. Reversion => negative forward return. Tests whether the "extreme pop in a low-vol
name reverts, routine pop doesn't" pattern holds with real sample size + a clean t-stat.
-> .data/studies/vol_pop_reversion.json + BacktestResult[vol_pop_reversion].
Run: MSYS_NO_PATHCONV=1 docker exec rotation-backend-1 python -u /app/vol_pop_reversion_study.py
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

OUT = Path("/app/.data/studies/vol_pop_reversion.json")
THRESHOLDS = [0.10, 0.15, 0.20, 0.25, 0.30]
HORIZONS = [1, 5, 10, 21]
VOL_BUCKETS = [("LOW <2%/d", 0.0, 0.02), ("MID 2-4%", 0.02, 0.04), ("HIGH >4%", 0.04, 9.9)]


def main():
    with connection.cursor() as c:
        c.execute("SET max_parallel_workers_per_gather=0")
    from core.models import Candle, Fundamental, Sector, BacktestResult
    from django.utils import timezone

    etfs = set(Sector.objects.values_list("etf", flat=True)) | {"SPY", "QQQ"}
    tickers = sorted(set(Fundamental.objects.values_list("ticker", flat=True)) - etfs)
    print(f"universe (stocks w/ fundamentals): {len(tickers)}", flush=True)

    events = []      # (thr_hit_flags..., vol, fwd returns...) captured per pop day
    n_names = 0
    for tk in tickers:
        q = list(Candle.objects.filter(ticker=tk, interval="1d").order_by("date").values_list("date", "close"))
        if len(q) < 120:
            continue
        s = pd.Series({pd.Timestamp(d): float(c) for d, c in q}).sort_index()
        s = s[s > 0]
        if len(s) < 120:
            continue
        n_names += 1
        ret = s.pct_change()
        tvol = ret.rolling(60).std().shift(1)          # trailing daily vol, PRE-pop (no look-ahead)
        logp = np.log(s)
        # forward cumulative returns at each horizon (from the pop-day close)
        fwd = {h: (s.shift(-h) / s - 1) for h in HORIZONS}
        # candidate pop days: any single-day return >= smallest threshold, with a valid trailing vol
        mask = (ret >= THRESHOLDS[0]) & tvol.notna() & (tvol > 0)
        idx = s.index[mask]
        for dt in idx:
            r = float(ret[dt]); v = float(tvol[dt])
            row = {"tk": tk, "ret": r, "vol": v}
            for h in HORIZONS:
                fv = fwd[h].get(dt)
                row[f"f{h}"] = float(fv) if pd.notna(fv) else np.nan
            events.append(row)
    df = pd.DataFrame(events)
    print(f"names used: {n_names} | raw pop events (>= {THRESHOLDS[0]*100:.0f}% 1-day): {len(df)}", flush=True)

    def bucket(v):
        for name, lo, hi in VOL_BUCKETS:
            if lo <= v < hi:
                return name
        return "HIGH >4%"
    df["vb"] = df["vol"].apply(bucket)

    out = {"computed_at": pd.Timestamp.utcnow().isoformat(), "n_names": n_names, "n_events": int(len(df)),
           "grid": {}}
    print("\n=== forward return after a single-day pop, by TRAILING-VOL bucket × threshold ===", flush=True)
    print("  (reversion => NEGATIVE forward return; f5 = 5-day fwd; t = t-stat of f5)\n", flush=True)
    for vb, lo, hi in VOL_BUCKETS:
        print(f"  --- trailing vol {vb} ---", flush=True)
        print(f"    {'pop>=':>6}{'n':>7}{'f1':>9}{'f5':>9}{'f10':>9}{'f21':>9}{'f5 t':>8}{'f5 win%':>9}", flush=True)
        for thr in THRESHOLDS:
            g = df[(df["vb"] == vb) & (df["ret"] >= thr)]
            g5 = g["f5"].dropna()
            if len(g5) < 10:
                continue
            row = {h: round(float(df.loc[g.index, f"f{h}"].mean()) * 100, 2) for h in HORIZONS}
            t5 = _tstat_from_returns(list(g5))
            win5 = float((g5 > 0).mean()) * 100
            out["grid"][f"{vb}|{int(thr*100)}"] = {"n": int(len(g)), **{f"f{h}": row[h] for h in HORIZONS},
                                                    "f5_t": round(t5, 2) if t5 is not None else None, "f5_win": round(win5, 1)}
            print(f"    {int(thr*100):>5}%{len(g):>7}{row[1]:>+8.2f}{row[5]:>+8.2f}{row[10]:>+8.2f}{row[21]:>+8.2f}"
                  f"{(round(t5,2) if t5 is not None else 0):>8}{win5:>8.0f}%", flush=True)
        print("", flush=True)

    # headline: LOW-vol names, extreme (>=20%) pop, 5-day forward
    low20 = df[(df["vb"] == VOL_BUCKETS[0][0]) & (df["ret"] >= 0.20)]["f5"].dropna()
    high20 = df[(df["vb"] == VOL_BUCKETS[2][0]) & (df["ret"] >= 0.20)]["f5"].dropna()
    if len(low20) >= 10:
        t = _tstat_from_returns(list(low20))
        out["verdict"] = (f"LOW-vol names, >=20% 1-day pop, 5-day fwd = {low20.mean()*100:+.2f}% (n={len(low20)}, "
                          f"t={t:.2f}, win {100*(low20>0).mean():.0f}%); HIGH-vol contrast = {high20.mean()*100:+.2f}% "
                          f"(n={len(high20)}). " + ("LOW-vol extreme pops REVERT (negative fwd) — supports the exit."
                          if low20.mean() < -0.002 else "No clean reversion at scale — the flagship signal was small-sample."))
        print("VERDICT:", out["verdict"], flush=True)

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(out, indent=2, default=str))
    try:
        BacktestResult.objects.update_or_create(kind="vol_pop_reversion",
            defaults={"payload": json.loads(json.dumps(out, default=str)), "computed_at": timezone.now()})
        print("Saved BacktestResult[vol_pop_reversion]", flush=True)
    except Exception as e:
        print("DB save failed:", e, flush=True)
    print("DONE_VOLPOP_REV", flush=True)


if __name__ == "__main__":
    main()
