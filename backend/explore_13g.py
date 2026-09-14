#!/usr/bin/env python3
"""EXPLORE the stake_13g signal (institutional 13G passive >5% stake filed) before designing a book.
Characterize: breadth (events/yr -> capacity), cap-bucket distribution, RAW forward-return distribution
(mean/median/win + percentiles -> is the edge broad or tail-driven?), and BY-YEAR robustness. Survivorship-aware
(exit at last available bar; -100% only if confirmed bankrupt delisting). PIT: event is .shift(1) (enter session
after filing). Segmented by cap bucket per HARD RULE. NOT demeaned here — we want ABSOLUTE deployable return.
Run: MSYS_NO_PATHCONV=1 docker exec rotation-backend-1 python -u /app/explore_13g.py"""
import os, sys
sys.path.insert(0, "/app")
os.environ.setdefault("DJANGO_SETTINGS_MODULE", "rotation.settings")
import django; django.setup()
import numpy as np, pandas as pd
from collections import defaultdict
from seq_fundamental_study import load_candles, load_filings, load_financial_reports, MIN_BARS, _chunk
from signal_discovery import _universe, DVOL_FLOOR, PRICE_FLOOR

HORIZONS = {"1m": 21, "3m": 63, "6m": 126, "12m": 252}
DELISTED = set()


def cap_bucket(mc):
    if mc is None or not np.isfinite(mc) or mc <= 0:
        return "cap?"
    b = mc / 1e9
    return "mega>200" if b >= 200 else "large10-200" if b >= 10 else "mid2-10" if b >= 2 else "small<2"


def shares_ff(reps, tk, dates):
    r = reps.get(tk)
    if r is None or "shares_outstanding" not in r.columns:
        return None
    d = r[["avail_date", "shares_outstanding"]].dropna().sort_values("avail_date")
    if d.empty:
        return None
    s = pd.Series(d["shares_outstanding"].values, index=pd.to_datetime(d["avail_date"]))
    s = s[~s.index.duplicated(keep="last")].sort_index()
    return s.reindex(s.index.union(dates)).ffill().reindex(dates).values


def main():
    universe, delisted = _universe()
    DELISTED.update(delisted)
    # restrict to names that actually have 13G filings (fast)
    from core.models import SecFiling
    have13g = set(SecFiling.objects.filter(form_group="13G").values_list("ticker", flat=True).distinct())
    names = [t for t in universe if t in have13g]
    print(f"universe {len(universe)} | with 13G filings {len(have13g)} | tradable ∩ {len(names)}", flush=True)

    # records: one per (event) -> dict of forward returns + cap + year + delist flag
    recs = []
    breadth = defaultdict(set)   # (year, month) -> set of tickers firing (for capacity)
    n_done = 0
    for ch in _chunk(names, 40):
        candles = load_candles(ch); filings = load_filings(ch); reps = load_financial_reports(ch)
        for tk, sdf in candles.items():
            if len(sdf) < MIN_BARS:
                continue
            close = sdf["Close"].values; vol = sdf["Volume"].values; n = len(close)
            dvol20 = pd.Series(close * vol).rolling(20, min_periods=10).mean().values
            fdf = filings.get(tk)
            if fdf is None or not len(fdf):
                continue
            sub = fdf[fdf["form_group"] == "13G"]
            if not len(sub):
                continue
            ev = pd.Series(1.0, index=pd.to_datetime(sub["filed_date"]))
            fired = ev.groupby(ev.index).sum().reindex(sdf.index).shift(1).fillna(0.0).values  # PIT shift
            dates = sdf.index
            sh = shares_ff(reps, tk, dates)
            is_delisted = tk in DELISTED
            for i in np.where(fired > 0)[0]:
                ep = float(close[i])
                if ep <= PRICE_FLOOR or not (dvol20[i] >= DVOL_FLOOR):
                    continue
                mc = (sh[i] * ep) if (sh is not None and np.isfinite(sh[i])) else None
                rec = {"tk": tk, "year": dates[i].year, "cap": cap_bucket(mc),
                       "ym": (dates[i].year, dates[i].month), "dvol": float(dvol20[i])}
                for hz, k in HORIZONS.items():
                    j = i + k
                    if j < n:
                        rec[hz] = (float(close[j]) - ep) / ep * 100.0
                    elif is_delisted:
                        # survivorship: exit at last bar (approx last/deal price; -100% only if truly bankrupt unknown here)
                        rec[hz] = (float(close[-1]) - ep) / ep * 100.0
                    else:
                        rec[hz] = np.nan
                recs.append(rec)
                breadth[rec["ym"]].add(tk)
        n_done += len(ch)
        print(f"  scanned {n_done}/{len(names)}  events {len(recs)}", flush=True)

    df = pd.DataFrame(recs)
    print(f"\nTOTAL 13G events (tradable, PIT): {len(df)}", flush=True)

    # ---- capacity / breadth ----
    bym = pd.Series({k: len(v) for k, v in breadth.items()})
    print(f"\n=== CAPACITY (distinct names firing per month) ===", flush=True)
    print(f"  median {bym.median():.0f}  mean {bym.mean():.1f}  p90 {bym.quantile(.9):.0f}  max {bym.max():.0f}", flush=True)
    print(f"  median $ADV of an event: ${df['dvol'].median()/1e6:.1f}M", flush=True)

    # ---- cap distribution ----
    print(f"\n=== CAP-BUCKET distribution of events ===", flush=True)
    print(df["cap"].value_counts().to_string(), flush=True)

    def dist(a):
        a = np.asarray(a, float); a = a[np.isfinite(a)]
        if len(a) < 30:
            return None
        return dict(n=len(a), mean=a.mean(), med=np.median(a), win=(a > 0).mean() * 100,
                    p10=np.percentile(a, 10), p90=np.percentile(a, 90),
                    trimmed=a[(a > np.percentile(a, 1)) & (a < np.percentile(a, 99))].mean())

    # ---- raw forward-return distribution overall + by cap ----
    for hz in HORIZONS:
        print(f"\n=== RAW forward {hz} return — ALL then by cap (ABSOLUTE, not demeaned) ===", flush=True)
        for seg, sub in [("ALL", df)] + [(c, df[df["cap"] == c]) for c in ["mega>200", "large10-200", "mid2-10", "small<2"]]:
            d = dist(sub[hz])
            if d:
                print(f"  {seg:14} n{d['n']:>6}  mean {d['mean']:+6.2f}%  trim99 {d['trimmed']:+6.2f}%  "
                      f"med {d['med']:+6.2f}%  win {d['win']:4.1f}%  p10 {d['p10']:+.1f}  p90 {d['p90']:+.1f}", flush=True)

    # ---- BY-YEAR (6m, the headline horizon) overall + small-cap ----
    print(f"\n=== BY-YEAR 6m return (is it broad or one era?) ===", flush=True)
    print(f"  {'year':>6}{'n_all':>7}{'mean_all':>10}{'win_all':>8}   |{'n_small':>8}{'mean_small':>12}{'win_small':>9}", flush=True)
    for y in sorted(df["year"].unique()):
        a = df[df["year"] == y]["6m"].dropna().values
        s = df[(df["year"] == y) & (df["cap"] == "small<2")]["6m"].dropna().values
        if len(a) < 10:
            continue
        sm = f"{len(s):>8}{np.mean(s):>12.2f}{(s>0).mean()*100:>9.1f}" if len(s) >= 5 else f"{'':>8}{'-':>12}{'-':>9}"
        print(f"  {y:>6}{len(a):>7}{np.mean(a):>10.2f}{(a>0).mean()*100:>8.1f}   |{sm}", flush=True)


if __name__ == "__main__":
    main()
