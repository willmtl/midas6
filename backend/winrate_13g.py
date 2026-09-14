#!/usr/bin/env python3
"""Per-trade WIN RATIO for the 13G book (exact book def: H=126, PIT .shift(1) entry, $ADV+price floors,
20bps round-trip, survivorship-aware exit at last bar). Reports, per variant (all/small) and per hold horizon,
the fraction of positions that closed net-positive + mean/median net return. Also the book's daily-return win rate.
Run: MSYS_NO_PATHCONV=1 docker exec rotation-backend-1 python -u /app/winrate_13g.py"""
import os, sys
sys.path.insert(0, "/app")
os.environ.setdefault("DJANGO_SETTINGS_MODULE", "rotation.settings")
import django; django.setup()
import numpy as np, pandas as pd
from seq_fundamental_study import load_candles, load_filings, load_financial_reports, MIN_BARS, _chunk
from signal_discovery import _universe, DVOL_FLOOR, PRICE_FLOOR

HORIZONS = {"1m": 21, "3m": 63, "6m": 126, "12m": 252}
COST_RT = 20 / 1e4            # round-trip fraction
SMALL_MAX = 2e9


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
    from core.models import SecFiling
    have = set(SecFiling.objects.filter(form_group="13G").values_list("ticker", flat=True).distinct())
    names = [t for t in universe if t in have]
    # rec: dict horizon->net return, + is_small flag
    rows = {"all": {h: [] for h in HORIZONS}, "small": {h: [] for h in HORIZONS}}
    done = 0
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
            fired = ev.groupby(ev.index).sum().reindex(sdf.index).shift(1).fillna(0.0).values
            dates = sdf.index; sh = shares_ff(reps, tk, dates); is_del = tk in delisted
            for i in np.where(fired > 0)[0]:
                ep = float(close[i])
                if ep <= PRICE_FLOOR or not (dvol20[i] >= DVOL_FLOOR):
                    continue
                mc = (sh[i] * ep) if (sh is not None and np.isfinite(sh[i])) else None
                small = mc is not None and np.isfinite(mc) and 0 < mc < SMALL_MAX
                for h, k in HORIZONS.items():
                    j = i + k
                    if j < n:
                        xp = float(close[j])
                    elif is_del:
                        xp = float(close[-1])
                    else:
                        continue
                    net = (xp - ep) / ep - COST_RT          # round-trip cost
                    rows["all"][h].append(net)
                    if small:
                        rows["small"][h].append(net)
        done += len(ch)
        print(f"  scanned {done}/{len(names)}", flush=True)

    print("\n=== 13G per-trade WIN RATIO (net of 20bps round-trip, survivorship-aware) ===", flush=True)
    print(f"  {'variant':8}{'horizon':8}{'n':>7}{'win%':>8}{'mean%':>9}{'median%':>9}", flush=True)
    for v in ("all", "small"):
        for h in HORIZONS:
            a = np.asarray(rows[v][h], float); a = a[np.isfinite(a)]
            if len(a) < 20:
                continue
            print(f"  {v:8}{h:8}{len(a):>7}{(a>0).mean()*100:>8.1f}{a.mean()*100:>9.2f}{np.median(a)*100:>9.2f}", flush=True)


if __name__ == "__main__":
    main()
