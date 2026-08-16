#!/usr/bin/env python3
"""Can we identify SPY-beating sectors in advance? Prospective hit-rate of momentum-ranked sectors.

At each month, rank the 93 sector ETFs by trailing 6mo return; take the top-10 and bottom-10. Measure what
fraction then BEAT SPY over the next 1/3/6/12 months, and the avg forward RELATIVE return, vs the all-sector
base rate. If top-momentum barely clears the base rate, sector winners are essentially un-timable at the ETF
level (which is why rotate-only loses to SPY).
Run: MSYS_NO_PATHCONV=1 docker exec rotation-backend-1 python -u /app/sector_hitrate.py
"""
import os, warnings
warnings.filterwarnings("ignore")
os.environ.setdefault("DJANGO_SETTINGS_MODULE", "rotation.settings")
import django
django.setup()
import numpy as np, pandas as pd
import config
from backtest_lowpb import _monthly_close, BENCH, CRYPTO
from seq_fundamental_study import load_candles

HZ = [1, 3, 6, 12]
WARMUP = 12


def hit(vals):
    a = np.asarray(vals, float); a = a[~np.isnan(a)]
    if not len(a):
        return (None, None, 0)
    return (round(float((a > 0).mean()) * 100, 1), round(float(a.mean()) * 100, 2), len(a))


def main():
    etfs = [e for e in config.SECTOR_ETFS.values() if e and e not in CRYPTO]
    data = load_candles(etfs + [BENCH])
    em = _monthly_close({t: d for t, d in data.items() if t in etfs})
    midx = em.index
    spy = data[BENCH]["Close"].resample("ME").last().reindex(midx)
    mom6 = em.pct_change(6)

    groups = {"top10 (mom)": [], "bottom10 (mom)": [], "all (base rate)": []}
    fwd = {g: {h: [] for h in HZ} for g in groups}
    for i in range(WARMUP, len(midx) - 1):
        d = midx[i]
        ranked = mom6.loc[d].dropna().sort_values(ascending=False)
        top = set(ranked.head(10).index); bot = set(ranked.tail(10).index)
        for h in HZ:
            if i + h >= len(midx):
                continue
            d2 = midx[i + h]
            srel = (em.loc[d2] / em.loc[d] - 1) - (spy.loc[d2] / spy.loc[d] - 1)
            for e in ranked.index:
                v = srel.get(e)
                if not np.isfinite(v):
                    continue
                fwd["all (base rate)"][h].append(v)
                if e in top:
                    fwd["top10 (mom)"][h].append(v)
                elif e in bot:
                    fwd["bottom10 (mom)"][h].append(v)

    print(f"\n=== SECTOR SPY-BEAT HIT-RATE ({len(midx)} months) — forward RELATIVE return vs SPY ===", flush=True)
    print(f"{'group':18} | " + "  ".join(f"+{h}mo (hit% / mean%)" for h in HZ), flush=True)
    for g in ["top10 (mom)", "all (base rate)", "bottom10 (mom)"]:
        cells = []
        for h in HZ:
            hr, mn, n = hit(fwd[g][h])
            cells.append(f"{hr}% / {mn:>+5}%")
        print(f"{g:18} | " + "   ".join(cells), flush=True)
    print("\n(lift = top10 hit% minus base-rate hit%; ~0 => momentum can't identify SPY-beaters at the ETF level)", flush=True)


if __name__ == "__main__":
    main()
