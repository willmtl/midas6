"""Isolate the METALS contribution now that Gold/Silver hold miners. Over the backtest months, when
Gold and/or Silver rank in the top-10 6mo-momentum, pick the cheapest positive PIT-P/B miner in that
sleeve and measure its forward relative return vs SPY. Answers: do the metal sleeves now fire, what do
they pick, and do the miner picks beat SPY? Run: docker exec rotation-backend-1 python /app/_metals_contribution.py
"""
import os, warnings
warnings.filterwarnings("ignore")
os.environ.setdefault("DJANGO_SETTINGS_MODULE", "rotation.settings")
import django
django.setup()
import numpy as np, pandas as pd
import config, sector_holdings
from seq_fundamental_study import load_candles, load_financial_reports
from trend_stock_studies import _pit_monthly_panel, _available_at, _ret_delist, CRYPTO
from backtest_lowpb import _monthly_close, BENCH
import price_basis

LOOKBACK, TOP_N = 6, 10
SLEEVES = {"Gold": "GLD", "Silver": "SLV"}


def _hit(v):
    a = np.asarray(v, float); a = a[~np.isnan(a)]
    return (round(float((a > 0).mean()) * 100, 1), round(float(a.mean()) * 100, 2), len(a)) if len(a) else (None, None, 0)


def main():
    etfs = {n: e for n, e in config.SECTOR_ETFS.items() if e not in CRYPTO}
    miners = {s: [t for t in sector_holdings.get_holdings(s) if t] for s in SLEEVES}
    allmin = sorted({t for lst in miners.values() for t in lst})

    daily = load_candles(list(etfs.values()) + [BENCH] + allmin)
    etf_m = _monthly_close({t: d for t, d in daily.items() if t in etfs.values()})
    midx = etf_m.index
    spy_m = daily[BENCH]["Close"].resample("ME").last().reindex(midx)
    etf_trail = etf_m.pct_change(LOOKBACK)
    min_m = _monthly_close({t: d for t, d in daily.items() if t in allmin}).reindex(midx)

    reps = load_financial_reports(allmin)
    shares = _pit_monthly_panel(reps, "shares_outstanding", midx)
    equity = _pit_monthly_panel(reps, "total_equity", midx)
    common = min_m.columns.intersection(shares.columns).intersection(equity.columns)
    pb = (price_basis.as_traded_close(min_m[common]) * shares[common]) / equity[common].where(equity[common] != 0)

    def pick(sleeve, date):
        cands = [t for t in miners[sleeve] if t in min_m.columns and _available_at(min_m[t], date) and t in pb.columns]
        if not cands or date not in pb.index:
            return None
        row = pb.loc[date, cands].dropna(); row = row[row > 0]
        return row.idxmin() if len(row) else None

    rows = {s: [] for s in SLEEVES}
    picked = {s: {} for s in SLEEVES}
    top_months = {s: 0 for s in SLEEVES}
    for i in range(LOOKBACK, len(midx) - 1):
        date, ndate = midx[i], midx[i + 1]
        ranks = etf_trail.loc[date].dropna().sort_values(ascending=False).head(TOP_N).index
        sp = spy_m.iloc[i + 1] / spy_m.iloc[i] - 1
        for s, etf in SLEEVES.items():
            if etf not in ranks:
                continue
            top_months[s] += 1
            pk = pick(s, date)
            if pk is None:
                continue
            r = _ret_delist(min_m[pk], date, ndate)
            if r is None or not np.isfinite(r):
                continue
            rows[s].append(float(r) - float(sp))
            picked[s][pk] = picked[s].get(pk, 0) + 1

    print(f"\n=== METALS CONTRIBUTION (fwd 1mo return vs SPY when the sleeve is top-{TOP_N} momentum) ===", flush=True)
    print(f"months: {len(midx)} | lookback {LOOKBACK}mo\n", flush=True)
    allr = []
    for s in SLEEVES:
        hr, mn, n = _hit(rows[s])
        allr += rows[s]
        names = ", ".join(f"{k}x{v}" for k, v in sorted(picked[s].items(), key=lambda x: -x[1]))
        print(f"{s:7} in top-{TOP_N}: {top_months[s]:2} months | picked {n} times | vs SPY hit {hr}% mean {mn}% | names: {names or '(none pickable)'}", flush=True)
    hr, mn, n = _hit(allr)
    print(f"\nMETALS combined: {n} sleeve-months | vs SPY hit {hr}% mean {mn}%", flush=True)
    print("(mean = avg forward 1mo return of the metal miner pick MINUS SPY that month)", flush=True)


if __name__ == "__main__":
    main()
