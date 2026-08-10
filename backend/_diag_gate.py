import os, django
os.environ.setdefault("DJANGO_SETTINGS_MODULE", "rotation.settings")
django.setup()

import numpy as np, pandas as pd
from seq_fundamental_study import build_universe, load_candles, load_financial_reports
from studies import SIGNALS
from all_on_all_study import _prepare_indicators
from pit_fundamentals import _ad_state
from core.models import Fundamental
from api.tasks import GICS2ETF

MAXH, TRAIL = 90, 20


def trail_exit(close, p, n):
    peak = close[p]
    for k in range(p + 1, min(p + MAXH, n)):
        if close[k] > peak:
            peak = close[k]
        if close[k] <= peak * (1 - TRAIL / 100):
            return k
    return min(p + MAXH, n - 1)


tks = build_universe()
sec = {r['ticker']: r['sector'] for r in Fundamental.objects.filter(ticker__in=tks).values('ticker', 'sector')}
t2etf = {t: GICS2ETF[s] for t, s in sec.items() if s in GICS2ETF}
reports = load_financial_reports(tks)
mkt = load_candles(sorted(set(t2etf.values())) + ['SPY'])
spy = mkt['SPY']['Close']
spy63 = spy.pct_change(63)

# per-ETF gate series
G = {}
for e in set(t2etf.values()):
    if e not in mkt:
        continue
    ec = mkt[e]['Close']
    spa = spy.reindex(ec.index).ffill()
    rs = ec / spa
    G[e] = {
        'lag': ec.pct_change(63) - spy63.reindex(ec.index).ffill(),  # >0 = already outperforming (lagging)
        'rsup': rs - rs.rolling(20).mean(),                           # >0 = RS above its own trend (leading)
        'accum': _ad_state(mkt[e]),                                   # ==2 = sector being accumulated (leading)
    }
cap = lambda df: (SIGNALS['new_52low'][1](df).fillna(False).values | SIGNALS['rsi_oversold20'][1](df).fillna(False).values)

GATES = ['none', 'lag(63d RS>0)', 'lead:RS-up', 'lead:sector-accum', 'lead:RS-up OR accum']
recs = []
cd = load_candles([t for t in tks if t in t2etf])
for tk, df in cd.items():
    if len(df) < 260:
        continue
    rep = reports.get(tk)
    pit_dates = pit_sh = None
    if rep is not None and len(rep):
        r2 = rep.dropna(subset=["avail_date", "shares_outstanding"]).sort_values("avail_date")
        if len(r2):
            pit_dates = pd.DatetimeIndex(pd.to_datetime(r2["avail_date"]).to_numpy())
            pit_sh = r2["shares_outstanding"].to_numpy(dtype=float)
    e = t2etf[tk]
    if e not in G:
        continue
    _prepare_indicators(df)
    close = df['Close'].values; n = len(close); idx = df.index
    st = _ad_state(df).values
    A = cap(df) & (st == 2)
    g = G[e]
    for p in range(252, n - 2):
        if not A[p] or close[p] < 1 or p + 5 >= n:
            continue
        d = idx[p]
        if pit_dates is not None:
            j = int(pit_dates.searchsorted(d, "right")) - 1
            if j < 0 or np.isnan(pit_sh[j]) or pit_sh[j] * close[p] < 300e6:
                continue
        else:
            continue
        lag = g['lag'].asof(d); rsup = g['rsup'].asof(d); ac = g['accum'].asof(d)
        gates = {
            'none': True,
            'lag(63d RS>0)': pd.notna(lag) and lag > 0,
            'lead:RS-up': pd.notna(rsup) and rsup > 0,
            'lead:sector-accum': ac == 2,
            'lead:RS-up OR accum': (pd.notna(rsup) and rsup > 0) or (ac == 2),
        }
        xi = trail_exit(close, p, n)
        ret = (close[xi] - close[p]) / close[p] * 100
        recs.append((gates, ret))

print(f"MODE A stock trades, trailing {TRAIL}%/{MAXH}b, under different SECTOR gates:\n")
print(f'{"gate":22}{"n":>6}{"win%":>7}{"median":>8}{"avg":>8}{">=50%":>7}{"expectancy":>11}')
for gname in GATES:
    a = np.array([r for gates, r in recs if gates[gname]])
    if not len(a):
        print(f'{gname:22} empty'); continue
    print(f'{gname:22}{len(a):6d}{(a>0).mean()*100:7.1f}{np.median(a):+8.1f}{a.mean():+8.1f}{(a>=50).mean()*100:7.1f}{a.mean():+11.1f}')
