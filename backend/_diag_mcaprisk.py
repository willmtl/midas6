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
spy = mkt['SPY']['Close']; spy63 = spy.pct_change(63)
cap = lambda df: (SIGNALS['new_52low'][1](df).fillna(False).values | SIGNALS['rsi_oversold20'][1](df).fillna(False).values)

recs = []
cd = load_candles([t for t in tks if t in t2etf])
for tk, df in cd.items():
    if len(df) < 260:
        continue
    rep = reports.get(tk)
    if rep is None or not len(rep):
        continue
    r2 = rep.dropna(subset=["avail_date", "shares_outstanding"]).sort_values("avail_date")
    if not len(r2):
        continue
    pit_dates = pd.DatetimeIndex(pd.to_datetime(r2["avail_date"]).to_numpy())
    pit_sh = r2["shares_outstanding"].to_numpy(dtype=float)
    _prepare_indicators(df)
    close = df['Close'].values; n = len(close); idx = df.index
    st = _ad_state(df).values
    e = mkt[t2etf[tk]]['Close']; e63 = e.pct_change(63)
    A = cap(df) & (st == 2)
    for p in range(252, n - 2):
        if not A[p] or close[p] < 1 or p + 5 >= n:
            continue
        d = idx[p]
        if not (pd.notna(e63.asof(d)) and pd.notna(spy63.asof(d)) and e63.asof(d) > spy63.asof(d)):
            continue
        j = int(pit_dates.searchsorted(d, "right")) - 1
        if j < 0 or np.isnan(pit_sh[j]):
            continue
        mc = pit_sh[j] * close[p]
        xi = trail_exit(close, p, n)
        recs.append((mc, (close[xi] - close[p]) / close[p] * 100))

D = pd.DataFrame(recs, columns=['mcap', 'ret'])
BINS = [(0, 100e6, '<100M'), (100e6, 300e6, '100-300M'), (300e6, 1e9, '300M-1B'),
        (1e9, 3e9, '1-3B'), (3e9, 10e9, '3-10B'), (10e9, 50e9, '10-50B'), (50e9, 1e15, '>=50B')]
print(f"RISK vs MARKET CAP — Mode A, trailing {TRAIL}%/{MAXH}b. n={len(D)}\n")
print(f'{"cap band":11}{"n":>5}{"win%":>7}{"avg":>7}{"median":>8}{"avgLoss":>8}{"worst":>7}{"<-33%":>7}{"vol(std)":>9}{">=50%":>7}')
for lo, hi, nm in BINS:
    a = D[(D['mcap'] >= lo) & (D['mcap'] < hi)]['ret'].values
    if len(a) < 5:
        print(f'{nm:11}{len(a):5d}  (too few)'); continue
    l = a[a <= 0]
    print(f'{nm:11}{len(a):5d}{(a>0).mean()*100:7.1f}{a.mean():+7.1f}{np.median(a):+8.1f}'
          f'{(l.mean() if len(l) else 0):+8.1f}{a.min():+7.1f}{(a<-33).mean()*100:7.1f}{a.std():9.1f}{(a>=50).mean()*100:7.1f}')
