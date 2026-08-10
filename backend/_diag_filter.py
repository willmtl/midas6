import os, django
os.environ.setdefault("DJANGO_SETTINGS_MODULE", "rotation.settings")
django.setup()

import numpy as np, pandas as pd
from seq_fundamental_study import build_universe, load_candles, load_fundamentals
from studies import SIGNALS
from all_on_all_study import _prepare_indicators
from pit_fundamentals import _ad_state
from core.models import Fundamental

G = {'Technology':'XLK','Healthcare':'XLV','Energy':'XLE','Financial Services':'XLF',
     'Consumer Cyclical':'XLY','Consumer Defensive':'XLP','Industrials':'XLI',
     'Basic Materials':'XLB','Real Estate':'XLRE','Communication Services':'XLC','Utilities':'XLU'}
tks = build_universe()
sec = {r['ticker']: r['sector'] for r in Fundamental.objects.filter(ticker__in=tks).values('ticker','sector')}
t2etf = {t: G[s] for t, s in sec.items() if s in G}
funds = load_fundamentals(tks)
mkt = load_candles(sorted(set(t2etf.values())) + ['SPY'])
r63 = {e: mkt[e]['Close'].pct_change(63) for e in mkt}
spy = r63['SPY']
f52 = SIGNALS['new_52low'][1]; frsi = SIGNALS['rsi_oversold20'][1]; fos30 = SIGNALS['rsi_oversold30'][1]

recs = []
cd = load_candles([t for t in tks if t in t2etf])
for tk, df in cd.items():
    if len(df) < 260:
        continue
    _prepare_indicators(df)
    close = df['Close'].values; n = len(close); idx = df.index
    st = _ad_state(df).values
    s52 = f52(df).fillna(False).values; srsi = frsi(df).fillna(False).values; so30 = fos30(df).fillna(False).values
    hi = pd.Series(close).rolling(252).max().values; lo = pd.Series(close).rolling(252).min().values
    posn = (close - lo) / np.where(hi - lo == 0, np.nan, hi - lo)
    A = (s52 | srsi) & (st == 2); Bm = so30 & (posn >= 0.75)
    e = r63.get(t2etf[tk]); f = funds.get(tk, {})
    for p in range(252, n - 2):
        if not (A[p] or Bm[p]) or p + 90 >= n:
            continue
        d = idx[p]; rv = e.asof(d); sv = spy.asof(d)
        if not (pd.notna(rv) and pd.notna(sv) and rv > sv):
            continue
        ret = (close[p + 90] - close[p]) / close[p] * 100
        mode = 'A' if A[p] else 'B'
        if A[p]:
            trig = '52low' if (s52[p] and not srsi[p]) else ('rsi20' if (srsi[p] and not s52[p]) else 'both')
        else:
            trig = 'Bdip'
        recs.append((mode, trig, ret, f.get('market_cap'), close[p], f.get('profit_margin')))

D = pd.DataFrame(recs, columns=['mode', 'trig', 'ret', 'mcap', 'price', 'margin'])
qual = (D['mcap'] >= 300e6) & (D['price'] >= 5) & ((D['margin'].isna()) | (D['margin'] >= 0))


def rep(name, sub):
    a = sub['ret'].values
    if len(a) == 0:
        print(f'{name:26} empty'); return
    micro = (sub['mcap'] < 300e6).mean() * 100
    print(f'{name:26} n={len(a):4d} win={(a>0).mean()*100:5.1f}% med={np.median(a):+6.1f}% worst={a.min():+7.1f}% <-33%={(a<-33).mean()*100:4.1f}% micro%={micro:4.0f}')


print('=== TRIGGER breakdown within Mode A (unfiltered) ===')
rep('52low only', D[(D['mode'] == 'A') & (D['trig'] == '52low')])
rep('rsi20 only', D[(D['mode'] == 'A') & (D['trig'] == 'rsi20')])
rep('both', D[(D['mode'] == 'A') & (D['trig'] == 'both')])
print('=== MODE B (uptrend dip): does the quality filter matter? ===')
rep('B all', D[D['mode'] == 'B'])
rep('B quality-filtered', D[(D['mode'] == 'B') & qual])
bb = D[D['mode'] == 'B']
ndis = int((bb['ret'] < -33).sum())
mshare = round((bb[bb['ret'] < -33]['mcap'] < 300e6).mean() * 100, 0) if ndis else 0
print(f'   B disasters(<-33): n={ndis}  microcap share={mshare}%')
print('=== MODE A filtered, by trigger ===')
rep('52low quality-filtered', D[(D['mode'] == 'A') & (D['trig'] == '52low') & qual])
rep('rsi20 quality-filtered', D[(D['mode'] == 'A') & (D['trig'] == 'rsi20') & qual])
