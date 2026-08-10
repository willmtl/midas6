import os, django
os.environ.setdefault("DJANGO_SETTINGS_MODULE", "rotation.settings")
django.setup()

import numpy as np, pandas as pd
from seq_fundamental_study import build_universe, load_candles, load_fundamentals
from studies import SIGNALS, _exit_sort_above
from all_on_all_study import _prepare_indicators
from core.models import Fundamental
from api.tasks import GICS2ETF, is_low_quality

tks = build_universe()
sec = {r['ticker']: r['sector'] for r in Fundamental.objects.filter(ticker__in=tks).values('ticker', 'sector')}
t2etf = {t: GICS2ETF[s] for t, s in sec.items() if s in GICS2ETF}
funds = load_fundamentals(tks)
mkt = load_candles(sorted(set(t2etf.values())) + ['SPY'])
r63 = {e: mkt[e]['Close'].pct_change(63) for e in mkt}
spy = r63['SPY']
fos30 = SIGNALS['rsi_oversold30'][1]
sg = lambda df, i: _exit_sort_above(df, i, 1)

# collect Mode B candidate entries (gated + quality) with position AND pct-below-high
recs = []          # (position, pct_below_high, ret)
below_high_at_075 = []
cd = load_candles([t for t in tks if t in t2etf])
for tk, df in cd.items():
    if len(df) < 260:
        continue
    f = funds.get(tk, {})
    if is_low_quality(f.get('market_cap'), None, f.get('profit_margin')):
        continue
    _prepare_indicators(df)
    close = df['Close'].values; n = len(close); idx = df.index
    hi = pd.Series(close).rolling(252).max().values
    lo = pd.Series(close).rolling(252).min().values
    pos = (close - lo) / np.where(hi - lo == 0, np.nan, hi - lo)
    pbh = (1 - close / hi) * 100    # % below the 52wk high
    os30 = fos30(df).fillna(False).values
    e = r63.get(t2etf[tk])
    for p in range(252, n - 2):
        if not os30[p] or close[p] < 5 or np.isnan(pos[p]):
            continue
        d = idx[p]; rv = e.asof(d); sv = spy.asof(d)
        if not (pd.notna(rv) and pd.notna(sv) and rv > sv):
            continue
        xi = sg(df, p)
        if xi is None or xi <= p or xi >= n:
            continue
        ret = (close[xi] - close[p]) / close[p] * 100
        recs.append((pos[p], pbh[p], ret))
        if pos[p] >= 0.75:
            below_high_at_075.append(pbh[p])

D = pd.DataFrame(recs, columns=['pos', 'pbh', 'ret'])
b = np.array(below_high_at_075)
print(f"At position>=0.75 (current Mode B), 'near the high' means on avg {b.mean():.0f}% below the 52wk high "
      f"(median {np.median(b):.0f}%, range {np.percentile(b,10):.0f}-{np.percentile(b,90):.0f}%).\n")


def rep(name, mask):
    a = D['ret'][mask].values
    if len(a) == 0:
        print(f'{name:34} empty'); return
    print(f'{name:34} n={len(a):4d}  win={(a>0).mean()*100:5.1f}%  median={np.median(a):+6.1f}%  avg={a.mean():+6.1f}%')


print("=== range-position thresholds (top X of 52wk range) ===")
for th in [0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85]:
    rep(f'position >= {th:.2f}', D['pos'] >= th)
print("=== range-position BANDS (monotonic check) ===")
for a, b2 in [(0.0, 0.25), (0.25, 0.50), (0.50, 0.65), (0.65, 0.75), (0.75, 0.85), (0.85, 1.01)]:
    rep(f'position {a:.2f}-{b2:.2f}', (D['pos'] >= a) & (D['pos'] < b2))
print("=== alt: within X% of the 52wk high ===")
for x in [30, 25, 20, 15, 12, 10, 8, 5]:
    rep(f'within {x}% of high', D['pbh'] <= x)
