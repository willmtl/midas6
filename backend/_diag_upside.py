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

MAXH = 90  # ~4 months, so a 50%+ move has room to develop under a trail
TRAIL = 20  # % trailing stop


def trail_exit(close, p, n, trail_pct=TRAIL, max_hold=MAXH):
    peak = close[p]
    for k in range(p + 1, min(p + max_hold, n)):
        if close[k] > peak:
            peak = close[k]
        if close[k] <= peak * (1 - trail_pct / 100):
            return k
    return min(p + max_hold, n - 1)


tks = build_universe()
sec = {r['ticker']: r['sector'] for r in Fundamental.objects.filter(ticker__in=tks).values('ticker', 'sector')}
t2etf = {t: GICS2ETF[s] for t, s in sec.items() if s in GICS2ETF}
reports = load_financial_reports(tks)
mkt = load_candles(sorted(set(t2etf.values())) + ['SPY'])
r63 = {e: mkt[e]['Close'].pct_change(63) for e in mkt}
spy = r63['SPY']
cap = lambda df: (SIGNALS['new_52low'][1](df).fillna(False).values | SIGNALS['rsi_oversold20'][1](df).fillna(False).values)

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
    _prepare_indicators(df)
    close = df['Close'].values; n = len(close); idx = df.index
    st = _ad_state(df).values
    e = r63.get(t2etf[tk])
    A = cap(df) & (st == 2)
    for p in range(252, n - 2):
        if not A[p] or close[p] < 1 or p + 5 >= n:  # tails back in: only sub-$1 excluded
            continue
        d = idx[p]; rv = e.asof(d); sv = spy.asof(d)
        if not (pd.notna(rv) and pd.notna(sv) and rv > sv):
            continue
        mcap = np.nan
        if pit_dates is not None:
            j = int(pit_dates.searchsorted(d, "right")) - 1
            if j >= 0 and not np.isnan(pit_sh[j]):
                mcap = pit_sh[j] * close[p]
        xi = trail_exit(close, p, n)
        recs.append({'mcap': mcap, 'ret': (close[xi] - close[p]) / close[p] * 100, 'hold': xi - p})

D = pd.DataFrame(recs)
a = D['ret'].values
w = a[a > 0]; l = a[a <= 0]
print(f"MODE A, TAILS BACK IN (no size filter, >$1), trailing {TRAIL}% / max {MAXH} bars. n={len(D)}\n")
print(f"  win={ (a>0).mean()*100:.1f}%  median={np.median(a):+.1f}%  avg={a.mean():+.1f}%  avg hold={D['hold'].mean():.0f}d")
print(f"  avg WIN={w.mean():+.1f}%  avg LOSS={l.mean():+.1f}%  win/loss size={w.mean()/abs(l.mean()):.2f}x")
print(f"  >=50%: {(a>=50).mean()*100:.1f}%   >=100%: {(a>=100).mean()*100:.1f}%   worst={a.min():+.1f}%")
print(f"  expectancy/trade = {a.mean():+.1f}%\n")
print("  by size at entry:")
for nm, m in [('micro <300M', D['mcap'] < 300e6), ('small .3-2B', (D['mcap'] >= 300e6) & (D['mcap'] < 2e9)),
              ('mid 2-10B', (D['mcap'] >= 2e9) & (D['mcap'] < 10e9)), ('large >=10B', D['mcap'] >= 10e9)]:
    s = D[m]['ret'].values
    if len(s):
        print(f'    {nm:12} n={len(s):5d}  win={(s>0).mean()*100:4.1f}%  avg={s.mean():+6.1f}%  >=50%:{(s>=50).mean()*100:4.1f}%  >=100%:{(s>=100).mean()*100:4.1f}%  worst={s.min():+6.1f}%')

# --- MSFT April->Aug 2025 case study ---
print("\n  MSFT case (Apr->Aug 2025):")
m = cd.get('MSFT')
if m is not None:
    sub = m[(m.index >= '2025-03-15') & (m.index <= '2025-08-15')]
    if len(sub):
        lo_i = sub['Close'].idxmin(); lo = sub['Close'].min()
        hi_after = sub[sub.index >= lo_i]['Close'].max()
        print(f"    low {lo:.2f} on {lo_i.date()} -> peak {hi_after:.2f} = {(hi_after-lo)/lo*100:+.1f}% by mid-Aug")
