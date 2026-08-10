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
BANDS = [(300e6, 1e9, '300M-1B'), (1e9, 3e9, '1-3B'), (3e9, 10e9, '3-10B'),
         (10e9, 50e9, '10-50B'), (50e9, 1e15, '>=50B')]


def band_of(mc):
    for lo, hi, nm in BANDS:
        if lo <= mc < hi:
            return nm
    return None


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

rows = []
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
    pd_ = pd.DatetimeIndex(pd.to_datetime(r2["avail_date"]).to_numpy()); sh = r2["shares_outstanding"].to_numpy(dtype=float)
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
        j = int(pd_.searchsorted(d, "right")) - 1
        if j < 0 or np.isnan(sh[j]):
            continue
        b = band_of(sh[j] * close[p])
        if b is None:
            continue
        xi = trail_exit(close, p, n)
        rows.append({'band': b, 'q': idx[p].to_period('Q'), 'ret': (close[xi] - close[p]) / close[p] * 100})

D = pd.DataFrame(rows)
names = [nm for _, _, nm in BANDS]
# per-band per-trade stats
print("Per-band (per-trade) return character:")
print(f'{"band":10}{"n":>5}{"mean%":>8}{"vol%":>7}{"Sharpe(r/σ)":>12}')
mu_t, sd_t = {}, {}
for nm in names:
    a = D[D['band'] == nm]['ret'].values
    mu_t[nm], sd_t[nm] = a.mean(), a.std()
    print(f'{nm:10}{len(a):5d}{a.mean():+8.1f}{a.std():7.1f}{a.mean()/a.std():12.2f}')

# quarterly return matrix per band -> covariance / correlation
Q = D.groupby(['q', 'band'])['ret'].mean().unstack('band').reindex(columns=names)
Q = Q.dropna(how='all')
corr = Q.corr()
print("\nBand correlation (quarterly return streams):")
print(corr.round(2).to_string())

mu = Q.mean().values      # quarterly expected return per band
cov = Q.cov().values
sd = np.sqrt(np.diag(cov))
valid = ~np.isnan(mu)
names_v = [n for n, v in zip(names, valid) if v]
mu, cov, sd = mu[valid], cov[np.ix_(valid, valid)], sd[valid]

# Monte-Carlo long-only frontier
rng = np.random.default_rng(1)
best_sh, best_mv = None, None
for _ in range(40000):
    w = rng.dirichlet(np.ones(len(mu)))
    r = w @ mu; v = np.sqrt(w @ cov @ w)
    sh = r / v if v > 0 else 0
    if best_sh is None or sh > best_sh[0]:
        best_sh = (sh, r, v, w)
    if best_mv is None or v < best_mv[2]:
        best_mv = (sh, r, v, w)
invv = (1 / sd) / (1 / sd).sum()   # risk-parity-lite (inverse vol)
ir = invv @ mu; iv = np.sqrt(invv @ cov @ invv)


def show(tag, r, v, w):
    print(f'\n{tag}: quarterly ret {r:+.1f}%  vol {v:.1f}%  Sharpe {r/v:.2f}')
    for nm, wt in zip(names_v, w):
        print(f'    {nm:10} {wt*100:5.1f}%')


print("\n=== MPT optimal allocations (long-only) ===")
show('MAX-SHARPE (tangency)', best_sh[1], best_sh[2], best_sh[3])
show('MIN-VARIANCE', best_mv[1], best_mv[2], best_mv[3])
show('INVERSE-VOL (robust)', ir, iv, invv)
ew = np.ones(len(mu)) / len(mu)
show('EQUAL-WEIGHT (baseline)', ew @ mu, np.sqrt(ew @ cov @ ew), ew)
