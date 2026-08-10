import os, django
os.environ.setdefault("DJANGO_SETTINGS_MODULE", "rotation.settings")
django.setup()

import numpy as np, pandas as pd
from datetime import timedelta
from seq_fundamental_study import build_universe, load_candles, load_financial_reports
from studies import SIGNALS, _exit_sort_above
from all_on_all_study import _prepare_indicators
from pit_fundamentals import _ad_state
from core.models import Fundamental, InsiderBuy, SecFiling
from api.tasks import GICS2ETF

COST, CAP0, MAXPOS, RM, GROSS_CAP = 0.003, 100_000.0, 8, 0.06, 1.5

tks = build_universe()
sec = {r['ticker']: r['sector'] for r in Fundamental.objects.filter(ticker__in=tks).values('ticker', 'sector')}
t2etf = {t: GICS2ETF[s] for t, s in sec.items() if s in GICS2ETF}
reports = load_financial_reports(tks)
mkt = load_candles(sorted(set(t2etf.values())) + ['SPY'])
spy = mkt['SPY']['Close']; spy63 = spy.pct_change(63)
cap = lambda df: (SIGNALS['new_52low'][1](df).fillna(False).values | SIGNALS['rsi_oversold20'][1](df).fillna(False).values)
sm = {}
for t, d in InsiderBuy.objects.filter(ticker__in=tks).values_list('ticker', 'filed_date'):
    sm.setdefault(t, []).append(pd.Timestamp(d))
for t, d in SecFiling.objects.filter(ticker__in=tks, form_group__in=['13D', '13G']).values_list('ticker', 'filed_date'):
    sm.setdefault(t, []).append(pd.Timestamp(d))
sm = {k: np.array(sorted(v)) for k, v in sm.items()}


def has_sm(tk, d):
    a = sm.get(tk)
    if a is None:
        return False
    return np.searchsorted(a, d, 'right') - np.searchsorted(a, d - pd.Timedelta(days=180), 'left') > 0


trades = []
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
    pdd = pd.DatetimeIndex(pd.to_datetime(r2["avail_date"]).to_numpy()); sh = r2["shares_outstanding"].to_numpy(dtype=float)
    _prepare_indicators(df)
    close = df['Close'].values; n = len(close); idx = df.index
    st = _ad_state(df).values
    e = mkt[t2etf[tk]]['Close']; e63 = e.pct_change(63)
    hi = pd.Series(close).rolling(252).max().values; lo = pd.Series(close).rolling(252).min().values
    pos = (close - lo) / np.where(hi - lo == 0, np.nan, hi - lo)
    A = cap(df) & (st == 2)
    try:
        B = SIGNALS['rsi_oversold30'][1](df).fillna(False).values & (pos >= 0.75)
    except Exception:
        B = np.zeros(n, bool)
    le = -1
    for p in range(252, n - 2):
        if p <= le or not (A[p] or B[p]) or close[p] < 5:
            continue
        d = idx[p]
        if not (pd.notna(e63.asof(d)) and pd.notna(spy63.asof(d)) and e63.asof(d) > spy63.asof(d)):
            continue
        j = int(pdd.searchsorted(d, "right")) - 1
        if j < 0 or np.isnan(sh[j]) or sh[j] * close[p] < 300e6:
            continue
        xi = _exit_sort_above(df, p, 1)
        smf = has_sm(tk, d)
        conv = 'HIGH' if (A[p] and smf) else ('MID' if A[p] else 'LOW')
        trades.append({'entry': d, 'conv': conv, 'ret': (close[xi] - close[p]) / close[p] * 100,
                       'ep': close[p], 'xp': close[xi], 'exit': idx[xi], 'path': pd.Series(close[p:xi + 1], index=idx[p:xi + 1])})
        le = xi
print(f"{len(trades)} trades")
for c in ['HIGH', 'MID', 'LOW']:
    a = np.array([t['ret'] for t in trades if t['conv'] == c])
    if len(a):
        print(f"  {c:5} n={len(a):4d} win={(a>0).mean()*100:4.1f}% avg={a.mean():+5.1f}%")

by_entry = {}
for t in trades:
    by_entry.setdefault(t['entry'], []).append(t)
cal = spy.index[spy.index >= min(t['entry'] for t in trades)]


def run(cal_slice, mult):
    spx = spy.reindex(cal_slice).ffill(); units = CAP0 / spx.iloc[0]; debt = 0.0; op = []; eq = []
    for d in cal_slice:
        sp = spx.loc[d]
        debt *= (1 + RM / 252)
        keep = []
        for o in op:
            if d >= o['exit']:
                proc = o['shares'] * o['xp'] * (1 - COST)
                pay = min(debt, proc); debt -= pay; units += (proc - pay) / sp
            else:
                keep.append(o)
        op = keep
        for t in sorted(by_entry.get(d, []), key=lambda x: {'HIGH': 0, 'MID': 1, 'LOW': 2}[x['conv']]):
            if len(op) >= MAXPOS:
                break
            E = units * sp + sum(o['shares'] * float(o['path'].asof(d)) for o in op) - debt
            gross = units * sp + sum(o['shares'] * float(o['path'].asof(d)) for o in op)
            size = (E / MAXPOS) * mult[t['conv']]
            if gross + size > GROSS_CAP * E:
                size = max(0, GROSS_CAP * E - gross)
            if size < 100:
                continue
            sell = min(size, units * sp); units -= sell / sp; debt += (size - sell)
            op.append({'shares': size * (1 - COST) / t['ep'], 'xp': t['xp'], 'exit': t['exit'], 'path': t['path']})
        eq.append(units * sp + sum(o['shares'] * float(o['path'].asof(d)) for o in op) - debt)
    return pd.Series(eq, index=cal_slice)


def stats(eq):
    r = eq.pct_change().dropna(); yrs = (eq.index[-1] - eq.index[0]).days / 365.25
    return ((eq.iloc[-1] / eq.iloc[0]) ** (1 / yrs) - 1, (eq / eq.cummax() - 1).min(),
            r.mean() / r.std() * np.sqrt(252) if r.std() > 0 else 0)


CONFIGS = {'baseline 1x': {'HIGH': 1, 'MID': 1, 'LOW': 1},
           'lever HIGH/MID': {'HIGH': 2.0, 'MID': 1.4, 'LOW': 0.7}}
print(f"\nPORTFOLIO (SPY overlay, sort_gt1, gross cap {GROSS_CAP}x, margin {RM*100:.0f}%):")
for name, mult in CONFIGS.items():
    for lab, sl in [('FULL', cal), ('OOS25-26', cal[cal >= '2025-01-01'])]:
        c, dd, s = stats(run(sl, mult))
        print(f"  {name:16} {lab:9} CAGR {c*100:+5.1f}%  DD {dd*100:6.1f}%  Sharpe {s:.2f}")
c, dd, s = stats(spy.reindex(cal).ffill())
print(f"  {'SPY buy-hold':16} FULL      CAGR {c*100:+5.1f}%  DD {dd*100:6.1f}%  Sharpe {s:.2f}")
