import os, django
os.environ.setdefault("DJANGO_SETTINGS_MODULE", "rotation.settings")
django.setup()

import numpy as np, pandas as pd
from seq_fundamental_study import build_universe, load_candles, load_financial_reports
from studies import SIGNALS, _exit_sort_above
from all_on_all_study import _prepare_indicators
from pit_fundamentals import _ad_state
from core.models import Fundamental
from api.tasks import GICS2ETF

MAXH, TRAIL, COST, CAP0, MAXPOS, TVOL = 90, 20, 0.003, 100_000.0, 8, 30.0


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
        sg = _exit_sort_above(df, p, 1); tr = trail_exit(close, p, n)
        trades.append({'entry': d, 'mode': 'A' if A[p] else 'B',
                       'sg': {'xp': close[sg], 'exit': idx[sg], 'ret': (close[sg] - close[p]) / close[p] * 100,
                              'path': pd.Series(close[p:sg + 1], index=idx[p:sg + 1]), 'ep': close[p]},
                       'tr': {'xp': close[tr], 'exit': idx[tr], 'ret': (close[tr] - close[p]) / close[p] * 100,
                              'path': pd.Series(close[p:tr + 1], index=idx[p:tr + 1]), 'ep': close[p]}})
        le = tr
print(f"generated {len(trades)} trades\n")

for ex in ['sg', 'tr']:
    a = np.array([t[ex]['ret'] for t in trades])
    nm = 'sort_gt1 (momentum-turn)' if ex == 'sg' else 'trailing 20%'
    hold = np.mean([(t[ex]['exit'] - t['entry']).days for t in trades])
    print(f"TRADE-LEVEL {nm:26} win={ (a>0).mean()*100:4.1f}%  median={np.median(a):+5.1f}%  avg={a.mean():+5.1f}%  >=50%={(a>=50).mean()*100:4.1f}%  hold~{hold:.0f}d")

cal = spy.index[spy.index >= min(t['entry'] for t in trades)]


def run_overlay(cal_slice, ex):
    be = {}
    for t in trades:
        be.setdefault(t['entry'], []).append(t)
    spx = spy.reindex(cal_slice).ffill(); units = CAP0 / spx.iloc[0]; op = []; eq = []
    for d in cal_slice:
        sp = spx.loc[d]; keep = []
        for o in op:
            if d >= o['exit']:
                units += o['shares'] * o['xp'] * (1 - COST) / sp
            else:
                keep.append(o)
        op = keep
        for t in sorted(be.get(d, []), key=lambda x: (0 if x['mode'] == 'A' else 1)):
            if len(op) >= MAXPOS:
                break
            leg = t[ex]
            E = units * sp + sum(o['shares'] * float(o['path'].asof(d)) for o in op)
            size = min(units * sp, E / MAXPOS)
            if size < 100:
                continue
            units -= size / sp
            op.append({'shares': size * (1 - COST) / leg['ep'], 'xp': leg['xp'], 'exit': leg['exit'], 'path': leg['path']})
        eq.append(units * sp + sum(o['shares'] * float(o['path'].asof(d)) for o in op))
    return pd.Series(eq, index=cal_slice)


def stats(eq):
    r = eq.pct_change().dropna(); yrs = (eq.index[-1] - eq.index[0]).days / 365.25
    return ((eq.iloc[-1] / eq.iloc[0]) ** (1 / yrs) - 1, (eq / eq.cummax() - 1).min(),
            r.mean() / r.std() * np.sqrt(252) if r.std() > 0 else 0)


print("\nPORTFOLIO (SPY overlay) by exit:")
for ex in ['sg', 'tr']:
    for lab, sl in [('FULL', cal), ('OOS25-26', cal[cal >= '2025-01-01'])]:
        eq = run_overlay(sl, ex); c, dd, s = stats(eq)
        print(f"  {('sort_gt1' if ex=='sg' else 'trailing20'):11} {lab:9} CAGR {c*100:+5.1f}%  DD {dd*100:5.1f}%  Sharpe {s:.2f}")
bc, bdd, bs = stats(spy.reindex(cal).ffill())
print(f"  SPY buy-hold FULL      CAGR {bc*100:+5.1f}%  DD {bdd*100:5.1f}%  Sharpe {bs:.2f}")
