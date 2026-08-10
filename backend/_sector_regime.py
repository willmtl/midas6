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
import rates as R

COST, CAP0, MAXPOS, SPREAD = 0.003, 100_000.0, 8, 0.015
try:
    rdf = R.get_rates("5y"); col = min(rdf.columns, key=lambda c: rdf[c].dropna().mean())
    rff = (rdf[col] / 100.0) if rdf[col].dropna().mean() > 1 else rdf[col]
except Exception:
    rff = None

tks = build_universe()
sec = {r['ticker']: r['sector'] for r in Fundamental.objects.filter(ticker__in=tks).values('ticker', 'sector')}
t2etf = {t: GICS2ETF[s] for t, s in sec.items() if s in GICS2ETF}
mkt = load_candles(sorted(set(t2etf.values())) + ['SPY'])
spy = mkt['SPY']['Close']; spy63 = spy.pct_change(63); spy200 = spy.rolling(200).mean()
etf200 = {e: mkt[e]['Close'].rolling(200).mean() for e in mkt}
cap = lambda df: (SIGNALS['new_52low'][1](df).fillna(False).values | SIGNALS['rsi_oversold20'][1](df).fillna(False).values)

trades = []
cd = load_candles([t for t in tks if t in t2etf])
for tk, df in cd.items():
    if len(df) < 260:
        continue
    rep = reports = None
    from seq_fundamental_study import load_financial_reports as _lfr
    rep = _lfr([tk]).get(tk)
    if rep is None or not len(rep):
        continue
    r2 = rep.dropna(subset=["avail_date", "shares_outstanding"]).sort_values("avail_date")
    if not len(r2):
        continue
    pdd = pd.DatetimeIndex(pd.to_datetime(r2["avail_date"]).to_numpy()); sh = r2["shares_outstanding"].to_numpy(dtype=float)
    _prepare_indicators(df)
    close = df['Close'].values; n = len(close); idx = df.index
    st = _ad_state(df).values
    etf = t2etf[tk]; e = mkt[etf]['Close']; e63 = e.pct_change(63); e200 = etf200[etf]
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
        etf_on = pd.notna(e200.asof(d)) and e.asof(d) > e200.asof(d)
        spy_on = pd.notna(spy200.asof(d)) and spy.asof(d) > spy200.asof(d)
        trades.append({'entry': d, 'conv': 'A' if A[p] else 'B', 'ep': close[p], 'xp': close[xi], 'exit': idx[xi],
                       'etf_on': etf_on, 'spy_on': spy_on, 'path': pd.Series(close[p:xi + 1], index=idx[p:xi + 1])})
        le = xi
print(f"{len(trades)} trades")
bear = [t for t in trades if not t['spy_on']]
bear_strong = [t for t in bear if t['etf_on']]
print(f"  in SPY-bear (<200dMA): {len(bear)} trades; of those in ABSOLUTELY-strong sectors (ETF>own200): {len(bear_strong)}")
a = np.array([(t['xp'] - t['ep']) / t['ep'] * 100 for t in bear_strong])
if len(a):
    print(f"  those bear+strong-sector trades: win={ (a>0).mean()*100:.0f}% avg={a.mean():+.1f}%")

by_entry = {}
for t in trades:
    by_entry.setdefault(t['entry'], []).append(t)
cal = spy.index[spy.index >= min(t['entry'] for t in trades)]
spret = spy.pct_change()


def rff_d(d):
    if rff is None:
        return 0.045 / 252
    v = rff.reindex([d], method='ffill').iloc[0]
    return (v if pd.notna(v) else 0.045) / 252


def run(cal_slice, entry_rule):
    base = CAP0; mode = 'spy'; op = []; eq = []
    for d in cal_slice:
        spy_on = pd.notna(spy200.asof(d)) and spy.asof(d) > spy200.asof(d)
        base *= (1 + (spret.asof(d) if mode == 'spy' else rff_d(d))) if pd.notna(spret.asof(d)) else 1
        want = 'spy' if spy_on else 'cash'
        if want != mode:
            base *= (1 - COST); mode = want
        keep = []
        for o in op:
            if d >= o['exit']:
                base += o['shares'] * o['xp'] * (1 - COST)
            else:
                keep.append(o)
        op = keep
        for t in sorted(by_entry.get(d, []), key=lambda x: {'A': 0, 'B': 1}[x['conv']]):
            if len(op) >= MAXPOS:
                break
            allow = spy_on if entry_rule == 'spy' else (t['etf_on'] if entry_rule == 'sector' else True)
            if not allow:
                continue
            stk = sum(o['shares'] * float(o['path'].asof(d)) for o in op)
            size = (base + stk) / MAXPOS
            size = min(size, base)
            if size < 100:
                continue
            base -= size
            op.append({'shares': size * (1 - COST) / t['ep'], 'xp': t['xp'], 'exit': t['exit'], 'path': t['path']})
        eq.append(base + sum(o['shares'] * float(o['path'].asof(d)) for o in op))
    return pd.Series(eq, index=cal_slice)


def stats(eq):
    r = eq.pct_change().dropna(); yrs = (eq.index[-1] - eq.index[0]).days / 365.25
    return ((eq.iloc[-1] / eq.iloc[0]) ** (1 / yrs) - 1, (eq / eq.cummax() - 1).min(),
            r.mean() / r.std() * np.sqrt(252) if r.std() > 0 else 0)


print("\nPORTFOLIO (base: SPY when SPY>200dMA else cash@T-bill):")
for name, rule in [('entries paused in SPY bear', 'spy'), ('entries if SECTOR>own 200dMA', 'sector'), ('entries always (relative gate only)', 'always')]:
    for lab, sl in [('FULL', cal), ('OOS25-26', cal[cal >= '2025-01-01'])]:
        c, dd, s = stats(run(sl, rule))
        print(f"  {name:34} {lab:9} CAGR {c*100:+5.1f}%  DD {dd*100:6.1f}%  Sharpe {s:.2f}")
c, dd, s = stats(spy.reindex(cal).ffill())
print(f"  {'SPY buy-hold':34} FULL      CAGR {c*100:+5.1f}%  DD {dd*100:6.1f}%  Sharpe {s:.2f}")
