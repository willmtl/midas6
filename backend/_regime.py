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

COST, CAP0, MAXPOS, SPREAD, GROSS_CAP = 0.003, 100_000.0, 8, 0.015, 1.5

# --- risk-free short rate series (annual %, e.g. 4.5 -> 0.045) ---
try:
    rdf = R.get_rates("5y")
    col = None
    for c in rdf.columns:
        m = rdf[c].dropna().mean()
        if 'short' in str(c).lower() or 'irx' in str(c).lower() or '3m' in str(c).lower() or '13' in str(c):
            col = c; break
    if col is None:  # fall back to the lower-valued rate column (short end)
        col = min(rdf.columns, key=lambda c: rdf[c].dropna().mean())
    rff_annual = (rdf[col] / 100.0) if rdf[col].dropna().mean() > 1 else rdf[col]
    print(f"risk-free col='{col}' mean={rff_annual.mean()*100:.2f}%")
except Exception as ex:
    print("rates load failed, flat 4.5%:", ex); rff_annual = None

tks = build_universe()
sec = {r['ticker']: r['sector'] for r in Fundamental.objects.filter(ticker__in=tks).values('ticker', 'sector')}
t2etf = {t: GICS2ETF[s] for t, s in sec.items() if s in GICS2ETF}
reports = load_financial_reports(tks)
mkt = load_candles(sorted(set(t2etf.values())) + ['SPY'])
spy = mkt['SPY']['Close']; spy63 = spy.pct_change(63); spy200 = spy.rolling(200).mean()
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
        xi = _exit_sort_above(df, p, 1)
        trades.append({'entry': d, 'conv': 'A' if A[p] else 'B', 'ep': close[p], 'xp': close[xi],
                       'exit': idx[xi], 'path': pd.Series(close[p:xi + 1], index=idx[p:xi + 1])})
        le = xi
print(f"{len(trades)} trades")
by_entry = {}
for t in trades:
    by_entry.setdefault(t['entry'], []).append(t)
cal = spy.index[spy.index >= min(t['entry'] for t in trades)]
spret = spy.pct_change()


def rff_d(d):
    if rff_annual is None:
        return 0.045 / 252
    v = rff_annual.reindex([d], method='ffill').iloc[0]
    return (v if pd.notna(v) else 0.045) / 252


rate_rising = (rff_annual > rff_annual.shift(126)) if rff_annual is not None else None  # Fed hiking (6mo)


def run(cal_slice, regime='none', lev=1.0):
    base = CAP0; mode = 'spy'; debt = 0.0; op = []; eq = []
    for d in cal_slice:
        spy_on = pd.notna(spy200.asof(d)) and spy.asof(d) > spy200.asof(d)
        fed_on = (rate_rising is None) or not bool(rate_rising.asof(d))  # risk-on when NOT hiking
        if regime == 'none':
            risk_on = True
        elif regime == 'spy':
            risk_on = spy_on
        elif regime == 'fed':
            risk_on = fed_on
        elif regime == 'fed_and_spy':
            risk_on = fed_on and spy_on
        else:
            risk_on = True
        # grow base
        base *= (1 + (spret.asof(d) if mode == 'spy' else rff_d(d))) if pd.notna(spret.asof(d)) else 1
        debt *= (1 + rff_d(d) + SPREAD / 252)
        # regime switch of base
        want = 'spy' if risk_on else 'cash'
        if want != mode:
            base *= (1 - COST); mode = want
        # close exits
        keep = []
        for o in op:
            if d >= o['exit']:
                proc = o['shares'] * o['xp'] * (1 - COST)
                pay = min(debt, proc); debt -= pay; base += (proc - pay)
            else:
                keep.append(o)
        op = keep
        # open (only risk-on)
        if risk_on:
            for t in sorted(by_entry.get(d, []), key=lambda x: {'A': 0, 'B': 1}[x['conv']]):
                if len(op) >= MAXPOS:
                    break
                stk = sum(o['shares'] * float(o['path'].asof(d)) for o in op)
                E = base + stk - debt
                size = (E / MAXPOS) * (lev if t['conv'] == 'A' else 1.0)
                if stk + size > GROSS_CAP * E:
                    size = max(0, GROSS_CAP * E - stk)
                if size < 100:
                    continue
                take = min(size, base); base -= take; debt += (size - take)
                op.append({'shares': size * (1 - COST) / t['ep'], 'xp': t['xp'], 'exit': t['exit'], 'path': t['path']})
        eq.append(base + sum(o['shares'] * float(o['path'].asof(d)) for o in op) - debt)
    return pd.Series(eq, index=cal_slice)


def stats(eq):
    r = eq.pct_change().dropna(); yrs = (eq.index[-1] - eq.index[0]).days / 365.25
    return ((eq.iloc[-1] / eq.iloc[0]) ** (1 / yrs) - 1, (eq / eq.cummax() - 1).min(),
            r.mean() / r.std() * np.sqrt(252) if r.std() > 0 else 0)


print("\nPORTFOLIO (sort_gt1; cash=T-bill yield; margin=T-bill+1.5%):")
CFG = [('no regime', dict(regime='none', lev=1.0)),
       ('cash when SPY<200dMA', dict(regime='spy', lev=1.0)),
       ('cash when Fed HIKING', dict(regime='fed', lev=1.0)),
       ('cash Fed-hiking OR SPY<200', dict(regime='fed_and_spy', lev=1.0)),
       ('Fed-regime + 1.3x Mode-A', dict(regime='fed', lev=1.3))]
for name, kw in CFG:
    for lab, sl in [('FULL', cal), ('OOS25-26', cal[cal >= '2025-01-01'])]:
        c, dd, s = stats(run(sl, **kw))
        print(f"  {name:26} {lab:9} CAGR {c*100:+5.1f}%  DD {dd*100:6.1f}%  Sharpe {s:.2f}")
c, dd, s = stats(spy.reindex(cal).ffill())
print(f"  {'SPY buy-hold':26} FULL      CAGR {c*100:+5.1f}%  DD {dd*100:6.1f}%  Sharpe {s:.2f}")
