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

COST, CAP0, MAXPOS = 0.003, 100_000.0, 8


def fred(series):
    try:
        df = pd.read_csv(f"https://fred.stlouisfed.org/graph/fredgraph.csv?id={series}", na_values=".")
        df.columns = ['date', 'v']
        df['date'] = pd.to_datetime(df['date'])
        return df.set_index('date')['v'].astype(float).dropna()
    except Exception as ex:
        print(f"  FRED {series} failed: {ex}"); return None


print("Fetching rate-expectations data...")
dgs2 = fred("DGS2"); dgs10 = fred("DGS10"); dgs3mo = fred("DGS3MO")
zq = None
try:
    import yfinance as yf
    z = yf.download("ZQ=F", period="6y", progress=False)
    zq = (100 - z['Close']).dropna()
    if isinstance(zq, pd.DataFrame):
        zq = zq.iloc[:, 0]
    print(f"  ZQ fed-funds-futures implied rate: {len(zq)} days, latest {zq.iloc[-1]:.2f}%")
except Exception as ex:
    print(f"  ZQ failed: {ex}")
for nm, s in [('DGS2', dgs2), ('DGS10', dgs10), ('DGS3MO', dgs3mo)]:
    if s is not None:
        print(f"  {nm}: {len(s)} obs, latest {s.iloc[-1]:.2f}%")

tks = build_universe()
sec = {r['ticker']: r['sector'] for r in Fundamental.objects.filter(ticker__in=tks).values('ticker', 'sector')}
t2etf = {t: GICS2ETF[s] for t, s in sec.items() if s in GICS2ETF}
reports = load_financial_reports(tks)
mkt = load_candles(sorted(set(t2etf.values())) + ['SPY'])
spy = mkt['SPY']['Close']; spy63 = spy.pct_change(63); spy200 = spy.rolling(200).mean(); spret = spy.pct_change()
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


def rff_d(d):
    if dgs3mo is None:
        return 0.045 / 252
    v = dgs3mo.reindex([d], method='ffill').iloc[0]
    return (v if pd.notna(v) else 4.5) / 100 / 252


# regime signals: risk_on(d) bool. Easing expected / not-tightening = risk-on.
def falling(series, d, lb=63):
    if series is None:
        return True
    now = series.reindex([d], method='ffill').iloc[0]
    past = series.reindex([d - pd.Timedelta(days=lb)], method='ffill').iloc[0]
    return pd.notna(now) and pd.notna(past) and now <= past


def curve_ok(d):
    if dgs10 is None or dgs3mo is None:
        return True
    c_now = dgs10.reindex([d], method='ffill').iloc[0] - dgs3mo.reindex([d], method='ffill').iloc[0]
    c_past = dgs10.reindex([d - pd.Timedelta(days=63)], method='ffill').iloc[0] - dgs3mo.reindex([d - pd.Timedelta(days=63)], method='ffill').iloc[0]
    return pd.notna(c_now) and pd.notna(c_past) and c_now >= c_past  # steepening = improving


REG = {
    'no regime': lambda d: True,
    'SPY<200dMA (baseline)': lambda d: pd.notna(spy200.asof(d)) and spy.asof(d) > spy200.asof(d),
    '1) FedFunds-fut easing': lambda d: falling(zq, d),
    '2) 2Y yield easing': lambda d: falling(dgs2, d),
    '3) yield-curve steepening': curve_ok,
}


def run(cal_slice, regime):
    base = CAP0; mode = 'spy'; op = []; eq = []
    for d in cal_slice:
        on = regime(d)
        base *= (1 + (spret.asof(d) if mode == 'spy' else rff_d(d))) if pd.notna(spret.asof(d)) else 1
        want = 'spy' if on else 'cash'
        if want != mode:
            base *= (1 - COST); mode = want
        keep = []
        for o in op:
            if d >= o['exit']:
                base += o['shares'] * o['xp'] * (1 - COST)
            else:
                keep.append(o)
        op = keep
        if on:
            for t in sorted(by_entry.get(d, []), key=lambda x: {'A': 0, 'B': 1}[x['conv']]):
                if len(op) >= MAXPOS:
                    break
                stk = sum(o['shares'] * float(o['path'].asof(d)) for o in op)
                size = min(base, (base + stk) / MAXPOS)
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


print("\nRATE-EXPECTATIONS REGIMES (base SPY when risk-on else cash@T-bill; sort_gt1):")
for name, fn in REG.items():
    for lab, sl in [('FULL', cal), ('OOS25-26', cal[cal >= '2025-01-01'])]:
        c, dd, s = stats(run(sl, fn))
        print(f"  {name:26} {lab:9} CAGR {c*100:+5.1f}%  DD {dd*100:6.1f}%  Sharpe {s:.2f}")
c, dd, s = stats(spy.reindex(cal).ffill())
print(f"  {'SPY buy-hold':26} FULL      CAGR {c*100:+5.1f}%  DD {dd*100:6.1f}%  Sharpe {s:.2f}")
