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
COST = 0.003          # 0.3% round-trip (commission + slippage), per side 0.15%
CAP0 = 100_000.0
MAXPOS = 8            # max concurrent positions
TARGET_VOL = 30.0     # inverse-vol sizing reference


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

# ---- generate trades (Mode A + B, gated, quality, PIT mcap) ----
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
    dret = pd.Series(close).pct_change().values
    A = cap(df) & (st == 2)
    try:
        B = SIGNALS['rsi_oversold30'][1](df).fillna(False).values & (pos >= 0.75)
    except Exception:
        B = np.zeros(n, bool)
    last_exit = -1
    for p in range(252, n - 2):
        if p <= last_exit:  # no overlapping same-ticker positions
            continue
        if not (A[p] or B[p]) or close[p] < 5:
            continue
        d = idx[p]
        if not (pd.notna(e63.asof(d)) and pd.notna(spy63.asof(d)) and e63.asof(d) > spy63.asof(d)):
            continue
        j = int(pdd.searchsorted(d, "right")) - 1
        if j < 0 or np.isnan(sh[j]) or sh[j] * close[p] < 300e6:
            continue
        xi = trail_exit(close, p, n)
        vol = np.nanstd(dret[p - 19:p + 1]) * 100 * np.sqrt(252)  # annualized-ish vol
        trades.append({'tk': tk, 'entry': d, 'exit': idx[xi], 'ep': close[p], 'xp': close[xi],
                       'mode': 'A' if A[p] else 'B', 'vol': vol if vol > 0 else TARGET_VOL,
                       'path': pd.Series(close[p:xi + 1], index=idx[p:xi + 1])})
        last_exit = xi
print(f"generated {len(trades)} trades")

# ---- portfolio simulation with daily mark-to-market ----
cal = spy.index
trades.sort(key=lambda t: t['entry'])
by_entry = {}
for t in trades:
    by_entry.setdefault(t['entry'], []).append(t)


def run_sim(cal_slice):
    cash = CAP0; open_pos = []; eq = []
    for d in cal_slice:
        # close exits
        still = []
        for op in open_pos:
            if d >= op['exit']:
                cash += op['shares'] * op['xp'] * (1 - COST)
            else:
                still.append(op)
        open_pos = still
        # open new
        for t in sorted(by_entry.get(d, []), key=lambda x: (0 if x['mode'] == 'A' else 1)):
            if len(open_pos) >= MAXPOS:
                break
            equity_now = cash + sum(o['shares'] * float(o['path'].asof(d)) for o in open_pos)
            base = equity_now / MAXPOS
            size = min(cash, base * min(2.0, TARGET_VOL / t['vol']))  # inverse-vol, capped 2x
            if size < 100:
                continue
            shares = size / t['ep']
            cash -= size * (1 + COST)
            open_pos.append({'shares': shares, 'xp': t['xp'], 'exit': t['exit'], 'path': t['path']})
        mtm = cash + sum(o['shares'] * float(o['path'].asof(d)) for o in open_pos)
        eq.append(mtm)
    return pd.Series(eq, index=cal_slice)


def stats(eq, bench):
    r = eq.pct_change().dropna()
    yrs = (eq.index[-1] - eq.index[0]).days / 365.25
    cagr = (eq.iloc[-1] / eq.iloc[0]) ** (1 / yrs) - 1
    dd = (eq / eq.cummax() - 1).min()
    sharpe = r.mean() / r.std() * np.sqrt(252) if r.std() > 0 else 0
    br = bench.reindex(eq.index).pct_change().dropna()
    bcagr = (bench.reindex(eq.index).iloc[-1] / bench.reindex(eq.index).iloc[0]) ** (1 / yrs) - 1
    bdd = (bench.reindex(eq.index) / bench.reindex(eq.index).cummax() - 1).min()
    bsharpe = br.mean() / br.std() * np.sqrt(252) if br.std() > 0 else 0
    return cagr, dd, sharpe, bcagr, bdd, bsharpe


for label, sl in [('FULL', cal), ('IN-SAMPLE 2021-24', cal[cal < '2025-01-01']), ('OUT-OF-SAMPLE 2025-26', cal[cal >= '2025-01-01'])]:
    sl = sl[sl >= trades[0]['entry']] if label == 'FULL' else sl
    if len(sl) < 30:
        continue
    eq = run_sim(sl)
    cagr, dd, sh, bcagr, bdd, bsh = stats(eq, spy)
    print(f"\n=== {label} ===")
    print(f"  STRATEGY:  final ${eq.iloc[-1]:,.0f}  CAGR {cagr*100:+.1f}%  maxDD {dd*100:.1f}%  Sharpe {sh:.2f}")
    print(f"  SPY hold:  CAGR {bcagr*100:+.1f}%  maxDD {bdd*100:.1f}%  Sharpe {bsh:.2f}")
