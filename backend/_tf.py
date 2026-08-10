import os, django
os.environ.setdefault("DJANGO_SETTINGS_MODULE", "rotation.settings")
django.setup()

import numpy as np, pandas as pd, ta
from seq_fundamental_study import build_universe, load_candles, load_financial_reports
from studies import SIGNALS, _exit_sort_above
from all_on_all_study import _prepare_indicators
from pit_fundamentals import _ad_state
from core.models import Fundamental
from api.tasks import GICS2ETF

COST, CAP0, MAXPOS = 0.003, 100_000.0, 8
tks = build_universe()
sec = {r['ticker']: r['sector'] for r in Fundamental.objects.filter(ticker__in=tks).values('ticker', 'sector')}
t2etf = {t: GICS2ETF[s] for t, s in sec.items() if s in GICS2ETF}
reports = load_financial_reports(tks)
mkt = load_candles(sorted(set(t2etf.values())) + ['SPY'])
spy = mkt['SPY']['Close']; spy63 = spy.pct_change(63); spy200 = spy.rolling(200).mean(); spret = spy.pct_change()

E = {'daily': [], 'weekly': [], 'mix': []}
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
    cap = (SIGNALS['new_52low'][1](df).fillna(False).values | SIGNALS['rsi_oversold20'][1](df).fillna(False).values)
    so30 = SIGNALS['rsi_oversold30'][1](df).fillna(False).values
    # weekly bars + weekly indicators
    wdf = df.resample('W-FRI').agg({'Open': 'first', 'High': 'max', 'Low': 'min', 'Close': 'last', 'Volume': 'sum'}).dropna()
    wrsi = ta.momentum.rsi(wdf['Close'], 10)
    wlow = wdf['Close'] == wdf['Close'].rolling(52).min()
    wpos = (wdf['Close'] - wdf['Close'].rolling(52).min()) / (wdf['Close'].rolling(52).max() - wdf['Close'].rolling(52).min())
    wst = pd.Series(_ad_state(wdf).values, index=wdf.index)
    wA = ((wrsi < 20) | wlow) & (wst == 2)                 # weekly Mode A
    wB = (wrsi < 30) & (wpos >= 0.75)                       # weekly Mode B
    wrsi_d = wrsi.reindex(idx, method='ffill')              # weekly RSI as-of each daily bar

    def gq(p):
        d = idx[p]
        if not (pd.notna(e63.asof(d)) and pd.notna(spy63.asof(d)) and e63.asof(d) > spy63.asof(d)):
            return False
        j = int(pdd.searchsorted(d, "right")) - 1
        return j >= 0 and not np.isnan(sh[j]) and sh[j] * close[p] >= 300e6 and close[p] >= 5

    def rec(lst, p, conv, last):
        if p - last[0] < 10:
            return
        xi = _exit_sort_above(df, p, 1); xi = min(max(xi, p + 1), n - 1)
        lst.append({'entry': idx[p], 'conv': conv, 'ep': float(close[p]),
                    'xp': float(close[xi]), 'xd': idx[xi], 'path': pd.Series(close[p:xi + 1], index=idx[p:xi + 1])})
        last[0] = p

    # DAILY
    last = [-99]
    for p in range(252, n - 2):
        aA = (cap[p] and st[p] == 2); b = (so30[p] and pos[p] >= 0.75)
        if (aA or b) and close[p] >= 5 and gq(p):
            rec(E['daily'], p, 'A' if aA else 'B', last)
    # WEEKLY (map weekly signal to the daily bar at week-end)
    last = [-99]
    for wd in wdf.index[52:]:
        if not (bool(wA.get(wd, False)) or bool(wB.get(wd, False))):
            continue
        loc = idx.searchsorted(wd)
        if loc >= n or loc < 252:
            continue
        p = loc
        if gq(p):
            rec(E['weekly'], p, 'A' if bool(wA.get(wd, False)) else 'B', last)
    # MIX: weekly oversold context + daily trigger
    last = [-99]
    for p in range(252, n - 2):
        d = idx[p]; wr = wrsi_d.iloc[p]
        aA = (cap[p] and st[p] == 2) and pd.notna(wr) and wr < 40      # daily trigger inside weekly-weak context
        b = (so30[p] and pos[p] >= 0.75) and pd.notna(wr) and wr < 55
        if (aA or b) and close[p] >= 5 and gq(p):
            rec(E['mix'], p, 'A' if aA else 'B', last)

for k in E:
    print(f"{k}: {len(E[k])} entries")
cal = spy.index[spy.index >= min(min(x['entry'] for x in E[k]) for k in E if E[k])]


def sim(lst, cal_slice):
    be = {}
    for en in lst:
        be.setdefault(en['entry'], []).append(en)
    base = CAP0; mode = 'spy'; op = []; eq = []
    for d in cal_slice:
        on = pd.notna(spy200.asof(d)) and spy.asof(d) > spy200.asof(d)
        base *= (1 + (spret.asof(d) if mode == 'spy' else 0.04 / 252)) if pd.notna(spret.asof(d)) else 1
        want = 'spy' if on else 'cash'
        if want != mode:
            base *= (1 - COST); mode = want
        keep = []
        for o in op:
            if d >= o['xd']:
                base += o['sh'] * o['xp'] * (1 - COST)
            else:
                keep.append(o)
        op = keep
        if on:
            for en in sorted(be.get(d, []), key=lambda x: 0 if x['conv'] == 'A' else 1):
                if len(op) >= MAXPOS:
                    break
                stk = sum(o['sh'] * float(o['path'].asof(d)) for o in op)
                size = min(base, (base + stk) / MAXPOS)
                if size < 100:
                    continue
                base -= size
                op.append({'sh': size * (1 - COST) / en['ep'], 'xp': en['xp'], 'xd': en['xd'], 'path': en['path']})
        eq.append(base + sum(o['sh'] * float(o['path'].asof(d)) for o in op))
    return pd.Series(eq, index=cal_slice)


def stats(eq):
    r = eq.pct_change().dropna(); yrs = (eq.index[-1] - eq.index[0]).days / 365.25
    return ((eq.iloc[-1] / eq.iloc[0]) ** (1 / yrs) - 1) * 100, (eq / eq.cummax() - 1).min() * 100, (r.mean() / r.std() * np.sqrt(252) if r.std() > 0 else 0)


oos = cal[cal >= '2025-01-01']
print(f"\nENTRY TIMEFRAME (exit=sort_gt1, SPY overlay + SPY<200dMA):")
print(f'{"entry":10}{"n":>6}{"win%":>6}{"CAGR":>7}{"DD":>7}{"Shrp":>6}{"|OOS CAGR":>10}{"Shrp":>6}')
for k in ['daily', 'weekly', 'mix']:
    if not E[k]:
        print(f'{k:10} none'); continue
    rets = np.array([(x['xp'] - x['ep']) / x['ep'] * 100 for x in E[k]])
    c, dd, s = stats(sim(E[k], cal)); co, ddo, so = stats(sim(E[k], oos))
    print(f'{k:10}{len(E[k]):6d}{(rets>0).mean()*100:6.0f}{c:+7.1f}{dd:7.1f}{s:6.2f}{co:+10.1f}{so:6.2f}')
bc, bdd, bs = stats(spy.reindex(cal).ffill())
print(f'{"SPY":10}{"":18}{bc:+7.1f}{bdd:7.1f}{bs:6.2f}')
