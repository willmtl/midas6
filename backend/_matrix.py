import os, django
os.environ.setdefault("DJANGO_SETTINGS_MODULE", "rotation.settings")
django.setup()

import numpy as np, pandas as pd
from seq_fundamental_study import build_universe, load_candles, load_financial_reports
from studies import SIGNALS, EXITS
from all_on_all_study import _prepare_indicators
from pit_fundamentals import _ad_state
from core.models import Fundamental
from api.tasks import GICS2ETF

COST, CAP0, MAXPOS = 0.003, 100_000.0, 8
EXIT_KEYS = ['sort_gt1', 'rsi_80', 'rsi_70_x_dn', 'trail_20', '6m']  # sort_gt1 kept as reference only
TRIGS = {'level (52low+RSI<20)': lambda al, ac: al,
         '+RSI-cross': lambda al, ac: al or ac,
         'RSI-cross only': lambda al, ac: ac}

tks = build_universe()
sec = {r['ticker']: r['sector'] for r in Fundamental.objects.filter(ticker__in=tks).values('ticker', 'sector')}
t2etf = {t: GICS2ETF[s] for t, s in sec.items() if s in GICS2ETF}
reports = load_financial_reports(tks)
mkt = load_candles(sorted(set(t2etf.values())) + ['SPY'])
spy = mkt['SPY']['Close']; spy63 = spy.pct_change(63); spy200 = spy.rolling(200).mean(); spret = spy.pct_change()

entries = []  # each: {entry, a_level, a_cross, b, exits:{key:(xp,exit_date,path)}}
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
    xcross = SIGNALS['seq_rsi20_rsi_10d'][1](df).fillna(False).values
    so30 = SIGNALS['rsi_oversold30'][1](df).fillna(False).values
    last = -99
    for p in range(252, n - 2):
        a_level = bool(cap[p] and st[p] == 2)
        a_cross = bool(xcross[p] and st[p] == 2)
        b = bool(so30[p] and pos[p] >= 0.75)
        if not (a_level or a_cross or b) or close[p] < 5 or p + 2 >= n or p - last < 10:
            continue
        d = idx[p]
        if not (pd.notna(e63.asof(d)) and pd.notna(spy63.asof(d)) and e63.asof(d) > spy63.asof(d)):
            continue
        j = int(pdd.searchsorted(d, "right")) - 1
        if j < 0 or np.isnan(sh[j]) or sh[j] * close[p] < 300e6:
            continue
        last = p
        ex = {}
        for k in EXIT_KEYS:
            try:
                xi = EXITS[k][1](df, p)
            except Exception:
                xi = min(p + 90, n - 1)
            if xi is None or xi <= p:
                xi = min(p + 1, n - 1)
            ex[k] = (float(close[xi]), idx[xi], pd.Series(close[p:xi + 1], index=idx[p:xi + 1]))
        entries.append({'entry': d, 'a_level': a_level, 'a_cross': a_cross, 'b': b, 'ep': float(close[p]), 'exits': ex})
print(f"{len(entries)} candidate entries collected")
cal = spy.index[spy.index >= min(e['entry'] for e in entries)]


def sim(trig_fn, exit_key, cal_slice):
    be = {}
    for en in entries:
        take_a = trig_fn(en['a_level'], en['a_cross'])
        if take_a or en['b']:
            be.setdefault(en['entry'], []).append((en, 'A' if take_a else 'B'))
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
            for en, conv in sorted(be.get(d, []), key=lambda x: 0 if x[1] == 'A' else 1):
                if len(op) >= MAXPOS:
                    break
                xp, xd, path = en['exits'][exit_key]
                stk = sum(o['sh'] * float(o['path'].asof(d)) for o in op)
                size = min(base, (base + stk) / MAXPOS)
                if size < 100:
                    continue
                base -= size
                op.append({'sh': size * (1 - COST) / en['ep'], 'xp': xp, 'xd': xd, 'path': path})
        eq.append(base + sum(o['sh'] * float(o['path'].asof(d)) for o in op))
    return pd.Series(eq, index=cal_slice)


def stats(eq):
    r = eq.pct_change().dropna(); yrs = (eq.index[-1] - eq.index[0]).days / 365.25
    return ((eq.iloc[-1] / eq.iloc[0]) ** (1 / yrs) - 1) * 100, (eq / eq.cummax() - 1).min() * 100, (r.mean() / r.std() * np.sqrt(252) if r.std() > 0 else 0)


oos = cal[cal >= '2025-01-01']
print(f"\nMATRIX: trigger x exit (SPY overlay + SPY<200dMA cash). FULL / OOS.")
print(f'{"trigger":22}{"exit":12}{"CAGR":>7}{"DD":>7}{"Shrp":>6}{"|OOS CAGR":>10}{"Shrp":>6}')
for tname, tfn in TRIGS.items():
    for ek in EXIT_KEYS:
        c, dd, s = stats(sim(tfn, ek, cal)); co, ddo, so = stats(sim(tfn, ek, oos))
        tag = ' (ref)' if ek == 'sort_gt1' else ''
        print(f'{tname:22}{ek+tag:12}{c:+7.1f}{dd:7.1f}{s:6.2f}{co:+10.1f}{so:6.2f}')
bc, bdd, bs = stats(spy.reindex(cal).ffill())
print(f'{"SPY buy-hold":22}{"":12}{bc:+7.1f}{bdd:7.1f}{bs:6.2f}')
