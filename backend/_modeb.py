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

A_recs, B_recs = [], []
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
    rsi = df['_rsi'].values if '_rsi' in df.columns else ta.momentum.rsi(df['Close'], 10).values
    e = mkt[t2etf[tk]]['Close']; e63 = e.pct_change(63)
    hi = pd.Series(close).rolling(252).max().values; lo = pd.Series(close).rolling(252).min().values
    pos = (close - lo) / np.where(hi - lo == 0, np.nan, hi - lo)
    s52 = SIGNALS['new_52low'][1](df).fillna(False).values
    o20 = SIGNALS['rsi_oversold20'][1](df).fillna(False).values

    def gq(p):
        d = idx[p]
        if not (pd.notna(e63.asof(d)) and pd.notna(spy63.asof(d)) and e63.asof(d) > spy63.asof(d)):
            return False
        j = int(pdd.searchsorted(d, "right")) - 1
        return j >= 0 and not np.isnan(sh[j]) and sh[j] * close[p] >= 300e6 and close[p] >= 5

    def leg(p):
        xi = _exit_sort_above(df, p, 1); xi = min(max(xi, p + 1), n - 1)
        return {'tk': tk, 'p': p, 'entry': idx[p], 'ep': float(close[p]), 'xp': float(close[xi]),
                'xd': idx[xi], 'path': pd.Series(close[p:xi + 1], index=idx[p:xi + 1])}
    for p in range(252, n - 2):
        aA = (s52[p] or o20[p]) and (not pd.isna(st[p]) and st[p] == 2)
        if aA and gq(p):
            A_recs.append({**leg(p), 'conv': 'A'})
        if pos[p] >= 0.75 and not np.isnan(rsi[p]) and rsi[p] < 45 and gq(p):
            B_recs.append({**leg(p), 'conv': 'B', 'rsi': float(rsi[p])})
print(f"Mode A entries (fixed): {len(A_recs)} | Mode B candidate bars (RSI<45 near-high): {len(B_recs)}")


def cooldown(recs):
    out = []; last = {}
    for r in sorted(recs, key=lambda z: (z['tk'], z['p'])):
        if r['p'] - last.get(r['tk'], -99) >= 10:
            last[r['tk']] = r['p']; out.append(r)
    return out


A_ent = cooldown(A_recs)
cal = spy.index[spy.index >= min(r['entry'] for r in A_recs + B_recs)]
oos = cal[cal >= '2025-01-01']


def sim(items, cal_slice):
    be = {}
    for en in items:
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


print(f"\n{'Mode B RSI<X':14}{'B n':>6}{'B win%':>7}{'B avg':>7}  |  FULL STRATEGY (A+B):{'':2}{'CAGR':>7}{'DD':>7}{'Shrp':>6}{'|OOS':>7}{'Shrp':>6}")
c, dd, s = stats(sim(A_ent, cal)); co, _, so = stats(sim(A_ent, oos))
print(f"{'(Mode A only)':14}{'-':>6}{'-':>7}{'-':>7}  |{'':23}{c:+7.1f}{dd:7.1f}{s:6.2f}{co:+7.1f}{so:6.2f}")
for X in [30, 35, 40, 45]:
    Bsub = cooldown([r for r in B_recs if r['rsi'] < X])
    br = np.array([(r['xp'] - r['ep']) / r['ep'] * 100 for r in Bsub])
    full = A_ent + Bsub
    c, dd, s = stats(sim(full, cal)); co, _, so = stats(sim(full, oos))
    print(f"{('RSI<'+str(X)):14}{len(Bsub):6d}{(br>0).mean()*100:7.0f}{br.mean():+7.1f}  |{'':23}{c:+7.1f}{dd:7.1f}{s:6.2f}{co:+7.1f}{so:6.2f}")
bc, bdd, bs = stats(spy.reindex(cal).ffill())
print(f"{'SPY':14}{'':20}  |{'':23}{bc:+7.1f}{bdd:7.1f}{bs:6.2f}")
