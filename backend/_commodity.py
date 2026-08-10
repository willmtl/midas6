import os, django
os.environ.setdefault("DJANGO_SETTINGS_MODULE", "rotation.settings")
django.setup()

import numpy as np, pandas as pd
from seq_fundamental_study import load_candles, load_financial_reports
from studies import SIGNALS, _exit_sort_above
from all_on_all_study import _prepare_indicators
from pit_fundamentals import _ad_state

# theme -> (trend anchor, [proxy stocks]).  anchor 'BASKET' = avg of proxies (no clean ETF).
THEMES = {
    'Gold': ('GLD', ['NEM']),
    'Silver': ('SLV', ['CDE']),
    'Copper': ('COPX', ['FCX', 'SCCO']),
    'Uranium': ('URA', ['UEC', 'DNN', 'UUUU']),
    'Oil/Energy': ('XLE', ['XOM', 'CVX', 'COP', 'OXY', 'DVN', 'EOG', 'FANG', 'MPC', 'SLB']),
    'NatGas': ('UNG', ['LNG']),
    'Steel': ('SLX', ['NUE', 'STLD', 'CLF', 'MT', 'RS']),
    'Aluminum': ('BASKET', ['AA']),
    'Lithium': ('LIT', ['ALB', 'SQM', 'LAC', 'SGML']),
    'Agriculture': ('DBA', ['ADM', 'BG', 'MOS', 'CF', 'CTVA', 'DE']),
    'RareEarth': ('REMX', ['MP']),
    'Lumber/Timber': ('BASKET', ['WY', 'RYN', 'LPX']),
}
anchors = [a for a, _ in THEMES.values() if a != 'BASKET']
proxies = sorted({p for _, ps in THEMES.values() for p in ps})
cd = load_candles(anchors + proxies + ['SPY'])
spy = cd['SPY']['Close']; spy63 = spy.pct_change(63)


def trend_series(theme):
    anchor, ps = THEMES[theme]
    if anchor == 'BASKET':
        cols = [cd[p]['Close'] for p in ps if p in cd]
        s = pd.concat(cols, axis=1).mean(axis=1) if cols else None
    else:
        s = cd[anchor]['Close'] if anchor in cd else None
    return s


rows = []
etf_bench = {}
for theme, (anchor, ps) in THEMES.items():
    ts = trend_series(theme)
    if ts is None:
        continue
    tr200 = ts.rolling(200).mean(); tr63 = ts.pct_change(63)
    # ETF trend-follow benchmark: hold anchor while >200dMA
    if anchor != 'BASKET' and anchor in cd:
        r = cd[anchor]['Close'].pct_change()
        onmask = (ts > tr200).reindex(r.index).fillna(False)
        etf_bench[theme] = float((1 + r[onmask].fillna(0)).prod() ** (252 / max(onmask.sum(), 1)) - 1) * 100
    for p in ps:
        if p not in cd:
            continue
        df = cd[p]
        if len(df) < 260:
            continue
        _prepare_indicators(df)
        close = df['Close'].values; nn = len(close); idx = df.index
        st = _ad_state(df).values
        hi = pd.Series(close).rolling(252).max().values; lo = pd.Series(close).rolling(252).min().values
        pos = (close - lo) / np.where(hi - lo == 0, np.nan, hi - lo)
        s52 = SIGNALS['new_52low'][1](df).fillna(False).values
        o20 = SIGNALS['rsi_oversold20'][1](df).fillna(False).values
        o30 = SIGNALS['rsi_oversold30'][1](df).fillna(False).values
        last = -99
        for q in range(252, nn - 2):
            d = idx[q]
            comm_on = (pd.notna(tr200.asof(d)) and ts.asof(d) > tr200.asof(d)) or (pd.notna(tr63.asof(d)) and pd.notna(spy63.asof(d)) and tr63.asof(d) > spy63.asof(d))
            if not comm_on:
                continue
            aA = (s52[q] or o20[q]) and (not pd.isna(st[q]) and st[q] == 2)
            aB = o30[q] and pos[q] >= 0.75
            if (aA or aB) and close[q] >= 5 and q - last >= 10:
                last = q
                xi = _exit_sort_above(df, q, 1); xi = min(max(xi, q + 1), nn - 1)
                rows.append({'theme': theme, 'tk': p, 'mode': 'A' if aA else 'B',
                             'ret': (close[xi] - close[q]) / close[q] * 100})
D = pd.DataFrame(rows)
print(f"{len(D)} proxy entries gated by commodity trend\n")
print(f'{"theme":15}{"n":>5}{"win%":>6}{"avg":>7}   ETF-trend-follow CAGR')
for theme in THEMES:
    sub = D[D['theme'] == theme]['ret'].values if len(D) else []
    b = etf_bench.get(theme)
    if len(sub) >= 3:
        print(f'{theme:15}{len(sub):5d}{(np.array(sub)>0).mean()*100:6.0f}{np.array(sub).mean():+7.1f}   {("%.1f%%" % b) if b is not None else "(basket)"}')
    else:
        print(f'{theme:15}{len(sub):5d}  (few)   {("%.1f%%" % b) if b is not None else "(basket)"}')
if len(D):
    a = D['ret'].values
    print(f'\nALL commodity-proxy entries: n={len(a)}  win={(a>0).mean()*100:.0f}%  avg={a.mean():+.1f}%  median={np.median(a):+.1f}%')
