import os, django
os.environ.setdefault("DJANGO_SETTINGS_MODULE", "rotation.settings")
django.setup()

import numpy as np, pandas as pd
from seq_fundamental_study import build_universe, load_candles, load_fundamentals
from studies import SIGNALS, _exit_sort_above
from all_on_all_study import _prepare_indicators
from pit_fundamentals import _ad_state
from core.models import Fundamental
from api.tasks import GICS2ETF, is_low_quality

TRIGGERS = ["rsi_os20_omega_lt1_sort_neg", "new_52low"]
tks = build_universe()
sec = {r['ticker']: r['sector'] for r in Fundamental.objects.filter(ticker__in=tks).values('ticker', 'sector')}
t2etf = {t: GICS2ETF[s] for t, s in sec.items() if s in GICS2ETF}
funds = load_fundamentals(tks)
mkt = load_candles(sorted(set(t2etf.values())) + ['SPY'])
r63 = {e: mkt[e]['Close'].pct_change(63) for e in mkt}
spy = r63['SPY']
sg = lambda df, i: _exit_sort_above(df, i, 1)

# per trigger: list of (state_code, ret)
data = {t: [] for t in TRIGGERS}
cd = load_candles([t for t in tks if t in t2etf])
for tk, df in cd.items():
    if len(df) < 260:
        continue
    f = funds.get(tk, {})
    if is_low_quality(f.get('market_cap'), None, f.get('profit_margin')):
        continue
    _prepare_indicators(df)
    close = df['Close'].values; n = len(close); idx = df.index
    st = _ad_state(df).values
    e = r63.get(t2etf[tk])
    gate = np.zeros(n, bool)
    for p in range(252, n - 2):
        d = idx[p]; rv = e.asof(d); sv = spy.asof(d)
        gate[p] = pd.notna(rv) and pd.notna(sv) and rv > sv
    for t in TRIGGERS:
        try:
            s = SIGNALS[t][1](df).fillna(False).values
        except Exception:
            continue
        for p in range(252, n - 2):
            if not (s[p] and gate[p]) or close[p] < 5 or np.isnan(st[p]):
                continue
            xi = sg(df, p)
            if xi is None or xi <= p or xi >= n:
                continue
            data[t].append((int(st[p]), (close[xi] - close[p]) / close[p] * 100))

STATES = [(2, 'accum divergence'), (1, 'accum trend-up'), (0, 'neutral'), (-1, 'distribution')]
for t in TRIGGERS:
    D = pd.DataFrame(data[t], columns=['state', 'ret'])
    print(f"\n=== {t}  (gated + quality-filtered + sort_gt1) — split by A/D state at entry ===")
    print(f'{"A/D state":22}{"n":>6}{"win%":>7}{"median":>8}{"avg":>8}{"worst":>8}')

    def line(name, mask):
        a = D['ret'][mask].values
        if len(a) == 0:
            print(f'{name:22}{"—":>6}'); return
        print(f'{name:22}{len(a):6d}{(a>0).mean()*100:7.1f}{np.median(a):+8.1f}{a.mean():+8.1f}{a.min():+8.1f}')
    for code, nm in STATES:
        line(nm, D['state'] == code)
    line('ANY accumulation(1/2)', D['state'].isin([1, 2]))
    line('ALL states (no A/D)', D['state'].notna())
