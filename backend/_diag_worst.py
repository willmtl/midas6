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

TRIGS = ["rsi_x_above_sma", "seq_rsi20_rsi_10d"]
tks = build_universe()
sec = {r['ticker']: r['sector'] for r in Fundamental.objects.filter(ticker__in=tks).values('ticker', 'sector')}
t2etf = {t: GICS2ETF[s] for t, s in sec.items() if s in GICS2ETF}
funds = load_fundamentals(tks)
mkt = load_candles(sorted(set(t2etf.values())) + ['SPY'])
r63 = {e: mkt[e]['Close'].pct_change(63) for e in mkt}
spy = r63['SPY']
sg = lambda df, i: _exit_sort_above(df, i, 1)

rows = {t: [] for t in TRIGS}
cd = load_candles([t for t in tks if t in t2etf])
for tk, df in cd.items():
    if len(df) < 260:
        continue
    f = funds.get(tk, {})
    cur_mcap = f.get('market_cap')
    if is_low_quality(cur_mcap, None, f.get('profit_margin')):
        continue
    _prepare_indicators(df)
    close = df['Close'].values; n = len(close); idx = df.index
    cur_px = close[-1]
    st = _ad_state(df).values
    e = r63.get(t2etf[tk])
    gate = np.zeros(n, bool)
    for p in range(252, n - 2):
        d = idx[p]; rv = e.asof(d); sv = spy.asof(d)
        gate[p] = pd.notna(rv) and pd.notna(sv) and rv > sv
    for t in TRIGS:
        try:
            s = SIGNALS[t][1](df).fillna(False).values
        except Exception:
            continue
        for p in range(252, n - 2):
            if not (s[p] and st[p] == 2 and gate[p]) or close[p] < 5:
                continue
            xi = sg(df, p)
            if xi is None or xi <= p or xi >= n:
                continue
            ret = (close[xi] - close[p]) / close[p] * 100
            # PIT market cap estimate: scale current mcap by price ratio (shares ~ const)
            pit_mcap = cur_mcap * close[p] / cur_px if (cur_mcap and cur_px) else None
            rows[t].append((tk, str(idx[p].date()), round(close[p], 2), round(ret, 1),
                            cur_mcap, pit_mcap))

for t in TRIGS:
    D = pd.DataFrame(rows[t], columns=['tk', 'date', 'entry_px', 'ret', 'cur_mcap', 'pit_mcap'])
    dis = D[D['ret'] < -33]
    print(f"\n=== {t}: {len(D)} trades, {len(dis)} disasters (<-33%) ===")
    if len(dis):
        pit_micro = (dis['pit_mcap'] < 300e6).mean() * 100
        cur_micro = (dis['cur_mcap'] < 300e6).mean() * 100
        print(f"  disasters that were micro-cap (<300M) AT ENTRY (PIT est): {pit_micro:.0f}%   ...by CURRENT mcap: {cur_micro:.0f}%")
        print(f"  median PIT mcap of disasters: ${np.nanmedian(dis['pit_mcap'])/1e6:.0f}M   vs current ${np.nanmedian(dis['cur_mcap'])/1e6:.0f}M")
        print("  worst 12 trades (ticker | date | entry$ | ret% | curMcap | PIT-mcap-est):")
        for _, r in dis.sort_values('ret').head(12).iterrows():
            cm = f"${r['cur_mcap']/1e9:.1f}B" if r['cur_mcap'] else "?"
            pm = f"${r['pit_mcap']/1e6:.0f}M" if r['pit_mcap'] else "?"
            print(f"    {r['tk']:10} {r['date']}  ${r['entry_px']:<8} {r['ret']:+7.1f}%  cur={cm:8} pit={pm}")
