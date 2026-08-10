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

# RSI-CROSS family (reversal-confirmed) vs the best LEVEL trigger baseline.
CANDIDATES = [
    ("rsi_os20_omega_lt1_sort_neg", "LEVEL baseline (RSI<20+conf)"),
    ("rsi_x_above_sma", "cross: RSI x above SMA"),
    ("rsi_x_sma_below50", "cross: RSI x SMA (from <50)"),
    ("rsi_sup10_x", "cross: suppressed 10d then x"),
    ("dd30_rsi_reversal", "cross: 30% off ATH + RSI x"),
    ("seq_rsi20_rsi_10d", "seq: RSI<20 then x (10d)"),
    ("seq_rsi25_rsi_10d", "seq: RSI<25 then x (10d)"),
    ("seq_rsi20_ad_rising_rsi", "seq: RSI<20 + A/D-rising + x"),
]
CANDIDATES = [(k, lbl) for k, lbl in CANDIDATES if k in SIGNALS]

tks = build_universe()
sec = {r['ticker']: r['sector'] for r in Fundamental.objects.filter(ticker__in=tks).values('ticker', 'sector')}
t2etf = {t: GICS2ETF[s] for t, s in sec.items() if s in GICS2ETF}
funds = load_fundamentals(tks)
mkt = load_candles(sorted(set(t2etf.values())) + ['SPY'])
r63 = {e: mkt[e]['Close'].pct_change(63) for e in mkt}
spy = r63['SPY']
sg = lambda df, i: _exit_sort_above(df, i, 1)

out = {k: [] for k, _ in CANDIDATES}
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
    for k, _ in CANDIDATES:
        try:
            s = SIGNALS[k][1](df).fillna(False).values
        except Exception:
            continue
        for p in range(252, n - 2):
            if not (s[p] and st[p] == 2 and gate[p]) or close[p] < 5:
                continue
            xi = sg(df, p)
            if xi is None or xi <= p or xi >= n:
                continue
            out[k].append((close[xi] - close[p]) / close[p] * 100)

print("=== Mode A: RSI-CROSS triggers vs LEVEL (all + accum-div, gated, quality, sort_gt1) ===")
print(f'{"trigger":34}{"n":>6}{"win%":>7}{"median":>8}{"avg":>8}{"worst":>8}{"<-33%":>7}')
rows = []
for k, lbl in CANDIDATES:
    a = np.array(out[k])
    if len(a) < 20:
        print(f'{lbl:34}{len(a):6d}  (too few)'); continue
    rows.append((lbl, len(a), (a > 0).mean() * 100, np.median(a), a.mean(), a.min(), (a < -33).mean() * 100))
for lbl, nn, win, med, avg, worst, d33 in sorted(rows, key=lambda r: -r[2]):
    print(f'{lbl:34}{nn:6d}{win:7.1f}{med:+8.1f}{avg:+8.1f}{worst:+8.1f}{d33:7.1f}')
