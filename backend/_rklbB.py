import os, django
os.environ.setdefault("DJANGO_SETTINGS_MODULE", "rotation.settings")
django.setup()

import numpy as np, pandas as pd, ta
from seq_fundamental_study import load_candles
from all_on_all_study import _prepare_indicators
from core.models import Fundamental
from api.tasks import GICS2ETF

TK = "RKLB"
f = Fundamental.objects.filter(ticker=TK).values("sector").first()
etf = GICS2ETF.get(f["sector"])
cd = load_candles([TK, etf, "SPY"])
df = cd[TK]; spy = cd["SPY"]["Close"]; spy63 = spy.pct_change(63)
e = cd[etf]["Close"]; e63 = e.pct_change(63); e200 = e.rolling(200).mean()
_prepare_indicators(df)
close = df["Close"].values; n = len(close); idx = df.index
rsi = df["_rsi"].values if "_rsi" in df.columns else ta.momentum.rsi(df["Close"], 10).values
hi = pd.Series(close).rolling(252).max().values; lo = pd.Series(close).rolling(252).min().values
pos = (close - lo) / np.where(hi - lo == 0, np.nan, hi - lo)

# Mode B = near 52wk high (pos>=0.75). What RSI do RKLB's pullbacks reach while near the high?
nearhigh = pos >= 0.75
print(f"RKLB: {int(np.nansum(nearhigh[252:]))} bars near 52wk-high (pos>=0.75) out of {n-252}")
for thr in [30, 35, 40, 45, 50]:
    cnt = int(np.nansum((rsi[252:] < thr) & nearhigh[252:]))
    print(f"  bars with RSI<{thr} AND near-high: {cnt}")

print("\nLocal RSI-dip lows while near the high (candidate Mode-B entries):")
last = -99
for p in range(252, n - 1):
    if not nearhigh[p] or np.isnan(rsi[p]):
        continue
    # local RSI trough: rsi turning back up
    if rsi[p] < 45 and rsi[p] <= rsi[p - 1] and rsi[p] < rsi[p + 1] and p - last >= 8:
        d = idx[p]
        gate = (pd.notna(e63.asof(d)) and e63.asof(d) > spy63.asof(d)) or (pd.notna(e200.asof(d)) and e.asof(d) > e200.asof(d))
        # forward 40-bar move (what the dip-buy would have made)
        fwd = (close[min(p + 40, n - 1)] - close[p]) / close[p] * 100
        print(f"  {d.date()}  ${close[p]:6.2f}  RSI={rsi[p]:4.0f}  pos={pos[p]*100:3.0f}%  gate={'OK' if gate else 'no'}  fwd40d={fwd:+.0f}%")
        last = p
