#!/usr/bin/env python
"""Compute the Sortino-RSI signal on BTC-USD vs SPY and dump a JSON series for visual verification."""
import os, json
import numpy as np, pandas as pd
os.environ.setdefault("DJANGO_SETTINGS_MODULE", "rotation.settings")
import django
django.setup()
from seq_fundamental_study import load_candles

WIN, RSI14, RSI20, SMA, OS_LOOKBACK, GREEN_THR, OS_THR = 14, 14, 20, 14, 20, 51.0, 20.0


def wilder_rsi(x, n):
    d = x.diff(); up = d.clip(lower=0.0); dn = (-d).clip(lower=0.0)
    ru = up.ewm(alpha=1.0 / n, adjust=False, min_periods=n).mean()
    rd = dn.ewm(alpha=1.0 / n, adjust=False, min_periods=n).mean()
    return 100.0 - 100.0 / (1.0 + ru / rd.replace(0.0, np.nan))


cd = load_candles(["BTC-USD", "SPY"])
spy = cd["SPY"]["Close"]
btc = cd["BTC-USD"]["Close"]
idx = btc.index.intersection(spy.index)                       # align to SPY trading days
btc = btc.reindex(idx); spy = spy.reindex(idx)
btc = btc[btc > 0]
ret = btc.pct_change(); spy_ret = spy.pct_change()
excess = ret - spy_ret
mean_ex = excess.rolling(WIN).mean()
dd = np.sqrt((excess.clip(upper=0.0) ** 2).rolling(WIN).mean())
sortino = (mean_ex / dd.replace(0.0, np.nan))
low = dd < 1e-10
sortino = sortino.mask(low & (mean_ex.abs() >= 1e-10), np.sign(mean_ex) * 9.99).mask(low & (mean_ex.abs() < 1e-10), np.nan)
rsi14 = wilder_rsi(sortino.ffill().fillna(0.0), RSI14)              # fast RSI of the rel-Sortino-vs-SPY
rsi20 = wilder_rsi(btc, RSI20)                                      # RSI(20) on the STOCK PRICE -> COLOR (green/red)
sma14 = rsi14.rolling(SMA).mean()
green = rsi20 > GREEN_THR                                           # price RSI20 > 51 = GREEN (uptrend)
below50 = rsi14 < 50.0                                              # rel-Sortino RSI14 below midline (the LOWER the better)
cross_up = (rsi14.shift(1) <= sma14.shift(1)) & (rsi14 > sma14)     # rsi14 crosses UP through its SMA14 (the turn)
buy = (green & below50 & cross_up).fillna(False)                    # BUY
print("buys =", int(buy.sum()))

df = pd.DataFrame({"close": btc, "sortino": sortino, "rsi14": rsi14, "rsi20": rsi20,
                   "sma14": sma14, "oversold": (rsi14 < OS_THR), "green": green, "buy": buy}).dropna()
df = df.tail(900)                                              # recent ~3.5y window for a readable chart
out = dict(
    dates=[d.strftime("%Y-%m-%d") for d in df.index],
    close=[round(float(x), 1) for x in df["close"]],
    sortino=[round(float(x), 3) for x in df["sortino"]],
    rsi14=[round(float(x), 1) for x in df["rsi14"]],
    rsi20=[round(float(x), 1) for x in df["rsi20"]],
    sma14=[round(float(x), 1) for x in df["sma14"]],
    oversold=[bool(x) for x in df["oversold"]],
    green=[bool(x) for x in df["green"]],
    buy=[bool(x) for x in df["buy"]],
    params=dict(win=WIN, rsi14=RSI14, rsi20=RSI20, sma=SMA, os_lookback=OS_LOOKBACK, green_thr=GREEN_THR, os_thr=OS_THR),
)
json.dump(out, open("/app/.data/studies/btc_sortino_viz.json", "w"))
print("buys in window:", int(df["buy"].sum()), "| dates:", df.index[0].date(), "->", df.index[-1].date())
print("buy dates:", [d.strftime("%Y-%m-%d") for d in df.index[df["buy"]]])
print("saved /app/.data/studies/btc_sortino_viz.json rows=", len(df))
