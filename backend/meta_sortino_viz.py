#!/usr/bin/env python
"""Sortino-RSI technique on META with ENTRY + EXIT('RSI20 goes red') state machine, dumped for a verification chart."""
import os, json
import numpy as np, pandas as pd
os.environ.setdefault("DJANGO_SETTINGS_MODULE", "rotation.settings")
import django
django.setup()
from seq_fundamental_study import load_candles

WIN, SMOOTH, RSI14, RSI20, SMA, GREEN_THR, RED_THR, GATE = 14, 14, 14, 20, 14, 51.0, 49.0, 50.0
TK = os.environ.get("TK", "META")


def wr(x, n):
    d = x.diff(); up = d.clip(lower=0.0); dn = (-d).clip(lower=0.0)
    ru = up.ewm(alpha=1.0 / n, adjust=False, min_periods=n).mean()
    rd = dn.ewm(alpha=1.0 / n, adjust=False, min_periods=n).mean()
    return 100.0 - 100.0 / (1.0 + ru / rd.replace(0.0, np.nan))


cd = load_candles([TK, "SPY"])
spy = cd["SPY"]["Close"]; px = cd[TK]["Close"]; hi = cd[TK]["High"]
ix = px.index.intersection(spy.index); px = px.reindex(ix); spy = spy.reindex(ix); hi = hi.reindex(ix)
px = px[px > 0]; hi = hi.reindex(px.index)
ex = px.pct_change() - spy.pct_change()
me = ex.rolling(WIN).mean(); dd = np.sqrt((ex.clip(upper=0.0) ** 2).rolling(WIN).mean())
sortino = (me / dd.replace(0.0, np.nan)).mask(dd < 1e-10, np.sign(me) * 9.99)
sortino_s = sortino.ffill().fillna(0.0).rolling(SMOOTH, min_periods=SMOOTH).mean()   # smoothing length on the Sortino
rsi14 = wr(sortino_s, RSI14)                              # RSI of the SMOOTHED Sortino
rsi20 = wr(px, RSI20)                                     # RSI(20) on CLOSE (user: RSI is on close, always) = color
sma14 = rsi14.rolling(SMA).mean()
entry = (rsi20 > GREEN_THR) & (rsi14 < GATE) & (rsi14.shift(1) <= sma14.shift(1)) & (rsi14 > sma14)
exit_red = rsi20 < RED_THR                                # EXIT when color goes RED (<49)

df = pd.DataFrame({"close": px, "sortino": sortino, "sortino_s": sortino_s, "rsi14": rsi14, "rsi20": rsi20, "sma14": sma14,
                   "entry": entry.fillna(False), "exit_red": exit_red.fillna(False)}).dropna()
buy = []; sell = []; held = []; in_pos = False; entry_px = None; trades = []; tnum = 0
for i in range(len(df)):
    was = in_pos; b = s = False
    if not in_pos and bool(df["entry"].iloc[i]):
        in_pos = True; b = True; tnum += 1
        entry_px = float(df["close"].iloc[i]); entry_dt = df.index[i]; entry_i = i; entry_r14 = float(df["rsi14"].iloc[i])
    elif in_pos and bool(df["exit_red"].iloc[i]):
        in_pos = False; s = True; xpx = float(df["close"].iloc[i])
        trades.append(dict(n=tnum, entry=str(entry_dt.date()), exit=str(df.index[i].date()),
                           entry_px=round(entry_px, 2), exit_px=round(xpx, 2), hold_days=int(i - entry_i),
                           rsi14_entry=round(entry_r14, 1), ret_pct=round((xpx / entry_px - 1) * 100, 1), open=False))
    buy.append(b); sell.append(s); held.append(was or b)
df["buy"] = buy; df["sell"] = sell; df["held"] = held
if in_pos:
    xpx = float(df["close"].iloc[-1])
    trades.append(dict(n=tnum, entry=str(entry_dt.date()), exit="(open)",
                       entry_px=round(entry_px, 2), exit_px=round(xpx, 2), hold_days=int(len(df) - 1 - entry_i),
                       rsi14_entry=round(entry_r14, 1), ret_pct=round((xpx / entry_px - 1) * 100, 1), open=True))

# strategy equity (long only while held) vs buy-and-hold, on this single ticker
dret = df["close"].pct_change().fillna(0.0)
heldser = df["held"].astype(bool)
strat_ret = dret.where(heldser, 0.0)
strat_eq = (1.0 + strat_ret).cumprod()
bh_eq = df["close"] / float(df["close"].iloc[0])

df = df.tail(1050)                                        # ~4y window incl the 2022 crash + 2023-24 recovery
strat_eq = strat_eq.reindex(df.index); strat_eq = strat_eq / float(strat_eq.iloc[0])
bh_eq = df["close"] / float(df["close"].iloc[0])
win = df.index[0]
vtrades = [t for t in trades if t["entry"] >= str(win.date())]   # trades visible in the plotted window
out = dict(ticker=TK,
           dates=[d.strftime("%Y-%m-%d") for d in df.index],
           close=[round(float(x), 2) for x in df["close"]],
           sortino=[round(float(x), 3) for x in df["sortino"]],
           sortino_s=[round(float(x), 3) for x in df["sortino_s"]],
           rsi14=[round(float(x), 1) for x in df["rsi14"]],
           rsi20=[round(float(x), 1) for x in df["rsi20"]],
           sma14=[round(float(x), 1) for x in df["sma14"]],
           strat_eq=[round(float(x), 4) for x in strat_eq],
           bh_eq=[round(float(x), 4) for x in bh_eq],
           buy=[bool(x) for x in df["buy"]], sell=[bool(x) for x in df["sell"]], held=[bool(x) for x in df["held"]],
           trades=trades, vtrades=vtrades,
           params=dict(win=WIN, smooth=SMOOTH, rsi14=RSI14, rsi20=RSI20, sma=SMA, green=GREEN_THR, red=RED_THR, gate=GATE))
json.dump(out, open("/app/.data/studies/meta_sortino_viz.json", "w"))
print(TK, "range:", df.index[0].date(), "->", df.index[-1].date(), "| buys:", int(df["buy"].sum()), "| sells:", int(df["sell"].sum()))
print("full-history round-trips:", len(trades), "| in-window:", len(vtrades))
print("last 12:", [(t["entry"], t["exit"], f'{t["ret_pct"]:+}%') for t in trades[-12:]])
print("saved /app/.data/studies/meta_sortino_viz.json rows=", len(df))
