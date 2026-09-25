#!/usr/bin/env python
"""Sortino-RSI viz for BTC using a FRESH EODHD fetch (BTC-USD.CC + SPY.US) — non-destructive, no DB writes.
Windows the plot from START (default 2024-01-01) but computes indicators on full history for correct warmup.
Dumps the same JSON shape that meta_sortino_png.py + build_served_page.py consume."""
import os, json, datetime as dt
import numpy as np, pandas as pd
os.environ.setdefault("DJANGO_SETTINGS_MODULE", "rotation.settings")
import django
django.setup()
from api.tasks import _eodhd_get

def _i(k, d): return int(os.environ.get(k, d))
def _f(k, d): return float(os.environ.get(k, d))
WIN = _i("WIN", 14)                                       # ADRIDEM "Rolling Window Length"
SMOOTH = _i("SMOOTH", 14)                                 # ADRIDEM "Smoothing Length" (SMA on the Sortino before its RSI)
RF = _f("RF", 0.02)                                       # ADRIDEM annual risk-free rate (~0 effect on daily)
RSI14 = _i("RSI14", 14)                                   # RSI length on the (smoothed) Sortino
RSI20 = _i("RSI20", 14)                                   # RSI length on the price (CLOSE) = green/red color (user: back to 14)
SMA = _i("SMA", 14)                                       # SMA of the Sortino-RSI (the line it must cross up)
GREEN_THR = _f("GREEN_THR", 51)                           # price RSI >= this = green (uptrend)
RED_THR = _f("RED_THR", 49)                               # price RSI < this = red (exit)
GATE = _f("GATE", 50)                                     # Sortino-RSI disarm level
CROSS_THR = _f("CROSS_THR", 30)                           # the arming up-cross must occur while Sortino-RSI < this (user: under 30)
ARM_WIN = _i("ARM_WIN", 10)                               # first GREEN may fire up to this many days AFTER the cross
OVERSOLD_THR = _f("OVERSOLD_THR", 30)                     # deep-oversold precondition level
OVERSOLD_MIN = _i("OVERSOLD_MIN", 5)                      # require >= this many days under OVERSOLD_THR in the run-up (0 = off)
OVERSOLD_WIN = _i("OVERSOLD_WIN", 20)                     # trailing window for the oversold-day count
OVERSOLD_DEEP = _f("OVERSOLD_DEEP", 10)                   # during the oversold run-up the Sortino-RSI must also dip BELOW this (capitulation; 0 = off)
EXIT_MODE = os.environ.get("EXIT", "red")                # exit: red (hold until color red) | downcross_red (needs the Sortino down-cross too)
SIMPLE = _i("SIMPLE", 1)                                  # SIMPLE mode: BUY on Sortino-RSI up-cross of its SMA, SELL on down-cross. Nothing else.
TREND_MA = _i("TREND_MA", 50)                             # trend filter: only BUY when price > its TREND_MA-day SMA (0 = off)
WEEKLY = _i("WEEKLY", 1)                                  # require the SAME entry arm on the WEEKLY timeframe too
TK = os.environ.get("TK", "BTC-USD")
LABEL = os.environ.get("LABEL", TK)                       # display name + output filename (lets us render 2 variants of one ticker)
START = os.environ.get("START", "2024-01-01")
SYM = {"BTC-USD": "BTC-USD.CC", "SPY": "SPY.US"}
def _sym(tk):
    return SYM.get(tk, f"{tk}.US")                        # default US stocks to <TK>.US


def wr(x, n):
    d = x.diff(); up = d.clip(lower=0.0); dn = (-d).clip(lower=0.0)
    ru = up.ewm(alpha=1.0 / n, adjust=False, min_periods=n).mean()
    rd = dn.ewm(alpha=1.0 / n, adjust=False, min_periods=n).mean()
    return 100.0 - 100.0 / (1.0 + ru / rd.replace(0.0, np.nan))


def fetch(sym):
    frm = (dt.date.today() - dt.timedelta(days=int(6 * 365.25))).isoformat()
    resp = _eodhd_get(f"eod/{sym}", **{"from": frm, "period": "d"})
    rows = {}
    for r in resp:
        d = r.get("date"); cl = r.get("close"); adj = r.get("adjusted_close", cl)
        if not d or cl in (None, "") or adj in (None, ""):
            continue
        cl = float(cl); adj = float(adj); fac = adj / cl if cl else 1.0
        rows[pd.Timestamp(d)] = dict(Open=float(r.get("open") or cl) * fac, High=float(r.get("high") or cl) * fac,
                                     Low=float(r.get("low") or cl) * fac, Close=adj)
    df = pd.DataFrame(rows).T.sort_index()
    return df


bt = fetch(_sym(TK))                                     # ADRIDEM Sortino uses the ticker's OWN returns (no SPY subtraction)
px = bt["Close"]; hi = bt["High"]
px = px[px > 0]; hi = hi.reindex(px.index)
print(f"{TK}: {px.index[0].date()} -> {px.index[-1].date()}  last close {px.iloc[-1]:.0f}")

# --- ADRIDEM "Rolling Sortino Ratio with Ref Ticker" (© adridem), exact:
ret = px.pct_change()                                     # ta.change(close)/close[1]
mean_ret = ret.rolling(WIN).mean()                        # ta.sma(dailyReturn, length)
downside = ret.clip(upper=0.0).rolling(WIN).std(ddof=0)   # ta.stdev(math.min(dailyReturn,0), length)  (population)
adj_rf = (1.0 + RF) ** (1.0 / 525600.0) - 1.0            # daily: minutes_per_year/multiplier -> ~0
sortino = (mean_ret - adj_rf) / downside.replace(0.0, np.nan)
sortino_s = sortino.rolling(SMOOTH, min_periods=SMOOTH).mean()   # ta.sma(sortinoRatio, smoothing_length)
rsi14 = wr(sortino_s.ffill().fillna(0.0), RSI14)         # RSI(14) of the SMOOTHED Sortino (matches TV)
# Reference-ticker (SPY) ADRIDEM Sortino — the indicator's 2nd line, shown red for comparison
sp = fetch(_sym("SPY")); spc = sp["Close"]; spc = spc[spc > 0]; sret = spc.pct_change()
spy_sortino = (sret.rolling(WIN).mean() - adj_rf) / sret.clip(upper=0.0).rolling(WIN).std(ddof=0).replace(0.0, np.nan)
spy_sortino_s = spy_sortino.rolling(SMOOTH, min_periods=SMOOTH).mean().reindex(px.index).ffill()
ohlc4 = (bt["Open"] + bt["High"] + bt["Low"] + bt["Close"]) / 4.0   # TradingView source = ohlc4
ohlc4 = ohlc4.reindex(px.index)
rsi20 = wr(ohlc4, RSI20)                                  # color-RSI on OHLC4 (matches user's TradingView, verified 48.1 on 2025-02-20)
sma14 = rsi14.rolling(SMA).mean()
# DEEP-OVERSOLD precondition: Sortino-RSI under OVERSOLD_THR for >= OVERSOLD_MIN of the last OVERSOLD_WIN days
oversold_ok = pd.Series(True, index=rsi14.index)
if OVERSOLD_MIN > 0:
    days_ok = (rsi14 < OVERSOLD_THR).astype(float).rolling(OVERSOLD_WIN, min_periods=OVERSOLD_MIN).sum() >= OVERSOLD_MIN
    oversold_ok = days_ok.fillna(False)
    if OVERSOLD_DEEP > 0:                                              # AND a capitulation dip below OVERSOLD_DEEP in that window
        oversold_ok = (days_ok & (rsi14.rolling(OVERSOLD_WIN, min_periods=1).min() < OVERSOLD_DEEP)).fillna(False)
# ENTRY arm: up-cross of Sortino-RSI thru its SMA WHILE < CROSS_THR (under 30), after a deep-oversold run-up
deep_cross = ((rsi14.shift(1) <= sma14.shift(1)) & (rsi14 > sma14) & (rsi14 < CROSS_THR) & oversold_ok).fillna(False)
exit_cross = ((rsi14.shift(1) >= sma14.shift(1)) & (rsi14 < sma14)).fillna(False)                    # EXIT arm: Sortino-RSI crosses DOWN its SMA
green = rsi20 > GREEN_THR                                 # entry confirm: price green (RSI20 on CLOSE)
red = rsi20 < RED_THR                                     # exit confirm: price red
if SIMPLE:                                                # CROSSOVER: up-cross buys (after >=OVERSOLD_MIN days under 30), down-cross sells
    up_s = ((rsi14.shift(1) <= sma14.shift(1)) & (rsi14 > sma14)).fillna(False)
    if OVERSOLD_MIN > 0:
        days_ok_s = ((rsi14 < OVERSOLD_THR).astype(float).rolling(OVERSOLD_WIN, min_periods=OVERSOLD_MIN).sum() >= OVERSOLD_MIN).fillna(False)
        up_s = up_s & days_ok_s
    if TREND_MA > 0:                                       # trend filter: only buy when price is above its TREND_MA-day SMA
        up_s = (up_s & (px > px.rolling(TREND_MA).mean())).fillna(False)
    deep_cross = up_s
    green = pd.Series(True, index=rsi20.index)
# WEEKLY entry confirmation — same ADRIDEM arm on weekly bars, mapped down to daily
wk_armed_d = pd.Series(True, index=px.index)
if WEEKLY and not SIMPLE:
    wk = px.resample("W-FRI").last(); wret = wk.pct_change()
    wsrt = (wret.rolling(WIN).mean() - adj_rf) / wret.clip(upper=0.0).rolling(WIN).std(ddof=0).replace(0.0, np.nan)
    wr14 = wr(wsrt.rolling(SMOOTH, min_periods=SMOOTH).mean().ffill().fillna(0.0), RSI14); wsma = wr14.rolling(SMA).mean()
    wa = pd.Series(False, index=wk.index); st = False; rv = wr14.to_numpy(); sv = wsma.to_numpy()
    for j in range(len(wk)):
        if j > 0 and rv[j - 1] <= sv[j - 1] and rv[j] > sv[j] and rv[j] < GATE:
            st = True
        elif st and (rv[j] >= GATE or rv[j] <= sv[j]):
            st = False
        wa.iloc[j] = st
    wk_armed_d = wa.reindex(px.index, method="ffill").fillna(False)

df = pd.DataFrame({"close": px, "sortino": sortino, "sortino_s": sortino_s, "rsi14": rsi14, "rsi20": rsi20, "sma14": sma14,
                   "deep_cross": deep_cross, "exit_cross": exit_cross, "green": green.fillna(False), "red": red.fillna(False),
                   "wk_armed": wk_armed_d.astype(bool)}).dropna()
def _entry_reason(cdt, cr14, e_rsi20, lag):
    when = "same day" if lag == 0 else f"{lag} bar{'s' if lag != 1 else ''} after the cross"
    if SIMPLE:
        return (f"<b>BUY — why</b><br>"
                f"1) Sortino RSI{RSI14} spent ≥{OVERSOLD_MIN}d under {OVERSOLD_THR:.0f} in the last {OVERSOLD_WIN}d ✓<br>"
                f"2) Sortino RSI{RSI14} crossed <b>up</b> its SMA{SMA} on {cdt} at <b>{cr14:.1f}</b> → BUY ✓")
    return (f"<b>BUY — why</b><br>"
            f"0) DEEP-OVERSOLD: Sortino RSI{RSI14} spent ≥{OVERSOLD_MIN}d under {OVERSOLD_THR:.0f} in the last {OVERSOLD_WIN}d ✓<br>"
            f"1) ARM: Sortino RSI{RSI14} crossed <b>up</b> its SMA{SMA} on {cdt} at <b>{cr14:.1f}</b> (&lt; {CROSS_THR:.0f}) ✓<br>"
            f"2) CONFIRM: price RSI{RSI20}(ohlc4) <b>{e_rsi20:.1f}</b> ≥ green {GREEN_THR:.0f} ({when}, ≤{ARM_WIN}d after) ✓<br>"
            f"3) WEEKLY: same Sortino-RSI arm active on the weekly ✓" + ("" if WEEKLY else " (off)"))


def _exit_reason(ddt, drsi14, x_rsi20, lag):
    when = "same day" if lag == 0 else f"{lag} bar{'s' if lag != 1 else ''} after the down-cross"
    if SIMPLE:
        return (f"<b>SELL — why</b><br>"
                f"Sortino RSI{RSI14} crossed <b>down</b> its SMA{SMA} on {ddt} at <b>{drsi14:.1f}</b> → SELL ✓")
    if EXIT_MODE == "red":
        return (f"<b>EXIT — why</b><br>"
                f"held until the color turned <b>RED</b>: price RSI{RSI20}(ohlc4) <b>{x_rsi20:.1f}</b> &lt; red {RED_THR:.0f} ✓")
    return (f"<b>EXIT — why</b><br>"
            f"1) Sortino RSI{RSI14} crossed <b>down</b> its SMA{SMA} on {ddt} at <b>{drsi14:.1f}</b> ✓<br>"
            f"2) price RSI{RSI20}(ohlc4) <b>{x_rsi20:.1f}</b> &lt; red {RED_THR:.0f} ({when}) ✓<br>"
            f"→ red alone does NOT exit; needs the down-cross too")


buy = []; sell = []; held = []; cross = []; dcross = []; in_pos = False; entry_px = None; trades = []; tnum = 0
entry_armed = False; exit_armed = False
last_cdt = None; last_cr14 = None; last_ci = None; last_ddt = None; last_dr14 = None; last_di = None
for i in range(len(df)):
    was = in_pos; b = s = False; xc = bool(df["deep_cross"].iloc[i]); xdc = bool(df["exit_cross"].iloc[i])
    r14i = float(df["rsi14"].iloc[i]); smi = float(df["sma14"].iloc[i])
    # ENTRY arm: a deep-oversold cross-under-30 arms; the arm lasts ARM_WIN bars (first GREEN in that window fires)
    if xc:
        entry_armed = True; last_cdt = str(df.index[i].date()); last_cr14 = r14i; last_ci = i
    elif entry_armed and last_ci is not None and (i - last_ci) > ARM_WIN:
        entry_armed = False                                             # 10-day window expired without a green
    # EXIT arm: down-cross arms; persists until it crosses back up (reset); no day limit
    if xdc:
        exit_armed = True; last_ddt = str(df.index[i].date()); last_dr14 = r14i; last_di = i
    elif exit_armed and r14i >= smi:
        exit_armed = False
    if not in_pos and entry_armed and bool(df["green"].iloc[i]) and (not WEEKLY or bool(df["wk_armed"].iloc[i])):
        in_pos = True; b = True; tnum += 1; entry_armed = False
        entry_px = float(df["close"].iloc[i]); entry_dt = df.index[i]; entry_i = i; entry_r14 = r14i
        e_rsi20 = float(df["rsi20"].iloc[i]); e_cdt = last_cdt; e_cr14 = last_cr14; e_lag = i - (last_ci if last_ci is not None else i)
        e_reason = _entry_reason(e_cdt, e_cr14, e_rsi20, e_lag)
    elif in_pos and (bool(df["exit_cross"].iloc[i]) if SIMPLE else (bool(df["red"].iloc[i]) if EXIT_MODE == "red" else (exit_armed and bool(df["red"].iloc[i])))):
        in_pos = False; s = True; exit_armed = False; xpx = float(df["close"].iloc[i]); x_rsi20 = float(df["rsi20"].iloc[i])
        x_ddt = last_ddt; x_dr14 = last_dr14; x_lag = i - (last_di if last_di is not None else i)
        x_reason = _exit_reason(x_ddt, x_dr14, x_rsi20, x_lag)
        trades.append(dict(n=tnum, entry=str(entry_dt.date()), exit=str(df.index[i].date()),
                           entry_px=round(entry_px, 2), exit_px=round(xpx, 2), hold_days=int(i - entry_i),
                           rsi14_entry=round(entry_r14, 1), ret_pct=round((xpx / entry_px - 1) * 100, 1), open=False,
                           cross_date=e_cdt, cross_rsi14=round(e_cr14, 1) if e_cr14 is not None else None,
                           entry_rsi20=round(e_rsi20, 1), green_lag=e_lag, exit_rsi20=round(x_rsi20, 1),
                           entry_reason=e_reason, exit_reason=x_reason))
    buy.append(b); sell.append(s); held.append(was or b); cross.append(xc); dcross.append(xdc)
df["buy"] = buy; df["sell"] = sell; df["held"] = held; df["cross"] = cross; df["dcross"] = dcross
if in_pos:
    xpx = float(df["close"].iloc[-1]); x_rsi20 = float(df["rsi20"].iloc[-1])
    below = bool(df["rsi14"].iloc[-1] < df["sma14"].iloc[-1])
    need = "Sortino-RSI already below its SMA; waiting for red" if below else "Sortino-RSI still above its SMA (no down-cross yet)"
    x_reason = (f"<b>OPEN</b><br>no exit yet — needs Sortino-RSI cross <b>down</b> AND red.<br>{need}; RSI{RSI20} {x_rsi20:.1f}")
    trades.append(dict(n=tnum, entry=str(entry_dt.date()), exit="(open)",
                       entry_px=round(entry_px, 2), exit_px=round(xpx, 2), hold_days=int(len(df) - 1 - entry_i),
                       rsi14_entry=round(entry_r14, 1), ret_pct=round((xpx / entry_px - 1) * 100, 1), open=True,
                       cross_date=e_cdt, cross_rsi14=round(e_cr14, 1) if e_cr14 is not None else None,
                       entry_rsi20=round(e_rsi20, 1), green_lag=e_lag, exit_rsi20=round(x_rsi20, 1),
                       entry_reason=e_reason, exit_reason=x_reason))

dret = df["close"].pct_change().fillna(0.0)
strat_eq_full = (1.0 + dret.where(df["held"].astype(bool), 0.0)).cumprod()

df = df[df.index >= pd.Timestamp(START)]                  # window for the plot
strat_eq = strat_eq_full.reindex(df.index); strat_eq = strat_eq / float(strat_eq.iloc[0])
bh_eq = df["close"] / float(df["close"].iloc[0])
vtrades = [t for t in trades if t["entry"] >= START]
out = dict(ticker=LABEL,
           dates=[d.strftime("%Y-%m-%d") for d in df.index],
           close=[round(float(x), 2) for x in df["close"]],
           sortino=[round(float(x), 3) for x in df["sortino"]],
           sortino_s=[round(float(x), 3) for x in df["sortino_s"]],
           spy_sortino_s=[round(float(x), 3) if pd.notna(x) else None for x in spy_sortino_s.reindex(df.index)],
           rsi14=[round(float(x), 1) for x in df["rsi14"]],
           rsi20=[round(float(x), 1) for x in df["rsi20"]],
           sma14=[round(float(x), 1) for x in df["sma14"]],
           strat_eq=[round(float(x), 4) for x in strat_eq],
           bh_eq=[round(float(x), 4) for x in bh_eq],
           buy=[bool(x) for x in df["buy"]], sell=[bool(x) for x in df["sell"]], held=[bool(x) for x in df["held"]],
           cross=[bool(x) for x in df["cross"]], dcross=[bool(x) for x in df["dcross"]],
           trades=vtrades, vtrades=vtrades,
           params=dict(win=WIN, smooth=SMOOTH, rsi14=RSI14, rsi20=RSI20, sma=SMA, green=GREEN_THR, red=RED_THR, gate=GATE, arm_win=ARM_WIN))
_safe = LABEL.replace("/", "_").replace(" ", "_")
json.dump(out, open(f"/app/.data/studies/viz_{_safe}.json", "w"))
json.dump(out, open("/app/.data/studies/meta_sortino_viz.json", "w"))   # back-compat (last rendered)
print(f"window {df.index[0].date()} -> {df.index[-1].date()}  rows={len(df)}  in-window trades={len(vtrades)}")
print(f"RSI(OHLC4) last = {df['rsi20'].iloc[-1]:.1f}   RSI14(Sortino) last = {df['rsi14'].iloc[-1]:.1f}")
