#!/usr/bin/env python
"""NEW CONCEPT strategy (user 2026-09-22, confirmed on BTC chart): rel-Sortino-vs-SPY RSI dip-and-turn, in an uptrend.

All daily:
  sortino     = ADRIDEM: (SMA(own_ret,14) - rf) / stdev(min(own_ret,0),14)   # ticker's OWN return, pop-stdev downside, NO SPY
  sortino_s   = SMA(SMOOTH=14) of sortino                              # smoothing length on the Sortino (user)
  rsi14_sort  = RSI(14) of sortino_s  ;  sma14 = SMA(14) of rsi14_sort
  rsi20_price = RSI(20) of the STOCK CLOSE  -> COLOR: >51 GREEN (uptrend), <49 red   (user: RSI is on close, always)

  BUY when:  rsi14_sort crosses UP through its SMA14 (the turn) WHILE < 50 (gate; lower=better)  -> ARMS the entry
             for ARM_WIN(5) bars; the BUY fires on the first bar in that window where rsi20_price > 51 (GREEN).
             (user: "it can cross a few days before as long as it crosses under the 50"; green can lag the turn.)
  SIZE:  depth-weighted -> weight = (50 - rsi14_sort_at_entry), so a deeper dip gets a bigger position (user).
  Entry: next trading day's close. Liquidity: 63d median $vol >= $5M. Exit: HOLD_D-day hold (PLACEHOLDER, no exit spec yet).
  Delisting-aware; one position per name at a time; cash when nothing open. Runs equal-weight AND depth-weight for comparison.

Run: MSYS_NO_PATHCONV=1 docker exec rotation-backend-1 python -u /app/sortino_rsi_strategy.py
Env: WIN(14) SMOOTH(14) RSI14(14) RSI20(20) SMA(14) HOLD_D(21) GREEN_THR(51) GATE(50) MINVOL(5e6) PANELS BENCH(SPY)
"""
import os, json, pickle
import numpy as np, pandas as pd
os.environ.setdefault("DJANGO_SETTINGS_MODULE", "rotation.settings")
import django
django.setup()
from seq_fundamental_study import load_candles

WIN = int(os.environ.get("WIN", 14)); RSI14 = int(os.environ.get("RSI14", 14)); RSI20 = int(os.environ.get("RSI20", 14))
SMA = int(os.environ.get("SMA", 14)); HOLD_D = int(os.environ.get("HOLD_D", 21)); SMOOTH = int(os.environ.get("SMOOTH", 14))
MAXHOLD = int(os.environ.get("MAXHOLD", 252))                        # safety cap only if the exit rule never fires
MAXPOS = int(os.environ.get("MAXPOS", 3))                            # concentrated book: hold at most this many names (deepest dips first)
TOP_MCAP = int(os.environ.get("TOP_MCAP", 50))                       # candidate pool = top-N by PIT market cap (0 = no cap filter)
WEEKLY = int(os.environ.get("WEEKLY", 1))                            # require the SAME entry arm on the WEEKLY timeframe too
SCENARIO = os.environ.get("SCENARIO", "")                            # if set, dump this run's result to scenarios/<name>.json
ARM_WIN = int(os.environ.get("ARM_WIN", 10))                         # cross arms the entry for this many bars; the first GREEN within the window fires (user: up to 10 days after)
RF = float(os.environ.get("RF", 0.02))                               # ADRIDEM annual risk-free rate (~0 effect on daily)
START_DATE = os.environ.get("START_DATE", "2016-01-01")              # portfolio floor — universe has <20 names before 2015, 971 by 2016
GREEN_THR = float(os.environ.get("GREEN_THR", 51)); RED_THR = float(os.environ.get("RED_THR", 49))
GATE = float(os.environ.get("GATE", 50)); MINVOL = float(os.environ.get("MINVOL", 5e6))
SIMPLE = int(os.environ.get("SIMPLE", 1))                             # SIMPLE mode (user 2026-09-23): BUY on Sortino-RSI up-cross of its SMA, SELL on down-cross. Nothing else.
TREND_MA = int(os.environ.get("TREND_MA", 50))                        # trend filter: only BUY when price > its TREND_MA-day SMA (0 = off). 50 splits MSFT win 77% vs 38%.
SRC = os.environ.get("SRC", "ohlc4")                                  # color-RSI source: ohlc4 (matches TradingView) or close
RANK = os.environ.get("RANK", "confirm")                              # pick among candidates: confirm (greenest turn) | depth (deepest dip) | confirm_srt
SORTINO_POS = int(os.environ.get("SORTINO_POS", 0))                   # entry gate: only buy when the smoothed Sortino > 0 (own trend already positive) — REFUTED (kills the dip entries)
EXIT = os.environ.get("EXIT", "red")                                  # exit rule: red (hold until the color turns red) | downcross_red | weekly_break | trail
TRAIL_PCT = float(os.environ.get("TRAIL_PCT", 0.15))                  # trailing-stop drawdown from entry-high (EXIT=trail)
PARK = os.environ.get("PARK", "cash")                                 # idle capital: cash (0%) | qqq (hold QQQ when a slot is empty)
COST_BPS = float(os.environ.get("COST_BPS", 0))                       # one-way transaction cost in bps, charged on every stock entry and exit
OVERSOLD_THR = float(os.environ.get("OVERSOLD_THR", 30))             # deep-oversold precondition: Sortino-RSI level to count as "under"
OVERSOLD_MIN = int(os.environ.get("OVERSOLD_MIN", 5))               # require >= this many days under OVERSOLD_THR in the run-up before the up-cross (0 = off)
OVERSOLD_WIN = int(os.environ.get("OVERSOLD_WIN", 20))              # trailing window (days) over which those oversold days are counted
CROSS_THR = float(os.environ.get("CROSS_THR", 30))                  # the arming up-cross must occur while Sortino-RSI < this level (user: under 30)
OVERSOLD_DEEP = float(os.environ.get("OVERSOLD_DEEP", 10))          # during the oversold run-up the Sortino-RSI must also dip BELOW this (capitulation touch; 0 = off)
PANELS = os.environ.get("PANELS", "/app/.data/panels.pkl"); BENCH = os.environ.get("BENCH", "SPY")


def wilder_rsi(x, n):
    d = x.diff(); up = d.clip(lower=0.0); dn = (-d).clip(lower=0.0)
    ru = up.ewm(alpha=1.0 / n, adjust=False, min_periods=n).mean()
    rd = dn.ewm(alpha=1.0 / n, adjust=False, min_periods=n).mean()
    return 100.0 - 100.0 / (1.0 + ru / rd.replace(0.0, np.nan))


def _cummax_col(a):
    for k in range(1, a.shape[0]):
        a[k] = np.maximum(a[k], a[k - 1])
    return a


def adridem_rsi_sma(close_df):
    """ADRIDEM Sortino -> RSI14 of the SMOOTHED sortino, its SMA, and the smoothed sortino itself.
    Works on any timeframe's close panel."""
    ret = close_df.pct_change()
    m = ret.rolling(WIN).mean(); dsd = ret.clip(upper=0.0).rolling(WIN).std(ddof=0)
    adj = (1.0 + RF) ** (1.0 / 525600.0) - 1.0
    srt = (m - adj) / dsd.replace(0.0, np.nan)
    srt_s = srt.rolling(SMOOTH, min_periods=SMOOTH).mean()
    r14 = wilder_rsi(srt_s.ffill().fillna(0.0), RSI14)
    return r14, r14.rolling(SMA).mean(), srt_s


def entry_armed_mask(r14, sma, precond=None, cross_thr=None, arm_win=None):
    """ENTRY arm from an up-cross of RSI thru its SMA that happens while RSI < cross_thr.
    precond (optional bool DataFrame): the arming cross ONLY counts on bars where precond is True (e.g. deep-oversold run-up).
    arm_win: if given, the arm lasts a FIXED window — armed on the cross bar and up to arm_win bars AFTER (green may fire late).
             if None, the arm is PERSISTENT (stays armed until RSI reaches GATE or crosses back down)."""
    thr = GATE if cross_thr is None else cross_thr
    r = r14.to_numpy(); s = sma.to_numpy(); idx = np.arange(len(r)).reshape(-1, 1)
    up = np.vstack([np.full((1, r.shape[1]), False), r[:-1] <= s[:-1]]) & (r > s) & (r < thr)
    if precond is not None:
        up = up & precond.to_numpy()
    up_df = pd.DataFrame(up, index=r14.index, columns=r14.columns)
    if arm_win is not None and arm_win > 0:                              # fixed-window arm: cross within the last arm_win bars
        armed = up_df.astype(float).rolling(arm_win + 1, min_periods=1).max().fillna(0.0).astype(bool)
        return armed, up_df
    disarm = (r >= GATE) | (r <= s)                                     # persistent arm (original behavior)
    lc = _cummax_col(np.where(up, idx, -1)); ld = _cummax_col(np.where(disarm, idx, -1))
    armed = pd.DataFrame((lc > ld) & (lc >= 0), index=r14.index, columns=r14.columns)
    return armed, up_df


def exit_armed_mask(r14, sma):
    """Persistent EXIT arm: cross DOWN through SMA arms; stays armed until it crosses back up."""
    r = r14.to_numpy(); s = sma.to_numpy(); idx = np.arange(len(r)).reshape(-1, 1)
    dn = np.vstack([np.full((1, r.shape[1]), False), r[:-1] >= s[:-1]]) & (r < s)
    disarm = (r >= s)
    lxc = _cummax_col(np.where(dn, idx, -1)); lxd = _cummax_col(np.where(disarm, idx, -1))
    return pd.DataFrame((lxc > lxd) & (lxc >= 0), index=r14.index, columns=r14.columns)


def rel_sortino(ret, bench_ret, window):
    excess = ret.sub(bench_ret, axis=0)
    mean_ex = excess.rolling(window).mean()
    dd = np.sqrt((excess.clip(upper=0.0) ** 2).rolling(window).mean())
    with np.errstate(divide="ignore", invalid="ignore"):
        s = mean_ex / dd.replace(0.0, np.nan)
    low = dd < 1e-10
    return s.mask(low & (mean_ex.abs() >= 1e-10), np.sign(mean_ex) * 9.99).mask(low & (mean_ex.abs() < 1e-10), np.nan)


def run_portfolio(events, scl, days, del_sector, bankrupt_tk, flm, weighted, exit_np):
    """events: list of (row, col, ticker, rsi14_entry). weighted=False -> equal; True -> weight=(GATE-rsi14).
    EXIT = user's rule: hold until the first bar (>= entry) where exit_np fires (Sortino-RSI crossed DOWN its SMA
    within ARM_WIN bars AND price red). MAXHOLD is only a safety cap if the exit signal never fires."""
    n = len(days); dws = np.zeros(n); dwt = np.zeros(n); dcnt = np.zeros(n); open_until = {}; trades = []
    for (r, c, t, r14e) in events:
        di = r + 1
        if di >= n or di < open_until.get(t, -1):
            continue
        sub = exit_np[di:, c]                                         # exit-signal for this name from entry onward
        exit_row = di + int(np.argmax(sub)) if sub.any() else min(di + MAXHOLD, n - 1)
        sc = scl[t]; seg = sc.iloc[di:exit_row + 1].dropna()
        if len(seg) < 2:
            continue
        w = max(1.0, GATE - r14e) if weighted else 1.0
        for d, rr in seg.pct_change().dropna().items():
            j = days.get_loc(d); dws[j] += w * rr; dwt[j] += w; dcnt[j] += 1
        entry_p = float(seg.iloc[0]); exit_p = float(seg.iloc[-1]); hold_ret = exit_p / entry_p - 1.0
        exit_di = days.get_loc(seg.index[-1])
        ended_early = seg.index[-1] < days[-1] and not bool(sub.any())  # data ran out before an exit signal
        is_bank = (t in bankrupt_tk) and ended_early and (sc.iloc[di:].dropna().index[-1] < days[-1] - pd.Timedelta(days=10))
        if is_bank:
            hold_ret = -1.0; kj = min(exit_di + 1, n - 1); dws[kj] += -w; dwt[kj] += w; dcnt[kj] += 1
        open_until[t] = exit_di + 1
        trades.append(dict(date=str(days[di].date()), exit=str(seg.index[-1].date()), ticker=t,
                           rsi14=round(float(r14e), 1), weight=round(w, 1), hold_days=int(exit_di - di),
                           ret_pct=round(hold_ret * 100, 2), delisted=(t in del_sector), bankrupt=is_bank))
    port = np.where(dwt > 0, dws / np.where(dwt > 0, dwt, 1), 0.0)
    pr = pd.Series(port, index=days).iloc[1:]; eq = (1 + pr).cumprod()
    total = float(eq.iloc[-1] - 1) * 100; yrs = (days[-1] - days[1]).days / 365.25
    cagr = ((1 + total / 100) ** (1 / yrs) - 1) * 100
    sh = float(pr.mean() / pr.std(ddof=1) * np.sqrt(252)) if pr.std(ddof=1) > 0 else float("nan")
    dd = float(((eq - eq.cummax()) / eq.cummax()).min()) * 100
    inv = float((dcnt[1:] > 0).mean()) * 100; avg = float(dcnt[dcnt > 0].mean()) if (dcnt > 0).any() else 0.0
    corr = None
    if flm:
        mo = (1 + pr).resample("ME").prod() - 1; idx = [str(x.date()) for x in mo.index]
        fv = np.array([flm.get(d, np.nan) for d in idx]); mv = mo.values.astype(float)
        m = np.isfinite(fv) & np.isfinite(mv)
        if m.sum() > 10:
            corr = round(float(np.corrcoef(mv[m], fv[m])[0, 1]), 3)
    rets = [x["ret_pct"] for x in trades]
    return dict(total=round(total, 1), cagr=round(cagr, 1), sharpe=round(sh, 2), dd=round(dd, 1), corr_flagship=corr,
                n_trades=len(trades), win_rate=round(100 * sum(1 for x in rets if x > 0) / max(1, len(rets)), 1),
                avg_concurrent=round(avg, 1), pct_days_invested=round(inv, 0),
                mean_ret=round(float(np.mean(rets)) if rets else 0, 2), median_ret=round(float(np.median(rets)) if rets else 0, 2),
                monthly=[[str(d.date()), float(v)] for d, v in ((1 + pr).resample("ME").prod() - 1).items()]), trades


def run_concentrated(sig_np, cross_np, rank_np, exit_np, scl, days, del_sector, bankrupt_tk, flm, weighted, maxpos,
                     park_np=None, trail_pct=0.0, cost_bps=0.0):
    """Concentrated book (user 2026-09-23): hold at most `maxpos` names, always trying to stay invested. When a slot is
    free, fill it with the BUY-signal candidate that has the HIGHEST rank_np score (best entry among all stocks). Exit on
    the per-name exit rule (exit_np) or a trailing stop (trail_pct drawdown from entry-high). Empty slots earn park_np
    (QQQ) instead of cash when provided. weighted=True -> size the held names by depth (GATE-cross_rsi)."""
    n = len(days); cols = list(scl.columns)
    px = scl.ffill().to_numpy(); valid = scl.notna().to_numpy()
    last_valid = np.where(valid.any(axis=0), n - 1 - np.argmax(valid[::-1], axis=0), -1)   # last real-data row per name
    daily = np.zeros(n); dcnt = np.zeros(n); openpos = {}; trades = []
    for t in range(1, n):
        active = [c for c in openpos if openpos[c]["ei"] < t and t <= last_valid[c]]
        park_r = float(park_np[t]) if park_np is not None else 0.0                          # idle-slot return (QQQ or 0=cash)
        if active:
            ws = [max(1.0, GATE - openpos[c]["cr"]) if weighted else 1.0 for c in active]
            stock_r = 0.0
            for c, w in zip(active, ws):
                bank_day = (t == last_valid[c]) and (cols[c] in bankrupt_tk) and (last_valid[c] < n - 3)
                rc = -1.0 if bank_day else (px[t, c] / px[t - 1, c] - 1.0 if px[t - 1, c] > 0 else 0.0)
                stock_r += w * rc
            stock_r = stock_r / sum(ws) if sum(ws) else 0.0                                 # weighted avg return of active names
            na = len(active); daily[t] = (na * stock_r + (maxpos - na) * park_r) / maxpos   # empty slots -> park_r
            dcnt[t] = na
        else:
            daily[t] = park_r                                                              # fully idle -> park (QQQ) or cash
        for c in list(openpos):                                                            # exits: rule fires, trailing stop, or delisted
            ei = openpos[c]["ei"]
            if ei < t:
                openpos[c]["hi"] = max(openpos[c].get("hi", openpos[c]["ep"]), px[t, c])
                trail_hit = trail_pct > 0 and px[t, c] <= openpos[c]["hi"] * (1.0 - trail_pct)
            else:
                trail_hit = False
            if ei < t and (bool(exit_np[t, c]) or trail_hit or t >= last_valid[c]):
                xr = min(t, last_valid[c]); ret = px[xr, c] / openpos[c]["ep"] - 1.0
                daily[xr] -= (cost_bps / 1e4) / maxpos                                      # exit transaction cost (per slot)
                is_bank = (t >= last_valid[c]) and not bool(exit_np[t, c]) and (cols[c] in bankrupt_tk) and (last_valid[c] < n - 3)
                if is_bank:
                    ret = -1.0
                trades.append(dict(ticker=cols[c], date=str(days[ei].date()), exit=str(days[min(t, last_valid[c])].date()),
                                   cross_rsi=round(openpos[c]["cr"], 1), weight=round(max(1.0, GATE - openpos[c]["cr"]) if weighted else 1.0, 1),
                                   hold_days=int(min(t, last_valid[c]) - ei), ret_pct=round(ret * 100, 2),
                                   delisted=bool(cols[c] in del_sector), bankrupt=bool(is_bank)))
                del openpos[c]
        if len(openpos) < maxpos:                                                          # fill free slots: BEST entry first (rank_np desc)
            row = np.where(sig_np[t])[0]
            cands = sorted(((rank_np[t, c], c) for c in row if c not in openpos
                            and np.isfinite(rank_np[t, c]) and np.isfinite(cross_np[t, c]) and t < last_valid[c]),
                           key=lambda x: -x[0])
            for sc, c in cands:
                if len(openpos) >= maxpos:
                    break
                openpos[c] = dict(ei=t, cr=float(cross_np[t, c]), ep=px[t, c])
                daily[t] -= (cost_bps / 1e4) / maxpos                                       # entry transaction cost (per slot)
    pr = pd.Series(daily, index=days).iloc[1:]; eq = (1 + pr).cumprod()
    total = float(eq.iloc[-1] - 1) * 100; yrs = (days[-1] - days[1]).days / 365.25
    cagr = ((1 + total / 100) ** (1 / yrs) - 1) * 100 if total > -100 else -100.0
    sh = float(pr.mean() / pr.std(ddof=1) * np.sqrt(252)) if pr.std(ddof=1) > 0 else float("nan")
    dd = float(((eq - eq.cummax()) / eq.cummax()).min()) * 100
    inv = float((dcnt[1:] > 0).mean()) * 100; avg = float(dcnt[dcnt > 0].mean()) if (dcnt > 0).any() else 0.0
    corr = None
    if flm:
        mo = (1 + pr).resample("ME").prod() - 1; idx = [str(x.date()) for x in mo.index]
        fv = np.array([flm.get(d, np.nan) for d in idx]); mv = mo.values.astype(float)
        m = np.isfinite(fv) & np.isfinite(mv)
        if m.sum() > 10:
            corr = round(float(np.corrcoef(mv[m], fv[m])[0, 1]), 3)
    rets = [x["ret_pct"] for x in trades]
    return dict(total=round(total, 1), cagr=round(cagr, 1), sharpe=round(sh, 2), dd=round(dd, 1), corr_flagship=corr,
                n_trades=len(trades), win_rate=round(100 * sum(1 for x in rets if x > 0) / max(1, len(rets)), 1),
                avg_concurrent=round(avg, 1), pct_days_invested=round(inv, 0),
                mean_ret=round(float(np.mean(rets)) if rets else 0, 2), median_ret=round(float(np.median(rets)) if rets else 0, 2),
                monthly=[[str(d.date()), float(v)] for d, v in ((1 + pr).resample("ME").prod() - 1).items()]), trades


TECH_ONLY = int(os.environ.get("TECH_ONLY", 1))                       # restrict to tech (XLK/XLC + tech thematics), like META
TECH_ETFS = set((os.environ.get("TECH_ETFS") or
                 "XLK,XLC,SOXX,SMH,IGV,SKYY,CIBR,HACK,FINX,BOTZ,ARKK,IBUY,UFO,QTUM,AIQ,WCLD,IPAY").split(","))


def main():
    b = pickle.load(open(PANELS, "rb"))
    universe = list(b["common"]); del_sector = b["delisted_sector"]; surv_sector = b["surv_sector"]; bankrupt_tk = set(b["bankrupt_tk"])
    mcap_m = b.get("mktcap_usd")                                      # monthly PIT market cap panel (for top-N filter)
    if TECH_ONLY:
        sec_of = {t: (surv_sector.get(t) or del_sector.get(t)) for t in universe}
        universe = [t for t in universe if sec_of.get(t) in TECH_ETFS]
        print(f"TECH universe: {len(universe)} names in {sorted(TECH_ETFS & set(sec_of.values()))}", flush=True)
    cd = load_candles(universe + [BENCH])
    scl = pd.DataFrame({t: cd[t]["Close"] for t in universe if t in cd and "Close" in cd[t]}).sort_index()
    vol = pd.DataFrame({t: cd[t]["Volume"] for t in universe if t in cd and "Volume" in cd[t]}).reindex(index=scl.index, columns=scl.columns)
    op = pd.DataFrame({t: cd[t]["Open"] for t in universe if t in cd and "Open" in cd[t]}).reindex(index=scl.index, columns=scl.columns)
    hp = pd.DataFrame({t: cd[t]["High"] for t in universe if t in cd and "High" in cd[t]}).reindex(index=scl.index, columns=scl.columns)
    lp = pd.DataFrame({t: cd[t]["Low"] for t in universe if t in cd and "Low" in cd[t]}).reindex(index=scl.index, columns=scl.columns)
    scl = scl[scl > 0]; days = scl.index
    ohlc4 = ((op + hp + lp + scl) / 4.0).reindex(index=scl.index, columns=scl.columns)   # TradingView color-RSI source
    ret = scl.pct_change()

    # ADRIDEM "Rolling Sortino Ratio with Ref Ticker" (user's TradingView indicator, exact replica): each ticker's
    # OWN return (NO SPY subtraction — the "Ref Ticker" is only a comparison line); downside = POPULATION stdev of
    # down-clipped returns; RSI is taken on the SMOOTHED sortino.
    rsi14, sma14, sortino_s = adridem_rsi_sma(scl)                     # DAILY ADRIDEM Sortino-RSI + its SMA + smoothed Sortino
    rsi20p = wilder_rsi(ohlc4 if SRC == "ohlc4" else scl, RSI20)       # color-RSI source (ohlc4 matches TradingView; verified 48.1 BTC 2025-02-20)
    dvol = (scl * vol).rolling(63, min_periods=20).mean()
    # ENTRY arm (persistent, no day limit; resets on down-cross or gate-reach). Depth sizing = RSI14 at the cross.
    # DEEP-OVERSOLD precondition (user 2026-09-23): Sortino-RSI must have spent >= OVERSOLD_MIN days under OVERSOLD_THR
    # in the trailing OVERSOLD_WIN window; AND the arming up-cross must itself occur while RSI < CROSS_THR; AND the first
    # GREEN bar may fire up to ARM_WIN days after that cross.
    oversold_ok = None
    if OVERSOLD_MIN > 0:
        below = (rsi14 < OVERSOLD_THR).astype(float)
        days_ok = below.rolling(OVERSOLD_WIN, min_periods=OVERSOLD_MIN).sum() >= OVERSOLD_MIN
        oversold_ok = days_ok
        if OVERSOLD_DEEP > 0:                                          # AND it must have dipped below OVERSOLD_DEEP (capitulation) in that window
            oversold_ok = days_ok & (rsi14.rolling(OVERSOLD_WIN, min_periods=1).min() < OVERSOLD_DEEP)
    armed, deep_cross = entry_armed_mask(rsi14, sma14, precond=oversold_ok, cross_thr=CROSS_THR, arm_win=ARM_WIN)
    cross_r14 = rsi14.where(deep_cross).ffill()
    green = rsi20p > GREEN_THR
    # SELECTION rank among all candidates that fire together (higher = better entry):
    #   confirm     = greenest turn now (color-RSI on ohlc4) -> catch the bounce once it's confirmed
    #   confirm_srt = greenest + how hard the Sortino-RSI is turning up above its SMA
    #   depth       = deepest dip (lowest Sortino-RSI at the cross) -> most oversold (falling-knife risk)
    if RANK == "depth":
        rank = -cross_r14
    elif RANK == "confirm_srt":
        rank = rsi20p + (rsi14 - sma14)
    else:
        rank = rsi20p
    top_mask = pd.DataFrame(True, index=scl.index, columns=scl.columns)
    if TOP_MCAP and mcap_m is not None:                                                   # PIT top-N by market cap
        tm = mcap_m.sort_index().reindex(columns=scl.columns)
        top = tm.rank(axis=1, ascending=False) <= TOP_MCAP
        top_mask = top.reindex(scl.index, method="ffill").fillna(False)
    # WEEKLY confirmation (user 2026-09-23): the SAME entry arm must also hold on the weekly timeframe.
    weekly_armed = pd.DataFrame(True, index=scl.index, columns=scl.columns)
    if WEEKLY:
        wk = scl.resample("W-FRI").last()
        w_r14, w_sma, _ = adridem_rsi_sma(wk)
        w_armed, _ = entry_armed_mask(w_r14, w_sma)
        weekly_armed = w_armed.reindex(scl.index, method="ffill").fillna(False)
    sortino_pos = (sortino_s > 0) if SORTINO_POS else pd.DataFrame(True, index=scl.index, columns=scl.columns)
    sig = (armed & green & (dvol >= MINVOL) & top_mask & weekly_armed & sortino_pos).fillna(False)
    # EXIT rule (B1 variants):
    if EXIT == "weekly_break":                                         # trend-trailing: HOLD while the WEEKLY Sortino-RSI is above its SMA
        wke = scl.resample("W-FRI").last(); we_r14, we_sma, _ = adridem_rsi_sma(wke)
        weekly_up = (we_r14 > we_sma).reindex(scl.index, method="ffill").fillna(False)
        exit_sig = (~weekly_up)                                        # exit only when the weekly momentum rolls over
    elif EXIT == "trail":                                             # pure price trailing stop, applied in the runner (TRAIL_PCT)
        exit_sig = pd.DataFrame(False, index=scl.index, columns=scl.columns)
    elif EXIT == "red":                                              # hold until the color turns RED (user 2026-09-23) — no down-cross needed
        exit_sig = (rsi20p < RED_THR).fillna(False)
    else:                                                             # downcross_red: DOWN-cross of RSI14 arms exit; RED bar fires it
        exit_armed = exit_armed_mask(rsi14, sma14); red = rsi20p < RED_THR
        exit_sig = (exit_armed & red).fillna(False)
    if SIMPLE:                                                        # CROSSOVER: buy on up-cross of Sortino-RSI thru SMA (after >=OVERSOLD_MIN days under 30), sell on down-cross
        up = ((rsi14.shift(1) <= sma14.shift(1)) & (rsi14 > sma14)).fillna(False)
        if OVERSOLD_MIN > 0:                                           # precondition: spent >= OVERSOLD_MIN days under OVERSOLD_THR in the run-up
            days_ok = (rsi14 < OVERSOLD_THR).astype(float).rolling(OVERSOLD_WIN, min_periods=OVERSOLD_MIN).sum() >= OVERSOLD_MIN
            up = (up & days_ok.fillna(False))
        if TREND_MA > 0:                                               # trend filter: only buy when price is above its TREND_MA-day SMA
            up = (up & (scl > scl.rolling(TREND_MA).mean())).fillna(False)
        dn = ((rsi14.shift(1) >= sma14.shift(1)) & (rsi14 < sma14)).fillna(False)
        deep_cross = up; cross_r14 = rsi14.where(up).ffill()
        rank = (-cross_r14) if RANK == "depth" else rsi20p
        sig = (up & (dvol >= MINVOL) & top_mask).fillna(False)         # keep only structural filters (liquidity + top-N universe)
        exit_sig = dn
    # floor the portfolio to the period where the universe actually has breadth (indicators keep full-history warmup)
    keep = scl.index >= pd.Timestamp(START_DATE)
    scl = scl.loc[keep]; days = scl.index; sig = sig.loc[keep]; cross_r14 = cross_r14.loc[keep]; exit_sig = exit_sig.loc[keep]; rank = rank.loc[keep]
    print(f"universe={scl.shape[1]}  days={len(days)} (from {days[0].date()})  deep-crosses={int(deep_cross.loc[keep].to_numpy().sum())}  "
          f"raw BUY signals={int(sig.to_numpy().sum())}", flush=True)

    cols = list(scl.columns)
    sig_np = sig.to_numpy(); cross_np = cross_r14.to_numpy(); rank_np = rank.to_numpy(); exit_np = exit_sig.to_numpy()
    park_np = None                                                     # idle-slot return: QQQ when PARK=qqq, else cash (0%)
    if PARK == "qqq":
        qd = load_candles(["QQQ"]).get("QQQ")
        if qd is not None and "Close" in qd:
            park_np = qd["Close"].pct_change().reindex(days).fillna(0.0).to_numpy()
            print(f"PARK=qqq: idle slots earn QQQ ({int(np.isfinite(park_np).sum())} days)", flush=True)
        else:
            print("PARK=qqq requested but QQQ candles missing -> cash", flush=True)
    trail = TRAIL_PCT if EXIT == "trail" else 0.0
    flm = {}
    try:
        flm = {d: r for d, r in json.load(open("/app/.data/studies/flagship_history.json")).get("monthly_net", [])}
    except Exception:
        pass

    for lab, wt in [("EQUAL-weight", False), ("DEPTH-weight (size by 50-rsi14)", True)]:
        res, trades = run_concentrated(sig_np, cross_np, rank_np, exit_np, scl, days, del_sector, bankrupt_tk, flm, wt, MAXPOS,
                                       park_np=park_np, trail_pct=trail, cost_bps=COST_BPS)
        print(f"\n=== SORTINO-RSI [{lab}]  max {MAXPOS} names, deepest-dip first, exit=down-cross+red ===", flush=True)
        print(f"  total={res['total']:,.0f}%  CAGR={res['cagr']:.1f}  Sharpe={res['sharpe']:.2f}  maxDD={res['dd']:.1f}%  "
              f"trades={res['n_trades']}  win={res['win_rate']}%  avg_concurrent={res['avg_concurrent']}  "
              f"%days_inv={res['pct_days_invested']:.0f}  CORR_flag={res['corr_flagship']}", flush=True)
        print(f"  per-trade: mean {res['mean_ret']:+.2f}%  median {res['median_ret']:+.2f}%", flush=True)
        if not wt:                                                      # persist EQUAL-weight (beats depth within the max-N book)
            summ = dict(strategy=f"sortinorsitechnique ADRIDEM: top{TOP_MCAP} mcap, max{MAXPOS} deepest-dip, weekly-confirm, exit=down-cross+red",
                        win=WIN, hold_d=HOLD_D, top_mcap=TOP_MCAP, maxpos=MAXPOS, weekly=WEEKLY,
                        **{k: res[k] for k in ("total", "cagr", "sharpe", "dd", "n_trades", "win_rate", "avg_concurrent", "pct_days_invested", "mean_ret", "median_ret")})
            summ["corr_flagship"] = res["corr_flagship"]
            json.dump(dict(summary=summ, trades=trades), open("/app/.data/studies/sortino_rsi_trades.json", "w"), indent=0)
            if SCENARIO:
                os.makedirs("/app/.data/studies/scenarios", exist_ok=True)
                json.dump(dict(name=SCENARIO, summary=summ, monthly=res["monthly"]),
                          open(f"/app/.data/studies/scenarios/{SCENARIO}.json", "w"))
                print(f"  scenario saved: {SCENARIO}", flush=True)
            try:
                from core.models import BacktestResult as BR
                from django.utils import timezone as tz
                BR.objects.update_or_create(kind="sortinorsitechnique", defaults=dict(computed_at=tz.now(), payload=dict(summ, monthly=res["monthly"])))
                print("  saved BacktestResult[sortinorsitechnique] + sortino_rsi_trades.json", flush=True)
            except Exception as e:
                print("  BR save skipped:", e, flush=True)


if __name__ == "__main__":
    main()
