"""Shared FRESH helpers: daily fresh (as in trend_analyzer) + a full weekly fresh."""
import warnings; warnings.filterwarnings("ignore")
import pandas as pd
import indicators

FRESH_WINDOW_D = 14   # trading days for daily fresh
FRESH_WINDOW_W = 8    # weeks for weekly fresh


def _state_from(conds: dict, rsi_days, rs_days, index, window):
    """Given the fresh conditions, return (state, days_ago, since_date, missing).
    FRESH = all conditions met; POTENTIAL = exactly one short. (MACD removed.)"""
    total = len(conds)
    n = sum(conds.values())
    since = None
    if n == total:
        days = max(rsi_days, rs_days)
        state = "FRESH" if days <= window else None
        if state:
            since = str(index[-1 - days].date())
        else:
            days = None
    elif n == total - 1:
        state, days = "POTENTIAL", None
    else:
        state, days = None, None
    missing = [k for k, v in conds.items() if not v]
    return state, days, since, missing


def daily_fresh(df):
    """Weekly-Sortino gate + daily RSI / RSI-of-Sortino crossovers. (MACD removed.)"""
    fd = indicators.compute_fresh_crossovers(df, window=FRESH_WINDOW_D, sortino_timeframe="W")
    f_rsi, f_rs = fd.get("fresh_rsi_x_days"), fd.get("fresh_rs_x_days")
    conds = {
        "wk_sortino>0": bool(fd.get("fresh_sortino_pos")),
        "rsi_x": f_rsi is not None,
        "rsisort_x": f_rs is not None,
    }
    return _state_from(conds, f_rsi, f_rs, df.index, FRESH_WINDOW_D)


def sector_signal_nomacd(r, window=FRESH_WINDOW_D):
    """Recompute a sector's signal + fresh_state WITHOUT MACD, from a _signals.json row.
    BULLISH = RSI>SMA AND Omega>1. FRESH = wk_sortino>0 + RSI x + RSI-of-Sortino x (3/3)."""
    if not r:
        return None, None
    rsi_above = bool(r.get("rsi_above_sma"))
    omega = r.get("omega") or 0
    cross = bool(r.get("rsi_crossover"))
    if rsi_above and omega > 1:
        sig = "ROTATE IN" if cross else "BULLISH"
    elif rsi_above:
        sig = "RSI ONLY"
    elif omega > 1:
        sig = "OMEGA ONLY"
    else:
        sig = "BEARISH"
    rx, rsx, wks = r.get("fresh_rsi_x_days"), r.get("fresh_rs_x_days"), r.get("fresh_wk_sortino")
    conds = [(wks is not None and wks > 0), rx is not None, rsx is not None]
    n = sum(conds)
    if n == 3:
        days = max(rx, rsx)
        fstate = "FRESH" if days <= window else None
    elif n == 2:
        fstate = "POTENTIAL"
    else:
        fstate = None
    return sig, fstate


def _to_weekly(df):
    """Resample daily OHLC to completed weekly (W-FRI) bars (drop in-progress week)."""
    agg = {}
    for col, how in (("Open", "first"), ("High", "max"), ("Low", "min"),
                     ("Close", "last"), ("Volume", "sum")):
        if col in df.columns:
            agg[col] = getattr(df[col].resample("W-FRI"), how)()
    wk = pd.DataFrame(agg).dropna(subset=["Close"])
    if len(df.index) and df.index[-1].dayofweek != 4:
        wk = wk.iloc[:-1]  # drop the still-forming week (no repaint)
    return wk


def weekly_fresh(df):
    """Same composite but computed entirely on WEEKLY bars (all crossovers weekly)."""
    wk = _to_weekly(df)
    if len(wk) < 40:   # need enough weekly bars for RSI(10)/Sortino(10)
        return None, None, None, ["insufficient_weekly_history"]
    # sortino_timeframe="D" => uses the passed (weekly) bars directly, no re-resample
    fd = indicators.compute_fresh_crossovers(wk, window=FRESH_WINDOW_W, sortino_timeframe="D")
    f_rsi, f_rs = fd.get("fresh_rsi_x_days"), fd.get("fresh_rs_x_days")
    conds = {
        "wk_sortino>0": bool(fd.get("fresh_sortino_pos")),
        "rsi_x": f_rsi is not None,
        "rsisort_x": f_rs is not None,
    }
    return _state_from(conds, f_rsi, f_rs, wk.index, FRESH_WINDOW_W)
