#!/usr/bin/env python3
"""
Stock Market Trend Bot - Studies Engine

Generates and runs trading studies across sector ETFs.
Entry signals × Exit conditions = hundreds of combinations.
Results saved to .data/studies/results.json
"""

import warnings
warnings.filterwarnings("ignore")

import json
import time
import numpy as np
import pandas as pd
import ta
from pathlib import Path
from datetime import datetime

import config
import data_fetcher
import indicators

STUDIES_DIR = Path(__file__).parent / ".data" / "studies"
STUDIES_DIR.mkdir(parents=True, exist_ok=True)


# ── Entry signal functions ──

def sig_gap_up(df, t=0.5):
    return (df["Open"] - df["High"].shift(1)) / df["High"].shift(1) * 100 >= t

def sig_gap_down(df, t=0.5):
    return (df["Open"] - df["Low"].shift(1)) / df["Low"].shift(1) * 100 <= -t

def sig_gap_up_large(df): return sig_gap_up(df, 2.0)
def sig_gap_down_large(df): return sig_gap_down(df, 2.0)
def sig_gap_up_medium(df): return sig_gap_up(df, 1.0)
def sig_gap_down_medium(df): return sig_gap_down(df, 1.0)

def sig_rsi_cross_above_sma(df):
    rsi = ta.momentum.rsi(df["Close"], window=10)
    sma = rsi.rolling(10).mean()
    return (rsi > sma) & (rsi.shift(1) <= sma.shift(1))

def sig_rsi_cross_below_sma(df):
    rsi = ta.momentum.rsi(df["Close"], window=10)
    sma = rsi.rolling(10).mean()
    return (rsi < sma) & (rsi.shift(1) >= sma.shift(1))

def sig_rsi_cross_above_sma_below50(df):
    rsi = ta.momentum.rsi(df["Close"], window=10)
    sma = rsi.rolling(10).mean()
    return (rsi > sma) & (rsi.shift(1) <= sma.shift(1)) & (rsi.shift(1) < 50)

SUPPRESS_MIN = 10  # min consecutive bars RSI stayed <50 AND <its SMA before the cross

def _suppressed_then_cross_series(close, min_streak=SUPPRESS_MIN, level=50):
    """Bool Series on `close`'s index: RSI(10) held < `level` AND < its SMA(10) for
    `min_streak`+ STRAIGHT bars, then crosses above its SMA(10) on that bar.
    The streak RESETS to 0 on any bar where RSI is >= level or >= its SMA (i.e. if RSI
    pops back above its average before the run completes, the count starts over)."""
    rsi = ta.momentum.rsi(close, window=10)
    sma = rsi.rolling(10).mean()
    cross = (rsi > sma) & (rsi.shift(1) <= sma.shift(1))
    suppressed = (rsi < level) & (rsi < sma)
    grp = (~suppressed).cumsum()                 # new group each time suppression breaks
    streak = suppressed.groupby(grp).cumsum()    # consecutive suppressed-bar count (resets on break)
    return (cross & (streak.shift(1) >= min_streak)).fillna(False).astype(bool)

def sig_rsi_suppressed_then_cross(df):
    """RSI(10) held <50 AND <its SMA(10) for SUPPRESS_MIN+ straight days, then
    crosses above its SMA(10) on that bar ('coiled spring' reversal)."""
    return _suppressed_then_cross_series(df["Close"])

def sig_rsi_suppressed_then_cross_ad(df):
    """rsi_sup10_x (RSI<50 streak -> cross) AND A/D line above its 10-bar SMA
    (accumulation trending up). NB: uses `_ad_trend_up`, not the divergence `_ad_rising`
    which is incompatible with the cross-up bar."""
    return (_suppressed_then_cross_series(df["Close"], level=50) & _ad_trend_up(df)).fillna(False).astype(bool)

def sig_rsi_suppressed_then_cross_vol(df):
    """rsi_sup10_x AND volume above its 20-bar average (conviction on the turn)."""
    vol_ok = df["Volume"] > df["Volume"].rolling(20).mean()
    return (_suppressed_then_cross_series(df["Close"], level=50) & vol_ok).fillna(False).astype(bool)

def sig_rsi_suppressed_then_cross_hl(df):
    """rsi_sup10_x AND a higher-low formed within the last 10 bars (price structure
    confirming the base is holding)."""
    hl = _higher_low(df).reindex(df.index).fillna(False).rolling(10, min_periods=1).max().astype(bool)
    return (_suppressed_then_cross_series(df["Close"], level=50) & hl).fillna(False).astype(bool)

def sig_rsi_suppressed_then_cross_mkt(df, spy_close=None):
    """rsi_sup10_x AND the broad market (SPY) turning up: SPY RSI(10) crossed above its
    SMA(10) within the last MKT_TURN_WINDOW days. Needs spy_close injected."""
    base = _suppressed_then_cross_series(df["Close"], level=50)
    if spy_close is None:
        return base
    spy_recent = (_rsi_cross_series(spy_close).reindex(df.index).fillna(False)
                  .rolling(MKT_TURN_WINDOW, min_periods=1).max().astype(bool))
    return (base & spy_recent).fillna(False).astype(bool)

def sig_rsi_suppressed_lt20_then_cross(df):
    """RSI(10) held <20 AND <its SMA(10) for SUPPRESS_MIN+ STRAIGHT days (streak resets
    if RSI pops back above its average), then crosses above its SMA(10) on that bar.
    Deeply-oversold 'coiled spring' variant of `rsi_sup10_x`."""
    return _suppressed_then_cross_series(df["Close"], level=20)

def sig_rsi_suppressed_then_cross_negsort(df):
    """Suppressed-then-cross AND Sortino(10) < 0 (deep-capitulation reversal:
    downside-risk-adjusted momentum still negative when RSI turns up)."""
    base = _suppressed_then_cross_series(df["Close"])
    return (base & (_rolling_sortino(df) < 0)).fillna(False).astype(bool)

def _suppressed_dd(df, dd_thresh):
    """Suppressed-then-cross AND >= `dd_thresh` (fraction) below the running ATH."""
    base = _suppressed_then_cross_series(df["Close"])
    dd = indicators.drawdown_from_high(df)
    return (base & (dd <= -dd_thresh)).fillna(False).astype(bool)

def sig_rsi_suppressed_then_cross_dd40(df):
    """Suppressed-then-cross AND >=40% below the running ATH."""
    return _suppressed_dd(df, 0.40)

def sig_rsi_suppressed_then_cross_dd50(df):
    """Suppressed-then-cross AND >=50% below the running ATH (deep drawdown)."""
    return _suppressed_dd(df, 0.50)

def sig_rsi_suppressed_then_cross_dd60(df):
    """Suppressed-then-cross AND >=60% below the running ATH."""
    return _suppressed_dd(df, 0.60)

def sig_rsi_suppressed_then_cross_dd70(df):
    """Suppressed-then-cross AND >=70% below the running ATH."""
    return _suppressed_dd(df, 0.70)

def _ad_trend_up(df, window=10):
    """A/D (Accumulation/Distribution) line above its `window`-bar SMA = accumulation
    trend rising (buyers stepping in). NB: distinct from the existing `_ad_rising`,
    which is a stricter price-flat/down divergence detector."""
    ad = ta.volume.AccDistIndexIndicator(
        df["High"], df["Low"], df["Close"], df["Volume"]).acc_dist_index()
    return ad > ad.rolling(window).mean()

def sig_rsi_suppressed_dd50_adrise(df):
    """dd50 deep-drawdown reversal AND A/D line in an uptrend (accumulation confirming)."""
    return (_suppressed_dd(df, 0.50) & _ad_trend_up(df)).fillna(False).astype(bool)

def sig_rsi_suppressed_dd50_vol(df):
    """dd50 deep-drawdown reversal AND volume above its 20-bar average (conviction)."""
    vol_ok = df["Volume"] > df["Volume"].rolling(20).mean()
    return (_suppressed_dd(df, 0.50) & vol_ok).fillna(False).astype(bool)

MKT_TURN_WINDOW = 5  # SPY RSI must have crossed up within this many days of the dip-buy

def sig_rsi_suppressed_dd50_mkt(df, spy_close=None):
    """dd50 deep-drawdown reversal AND the broad market (SPY) is turning up:
    SPY RSI(10) crossed above its SMA(10) within the last MKT_TURN_WINDOW days.
    Needs spy_close injected (like `rsi_x_pos_updn`); without it, falls back to plain dd50."""
    dd50 = _suppressed_dd(df, 0.50)
    if spy_close is None:
        return dd50
    spy_recent = (_rsi_cross_series(spy_close).reindex(df.index).fillna(False)
                  .rolling(MKT_TURN_WINDOW, min_periods=1).max().astype(bool))
    return (dd50 & spy_recent).fillna(False).astype(bool)

def sig_rsi_suppressed_then_cross_weekly(df):
    """Weekly (W-FRI) suppressed-then-cross, mapped to a single daily entry on the
    last trading day of the crossing week. SUPPRESS_MIN is in WEEKLY bars (~weeks)."""
    close = df["Close"]
    out = pd.Series(False, index=close.index)
    wk = close.resample("W-FRI").last().dropna()
    if len(wk) < 15 + SUPPRESS_MIN:
        return out
    wk_fire = _suppressed_then_cross_series(wk)
    for wend in wk_fire[wk_fire].index:
        mask = (close.index > wend - pd.Timedelta(days=7)) & (close.index <= wend)
        if mask.any():
            out.loc[close.index[mask][-1]] = True
    return out

def sig_rsi_oversold(df):
    rsi = ta.momentum.rsi(df["Close"], window=10)
    return (rsi < 30) & (rsi.shift(1) >= 30)

def sig_rsi_oversold_20(df):
    rsi = ta.momentum.rsi(df["Close"], window=10)
    return (rsi < 20) & (rsi.shift(1) >= 20)

def sig_rsi_overbought(df):
    rsi = ta.momentum.rsi(df["Close"], window=10)
    return (rsi > 70) & (rsi.shift(1) <= 70)

def sig_rsi_cross_50_up(df):
    rsi = ta.momentum.rsi(df["Close"], window=10)
    return (rsi > 50) & (rsi.shift(1) <= 50)

def sig_rsi_cross_50_down(df):
    rsi = ta.momentum.rsi(df["Close"], window=10)
    return (rsi < 50) & (rsi.shift(1) >= 50)

def sig_price_cross_sma20_up(df):
    sma = df["Close"].rolling(20).mean()
    return (df["Close"] > sma) & (df["Close"].shift(1) <= sma.shift(1))

def sig_price_cross_sma20_down(df):
    sma = df["Close"].rolling(20).mean()
    return (df["Close"] < sma) & (df["Close"].shift(1) >= sma.shift(1))

def sig_price_cross_sma50_up(df):
    sma = df["Close"].rolling(50).mean()
    return (df["Close"] > sma) & (df["Close"].shift(1) <= sma.shift(1))

def sig_golden_cross(df):
    s20 = df["Close"].rolling(20).mean()
    s50 = df["Close"].rolling(50).mean()
    return (s20 > s50) & (s20.shift(1) <= s50.shift(1))

def sig_death_cross(df):
    s20 = df["Close"].rolling(20).mean()
    s50 = df["Close"].rolling(50).mean()
    return (s20 < s50) & (s20.shift(1) >= s50.shift(1))

def sig_vol_spike_up(df):
    avg = df["Volume"].rolling(20).mean()
    return (df["Volume"] > avg * 2) & (df["Close"] > df["Open"])

def sig_vol_spike_down(df):
    avg = df["Volume"].rolling(20).mean()
    return (df["Volume"] > avg * 2) & (df["Close"] < df["Open"])

def sig_3down(df):
    return (df["Close"] < df["Close"].shift(1)) & (df["Close"].shift(1) < df["Close"].shift(2)) & (df["Close"].shift(2) < df["Close"].shift(3))

def sig_3up(df):
    return (df["Close"] > df["Close"].shift(1)) & (df["Close"].shift(1) > df["Close"].shift(2)) & (df["Close"].shift(2) > df["Close"].shift(3))

def sig_5down(df):
    return sum(df["Close"].shift(i) < df["Close"].shift(i+1) for i in range(5)) == 5

def sig_new_20high(df):
    h = df["High"].rolling(20).max()
    return (df["High"] >= h) & (df["High"].shift(1) < h.shift(1))

def sig_new_20low(df):
    l = df["Low"].rolling(20).min()
    return (df["Low"] <= l) & (df["Low"].shift(1) > l.shift(1))

def sig_new_52high(df):
    h = df["High"].rolling(min(252, len(df))).max()
    return (df["High"] >= h) & (df["High"].shift(1) < h.shift(1))

def sig_new_52low(df):
    l = df["Low"].rolling(min(252, len(df))).min()
    return (df["Low"] <= l) & (df["Low"].shift(1) > l.shift(1))

def sig_bullish_engulf(df):
    return (df["Close"].shift(1) < df["Open"].shift(1)) & (df["Close"] > df["Open"]) & (df["Close"] > df["Open"].shift(1)) & (df["Open"] < df["Close"].shift(1))

def sig_bearish_engulf(df):
    return (df["Close"].shift(1) > df["Open"].shift(1)) & (df["Close"] < df["Open"]) & (df["Close"] < df["Open"].shift(1)) & (df["Open"] > df["Close"].shift(1))

def sig_macd_cross_up(df):
    m = ta.trend.MACD(df["Close"])
    return (m.macd() > m.macd_signal()) & (m.macd().shift(1) <= m.macd_signal().shift(1))

def sig_macd_cross_down(df):
    m = ta.trend.MACD(df["Close"])
    return (m.macd() < m.macd_signal()) & (m.macd().shift(1) >= m.macd_signal().shift(1))

def _macd_great(df):
    """MACD daily 'great' state: MACD line above signal line AND above zero (bullish)."""
    m = ta.trend.MACD(df["Close"])
    macd, sig = m.macd(), m.macd_signal()
    return (macd > sig) & (macd > 0)

def sig_macd_great(df):
    """Day the MACD enters its 'great' state (MACD > signal AND MACD > 0)."""
    great = _macd_great(df)
    return great & (~great.shift(1).fillna(False))


DD_THRESHOLD = 0.30  # min drawdown from running ATH to qualify (0.30 = 30% below high)


def sig_dd30_rsi_reversal(df):
    """>=30% below running ATH AND RSI(10) crosses above its SMA(10) on that bar."""
    dd = indicators.drawdown_from_high(df)
    rsi = ta.momentum.rsi(df["Close"], window=10)
    sma = rsi.rolling(10).mean()
    cross = (rsi > sma) & (rsi.shift(1) <= sma.shift(1))
    return ((dd <= -DD_THRESHOLD) & cross).fillna(False).astype(bool)

# ── Market (SPY/QQQ) signal constants & helpers ──
CB_WINDOW = 60          # rolling window (trading days) for correlation & beta
HI_CORR = 0.7           # "high correlation" threshold
HI_BETA = 1.0           # "high beta" threshold
BOTH_WINDOW_D = 3       # daily bars: SPY & QQQ must each have crossed within this window
BOTH_WINDOW_W = 2       # weekly bars: same, for weekly variants

# Entry-quality / max-adverse-excursion (MAE). A trade's MAE is the worst intraday drawdown
# from its entry price over the hold (min of Low/entry-1, in %, <= 0). A trade is "clean" if
# its MAE never breached this threshold — it barely dipped before working out. Shared by every
# backtest locus (this module, all_on_all_study, tasks._run_one, tasks.compute_sector_drilldown)
# so the AvgDip / Clean% columns mean the same thing everywhere.
CLEAN_MAE_THRESH = -2.0   # % — dip shallower than this counts as a clean entry

# Independence gap for significance stats. Level-condition signals stay True for runs of
# consecutive bars, so each bar opens a near-duplicate trade sharing the same forward window;
# pooling them as i.i.d. inflates the apparent sample and overstates significance. We collapse
# fires within EFFECTIVE_GAP trading bars of the previous accepted fire (per ticker/sector) into
# ONE independent observation before computing the t-stat. A trading week is a coarse but
# defensible proxy for "distinct setup" (not a full overlap/Newey-West correction).
EFFECTIVE_GAP = 5


def _tstat_from_returns(returns):
    """One-sample t-stat of a return list vs 0: mean / (std_ddof1 / sqrt(n)). Returns None if
    fewer than 3 observations or zero dispersion. |t|>=2 ≈ a real edge; small |t| = noise."""
    n = len(returns)
    if n < 3:
        return None
    arr = np.asarray(returns, dtype=float)
    arr = arr[np.isfinite(arr)]
    if len(arr) < 3:
        return None
    sd = arr.std(ddof=1)
    if not (sd > 0):
        return None
    return round(float(arr.mean() / (sd / np.sqrt(len(arr)))), 2)


def _episode_starts(entry_idxs, gap=EFFECTIVE_GAP):
    """Given ascending entry bar positions, return the subset that start a new independent
    'episode' — the first fire, then any fire >= `gap` bars after the previous accepted one.
    Collapses runs of consecutive/near-consecutive fires into single observations."""
    starts = set()
    last = -10 ** 9
    for i in entry_idxs:
        if i - last >= gap:
            starts.add(i)
            last = i
    return starts


def trade_mae(entry_price, low_slice):
    """Worst intraday adverse excursion (%) from entry over the hold, vectorized.

    low_slice is the ndarray of Lows for the bars AFTER entry up to and including exit.
    Returns 0.0 when there are no post-entry bars (nothing to draw down through)."""
    if entry_price <= 0 or low_slice.size == 0:
        return 0.0
    return float((low_slice / entry_price - 1.0).min() * 100.0)


def _rsi_cross_series(close):
    """Bool Series: RSI(10) crosses above its SMA(10) on that bar."""
    rsi = ta.momentum.rsi(close, window=10)
    sma = rsi.rolling(10).mean()
    return ((rsi > sma) & (rsi.shift(1) <= sma.shift(1))).fillna(False).astype(bool)


def _rsi_cross_series_weekly(close):
    """Bool Series (daily index of `close`): weekly RSI(10) cross mapped to a single
    daily entry on the last trading day of the crossing week."""
    out = pd.Series(False, index=close.index)
    wk = close.resample("W-FRI").last().dropna()
    if len(wk) < 15:
        return out
    rsi = ta.momentum.rsi(wk, window=10)
    sma = rsi.rolling(10).mean()
    wk_cross = (rsi > sma) & (rsi.shift(1) <= sma.shift(1))
    for wend in wk_cross[wk_cross.fillna(False)].index:
        mask = (close.index > wend - pd.Timedelta(days=7)) & (close.index <= wend)
        if mask.any():
            out.loc[close.index[mask][-1]] = True
    return out


def _rolling_corr(close, ref_close, w=CB_WINDOW):
    """Rolling Pearson correlation of daily returns of `close` vs `ref_close`."""
    a = close.pct_change()
    b = ref_close.pct_change().reindex(a.index)
    return a.rolling(w).corr(b)


def _rolling_beta_series(close, ref_close, w=CB_WINDOW):
    """Rolling beta = cov(asset, ref) / var(ref) on daily returns."""
    a = close.pct_change()
    b = ref_close.pct_change().reindex(a.index)
    cov = a.rolling(w).cov(b)
    var = b.rolling(w).var()
    return (cov / var).replace([np.inf, -np.inf], np.nan)


def _min_corr_spyqqq(close, spy_close, qqq_close):
    """Elementwise min of corr-to-SPY and corr-to-QQQ (reindexed to close)."""
    cs = _rolling_corr(close, spy_close).reindex(close.index)
    cq = _rolling_corr(close, qqq_close).reindex(close.index)
    return pd.concat([cs, cq], axis=1).min(axis=1)


def _min_beta_spyqqq(close, spy_close, qqq_close):
    """Elementwise min of beta-to-SPY and beta-to-QQQ (reindexed to close)."""
    bs = _rolling_beta_series(close, spy_close).reindex(close.index)
    bq = _rolling_beta_series(close, qqq_close).reindex(close.index)
    return pd.concat([bs, bq], axis=1).min(axis=1)


# ── Market (SPY/QQQ) entry signals. All accept (df, spy_close, qqq_close). ──
def _empty(df):
    return pd.Series(False, index=df.index)


def sig_spy_rsi_x(df, spy_close=None, qqq_close=None):
    if spy_close is None:
        return _empty(df)
    return _rsi_cross_series(spy_close).reindex(df.index).fillna(False).astype(bool)


def sig_qqq_rsi_x(df, spy_close=None, qqq_close=None):
    if qqq_close is None:
        return _empty(df)
    return _rsi_cross_series(qqq_close).reindex(df.index).fillna(False).astype(bool)


def _both_recent(a_cross, b_cross, index, window):
    """True where BOTH cross series have fired within the last `window` bars."""
    a = a_cross.reindex(index).fillna(False).rolling(window, min_periods=1).max().astype(bool)
    b = b_cross.reindex(index).fillna(False).rolling(window, min_periods=1).max().astype(bool)
    fresh = (a_cross.reindex(index).fillna(False)) | (b_cross.reindex(index).fillna(False))
    return (a & b & fresh).astype(bool)


def sig_spy_qqq_rsi_x_both(df, spy_close=None, qqq_close=None):
    if spy_close is None or qqq_close is None:
        return _empty(df)
    return _both_recent(_rsi_cross_series(spy_close), _rsi_cross_series(qqq_close),
                        df.index, BOTH_WINDOW_D)


def sig_spy_rsi_x_wk(df, spy_close=None, qqq_close=None):
    if spy_close is None:
        return _empty(df)
    return _rsi_cross_series_weekly(spy_close).reindex(df.index).fillna(False).astype(bool)


def sig_qqq_rsi_x_wk(df, spy_close=None, qqq_close=None):
    if qqq_close is None:
        return _empty(df)
    return _rsi_cross_series_weekly(qqq_close).reindex(df.index).fillna(False).astype(bool)


def sig_spy_qqq_rsi_x_both_wk(df, spy_close=None, qqq_close=None):
    if spy_close is None or qqq_close is None:
        return _empty(df)
    return _both_recent(_rsi_cross_series_weekly(spy_close), _rsi_cross_series_weekly(qqq_close),
                        df.index, BOTH_WINDOW_W)


def sig_corr_spyqqq_x_high(df, spy_close=None, qqq_close=None):
    if spy_close is None or qqq_close is None:
        return _empty(df)
    mc = _min_corr_spyqqq(df["Close"], spy_close, qqq_close)
    return ((mc > HI_CORR) & (mc.shift(1) <= HI_CORR)).fillna(False).astype(bool)


def sig_beta_spyqqq_x_high(df, spy_close=None, qqq_close=None):
    if spy_close is None or qqq_close is None:
        return _empty(df)
    mb = _min_beta_spyqqq(df["Close"], spy_close, qqq_close)
    return ((mb > HI_BETA) & (mb.shift(1) <= HI_BETA)).fillna(False).astype(bool)


def _gate(both, df, spy_close, qqq_close, use_beta, use_corr):
    cond = both.copy()
    if use_beta:
        mb = _min_beta_spyqqq(df["Close"], spy_close, qqq_close).reindex(df.index)
        cond = cond & (mb > HI_BETA)
    if use_corr:
        mc = _min_corr_spyqqq(df["Close"], spy_close, qqq_close).reindex(df.index)
        cond = cond & (mc > HI_CORR)
    return cond.fillna(False).astype(bool)


def sig_spy_qqq_rsi_x_hibeta(df, spy_close=None, qqq_close=None):
    if spy_close is None or qqq_close is None:
        return _empty(df)
    return _gate(sig_spy_qqq_rsi_x_both(df, spy_close, qqq_close), df, spy_close, qqq_close, True, False)


def sig_spy_qqq_rsi_x_hicorr(df, spy_close=None, qqq_close=None):
    if spy_close is None or qqq_close is None:
        return _empty(df)
    return _gate(sig_spy_qqq_rsi_x_both(df, spy_close, qqq_close), df, spy_close, qqq_close, False, True)


def sig_spy_qqq_rsi_x_hibeta_hicorr(df, spy_close=None, qqq_close=None):
    if spy_close is None or qqq_close is None:
        return _empty(df)
    return _gate(sig_spy_qqq_rsi_x_both(df, spy_close, qqq_close), df, spy_close, qqq_close, True, True)


def sig_spy_qqq_rsi_x_hibeta_wk(df, spy_close=None, qqq_close=None):
    if spy_close is None or qqq_close is None:
        return _empty(df)
    return _gate(sig_spy_qqq_rsi_x_both_wk(df, spy_close, qqq_close), df, spy_close, qqq_close, True, False)


def sig_spy_qqq_rsi_x_hicorr_wk(df, spy_close=None, qqq_close=None):
    if spy_close is None or qqq_close is None:
        return _empty(df)
    return _gate(sig_spy_qqq_rsi_x_both_wk(df, spy_close, qqq_close), df, spy_close, qqq_close, False, True)


def sig_spy_qqq_rsi_x_hibeta_hicorr_wk(df, spy_close=None, qqq_close=None):
    if spy_close is None or qqq_close is None:
        return _empty(df)
    return _gate(sig_spy_qqq_rsi_x_both_wk(df, spy_close, qqq_close), df, spy_close, qqq_close, True, True)


MARKET_SIGNAL_KEYS = {
    "spy_rsi_x", "qqq_rsi_x", "spy_qqq_rsi_x_both",
    "spy_rsi_x_wk", "qqq_rsi_x_wk", "spy_qqq_rsi_x_both_wk",
    "corr_spyqqq_x_high", "beta_spyqqq_x_high",
    "spy_qqq_rsi_x_hibeta", "spy_qqq_rsi_x_hicorr", "spy_qqq_rsi_x_hibeta_hicorr",
    "spy_qqq_rsi_x_hibeta_wk", "spy_qqq_rsi_x_hicorr_wk", "spy_qqq_rsi_x_hibeta_hicorr_wk",
}

def sig_boll_lower(df):
    bb = ta.volatility.BollingerBands(df["Close"], window=20)
    return df["Low"] <= bb.bollinger_lband()

def sig_boll_upper(df):
    bb = ta.volatility.BollingerBands(df["Close"], window=20)
    return df["High"] >= bb.bollinger_hband()

def sig_monday(df):
    return pd.Series(df.index.dayofweek == 0, index=df.index)

def sig_friday(df):
    return pd.Series(df.index.dayofweek == 4, index=df.index)

def sig_doji(df):
    body = abs(df["Close"] - df["Open"])
    wick = df["High"] - df["Low"]
    return (body < wick * 0.1) & (wick > 0)

def sig_hammer(df):
    body = abs(df["Close"] - df["Open"])
    lower = pd.concat([df["Close"], df["Open"]], axis=1).min(axis=1) - df["Low"]
    upper = df["High"] - pd.concat([df["Close"], df["Open"]], axis=1).max(axis=1)
    return (lower > body * 2) & (upper < body * 0.5) & (body > 0)

def sig_big_red(df):
    ret = (df["Close"] - df["Open"]) / df["Open"] * 100
    return ret < -3

def sig_big_green(df):
    ret = (df["Close"] - df["Open"]) / df["Open"] * 100
    return ret > 3

# Volatility-NORMALIZED big day: today's close-to-close return in units of the stock's own
# trailing 20d realized vol (z-score). A "shock" is a move large RELATIVE to how much this name
# usually moves — unlike big_red/big_green which use a fixed 3% for every ticker. Vol is trailing
# (shifted 1 bar) so the day itself doesn't deflate its own z.
def _vol_shock_z(df, win=20):
    ret = df["Close"].pct_change()
    vol = ret.rolling(win).std().shift(1)
    return ret / vol

def sig_vol_shock_up(df):   return _vol_shock_z(df) >= 2.0
def sig_vol_shock_dn(df):   return _vol_shock_z(df) <= -2.0
def sig_vol_shock_dn3(df):  return _vol_shock_z(df) <= -3.0

# Volume-CONFIRMED shock: the move also came on above-average volume (>1.5x trailing-20d avg,
# shifted). In the study, volume confirmation made the shock more RELIABLE (higher t) even though
# the point estimate shrank — genuine participation behind the move, not a thin-tape print.
def _hi_vol(df, win=20):
    return df["Volume"] > 1.5 * df["Volume"].rolling(win).mean().shift(1)

def sig_vol_shock_up_hivol(df):   return sig_vol_shock_up(df) & _hi_vol(df)
def sig_vol_shock_dn_hivol(df):   return sig_vol_shock_dn(df) & _hi_vol(df)
def sig_vol_shock_dn3_hivol(df):  return sig_vol_shock_dn3(df) & _hi_vol(df)

def sig_weekly_up_3pct(df):
    ret5 = df["Close"].pct_change(5) * 100
    return ret5 > 3

def sig_weekly_down_3pct(df):
    ret5 = df["Close"].pct_change(5) * 100
    return ret5 < -3

def sig_above_avg_volume(df):
    avg = df["Volume"].rolling(20).mean()
    return df["Volume"] > avg * 1.5

def sig_below_avg_volume(df):
    avg = df["Volume"].rolling(20).mean()
    return df["Volume"] < avg * 0.5

def sig_price_at_sma20(df):
    sma = df["Close"].rolling(20).mean()
    dist = abs(df["Close"] - sma) / sma * 100
    return dist < 0.5

def sig_narrow_range(df):
    rng = (df["High"] - df["Low"]) / df["Low"] * 100
    avg_rng = rng.rolling(20).mean()
    return rng < avg_rng * 0.5

def sig_rsi_cross_sma_from_below50(df):
    """RSI crosses above SMA AND RSI was below 50 at time of cross."""
    rsi = ta.momentum.rsi(df["Close"], window=10)
    sma = rsi.rolling(10).mean()
    return (rsi > sma) & (rsi.shift(1) <= sma.shift(1)) & (rsi.shift(1) < 50)

def _rolling_sortino(df, w=10):
    """Fast vectorized rolling Sortino using stride tricks."""
    if "_sortino" in df.columns:
        return df["_sortino"]
    ret = np.log(df["Close"] / df["Close"].shift(1)).values
    daily_rf = 0.05 / 252
    n = len(ret)
    result = np.full(n, np.nan)

    # Use stride tricks for fast windowed computation
    if n > w:
        from numpy.lib.stride_tricks import sliding_window_view
        valid_ret = np.nan_to_num(ret, nan=0.0)
        windows = sliding_window_view(valid_ret, w)
        # Vectorized across all windows at once
        excess = windows - daily_rf
        mean_excess = excess.mean(axis=1)
        downside = np.minimum(excess, 0)
        dd = np.sqrt(np.mean(downside**2, axis=1))
        mask = dd > 1e-10
        ratios = np.where(mask, mean_excess / dd, np.nan)
        # window k covers returns k..k+w-1 → ends at bar k+w-1, so the ratio
        # belongs at result[k+w-1]. result[w-1:] has len n-w+1 == len(ratios).
        result[w - 1:] = ratios

    return pd.Series(result, index=df.index)

def sig_rsi_cross_sma_pos_sortino(df):
    """RSI crosses above SMA from <50 AND Sortino > 0."""
    rsi = ta.momentum.rsi(df["Close"], window=10)
    sma = rsi.rolling(10).mean()
    sortino = _rolling_sortino(df)
    cross = (rsi > sma) & (rsi.shift(1) <= sma.shift(1)) & (rsi.shift(1) < 50)
    return cross & (sortino > 0)

def _rolling_omega(df, w=10):
    """Fast vectorized rolling Omega using stride tricks."""
    if "_omega" in df.columns:
        return df["_omega"]
    ret = np.log(df["Close"] / df["Close"].shift(1)).values
    n = len(ret)
    result = np.full(n, np.nan)

    if n > w:
        from numpy.lib.stride_tricks import sliding_window_view
        valid_ret = np.nan_to_num(ret, nan=0.0)
        windows = sliding_window_view(valid_ret, w)
        gains = np.sum(np.maximum(windows, 0), axis=1)
        losses = -np.sum(np.minimum(windows, 0), axis=1)
        mask = losses > 1e-10
        ratios = np.where(mask, gains / losses, np.nan)
        # window k ends at bar k+w-1 → ratio belongs at result[k+w-1].
        result[w - 1:] = ratios

    return pd.Series(result, index=df.index)

def _dn_capture_signal(df, threshold, direction):
    """AVERAGE DOWN-DAY RETURN (in %) crossing a level. NOTE: this is NOT a downside-capture ratio
    (there is no SPY benchmark) — it is the mean of the trailing-10-bar down-day LOG returns, in %.
    `threshold` is in the same %/100 units: 50 → the -0.5% level, 150 → the -1.5% level, 0 → 0%.
    min_periods=3 so the mean forms from >=3 down days instead of requiring 10 CONSECUTIVE down days
    (default min_periods=window left the masked series ~all-NaN → the signal never fired)."""
    ret = np.log(df["Close"] / df["Close"].shift(1))
    neg_ret = ret.copy()
    neg_ret[neg_ret >= 0] = np.nan
    dn_avg = neg_ret.rolling(10, min_periods=3).mean() * 100  # avg down-day return, in %

    if direction == "below":
        return (dn_avg < -abs(threshold/100)) & (dn_avg.shift(1) >= -abs(threshold/100))
    else:
        return (dn_avg > -abs(threshold/100)) & (dn_avg.shift(1) <= -abs(threshold/100))


def _dn_trending(df, direction):
    """Trend change in the avg down-day return (its 10-bar mean vs a 5-bar smoothing of that mean).
    min_periods added so the sparse masked series produces a live series (see _dn_capture_signal)."""
    ret = np.log(df["Close"] / df["Close"].shift(1))
    neg_ret = ret.copy()
    neg_ret[neg_ret >= 0] = np.nan
    dn_avg = neg_ret.rolling(10, min_periods=3).mean()
    dn_sma = dn_avg.rolling(5, min_periods=2).mean()

    if direction == "down":  # improving = dn going less negative
        return (dn_avg > dn_sma) & (dn_avg.shift(1) <= dn_sma.shift(1))
    else:  # worsening = dn going more negative
        return (dn_avg < dn_sma) & (dn_avg.shift(1) >= dn_sma.shift(1))


def _updn_spread_signal(df, threshold, direction):
    """Spread between the avg UP-day and avg DOWN-day return (both in %), i.e. avg_up - avg_dn.
    min_periods=3 on each masked mean so the series is live; the previous .fillna(0) was dropped
    because zero-filling a missing half biased the spread toward 0 and pinned the cross tests."""
    ret = np.log(df["Close"] / df["Close"].shift(1))
    pos_ret = ret.copy()
    pos_ret[pos_ret <= 0] = np.nan
    neg_ret = ret.copy()
    neg_ret[neg_ret >= 0] = np.nan

    up_avg = pos_ret.rolling(10, min_periods=3).mean() * 100
    dn_avg = neg_ret.rolling(10, min_periods=3).mean() * 100
    spread = up_avg - dn_avg  # positive = more upside than downside

    if direction == "above":
        return (spread > threshold/100) & (spread.shift(1) <= threshold/100)
    else:
        return (spread < threshold/100) & (spread.shift(1) >= threshold/100)


def _obv(df):
    """On-Balance Volume."""
    sign = np.where(df["Close"] > df["Close"].shift(1), 1, np.where(df["Close"] < df["Close"].shift(1), -1, 0))
    return (df["Volume"] * sign).cumsum()

def _obv_divergence(df, lookback=20):
    """Price makes lower low but OBV makes higher low."""
    obv = _obv(df)
    price_ll = df["Low"] <= df["Low"].rolling(lookback).min()
    obv_hl = obv > obv.rolling(lookback).min()
    return price_ll & obv_hl

def _ad_line(df):
    """Accumulation/Distribution line."""
    mfm = ((df["Close"] - df["Low"]) - (df["High"] - df["Close"])) / (df["High"] - df["Low"]).replace(0, 1)
    return (mfm * df["Volume"]).cumsum()

def _ad_rising(df, period=20):
    """A/D line SMA trending up while price flat/down."""
    ad = _ad_line(df)
    ad_sma = ad.rolling(period).mean()
    ad_up = ad_sma > ad_sma.shift(1)
    price_flat = df["Close"] <= df["Close"].shift(1)
    return ad_up & price_flat & (ad_up.shift(1) == False)

def _cmf(df, period=20):
    """Chaikin Money Flow."""
    mfm = ((df["Close"] - df["Low"]) - (df["High"] - df["Close"])) / (df["High"] - df["Low"]).replace(0, 1)
    mfv = mfm * df["Volume"]
    # Guard a fully-zero-volume (halted/illiquid) window: 0/0 → NaN → no fire.
    return mfv.rolling(period).sum() / df["Volume"].rolling(period).sum().replace(0, np.nan)

def _cmf_cross_zero(df, period=20):
    """CMF crosses above zero."""
    cmf = _cmf(df, period)
    return (cmf > 0) & (cmf.shift(1) <= 0)

def _vwap_reclaim(df, period=20):
    """Price crosses back above VWAP after being below."""
    typical = (df["High"] + df["Low"] + df["Close"]) / 3
    vwap = (typical * df["Volume"]).rolling(period).sum() / df["Volume"].rolling(period).sum()
    return (df["Close"] > vwap) & (df["Close"].shift(1) <= vwap.shift(1))

def _vol_shrink_pullback(df, period=10):
    """Volume shrinks on red days, green day with volume expansion follows."""
    avg_vol = df["Volume"].rolling(20).mean()
    red = df["Close"] < df["Open"]
    green = df["Close"] > df["Open"]
    # Check last 3 red days had below-avg volume, today is green with above-avg
    red_low_vol = red & (df["Volume"] < avg_vol * 0.8)
    consecutive_low = red_low_vol.rolling(3).sum() >= 2
    today_green_vol = green & (df["Volume"] > avg_vol * 1.3)
    return consecutive_low.shift(1) & today_green_vol

def _pocket_pivot(df, lookback=10):
    """Price up day with volume > highest volume of any down day in last N days."""
    green = df["Close"] > df["Close"].shift(1)
    red_vol = df["Volume"].where(df["Close"] < df["Close"].shift(1))
    max_red_vol = red_vol.rolling(lookback, min_periods=1).max().fillna(0)
    return green & (df["Volume"] > max_red_vol) & (max_red_vol > 0)

def _tight_closes(df, period=5, threshold=1.5):
    """Multiple days closing within narrow range near the high."""
    close_range = (df["Close"].rolling(period).max() - df["Close"].rolling(period).min()) / df["Close"] * 100
    near_high = (df["Close"] - df["Low"]) / (df["High"] - df["Low"]).replace(0, 1) > 0.6
    tight = close_range < threshold
    return tight & near_high & (tight.shift(1) == False)

def _volume_dry_up_reversal(df, lookback=10):
    """Volume drops to low after decline, then high-volume green day."""
    avg_vol = df["Volume"].rolling(20).mean()
    vol_ratio = df["Volume"] / avg_vol
    # Low volume for several days
    low_vol = vol_ratio.rolling(lookback).mean() < 0.8
    # Today: high volume green
    green_spike = (df["Close"] > df["Open"]) & (vol_ratio > 1.3)
    # Was declining
    declining = df["Close"].shift(1) < df["Close"].shift(lookback)
    return low_vol.shift(1) & green_spike & declining

def _sma50_bounce_volume(df):
    """Price bounces off 50-SMA with above-average volume."""
    sma50 = df["Close"].rolling(50).mean()
    avg_vol = df["Volume"].rolling(20).mean()
    # Price was at or below SMA50 yesterday, above today
    bounce = (df["Close"] > sma50) & (df["Low"] <= sma50 * 1.01)
    high_vol = df["Volume"] > avg_vol * 1.5
    return bounce & high_vol

def _high_vol_narrow_range(df):
    """Very high volume on narrow range day — absorption."""
    avg_vol = df["Volume"].rolling(20).mean()
    rng = (df["High"] - df["Low"]) / df["Close"] * 100
    avg_rng = rng.rolling(20).mean()
    return (df["Volume"] > avg_vol * 1.5) & (rng < avg_rng * 0.7)


def _seq_rs_hl_rsi(df, window=10):
    """Sequential: RSI of Sortino crossed up, then Higher Low, then RSI crossed up — all within N days.
    Fires on the day the RSI cross happens (final confirmation)."""
    rs = _rsi_of_sortino(df).pipe(lambda r: (r > r.rolling(14).mean()) & (r.shift(1) <= r.rolling(14).mean().shift(1)))
    hl = _higher_low(df)
    rsi = ta.momentum.rsi(df["Close"], window=10)
    rsi_sma = rsi.rolling(10).mean()
    rsi_x = (rsi > rsi_sma) & (rsi.shift(1) <= rsi_sma.shift(1))

    # RSI cross fires. Check if HL happened in last N days AND RS happened before HL
    result = pd.Series(False, index=df.index)
    for i in range(window * 2, len(df)):
        if not rsi_x.iloc[i]:
            continue
        # Look back for HL in last window days
        hl_found = False
        hl_day = -1
        for j in range(1, window + 1):
            if i - j >= 0 and hl.iloc[i - j]:
                hl_found = True
                hl_day = i - j
                break
        if not hl_found:
            continue
        # Look back from HL for RS cross
        for j in range(1, window + 1):
            if hl_day - j >= 0 and rs.iloc[hl_day - j]:
                result.iloc[i] = True
                break
    return result


def _seq_rs_rsi(df, window=10):
    """Sequential: RSI of Sortino crossed up, then RSI crossed up within N days."""
    rs = _rsi_of_sortino(df).pipe(lambda r: (r > r.rolling(14).mean()) & (r.shift(1) <= r.rolling(14).mean().shift(1)))
    rsi = ta.momentum.rsi(df["Close"], window=10)
    rsi_sma = rsi.rolling(10).mean()
    rsi_x = (rsi > rsi_sma) & (rsi.shift(1) <= rsi_sma.shift(1))

    result = pd.Series(False, index=df.index)
    for i in range(window, len(df)):
        if not rsi_x.iloc[i]:
            continue
        for j in range(1, window + 1):
            if i - j >= 0 and rs.iloc[i - j]:
                result.iloc[i] = True
                break
    return result


def _seq_hl_rsi(df, window=10):
    """Sequential: Higher Low, then RSI crossed up within N days."""
    hl = _higher_low(df)
    rsi = ta.momentum.rsi(df["Close"], window=10)
    rsi_sma = rsi.rolling(10).mean()
    rsi_x = (rsi > rsi_sma) & (rsi.shift(1) <= rsi_sma.shift(1))

    result = pd.Series(False, index=df.index)
    for i in range(window, len(df)):
        if not rsi_x.iloc[i]:
            continue
        for j in range(1, window + 1):
            if i - j >= 0 and hl.iloc[i - j]:
                result.iloc[i] = True
                break
    return result


def _seq_rs_hl(df, window=10):
    """Sequential: RSI of Sortino crossed up, then Higher Low within N days."""
    rs = _rsi_of_sortino(df).pipe(lambda r: (r > r.rolling(14).mean()) & (r.shift(1) <= r.rolling(14).mean().shift(1)))
    hl = _higher_low(df)

    result = pd.Series(False, index=df.index)
    for i in range(window, len(df)):
        if not hl.iloc[i]:
            continue
        for j in range(1, window + 1):
            if i - j >= 0 and rs.iloc[i - j]:
                result.iloc[i] = True
                break
    return result


def _seq_sort_neg_rs_rsi(df, window=10):
    """Sortino was <0, then RSI of Sortino crossed up, then RSI crossed up — all within N days."""
    sortino = _rolling_sortino(df)
    rs = _rsi_of_sortino(df).pipe(lambda r: (r > r.rolling(10).mean()) & (r.shift(1) <= r.rolling(10).mean().shift(1)))
    rsi = ta.momentum.rsi(df["Close"], window=10)
    rsi_sma = rsi.rolling(10).mean()
    rsi_x = (rsi > rsi_sma) & (rsi.shift(1) <= rsi_sma.shift(1))

    result = pd.Series(False, index=df.index)
    for i in range(window * 2, len(df)):
        if not rsi_x.iloc[i]:
            continue
        # Look back for RS cross
        rs_day = -1
        for j in range(1, window + 1):
            if i - j >= 0 and rs.iloc[i - j]:
                rs_day = i - j
                break
        if rs_day < 0:
            continue
        # Look back from RS cross for Sortino < 0
        for j in range(0, window + 1):
            if rs_day - j >= 0 and sortino.iloc[rs_day - j] < 0:
                result.iloc[i] = True
                break
    return result


def _seq_triple_os_recovery(df, window=10, rsi_thresh=30, rs_thresh=30, need_hl=True, need_rsi_x=True, need_rs_x=True):
    """Triple oversold (RSI<X + Sortino<0 + RS<X) then recovery signals within N days."""
    rsi = ta.momentum.rsi(df["Close"], window=10)
    rsi_sma = rsi.rolling(10).mean()
    sortino = _rolling_sortino(df)
    rs = _rsi_of_sortino(df)
    rs_sma = rs.rolling(14).mean()
    hl = _higher_low(df) if need_hl else pd.Series(True, index=df.index)

    # Phase 1: all three oversold on same day
    phase1 = (rsi < rsi_thresh) & (sortino < 0) & (rs < rs_thresh)

    result = pd.Series(False, index=df.index)
    for i in range(len(df)):
        if not phase1.iloc[i]:
            continue
        # Track which recovery conditions have been met
        rsi_crossed = not need_rsi_x
        rs_crossed = not need_rs_x
        hl_hit = not need_hl
        for j in range(i + 1, min(i + window + 1, len(df))):
            if need_rsi_x and not rsi_crossed:
                if rsi.iloc[j] > rsi_sma.iloc[j] and rsi.iloc[j-1] <= rsi_sma.iloc[j-1]:
                    rsi_crossed = True
            if need_rs_x and not rs_crossed:
                if rs.iloc[j] > rs_sma.iloc[j] and rs.iloc[j-1] <= rs_sma.iloc[j-1]:
                    rs_crossed = True
            if need_hl and not hl_hit:
                if hl.iloc[j]:
                    hl_hit = True
            if rsi_crossed and rs_crossed and hl_hit:
                result.iloc[j] = True
                break
    return result


def _seq_a_then_b(df, sig_a_fn, sig_b_fn, window):
    """Generic sequential: signal A fires, then signal B fires within N days."""
    a = sig_a_fn(df).fillna(False)
    b = sig_b_fn(df).fillna(False)
    result = pd.Series(False, index=df.index)
    for i in range(len(df)):
        if not a.iloc[i]:
            continue
        for j in range(i + 1, min(i + window + 1, len(df))):
            if b.iloc[j]:
                result.iloc[j] = True
                break
    return result


def _seq_both_os_then_cross(df, window=5):
    """RSI of Sortino <30 AND RSI<30 both happen, then both cross above their SMA(14) within N days."""
    rs = _rsi_of_sortino(df)
    rs_sma = rs.rolling(14).mean()
    rsi = ta.momentum.rsi(df["Close"], window=10)
    rsi_sma = rsi.rolling(14).mean()

    # Find days where both are below 30
    both_os = (rs < 30) & (rsi < 30)
    result = pd.Series(False, index=df.index)

    for i in range(len(df)):
        if not both_os.iloc[i]:
            continue
        # Look forward for both crossing above their SMAs
        for j in range(i + 1, min(i + window + 1, len(df))):
            rs_crossed = rs.iloc[j] > rs_sma.iloc[j] and rs.iloc[j-1] <= rs_sma.iloc[j-1]
            rsi_crossed = rsi.iloc[j] > rsi_sma.iloc[j] and rsi.iloc[j-1] <= rsi_sma.iloc[j-1]
            # Check if both have crossed at some point (not necessarily same day)
            rs_above = rs.iloc[j] > rs_sma.iloc[j]
            rsi_above = rsi.iloc[j] > rsi_sma.iloc[j]
            if rs_above and rsi_above and (rs_crossed or rsi_crossed):
                result.iloc[j] = True
                break
    return result


def _seq_oversold_vol_rsi(df, window=10, rsi_threshold=20):
    """RSI<threshold + Sortino<0 happened, then volume spike up + RSI crossover within N days."""
    rsi = ta.momentum.rsi(df["Close"], window=10)
    rsi_sma = rsi.rolling(10).mean()
    sortino = _rolling_sortino(df)
    avg_vol = df["Volume"].rolling(20).mean()

    # Oversold condition: RSI < threshold AND Sortino < 0
    oversold = (rsi < rsi_threshold) & (sortino < 0)

    # Volume spike up + RSI crossover
    vol_spike_up = (df["Volume"] > avg_vol * 1.5) & (df["Close"] > df["Open"])
    rsi_x = (rsi > rsi_sma) & (rsi.shift(1) <= rsi_sma.shift(1))
    trigger = vol_spike_up & rsi_x

    result = pd.Series(False, index=df.index)
    for i in range(window * 2, len(df)):
        if not trigger.iloc[i]:
            continue
        # Was oversold in the last N days?
        for j in range(1, window + 1):
            if i - j >= 0 and oversold.iloc[i - j]:
                result.iloc[i] = True
                break
    return result


def _seq_oversold_trigger(df, window=10, rsi_threshold=20, trigger_fn=None, use_sortino_only=False):
    """Generic: oversold condition happened, then trigger_fn fires within N days."""
    rsi = ta.momentum.rsi(df["Close"], window=10)
    sortino = _rolling_sortino(df)

    if use_sortino_only:
        oversold = sortino < 0
    else:
        oversold = (rsi < rsi_threshold) & (sortino < 0)

    trigger = trigger_fn(df).fillna(False) if trigger_fn else pd.Series(False, index=df.index)

    result = pd.Series(False, index=df.index)
    for i in range(window * 2, len(df)):
        if not trigger.iloc[i]:
            continue
        for j in range(1, window + 1):
            if i - j >= 0 and oversold.iloc[i - j]:
                result.iloc[i] = True
                break
    return result


def _higher_low(df, lookback=10):
    """Detect higher local low — current swing low > previous swing low.
    Uses rolling min over lookback window to find local lows."""
    low_series = df["Low"]
    # Current local low (last `lookback` bars)
    curr_low = low_series.rolling(lookback).min()
    # Previous local low (lookback before that)
    prev_low = low_series.shift(lookback).rolling(lookback).min()
    # Signal when current low just became higher than previous low
    is_higher = curr_low > prev_low
    was_not = curr_low.shift(1) <= prev_low.shift(1)
    return is_higher & was_not

def _lower_high(df, lookback=10):
    """Detect lower local high — current swing high < previous swing high."""
    high_series = df["High"]
    curr_high = high_series.rolling(lookback).max()
    prev_high = high_series.shift(lookback).rolling(lookback).max()
    is_lower = curr_high < prev_high
    was_not = curr_high.shift(1) >= prev_high.shift(1)
    return is_lower & was_not

def _rsi_of_sortino(df, w=10):
    """RSI(10) computed on the Sortino ratio series. Uses ffill to match TradingView."""
    sortino = _rolling_sortino(df)
    sortino_filled = sortino.ffill().fillna(0)
    if sortino_filled.notna().sum() < w + 5:
        return pd.Series(np.nan, index=df.index)
    return ta.momentum.rsi(sortino_filled, window=w)

def sig_rsi_cross_sma_pos_omega(df):
    """RSI crosses above SMA from <50 AND Omega > 1."""
    rsi = ta.momentum.rsi(df["Close"], window=10)
    sma = rsi.rolling(10).mean()
    omega = _rolling_omega(df)
    cross = (rsi > sma) & (rsi.shift(1) <= sma.shift(1)) & (rsi.shift(1) < 50)
    return cross & (omega > 1)

def _rolling_updn_spread(df, spy_close, w=10):
    """Compute Up% - Dn% spread vs SPY."""
    ret = df["Close"].pct_change()
    spy_ret = spy_close.pct_change().reindex(ret.index)
    spread = pd.Series(np.nan, index=ret.index)
    for i in range(w - 1, len(ret)):
        # Trailing w-bar window ENDING AT and INCLUDING bar i (iloc[i-w+1:i+1]);
        # iloc[i-w:i] excluded the current bar and shifted the value one bar early.
        window_r = ret.iloc[i-w+1:i+1]
        window_s = spy_ret.iloc[i-w+1:i+1]
        up_mask = window_s > 0
        dn_mask = window_s < 0
        if up_mask.sum() >= 2 and dn_mask.sum() >= 2:
            up_cap = window_r[up_mask].mean() / window_s[up_mask].mean() * 100
            dn_cap = window_r[dn_mask].mean() / window_s[dn_mask].mean() * 100
            spread.iloc[i] = up_cap - dn_cap
    return spread

# These need SPY data — will be handled specially in run_study
def sig_rsi_cross_sma_pos_updn(df, spy_close=None):
    """RSI crosses above SMA from <50 AND Up%-Dn% spread > 0 vs SPY."""
    rsi = ta.momentum.rsi(df["Close"], window=10)
    sma = rsi.rolling(10).mean()
    cross = (rsi > sma) & (rsi.shift(1) <= sma.shift(1)) & (rsi.shift(1) < 50)
    if spy_close is None:
        return cross
    spread = _rolling_updn_spread(df, spy_close)
    return cross & (spread > 0)

def sig_rsi_cross_sma_neg_sortino(df):
    """RSI crosses below SMA AND Sortino is negative (momentum dying)."""
    rsi = ta.momentum.rsi(df["Close"], window=10)
    sma = rsi.rolling(10).mean()
    sortino = _rolling_sortino(df)
    cross = (rsi < sma) & (rsi.shift(1) >= sma.shift(1))
    return cross & (sortino < 0)

def sig_pos_sortino_only(df):
    """Sortino turns positive (crosses above 0)."""
    sortino = _rolling_sortino(df)
    return (sortino > 0) & (sortino.shift(1) <= 0)

def sig_omega_above_1(df):
    """Omega crosses above 1."""
    ret = np.log(df["Close"] / df["Close"].shift(1))
    def _omega(r):
        gains = r[r > 0].sum()
        losses = -r[r < 0].sum()
        return gains / losses if losses > 1e-10 else np.nan
    omega = ret.rolling(10).apply(_omega, raw=False)
    return (omega > 1) & (omega.shift(1) <= 1)

def sig_rsi_cross_sma_omega_sortino(df):
    """Triple: RSI crosses SMA from <50 + Omega>1 + Sortino>0."""
    rsi = ta.momentum.rsi(df["Close"], window=10)
    sma = rsi.rolling(10).mean()
    sortino = _rolling_sortino(df)
    omega = _rolling_omega(df)
    cross = (rsi > sma) & (rsi.shift(1) <= sma.shift(1)) & (rsi.shift(1) < 50)
    return cross & (omega > 1) & (sortino > 0)

def sig_rsi_cross_sma_macd_great(df):
    """RSI crosses above SMA from <50 AND MACD daily is great (>signal and >0)."""
    rsi = ta.momentum.rsi(df["Close"], window=10)
    sma = rsi.rolling(10).mean()
    cross = (rsi > sma) & (rsi.shift(1) <= sma.shift(1)) & (rsi.shift(1) < 50)
    return cross & _macd_great(df)

def sig_rsi_omega_macd_great(df):
    """Dashboard rule: RSI crosses SMA from <50 + Omega>1 + MACD daily great."""
    rsi = ta.momentum.rsi(df["Close"], window=10)
    sma = rsi.rolling(10).mean()
    omega = _rolling_omega(df)
    cross = (rsi > sma) & (rsi.shift(1) <= sma.shift(1)) & (rsi.shift(1) < 50)
    return cross & (omega > 1) & _macd_great(df)

def sig_wide_range(df):
    rng = (df["High"] - df["Low"]) / df["Low"] * 100
    avg_rng = rng.rolling(20).mean()
    return rng > avg_rng * 2


# ── Alt-data event signals (Phase D). Read columns attached per-stock in the all-on-all
# worker (_insider_buy / _filed_13d / _filed_13g). Guarded: return all-False when the
# column is absent (e.g. the ETF engine), so they no-op safely there. ─────────────────
def _altfalse(df):
    return pd.Series(False, index=df.index)

def sig_insider_buy(df):
    if "_insider_buy" not in df.columns: return _altfalse(df)
    return df["_insider_buy"] > 0

def sig_insider_buy_rsi_os(df):
    if "_insider_buy" not in df.columns: return _altfalse(df)
    rsi = ta.momentum.rsi(df["Close"], window=10)
    recent = df["_insider_buy"].rolling(10, min_periods=1).sum() > 0
    return recent & (rsi < 35)

def sig_insider_buy_dd25(df):
    if "_insider_buy" not in df.columns: return _altfalse(df)
    recent = df["_insider_buy"].rolling(10, min_periods=1).sum() > 0
    dd = df["Close"] / df["Close"].cummax() - 1.0
    return recent & (dd <= -0.25)

def sig_activist_13d(df):
    if "_filed_13d" not in df.columns: return _altfalse(df)
    return df["_filed_13d"] > 0

def sig_activist_13d_os(df):
    if "_filed_13d" not in df.columns: return _altfalse(df)
    rsi = ta.momentum.rsi(df["Close"], window=10)
    recent = df["_filed_13d"].rolling(20, min_periods=1).sum() > 0
    return recent & (rsi < 40)

def sig_stake_13g(df):
    if "_filed_13g" not in df.columns: return _altfalse(df)
    return df["_filed_13g"] > 0

# Per-stock only; skipped by the ETF studies engine (meaningless on sector ETFs).
ALT_SIGNAL_KEYS = {
    "insider_buy", "insider_buy_rsi_os", "insider_buy_dd25",
    "activist_13d", "activist_13d_os", "stake_13g",
}

SIGNALS = {
    "insider_buy": ("Insider Open-Market Buy", sig_insider_buy),
    "insider_buy_rsi_os": ("Insider Buy + RSI<35", sig_insider_buy_rsi_os),
    "insider_buy_dd25": ("Insider Buy + Drawdown<=-25%", sig_insider_buy_dd25),
    "activist_13d": ("Activist 13D Filed", sig_activist_13d),
    "activist_13d_os": ("Activist 13D + RSI<40", sig_activist_13d_os),
    "stake_13g": ("Institutional 13G Filed", sig_stake_13g),
    "gap_up": ("Gap Up >0.5%", sig_gap_up),
    "gap_down": ("Gap Down >0.5%", sig_gap_down),
    "gap_up_med": ("Gap Up >1%", sig_gap_up_medium),
    "gap_down_med": ("Gap Down >1%", sig_gap_down_medium),
    "gap_up_large": ("Gap Up >2%", sig_gap_up_large),
    "gap_down_large": ("Gap Down >2%", sig_gap_down_large),
    "rsi_x_above_sma": ("RSI Crosses Above SMA", sig_rsi_cross_above_sma),
    "rsi_x_below_sma": ("RSI Crosses Below SMA", sig_rsi_cross_below_sma),
    "rsi_x_sma_below50": ("RSI Cross SMA (from <50)", sig_rsi_cross_above_sma_below50),
    "rsi_oversold30": ("RSI Below 30", sig_rsi_oversold),
    "rsi_oversold20": ("RSI Below 20", sig_rsi_oversold_20),
    "rsi_overbought": ("RSI Above 70", sig_rsi_overbought),
    "rsi_x50_up": ("RSI Cross 50 Up", sig_rsi_cross_50_up),
    "rsi_x50_down": ("RSI Cross 50 Down", sig_rsi_cross_50_down),
    "price_x_sma20_up": ("Price Cross SMA20 Up", sig_price_cross_sma20_up),
    "price_x_sma20_dn": ("Price Cross SMA20 Down", sig_price_cross_sma20_down),
    "price_x_sma50_up": ("Price Cross SMA50 Up", sig_price_cross_sma50_up),
    "golden_cross": ("Golden Cross", sig_golden_cross),
    "death_cross": ("Death Cross", sig_death_cross),
    "vol_spike_up": ("Volume Spike + Up", sig_vol_spike_up),
    "vol_spike_down": ("Volume Spike + Down", sig_vol_spike_down),
    "3down": ("3 Down Days", sig_3down),
    "3up": ("3 Up Days", sig_3up),
    "5down": ("5 Down Days", sig_5down),
    "new_20high": ("New 20-Day High", sig_new_20high),
    "new_20low": ("New 20-Day Low", sig_new_20low),
    "new_52high": ("New 52-Week High", sig_new_52high),
    "new_52low": ("New 52-Week Low", sig_new_52low),
    "bull_engulf": ("Bullish Engulfing", sig_bullish_engulf),
    "bear_engulf": ("Bearish Engulfing", sig_bearish_engulf),
    "macd_x_up": ("MACD Cross Up", sig_macd_cross_up),
    "macd_x_down": ("MACD Cross Down", sig_macd_cross_down),
    "macd_great": ("MACD Great (>signal & >0)", sig_macd_great),
    "boll_lower": ("Touch Lower Bollinger", sig_boll_lower),
    "boll_upper": ("Touch Upper Bollinger", sig_boll_upper),
    "monday": ("Buy Monday", sig_monday),
    "friday": ("Buy Friday", sig_friday),
    "doji": ("Doji Candle", sig_doji),
    "hammer": ("Hammer Candle", sig_hammer),
    "big_red": ("Big Red Day (>3%)", sig_big_red),
    "big_green": ("Big Green Day (>3%)", sig_big_green),
    "vol_shock_up": ("Vol-Shock Up (+2σ day)", sig_vol_shock_up),
    "vol_shock_dn": ("Vol-Shock Down (-2σ day)", sig_vol_shock_dn),
    "vol_shock_dn3": ("Vol-Shock Down (-3σ day)", sig_vol_shock_dn3),
    "vol_shock_up_hivol": ("Vol-Shock Up +2σ (hi-vol)", sig_vol_shock_up_hivol),
    "vol_shock_dn_hivol": ("Vol-Shock Down -2σ (hi-vol)", sig_vol_shock_dn_hivol),
    "vol_shock_dn3_hivol": ("Vol-Shock Down -3σ (hi-vol)", sig_vol_shock_dn3_hivol),
    "weekly_up3": ("Weekly Up >3%", sig_weekly_up_3pct),
    "weekly_down3": ("Weekly Down >3%", sig_weekly_down_3pct),
    "high_vol": ("Above Avg Volume (1.5x)", sig_above_avg_volume),
    "low_vol": ("Below Avg Volume (0.5x)", sig_below_avg_volume),
    "at_sma20": ("Price At SMA20", sig_price_at_sma20),
    "narrow_range": ("Narrow Range Day", sig_narrow_range),
    "wide_range": ("Wide Range Day", sig_wide_range),
    "rsi_x_sma_b50": ("RSI Cross SMA (from <50)", sig_rsi_cross_sma_from_below50),
    "rsi_sup10_x": ("RSI <50 & <avg 10d+ then Cross", sig_rsi_suppressed_then_cross),
    "rsi_sup10_x_lt20": ("RSI <20 & <avg 10d+ then Cross", sig_rsi_suppressed_lt20_then_cross),
    "rsi_sup10_x_ad": ("RSI Sup10 Cross + A/D Trend Up", sig_rsi_suppressed_then_cross_ad),
    "rsi_sup10_x_vol": ("RSI Sup10 Cross + Vol>Avg", sig_rsi_suppressed_then_cross_vol),
    "rsi_sup10_x_hl": ("RSI Sup10 Cross + Higher Low", sig_rsi_suppressed_then_cross_hl),
    "rsi_sup10_x_mkt": ("RSI Sup10 Cross + SPY Turning Up", sig_rsi_suppressed_then_cross_mkt),
    "rsi_sup10_x_negsort": ("RSI Sup10 Cross + Sortino<0", sig_rsi_suppressed_then_cross_negsort),
    "rsi_sup10_x_dd40": ("RSI Sup10 Cross + 40% off ATH", sig_rsi_suppressed_then_cross_dd40),
    "rsi_sup10_x_dd50": ("RSI Sup10 Cross + 50% off ATH", sig_rsi_suppressed_then_cross_dd50),
    "rsi_sup10_x_dd60": ("RSI Sup10 Cross + 60% off ATH", sig_rsi_suppressed_then_cross_dd60),
    "rsi_sup10_x_dd70": ("RSI Sup10 Cross + 70% off ATH", sig_rsi_suppressed_then_cross_dd70),
    "rsi_sup10_x_dd50_ad": ("RSI Sup10 dd50 + A/D Rising", sig_rsi_suppressed_dd50_adrise),
    "rsi_sup10_x_dd50_vol": ("RSI Sup10 dd50 + Vol>Avg", sig_rsi_suppressed_dd50_vol),
    "rsi_sup10_x_dd50_mkt": ("RSI Sup10 dd50 + SPY Turning Up", sig_rsi_suppressed_dd50_mkt),
    "rsi_sup10_x_wk": ("RSI Sup10 Cross (Weekly)", sig_rsi_suppressed_then_cross_weekly),
    "rsi_x_pos_sort": ("RSI Cross SMA <50 + Sortino>0", sig_rsi_cross_sma_pos_sortino),
    "rsi_x_pos_omega": ("RSI Cross SMA <50 + Omega>1", sig_rsi_cross_sma_pos_omega),
    "rsi_x_pos_updn": ("RSI Cross SMA <50 + UpDn>0", sig_rsi_cross_sma_pos_updn),
    "rsi_x_neg_sort": ("RSI Cross Below SMA + Sortino<0", sig_rsi_cross_sma_neg_sortino),
    "sort_pos": ("Sortino Turns Positive", sig_pos_sortino_only),
    "omega_x1": ("Omega Crosses Above 1", sig_omega_above_1),
    "rsi_x_triple": ("RSI<50 + Omega>1 + Sortino>0", sig_rsi_cross_sma_omega_sortino),
    "rsi_x_macd_great": ("RSI Cross SMA <50 + MACD Great", sig_rsi_cross_sma_macd_great),
    "rsi_omega_macd_great": ("RSI<50 + Omega>1 + MACD Great", sig_rsi_omega_macd_great),
    "sort_neg": ("Sortino Turns Negative", lambda df: _rolling_sortino(df).pipe(lambda s: (s < 0) & (s.shift(1) >= 0))),
    "sort_above_1": ("Sortino Crosses Above 1", lambda df: _rolling_sortino(df).pipe(lambda s: (s > 1) & (s.shift(1) <= 1))),
    "sort_below_neg1": ("Sortino Below -1", lambda df: _rolling_sortino(df).pipe(lambda s: (s < -1) & (s.shift(1) >= -1))),
    "omega_below_1": ("Omega Drops Below 1", lambda df: _rolling_omega(df).pipe(lambda s: (s < 1) & (s.shift(1) >= 1))),
    "omega_above_2": ("Omega Crosses Above 2", lambda df: _rolling_omega(df).pipe(lambda s: (s > 2) & (s.shift(1) <= 2))),
    "rsi_os30_sort_neg": ("RSI<30 + Sortino<0", lambda df: (
        (ta.momentum.rsi(df["Close"], window=10) < 30) & (ta.momentum.rsi(df["Close"], window=10).shift(1) >= 30) & (_rolling_sortino(df) < 0)
    )),
    "rsi_ob70_sort_pos": ("RSI>70 + Sortino>0", lambda df: (
        (ta.momentum.rsi(df["Close"], window=10) > 70) & (ta.momentum.rsi(df["Close"], window=10).shift(1) <= 70) & (_rolling_sortino(df) > 0)
    )),
    "rsi_x50_omega_pos": ("RSI Cross 50 Up + Omega>1", lambda df: (
        (ta.momentum.rsi(df["Close"], window=10) > 50) & (ta.momentum.rsi(df["Close"], window=10).shift(1) <= 50) & (_rolling_omega(df) > 1)
    )),
    "gap_down_rsi_os": ("Gap Down + RSI<40", lambda df: (
        ((df["Open"] - df["Low"].shift(1)) / df["Low"].shift(1) * 100 <= -0.5) & (ta.momentum.rsi(df["Close"], window=10) < 40)
    )),
    "gap_up_omega_pos": ("Gap Up + Omega>1", lambda df: (
        ((df["Open"] - df["High"].shift(1)) / df["High"].shift(1) * 100 >= 0.5) & (_rolling_omega(df) > 1)
    )),
    "gap_down_sort_neg": ("Gap Down + Sortino<0", lambda df: (
        ((df["Open"] - df["Low"].shift(1)) / df["Low"].shift(1) * 100 <= -0.5) & (_rolling_sortino(df) < 0)
    )),
    "rsi50_x_sma50_up": ("RSI(50) Cross SMA(50) Up", lambda df: (
        ta.momentum.rsi(df["Close"], window=50).pipe(lambda r: (r > r.rolling(50).mean()) & (r.shift(1) <= r.rolling(50).mean().shift(1)))
    )),
    "rsi_weekly_x_up": ("Weekly RSI Cross (50d proxy)", lambda df: (
        ta.momentum.rsi(df["Close"], window=50).pipe(lambda r: (r > r.rolling(50).mean()) & (r.shift(5) <= r.rolling(50).mean().shift(5)) & (r.shift(5) < 50))
    )),
    "sort_pos_omega_pos": ("Sortino>0 + Omega>1 (both cross)", lambda df: (
        _rolling_sortino(df).pipe(lambda s: (s > 0) & (s.shift(1) <= 0)) & (_rolling_omega(df) > 1)
    )),
    "rsi_recovery": ("RSI Recovers from <30 to >50", lambda df: (
        (ta.momentum.rsi(df["Close"], window=10) > 50) & (ta.momentum.rsi(df["Close"], window=10).shift(1) <= 50) & (ta.momentum.rsi(df["Close"], window=10).rolling(10).min().shift(1) < 30)
    )),
    "rsi_bull_div": ("Bullish RSI Divergence (approx)", lambda df: (
        (df["Low"] <= df["Low"].rolling(20).min()) & (ta.momentum.rsi(df["Close"], window=10) > ta.momentum.rsi(df["Close"], window=10).rolling(20).min())
    )),
    # ── Sequential signals (A then B then C within N days) ──
    "seq_rs_hl_rsi_5d": ("Seq: RSI-Sort -> HL -> RSI Cross (5d)", lambda df: _seq_rs_hl_rsi(df, 5)),
    "seq_rs_hl_rsi_10d": ("Seq: RSI-Sort -> HL -> RSI Cross (10d)", lambda df: _seq_rs_hl_rsi(df, 10)),
    "seq_rs_hl_rsi_20d": ("Seq: RSI-Sort -> HL -> RSI Cross (20d)", lambda df: _seq_rs_hl_rsi(df, 20)),
    "seq_rs_rsi_5d": ("Seq: RSI-Sort -> RSI Cross (5d)", lambda df: _seq_rs_rsi(df, 5)),
    "seq_rs_rsi_10d": ("Seq: RSI-Sort -> RSI Cross (10d)", lambda df: _seq_rs_rsi(df, 10)),
    "seq_hl_rsi_5d": ("Seq: Higher Low -> RSI Cross (5d)", lambda df: _seq_hl_rsi(df, 5)),
    "seq_hl_rsi_10d": ("Seq: Higher Low -> RSI Cross (10d)", lambda df: _seq_hl_rsi(df, 10)),
    "seq_rs_hl_5d": ("Seq: RSI-Sort -> Higher Low (5d)", lambda df: _seq_rs_hl(df, 5)),
    "seq_rs_hl_10d": ("Seq: RSI-Sort -> Higher Low (10d)", lambda df: _seq_rs_hl(df, 10)),
    "seq_rs_hl_rsi_10d_omega": ("Seq: RS->HL->RSI (10d) + Omega>1", lambda df: _seq_rs_hl_rsi(df, 10) & (_rolling_omega(df) > 1)),
    "seq_rs_hl_rsi_10d_sort": ("Seq: RS->HL->RSI (10d) + Sortino>0", lambda df: _seq_rs_hl_rsi(df, 10) & (_rolling_sortino(df) > 0)),
    "seq_rs_hl_rsi_20d_omega": ("Seq: RS->HL->RSI (20d) + Omega>1", lambda df: _seq_rs_hl_rsi(df, 20) & (_rolling_omega(df) > 1)),
    "seq_rs_hl_rsi_20d_sort": ("Seq: RS->HL->RSI (20d) + Sortino>0", lambda df: _seq_rs_hl_rsi(df, 20) & (_rolling_sortino(df) > 0)),
    # Sortino<0 + RSI-Sort cross + RSI cross (sequential)
    "seq_sort_neg_rs_rsi_10d": ("Seq: Sortino<0 -> RS Cross -> RSI Cross (10d)", lambda df: (
        _seq_sort_neg_rs_rsi(df, 10)
    )),
    "seq_sort_neg_rs_rsi_20d": ("Seq: Sortino<0 -> RS Cross -> RSI Cross (20d)", lambda df: (
        _seq_sort_neg_rs_rsi(df, 20)
    )),
    "seq_sort_neg_rs_rsi_10d_omega": ("Seq: Sort<0->RS->RSI (10d) + Omega>1", lambda df: (
        _seq_sort_neg_rs_rsi(df, 10) & (_rolling_omega(df) > 1)
    )),
    "seq_sort_neg_rs_rsi_10d_hl": ("Seq: Sort<0->RS->RSI (10d) + Higher Low", lambda df: (
        _seq_sort_neg_rs_rsi(df, 10) & _higher_low(df)
    )),
    # RSI<20 + Sortino<0 then volume spike up with RSI crossover
    "seq_rsi20_sort_neg_vol_rsi": ("Seq: RSI<20+Sort<0 -> VolSpike+RSI Cross (10d)", lambda df: (
        _seq_oversold_vol_rsi(df, 10)
    )),
    "seq_rsi20_sort_neg_vol_rsi_20d": ("Seq: RSI<20+Sort<0 -> VolSpike+RSI Cross (20d)", lambda df: (
        _seq_oversold_vol_rsi(df, 20)
    )),
    "seq_rsi30_sort_neg_vol_rsi": ("Seq: RSI<30+Sort<0 -> VolSpike+RSI Cross (10d)", lambda df: (
        _seq_oversold_vol_rsi(df, 10, rsi_threshold=30)
    )),
    # More sequential combos
    "seq_sort_neg_rs_hl_rsi_10d": ("Seq: Sort<0->RS Cross->HL->RSI Cross (10d)", lambda df: (
        _seq_sort_neg_rs_rsi(df, 10) & _higher_low(df)
    )),
    "seq_rsi20_vol_hl_rsi": ("Seq: RSI<20 -> VolSpike -> Higher Low -> RSI Cross", lambda df: (
        _seq_oversold_vol_rsi(df, 20, rsi_threshold=20) & _higher_low(df)
    )),
    "seq_rsi20_sort_neg_pocket": ("Seq: RSI<20+Sort<0 -> Pocket Pivot (10d)", lambda df: (
        _seq_oversold_trigger(df, 10, 20, _pocket_pivot)
    )),
    "seq_rsi30_sort_neg_obv_div": ("Seq: RSI<30+Sort<0 -> OBV Divergence (10d)", lambda df: (
        _seq_oversold_trigger(df, 10, 30, _obv_divergence)
    )),
    "seq_rsi20_sort_neg_cmf": ("Seq: RSI<20+Sort<0 -> CMF>0 (10d)", lambda df: (
        _seq_oversold_trigger(df, 10, 20, _cmf_cross_zero)
    )),
    "seq_rsi30_vwap_rsi": ("Seq: RSI<30+Sort<0 -> VWAP Reclaim+RSI Cross (10d)", lambda df: (
        _seq_oversold_trigger(df, 10, 30, lambda d: _vwap_reclaim(d) & (ta.momentum.rsi(d["Close"], window=10) > ta.momentum.rsi(d["Close"], window=10).rolling(10).mean()))
    )),
    "seq_sort_neg_sma50_bounce": ("Seq: Sortino<0 -> SMA50 Bounce + RSI Cross (10d)", lambda df: (
        _seq_oversold_trigger(df, 10, 50, lambda d: _sma50_bounce_volume(d) & (ta.momentum.rsi(d["Close"], window=10) > ta.momentum.rsi(d["Close"], window=10).rolling(10).mean()), use_sortino_only=True)
    )),
    "seq_rsi20_tight_closes_breakout": ("Seq: RSI<20+Sort<0 -> Tight Closes -> Breakout", lambda df: (
        _seq_oversold_trigger(df, 20, 20, lambda d: _tight_closes(d) & (d["Close"] > d["Close"].shift(1)))
    )),
    "seq_rsi30_vol_dry_up_reversal": ("Seq: RSI<30+Sort<0 -> Vol Dry Up Reversal (20d)", lambda df: (
        _seq_oversold_trigger(df, 20, 30, _volume_dry_up_reversal)
    )),
    "seq_rsi20_ad_rising_rsi": ("Seq: RSI<20+Sort<0 -> A/D Rising + RSI Cross (10d)", lambda df: (
        _seq_oversold_trigger(df, 10, 20, lambda d: _ad_rising(d) & (ta.momentum.rsi(d["Close"], window=10) > ta.momentum.rsi(d["Close"], window=10).rolling(10).mean()))
    )),
    # ── Iterations on the seq winner (RSI<20+Sort<0 -> A/D Diverg + RSI cross, 10d) ──
    # Oversold-threshold sweep
    "seq_rsi15_ad_rsi": ("Seq: RSI<15+Sort<0 -> A/D Diverg + RSI Cross (10d)", lambda df: (
        _seq_oversold_trigger(df, 10, 15, lambda d: _ad_rising(d) & (ta.momentum.rsi(d["Close"], window=10) > ta.momentum.rsi(d["Close"], window=10).rolling(10).mean())))),
    "seq_rsi25_ad_rsi": ("Seq: RSI<25+Sort<0 -> A/D Diverg + RSI Cross (10d)", lambda df: (
        _seq_oversold_trigger(df, 10, 25, lambda d: _ad_rising(d) & (ta.momentum.rsi(d["Close"], window=10) > ta.momentum.rsi(d["Close"], window=10).rolling(10).mean())))),
    "seq_rsi30_ad_rsi": ("Seq: RSI<30+Sort<0 -> A/D Diverg + RSI Cross (10d)", lambda df: (
        _seq_oversold_trigger(df, 10, 30, lambda d: _ad_rising(d) & (ta.momentum.rsi(d["Close"], window=10) > ta.momentum.rsi(d["Close"], window=10).rolling(10).mean())))),
    # Window sweep (how long after oversold the A/D+RSI trigger may come)
    "seq_rsi20_ad_rsi_5d": ("Seq: RSI<20+Sort<0 -> A/D Diverg + RSI Cross (5d)", lambda df: (
        _seq_oversold_trigger(df, 5, 20, lambda d: _ad_rising(d) & (ta.momentum.rsi(d["Close"], window=10) > ta.momentum.rsi(d["Close"], window=10).rolling(10).mean())))),
    "seq_rsi20_ad_rsi_15d": ("Seq: RSI<20+Sort<0 -> A/D Diverg + RSI Cross (15d)", lambda df: (
        _seq_oversold_trigger(df, 15, 20, lambda d: _ad_rising(d) & (ta.momentum.rsi(d["Close"], window=10) > ta.momentum.rsi(d["Close"], window=10).rolling(10).mean())))),
    "seq_rsi20_ad_rsi_20d": ("Seq: RSI<20+Sort<0 -> A/D Diverg + RSI Cross (20d)", lambda df: (
        _seq_oversold_trigger(df, 20, 20, lambda d: _ad_rising(d) & (ta.momentum.rsi(d["Close"], window=10) > ta.momentum.rsi(d["Close"], window=10).rolling(10).mean())))),
    # Trigger variant: looser A/D "trend up" (ADL>SMA) instead of the strict divergence
    "seq_rsi20_adtrend_rsi": ("Seq: RSI<20+Sort<0 -> A/D TrendUp + RSI Cross (10d)", lambda df: (
        _seq_oversold_trigger(df, 10, 20, lambda d: _ad_trend_up(d) & (ta.momentum.rsi(d["Close"], window=10) > ta.momentum.rsi(d["Close"], window=10).rolling(10).mean())))),
    # Isolate: drop the RSI<20 level, keep only Sortino<0 as the oversold precondition
    "seq_sort_ad_rsi": ("Seq: Sortino<0 -> A/D Diverg + RSI Cross (10d)", lambda df: (
        _seq_oversold_trigger(df, 10, 20, lambda d: _ad_rising(d) & (ta.momentum.rsi(d["Close"], window=10) > ta.momentum.rsi(d["Close"], window=10).rolling(10).mean()), use_sortino_only=True))),
    # ── Grid fill: cross the best knobs (threshold x window x best triggers) ──
    "seq_rsi25_ad_rsi_15d": ("Seq: RSI<25+Sort<0 -> A/D Diverg + RSI Cross (15d)", lambda df: (
        _seq_oversold_trigger(df, 15, 25, lambda d: _ad_rising(d) & (ta.momentum.rsi(d["Close"], window=10) > ta.momentum.rsi(d["Close"], window=10).rolling(10).mean())))),
    "seq_rsi25_ad_rsi_20d": ("Seq: RSI<25+Sort<0 -> A/D Diverg + RSI Cross (20d)", lambda df: (
        _seq_oversold_trigger(df, 20, 25, lambda d: _ad_rising(d) & (ta.momentum.rsi(d["Close"], window=10) > ta.momentum.rsi(d["Close"], window=10).rolling(10).mean())))),
    "seq_rsi20_3ph_ad_omega_15d": ("Seq3: RSI<20+Sort<0 -> A/D Diverg -> Omega>1 (15d/10d)", lambda df: (
        _seq_oversold_trigger(df, 15, 20, lambda d: _seq_a_then_b(d, lambda x: _ad_rising(x), lambda x: (_rolling_omega(x) > 1) & (_rolling_omega(x).shift(1) <= 1), 10)))),
    "seq_rsi20_3ph_ad_omega_20d": ("Seq3: RSI<20+Sort<0 -> A/D Diverg -> Omega>1 (20d/10d)", lambda df: (
        _seq_oversold_trigger(df, 20, 20, lambda d: _seq_a_then_b(d, lambda x: _ad_rising(x), lambda x: (_rolling_omega(x) > 1) & (_rolling_omega(x).shift(1) <= 1), 10)))),
    "seq_rsi25_3ph_ad_omega": ("Seq3: RSI<25+Sort<0 -> A/D Diverg -> Omega>1 (10d/10d)", lambda df: (
        _seq_oversold_trigger(df, 10, 25, lambda d: _seq_a_then_b(d, lambda x: _ad_rising(x), lambda x: (_rolling_omega(x) > 1) & (_rolling_omega(x).shift(1) <= 1), 10)))),
    "seq_rsi20_3ph_ad_rsi_15d": ("Seq3: RSI<20+Sort<0 -> A/D Diverg -> RSI Cross (15d/10d)", lambda df: (
        _seq_oversold_trigger(df, 15, 20, lambda d: _seq_a_then_b(d, lambda x: _ad_rising(x), lambda x: _rsi_cross_series(x["Close"]), 10)))),
    # ── More sequential combos built on the winner (RSI<20+Sort<0 oversold theme) ──
    # 3-phase: oversold -> A/D divergence -> RSI cross (A/D and cross fully decoupled)
    "seq_rsi20_3ph_ad_rsi": ("Seq3: RSI<20+Sort<0 -> A/D Diverg -> RSI Cross (10d/10d)", lambda df: (
        _seq_oversold_trigger(df, 10, 20, lambda d: _seq_a_then_b(d, lambda x: _ad_rising(x), lambda x: _rsi_cross_series(x["Close"]), 10)))),
    # oversold -> VWAP reclaim + RSI cross
    "seq_rsi20_vwap_rsi": ("Seq: RSI<20+Sort<0 -> VWAP Reclaim + RSI Cross (10d)", lambda df: (
        _seq_oversold_trigger(df, 10, 20, lambda d: _vwap_reclaim(d) & (ta.momentum.rsi(d["Close"], window=10) > ta.momentum.rsi(d["Close"], window=10).rolling(10).mean())))),
    # oversold -> higher-low + RSI cross
    "seq_rsi20_hl_rsi": ("Seq: RSI<20+Sort<0 -> Higher Low + RSI Cross (10d)", lambda df: (
        _seq_oversold_trigger(df, 10, 20, lambda d: _higher_low(d) & (ta.momentum.rsi(d["Close"], window=10) > ta.momentum.rsi(d["Close"], window=10).rolling(10).mean())))),
    # 3-phase: oversold -> A/D divergence -> Omega crosses above 1
    "seq_rsi20_3ph_ad_omega": ("Seq3: RSI<20+Sort<0 -> A/D Diverg -> Omega>1 (10d/10d)", lambda df: (
        _seq_oversold_trigger(df, 10, 20, lambda d: _seq_a_then_b(d, lambda x: _ad_rising(x), lambda x: (_rolling_omega(x) > 1) & (_rolling_omega(x).shift(1) <= 1), 10)))),
    # oversold -> SMA50 bounce (with volume) + RSI cross
    "seq_rsi20_sma50_rsi": ("Seq: RSI<20+Sort<0 -> SMA50 Bounce + RSI Cross (10d)", lambda df: (
        _seq_oversold_trigger(df, 10, 20, lambda d: _sma50_bounce_volume(d) & (ta.momentum.rsi(d["Close"], window=10) > ta.momentum.rsi(d["Close"], window=10).rolling(10).mean())))),
    # ── A/D divergence THEN RSI cross (decoupled: divergence needs price flat/down,
    #    so it can't share the cross-up bar — make it sequential instead) ──
    "seq_ad_rsi_x_5d": ("Seq: A/D Divergence -> RSI Cross (5d)", lambda df: _seq_a_then_b(df,
        lambda d: _ad_rising(d), lambda d: _rsi_cross_series(d["Close"]), 5)),
    "seq_ad_rsi_x_10d": ("Seq: A/D Divergence -> RSI Cross (10d)", lambda df: _seq_a_then_b(df,
        lambda d: _ad_rising(d), lambda d: _rsi_cross_series(d["Close"]), 10)),
    # ── RSI<20 oversold -> RSI Cross family (built on the seq winner) ──
    # Pure version: does the A/D gate in the winner actually add anything?
    "seq_rsi20_rsi_10d": ("Seq: RSI<20+Sort<0 -> RSI Cross (10d)", lambda df: (
        _seq_oversold_trigger(df, 10, 20, lambda d: _rsi_cross_series(d["Close"])))),
    # Oversold-depth sweep (trigger = RSI cross up)
    "seq_rsi15_rsi_10d": ("Seq: RSI<15+Sort<0 -> RSI Cross (10d)", lambda df: (
        _seq_oversold_trigger(df, 10, 15, lambda d: _rsi_cross_series(d["Close"])))),
    "seq_rsi25_rsi_10d": ("Seq: RSI<25+Sort<0 -> RSI Cross (10d)", lambda df: (
        _seq_oversold_trigger(df, 10, 25, lambda d: _rsi_cross_series(d["Close"])))),
    "seq_rsi30_rsi_10d": ("Seq: RSI<30+Sort<0 -> RSI Cross (10d)", lambda df: (
        _seq_oversold_trigger(df, 10, 30, lambda d: _rsi_cross_series(d["Close"])))),
    # Window sweep (how long after oversold the cross may come)
    "seq_rsi20_rsi_5d": ("Seq: RSI<20+Sort<0 -> RSI Cross (5d)", lambda df: (
        _seq_oversold_trigger(df, 5, 20, lambda d: _rsi_cross_series(d["Close"])))),
    "seq_rsi20_rsi_20d": ("Seq: RSI<20+Sort<0 -> RSI Cross (20d)", lambda df: (
        _seq_oversold_trigger(df, 20, 20, lambda d: _rsi_cross_series(d["Close"])))),
    # Confirmation gates on the RSI cross
    "seq_rsi20_rsi_omega_10d": ("Seq: RSI<20+Sort<0 -> RSI Cross + Omega>1 (10d)", lambda df: (
        _seq_oversold_trigger(df, 10, 20, lambda d: _rsi_cross_series(d["Close"]) & (_rolling_omega(d) > 1)))),
    "seq_rsi20_rsi_vol_10d": ("Seq: RSI<20+Sort<0 -> RSI Cross + Vol>Avg (10d)", lambda df: (
        _seq_oversold_trigger(df, 10, 20, lambda d: _rsi_cross_series(d["Close"]) & (d["Volume"] > d["Volume"].rolling(20).mean())))),
    # ── Missing sequential: RSI-Sort cross -> Omega cross ──
    "seq_rs_omega_5d": ("Seq: RSI-Sort Cross -> Omega>1 (5d)", lambda df: _seq_a_then_b(df,
        lambda d: _rsi_of_sortino(d).pipe(lambda r: (r > r.rolling(14).mean()) & (r.shift(1) <= r.rolling(14).mean().shift(1))),
        lambda d: (_rolling_omega(d) > 1) & (_rolling_omega(d).shift(1) <= 1), 5)),
    "seq_rs_omega_10d": ("Seq: RSI-Sort Cross -> Omega>1 (10d)", lambda df: _seq_a_then_b(df,
        lambda d: _rsi_of_sortino(d).pipe(lambda r: (r > r.rolling(14).mean()) & (r.shift(1) <= r.rolling(14).mean().shift(1))),
        lambda d: (_rolling_omega(d) > 1) & (_rolling_omega(d).shift(1) <= 1), 10)),
    # ── Missing sequential: Omega cross -> RSI cross ──
    "seq_omega_rsi_5d": ("Seq: Omega>1 -> RSI Cross (5d)", lambda df: _seq_a_then_b(df,
        lambda d: (_rolling_omega(d) > 1) & (_rolling_omega(d).shift(1) <= 1),
        lambda d: (ta.momentum.rsi(d["Close"], window=10) > ta.momentum.rsi(d["Close"], window=10).rolling(10).mean()) & (ta.momentum.rsi(d["Close"], window=10).shift(1) <= ta.momentum.rsi(d["Close"], window=10).rolling(10).mean().shift(1)), 5)),
    "seq_omega_rsi_10d": ("Seq: Omega>1 -> RSI Cross (10d)", lambda df: _seq_a_then_b(df,
        lambda d: (_rolling_omega(d) > 1) & (_rolling_omega(d).shift(1) <= 1),
        lambda d: (ta.momentum.rsi(d["Close"], window=10) > ta.momentum.rsi(d["Close"], window=10).rolling(10).mean()) & (ta.momentum.rsi(d["Close"], window=10).shift(1) <= ta.momentum.rsi(d["Close"], window=10).rolling(10).mean().shift(1)), 10)),
    # ── Missing sequential: Sortino>0 -> RSI cross ──
    "seq_sort_pos_rsi_5d": ("Seq: Sortino>0 -> RSI Cross (5d)", lambda df: _seq_a_then_b(df,
        lambda d: (_rolling_sortino(d) > 0) & (_rolling_sortino(d).shift(1) <= 0),
        lambda d: (ta.momentum.rsi(d["Close"], window=10) > ta.momentum.rsi(d["Close"], window=10).rolling(10).mean()) & (ta.momentum.rsi(d["Close"], window=10).shift(1) <= ta.momentum.rsi(d["Close"], window=10).rolling(10).mean().shift(1)), 5)),
    "seq_sort_pos_rsi_10d": ("Seq: Sortino>0 -> RSI Cross (10d)", lambda df: _seq_a_then_b(df,
        lambda d: (_rolling_sortino(d) > 0) & (_rolling_sortino(d).shift(1) <= 0),
        lambda d: (ta.momentum.rsi(d["Close"], window=10) > ta.momentum.rsi(d["Close"], window=10).rolling(10).mean()) & (ta.momentum.rsi(d["Close"], window=10).shift(1) <= ta.momentum.rsi(d["Close"], window=10).rolling(10).mean().shift(1)), 10)),
    # ── Missing sequential: Higher Low -> RSI-Sort cross ──
    "seq_hl_rs_5d": ("Seq: Higher Low -> RSI-Sort Cross (5d)", lambda df: _seq_a_then_b(df,
        lambda d: _higher_low(d),
        lambda d: _rsi_of_sortino(d).pipe(lambda r: (r > r.rolling(14).mean()) & (r.shift(1) <= r.rolling(14).mean().shift(1))), 5)),
    "seq_hl_rs_10d": ("Seq: Higher Low -> RSI-Sort Cross (10d)", lambda df: _seq_a_then_b(df,
        lambda d: _higher_low(d),
        lambda d: _rsi_of_sortino(d).pipe(lambda r: (r > r.rolling(14).mean()) & (r.shift(1) <= r.rolling(14).mean().shift(1))), 10)),
    # ── Missing sequential: RSI cross -> Higher Low ──
    "seq_rsi_hl_5d": ("Seq: RSI Cross -> Higher Low (5d)", lambda df: _seq_a_then_b(df,
        lambda d: (ta.momentum.rsi(d["Close"], window=10) > ta.momentum.rsi(d["Close"], window=10).rolling(10).mean()) & (ta.momentum.rsi(d["Close"], window=10).shift(1) <= ta.momentum.rsi(d["Close"], window=10).rolling(10).mean().shift(1)),
        lambda d: _higher_low(d), 5)),
    "seq_rsi_hl_10d": ("Seq: RSI Cross -> Higher Low (10d)", lambda df: _seq_a_then_b(df,
        lambda d: (ta.momentum.rsi(d["Close"], window=10) > ta.momentum.rsi(d["Close"], window=10).rolling(10).mean()) & (ta.momentum.rsi(d["Close"], window=10).shift(1) <= ta.momentum.rsi(d["Close"], window=10).rolling(10).mean().shift(1)),
        lambda d: _higher_low(d), 10)),
    # ── Missing sequential: OBV div -> RSI cross ──
    "seq_obv_rsi_10d": ("Seq: OBV Divergence -> RSI Cross (10d)", lambda df: _seq_a_then_b(df,
        lambda d: _obv_divergence(d),
        lambda d: (ta.momentum.rsi(d["Close"], window=10) > ta.momentum.rsi(d["Close"], window=10).rolling(10).mean()) & (ta.momentum.rsi(d["Close"], window=10).shift(1) <= ta.momentum.rsi(d["Close"], window=10).rolling(10).mean().shift(1)), 10)),
    # ── Missing sequential: CMF>0 -> RSI cross ──
    "seq_cmf_rsi_10d": ("Seq: CMF>0 -> RSI Cross (10d)", lambda df: _seq_a_then_b(df,
        lambda d: _cmf_cross_zero(d),
        lambda d: (ta.momentum.rsi(d["Close"], window=10) > ta.momentum.rsi(d["Close"], window=10).rolling(10).mean()) & (ta.momentum.rsi(d["Close"], window=10).shift(1) <= ta.momentum.rsi(d["Close"], window=10).rolling(10).mean().shift(1)), 10)),
    # ── Missing sequential: Pocket Pivot -> RSI cross ──
    "seq_pocket_rsi_10d": ("Seq: Pocket Pivot -> RSI Cross (10d)", lambda df: _seq_a_then_b(df,
        lambda d: _pocket_pivot(d),
        lambda d: (ta.momentum.rsi(d["Close"], window=10) > ta.momentum.rsi(d["Close"], window=10).rolling(10).mean()) & (ta.momentum.rsi(d["Close"], window=10).shift(1) <= ta.momentum.rsi(d["Close"], window=10).rolling(10).mean().shift(1)), 10)),
    # ── Missing sequential: MACD cross -> RSI cross ──
    "seq_macd_rsi_5d": ("Seq: MACD Cross Up -> RSI Cross (5d)", lambda df: _seq_a_then_b(df,
        lambda d: sig_macd_cross_up(d),
        lambda d: (ta.momentum.rsi(d["Close"], window=10) > ta.momentum.rsi(d["Close"], window=10).rolling(10).mean()) & (ta.momentum.rsi(d["Close"], window=10).shift(1) <= ta.momentum.rsi(d["Close"], window=10).rolling(10).mean().shift(1)), 5)),
    "seq_macd_rsi_10d": ("Seq: MACD Cross Up -> RSI Cross (10d)", lambda df: _seq_a_then_b(df,
        lambda d: sig_macd_cross_up(d),
        lambda d: (ta.momentum.rsi(d["Close"], window=10) > ta.momentum.rsi(d["Close"], window=10).rolling(10).mean()) & (ta.momentum.rsi(d["Close"], window=10).shift(1) <= ta.momentum.rsi(d["Close"], window=10).rolling(10).mean().shift(1)), 10)),
    # ── Missing sequential: Gap down -> RSI cross ──
    "seq_gap_dn_rsi_5d": ("Seq: Gap Down -> RSI Cross (5d)", lambda df: _seq_a_then_b(df,
        lambda d: sig_gap_down(d),
        lambda d: (ta.momentum.rsi(d["Close"], window=10) > ta.momentum.rsi(d["Close"], window=10).rolling(10).mean()) & (ta.momentum.rsi(d["Close"], window=10).shift(1) <= ta.momentum.rsi(d["Close"], window=10).rolling(10).mean().shift(1)), 5)),
    "seq_gap_dn_rsi_10d": ("Seq: Gap Down -> RSI Cross (10d)", lambda df: _seq_a_then_b(df,
        lambda d: sig_gap_down(d),
        lambda d: (ta.momentum.rsi(d["Close"], window=10) > ta.momentum.rsi(d["Close"], window=10).rolling(10).mean()) & (ta.momentum.rsi(d["Close"], window=10).shift(1) <= ta.momentum.rsi(d["Close"], window=10).rolling(10).mean().shift(1)), 10)),
    # ── Missing sequential: Volume spike -> RSI-Sort cross ──
    "seq_vol_rs_10d": ("Seq: Volume Spike -> RSI-Sort Cross (10d)", lambda df: _seq_a_then_b(df,
        lambda d: sig_vol_spike_up(d),
        lambda d: _rsi_of_sortino(d).pipe(lambda r: (r > r.rolling(14).mean()) & (r.shift(1) <= r.rolling(14).mean().shift(1))), 10)),
    # ── Missing sequential: RSI<30 -> RSI-Sort cross ──
    "seq_rsi30_rs_10d": ("Seq: RSI<30 -> RSI-Sort Cross (10d)", lambda df: _seq_a_then_b(df,
        lambda d: (ta.momentum.rsi(d["Close"], window=10) < 30) & (ta.momentum.rsi(d["Close"], window=10).shift(1) >= 30),
        lambda d: _rsi_of_sortino(d).pipe(lambda r: (r > r.rolling(14).mean()) & (r.shift(1) <= r.rolling(14).mean().shift(1))), 10)),
    "seq_rsi30_rs_20d": ("Seq: RSI<30 -> RSI-Sort Cross (20d)", lambda df: _seq_a_then_b(df,
        lambda d: (ta.momentum.rsi(d["Close"], window=10) < 30) & (ta.momentum.rsi(d["Close"], window=10).shift(1) >= 30),
        lambda d: _rsi_of_sortino(d).pipe(lambda r: (r > r.rolling(14).mean()) & (r.shift(1) <= r.rolling(14).mean().shift(1))), 20)),
    # ── Missing sequential: Bollinger lower -> RSI cross ──
    "seq_boll_rsi_10d": ("Seq: Touch Lower Boll -> RSI Cross (10d)", lambda df: _seq_a_then_b(df,
        lambda d: sig_boll_lower(d),
        lambda d: (ta.momentum.rsi(d["Close"], window=10) > ta.momentum.rsi(d["Close"], window=10).rolling(10).mean()) & (ta.momentum.rsi(d["Close"], window=10).shift(1) <= ta.momentum.rsi(d["Close"], window=10).rolling(10).mean().shift(1)), 10)),
    # ── Missing sequential: SMA20 cross -> RSI-Sort cross ──
    "seq_sma20_rs_10d": ("Seq: Price Cross SMA20 -> RSI-Sort Cross (10d)", lambda df: _seq_a_then_b(df,
        lambda d: sig_price_cross_sma20_up(d),
        lambda d: _rsi_of_sortino(d).pipe(lambda r: (r > r.rolling(14).mean()) & (r.shift(1) <= r.rolling(14).mean().shift(1))), 10)),
    # ── Both oversold then both cross (different windows) ──
    "seq_both_os_cross_5d": ("Seq: RS<30+RSI<30 -> Both Cross SMA (5d)", lambda df: _seq_both_os_then_cross(df, 5)),
    "seq_both_os_cross_30d": ("Seq: RS<30+RSI<30 -> Both Cross SMA (30d)", lambda df: _seq_both_os_then_cross(df, 30)),
    # ── Triple oversold recovery combos ──
    # Full combo: RSI<30 + Sortino<0 + RS<30 -> RSI cross + RS cross + Higher Low
    "seq_triple_os_full_10d": ("Seq: RSI<30+Sort<0+RS<30 -> RSI×+RS×+HL (10d)", lambda df: _seq_triple_os_recovery(df, 10)),
    "seq_triple_os_full_20d": ("Seq: RSI<30+Sort<0+RS<30 -> RSI×+RS×+HL (20d)", lambda df: _seq_triple_os_recovery(df, 20)),
    "seq_triple_os_full_30d": ("Seq: RSI<30+Sort<0+RS<30 -> RSI×+RS×+HL (30d)", lambda df: _seq_triple_os_recovery(df, 30)),
    # Without Higher Low
    "seq_triple_os_2x_10d": ("Seq: RSI<30+Sort<0+RS<30 -> RSI×+RS× (10d)", lambda df: _seq_triple_os_recovery(df, 10, need_hl=False)),
    "seq_triple_os_2x_20d": ("Seq: RSI<30+Sort<0+RS<30 -> RSI×+RS× (20d)", lambda df: _seq_triple_os_recovery(df, 20, need_hl=False)),
    # RSI cross + Higher Low only (no RS cross)
    "seq_triple_os_rsi_hl_10d": ("Seq: RSI<30+Sort<0+RS<30 -> RSI×+HL (10d)", lambda df: _seq_triple_os_recovery(df, 10, need_rs_x=False)),
    "seq_triple_os_rsi_hl_20d": ("Seq: RSI<30+Sort<0+RS<30 -> RSI×+HL (20d)", lambda df: _seq_triple_os_recovery(df, 20, need_rs_x=False)),
    # RS cross + Higher Low only (no RSI cross)
    "seq_triple_os_rs_hl_10d": ("Seq: RSI<30+Sort<0+RS<30 -> RS×+HL (10d)", lambda df: _seq_triple_os_recovery(df, 10, need_rsi_x=False)),
    "seq_triple_os_rs_hl_20d": ("Seq: RSI<30+Sort<0+RS<30 -> RS×+HL (20d)", lambda df: _seq_triple_os_recovery(df, 20, need_rsi_x=False)),
    # RSI cross only
    "seq_triple_os_rsi_10d": ("Seq: RSI<30+Sort<0+RS<30 -> RSI Cross (10d)", lambda df: _seq_triple_os_recovery(df, 10, need_rs_x=False, need_hl=False)),
    "seq_triple_os_rsi_20d": ("Seq: RSI<30+Sort<0+RS<30 -> RSI Cross (20d)", lambda df: _seq_triple_os_recovery(df, 20, need_rs_x=False, need_hl=False)),
    # RS cross only
    "seq_triple_os_rs_10d": ("Seq: RSI<30+Sort<0+RS<30 -> RS Cross (10d)", lambda df: _seq_triple_os_recovery(df, 10, need_rsi_x=False, need_hl=False)),
    "seq_triple_os_rs_20d": ("Seq: RSI<30+Sort<0+RS<30 -> RS Cross (20d)", lambda df: _seq_triple_os_recovery(df, 20, need_rsi_x=False, need_hl=False)),
    # Higher Low only
    "seq_triple_os_hl_10d": ("Seq: RSI<30+Sort<0+RS<30 -> Higher Low (10d)", lambda df: _seq_triple_os_recovery(df, 10, need_rsi_x=False, need_rs_x=False)),
    "seq_triple_os_hl_20d": ("Seq: RSI<30+Sort<0+RS<30 -> Higher Low (20d)", lambda df: _seq_triple_os_recovery(df, 20, need_rsi_x=False, need_rs_x=False)),
    # Looser thresholds: RSI<40 + RS<40
    "seq_os40_full_10d": ("Seq: RSI<40+Sort<0+RS<40 -> RSI×+RS×+HL (10d)", lambda df: _seq_triple_os_recovery(df, 10, rsi_thresh=40, rs_thresh=40)),
    "seq_os40_full_20d": ("Seq: RSI<40+Sort<0+RS<40 -> RSI×+RS×+HL (20d)", lambda df: _seq_triple_os_recovery(df, 20, rsi_thresh=40, rs_thresh=40)),
    "seq_os40_2x_10d": ("Seq: RSI<40+Sort<0+RS<40 -> RSI×+RS× (10d)", lambda df: _seq_triple_os_recovery(df, 10, rsi_thresh=40, rs_thresh=40, need_hl=False)),
    "seq_os40_2x_20d": ("Seq: RSI<40+Sort<0+RS<40 -> RSI×+RS× (20d)", lambda df: _seq_triple_os_recovery(df, 20, rsi_thresh=40, rs_thresh=40, need_hl=False)),
    # Tighter: RSI<20 + RS<20
    "seq_os20_full_10d": ("Seq: RSI<20+Sort<0+RS<20 -> RSI×+RS×+HL (10d)", lambda df: _seq_triple_os_recovery(df, 10, rsi_thresh=20, rs_thresh=20)),
    "seq_os20_full_20d": ("Seq: RSI<20+Sort<0+RS<20 -> RSI×+RS×+HL (20d)", lambda df: _seq_triple_os_recovery(df, 20, rsi_thresh=20, rs_thresh=20)),
    "seq_os20_full_30d": ("Seq: RSI<20+Sort<0+RS<20 -> RSI×+RS×+HL (30d)", lambda df: _seq_triple_os_recovery(df, 30, rsi_thresh=20, rs_thresh=20)),
    "seq_os20_2x_20d": ("Seq: RSI<20+Sort<0+RS<20 -> RSI×+RS× (20d)", lambda df: _seq_triple_os_recovery(df, 20, rsi_thresh=20, rs_thresh=20, need_hl=False)),
    # ── Double oversold (without Sortino) combos ──
    "seq_rsi30_rs30_rsi_x_10d": ("Seq: RSI<30+RS<30 -> RSI Cross (10d)", lambda df: _seq_a_then_b(df,
        lambda d: (ta.momentum.rsi(d["Close"], window=10) < 30) & (_rsi_of_sortino(d) < 30),
        lambda d: (ta.momentum.rsi(d["Close"], window=10) > ta.momentum.rsi(d["Close"], window=10).rolling(10).mean()) & (ta.momentum.rsi(d["Close"], window=10).shift(1) <= ta.momentum.rsi(d["Close"], window=10).rolling(10).mean().shift(1)), 10)),
    "seq_rsi30_rs30_both_x_10d": ("Seq: RSI<30+RS<30 -> RSI×+RS× (10d)", lambda df: _seq_both_os_then_cross(df, 10)),
    "seq_rsi30_rs30_both_x_hl_20d": ("Seq: RSI<30+RS<30 -> RSI×+RS×+HL (20d)", lambda df: (
        _seq_both_os_then_cross(df, 20) & _higher_low(df))),
    # ── Higher Low / Lower High (structure) ──
    "higher_low": ("Higher Local Low", lambda df: _higher_low(df)),
    "lower_high": ("Lower Local High", lambda df: _lower_high(df)),
    "higher_low_rsi_x": ("Higher Low + RSI Cross SMA", lambda df: (
        _higher_low(df) &
        (ta.momentum.rsi(df["Close"], window=10) > ta.momentum.rsi(df["Close"], window=10).rolling(10).mean()) &
        (ta.momentum.rsi(df["Close"], window=10).shift(1) <= ta.momentum.rsi(df["Close"], window=10).rolling(10).mean().shift(1))
    )),
    "higher_low_omega_gt1": ("Higher Low + Omega>1", lambda df: (
        _higher_low(df) & (_rolling_omega(df) > 1)
    )),
    "higher_low_sort_pos": ("Higher Low + Sortino>0", lambda df: (
        _higher_low(df) & (_rolling_sortino(df) > 0)
    )),
    "higher_low_rsi_x_omega": ("Higher Low + RSI Cross + Omega>1", lambda df: (
        _higher_low(df) &
        (ta.momentum.rsi(df["Close"], window=10) > ta.momentum.rsi(df["Close"], window=10).rolling(10).mean()) &
        (ta.momentum.rsi(df["Close"], window=10).shift(1) <= ta.momentum.rsi(df["Close"], window=10).rolling(10).mean().shift(1)) &
        (_rolling_omega(df) > 1)
    )),
    # ── RSI of Sortino (meta-indicator) ──
    "rsi_of_sortino_x_up": ("RSI(14) of Sortino Cross SMA Up", lambda df: (
        _rsi_of_sortino(df).pipe(lambda r: (r > r.rolling(14).mean()) & (r.shift(1) <= r.rolling(14).mean().shift(1)))
    )),
    "rsi_of_sortino_x_up_b50": ("RSI of Sortino Cross SMA <50", lambda df: (
        _rsi_of_sortino(df).pipe(lambda r: (r > r.rolling(14).mean()) & (r.shift(1) <= r.rolling(14).mean().shift(1)) & (r.shift(1) < 50))
    )),
    "rsi_of_sortino_x_dn": ("RSI of Sortino Cross SMA Down", lambda df: (
        _rsi_of_sortino(df).pipe(lambda r: (r < r.rolling(14).mean()) & (r.shift(1) >= r.rolling(14).mean().shift(1)))
    )),
    "rsi_of_sortino_x_up_b30": ("RSI of Sortino Cross SMA (from <30)", lambda df: (
        _rsi_of_sortino(df).pipe(lambda r: (r > r.rolling(14).mean()) & (r.shift(1) <= r.rolling(14).mean().shift(1)) & (r.rolling(14).min().shift(1) < 30))
    )),
    "rsi_of_sortino_x_up_b20": ("RSI of Sortino Cross SMA (from <20)", lambda df: (
        _rsi_of_sortino(df).pipe(lambda r: (r > r.rolling(14).mean()) & (r.shift(1) <= r.rolling(14).mean().shift(1)) & (r.rolling(14).min().shift(1) < 20))
    )),
    "rsi_of_sortino_x50_up": ("RSI of Sortino Cross 50 Up", lambda df: (
        _rsi_of_sortino(df).pipe(lambda r: (r > 50) & (r.shift(1) <= 50))
    )),
    "rsi_of_sortino_x50_dn": ("RSI of Sortino Cross 50 Down", lambda df: (
        _rsi_of_sortino(df).pipe(lambda r: (r < 50) & (r.shift(1) >= 50))
    )),
    "rsi_of_sortino_x30_up": ("RSI of Sortino Cross 30 Up (recovery)", lambda df: (
        _rsi_of_sortino(df).pipe(lambda r: (r > 30) & (r.shift(1) <= 30))
    )),
    "rsi_of_sortino_x70_dn": ("RSI of Sortino Cross 70 Down", lambda df: (
        _rsi_of_sortino(df).pipe(lambda r: (r < 70) & (r.shift(1) >= 70))
    )),
    "rsi_of_sortino_b30_rsi_x": ("RSI of Sortino <30 + RSI Cross SMA", lambda df: (
        (_rsi_of_sortino(df) < 30) &
        (ta.momentum.rsi(df["Close"], window=10) > ta.momentum.rsi(df["Close"], window=10).rolling(10).mean()) &
        (ta.momentum.rsi(df["Close"], window=10).shift(1) <= ta.momentum.rsi(df["Close"], window=10).rolling(10).mean().shift(1))
    )),
    "rsi_of_sortino_b30_omega": ("RSI of Sortino <30 + Omega>1", lambda df: (
        (_rsi_of_sortino(df) < 30) & (_rsi_of_sortino(df).shift(1) >= 30) & (_rolling_omega(df) > 1)
    )),
    "rsi_of_sortino_recover_50": ("RSI of Sortino Recovers from <30 to >50", lambda df: (
        (_rsi_of_sortino(df) > 50) & (_rsi_of_sortino(df).shift(1) <= 50) & (_rsi_of_sortino(df).rolling(14).min().shift(1) < 30)
    )),
    "rsi_of_sortino_os": ("RSI of Sortino < 40", lambda df: (
        _rsi_of_sortino(df).pipe(lambda r: (r < 40) & (r.shift(1) >= 40))
    )),
    "rsi_of_sortino_ob": ("RSI of Sortino > 60", lambda df: (
        _rsi_of_sortino(df).pipe(lambda r: (r > 60) & (r.shift(1) <= 60))
    )),
    # ── Sequential: RS Cross Up from <30 + confirmation ──
    "seq_rs30_rsi_x_5d": ("Seq: RS Cross from <30 -> RSI Cross (5d)", lambda df: _seq_a_then_b(df,
        lambda d: _rsi_of_sortino(d).pipe(lambda r: (r > r.rolling(14).mean()) & (r.shift(1) <= r.rolling(14).mean().shift(1)) & (r.rolling(14).min().shift(1) < 30)),
        lambda d: (ta.momentum.rsi(d["Close"], window=10) > ta.momentum.rsi(d["Close"], window=10).rolling(14).mean()) & (ta.momentum.rsi(d["Close"], window=10).shift(1) <= ta.momentum.rsi(d["Close"], window=10).rolling(14).mean().shift(1)), 5)),
    "seq_rs30_rsi_x_10d": ("Seq: RS Cross from <30 -> RSI Cross (10d)", lambda df: _seq_a_then_b(df,
        lambda d: _rsi_of_sortino(d).pipe(lambda r: (r > r.rolling(14).mean()) & (r.shift(1) <= r.rolling(14).mean().shift(1)) & (r.rolling(14).min().shift(1) < 30)),
        lambda d: (ta.momentum.rsi(d["Close"], window=10) > ta.momentum.rsi(d["Close"], window=10).rolling(14).mean()) & (ta.momentum.rsi(d["Close"], window=10).shift(1) <= ta.momentum.rsi(d["Close"], window=10).rolling(14).mean().shift(1)), 10)),
    "seq_rs30_omega_5d": ("Seq: RS Cross from <30 -> Omega>1 (5d)", lambda df: _seq_a_then_b(df,
        lambda d: _rsi_of_sortino(d).pipe(lambda r: (r > r.rolling(14).mean()) & (r.shift(1) <= r.rolling(14).mean().shift(1)) & (r.rolling(14).min().shift(1) < 30)),
        lambda d: (_rolling_omega(d) > 1) & (_rolling_omega(d).shift(1) <= 1), 5)),
    "seq_rs30_omega_10d": ("Seq: RS Cross from <30 -> Omega>1 (10d)", lambda df: _seq_a_then_b(df,
        lambda d: _rsi_of_sortino(d).pipe(lambda r: (r > r.rolling(14).mean()) & (r.shift(1) <= r.rolling(14).mean().shift(1)) & (r.rolling(14).min().shift(1) < 30)),
        lambda d: (_rolling_omega(d) > 1) & (_rolling_omega(d).shift(1) <= 1), 10)),
    "seq_rs30_sort_pos_5d": ("Seq: RS Cross from <30 -> Sortino>0 (5d)", lambda df: _seq_a_then_b(df,
        lambda d: _rsi_of_sortino(d).pipe(lambda r: (r > r.rolling(14).mean()) & (r.shift(1) <= r.rolling(14).mean().shift(1)) & (r.rolling(14).min().shift(1) < 30)),
        lambda d: (_rolling_sortino(d) > 0) & (_rolling_sortino(d).shift(1) <= 0), 5)),
    "seq_rs30_sort_pos_10d": ("Seq: RS Cross from <30 -> Sortino>0 (10d)", lambda df: _seq_a_then_b(df,
        lambda d: _rsi_of_sortino(d).pipe(lambda r: (r > r.rolling(14).mean()) & (r.shift(1) <= r.rolling(14).mean().shift(1)) & (r.rolling(14).min().shift(1) < 30)),
        lambda d: (_rolling_sortino(d) > 0) & (_rolling_sortino(d).shift(1) <= 0), 10)),
    "seq_rs30_hl_5d": ("Seq: RS Cross from <30 -> Higher Low (5d)", lambda df: _seq_a_then_b(df,
        lambda d: _rsi_of_sortino(d).pipe(lambda r: (r > r.rolling(14).mean()) & (r.shift(1) <= r.rolling(14).mean().shift(1)) & (r.rolling(14).min().shift(1) < 30)),
        lambda d: _higher_low(d), 5)),
    "seq_rs30_hl_10d": ("Seq: RS Cross from <30 -> Higher Low (10d)", lambda df: _seq_a_then_b(df,
        lambda d: _rsi_of_sortino(d).pipe(lambda r: (r > r.rolling(14).mean()) & (r.shift(1) <= r.rolling(14).mean().shift(1)) & (r.rolling(14).min().shift(1) < 30)),
        lambda d: _higher_low(d), 10)),
    "seq_rs30_rsi_x_hl_10d": ("Seq: RS Cross from <30 -> RSI Cross + HL (10d)", lambda df: _seq_a_then_b(df,
        lambda d: _rsi_of_sortino(d).pipe(lambda r: (r > r.rolling(14).mean()) & (r.shift(1) <= r.rolling(14).mean().shift(1)) & (r.rolling(14).min().shift(1) < 30)),
        lambda d: (ta.momentum.rsi(d["Close"], window=10) > ta.momentum.rsi(d["Close"], window=10).rolling(14).mean()) & _higher_low(d), 10)),
    "seq_rs30_rsi_x_omega_10d": ("Seq: RS Cross from <30 -> RSI Cross + Omega>1 (10d)", lambda df: _seq_a_then_b(df,
        lambda d: _rsi_of_sortino(d).pipe(lambda r: (r > r.rolling(14).mean()) & (r.shift(1) <= r.rolling(14).mean().shift(1)) & (r.rolling(14).min().shift(1) < 30)),
        lambda d: (ta.momentum.rsi(d["Close"], window=10) > ta.momentum.rsi(d["Close"], window=10).rolling(14).mean()) & (_rolling_omega(d) > 1), 10)),
    "seq_rs30_vol_spike_5d": ("Seq: RS Cross from <30 -> Volume Spike (5d)", lambda df: _seq_a_then_b(df,
        lambda d: _rsi_of_sortino(d).pipe(lambda r: (r > r.rolling(14).mean()) & (r.shift(1) <= r.rolling(14).mean().shift(1)) & (r.rolling(14).min().shift(1) < 30)),
        lambda d: (d["Volume"] > d["Volume"].rolling(20).mean() * 1.5) & (d["Close"] > d["Open"]), 5)),
    "seq_rs30_pocket_10d": ("Seq: RS Cross from <30 -> Pocket Pivot (10d)", lambda df: _seq_a_then_b(df,
        lambda d: _rsi_of_sortino(d).pipe(lambda r: (r > r.rolling(14).mean()) & (r.shift(1) <= r.rolling(14).mean().shift(1)) & (r.rolling(14).min().shift(1) < 30)),
        lambda d: _pocket_pivot(d), 10)),
    "seq_rs30_sma20_10d": ("Seq: RS Cross from <30 -> Price Above SMA20 (10d)", lambda df: _seq_a_then_b(df,
        lambda d: _rsi_of_sortino(d).pipe(lambda r: (r > r.rolling(14).mean()) & (r.shift(1) <= r.rolling(14).mean().shift(1)) & (r.rolling(14).min().shift(1) < 30)),
        lambda d: (d["Close"] > d["Close"].rolling(20).mean()) & (d["Close"].shift(1) <= d["Close"].rolling(20).mean().shift(1)), 10)),
    "seq_rs30_macd_10d": ("Seq: RS Cross from <30 -> MACD Cross Up (10d)", lambda df: _seq_a_then_b(df,
        lambda d: _rsi_of_sortino(d).pipe(lambda r: (r > r.rolling(14).mean()) & (r.shift(1) <= r.rolling(14).mean().shift(1)) & (r.rolling(14).min().shift(1) < 30)),
        lambda d: sig_macd_cross_up(d), 10)),
    "seq_rs30_obv_div_10d": ("Seq: RS Cross from <30 -> OBV Divergence (10d)", lambda df: _seq_a_then_b(df,
        lambda d: _rsi_of_sortino(d).pipe(lambda r: (r > r.rolling(14).mean()) & (r.shift(1) <= r.rolling(14).mean().shift(1)) & (r.rolling(14).min().shift(1) < 30)),
        lambda d: _obv_divergence(d), 10)),
    "seq_rs30_cmf_10d": ("Seq: RS Cross from <30 -> CMF>0 (10d)", lambda df: _seq_a_then_b(df,
        lambda d: _rsi_of_sortino(d).pipe(lambda r: (r > r.rolling(14).mean()) & (r.shift(1) <= r.rolling(14).mean().shift(1)) & (r.rolling(14).min().shift(1) < 30)),
        lambda d: _cmf_cross_zero(d), 10)),
    "seq_rs30_gap_up_5d": ("Seq: RS Cross from <30 -> Gap Up (5d)", lambda df: _seq_a_then_b(df,
        lambda d: _rsi_of_sortino(d).pipe(lambda r: (r > r.rolling(14).mean()) & (r.shift(1) <= r.rolling(14).mean().shift(1)) & (r.rolling(14).min().shift(1) < 30)),
        lambda d: sig_gap_up(d), 5)),
    "seq_rs30_3up_5d": ("Seq: RS Cross from <30 -> 3 Up Days (5d)", lambda df: _seq_a_then_b(df,
        lambda d: _rsi_of_sortino(d).pipe(lambda r: (r > r.rolling(14).mean()) & (r.shift(1) <= r.rolling(14).mean().shift(1)) & (r.rolling(14).min().shift(1) < 30)),
        lambda d: sig_3up(d), 5)),
    "seq_rs30_rsi_x_sort_pos_10d": ("Seq: RS Cross <30 -> RSI× + Sortino>0 (10d)", lambda df: _seq_a_then_b(df,
        lambda d: _rsi_of_sortino(d).pipe(lambda r: (r > r.rolling(14).mean()) & (r.shift(1) <= r.rolling(14).mean().shift(1)) & (r.rolling(14).min().shift(1) < 30)),
        lambda d: (ta.momentum.rsi(d["Close"], window=10) > ta.momentum.rsi(d["Close"], window=10).rolling(14).mean()) & (_rolling_sortino(d) > 0), 10)),
    "seq_rs30_rsi_x_omega_sort_10d": ("Seq: RS Cross <30 -> RSI× + Omega>1 + Sort>0 (10d)", lambda df: _seq_a_then_b(df,
        lambda d: _rsi_of_sortino(d).pipe(lambda r: (r > r.rolling(14).mean()) & (r.shift(1) <= r.rolling(14).mean().shift(1)) & (r.rolling(14).min().shift(1) < 30)),
        lambda d: (ta.momentum.rsi(d["Close"], window=10) > ta.momentum.rsi(d["Close"], window=10).rolling(14).mean()) & (_rolling_omega(d) > 1) & (_rolling_sortino(d) > 0), 10)),
    # ── Avg down-day return signals (mean of trailing-10 down-day returns, in %; NOT a capture ratio) ──
    "dn_low": ("Avg Dn-Day Ret Crosses < -0.5%", lambda df: _dn_capture_signal(df, 50, "below")),
    "dn_very_low": ("Avg Dn-Day Ret Turns Negative", lambda df: _dn_capture_signal(df, 0, "below")),
    "dn_high": ("Avg Dn-Day Ret Recovers > -1.5%", lambda df: _dn_capture_signal(df, 150, "above")),
    "dn_improving": ("Avg Dn-Day Ret Improving", lambda df: _dn_trending(df, "down")),
    "dn_worsening": ("Avg Dn-Day Ret Worsening", lambda df: _dn_trending(df, "up")),
    "dn_low_rsi_x": ("Avg Dn-Day Ret<-0.5% + RSI Cross SMA", lambda df: (
        _dn_capture_signal(df, 50, "below") &
        (ta.momentum.rsi(df["Close"], window=10) > ta.momentum.rsi(df["Close"], window=10).rolling(10).mean()) &
        (ta.momentum.rsi(df["Close"], window=10).shift(1) <= ta.momentum.rsi(df["Close"], window=10).rolling(10).mean().shift(1))
    )),
    "dn_low_omega": ("Avg Dn-Day Ret<-0.5% + Omega>1", lambda df: (
        _dn_capture_signal(df, 50, "below") & (_rolling_omega(df) > 1)
    )),
    "dn_low_sort_pos": ("Avg Dn-Day Ret<-0.5% + Sortino>0", lambda df: (
        _dn_capture_signal(df, 50, "below") & (_rolling_sortino(df) > 0)
    )),
    "dn_neg_rsi_x": ("Avg Dn-Day Ret<0 + RSI Cross SMA", lambda df: (
        _dn_capture_signal(df, 0, "below") &
        (ta.momentum.rsi(df["Close"], window=10) > ta.momentum.rsi(df["Close"], window=10).rolling(10).mean()) &
        (ta.momentum.rsi(df["Close"], window=10).shift(1) <= ta.momentum.rsi(df["Close"], window=10).rolling(10).mean().shift(1))
    )),
    "dn_neg_omega_sort": ("Avg Dn-Day Ret<0 + Omega>1 + Sortino>0", lambda df: (
        _dn_capture_signal(df, 0, "below") & (_rolling_omega(df) > 1) & (_rolling_sortino(df) > 0)
    )),
    "dn_improving_rsi_x": ("Avg Dn-Day Ret Improving + RSI Cross", lambda df: (
        _dn_trending(df, "down") &
        (ta.momentum.rsi(df["Close"], window=10) > ta.momentum.rsi(df["Close"], window=10).rolling(10).mean()) &
        (ta.momentum.rsi(df["Close"], window=10).shift(1) <= ta.momentum.rsi(df["Close"], window=10).rolling(10).mean().shift(1))
    )),
    "dn_improving_omega_sort": ("Avg Dn-Day Ret Improving + Omega>1 + Sortino>0", lambda df: (
        _dn_trending(df, "down") & (_rolling_omega(df) > 1) & (_rolling_sortino(df) > 0)
    )),
    "spread_pos": ("Up-Dn Ret Spread > 0.5%", lambda df: _updn_spread_signal(df, 50, "above")),
    "spread_turns_pos": ("Up-Dn Ret Spread Turns Positive", lambda df: _updn_spread_signal(df, 0, "above")),
    "spread_turns_pos_rsi_x": ("Up-Dn Spread Positive + RSI Cross", lambda df: (
        _updn_spread_signal(df, 0, "above") &
        (ta.momentum.rsi(df["Close"], window=10) > ta.momentum.rsi(df["Close"], window=10).rolling(14).mean())
    )),
    "spread_pos_rsi_x": ("Up-Dn Spread>0.5% + RSI Cross SMA", lambda df: (
        _updn_spread_signal(df, 50, "above") &
        (ta.momentum.rsi(df["Close"], window=10) > ta.momentum.rsi(df["Close"], window=10).rolling(10).mean()) &
        (ta.momentum.rsi(df["Close"], window=10).shift(1) <= ta.momentum.rsi(df["Close"], window=10).rolling(10).mean().shift(1))
    )),
    # ── Institutional Accumulation Patterns ──
    "obv_divergence": ("OBV Bullish Divergence", lambda df: _obv_divergence(df)),
    "ad_rising": ("A/D Line Rising + Price Flat", lambda df: _ad_rising(df)),
    "cmf_cross_zero": ("CMF Crosses Above Zero", lambda df: _cmf_cross_zero(df)),
    "vwap_reclaim": ("VWAP Reclaim", lambda df: _vwap_reclaim(df)),
    "vol_shrink_pullback": ("Volume Shrink Pullback + Green", lambda df: _vol_shrink_pullback(df)),
    "pocket_pivot": ("Pocket Pivot", lambda df: _pocket_pivot(df)),
    "tight_closes": ("Tight Closes Near High", lambda df: _tight_closes(df)),
    "vol_dry_up_reversal": ("Volume Dry Up Reversal", lambda df: _volume_dry_up_reversal(df)),
    "sma50_bounce_vol": ("SMA50 Bounce + Volume", lambda df: _sma50_bounce_volume(df)),
    "high_vol_narrow": ("High Volume Narrow Range", lambda df: _high_vol_narrow_range(df)),
    # Institutional + RSI combos
    "obv_div_rsi_x": ("OBV Divergence + RSI Cross", lambda df: (
        _obv_divergence(df) &
        (ta.momentum.rsi(df["Close"], window=10) > ta.momentum.rsi(df["Close"], window=10).rolling(10).mean()) &
        (ta.momentum.rsi(df["Close"], window=10).shift(1) <= ta.momentum.rsi(df["Close"], window=10).rolling(10).mean().shift(1))
    )),
    "cmf_pos_rsi_x": ("CMF>0 + RSI Cross", lambda df: (
        (_cmf(df) > 0) &
        (ta.momentum.rsi(df["Close"], window=10) > ta.momentum.rsi(df["Close"], window=10).rolling(10).mean()) &
        (ta.momentum.rsi(df["Close"], window=10).shift(1) <= ta.momentum.rsi(df["Close"], window=10).rolling(10).mean().shift(1))
    )),
    "pocket_pivot_rsi_x": ("Pocket Pivot + RSI Cross", lambda df: (
        _pocket_pivot(df) &
        (ta.momentum.rsi(df["Close"], window=10) > ta.momentum.rsi(df["Close"], window=10).rolling(10).mean())
    )),
    "vwap_reclaim_omega": ("VWAP Reclaim + Omega>1", lambda df: (
        _vwap_reclaim(df) & (_rolling_omega(df) > 1)
    )),
    "sma50_bounce_rsi_x": ("SMA50 Bounce + RSI Cross", lambda df: (
        _sma50_bounce_volume(df) &
        (ta.momentum.rsi(df["Close"], window=10) > ta.momentum.rsi(df["Close"], window=10).rolling(10).mean())
    )),
    "vol_dry_up_rsi_x": ("Vol Dry Up + RSI Cross", lambda df: (
        _volume_dry_up_reversal(df) &
        (ta.momentum.rsi(df["Close"], window=10) > ta.momentum.rsi(df["Close"], window=10).rolling(10).mean())
    )),
    # Institutional + Sortino/Omega combos
    "obv_div_sort_pos": ("OBV Divergence + Sortino>0", lambda df: _obv_divergence(df) & (_rolling_sortino(df) > 0)),
    "cmf_pos_omega_sort": ("CMF>0 + Omega>1 + Sortino>0", lambda df: (_cmf(df) > 0) & (_rolling_omega(df) > 1) & (_rolling_sortino(df) > 0)),
    "pocket_pivot_omega": ("Pocket Pivot + Omega>1", lambda df: _pocket_pivot(df) & (_rolling_omega(df) > 1)),
    "tight_closes_rsi_x": ("Tight Closes + RSI Cross", lambda df: (
        _tight_closes(df) &
        (ta.momentum.rsi(df["Close"], window=10) > ta.momentum.rsi(df["Close"], window=10).rolling(10).mean())
    )),
    "high_vol_narrow_rsi_x": ("High Vol Narrow + RSI Cross", lambda df: (
        _high_vol_narrow_range(df) &
        (ta.momentum.rsi(df["Close"], window=10) > ta.momentum.rsi(df["Close"], window=10).rolling(10).mean())
    )),
    # Institutional + Higher Low + RSI of Sortino
    "obv_div_hl_rsi": ("OBV Div + Higher Low + RSI Cross", lambda df: (
        _obv_divergence(df) & _higher_low(df) &
        (ta.momentum.rsi(df["Close"], window=10) > ta.momentum.rsi(df["Close"], window=10).rolling(10).mean())
    )),
    "pocket_pivot_hl": ("Pocket Pivot + Higher Low", lambda df: _pocket_pivot(df) & _higher_low(df)),
    "cmf_pos_hl_rsi": ("CMF>0 + Higher Low + RSI Cross", lambda df: (
        (_cmf(df) > 0) & _higher_low(df) &
        (ta.momentum.rsi(df["Close"], window=10) > ta.momentum.rsi(df["Close"], window=10).rolling(10).mean())
    )),
    # RSI of Sortino + regular RSI combo
    "rsi_sort_x_up_rsi_x_up": ("RSI of Sortino + RSI Both Cross Up", lambda df: (
        _rsi_of_sortino(df).pipe(lambda r: (r > r.rolling(14).mean()) & (r.shift(1) <= r.rolling(14).mean().shift(1))) &
        (ta.momentum.rsi(df["Close"], window=10) > ta.momentum.rsi(df["Close"], window=10).rolling(10).mean()) &
        (ta.momentum.rsi(df["Close"], window=10).shift(1) <= ta.momentum.rsi(df["Close"], window=10).rolling(10).mean().shift(1))
    )),
    # RSI of Sortino + Omega combos
    "rs_x_up_omega_gt1": ("RSI Sortino Cross Up + Omega>1", lambda df: (
        _rsi_of_sortino(df).pipe(lambda r: (r > r.rolling(14).mean()) & (r.shift(1) <= r.rolling(14).mean().shift(1))) &
        (_rolling_omega(df) > 1)
    )),
    "rs_x_up_b50_omega_gt1": ("RSI Sortino Cross <50 + Omega>1", lambda df: (
        _rsi_of_sortino(df).pipe(lambda r: (r > r.rolling(14).mean()) & (r.shift(1) <= r.rolling(14).mean().shift(1)) & (r.shift(1) < 50)) &
        (_rolling_omega(df) > 1)
    )),
    "rs_x_up_omega_lt1": ("RSI Sortino Cross Up + Omega<1", lambda df: (
        _rsi_of_sortino(df).pipe(lambda r: (r > r.rolling(14).mean()) & (r.shift(1) <= r.rolling(14).mean().shift(1))) &
        (_rolling_omega(df) < 1)
    )),
    # RSI of Sortino + regular RSI combos
    "rs_x_up_rsi_gt50": ("RSI Sortino Cross + RSI>50", lambda df: (
        _rsi_of_sortino(df).pipe(lambda r: (r > r.rolling(14).mean()) & (r.shift(1) <= r.rolling(14).mean().shift(1))) &
        (ta.momentum.rsi(df["Close"], window=10) > 50)
    )),
    "rs_x_up_rsi_lt50": ("RSI Sortino Cross + RSI<50", lambda df: (
        _rsi_of_sortino(df).pipe(lambda r: (r > r.rolling(14).mean()) & (r.shift(1) <= r.rolling(14).mean().shift(1))) &
        (ta.momentum.rsi(df["Close"], window=10) < 50)
    )),
    "rs_os_rsi_os": ("RSI Sortino <30 + RSI<30", lambda df: (
        _rsi_of_sortino(df).pipe(lambda r: (r < 30) & (r.shift(1) >= 30)) &
        (ta.momentum.rsi(df["Close"], window=10) < 30)
    )),
    "rs_os_rsi_os_both_cross": ("RS<30 + RSI<30 Then Both Cross Up", lambda df: _seq_both_os_then_cross(df)),
    "rs_os_rsi_os_both_cross_10d": ("RS<30 + RSI<30 Then Both Cross (10d)", lambda df: _seq_both_os_then_cross(df, window=10)),
    "rs_os_rsi_os_both_cross_20d": ("RS<30 + RSI<30 Then Both Cross (20d)", lambda df: _seq_both_os_then_cross(df, window=20)),
    # RSI of Sortino + Sortino value combos
    "rs_x_up_sort_pos": ("RSI Sortino Cross + Sortino>0", lambda df: (
        _rsi_of_sortino(df).pipe(lambda r: (r > r.rolling(14).mean()) & (r.shift(1) <= r.rolling(14).mean().shift(1))) &
        (_rolling_sortino(df) > 0)
    )),
    "rs_x_up_sort_neg": ("RSI Sortino Cross + Sortino<0", lambda df: (
        _rsi_of_sortino(df).pipe(lambda r: (r > r.rolling(14).mean()) & (r.shift(1) <= r.rolling(14).mean().shift(1))) &
        (_rolling_sortino(df) < 0)
    )),
    "rs_os_sort_neg": ("RSI Sortino <30 + Sortino<0", lambda df: (
        _rsi_of_sortino(df).pipe(lambda r: (r < 30) & (r.shift(1) >= 30)) &
        (_rolling_sortino(df) < 0)
    )),
    # RSI of Sortino + Sortino MA trending
    "rs_x_up_sort_ma5_up": ("RSI Sortino Cross + Sortino MA5 Up", lambda df: (
        _rsi_of_sortino(df).pipe(lambda r: (r > r.rolling(14).mean()) & (r.shift(1) <= r.rolling(14).mean().shift(1))) &
        _rolling_sortino(df).pipe(lambda s: s.rolling(5).mean() > s.rolling(5).mean().shift(1))
    )),
    "rs_x_up_sort_3x10_up": ("RSI Sortino Cross + Sortino 3x10 Up", lambda df: (
        _rsi_of_sortino(df).pipe(lambda r: (r > r.rolling(14).mean()) & (r.shift(1) <= r.rolling(14).mean().shift(1))) &
        _rolling_sortino(df).pipe(lambda s: (s.rolling(3).mean() > s.rolling(10).mean()) & (s.rolling(3).mean().shift(1) <= s.rolling(10).mean().shift(1)))
    )),
    # RSI of Sortino + Gap combos
    "rs_x_up_gap_dn": ("RSI Sortino Cross + Gap Down", lambda df: (
        _rsi_of_sortino(df).pipe(lambda r: (r > r.rolling(14).mean()) & (r.shift(1) <= r.rolling(14).mean().shift(1))) &
        ((df["Open"] - df["Low"].shift(1)) / df["Low"].shift(1) * 100 <= -0.5)
    )),
    "rs_os_gap_dn": ("RSI Sortino <30 + Gap Down", lambda df: (
        _rsi_of_sortino(df).pipe(lambda r: (r < 30) & (r.shift(1) >= 30)) &
        ((df["Open"] - df["Low"].shift(1)) / df["Low"].shift(1) * 100 <= -0.5)
    )),
    # RSI of Sortino + MACD
    "rs_x_up_macd_up": ("RSI Sortino Cross + MACD Bull", lambda df: (
        _rsi_of_sortino(df).pipe(lambda r: (r > r.rolling(14).mean()) & (r.shift(1) <= r.rolling(14).mean().shift(1))) &
        ta.trend.MACD(df["Close"]).macd().pipe(lambda m: m > ta.trend.MACD(df["Close"]).macd_signal())
    )),
    # RSI of Sortino + Volume
    "rs_x_up_vol_spike": ("RSI Sortino Cross + Volume Spike", lambda df: (
        _rsi_of_sortino(df).pipe(lambda r: (r > r.rolling(14).mean()) & (r.shift(1) <= r.rolling(14).mean().shift(1))) &
        (df["Volume"] > df["Volume"].rolling(20).mean() * 2)
    )),
    # Triple: RSI of Sortino + regular RSI + Omega
    "rs_x_rsi_x_omega": ("RSI Sortino + RSI Cross + Omega>1", lambda df: (
        _rsi_of_sortino(df).pipe(lambda r: (r > r.rolling(14).mean()) & (r.shift(1) <= r.rolling(14).mean().shift(1))) &
        (ta.momentum.rsi(df["Close"], window=10) > ta.momentum.rsi(df["Close"], window=10).rolling(10).mean()) &
        (ta.momentum.rsi(df["Close"], window=10).shift(1) <= ta.momentum.rsi(df["Close"], window=10).rolling(10).mean().shift(1)) &
        (_rolling_omega(df) > 1)
    )),
    # RSI of Sortino + Price above SMA
    "rs_x_up_above_sma20": ("RSI Sortino Cross + Price>SMA20", lambda df: (
        _rsi_of_sortino(df).pipe(lambda r: (r > r.rolling(14).mean()) & (r.shift(1) <= r.rolling(14).mean().shift(1))) &
        (df["Close"] > df["Close"].rolling(20).mean())
    )),
    "rs_x_up_below_sma20": ("RSI Sortino Cross + Price<SMA20", lambda df: (
        _rsi_of_sortino(df).pipe(lambda r: (r > r.rolling(14).mean()) & (r.shift(1) <= r.rolling(14).mean().shift(1))) &
        (df["Close"] < df["Close"].rolling(20).mean())
    )),
    # RSI of Sortino divergence from regular RSI
    "rs_up_rsi_dn": ("RSI Sortino Rising + RSI Falling", lambda df: (
        _rsi_of_sortino(df).pipe(lambda r: (r > r.shift(1)) & (r.shift(1) <= r.shift(2))) &
        (ta.momentum.rsi(df["Close"], window=10) < ta.momentum.rsi(df["Close"], window=10).shift(1))
    )),
    "rs_dn_rsi_up": ("RSI Sortino Falling + RSI Rising", lambda df: (
        _rsi_of_sortino(df).pipe(lambda r: (r < r.shift(1)) & (r.shift(1) >= r.shift(2))) &
        (ta.momentum.rsi(df["Close"], window=10) > ta.momentum.rsi(df["Close"], window=10).shift(1))
    )),
    # ── Triple weakness/strength combos ──
    # RSI drops below 50 + Omega<1 + Sortino<0
    "rsi_x50dn_omega_lt1_sort_neg": ("RSI Cross 50 Down + Omega<1 + Sortino<0", lambda df: (
        (ta.momentum.rsi(df["Close"], window=10) < 50) & (ta.momentum.rsi(df["Close"], window=10).shift(1) >= 50) &
        (_rolling_omega(df) < 1) & (_rolling_sortino(df) < 0)
    )),
    # RSI crosses above SMA from <50 + Omega<1 + Sortino<0 (crossover into weakness)
    "rsi_xsma_omega_lt1_sort_neg": ("RSI Cross SMA <50 + Omega<1 + Sortino<0", lambda df: (
        (ta.momentum.rsi(df["Close"], window=10) > ta.momentum.rsi(df["Close"], window=10).rolling(10).mean()) &
        (ta.momentum.rsi(df["Close"], window=10).shift(1) <= ta.momentum.rsi(df["Close"], window=10).rolling(10).mean().shift(1)) &
        (ta.momentum.rsi(df["Close"], window=10).shift(1) < 50) &
        (_rolling_omega(df) < 1) & (_rolling_sortino(df) < 0)
    )),
    # RSI crosses above SMA from <50 + Omega>1 + Sortino<0 (momentum but bad risk)
    "rsi_xsma_omega_gt1_sort_neg": ("RSI Cross SMA <50 + Omega>1 + Sortino<0", lambda df: (
        (ta.momentum.rsi(df["Close"], window=10) > ta.momentum.rsi(df["Close"], window=10).rolling(10).mean()) &
        (ta.momentum.rsi(df["Close"], window=10).shift(1) <= ta.momentum.rsi(df["Close"], window=10).rolling(10).mean().shift(1)) &
        (ta.momentum.rsi(df["Close"], window=10).shift(1) < 50) &
        (_rolling_omega(df) > 1) & (_rolling_sortino(df) < 0)
    )),
    # RSI crosses above SMA from <50 + Omega<1 + Sortino>0 (good risk, bad gain/loss)
    "rsi_xsma_omega_lt1_sort_pos": ("RSI Cross SMA <50 + Omega<1 + Sortino>0", lambda df: (
        (ta.momentum.rsi(df["Close"], window=10) > ta.momentum.rsi(df["Close"], window=10).rolling(10).mean()) &
        (ta.momentum.rsi(df["Close"], window=10).shift(1) <= ta.momentum.rsi(df["Close"], window=10).rolling(10).mean().shift(1)) &
        (ta.momentum.rsi(df["Close"], window=10).shift(1) < 50) &
        (_rolling_omega(df) < 1) & (_rolling_sortino(df) > 0)
    )),
    # RSI<30 + Omega<1 + Sortino<0 (deep oversold + all negative)
    "rsi_lt30_omega_lt1_sort_neg": ("RSI<30 + Omega<1 + Sortino<0", lambda df: (
        (ta.momentum.rsi(df["Close"], window=10) < 30) & (ta.momentum.rsi(df["Close"], window=10).shift(1) >= 30) &
        (_rolling_omega(df) < 1) & (_rolling_sortino(df) < 0)
    )),
    # RSI<30 + Omega<1 + Sortino<-1 (extreme weakness)
    "rsi_lt30_omega_lt1_sort_ltn1": ("RSI<30 + Omega<1 + Sortino<-1", lambda df: (
        (ta.momentum.rsi(df["Close"], window=10) < 30) & (ta.momentum.rsi(df["Close"], window=10).shift(1) >= 30) &
        (_rolling_omega(df) < 1) & (_rolling_sortino(df) < -1)
    )),
    # RSI crosses above 50 + Omega>1 + Sortino>0 (all positive, momentum confirmed)
    "rsi_x50up_omega_gt1_sort_pos": ("RSI Cross 50 Up + Omega>1 + Sortino>0", lambda df: (
        (ta.momentum.rsi(df["Close"], window=10) > 50) & (ta.momentum.rsi(df["Close"], window=10).shift(1) <= 50) &
        (_rolling_omega(df) > 1) & (_rolling_sortino(df) > 0)
    )),
    # RSI crosses above 50 + Omega<1 + Sortino<0 (momentum into weakness)
    "rsi_x50up_omega_lt1_sort_neg": ("RSI Cross 50 Up + Omega<1 + Sortino<0", lambda df: (
        (ta.momentum.rsi(df["Close"], window=10) > 50) & (ta.momentum.rsi(df["Close"], window=10).shift(1) <= 50) &
        (_rolling_omega(df) < 1) & (_rolling_sortino(df) < 0)
    )),
    # RSI>70 + Omega>2 + Sortino>1 (strong everything)
    "rsi_gt70_omega_gt2_sort_gt1": ("RSI>70 + Omega>2 + Sortino>1", lambda df: (
        (ta.momentum.rsi(df["Close"], window=10) > 70) & (ta.momentum.rsi(df["Close"], window=10).shift(1) <= 70) &
        (_rolling_omega(df) > 2) & (_rolling_sortino(df) > 1)
    )),
    # RSI crosses below SMA + Omega<1 + Sortino<0 (exit signal as entry — mean reversion)
    "rsi_xsma_dn_omega_lt1_sort_neg": ("RSI Cross SMA Down + Omega<1 + Sortino<0", lambda df: (
        (ta.momentum.rsi(df["Close"], window=10) < ta.momentum.rsi(df["Close"], window=10).rolling(10).mean()) &
        (ta.momentum.rsi(df["Close"], window=10).shift(1) >= ta.momentum.rsi(df["Close"], window=10).rolling(10).mean().shift(1)) &
        (_rolling_omega(df) < 1) & (_rolling_sortino(df) < 0)
    )),
    # ── Sortino trending up (MA crossover on Sortino) ──
    "sort_ma3_up": ("Sortino SMA(3) Trending Up", lambda df: _rolling_sortino(df).pipe(lambda s: s.rolling(3).mean().pipe(lambda m: (m > m.shift(1)) & (m.shift(1) <= m.shift(2))))),
    "sort_ma5_up": ("Sortino SMA(5) Trending Up", lambda df: _rolling_sortino(df).pipe(lambda s: s.rolling(5).mean().pipe(lambda m: (m > m.shift(1)) & (m.shift(1) <= m.shift(2))))),
    "sort_ma7_up": ("Sortino SMA(7) Trending Up", lambda df: _rolling_sortino(df).pipe(lambda s: s.rolling(7).mean().pipe(lambda m: (m > m.shift(1)) & (m.shift(1) <= m.shift(2))))),
    "sort_ma10_up": ("Sortino SMA(10) Trending Up", lambda df: _rolling_sortino(df).pipe(lambda s: s.rolling(10).mean().pipe(lambda m: (m > m.shift(1)) & (m.shift(1) <= m.shift(2))))),
    "sort_ma20_up": ("Sortino SMA(20) Trending Up", lambda df: _rolling_sortino(df).pipe(lambda s: s.rolling(20).mean().pipe(lambda m: (m > m.shift(1)) & (m.shift(1) <= m.shift(2))))),
    # Sortino trending down
    "sort_ma3_dn": ("Sortino SMA(3) Trending Down", lambda df: _rolling_sortino(df).pipe(lambda s: s.rolling(3).mean().pipe(lambda m: (m < m.shift(1)) & (m.shift(1) >= m.shift(2))))),
    "sort_ma5_dn": ("Sortino SMA(5) Trending Down", lambda df: _rolling_sortino(df).pipe(lambda s: s.rolling(5).mean().pipe(lambda m: (m < m.shift(1)) & (m.shift(1) >= m.shift(2))))),
    "sort_ma10_dn": ("Sortino SMA(10) Trending Down", lambda df: _rolling_sortino(df).pipe(lambda s: s.rolling(10).mean().pipe(lambda m: (m < m.shift(1)) & (m.shift(1) >= m.shift(2))))),
    # Sortino fast MA crosses slow MA (golden/death cross on Sortino)
    "sort_3x10_up": ("Sortino SMA(3) Cross SMA(10) Up", lambda df: _rolling_sortino(df).pipe(lambda s: (s.rolling(3).mean() > s.rolling(10).mean()) & (s.rolling(3).mean().shift(1) <= s.rolling(10).mean().shift(1)))),
    "sort_5x20_up": ("Sortino SMA(5) Cross SMA(20) Up", lambda df: _rolling_sortino(df).pipe(lambda s: (s.rolling(5).mean() > s.rolling(20).mean()) & (s.rolling(5).mean().shift(1) <= s.rolling(20).mean().shift(1)))),
    "sort_3x10_dn": ("Sortino SMA(3) Cross SMA(10) Down", lambda df: _rolling_sortino(df).pipe(lambda s: (s.rolling(3).mean() < s.rolling(10).mean()) & (s.rolling(3).mean().shift(1) >= s.rolling(10).mean().shift(1)))),
    # Sortino trending up + RSI combos
    "sort_ma5_up_rsi_xsma": ("Sortino MA(5) Up + RSI Cross SMA", lambda df: (
        _rolling_sortino(df).pipe(lambda s: s.rolling(5).mean() > s.rolling(5).mean().shift(1)) &
        (ta.momentum.rsi(df["Close"], window=10) > ta.momentum.rsi(df["Close"], window=10).rolling(10).mean()) &
        (ta.momentum.rsi(df["Close"], window=10).shift(1) <= ta.momentum.rsi(df["Close"], window=10).rolling(10).mean().shift(1))
    )),
    "sort_ma5_up_rsi_lt50": ("Sortino MA(5) Up + RSI<50", lambda df: (
        _rolling_sortino(df).pipe(lambda s: s.rolling(5).mean() > s.rolling(5).mean().shift(1)) &
        (ta.momentum.rsi(df["Close"], window=10) < 50) &
        (ta.momentum.rsi(df["Close"], window=10) > ta.momentum.rsi(df["Close"], window=10).rolling(10).mean()) &
        (ta.momentum.rsi(df["Close"], window=10).shift(1) <= ta.momentum.rsi(df["Close"], window=10).rolling(10).mean().shift(1))
    )),
    "sort_ma10_up_rsi_xsma": ("Sortino MA(10) Up + RSI Cross SMA", lambda df: (
        _rolling_sortino(df).pipe(lambda s: s.rolling(10).mean() > s.rolling(10).mean().shift(1)) &
        (ta.momentum.rsi(df["Close"], window=10) > ta.momentum.rsi(df["Close"], window=10).rolling(10).mean()) &
        (ta.momentum.rsi(df["Close"], window=10).shift(1) <= ta.momentum.rsi(df["Close"], window=10).rolling(10).mean().shift(1))
    )),
    # Sortino trending up + Omega combos
    "sort_ma5_up_omega_gt1": ("Sortino MA(5) Up + Omega>1", lambda df: (
        _rolling_sortino(df).pipe(lambda s: (s.rolling(5).mean() > s.rolling(5).mean().shift(1)) & (s.rolling(5).mean().shift(1) <= s.rolling(5).mean().shift(2))) &
        (_rolling_omega(df) > 1)
    )),
    "sort_ma10_up_omega_gt1": ("Sortino MA(10) Up + Omega>1", lambda df: (
        _rolling_sortino(df).pipe(lambda s: (s.rolling(10).mean() > s.rolling(10).mean().shift(1)) & (s.rolling(10).mean().shift(1) <= s.rolling(10).mean().shift(2))) &
        (_rolling_omega(df) > 1)
    )),
    # Sortino trending up + RSI + Omega triple
    "sort_ma5_up_rsi_xsma_omega": ("Sortino MA(5) Up + RSI Cross + Omega>1", lambda df: (
        _rolling_sortino(df).pipe(lambda s: s.rolling(5).mean() > s.rolling(5).mean().shift(1)) &
        (ta.momentum.rsi(df["Close"], window=10) > ta.momentum.rsi(df["Close"], window=10).rolling(10).mean()) &
        (ta.momentum.rsi(df["Close"], window=10).shift(1) <= ta.momentum.rsi(df["Close"], window=10).rolling(10).mean().shift(1)) &
        (_rolling_omega(df) > 1)
    )),
    "sort_3x10_up_rsi_xsma": ("Sortino 3x10 Cross Up + RSI Cross SMA", lambda df: (
        _rolling_sortino(df).pipe(lambda s: (s.rolling(3).mean() > s.rolling(10).mean()) & (s.rolling(3).mean().shift(1) <= s.rolling(10).mean().shift(1))) &
        (ta.momentum.rsi(df["Close"], window=10) > ta.momentum.rsi(df["Close"], window=10).rolling(10).mean()) &
        (ta.momentum.rsi(df["Close"], window=10).shift(1) <= ta.momentum.rsi(df["Close"], window=10).rolling(10).mean().shift(1))
    )),
    "sort_3x10_up_omega_rsi": ("Sortino 3x10 Up + Omega>1 + RSI Cross", lambda df: (
        _rolling_sortino(df).pipe(lambda s: (s.rolling(3).mean() > s.rolling(10).mean()) & (s.rolling(3).mean().shift(1) <= s.rolling(10).mean().shift(1))) &
        (_rolling_omega(df) > 1) &
        (ta.momentum.rsi(df["Close"], window=10) > ta.momentum.rsi(df["Close"], window=10).rolling(10).mean()) &
        (ta.momentum.rsi(df["Close"], window=10).shift(1) <= ta.momentum.rsi(df["Close"], window=10).rolling(10).mean().shift(1))
    )),
    # Sortino trending up from negative territory
    "sort_ma5_up_from_neg": ("Sortino MA(5) Up from <0", lambda df: (
        _rolling_sortino(df).pipe(lambda s: (s.rolling(5).mean() > s.rolling(5).mean().shift(1)) & (s.rolling(5).mean().shift(1) <= s.rolling(5).mean().shift(2)) & (s < 0))
    )),
    "sort_ma10_up_from_neg": ("Sortino MA(10) Up from <0", lambda df: (
        _rolling_sortino(df).pipe(lambda s: (s.rolling(10).mean() > s.rolling(10).mean().shift(1)) & (s.rolling(10).mean().shift(1) <= s.rolling(10).mean().shift(2)) & (s < 0))
    )),
    # Gap + Sortino trending
    "gap_dn_sort_ma5_up": ("Gap Down + Sortino MA(5) Up", lambda df: (
        ((df["Open"] - df["Low"].shift(1)) / df["Low"].shift(1) * 100 <= -0.5) &
        _rolling_sortino(df).pipe(lambda s: s.rolling(5).mean() > s.rolling(5).mean().shift(1))
    )),
    "gap_dn_sort_ma5_dn": ("Gap Down + Sortino MA(5) Down", lambda df: (
        ((df["Open"] - df["Low"].shift(1)) / df["Low"].shift(1) * 100 <= -0.5) &
        _rolling_sortino(df).pipe(lambda s: s.rolling(5).mean() < s.rolling(5).mean().shift(1))
    )),
    # ── New entry signals ──
    # Price action
    "inside_day": ("Inside Day (HL within prev HL)", lambda df: (df["High"] < df["High"].shift(1)) & (df["Low"] > df["Low"].shift(1))),
    "outside_day": ("Outside Day (HL exceeds prev HL)", lambda df: (df["High"] > df["High"].shift(1)) & (df["Low"] < df["Low"].shift(1)) & (df["Close"] > df["Open"])),
    "3_tight_closes": ("3 Days Tight Closes (<1% range)", lambda df: (
        ((df["Close"].rolling(3).max() - df["Close"].rolling(3).min()) / df["Close"].rolling(3).min() * 100 < 1) &
        (df["Close"].shift(3).rolling(3).std() > df["Close"].rolling(3).std())
    )),
    "price_above_sma200": ("Price Crosses Above SMA200", lambda df: (df["Close"] > df["Close"].rolling(200).mean()) & (df["Close"].shift(1) <= df["Close"].rolling(200).mean().shift(1))),
    "price_below_sma200": ("Price Crosses Below SMA200", lambda df: (df["Close"] < df["Close"].rolling(200).mean()) & (df["Close"].shift(1) >= df["Close"].rolling(200).mean().shift(1))),
    # Volume patterns
    "vol_climax_dn": ("Volume Climax Down (3x vol + red)", lambda df: (df["Volume"] > df["Volume"].rolling(20).mean() * 3) & (df["Close"] < df["Open"])),
    "vol_climax_up": ("Volume Climax Up (3x vol + green)", lambda df: (df["Volume"] > df["Volume"].rolling(20).mean() * 3) & (df["Close"] > df["Open"])),
    "vol_dry_up_3d": ("3 Days Declining Volume + Down", lambda df: (
        (df["Volume"] < df["Volume"].shift(1)) & (df["Volume"].shift(1) < df["Volume"].shift(2)) &
        (df["Close"] < df["Close"].shift(2)) & (df["Close"] > df["Close"].shift(1))
    )),
    # Multi-indicator combos
    "rsi_os30_omega_lt1": ("RSI<30 + Omega<1", lambda df: (
        (ta.momentum.rsi(df["Close"], window=10) < 30) & (_rolling_omega(df) < 1)
    )),
    "rsi_os20_omega_lt1_sort_neg": ("RSI<20 + Omega<1 + Sortino<0", lambda df: (
        (ta.momentum.rsi(df["Close"], window=10) < 20) & (_rolling_omega(df) < 1) & (_rolling_sortino(df) < 0)
    )),
    "rsi_x_sma_omega_sort_hl": ("RSI Cross + Omega>1 + Sortino>0 + HL", lambda df: (
        (ta.momentum.rsi(df["Close"], window=10) > ta.momentum.rsi(df["Close"], window=10).rolling(10).mean()) &
        (ta.momentum.rsi(df["Close"], window=10).shift(1) <= ta.momentum.rsi(df["Close"], window=10).rolling(10).mean().shift(1)) &
        (_rolling_omega(df) > 1) & (_rolling_sortino(df) > 0) & _higher_low(df)
    )),
    "all_bullish": ("All Bullish (RSI× + Omega>1 + Sort>0 + RS>50)", lambda df: (
        (ta.momentum.rsi(df["Close"], window=10) > ta.momentum.rsi(df["Close"], window=10).rolling(10).mean()) &
        (ta.momentum.rsi(df["Close"], window=10).shift(1) <= ta.momentum.rsi(df["Close"], window=10).rolling(10).mean().shift(1)) &
        (_rolling_omega(df) > 1) & (_rolling_sortino(df) > 0) & (_rsi_of_sortino(df) > 50)
    )),
    "all_bearish_reversal": ("All Bearish Then RSI Cross (RSI<30+Omega<1+Sort<0+RS<30 -> RSI×)", lambda df: _seq_a_then_b(df,
        lambda d: (ta.momentum.rsi(d["Close"], window=10) < 30) & (_rolling_omega(d) < 1) & (_rolling_sortino(d) < 0) & (_rsi_of_sortino(d) < 30),
        lambda d: (ta.momentum.rsi(d["Close"], window=10) > ta.momentum.rsi(d["Close"], window=10).rolling(10).mean()) & (ta.momentum.rsi(d["Close"], window=10).shift(1) <= ta.momentum.rsi(d["Close"], window=10).rolling(10).mean().shift(1)), 10)),
    # Mean reversion
    "rsi_os20_3down": ("RSI<20 + 3 Down Days", lambda df: (
        (ta.momentum.rsi(df["Close"], window=10) < 20) & (df["Close"] < df["Close"].shift(1)) & (df["Close"].shift(1) < df["Close"].shift(2)) & (df["Close"].shift(2) < df["Close"].shift(3))
    )),
    "boll_lower_rsi_os": ("Lower Bollinger + RSI<30", lambda df: sig_boll_lower(df) & (ta.momentum.rsi(df["Close"], window=10) < 30)),
    "boll_lower_vol_spike": ("Lower Bollinger + Volume Spike", lambda df: sig_boll_lower(df) & (df["Volume"] > df["Volume"].rolling(20).mean() * 1.5)),
    # Momentum continuation
    "golden_cross_rsi_x": ("Golden Cross + RSI Cross SMA", lambda df: (
        sig_golden_cross(df) &
        (ta.momentum.rsi(df["Close"], window=10) > ta.momentum.rsi(df["Close"], window=10).rolling(10).mean())
    )),
    "new_20high_vol": ("New 20-Day High + Volume Spike", lambda df: sig_new_20high(df) & (df["Volume"] > df["Volume"].rolling(20).mean() * 1.5)),
    "new_52high_rsi_x": ("New 52-Week High + RSI Cross", lambda df: (
        sig_new_52high(df) &
        (ta.momentum.rsi(df["Close"], window=10) > ta.momentum.rsi(df["Close"], window=10).rolling(10).mean())
    )),
    # ── SPY/QQQ market signals (injected with spy_close + qqq_close in run_study) ──
    "spy_rsi_x": ("SPY RSI Cross Above SMA", sig_spy_rsi_x),
    "qqq_rsi_x": ("QQQ RSI Cross Above SMA", sig_qqq_rsi_x),
    "spy_qqq_rsi_x_both": ("SPY+QQQ RSI Cross (both, 3d)", sig_spy_qqq_rsi_x_both),
    "spy_rsi_x_wk": ("SPY Weekly RSI Cross", sig_spy_rsi_x_wk),
    "qqq_rsi_x_wk": ("QQQ Weekly RSI Cross", sig_qqq_rsi_x_wk),
    "spy_qqq_rsi_x_both_wk": ("SPY+QQQ Weekly RSI Cross (both)", sig_spy_qqq_rsi_x_both_wk),
    "corr_spyqqq_x_high": ("Corr->SPY&QQQ Cross >0.7", sig_corr_spyqqq_x_high),
    "beta_spyqqq_x_high": ("Beta->SPY&QQQ Cross >1.0", sig_beta_spyqqq_x_high),
    "spy_qqq_rsi_x_hibeta": ("SPY+QQQ Cross + HiBeta", sig_spy_qqq_rsi_x_hibeta),
    "spy_qqq_rsi_x_hicorr": ("SPY+QQQ Cross + HiCorr", sig_spy_qqq_rsi_x_hicorr),
    "spy_qqq_rsi_x_hibeta_hicorr": ("SPY+QQQ Cross + HiBeta + HiCorr", sig_spy_qqq_rsi_x_hibeta_hicorr),
    "spy_qqq_rsi_x_hibeta_wk": ("SPY+QQQ Weekly Cross + HiBeta", sig_spy_qqq_rsi_x_hibeta_wk),
    "spy_qqq_rsi_x_hicorr_wk": ("SPY+QQQ Weekly Cross + HiCorr", sig_spy_qqq_rsi_x_hicorr_wk),
    "spy_qqq_rsi_x_hibeta_hicorr_wk": ("SPY+QQQ Weekly Cross + HiBeta + HiCorr", sig_spy_qqq_rsi_x_hibeta_hicorr_wk),
    # ── Stock drawdown signal ──
    "dd30_rsi_reversal": ("30% off ATH + RSI Reversal", sig_dd30_rsi_reversal),
}


# ── Exit conditions ──

def exit_fixed(df, entry_idx, days):
    """Exit after a fixed number of days. Returns None when the full horizon runs past the end
    of the series so an incomplete trade is DROPPED rather than silently clamped to the last bar
    and counted as a finished `days`-bar trade. Clamping made every ticker's final ~`days` entries
    book a truncated hold / compressed return, biasing avg_hold down and avg_return toward 0 for
    long fixed-horizon exits (6m/1y)."""
    exit_idx = entry_idx + days
    return exit_idx if exit_idx < len(df) else None

def exit_next_gap_down(df, entry_idx, max_hold=60):
    """Hold until next gap down, max 60 days."""
    for i in range(entry_idx + 1, min(entry_idx + max_hold, len(df))):
        prev_low = df["Low"].iloc[i - 1]
        cur_open = df["Open"].iloc[i]
        if prev_low > 0 and (cur_open - prev_low) / prev_low * 100 <= -0.5:
            return i
    return min(entry_idx + max_hold, len(df) - 1)

def exit_next_gap_up(df, entry_idx, max_hold=60):
    """Hold until next gap up, max 60 days."""
    for i in range(entry_idx + 1, min(entry_idx + max_hold, len(df))):
        prev_high = df["High"].iloc[i - 1]
        cur_open = df["Open"].iloc[i]
        if prev_high > 0 and (cur_open - prev_high) / prev_high * 100 >= 0.5:
            return i
    return min(entry_idx + max_hold, len(df) - 1)

def exit_rsi_cross_down(df, entry_idx, max_hold=60):
    """Hold until RSI crosses below its SMA."""
    rsi = df["_rsi"] if "_rsi" in df.columns else ta.momentum.rsi(df["Close"], window=10)
    sma = df["_rsi_sma"] if "_rsi_sma" in df.columns else rsi.rolling(10).mean()
    for i in range(entry_idx + 2, min(entry_idx + max_hold, len(df))):
        if rsi.iloc[i] < sma.iloc[i] and rsi.iloc[i-1] >= sma.iloc[i-1]:
            return i
    return min(entry_idx + max_hold, len(df) - 1)

def exit_rsi_cross_up(df, entry_idx, max_hold=60):
    """Hold until RSI crosses above its SMA."""
    rsi = df["_rsi"] if "_rsi" in df.columns else ta.momentum.rsi(df["Close"], window=10)
    sma = df["_rsi_sma"] if "_rsi_sma" in df.columns else rsi.rolling(10).mean()
    for i in range(entry_idx + 2, min(entry_idx + max_hold, len(df))):
        if rsi.iloc[i] > sma.iloc[i] and rsi.iloc[i-1] <= sma.iloc[i-1]:
            return i
    return min(entry_idx + max_hold, len(df) - 1)

def exit_rsi_overbought(df, entry_idx, max_hold=60):
    """Hold until RSI > 70."""
    rsi = df["_rsi"] if "_rsi" in df.columns else ta.momentum.rsi(df["Close"], window=10)
    for i in range(entry_idx + 1, min(entry_idx + max_hold, len(df))):
        if rsi.iloc[i] > 70:
            return i
    return min(entry_idx + max_hold, len(df) - 1)

def exit_rsi_oversold(df, entry_idx, max_hold=60):
    """Hold until RSI < 30."""
    rsi = df["_rsi"] if "_rsi" in df.columns else ta.momentum.rsi(df["Close"], window=10)
    for i in range(entry_idx + 1, min(entry_idx + max_hold, len(df))):
        if rsi.iloc[i] < 30:
            return i
    return min(entry_idx + max_hold, len(df) - 1)

def exit_trailing_stop(df, entry_idx, stop_pct=5, max_hold=120):
    """Trailing stop loss."""
    peak = df["Close"].iloc[entry_idx]
    for i in range(entry_idx + 1, min(entry_idx + max_hold, len(df))):
        price = df["Close"].iloc[i]
        if price > peak:
            peak = price
        if (price - peak) / peak * 100 <= -stop_pct:
            return i
    return min(entry_idx + max_hold, len(df) - 1)

def exit_stop_loss(df, entry_idx, stop_pct=3, max_hold=60):
    """Fixed stop loss from entry."""
    entry_price = df["Close"].iloc[entry_idx]
    for i in range(entry_idx + 1, min(entry_idx + max_hold, len(df))):
        if (df["Close"].iloc[i] - entry_price) / entry_price * 100 <= -stop_pct:
            return i
    return min(entry_idx + max_hold, len(df) - 1)

def exit_take_profit(df, entry_idx, target_pct=5, max_hold=60):
    """Take profit at target %."""
    entry_price = df["Close"].iloc[entry_idx]
    for i in range(entry_idx + 1, min(entry_idx + max_hold, len(df))):
        if (df["Close"].iloc[i] - entry_price) / entry_price * 100 >= target_pct:
            return i
    return min(entry_idx + max_hold, len(df) - 1)

def exit_macd_cross_down(df, entry_idx, max_hold=60):
    """Hold until MACD crosses below signal."""
    m = ta.trend.MACD(df["Close"])
    macd, sig = m.macd(), m.macd_signal()
    for i in range(entry_idx + 2, min(entry_idx + max_hold, len(df))):
        if macd.iloc[i] < sig.iloc[i] and macd.iloc[i-1] >= sig.iloc[i-1]:
            return i
    return min(entry_idx + max_hold, len(df) - 1)


def _exit_rsi_sort_cross_down(df, entry_idx, max_hold=90):
    """Hold until RSI of Sortino crosses below its SMA."""
    rs = df["_rsi_sort"] if "_rsi_sort" in df.columns else _rsi_of_sortino(df)
    rs_sma = df["_rsi_sort_sma"] if "_rsi_sort_sma" in df.columns else rs.rolling(10).mean()
    for i in range(entry_idx + 2, min(entry_idx + max_hold, len(df))):
        if rs.iloc[i] < rs_sma.iloc[i] and rs.iloc[i-1] >= rs_sma.iloc[i-1]:
            return i
    return min(entry_idx + max_hold, len(df) - 1)


def _exit_rsi_sort_cross_up(df, entry_idx, max_hold=90):
    """Hold until RSI of Sortino crosses above its SMA."""
    rs = df["_rsi_sort"] if "_rsi_sort" in df.columns else _rsi_of_sortino(df)
    rs_sma = df["_rsi_sort_sma"] if "_rsi_sort_sma" in df.columns else rs.rolling(10).mean()
    for i in range(entry_idx + 2, min(entry_idx + max_hold, len(df))):
        if rs.iloc[i] > rs_sma.iloc[i] and rs.iloc[i-1] <= rs_sma.iloc[i-1]:
            return i
    return min(entry_idx + max_hold, len(df) - 1)


def _exit_rsi_sort_below50(df, entry_idx, max_hold=90):
    """Hold until RSI of Sortino drops below 50."""
    rs = df["_rsi_sort"] if "_rsi_sort" in df.columns else _rsi_of_sortino(df)
    for i in range(entry_idx + 2, min(entry_idx + max_hold, len(df))):
        if rs.iloc[i] < 50 and rs.iloc[i-1] >= 50:
            return i
    return min(entry_idx + max_hold, len(df) - 1)


def _exit_rsi_sort_below30(df, entry_idx, max_hold=90):
    """Hold until RSI of Sortino drops below 30."""
    rs = df["_rsi_sort"] if "_rsi_sort" in df.columns else _rsi_of_sortino(df)
    for i in range(entry_idx + 2, min(entry_idx + max_hold, len(df))):
        if rs.iloc[i] < 30:
            return i
    return min(entry_idx + max_hold, len(df) - 1)


def _exit_rsi_sort_above70(df, entry_idx, max_hold=90):
    """Hold until RSI of Sortino rises above 70."""
    rs = df["_rsi_sort"] if "_rsi_sort" in df.columns else _rsi_of_sortino(df)
    for i in range(entry_idx + 2, min(entry_idx + max_hold, len(df))):
        if rs.iloc[i] > 70:
            return i
    return min(entry_idx + max_hold, len(df) - 1)


def _exit_rsi_level(df, entry_idx, level, max_hold=90):
    """Hold until RSI crosses above a level."""
    rsi = df["_rsi"] if "_rsi" in df.columns else ta.momentum.rsi(df["Close"], window=10)
    for i in range(entry_idx + 1, min(entry_idx + max_hold, len(df))):
        if rsi.iloc[i] > level:
            return i
    return min(entry_idx + max_hold, len(df) - 1)


def _exit_rsi_ob70_then_cross_dn(df, entry_idx, max_hold=120):
    """Hold until RSI goes above 70 then crosses below its SMA(14)."""
    rsi = df["_rsi"] if "_rsi" in df.columns else ta.momentum.rsi(df["Close"], window=10)
    rsi_sma = rsi.rolling(14).mean()
    hit_70 = False
    for i in range(entry_idx + 1, min(entry_idx + max_hold, len(df))):
        if rsi.iloc[i] > 70:
            hit_70 = True
        if hit_70 and i > entry_idx + 1 and rsi.iloc[i] < rsi_sma.iloc[i] and rsi.iloc[i-1] >= rsi_sma.iloc[i-1]:
            return i
    return min(entry_idx + max_hold, len(df) - 1)


def _exit_rs_ob70_then_cross_dn(df, entry_idx, max_hold=120):
    """Hold until RSI of Sortino goes above 70 then crosses below its SMA(14)."""
    rs = df["_rsi_sort"] if "_rsi_sort" in df.columns else _rsi_of_sortino(df)
    rs_sma = rs.rolling(14).mean()
    hit_70 = False
    for i in range(entry_idx + 1, min(entry_idx + max_hold, len(df))):
        if rs.iloc[i] > 70:
            hit_70 = True
        if hit_70 and i > entry_idx + 1 and rs.iloc[i] < rs_sma.iloc[i] and rs.iloc[i-1] >= rs_sma.iloc[i-1]:
            return i
    return min(entry_idx + max_hold, len(df) - 1)


def _exit_both_ob70_then_cross_dn(df, entry_idx, max_hold=120):
    """Hold until both RSI and RSI-Sort go above 70 then both cross below their SMA(14)."""
    rsi = df["_rsi"] if "_rsi" in df.columns else ta.momentum.rsi(df["Close"], window=10)
    rsi_sma = rsi.rolling(14).mean()
    rs = df["_rsi_sort"] if "_rsi_sort" in df.columns else _rsi_of_sortino(df)
    rs_sma = rs.rolling(14).mean()
    rsi_hit_70 = False
    rs_hit_70 = False
    rsi_crossed = False
    rs_crossed = False
    for i in range(entry_idx + 1, min(entry_idx + max_hold, len(df))):
        if rsi.iloc[i] > 70:
            rsi_hit_70 = True
        if rs.iloc[i] > 70:
            rs_hit_70 = True
        if rsi_hit_70 and i > entry_idx + 1 and rsi.iloc[i] < rsi_sma.iloc[i] and rsi.iloc[i-1] >= rsi_sma.iloc[i-1]:
            rsi_crossed = True
        if rs_hit_70 and i > entry_idx + 1 and rs.iloc[i] < rs_sma.iloc[i] and rs.iloc[i-1] >= rs_sma.iloc[i-1]:
            rs_crossed = True
        if rsi_crossed and rs_crossed:
            return i
    return min(entry_idx + max_hold, len(df) - 1)


def _exit_seq_above_then_below(df, entry_idx, col, level, max_hold=120):
    """Hold until indicator goes above level then drops below it."""
    s = df[col] if col in df.columns else None
    if s is None:
        return min(entry_idx + max_hold, len(df) - 1)
    went_above = False
    for i in range(entry_idx + 1, min(entry_idx + max_hold, len(df))):
        if s.iloc[i] > level:
            went_above = True
        if went_above and s.iloc[i] < level and s.iloc[i-1] >= level:
            return i
    return min(entry_idx + max_hold, len(df) - 1)


def _exit_level_then_cross_dn(df, entry_idx, col, level, sma_period, max_hold=120):
    """Hold until indicator goes above level then crosses below its SMA."""
    s = df[col] if col in df.columns else None
    if s is None:
        return min(entry_idx + max_hold, len(df) - 1)
    sma = s.rolling(sma_period).mean()
    hit_level = False
    for i in range(entry_idx + 1, min(entry_idx + max_hold, len(df))):
        if s.iloc[i] > level:
            hit_level = True
        if hit_level and i > entry_idx + 1 and s.iloc[i] < sma.iloc[i] and s.iloc[i-1] >= sma.iloc[i-1]:
            return i
    return min(entry_idx + max_hold, len(df) - 1)


def _exit_price_above_then_below_sma(df, entry_idx, period, max_hold=120):
    """Hold until price goes above SMA then crosses below it."""
    close = df["Close"]
    sma = close.rolling(period).mean()
    went_above = False
    for i in range(entry_idx + 1, min(entry_idx + max_hold, len(df))):
        if close.iloc[i] > sma.iloc[i]:
            went_above = True
        if went_above and close.iloc[i] < sma.iloc[i] and close.iloc[i-1] >= sma.iloc[i-1]:
            return i
    return min(entry_idx + max_hold, len(df) - 1)


def _exit_rs_cross_up_then_down(df, entry_idx, max_hold=120):
    """Hold until RSI of Sortino crosses above SMA then crosses back below."""
    rs = df["_rsi_sort"] if "_rsi_sort" in df.columns else _rsi_of_sortino(df)
    rs_sma = rs.rolling(14).mean()
    crossed_up = False
    for i in range(entry_idx + 2, min(entry_idx + max_hold, len(df))):
        if rs.iloc[i] > rs_sma.iloc[i] and rs.iloc[i-1] <= rs_sma.iloc[i-1]:
            crossed_up = True
        if crossed_up and rs.iloc[i] < rs_sma.iloc[i] and rs.iloc[i-1] >= rs_sma.iloc[i-1]:
            return i
    return min(entry_idx + max_hold, len(df) - 1)


def _exit_rsi_cross_up_then_down(df, entry_idx, max_hold=120):
    """Hold until RSI crosses above SMA(14) then crosses back below."""
    rsi = df["_rsi"] if "_rsi" in df.columns else ta.momentum.rsi(df["Close"], window=10)
    rsi_sma = rsi.rolling(14).mean()
    crossed_up = False
    for i in range(entry_idx + 2, min(entry_idx + max_hold, len(df))):
        if rsi.iloc[i] > rsi_sma.iloc[i] and rsi.iloc[i-1] <= rsi_sma.iloc[i-1]:
            crossed_up = True
        if crossed_up and rsi.iloc[i] < rsi_sma.iloc[i] and rsi.iloc[i-1] >= rsi_sma.iloc[i-1]:
            return i
    return min(entry_idx + max_hold, len(df) - 1)


def _exit_rsi_sort_cross_level(df, entry_idx, level, direction, max_hold=90):
    """Hold until RSI of Sortino crosses a fixed level."""
    rs = df["_rsi_sort"] if "_rsi_sort" in df.columns else _rsi_of_sortino(df)
    for i in range(entry_idx + 2, min(entry_idx + max_hold, len(df))):
        if direction == "down" and rs.iloc[i] < level and rs.iloc[i-1] >= level:
            return i
        if direction == "up" and rs.iloc[i] > level and rs.iloc[i-1] <= level:
            return i
    return min(entry_idx + max_hold, len(df) - 1)


def _exit_rsi_sort_b30_cross_up(df, entry_idx, max_hold=90):
    """Hold until RSI of Sortino drops below 30 then crosses above SMA."""
    rs = df["_rsi_sort"] if "_rsi_sort" in df.columns else _rsi_of_sortino(df)
    rs_sma = df["_rsi_sort_sma"] if "_rsi_sort_sma" in df.columns else rs.rolling(10).mean()
    went_below_30 = False
    for i in range(entry_idx + 2, min(entry_idx + max_hold, len(df))):
        if rs.iloc[i] < 30:
            went_below_30 = True
        if went_below_30 and rs.iloc[i] > rs_sma.iloc[i] and rs.iloc[i-1] <= rs_sma.iloc[i-1]:
            return i
    return min(entry_idx + max_hold, len(df) - 1)


def _exit_sort_above(df, entry_idx, level, max_hold=90):
    """Hold until Sortino crosses above a level."""
    sortino = df["_sortino"] if "_sortino" in df.columns else _rolling_sortino(df)
    for i in range(entry_idx + 2, min(entry_idx + max_hold, len(df))):
        if sortino.iloc[i] > level:
            return i
    return min(entry_idx + max_hold, len(df) - 1)


def _exit_sort_positive(df, entry_idx, max_hold=90):
    """Hold until Sortino turns positive."""
    sortino = df["_sortino"] if "_sortino" in df.columns else _rolling_sortino(df)
    for i in range(entry_idx + 2, min(entry_idx + max_hold, len(df))):
        if sortino.iloc[i] > 0 and sortino.iloc[i-1] <= 0:
            return i
    return min(entry_idx + max_hold, len(df) - 1)


def _exit_omega_above1(df, entry_idx, max_hold=90):
    """Hold until Omega crosses above 1."""
    omega = df["_omega"] if "_omega" in df.columns else _rolling_omega(df)
    for i in range(entry_idx + 2, min(entry_idx + max_hold, len(df))):
        if omega.iloc[i] > 1 and omega.iloc[i-1] <= 1:
            return i
    return min(entry_idx + max_hold, len(df) - 1)


def _exit_first_of(df, entry_idx, exit_a, exit_b):
    """Exit at whichever condition fires first."""
    a = exit_a(df, entry_idx)
    b = exit_b(df, entry_idx)
    if a is None:
        return b
    if b is None:
        return a
    return min(a, b)


def _exit_price_below_sma(df, entry_idx, period, max_hold=90):
    """Hold until price crosses below SMA."""
    close = df["Close"]
    sma = close.rolling(period).mean()
    for i in range(entry_idx + 2, min(entry_idx + max_hold, len(df))):
        if close.iloc[i] < sma.iloc[i] and close.iloc[i-1] >= sma.iloc[i-1]:
            return i
    return min(entry_idx + max_hold, len(df) - 1)


def _exit_lower_low(df, entry_idx, max_hold=90, lookback=10):
    """Hold until a lower local low forms."""
    low = df["Low"]
    for i in range(entry_idx + lookback + 1, min(entry_idx + max_hold, len(df))):
        curr_min = low.iloc[i-lookback:i].min()
        prev_min = low.iloc[max(0,i-2*lookback):i-lookback].min()
        if curr_min < prev_min:
            return i
    return min(entry_idx + max_hold, len(df) - 1)


def _exit_sortino_neg(df, entry_idx, max_hold=60):
    """Hold until Sortino turns negative."""
    sortino = df["_sortino"] if "_sortino" in df.columns else _rolling_sortino(df)
    for i in range(entry_idx + 2, min(entry_idx + max_hold, len(df))):
        if sortino.iloc[i] < 0:
            return i
    return min(entry_idx + max_hold, len(df) - 1)

def _exit_omega_below1(df, entry_idx, max_hold=60):
    """Hold until Omega drops below 1."""
    omega = df["_omega"] if "_omega" in df.columns else _rolling_omega(df)
    for i in range(entry_idx + 2, min(entry_idx + max_hold, len(df))):
        if omega.iloc[i] < 1:
            return i
    return min(entry_idx + max_hold, len(df) - 1)

def _exit_rsi_below50(df, entry_idx, max_hold=60):
    """Hold until RSI drops below 50."""
    rsi = df["_rsi"] if "_rsi" in df.columns else ta.momentum.rsi(df["Close"], window=10)
    for i in range(entry_idx + 1, min(entry_idx + max_hold, len(df))):
        if rsi.iloc[i] < 50:
            return i
    return min(entry_idx + max_hold, len(df) - 1)


EXITS = {
    "1d": ("Hold 1 day", lambda df, idx: exit_fixed(df, idx, 1)),
    "3d": ("Hold 3 days", lambda df, idx: exit_fixed(df, idx, 3)),
    "1w": ("Hold 1 week", lambda df, idx: exit_fixed(df, idx, 5)),
    "2w": ("Hold 2 weeks", lambda df, idx: exit_fixed(df, idx, 10)),
    "4w": ("Hold 4 weeks", lambda df, idx: exit_fixed(df, idx, 20)),
    "8w": ("Hold 8 weeks", lambda df, idx: exit_fixed(df, idx, 40)),
    "gap_down": ("Till Next Gap Down", exit_next_gap_down),
    "gap_up": ("Till Next Gap Up", exit_next_gap_up),
    "rsi_x_dn": ("Till RSI Crosses Down", exit_rsi_cross_down),
    "rsi_x_up": ("Till RSI Crosses Up", exit_rsi_cross_up),
    "rsi_ob": ("Till RSI > 70", exit_rsi_overbought),
    "rsi_80": ("Till RSI > 80", lambda df, idx: _exit_rsi_level(df, idx, 80)),
    "rsi_os": ("Till RSI < 30", exit_rsi_oversold),
    "trail_3": ("3% Trailing Stop", lambda df, idx: exit_trailing_stop(df, idx, 3)),
    "trail_5": ("5% Trailing Stop", lambda df, idx: exit_trailing_stop(df, idx, 5)),
    "trail_10": ("10% Trailing Stop", lambda df, idx: exit_trailing_stop(df, idx, 10)),
    "sl_3": ("3% Stop Loss", lambda df, idx: exit_stop_loss(df, idx, 3)),
    "tp_5": ("5% Take Profit", lambda df, idx: exit_take_profit(df, idx, 5)),
    "tp_10": ("10% Take Profit", lambda df, idx: exit_take_profit(df, idx, 10)),
    "macd_x_dn": ("Till MACD Cross Down", exit_macd_cross_down),
    "rs_x_dn": ("Till RSI of Sortino Crosses Below SMA", lambda df, idx: _exit_rsi_sort_cross_down(df, idx)),
    "rs_x_up": ("Till RSI of Sortino Crosses Above SMA", lambda df, idx: _exit_rsi_sort_cross_up(df, idx)),
    "rs_b50": ("Till RSI of Sortino < 50", lambda df, idx: _exit_rsi_sort_below50(df, idx)),
    "rs_b30": ("Till RSI of Sortino < 30", lambda df, idx: _exit_rsi_sort_below30(df, idx)),
    "rs_a70": ("Till RSI of Sortino > 70", lambda df, idx: _exit_rsi_sort_above70(df, idx)),
    "rs_b30_x_up": ("Till RSI of Sortino <30 Then Cross SMA", lambda df, idx: _exit_rsi_sort_b30_cross_up(df, idx)),
    "rs_x50_dn": ("Till RSI of Sortino Crosses 50 Down", lambda df, idx: _exit_rsi_sort_cross_level(df, idx, 50, "down")),
    "rs_x50_up": ("Till RSI of Sortino Crosses 50 Up", lambda df, idx: _exit_rsi_sort_cross_level(df, idx, 50, "up")),
    "rs_x30_up": ("Till RSI of Sortino Crosses 30 Up", lambda df, idx: _exit_rsi_sort_cross_level(df, idx, 30, "up")),
    "sort_neg": ("Till Sortino < 0", lambda df, idx: _exit_sortino_neg(df, idx)),
    "sort_gt2": ("Till Sortino > 2", lambda df, idx: _exit_sort_above(df, idx, 2)),
    "sort_gt1": ("Till Sortino > 1", lambda df, idx: _exit_sort_above(df, idx, 1)),
    "omega_lt1": ("Till Omega < 1", lambda df, idx: _exit_omega_below1(df, idx)),
    "12w": ("Hold 12 weeks", lambda df, idx: exit_fixed(df, idx, 60)),
    "rsi_x50_dn": ("Till RSI < 50", lambda df, idx: _exit_rsi_below50(df, idx)),
    "sl_5": ("5% Stop Loss", lambda df, idx: exit_stop_loss(df, idx, 5)),
    "sl_10": ("10% Stop Loss", lambda df, idx: exit_stop_loss(df, idx, 10)),
    "tp_3": ("3% Take Profit", lambda df, idx: exit_take_profit(df, idx, 3)),
    "tp_15": ("15% Take Profit", lambda df, idx: exit_take_profit(df, idx, 15)),
    "trail_7": ("7% Trailing Stop", lambda df, idx: exit_trailing_stop(df, idx, 7)),
    "trail_15": ("15% Trailing Stop", lambda df, idx: exit_trailing_stop(df, idx, 15)),
    # ── New exits ──
    "16w": ("Hold 16 weeks", lambda df, idx: exit_fixed(df, idx, 80)),
    "120d": ("Hold 120 days", lambda df, idx: exit_fixed(df, idx, 120)),
    "6m": ("Hold 6 months", lambda df, idx: exit_fixed(df, idx, 126)),
    "tp_20": ("20% Take Profit", lambda df, idx: exit_take_profit(df, idx, 20)),
    "tp_30": ("30% Take Profit", lambda df, idx: exit_take_profit(df, idx, 30)),
    "sl_7": ("7% Stop Loss", lambda df, idx: exit_stop_loss(df, idx, 7)),
    "sl_15": ("15% Stop Loss", lambda df, idx: exit_stop_loss(df, idx, 15)),
    "trail_20": ("20% Trailing Stop", lambda df, idx: exit_trailing_stop(df, idx, 20)),
    "sort_pos": ("Till Sortino > 0", lambda df, idx: _exit_sort_positive(df, idx)),
    "omega_gt1": ("Till Omega > 1", lambda df, idx: _exit_omega_above1(df, idx)),
    "rsi_x_dn_or_sl5": ("RSI Cross Down OR 5% Stop", lambda df, idx: _exit_first_of(df, idx, exit_rsi_cross_down, lambda d, i: exit_stop_loss(d, i, 5))),
    "rsi_x_dn_or_sl10": ("RSI Cross Down OR 10% Stop", lambda df, idx: _exit_first_of(df, idx, exit_rsi_cross_down, lambda d, i: exit_stop_loss(d, i, 10))),
    "rs_x_dn_or_sl5": ("RSI-Sort Cross Down OR 5% Stop", lambda df, idx: _exit_first_of(df, idx, lambda d, i: _exit_rsi_sort_cross_down(d, i), lambda d, i: exit_stop_loss(d, i, 5))),
    "rs_x_dn_or_sl10": ("RSI-Sort Cross Down OR 10% Stop", lambda df, idx: _exit_first_of(df, idx, lambda d, i: _exit_rsi_sort_cross_down(d, i), lambda d, i: exit_stop_loss(d, i, 10))),
    "sma20_dn": ("Till Price Crosses Below SMA20", lambda df, idx: _exit_price_below_sma(df, idx, 20)),
    "sma50_dn": ("Till Price Crosses Below SMA50", lambda df, idx: _exit_price_below_sma(df, idx, 50)),
    "lower_low": ("Till Lower Local Low", lambda df, idx: _exit_lower_low(df, idx)),
    # Sequential exits: overbought then cross down
    "rsi_70_x_dn": ("Till RSI >70 Then Crosses Below SMA(14)", lambda df, idx: _exit_rsi_ob70_then_cross_dn(df, idx)),
    "rs_70_x_dn": ("Till RSI-Sort >70 Then Crosses Below SMA(14)", lambda df, idx: _exit_rs_ob70_then_cross_dn(df, idx)),
    "both_70_x_dn": ("Till Both >70 Then Both Cross Below SMA(14)", lambda df, idx: _exit_both_ob70_then_cross_dn(df, idx)),
    # Sequential: sortino positive then turns negative
    "sort_pos_neg": ("Till Sortino >0 Then <0", lambda df, idx: _exit_seq_above_then_below(df, idx, "_sortino", 0)),
    # Sequential: omega above 1 then below 1
    "omega_1_lt1": ("Till Omega >1 Then <1", lambda df, idx: _exit_seq_above_then_below(df, idx, "_omega", 1)),
    # Sequential: RSI above 50 then cross down
    "rsi_50_x_dn": ("Till RSI >50 Then Crosses Below SMA(14)", lambda df, idx: _exit_level_then_cross_dn(df, idx, "_rsi", 50, 14)),
    # Sequential: RS above 50 then cross down
    "rs_50_x_dn": ("Till RSI-Sort >50 Then Crosses Below SMA(14)", lambda df, idx: _exit_level_then_cross_dn(df, idx, "_rsi_sort", 50, 14)),
    # Sequential: RSI above 60 then cross down
    "rsi_60_x_dn": ("Till RSI >60 Then Crosses Below SMA(14)", lambda df, idx: _exit_level_then_cross_dn(df, idx, "_rsi", 60, 14)),
    # Sequential: price above SMA20 then below SMA20
    "sma20_up_dn": ("Till Price >SMA20 Then <SMA20", lambda df, idx: _exit_price_above_then_below_sma(df, idx, 20)),
    # Sequential: price above SMA50 then below SMA50
    "sma50_up_dn": ("Till Price >SMA50 Then <SMA50", lambda df, idx: _exit_price_above_then_below_sma(df, idx, 50)),
    # RSI of Sortino cross up then cross down (full cycle)
    "rs_cycle": ("Till RSI-Sort Crosses Up Then Down", lambda df, idx: _exit_rs_cross_up_then_down(df, idx)),
    # RSI cross up then cross down (full cycle)
    "rsi_cycle": ("Till RSI Crosses Up Then Down", lambda df, idx: _exit_rsi_cross_up_then_down(df, idx)),
}


def _categorize(sig_key):
    if "macd_great" in sig_key and ("rsi" in sig_key or "omega" in sig_key): return "Multi-Indicator"
    if sig_key.startswith("rsi_of_sortino") or sig_key.startswith("rs_"): return "RSI of Sortino"
    if "rsi_sort_x" in sig_key or "rsi_sort" in sig_key: return "RSI of Sortino"
    if "higher_low" in sig_key or "lower_high" in sig_key: return "Higher Low"
    if "sort" in sig_key and "rsi" not in sig_key and "omega" not in sig_key: return "Sortino"
    if "omega" in sig_key and "rsi" not in sig_key and "sort" not in sig_key: return "Omega"
    if "triple" in sig_key or ("sort" in sig_key and "omega" in sig_key) or ("sort" in sig_key and "rsi" in sig_key and "rsi_of" not in sig_key) or ("omega" in sig_key and "rsi" in sig_key): return "Multi-Indicator"
    if "seq_" in sig_key: return "Sequential"
    if "obv_" in sig_key or "ad_" in sig_key or "cmf_" in sig_key or "vwap_" in sig_key or "pocket_" in sig_key or "tight_" in sig_key or "sma50_bounce" in sig_key or "high_vol_narrow" in sig_key or "vol_dry" in sig_key or "vol_shrink" in sig_key: return "Institutional"
    if "dn_" in sig_key or "spread_" in sig_key: return "Capture Ratio"
    if "weekly" in sig_key or "50_x_sma50" in sig_key or "rsi50" in sig_key: return "Multi-Timeframe"
    if "recovery" in sig_key or "div" in sig_key: return "Divergence"
    if "gap" in sig_key: return "Gap"
    if "rsi" in sig_key: return "RSI"
    if "sma" in sig_key or "cross" in sig_key or "golden" in sig_key or "death" in sig_key: return "Moving Average"
    if "vol" in sig_key: return "Volume"
    if "down" in sig_key and "gap" not in sig_key and len(sig_key) < 8: return "Streak"
    if "up" in sig_key and "gap" not in sig_key and len(sig_key) < 6: return "Streak"
    if "52" in sig_key or "20" in sig_key: return "Breakout"
    if "engulf" in sig_key or "doji" in sig_key or "hammer" in sig_key or "big_" in sig_key: return "Candlestick"
    if "macd" in sig_key: return "MACD"
    if "boll" in sig_key: return "Bollinger"
    if sig_key in ("monday", "friday"): return "Calendar"
    return "Other"


def generate_studies():
    """Generate ALL signal × exit combinations."""
    studies = []
    sid = 0

    # Generate every possible combination
    for sig_key in SIGNALS:
        if sig_key in ALT_SIGNAL_KEYS:
            continue  # per-stock alt-data signals are meaningless on sector ETFs
        sig_name, _ = SIGNALS[sig_key]
        for exit_key in EXITS:
            exit_name, _ = EXITS[exit_key]
            sid += 1
            studies.append({
                "id": sid,
                "signal": sig_key,
                "signal_name": sig_name,
                "exit": exit_key,
                "exit_name": exit_name,
                "name": f"{sig_name} -> {exit_name}",
                "category": _categorize(sig_key),
            })

    return studies


def _generate_studies_old():
    """Old hand-picked combos - kept for reference."""
    studies = []
    sid = 0

    combos = [
        # Gaps
        ("gap_up", ["1d", "3d", "1w", "2w", "4w", "rsi_x_dn", "gap_down", "trail_5"]),
        ("gap_down", ["1d", "3d", "1w", "2w", "4w", "8w", "rsi_x_up", "gap_up", "trail_5"]),
        ("gap_up_large", ["1d", "3d", "1w", "2w", "gap_down", "rsi_x_dn"]),
        ("gap_down_large", ["1d", "3d", "1w", "2w", "4w", "8w", "gap_up", "rsi_x_up"]),
        # RSI
        ("rsi_x_above_sma", ["1w", "2w", "4w", "rsi_x_dn", "trail_5", "tp_5"]),
        ("rsi_x_below_sma", ["1w", "2w", "4w", "rsi_x_up"]),
        ("rsi_x_sma_below50", ["1w", "2w", "4w", "8w", "rsi_x_dn", "rsi_ob", "trail_5"]),
        ("rsi_oversold30", ["1d", "3d", "1w", "2w", "4w", "8w", "rsi_x_up", "rsi_ob", "trail_5"]),
        ("rsi_oversold20", ["1d", "3d", "1w", "2w", "4w", "rsi_x_up"]),
        ("rsi_overbought", ["1d", "3d", "1w", "2w", "rsi_x_dn", "rsi_os"]),
        ("rsi_x50_up", ["1w", "2w", "4w", "rsi_x_dn"]),
        # Moving averages
        ("price_x_sma20_up", ["1w", "2w", "4w", "8w", "rsi_x_dn", "trail_5"]),
        ("price_x_sma20_dn", ["1w", "2w", "4w", "rsi_x_up"]),
        ("golden_cross", ["2w", "4w", "8w", "trail_10", "macd_x_dn"]),
        ("death_cross", ["2w", "4w", "8w"]),
        # Volume
        ("vol_spike_up", ["1d", "3d", "1w", "2w", "trail_5"]),
        ("vol_spike_down", ["1d", "3d", "1w", "2w", "4w", "trail_5"]),
        # Streaks
        ("3down", ["1d", "3d", "1w", "2w", "4w", "rsi_x_up"]),
        ("3up", ["1d", "3d", "1w", "rsi_x_dn"]),
        ("5down", ["1d", "3d", "1w", "2w", "4w", "8w"]),
        # Breakouts
        ("new_20high", ["1w", "2w", "4w", "trail_5", "rsi_x_dn"]),
        ("new_20low", ["1w", "2w", "4w", "8w", "rsi_x_up"]),
        ("new_52high", ["2w", "4w", "8w", "trail_10"]),
        ("new_52low", ["2w", "4w", "8w", "rsi_x_up"]),
        # Candlestick
        ("bull_engulf", ["1d", "3d", "1w", "2w"]),
        ("bear_engulf", ["1d", "3d", "1w", "2w"]),
        ("doji", ["1d", "3d", "1w"]),
        ("hammer", ["1d", "3d", "1w", "2w"]),
        ("big_red", ["1d", "3d", "1w", "2w", "4w"]),
        ("big_green", ["1d", "3d", "1w"]),
        # MACD
        ("macd_x_up", ["1w", "2w", "4w", "macd_x_dn", "trail_5"]),
        ("macd_x_down", ["1w", "2w", "4w"]),
        # Bollinger
        ("boll_lower", ["1d", "3d", "1w", "2w", "boll_upper" if "boll_upper" in EXITS else "2w"]),
        ("boll_upper", ["1d", "3d", "1w"]),
        # Calendar
        ("monday", ["1d", "3d", "1w"]),
        ("friday", ["1d", "3d"]),
        # Weekly momentum
        ("weekly_up3", ["1w", "2w", "4w", "rsi_x_dn", "trail_5"]),
        ("weekly_down3", ["1w", "2w", "4w", "8w", "rsi_x_up"]),
        # Volume
        ("high_vol", ["1d", "3d", "1w", "2w"]),
        ("low_vol", ["1d", "3d", "1w"]),
        # Mean reversion
        ("at_sma20", ["1d", "3d", "1w"]),
        # Range
        ("narrow_range", ["1d", "3d", "1w", "2w"]),
        ("wide_range", ["1d", "3d", "1w"]),
        # Gap combos with longer exits
        ("gap_down_med", ["4w", "8w", "rsi_x_up", "rsi_ob", "trail_5"]),
        ("gap_up_med", ["rsi_x_dn", "gap_down", "trail_3"]),
        # RSI crossover combos
        ("rsi_x_sma_b50", ["1w", "2w", "4w", "8w", "12w", "rsi_x_dn", "rsi_ob", "sort_neg", "omega_lt1", "trail_5", "trail_10", "tp_5", "tp_10"]),
        ("rsi_x_pos_sort", ["1w", "2w", "4w", "8w", "rsi_x_dn", "sort_neg", "omega_lt1", "trail_5", "tp_5"]),
        ("rsi_x_pos_omega", ["1w", "2w", "4w", "8w", "rsi_x_dn", "sort_neg", "omega_lt1", "trail_5", "tp_5"]),
        ("rsi_x_neg_sort", ["1d", "3d", "1w", "2w", "4w", "rsi_x_up"]),
        ("sort_pos", ["1w", "2w", "4w", "8w", "rsi_x_dn", "sort_neg", "trail_5", "trail_10"]),
        ("omega_x1", ["1w", "2w", "4w", "8w", "rsi_x_dn", "omega_lt1", "trail_5"]),
        ("rsi_x_triple", ["1w", "2w", "4w", "8w", "12w", "rsi_x_dn", "sort_neg", "omega_lt1", "trail_5", "trail_10", "tp_5", "tp_10"]),
        # Sortino-only entries
        ("sort_neg", ["1d", "3d", "1w", "2w", "4w", "rsi_x_up", "sort_pos"]),
        ("sort_above_1", ["1w", "2w", "4w", "8w", "rsi_x_dn", "sort_neg", "trail_5"]),
        ("sort_below_neg1", ["1d", "3d", "1w", "2w", "4w", "8w", "rsi_x_up"]),
        # Omega-only entries
        ("omega_below_1", ["1d", "3d", "1w", "2w", "4w", "rsi_x_up", "omega_lt1"]),
        ("omega_above_2", ["1w", "2w", "4w", "rsi_x_dn", "omega_lt1", "trail_5"]),
        # RSI + Sortino/Omega combos
        ("rsi_os30_sort_neg", ["1d", "3d", "1w", "2w", "4w", "8w", "rsi_x_up", "sort_pos", "rsi_ob"]),
        ("rsi_ob70_sort_pos", ["1d", "3d", "1w", "2w", "rsi_x_dn", "sort_neg"]),
        ("rsi_x50_omega_pos", ["1w", "2w", "4w", "8w", "rsi_x_dn", "omega_lt1", "trail_5"]),
        # Gap + indicator combos
        ("gap_down_rsi_os", ["1d", "3d", "1w", "2w", "4w", "8w", "rsi_x_up", "rsi_ob"]),
        ("gap_up_omega_pos", ["1d", "3d", "1w", "2w", "rsi_x_dn", "omega_lt1"]),
        ("gap_down_sort_neg", ["1d", "3d", "1w", "2w", "4w", "8w", "rsi_x_up", "sort_pos"]),
        # Sortino + Omega together
        ("sort_pos_omega_pos", ["1w", "2w", "4w", "8w", "rsi_x_dn", "sort_neg", "omega_lt1", "trail_5"]),
        # Multi-timeframe (weekly proxy)
        ("rsi50_x_sma50_up", ["2w", "4w", "8w", "12w", "rsi_x_dn", "sort_neg", "trail_10"]),
        ("rsi_weekly_x_up", ["2w", "4w", "8w", "12w", "rsi_x_dn", "trail_10"]),
        # RSI recovery / divergence
        ("rsi_recovery", ["1w", "2w", "4w", "8w", "rsi_x_dn", "rsi_ob", "trail_5"]),
        ("rsi_bull_div", ["1d", "3d", "1w", "2w", "4w", "rsi_x_up", "rsi_ob"]),
        # More risk management combos on best signals
        ("rsi_oversold30", ["sort_neg", "omega_lt1", "rsi_x50_dn", "sl_5", "sl_10", "tp_3", "tp_15", "trail_7", "trail_15"]),
        ("gap_down_large", ["sort_neg", "omega_lt1", "sl_5", "tp_15", "trail_7", "trail_15", "12w"]),
        ("new_52low", ["sort_neg", "omega_lt1", "trail_7", "trail_15", "12w", "sl_5", "tp_15"]),
        # Cross combos with Sortino/Omega exits
        ("golden_cross", ["sort_neg", "omega_lt1", "rsi_x_dn", "12w", "trail_15"]),
        ("macd_x_up", ["sort_neg", "omega_lt1", "rsi_x_dn", "8w", "12w"]),
        ("price_x_sma20_up", ["sort_neg", "omega_lt1", "rsi_x_dn", "rsi_x50_dn"]),
        ("price_x_sma50_up", ["sort_neg", "omega_lt1", "4w", "8w", "12w"]),
        # All triple weakness/strength combos
        ("rsi_x50dn_omega_lt1_sort_neg", ["1d", "3d", "1w", "2w", "4w", "8w", "rsi_x_up", "sort_pos", "omega_lt1"]),
        ("rsi_xsma_omega_lt1_sort_neg", ["1d", "3d", "1w", "2w", "4w", "8w", "rsi_x_up", "sort_pos"]),
        ("rsi_xsma_omega_gt1_sort_neg", ["1w", "2w", "4w", "rsi_x_dn", "sort_pos"]),
        ("rsi_xsma_omega_lt1_sort_pos", ["1w", "2w", "4w", "rsi_x_dn", "omega_lt1"]),
        ("rsi_lt30_omega_lt1_sort_neg", ["1d", "3d", "1w", "2w", "4w", "8w", "12w", "rsi_x_up", "rsi_ob", "sort_pos"]),
        ("rsi_lt30_omega_lt1_sort_ltn1", ["1d", "3d", "1w", "2w", "4w", "8w", "12w", "rsi_x_up", "sort_pos"]),
        ("rsi_x50up_omega_gt1_sort_pos", ["1w", "2w", "4w", "8w", "rsi_x_dn", "sort_neg", "omega_lt1", "trail_5"]),
        ("rsi_x50up_omega_lt1_sort_neg", ["1d", "3d", "1w", "2w", "4w", "rsi_x_up", "sort_pos"]),
        ("rsi_gt70_omega_gt2_sort_gt1", ["1d", "3d", "1w", "2w", "rsi_x_dn", "sort_neg", "omega_lt1"]),
        ("rsi_xsma_dn_omega_lt1_sort_neg", ["1d", "3d", "1w", "2w", "4w", "8w", "rsi_x_up", "sort_pos"]),
        # RSI of Sortino combos
        ("rsi_of_sortino_x_up", ["1w", "2w", "4w", "8w", "rsi_x_dn", "sort_neg", "omega_lt1", "trail_5"]),
        ("rsi_of_sortino_x_up_b50", ["1w", "2w", "4w", "8w", "12w", "rsi_x_dn", "sort_neg", "trail_5", "trail_10"]),
        ("rsi_of_sortino_x_dn", ["1d", "3d", "1w", "2w", "rsi_x_up"]),
        ("rsi_of_sortino_os", ["1d", "3d", "1w", "2w", "4w", "8w", "rsi_x_up", "sort_pos"]),
        ("rsi_of_sortino_ob", ["1d", "3d", "1w", "rsi_x_dn", "sort_neg"]),
        ("rsi_sort_x_up_rsi_x_up", ["1w", "2w", "4w", "8w", "rsi_x_dn", "sort_neg", "omega_lt1", "trail_5"]),
        # Sortino trending up (pure)
        ("sort_ma3_up", ["1d", "3d", "1w", "2w", "4w", "sort_neg", "rsi_x_dn", "trail_5"]),
        ("sort_ma5_up", ["1d", "3d", "1w", "2w", "4w", "8w", "sort_neg", "rsi_x_dn", "trail_5"]),
        ("sort_ma7_up", ["1w", "2w", "4w", "8w", "sort_neg", "rsi_x_dn"]),
        ("sort_ma10_up", ["1w", "2w", "4w", "8w", "sort_neg", "rsi_x_dn", "trail_5"]),
        ("sort_ma20_up", ["2w", "4w", "8w", "12w", "sort_neg", "trail_10"]),
        # Sortino trending down (mean reversion buys)
        ("sort_ma3_dn", ["1d", "3d", "1w", "2w", "4w", "rsi_x_up", "sort_pos"]),
        ("sort_ma5_dn", ["1d", "3d", "1w", "2w", "4w", "rsi_x_up", "sort_pos"]),
        ("sort_ma10_dn", ["1d", "3d", "1w", "2w", "4w", "8w", "rsi_x_up"]),
        # Sortino golden/death cross
        ("sort_3x10_up", ["1w", "2w", "4w", "8w", "sort_neg", "omega_lt1", "rsi_x_dn", "trail_5"]),
        ("sort_5x20_up", ["2w", "4w", "8w", "12w", "sort_neg", "rsi_x_dn", "trail_10"]),
        ("sort_3x10_dn", ["1d", "3d", "1w", "2w", "4w", "rsi_x_up", "sort_pos"]),
        # Sortino trending + RSI combos
        ("sort_ma5_up_rsi_xsma", ["1w", "2w", "4w", "8w", "rsi_x_dn", "sort_neg", "omega_lt1", "trail_5"]),
        ("sort_ma5_up_rsi_lt50", ["1w", "2w", "4w", "8w", "rsi_x_dn", "sort_neg", "trail_5"]),
        ("sort_ma10_up_rsi_xsma", ["1w", "2w", "4w", "8w", "rsi_x_dn", "sort_neg", "trail_5"]),
        # Sortino trending + Omega
        ("sort_ma5_up_omega_gt1", ["1w", "2w", "4w", "8w", "rsi_x_dn", "sort_neg", "omega_lt1", "trail_5"]),
        ("sort_ma10_up_omega_gt1", ["1w", "2w", "4w", "8w", "sort_neg", "omega_lt1"]),
        # Triple: Sortino trend + RSI + Omega
        ("sort_ma5_up_rsi_xsma_omega", ["1w", "2w", "4w", "8w", "rsi_x_dn", "sort_neg", "omega_lt1", "trail_5"]),
        ("sort_3x10_up_rsi_xsma", ["1w", "2w", "4w", "8w", "rsi_x_dn", "sort_neg", "trail_5"]),
        ("sort_3x10_up_omega_rsi", ["1w", "2w", "4w", "8w", "rsi_x_dn", "sort_neg", "omega_lt1"]),
        # Sortino trending up from negative
        ("sort_ma5_up_from_neg", ["1d", "3d", "1w", "2w", "4w", "8w", "rsi_x_up", "sort_pos", "rsi_ob"]),
        ("sort_ma10_up_from_neg", ["1d", "3d", "1w", "2w", "4w", "8w", "rsi_x_up", "sort_pos"]),
        # Gap + Sortino trending
        ("gap_dn_sort_ma5_up", ["1d", "3d", "1w", "2w", "4w", "rsi_x_up", "sort_neg"]),
        ("gap_dn_sort_ma5_dn", ["1d", "3d", "1w", "2w", "4w", "8w", "rsi_x_up", "sort_pos"]),

        # ── 200 MORE COMBOS ──

        # Existing best signals with ALL exit types
        ("new_52low", ["1d", "3d", "1w", "rsi_x_up", "rsi_ob", "gap_up", "trail_3", "sl_3", "tp_3", "macd_x_dn"]),
        ("gap_down_large", ["1d", "3d", "rsi_x_up", "rsi_ob", "gap_up", "trail_3", "macd_x_dn", "rsi_x50_dn"]),
        ("rsi_oversold30", ["12w", "gap_up", "gap_down", "trail_3", "trail_7", "macd_x_dn"]),
        ("rsi_oversold20", ["8w", "12w", "rsi_ob", "trail_5", "trail_10", "sort_neg", "omega_lt1"]),

        # Omega trending up (MA on Omega)
        ("omega_above_2", ["1d", "3d", "8w", "12w", "sort_neg", "rsi_x_dn", "trail_10", "macd_x_dn"]),
        ("omega_x1", ["1d", "3d", "sort_neg", "omega_lt1", "macd_x_dn", "trail_3", "trail_7"]),

        # Volume + RSI + Sortino combos
        ("vol_spike_up", ["4w", "8w", "rsi_x_dn", "sort_neg", "omega_lt1", "trail_10"]),
        ("vol_spike_down", ["8w", "12w", "rsi_x_up", "sort_pos", "omega_lt1", "trail_10"]),

        # Streaks + indicator combos
        ("3down", ["8w", "12w", "sort_pos", "omega_lt1", "trail_10", "trail_15"]),
        ("5down", ["12w", "sort_pos", "rsi_ob", "trail_10", "trail_15", "tp_15"]),
        ("3up", ["4w", "8w", "sort_neg", "omega_lt1", "trail_5", "trail_10"]),

        # Breakout + indicator combos
        ("new_20high", ["8w", "12w", "sort_neg", "omega_lt1", "trail_10", "trail_15", "macd_x_dn"]),
        ("new_20low", ["8w", "12w", "sort_pos", "trail_10", "trail_15", "tp_15"]),
        ("new_52high", ["1d", "3d", "1w", "sort_neg", "omega_lt1", "trail_5", "macd_x_dn"]),

        # Candlestick + indicator combos
        ("bull_engulf", ["4w", "8w", "rsi_x_dn", "sort_neg", "omega_lt1", "trail_5"]),
        ("bear_engulf", ["4w", "8w", "rsi_x_up", "sort_pos"]),
        ("hammer", ["4w", "8w", "rsi_x_dn", "sort_neg", "trail_5"]),
        ("doji", ["4w", "8w", "rsi_x_dn", "rsi_x_up"]),
        ("big_red", ["8w", "12w", "sort_pos", "rsi_ob", "trail_10", "trail_15"]),
        ("big_green", ["4w", "8w", "sort_neg", "omega_lt1", "rsi_x_dn"]),

        # MACD + Sortino/Omega combos
        ("macd_x_up", ["trail_3", "trail_7", "trail_15", "tp_3", "tp_15", "sl_3", "sl_10"]),
        ("macd_x_down", ["8w", "12w", "rsi_x_up", "sort_pos", "trail_10"]),

        # Bollinger + indicator combos
        ("boll_lower", ["4w", "8w", "rsi_x_up", "sort_pos", "omega_lt1", "trail_5", "trail_10"]),
        ("boll_upper", ["4w", "rsi_x_dn", "sort_neg", "omega_lt1"]),

        # Golden/Death cross + more exits
        ("golden_cross", ["1w", "2w", "4w", "trail_3", "trail_7", "sl_3", "sl_5", "tp_3", "tp_5"]),
        ("death_cross", ["1d", "3d", "1w", "rsi_x_up", "sort_pos", "trail_5"]),

        # SMA crosses + all exit types
        ("price_x_sma20_up", ["12w", "trail_3", "trail_7", "trail_15", "sl_3", "tp_3", "tp_15", "macd_x_dn"]),
        ("price_x_sma20_dn", ["8w", "12w", "rsi_x_up", "sort_pos", "omega_lt1", "trail_10"]),
        ("price_x_sma50_up", ["1w", "2w", "rsi_x_dn", "trail_3", "trail_5", "trail_7"]),

        # RSI crossover from different levels + all exits
        ("rsi_x_above_sma", ["8w", "12w", "sort_neg", "omega_lt1", "trail_3", "trail_7", "trail_10", "sl_3", "sl_5"]),
        ("rsi_x_below_sma", ["8w", "12w", "sort_pos", "omega_lt1", "trail_5", "trail_10"]),
        ("rsi_x50_up", ["8w", "12w", "sort_neg", "omega_lt1", "trail_5", "trail_10", "macd_x_dn"]),
        ("rsi_x50_down", ["8w", "12w", "rsi_x_up", "sort_pos", "trail_5"]),
        ("rsi_overbought", ["4w", "8w", "sort_neg", "omega_lt1", "trail_5", "trail_10"]),

        # Gap up combos with all exit types
        ("gap_up", ["8w", "12w", "sort_neg", "omega_lt1", "rsi_x50_dn", "trail_10", "trail_15", "macd_x_dn"]),
        ("gap_down", ["12w", "sort_pos", "rsi_ob", "trail_10", "trail_15", "tp_15", "macd_x_dn"]),
        ("gap_up_med", ["1d", "3d", "1w", "4w", "8w", "sort_neg", "omega_lt1", "trail_5"]),
        ("gap_down_med", ["1d", "3d", "1w", "2w", "sort_pos", "trail_3"]),

        # Sortino trending + more exits
        ("sort_ma3_up", ["8w", "12w", "omega_lt1", "trail_10", "trail_15", "tp_5", "tp_10"]),
        ("sort_ma5_up", ["12w", "omega_lt1", "trail_10", "trail_15", "tp_10", "tp_15", "macd_x_dn"]),
        ("sort_ma10_up", ["12w", "omega_lt1", "trail_15", "tp_15", "macd_x_dn"]),
        ("sort_ma3_dn", ["8w", "12w", "trail_10", "trail_15"]),
        ("sort_ma5_dn", ["8w", "12w", "trail_10"]),
        ("sort_3x10_up", ["12w", "trail_15", "tp_10", "tp_15", "macd_x_dn"]),
        ("sort_5x20_up", ["sort_neg", "omega_lt1", "trail_5", "trail_7", "macd_x_dn"]),

        # Weekly/multi-timeframe + more exits
        ("rsi50_x_sma50_up", ["1w", "2w", "trail_5", "trail_7", "sl_5", "tp_5"]),
        ("rsi_weekly_x_up", ["1w", "2w", "trail_5", "trail_7", "sl_5"]),
        ("weekly_up3", ["8w", "12w", "sort_neg", "omega_lt1", "trail_10", "trail_15"]),
        ("weekly_down3", ["12w", "rsi_x_up", "sort_pos", "trail_10", "trail_15"]),

        # Calendar + more exits
        ("monday", ["2w", "4w", "rsi_x_dn", "trail_5"]),
        ("friday", ["1w", "2w", "rsi_x_dn", "trail_5"]),

        # Narrow/Wide range + more exits
        ("narrow_range", ["4w", "8w", "rsi_x_dn", "sort_neg", "trail_5"]),
        ("wide_range", ["4w", "8w", "rsi_x_dn", "sort_neg", "trail_5"]),

        # RSI of Sortino + more exits
        ("rsi_of_sortino_x_up", ["12w", "trail_10", "trail_15", "tp_10", "tp_15", "macd_x_dn"]),
        ("rsi_of_sortino_x_up_b50", ["sort_neg", "omega_lt1", "trail_3", "trail_7", "macd_x_dn"]),
        ("rsi_of_sortino_os", ["12w", "trail_10", "trail_15", "tp_15"]),

        # Triple combos + more exits
        ("rsi_x_triple", ["trail_3", "trail_7", "trail_15", "sl_3", "sl_5", "sl_10", "tp_3", "tp_15", "macd_x_dn"]),
        ("rsi_x_pos_sort", ["12w", "trail_3", "trail_7", "trail_10", "trail_15", "sl_5", "tp_3", "tp_15"]),
        ("rsi_x_pos_omega", ["12w", "trail_3", "trail_7", "trail_15", "sl_5", "tp_3", "tp_15"]),
        ("sort_pos_omega_pos", ["12w", "trail_3", "trail_7", "trail_10", "trail_15", "macd_x_dn"]),

        # Sortino trending + RSI + more exits
        ("sort_ma5_up_rsi_xsma", ["12w", "trail_3", "trail_7", "trail_10", "trail_15", "tp_10", "tp_15"]),
        ("sort_ma5_up_rsi_xsma_omega", ["12w", "trail_3", "trail_7", "trail_10", "trail_15", "macd_x_dn"]),
        ("sort_3x10_up_rsi_xsma", ["12w", "trail_3", "trail_10", "trail_15", "macd_x_dn"]),
        ("sort_3x10_up_omega_rsi", ["12w", "trail_3", "trail_10", "trail_15"]),

        # Triple weakness/strength + more exits
        ("rsi_lt30_omega_lt1_sort_neg", ["trail_3", "trail_5", "trail_7", "trail_10", "trail_15", "tp_3", "tp_5", "tp_10", "tp_15"]),
        ("rsi_lt30_omega_lt1_sort_ltn1", ["trail_3", "trail_5", "trail_7", "trail_10", "trail_15", "tp_10", "tp_15"]),
        ("rsi_x50up_omega_gt1_sort_pos", ["12w", "trail_3", "trail_7", "trail_10", "trail_15", "macd_x_dn"]),
        ("rsi_xsma_omega_lt1_sort_neg", ["12w", "trail_5", "trail_10", "trail_15", "tp_10", "tp_15"]),
        ("rsi_x50dn_omega_lt1_sort_neg", ["12w", "trail_5", "trail_10", "trail_15"]),

        # Divergence + more exits
        ("rsi_recovery", ["12w", "sort_neg", "omega_lt1", "trail_10", "trail_15", "tp_10", "tp_15"]),
        ("rsi_bull_div", ["8w", "12w", "sort_pos", "trail_10", "trail_15", "tp_10"]),

        # Sortino from negative + more exits
        ("sort_ma5_up_from_neg", ["12w", "trail_5", "trail_10", "trail_15", "tp_10", "tp_15"]),
        ("sort_ma10_up_from_neg", ["12w", "trail_5", "trail_10", "trail_15", "tp_10"]),

        # Gap + Sortino trending + more exits
        ("gap_dn_sort_ma5_up", ["8w", "12w", "trail_5", "trail_10", "trail_15", "sort_pos"]),
        ("gap_dn_sort_ma5_dn", ["12w", "trail_10", "trail_15", "tp_15"]),
    ]

    for sig_key, exit_keys in combos:
        if sig_key not in SIGNALS:
            continue
        sig_name, _ = SIGNALS[sig_key]
        for exit_key in exit_keys:
            if exit_key not in EXITS:
                continue
            sid += 1
            exit_name, _ = EXITS[exit_key]
            studies.append({
                "id": sid,
                "signal": sig_key,
                "signal_name": sig_name,
                "exit": exit_key,
                "exit_name": exit_name,
                "name": f"{sig_name} -> {exit_name}",
                "category": _categorize(sig_key),
            })

    return studies


def _precompute_indicators(all_data):
    """Pre-compute RSI, Sortino, Omega for all tickers once."""
    cache = {}
    for ticker, df in all_data.items():
        if len(df) < 20:
            continue
        rsi = ta.momentum.rsi(df["Close"], window=10)
        sortino = _rolling_sortino(df)
        omega = _rolling_omega(df)
        cache[ticker] = {"rsi": rsi, "sortino": sortino, "omega": omega}
    return cache

# Module-level indicator cache
_indicator_cache = {}

_rates_cache = None
_market_cache = None

def _get_rates():
    global _rates_cache
    if _rates_cache is None:
        try:
            import rates as rates_mod
            _rates_cache = rates_mod.get_rates()
        except Exception:
            _rates_cache = pd.DataFrame()
    return _rates_cache

def _get_market():
    global _market_cache
    if _market_cache is None:
        try:
            import market_regime
            _market_cache = market_regime.get_market_data()
        except Exception:
            _market_cache = pd.DataFrame()
    return _market_cache


def run_study(study: dict, all_data: dict) -> dict:
    """Run a single study across all sector ETFs."""
    global _indicator_cache
    sig_key = study["signal"]
    exit_key = study["exit"]
    _, sig_fn = SIGNALS[sig_key]
    _, exit_fn = EXITS[exit_key]

    etf_to_sector = {v: k for k, v in config.SECTOR_ETFS.items()}
    sector_results = {}
    all_peak_rets = []
    all_peak_days = []
    all_ret_90ds = []
    all_regime_trades = {"LOW": [], "MEDIUM": [], "HIGH": []}
    all_curve_trades = {"NORMAL": [], "INVERTED": []}
    all_vix_trades = {"LOW_VIX": [], "MED_VIX": [], "HIGH_VIX": []}
    all_spy_trend_trades = {"BULL": [], "BEAR": []}
    all_season_trades = {"NOV_APR": [], "MAY_OCT": []}
    all_eff_returns = []  # one return per independent episode (overlap-deduped), pooled for t-stat

    for ticker, df in all_data.items():
        if ticker == config.BENCHMARK or len(df) < 60:
            continue

        sector = etf_to_sector.get(ticker, ticker)

        try:
            if sig_key == "rsi_x_pos_updn":
                spy_df = all_data.get(config.BENCHMARK)
                spy_close = spy_df["Close"] if spy_df is not None else None
                signals = sig_fn(df, spy_close=spy_close).fillna(False)
            elif sig_key in MARKET_SIGNAL_KEYS:
                spy_df = all_data.get(config.BENCHMARK)
                qqq_df = all_data.get("QQQ")
                spy_close = spy_df["Close"] if spy_df is not None else None
                qqq_close = qqq_df["Close"] if qqq_df is not None else None
                signals = sig_fn(df, spy_close=spy_close, qqq_close=qqq_close).fillna(False)
            else:
                signals = sig_fn(df).fillna(False)
        except Exception:
            continue

        entry_dates = signals[signals].index.tolist()
        entry_idxs = [df.index.get_loc(d) for d in entry_dates]
        episode_starts = _episode_starts(entry_idxs)  # overlap-dedup for the significance stat
        wins = losses = 0
        total_ret = 0
        trades = 0
        returns_list = []
        hold_days_list = []
        mae_list = []   # per-trade max adverse excursion (%)
        clean_list = [] # per-trade 1/0 clean-entry flag
        peak_rets = []  # per-trade peak return within 90d
        peak_days = []  # per-trade peak day
        ret_90ds = []   # per-trade 90d return
        close_arr = df["Close"].values
        low_arr = df["Low"].values
        n_bars = len(close_arr)

        for entry_date, idx in zip(entry_dates, entry_idxs):
            exit_idx = exit_fn(df, idx)
            if exit_idx is None or exit_idx <= idx:
                continue

            entry_price = float(close_arr[idx])
            if entry_price <= 0:
                continue
            exit_price = float(close_arr[exit_idx])
            ret = (exit_price - entry_price) / entry_price * 100
            hold = exit_idx - idx

            # Max adverse excursion over the hold (vectorized): worst intraday dip from entry.
            mae = trade_mae(entry_price, low_arr[idx + 1:exit_idx + 1])
            mae_list.append(mae)
            clean_list.append(1 if mae >= CLEAN_MAE_THRESH else 0)

            # Per-trade peak return + the day it peaked, over the FULL 90d forward window only
            # (gated identically to ret_90d below) so peak_avg / peak_day / ret_90d are all computed
            # over the SAME trade sample. Recent trades without a full 90d of forward data are
            # excluded rather than peaked over a truncated (biased-low) window.
            if idx + 90 < n_bars:
                fwd = (close_arr[idx + 1:idx + 91] / entry_price - 1.0) * 100
                d_best = int(fwd.argmax())
                if fwd[d_best] > 0:
                    peak_rets.append(float(fwd[d_best]))
                    peak_days.append(d_best + 1)
                else:
                    # Never went green within 90d: peak return floored at 0, but there is no
                    # meaningful "peak day" — record None so it doesn't pollute the day mode
                    # (previously stored as 0, which made peak_day report "peak on entry day").
                    peak_rets.append(0.0)
                    peak_days.append(None)
                ret_90ds.append((float(close_arr[idx + 90]) - entry_price) / entry_price * 100)

            trades += 1
            total_ret += ret
            returns_list.append(ret)
            is_ep = idx in episode_starts   # independent-episode start (overlap-dedup)
            if is_ep:
                all_eff_returns.append(ret)   # → parent significance pool
            hold_days_list.append(hold)
            if ret > 0:
                wins += 1
            else:
                losses += 1

            # Tag with pre-computed regime arrays (fast array lookup). Carry the episode flag
            # so each sub-bucket gets its own eff_trades + t_stat, not just a raw overlapping mean.
            if "_regime" in df.columns:
                regime = df["_regime"].iloc[idx]
                if regime in all_regime_trades:
                    all_regime_trades[regime].append((ret, is_ep))
            if "_curve" in df.columns:
                curve = df["_curve"].iloc[idx]
                if curve in all_curve_trades:
                    all_curve_trades[curve].append((ret, is_ep))
            if "_vix" in df.columns:
                vr = df["_vix"].iloc[idx]
                if vr in all_vix_trades:
                    all_vix_trades[vr].append((ret, is_ep))
            if "_spy_trend" in df.columns:
                st = df["_spy_trend"].iloc[idx]
                if st in all_spy_trend_trades:
                    all_spy_trend_trades[st].append((ret, is_ep))
            if "_season" in df.columns:
                ss = df["_season"].iloc[idx]
                if ss in all_season_trades:
                    all_season_trades[ss].append((ret, is_ep))

        if trades > 0:
            avg_ret = total_ret / trades
            win_rate = wins / trades * 100
            avg_hold = sum(hold_days_list) / len(hold_days_list)
            max_gain = max(returns_list)
            max_loss = min(returns_list)
            avg_mae = sum(mae_list) / len(mae_list)
            clean_pct = sum(clean_list) / len(clean_list) * 100
            sector_results[sector] = {
                "trades": trades, "avg_return": round(avg_ret, 3),
                "total_return": round(total_ret, 2), "win_rate": round(win_rate, 1),
                "wins": wins, "losses": losses,
                "avg_hold": round(avg_hold, 1), "max_gain": round(max_gain, 2),
                "max_loss": round(max_loss, 2),
                "avg_mae": round(avg_mae, 2), "clean_pct": round(clean_pct, 1),
            }
            all_peak_rets.extend(peak_rets)
            all_peak_days.extend(peak_days)
            all_ret_90ds.extend(ret_90ds)

    total_trades = sum(r["trades"] for r in sector_results.values())
    if total_trades == 0:
        return {**study, "total_trades": 0, "avg_return": 0, "win_rate": 0, "avg_hold": 0, "sectors": {}}

    weighted_ret = sum(r["avg_return"] * r["trades"] for r in sector_results.values()) / total_trades
    total_wins = sum(r["wins"] for r in sector_results.values())
    avg_hold = sum(r["avg_hold"] * r["trades"] for r in sector_results.values()) / total_trades
    avg_mae = sum(r["avg_mae"] * r["trades"] for r in sector_results.values()) / total_trades
    clean_pct = sum(r["clean_pct"] * r["trades"] for r in sector_results.values()) / total_trades

    sorted_s = sorted(sector_results.items(), key=lambda x: x[1]["avg_return"], reverse=True)
    # Floor the best/worst-sector ranking: sector_results keeps any sector with >=1 trade,
    # so an unfloored top-3/bottom-3 over ~93 correlated ETFs can headline a 1-2 trade noise
    # max. Require >=5 trades to be eligible; fall back to the full list if none qualify.
    MIN_HL_SECTOR_TRADES = 5
    ranked_s = [(s, d) for s, d in sorted_s if d["trades"] >= MIN_HL_SECTOR_TRADES] or sorted_s

    def _regime_stats(pairs):
        # pairs: list of (ret, is_episode). eff_trades/t_stat come from the deduped episode
        # subset so a small overlapping sub-bucket isn't read as significant.
        if not pairs:
            return None
        rets = [r for r, _ in pairs]
        eff = [r for r, is_ep in pairs if is_ep]
        w = sum(1 for r in rets if r > 0)
        return {"trades": len(rets), "avg_return": round(sum(rets)/len(rets), 3),
                "win_rate": round(w/len(rets)*100, 1),
                "eff_trades": len(eff), "t_stat": _tstat_from_returns(eff)}

    by_regime = {k: _regime_stats(v) for k, v in all_regime_trades.items() if v}
    by_curve = {k: _regime_stats(v) for k, v in all_curve_trades.items() if v}
    by_vix = {k: _regime_stats(v) for k, v in all_vix_trades.items() if v}
    by_spy_trend = {k: _regime_stats(v) for k, v in all_spy_trend_trades.items() if v}
    by_season = {k: _regime_stats(v) for k, v in all_season_trades.items() if v}

    # Peak / 90d aggregation
    peak_day = None
    peak_avg = None
    ret_90d = None
    best_peak_day = None
    best_peak_ret = None
    best_ret_90d = None
    if all_peak_rets:
        peak_avg = round(sum(all_peak_rets) / len(all_peak_rets), 2)
        best_peak_ret = round(max(all_peak_rets), 2)
        # Most common peak day (mode) over trades that actually peaked positive; never-green
        # trades carry peak_day=None and are excluded so the mode isn't dragged to 0.
        from collections import Counter
        valid_peak_days = [d for d in all_peak_days if d is not None]
        peak_day = Counter(valid_peak_days).most_common(1)[0][0] if valid_peak_days else None
        best_peak_day = all_peak_days[all_peak_rets.index(max(all_peak_rets))] if all_peak_rets else None
    if all_ret_90ds:
        ret_90d = round(sum(all_ret_90ds) / len(all_ret_90ds), 2)
        best_ret_90d = round(sorted(all_ret_90ds, reverse=True)[int(len(all_ret_90ds) * 0.1)] if len(all_ret_90ds) >= 10 else max(all_ret_90ds), 2)

    return {
        **study,
        "total_trades": total_trades,
        "eff_trades": len(all_eff_returns),
        "t_stat": _tstat_from_returns(all_eff_returns),
        "avg_return": round(weighted_ret, 3),
        "win_rate": round(total_wins / total_trades * 100, 1),
        "avg_hold": round(avg_hold, 1),
        "avg_mae": round(avg_mae, 2),
        "clean_pct": round(clean_pct, 1),
        "best_sectors": [{"sector": s, **d} for s, d in ranked_s[:3]],
        "worst_sectors": [{"sector": s, **d} for s, d in ranked_s[-3:]] if len(ranked_s) > 3 else [],
        "sector_count": len(sector_results),
        "peak_day": peak_day,
        "peak_avg": peak_avg,
        "ret_90d": ret_90d,
        "best_peak_day": best_peak_day,
        "best_peak_ret": best_peak_ret,
        "best_ret_90d": best_ret_90d,
        "by_regime": by_regime,
        "by_curve": by_curve,
        "by_vix": by_vix,
        "by_spy_trend": by_spy_trend,
        "by_season": by_season,
    }


_ALL_DATA = None

def _run_one_mp(study):
    """Top-level function for multiprocessing — uses forked global _ALL_DATA."""
    return run_study(study, _ALL_DATA)


def run_all():
    print("Generating studies...")
    studies = generate_studies()
    print(f"Generated {len(studies)} studies")

    print("Loading data...")
    all_data = data_fetcher.fetch_all()
    print(f"Loaded {len(all_data)} tickers")

    print("Pre-computing indicators (RSI, Sortino, Omega, regimes)...")
    t0 = time.time()

    # Pre-compute rate and market regime — vectorized with reindex
    rates_df = _get_rates()
    market_df = _get_market()

    # Build regime lookup series (reindexable)
    rates_regime = rates_df["regime"].astype(str) if len(rates_df) > 0 and "regime" in rates_df.columns else None
    rates_curve = rates_df["curve"].astype(str) if len(rates_df) > 0 and "curve" in rates_df.columns else None
    market_vix = market_df["vix_regime"].astype(str) if len(market_df) > 0 and "vix_regime" in market_df.columns else None
    market_spy = market_df["spy_trend"].astype(str) if len(market_df) > 0 and "spy_trend" in market_df.columns else None
    market_season = market_df["sell_in_may"].astype(str) if len(market_df) > 0 and "sell_in_may" in market_df.columns else None

    for ticker, df in all_data.items():
        if len(df) < 20:
            continue
        df["_sortino"] = _rolling_sortino(df)
        df["_omega"] = _rolling_omega(df)
        df["_rsi"] = ta.momentum.rsi(df["Close"], window=10)
        df["_rsi_sma"] = df["_rsi"].rolling(10).mean()
        df["_rsi_sort"] = _rsi_of_sortino(df)
        df["_rsi_sort_sma"] = df["_rsi_sort"].rolling(10).mean()

        # Vectorized regime assignment via reindex (no per-bar loop)
        idx = df.index
        if hasattr(idx, 'tz') and idx.tz is not None:
            idx = idx.tz_localize(None)

        if rates_regime is not None:
            df["_regime"] = rates_regime.reindex(idx, method="ffill").fillna("").values
        if rates_curve is not None:
            df["_curve"] = rates_curve.reindex(idx, method="ffill").fillna("").values
        if market_vix is not None:
            df["_vix"] = market_vix.reindex(idx, method="ffill").fillna("").values
        if market_spy is not None:
            df["_spy_trend"] = market_spy.reindex(idx, method="ffill").fillna("").values
        if market_season is not None:
            df["_season"] = market_season.reindex(idx, method="ffill").fillna("").values

    print(f"Pre-computed in {time.time() - t0:.1f}s")

    # Load existing from DB to skip already-computed studies
    existing = set()
    try:
        from core.models import Study as StudyModel
        for s in StudyModel.objects.filter(is_computed=True).values_list("signal_key", "exit_key"):
            existing.add(f"{s[0]}|{s[1]}")
        print(f"Loaded {len(existing)} existing studies from DB")
    except Exception as e:
        print(f"DB load failed, computing all: {e}")

    new_studies = [s for s in studies if f"{s['signal']}|{s['exit']}" not in existing]
    print(f"Existing: {len(existing)}, New to compute: {len(new_studies)}")

    start = time.time()
    import os
    import multiprocessing as mp

    # Store all_data in module global so forked processes inherit it
    global _ALL_DATA
    _ALL_DATA = all_data

    num_workers = min(os.cpu_count() or 4, 24)
    use_mp = os.name != 'nt'  # Use multiprocessing on Linux, threads on Windows

    if use_mp:
        mp.set_start_method('fork', force=True)
        print(f"Running {len(new_studies)} studies with {num_workers} processes (fork)...")
    else:
        from concurrent.futures import ThreadPoolExecutor, as_completed
        print(f"Running {len(new_studies)} studies with {num_workers} threads...")

    batch_size = 200
    computed_count = 0
    for batch_start in range(0, len(new_studies), batch_size):
        batch = new_studies[batch_start:batch_start + batch_size]
        batch_results = []

        if use_mp:
            with mp.Pool(processes=num_workers) as pool:
                batch_results = pool.map(_run_one_mp, batch)
        else:
            with ThreadPoolExecutor(max_workers=num_workers) as pool:
                futures = {pool.submit(lambda s: run_study(s, all_data), s): s for s in batch}
                for f in as_completed(futures):
                    batch_results.append(f.result())

        # Save batch to DB
        _save_batch_to_db(batch_results)
        computed_count += len(batch_results)

        elapsed = time.time() - start
        done = min(batch_start + batch_size, len(new_studies))
        print(f"  [{done}/{len(new_studies)}] {elapsed:.1f}s", flush=True)

    elapsed = time.time() - start
    total = len(existing) + computed_count
    print(f"\n{computed_count} new studies computed in {elapsed:.1f}s")
    print(f"Total in DB: {total}")

    # Print top 5 from DB
    try:
        from core.models import Study as StudyModel
        top5 = StudyModel.objects.filter(is_computed=True).order_by("-avg_return")[:5]
        print("\nTop 5:")
        for s in top5:
            print(f"  {s.name[:50]:50s}  avg={s.avg_return:+.3f}%  wr={s.win_rate:.0f}%  hold={s.avg_hold:.0f}d  trades={s.total_trades}")
        profitable = StudyModel.objects.filter(is_computed=True, avg_return__gt=0).count()
        total_db = StudyModel.objects.filter(is_computed=True).count()
        print(f"\nProfitable: {profitable}/{total_db}")
    except Exception:
        pass


def _save_batch_to_db(results):
    """Save a batch of study results to PostgreSQL."""
    try:
        from core.models import Study as StudyModel
        from django.utils import timezone

        now = timezone.now()
        for sd in results:
            fields = {
                "name": sd.get("name", ""),
                "signal_name": sd.get("signal_name", ""),
                "exit_name": sd.get("exit_name", ""),
                "category": sd.get("category", ""),
                "total_trades": sd.get("total_trades", 0),
                "avg_return": sd.get("avg_return", 0),
                "win_rate": sd.get("win_rate", 0),
                "avg_hold": sd.get("avg_hold", 0),
                "sector_count": sd.get("sector_count", 0),
                "peak_day": sd.get("peak_day"),
                "peak_avg": sd.get("peak_avg"),
                "ret_90d": sd.get("ret_90d"),
                "best_peak_day": sd.get("best_peak_day"),
                "best_peak_ret": sd.get("best_peak_ret"),
                "best_ret_90d": sd.get("best_ret_90d"),
                "by_regime": sd.get("by_regime"),
                "by_curve": sd.get("by_curve"),
                "by_vix": sd.get("by_vix"),
                "by_spy_trend": sd.get("by_spy_trend"),
                "by_season": sd.get("by_season"),
                "best_sectors": sd.get("best_sectors"),
                "worst_sectors": sd.get("worst_sectors"),
                "is_computed": True,
                "computed_at": now,
            }
            StudyModel.objects.update_or_create(
                signal_key=sd.get("signal", ""),
                exit_key=sd.get("exit", ""),
                defaults=fields,
            )
    except Exception as e:
        print(f"DB batch save error: {e}")


if __name__ == "__main__":
    import os, sys, django
    os.environ.setdefault("DJANGO_SETTINGS_MODULE", "rotation.settings")
    # Support running from repo root or from /app inside Docker
    for p in [Path(__file__).parent / "backend", Path("/app")]:
        if p.exists() and str(p) not in sys.path:
            sys.path.insert(0, str(p))
    django.setup()
    run_all()
