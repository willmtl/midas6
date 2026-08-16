#!/usr/bin/env python3
"""H4 (4-hour) short-horizon studies engine. See docs/superpowers/specs/2026-08-16-h4-short-horizon-engine-design.md.
Signal registry × bar-based exit ladder over the liquid top-250 universe; magnitude-bucketed
(tail-not-average); daily-trend split; daily-candle benchmark; saved to BacktestResult[h4_study].
Pure signal/agg code imports without Django; main() calls django.setup()."""
import numpy as np
import pandas as pd

RTH_HOURS = 6.5
TF_HOURS = 4
GAP = 3                                   # episode-dedup gap in bars
FIXED_BARS = [1, 2, 3, 4, 5, 6, 8, 10]    # 0-3 day focus (~½ day → ~6 days on 4h)


def day_label(bars):
    days = bars * TF_HOURS / RTH_HOURS
    return f"~{days:.1f}d" if days >= 1 else "~½ day"


import ta


def bucket_of(v, buckets):
    """Map a magnitude to its bucket label, or None if non-finite / out of range."""
    if v is None or not np.isfinite(v):
        return None
    for label, lo, hi in buckets:
        if lo <= v < hi:
            return label
    return None


def _fresh(cond):
    """True only on the bar a boolean condition first becomes True (rising edge)."""
    c = np.asarray(cond, bool)
    prev = np.concatenate([[False], c[:-1]])
    return c & ~prev


# ── bucket schemes ───────────────────────────────────────────────────────────
RSI_BUCKETS = [("<25", 0, 25), ("25-35", 25, 35), ("35-45", 35, 45), ("45-55", 45, 55), ("55+", 55, 200)]
Z_DN_BUCKETS = [("z<-3", -100, -3), ("-3..-2.5", -3, -2.5), ("-2.5..-2", -2.5, -2)]
GAP_DN_BUCKETS = [("<-4%", -100, -4), ("-4..-3%", -4, -3), ("-3..-2%", -3, -2)]
GAP_UP_BUCKETS = [("2..3%", 2, 3), ("3..4%", 3, 4), (">4%", 4, 100)]
PCTB_BUCKETS = [("<-0.2", -100, -0.2), ("-0.2..-0.1", -0.2, -0.1), ("-0.1..0", -0.1, 0)]
DEPTH_DN_BUCKETS = [("<-6%", -100, -6), ("-6..-3%", -6, -3), ("-3..0%", -3, 0)]
UP_BUCKETS = [("2..4%", 2, 4), ("4..7%", 4, 7), (">7%", 7, 200)]
KDOWN_BUCKETS = [("3", 3, 4), ("4", 4, 5), ("5+", 5, 100)]
RSI_OB_BUCKETS = [("60-70", 60, 70), ("70-80", 70, 80), ("80+", 80, 200)]
BREAK_BUCKETS = [("0..1%", 0, 1), ("1..3%", 1, 3), (">3%", 3, 200)]
DIV_BUCKETS = [("<-6%", -100, -6), ("-6..-3%", -6, -3), ("-3..0%", -3, 0)]
MAPULL_BUCKETS = [("<-2%", -100, -2), ("-2..-0.5%", -2, -0.5), ("-0.5..0.5%", -0.5, 0.5)]


# ── mean-reversion ───────────────────────────────────────────────────────────
def sig_mr_rsi_os(df):
    """RSI(14) crosses above its SMA(14); bucket by RSI level at the cross (mirrors rsi_4h_study)."""
    close = df["Close"]
    rsi = ta.momentum.rsi(close, window=14)
    sma = rsi.rolling(14).mean()
    up = (rsi > sma) & (rsi.shift(1) <= sma.shift(1))
    entry = up.fillna(False).values
    mag = np.where(entry, rsi.values, np.nan)
    return entry, mag


def sig_mr_volshock_dn(df):
    """Vol-normalized down shock: z = ret / trailing-vol(20) <= -2."""
    close = df["Close"]
    ret = close.pct_change()
    vol = ret.rolling(20).std()
    z = (ret / vol).replace([np.inf, -np.inf], np.nan)
    entry = (z <= -2).fillna(False).values
    mag = np.where(entry, z.values, np.nan)
    return entry, mag


def sig_mr_gap_dn(df):
    """Bar opens >=2% below the prior bar's close (gap-down)."""
    gap = (df["Open"] / df["Close"].shift(1) - 1) * 100
    entry = (gap <= -2).fillna(False).values
    mag = np.where(entry, gap.values, np.nan)
    return entry, mag


def sig_mr_bb_low(df):
    """Fresh close below the lower Bollinger(20,2) band (%B < 0)."""
    close = df["Close"]
    bb = ta.volatility.BollingerBands(close, window=20, window_dev=2)
    pctb = (close - bb.bollinger_lband()) / (bb.bollinger_hband() - bb.bollinger_lband())
    entry = _fresh((pctb < 0).fillna(False).values)
    mag = np.where(entry, pctb.values, np.nan)
    return entry, mag


def _newlow(df, n):
    close = df["Close"]
    prior_min = close.shift(1).rolling(n).min()
    fresh_low = _fresh((close < prior_min).fillna(False).values)
    depth = (close / prior_min - 1) * 100
    mag = np.where(fresh_low, depth.values, np.nan)
    return fresh_low, mag


def sig_mr_newlow30(df):
    """Fresh new 30-bar low; bucket by depth below the prior 30-bar low."""
    return _newlow(df, 30)


def sig_mr_newlow60(df):
    """Fresh new 60-bar low; bucket by depth below the prior 60-bar low."""
    return _newlow(df, 60)


def sig_mr_ndown(df):
    """A run of K>=3 consecutive down bars, entered on the REVERSAL bar (first non-down bar after the
    run ends); bucket by the completed run length K. No look-ahead: at the entry bar we only use the
    past run length and the current bar being non-down. Buckets stay meaningful (deeper runs -> higher K)."""
    close = df["Close"].values
    n = len(close)
    down = np.concatenate([[False], close[1:] < close[:-1]])
    run = np.zeros(n)
    for i in range(1, n):
        run[i] = run[i - 1] + 1 if down[i] else 0
    entry = np.zeros(n, bool)
    mag = np.full(n, np.nan)
    for i in range(n - 1):
        if run[i] >= 3 and not down[i + 1]:   # run just ended; enter the reversal bar i+1
            entry[i + 1] = True
            mag[i + 1] = run[i]
    return entry, mag


# ── momentum / breakout ──────────────────────────────────────────────────────
def sig_mo_burst(df):
    """2-bar cumulative up-burst >= +4%; bucket by burst magnitude."""
    close = df["Close"]
    two = (close / close.shift(2) - 1) * 100
    entry = _fresh((two >= 4).fillna(False).values)
    mag = np.where(entry, two.values, np.nan)
    return entry, mag


def sig_mo_break_hi(df):
    """Fresh new 30-bar high breakout; bucket by distance above the prior high."""
    close = df["Close"]
    prior_max = close.shift(1).rolling(30).max()
    entry = _fresh((close > prior_max).fillna(False).values)
    dist = (close / prior_max - 1) * 100
    mag = np.where(entry, dist.values, np.nan)
    return entry, mag


def sig_mo_rsi_ob(df):
    """RSI(14) crosses above 60 (momentum regime flip up); bucket by RSI level."""
    rsi = ta.momentum.rsi(df["Close"], window=14)
    entry = ((rsi > 60) & (rsi.shift(1) <= 60)).fillna(False).values
    mag = np.where(entry, rsi.values, np.nan)
    return entry, mag


def sig_mo_gap_up(df):
    """Bar opens >=2% above the prior bar's close (gap-up)."""
    gap = (df["Open"] / df["Close"].shift(1) - 1) * 100
    entry = (gap >= 2).fillna(False).values
    mag = np.where(entry, gap.values, np.nan)
    return entry, mag


# ── event-driven (price-based) ───────────────────────────────────────────────
def sig_ev_open_gap(df):
    """First 4h bar of each session; bucket by the signed overnight gap vs the prior session's close."""
    dates = df.index.normalize()
    first_bar = np.concatenate([[True], dates[1:] != dates[:-1]])
    gap = (df["Open"] / df["Close"].shift(1) - 1) * 100
    entry = first_bar & np.isfinite(gap.values)
    mag = np.where(entry, gap.values, np.nan)
    return entry, mag


# ── trend / structure ────────────────────────────────────────────────────────
def sig_st_ad_div(df):
    """A/D (ta.accdist / ADL) rising over 10 bars while price falls over 10 bars — bullish divergence.
    Bucket by how far price fell (deeper drop w/ rising ADL = stronger). Mirrors the daily _ad_rising edge."""
    adl = ta.volume.acc_dist_index(df["High"], df["Low"], df["Close"], df["Volume"])
    price_chg = (df["Close"] / df["Close"].shift(10) - 1) * 100
    adl_chg = adl - adl.shift(10)
    cond = (adl_chg > 0) & (price_chg < 0)
    entry = _fresh(cond.fillna(False).values)
    mag = np.where(entry, price_chg.values, np.nan)
    return entry, mag


def sig_st_ma_pull(df):
    """Pullback to a rising MA(20): price crosses down to <= MA while MA is rising and price was above."""
    close = df["Close"]
    ma = close.rolling(20).mean()
    rising = ma > ma.shift(5)
    cross_dn = (close <= ma) & (close.shift(1) > ma.shift(1))
    entry = (cross_dn & rising).fillna(False).values
    dist = (close / ma - 1) * 100
    mag = np.where(entry, dist.values, np.nan)
    return entry, mag


# ── registry ─────────────────────────────────────────────────────────────────
SIGNALS = {
    "mr_rsi_os":     {"name": "RSI(14) cross-up (oversold buckets)", "family": "mean_reversion",
                      "fn": sig_mr_rsi_os, "buckets": RSI_BUCKETS, "exit_fn": "rsi_x_dn"},
    "mr_volshock_dn":{"name": "Vol-normalized down shock (z<=-2)", "family": "mean_reversion",
                      "fn": sig_mr_volshock_dn, "buckets": Z_DN_BUCKETS, "exit_fn": None},
    "mr_gap_dn":     {"name": "Gap-down bar (>=2%)", "family": "mean_reversion",
                      "fn": sig_mr_gap_dn, "buckets": GAP_DN_BUCKETS, "exit_fn": None},
    "mr_bb_low":     {"name": "Below lower Bollinger (%B<0)", "family": "mean_reversion",
                      "fn": sig_mr_bb_low, "buckets": PCTB_BUCKETS, "exit_fn": None},
    "mr_newlow30":   {"name": "New 30-bar low", "family": "mean_reversion",
                      "fn": sig_mr_newlow30, "buckets": DEPTH_DN_BUCKETS, "exit_fn": None},
    "mr_newlow60":   {"name": "New 60-bar low", "family": "mean_reversion",
                      "fn": sig_mr_newlow60, "buckets": DEPTH_DN_BUCKETS, "exit_fn": None},
    "mr_ndown":      {"name": "K consecutive down bars (>=3)", "family": "mean_reversion",
                      "fn": sig_mr_ndown, "buckets": KDOWN_BUCKETS, "exit_fn": None},
    "mo_burst":      {"name": "2-bar up-burst (>=4%)", "family": "momentum",
                      "fn": sig_mo_burst, "buckets": UP_BUCKETS, "exit_fn": None},
    "mo_break_hi":   {"name": "New 30-bar high breakout", "family": "momentum",
                      "fn": sig_mo_break_hi, "buckets": BREAK_BUCKETS, "exit_fn": None},
    "mo_rsi_ob":     {"name": "RSI(14) cross above 60", "family": "momentum",
                      "fn": sig_mo_rsi_ob, "buckets": RSI_OB_BUCKETS, "exit_fn": None},
    "mo_gap_up":     {"name": "Gap-up bar (>=2%)", "family": "momentum",
                      "fn": sig_mo_gap_up, "buckets": GAP_UP_BUCKETS, "exit_fn": None},
    "ev_open_gap":   {"name": "Session-open gap reaction", "family": "event",
                      "fn": sig_ev_open_gap, "buckets": GAP_DN_BUCKETS + GAP_UP_BUCKETS, "exit_fn": None},
    "st_ad_div":     {"name": "A/D bullish divergence (10-bar)", "family": "structure",
                      "fn": sig_st_ad_div, "buckets": DIV_BUCKETS, "exit_fn": None},
    "st_ma_pull":    {"name": "Pullback to rising MA(20)", "family": "structure",
                      "fn": sig_st_ma_pull, "buckets": MAPULL_BUCKETS, "exit_fn": None},
}

FAMILIES = {}
for _k, _m in SIGNALS.items():
    FAMILIES.setdefault(_m["family"], []).append(_k)


# Pure stat helpers copied from studies.py so this module stays Django-free (unit-testable).
def _tstat_from_returns(returns):
    arr = np.asarray(returns, dtype=float)
    arr = arr[np.isfinite(arr)]
    if len(arr) < 3:
        return None
    sd = arr.std(ddof=1)
    if not (sd > 0):
        return None
    return round(float(arr.mean() / (sd / np.sqrt(len(arr)))), 2)


def _episode_starts(entry_idxs, gap=GAP):
    starts, last = set(), -10 ** 9
    for i in entry_idxs:
        if i - last >= gap:
            starts.add(i)
            last = i
    return starts


EXITS = [(f"{b}b", b, day_label(b)) for b in FIXED_BARS]
EXIT_LABEL = {k: f"Hold {k} ({d})" for k, b, d in EXITS}
EXIT_LABEL["rsi_x_dn"] = "Till RSI crosses back below SMA"


def exit_keys_for(sig):
    keys = [k for k, _, _ in EXITS]
    if SIGNALS[sig].get("exit_fn") == "rsi_x_dn":
        keys = keys + ["rsi_x_dn"]
    return keys


def _rsi_x_dn_exit(df):
    """Exit bars where RSI(14) crosses back below its SMA(14) — the native MR exit for RSI signals."""
    rsi = ta.momentum.rsi(df["Close"], window=14)
    sma = rsi.rolling(14).mean()
    dn = (rsi < sma) & (rsi.shift(1) >= sma.shift(1))
    return dn.fillna(False).values


def _empty_exit_pool(sig):
    return {k: [] for k in exit_keys_for(sig)}


def backtest_ticker(df, dtrend=None):
    """Backtest every signal on one 4h frame. dtrend: {date -> 'up'|'dn'} daily-trend map (optional)."""
    close = df["Close"].values
    n = len(close)
    dates = df.index.normalize()
    out = {}
    for sig, meta in SIGNALS.items():
        entry, mag = meta["fn"](df)
        buckets = meta["buckets"]
        idxs = sorted(_episode_starts([i for i in range(n) if entry[i]], gap=GAP))
        flat = _empty_exit_pool(sig)
        by_bucket = {b[0]: _empty_exit_pool(sig) for b in buckets}
        by_dtrend = {"up": _empty_exit_pool(sig), "dn": _empty_exit_pool(sig)}
        dn_exit = _rsi_x_dn_exit(df) if meta.get("exit_fn") == "rsi_x_dn" else None
        for i in idxs:
            ep = float(close[i])
            if ep <= 0:
                continue
            blab = bucket_of(mag[i], buckets)
            dstate = None
            if dtrend is not None:
                dstate = dtrend.get(dates[i].date())
            for k, bars, _ in EXITS:
                j = i + bars
                if j < n:
                    r = (close[j] - ep) / ep * 100
                    flat[k].append(r)
                    if blab is not None:
                        by_bucket[blab][k].append(r)
                    if dstate in ("up", "dn"):
                        by_dtrend[dstate][k].append(r)
            if dn_exit is not None:
                j = next((q for q in range(i + 1, n) if dn_exit[q]), None)
                if j is not None:
                    r = (close[j] - ep) / ep * 100
                    flat["rsi_x_dn"].append(r)
                    if blab is not None:
                        by_bucket[blab]["rsi_x_dn"].append(r)
                    if dstate in ("up", "dn"):
                        by_dtrend[dstate]["rsi_x_dn"].append(r)
        out[sig] = {"flat": flat, "by_bucket": by_bucket, "by_dtrend": by_dtrend}
    return out


def agg_rows(pool, exit_keys, min_trades=20):
    """Aggregate {exit_key: [returns]} into sorted ladder rows (n>=min_trades)."""
    rows = []
    for k in exit_keys:
        r = pool.get(k, [])
        if len(r) < min_trades:
            continue
        a = np.array(r, dtype=float)
        a = a[np.isfinite(a)]
        if len(a) < min_trades:
            continue
        rows.append({"exit": k, "name": EXIT_LABEL.get(k, k), "trades": int(len(a)),
                     "avg_pct": round(float(a.mean()), 3), "median_pct": round(float(np.median(a)), 3),
                     "win_pct": round(float((a > 0).mean() * 100), 1),
                     "t": _tstat_from_returns(list(a))})
    rows.sort(key=lambda x: -x["avg_pct"])
    return rows
