#!/usr/bin/env python3
"""Sweep the full studies.py indicator library + dip->confirmation combos as H4 0-3 day entries
inside daily-C candidate windows. See docs/superpowers/specs/2026-08-17-h4-c-indicator-sweep-design.md.
Reuses h4_on_signals_study + h4_study + studies read-only. Pure funcs import without Django."""
import math
import numpy as np
import pandas as pd
import ta
import h4_study as H

K_SEQ = 5  # precondition must have fired within the prior K bars


def _rsi(df):
    return ta.momentum.rsi(df["Close"], window=14)


def _preconditions(df):
    c = df["Close"]
    gap = (df["Open"] / c.shift(1) - 1) * 100
    rsi = _rsi(df)
    prior_min = c.shift(1).rolling(30).min()
    down = c < c.shift(1)
    ndown3 = down & down.shift(1) & down.shift(2)
    return {
        "gap_dn": (gap <= -2).fillna(False),
        "rsi_lt30": (rsi < 30).fillna(False),
        "newlow30": (c < prior_min).fillna(False),
        "ndown3": ndown3.fillna(False),
    }


def _triggers(df):
    c = df["Close"]
    rsi = _rsi(df)
    sma = rsi.rolling(14).mean()
    rsi_x = (rsi > sma) & (rsi.shift(1) <= sma.shift(1))
    adl = ta.volume.acc_dist_index(df["High"], df["Low"], df["Close"], df["Volume"])
    ad_sma = adl.rolling(20).mean()
    ad_rising = (ad_sma > ad_sma.shift(1)) & (ad_sma.shift(1) <= ad_sma.shift(2))
    break_hi = c > df["High"].shift(1)
    return {
        "rsi_x": rsi_x.fillna(False),
        "ad_up": ad_rising.fillna(False),
        "break_hi": break_hi.fillna(False),
    }


def _seq(P, T, k=K_SEQ):
    """T fires now AND P fired within the prior k bars (P strictly before T)."""
    Pr = P.rolling(k).max().shift(1).fillna(0).astype(bool)   # P in [i-k, i-1]
    return (T & Pr).fillna(False).values


def combo_entries(df):
    """All precondition x trigger sequences, keyed 'P__T'. gap_dn__rsi_x is the headline."""
    P = _preconditions(df)
    T = _triggers(df)
    out = {}
    for pk, pv in P.items():
        for tk, tv in T.items():
            out[f"{pk}__{tk}"] = _seq(pv, tv)
    return out


def ladder(entry_arr, close, dates, allowed_dates):
    """Masked, episode-deduped fixed-bar exit ladder for one boolean entry series on one frame.
    Returns {exit_key: [returns%]}. allowed_dates=set of datetime.date (C window); None=all."""
    n = len(close)
    cand = [i for i in range(n) if entry_arr[i] and
            (allowed_dates is None or dates[i].date() in allowed_dates)]
    idxs = sorted(H._episode_starts(cand, gap=H.GAP))
    bars_list = [e[1] for e in H.EXITS]
    out = {f"{b}b": [] for b in bars_list}
    keymap = {b: f"{b}b" for b in bars_list}
    for i in idxs:
        ep = float(close[i])
        if ep <= 0:
            continue
        for _, bars, _ in H.EXITS:
            j = i + bars
            if j < n:
                out[keymap[bars]].append((close[j] - ep) / ep * 100)
    return out
