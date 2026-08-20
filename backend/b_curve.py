#!/usr/bin/env python3
"""Combined B + B-2 + B-3 capitulation book — time-stepped equity curve (proper concurrency, net of ~10bps).
Within the seq-w15 capitulation windows, take EVERY one of the three orthogonal entries — B gap-up (hold 3),
B-2 st_ad_div (hold 4), B-3 mr_ndown (hold 6) — one position per name, equal-weight capped, gross 1x. Reports
total / maxDD / Sharpe / $10k-> vs base-B-alone. Run: MSYS_NO_PATHCONV=1 docker exec rotation-backend-1 python -u /app/b_curve.py"""
import os, warnings
warnings.filterwarnings("ignore")
os.environ.setdefault("DJANGO_SETTINGS_MODULE", "rotation.settings")
import django
django.setup()
import numpy as np, pandas as pd
import h4_on_signals_study as S
import h4_study as H
from seq_fundamental_study import load_candles
from studies import SIGNALS as STUDY_SIGNALS
from intraday_data import get_4h

_n, seq_fn = STUDY_SIGNALS["seq_rsi20_ad_rising_rsi"]
ENTRIES = [("mo_gap_up", 3), ("st_ad_div", 4), ("mr_ndown", 6)]   # (signal, hold bars)

daily = load_candles(S._stock_universe())
winset = {}
for tk, df in daily.items():
    if len(df) < 60:
        continue
    sig = seq_fn(df).fillna(False); idx = df.index
    dv = (df["Close"] * df["Volume"]).rolling(20).mean()
    s = set()
    for i in np.flatnonzero(sig.values):
        if dv.iloc[i] < 5e6:
            continue
        for j in range(i, min(i + 15, len(idx))):
            s.add(idx[j].date())
    if s:
        winset[tk] = s

held_cols, retB_cols, heldB_cols = {}, {}, {}
for tk, wd in winset.items():
    df = get_4h(tk, 5, False)
    if df is None or len(df) < 120:
        continue
    ts = df.index
    inwin = pd.Series([d.date() in wd for d in ts], index=ts)
    ret = df["Close"].pct_change()
    held_any = pd.Series(False, index=ts)
    held_b = pd.Series(False, index=ts)
    for sg, hz in ENTRIES:
        entry, _m = H.SIGNALS[sg]["fn"](df)
        e = pd.Series(entry, index=ts).fillna(False) & inwin
        h = e.shift(1).rolling(hz, min_periods=1).max().fillna(0) > 0
        held_any = held_any | h
        if sg == "mo_gap_up":
            held_b = h
    held_cols[tk] = held_any
    heldB_cols[tk] = held_b
    retB_cols[tk] = ret

held = pd.DataFrame(held_cols).sort_index().fillna(False)
heldB = pd.DataFrame(heldB_cols).reindex_like(held).fillna(False)
R = pd.DataFrame(retB_cols).reindex_like(held).fillna(0.0)
bars_per_yr = len(held) / max(0.1, (held.index[-1] - held.index[0]).days / 365.25)


def curve(H_mask, w_name, fee=0.001):
    W = H_mask.astype(float) * w_name
    row = W.sum(axis=1)
    scale = np.where(row > 1.0, 1.0 / row.replace(0, np.nan), 1.0)
    W = W.mul(pd.Series(scale, index=W.index).fillna(1.0), axis=0)
    port = (W.shift(1) * R).sum(axis=1)
    turn = (W - W.shift(1)).abs().sum(axis=1)
    net = port - turn * fee
    eq = (1 + net).cumprod()
    tot = (eq.iloc[-1] - 1) * 100
    dd = ((eq / eq.cummax()) - 1).min() * 100
    sd = net.std()
    sh = net.mean() / sd * np.sqrt(bars_per_yr) if sd > 0 else 0.0
    return tot, dd, sh, eq.iloc[-1]


print(f"names {held.shape[1]}, 4h bars {held.shape[0]} (~{bars_per_yr:.0f}/yr)\n", flush=True)
print(f"  {'book':34}{'total%':>10}{'maxDD%':>9}{'Sharpe':>8}{'$10k ->':>13}", flush=True)
for lab, mask, w in [
    ("B alone (gap-up)  K10 (10%)", heldB, 0.10),
    ("B alone (gap-up)  K20 (5%)",  heldB, 0.05),
    ("B+B2+B3 combined  K10 (10%)", held, 0.10),
    ("B+B2+B3 combined  K20 (5%)",  held, 0.05),
    ("B+B2+B3 combined  K33 (3%)",  held, 0.03),
]:
    tot, dd, sh, mult = curve(mask, w)
    print(f"  {lab:34}{tot:>10.0f}{dd:>9.1f}{sh:>8.2f}{10000*mult:>13,.0f}", flush=True)
