#!/usr/bin/env python3
"""FLAGSHIP-A time-stepped equity curve → total / max-DD / Sharpe (net of 10bps turnover cost), plus beat
variants: wider K (smaller per-name weight), hold fine-tune, trigger union (gap-up ∪ new-20-high), and a QQQ
stress-hedge. Vectorized: aligned 4h close/volume panels, position = entered in last H bars, equal-weight
capped per name and to gross 1x.  Run: MSYS_NO_PATHCONV=1 docker exec rotation-backend-1 python -u /app/a_curve.py"""
import os, warnings
warnings.filterwarnings("ignore")
os.environ.setdefault("DJANGO_SETTINGS_MODULE", "rotation.settings")
import django
django.setup()
import numpy as np, pandas as pd
import h4_on_signals_study as S
from intraday_data import get_4h

VOL_LO, VOL_HI, DVOL, GAP = 0.50, 3.00, 5e6, 0.05

# ---- build aligned panels ----
cl, vo = {}, {}
for tk in S._stock_universe():
    df = get_4h(tk, 5, False)
    if df is not None and len(df) >= 120:
        cl[tk] = df["Close"]; vo[tk] = df["Volume"]
qdf = get_4h("QQQ", 5, False)
close = pd.DataFrame(cl).sort_index()
vol_ = pd.DataFrame(vo).reindex_like(close)
R = close.pct_change()
ann_vol = R.rolling(20).std() * (252 ** 0.5)
dvol = (close * vol_).rolling(20).mean()
gap = close / close.shift(1) - 1.0
new20 = close >= close.rolling(20).max()
qual = (ann_vol >= VOL_LO) & (ann_vol <= VOL_HI) & (dvol >= DVOL)
bars_per_yr = len(close) / max(0.1, (close.index[-1] - close.index[0]).days / 365.25)
qret = qdf["Close"].reindex(close.index).pct_change() if qdf is not None else pd.Series(0.0, index=close.index)

Rf = R.fillna(0.0)


def entries(kind):
    e = (gap >= GAP) & qual
    if kind == "union":
        e = e | (new20 & qual)
    return e.fillna(False)


def run(kind="gap", H=8, w_name=0.02, fee=1e-3, hedge=0.0):
    E = entries(kind)
    held = E.shift(1).rolling(H, min_periods=1).max().fillna(0) > 0     # in a position for H bars after entry
    W = held.astype(float) * w_name
    row = W.sum(axis=1)                                                 # scale down if gross would exceed 1x
    scale = np.where(row > 1.0, 1.0 / row.replace(0, np.nan), 1.0)
    W = W.mul(pd.Series(scale, index=W.index).fillna(1.0), axis=0)
    port = (W.shift(1) * Rf).sum(axis=1)
    turn = (W - W.shift(1)).abs().sum(axis=1)
    net = port - turn * fee
    if hedge:
        # PIT: decide the short from PRIOR bars (no same-bar hindsight). Stress regime = QQQ below its 50-bar MA
        # OR QQQ fell >1% on the PREVIOUS bar; short QQQ this bar (P&L = -hedge * this-bar QQQ return).
        qma = qret.add(1).cumprod()
        regime = (qma < qma.rolling(50).mean()) | (qret.shift(1) < -0.01)
        net = net - hedge * (qret * regime.shift(1).fillna(False).astype(float))
    eq = (1 + net).cumprod()
    total = (eq.iloc[-1] - 1) * 100
    dd = ((eq / eq.cummax()) - 1).min() * 100
    sd = net.std()
    sh = net.mean() / sd * np.sqrt(bars_per_yr) if sd > 0 else 0.0
    return total, dd, sh


print(f"names {close.shape[1]}, 4h bars {close.shape[0]} (~{bars_per_yr:.0f}/yr)\n", flush=True)
print(f"  {'variant':34}{'total%(net10)':>15}{'maxDD%':>10}{'Sharpe':>9}", flush=True)
variants = [
    ("★ gap-up H8 K50 (2%)", dict(kind="gap", H=8, w_name=0.02)),
    ("gap-up H8 K80 (1.25%)", dict(kind="gap", H=8, w_name=0.0125)),
    ("gap-up H8 K33 (3%)",    dict(kind="gap", H=8, w_name=0.03)),
    ("gap-up H7 K50",         dict(kind="gap", H=7, w_name=0.02)),
    ("gap-up H10 K50",        dict(kind="gap", H=10, w_name=0.02)),
    ("gap∪new20hi H8 K50",    dict(kind="union", H=8, w_name=0.02)),
    ("★ + QQQ hedge 50%",     dict(kind="gap", H=8, w_name=0.02, hedge=0.5)),
    ("★ + QQQ hedge 100%",    dict(kind="gap", H=8, w_name=0.02, hedge=1.0)),
]
for lab, kw in variants:
    t, dd, sh = run(**kw)
    print(f"  {lab:34}{t:>15.0f}{dd:>10.1f}{sh:>9.2f}", flush=True)
