#!/usr/bin/env python3
"""ITERATION-2 A time-step: does A_rs (relative-strength gap-up) actually cut A's drawdown / lift return vs the
gap-up baseline in an HONEST time-stepped portfolio (net 10bps)? A_rs = gap-up AND the 4h gap beats SPY's same-bar
move by >=RS (idiosyncratic momentum -> less market beta). Also A_vol (volume-confirmed). Same engine as a_curve.py.
Run: MSYS_NO_PATHCONV=1 docker exec rotation-backend-1 python -u /app/a_rs_curve.py"""
import os, warnings
warnings.filterwarnings("ignore")
os.environ.setdefault("DJANGO_SETTINGS_MODULE", "rotation.settings")
import django
django.setup()
import numpy as np, pandas as pd
import h4_on_signals_study as S
from intraday_data import get_4h

VOL_LO, VOL_HI, DVOL, GAP, RS, VOLX = 0.50, 3.00, 5e6, 0.05, 0.05, 2.0

cl, vo = {}, {}
for tk in S._stock_universe():
    df = get_4h(tk, 5, False)
    if df is not None and len(df) >= 120:
        cl[tk] = df["Close"]; vo[tk] = df["Volume"]
close = pd.DataFrame(cl).sort_index()
vol_ = pd.DataFrame(vo).reindex_like(close)
R = close.pct_change()
ann_vol = R.rolling(20).std() * (252 ** 0.5)
dvol = (close * vol_).rolling(20).mean()
volavg = vol_.rolling(20).mean()
gap = close / close.shift(1) - 1.0
qual = (ann_vol >= VOL_LO) & (ann_vol <= VOL_HI) & (dvol >= DVOL)
sdf = get_4h("SPY", 5, False)
sret = sdf["Close"].reindex(close.index).pct_change() if sdf is not None else pd.Series(0.0, index=close.index)
qdf = get_4h("QQQ", 5, False)
qret = qdf["Close"].reindex(close.index).pct_change() if qdf is not None else pd.Series(0.0, index=close.index)
bars_per_yr = len(close) / max(0.1, (close.index[-1] - close.index[0]).days / 365.25)
Rf = R.fillna(0.0)


def entries(kind):
    e = (gap >= GAP) & qual
    if kind == "rs":
        e = e & (gap.sub(sret, axis=0) >= RS)
    elif kind == "vol":
        e = e & (vol_ >= VOLX * volavg)
    return e.fillna(False)


def run(kind, H=8, w_name=0.02, fee=1e-3, hedge=0.0):
    E = entries(kind)
    held = E.shift(1).rolling(H, min_periods=1).max().fillna(0) > 0
    W = held.astype(float) * w_name
    row = W.sum(axis=1)
    scale = np.where(row > 1.0, 1.0 / row.replace(0, np.nan), 1.0)
    W = W.mul(pd.Series(scale, index=W.index).fillna(1.0), axis=0)
    port = (W.shift(1) * Rf).sum(axis=1)
    turn = (W - W.shift(1)).abs().sum(axis=1)
    net = port - turn * fee
    if hedge:
        qma = qret.add(1).cumprod()
        regime = (qma < qma.rolling(50).mean()) | (qret.shift(1) < -0.01)
        net = net - hedge * (qret * regime.shift(1).fillna(False).astype(float))
    eq = (1 + net).cumprod()
    total = (eq.iloc[-1] - 1) * 100
    dd = ((eq / eq.cummax()) - 1).min() * 100
    sd = net.std()
    sh = net.mean() / sd * np.sqrt(bars_per_yr) if sd > 0 else 0.0
    avg_names = float(W.gt(0).sum(axis=1).mean())
    return total, dd, sh, avg_names


print(f"names {close.shape[1]}, 4h bars {close.shape[0]} (~{bars_per_yr:.0f}/yr)\n", flush=True)
print(f"  {'variant':32}{'total%(net10)':>15}{'maxDD%':>10}{'Sharpe':>9}{'avg#held':>10}", flush=True)
variants = [
    ("baseline gap H8 K50",   dict(kind="gap", H=8, w_name=0.02)),
    ("A_rs  H8 K50",          dict(kind="rs",  H=8, w_name=0.02)),
    ("A_vol H8 K50",          dict(kind="vol", H=8, w_name=0.02)),
    ("baseline gap H8 K33",   dict(kind="gap", H=8, w_name=0.03)),
    ("A_rs  H8 K33",          dict(kind="rs",  H=8, w_name=0.03)),
    ("A_rs  H8 K50 +QQQh50",  dict(kind="rs",  H=8, w_name=0.02, hedge=0.5)),
    ("A_vol H8 K33",          dict(kind="vol", H=8, w_name=0.03)),
]
for lab, kw in variants:
    t, dd, sh, an = run(**kw)
    print(f"  {lab:32}{t:>15.0f}{dd:>10.1f}{sh:>9.2f}{an:>10.1f}", flush=True)
