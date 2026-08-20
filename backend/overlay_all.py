#!/usr/bin/env python3
"""Full overlay test, DAILY-marked (honest DD): C (reconstructed daily) + A + B.
 - C: flagship daily equity (replicate flagship_daily_dd) -> daily returns.
 - A: gap-up momentum, capacity-capped SLEEVE (a_curve K50) overlaid at fraction fA (a real 1x book -> adds
      fA of book; A is a firehose so it needs its own capacity, not per-trade margin).
 - B: capitulation->gap-up, INTERMITTENT MARGIN overlay at fB (rare -> take each trade at fB notional on C).
Compares C / C+A / C+B / C+A+B over the common daily window. Run:
 MSYS_NO_PATHCONV=1 docker exec rotation-backend-1 python -u /app/overlay_all.py"""
import os, json, warnings
warnings.filterwarnings("ignore")
os.environ.setdefault("DJANGO_SETTINGS_MODULE", "rotation.settings")
import django
django.setup()
import numpy as np, pandas as pd
from django.db import connection
with connection.cursor() as cur:
    cur.execute("SET max_parallel_workers_per_gather = 0")
import h4_on_signals_study as S
import h4_study as H
from seq_fundamental_study import load_candles
from studies import SIGNALS as STUDY_SIGNALS

FEE = 0.001

# ---------- C daily returns (replicate flagship_daily_dd) ----------
D = json.load(open("/app/.data/studies/flagship_history.json"))
months = [m for m in D["months"] if m.get("ndate") and m.get("picks")]
ctk = sorted({p["ticker"] for m in months for p in m["picks"] if p.get("ticker")})
cand = load_candles(ctk + ["SPY"])
spy = cand["SPY"]["Close"]
val = 1.0; dc = []
for m in months:
    d0 = pd.Timestamp(m["date"]); d1 = pd.Timestamp(m["ndate"])
    picks = [(p["ticker"], float(p["weight"])) for p in m["picks"] if p.get("ticker") and p.get("weight")]
    days = spy.loc[d0:d1].index
    if len(days) < 2 or not picks:
        continue
    w = pd.Series(dict(picks), dtype=float)
    px = pd.DataFrame({t: cand[t]["Close"].reindex(days, method="ffill") for t, _ in picks if t in cand})
    w = w.reindex(px.columns).dropna(); px = px[w.index]
    ent = px.iloc[0]; ok = ent[ent > 0].index; px = px[ok]; w = w[ok]; ent = ent[ok]
    if not len(w):
        continue
    book = (px.div(ent, axis=1) - 1.0).mul(w, axis=1).sum(axis=1) / w.sum()
    vals = val * (1.0 + book)
    for ts, v in vals.iloc[1:].items():
        dc.append((ts, float(v)))
    val = float(vals.iloc[-1])
cval = pd.Series(dict(dc)).sort_index()
rC = cval.pct_change().dropna()
print(f"C daily reconstructed: {len(rC)} days {rC.index[0].date()}..{rC.index[-1].date()}", flush=True)

# ---------- universe 4h (A + B) ----------
uni = S._stock_universe()
daily = load_candles(uni)
frames = {}
from intraday_data import get_4h
for tk in uni:
    f = get_4h(tk, 5, False)
    if f is not None and len(f) >= 120:
        frames[tk] = f

# ---------- A sleeve daily returns (gap-up H8, K50 2% cap, gross 1x) ----------
cl = pd.DataFrame({t: f["Close"] for t, f in frames.items()}).sort_index()
vo = pd.DataFrame({t: f["Volume"] for t, f in frames.items()}).reindex_like(cl)
R4 = cl.pct_change().fillna(0.0)
annv = cl.pct_change().rolling(20).std() * (252 ** 0.5)
dvol = (cl * vo).rolling(20).mean()
gap = cl / cl.shift(1) - 1.0
qualA = (annv >= 0.5) & (annv <= 3.0) & (dvol >= 5e6) & (gap >= 0.05)
heldA = qualA.shift(1).rolling(8, min_periods=1).max().fillna(0) > 0
WA = heldA.astype(float) * 0.02
sc = np.where(WA.sum(1) > 1, 1.0 / WA.sum(1).replace(0, np.nan), 1.0)
WA = WA.mul(pd.Series(sc, index=WA.index).fillna(1.0), axis=0)
netA4 = (WA.shift(1) * R4).sum(1) - (WA - WA.shift(1)).abs().sum(1) * FEE
rA = (1 + netA4).groupby(netA4.index.normalize()).prod() - 1     # -> daily

# ---------- B intermittent-margin daily P&L (capitulation w15 -> gap-up @3b) ----------
_n, seq_fn = STUDY_SIGNALS["seq_rsi20_ad_rising_rsi"]
win = {}
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
        win[tk] = s
bpnl = {}
for tk, dates in win.items():
    f = frames.get(tk)
    if f is None:
        continue
    entry, _m = H.SIGNALS["mo_gap_up"]["fn"](f); c = f["Close"].values; ts = f.index
    for i in range(len(c) - 3):
        if entry[i] and ts[i].date() in dates:
            ex = ts[i + 3].normalize()
            bpnl[ex] = bpnl.get(ex, 0.0) + ((c[i + 3] - c[i]) / c[i] - FEE)
rB = pd.Series(bpnl).sort_index()

# ---------- align on C daily calendar & combine (normalize tz so 4h-derived dates match C's naive daily) ----------
def _naive(s):
    idx = pd.DatetimeIndex(s.index)
    if idx.tz is not None:
        idx = idx.tz_convert("UTC").tz_localize(None)
    s = s.copy(); s.index = idx.normalize()
    return s.groupby(level=0).sum()

rC = _naive(rC); rA = _naive(rA); rB = _naive(rB)
idxC = rC.index
rA = rA.reindex(idxC).fillna(0.0)
rB = rB.reindex(idxC).fillna(0.0)
print(f"  overlap check: A nonzero days {int((rA!=0).sum())}, B nonzero days {int((rB!=0).sum())}", flush=True)


def stat(r):
    r = r.dropna()
    tot = (np.prod(1 + r) - 1) * 100
    yrs = (r.index[-1] - r.index[0]).days / 365.25
    cagr = ((1 + tot / 100) ** (1 / yrs) - 1) * 100 if yrs > 0 and tot > -100 else float("nan")
    eq = (1 + r).cumprod(); dd = ((eq / eq.cummax()) - 1).min() * 100
    sh = r.mean() / r.std() * np.sqrt(252) if r.std() > 0 else 0
    return tot, cagr, dd, sh


fA = fB = 0.5
print(f"\n  overlay f=0.5  (A=capacity sleeve, B=intermittent margin)  DAILY-marked", flush=True)
print(f"  {'book':16}{'total%':>12}{'CAGR%':>8}{'DAILY DD%':>11}{'Sharpe':>8}", flush=True)
for lab, r in [("C alone", rC), ("C + A", rC + fA * rA), ("C + B", rC + fB * rB),
               ("C + A + B", rC + fA * rA + fB * rB)]:
    t, c, dd, sh = stat(r)
    print(f"  {lab:16}{t:>12.0f}{c:>8.1f}{dd:>11.1f}{sh:>8.2f}", flush=True)
