#!/usr/bin/env python3
"""D — COST-AWARE OVERNIGHT harvest. The overnight premium is real (+0.067%/night all, +0.14% hi-vol; screen)
but harvesting = buy-at-close/sell-at-open EVERY night = 100% turnover/night. Test if ANY survives realistic
round-trip cost. Equal-weight overnight book on liquid ($5M) names; net = overnight_ret - 2*half_spread; sweep
half-spread 0/1/2/5/10 bps. Also high-vol subset (bigger gross, but wider real spreads). Benchmark = full-day
buy-hold of the same universe. Run: MSYS_NO_PATHCONV=1 docker exec rotation-backend-1 python -u /app/d_overnight_book.py"""
import os, warnings
warnings.filterwarnings("ignore")
os.environ.setdefault("DJANGO_SETTINGS_MODULE", "rotation.settings")
import django
django.setup()
import numpy as np, pandas as pd
from django.db import connection
with connection.cursor() as cur:
    cur.execute("SET max_parallel_workers_per_gather = 0")
from seq_fundamental_study import load_candles
import h4_on_signals_study as S

DVOL_FLOOR = 5e6
uni = S._stock_universe()
print("loading candles...", flush=True)
daily = load_candles(uni)
close = pd.DataFrame({t: d["Close"] for t, d in daily.items()}).sort_index()
opn = pd.DataFrame({t: d["Open"] for t, d in daily.items()}).reindex_like(close)
vol = pd.DataFrame({t: d["Volume"] for t, d in daily.items()}).reindex_like(close)
idx = close.index
dvol = (close * vol).rolling(20).mean()
annv = close.pct_change().rolling(20).std() * (252 ** 0.5)
liq = dvol >= DVOL_FLOOR
overnight = (opn - close.shift(1)) / close.shift(1)          # close_{t-1} -> open_t
intraday = (close - opn) / opn                               # open_t -> close_t
fullday = close.pct_change()
overnight = overnight.clip(-0.5, 0.5); intraday = intraday.clip(-0.5, 0.5); fullday = fullday.clip(-0.5, 0.5)
bars_yr = len(idx) / max(0.1, (idx[-1] - idx[0]).days / 365.25)


def book(sig_mask):
    on = overnight.where(sig_mask); return on.mean(axis=1)     # equal-weight across qualifying names each day


def stats(r, cost_bps):
    r = (r - 2 * cost_bps / 1e4).dropna()                      # 2 * half-spread per night (in + out)
    if not len(r):
        return None
    eq = (1 + r).cumprod()
    tot = (eq.iloc[-1] - 1) * 100
    yrs = (r.index[-1] - r.index[0]).days / 365.25
    cagr = (eq.iloc[-1] ** (1 / yrs) - 1) * 100 if yrs > 0 else float("nan")
    dd = ((eq / eq.cummax()) - 1).min() * 100
    sh = r.mean() / r.std() * np.sqrt(bars_yr) if r.std() > 0 else 0
    return tot, cagr, dd, sh


on_all = book(liq)
on_hiv = book(liq & (annv >= 0.50))
fd_all = fullday.where(liq).mean(axis=1)
print(f"window {idx[0].date()}..{idx[-1].date()}  names~{close.shape[1]}", flush=True)
print(f"  gross/night: overnight ALL {on_all.mean()*1e4:+.1f}bp  hi-vol {on_hiv.mean()*1e4:+.1f}bp  "
      f"(intraday ALL {intraday.where(liq).mean(axis=1).mean()*1e4:+.1f}bp)\n", flush=True)
print(f"  {'book':22}{'half-spread':>12}{'CAGR%':>8}{'total%':>9}{'maxDD%':>9}{'Sharpe':>8}", flush=True)
for lab, series in (("overnight ALL", on_all), ("overnight hi-vol", on_hiv)):
    for hs in (0, 1, 2, 5, 10):
        s = stats(series, hs)
        if s:
            print(f"  {lab:22}{str(hs)+'bp':>12}{s[1]:>8.1f}{s[0]:>9.0f}{s[2]:>9.1f}{s[3]:>8.2f}", flush=True)
    print("", flush=True)
s = stats(fd_all, 0)
print(f"  {'full-day buy-hold':22}{'0bp':>12}{s[1]:>8.1f}{s[0]:>9.0f}{s[2]:>9.1f}{s[3]:>8.2f}  (reference, no daily turnover)", flush=True)
