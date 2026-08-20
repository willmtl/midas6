#!/usr/bin/env python3
"""What happens if we OVERLAY flagship-B on flagship-C? C stays fully invested (monthly small-cap value); B's
capitulation->gap-up trades are taken as a MARGIN overlay at fraction f of book notional, so B's per-trade edge
ADDS to C instead of idling in cash. Compare C-alone vs C+B over the common window (2021-09+, where 4h/B exists),
net ~10bps on B. Run: MSYS_NO_PATHCONV=1 docker exec rotation-backend-1 python -u /app/c_b_overlay.py"""
import os, json, warnings, datetime as dt
warnings.filterwarnings("ignore")
os.environ.setdefault("DJANGO_SETTINGS_MODULE", "rotation.settings")
import django
django.setup()
import numpy as np
import h4_on_signals_study as S
import h4_study as H
from seq_fundamental_study import load_candles
from studies import SIGNALS as STUDY_SIGNALS
from intraday_data import get_4h

# ---- C monthly returns (flagship adaptive) ----
Cj = json.load(open("/app/.data/studies/flagship_history.json"))
c_month = {}
for m in Cj.get("months", []):
    d = str(m.get("date"))[:7]        # YYYY-MM
    br = m.get("basket_ret")
    if d and br is not None:
        c_month[d] = float(br)

# ---- B trades: capitulation w15 -> gap-up @3b, bucket returns (fraction) by month ----
_n, seq_fn = STUDY_SIGNALS["seq_rsi20_ad_rising_rsi"]
daily = load_candles(S._stock_universe())
allowed = {}
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
        allowed[tk] = s
b_month = {}
FEE = 0.001
for tk, dates in allowed.items():
    df = get_4h(tk, 5, False)
    if df is None or len(df) < 120:
        continue
    entry, _m = H.SIGNALS["mo_gap_up"]["fn"](df)
    c = df["Close"].values; ts = df.index
    for i in range(len(c) - 3):
        if entry[i] and ts[i].date() in dates:
            r = (c[i + 3] - c[i]) / c[i] - FEE
            b_month.setdefault(str(ts[i].date())[:7], []).append(r)

# common months (both C and B exist) — B era 2021-09+
months = sorted(m for m in c_month if m >= "2021-09" and m in {**{k: 1 for k in c_month}})
def stats(series):
    r = np.array(series, float)
    tot = (np.prod(1 + r) - 1) * 100
    n = len(r); cagr = ((1 + tot / 100) ** (12.0 / n) - 1) * 100 if n else 0
    eq = np.cumprod(1 + r); dd = ((eq / np.maximum.accumulate(eq)) - 1).min() * 100
    sh = r.mean() / r.std(ddof=1) * np.sqrt(12) if r.std(ddof=1) > 0 else 0
    return tot, cagr, dd, sh

c_series = [c_month[m] for m in months]
tot, cagr, dd, sh = stats(c_series)
nb = sum(len(b_month.get(m, [])) for m in months)
print(f"common window {months[0]}..{months[-1]} ({len(months)} mo); B trades in window: {nb} (~{nb/len(months)*12:.0f}/yr)\n", flush=True)
print(f"  {'book':28}{'total%':>10}{'CAGR%':>8}{'maxDD%':>9}{'Sharpe':>8}", flush=True)
print(f"  {'C alone':28}{tot:>10.0f}{cagr:>8.1f}{dd:>9.1f}{sh:>8.2f}", flush=True)
for f in (0.25, 0.5, 1.0):
    comb = [c_month[m] + f * sum(b_month.get(m, [])) for m in months]
    tot, cagr, dd, sh = stats(comb)
    print(f"  {'C + B overlay f=' + str(f):28}{tot:>10.0f}{cagr:>8.1f}{dd:>9.1f}{sh:>8.2f}", flush=True)
