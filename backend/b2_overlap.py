#!/usr/bin/env python3
"""Is there a genuinely DIFFERENT, profitable B-2? Within the capitulation (seq w15) windows, compare the
entry sets of several H4 signals: are their trades distinct from B's gap-up (low overlap) AND profitable?
A distinct+profitable signal = a real B-2 to run ALONGSIDE B; a high-overlap one is just B again.
Run: MSYS_NO_PATHCONV=1 docker exec rotation-backend-1 python -u /app/b2_overlap.py"""
import os, warnings
warnings.filterwarnings("ignore")
os.environ.setdefault("DJANGO_SETTINGS_MODULE", "rotation.settings")
import django
django.setup()
import numpy as np
import h4_on_signals_study as S
import h4_study as H
from seq_fundamental_study import load_candles
from studies import SIGNALS as STUDY_SIGNALS

_n, seq_fn = STUDY_SIGNALS["seq_rsi20_ad_rising_rsi"]
# signal -> exit horizon (bars) from the exit-ladder sweep
SIGS = {"mo_gap_up": 3, "st_ad_div": 4, "mr_ndown": 6, "mr_newlow60": 5, "mr_rsi_os": 3, "mr_gap_dn": 3}


def _t(a):
    a = np.asarray(a, float); a = a[np.isfinite(a)]
    if len(a) < 8:
        return None
    sd = a.std(ddof=1)
    return round(float(a.mean() / (sd / np.sqrt(len(a)))), 2) if sd > 0 else None


# B capitulation w15 windows
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

from intraday_data import get_4h
entset = {k: set() for k in SIGS}       # (ticker, entry_ts) per signal
rets = {k: [] for k in SIGS}
for tk, dates in allowed.items():
    df = get_4h(tk, 5, False)
    if df is None or len(df) < 120:
        continue
    c = df["Close"].values; ts = df.index
    for sg, hz in SIGS.items():
        entry, _mag = H.SIGNALS[sg]["fn"](df)
        for i in range(len(c) - hz):
            if entry[i] and ts[i].date() in dates:
                entset[sg].add((tk, int(ts[i].value)))
                rets[sg].append((c[i + hz] - c[i]) / c[i] * 100)

B = entset["mo_gap_up"]
print(f"B (gap-up@3b) base: n{len(B)}\n", flush=True)
print(f"  {'signal':12}{'exit':>5}{'n':>6}{'avg%':>8}{'win%':>7}{'t':>7}{'overlap w/B':>13}{'distinct?':>11}", flush=True)
for sg, hz in SIGS.items():
    a = np.asarray(rets[sg], float); a = a[np.isfinite(a)]
    E = entset[sg]
    ov = (len(E & B) / len(E | B) * 100) if (E | B) else 0
    distinct = "—" if sg == "mo_gap_up" else ("YES" if ov < 25 else "no (⊂B)")
    prof = "" if _t(a) is None else ("★" if (a.mean() > 0 and (_t(a) or 0) >= 2 and (sg == "mo_gap_up" or ov < 25)) else "")
    print(f"  {sg:12}{hz:>4}b{len(a):>6}{a.mean():>+8.2f}{(a>0).mean()*100:>7.0f}{str(_t(a)):>7}{ov:>12.0f}%{distinct:>11} {prof}", flush=True)
