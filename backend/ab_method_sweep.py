#!/usr/bin/env python3
"""C-style sweep for the SHORT-HORIZON selectors: vary A_plus's trigger / vol-band / liquidity / window and
B_plus's window / liquidity, and rank each variant by the pooled H4 3-bar bounce (n / avg% / win% / t) on the
relevant signals. Daily candles loaded once; 4h cached (--no-fetch style, cached only). This is the A/B analog
of C's LAB parameter sweeps — the fundamental/value methods are excluded (proven to hurt short-horizon).
Run: MSYS_NO_PATHCONV=1 docker exec rotation-backend-1 python -u /app/ab_method_sweep.py"""
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

WIN_A = 5
_seq_name, _seq_fn = STUDY_SIGNALS["seq_rsi20_ad_rising_rsi"]


def _t(a):
    a = np.asarray(a, float); a = a[np.isfinite(a)]
    if len(a) < 5:
        return None
    sd = a.std(ddof=1)
    return round(float(a.mean() / (sd / np.sqrt(len(a)))), 2) if sd > 0 else None


# ---- variant window builders: return {ticker: set(date)} from precomputed daily features ----
def build_A(daily, burst_thr, vol_lo, vol_hi, dvol_min, win):
    allowed = {}
    for tk, d in daily.items():
        c = d["close"]; idx = d["idx"]
        ok = (d["burst"] >= burst_thr) & (d["vol"].between(vol_lo, vol_hi)) & (d["dvol"] >= dvol_min)
        s = set()
        for i in np.flatnonzero(ok.values):
            for j in range(i, min(i + win, len(idx))):
                s.add(idx[j].date())
        if s:
            allowed[tk] = s
    return allowed


def build_A_gap(daily, gap_thr, vol_lo, vol_hi, dvol_min, win):
    allowed = {}
    for tk, d in daily.items():
        idx = d["idx"]
        ok = (d["gap1"] >= gap_thr) & (d["vol"].between(vol_lo, vol_hi)) & (d["dvol"] >= dvol_min)
        s = set()
        for i in np.flatnonzero(ok.values):
            for j in range(i, min(i + win, len(idx))):
                s.add(idx[j].date())
        if s:
            allowed[tk] = s
    return allowed


def build_A_newhi(daily, vol_lo, vol_hi, dvol_min, win):
    allowed = {}
    for tk, d in daily.items():
        idx = d["idx"]
        ok = d["new20hi"] & (d["vol"].between(vol_lo, vol_hi)) & (d["dvol"] >= dvol_min)
        s = set()
        for i in np.flatnonzero(ok.values):
            for j in range(i, min(i + win, len(idx))):
                s.add(idx[j].date())
        if s:
            allowed[tk] = s
    return allowed


def build_B(daily, dvol_min, win):
    allowed = {}
    for tk, d in daily.items():
        idx = d["idx"]; sig = d["seq"]
        s = set()
        for i in np.flatnonzero(sig.values):
            if d["dvol"].iloc[i] < dvol_min:
                continue
            for j in range(i, min(i + win, len(idx))):
                s.add(idx[j].date())
        if s:
            allowed[tk] = s
    return allowed


def pooled(allowed, frames, sigset):
    per = {s: [] for s in sigset}
    names = 0
    for tk, dates in allowed.items():
        df = frames.get(tk)
        if df is None:
            continue
        names += 1
        res = S.backtest_ticker_masked(df, dates)
        for s in sigset:
            per[s].extend(res[s]["flat"].get("3b", []))
    return per, names


def main():
    uni = S._stock_universe()
    raw = load_candles(uni)
    print(f"universe {len(uni)} → candles {len(raw)}", flush=True)
    daily = {}
    for tk, df in raw.items():
        if len(df) < 60:
            continue
        c = df["Close"]
        daily[tk] = {
            "idx": df.index, "close": c,
            "vol": c.pct_change().rolling(20).std() * (252 ** 0.5),
            "dvol": (c * df["Volume"]).rolling(20).mean(),
            "burst": c / c.shift(2) - 1.0,
            "gap1": c / c.shift(1) - 1.0,
            "new20hi": c >= c.rolling(20).max(),
            "seq": _seq_fn(df).fillna(False),
        }

    A_SIGS = ["mo_burst", "mo_gap_up", "mo_break_hi", "mo_rsi_ob", "mr_rsi_os"]
    B_SIGS = ["mo_gap_up", "mr_ndown", "st_ad_div", "mr_newlow60", "mr_rsi_os"]

    variants = [
        ("A: burst8 vol50-300 $5M w5",  lambda: build_A(daily, .08, .50, 3.0, 5e6, 5),  A_SIGS),
        ("A: burst15 vol50-300 $5M w5", lambda: build_A(daily, .15, .50, 3.0, 5e6, 5),  A_SIGS),
        ("A: burst8 vol ALL $5M w5",    lambda: build_A(daily, .08, .0, 99., 5e6, 5),   A_SIGS),
        ("A: burst8 vol80-300 $5M w5",  lambda: build_A(daily, .08, .80, 3.0, 5e6, 5),  A_SIGS),
        ("A: burst8 vol50-300 $20M w5", lambda: build_A(daily, .08, .50, 3.0, 2e7, 5),  A_SIGS),
        ("A: burst8 vol50-300 $5M w3",  lambda: build_A(daily, .08, .50, 3.0, 5e6, 3),  A_SIGS),
        ("A: gap5 vol50-300 $5M w5",    lambda: build_A_gap(daily, .05, .50, 3.0, 5e6, 5), A_SIGS),
        ("A: new20hi vol50-300 $5M w5", lambda: build_A_newhi(daily, .50, 3.0, 5e6, 5),  A_SIGS),
        ("B: seq $5M w10",              lambda: build_B(daily, 5e6, 10),  B_SIGS),
        ("B: seq $5M w5",               lambda: build_B(daily, 5e6, 5),   B_SIGS),
        ("B: seq $5M w15",              lambda: build_B(daily, 5e6, 15),  B_SIGS),
        ("B: seq $20M w10",             lambda: build_B(daily, 2e7, 10),  B_SIGS),
    ]

    # build all windows, collect union of names, fetch 4h (cached) once
    built = [(lab, fn(), sigs) for lab, fn, sigs in variants]
    allnames = sorted({tk for _, aw, _ in built for tk in aw})
    frames = {}
    for tk in allnames:
        df = get_4h(tk, 5, False)
        if df is not None and len(df) >= 120:
            frames[tk] = df
    print(f"4h cached for {len(frames)}/{len(allnames)} names\n", flush=True)

    print(f"{'variant':32}{'windows':>9}{'names4h':>9}   best-signal rows (avg%/win%/t) @3b", flush=True)
    for lab, aw, sigs in built:
        per, n4h = pooled(aw, frames, sigs)
        cells = []
        for s in sigs:
            a = np.asarray(per[s], float); a = a[np.isfinite(a)]
            if len(a) >= 20:
                cells.append((s, len(a), a.mean(), (a > 0).mean() * 100, _t(a)))
        cells.sort(key=lambda x: -(x[4] or -9))
        nwin = sum(len(v) for v in aw.values())
        top = "  ".join(f"{s}:{n}/{av:+.2f}/{w:.0f}/{t}" for s, n, av, w, t in cells[:3])
        print(f"{lab:32}{nwin:>9}{n4h:>9}   {top}", flush=True)


if __name__ == "__main__":
    main()
