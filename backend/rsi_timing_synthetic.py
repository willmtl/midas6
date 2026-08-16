#!/usr/bin/env python3
"""Is RSI(10)>50 a LATE entry on the RS bar? Compare entry timings on all 93 synthetic etf/spy candles.

For each synthetic relative-strength candle, fire events and measure the RELATIVE forward return
(etf minus SPY) at short horizons, plus the relative RUN-UP that already happened BEFORE the event:
  - RSI(10) crosses up through 30  (early — near the bottom of the relative dip)
  - RSI(10) crosses up through 35
  - RSI(10) crosses up through 50  (the sweep's signal — mid-momentum)
  - SMA 20/50 golden cross         (lagging confirmation, for reference)

If the oversold crosses show POSITIVE short-term forward relative return while >50 / golden are already
negative, that confirms >50 is late and the edge is the early turn. Diagnostic (forward windows), no fees.
Run: MSYS_NO_PATHCONV=1 docker exec rotation-backend-1 python -u /app/rsi_timing_synthetic.py
"""
import os, warnings
warnings.filterwarnings("ignore")
os.environ.setdefault("DJANGO_SETTINGS_MODULE", "rotation.settings")
import django
django.setup()
import numpy as np, pandas as pd, ta
import config
from seq_fundamental_study import load_candles

BENCH = getattr(config, "BENCHMARK", "SPY")
HORIZONS = [1, 3, 5, 10, 21]
PRE = [5, 10]


def _t(a):
    a = np.asarray(a, float); a = a[~np.isnan(a)]
    if len(a) < 3 or a.std(ddof=1) == 0:
        return None
    return round(float(a.mean() / (a.std(ddof=1) / np.sqrt(len(a)))), 1)


def _m(a):
    a = np.asarray(a, float); a = a[~np.isnan(a)]
    return (round(float(a.mean()) * 100, 2), round(float((a > 0).mean()) * 100, 1), len(a)) if len(a) else (None, None, 0)


def main():
    etfs = [e for e in config.SECTOR_ETFS.values() if e]
    data = load_candles(etfs + [BENCH])
    spy = data[BENCH]["Close"]

    events = ["RSI x30 (early)", "RSI x35", "RSI x50 (sweep)", "SMA 20/50 golden"]
    fwd = {ev: {h: [] for h in HORIZONS} for ev in events}
    pre = {ev: {w: [] for w in PRE} for ev in events}
    n = 0

    for etf in etfs:
        df = data.get(etf)
        if df is None or len(df) < 260:
            continue
        rs = (df["Close"] / spy.reindex(df.index)).dropna()
        if len(rs) < 260:
            continue
        n += 1
        rsi = ta.momentum.rsi(rs, window=10)
        f50, s50 = rs.rolling(20).mean(), rs.rolling(50).mean()
        bull = f50 > s50
        masks = {
            "RSI x30 (early)": (rsi > 30) & (rsi.shift(1) <= 30),
            "RSI x35": (rsi > 35) & (rsi.shift(1) <= 35),
            "RSI x50 (sweep)": (rsi > 50) & (rsi.shift(1) <= 50),
            "SMA 20/50 golden": bull & ~bull.shift(1).fillna(False),
        }
        for ev, mask in masks.items():
            m = mask.fillna(False)
            for h in HORIZONS:
                fwd[ev][h] += list((rs.shift(-h) / rs - 1)[m].dropna().values)
            for w in PRE:
                pre[ev][w] += list((rs / rs.shift(w) - 1)[m].dropna().values)

    print(f"\n=== ENTRY TIMING on {n} synthetic RS candles — RELATIVE fwd return (etf minus SPY) ===", flush=True)
    hdr = "  ".join(f"+{h}d" for h in HORIZONS)
    print(f"{'event':18} {'pre5d':>7} {'pre10d':>7} | {hdr}   (mean% / %pos / t)", flush=True)
    for ev in events:
        p5 = _m(pre[ev][5])[0]; p10 = _m(pre[ev][10])[0]
        cells = []
        for h in HORIZONS:
            mean, pos, cnt = _m(fwd[ev][h])
            cells.append(f"{mean:>+5}/{pos:>4}/{str(_t(fwd[ev][h])):>4}")
        print(f"{ev:18} {p5:>+7} {p10:>+7} | " + "  ".join(cells), flush=True)
    print("\n(pre = relative run-up BEFORE the event; positive fwd + = the entry still has upside)", flush=True)


if __name__ == "__main__":
    main()
