#!/usr/bin/env python3
"""OVERSOLD BOUNCE — short-term absolute single-stock reversal entry (the edge the RS bar does NOT have).

RSI(10) crossing up out of oversold on ABSOLUTE price gives a positive short-term forward return, and
DEEPER oversold pays more (the tail). Confirmed vs the RS bar, where the same signal is a falling knife
(relative strength is trend-persistent). This surfaces (a) the entry backtest by threshold + by oversold
depth, and (b) a LIVE firing list — stocks that just crossed up out of oversold, most-oversold first.

-> BacktestResult[oversold_bounce] + JSON. Diagnostic/directional: forward windows, no fees; stock-universe
survivorship applies. Not wired into risk rating.
Run: MSYS_NO_PATHCONV=1 docker exec rotation-backend-1 python -u /app/oversold_bounce.py
"""
import os, json, warnings
warnings.filterwarnings("ignore")
os.environ.setdefault("DJANGO_SETTINGS_MODULE", "rotation.settings")
import django
django.setup()
import numpy as np, pandas as pd, ta
import config, sector_holdings
from seq_fundamental_study import load_candles

HORIZONS = [1, 3, 5, 10, 21]
THRESHOLDS = [("RSI x20 (deep)", 20), ("RSI x30", 30), ("RSI x35", 35), ("RSI x50", 50)]
DEPTH_BUCKETS = [(0, 15), (15, 20), (20, 25), (25, 30)]   # min RSI in prior 5 bars, for x30 crosses
FIRING_LOOKBACK = 2      # a cross within the last N bars = "firing now"


def _stats(a):
    a = np.asarray(a, float); a = a[~np.isnan(a)]
    if not len(a):
        return {"mean_pct": None, "pos_pct": None, "t": None, "n": 0}
    t = None
    if len(a) >= 3 and a.std(ddof=1) > 0:
        t = round(float(a.mean() / (a.std(ddof=1) / np.sqrt(len(a)))), 1)
    return {"mean_pct": round(float(a.mean()) * 100, 2), "pos_pct": round(float((a > 0).mean()) * 100, 1),
            "t": t, "n": int(len(a))}


def build():
    from core.models import Candle, Fundamental
    etfs = set(config.SECTOR_ETFS.values()) | {"SPY", "QQQ"}
    cand = set(Candle.objects.values_list("ticker", flat=True).distinct())
    fund = set(Fundamental.objects.values_list("ticker", flat=True).distinct())
    universe = sorted((cand & fund) - etfs)
    data = load_candles(universe)
    caps = {r["ticker"]: r["market_cap"] for r in
            Fundamental.objects.filter(ticker__in=universe).order_by("ticker", "-date")
            .values("ticker", "market_cap")}   # first (latest) per ticker via ordering
    # keep the LATEST market cap per ticker
    caps = {}
    for r in (Fundamental.objects.filter(ticker__in=universe).order_by("ticker", "-date")
              .values("ticker", "market_cap")):
        caps.setdefault(r["ticker"], r["market_cap"])

    fwd = {ev: {h: [] for h in HORIZONS} for ev, _ in THRESHOLDS}
    depth = {b: {5: [], 10: []} for b in DEPTH_BUCKETS}
    firing = []
    n = 0

    for tkr in universe:
        df = data.get(tkr)
        if df is None or len(df) < 120:
            continue
        c = df["Close"]
        rsi = ta.momentum.rsi(c, window=10)
        if rsi.notna().sum() < 60:
            continue
        n += 1
        fret = {h: c.shift(-h) / c - 1 for h in HORIZONS}
        for ev, thr in THRESHOLDS:
            m = ((rsi > thr) & (rsi.shift(1) <= thr)).fillna(False)
            for h in HORIZONS:
                fwd[ev][h] += list(fret[h][m].dropna().values)
        # depth buckets on the x30 cross
        m30 = ((rsi > 30) & (rsi.shift(1) <= 30)).fillna(False)
        prior_min = rsi.rolling(5).min()
        for idx in np.where(m30.values)[0]:
            pm = prior_min.iloc[idx]
            if not np.isfinite(pm):
                continue
            for lo, hi in DEPTH_BUCKETS:
                if lo <= pm < hi:
                    for h in (5, 10):
                        v = fret[h].iloc[idx]
                        if np.isfinite(v):
                            depth[(lo, hi)][h].append(v)
                    break
        # live firing: crossed up through 30 (preferred) or 35 within the last FIRING_LOOKBACK bars
        for thr in (30, 35):
            m = ((rsi > thr) & (rsi.shift(1) <= thr)).fillna(False)
            recent = m.iloc[-FIRING_LOOKBACK:]
            if recent.any():
                days_ago = int(len(recent) - 1 - np.where(recent.values)[0][-1])
                dmin = float(rsi.iloc[-7:].min())
                firing.append({
                    "ticker": tkr, "threshold": thr, "days_ago": days_ago,
                    "rsi_now": round(float(rsi.iloc[-1]), 1), "min_rsi_7d": round(dmin, 1),
                    "close": round(float(c.iloc[-1]), 2),
                    "market_cap": caps.get(tkr),
                    "sectors": sector_holdings.get_sectors_for_ticker(tkr)[:2],
                })
                break

    firing.sort(key=lambda x: x["min_rsi_7d"])   # most oversold first = the tail
    payload = {
        "computed_at": pd.Timestamp.utcnow().isoformat(),
        "params": {"universe_n": n, "horizons_days": HORIZONS, "rsi_window": 10,
                   "firing_lookback_bars": FIRING_LOOKBACK,
                   "rule": "RSI(10) crosses UP out of oversold on absolute price; deeper = bigger bounce"},
        "by_threshold": [{"event": ev, "fwd": {f"{h}d": _stats(fwd[ev][h]) for h in HORIZONS}}
                         for ev, _ in THRESHOLDS],
        "by_depth": [{"bucket": f"{lo}-{hi}", "fwd5": _stats(depth[(lo, hi)][5]),
                      "fwd10": _stats(depth[(lo, hi)][10])} for lo, hi in DEPTH_BUCKETS],
        "firing": firing,
        "note": ("Absolute single-stock oversold-reversal entry — positive short-term forward return, deeper "
                 "oversold pays more. The MIRROR of the RS bar (where oversold is a falling knife). Directional, "
                 "forward windows, no fees; stock-universe survivorship. NOT wired into risk rating."),
    }
    return payload


def main():
    from pathlib import Path
    payload = build()
    out = Path(__file__).resolve().parent / ".data" / "studies" / "oversold_bounce.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2, default=str))
    try:
        from core.models import BacktestResult
        from django.utils import timezone
        BacktestResult.objects.update_or_create(
            kind="oversold_bounce",
            defaults={"payload": json.loads(json.dumps(payload, default=str)),
                      "computed_at": timezone.now()})
        print("Saved BacktestResult[oversold_bounce]", flush=True)
    except Exception as e:
        print("DB save failed:", e, flush=True)

    print(f"\n=== OVERSOLD BOUNCE — {payload['params']['universe_n']} stocks, forward ABSOLUTE return ===", flush=True)
    print(f"{'event':16} | " + "  ".join(f"+{h}d" for h in HORIZONS), flush=True)
    for row in payload["by_threshold"]:
        cells = [f"{row['fwd'][f'{h}d']['mean_pct']:>+5}/{row['fwd'][f'{h}d']['pos_pct']}%/t{row['fwd'][f'{h}d']['t']}" for h in HORIZONS]
        print(f"{row['event']:16} | " + "  ".join(cells), flush=True)
    print("\nTHE TAIL (x30 by prior-min RSI):", flush=True)
    for row in payload["by_depth"]:
        print(f"  {row['bucket']:8} +5d {row['fwd5']['mean_pct']:>+5}% (t{row['fwd5']['t']}, n{row['fwd5']['n']})  "
              f"+10d {row['fwd10']['mean_pct']:>+5}% (t{row['fwd10']['t']}, n{row['fwd10']['n']})", flush=True)
    print(f"\nFIRING NOW: {len(payload['firing'])} stocks crossed up out of oversold in last {FIRING_LOOKBACK} bars. "
          f"Most oversold: " + ", ".join(f"{f['ticker']}(min{f['min_rsi_7d']})" for f in payload["firing"][:10]), flush=True)


if __name__ == "__main__":
    main()
