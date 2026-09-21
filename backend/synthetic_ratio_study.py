#!/usr/bin/env python3
"""SYNTHETIC relative-strength 'stocks': ratio = sleeve-ETF / benchmark (SPY, QQQ). Run indicators ON THE RATIO LINE
(not the raw price) and test whether they predict the ETF's forward ABSOLUTE return and its forward RELATIVE (ratio)
move — i.e. does an RS-line signal time rotation INTO that sleeve? Daily AND weekly bars (user 2026-09-21).

Indicators on the ratio:
  • rsi10_oscross / rsi14_oscross — Wilder RSI crosses UP through 35 (RS turning up from a relative-low)
  • ha_flip_up                    — Heikin-Ashi candle flips red->green on the ratio (RS trend change up)
Forward measures per signal: ETF absolute return, and ratio (relative) return, over several horizons. Compared to the
UNCONDITIONAL baseline (all bars) so we see if the signal adds. Pooled across all sleeves; t-stats + win-rates + n.
Read-only. Saves BacktestResult[synthetic_ratio] + /app/.data/studies/synthetic_ratio.json.
Run: MSYS_NO_PATHCONV=1 docker exec rotation-backend-1 python -u /app/synthetic_ratio_study.py"""
import os, json
os.environ.setdefault("DJANGO_SETTINGS_MODULE", "rotation.settings")
import django; django.setup()
import numpy as np, pandas as pd
from core.models import Candle, BacktestResult
import config

BENCHES = ["SPY", "QQQ"]
OS_LEVELS = [20.0, 30.0]                    # DEEP-oversold thresholds on the ratio RSI (user: below 20; 30 for compare)
OUT = "/app/.data/studies/synthetic_ratio.json"
HZ = {"daily": [5, 10, 21, 63], "weekly": [1, 2, 4, 8]}      # forward horizons (bars)


def _t(x):
    x = np.asarray(x, float); x = x[np.isfinite(x)]
    if len(x) < 3 or x.std(ddof=1) == 0:
        return float("nan")
    return float(x.mean() / (x.std(ddof=1) / np.sqrt(len(x))))


def wilder_rsi(close, n):
    d = close.diff()
    up = d.clip(lower=0.0); dn = (-d).clip(lower=0.0)
    ru = up.ewm(alpha=1.0 / n, adjust=False, min_periods=n).mean()
    rd = dn.ewm(alpha=1.0 / n, adjust=False, min_periods=n).mean()
    rs = ru / rd.replace(0.0, np.nan)
    return (100.0 - 100.0 / (1.0 + rs)).fillna(50.0)


def heikin_ashi_open(o, h, l, c):
    ha_c = (o + h + l + c) / 4.0
    ha_o = ha_c.to_numpy(dtype=float, copy=True)
    ha_o[0] = (o.iloc[0] + c.iloc[0]) / 2.0
    cvals = ha_c.to_numpy(dtype=float)
    for i in range(1, len(ha_o)):
        ha_o[i] = (ha_o[i - 1] + cvals[i - 1]) / 2.0
    return pd.Series(ha_o, index=c.index), ha_c


def load(tk):
    rows = list(Candle.objects.filter(ticker=tk).values_list("date", "open", "high", "low", "close"))
    if not rows:
        return None
    df = pd.DataFrame(rows, columns=["date", "o", "h", "l", "c"])
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").set_index("date")
    return df[df["c"] > 0]


def weekly(df):
    return df.resample("W-FRI").agg(o=("o", "first"), h=("h", "max"), l=("l", "min"), c=("c", "last")).dropna()


def ratio_ohlc(etf, bench):
    idx = etf.index.intersection(bench.index)
    e, b = etf.loc[idx], bench.loc[idx]
    r = pd.DataFrame(index=idx)
    r["c"] = e["c"] / b["c"]
    r["o"] = e["o"] / b["o"]
    r["h"] = e["h"] / b["l"]                 # ratio intraday max ≈ etf-high / bench-low
    r["l"] = e["l"] / b["h"]                 # ratio intraday min ≈ etf-low / bench-high
    return r, e["c"]                         # ratio frame + the ETF's own close (for absolute fwd return)


def signals(r):
    out = {}
    for n in (10, 14):
        rsi = wilder_rsi(r["c"], n)
        for lvl in OS_LEVELS:
            out[f"rsi{n}_os{int(lvl)}"] = (rsi.shift(1) < lvl) & (rsi >= lvl)   # cross UP through deep-oversold on RS line
    ha_o, ha_c = heikin_ashi_open(r["o"], r["h"], r["l"], r["c"])
    green = ha_c > ha_o
    out["ha_flip_up"] = green & (~green.shift(1).fillna(False))
    return out


def main():
    sleeves = sorted(set(config.SECTOR_ETFS.values()))
    px = {}
    for tk in set(sleeves) | set(BENCHES):
        d = load(tk)
        if d is not None and len(d) > 200:
            px[tk] = d
    print(f"loaded {len(px)} series ({len(sleeves)} sleeves + benches)", flush=True)

    # accumulators: acc[(bench,tf,ind,H)] -> {"etf":[...], "ratio":[...]}; base[(bench,tf,H)] -> [unconditional etf fwd]
    from collections import defaultdict
    acc = defaultdict(lambda: {"etf": [], "ratio": []})
    base = defaultdict(list)
    for bench in BENCHES:
        if bench not in px:
            continue
        bd = px[bench]; bw = weekly(bd)
        for tk in sleeves:
            if tk not in px or tk == bench:
                continue
            for tf, bpx in [("daily", bd), ("weekly", bw)]:
                epx = px[tk] if tf == "daily" else weekly(px[tk])
                r, ec = ratio_ohlc(epx, bpx)
                if len(r) < 120:
                    continue
                sig = signals(r)
                for H in HZ[tf]:
                    efwd = ec.shift(-H) / ec - 1.0            # ETF absolute forward return
                    rfwd = r["c"].shift(-H) / r["c"] - 1.0    # ratio (relative) forward return
                    base[(bench, tf, H)].extend(efwd.dropna().tolist())
                    for ind, s in sig.items():
                        m = s.fillna(False) & efwd.notna()
                        acc[(bench, tf, ind, H)]["etf"].extend(efwd[m].tolist())
                        acc[(bench, tf, ind, H)]["ratio"].extend(rfwd[m].tolist())

    res = {"baseline": {}, "signals": {}}
    for (bench, tf, H), v in base.items():
        a = np.asarray(v, float)
        res["baseline"][f"{bench}|{tf}|{H}"] = dict(n=len(a), mean_pct=round(float(np.nanmean(a)) * 100, 3),
                                                     win_pct=round(float((a > 0).mean()) * 100, 1))
    for (bench, tf, ind, H), v in acc.items():
        e = np.asarray(v["etf"], float); rr = np.asarray(v["ratio"], float)
        if len(e) < 20:
            continue
        b = res["baseline"].get(f"{bench}|{tf}|{H}", {})
        res["signals"][f"{bench}|{tf}|{ind}|{H}"] = dict(
            n=len(e),
            etf_mean_pct=round(float(np.nanmean(e)) * 100, 3), etf_win_pct=round(float((e > 0).mean()) * 100, 1),
            etf_t=round(_t(e), 2), etf_edge_vs_base_pct=round(float(np.nanmean(e)) * 100 - b.get("mean_pct", 0), 3),
            ratio_mean_pct=round(float(np.nanmean(rr)) * 100, 3), ratio_win_pct=round(float((rr > 0).mean()) * 100, 1),
            ratio_t=round(_t(rr), 2))

    # rank signals by ETF edge vs baseline (does the RS signal beat just holding the sleeve?)
    ranked = sorted(res["signals"].items(), key=lambda kv: -kv[1]["etf_edge_vs_base_pct"])
    print("\n=== TOP RS-ratio signals by ETF edge vs baseline (bench|tf|indicator|horizon) ===", flush=True)
    for k, s in ranked[:20]:
        print(f"  {k:34} n={s['n']:>5}  ETFfwd={s['etf_mean_pct']:+.2f}% (base {s['etf_mean_pct']-s['etf_edge_vs_base_pct']:+.2f}, "
              f"edge {s['etf_edge_vs_base_pct']:+.2f}, t{s['etf_t']}, win{s['etf_win_pct']}%)  RATIOfwd={s['ratio_mean_pct']:+.2f}% (t{s['ratio_t']})", flush=True)

    json.dump(res, open(OUT, "w"), indent=1)
    try:
        from django.utils import timezone
        BacktestResult.objects.update_or_create(kind="synthetic_ratio", defaults=dict(payload=res, computed_at=timezone.now()))
        print("\nsaved BacktestResult[synthetic_ratio] +", OUT, flush=True)
    except Exception as e:
        print("save skipped:", e, flush=True)


if __name__ == "__main__":
    main()
