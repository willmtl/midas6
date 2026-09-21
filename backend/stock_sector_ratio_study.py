#!/usr/bin/env python3
"""Synthetic RS ratio at the STOCK level: ratio = stock / its SECTOR ETF (not sleeve/SPY). Mean-reversion should be
stronger here — a stock oversold vs its own sector is a cleaner snap-back setup than a whole sleeve vs SPY. Tests the
same deep-oversold RS-line signals (user follow-up 2026-09-21): RSI(14)/RSI(10) cross UP through 20/30 on stock/sector,
+ Heikin-Ashi flip. Measures forward STOCK absolute return AND ratio (vs-sector) return, vs unconditional baseline.
Daily + weekly. Reuses helpers from synthetic_ratio_study. Read-only; BacktestResult[stock_sector_ratio] + JSON.
Run: MSYS_NO_PATHCONV=1 docker exec rotation-backend-1 python -u /app/stock_sector_ratio_study.py"""
import os, json
os.environ.setdefault("DJANGO_SETTINGS_MODULE", "rotation.settings")
import django; django.setup()
import numpy as np, pandas as pd
from collections import defaultdict
from core.models import BacktestResult
import config, sector_holdings as sh
from synthetic_ratio_study import wilder_rsi, heikin_ashi_open, weekly, signals, HZ, _t, OS_LEVELS
from seq_fundamental_study import load_candles

OUT = "/app/.data/studies/stock_sector_ratio.json"


def _bulk_load(tickers):
    """One-query bulk load (vs per-ticker) -> {tk: df[o,h,l,c]} with c>0. Reuses the flagship candle loader."""
    raw = load_candles(sorted(set(tickers)))          # {tk: df[Open,High,Low,Close,Volume]} in a single DB query
    out = {}
    for tk, df in raw.items():
        if df is None or len(df) < 200:
            continue
        g = df.rename(columns={"Open": "o", "High": "h", "Low": "l", "Close": "c"})[["o", "h", "l", "c"]]
        g = g[g["c"] > 0]
        if len(g) >= 200:
            out[tk] = g
    return out


def ratio_ohlc(stk, etf):
    idx = stk.index.intersection(etf.index)
    s, e = stk.loc[idx], etf.loc[idx]
    r = pd.DataFrame(index=idx)
    r["c"] = s["c"] / e["c"]; r["o"] = s["o"] / e["o"]
    r["h"] = s["h"] / e["l"]; r["l"] = s["l"] / e["h"]
    return r, s["c"]


def main():
    print(f"OS levels {OS_LEVELS}", flush=True)
    # ETF -> its holdings (stocks). Cache candles as we go.
    etf_of = {}                                            # stock -> sector ETF
    for name, etf in config.SECTOR_ETFS.items():
        for tk in sh.get_holdings(name):
            etf_of.setdefault(tk, etf)                     # first sector wins (stable)
    etfs = sorted(set(config.SECTOR_ETFS.values()))
    stocks = sorted(etf_of)
    print(f"{len(stocks)} stocks across {len(etfs)} sector ETFs", flush=True)

    px = _bulk_load(set(etfs) | set(stocks))
    print(f"loaded {len(px)} series (bulk)", flush=True)

    acc = defaultdict(lambda: {"stk": [], "ratio": []})
    base = defaultdict(list)
    n_pairs = 0
    for stk in stocks:
        etf = etf_of[stk]
        if stk not in px or etf not in px or stk == etf:
            continue
        for tf in ("daily", "weekly"):
            spx = px[stk] if tf == "daily" else weekly(px[stk])
            epx = px[etf] if tf == "daily" else weekly(px[etf])
            r, sc = ratio_ohlc(spx, epx)
            if len(r) < 120:
                continue
            sig = signals(r)
            for H in HZ[tf]:
                sfwd = sc.shift(-H) / sc - 1.0            # STOCK absolute forward return
                rfwd = r["c"].shift(-H) / r["c"] - 1.0    # ratio (vs-sector) forward return
                base[(tf, H)].extend(sfwd.dropna().tolist())
                for ind, s in sig.items():
                    m = s.fillna(False) & sfwd.notna()
                    acc[(tf, ind, H)]["stk"].extend(sfwd[m].tolist())
                    acc[(tf, ind, H)]["ratio"].extend(rfwd[m].tolist())
        n_pairs += 1
    print(f"processed {n_pairs} stock/sector pairs", flush=True)

    res = {"baseline": {}, "signals": {}}
    for (tf, H), v in base.items():
        a = np.asarray(v, float)
        res["baseline"][f"{tf}|{H}"] = dict(n=len(a), mean_pct=round(float(np.nanmean(a)) * 100, 3),
                                             win_pct=round(float((a > 0).mean()) * 100, 1))
    for (tf, ind, H), v in acc.items():
        s = np.asarray(v["stk"], float); rr = np.asarray(v["ratio"], float)
        if len(s) < 20:
            continue
        b = res["baseline"].get(f"{tf}|{H}", {})
        res["signals"][f"{tf}|{ind}|{H}"] = dict(
            n=len(s), stk_mean_pct=round(float(np.nanmean(s)) * 100, 3), stk_win_pct=round(float((s > 0).mean()) * 100, 1),
            stk_t=round(_t(s), 2), stk_edge_vs_base_pct=round(float(np.nanmean(s)) * 100 - b.get("mean_pct", 0), 3),
            ratio_mean_pct=round(float(np.nanmean(rr)) * 100, 3), ratio_win_pct=round(float((rr > 0).mean()) * 100, 1),
            ratio_t=round(_t(rr), 2))

    ranked = sorted(res["signals"].items(), key=lambda kv: -kv[1]["stk_edge_vs_base_pct"])
    print("\n=== TOP stock/sector RS-ratio signals by STOCK edge vs baseline (tf|indicator|horizon) ===", flush=True)
    for k, s in ranked[:22]:
        print(f"  {k:26} n={s['n']:>6}  STKfwd={s['stk_mean_pct']:+.2f}% (base {s['stk_mean_pct']-s['stk_edge_vs_base_pct']:+.2f}, "
              f"edge {s['stk_edge_vs_base_pct']:+.2f}, t{s['stk_t']}, win{s['stk_win_pct']}%)  RATIOfwd={s['ratio_mean_pct']:+.2f}% (t{s['ratio_t']})", flush=True)

    json.dump(res, open(OUT, "w"), indent=1)
    try:
        from django.utils import timezone
        BacktestResult.objects.update_or_create(kind="stock_sector_ratio", defaults=dict(payload=res, computed_at=timezone.now()))
        print("\nsaved BacktestResult[stock_sector_ratio] +", OUT, flush=True)
    except Exception as e:
        print("save skipped:", e, flush=True)


if __name__ == "__main__":
    main()
