#!/usr/bin/env python3
"""EXPAND the synthetic-stock (RS-ratio) concept to hunt more alpha (user 2026-09-21). Two axes:
  RATIO TYPES: stock/sector-ETF (proven winner) AND stock/SPY (RS vs the whole market).
  INDICATOR FAMILIES on the RS line — MEAN-REVERSION vs MOMENTUM (does relative strength PERSIST or REVERT?):
    MR : rsi14_os20, rsi14_os30      (cross UP through deep-oversold — proven)
         rsi14_ob80                  (cross DOWN through 80 = overbought fade / short-or-exit)
    MOM: rs_newhi252                 (ratio breaks its prior 252d high = RS breakout)
         rs_ma_gx                    (SMA20 of ratio crosses ABOVE SMA100 = RS golden cross)
         rs_reclaim100               (ratio crosses back above its own SMA100 = RS uptrend resume)
Forward STOCK absolute return + ratio (relative) return, edge vs unconditional baseline, t-stats, and BOTH-HALVES
(2016-21 / 2021-26) so a hit must hold out-of-sample to count. Daily, horizons 21 & 63d (the proven window).
Read-only; BacktestResult[synthetic_alpha_scan] + JSON. Run: docker exec ... python -u /app/synthetic_alpha_scan.py"""
import os, json
os.environ.setdefault("DJANGO_SETTINGS_MODULE", "rotation.settings")
import django; django.setup()
import numpy as np, pandas as pd
from collections import defaultdict
from core.models import BacktestResult
import config, sector_holdings as sh
from synthetic_ratio_study import wilder_rsi, _t
from stock_sector_ratio_study import _bulk_load, ratio_ohlc

OUT = "/app/.data/studies/synthetic_alpha_scan.json"
HZS = [21, 63]
SPLIT = pd.Timestamp("2021-05-31")     # H1 <= SPLIT < H2


def sma(s, n):
    return s.rolling(n, min_periods=n).mean()


def indicators(r):
    """All signals on the ratio close (event-based crossings)."""
    c = r["c"]; out = {}
    r10 = wilder_rsi(c, 10); r14 = wilder_rsi(c, 14)
    out["rsi14_os20"] = (r14.shift(1) < 20) & (r14 >= 20)
    out["rsi14_os30"] = (r14.shift(1) < 30) & (r14 >= 30)
    out["rsi10_os20"] = (r10.shift(1) < 20) & (r10 >= 20)
    out["rsi14_ob80"] = (r14.shift(1) > 80) & (r14 <= 80)                  # overbought fade (down-cross)
    pm = c.shift(1).rolling(252, min_periods=200).max()                   # prior 252d high (excl today)
    out["rs_newhi252"] = (c > pm) & (c.shift(1) <= pm.shift(0))           # RS breakout to new 1y high
    s20, s100 = sma(c, 20), sma(c, 100)
    out["rs_ma_gx"] = (s20 > s100) & (s20.shift(1) <= s100.shift(1))      # RS golden cross (20>100)
    out["rs_reclaim100"] = (c > s100) & (c.shift(1) <= s100.shift(1))     # reclaim the 100d RS mean
    return out


def _agg(a):
    a = np.asarray(a, float); a = a[np.isfinite(a)]
    if len(a) == 0:
        return None
    return dict(n=len(a), mean=round(float(a.mean()) * 100, 3), win=round(float((a > 0).mean()) * 100, 1), t=round(_t(a), 2))


def main():
    etf_of = {}
    for name, etf in config.SECTOR_ETFS.items():
        for tk in sh.get_holdings(name):
            etf_of.setdefault(tk, etf)
    etfs = sorted(set(config.SECTOR_ETFS.values()))
    stocks = sorted(etf_of)
    px = _bulk_load(set(etfs) | set(stocks) | {"SPY"})
    print(f"loaded {len(px)} series; {len(stocks)} stocks", flush=True)

    # acc[(pair,ind,H,half)] -> {"stk":[],"ratio":[]}; base[(pair,H,half)] -> stk fwd
    acc = defaultdict(lambda: {"stk": [], "ratio": []}); base = defaultdict(list)
    PAIRS = [("sector", None), ("spy", "SPY")]
    n = 0
    for stk in stocks:
        if stk not in px:
            continue
        for pair, fixed in PAIRS:
            bench = fixed or etf_of[stk]
            if bench not in px or bench == stk:
                continue
            r, sc = ratio_ohlc(px[stk], px[bench])
            if len(r) < 300:
                continue
            sig = indicators(r)
            dates = r.index
            for H in HZS:
                sfwd = sc.shift(-H) / sc - 1.0
                rfwd = r["c"].shift(-H) / r["c"] - 1.0
                for half, mask in [("full", pd.Series(True, index=dates)),
                                   ("H1", dates <= SPLIT), ("H2", dates > SPLIT)]:
                    hm = pd.Series(mask, index=dates) if not isinstance(mask, pd.Series) else mask
                    base[(pair, H, half)].extend(sfwd[hm & sfwd.notna()].tolist())
                    for ind, s in sig.items():
                        m = s.fillna(False) & sfwd.notna() & hm
                        acc[(pair, ind, H, half)]["stk"].extend(sfwd[m].tolist())
                        acc[(pair, ind, H, half)]["ratio"].extend(rfwd[m].tolist())
        n += 1
    print(f"processed {n} stocks x {len(PAIRS)} ratio types", flush=True)

    res = {"baseline": {}, "signals": {}}
    for k, v in base.items():
        res["baseline"]["|".join(map(str, k))] = _agg(v)
    rows = []
    for (pair, ind, H, half), v in acc.items():
        s = _agg(v["stk"]); rr = _agg(v["ratio"])
        if not s or s["n"] < 30:
            continue
        b = res["baseline"].get(f"{pair}|{H}|{half}") or {}
        edge = round(s["mean"] - (b.get("mean") or 0), 3)
        rec = dict(pair=pair, ind=ind, H=H, half=half, n=s["n"], stk_mean=s["mean"], stk_t=s["t"],
                   stk_win=s["win"], edge=edge, ratio_mean=(rr or {}).get("mean"), ratio_t=(rr or {}).get("t"))
        res["signals"]["|".join([pair, ind, str(H), half])] = rec
        if half == "full":
            rows.append(rec)

    # robustness: for each full-sample hit, pull its H1/H2 edge
    def half_edge(pair, ind, H, half):
        k = "|".join([pair, ind, str(H), half]); r = res["signals"].get(k)
        b = res["baseline"].get(f"{pair}|{H}|{half}") or {}
        return round((r["stk_mean"] - (b.get("mean") or 0)), 3) if r else None

    print("\n=== FULL-sample RS signals by edge, with H1/H2 robustness (both must be + to count) ===", flush=True)
    for rec in sorted(rows, key=lambda r: -r["edge"])[:30]:
        e1 = half_edge(rec["pair"], rec["ind"], rec["H"], "H1"); e2 = half_edge(rec["pair"], rec["ind"], rec["H"], "H2")
        ok = "OK " if (e1 is not None and e2 is not None and e1 > 0 and e2 > 0) else "   "
        print(f"  {ok}{rec['pair']:6}|{rec['ind']:13}|{rec['H']:>2}d  n={rec['n']:>6}  edge={rec['edge']:+.2f} "
              f"(H1{e1:+} H2{e2:+})  stk_t={rec['stk_t']}  win={rec['stk_win']}  ratio={rec['ratio_mean']}(t{rec['ratio_t']})", flush=True)

    json.dump(res, open(OUT, "w"), indent=1)
    try:
        from django.utils import timezone
        BacktestResult.objects.update_or_create(kind="synthetic_alpha_scan", defaults=dict(payload=res, computed_at=timezone.now()))
        print("\nsaved BacktestResult[synthetic_alpha_scan] +", OUT, flush=True)
    except Exception as e:
        print("save skipped:", e, flush=True)


if __name__ == "__main__":
    main()
