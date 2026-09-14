#!/usr/bin/env python3
"""RSI(14) suppressed-then-cross, ISOLATED on large-cap (S&P 500 proxy) individual STOCKS.

User ask: the `rsi_sup10_x` "coiled spring" (RSI held <50 AND <its own SMA for X+ consecutive
days, then crosses above its SMA) but with RSI(14) instead of RSI(10), tested on SPY / S&P 500
stocks only (large-cap), standalone.

We have NO SPY constituent list, so "S&P 500 stocks" = top-N US names by current market cap
(N=500). We run RSI(10) AND RSI(14) on the SAME universe so the comparison isolates the PERIOD,
not the universe. Also report the full stock universe for context, and a couple of exits.

Signal (exactly the studies.py family, window parametrized):
  suppressed[t] = RSI(win)[t] < LEVEL AND RSI(win)[t] < SMA(win of RSI)[t]
  cross[t]      = RSI[t] > SMA[t] AND RSI[t-1] <= SMA[t-1]
  fire[t]       = cross[t] AND consecutive-suppressed-streak ending at t-1 >= MIN_STREAK

-> BacktestResult[rsi14_suppressed_largecap] + JSON. $5 price floor, $5M median $-vol floor.
Run: MSYS_NO_PATHCONV=1 docker exec rotation-backend-1 python -u /app/rsi14_suppressed_largecap.py
"""
import os, json, warnings
warnings.filterwarnings("ignore")
import numpy as np, pandas as pd, ta
os.environ.setdefault("DJANGO_SETTINGS_MODULE", "rotation.settings")
import django
django.setup()
from pathlib import Path

from seq_fundamental_study import build_universe, load_candles, load_financial_reports

OUT = Path(__file__).resolve().parent / ".data" / "studies" / "rsi14_suppressed_largecap.json"
MIN_STREAK, LEVEL = 10, 50
HOLDS = {"6m": 126, "3m": 63, "12m": 252}      # fixed-horizon exits (trading days)
PRICE_FLOOR, DVOL_FLOOR = 5.0, 5_000_000.0
TOP_N = 500                                     # S&P 500 proxy


def suppressed_then_cross(close, win):
    rsi = ta.momentum.rsi(close, window=win)
    sma = rsi.rolling(win).mean()
    cross = (rsi > sma) & (rsi.shift(1) <= sma.shift(1))
    suppressed = (rsi < LEVEL) & (rsi < sma)
    grp = (~suppressed).cumsum()
    streak = suppressed.groupby(grp).cumsum()
    return (cross & (streak.shift(1) >= MIN_STREAK)).fillna(False)


def episode_dedup(idxs, gap):
    """Keep entries at least `gap` bars apart per ticker (independent episodes for the t-stat)."""
    out, last = [], -10**9
    for i in sorted(idxs):
        if i - last >= gap:
            out.append(i); last = i
    return out


def run(candles, tickers, win, hold_key):
    hold = HOLDS[hold_key]
    rets = []
    for t in tickers:
        df = candles.get(t)
        if df is None or len(df) < 60 + win:
            continue
        close = df["Close"].values
        dvol = (df["Close"] * df["Volume"]).rolling(20).median().values
        n = len(close)
        sig = suppressed_then_cross(df["Close"], win).values
        idxs = [i for i in range(n) if sig[i]]
        idxs = episode_dedup(idxs, hold)          # non-overlapping episodes for this horizon
        for i in idxs:
            j = i + hold
            if j >= n:
                continue
            ep = close[i]
            if ep < PRICE_FLOOR or not np.isfinite(dvol[i]) or dvol[i] < DVOL_FLOOR:
                continue
            r = (close[j] - ep) / ep * 100
            if np.isfinite(r):
                rets.append(r)
    a = np.array(rets)
    if len(a) == 0:
        return {"trades": 0}
    t_stat = float(a.mean() / (a.std(ddof=1) / np.sqrt(len(a)))) if len(a) > 1 and a.std() > 0 else None
    return {"trades": int(len(a)), "avg_ret": round(float(a.mean()), 2),
            "win": round(float((a > 0).mean() * 100), 1),
            "t_stat": round(t_stat, 2) if t_stat is not None else None,
            "median_ret": round(float(np.median(a)), 2)}


def main():
    import time
    from django.db import connection

    def robust_load(chunk, tries=6):
        for a in range(tries):
            try:
                return load_candles(chunk)
            except Exception as e:
                connection.close()                 # drop the dead connection
                wait = 5 * (a + 1)
                print(f"    load retry {a+1}/{tries} after error ({type(e).__name__}); "
                      f"waiting {wait}s for DB", flush=True)
                time.sleep(wait)
        # last resort: per-ticker so one bad/heavy ticker can't sink the whole chunk
        out = {}
        for t in chunk:
            try:
                out.update(load_candles([t]))
            except Exception:
                connection.close(); time.sleep(2)
        return out

    tks = build_universe()
    print(f"universe {len(tks)} tickers; loading candles in chunks...", flush=True)
    candles = {}
    CH = 25
    for k in range(0, len(tks), CH):
        chunk = tks[k:k + CH]
        candles.update(robust_load(chunk))
        connection.close()                          # release between chunks
        if (k // CH) % 8 == 0:
            print(f"  loaded {min(k + CH, len(tks))}/{len(tks)}", flush=True)
    print(f"  loaded {len(candles)} tickers total", flush=True)
    reps = load_financial_reports(tks)

    # current market cap = latest reported shares_outstanding * latest close
    mcap = {}
    for t in tks:
        df = candles.get(t); rep = reps.get(t)
        if df is None or len(df) == 0 or rep is None or not len(rep):
            continue
        r2 = rep.dropna(subset=["shares_outstanding"]).sort_values("avail_date")
        if not len(r2):
            continue
        sh = float(r2["shares_outstanding"].iloc[-1])
        px = float(df["Close"].iloc[-1])
        if sh > 0 and px > 0:
            mcap[t] = sh * px
    ranked = sorted(mcap, key=lambda t: mcap[t], reverse=True)
    largecap = ranked[:TOP_N]
    cutoff = mcap[largecap[-1]] / 1e9 if largecap else 0
    print(f"large-cap universe = top {len(largecap)} by mktcap (floor ~${cutoff:.1f}B); "
          f"full universe {len(tks)}", flush=True)

    results = {}
    for uni_name, uni in [("largecap_top500", largecap), ("full_universe", tks)]:
        for win in (10, 14):
            for hk in ("6m", "3m", "12m"):
                key = f"{uni_name}|rsi{win}|{hk}"
                results[key] = run(candles, uni, win, hk)

    print("\n=== RSI suppressed-then-cross (>=10 consecutive days <50 & <SMA, then cross up) ===", flush=True)
    print(f"{'universe':16} {'rsi':>4} {'exit':>5} {'trades':>7} {'avg%':>7} {'win%':>6} {'t':>6} {'med%':>7}", flush=True)
    for uni_name in ("largecap_top500", "full_universe"):
        for win in (10, 14):
            for hk in ("6m", "3m", "12m"):
                r = results[f"{uni_name}|rsi{win}|{hk}"]
                if r.get("trades", 0) == 0:
                    print(f"{uni_name:16} {win:>4} {hk:>5} {'0':>7}", flush=True); continue
                print(f"{uni_name:16} {win:>4} {hk:>5} {r['trades']:>7} {r['avg_ret']:>+7.2f} "
                      f"{r['win']:>6.1f} {str(r['t_stat']):>6} {r['median_ret']:>+7.2f}", flush=True)

    payload = {"computed_at": pd.Timestamp.utcnow().isoformat(),
               "params": {"min_streak": MIN_STREAK, "level": LEVEL, "top_n": TOP_N,
                          "price_floor": PRICE_FLOOR, "dvol_floor": DVOL_FLOOR,
                          "largecap_floor_bn": round(cutoff, 2)},
               "results": results,
               "caveat": "S&P 500 proxied as top-500 by CURRENT mktcap (survivorship + look-ahead on "
                         "membership). Standalone signal, fixed-horizon exits, in-sample, no fees. RSI(10) "
                         "run on the SAME universe as control to isolate the period effect."}
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(payload, indent=2, default=str))
    try:
        from core.models import BacktestResult
        from django.utils import timezone
        BacktestResult.objects.update_or_create(
            kind="rsi14_suppressed_largecap",
            defaults={"payload": json.loads(json.dumps(payload, default=str)), "computed_at": timezone.now()})
        print("\nSaved BacktestResult[rsi14_suppressed_largecap]", flush=True)
    except Exception as e:
        print("DB save failed:", e, flush=True)


if __name__ == "__main__":
    main()
