#!/usr/bin/env python3
"""EVENT STUDY on the LuxAlgo SMC indicator series fetched from TradingView (tv/tv_smc_batch.js ->
/app/.data/tradingview/smc/*.json). Question: do SMC structural events (BOS / CHoCH / OB-breakout / FVG /
equal H-L, internal + swing) carry any FORWARD-RETURN edge on our stocks?

For each event type x timeframe (daily/weekly) we take every bar where the event fires and measure the
MARKET-ADJUSTED forward return (stock fwd return minus SPY same-window) at several horizons, over all names,
segmented by cap bucket. Bullish events should skew +, bearish -; a real edge = big |mean| with a real t-stat.
Forward returns use OUR split-adjusted daily closes (not TV's), so it plugs into the same return basis as the
flagship. Saves BacktestResult[smc_signal_study] + JSON.

Run: MSYS_NO_PATHCONV=1 docker exec rotation-backend-1 python -u /app/smc_signal_study.py"""
import os, json, math, glob
os.environ.setdefault("DJANGO_SETTINGS_MODULE", "rotation.settings")
import django; django.setup()
import numpy as np, pandas as pd
from collections import defaultdict
from core.models import Candle, Fundamental

SMC_DIR = "/app/.data/tradingview/smc"
OUT = "/app/.data/studies/smc_signal_study.json"
EVENTS = ["Bullish_BOS", "Bearish_BOS", "Bullish_CHoCH", "Bearish_CHoCH",
          "Internal_Bullish_BOS", "Internal_Bearish_BOS", "Internal_Bullish_CHoCH", "Internal_Bearish_CHoCH",
          "Bullish_Swing_OB_Breakout", "Bearish_Swing_OB_Breakout",
          "Bullish_Internal_OB_Breakout", "Bearish_Internal_OB_Breakout",
          "Equal_Highs", "Equal_Lows", "Bullish_FVG", "Bearish_FVG"]
H_DAILY = [5, 20]      # trading-day horizons for daily events
H_WEEKLY = [20, 60]    # trading-day horizons for weekly events


def nz(v):
    return v is not None and v == v and v != 0 and not (isinstance(v, (int, float)) and abs(v) > 1e50)


def main():
    files = sorted(glob.glob(SMC_DIR + "/*.json"))
    if not files:
        print("no SMC files in", SMC_DIR, "- run tv_smc_batch.js first", flush=True)
        return
    recs = []
    for f in files:
        try:
            recs.append(json.load(open(f)))
        except Exception:
            pass
    tickers = [r["ticker"] for r in recs]
    print(f"loaded {len(recs)} SMC files", flush=True)

    mcap = dict(Fundamental.objects.filter(ticker__in=tickers).values_list("ticker", "market_cap"))

    def cap_bucket(tk):
        mc = mcap.get(tk)
        if not mc or mc <= 0:
            return "unknown"
        return "micro(<2B)" if mc < 2e9 else "mid(2-10B)" if mc < 1e10 else "mega(>=10B)"

    # our split-adjusted daily closes for these names + SPY
    from django.db import connection
    with connection.cursor() as cur:
        cur.execute("SET max_parallel_workers_per_gather = 0")
    close = {}
    allt = tickers + ["SPY"]
    for i in range(0, len(allt), 200):
        rows = Candle.objects.filter(ticker__in=allt[i:i + 200], interval="1d", date__gte="2014-01-01").values_list("ticker", "date", "close")
        df = pd.DataFrame(list(rows), columns=["ticker", "date", "close"])
        if df.empty:
            continue
        df["date"] = pd.to_datetime(df["date"]); df["close"] = df["close"].astype(float)
        for tk, g in df.groupby("ticker", sort=False):
            close[tk] = g.sort_values("date").set_index("date")["close"]
    close = pd.DataFrame(close).sort_index()
    if "SPY" not in close.columns:
        print("no SPY candles", flush=True); return
    spy = close["SPY"]
    idx = close.index
    fwd = {h: close.shift(-h) / close - 1.0 for h in set(H_DAILY + H_WEEKLY)}
    spyfwd = {h: spy.shift(-h) / spy - 1.0 for h in set(H_DAILY + H_WEEKLY)}
    pos_of = {d: i for i, d in enumerate(idx)}

    # accumulate excess returns per (event, timeframe, horizon, cap)
    acc = defaultdict(list)

    def add_events(rec, tf, horizons):
        tk = rec["ticker"]
        if tk not in close.columns:
            return
        series = rec.get(tf) or []
        cb = cap_bucket(tk)
        for row in series:
            t = row.get("t")
            if t is None:
                continue
            d = pd.Timestamp(int(t), unit="s").normalize()
            # map to a trading day in our index (exact, else next available)
            if d in pos_of:
                dd = d
            else:
                p = idx.searchsorted(d)
                if p >= len(idx):
                    continue
                dd = idx[p]
            for ev in EVENTS:
                if nz(row.get(ev)):
                    for h in horizons:
                        fr = fwd[h].at[dd, tk] if (dd in fwd[h].index and tk in fwd[h].columns) else np.nan
                        sr = spyfwd[h].at[dd] if dd in spyfwd[h].index else np.nan
                        if pd.notna(fr) and pd.notna(sr):
                            acc[(ev, tf, h, cb)].append(float(fr - sr))
                            acc[(ev, tf, h, "ALL")].append(float(fr - sr))

    for rec in recs:
        add_events(rec, "daily", H_DAILY)
        add_events(rec, "weekly", H_WEEKLY)

    def summarize(vals):
        a = np.array(vals, dtype=float)
        a = a[np.isfinite(a)]
        if len(a) < 20:
            return {"n": int(len(a))}
        m = float(a.mean()); s = float(a.std(ddof=1))
        t = m / (s / math.sqrt(len(a))) if s > 0 else 0.0
        return {"n": int(len(a)), "mean_excess_pct": round(m * 100, 3), "hit_pct": round(float((a > 0).mean()) * 100, 1),
                "t_stat": round(t, 2), "median_pct": round(float(np.median(a)) * 100, 3)}

    res = {"n_files": len(recs), "events": EVENTS, "note": "market-adjusted (vs SPY) fwd returns after each SMC event; our split-adj closes; static cap buckets (caveat)",
           "by_event": {}}
    print("\n=== SMC event study — market-adjusted forward returns (ALL cap) ===", flush=True)
    print(f"{'event':>30} {'tf':>6} {'h(td)':>5} {'n':>6} {'meanExc%':>9} {'hit%':>6} {'t':>6}", flush=True)
    rows_sorted = []
    for ev in EVENTS:
        res["by_event"][ev] = {}
        for tf, hs in (("daily", H_DAILY), ("weekly", H_WEEKLY)):
            for h in hs:
                st = summarize(acc.get((ev, tf, h, "ALL"), []))
                res["by_event"][ev][f"{tf}_{h}"] = {"ALL": st, "by_cap": {cb: summarize(acc.get((ev, tf, h, cb), []))
                                                                          for cb in ["micro(<2B)", "mid(2-10B)", "mega(>=10B)"]}}
                if st.get("n", 0) >= 20:
                    rows_sorted.append((abs(st.get("t_stat", 0)), ev, tf, h, st))
    for _, ev, tf, h, st in sorted(rows_sorted, reverse=True):
        print(f"{ev:>30} {tf:>6} {h:>5} {st['n']:>6} {st['mean_excess_pct']:>+9.3f} {st['hit_pct']:>6.1f} {st['t_stat']:>+6.2f}", flush=True)

    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    open(OUT, "w").write(json.dumps(res, indent=2, default=float))
    print(f"\nwrote {OUT}", flush=True)
    try:
        from core.models import BacktestResult
        from django.utils import timezone
        BacktestResult.objects.update_or_create(kind="smc_signal_study",
            defaults={"payload": res, "computed_at": timezone.now()})
        print("Saved BacktestResult[smc_signal_study]", flush=True)
    except Exception as e:
        print("save skipped:", e, flush=True)


if __name__ == "__main__":
    main()
