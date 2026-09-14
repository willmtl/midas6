#!/usr/bin/env python3
"""EXECUTION-TIMING STUDY — morning (open) vs evening (close) fills on the REAL flagship rebalance history.

The flagship backtest strikes every trade at the MONTH-END CLOSE (survivorship_smallcap_study.py resample('ME').last()).
The user asks: given the signal fires at that close, is it better to buy in the EVENING (market-on-close, signal day)
or the MORNING (next session's open)? And is there any executable intraday edge at all?

Method (no look-ahead, FX-cancelling): take the canonical blotter flagship_history.json — every month's held names
with buy date `date`, sell date `ndate`, weight, and the engine's close->close USD return `ret`. Overlay ONLY the
intraday fill RATIO from native daily OHLC (open vs close, same ticker, adjacent days -> FX + adjustment factor cancel):

    gross_cc   = 1 + ret                                   # engine, EVENING entry / EVENING exit  (baseline)
    entry_fac  = close(date) / fill_entry_price            # buy cheaper -> >1 -> more return
    exit_fac   = fill_exit_price / close(ndate)            # sell dearer -> >1 -> more return
    gross_alt  = gross_cc * entry_fac * exit_fac

Conventions:
  EVENING            : buy close(date),      sell close(ndate)        == engine baseline
  MORNING            : buy open(next mkt day after date), sell close(ndate)   # earliest executable, entry-only
  MORNING_OO         : buy next-open,        sell next-open after ndate       # morning both ends
  EXIT_NEXTOPEN      : buy close(date),      sell next-open after ndate       # exit-timing only
  SAMEDAY_OPEN*      : buy open(date),       sell close(ndate)   # *LOOK-AHEAD (signal not known at open) — diagnostic only

Monthly return = weighted mean across that month's picks (matches engine rr/wsum). Chain -> equity curve.
Saves BacktestResult[exec_timing] + /app/.data/studies/exec_timing.json.  Run:
  MSYS_NO_PATHCONV=1 docker exec rotation-backend-1 python -u /app/exec_timing_study.py
"""
import os, json, math, datetime as dt
os.environ.setdefault("DJANGO_SETTINGS_MODULE", "rotation.settings")
import django; django.setup()
import numpy as np, pandas as pd
from seq_fundamental_study import load_candles

TRACE = "/app/.data/studies/flagship_history.json"
OUT = "/app/.data/studies/exec_timing.json"


def _cap_bucket(mc):
    if mc is None or not np.isfinite(mc): return "unknown"
    if mc < 5e8: return "micro(<0.5B)"
    if mc < 2e9: return "small(<2B)"
    return "large(>=2B)"


def main():
    d = json.load(open(TRACE))
    months = d["months"]
    print(f"trace: {len(months)} months, computed_at={d.get('computed_at')}, arm={d.get('arm')}", flush=True)

    # ---- collect trades ----
    trades, tickers = [], set()
    for m in months:
        bd, sd = m.get("date"), m.get("ndate")
        if not sd:
            continue  # live month, no forward return
        for pk in m.get("picks") or []:
            tk, w, ret = pk.get("ticker"), pk.get("weight") or 1.0, pk.get("ret")
            if not tk or ret is None or not np.isfinite(ret):
                continue
            trades.append({"tk": tk, "bd": bd, "sd": sd, "w": float(w), "cc": float(ret),
                           "cap": _cap_bucket(pk.get("mktcap_usd")), "delisted": bool(pk.get("delisted"))})
            tickers.add(tk)
    print(f"{len(trades)} trades across {len(tickers)} tickers", flush=True)

    # ---- load daily OHLC (batched + parallel-gather off to dodge the db /dev/shm=64MB limit on the hypertable) ----
    from django.db import connection
    with connection.cursor() as cur:
        cur.execute("SET max_parallel_workers_per_gather = 0")
    cand, tks = {}, sorted(tickers)
    for i in range(0, len(tks), 40):
        cand.update(load_candles(tks[i:i + 40]))
    px = {}
    for tk, df in cand.items():
        if df is None or df.empty:
            continue
        s = df[["Open", "Close"]].copy()
        s.index = pd.to_datetime(s.index).normalize()
        s = s[~s.index.duplicated(keep="last")].sort_index()
        px[tk] = s

    def close_on(tk, d0):
        s = px.get(tk)
        if s is None: return None
        d0 = pd.Timestamp(d0).normalize()
        if d0 in s.index:
            v = s.at[d0, "Close"]; return float(v) if pd.notna(v) and v > 0 else None
        return None

    def open_on(tk, d0):
        s = px.get(tk)
        if s is None: return None
        d0 = pd.Timestamp(d0).normalize()
        if d0 in s.index:
            v = s.at[d0, "Open"]; return float(v) if pd.notna(v) and v > 0 else None
        return None

    def next_open(tk, d0):
        """Open of the first trading day STRICTLY AFTER d0."""
        s = px.get(tk)
        if s is None: return None
        d0 = pd.Timestamp(d0).normalize()
        fut = s.index[s.index > d0]
        if len(fut) == 0: return None
        v = s.at[fut[0], "Open"]; return float(v) if pd.notna(v) and v > 0 else None

    # ---- per-trade fill factors ----
    CONV = ["EVENING", "MORNING", "MORNING_OO", "EXIT_NEXTOPEN", "SAMEDAY_OPEN"]
    cov = {c: 0 for c in CONV}
    for t in trades:
        tk, bd, sd, g = t["tk"], t["bd"], t["sd"], 1.0 + t["cc"]
        c_bd, o_bd = close_on(tk, bd), open_on(tk, bd)
        no_bd = next_open(tk, bd)
        c_sd, no_sd = close_on(tk, sd), next_open(tk, sd)
        r = {}
        # EVENING baseline (always available = engine)
        r["EVENING"] = t["cc"]; cov["EVENING"] += 1
        # MORNING: entry next-open, exit close(sd)
        if c_bd and no_bd:
            r["MORNING"] = g * (c_bd / no_bd) - 1.0; cov["MORNING"] += 1
        else:
            r["MORNING"] = t["cc"]
        # MORNING_OO: entry next-open, exit next-open after sd
        if c_bd and no_bd and c_sd and no_sd:
            r["MORNING_OO"] = g * (c_bd / no_bd) * (no_sd / c_sd) - 1.0; cov["MORNING_OO"] += 1
        else:
            r["MORNING_OO"] = t["cc"]
        # EXIT_NEXTOPEN: entry close(bd), exit next-open after sd
        if c_sd and no_sd:
            r["EXIT_NEXTOPEN"] = g * (no_sd / c_sd) - 1.0; cov["EXIT_NEXTOPEN"] += 1
        else:
            r["EXIT_NEXTOPEN"] = t["cc"]
        # SAMEDAY_OPEN (look-ahead diagnostic): entry open(bd), exit close(sd)
        if c_bd and o_bd:
            r["SAMEDAY_OPEN"] = g * (c_bd / o_bd) - 1.0; cov["SAMEDAY_OPEN"] += 1
        else:
            r["SAMEDAY_OPEN"] = t["cc"]
        t["r"] = r

    print("per-trade fill coverage:", {c: f"{cov[c]}/{len(trades)}" for c in CONV}, flush=True)

    # ---- aggregate to monthly weighted returns, chain to equity ----
    by_month = {}
    for t in trades:
        by_month.setdefault(t["bd"], []).append(t)
    dates = sorted(by_month)

    def curve(conv):
        rets = []
        for d0 in dates:
            g = by_month[d0]
            wsum = sum(x["w"] for x in g)
            rr = sum(x["w"] * x["r"][conv] for x in g)
            rets.append(rr / wsum if wsum else 0.0)
        return np.array(rets)

    def stats(rets):
        eq = np.cumprod(1.0 + rets)
        total = eq[-1] - 1.0
        yrs = len(rets) / 12.0
        cagr = eq[-1] ** (1.0 / yrs) - 1.0 if eq[-1] > 0 else -1.0
        sh = (rets.mean() / rets.std() * math.sqrt(12)) if rets.std() > 0 else 0.0
        peak = np.maximum.accumulate(eq); dd = (eq / peak - 1.0).min()
        return {"total_ret": float(total), "cagr": float(cagr), "sharpe": float(sh),
                "maxdd": float(dd), "final_mult": float(eq[-1])}

    res = {c: stats(curve(c)) for c in CONV}
    base = curve("EVENING")

    # ---- paired per-trade edge vs EVENING, COVERED trades only (uncovered fall back to close -> diff=0, dilutes) ----
    def paired(conv):
        v = np.array([t["r"][conv] - t["r"]["EVENING"] for t in trades if abs(t["r"][conv] - t["r"]["EVENING"]) > 1e-12])
        if len(v) < 2:
            return {"n": len(v), "mean_bps": 0.0, "tstat": 0.0}
        return {"n": len(v), "mean_bps": float(v.mean() * 1e4),
                "tstat": float(v.mean() / (v.std() / math.sqrt(len(v))))}
    paired_stats = {c: paired(c) for c in ("MORNING", "MORNING_OO", "EXIT_NEXTOPEN", "SAMEDAY_OPEN")}

    # entry-timing per-trade edge over ALL trades (zeros included) + cap segmentation
    diffs = np.array([t["r"]["MORNING"] - t["r"]["EVENING"] for t in trades])
    tstat = float(diffs.mean() / (diffs.std() / math.sqrt(len(diffs)))) if diffs.std() > 0 else 0.0
    seg = {}
    for t in trades:
        seg.setdefault(t["cap"], []).append(t["r"]["MORNING"] - t["r"]["EVENING"])
    seg_stats = {k: {"n": len(v), "mean_bps": float(np.mean(v) * 1e4),
                     "tstat": float(np.mean(v) / (np.std(v) / math.sqrt(len(v)))) if np.std(v) > 0 and len(v) > 1 else 0.0}
                 for k, v in seg.items()}

    print("\n=== EXECUTION-TIMING RESULTS (flagship real rebalances, 2016-2026) ===", flush=True)
    print(f"{'convention':16s} {'final_x':>10s} {'CAGR':>8s} {'total':>10s} {'Sharpe':>7s} {'maxDD':>8s}", flush=True)
    for c in CONV:
        r = res[c]
        star = " *LOOK-AHEAD" if c == "SAMEDAY_OPEN" else ""
        print(f"{c:16s} {r['final_mult']:>10.2f} {r['cagr']*100:>7.1f}% {r['total_ret']*100:>9.0f}% "
              f"{r['sharpe']:>7.2f} {r['maxdd']*100:>7.1f}%{star}", flush=True)
    # ---- period-stability: split trades by buy-year into two halves, re-test entry + exit edges ----
    def half_paired(conv, lo, hi):
        v = np.array([t["r"][conv] - t["r"]["EVENING"] for t in trades
                      if lo <= int(t["bd"][:4]) <= hi and abs(t["r"][conv] - t["r"]["EVENING"]) > 1e-12])
        if len(v) < 2:
            return {"n": len(v), "mean_bps": 0.0, "tstat": 0.0}
        return {"n": len(v), "mean_bps": float(v.mean() * 1e4),
                "tstat": float(v.mean() / (v.std() / math.sqrt(len(v))))}
    halves = {"2016-2020": (2016, 2020), "2021-2026": (2021, 2026)}
    period_stab = {c: {k: half_paired(c, *rng) for k, rng in halves.items()} for c in ("MORNING", "EXIT_NEXTOPEN")}

    print("\npaired per-trade vs EVENING (covered fills only):", flush=True)
    for c in ("MORNING", "MORNING_OO", "EXIT_NEXTOPEN", "SAMEDAY_OPEN"):
        p = paired_stats[c]
        print(f"  {c:14s} n={p['n']:>4d}  mean={p['mean_bps']:+7.1f} bps  t={p['tstat']:+.2f}", flush=True)
    print(f"\nMORNING - EVENING per-trade (all 512, uncovered=0): mean={diffs.mean()*1e4:+.1f} bps  t={tstat:+.2f}", flush=True)
    print("period stability (both halves must agree for a real edge):", flush=True)
    for c in ("MORNING", "EXIT_NEXTOPEN"):
        for k in halves:
            s = period_stab[c][k]
            print(f"  {c:14s} {k}: n={s['n']:>4d}  mean={s['mean_bps']:+7.1f} bps  t={s['tstat']:+.2f}", flush=True)
    print("segmented by cap bucket (entry morning-evening):", flush=True)
    for k in sorted(seg_stats):
        s = seg_stats[k]
        print(f"  {k:14s} n={s['n']:>4d}  mean={s['mean_bps']:+7.1f} bps  t={s['tstat']:+.2f}", flush=True)

    verdict = ("EVENING (buy at signal-day close) is the anchor; "
               f"MORNING entry edge = {diffs.mean()*1e4:+.1f} bps/trade (t={tstat:+.2f}). ")
    if abs(tstat) < 2:
        verdict += "NOT statistically distinguishable -> no executable morning-vs-evening edge."
    else:
        verdict += ("MORNING better." if diffs.mean() > 0 else "EVENING better.")
    print("\nVERDICT:", verdict, flush=True)

    payload = {"computed_at": dt.datetime.now().isoformat(), "trace_computed_at": d.get("computed_at"),
               "n_trades": len(trades), "n_months": len(dates), "span": [dates[0], dates[-1]],
               "coverage": {c: cov[c] for c in CONV}, "results": res,
               "morning_minus_evening": {"mean_bps": float(diffs.mean() * 1e4), "tstat": tstat, "n": len(diffs)},
               "paired_vs_evening": paired_stats, "period_stability": period_stab,
               "segmented_cap": seg_stats, "verdict": verdict}
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    json.dump(payload, open(OUT, "w"), indent=2, default=float)
    print(f"\nwrote {OUT}", flush=True)

    # ---- persist BacktestResult ----
    try:
        from core.models import BacktestResult
        from django.utils import timezone
        BacktestResult.objects.create(kind="exec_timing_morning_vs_evening", payload=payload,
                                      computed_at=timezone.now())
        print("Saved BacktestResult[exec_timing_morning_vs_evening]", flush=True)
    except Exception as e:
        print("BacktestResult save skipped:", e, flush=True)


if __name__ == "__main__":
    main()
