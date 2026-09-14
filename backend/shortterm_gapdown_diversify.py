#!/usr/bin/env python3
"""Does DIVERSIFICATION (looser gap -> more names/night) cut the gap-down overnight book's
drawdown, with skip-crash-days protecting return? The regime run showed DD is a variance/names
problem (crisis nights = 47 names = only −21% DD), and 'skip SPY-down>1% days' lifts return &
cost-robustness. Combine: sweep gap threshold (looser=more names) x skip-crash gate, liquid-only.

Base universe = liquid (>$100M/day). Skip-crash = don't enter when SPY fell >1% that day (as-of
close, no look-ahead). Report entries, names/night, gross, CAGR@5/10, MAX DRAWDOWN, win%, worst night.
-> BacktestResult[shortterm_gapdown_diversify]. Cache /tmp/beat_candles.pkl.
"""
import os, json, warnings, pickle
warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
os.environ.setdefault("DJANGO_SETTINGS_MODULE", "rotation.settings")
import django
django.setup()
from pathlib import Path
from seq_fundamental_study import load_candles

CACHE = Path("/tmp/beat_candles.pkl")
LIQ = 100e6


def robust(fn, tries=6):
    from django.db import connection
    import time
    for a in range(tries):
        try:
            return fn()
        except Exception:
            connection.close(); time.sleep(4 * (a + 1))
    return None


def main():
    candles = pickle.loads(CACHE.read_bytes())
    mkt = robust(lambda: load_candles(["SPY", "^VIX"])) or {}
    spy = mkt["SPY"]["Close"]; vix = mkt["^VIX"]["Close"]
    spy_ret = spy.pct_change()
    print(f"{len(candles)} tickers", flush=True)

    ev = []   # (date, overnight, gap, spyret, vix)
    for tk, df in candles.items():
        if df is None or len(df) < 40:
            continue
        o = df["Open"].values.astype(float); c = df["Close"].values.astype(float)
        idx = df.index; n = len(c)
        dvol = (df["Close"] * df["Volume"]).rolling(20).median().values
        for t in range(20, n - 1):
            if not (np.isfinite(dvol[t]) and dvol[t] >= LIQ and c[t] > 5 and o[t] > 0 and o[t + 1] > 0 and c[t - 1] > 0):
                continue
            gap = o[t] / c[t - 1] - 1.0
            if gap >= -0.01:
                continue
            on = o[t + 1] / c[t] - 1.0
            if abs(on) > 0.5:
                continue
            d = idx[t]; sr = spy_ret.asof(d); vx = vix.asof(d)
            ev.append((d, on, gap, float(sr) if pd.notna(sr) else 0.0, float(vx) if pd.notna(vx) else np.nan))

    E_date = pd.DatetimeIndex([e[0] for e in ev])
    E_on = np.array([e[1] for e in ev]); E_gap = np.array([e[2] for e in ev])
    E_sr = np.array([e[3] for e in ev]); E_vix = np.array([e[4] for e in ev])
    span = (E_date.max() - E_date.min()).days / 365.25
    print(f"total liquid gap<-1% entries: {len(ev)}\n", flush=True)

    def book(mask):
        idxs = np.where(mask)[0]
        if len(idxs) == 0:
            return None
        bybn = {}
        for i in idxs:
            bybn.setdefault(E_date[i], []).append(E_on[i])
        nights = sorted(bybn)
        gross = np.array([np.mean(bybn[d]) for d in nights])
        npn = np.mean([len(bybn[d]) for d in nights])
        res = {"entries": int(len(idxs)), "nights": len(nights), "names_per_night": round(float(npn), 1),
               "gross_bps": round(float(gross.mean()) * 1e4, 1)}
        for cost in (5, 10):
            net = gross - cost / 1e4
            eq = float(np.prod(1 + net) - 1)
            cagr = ((1 + eq) ** (1 / span) - 1) * 100 if span > 0 else None
            ec = np.cumprod(1 + net); maxdd = float((ec / np.maximum.accumulate(ec) - 1).min() * 100)
            res[f"cagr_{cost}bps"] = round(cagr, 1) if cagr is not None else None
            res[f"maxdd_{cost}bps"] = round(maxdd, 1)
            if cost == 5:
                res["win%"] = round(float((net > 0).mean() * 100), 1)
                res["worst_night_bps"] = round(float(net.min()) * 1e4, 0)
                res["calmar@5"] = round(res["cagr_5bps"] / abs(maxdd), 2) if maxdd < 0 else None
        return res

    GAPS = [(-0.01, "gap<-1%"), (-0.015, "gap<-1.5%"), (-0.02, "gap<-2%"), (-0.03, "gap<-3%"), (-0.05, "gap<-5%")]
    nocrash = E_sr > -0.01
    print(f"{'book':28} {'entr':>6} {'nm/nt':>6} {'gross':>6} | {'CAGR@5':>7} {'DD@5':>7} {'Calmar':>6} {'win%':>5} | {'CAGR@10':>8} {'DD@10':>7}", flush=True)
    out = {}
    for thr, glabel in GAPS:
        for gate_on, gate_m, glabel2 in ((False, np.ones(len(ev), bool), ""), (True, nocrash, " +skipcrash")):
            m = (E_gap < thr) & gate_m
            r = book(m)
            if r is None:
                continue
            key = glabel + glabel2
            out[key] = r
            print(f"{key:28} {r['entries']:>6} {r['names_per_night']:>6} {r['gross_bps']:>6} | "
                  f"{str(r['cagr_5bps']):>7} {str(r['maxdd_5bps']):>7} {str(r['calmar@5']):>6} {str(r['win%']):>5} | "
                  f"{str(r['cagr_10bps']):>8} {str(r['maxdd_10bps']):>7}", flush=True)

    # best-of triple: looser gap + skipcrash + VIX<30 (avoid the very worst crisis contagion)
    print("\n=== triple: gap x skipcrash x VIX<30 ===", flush=True)
    for thr, glabel in [(-0.015, "gap<-1.5%"), (-0.02, "gap<-2%")]:
        m = (E_gap < thr) & (E_sr > -0.01) & np.isfinite(E_vix) & (E_vix < 30)
        r = book(m)
        if r:
            out[glabel + " +skipcrash +VIX<30"] = r
            print(f"  {glabel} +skipcrash +VIX<30: entr {r['entries']} nm/nt {r['names_per_night']} | "
                  f"CAGR@5 {r['cagr_5bps']}% DD@5 {r['maxdd_5bps']}% Calmar {r['calmar@5']} | "
                  f"CAGR@10 {r['cagr_10bps']}% DD@10 {r['maxdd_10bps']}%", flush=True)

    payload = {"computed_at": pd.Timestamp.utcnow().isoformat(), "total_entries": len(ev), "results": out,
               "caveat": "Liquid >$100M, buy close/sell open, nightly equal-weight net, flat spread cost. "
                         "Skipcrash=skip entries when SPY fell >1% that day (as-of close). Looser gap=more names/"
                         "night=more diversification. Calmar=CAGR/|maxDD|. 1-night holds ~survivorship-insensitive."}
    try:
        from core.models import BacktestResult
        from django.utils import timezone
        BacktestResult.objects.update_or_create(
            kind="shortterm_gapdown_diversify",
            defaults={"payload": json.loads(json.dumps(payload, default=str)), "computed_at": timezone.now()})
        print("\nSaved BacktestResult[shortterm_gapdown_diversify]", flush=True)
    except Exception as e:
        print("DB save failed:", e, flush=True)


if __name__ == "__main__":
    main()
