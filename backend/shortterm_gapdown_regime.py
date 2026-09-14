#!/usr/bin/env python3
"""ADAPT THE GAP-DOWN OVERNIGHT BOOK TO MARKET CONDITIONS. The −55% drawdown is a REGIME event
(2020 COVID, 2025 tariff crash — market-wide, not name-quality; the P/B filter refuted names).
So gate/scale the book by VIX + market trend, measured at the entry day's CLOSE (no look-ahead).

Base = liquid (>$100M) gap<-3% overnight book (buy close / sell open).
 (A) VIX & trend GATES: only trade nights where the condition holds.
 (B) VIX-SCALED exposure: size = clip(target_vix / VIX, 0, 1) (adapt, don't gate — scalers beat
     cash-gates historically). Net nightly = size*(gross-cost); idle capital earns 0.
Report entries, names/night, gross, CAGR@5/10bps, MAX DRAWDOWN, win%.
-> BacktestResult[shortterm_gapdown_regime]. Cache /tmp/beat_candles.pkl.
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
LIQ, GAP = 100e6, -0.03


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
    spy200 = spy.rolling(200).mean(); spy_ret = spy.pct_change()
    print(f"{len(candles)} tickers; VIX {vix.index.min().date()}..{vix.index.max().date()}", flush=True)

    # collect liquid gap<-3% events + entry-day regime state (as-of close, no look-ahead)
    ev = []
    for tk, df in candles.items():
        if df is None or len(df) < 40:
            continue
        o = df["Open"].values.astype(float); c = df["Close"].values.astype(float)
        idx = df.index; n = len(c)
        dvol = (df["Close"] * df["Volume"]).rolling(20).median().values
        for t in range(20, n - 1):
            if not (np.isfinite(dvol[t]) and dvol[t] >= LIQ and c[t] > 5 and o[t] > 0 and o[t + 1] > 0 and c[t - 1] > 0):
                continue
            if o[t] / c[t - 1] - 1.0 >= GAP:
                continue
            on = o[t + 1] / c[t] - 1.0
            if abs(on) > 0.5:
                continue
            d = idx[t]
            vx = vix.asof(d); bull = bool(spy.asof(d) > spy200.asof(d)) if pd.notna(spy200.asof(d)) else True
            sr = spy_ret.asof(d)
            ev.append((d, on, float(vx) if pd.notna(vx) else np.nan, bull,
                       float(sr) if pd.notna(sr) else 0.0))

    E_date = pd.DatetimeIndex([e[0] for e in ev])
    E_on = np.array([e[1] for e in ev]); E_vix = np.array([e[2] for e in ev])
    E_bull = np.array([e[3] for e in ev]); E_spyret = np.array([e[4] for e in ev])
    span = (E_date.max() - E_date.min()).days / 365.25
    print(f"total gap<-3% liquid entries: {len(ev)}; median VIX at entry {np.nanmedian(E_vix):.1f}\n", flush=True)

    def book(mask, scale=None):
        idxs = np.where(mask)[0]
        if len(idxs) == 0:
            return None
        bybn = {}
        for i in idxs:
            bybn.setdefault(E_date[i], []).append((E_on[i], E_vix[i]))
        nights = sorted(bybn)
        gross = np.array([np.mean([x[0] for x in bybn[d]]) for d in nights])
        vixn = np.array([np.nanmean([x[1] for x in bybn[d]]) for d in nights])
        npn = np.mean([len(bybn[d]) for d in nights])
        res = {"entries": int(len(idxs)), "nights": len(nights), "names_per_night": round(float(npn), 1),
               "gross_bps": round(float(gross.mean()) * 1e4, 1)}
        for cost in (5, 10):
            net = gross - cost / 1e4
            if scale is not None:                       # VIX-scaled exposure
                exp = np.clip(scale / np.where(vixn > 0, vixn, np.nan), 0, 1)
                exp = np.nan_to_num(exp, nan=0.0)
                net = exp * net
            eq = float(np.prod(1 + net) - 1)
            cagr = ((1 + eq) ** (1 / span) - 1) * 100 if span > 0 else None
            ec = np.cumprod(1 + net); maxdd = float((ec / np.maximum.accumulate(ec) - 1).min() * 100)
            res[f"cagr_{cost}bps"] = round(cagr, 1) if cagr is not None else None
            res[f"maxdd_{cost}bps"] = round(maxdd, 1)
            if cost == 5:
                res["win%"] = round(float((net > 0).mean() * 100), 1)
        return res

    fin = np.isfinite(E_vix)
    GATES = {
        "all (baseline)":        np.ones(len(ev), bool),
        "VIX < 15":              fin & (E_vix < 15),
        "VIX < 20":              fin & (E_vix < 20),
        "VIX < 25":              fin & (E_vix < 25),
        "VIX < 30":              fin & (E_vix < 30),
        "VIX > 30 (crisis)":     fin & (E_vix > 30),
        "SPY > 200sma (bull)":   E_bull,
        "SPY day > -1%":         (E_spyret > -0.01),
        "bull & VIX<25":         E_bull & fin & (E_vix < 25),
        "bull & VIX<20":         E_bull & fin & (E_vix < 20),
        "bull & SPYday>-1%":     E_bull & (E_spyret > -0.01),
    }
    print(f"{'gate':22} {'entries':>7} {'nm/nt':>6} {'gross':>6} | {'CAGR@5':>7} {'DD@5':>7} {'win%':>5} | "
          f"{'CAGR@10':>8} {'DD@10':>7}", flush=True)
    out = {}
    for name, m in GATES.items():
        r = book(m)
        if r is None:
            print(f"{name:22} (no nights)", flush=True); continue
        out[name] = r
        print(f"{name:22} {r['entries']:>7} {r['names_per_night']:>6} {r['gross_bps']:>6} | "
              f"{str(r['cagr_5bps']):>7} {str(r['maxdd_5bps']):>7} {str(r['win%']):>5} | "
              f"{str(r['cagr_10bps']):>8} {str(r['maxdd_10bps']):>7}", flush=True)

    print("\n=== VIX-SCALED exposure (adapt, not gate) ===", flush=True)
    for tgt in (15, 18, 20):
        r = book(np.ones(len(ev), bool), scale=tgt)
        out[f"vix_scaled_t{tgt}"] = r
        print(f"  target VIX {tgt:>3}: gross {r['gross_bps']} | CAGR@5 {r['cagr_5bps']}% DD@5 {r['maxdd_5bps']}% "
              f"| CAGR@10 {r['cagr_10bps']}% DD@10 {r['maxdd_10bps']}%", flush=True)

    payload = {"computed_at": pd.Timestamp.utcnow().isoformat(), "total_entries": len(ev), "results": out,
               "caveat": "Regime measured at entry-day CLOSE (VIX close / SPY vs 200sma / SPY day ret) — known "
                         "when entering MOC, no look-ahead. Nightly equal-weight net, flat spread cost. VIX-scaled "
                         "sizes exposure = clip(target/VIX,0,1), idle capital=0. 1-night holds ~survivorship-insensitive."}
    try:
        from core.models import BacktestResult
        from django.utils import timezone
        BacktestResult.objects.update_or_create(
            kind="shortterm_gapdown_regime",
            defaults={"payload": json.loads(json.dumps(payload, default=str)), "computed_at": timezone.now()})
        print("\nSaved BacktestResult[shortterm_gapdown_regime]", flush=True)
    except Exception as e:
        print("DB save failed:", e, flush=True)


if __name__ == "__main__":
    main()
