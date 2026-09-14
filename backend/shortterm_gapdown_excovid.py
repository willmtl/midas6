#!/usr/bin/env python3
"""DIAGNOSTIC: how much of the gap-down overnight book's ~-50% drawdown is COVID (2020) alone?
Re-run the best books (liquid >$100M, gap<-2/-3/-5%, +skipcrash) over: FULL sample / ex-COVID /
ex-COVID&2025-tariff-crash. NOTE: excluding calendar windows is LOOK-AHEAD (not tradeable) — this
only isolates whether the tail is COVID-specific or recurring. The skip-crash gate is the real
tradeable crash defense.
-> BacktestResult[shortterm_gapdown_excovid]. Cache /tmp/beat_candles.pkl.
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
COVID = (pd.Timestamp("2020-02-15"), pd.Timestamp("2020-06-30"))
TARIFF = (pd.Timestamp("2025-03-25"), pd.Timestamp("2025-05-15"))


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
    mkt = robust(lambda: load_candles(["SPY"])) or {}
    spy_ret = mkt["SPY"]["Close"].pct_change()
    print(f"{len(candles)} tickers", flush=True)

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
            gap = o[t] / c[t - 1] - 1.0
            if gap >= -0.02:
                continue
            on = o[t + 1] / c[t] - 1.0
            if abs(on) > 0.5:
                continue
            d = idx[t]; sr = spy_ret.asof(d)
            ev.append((d, on, gap, float(sr) if pd.notna(sr) else 0.0))
    E_date = pd.DatetimeIndex([e[0] for e in ev]); E_on = np.array([e[1] for e in ev])
    E_gap = np.array([e[2] for e in ev]); E_sr = np.array([e[3] for e in ev])
    print(f"liquid gap<-2% entries: {len(ev)}\n", flush=True)

    def in_win(dates, win):
        return (dates >= win[0]) & (dates <= win[1])

    def book(mask):
        idxs = np.where(mask)[0]
        if len(idxs) == 0:
            return None
        bybn = {}
        for i in idxs:
            bybn.setdefault(E_date[i], []).append(E_on[i])
        nights = pd.DatetimeIndex(sorted(bybn))
        gross = np.array([np.mean(bybn[d]) for d in nights])
        span = (nights.max() - nights.min()).days / 365.25
        net = gross - 5 / 1e4
        eq = float(np.prod(1 + net) - 1); cagr = ((1 + eq) ** (1 / span) - 1) * 100
        ec = np.cumprod(1 + net); maxdd = float((ec / np.maximum.accumulate(ec) - 1).min() * 100)
        net10 = gross - 10 / 1e4
        eq10 = float(np.prod(1 + net10) - 1); cagr10 = ((1 + eq10) ** (1 / span) - 1) * 100
        dd10 = float((np.cumprod(1 + net10) / np.maximum.accumulate(np.cumprod(1 + net10)) - 1).min() * 100)
        return {"nights": len(nights), "nm_nt": round(float(np.mean([len(bybn[d]) for d in nights])), 1),
                "cagr5": round(cagr, 1), "dd5": round(maxdd, 1), "calmar5": round(cagr / abs(maxdd), 2) if maxdd else None,
                "cagr10": round(cagr10, 1), "dd10": round(dd10, 1),
                "worst_bps": round(float(net.min()) * 1e4, 0)}

    skip = E_sr > -0.01
    excov = ~in_win(E_date, COVID)
    exboth = excov & ~in_win(E_date, TARIFF)
    periods = {"FULL": np.ones(len(ev), bool), "ex-COVID": excov, "ex-COVID&tariff": exboth}
    print(f"{'book / period':30} {'nights':>6} {'nm/nt':>6} | {'CAGR@5':>7} {'DD@5':>7} {'Calmar':>6} | "
          f"{'CAGR@10':>8} {'DD@10':>7} | {'worst':>6}", flush=True)
    out = {}
    for thr, gl in ((-0.02, "gap<-2%"), (-0.03, "gap<-3%"), (-0.05, "gap<-5%")):
        for pl, pm in periods.items():
            m = (E_gap < thr) & skip & pm
            r = book(m)
            if not r:
                continue
            out[f"{gl}|{pl}"] = r
            print(f"{gl+' +skip / '+pl:30} {r['nights']:>6} {r['nm_nt']:>6} | {r['cagr5']:>7} {r['dd5']:>7} "
                  f"{str(r['calmar5']):>6} | {r['cagr10']:>8} {r['dd10']:>7} | {r['worst_bps']:>6}", flush=True)
        print("", flush=True)

    payload = {"computed_at": pd.Timestamp.utcnow().isoformat(), "results": out,
               "covid_window": [str(COVID[0].date()), str(COVID[1].date())],
               "tariff_window": [str(TARIFF[0].date()), str(TARIFF[1].date())],
               "caveat": "Excluding calendar windows is LOOK-AHEAD, NOT tradeable — diagnostic only for how much "
                         "of the DD is COVID/tariff-specific vs recurring. skip-crash gate (SPY day>-1%) IS the "
                         "tradeable crash defense and is applied throughout. Liquid>$100M, buy close/sell open."}
    try:
        from core.models import BacktestResult
        from django.utils import timezone
        BacktestResult.objects.update_or_create(
            kind="shortterm_gapdown_excovid",
            defaults={"payload": json.loads(json.dumps(payload, default=str)), "computed_at": timezone.now()})
        print("Saved BacktestResult[shortterm_gapdown_excovid]", flush=True)
    except Exception as e:
        print("DB save failed:", e, flush=True)


if __name__ == "__main__":
    main()
