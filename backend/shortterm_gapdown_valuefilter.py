#!/usr/bin/env python3
"""Can a BOOK-RATIO / quality gate fix the gap-down overnight book's −55% drawdown?
Hypothesis: the fat left tail = gap-downs on broken/expensive names that KEEP falling. Gate
entries on PIT fundamentals (low P/B = asset-backed value, positive equity, profitable) → should
bounce cleaner, cut the tail — fewer entries but better risk/reward. Shows entry-count effect too.

Base = liquid (>$100M/day) gap<-3% (the ~10-names/night book), buy close / sell next open.
Per filter: entries, names/night, gross bps, net CAGR@5/10bps, MAX DRAWDOWN, win%, worst nights.
P/B = as_traded_close * PIT shares_outstanding / PIT total_equity (latest report avail<=entry).
-> BacktestResult[shortterm_gapdown_valuefilter]. Cache /tmp/beat_candles.pkl.
"""
import os, json, warnings, pickle
warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
os.environ.setdefault("DJANGO_SETTINGS_MODULE", "rotation.settings")
import django
django.setup()
from pathlib import Path
from seq_fundamental_study import load_candles, load_financial_reports
import price_basis

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
    print("loading cache + reports...", flush=True)
    candles = pickle.loads(CACHE.read_bytes())
    tks = sorted(candles)
    reps = robust(lambda: load_financial_reports(tks)) or {}
    print(f"  {len(candles)} tickers, {len(reps)} with reports", flush=True)

    # collect gap-down events with PIT P/B + quality flags
    ev = []   # (date, overnight, pb, equity_pos, profitable)
    for tk, df in candles.items():
        if df is None or len(df) < 40:
            continue
        o = df["Open"].values.astype(float); c = df["Close"].values.astype(float)
        idx = df.index; n = len(c)
        dvol = (df["Close"] * df["Volume"]).rolling(20).median().values
        rep = reps.get(tk)
        rad = eqa = sha = nittm = None
        if rep is not None and len(rep):
            r2 = rep.dropna(subset=["avail_date"]).sort_values("avail_date").reset_index(drop=True)
            rad = pd.DatetimeIndex(pd.to_datetime(r2["avail_date"]))
            eqa = pd.to_numeric(r2.get("total_equity"), errors="coerce").to_numpy(dtype=float) if "total_equity" in r2 else None
            sha = pd.to_numeric(r2.get("shares_outstanding"), errors="coerce").to_numpy(dtype=float) if "shares_outstanding" in r2 else None
            ni = pd.to_numeric(r2.get("net_income"), errors="coerce").to_numpy(dtype=float) if "net_income" in r2 else None
            nittm = pd.Series(ni).rolling(4).sum().to_numpy() if ni is not None else None
        for t in range(20, n - 1):
            if not (np.isfinite(dvol[t]) and dvol[t] >= LIQ and c[t] > 5 and o[t] > 0 and o[t + 1] > 0 and c[t - 1] > 0):
                continue
            if o[t] / c[t - 1] - 1.0 >= GAP:
                continue
            on = o[t + 1] / c[t] - 1.0
            if abs(on) > 0.5:
                continue
            pb = np.nan; eq_pos = False; prof = False
            if rad is not None and len(rad):
                q = int(rad.searchsorted(idx[t], "right")) - 1
                if q >= 0:
                    eq = eqa[q] if eqa is not None else np.nan
                    sh = sha[q] if sha is not None else np.nan
                    if np.isfinite(eq) and eq > 0 and np.isfinite(sh) and sh > 0:
                        pb = c[t] * sh / eq; eq_pos = True
                    prof = bool(nittm is not None and np.isfinite(nittm[q]) and nittm[q] > 0)
            ev.append((idx[t], on, pb, eq_pos, prof))

    E_date = pd.DatetimeIndex([e[0] for e in ev])
    E_on = np.array([e[1] for e in ev]); E_pb = np.array([e[2] for e in ev])
    E_eqpos = np.array([e[3] for e in ev]); E_prof = np.array([e[4] for e in ev])
    span_yrs = (E_date.max() - E_date.min()).days / 365.25
    tot = len(ev)
    print(f"  total gap<-3% liquid entries: {tot} ; with valid P/B: {int(np.isfinite(E_pb).sum())} "
          f"({round(np.isfinite(E_pb).mean()*100)}%)\n", flush=True)

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
            cagr = ((1 + eq) ** (1 / span_yrs) - 1) * 100 if span_yrs > 0 else None
            ec = np.cumprod(1 + net); maxdd = float((ec / np.maximum.accumulate(ec) - 1).min() * 100)
            res[f"cagr_{cost}bps"] = round(cagr, 1) if cagr is not None else None
            res[f"maxdd_{cost}bps"] = round(maxdd, 1)
            if cost == 5:
                res["win%"] = round(float((net > 0).mean() * 100), 1)
        return res

    fin = np.isfinite(E_pb)
    FILTERS = {
        "all (baseline)":       np.ones(tot, bool),
        "P/B valid only":       fin,
        "P/B < 1":              fin & (E_pb < 1),
        "P/B < 2":              fin & (E_pb < 2),
        "P/B < 3":              fin & (E_pb < 3),
        "P/B > 3 (contrast)":   fin & (E_pb > 3),
        "equity > 0":           E_eqpos,
        "profitable(TTM)":      E_prof,
        "P/B<3 & profitable":   fin & (E_pb < 3) & E_prof,
        "P/B<2 & profitable":   fin & (E_pb < 2) & E_prof,
    }
    print(f"{'filter':22} {'entries':>7} {'nm/nt':>6} {'gross':>6} | {'CAGR@5':>7} {'DD@5':>7} {'win%':>5} | "
          f"{'CAGR@10':>8} {'DD@10':>7}", flush=True)
    out = {}
    for name, m in FILTERS.items():
        r = book(m)
        if r is None:
            continue
        out[name] = r
        print(f"{name:22} {r['entries']:>7} {r['names_per_night']:>6} {r['gross_bps']:>6} | "
              f"{str(r['cagr_5bps']):>7} {str(r['maxdd_5bps']):>7} {str(r['win%']):>5} | "
              f"{str(r['cagr_10bps']):>8} {str(r['maxdd_10bps']):>7}", flush=True)

    payload = {"computed_at": pd.Timestamp.utcnow().isoformat(), "base": f"liquid>$100M gap<{int(GAP*100)}%",
               "total_entries": tot, "results": out,
               "caveat": "P/B = as-traded close * PIT shares / PIT total_equity (latest report avail<=entry). "
                         "Nightly equal-weight net, flat spread cost. Filters reduce entries; watch CAGR vs MAXDD "
                         "tradeoff. Residual delisted survivorship but 1-night holds ~insensitive."}
    try:
        from core.models import BacktestResult
        from django.utils import timezone
        BacktestResult.objects.update_or_create(
            kind="shortterm_gapdown_valuefilter",
            defaults={"payload": json.loads(json.dumps(payload, default=str)), "computed_at": timezone.now()})
        print("\nSaved BacktestResult[shortterm_gapdown_valuefilter]", flush=True)
    except Exception as e:
        print("DB save failed:", e, flush=True)


if __name__ == "__main__":
    main()
