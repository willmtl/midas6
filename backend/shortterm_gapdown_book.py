#!/usr/bin/env python3
"""VALIDATE the one short-term candidate found: gap-down OVERNIGHT bounce on liquid names.
(buy liquid large-cap at close on a day it gapped down >X%, sell next open). Unconditioned
overnight was too thin after cost; gap-down lifted it to +16.85 bps/night. Now stress it:

 (A) PERIOD STABILITY  — mean net overnight by YEAR and by bull/bear regime. Is it COVID-2020
     only? A real edge shows up in most years, not one.
 (B) THRESHOLD SWEEP   — gap < -1/-2/-3/-5%; monotone (deeper gap -> bigger pop) = real.
 (C) NIGHTLY BOOK      — each night equal-weight the liquid names that gapped down >2% that day,
     hold overnight, NET of cost (5 & 10 bps round-trip). Chain -> equity curve, annualized on
     nights-traded and on the full calendar. Worst nights + names/night.

Liquid = trailing-20d median $vol >= $20M (tiers A+B, spread ~4-10bps). Full-history daily cache.
-> BacktestResult[shortterm_gapdown_book] + JSON.
Run: MSYS_NO_PATHCONV=1 docker exec rotation-celery-worker-1 python -u /app/shortterm_gapdown_book.py
"""
import os, json, warnings, pickle
warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
os.environ.setdefault("DJANGO_SETTINGS_MODULE", "rotation.settings")
import django
django.setup()
from pathlib import Path
from seq_fundamental_study import load_candles

OUT = Path(__file__).resolve().parent / ".data" / "studies" / "shortterm_gapdown_book.json"
CACHE = Path("/tmp/beat_candles.pkl")
LIQ_MIN = 20e6          # tiers A+B
GAP = -0.02             # book trigger
COSTS = [5, 10]         # round-trip bps


def robust(fn, tries=6):
    from django.db import connection
    import time
    for a in range(tries):
        try:
            return fn()
        except Exception:
            connection.close(); time.sleep(4 * (a + 1))
    return None


def _stat(a):
    a = np.asarray(a, float)
    if len(a) == 0:
        return {"n": 0}
    t = a.mean() / (a.std(ddof=1) / np.sqrt(len(a))) if len(a) > 1 and a.std() > 0 else None
    return {"n": int(len(a)), "mean_bps": round(float(a.mean()) * 1e4, 2),
            "hit%": round(float((a > 0).mean() * 100), 1), "t": round(float(t), 1) if t is not None else None}


def main():
    print("loading cache...", flush=True)
    candles = pickle.loads(CACHE.read_bytes())
    spy = robust(lambda: load_candles(["SPY"]))["SPY"]["Close"]
    spy_bull = (spy > spy.rolling(200).mean())
    print(f"  {len(candles)} tickers", flush=True)

    # collect ALL liquid gap<-1% events as arrays so we can rebuild any (dvol,gap) book
    E_date, E_on, E_dvol, E_gap = [], [], [], []
    for tk, df in candles.items():
        if df is None or len(df) < 40:
            continue
        o = df["Open"].values.astype(float); c = df["Close"].values.astype(float)
        idx = df.index; n = len(c)
        dvol = (df["Close"] * df["Volume"]).rolling(20).median().values
        for t in range(20, n - 1):
            if not (np.isfinite(dvol[t]) and dvol[t] >= LIQ_MIN and c[t] > 5 and o[t] > 0 and o[t + 1] > 0 and c[t - 1] > 0):
                continue
            gap = o[t] / c[t - 1] - 1.0
            if gap >= -0.01:
                continue
            overnight = o[t + 1] / c[t] - 1.0
            if abs(overnight) > 0.5:
                continue
            E_date.append(idx[t]); E_on.append(overnight); E_dvol.append(dvol[t]); E_gap.append(gap)
    E_date = pd.DatetimeIndex(E_date); E_on = np.array(E_on); E_dvol = np.array(E_dvol); E_gap = np.array(E_gap)
    events = {thr: [(E_date[i], E_on[i]) for i in np.where(E_gap < thr)[0]] for thr in (-0.01, -0.02, -0.03, -0.05)}
    book_by_night = {}
    for i in np.where(E_gap < GAP)[0]:
        book_by_night.setdefault(E_date[i], []).append(E_on[i])

    # (B) threshold sweep
    print("\n=== (B) threshold sweep (gross overnight, liquid A+B) ===", flush=True)
    sweep = {}
    for thr in sorted(events):
        rets = [r for _, r in events[thr]]
        s = _stat(rets); sweep[f"gap<{int(thr*100)}%"] = s
        print(f"  gap<{int(thr*100):>3}%   n{s['n']:>7}  {s['mean_bps']:>7} bps  hit {s['hit%']}%  t{s['t']}", flush=True)

    # (A) period + regime for the -2% book trigger
    ev2 = events[-0.02]
    byyear = {}
    for d, r in ev2:
        byyear.setdefault(d.year, []).append(r)
    reg = {"bull": [], "bear": []}
    for d, r in ev2:
        reg["bull" if bool(spy_bull.asof(d)) else "bear"].append(r)
    print("\n=== (A) period stability (gap<-2% overnight, gross) ===", flush=True)
    year_res = {}
    for y in sorted(byyear):
        s = _stat(byyear[y]); year_res[str(y)] = s
        print(f"  {y}   n{s['n']:>6}  {s['mean_bps']:>7} bps  hit {s['hit%']}%  t{s['t']}", flush=True)
    regr = {k: _stat(v) for k, v in reg.items()}
    print(f"  BULL n{regr['bull']['n']} {regr['bull']['mean_bps']} bps t{regr['bull']['t']}  |  "
          f"BEAR n{regr['bear']['n']} {regr['bear']['mean_bps']} bps t{regr['bear']['t']}", flush=True)

    # (C) nightly equal-weight book, net of cost
    nights = sorted(book_by_night)
    gross_nightly = np.array([np.mean(book_by_night[d]) for d in nights])
    names_per_night = np.array([len(book_by_night[d]) for d in nights])
    print("\n=== (C) nightly equal-weight book (gap<-2%, liquid) ===", flush=True)
    print(f"  nights traded: {len(nights)}  |  avg names/night: {names_per_night.mean():.1f}  "
          f"|  median: {int(np.median(names_per_night))}", flush=True)
    span_days = (nights[-1] - nights[0]).days if len(nights) > 1 else 1
    book = {}
    for cost in COSTS:
        net = gross_nightly - cost / 1e4
        eq = float(np.prod(1 + net) - 1) * 100
        # calendar-annualized (compounding only on nights traded, capital idle otherwise)
        yrs = span_days / 365.25
        cagr = ((1 + eq / 100) ** (1 / yrs) - 1) * 100 if yrs > 0 else None
        winnights = float((net > 0).mean() * 100)
        worst = float(np.min(net) * 1e4)
        s = _stat(net)
        book[f"cost_{cost}bps"] = {"mean_net_bps": s["mean_bps"], "t": s["t"], "win_nights%": round(winnights, 1),
                                   "total_return%": round(eq, 1), "cagr%": round(cagr, 1) if cagr else None,
                                   "worst_night_bps": round(worst, 1), "nights": len(nights)}
        print(f"  @ {cost}bps cost: net {s['mean_bps']:>6} bps/night (t{s['t']})  win {winnights:.0f}%  "
              f"-> total {eq:.0f}% over {yrs:.1f}y = CAGR {cagr:.1f}%  (worst night {worst:.0f} bps)", flush=True)

    # (D) TIGHT books: tier x gap-depth x cost -> where does it clear cost with cushion?
    print("\n=== (D) tight books: tier x gap-depth (net CAGR at cost) ===", flush=True)
    span_yrs = (E_date.max() - E_date.min()).days / 365.25
    tight = {}
    print(f"{'book':22} {'nights':>6} {'nm/nt':>6} {'gross_bps':>9} | " + " ".join(f"CAGR@{c}b" for c in (5, 10, 15)), flush=True)
    for dvmin, dvlabel in ((20e6, "A+B>=20M"), (100e6, "A>=100M")):
        for gthr in (-0.02, -0.03, -0.05):
            mask = (E_dvol >= dvmin) & (E_gap < gthr)
            if mask.sum() == 0:
                continue
            bybn = {}
            for i in np.where(mask)[0]:
                bybn.setdefault(E_date[i], []).append(E_on[i])
            nts = sorted(bybn)
            gn = np.array([np.mean(bybn[d]) for d in nts])
            npn = np.mean([len(bybn[d]) for d in nts])
            cagrs = []
            row = {"nights": len(nts), "names_per_night": round(float(npn), 1),
                   "gross_bps": round(float(gn.mean()) * 1e4, 1)}
            for cost in (5, 10, 15):
                net = gn - cost / 1e4
                eq = float(np.prod(1 + net) - 1)
                cagr = ((1 + eq) ** (1 / span_yrs) - 1) * 100 if span_yrs > 0 else None
                row[f"cagr_{cost}bps"] = round(cagr, 1) if cagr is not None else None
                cagrs.append(cagr)
            label = f"{dvlabel} gap<{int(gthr*100)}%"
            tight[label] = row
            print(f"{label:22} {len(nts):>6} {npn:>6.1f} {row['gross_bps']:>9} | "
                  + " ".join(f"{c:>7.1f}" for c in cagrs), flush=True)

    verdict = (f"Gap-down overnight bounce (liquid, gap<-2%): "
               + ("period-STABLE (positive most years) " if sum(1 for y in year_res.values()
                    if y.get('mean_bps', 0) and y['mean_bps'] > 0) >= 0.6 * len(year_res) else "period-UNSTABLE (few years carry it) ")
               + f"| book @5bps CAGR {book['cost_5bps']['cagr%']}% / @10bps {book['cost_10bps']['cagr%']}%. "
               + "Long-only overnight, no leverage/shorting (fits return-priority). Capital idle intraday -> "
                 "pairing with an intraday use would raise capital efficiency. Costs are quoted-spread proxies; "
                 "MOC/MOO impact on a real basket must be measured live before sizing.")
    print("\n" + verdict, flush=True)

    payload = {"computed_at": pd.Timestamp.utcnow().isoformat(), "threshold_sweep": sweep,
               "by_year": year_res, "by_regime": regr, "book": book, "verdict": verdict,
               "params": {"liq_min_dvol": LIQ_MIN, "book_gap": GAP, "costs_bps": COSTS},
               "caveat": "Full-history daily cache; overnight=open[t+1]/close[t]-1 (adj cancels). Liquid=trailing "
                         "20d median $vol>=$20M. NET uses flat quoted-spread proxy (5/10bps); real MOC/MOO market "
                         "impact on a basket is NOT modeled and could be larger. Residual delisted survivorship "
                         "(current-listed universe) but 1-night holds are structurally survivorship-insensitive."}
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(payload, indent=2, default=str))
    try:
        from core.models import BacktestResult
        from django.utils import timezone
        BacktestResult.objects.update_or_create(
            kind="shortterm_gapdown_book",
            defaults={"payload": json.loads(json.dumps(payload, default=str)), "computed_at": timezone.now()})
        print("Saved BacktestResult[shortterm_gapdown_book]", flush=True)
    except Exception as e:
        print("DB save failed:", e, flush=True)


if __name__ == "__main__":
    main()
