#!/usr/bin/env python3
"""SCREEN: short-interest as forward-return signal (.data/short_interest.jsonl, bi-monthly 2017-12+, multi-regime).
Classic anomaly: HIGH short interest -> LOW forward returns (informed shorts). Long-only usable end = long LOW-SI
or EXCLUDE high-SI. Signals: days_to_cover (given, = SI/ADV crowding) and si_pct_float (SI/shares_out, PIT).

DISCIPLINE: PIT entry LAGGED PUB_LAG=10 trading days after settlement_date (FINRA disseminates ~8 bd later) so
NO look-ahead; $5M dvol + price floor; segment by CAP tier; base-rate subtract (EXCESS vs SPY same window);
winsorize fwd ±50%. SCREEN ONLY — a spread must still pass the time-stepped book + regime gate. Horizons 21/42/63d
(bi-monthly cadence). Persists BacktestResult[si_screen]+JSON.
Run: MSYS_NO_PATHCONV=1 docker exec rotation-backend-1 python -u /app/si_screen.py"""
import os, sys, json, math, bisect
sys.path.insert(0, "/app")
os.environ.setdefault("DJANGO_SETTINGS_MODULE", "rotation.settings")
import django; django.setup()
import numpy as np, pandas as pd
from pathlib import Path
from collections import defaultdict
from seq_fundamental_study import load_candles, load_financial_reports, MIN_BARS, _chunk
from signal_discovery import _universe, PRICE_FLOOR

SI_FILE = Path("/app/.data/short_interest.jsonl")
DVOL_FLOOR = 5e6
PUB_LAG = 10            # trading days: settlement -> public dissemination (conservative)
HORIZONS = [21, 42, 63]
WINSOR = 0.50


def cap_tier(dvol):
    m = dvol / 1e6
    return "mega>1B" if m >= 1000 else "large200M-1B" if m >= 200 else "mid20-200M" if m >= 20 else "small<20M"


def main():
    universe, _ = _universe()
    from core.models import Candle
    si = defaultdict(list)
    with SI_FILE.open() as f:
        for line in f:
            r = json.loads(line)
            if r["ticker"] in set(universe) and r.get("settlement_date"):
                si[r["ticker"]].append((pd.Timestamp(r["settlement_date"]), r.get("days_to_cover"),
                                        r.get("short_interest"), r.get("avg_daily_volume")))
    for tk in si:
        si[tk].sort()
    names = sorted(si)
    print(f"SI names ∩ universe: {len(names)}", flush=True)

    spy = pd.DataFrame(list(Candle.objects.filter(ticker="SPY").values_list("date", "close")), columns=["d", "c"])
    spy["d"] = pd.to_datetime(spy["d"]); spy_c = spy.set_index("d")["c"].astype(float).sort_index()

    def shares_ff(reps, tk, dates):
        r = reps.get(tk)
        if r is None or "shares_outstanding" not in r.columns:
            return None
        d = r[["avail_date", "shares_outstanding"]].dropna().sort_values("avail_date")
        if d.empty:
            return None
        s = pd.Series(d["shares_outstanding"].values, index=pd.to_datetime(d["avail_date"]))
        s = s[~s.index.duplicated(keep="last")].sort_index()
        return s.reindex(s.index.union(dates)).ffill().reindex(dates).values

    acc = {s: defaultdict(lambda: defaultdict(list)) for s in ("days_to_cover", "si_pct_float", "dtc_change")}
    done = 0
    for ch in _chunk(names, 40):
        candles = load_candles(ch); reps = load_financial_reports(ch)
        for tk, sdf in candles.items():
            if len(sdf) < MIN_BARS or tk not in si:
                continue
            dates = sdf.index; close = sdf["Close"].values; vol = sdf["Volume"].values; n = len(close)
            dvol20 = pd.Series(close * vol).rolling(20, min_periods=10).mean().values
            di = dates.values; spy_al = spy_c.reindex(dates).ffill().values
            sh = shares_ff(reps, tk, dates)
            obs = si[tk]
            for oi, (sd, dtc, sint, adv) in enumerate(obs):
                # PIT: settlement -> first trading day, then +PUB_LAG trading days
                j0 = int(np.searchsorted(di, np.datetime64(sd), side="left"))
                j = j0 + PUB_LAG
                if j >= n or close[j] <= PRICE_FLOOR or not (np.isfinite(dvol20[j]) and dvol20[j] >= DVOL_FLOOR):
                    continue
                cap = cap_tier(float(dvol20[j]))
                sigs = {}
                if dtc is not None and np.isfinite(dtc):
                    sigs["days_to_cover"] = float(dtc)
                if sint and sh is not None and np.isfinite(sh[j]) and sh[j] > 0:
                    sigs["si_pct_float"] = 100.0 * float(sint) / float(sh[j])
                if oi > 0 and dtc is not None and obs[oi - 1][1] is not None and np.isfinite(dtc) and np.isfinite(obs[oi - 1][1]):
                    sigs["dtc_change"] = float(dtc) - float(obs[oi - 1][1])
                if not sigs:
                    continue
                for h in HORIZONS:
                    k = j + h
                    if k >= n or not np.isfinite(close[k]) or close[j] <= 0:
                        continue
                    r = close[k] / close[j] - 1.0
                    sr = spy_al[k] / spy_al[j] - 1.0 if (spy_al[j] > 0 and np.isfinite(spy_al[k])) else 0.0
                    ex = float(np.clip(r - sr, -WINSOR, WINSOR))
                    for sname, sval in sigs.items():
                        acc[sname][cap][h].append((sval, ex))
        done += len(ch)
        if done % 200 < 40:
            print(f"  scanned {done}/{len(names)}", flush=True)

    def quintile_report(pairs):
        if len(pairs) < 250:
            return None
        arr = np.array(pairs, float); v, e = arr[:, 0], arr[:, 1]
        qe = np.unique(np.quantile(v, np.linspace(0, 1, 6)))
        if len(qe) < 6:
            return None
        qi = np.clip(np.searchsorted(qe, v, side="right") - 1, 0, 4)
        rows = []
        for q in range(5):
            m = qi == q
            if m.sum() < 30:
                rows.append(None); continue
            ee = e[m]
            rows.append(dict(n=int(m.sum()), sig_med=round(float(np.median(v[m])), 3),
                             ex_mean=round(float(ee.mean()) * 100, 3), win=round(float((ee > 0).mean() * 100), 1),
                             t=round(float(ee.mean() / (ee.std() / math.sqrt(len(ee)))), 2) if ee.std() > 0 else 0.0))
        spread = (rows[0]["ex_mean"] - rows[4]["ex_mean"]) if (rows[0] and rows[4]) else None
        return dict(quintiles=rows, spread_Q1_minus_Q5=spread)

    out = {}
    CAPS = ["mega>1B", "large200M-1B", "mid20-200M", "small<20M"]
    for sname in ("days_to_cover", "si_pct_float", "dtc_change"):
        print(f"\n########## {sname}  (excess-vs-SPY %, quintiles; Q1=lowest) ##########", flush=True)
        out[sname] = {}
        for cap in CAPS:
            for h in HORIZONS:
                rep = quintile_report(acc[sname][cap][h])
                if not rep:
                    continue
                out[sname].setdefault(cap, {})[h] = rep
                qs = rep["quintiles"]
                cells = " ".join(f"Q{i+1}:{(qs[i]['ex_mean'] if qs[i] else float('nan')):+.2f}(t{qs[i]['t'] if qs[i] else 0:+.1f})" for i in range(5))
                sp = rep["spread_Q1_minus_Q5"]
                sp_s = f"{sp:+.2f}" if sp is not None else "  n/a"
                print(f"  {cap:14} h{h:<3} n~{sum(q['n'] for q in qs if q):>6}  {cells}  | Q1-Q5={sp_s}", flush=True)

    payload = dict(computed_at=pd.Timestamp.utcnow().isoformat(), dvol_floor=DVOL_FLOOR, pub_lag=PUB_LAG,
                   horizons=HORIZONS, winsor=WINSOR, results=out,
                   caveat="Short-interest SCREEN (bi-monthly 2017-12+, multi-regime). Excess-vs-SPY fwd by signal "
                   "quintile within cap tier. PIT entry lagged 10td after settlement (dissemination). $5M floor, "
                   "winsor ±50%. Q1=lowest SI. SCREEN ONLY — needs book+regime gate.")
    Path("/app/.data/studies/si_screen.json").write_text(json.dumps(payload, default=str))
    try:
        from core.models import BacktestResult
        from django.utils import timezone
        BacktestResult.objects.update_or_create(kind="si_screen",
            defaults={"payload": json.loads(json.dumps(payload, default=str)), "computed_at": timezone.now()})
        print("\nSaved BacktestResult[si_screen]", flush=True)
    except Exception as e:
        print("DB save failed:", e, flush=True)


if __name__ == "__main__":
    main()
