#!/usr/bin/env python3
"""SCREEN: is there short-term alpha in OPTIONS POSITIONING (OptionSnapshot, 2022-09+)? Hypothesis: unusually
CALL-HEAVY option volume (low put/call vol ratio vs the name's own baseline) = informed bullish positioning ->
positive forward drift. Usable historical fields: pc_vol (98%), atm_iv (86%), iv_skew (74%). pc_oi/gex empty.

Discipline (per house rules): segment by CAP tier ($ADV), $5M dvol floor + price floor, PIT entry at the CLOSE
of the trading day AFTER the snapshot, base-rate subtract (EXCESS over SPY same window), winsorize fwd ±50%,
test STANDALONE, report the magnitude TAIL not just the mean. This is only a SCREEN — a positive result must
still pass a time-stepped net-of-cost book + regime attribution (the test that killed the insider signal).

Signals tested (each bucketed into quintiles within cap tier):
  pc_vol      : raw put/call volume ratio (LOW = call-heavy = bullish)
  pc_vol_z    : (pc_vol - trailing60 mean)/std  (a call-volume SPIKE vs the name's own norm)
  iv_skew     : raw skew
Forward horizons 5/10/20d. Persists BacktestResult[uoa_screen]+JSON.
Run: MSYS_NO_PATHCONV=1 docker exec rotation-backend-1 python -u /app/uoa_screen.py"""
import os, sys, json, math
sys.path.insert(0, "/app")
os.environ.setdefault("DJANGO_SETTINGS_MODULE", "rotation.settings")
import django; django.setup()
import numpy as np, pandas as pd
from pathlib import Path
from collections import defaultdict
from seq_fundamental_study import load_candles, MIN_BARS, _chunk
from signal_discovery import _universe, PRICE_FLOOR

DVOL_FLOOR = 5e6
HORIZONS = [5, 10, 20]
WINSOR = 0.50


def cap_tier(dvol):
    m = dvol / 1e6
    return "mega>1B" if m >= 1000 else "large200M-1B" if m >= 200 else "mid20-200M" if m >= 20 else "small<20M"


def main():
    universe, _ = _universe()
    from core.models import OptionSnapshot, Candle
    # pull snapshots
    print("loading OptionSnapshot...", flush=True)
    snap = pd.DataFrame(list(OptionSnapshot.objects.values_list("ticker", "date", "pc_vol", "atm_iv", "iv_skew")),
                        columns=["ticker", "date", "pc_vol", "atm_iv", "iv_skew"])
    snap["date"] = pd.to_datetime(snap["date"])
    names = sorted(set(snap["ticker"]) & set(universe))
    snap = snap[snap["ticker"].isin(names)]
    print(f"snapshots {len(snap)} over {len(names)} names ∩ universe", flush=True)

    spy = pd.DataFrame(list(Candle.objects.filter(ticker="SPY").values_list("date", "close")), columns=["d", "c"])
    spy["d"] = pd.to_datetime(spy["d"]); spy_c = spy.set_index("d")["c"].astype(float).sort_index()

    snap_by_tk = {tk: g.sort_values("date") for tk, g in snap.groupby("ticker")}

    # accumulate: sig -> cap -> horizon -> quintile -> list of excess fwd returns
    acc = {s: defaultdict(lambda: defaultdict(lambda: defaultdict(list))) for s in ("pc_vol", "pc_vol_z", "iv_skew")}
    done = 0
    for ch in _chunk(names, 40):
        candles = load_candles(ch)
        for tk, sdf in candles.items():
            if len(sdf) < MIN_BARS or tk not in snap_by_tk:
                continue
            dates = sdf.index; close = sdf["Close"].values; vol = sdf["Volume"].values; n = len(close)
            dvol20 = pd.Series(close * vol).rolling(20, min_periods=10).mean().values
            di = dates.values
            spy_al = spy_c.reindex(dates).ffill().values
            g = snap_by_tk[tk]
            pcv = g["pc_vol"].values.astype(float)
            pcv_mean = pd.Series(pcv).rolling(60, min_periods=20).mean().values
            pcv_std = pd.Series(pcv).rolling(60, min_periods=20).std().values
            for i, srow in enumerate(g.itertuples(index=False)):
                sd = np.datetime64(srow.date)
                # PIT: enter at close of the trading day AFTER the snapshot
                j = int(np.searchsorted(di, sd, side="right"))
                if j >= n:
                    continue
                if close[j] <= PRICE_FLOOR or not (np.isfinite(dvol20[j]) and dvol20[j] >= DVOL_FLOOR):
                    continue
                cap = cap_tier(float(dvol20[j]))
                sigs = {}
                if np.isfinite(srow.pc_vol) and srow.pc_vol > 0:
                    sigs["pc_vol"] = float(srow.pc_vol)
                if np.isfinite(pcv_mean[i]) and np.isfinite(pcv_std[i]) and pcv_std[i] > 0:
                    sigs["pc_vol_z"] = (pcv[i] - pcv_mean[i]) / pcv_std[i]
                if np.isfinite(srow.iv_skew):
                    sigs["iv_skew"] = float(srow.iv_skew)
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
                        acc[sname][cap][h][None].append((sval, ex))
        done += len(ch)
        if done % 200 < 40:
            print(f"  scanned {done}/{len(names)}", flush=True)

    def quintile_report(pairs):
        if len(pairs) < 250:
            return None
        arr = np.array(pairs, float)
        v, e = arr[:, 0], arr[:, 1]
        qe = np.quantile(v, np.linspace(0, 1, 6))
        qi = np.clip(np.searchsorted(qe, v, side="right") - 1, 0, 4)
        rows = []
        for q in range(5):
            m = qi == q
            if m.sum() < 30:
                rows.append(None); continue
            ee = e[m]
            rows.append(dict(n=int(m.sum()), sig_mean=round(float(v[m].mean()), 3),
                             ex_mean=round(float(ee.mean()) * 100, 3), ex_med=round(float(np.median(ee)) * 100, 3),
                             win=round(float((ee > 0).mean() * 100), 1),
                             t=round(float(ee.mean() / (ee.std() / math.sqrt(len(ee)))), 2) if ee.std() > 0 else 0.0))
        # Q1-Q5 spread on excess mean (informational; long-only can only use one end)
        if rows[0] and rows[4]:
            spread = rows[0]["ex_mean"] - rows[4]["ex_mean"]
        else:
            spread = None
        return dict(quintiles=rows, spread_Q1_minus_Q5=spread)

    out = {}
    CAPS = ["mega>1B", "large200M-1B", "mid20-200M", "small<20M"]
    for sname in ("pc_vol", "pc_vol_z", "iv_skew"):
        print(f"\n########## SIGNAL: {sname}  (excess-vs-SPY %, quintiles; Q1=lowest signal value) ##########", flush=True)
        out[sname] = {}
        for cap in CAPS:
            for h in HORIZONS:
                rep = quintile_report(acc[sname][cap][h][None])
                if not rep:
                    continue
                out[sname].setdefault(cap, {})[h] = rep
                qs = rep["quintiles"]
                cells = " ".join(f"Q{i+1}:{(qs[i]['ex_mean'] if qs[i] else float('nan')):+.2f}(w{qs[i]['win'] if qs[i] else 0:.0f},t{qs[i]['t'] if qs[i] else 0:+.1f})" for i in range(5))
                print(f"  {cap:14} h{h:<3} n~{sum(q['n'] for q in qs if q):>6}  {cells}  | Q1-Q5={rep['spread_Q1_minus_Q5']:+.2f}", flush=True)

    payload = dict(computed_at=pd.Timestamp.utcnow().isoformat(), dvol_floor=DVOL_FLOOR, horizons=HORIZONS,
                   winsor=WINSOR, results=out,
                   caveat="Options-positioning SCREEN (OptionSnapshot 2022-09+). Excess-vs-SPY fwd return by "
                   "signal quintile within cap tier. PIT entry at close of day AFTER snapshot, $5M floor, "
                   "winsorized ±50%. SCREEN ONLY — needs time-stepped book + regime attribution before trust.")
    Path("/app/.data/studies/uoa_screen.json").write_text(json.dumps(payload, default=str))
    try:
        from core.models import BacktestResult
        from django.utils import timezone
        BacktestResult.objects.update_or_create(kind="uoa_screen",
            defaults={"payload": json.loads(json.dumps(payload, default=str)), "computed_at": timezone.now()})
        print("\nSaved BacktestResult[uoa_screen]", flush=True)
    except Exception as e:
        print("DB save failed:", e, flush=True)


if __name__ == "__main__":
    main()
