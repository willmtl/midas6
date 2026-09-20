#!/usr/bin/env python3
"""GROWTH-SLEEVE PICK RULE (user: ARKK/QTUM-type sleeves field no VALUE pick — can a DIFFERENT rule field a good
pick when they accelerate? reverse-engineer the criterion, THEN walk-forward-gate it so we don't relearn the
'descriptive-great / WF-fails' lesson [[sector-conditional-rule-refuted]]).

Setup: a GROWTH sleeve is 'in play' in month M when its ETF is among the top-10 accelerating sleeves (same accel =
pct_change(3)-pct_change(3).shift(3) and top_n=10 the flagship uses). Its US/CA, $5M-liquid constituents are the
candidate pool. For each (month, in-play growth sleeve) we pick ONE constituent by a candidate criterion and hold
1 month. Target = forward return, measured as EXCESS vs SPY and vs the sleeve equal-weight (does the RULE beat just
buying the whole sleeve / the index / skipping?).

Criteria tested (PIT, known at month-end): mom3/mom6/mom12 (price momentum, hi & lo), rs3/rs6 (constituent minus
sleeve-ETF return = relative strength, hi & lo), rev1 (1mo reversal = prior-month loser), rsi14 (hi & lo), dvol
(biggest/smallest name). PHASE 1 descriptive: full-sample mean excess per criterion. PHASE 2 walk-forward: pick the
BEST criterion on H1 (2016-2020) ONLY, report its H2 (2021-2026) out-of-sample excess — survives only if H2 excess
> 0 AND beats sleeve-EW OOS. Saves BacktestResult[growth_sleeve_pick].
Run: MSYS_NO_PATHCONV=1 docker exec rotation-backend-1 python -u /app/growth_sleeve_pick.py"""
import os, json, math
os.environ.setdefault("DJANGO_SETTINGS_MODULE", "rotation.settings")
import django; django.setup()
import numpy as np, pandas as pd
from collections import defaultdict
import config, sector_holdings
from seq_fundamental_study import load_candles

MIN_DVOL = 5e6; TOP_ACCEL = 10; SPLIT = "2021-01-01"
GROWTH = ['Semiconductors', 'Biotech', 'Genomics', 'Cybersecurity', 'AI & Robotics', 'Cloud Computing', 'Space',
          'Clean Energy', 'Solar', 'Fintech', 'Electric Vehicles', 'Internet', 'Software', 'Lithium & Battery',
          'Nanotechnology', 'Cannabis', 'Hydrogen', 'Psychedelics', 'E-Commerce', 'Social Media',
          'Gaming & Esports', 'Uranium', 'Digital Infrastructure']


def is_usca(tk):
    return ("." not in tk) or tk.rsplit(".", 1)[1] in ("TO", "V")


def rsi(c, n=14):
    d = c.diff(); up = d.clip(lower=0).ewm(alpha=1 / n, adjust=False).mean(); dn = (-d.clip(upper=0)).ewm(alpha=1 / n, adjust=False).mean()
    return 100 - 100 / (1 + up / dn.replace(0, np.nan))


def main():
    growth = [g for g in GROWTH if g in config.SECTOR_ETFS]
    # ---- sleeve ETF accel panel (ALL sleeves, to rank top-10 accel like the flagship) ----
    all_etfs = {n: e for n, e in config.SECTOR_ETFS.items()}
    etf_daily = load_candles(sorted(set(all_etfs.values())) + ["SPY"])
    def mclose(d):
        return d["Close"].resample("ME").last()
    etf_m = pd.DataFrame({n: mclose(etf_daily[e]) for n, e in all_etfs.items() if e in etf_daily and etf_daily[e] is not None})
    accel = etf_m.pct_change(3) - etf_m.pct_change(3).shift(3)
    midx = etf_m.index[etf_m.index >= "2016-01-01"]
    spy_m = etf_daily["SPY"]["Close"].resample("ME").last().reindex(midx)
    spy_fwd = spy_m.shift(-1) / spy_m - 1.0
    # per-month top-10 accelerating sleeves
    topaccel = {}
    for d in midx:
        row = accel.loc[d].dropna() if d in accel.index else pd.Series(dtype=float)
        topaccel[d] = set(row.nlargest(TOP_ACCEL).index) if len(row) else set()

    # ---- constituents of growth sleeves (US/CA) ----
    cons = {}
    for g in growth:
        cons[g] = [t for t in sector_holdings.get_holdings(g) if is_usca(t)]
    allc = sorted({t for v in cons.values() for t in v})
    cand = {}
    for i in range(0, len(allc), 40):
        cand.update(load_candles(allc[i:i + 40]))
    close_m, dvol_m, rsi_m = {}, {}, {}
    for tk, df in cand.items():
        if df is None or df.empty:
            continue
        close_m[tk] = df["Close"].resample("ME").last().reindex(midx)
        dvol_m[tk] = (df["Close"] * df["Volume"]).resample("ME").mean().reindex(midx)
        rsi_m[tk] = rsi(df["Close"]).resample("ME").last().reindex(midx)
    close_m = pd.DataFrame(close_m); dvol_m = pd.DataFrame(dvol_m); rsi_m = pd.DataFrame(rsi_m)
    fret = close_m.shift(-1) / close_m - 1.0
    mom3 = close_m / close_m.shift(3) - 1; mom6 = close_m / close_m.shift(6) - 1; mom12 = close_m / close_m.shift(12) - 1
    rev1 = close_m / close_m.shift(1) - 1
    etf_ret3 = etf_m.pct_change(3)

    # ---- build event pool: (month, sleeve) where sleeve in top-10 accel ----
    # criteria -> function returning a Series over the pool (higher = picked); we test hi and lo
    def feat_panels():
        return {"mom3": mom3, "mom6": mom6, "mom12": mom12, "rev1": rev1, "rsi14": rsi_m}
    feats = feat_panels()

    events = []   # each: dict(date, sleeve, names, feat_vals per name, fret, sleeve_ew_fwd, spy_fwd)
    for d in midx[:-1]:
        for g in growth:
            e = config.SECTOR_ETFS[g]
            if g not in topaccel.get(d, set()):
                continue
            pool = [t for t in cons[g] if t in close_m.columns and pd.notna(fret.loc[d, t])
                    and (dvol_m.loc[d, t] if pd.notna(dvol_m.loc[d, t]) else 0) >= MIN_DVOL]
            if len(pool) < 3:
                continue
            sew = float(np.nanmean([fret.loc[d, t] for t in pool]))
            # relative strength vs sleeve ETF (3m)
            er3 = etf_ret3.loc[d, g] if g in etf_ret3.columns and pd.notna(etf_ret3.loc[d, g]) else np.nan
            rs3 = {t: (mom3.loc[d, t] - er3) if pd.notna(mom3.loc[d, t]) and pd.notna(er3) else np.nan for t in pool}
            fv = {"mom3": {t: mom3.loc[d, t] for t in pool}, "mom6": {t: mom6.loc[d, t] for t in pool},
                  "mom12": {t: mom12.loc[d, t] for t in pool}, "rev1": {t: rev1.loc[d, t] for t in pool},
                  "rsi14": {t: rsi_m.loc[d, t] for t in pool}, "rs3": rs3,
                  "dvol": {t: dvol_m.loc[d, t] for t in pool}}
            events.append({"date": d, "sleeve": g, "pool": pool, "fret": {t: float(fret.loc[d, t]) for t in pool},
                           "sew": sew, "spy": float(spy_fwd.loc[d]) if pd.notna(spy_fwd.loc[d]) else np.nan, "fv": fv})
    print(f"growth sleeves in play: {len(events)} (month,sleeve) events across {len({e['date'] for e in events})} months", flush=True)

    CRITERIA = []
    for f in ["mom3", "mom6", "mom12", "rev1", "rsi14", "rs3", "dvol"]:
        CRITERIA += [(f, "hi"), (f, "lo")]

    def pick(ev, feat, direction):
        vals = [(t, ev["fv"][feat].get(t)) for t in ev["pool"] if pd.notna(ev["fv"][feat].get(t))]
        if not vals:
            return None
        vals.sort(key=lambda x: x[1], reverse=(direction == "hi"))
        return vals[0][0]

    def evaluate(evs):
        out = {}
        for feat, dr in CRITERIA:
            exc_spy, exc_sew, picks = [], [], 0
            for ev in evs:
                t = pick(ev, feat, dr)
                if t is None or not pd.notna(ev["spy"]):
                    continue
                r = ev["fret"][t]; exc_spy.append(r - ev["spy"]); exc_sew.append(r - ev["sew"]); picks += 1
            if picks >= 10:
                out[f"{feat}_{dr}"] = {"n": picks, "exc_spy_pct": round(float(np.mean(exc_spy)) * 100, 3),
                                       "exc_sew_pct": round(float(np.mean(exc_sew)) * 100, 3),
                                       "beat_spy_pct": round(float(np.mean(np.array(exc_spy) > 0)) * 100, 1)}
        return out

    _sp = pd.Timestamp(SPLIT)
    h1 = [e for e in events if e["date"] < _sp]; h2 = [e for e in events if e["date"] >= _sp]
    # sleeve-EW and SPY baselines
    def base(evs):
        sew = [ev["sew"] - ev["spy"] for ev in evs if pd.notna(ev["spy"])]
        return {"sleeve_ew_exc_spy_pct": round(float(np.mean(sew)) * 100, 3), "n": len(sew)}
    res = {"params": {"top_accel": TOP_ACCEL, "split": SPLIT, "min_dvol": MIN_DVOL, "growth_sleeves": growth},
           "n_events": len(events), "baseline_full": base(events), "baseline_h1": base(h1), "baseline_h2": base(h2)}

    print(f"\n=== PHASE 1 DESCRIPTIVE (full sample) — pick-rule mean fwd EXCESS ===", flush=True)
    print(f"  sleeve-EW vs SPY: {res['baseline_full']['sleeve_ew_exc_spy_pct']:+.3f}%/mo (n={res['baseline_full']['n']})  <- the 'just buy the sleeve' benchmark", flush=True)
    full = evaluate(events); res["descriptive_full"] = full
    print(f"{'criterion':>12} {'n':>4} {'excSPY%':>9} {'excSleeve%':>11} {'beatSPY%':>9}", flush=True)
    for k, v in sorted(full.items(), key=lambda x: -x[1]["exc_spy_pct"]):
        print(f"{k:>12} {v['n']:>4} {v['exc_spy_pct']:>+9.3f} {v['exc_sew_pct']:>+11.3f} {v['beat_spy_pct']:>8.1f}%", flush=True)

    # PHASE 2 walk-forward: rank on H1, apply best to H2
    h1e = evaluate(h1); h2e = evaluate(h2)
    res["h1_insample"] = h1e; res["h2_oos"] = h2e
    ranked = sorted([k for k in h1e if h1e[k]["n"] >= 10], key=lambda k: -h1e[k]["exc_sew_pct"])  # beat-the-sleeve in-sample
    print(f"\n=== PHASE 2 WALK-FORWARD (select on H1 2016-2020 by excess-vs-sleeve, test H2 2021-2026) ===", flush=True)
    print(f"  H1 sleeve-EW vsSPY {res['baseline_h1']['sleeve_ew_exc_spy_pct']:+.3f}% | H2 sleeve-EW vsSPY {res['baseline_h2']['sleeve_ew_exc_spy_pct']:+.3f}%", flush=True)
    if ranked:
        best = ranked[0]; h1b = h1e[best]; h2b = h2e.get(best, {})
        res["wf_selected"] = best; res["wf_h1"] = h1b; res["wf_h2"] = h2b
        print(f"  BEST on H1: {best}  (H1 excSleeve {h1b['exc_sew_pct']:+.3f}%, excSPY {h1b['exc_spy_pct']:+.3f}%)", flush=True)
        if h2b:
            surv = h2b["exc_spy_pct"] > 0 and h2b["exc_sew_pct"] > 0
            res["wf_verdict"] = "SURVIVES" if surv else "FAILS"
            print(f"  OOS H2: excSPY {h2b['exc_spy_pct']:+.3f}%  excSleeve {h2b['exc_sew_pct']:+.3f}%  beatSPY {h2b['beat_spy_pct']:.1f}%  -> {res['wf_verdict']}", flush=True)
        else:
            res["wf_verdict"] = "NO_H2_DATA"; print("  OOS H2: insufficient events", flush=True)
        print(f"\n  top-5 H1 criteria and their OOS H2 (consistency check):", flush=True)
        for k in ranked[:5]:
            h2k = h2e.get(k, {})
            print(f"    {k:>10}  H1 excSleeve {h1e[k]['exc_sew_pct']:+.3f}%  ->  H2 excSleeve {h2k.get('exc_sew_pct','NA')}  excSPY {h2k.get('exc_spy_pct','NA')}", flush=True)

    open("/app/.data/studies/growth_sleeve_pick.json", "w").write(json.dumps(res, indent=2, default=float))
    try:
        from core.models import BacktestResult
        from django.utils import timezone
        BacktestResult.objects.update_or_create(kind="growth_sleeve_pick", defaults={"payload": res, "computed_at": timezone.now()})
        print("\nSaved BacktestResult[growth_sleeve_pick]", flush=True)
    except Exception as e:
        print("save skipped:", e, flush=True)


if __name__ == "__main__":
    main()
