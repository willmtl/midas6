#!/usr/bin/env python3
"""Is the H4 dip-buy edge driven by VOLATILITY, not VALUE? — decisive test of "C is the wrong universe".

Hypothesis (user): removing ARKG/ARKK cost return because the H4 mean-reversion dip-buy pays on HIGH-VOL
names (bigger reversion amplitude), and C only caught them by accident (cheapest-P/B inside a crashed-growth
basket). If true, a volatility universe beats the C value basket for this trade.

Test: run the combined oversold dip-buy (rsi_os + newlow60 + ndown, 3-bar hold) across the BROAD cached-4h
universe (NOT C-gated), tag each entry with the name's point-in-time trailing volatility AND whether it was
in a C window at the time, then bucket 3b returns by vol quintile and by the vol x C-membership 2x2.
Answers: does the edge rise with vol? and is a high-vol non-C name as good as a high-vol C name?
-> BacktestResult[h4_vol_universe]. Run: MSYS_NO_PATHCONV=1 docker exec rotation-backend-1 python -u /app/h4_vol_universe.py
"""
import os, json, warnings
warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
import h4_study as H

SIGS = ["mr_rsi_os", "mr_newlow60", "mr_ndown"]
HOLD = 3
VOLWIN = 30            # trailing 4h bars for the point-in-time realized-vol measure


def _tstat(a):
    a = np.asarray(a, float); a = a[np.isfinite(a)]
    if len(a) < 3:
        return None
    sd = a.std(ddof=1)
    return round(float(a.mean() / (sd / np.sqrt(len(a)))), 2) if sd > 0 else None


def _agg(rets):
    r = np.asarray(rets, float)
    if len(r) == 0:
        return {"n": 0, "avg": None, "win": None, "t": None}
    return {"n": len(r), "avg": round(float(r.mean()), 3),
            "win": round(float((r > 0).mean() * 100), 1), "t": _tstat(r)}


def main():
    os.environ.setdefault("DJANGO_SETTINGS_MODULE", "rotation.settings")
    import django; django.setup()
    from pathlib import Path
    import intraday_data as ID
    from h4_on_signals_study import candidate_windows

    d4 = ID.DATA / "4h"
    universe = sorted(p.stem for p in d4.glob("*.parquet"))
    allowedC, _ = candidate_windows("C")        # {ticker: set(dates)} for the C-membership tag

    entries = []        # (ret3b, trailing_vol, in_C)
    n_names = 0
    for tk in universe:
        df = ID.get_4h(tk, 5, False)
        if df is None or len(df) < 120:
            continue
        n_names += 1
        close = df["Close"].values
        ts = df.index
        n = len(close)
        ret = np.zeros(n); ret[1:] = close[1:] / close[:-1] - 1
        # point-in-time trailing realized vol (std of last VOLWIN bar returns, annualized to daily-ish scale)
        vol = pd.Series(ret).rolling(VOLWIN).std().values * np.sqrt(2 * 252)  # ~2 bars/day
        cwin = allowedC.get(tk, set())
        fire = np.zeros(n, dtype=bool)
        for s in SIGS:
            e, _ = H.SIGNALS[s]["fn"](df); fire |= np.asarray(e, dtype=bool)
        idxs = sorted(H._episode_starts([i for i in range(n) if fire[i]], gap=H.GAP))
        for i in idxs:
            if i + HOLD >= n or i < VOLWIN or not np.isfinite(vol[i]) or close[i] <= 0:
                continue
            r3 = (close[i + HOLD] - close[i]) / close[i] * 100
            entries.append((r3, float(vol[i]), ts[i].date() in cwin))

    rets = np.array([e[0] for e in entries])
    vols = np.array([e[1] for e in entries])
    inC = np.array([e[2] for e in entries])
    # global vol quintile edges
    qs = np.quantile(vols, [0.2, 0.4, 0.6, 0.8])
    qlab = ["Q1 low-vol", "Q2", "Q3", "Q4", "Q5 high-vol"]
    def qof(v):
        return int(np.searchsorted(qs, v, side="right"))
    by_vol = {}
    for q in range(5):
        m = np.array([qof(v) == q for v in vols])
        by_vol[qlab[q]] = {"vol_range_pct": [round(float(vols[m].min()) * 100, 1), round(float(vols[m].max()) * 100, 1)] if m.any() else None,
                           **_agg(rets[m])}
    # 2x2: (low-vol = Q1-Q2, high-vol = Q4-Q5) x (in C / not C)
    hv = np.array([qof(v) >= 3 for v in vols])
    lv = np.array([qof(v) <= 1 for v in vols])
    grid = {
        "high_vol_in_C":  _agg(rets[hv & inC]),
        "high_vol_not_C": _agg(rets[hv & ~inC]),
        "low_vol_in_C":   _agg(rets[lv & inC]),
        "low_vol_not_C":  _agg(rets[lv & ~inC]),
    }
    overall_C = _agg(rets[inC]); overall_notC = _agg(rets[~inC])

    payload = {"computed_at": pd.Timestamp.utcnow().isoformat(),
               "n_names": n_names, "n_entries": len(entries), "hold_bars": HOLD,
               "by_vol_quintile": by_vol, "vol_x_membership": grid,
               "overall_in_C": overall_C, "overall_not_C": overall_notC,
               "note": ("H4 combined oversold dip-buy (rsi_os+newlow60+ndown, 3b) across the broad cached-4h "
                        "universe, NOT C-gated. Entries bucketed by point-in-time trailing realized vol (30-bar) "
                        "and by C-membership. Tests whether the edge is driven by VOLATILITY vs the C value "
                        "selector. Gross of fees; cached-4h universe only.")}
    Path("/app/.data/studies").mkdir(parents=True, exist_ok=True)
    Path("/app/.data/studies/h4_vol_universe.json").write_text(json.dumps(payload, indent=2, default=str))
    try:
        from core.models import BacktestResult
        from django.utils import timezone
        BacktestResult.objects.update_or_create(kind="h4_vol_universe",
            defaults={"payload": json.loads(json.dumps(payload, default=str)), "computed_at": timezone.now()})
        print("saved BacktestResult[h4_vol_universe]", flush=True)
    except Exception as e:
        print("DB save failed:", e, flush=True)
    print(f"\n=== H4 DIP-BUY by VOLATILITY ({n_names} names, {len(entries)} entries, 3b) ===", flush=True)
    print(f"  {'vol bucket':14}{'n':>7}{'avg3b':>9}{'win%':>7}{'t':>7}   vol range", flush=True)
    for k in qlab:
        v = by_vol[k]
        print(f"  {k:14}{v['n']:>7}{(v['avg'] or 0):>+9.3f}{(v['win'] or 0):>7}{(v['t'] or 0):>7}   {v['vol_range_pct']}", flush=True)
    print(f"\n=== VOL x C-MEMBERSHIP 2x2 (avg3b / n / t) ===", flush=True)
    for k, v in grid.items():
        print(f"  {k:16} avg {(v['avg'] or 0):>+.3f}%  win {v['win']}%  t {v['t']}  n {v['n']}", flush=True)
    print(f"\n  overall in-C {overall_C['avg']}% (n{overall_C['n']}) vs not-C {overall_notC['avg']}% (n{overall_notC['n']})", flush=True)


if __name__ == "__main__":
    main()
