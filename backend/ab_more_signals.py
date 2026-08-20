#!/usr/bin/env python3
"""ITERATION 2 bake-off: screen new short-horizon SETUPS for A & B against the flagship baselines.
Each selector builds candidate windows; the H4 engine's signals are masked to them and the pooled bounce
is reported at 3b (and 8b for A's hold) with the candidate base rate. Cheap first-pass screen (pooled avg/t);
promote anything with real edge to a time-stepped a_curve run.
  A group: A_plus (baseline gap-up) vs A_rs (relative-strength) / A_vol (volume-confirmed) / A_pead (earnings drift)
  B group: B_plus (baseline capitulation) vs B_climax (volume-climax capitulation)
Run: MSYS_NO_PATHCONV=1 docker exec rotation-backend-1 python -u /app/ab_more_signals.py [--fetch] [--b-limit N]"""
import os, sys, warnings
warnings.filterwarnings("ignore")
os.environ.setdefault("DJANGO_SETTINGS_MODULE", "rotation.settings")
import django
django.setup()
import numpy as np
from django.db import connection
with connection.cursor() as cur:
    cur.execute("SET max_parallel_workers_per_gather = 0")
import h4_on_signals_study as S
import h4_study as H
from intraday_data import get_4h

FETCH = "--fetch" in sys.argv
BL = int(sys.argv[sys.argv.index("--b-limit") + 1]) if "--b-limit" in sys.argv else None
A_GROUP = ["A_plus", "A_rs", "A_vol", "A_pead"]
B_GROUP = ["B_plus", "B_climax"]
EXITS = [k for (k, _b, _) in H.EXITS]
E3 = "3b" if "3b" in EXITS else EXITS[min(2, len(EXITS) - 1)]
E8 = "8b" if "8b" in EXITS else EXITS[-1]


def _t(a):
    a = np.asarray(a, float); a = a[np.isfinite(a)]
    if len(a) < 3:
        return None
    sd = a.std(ddof=1)
    return round(float(a.mean() / (sd / np.sqrt(len(a)))), 2) if sd > 0 else None


def _base_rate(frames, bars):
    pool = []
    for df in frames.values():
        c = df["Close"].values
        if len(c) > bars:
            r = (c[bars:] - c[:-bars]) / c[:-bars] * 100
            pool.extend([x for x in r if np.isfinite(x)])
    return float(np.mean(pool)) if pool else float("nan")


def pooled(selector):
    allowed, meta = S.candidate_windows(selector, b_limit=BL)
    per = {s: {} for s in H.SIGNALS}
    frames = {}
    got = 0
    for tk in sorted(allowed):
        df = get_4h(tk, 5, FETCH)
        if df is None or len(df) < 120:
            continue
        got += 1
        frames[tk] = df
        res = S.backtest_ticker_masked(df, allowed[tk])
        for s in H.SIGNALS:
            for k, v in res[s]["flat"].items():
                per[s].setdefault(k, []).extend(v)
    return meta, got, per, frames


def cell(v):
    a = np.asarray(v, float); a = a[np.isfinite(a)]
    if not len(a):
        return "—"
    return f"{len(a)}/{a.mean():+.2f}/{(a>0).mean()*100:.0f}/{_t(a)}"


def report(group, families, e_extra):
    print(f"\n{'='*96}\n== {group[0]} baseline vs {group[1:]}   (n/avg%/win%/t)  exits={E3}"
          + (f",{e_extra}" if e_extra else "") + f"   fetch={'on' if FETCH else 'off'}\n{'='*96}", flush=True)
    data = {}
    for sel in group:
        meta, got, per, frames = pooled(sel)
        data[sel] = (meta, got, per, frames)
        print(f"[{sel:9}] windows={meta.get('n_windows')} names={meta.get('n_names')} "
              f"names_with_4h={got} base3b={_base_rate(frames,3):+.3f}% base8b={_base_rate(frames,8):+.3f}%",
              flush=True)
    sigs = [s for s in H.SIGNALS if H.SIGNALS[s]["family"] in families]
    hdr = f"\n  {'signal':16}" + "".join(f"{sel:>24}" for sel in group)
    print(hdr, flush=True)
    print(f"  --- {E3} ---", flush=True)
    for s in sigs:
        print(f"  {s:16}" + "".join(f"{cell(data[sel][2][s].get(E3, [])):>24}" for sel in group), flush=True)
    if e_extra:
        print(f"  --- {e_extra} ---", flush=True)
        for s in sigs:
            print(f"  {s:16}" + "".join(f"{cell(data[sel][2][s].get(e_extra, [])):>24}" for sel in group), flush=True)


# A = momentum families (hold to 8b matters); B = mean-reversion families
MOM = {"momentum", "trend", "breakout"}
MR = {"mean_reversion", "reversal", "oversold"}
fams = {H.SIGNALS[s]["family"] for s in H.SIGNALS}
print(f"available families: {sorted(fams)}", flush=True)
report(A_GROUP, fams & MOM or fams, E8)      # A: show momentum signals at 3b + 8b
report(B_GROUP, fams & MR or fams, None)      # B: show mean-reversion signals at 3b
