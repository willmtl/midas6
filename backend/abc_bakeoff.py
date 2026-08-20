#!/usr/bin/env python3
"""A vs A_plus, B vs B_plus bake-off: run the H4 masked backtest over each selector's windows (shared 4h
cache) and print the pooled short-horizon bounce per H4 signal (n / avg% / win% / t) at the standard exits.
Run: MSYS_NO_PATHCONV=1 docker exec rotation-backend-1 python -u /app/abc_bakeoff.py [--b-limit N] [--no-fetch]"""
import os, sys, warnings
warnings.filterwarnings("ignore")
os.environ.setdefault("DJANGO_SETTINGS_MODULE", "rotation.settings")
import django
django.setup()
import numpy as np
import h4_on_signals_study as S
import h4_study as H

BL = int(sys.argv[sys.argv.index("--b-limit") + 1]) if "--b-limit" in sys.argv else None
FETCH = "--no-fetch" not in sys.argv


def _t(a):
    a = np.asarray(a, float); a = a[np.isfinite(a)]
    if len(a) < 3:
        return None
    sd = a.std(ddof=1)
    return round(float(a.mean() / (sd / np.sqrt(len(a)))), 2) if sd > 0 else None


def pooled(selector):
    """Pooled H4 returns per signal across the selector's candidate windows, at each standard exit."""
    from intraday_data import get_4h
    allowed, meta = S.candidate_windows(selector, b_limit=BL)
    per_sig = {s: {} for s in H.SIGNALS}          # sig -> exit_key -> [returns]
    got = 0
    for tk in sorted(allowed):
        df = get_4h(tk, 5, FETCH)
        if df is None or len(df) < 120:
            continue
        got += 1
        res = S.backtest_ticker_masked(df, allowed[tk])
        for s in H.SIGNALS:
            for k, v in res[s]["flat"].items():
                per_sig[s].setdefault(k, []).extend(v)
    return meta, got, per_sig


EXITS = [k for (k, _b, _) in H.EXITS]
print(f"exits={EXITS}  fetch={'on' if FETCH else 'off'}  b_limit={BL}\n", flush=True)
results = {}
for sel in ["A", "A_plus", "B", "B_plus"]:
    meta, got, per_sig = pooled(sel)
    results[sel] = (meta, got, per_sig)
    print(f"[{sel}] windows={meta.get('n_windows')} names={meta.get('n_names')} "
          f"dropped_fires={meta.get('dropped_fires','-')} names_with_4h={got}", flush=True)

# pick a mid exit to summarize (prefer a ~3-bar/6-bar horizon key if present, else the middle one)
key = EXITS[min(len(EXITS) - 1, 2)]
print(f"\n=== pooled bounce at exit '{key}'  (n / avg% / win% / t) ===", flush=True)
print(f"  {'signal':16}{'A':>22}{'A_plus':>22}", flush=True)
for s in H.SIGNALS:
    row = f"  {s:16}"
    for sel in ("A", "A_plus"):
        v = results[sel][2][s].get(key, [])
        a = np.asarray(v, float); a = a[np.isfinite(a)]
        cell = f"{len(a)}/{a.mean():+.2f}/{(a>0).mean()*100:.0f}/{_t(a)}" if len(a) else "—"
        row += f"{cell:>22}"
    print(row, flush=True)
print(f"  {'signal':16}{'B':>22}{'B_plus':>22}", flush=True)
for s in H.SIGNALS:
    row = f"  {s:16}"
    for sel in ("B", "B_plus"):
        v = results[sel][2][s].get(key, [])
        a = np.asarray(v, float); a = a[np.isfinite(a)]
        cell = f"{len(a)}/{a.mean():+.2f}/{(a>0).mean()*100:.0f}/{_t(a)}" if len(a) else "—"
        row += f"{cell:>22}"
    print(row, flush=True)
