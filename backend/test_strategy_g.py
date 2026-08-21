#!/usr/bin/env python3
"""Strategy G reproduction/invariant checks (plain-script harness — the container has no pytest).
Run all:            MSYS_NO_PATHCONV=1 docker exec rotation-backend-1 python -u /app/test_strategy_g.py
Run one:            ... python -u /app/test_strategy_g.py <test_name>
Panels are built once and shared across checks. Exit code 0 = all pass, 1 = a failure."""
import os, sys
os.environ.setdefault("DJANGO_SETTINGS_MODULE", "rotation.settings")
import django; django.setup()
import numpy as np, pandas as pd
import strategy_g as G

_P = None


def _panels():
    global _P
    if _P is None:
        uni, tgts = G.build_universe()
        _P = (uni, tgts, G.build_panels(uni, tgts))
    return _P


def test_panels_pit_and_shapes():
    uni, tgts, P = _panels()
    assert len(uni) > 3000, f"covered universe too small: {len(uni)}"
    assert "SPY" in P["mclose"].columns
    assert P["midx"][0].year <= 2016 and P["midx"][-1].year >= 2025
    assert P["ups"].notna().values.any()
    tk = "AAPL" if "AAPL" in P["ttm_ni"].columns else P["ttm_ni"].columns[0]
    d = P["midx"][60]
    r = G._reports[tk][["period_end", "avail_date", "net_income"]].dropna()
    r = r.sort_values("period_end"); r["ttm"] = r["net_income"].rolling(4).sum()
    avail = r[pd.to_datetime(r["avail_date"]) <= d]
    expected = avail["ttm"].iloc[-1] if len(avail) else np.nan
    got = P["ttm_ni"].loc[d, tk]
    assert (pd.isna(expected) and pd.isna(got)) or abs(got - expected) < 1e-6, f"PIT breach: {got} vs {expected}"


TESTS = [test_panels_pit_and_shapes]


def _run(fns):
    fails = 0
    for fn in fns:
        try:
            fn(); print(f"PASS {fn.__name__}", flush=True)
        except AssertionError as e:
            fails += 1; print(f"FAIL {fn.__name__}: {e}", flush=True)
        except Exception as e:
            fails += 1; print(f"ERROR {fn.__name__}: {type(e).__name__}: {e}", flush=True)
    print(f"\n{len(fns) - fails}/{len(fns)} passed", flush=True)
    return fails


if __name__ == "__main__":
    reg = {fn.__name__: fn for fn in TESTS}
    sel = [reg[a] for a in sys.argv[1:] if a in reg] or TESTS
    sys.exit(1 if _run(sel) else 0)
