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


def test_gcore_reproduces_prototype():
    uni, tgts, P = _panels()
    r = G.sim(P, cost_bps=25.0)                          # prototype G-core @ 25bps: CAGR 29.5, DD -24.8, hit 69.5
    m = G.perf(r["ret"], P["spy_ret"], r["pos_win"])
    assert 27.0 <= m["cagr"] <= 32.0, f"CAGR off: {m['cagr']}"
    assert -27.0 <= m["maxdd"] <= -22.0, f"DD off: {m['maxdd']}"
    assert m["hit"] >= 66.0, f"hit rate off: {m['hit']}"
    assert 0.15 <= r["turnover"] <= 0.30, f"turnover off: {r['turnover']}"
    assert 55 <= r["avg_n"] <= 70, f"name count off: {r['avg_n']}"
    spy = G.perf(P["spy_ret"], P["spy_ret"])
    assert m["cagr"] > spy["cagr"] + 8, f"G-core does not clear SPY: {m['cagr']} vs {spy['cagr']}"


def test_walk_forward_gate():
    uni, tgts, P = _panels()
    wf = G.walk_forward(P, cost_bps=25.0)
    assert len(wf["subperiods"]) >= 5, f"too few subperiods: {len(wf['subperiods'])}"
    multi = [s for s in wf["subperiods"] if not s["label"].startswith("covid")]
    beats = sum(1 for s in multi if s["cagr"] > s["spy_cagr"])
    for s in wf["subperiods"]:
        print(f"    {s['label']:14} G CAGR {s['cagr']:+7.1f}%  SPY {s['spy_cagr']:+7.1f}%  DD {s['maxdd']:6.1f}%  hit {s['hit']:.0f}%", flush=True)
    assert beats >= 3, f"not robust: beats SPY in only {beats}/{len(multi)} windows"
    assert wf["verdict"] == "robust"


TESTS = [test_panels_pit_and_shapes, test_gcore_reproduces_prototype, test_walk_forward_gate]


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
