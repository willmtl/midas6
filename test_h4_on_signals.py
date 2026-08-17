#!/usr/bin/env python3
"""Unit tests for h4_on_signals_study. Run:
  docker exec rotation-backend-1 python -u /app/test_h4_on_signals.py            # all
  docker exec rotation-backend-1 python -u /app/test_h4_on_signals.py masknoop   # one"""
import sys
import numpy as np
import pandas as pd


def _frame(n=200, seed_shift=0):
    # deterministic pseudo-random-ish walk without Math.random/Date: sine+ramp mix
    t = np.arange(n) + seed_shift
    close = 100 * (1 + 0.06*np.sin(t/7.0) + 0.0008*t + 0.02*np.sin(t/2.0))
    idx = pd.date_range("2023-01-02 08:00", periods=n, freq="4h", tz="UTC")
    return pd.DataFrame({"Open": close, "High": close*1.003, "Low": close*0.997,
                         "Close": close, "Volume": np.full(n, 1000.0)}, index=idx)


def test_masknoop():
    """allowed_dates=None (and a set of ALL dates) must reproduce h4_study.backtest_ticker exactly."""
    import h4_study as H
    from h4_on_signals_study import backtest_ticker_masked
    df = _frame()
    base = H.backtest_ticker(df)                    # no dtrend -> flat/by_bucket populated
    m_none = backtest_ticker_masked(df, None)
    all_dates = {d.date() for d in df.index}
    m_all = backtest_ticker_masked(df, all_dates)
    for sig in H.SIGNALS:
        for exitk, rows in base[sig]["flat"].items():
            assert m_none[sig]["flat"][exitk] == rows, f"none mismatch {sig}/{exitk}"
            assert m_all[sig]["flat"][exitk] == rows, f"all mismatch {sig}/{exitk}"
    print("test_masknoop OK")


def test_maskempty():
    """An empty allowed set yields zero trades everywhere."""
    from h4_on_signals_study import backtest_ticker_masked
    df = _frame()
    m = backtest_ticker_masked(df, set())
    total = sum(len(v) for s in m.values() for v in s["flat"].values())
    assert total == 0
    print("test_maskempty OK")


def test_candwindows():
    import os, datetime as dt
    os.environ.setdefault("DJANGO_SETTINGS_MODULE", "rotation.settings")
    import django; django.setup()
    from h4_on_signals_study import candidate_windows
    # C: full (fast — reads saved rotation_history). B: capped universe for speed.
    for sel, kw in (("C", {}), ("B", {"b_limit": 40})):
        allowed, meta = candidate_windows(sel, **kw)
        assert meta["n_names"] > 0 and meta["n_windows"] > 0, f"{sel} empty"
        tk = next(iter(allowed))
        assert isinstance(next(iter(allowed[tk])), dt.date), f"{sel} not date"
    print("test_candwindows OK")


def test_cwc():
    """C-only diagnostic with step timing (C is the fast selector; isolates load_candles cost)."""
    import os, time
    os.environ.setdefault("DJANGO_SETTINGS_MODULE", "rotation.settings")
    import django; django.setup()
    print("django ready", flush=True)
    from h4_on_signals_study import candidate_windows
    t = time.time()
    allowed, meta = candidate_windows("C")
    print(f"C done in {time.time()-t:.1f}s: {meta}", flush=True)
    assert meta["n_names"] > 0
    print("test_cwc OK")


CHECKS = {"masknoop": test_masknoop, "maskempty": test_maskempty,
          "candwindows": test_candwindows, "cwc": test_cwc}


def main():
    names = [a for a in sys.argv[1:] if a in CHECKS] or list(CHECKS)
    for nm in names:
        CHECKS[nm]()
    print(f"\n{len(names)} check(s) passed.")


if __name__ == "__main__":
    main()
