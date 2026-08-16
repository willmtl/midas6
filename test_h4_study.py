#!/usr/bin/env python3
"""Unit tests for the pure H4 engine logic. Run in-container against the live bind mount:
  docker exec rotation-backend-1 python -u /app/test_h4_study.py            # all
  docker exec rotation-backend-1 python -u /app/test_h4_study.py resample   # one check
Also collected by pytest (test_* functions)."""
import sys
import numpy as np
import pandas as pd


def _idx(n, start="2023-01-02 08:00", freq="1h"):
    # start on a 4h wall-clock boundary (08:00 UTC) so 8 1h bars form exactly two 4h bins;
    # pandas resample bins on fixed boundaries, not from the first timestamp.
    return pd.date_range(start, periods=n, freq=freq, tz="UTC")


def test_resample():
    from intraday_data import resample_ohlc
    df = pd.DataFrame({
        "open":   [1, 2, 3, 4, 5, 6, 7, 8],
        "high":   [2, 3, 4, 5, 6, 7, 8, 9],
        "low":    [0, 1, 2, 3, 4, 5, 6, 7],
        "close":  [1.5, 2.5, 3.5, 4.5, 5.5, 6.5, 7.5, 8.5],
        "volume": [10, 10, 10, 10, 20, 20, 20, 20],
    }, index=_idx(8))
    out = resample_ohlc(df, 4, from_1h=True)
    assert list(out.columns) == ["Open", "High", "Low", "Close", "Volume"]
    assert len(out) == 2
    assert out.iloc[0]["Open"] == 1 and out.iloc[0]["High"] == 5 and out.iloc[0]["Low"] == 0
    assert out.iloc[0]["Close"] == 4.5 and out.iloc[0]["Volume"] == 40
    print("test_resample OK")


CHECKS = {"resample": test_resample}


def main():
    names = [a for a in sys.argv[1:] if a in CHECKS] or list(CHECKS)
    for nm in names:
        CHECKS[nm]()
    print(f"\n{len(names)} check(s) passed.")


if __name__ == "__main__":
    main()
