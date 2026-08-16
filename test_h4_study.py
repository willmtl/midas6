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


def _ohlc_from_close(close, idx=None):
    close = np.asarray(close, float)
    n = len(close)
    idx = idx if idx is not None else _idx(n, freq="4h")
    return pd.DataFrame({
        "Open": close, "High": close * 1.001, "Low": close * 0.999,
        "Close": close, "Volume": np.full(n, 1000.0),
    }, index=idx)


def test_bucket_of():
    from h4_study import bucket_of
    b = [("a", -100, -3), ("b", -3, -2), ("c", -2, 0)]
    assert bucket_of(-5, b) == "a"
    assert bucket_of(-2.5, b) == "b"
    assert bucket_of(-1, b) == "c"
    assert bucket_of(5, b) is None
    print("test_bucket_of OK")


def test_sig_volshock_dn():
    from h4_study import sig_mr_volshock_dn
    close = np.concatenate([np.full(40, 100.0), [90.0], np.full(5, 90.0)])  # -10% shock at idx 40
    entry, mag = sig_mr_volshock_dn(_ohlc_from_close(close))
    assert entry[40] and mag[40] < -2
    assert not entry[:40].any()
    print("test_sig_volshock_dn OK")


def test_sig_ndown():
    from h4_study import sig_mr_ndown
    close = np.array([100, 99, 98, 97, 96, 97, 98], float)  # 4-down run ends at idx4; reversal bar = idx5
    entry, mag = sig_mr_ndown(_ohlc_from_close(close))
    assert entry[5] and mag[5] == 4          # enter the reversal bar, bucketed by completed run length
    assert not entry[4] and not entry[6]     # no entry mid-run and no look-ahead
    print("test_sig_ndown OK")


def test_sig_gap_dn():
    from h4_study import sig_mr_gap_dn
    df = _ohlc_from_close(np.full(30, 100.0))
    df.iloc[20, df.columns.get_loc("Open")] = 96.0   # -4% gap-down open at idx 20
    df.iloc[20, df.columns.get_loc("Close")] = 96.0
    entry, mag = sig_mr_gap_dn(df)
    assert entry[20] and mag[20] < -2
    print("test_sig_gap_dn OK")


CHECKS = {"resample": test_resample, "bucket": test_bucket_of,
          "volshock": test_sig_volshock_dn, "ndown": test_sig_ndown, "gap": test_sig_gap_dn}


def main():
    names = [a for a in sys.argv[1:] if a in CHECKS] or list(CHECKS)
    for nm in names:
        CHECKS[nm]()
    print(f"\n{len(names)} check(s) passed.")


if __name__ == "__main__":
    main()
