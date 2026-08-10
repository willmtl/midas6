"""Standalone verification for sig_rsi_suppressed_then_cross.

Run: python test_rsi_suppressed.py
"""
import numpy as np
import pandas as pd
import ta

import studies


def _frame(closes):
    idx = pd.date_range("2020-01-01", periods=len(closes), freq="B")
    c = pd.Series(closes, index=idx, dtype=float)
    return pd.DataFrame({"Open": c, "High": c, "Low": c, "Close": c, "Volume": 1_000_000})


def _manual(df):
    """Independent re-implementation of the rule for cross-checking."""
    rsi = ta.momentum.rsi(df["Close"], window=10)
    sma = rsi.rolling(10).mean()
    out = pd.Series(False, index=df.index)
    for i in range(1, len(df)):
        crossed = rsi.iloc[i] > sma.iloc[i] and rsi.iloc[i - 1] <= sma.iloc[i - 1]
        if not crossed:
            continue
        # count consecutive suppressed days ending at i-1
        streak = 0
        j = i - 1
        while j >= 0:
            s = rsi.iloc[j] < 50 and rsi.iloc[j] < sma.iloc[j]
            if not (s == True):  # NaN or False breaks it
                break
            streak += 1
            j -= 1
        if streak >= studies.SUPPRESS_MIN:
            out.iloc[i] = True
    return out


def main():
    # A rounded top that rolls over into an accelerating decline: RSI eases down SMOOTHLY
    # (staying below its own SMA for a long unbroken run, and <50), then a sharp rally
    # forces RSI to cross back above its SMA. A monotonic crash pins RSI to 0 (== its SMA,
    # so not strictly below) and a choppy decline pops RSI above its SMA every few bars —
    # only a smooth accelerating decline yields the 10+ consecutive suppressed days.
    t = np.arange(40)
    down = list(100 + 6 * np.sin(t * 0.18) - 0.06 * t ** 2)
    p = down[-1]
    up = []
    for _ in range(18):
        p *= 1.04                                # sharp recovery -> RSI crosses its SMA
        up.append(p)
    df = _frame(down + up)

    sig = studies.sig_rsi_suppressed_then_cross(df)
    ref = _manual(df)

    assert sig.dtype == bool, f"dtype should be bool, got {sig.dtype}"
    assert not sig.isna().any(), "signal must not contain NaN"
    assert sig.equals(ref), "signal disagrees with the independent re-implementation"

    fires = list(df.index[sig])
    print(f"fire count: {sig.sum()}")
    print(f"fire dates: {[d.date().isoformat() for d in fires]}")
    assert sig.sum() >= 1, "expected at least one trigger on the rally cross"

    # On every fire bar: RSI just crossed above its SMA, and the 10 prior bars were suppressed.
    rsi = ta.momentum.rsi(df["Close"], window=10)
    sma = rsi.rolling(10).mean()
    for d in fires:
        i = df.index.get_loc(d)
        assert rsi.iloc[i] > sma.iloc[i] and rsi.iloc[i - 1] <= sma.iloc[i - 1], "not a cross bar"
        for k in range(i - studies.SUPPRESS_MIN, i):
            assert rsi.iloc[k] < 50 and rsi.iloc[k] < sma.iloc[k], f"bar {k} not suppressed"

    # Negative control: a pure uptrend never suppresses, so it must never fire.
    up_only = _frame(list(np.linspace(50, 150, 80)))
    assert studies.sig_rsi_suppressed_then_cross(up_only).sum() == 0, "should not fire in an uptrend"

    print("ALL CHECKS PASSED")


if __name__ == "__main__":
    main()
