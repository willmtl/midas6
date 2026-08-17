#!/usr/bin/env python3
"""Unit tests for h4_c_indicators. Run:
  docker exec rotation-backend-1 python -u /app/test_h4_c_indicators.py"""
import sys
import numpy as np
import pandas as pd


def _frame(n=120):
    t = np.arange(n)
    close = 100 * (1 + 0.05*np.sin(t/6.0) + 0.0005*t)
    idx = pd.date_range("2023-01-02 08:00", periods=n, freq="4h", tz="UTC")
    return pd.DataFrame({"Open": close, "High": close*1.004, "Low": close*0.996,
                         "Close": close, "Volume": np.full(n, 1e6)}, index=idx)


def test_combo_keys():
    from h4_c_indicators import combo_entries
    df = _frame()
    e = combo_entries(df)
    assert "gap_dn__rsi_x" in e, "headline combo missing"
    assert len(e) == 12, f"expected 12 combos, got {len(e)}"
    for k, v in e.items():
        assert len(v) == len(df) and v.dtype == bool, f"{k} bad shape/dtype"
    print("test_combo_keys OK")


def test_combo_seq_fires():
    """A gap-down at bar 40 then an RSI cross a few bars later fires gap_dn__rsi_x once, near the cross."""
    from h4_c_indicators import combo_entries, _preconditions, _triggers
    df = _frame(120)
    # force a gap-down at bar 40
    df.iloc[40, df.columns.get_loc("Open")] = df["Close"].iloc[39] * 0.95
    e = combo_entries(df)
    P = _preconditions(df); T = _triggers(df)
    assert P["gap_dn"].values[40], "gap_dn precondition should fire at 40"
    # any gap_dn__rsi_x fire must have an rsi_x trigger at that bar and a gap_dn within prior 5 bars
    fired = np.where(e["gap_dn__rsi_x"])[0]
    for i in fired:
        assert T["rsi_x"].values[i], "combo fired without trigger"
        assert P["gap_dn"].values[i-5:i].any(), "combo fired without recent precondition"
    print("test_combo_seq_fires OK")


def test_ladder_mask():
    from h4_c_indicators import ladder
    df = _frame(60)
    entry = np.zeros(60, bool); entry[10] = entry[30] = True
    close = df["Close"].values; dates = df.index.normalize()
    allow = {dates[10].date()}          # only the first entry allowed
    out = ladder(entry, close, dates, allow)
    assert len(out["1b"]) == 1, "mask should keep only the allowed entry"
    out_all = ladder(entry, close, dates, None)
    assert len(out_all["1b"]) == 2, "None mask keeps both"
    print("test_ladder_mask OK")


CHECKS = {"combo_keys": test_combo_keys, "combo_seq": test_combo_seq_fires, "ladder": test_ladder_mask}


def main():
    names = [a for a in sys.argv[1:] if a in CHECKS] or list(CHECKS)
    for nm in names:
        CHECKS[nm]()
    print(f"\n{len(names)} check(s) passed.")


if __name__ == "__main__":
    main()
