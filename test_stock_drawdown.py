"""Standalone tests for the 30%-off-ATH + RSI reversal strategy.
Run: python test_stock_drawdown.py"""
import numpy as np
import pandas as pd
import indicators
import studies as S


def _mk_df(closes):
    idx = pd.bdate_range("2022-01-03", periods=len(closes))
    c = pd.Series(closes, index=idx, dtype=float)
    return pd.DataFrame({"Open": c, "High": c * 1.005, "Low": c * 0.995,
                         "Close": c, "Volume": 1e6}, index=idx)


def test_drawdown_from_high_basic():
    df = _mk_df([100, 110, 120, 90, 78])   # peak 120, last = 78 -> 78/120-1 = -0.35
    dd = indicators.drawdown_from_high(df)
    assert dd.index.equals(df.index)
    assert abs(dd.iloc[-1] - (78 / 120 - 1)) < 1e-9
    assert dd.iloc[2] == 0.0                # at the running high -> 0
    assert (dd <= 1e-12).all()              # never positive


def _rise_fall_recover():
    """Rise to a peak, fall >35%, then turn up enough for RSI to cross its SMA."""
    up = list(np.linspace(50, 120, 40))     # peak 120
    down = list(np.linspace(120, 74, 30))   # -38% from peak (74/120-1)
    recover = list(np.linspace(74, 88, 20))  # turn up while still >30% down
    return _mk_df(up + down + recover)


def test_signal_fires_on_reversal_while_down():
    df = _rise_fall_recover()
    sig = S.sig_dd30_rsi_reversal(df)
    assert sig.index.equals(df.index) and sig.dtype == bool
    fire = sig[sig].index
    assert len(fire) >= 1                      # fires at least once
    dd = indicators.drawdown_from_high(df)
    for d in fire:                             # every fire is >=30% below ATH
        assert dd.loc[d] <= -0.30 + 1e-9


def test_no_fire_while_falling():
    df = _rise_fall_recover()
    sig = S.sig_dd30_rsi_reversal(df)
    # during the straight decline (bars 41..68) momentum is down -> no fire
    assert sig.iloc[41:68].sum() == 0


def test_no_fire_when_shallow_drawdown():
    # peak 100 -> trough 78 = only 22% down, then recover; must NOT fire (<30%)
    df = _mk_df(list(np.linspace(60, 100, 40)) + list(np.linspace(100, 78, 20))
                + list(np.linspace(78, 92, 20)))
    sig = S.sig_dd30_rsi_reversal(df)
    dd = indicators.drawdown_from_high(df)
    assert dd.min() > -0.30                    # never reached 30% down
    assert sig.sum() == 0


def test_registered_in_signals():
    assert "dd30_rsi_reversal" in S.SIGNALS
    assert S.SIGNALS["dd30_rsi_reversal"][1] is S.sig_dd30_rsi_reversal
    assert S.DD_THRESHOLD == 0.30


def test_run_one_exit_aggregates_trades():
    import stock_drawdown_study as sd
    # Two synthetic stocks that each produce a dd30 reversal entry
    a = _rise_fall_recover()
    b = _rise_fall_recover()
    stock_data = {"AAA": a, "BBB": b}
    t2s = {"AAA": "Tech", "BBB": "Energy"}
    res = sd.run_one_exit("1w", stock_data, t2s)
    assert res["exit_key"] == "1w"
    assert res["total_trades"] >= 2            # at least one entry per stock
    assert isinstance(res["avg_return"], float)
    assert res["sector_count"] >= 1
    assert isinstance(res["best_sectors"], list)


def test_build_universe_dedupes_and_drops_dotted():
    import stock_drawdown_study as sd
    tickers, t2s = sd.build_universe()
    assert len(tickers) == len(set(tickers))   # deduped
    assert all("." not in t for t in tickers)  # no foreign tickers
    assert all(t in t2s for t in tickers)      # every ticker mapped to a sector


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in fns:
        fn()
        print(f"PASS {fn.__name__}")
    print(f"\nAll {len(fns)} tests passed.")
