"""Standalone tests for market (SPY/QQQ) signal helpers. Run: python test_market_signals.py"""
import numpy as np
import pandas as pd
import studies as S


def _series_with_late_cross(n=120):
    """Downtrend then sharp uptrend so RSI crosses above its SMA near the end."""
    idx = pd.bdate_range("2023-01-02", periods=n)
    down = np.linspace(100, 70, n // 2)
    up = np.linspace(70, 110, n - n // 2)
    return pd.Series(np.concatenate([down, up]), index=idx)


def test_rsi_cross_series_fires_on_uptrend():
    close = _series_with_late_cross()
    x = S._rsi_cross_series(close)
    assert x.dtype == bool
    assert x.index.equals(close.index)
    assert x.iloc[: len(close) // 2].sum() == 0  # no up-cross during the downtrend
    assert x.iloc[len(close) // 2 :].sum() >= 1  # at least one up-cross during recovery


def test_rsi_cross_weekly_marks_one_daily_bar_per_week():
    close = _series_with_late_cross(200)
    xw = S._rsi_cross_series_weekly(close)
    assert xw.dtype == bool
    assert xw.index.equals(close.index)
    # every True must be the last daily bar of its ISO week (no two Trues in same week)
    trues = xw[xw].index
    weeks = [(d.isocalendar().year, d.isocalendar().week) for d in trues]
    assert len(weeks) == len(set(weeks))


def test_rolling_corr_and_beta_in_range():
    close = _series_with_late_cross()
    ref = close * 1.01 + 0.5  # near-perfectly correlated
    corr = S._rolling_corr(close, ref, w=30)
    beta = S._rolling_beta_series(close, ref, w=30)
    valid = corr.dropna()
    assert (valid <= 1.0001).all() and (valid >= -1.0001).all()
    assert corr.dropna().iloc[-1] > 0.9  # highly correlated
    assert beta.dropna().shape[0] > 0


def test_min_helpers_take_elementwise_min():
    close = _series_with_late_cross()
    spy = close * 1.00 + 0.1
    qqq = close * 1.20 + 0.1  # higher beta vs qqq-ish
    mc = S._min_corr_spyqqq(close, spy, qqq)
    mb = S._min_beta_spyqqq(close, spy, qqq)
    assert mc.index.equals(close.index) and mb.index.equals(close.index)
    # min is <= each component where both defined
    cs = S._rolling_corr(close, spy)
    cq = S._rolling_corr(close, qqq)
    both = mc.notna() & cs.notna() & cq.notna()
    assert (mc[both] <= cs[both] + 1e-9).all() and (mc[both] <= cq[both] + 1e-9).all()


def _df_from_close(close):
    return pd.DataFrame({"Open": close, "High": close * 1.01,
                         "Low": close * 0.99, "Close": close, "Volume": 1e6}, index=close.index)


def test_spy_rsi_x_matches_helper():
    close = _series_with_late_cross()
    spy = _series_with_late_cross()
    df = _df_from_close(close)
    out = S.sig_spy_rsi_x(df, spy_close=spy, qqq_close=close)
    assert out.index.equals(df.index) and out.dtype == bool
    # signal follows SPY's cross, not the sector's
    assert out.equals(S._rsi_cross_series(spy).reindex(df.index).fillna(False))


def test_missing_market_series_is_all_false():
    df = _df_from_close(_series_with_late_cross())
    out = S.sig_spy_qqq_rsi_x_both(df, spy_close=None, qqq_close=None)
    assert out.dtype == bool and out.sum() == 0


def test_both_requires_both_indices():
    close = _series_with_late_cross()
    df = _df_from_close(close)
    flat = pd.Series(100.0, index=close.index)  # never crosses
    # SPY crosses, QQQ flat → both-signal must be empty
    out = S.sig_spy_qqq_rsi_x_both(df, spy_close=close, qqq_close=flat)
    assert out.sum() == 0


def test_gated_is_subset_of_both():
    close = _series_with_late_cross()
    df = _df_from_close(close)
    both = S.sig_spy_qqq_rsi_x_both(df, spy_close=close, qqq_close=close)
    gated = S.sig_spy_qqq_rsi_x_hibeta(df, spy_close=close, qqq_close=close)
    # gated entries can only occur where the both-signal fired
    assert (gated & ~both).sum() == 0


def test_all_14_keys_registered_and_injected():
    keys = ["spy_rsi_x", "qqq_rsi_x", "spy_qqq_rsi_x_both", "spy_rsi_x_wk",
            "qqq_rsi_x_wk", "spy_qqq_rsi_x_both_wk", "corr_spyqqq_x_high",
            "beta_spyqqq_x_high", "spy_qqq_rsi_x_hibeta", "spy_qqq_rsi_x_hicorr",
            "spy_qqq_rsi_x_hibeta_hicorr", "spy_qqq_rsi_x_hibeta_wk",
            "spy_qqq_rsi_x_hicorr_wk", "spy_qqq_rsi_x_hibeta_hicorr_wk"]
    for k in keys:
        assert k in S.SIGNALS, f"{k} missing from SIGNALS"
        assert k in S.MARKET_SIGNAL_KEYS, f"{k} missing from MARKET_SIGNAL_KEYS"
    assert len(S.MARKET_SIGNAL_KEYS) == 14


def test_rolling_correlation_helper():
    import indicators
    idx = pd.bdate_range("2023-01-02", periods=90)
    a = pd.Series(np.linspace(1, 2, 90), index=idx).pct_change()
    b = (pd.Series(np.linspace(1, 2, 90), index=idx) * 1.01).pct_change()
    corr = indicators.rolling_correlation(a, b, window=20)
    v = corr.dropna()
    assert (v <= 1.0001).all() and (v >= -1.0001).all()


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in fns:
        fn()
        print(f"PASS {fn.__name__}")
    print(f"\nAll {len(fns)} tests passed.")
