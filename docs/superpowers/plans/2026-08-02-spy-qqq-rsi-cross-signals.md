# SPY + QQQ RSI-Cross, Correlation & Beta Signals — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add 14 SPY/QQQ-driven entry signals (daily + weekly RSI cross, correlation/beta standalone and gated combos) to the studies engine, plus per-sector correlation/beta report columns, then run a full data → studies → report update.

**Architecture:** New market-referencing signals live in `studies.py`, following the existing `sig_rsi_cross_sma_pos_updn` / `rsi_x_pos_updn` precedent where SPY (and now QQQ) close series are injected in `run_study`. Correlation/beta context columns go through `indicators.py` → `trend_analyzer.py` → `report.py`. A standalone `test_market_signals.py` validates signal logic on synthetic data.

**Tech Stack:** Python, pandas, numpy, `ta` (technical-analysis lib). No pytest — tests are standalone runnable scripts. Not a git repo — no commit steps.

## Global Constraints

- RSI cross = `ta.momentum.rsi(close, window=10)` vs its `.rolling(10).mean()`; cross = `(rsi > sma) & (rsi.shift(1) <= sma.shift(1))`.
- Correlation & beta window: `CB_WINDOW = 60` trading days. Thresholds `HI_CORR = 0.7`, `HI_BETA = 1.0`.
- "min-corr"/"min-β" = elementwise `min` of the metric vs SPY and vs QQQ.
- Weekly resample bucket: `W-FRI`. Weekly cross maps to ONE daily entry = last trading day of the crossing week.
- "Both" windows: `BOTH_WINDOW_D = 3` daily bars, `BOTH_WINDOW_W = 2` weekly bars.
- Every signal function returns a `bool` Series reindexed to `df.index`; missing market series → all-False Series.
- QQQ is already fetched (`"Nasdaq 100": "QQQ"` in `config.SECTOR_ETFS`) — no config change.
- All new signal keys must be added to a module-level `MARKET_SIGNAL_KEYS` set, and `run_study` injects `spy_close` + `qqq_close` for those keys.

---

### Task 1: Signal + metric helpers in `studies.py`

**Files:**
- Modify: `studies.py` — add helpers + constants after the existing indicator helpers (near `_macd_great`, line ~145) and before the `SIGNALS` dict.
- Create: `test_market_signals.py` (project root).

**Interfaces:**
- Consumes: `ta`, `numpy as np`, `pandas as pd` (already imported at top of `studies.py`).
- Produces (all used by Task 2):
  - `_rsi_cross_series(close: pd.Series) -> pd.Series`  (bool, indexed like `close`)
  - `_rsi_cross_series_weekly(close: pd.Series) -> pd.Series`  (bool, daily index of `close`)
  - `_rolling_corr(close: pd.Series, ref_close: pd.Series, w: int = CB_WINDOW) -> pd.Series`
  - `_rolling_beta_series(close: pd.Series, ref_close: pd.Series, w: int = CB_WINDOW) -> pd.Series`
  - `_min_corr_spyqqq(close, spy_close, qqq_close) -> pd.Series`
  - `_min_beta_spyqqq(close, spy_close, qqq_close) -> pd.Series`
  - Module constants: `CB_WINDOW=60`, `HI_CORR=0.7`, `HI_BETA=1.0`, `BOTH_WINDOW_D=3`, `BOTH_WINDOW_W=2`

- [ ] **Step 1: Write the failing test** — create `test_market_signals.py`:

```python
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


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in fns:
        fn()
        print(f"PASS {fn.__name__}")
    print(f"\nAll {len(fns)} tests passed.")
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python test_market_signals.py`
Expected: FAIL — `AttributeError: module 'studies' has no attribute '_rsi_cross_series'`.

- [ ] **Step 3: Add constants + helpers to `studies.py`** (insert after `_macd_great`, before `SIGNALS = {`):

```python
# ── Market (SPY/QQQ) signal constants & helpers ──
CB_WINDOW = 60          # rolling window (trading days) for correlation & beta
HI_CORR = 0.7           # "high correlation" threshold
HI_BETA = 1.0           # "high beta" threshold
BOTH_WINDOW_D = 3       # daily bars: SPY & QQQ must each have crossed within this window
BOTH_WINDOW_W = 2       # weekly bars: same, for weekly variants


def _rsi_cross_series(close):
    """Bool Series: RSI(10) crosses above its SMA(10) on that bar."""
    rsi = ta.momentum.rsi(close, window=10)
    sma = rsi.rolling(10).mean()
    return ((rsi > sma) & (rsi.shift(1) <= sma.shift(1))).fillna(False).astype(bool)


def _rsi_cross_series_weekly(close):
    """Bool Series (daily index of `close`): weekly RSI(10) cross mapped to a single
    daily entry on the last trading day of the crossing week."""
    out = pd.Series(False, index=close.index)
    wk = close.resample("W-FRI").last().dropna()
    if len(wk) < 15:
        return out
    rsi = ta.momentum.rsi(wk, window=10)
    sma = rsi.rolling(10).mean()
    wk_cross = (rsi > sma) & (rsi.shift(1) <= sma.shift(1))
    for wend in wk_cross[wk_cross.fillna(False)].index:
        mask = (close.index > wend - pd.Timedelta(days=7)) & (close.index <= wend)
        if mask.any():
            out.loc[close.index[mask][-1]] = True
    return out


def _rolling_corr(close, ref_close, w=CB_WINDOW):
    """Rolling Pearson correlation of daily returns of `close` vs `ref_close`."""
    a = close.pct_change()
    b = ref_close.pct_change().reindex(a.index)
    return a.rolling(w).corr(b)


def _rolling_beta_series(close, ref_close, w=CB_WINDOW):
    """Rolling beta = cov(asset, ref) / var(ref) on daily returns."""
    a = close.pct_change()
    b = ref_close.pct_change().reindex(a.index)
    cov = a.rolling(w).cov(b)
    var = b.rolling(w).var()
    return (cov / var).replace([np.inf, -np.inf], np.nan)


def _min_corr_spyqqq(close, spy_close, qqq_close):
    """Elementwise min of corr-to-SPY and corr-to-QQQ (reindexed to close)."""
    cs = _rolling_corr(close, spy_close).reindex(close.index)
    cq = _rolling_corr(close, qqq_close).reindex(close.index)
    return pd.concat([cs, cq], axis=1).min(axis=1)


def _min_beta_spyqqq(close, spy_close, qqq_close):
    """Elementwise min of beta-to-SPY and beta-to-QQQ (reindexed to close)."""
    bs = _rolling_beta_series(close, spy_close).reindex(close.index)
    bq = _rolling_beta_series(close, qqq_close).reindex(close.index)
    return pd.concat([bs, bq], axis=1).min(axis=1)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python test_market_signals.py`
Expected: `PASS` for the 4 helper tests, ending `All 4 tests passed.`
(If `test_min_helpers_take_elementwise_min` fails on NaN alignment, confirm `.reindex(close.index)` is present in the min helpers.)

- [ ] **Step 5: Verify no import regression**

Run: `python -c "import studies; print('ok', studies.CB_WINDOW, studies.HI_BETA)"`
Expected: `ok 60 1.0`

---

### Task 2: Signal functions + `SIGNALS` entries + `MARKET_SIGNAL_KEYS`

**Files:**
- Modify: `studies.py` — add 14 `sig_*` functions after the helpers from Task 1; add a `MARKET_SIGNAL_KEYS` set; add 14 entries to the `SIGNALS` dict (before the closing `}` at line ~1957).
- Modify: `test_market_signals.py` — add signal-function tests.

**Interfaces:**
- Consumes: all helpers + constants from Task 1.
- Produces (used by Task 3 via the `SIGNALS` dict and `MARKET_SIGNAL_KEYS`):
  - Daily fns: `sig_spy_rsi_x(df, spy_close=None, qqq_close=None)`, `sig_qqq_rsi_x(...)`, `sig_spy_qqq_rsi_x_both(...)`, `sig_corr_spyqqq_x_high(...)`, `sig_beta_spyqqq_x_high(...)`, `sig_spy_qqq_rsi_x_hibeta(...)`, `sig_spy_qqq_rsi_x_hicorr(...)`, `sig_spy_qqq_rsi_x_hibeta_hicorr(...)`
  - Weekly fns: `sig_spy_rsi_x_wk(...)`, `sig_qqq_rsi_x_wk(...)`, `sig_spy_qqq_rsi_x_both_wk(...)`, `sig_spy_qqq_rsi_x_hibeta_wk(...)`, `sig_spy_qqq_rsi_x_hicorr_wk(...)`, `sig_spy_qqq_rsi_x_hibeta_hicorr_wk(...)`
  - `MARKET_SIGNAL_KEYS: set[str]` — the 14 keys.
  - All fns accept `(df, spy_close=None, qqq_close=None)` and return a bool Series on `df.index`.

- [ ] **Step 1: Write the failing test** — append to `test_market_signals.py` (before the `__main__` block):

```python
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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python test_market_signals.py`
Expected: FAIL — `AttributeError: module 'studies' has no attribute 'sig_spy_rsi_x'`.

- [ ] **Step 3: Add the 14 signal functions** (in `studies.py`, after Task 1 helpers):

```python
# ── Market (SPY/QQQ) entry signals. All accept (df, spy_close, qqq_close). ──
def _empty(df):
    return pd.Series(False, index=df.index)


def sig_spy_rsi_x(df, spy_close=None, qqq_close=None):
    if spy_close is None:
        return _empty(df)
    return _rsi_cross_series(spy_close).reindex(df.index).fillna(False).astype(bool)


def sig_qqq_rsi_x(df, spy_close=None, qqq_close=None):
    if qqq_close is None:
        return _empty(df)
    return _rsi_cross_series(qqq_close).reindex(df.index).fillna(False).astype(bool)


def _both_recent(a_cross, b_cross, index, window):
    """True where BOTH cross series have fired within the last `window` bars."""
    a = a_cross.reindex(index).fillna(False).rolling(window, min_periods=1).max().astype(bool)
    b = b_cross.reindex(index).fillna(False).rolling(window, min_periods=1).max().astype(bool)
    fresh = (a_cross.reindex(index).fillna(False)) | (b_cross.reindex(index).fillna(False))
    return (a & b & fresh).astype(bool)


def sig_spy_qqq_rsi_x_both(df, spy_close=None, qqq_close=None):
    if spy_close is None or qqq_close is None:
        return _empty(df)
    return _both_recent(_rsi_cross_series(spy_close), _rsi_cross_series(qqq_close),
                        df.index, BOTH_WINDOW_D)


def sig_spy_rsi_x_wk(df, spy_close=None, qqq_close=None):
    if spy_close is None:
        return _empty(df)
    return _rsi_cross_series_weekly(spy_close).reindex(df.index).fillna(False).astype(bool)


def sig_qqq_rsi_x_wk(df, spy_close=None, qqq_close=None):
    if qqq_close is None:
        return _empty(df)
    return _rsi_cross_series_weekly(qqq_close).reindex(df.index).fillna(False).astype(bool)


def sig_spy_qqq_rsi_x_both_wk(df, spy_close=None, qqq_close=None):
    if spy_close is None or qqq_close is None:
        return _empty(df)
    return _both_recent(_rsi_cross_series_weekly(spy_close), _rsi_cross_series_weekly(qqq_close),
                        df.index, BOTH_WINDOW_W)


def sig_corr_spyqqq_x_high(df, spy_close=None, qqq_close=None):
    if spy_close is None or qqq_close is None:
        return _empty(df)
    mc = _min_corr_spyqqq(df["Close"], spy_close, qqq_close)
    return ((mc > HI_CORR) & (mc.shift(1) <= HI_CORR)).fillna(False).astype(bool)


def sig_beta_spyqqq_x_high(df, spy_close=None, qqq_close=None):
    if spy_close is None or qqq_close is None:
        return _empty(df)
    mb = _min_beta_spyqqq(df["Close"], spy_close, qqq_close)
    return ((mb > HI_BETA) & (mb.shift(1) <= HI_BETA)).fillna(False).astype(bool)


def _gate(both, df, spy_close, qqq_close, use_beta, use_corr):
    cond = both.copy()
    if use_beta:
        mb = _min_beta_spyqqq(df["Close"], spy_close, qqq_close).reindex(df.index)
        cond = cond & (mb > HI_BETA)
    if use_corr:
        mc = _min_corr_spyqqq(df["Close"], spy_close, qqq_close).reindex(df.index)
        cond = cond & (mc > HI_CORR)
    return cond.fillna(False).astype(bool)


def sig_spy_qqq_rsi_x_hibeta(df, spy_close=None, qqq_close=None):
    if spy_close is None or qqq_close is None:
        return _empty(df)
    return _gate(sig_spy_qqq_rsi_x_both(df, spy_close, qqq_close), df, spy_close, qqq_close, True, False)


def sig_spy_qqq_rsi_x_hicorr(df, spy_close=None, qqq_close=None):
    if spy_close is None or qqq_close is None:
        return _empty(df)
    return _gate(sig_spy_qqq_rsi_x_both(df, spy_close, qqq_close), df, spy_close, qqq_close, False, True)


def sig_spy_qqq_rsi_x_hibeta_hicorr(df, spy_close=None, qqq_close=None):
    if spy_close is None or qqq_close is None:
        return _empty(df)
    return _gate(sig_spy_qqq_rsi_x_both(df, spy_close, qqq_close), df, spy_close, qqq_close, True, True)


def sig_spy_qqq_rsi_x_hibeta_wk(df, spy_close=None, qqq_close=None):
    if spy_close is None or qqq_close is None:
        return _empty(df)
    return _gate(sig_spy_qqq_rsi_x_both_wk(df, spy_close, qqq_close), df, spy_close, qqq_close, True, False)


def sig_spy_qqq_rsi_x_hicorr_wk(df, spy_close=None, qqq_close=None):
    if spy_close is None or qqq_close is None:
        return _empty(df)
    return _gate(sig_spy_qqq_rsi_x_both_wk(df, spy_close, qqq_close), df, spy_close, qqq_close, False, True)


def sig_spy_qqq_rsi_x_hibeta_hicorr_wk(df, spy_close=None, qqq_close=None):
    if spy_close is None or qqq_close is None:
        return _empty(df)
    return _gate(sig_spy_qqq_rsi_x_both_wk(df, spy_close, qqq_close), df, spy_close, qqq_close, True, True)


MARKET_SIGNAL_KEYS = {
    "spy_rsi_x", "qqq_rsi_x", "spy_qqq_rsi_x_both",
    "spy_rsi_x_wk", "qqq_rsi_x_wk", "spy_qqq_rsi_x_both_wk",
    "corr_spyqqq_x_high", "beta_spyqqq_x_high",
    "spy_qqq_rsi_x_hibeta", "spy_qqq_rsi_x_hicorr", "spy_qqq_rsi_x_hibeta_hicorr",
    "spy_qqq_rsi_x_hibeta_wk", "spy_qqq_rsi_x_hicorr_wk", "spy_qqq_rsi_x_hibeta_hicorr_wk",
}
```

Note: the gated `sig_*` call `sig_spy_qqq_rsi_x_both(df, spy_close, qqq_close)` positionally — those functions accept `(df, spy_close=None, qqq_close=None)`, so positional passing is correct.

- [ ] **Step 4: Register the 14 entries in the `SIGNALS` dict** — insert before the closing `}` of `SIGNALS` (line ~1957, right after the last `seq_*` entry):

```python
    # ── SPY/QQQ market signals (injected with spy_close + qqq_close in run_study) ──
    "spy_rsi_x": ("SPY RSI Cross Above SMA", sig_spy_rsi_x),
    "qqq_rsi_x": ("QQQ RSI Cross Above SMA", sig_qqq_rsi_x),
    "spy_qqq_rsi_x_both": ("SPY+QQQ RSI Cross (both, 3d)", sig_spy_qqq_rsi_x_both),
    "spy_rsi_x_wk": ("SPY Weekly RSI Cross", sig_spy_rsi_x_wk),
    "qqq_rsi_x_wk": ("QQQ Weekly RSI Cross", sig_qqq_rsi_x_wk),
    "spy_qqq_rsi_x_both_wk": ("SPY+QQQ Weekly RSI Cross (both)", sig_spy_qqq_rsi_x_both_wk),
    "corr_spyqqq_x_high": ("Corr->SPY&QQQ Cross >0.7", sig_corr_spyqqq_x_high),
    "beta_spyqqq_x_high": ("Beta->SPY&QQQ Cross >1.0", sig_beta_spyqqq_x_high),
    "spy_qqq_rsi_x_hibeta": ("SPY+QQQ Cross + HiBeta", sig_spy_qqq_rsi_x_hibeta),
    "spy_qqq_rsi_x_hicorr": ("SPY+QQQ Cross + HiCorr", sig_spy_qqq_rsi_x_hicorr),
    "spy_qqq_rsi_x_hibeta_hicorr": ("SPY+QQQ Cross + HiBeta + HiCorr", sig_spy_qqq_rsi_x_hibeta_hicorr),
    "spy_qqq_rsi_x_hibeta_wk": ("SPY+QQQ Weekly Cross + HiBeta", sig_spy_qqq_rsi_x_hibeta_wk),
    "spy_qqq_rsi_x_hicorr_wk": ("SPY+QQQ Weekly Cross + HiCorr", sig_spy_qqq_rsi_x_hicorr_wk),
    "spy_qqq_rsi_x_hibeta_hicorr_wk": ("SPY+QQQ Weekly Cross + HiBeta + HiCorr", sig_spy_qqq_rsi_x_hibeta_hicorr_wk),
```

- [ ] **Step 5: Run test to verify it passes**

Run: `python test_market_signals.py`
Expected: all tests PASS, ending `All 9 tests passed.`

---

### Task 3: Inject `spy_close` + `qqq_close` in `run_study`

**Files:**
- Modify: `studies.py` — the signal-dispatch `try` block in `run_study` (line ~2461-2469).

**Interfaces:**
- Consumes: `MARKET_SIGNAL_KEYS` (Task 2), `config.BENCHMARK`, `all_data["QQQ"]`.
- Produces: `run_study` correctly evaluates all 14 market signals.

- [ ] **Step 1: Replace the dispatch block** — change:

```python
        try:
            if sig_key == "rsi_x_pos_updn":
                spy_df = all_data.get(config.BENCHMARK)
                spy_close = spy_df["Close"] if spy_df is not None else None
                signals = sig_fn(df, spy_close=spy_close).fillna(False)
            else:
                signals = sig_fn(df).fillna(False)
        except Exception:
            continue
```

to:

```python
        try:
            if sig_key == "rsi_x_pos_updn":
                spy_df = all_data.get(config.BENCHMARK)
                spy_close = spy_df["Close"] if spy_df is not None else None
                signals = sig_fn(df, spy_close=spy_close).fillna(False)
            elif sig_key in MARKET_SIGNAL_KEYS:
                spy_df = all_data.get(config.BENCHMARK)
                qqq_df = all_data.get("QQQ")
                spy_close = spy_df["Close"] if spy_df is not None else None
                qqq_close = qqq_df["Close"] if qqq_df is not None else None
                signals = sig_fn(df, spy_close=spy_close, qqq_close=qqq_close).fillna(False)
            else:
                signals = sig_fn(df).fillna(False)
        except Exception:
            continue
```

- [ ] **Step 2: Verify study count grew by 14×70**

Run: `python -c "import studies; s=studies.generate_studies(); mk=[x for x in s if x['signal'] in studies.MARKET_SIGNAL_KEYS]; print('total', len(s), 'market', len(mk))"`
Expected: `market 980` (14 signals × 70 exits), and `total` is the previous total + 980.

- [ ] **Step 3: Smoke-run ONE market study end-to-end on real data**

Run:
```bash
python -c "import studies, data_fetcher, config; ad=data_fetcher.fetch_all(); \
st={'signal':'spy_qqq_rsi_x_both','exit':'1w','name':'x','signal_name':'x','exit_name':'x','category':'x'}; \
r=studies.run_study(st, ad); print('trades', r.get('total_trades'), 'avg_ret', r.get('avg_return'))"
```
Expected: prints a non-negative `trades` count (>0 in normal markets) and a numeric `avg_ret`. No exception.

- [ ] **Step 4: Smoke-run one WEEKLY study**

Run: same as Step 3 but `'signal':'spy_rsi_x_wk'`.
Expected: `trades` > 0, numeric `avg_ret`, no exception. (Weekly fires less often than daily — a smaller trade count is expected.)

---

### Task 4: Correlation helper + report columns

**Files:**
- Modify: `indicators.py` — add `rolling_correlation` next to `rolling_beta` (line ~189).
- Modify: `trend_analyzer.py` — compute `corr_spy`, `corr_qqq`, `beta_qqq` in `analyze()`; add to the result dict (line ~121-170).
- Modify: `report.py` — add the three columns to the output table.
- Modify: `test_market_signals.py` — add a test for `rolling_correlation`.

**Interfaces:**
- Consumes: `config`, `numpy`, `pandas` (already imported in `indicators.py`); `all_data["QQQ"]` inside `analyze()`.
- Produces: `indicators.rolling_correlation(asset_returns, bench_returns, window=config.SORTINO_WINDOW) -> pd.Series`; new result keys `corr_spy`, `corr_qqq`, `beta_qqq` (floats rounded to 2, or `None`).

- [ ] **Step 1: Write the failing test** — append to `test_market_signals.py` (before `__main__`):

```python
def test_rolling_correlation_helper():
    import indicators
    idx = pd.bdate_range("2023-01-02", periods=90)
    a = pd.Series(np.linspace(1, 2, 90), index=idx).pct_change()
    b = (pd.Series(np.linspace(1, 2, 90), index=idx) * 1.01).pct_change()
    corr = indicators.rolling_correlation(a, b, window=20)
    v = corr.dropna()
    assert (v <= 1.0001).all() and (v >= -1.0001).all()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python test_market_signals.py`
Expected: FAIL — `AttributeError: module 'indicators' has no attribute 'rolling_correlation'`.

- [ ] **Step 3: Add `rolling_correlation` to `indicators.py`** (after `rolling_beta`, line ~189):

```python
def rolling_correlation(
    asset_returns: pd.Series,
    bench_returns: pd.Series,
    window: int = config.SORTINO_WINDOW,
) -> pd.Series:
    """Rolling Pearson correlation of asset vs benchmark daily returns."""
    aligned = pd.DataFrame({"asset": asset_returns, "bench": bench_returns}).dropna()
    return aligned["asset"].rolling(window).corr(aligned["bench"])
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python test_market_signals.py`
Expected: all PASS, ending `All 10 tests passed.`

- [ ] **Step 5: Compute the columns in `trend_analyzer.analyze()`** — fetch QQQ once before the loop (after `spy_df = all_data.get(config.BENCHMARK)`, line ~25):

```python
    qqq_df = all_data.get("QQQ")
    spy_ret_full = indicators.daily_returns(spy_df)
    qqq_ret_full = indicators.daily_returns(qqq_df) if qqq_df is not None else None
```

Inside the per-sector loop, after `metrics = indicators.compute_all_risk_metrics(...)` (line ~38), add:

```python
        etf_ret = indicators.daily_returns(etf_df)
        corr_spy = indicators.rolling_correlation(etf_ret, spy_ret_full, window).iloc[-1] \
            if len(etf_ret) > window else float("nan")
        if qqq_ret_full is not None and len(etf_ret) > window:
            corr_qqq = indicators.rolling_correlation(etf_ret, qqq_ret_full, window).iloc[-1]
            beta_qqq = indicators.rolling_beta(etf_ret, qqq_ret_full, window).iloc[-1]
        else:
            corr_qqq = float("nan")
            beta_qqq = float("nan")
```

Then add to the appended result dict (near `"beta": metrics.get("beta"),`, line ~145):

```python
            "corr_spy": round(float(corr_spy), 2) if corr_spy == corr_spy else None,
            "corr_qqq": round(float(corr_qqq), 2) if corr_qqq == corr_qqq else None,
            "beta_qqq": round(float(beta_qqq), 2) if beta_qqq == beta_qqq else None,
```

(`x == x` is the NaN check — NaN is the only value not equal to itself.)

- [ ] **Step 6: Verify `analyze()` returns the new keys**

Run:
```bash
python -c "import trend_analyzer as t; r=t.analyze(); row=r[0]; print({k:row[k] for k in ('sector','beta','beta_qqq','corr_spy','corr_qqq')})"
```
Expected: dict prints with `corr_spy` in roughly [-1, 1], `beta`/`beta_qqq` numeric or `None`. No exception.

- [ ] **Step 7: Add columns to `report.py`**

First inspect the table construction:
Run: `grep -n "beta\|rsi_spread\|headers\|f\"{" report.py | head -40`

Then add `corr_spy`, `corr_qqq`, `beta_qqq` alongside the existing `beta` column — same formatting pattern (numeric, 2 decimals, blank/`-` when `None`). Match the surrounding header + row-format code exactly; do not restructure the table.

- [ ] **Step 8: Verify the report renders**

Run: `python report.py`
Expected: table prints with the new corr/beta columns populated, no exception, no misaligned rows.

---

### Task 5: "Update everything" run

**Files:** none modified — this task runs the pipeline.

**Interfaces:** Consumes everything above.

- [ ] **Step 1: Refresh data + regenerate the trend report** (fetch is incremental inside `report.py`)

Run: `python report.py`
Expected: completes; the "up to date" / incremental-download log appears; table renders with corr/beta columns.

- [ ] **Step 2: DB import (candles) — if the DB is reachable**

Run (from `backend/`): `python manage.py import_json`
Expected: imports refreshed JSON into Postgres. If it errors with a DB connection failure in this environment, record the error and skip — note it in the final summary.

- [ ] **Step 3: Re-run studies (computes the ~980 new combos only)**

Run: `python studies.py` (or the module's `run_all()` entrypoint — confirm with `grep -n "__main__" studies.py`).
Expected: logs `Generated <N> studies`, `New to compute: 980` (existing skipped), then completes and saves. Confirm the "new to compute" count is 980 (14×70); if it is larger, existing studies were missing from the DB (acceptable) — if smaller, a signal key failed to register (go back to Task 2).

- [ ] **Step 4: Spot-check saved market studies**

Run:
```bash
python -c "import json,glob,os; f=sorted(glob.glob('.data/studies/*.json'))[-1]; \
d=json.load(open(f)); mk=[s for s in d if str(s.get('signal','')).startswith(('spy_','qqq_','corr_spyqqq','beta_spyqqq'))]; \
print('market studies saved:', len(mk)); print(mk[0]['signal'], mk[0].get('avg_return')) if mk else print('NONE')"
```
Expected: prints a non-zero count of saved market studies and a sample signal + avg_return. (Adjust the glob if the studies runner writes to the DB only — in that case verify via the Django `Study` model instead.)

- [ ] **Step 5: Final summary**

Report: number of new studies computed, whether DB import ran or was skipped (with reason), top few market-signal studies by `avg_return`, and confirmation the trend report renders with corr/beta columns.

---

## Self-Review notes

- **Spec coverage:** Group A daily (Task 2), Group A weekly (Task 2), Group B standalone (Task 2), Group C gated daily+weekly (Task 2), injection (Task 3), report columns/Group 4 (Task 4), update sequence (Task 5). All spec sections mapped.
- **Type consistency:** every `sig_*` has signature `(df, spy_close=None, qqq_close=None) -> pd.Series[bool]`; gated fns call the "both" fns positionally, consistent with that signature. Helper names match between Task 1 (definitions), Task 2 (usage), and tests.
- **`both` semantics:** `_both_recent` requires each index to have crossed within `window` bars AND at least one of them to have crossed on the current bar (`fresh`), so the signal fires on the completing bar rather than persisting for `window` bars — one entry per confirmation, matching "single daily entry" intent.
- **No pytest / no git:** tests are a standalone script; there are no commit steps (environment is not a git repo).
