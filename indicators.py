"""
Stock Market Trend Bot - Indicators

Rolling Sortino ratio and RSI with SMA crossover detection.
"""

import numpy as np
import pandas as pd
import ta

import config

RSI_PERIOD = 10
RSI_SMA_PERIOD = 10

# MACD parameters (standard 12/26/9 on daily closes)
MACD_FAST = 12
MACD_SLOW = 26
MACD_SIGNAL = 9


def daily_returns(df: pd.DataFrame) -> pd.Series:
    """Compute daily log returns from Close prices."""
    return np.log(df["Close"] / df["Close"].shift(1)).dropna()


def rolling_sortino(
    returns: pd.Series,
    window: int = config.SORTINO_WINDOW,
    risk_free_rate: float = config.RISK_FREE_RATE,
) -> pd.Series:
    """
    Rolling Sortino ratio over a window of daily returns.

    Sortino = mean(excess) / downside_deviation

    Downside deviation uses ALL observations in the window,
    replacing positive excess returns with 0 before computing std.
    No annualization — keeps values in a sensible range for short windows.
    """
    daily_rf = risk_free_rate / config.TRADING_DAYS

    def _sortino(window_returns):
        excess = window_returns - daily_rf
        mean_excess = excess.mean()

        # Proper downside deviation: zero out positive returns, compute std over full window
        downside = np.minimum(excess, 0)
        dd = np.sqrt(np.mean(downside ** 2))

        if dd < 1e-10:
            # No downside in the window. The ratio is ±inf, NOT "undefined" — returning NaN made
            # the caller (_last) reach back to a STALE earlier bar and mislabel a ripping ETF.
            # Flat window (no up moves either) is genuinely undefined → NaN; otherwise a large
            # JSON-safe finite sentinel so the sortino>0 gate reads correctly.
            if abs(mean_excess) < 1e-10:
                return np.nan
            return 9.99 if mean_excess > 0 else -9.99

        return mean_excess / dd

    return returns.rolling(window=window).apply(_sortino, raw=False)


def compute_sortino_comparison(
    sector_data: dict[str, pd.DataFrame],
    benchmark_df: pd.DataFrame,
    window: int = config.SORTINO_WINDOW,
) -> pd.DataFrame:
    """
    Compute rolling Sortino for each sector ETF and the benchmark.
    Returns a DataFrame with columns = tickers, values = rolling Sortino.
    """
    bench_ret = daily_returns(benchmark_df)
    result = pd.DataFrame(index=bench_ret.index)
    result[config.BENCHMARK] = rolling_sortino(bench_ret, window)

    for sector_name, etf_ticker in config.SECTOR_ETFS.items():
        df = sector_data.get(etf_ticker)
        if df is None:
            continue
        ret = daily_returns(df)
        # Align to benchmark index
        ret = ret.reindex(bench_ret.index)
        result[etf_ticker] = rolling_sortino(ret, window)

    return result.dropna(how="all")


def rolling_omega(
    returns: pd.Series,
    window: int = config.SORTINO_WINDOW,
    threshold: float = 0.0,
) -> pd.Series:
    """
    Rolling Omega ratio. Sum of gains above threshold / sum of losses below threshold.
    Higher = better. >1 means more gain than pain.
    """
    def _omega(r):
        excess = r - threshold
        gains = excess[excess > 0].sum()
        losses = -excess[excess < 0].sum()
        if losses < 1e-10:
            # No losses in the window → Omega is +inf (all upside), NOT undefined. Returning NaN
            # made _last reach back to a stale bar and downgrade a strong ETF from BULLISH to
            # "RSI ONLY". Flat window (no gains either) is genuinely undefined → NaN; otherwise a
            # large JSON-safe finite sentinel so the omega>1 gate reads bullish.
            return np.nan if gains < 1e-10 else 99.0
        return gains / losses

    return returns.rolling(window=window).apply(_omega, raw=False)


def rolling_cvar(
    returns: pd.Series,
    window: int = config.SORTINO_WINDOW,
    alpha: float = 0.05,
) -> pd.Series:
    """
    Rolling CVaR (Expected Shortfall) at alpha level: mean of the worst alpha% of returns.
    NOTE: at the default 10-day window, int(10*0.05)=0 → n=1, so this DEGENERATES to the single
    worst day (a 1-sample VaR, not an averaged tail). Fine as a relative ETF-vs-SPY tail gauge
    (both use the same method); do NOT read the absolute number as a true multi-sample ES unless
    the window is large enough that len*alpha >= ~2.
    """
    def _cvar(r):
        sorted_r = np.sort(r)
        n = max(1, int(len(sorted_r) * alpha))
        return sorted_r[:n].mean()

    return returns.rolling(window=window).apply(_cvar, raw=False)


def rolling_ulcer(
    prices: pd.Series,
    window: int = config.SORTINO_WINDOW,
) -> pd.Series:
    """
    Rolling Ulcer Index. Measures depth and duration of drawdowns.
    Lower = less painful drawdowns.
    """
    def _ulcer(p):
        peak = np.maximum.accumulate(p)
        dd_pct = (p - peak) / peak * 100
        return np.sqrt(np.mean(dd_pct ** 2))

    return prices.rolling(window=window).apply(_ulcer, raw=False)


def rolling_updown_capture(
    asset_returns: pd.Series,
    bench_returns: pd.Series,
    window: int = config.SORTINO_WINDOW,
) -> tuple[pd.Series, pd.Series]:
    """
    Rolling Upside and Downside capture ratios vs a benchmark.
    Upside capture > 100% = captures more upside than benchmark.
    Downside capture < 100% = captures less downside than benchmark.
    """
    aligned = pd.DataFrame({"asset": asset_returns, "bench": bench_returns}).dropna()

    def _up_capture(idx):
        rows = aligned.iloc[idx]
        up_days = rows[rows["bench"] > 0]
        if len(up_days) < 2:
            return np.nan
        return (up_days["asset"].mean() / up_days["bench"].mean()) * 100

    def _down_capture(idx):
        rows = aligned.iloc[idx]
        down_days = rows[rows["bench"] < 0]
        if len(down_days) < 2:
            return np.nan
        return (down_days["asset"].mean() / down_days["bench"].mean()) * 100

    n = len(aligned)
    up_caps = []
    down_caps = []
    for i in range(n):
        if i < window - 1:
            up_caps.append(np.nan)
            down_caps.append(np.nan)
        else:
            idx = range(i - window + 1, i + 1)
            up_caps.append(_up_capture(idx))
            down_caps.append(_down_capture(idx))

    up_series = pd.Series(up_caps, index=aligned.index)
    down_series = pd.Series(down_caps, index=aligned.index)
    return up_series, down_series


def rolling_beta(
    asset_returns: pd.Series,
    bench_returns: pd.Series,
    window: int = config.SORTINO_WINDOW,
) -> pd.Series:
    """Rolling beta = cov(asset, bench) / var(bench)."""
    aligned = pd.DataFrame({"asset": asset_returns, "bench": bench_returns}).dropna()
    cov = aligned["asset"].rolling(window).cov(aligned["bench"])
    var = aligned["bench"].rolling(window).var()
    return (cov / var).replace([np.inf, -np.inf], np.nan)


def rolling_correlation(
    asset_returns: pd.Series,
    bench_returns: pd.Series,
    window: int = config.SORTINO_WINDOW,
) -> pd.Series:
    """Rolling Pearson correlation of asset vs benchmark daily returns."""
    aligned = pd.DataFrame({"asset": asset_returns, "bench": bench_returns}).dropna()
    return aligned["asset"].rolling(window).corr(aligned["bench"])


def drawdown_from_high(df: pd.DataFrame) -> pd.Series:
    """Fractional drop from the running all-time-high close.
    0.0 at a new high; -0.30 means 30% below the running ATH."""
    close = df["Close"]
    return close / close.cummax() - 1.0


def compute_all_risk_metrics(
    etf_df: pd.DataFrame,
    spy_df: pd.DataFrame,
    window: int = config.SORTINO_WINDOW,
) -> dict:
    """
    Compute all risk metrics for a single ETF vs SPY.
    Returns latest values for: Sortino, Omega, CVaR, Ulcer Index,
    Upside Capture, Downside Capture.
    """
    etf_ret = daily_returns(etf_df)
    spy_ret = daily_returns(spy_df)

    # Align
    aligned = pd.DataFrame({"etf": etf_ret, "spy": spy_ret}).dropna()
    if len(aligned) < window + 5:
        return {}

    etf_r = aligned["etf"]
    spy_r = aligned["spy"]

    # Sortino
    etf_sortino = rolling_sortino(etf_r, window)
    spy_sortino = rolling_sortino(spy_r, window)

    # Omega
    etf_omega = rolling_omega(etf_r, window)
    spy_omega = rolling_omega(spy_r, window)

    # CVaR
    etf_cvar = rolling_cvar(etf_r, window)
    spy_cvar = rolling_cvar(spy_r, window)

    # Ulcer Index
    etf_ulcer = rolling_ulcer(etf_df["Close"].reindex(aligned.index), window)
    spy_ulcer = rolling_ulcer(spy_df["Close"].reindex(aligned.index), window)

    # Upside/Downside Capture
    up_cap, down_cap = rolling_updown_capture(etf_r, spy_r, window)

    # Beta
    etf_beta = rolling_beta(etf_r, spy_r, window)

    def _last(s):
        # Return the LATEST bar's value (or None) — never reach back past a trailing NaN, which
        # silently returned a stale earlier-date metric as if it were current.
        if len(s) == 0:
            return None
        v = s.iloc[-1]
        return round(float(v), 3) if pd.notna(v) else None

    def _trend(s):
        """
        Smooth trend: compare 3-day SMA vs 5-day SMA of the metric.
        If fast > slow = trending up, else down.
        """
        v = s.dropna()
        if len(v) < 5:
            return "flat"
        fast = v.iloc[-3:].mean()
        slow = v.iloc[-5:].mean()
        if pd.isna(fast) or pd.isna(slow):
            return "flat"
        diff = fast - slow
        if abs(diff) < 1e-6:
            return "flat"
        return "up" if diff > 0 else "down"

    # Store omega series for crossover validation
    result = {
        "sortino": _last(etf_sortino),
        "spy_sortino": _last(spy_sortino),
        "sortino_trend": _trend(etf_sortino),
        "omega": _last(etf_omega),
        "spy_omega": _last(spy_omega),
        "omega_trend": _trend(etf_omega),
        "omega_series": etf_omega,
        "cvar": _last(etf_cvar),
        "spy_cvar": _last(spy_cvar),
        "ulcer": _last(etf_ulcer),
        "spy_ulcer": _last(spy_ulcer),
        "ulcer_trend": _trend(etf_ulcer),
        "up_capture": _last(up_cap),
        "down_capture": _last(down_cap),
        "down_capture_trend": _trend(down_cap),
        "beta": _last(etf_beta),
    }
    return result


def compute_rsi_crossover(df: pd.DataFrame, omega_series: pd.Series = None) -> dict:
    """
    Compute RSI(RSI_PERIOD=10) and its RSI_SMA_PERIOD(=10)-period SMA.
    Returns latest RSI, RSI SMA, and whether RSI just crossed above its SMA.
    """
    rsi = ta.momentum.rsi(df["Close"], window=RSI_PERIOD)
    rsi_sma = rsi.rolling(window=RSI_SMA_PERIOD).mean()

    latest_rsi = rsi.iloc[-1]
    latest_rsi_sma = rsi_sma.iloc[-1]
    above = latest_rsi > latest_rsi_sma

    # Check if crossover happened in last 3 days
    # Crossover = RSI was below SMA at some point in last 3 days, now above
    # Conditions at time of cross: RSI < 50 AND Omega > 1
    crossover = False
    crossover_days_ago = None
    if above and len(rsi) >= 4 and len(rsi_sma) >= 4:
        for days_back in range(1, 4):  # check 1, 2, 3 days ago
            idx = -(days_back + 1)
            if abs(idx) <= len(rsi):
                prev_r = rsi.iloc[idx]
                prev_s = rsi_sma.iloc[idx]
                if np.isnan(prev_r) or np.isnan(prev_s):
                    continue
                if prev_r > prev_s or prev_r >= 50:
                    continue
                # Check Omega > 1 at the time of cross
                if omega_series is not None:
                    cross_date = rsi.index[idx]
                    if cross_date in omega_series.index:
                        omega_val = omega_series.loc[cross_date]
                        if np.isnan(omega_val) or omega_val <= 1.0:
                            continue
                    else:
                        continue
                crossover = True
                crossover_days_ago = days_back
                break

    return {
        "rsi": round(float(latest_rsi), 2) if not np.isnan(latest_rsi) else None,
        "rsi_sma": round(float(latest_rsi_sma), 2) if not np.isnan(latest_rsi_sma) else None,
        "rsi_above_sma": bool(above),
        "rsi_crossover": bool(crossover),
        "crossover_days_ago": crossover_days_ago,
    }


def compute_rsi_of_sortino(df: pd.DataFrame, window: int = config.SORTINO_WINDOW) -> dict:
    """
    Compute RSI(10) on the rolling Sortino ratio, then its 10-period SMA.
    Same crossover logic as regular RSI: cross above SMA from below 50 = buy.
    """
    ret = daily_returns(df)
    sortino_series = rolling_sortino(ret, window).reindex(df.index)

    # Forward-fill NaN then compute RSI on the full series (matches TradingView)
    sortino_filled = sortino_series.ffill().fillna(0)
    if sortino_filled.notna().sum() < RSI_PERIOD + RSI_SMA_PERIOD + 5:
        return {
            "rsi_sort": None, "rsi_sort_sma": None,
            "rsi_sort_above_sma": False, "rsi_sort_crossover": False,
            "rsi_sort_cross_days_ago": None,
        }

    rsi_sort = ta.momentum.rsi(sortino_filled, window=RSI_PERIOD)
    rsi_sort_sma = rsi_sort.rolling(window=RSI_SMA_PERIOD).mean()

    latest = rsi_sort.iloc[-1]
    latest_sma = rsi_sort_sma.iloc[-1]

    if np.isnan(latest) or np.isnan(latest_sma):
        return {
            "rsi_sort": None, "rsi_sort_sma": None,
            "rsi_sort_above_sma": False, "rsi_sort_crossover": False,
            "rsi_sort_cross_days_ago": None,
        }

    above = latest > latest_sma

    # Crossover in last 3 days — RSI of Sortino crossed above its SMA from below 50
    crossover = False
    cross_days_ago = None
    if above and len(rsi_sort) >= 4:
        for days_back in range(1, 4):
            idx = -(days_back + 1)
            if abs(idx) <= len(rsi_sort):
                prev_r = rsi_sort.iloc[idx]
                prev_s = rsi_sort_sma.iloc[idx]
                if np.isnan(prev_r) or np.isnan(prev_s):
                    continue
                if prev_r <= prev_s and prev_r < 50:
                    crossover = True
                    cross_days_ago = days_back
                    break

    return {
        "rsi_sort": round(float(latest), 2),
        "rsi_sort_sma": round(float(latest_sma), 2),
        "rsi_sort_above_sma": bool(above),
        "rsi_sort_crossover": bool(crossover),
        "rsi_sort_cross_days_ago": cross_days_ago,
        "rsi_sort_series": rsi_sort,
        "rsi_sort_sma_series": rsi_sort_sma,
    }


def detect_gap(df: pd.DataFrame, min_gap_pct: float = 0.5) -> dict:
    """
    Detect if a gap up or gap down happened in the last 3 days.
    Gap up = today's Open > yesterday's High.
    Gap down = today's Open < yesterday's Low.
    Returns the most recent gap found.
    """
    if len(df) < 4:
        return {"gap": False, "gap_dir": None, "gap_days_ago": None, "gap_pct": None}

    for days_back in range(0, 3):
        idx = -(days_back + 1)
        prev_idx = idx - 1
        if abs(prev_idx) > len(df):
            continue

        today_open = df["Open"].iloc[idx]
        prev_high = df["High"].iloc[prev_idx]
        prev_low = df["Low"].iloc[prev_idx]

        if pd.isna(today_open) or pd.isna(prev_high) or pd.isna(prev_low) or prev_high == 0 or prev_low == 0:
            continue

        # Gap up
        gap_up_pct = (today_open - prev_high) / prev_high * 100
        if gap_up_pct >= min_gap_pct:
            return {
                "gap": True,
                "gap_dir": "up",
                "gap_days_ago": days_back,
                "gap_pct": round(gap_up_pct, 2),
            }

        # Gap down
        gap_dn_pct = (today_open - prev_low) / prev_low * 100
        if gap_dn_pct <= -min_gap_pct:
            return {
                "gap": True,
                "gap_dir": "down",
                "gap_days_ago": days_back,
                "gap_pct": round(gap_dn_pct, 2),
            }

    return {"gap": False, "gap_dir": None, "gap_days_ago": None, "gap_pct": None}


def compute_fresh_crossovers(df: pd.DataFrame, window: int = 14,
                             sortino_timeframe: str = "W",
                             weekly_closed_only: bool = True) -> dict:
    """
    Detect the "fresh" composite setup within the last `window` trading days:
      - RSI(10) crossed above its SMA(10), AND
      - RSI-of-Sortino crossed above its SMA(10), AND
      - rolling Sortino(10) is currently > 0.
    (Green MACD is combined by the caller so MACD stays single-sourced.)

    The standalone Sortino>0 gate is evaluated on `sortino_timeframe`:
      "W" = weekly (resample closes to weekly bars, then rolling Sortino) [default]
      "D" = daily.
    RSI and RSI-of-Sortino crossovers stay on the daily timeframe.

    Returns days-ago for each crossover (most recent within the window), the
    current Sortino sign, and which timeframe the Sortino gate used.
    """
    def _cross_days_ago(x, sma):
        cross = (x > sma) & (x.shift(1) <= sma.shift(1))
        tail = cross.iloc[-window:]
        days = None
        for pos, v in enumerate(tail.values):
            if bool(v):
                days = (len(tail) - 1) - pos  # keep most recent -> smallest days-ago
        return days

    # RSI crossover (daily)
    rsi = ta.momentum.rsi(df["Close"], window=RSI_PERIOD)
    rsi_sma = rsi.rolling(RSI_SMA_PERIOD).mean()
    rsi_x_days = _cross_days_ago(rsi, rsi_sma)

    # RSI-of-Sortino crossover (daily Sortino)
    ret = daily_returns(df)
    sortino_series = rolling_sortino(ret, config.SORTINO_WINDOW).reindex(df.index)
    sortino_filled = sortino_series.ffill().fillna(0)
    rs = ta.momentum.rsi(sortino_filled, window=RSI_PERIOD)
    rs_sma = rs.rolling(RSI_SMA_PERIOD).mean()
    rs_x_days = _cross_days_ago(rs, rs_sma)

    # Sortino>0 gate — daily or weekly
    if sortino_timeframe.upper().startswith("W"):
        wk_close = df["Close"].resample("W-FRI").last().dropna()
        # Drop the in-progress week so confirmation uses only COMPLETED weekly
        # bars (no repaint). A week is complete once its last daily bar is Friday.
        if weekly_closed_only and len(df.index) and df.index[-1].dayofweek != 4:
            wk_close = wk_close.iloc[:-1]
        wk_ret = np.log(wk_close / wk_close.shift(1)).dropna()
        wk_sortino = rolling_sortino(wk_ret, config.SORTINO_WINDOW).dropna()
        sortino_val = float(wk_sortino.iloc[-1]) if len(wk_sortino) else None
        tf = "W"
    else:
        sv = sortino_series.dropna()
        sortino_val = float(sv.iloc[-1]) if len(sv) else None
        tf = "D"
    sortino_pos = bool(sortino_val is not None and sortino_val > 0)

    return {
        "fresh_rsi_x_days": rsi_x_days,
        "fresh_rs_x_days": rs_x_days,
        "fresh_sortino_pos": sortino_pos,
        "fresh_sortino_val": round(sortino_val, 3) if sortino_val is not None else None,
        "fresh_sortino_tf": tf,
    }


def compute_macd(df: pd.DataFrame) -> dict:
    """
    Compute daily MACD (12/26/9) and judge whether it is "great".

    "Great" = bullish and confirmed:
      - MACD line is ABOVE its signal line (bullish crossover state), AND
      - MACD line is ABOVE zero (established uptrend, not just a bounce in a downtrend).

    Also reports the histogram (macd - signal) and whether it is expanding
    (momentum accelerating) so the dashboard can show the trend.
    """
    macd_obj = ta.trend.MACD(
        df["Close"],
        window_fast=MACD_FAST,
        window_slow=MACD_SLOW,
        window_sign=MACD_SIGNAL,
    )
    macd_line = macd_obj.macd()
    signal_line = macd_obj.macd_signal()
    hist = macd_obj.macd_diff()

    latest_macd = macd_line.iloc[-1]
    latest_signal = signal_line.iloc[-1]
    latest_hist = hist.iloc[-1]

    if np.isnan(latest_macd) or np.isnan(latest_signal):
        return {
            "macd": None, "macd_signal": None, "macd_hist": None,
            "macd_great": False, "macd_hist_trend": "flat",
        }

    great = bool((latest_macd > latest_signal) and (latest_macd > 0))

    # Histogram trend: is momentum expanding (rising) or contracting (falling)?
    h = hist.dropna()
    hist_trend = "flat"
    if len(h) >= 5:
        fast = h.iloc[-3:].mean()
        slow = h.iloc[-5:].mean()
        if not (pd.isna(fast) or pd.isna(slow)) and abs(fast - slow) >= 1e-9:
            hist_trend = "up" if fast > slow else "down"

    return {
        "macd": round(float(latest_macd), 4),
        "macd_signal": round(float(latest_signal), 4),
        "macd_hist": round(float(latest_hist), 4) if not np.isnan(latest_hist) else None,
        "macd_great": great,
        "macd_hist_trend": hist_trend,
    }
