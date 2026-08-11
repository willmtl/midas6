"""
Stock Market Trend Bot - Trend Analyzer

Two conditions for a sector to be bullish:
  1. RSI(14) > its 14-period SMA
  2. 14-day rolling Sortino > SPY's 14-day rolling Sortino

Also computes: Omega, CVaR, Ulcer Index, Upside/Downside Capture.
"""

import pandas as pd

import config
import data_fetcher
import indicators


def analyze(
    period: str = config.DEFAULT_PERIOD,
    interval: str = config.DEFAULT_INTERVAL,
    window: int = config.SORTINO_WINDOW,
) -> list[dict]:
    all_data = data_fetcher.fetch_all(period, interval)

    spy_df = all_data.get(config.BENCHMARK)
    if spy_df is None:
        raise RuntimeError(f"Could not fetch benchmark data ({config.BENCHMARK})")

    qqq_df = all_data.get("QQQ")
    spy_ret_full = indicators.daily_returns(spy_df)
    qqq_ret_full = indicators.daily_returns(qqq_df) if qqq_df is not None else None

    etf_to_sector = {v: k for k, v in config.SECTOR_ETFS.items()}

    results = []
    for etf_ticker, sector_name in etf_to_sector.items():
        etf_df = all_data.get(etf_ticker)
        if etf_df is None:
            continue

        # All risk metrics (includes omega_series)
        metrics = indicators.compute_all_risk_metrics(etf_df, spy_df, window)
        if not metrics:
            continue

        # Correlation / beta vs both indices (beta-vs-SPY already in metrics as "beta")
        etf_ret = indicators.daily_returns(etf_df)
        corr_spy = indicators.rolling_correlation(etf_ret, spy_ret_full, window).iloc[-1] \
            if len(etf_ret) > window else float("nan")
        if qqq_ret_full is not None and len(etf_ret) > window:
            corr_qqq = indicators.rolling_correlation(etf_ret, qqq_ret_full, window).iloc[-1]
            beta_qqq = indicators.rolling_beta(etf_ret, qqq_ret_full, window).iloc[-1]
        else:
            corr_qqq = float("nan")
            beta_qqq = float("nan")

        # RSI crossover — pass omega series for validation
        rsi_data = indicators.compute_rsi_crossover(etf_df, omega_series=metrics.get("omega_series"))
        # Skip when RSI or its SMA is unavailable (SMA warmup): a None SMA
        # coerced to 0 would make rsi_spread ≈ raw RSI (~60) and float an
        # under-warmed ETF to the top of the bullish ranking.
        if rsi_data.get("rsi") is None or rsi_data.get("rsi_sma") is None:
            continue

        rsi = rsi_data["rsi"]
        rsi_sma = rsi_data["rsi_sma"]
        rsi_spread = round(rsi - rsi_sma, 2)
        rsi_above = rsi_data.get("rsi_above_sma", False)
        crossover = rsi_data.get("rsi_crossover", False)
        crossover_days_ago = rsi_data.get("crossover_days_ago")

        sortino = metrics.get("sortino") or 0
        spy_sortino = metrics.get("spy_sortino") or 0
        sortino_above = sortino > spy_sortino

        omega = metrics.get("omega") or 0
        omega_above_1 = omega > 1.0

        # MACD (daily) — must be "great" (bullish: MACD > signal AND MACD > 0)
        macd_data = indicators.compute_macd(etf_df)
        macd_great = macd_data.get("macd_great", False)
        _m, _ms = macd_data.get("macd"), macd_data.get("macd_signal")
        macd_green = bool(_m is not None and _ms is not None and _m > _ms)

        # ── "Fresh" composite: Sortino>0 + RSI crossover + RSI-of-Sortino
        #    crossover, all inside the last 14 trading days. (MACD removed.)
        #    After 14 days it is no longer considered fresh. ──
        FRESH_WINDOW = 14
        fresh_data = indicators.compute_fresh_crossovers(
            etf_df, window=FRESH_WINDOW, sortino_timeframe="W")
        _f_rsi = fresh_data.get("fresh_rsi_x_days")
        _f_rs = fresh_data.get("fresh_rs_x_days")
        _f_sort = fresh_data.get("fresh_sortino_pos", False)
        # Three component conditions (Sortino gate is WEEKLY)
        _conds = {
            "wk_sortino>0": bool(_f_sort),
            "rsi_x": _f_rsi is not None,
            "rsisort_x": _f_rs is not None,
        }
        _n_pass = sum(_conds.values())
        if _n_pass == 3:
            # completing crossover = the later (larger days-ago) of the two crosses
            fresh_days = max(_f_rsi, _f_rs)
            # Expire: after FRESH_WINDOW days it is no longer fresh
            fresh_state = "FRESH" if fresh_days <= FRESH_WINDOW else None
            if fresh_state is None:
                fresh_days = None
                fresh_since = None
            else:
                # calendar date of the completing crossover bar
                fresh_since = str(etf_df.index[-1 - fresh_days].date())
        elif _n_pass == 2:
            fresh_state = "POTENTIAL"
            fresh_days = None
            fresh_since = None
        else:
            fresh_state = None
            fresh_days = None
            fresh_since = None
        fresh_missing = [k for k, v in _conds.items() if not v]

        # Signal: RSI crossed above SMA AND Omega > 1 (MACD removed)
        both_pass = rsi_above and omega_above_1
        if both_pass:
            signal = "ROTATE IN" if crossover else "BULLISH"
        elif rsi_above and not omega_above_1:
            signal = "RSI ONLY"
        elif omega_above_1 and not rsi_above:
            signal = "OMEGA ONLY"
        else:
            signal = "BEARISH"

        # RSI of Sortino
        rsi_sort_data = indicators.compute_rsi_of_sortino(etf_df, window)

        # Gap detection
        gap_data = indicators.detect_gap(etf_df)

        results.append({
            "sector": sector_name,
            "etf": etf_ticker,
            "rsi": rsi,
            "rsi_sma": rsi_sma,
            "rsi_spread": rsi_spread,
            "rsi_above_sma": rsi_above,
            "rsi_crossover": crossover,
            "crossover_days_ago": crossover_days_ago,
            "sortino": sortino,
            "spy_sortino": spy_sortino,
            "sortino_above": sortino_above,
            "sortino_trend": metrics.get("sortino_trend", "flat"),
            "omega": metrics.get("omega"),
            "spy_omega": metrics.get("spy_omega"),
            "omega_trend": metrics.get("omega_trend", "flat"),
            "cvar": metrics.get("cvar"),
            "spy_cvar": metrics.get("spy_cvar"),
            "ulcer": metrics.get("ulcer"),
            "spy_ulcer": metrics.get("spy_ulcer"),
            "ulcer_trend": metrics.get("ulcer_trend", "flat"),
            "up_capture": metrics.get("up_capture"),
            "down_capture": metrics.get("down_capture"),
            "down_capture_trend": metrics.get("down_capture_trend", "flat"),
            "beta": metrics.get("beta"),
            "corr_spy": round(float(corr_spy), 2) if corr_spy == corr_spy else None,
            "corr_qqq": round(float(corr_qqq), 2) if corr_qqq == corr_qqq else None,
            "beta_qqq": round(float(beta_qqq), 2) if beta_qqq == beta_qqq else None,
            "rsi_sort": rsi_sort_data.get("rsi_sort"),
            "rsi_sort_sma": rsi_sort_data.get("rsi_sort_sma"),
            "rsi_sort_above_sma": rsi_sort_data.get("rsi_sort_above_sma", False),
            "rsi_sort_crossover": rsi_sort_data.get("rsi_sort_crossover", False),
            "rsi_sort_cross_days_ago": rsi_sort_data.get("rsi_sort_cross_days_ago"),
            "gap": gap_data.get("gap", False),
            "gap_dir": gap_data.get("gap_dir"),
            "gap_days_ago": gap_data.get("gap_days_ago"),
            "gap_pct": gap_data.get("gap_pct"),
            "macd": macd_data.get("macd"),
            "macd_signal": macd_data.get("macd_signal"),
            "macd_hist": macd_data.get("macd_hist"),
            "macd_great": macd_great,
            "macd_green": macd_green,
            "macd_hist_trend": macd_data.get("macd_hist_trend", "flat"),
            "fresh_state": fresh_state,
            "fresh_days": fresh_days,
            "fresh_since": fresh_since,
            "fresh_wk_sortino": fresh_data.get("fresh_sortino_val"),
            "fresh_rsi_x_days": _f_rsi,
            "fresh_rs_x_days": _f_rs,
            "fresh_missing": fresh_missing,
            "bullish": both_pass,
            "signal": signal,
        })

    results.sort(key=lambda x: x["rsi_spread"], reverse=True)
    return results
