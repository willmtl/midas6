"""
Background tasks for data import, scan computation, and study execution.
"""

import json
import logging
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import date, datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import ta
import yfinance as yf
from django.utils import timezone

from core.models import Sector, Candle, Study, Trade, ScanResult, StudySectorResult

logger = logging.getLogger(__name__)

# Import sector config
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
try:
    import config
    import indicators
    import sector_holdings
except ImportError:
    config = None
    indicators = None
    sector_holdings = None


# ── Data Import ──

def import_candles_task():
    """Refresh candle data from Yahoo Finance for EVERY tracked ticker — sector ETFs + SPY,
    all commodity anchors/proxies/futures, AND the full individual-stock universe the scanner
    uses. HARD RULE: the updater must cover everything the strategy reads, or the live signals
    silently run on stale prices (that bug left ~1000 stocks frozen for weeks). Ends with a
    freshness assertion that logs any ticker still behind."""
    if config is None:
        logger.error("config module not found")
        return

    tickers = list(config.SECTOR_ETFS.values()) + [config.BENCHMARK]
    # Track ALL commodity prices: theme anchors + every proxy stock + key underlying futures
    # (futures give a real price for no-ETF commodities like lumber/copper/gas).
    comm = set()
    for anchor, ps in COMMODITY_THEMES.values():
        if anchor != "BASKET":
            comm.add(anchor)
        comm.update(ps)
    futures = ["GC=F", "SI=F", "HG=F", "CL=F", "NG=F", "PL=F", "PA=F", "ZC=F", "ZW=F", "ZS=F", "LBS=F", "KC=F", "SB=F", "CT=F", "^VIX"]
    # EVERY ticker that already has candles (⊇ the scanner universe) + the fundamentals universe,
    # so nothing the strategy reads is ever left un-refreshed.
    existing = list(Candle.objects.filter(interval="1d").values_list("ticker", flat=True).distinct())
    try:
        from seq_fundamental_study import build_universe
        universe = list(build_universe())
    except Exception as e:
        logger.warning("build_universe failed in candle import: %s", e); universe = []
    tickers = list(dict.fromkeys(tickers + sorted(comm) + futures + existing + universe))
    today = date.today()

    # Find which tickers need updating. ONE aggregate query for the latest bar of every ticker
    # (was an N+1: one .first() per ticker → ~1150 round-trips every hour). Same pattern as the
    # freshness check below. NB: Candle is a TimescaleDB hypertable with no `id` column.
    from django.db.models import Max
    last_by_ticker = {r["ticker"]: r["last"] for r in
                      Candle.objects.filter(ticker__in=tickers, interval="1d")
                      .values("ticker").annotate(last=Max("date"))}
    need_full = []
    need_incremental = {}
    for ticker in tickers:
        last_date = last_by_ticker.get(ticker)
        if last_date is None:
            need_full.append(ticker)
        elif last_date < today:
            need_incremental[ticker] = last_date
        # else: already up to date

    logger.info(f"Import: {len(need_full)} full, {len(need_incremental)} incremental, "
                f"{len(tickers) - len(need_full) - len(need_incremental)} up to date")

    def _chunks(seq, n=150):
        for i in range(0, len(seq), n):
            yield seq[i:i + n]

    # Full downloads (brand-new tickers) — chunked so a big list doesn't overwhelm one yf call.
    for ch in _chunks(need_full):
        _batch_import(ch, period="5y")

    # Incremental — refetch from just before the OLDEST stale bar. Candle is a managed=False
    # TimescaleDB hypertable (no id col), so bulk_create(update_conflicts=...) can't upsert and
    # plain ignore_conflicts would KEEP the existing row. That silently FREEZES a bar first written
    # mid-session: the hourly job runs during US hours, writes the current day's partial (wrong
    # close/high/low/volume) bar, and every later refetch of that date is dropped as a conflict — so
    # the most decision-relevant bar (latest) never gets its final values. Delete the trailing window
    # first, then re-insert the fresh bars so finals replace partials. (Self-heals next run if a
    # fetch blips.)
    if need_incremental:
        start_date = min(need_incremental.values()) - timedelta(days=2)
        start = start_date.isoformat()
        for ch in _chunks(list(need_incremental.keys())):
            Candle.objects.filter(ticker__in=ch, interval="1d", date__gte=start_date).delete()
            _batch_import(ch, start=start)

    # Also seed Sector table if empty
    if Sector.objects.count() == 0:
        _seed_sectors()

    # HARD FRESHNESS RULE: after the run, verify nothing is left stale (allow a small weekend/
    # holiday grace). Log loudly if any ticker is behind so the pipeline never silently rots.
    from django.db.models import Max
    fresh_cut = today - timedelta(days=5)
    stale = sorted(r["ticker"] for r in
                   Candle.objects.filter(ticker__in=tickers, interval="1d")
                   .values("ticker").annotate(last=Max("date")) if r["last"] and r["last"] < fresh_cut)
    if stale:
        logger.warning("CANDLE FRESHNESS: %d/%d tickers still stale (>5d behind) after import: %s%s",
                       len(stale), len(tickers), stale[:25], " …" if len(stale) > 25 else "")
    else:
        logger.info("CANDLE FRESHNESS: all %d tickers current (<=5d).", len(tickers))
    return {"tickers": len(tickers), "full": len(need_full),
            "incremental": len(need_incremental), "stale_after": len(stale)}


def _batch_import(tickers, period=None, start=None):
    """Download and save candles to DB."""
    kwargs = {"tickers": tickers, "interval": "1d", "group_by": "ticker", "auto_adjust": True, "threads": True, "progress": False}
    if period:
        kwargs["period"] = period
    if start:
        kwargs["start"] = start

    data = yf.download(**kwargs)
    if data is None or data.empty:
        return

    bulk = []
    for ticker in tickers:
        try:
            if len(tickers) == 1:
                df = data.copy()
            else:
                df = data[ticker].copy()

            if isinstance(df.columns, pd.MultiIndex):
                level0 = [c[0] for c in df.columns]
                level1 = [c[1] if len(c) > 1 else c[0] for c in df.columns]
                if len(set(level1)) == 1:
                    df.columns = level0
                elif len(set(level0)) == 1:
                    df.columns = level1
                else:
                    df.columns = level0

            df = df.dropna(how="all")
            for dt, row in df.iterrows():
                d = dt.date() if hasattr(dt, 'date') else dt
                cl = row.get("Close")
                if pd.isna(cl):
                    continue
                cl = float(cl)
                # O/H/L can be NaN even when Close is present (partial vendor rows); float(nan)
                # would store NaN and poison any rolling max/min. Coalesce to the valid close.
                o, h, lo, vol = row.get("Open"), row.get("High"), row.get("Low"), row.get("Volume")
                bulk.append(Candle(
                    ticker=ticker, date=d, interval="1d",
                    open=float(o) if not pd.isna(o) else cl,
                    high=float(h) if not pd.isna(h) else cl,
                    low=float(lo) if not pd.isna(lo) else cl,
                    close=cl,
                    volume=int(vol) if not pd.isna(vol) else 0,
                ))
        except Exception as e:
            logger.warning(f"Failed to import {ticker}: {e}")

    if bulk:
        Candle.objects.bulk_create(bulk, ignore_conflicts=True, batch_size=5000)
        logger.info(f"Imported {len(bulk)} candles")


def _seed_sectors():
    """Create Sector records from config."""
    if config is None:
        return
    for name, etf in config.SECTOR_ETFS.items():
        Sector.objects.get_or_create(name=name, defaults={"etf": etf})
    logger.info(f"Seeded {Sector.objects.count()} sectors")


# ── Scan Computation ──

def _get_df(ticker, interval="1d"):
    """Load candles from DB into a pandas DataFrame."""
    candles = Candle.objects.filter(ticker=ticker, interval=interval).order_by("date").values("date", "open", "high", "low", "close", "volume")
    if not candles:
        return None
    df = pd.DataFrame(list(candles))
    df["date"] = pd.to_datetime(df["date"])
    df = df.set_index("date")
    df.columns = [c.title() for c in df.columns]
    return df


def _get_dfs(tickers, interval="1d"):
    """Bulk-load candles for many tickers in ONE query, grouped into per-ticker DataFrames.
    Avoids the N+1 of calling _get_df per ticker."""
    tickers = list(tickers)
    if not tickers:
        return {}
    rows = (Candle.objects.filter(ticker__in=tickers, interval=interval)
            .values_list("ticker", "date", "open", "high", "low", "close", "volume"))
    big = pd.DataFrame.from_records(
        list(rows), columns=["ticker", "date", "Open", "High", "Low", "Close", "Volume"])
    if big.empty:
        return {}
    big["date"] = pd.to_datetime(big["date"])
    out = {}
    for tk, g in big.groupby("ticker", sort=False):
        out[tk] = g.sort_values("date").set_index("date")[["Open", "High", "Low", "Close", "Volume"]]
    return out


def _last_val(series, ndigits=2):
    """Last value of a Series rounded, or None if empty/NaN."""
    if series is None or len(series) == 0:
        return None
    v = series.iloc[-1]
    return round(float(v), ndigits) if v == v else None  # v == v is False for NaN


def _prepare_study_indicators(df):
    """Attach the precomputed indicator columns signal/exit functions read (so exits
    don't recompute RSI/Sortino every call). Mirrors all_on_all_study._prepare_indicators."""
    import ta
    from studies import _rolling_sortino, _rolling_omega, _rsi_of_sortino
    df["_sortino"] = _rolling_sortino(df)
    df["_omega"] = _rolling_omega(df)
    df["_rsi"] = ta.momentum.rsi(df["Close"], window=10)
    df["_rsi_sma"] = df["_rsi"].rolling(10).mean()
    df["_rsi_sort"] = _rsi_of_sortino(df)
    df["_rsi_sort_sma"] = df["_rsi_sort"].rolling(10).mean()
    return df


def compute_sector_drilldown(sector_name, signal_key, exit_key, recent_window=10):
    """"A signal on a sector becomes a signal on specific stocks in that sector."

    For every stock in the sector's holdings: backtest the given signal×exit, and flag
    whether the signal is FIRING right now (within the last `recent_window` bars). Returns
    per-stock results ranked by avg return, plus the currently-firing shortlist — turning a
    sector-level signal into actionable per-stock signals. Fast enough to run inline (a
    sector is ~20 stocks)."""
    from studies import SIGNALS, EXITS, CLEAN_MAE_THRESH, trade_mae
    import sector_holdings
    if signal_key not in SIGNALS:
        return {"error": f"unknown signal {signal_key}"}
    if exit_key not in EXITS:
        return {"error": f"unknown exit {exit_key}"}
    _, sig_fn = SIGNALS[signal_key]
    _, exit_fn = EXITS[exit_key]

    tickers = sector_holdings.get_holdings(sector_name)
    if not tickers:
        return {"error": f"no holdings for sector {sector_name}", "stocks": [], "firing_now": []}

    dfs = _get_dfs(tickers)  # one bulk query for all holdings, not N+1
    stocks = []
    firing_now = []
    for tk in tickers:
        df = dfs.get(tk)
        if df is None or len(df) < 60:
            continue
        _prepare_study_indicators(df)
        try:
            sig = sig_fn(df).fillna(False)
        except Exception:
            continue
        close = df["Close"].values
        low = df["Low"].values
        n = len(close)
        entry_idxs = [df.index.get_loc(d) for d in sig[sig].index]
        from studies import _episode_starts, _tstat_from_returns
        episode = _episode_starts(entry_idxs)   # overlap-dedup for the significance stat

        rets, holds, maes, eff = [], [], [], []
        for idx in entry_idxs:
            try:
                exit_idx = exit_fn(df, idx)
            except Exception:
                continue
            if exit_idx is None or exit_idx <= idx or exit_idx >= n:
                continue
            ep = float(close[idx])
            if ep <= 0:
                continue
            ret = (float(close[exit_idx]) - ep) / ep * 100
            rets.append(ret)
            holds.append(exit_idx - idx)
            maes.append(trade_mae(ep, low[idx + 1:exit_idx + 1]))
            if idx in episode:
                eff.append(ret)

        # Currently firing? True anywhere in the last recent_window bars.
        recent = sig.iloc[-recent_window:]
        fired = bool(recent.any())
        days_ago = None
        if fired:
            # bars since the most recent True (0 = today's bar)
            true_positions = [i for i, v in enumerate(reversed(recent.tolist())) if v]
            days_ago = true_positions[0] if true_positions else None

        row = {
            "ticker": tk,
            "trades": len(rets),
            "eff_trades": len(eff),
            "t_stat": _tstat_from_returns(eff),
            "avg_return": round(sum(rets) / len(rets), 2) if rets else 0.0,
            "win_rate": round(sum(1 for r in rets if r > 0) / len(rets) * 100, 1) if rets else 0.0,
            "avg_hold": round(sum(holds) / len(holds), 1) if holds else 0.0,
            "avg_mae": round(sum(maes) / len(maes), 2) if maes else 0.0,
            "clean_pct": round(sum(1 for m in maes if m >= CLEAN_MAE_THRESH) / len(maes) * 100, 1) if maes else 0.0,
            "firing": fired,
            "days_ago": days_ago,
            "last_close": round(float(close[-1]), 2),
        }
        stocks.append(row)
        if fired:
            firing_now.append(row)

    stocks.sort(key=lambda r: r["avg_return"], reverse=True)
    firing_now.sort(key=lambda r: (r["days_ago"] if r["days_ago"] is not None else 999, -r["avg_return"]))
    return {
        "sector": sector_name,
        "signal_key": signal_key, "signal_name": SIGNALS[signal_key][0],
        "exit_key": exit_key, "exit_name": EXITS[exit_key][0],
        "recent_window": recent_window,
        "n_stocks": len(stocks), "n_firing": len(firing_now),
        "stocks": stocks, "firing_now": firing_now,
    }


def compute_scan(interval="1d"):
    """Compute scan results for all sectors."""
    if indicators is None:
        return

    # Bulk-load benchmarks + every sector ETF in ONE query (was an N+1: _get_df per sector,
    # ~93 round-trips every hour).
    sectors = list(Sector.objects.all())
    dfs = _get_dfs([config.BENCHMARK, "QQQ"] + [s.etf for s in sectors], interval)

    spy_df = dfs.get(config.BENCHMARK)
    if spy_df is None or len(spy_df) < 30:
        return

    qqq_df = dfs.get("QQQ")
    spy_ret = indicators.daily_returns(spy_df)
    qqq_ret = indicators.daily_returns(qqq_df) if qqq_df is not None else None

    ScanResult.objects.filter(interval=interval).delete()

    for sector in sectors:
        etf_df = dfs.get(sector.etf)
        if etf_df is None or len(etf_df) < 30:
            continue

        try:
            metrics = indicators.compute_all_risk_metrics(etf_df, spy_df, config.SORTINO_WINDOW)
            if not metrics:
                continue

            rsi_data = indicators.compute_rsi_crossover(etf_df, omega_series=metrics.get("omega_series"))
            if rsi_data.get("rsi") is None:
                continue

            gap_data = indicators.detect_gap(etf_df)

            # Correlation / beta vs both indices (beta-vs-SPY already in metrics)
            w = config.SORTINO_WINDOW
            etf_ret = indicators.daily_returns(etf_df)
            corr_spy = _last_val(indicators.rolling_correlation(etf_ret, spy_ret, w)) \
                if len(etf_ret) > w else None
            if qqq_ret is not None and len(etf_ret) > w:
                corr_qqq = _last_val(indicators.rolling_correlation(etf_ret, qqq_ret, w))
                beta_qqq = _last_val(indicators.rolling_beta(etf_ret, qqq_ret, w))
            else:
                corr_qqq = beta_qqq = None

            omega = metrics.get("omega") or 0
            rsi_above = rsi_data.get("rsi_above_sma", False)
            omega_above = omega > 1
            both = rsi_above and omega_above
            crossover = rsi_data.get("rsi_crossover", False)

            if both:
                signal = "ROTATE IN" if crossover else "BULLISH"
            elif rsi_above:
                signal = "RSI ONLY"
            elif omega_above:
                signal = "OMEGA ONLY"
            else:
                signal = "BEARISH"

            ScanResult.objects.create(
                sector=sector, interval=interval,
                rsi=rsi_data.get("rsi"),
                rsi_sma=rsi_data.get("rsi_sma"),
                rsi_spread=round((rsi_data.get("rsi") or 0) - (rsi_data.get("rsi_sma") or 0), 2),
                rsi_above_sma=rsi_above,
                rsi_crossover=crossover,
                crossover_days_ago=rsi_data.get("crossover_days_ago"),
                sortino=metrics.get("sortino"),
                spy_sortino=metrics.get("spy_sortino"),
                sortino_trend=metrics.get("sortino_trend", "flat"),
                omega=metrics.get("omega"),
                spy_omega=metrics.get("spy_omega"),
                omega_trend=metrics.get("omega_trend", "flat"),
                cvar=metrics.get("cvar"),
                spy_cvar=metrics.get("spy_cvar"),
                ulcer=metrics.get("ulcer"),
                spy_ulcer=metrics.get("spy_ulcer"),
                ulcer_trend=metrics.get("ulcer_trend", "flat"),
                up_capture=metrics.get("up_capture"),
                down_capture=metrics.get("down_capture"),
                down_capture_trend=metrics.get("down_capture_trend", "flat"),
                beta=metrics.get("beta"),
                corr_spy=corr_spy,
                corr_qqq=corr_qqq,
                beta_qqq=beta_qqq,
                gap=gap_data.get("gap", False),
                gap_dir=gap_data.get("gap_dir"),
                gap_days_ago=gap_data.get("gap_days_ago"),
                gap_pct=gap_data.get("gap_pct"),
                signal=signal,
                bullish=both,
            )
        except Exception as e:
            logger.warning(f"Scan failed for {sector.name}: {e}")


def compute_drilldown(sector):
    """Compute stock-level drill-down for a sector."""
    if sector_holdings is None:
        return {"sector": sector.name, "etf": sector.etf, "stocks": []}

    holdings = sector_holdings.get_holdings(sector.name)
    if not holdings:
        return {"sector": sector.name, "etf": sector.etf, "stocks": []}

    us_tickers = [t for t in holdings if "." not in t][:20]
    results = []

    # Bulk-load all holdings in ONE query (was an N+1: _get_df per holding). Missing tickers are
    # simply skipped — the nightly candle task backfills the whole universe, so we NEVER trigger a
    # live yfinance download on the request thread (that could stall a web request for seconds).
    dfs = _get_dfs(us_tickers)

    for ticker in us_tickers:
        df = dfs.get(ticker)
        if df is None or len(df) < 25:
            continue

        try:
            rsi_data = indicators.compute_rsi_crossover(df)
            if rsi_data.get("rsi") is None:
                continue

            gap_data = indicators.detect_gap(df)
            close = float(df["Close"].iloc[-1])
            ret_1w = float((df["Close"].iloc[-1] / df["Close"].iloc[-6] - 1) * 100) if len(df) > 6 else 0
            ret_1m = float((df["Close"].iloc[-1] / df["Close"].iloc[-22] - 1) * 100) if len(df) > 22 else 0

            rsi = rsi_data["rsi"]
            rsi_sma = rsi_data["rsi_sma"] or 0
            rsi_above = rsi_data.get("rsi_above_sma", False)
            crossover = rsi_data.get("rsi_crossover", False)
            signal = "ROTATE IN" if (rsi_above and crossover) else ("BULLISH" if rsi_above else "BEARISH")

            results.append({
                "ticker": ticker, "price": round(close, 2),
                "return_1w": round(ret_1w, 1), "return_1m": round(ret_1m, 1),
                "rsi": rsi, "rsi_sma": rsi_sma,
                "rsi_spread": round(rsi - rsi_sma, 2),
                "rsi_above_sma": rsi_above, "rsi_crossover": crossover,
                "crossover_days_ago": rsi_data.get("crossover_days_ago"),
                "gap": gap_data.get("gap", False),
                "gap_dir": gap_data.get("gap_dir"),
                "gap_days_ago": gap_data.get("gap_days_ago"),
                "gap_pct": gap_data.get("gap_pct"),
                "signal": signal,
            })
        except Exception:
            continue

    results.sort(key=lambda x: x["rsi_spread"], reverse=True)
    return {"sector": sector.name, "etf": sector.etf, "stocks": results}


def get_chart_data(ticker, interval="1d", sector_etf=None, period="5y"):
    """Get chart candle data with indicators."""
    df = _get_df(ticker, interval)
    if df is None or len(df) < 20:
        return None

    w = config.SORTINO_WINDOW if config else 10
    rsi = ta.momentum.rsi(df["Close"], window=10)
    rsi_sma = rsi.rolling(10).mean()
    ret = indicators.daily_returns(df)

    sortino = indicators.rolling_sortino(ret, w).reindex(df.index)
    omega = indicators.rolling_omega(ret, w).reindex(df.index)
    cvar = indicators.rolling_cvar(ret, w).reindex(df.index)
    ulcer = indicators.rolling_ulcer(df["Close"], w)

    spy_df = _get_df(config.BENCHMARK, interval)
    beta_s = spy_sortino_s = spy_omega_s = spy_norm = None
    if spy_df is not None:
        spy_ret = indicators.daily_returns(spy_df)
        beta_s = indicators.rolling_beta(ret, spy_ret, w).reindex(df.index)
        spy_sortino_s = indicators.rolling_sortino(spy_ret, w).reindex(df.index)
        spy_omega_s = indicators.rolling_omega(spy_ret, w).reindex(df.index)
        up_s, dn_s = indicators.rolling_updown_capture(ret, spy_ret, w)
        up_s = up_s.reindex(df.index)
        dn_s = dn_s.reindex(df.index)
        spy_close = spy_df["Close"].reindex(df.index)
        fv = spy_close.first_valid_index()
        if fv:
            spy_norm = spy_close / spy_close.loc[fv] * df["Close"].loc[fv]
    else:
        up_s = dn_s = None

    sect_norm = sect_sortino_s = sect_omega_s = None
    if sector_etf:
        sect_df = _get_df(sector_etf.upper(), interval)
        if sect_df is not None:
            sect_ret = indicators.daily_returns(sect_df)
            sect_sortino_s = indicators.rolling_sortino(sect_ret, w).reindex(df.index)
            sect_omega_s = indicators.rolling_omega(sect_ret, w).reindex(df.index)
            sect_close = sect_df["Close"].reindex(df.index)
            fv = sect_close.first_valid_index()
            if fv:
                sect_norm = sect_close / sect_close.loc[fv] * df["Close"].loc[fv]

    def _v(s, i):
        if s is None or i >= len(s): return None
        v = s.iloc[i]
        return round(float(v), 3) if v == v else None

    candles = []
    for i, (dt, row) in enumerate(df.iterrows()):
        rec = {"date": str(dt)[:10], "open": round(float(row["Open"]), 2), "high": round(float(row["High"]), 2), "low": round(float(row["Low"]), 2), "close": round(float(row["Close"]), 2), "volume": int(row["Volume"])}
        if i < len(rsi) and rsi.iloc[i] == rsi.iloc[i]: rec["rsi"] = round(float(rsi.iloc[i]), 2)
        if i < len(rsi_sma) and rsi_sma.iloc[i] == rsi_sma.iloc[i]: rec["rsi_sma"] = round(float(rsi_sma.iloc[i]), 2)
        for key, series in [("sortino", sortino), ("omega", omega), ("cvar", cvar), ("ulcer", ulcer), ("beta", beta_s), ("spy_sortino", spy_sortino_s), ("spy_omega", spy_omega_s), ("sect_sortino", sect_sortino_s), ("sect_omega", sect_omega_s)]:
            v = _v(series, i)
            if v is not None: rec[key] = v
        for key, series in [("up_capture", up_s), ("dn_capture", dn_s), ("spy_price", spy_norm), ("sect_price", sect_norm)]:
            v = _v(series, i)
            if v is not None: rec[key] = round(v, 2 if "price" in key else 1)
        if i > 0:
            prev_high, prev_low = df["High"].iloc[i-1], df["Low"].iloc[i-1]
            cur_open = row["Open"]
            if prev_high > 0 and prev_low > 0:
                gu = (cur_open - prev_high) / prev_high * 100
                gd = (cur_open - prev_low) / prev_low * 100
                if gu >= 0.5: rec["gap"] = round(float(gu), 2)
                elif gd <= -0.5: rec["gap"] = round(float(gd), 2)
        candles.append(rec)

    return {"ticker": ticker, "sector_etf": sector_etf, "candles": candles}


# ── Study Runner (multithreaded) ──

def run_stock_studies_task(jobs=16, min_trades=20):
    # jobs capped at 16: each worker holds its own full buckets dict (~1GB at 33 dims),
    # so 32 workers OOM the box; 16 fits ~16GB. Raise only if dims shrink or RAM grows.
    """Run the all-on-all stock sweep (every signal × exit over ~1035 stocks + fundamental
    buckets) as a clean SUBPROCESS.

    Why a subprocess and not an in-process import: all_on_all_study.py parallelizes with
    multiprocessing 'spawn'. Spawn re-imports the child's __main__; if we ran the sweep
    inside this Django/Celery process, the spawned workers would try to re-import the
    server's __main__ (runserver/celery) instead of the script. Launching it as
    `python all_on_all_study.py` gives spawn a clean script __main__, exactly like the
    manual command. Output JSON lands at /app/.data/studies/stock_studies_all.json, which
    StockStudiesView serves.
    """
    import subprocess
    script = "/app/all_on_all_study.py"
    if not os.path.exists(script):
        logger.error("run_stock_studies_task: %s not found (mount it in docker-compose)", script)
        return
    cmd = ["python", "-u", script, "--db", "--min-trades", str(min_trades)]
    if jobs:
        cmd += ["--jobs", str(jobs)]
    logger.info("run_stock_studies_task: launching %s", " ".join(cmd))
    proc = subprocess.run(cmd, cwd="/app", capture_output=True, text=True)
    if proc.returncode != 0:
        logger.error("stock studies sweep failed (rc=%s): %s", proc.returncode, proc.stderr[-2000:])
    else:
        logger.info("stock studies sweep done: %s", proc.stdout[-1000:])
    return proc.returncode


def run_live_firing_task(recent=10, top_n=12, jobs=None):
    """Run the firing-now scan as a SUBPROCESS (same spawn reasoning as
    run_stock_studies_task). Writes LiveSignal rows + JSON cache."""
    import subprocess
    script = "/app/live_firing_scan.py"
    if not os.path.exists(script):
        logger.error("run_live_firing_task: %s not found", script)
        return
    cmd = ["python", "-u", script, "--db", "--recent", str(recent), "--top", str(top_n)]
    if jobs:
        cmd += ["--jobs", str(jobs)]
    logger.info("run_live_firing_task: %s", " ".join(cmd))
    proc = subprocess.run(cmd, cwd="/app", capture_output=True, text=True)
    if proc.returncode != 0:
        logger.error("firing scan failed (rc=%s): %s", proc.returncode, proc.stderr[-2000:])
    else:
        logger.info("firing scan done: %s", proc.stdout[-800:])
    return proc.returncode


def run_news_horizon_scan_task():
    """Run the news-horizon scan (recent material news joined to horizon-conditioned drift) as a
    SUBPROCESS. Writes NewsHorizonSignal rows. Depends on classified news being present."""
    import subprocess
    script = "/app/news_horizon_scan.py"
    if not os.path.exists(script):
        logger.error("run_news_horizon_scan_task: %s not found", script)
        return
    proc = subprocess.run(["python", "-u", script], cwd="/app", capture_output=True, text=True)
    if proc.returncode != 0:
        logger.error("news-horizon scan failed (rc=%s): %s", proc.returncode, proc.stderr[-2000:])
    else:
        logger.info("news-horizon scan done: %s", proc.stdout[-400:])
    return proc.returncode


AD_CAPIT_SIGNALS = ["new_52low", "rsi_oversold20"]  # pure price-capitulation triggers

def is_low_quality(market_cap, price, margin):
    """Landmine filter. The −33%+ disasters were overwhelmingly tiny, cheap, unprofitable,
    hyper-volatile microcaps already deep in a crash. Flag (and by default exclude) names that
    are sub-$300M, penny (<$5), or explicitly unprofitable — knowable at entry, and it collapses
    the catastrophic tail (<−33% from 5.7%→1.3%) while raising win rate and median."""
    if market_cap is None or market_cap < 300e6:
        return True
    if price is not None and price < 5:
        return True
    if margin is not None and margin < 0:
        return True
    return False


def collect_options_snapshot(min_mcap=2e9, limit=None):
    """Nightly: snapshot a compact options summary (30d ATM IV, put/call volume & OI ratios) for
    the LIQUID US universe, so we accumulate our OWN history (yfinance has no options history to
    backtest). US-listed only (no '.' foreign suffix), mcap>=min_mcap (options need liquidity).
    Upserts one OptionSnapshot per ticker per day. No signal yet — this is the seed; in ~6-12
    months we can validate a single-stock IV/put-call signal the way we validated VIX."""
    import numpy as np, datetime as dt, time as _t
    import yfinance as yf
    from core.models import OptionSnapshot
    from seq_fundamental_study import build_universe, load_fundamentals

    tks = build_universe()
    funds = load_fundamentals(tks)
    us = [t for t in tks if "." not in t and (funds.get(t, {}).get("market_cap") or 0) >= min_mcap]
    if limit:
        us = us[:limit]
    today = date.today()
    saved = 0; errs = 0
    for tk in us:
        try:
            t = yf.Ticker(tk)
            exps = t.options
            if not exps:
                continue
            spot = None
            try:
                spot = float(t.fast_info["last_price"])
            except Exception:
                pass
            near = [e for e in exps if 0 <= (dt.date.fromisoformat(e) - today).days <= 45] or list(exps[:2])
            target = min(exps, key=lambda e: abs((dt.date.fromisoformat(e) - today).days - 30))
            pv = cv = poi = coi = 0.0; ivs = []
            for e in near:
                try:
                    ch = t.option_chain(e)
                except Exception:
                    continue
                pv += ch.puts["volume"].fillna(0).sum(); cv += ch.calls["volume"].fillna(0).sum()
                poi += ch.puts["openInterest"].fillna(0).sum(); coi += ch.calls["openInterest"].fillna(0).sum()
                if e == target and spot:
                    for dfc in (ch.calls, ch.puts):
                        if len(dfc):
                            i = (dfc["strike"] - spot).abs().idxmin()
                            iv = dfc.loc[i, "impliedVolatility"]
                            if iv and iv > 0:
                                ivs.append(float(iv))
            OptionSnapshot.objects.update_or_create(
                ticker=tk, date=today,
                defaults=dict(spot=round(spot, 2) if spot else None,
                              atm_iv=round(float(np.mean(ivs)) * 100, 1) if ivs else None,
                              pc_vol=round(pv / cv, 3) if cv else None,
                              pc_oi=round(poi / coi, 3) if coi else None,
                              n_exp=len(exps)))
            saved += 1
            _t.sleep(0.2)
        except Exception as ex:
            errs += 1
            if errs <= 5:
                logger.warning("options snapshot %s failed: %s", tk, ex)
    logger.info("collect_options_snapshot: %d saved, %d errors, universe=%d US>=%.0fB",
                saved, errs, len(us), min_mcap / 1e9)
    return {"saved": saved, "errors": errs, "universe": len(us)}


# ── Polygon / Massive integration (options IV/OI/Greeks + dark-pool off-exchange flow) ──
# All key-driven via POLYGON_API_KEY env; nothing runs without it. Advanced plan covers both
# options snapshots and trade-level (dark-pool) data. Endpoints/field names verified against the
# public API docs; confirm on first live run and adjust if the response shape differs.

def _polygon_get(path, **params):
    """GET the Polygon/Massive REST API (https://api.polygon.io). Key from POLYGON_API_KEY env.
    `path` may be a leading-slash path or a full next_url. Returns parsed JSON or None."""
    import os, json, urllib.request, urllib.parse
    key = os.environ.get("POLYGON_API_KEY")
    if not key:
        logger.warning("POLYGON_API_KEY not set — Polygon call skipped")
        return None
    if path.startswith("http"):
        url = path + ("&" if "?" in path else "?") + "apiKey=" + key
    else:
        params["apiKey"] = key
        url = "https://api.polygon.io" + path + "?" + urllib.parse.urlencode(params)
    req = urllib.request.Request(url, headers={"User-Agent": "rotation/1.0"})
    try:
        with urllib.request.urlopen(req, timeout=30) as r:
            return json.loads(r.read().decode("utf-8", "replace"))
    except Exception as e:
        logger.warning("polygon GET failed (%s): %s", path[:60], e)
        return None


def _polygon_paginate(path, cap=8000, **params):
    """Follow next_url pagination, returning the concatenated results[] (up to `cap`)."""
    import time as _t
    out = []
    d = _polygon_get(path, **params)
    while d and d.get("results"):
        out.extend(d["results"])
        if len(out) >= cap or not d.get("next_url"):
            break
        _t.sleep(0.05)
        d = _polygon_get(d["next_url"])
    return out


def collect_options_polygon(min_mcap=2e9, limit=None):
    """FORWARD daily options collector via Polygon's option-chain SNAPSHOT — OI + IV + Greeks in one
    call per underlying. Aggregates to: 30d ATM IV, put/call OI & volume ratios, IV skew (put−call
    near-ATM), and dealer GEX (Σ gamma·OI·100·spot²·0.01, calls + / puts −). A clean upgrade over the
    yfinance collector (real OI + Greeks). Snapshot is CURRENT → builds history forward; for backtest
    history use backfill_options_polygon(). Stores OptionSnapshot(source='polygon')."""
    import os, numpy as np, time as _t
    from datetime import date
    from core.models import OptionSnapshot
    from seq_fundamental_study import build_universe, load_fundamentals
    if not os.environ.get("POLYGON_API_KEY"):
        return {"error": "POLYGON_API_KEY not set"}
    tks = build_universe(); funds = load_fundamentals(tks)
    us = [t for t in tks if "." not in t and (funds.get(t, {}).get("market_cap") or 0) >= min_mcap]
    if limit:
        us = us[:limit]
    today = date.today(); saved = 0; errs = 0
    for tk in us:
        try:
            rows = _polygon_paginate(f"/v3/snapshot/options/{tk}", limit=250)
            if not rows:
                continue
            spot = next((float(r["underlying_asset"]["price"]) for r in rows
                         if r.get("underlying_asset", {}).get("price")), None)
            c_oi = p_oi = c_vol = p_vol = 0.0; gex = 0.0; c_iv = []; p_iv = []; exps = set()
            for r in rows:
                det = r.get("details", {}); typ = det.get("contract_type")
                exps.add(det.get("expiration_date"))
                oi = r.get("open_interest") or 0
                dv = (r.get("day") or {}).get("volume") or 0
                iv = r.get("implied_volatility"); gamma = (r.get("greeks") or {}).get("gamma")
                strike = det.get("strike_price")
                near = bool(spot and strike and abs(strike / spot - 1) <= 0.05)
                if typ == "call":
                    c_oi += oi; c_vol += dv
                    if near and iv: c_iv.append(iv)
                    if gamma and spot: gex += gamma * oi * 100 * spot * spot * 0.01
                elif typ == "put":
                    p_oi += oi; p_vol += dv
                    if near and iv: p_iv.append(iv)
                    if gamma and spot: gex -= gamma * oi * 100 * spot * spot * 0.01
            atm = (c_iv + p_iv)
            OptionSnapshot.objects.update_or_create(
                ticker=tk, date=today, defaults=dict(
                    spot=round(spot, 2) if spot else None,
                    atm_iv=round(float(np.mean(atm)) * 100, 1) if atm else None,
                    pc_vol=round(p_vol / c_vol, 3) if c_vol else None,
                    pc_oi=round(p_oi / c_oi, 3) if c_oi else None,
                    iv_skew=round((float(np.mean(p_iv)) - float(np.mean(c_iv))) * 100, 2) if (c_iv and p_iv) else None,
                    gex=round(gex, 0) if gex else None,
                    n_exp=len([e for e in exps if e]), source="polygon"))
            saved += 1; _t.sleep(0.1)
        except Exception as ex:
            errs += 1
            if errs <= 5:
                logger.warning("polygon options %s: %s", tk, ex)
    logger.info("collect_options_polygon: %d saved, %d errs", saved, errs)
    return {"saved": saved, "errors": errs, "source": "polygon"}


def _s3_massive():
    import os, boto3
    return boto3.client("s3", endpoint_url=os.environ.get("POLYGON_S3_ENDPOINT", "https://files.massive.com"),
                        aws_access_key_id=os.environ.get("POLYGON_S3_KEY"),
                        aws_secret_access_key=os.environ.get("POLYGON_S3_SECRET"))


def _parse_opt(t):
    """'O:AAPL260810C00205000' → ('AAPL', date(2026,8,10), 'call', 205.0). Underlying is the
    variable-length prefix; last 8 digits = strike×1000, preceding char = C/P, preceding 6 = YYMMDD."""
    from datetime import date as _d
    s = t[2:] if t.startswith("O:") else t
    try:
        strike = int(s[-8:]) / 1000.0
        cp = s[-9]
        ymd = s[-15:-9]
        und = s[:-15]
        exp = _d(2000 + int(ymd[:2]), int(ymd[2:4]), int(ymd[4:6]))
        return und, exp, ("call" if cp == "C" else "put"), strike
    except Exception:
        return None


def _bs_iv(price, S, K, T, r, typ):
    """Black–Scholes implied vol via bisection (European; fine for near-ATM/near-dated aggregates)."""
    from math import log, sqrt, exp
    from statistics import NormalDist
    if not price or price <= 0 or S <= 0 or K <= 0 or T <= 0:
        return None
    N = NormalDist().cdf
    intrinsic = max(0.0, (S - K) if typ == "call" else (K - S))
    if price < intrinsic - 0.02:
        return None

    def bs(sig):
        d1 = (log(S / K) + (r + sig * sig / 2) * T) / (sig * sqrt(T)); d2 = d1 - sig * sqrt(T)
        return (S * N(d1) - K * exp(-r * T) * N(d2)) if typ == "call" else (K * exp(-r * T) * N(-d2) - S * N(-d1))
    lo, hi = 1e-3, 5.0
    if bs(hi) < price:
        return None
    for _ in range(48):
        mid = (lo + hi) / 2
        if bs(mid) > price:
            hi = mid
        else:
            lo = mid
    return (lo + hi) / 2


def backfill_options_polygon(start, end, tickers=None, r=0.043, dmin=10, dmax=55, band=0.08):
    """HISTORICAL options backfill from Massive day-aggregate FLAT FILES (one gz file/day, all OPRA
    contracts: ticker,volume,open,close,high,low,...). Per underlying per day computes: put/call
    VOLUME ratio, and BSM-implied ATM IV + skew (put_IV−call_IV) from near-ATM (|K/S−1|≤band),
    near-dated (dmin..dmax days) contracts, using the underlying close from our Candle DB.
    NO OI in flat files → pc_oi & GEX are NOT backfilled (forward-only). Stores
    OptionSnapshot(source='polygon_hist'). Iterate business days start..end (YYYY-MM-DD)."""
    import os, gzip, io, csv
    import numpy as np, pandas as pd
    from datetime import datetime, timedelta
    from core.models import OptionSnapshot, Candle
    from seq_fundamental_study import build_universe
    if not os.environ.get("POLYGON_S3_KEY"):
        return {"error": "POLYGON_S3_KEY not set"}
    uni = set(tickers) if tickers else set(t for t in build_universe() if "." not in t)
    d0 = datetime.strptime(start, "%Y-%m-%d").date(); d1 = datetime.strptime(end, "%Y-%m-%d").date()
    # underlying closes for the whole window, one query → {(ticker,date): close}
    cl = {(c["ticker"], c["date"]): c["close"] for c in
          Candle.objects.filter(ticker__in=uni, interval="1d", date__range=(d0 - timedelta(days=5), d1))
          .values("ticker", "date", "close")}
    s3 = _s3_massive(); days = 0; saved = 0
    dd = d0
    while dd <= d1:
        if dd.weekday() >= 5:
            dd += timedelta(days=1); continue
        key = f"us_options_opra/day_aggs_v1/{dd:%Y/%m/%Y-%m-%d}.csv.gz"
        try:
            obj = s3.get_object(Bucket="flatfiles", Key=key)
        except Exception:
            dd += timedelta(days=1); continue   # holiday / missing
        rd = csv.reader(io.StringIO(gzip.decompress(obj["Body"].read()).decode("utf-8", "replace")))
        next(rd, None)
        acc = {}   # und -> dict of sums/lists
        for row in rd:
            if not row:
                continue
            t = row[0]
            if not t.startswith("O:"):
                continue
            p = _parse_opt(t)
            if not p:
                continue
            und, exp, typ, strike = p
            if und not in uni:
                continue
            T = (exp - dd).days
            if T < dmin or T > dmax:
                continue
            S = cl.get((und, dd))
            if not S:
                continue
            # columns: 0=ticker,1=volume,2=open,3=close,4=high,5=low
            vol = int(row[1] or 0); close = float(row[3] or 0)
            a = acc.setdefault(und, {"cv": 0, "pv": 0, "civ": [], "piv": []})
            if typ == "call":
                a["cv"] += vol
            else:
                a["pv"] += vol
            if abs(strike / S - 1) <= band:
                iv = _bs_iv(close, S, strike, T / 365.0, r, typ)
                if iv and 0.03 < iv < 3.0:
                    (a["civ"] if typ == "call" else a["piv"]).append(iv)
        for und, a in acc.items():
            atm = a["civ"] + a["piv"]
            OptionSnapshot.objects.update_or_create(
                ticker=und, date=dd, defaults=dict(
                    spot=round(cl.get((und, dd)), 2) if cl.get((und, dd)) else None,
                    atm_iv=round(float(np.mean(atm)) * 100, 1) if atm else None,
                    pc_vol=round(a["pv"] / a["cv"], 3) if a["cv"] else None,
                    pc_oi=None,
                    iv_skew=round((float(np.mean(a["piv"])) - float(np.mean(a["civ"]))) * 100, 2) if (a["civ"] and a["piv"]) else None,
                    gex=None, n_exp=0, source="polygon_hist"))
            saved += 1
        days += 1
        dd += timedelta(days=1)
    logger.info("backfill_options_polygon: %d day-files, %d ticker-days saved", days, saved)
    return {"day_files": days, "saved": saved}


def collect_darkpool_polygon(tickers, day=None, block_min=5000):
    """Daily off-exchange (dark-pool) volume per stock from Polygon's trade tape. Off-exchange =
    trades with exchange==4 AND a trf_id (TRF-reported). Sums off-exchange vs total volume →
    off_pct; and off-exchange volume in ≥block_min-share prints → block_off_vol (institutional
    proxy; retail internalization is small-lot). Uses /v3/trades/{ticker} for one day. Heavy per
    ticker (thousands of prints) → for BULK history use backfill_darkpool_flatfiles(). Stores
    DarkPoolDay. NOTE: blended tape (dark pool + retail), no per-ATS, no signed side."""
    import os
    from datetime import date as _date
    from core.models import DarkPoolDay
    if not os.environ.get("POLYGON_API_KEY"):
        return {"error": "POLYGON_API_KEY not set"}
    day = day or _date.today().isoformat()
    saved = 0; errs = 0
    for tk in tickers:
        try:
            rows = _polygon_paginate(f"/v3/trades/{tk}", cap=200000,
                                     timestamp=day, limit=50000, order="asc")
            if not rows:
                continue
            tot = off = boff = 0
            for tr in rows:
                sz = tr.get("size") or 0
                exs = tr.get("exchange")
                is_off = (exs == 4) and bool(tr.get("trf_id") is not None)
                tot += sz
                if is_off:
                    off += sz
                    if sz >= block_min:
                        boff += sz
            DarkPoolDay.objects.update_or_create(
                ticker=tk, date=day, defaults=dict(
                    total_vol=tot, off_vol=off, off_pct=round(off / tot, 4) if tot else None,
                    block_off_vol=boff, block_min=block_min, source="polygon"))
            saved += 1
        except Exception as ex:
            errs += 1
            if errs <= 5:
                logger.warning("polygon darkpool %s: %s", tk, ex)
    logger.info("collect_darkpool_polygon: %d saved, %d errs (day=%s)", saved, errs, day)
    return {"saved": saved, "errors": errs, "day": day}


def backfill_darkpool_flatfiles(dates, tickers=None, block_min=5000, chunk_rows=4_000_000):
    """BULK historical dark-pool backfill from Polygon's daily trade FLAT FILES (S3 — the whole US
    consolidated tape, ~3.4GB gzip/day). For each day, STREAM the gzip and pandas-parse in chunks
    (memory-safe), summing off-exchange (exchange==4 & trf_id!=0) vs total volume per ticker →
    DarkPoolDay. ONE S3 GET per day covers all tickers (vs 668 REST calls/day).

    RESUMABLE: any date already present in DarkPoolDay is skipped, so this can grind through years
    of history across restarts without redoing work — process most-recent-first so the useful data
    lands early. `dates` = iterable of 'YYYY-MM-DD'. Needs POLYGON_S3_KEY/POLYGON_S3_SECRET + boto3
    + pandas. Schema: ticker,conditions,correction,exchange,id,participant_timestamp,price,
    sequence_number,sip_timestamp,size,tape,trf_id,trf_timestamp (size is float-formatted)."""
    import os, gzip
    import pandas as pd
    from core.models import DarkPoolDay
    if not (os.environ.get("POLYGON_S3_KEY") and os.environ.get("POLYGON_S3_SECRET")):
        return {"error": "POLYGON_S3_KEY/POLYGON_S3_SECRET not set"}
    if tickers is None:
        from seq_fundamental_study import build_universe, load_fundamentals
        tks = build_universe(); f = load_fundamentals(tks)
        tickers = [t for t in tks if "." not in t and (f.get(t, {}).get("market_cap") or 0) >= 2e9]
    tset = set(tickers)
    s3 = _s3_massive()
    # The endpoint caps a SINGLE stream at ~1 MB/s but 8-way parallel hits ~6-7 MB/s, so use boto3's
    # managed multipart transfer (concurrent ranged GETs) to download each day → ~9 min vs ~60 min.
    import tempfile
    from boto3.s3.transfer import TransferConfig
    xfer = TransferConfig(max_concurrency=8, multipart_threshold=16 * 1024 * 1024,
                          multipart_chunksize=16 * 1024 * 1024, use_threads=True)
    days_done = skipped = missing = saved = 0
    for day in dates:
        # Resumable, but only skip days THIS backfiller already did — a day with just partial
        # REST rows (source='polygon') must still be fully backfilled from the flat file.
        if DarkPoolDay.objects.filter(date=day, source="polygon_flatfile").exists():
            skipped += 1; continue
        y, m, _d = day.split("-")
        key = f"us_stocks_sip/trades_v1/{y}/{m}/{day}.csv.gz"
        tmp = os.path.join(tempfile.gettempdir(), f"dp_{day}.csv.gz")
        try:
            s3.download_file("flatfiles", key, tmp, Config=xfer)   # parallel multipart → disk
        except Exception as e:
            missing += 1
            logger.info("darkpool flatfile %s absent (holiday/weekend?): %s", day, str(e)[:60])
            continue
        tot = {}; off = {}; boff = {}
        try:
            for chunk in pd.read_csv(tmp, compression="gzip",
                                     usecols=["ticker", "size", "exchange", "trf_id"],
                                     dtype={"ticker": "string"}, chunksize=chunk_rows):
                chunk = chunk[chunk["ticker"].isin(tset)]
                if chunk.empty:
                    continue
                sz = chunk["size"].astype("float64")
                isoff = (chunk["exchange"] == 4) & (chunk["trf_id"].fillna(0) != 0)
                for t, v in sz.groupby(chunk["ticker"]).sum().items():
                    tot[t] = tot.get(t, 0.0) + float(v)
                for t, v in sz[isoff].groupby(chunk["ticker"][isoff]).sum().items():
                    off[t] = off.get(t, 0.0) + float(v)
                blk = isoff & (sz >= block_min)
                for t, v in sz[blk].groupby(chunk["ticker"][blk]).sum().items():
                    boff[t] = boff.get(t, 0.0) + float(v)
        except Exception as e:
            logger.warning("darkpool flatfile %s parse err: %s", day, str(e)[:150])
            continue
        finally:
            try:
                os.remove(tmp)
            except OSError:
                pass
        rows = [DarkPoolDay(
            ticker=t, date=day, total_vol=int(tot[t]), off_vol=int(off.get(t, 0)),
            off_pct=round(off.get(t, 0) / tot[t], 4) if tot[t] else None,
            block_off_vol=int(boff.get(t, 0)), block_min=block_min, source="polygon_flatfile")
            for t in tot]
        # Clean-replace the day (drops any partial REST rows) so coverage is uniform & full.
        DarkPoolDay.objects.filter(date=day).delete()
        DarkPoolDay.objects.bulk_create(rows, batch_size=1000, ignore_conflicts=True)
        saved += len(rows); days_done += 1
        logger.info("darkpool flatfile %s: %d tickers | %d days done, %d skipped, %d missing",
                    day, len(rows), days_done, skipped, missing)
    return {"days": days_done, "skipped": skipped, "missing": missing, "rows": saved}


# ── FINRA OTC Transparency (weekly ATS / dark-pool volume — public, no auth) ──
_FINRA_WEEKLY_URL = "https://api.finra.org/data/group/otcMarket/name/weeklySummary"


def _finra_num(x):
    """CSV cell → float, treating blank/NaN/garbage as 0.0."""
    try:
        v = float(x)
        return 0.0 if v != v else v   # NaN != NaN
    except (TypeError, ValueError):
        return 0.0


def _finra_ats_get(symbol, offset=0, limit=5000):
    """POST FINRA weeklySummary for ONE symbol's ATS per-symbol weekly rows (summaryTypeCode
    ATS_W_SMBL = dark-pool total across all venues). Public, no auth. Returns list-of-dict rows
    (CSV parsed) or None on transport error."""
    import io, urllib.request
    body = json.dumps({
        "limit": limit, "offset": offset,
        "compareFilters": [
            {"compareType": "EQUAL", "fieldName": "issueSymbolIdentifier", "fieldValue": symbol},
            {"compareType": "EQUAL", "fieldName": "summaryTypeCode", "fieldValue": "ATS_W_SMBL"},
        ],
    }).encode()
    req = urllib.request.Request(
        _FINRA_WEEKLY_URL, data=body, method="POST",
        headers={"Content-Type": "application/json", "Accept": "text/plain",
                 "User-Agent": "rotation/1.0"})
    try:
        with urllib.request.urlopen(req, timeout=30) as r:
            text = r.read().decode("utf-8", "replace")
    except Exception as e:
        logger.warning("finra ATS %s failed: %s", symbol, str(e)[:80])
        return None
    if not text.strip():
        return []
    return pd.read_csv(io.StringIO(text)).to_dict("records")


def _finra_ats_fetch_all(symbol):
    """Page through a symbol's full ATS weekly history (one 5000-row page covers ~5y). Pure I/O —
    safe to run in a worker thread. Returns list-of-dict rows."""
    offset, rows = 0, []
    while True:
        page = _finra_ats_get(symbol, offset=offset)
        if not page:
            break
        rows.extend(page)
        if len(page) < 5000:
            break
        offset += 5000
    return rows


def import_finra_ats(tickers=None, only_missing=False, workers=10):
    """Backfill weekly ATS (dark-pool) volume per ticker from FINRA OTC Transparency → DarkPoolWeek.
    Public API, no auth. ONE pass pulls FULL available history per ticker (a name's ~260 weekly rows
    fit a single 5000-row page). `off_pct` = ATS shares / that week's consolidated Candle volume
    (Monday-anchored). Idempotent: clean-replaces each ticker's finra rows. `only_missing` skips
    tickers that already have rows (use for resuming an interrupted first backfill; leave False for
    the weekly refresh so newly-published weeks are picked up).

    FINRA answers ~0.5s for names it has but ~6s (HTTP 204) for names it lacks, and our universe has
    hundreds of the latter — so the HTTP fetch is parallelized across `workers` threads. DB writes
    stay single-threaded (in the as_completed loop) to keep Django's ORM connection thread-safe."""
    from core.models import DarkPoolWeek, CorporateAction
    from seq_fundamental_study import build_universe
    tickers = tickers or build_universe()
    tickers = [t for t in tickers if "." not in t]   # FINRA covers US-listed only
    skipped = 0
    if only_missing:
        have = set(DarkPoolWeek.objects.values_list("ticker", flat=True).distinct())
        skipped = sum(1 for t in tickers if t in have)
        tickers = [t for t in tickers if t not in have]
    dfs = _get_dfs(tickers)                            # candles → weekly-volume denominator
    # precompute weekly Monday-anchored consolidated volume per ticker (denominator for off_pct)
    wvols = {}
    for tk in tickers:
        cdf = dfs.get(tk)
        if cdf is None or cdf.empty:
            continue
        wv = {}
        for d, v in cdf["Volume"].items():
            mon = (d - pd.Timedelta(days=int(d.dayofweek))).date()
            wv[mon] = wv.get(mon, 0.0) + float(v)
        wvols[tk] = wv
    # Split-adjust the off_pct numerator: yfinance candle VOLUME is retroactively split-adjusted
    # (a pre-split week's volume is scaled UP by the split ratio), but FINRA totalWeeklyShareQuantity
    # is the RAW as-reported count. So for weeks before a split the ratio ats/consolidated is
    # understated by the cumulative split factor. Use clean CorporateAction split factors to scale
    # the FINRA shares onto the same adjusted basis before dividing. (Fixes the known off_pct bug.)
    split_map = {}
    for ca in (CorporateAction.objects.filter(action_type="split", split_ratio__gt=0)
               .values_list("ticker", "ex_date", "split_ratio")):
        split_map.setdefault(ca[0], []).append((ca[1], float(ca[2])))

    def _cum_split_factor(tk, week_date):
        """Product of split ratios with ex_date AFTER week_date (matches the candle-volume scaling)."""
        f = 1.0
        for ex, ratio in split_map.get(tk, ()):
            if ex > week_date and ratio > 0:
                f *= ratio
        return f
    saved = done = 0
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futs = {ex.submit(_finra_ats_fetch_all, tk): tk for tk in tickers}
        for fut in as_completed(futs):
            tk = futs[fut]
            rows = fut.result()
            if not rows:
                continue
            wvol = wvols.get(tk, {})
            objs = []
            for r in rows:
                ws = str(r.get("weekStartDate") or "").strip()[:10]
                try:
                    ws_date = datetime.strptime(ws, "%Y-%m-%d").date()
                except ValueError:
                    continue
                shares = _finra_num(r.get("totalWeeklyShareQuantity"))
                tot = wvol.get(ws_date)
                pub = str(r.get("initialPublishedDate") or "").strip()[:10]
                adj_shares = shares * _cum_split_factor(tk, ws_date)   # onto split-adjusted basis
                objs.append(DarkPoolWeek(
                    ticker=tk, week_start=ws_date, ats_shares=int(shares),
                    ats_trades=int(_finra_num(r.get("totalWeeklyTradeCount"))),
                    ats_notional=(_finra_num(r.get("totalNotionalSum")) or None),
                    off_pct=(round(adj_shares / tot, 4) if tot else None),
                    tier=str(r.get("tierDescription") or "")[:16],
                    published_date=(datetime.strptime(pub, "%Y-%m-%d").date() if len(pub) == 10 else None),
                    source="finra_ats"))
            DarkPoolWeek.objects.filter(ticker=tk, source="finra_ats").delete()
            DarkPoolWeek.objects.bulk_create(objs, batch_size=1000, ignore_conflicts=True)
            saved += len(objs); done += 1
            if done % 100 == 0:
                logger.info("finra ATS: %d tickers, %d weekly rows", done, saved)
    logger.info("import_finra_ats: %d tickers, %d weekly rows (%d skipped)", done, saved, skipped)
    return {"tickers": done, "rows": saved, "skipped": skipped}


# ── EODHD integration (news + sentiment, earnings surprises, estimate revisions) ──
# Key-driven via EODHD_API_KEY. Uses the base plan (fundamentals/news/calendar), which the token
# already has. Ticker mapping: US names → TICKER.US; foreign keep their suffix (SHOP.TO).

# Yahoo-style exchange suffix → EODHD exchange code (they differ for several venues). Verified:
# Korea .KS→.KO (Samsung), Shenzhen .SZ→.SHE, plus the standard EODHD codes for LSE/Xetra/etc.
# Suffixes not listed pass through unchanged (.HK, .TW, .PA, .TO, .MI, .SA, .AS, .SW, … are native).
_EODHD_SUFFIX_REMAP = {
    "KS": "KO", "KQ": "KO",   # Korea (KOSPI / KOSDAQ)
    "SZ": "SHE", "SS": "SHG",  # Shenzhen / Shanghai
    "T": "TSE",                # Tokyo (yahoo .T → EODHD .TSE)
    "NS": "NSE",               # India NSE (yahoo .NS → EODHD .NSE)
    "BO": "BSE",               # India BSE (yahoo .BO → EODHD .BSE)
    "L": "LSE",                # London
    "DE": "XETRA", "F": "XETRA",  # Frankfurt / Xetra
    "AX": "AU",                # Australia
    "JO": "JSE",               # Johannesburg (verified)
    "WA": "WAR",               # Warsaw (verified)
    # Suffixes NOT listed pass through unchanged: .TO (Toronto), .HK, .PA, .SW, .MI, .SA, .AS, .TW …
    # are already the native EODHD exchange codes.
}

# Per-ticker overrides for edge cases the suffix map can't express (populate as needed).
_EODHD_SYM_OVERRIDE = {}


def _eodhd_sym(tk):
    """Map a plain/yahoo ticker to an EODHD `TICKER.EXCHANGE` symbol.

    EODHD requires the exchange-qualified form (AAPL.US, 7203.TSE, WIPRO.NSE, SHOP.TO, RIO.LSE);
    the plain/yahoo form (AAPL, 7203.T, WIPRO.NS) will NOT resolve. Returns None for symbols that
    have no fundamentals on EODHD (e.g. futures ending in `=F`)."""
    if tk in _EODHD_SYM_OVERRIDE:
        return _EODHD_SYM_OVERRIDE[tk]
    if tk.endswith("=F"):
        return None  # futures — no fundamentals to fetch, skip
    if "." not in tk:
        return f"{tk}.US"  # plain ticker → US listing
    base, suf = tk.rsplit(".", 1)
    return f"{base}.{_EODHD_SUFFIX_REMAP.get(suf.upper(), suf)}"


def _eodhd_get(path, **params):
    import os, json, urllib.request, urllib.parse
    key = os.environ.get("EODHD_API_KEY")
    if not key:
        return None
    params.setdefault("api_token", key); params.setdefault("fmt", "json")
    url = "https://eodhd.com/api/" + path + "?" + urllib.parse.urlencode(params)
    try:
        with urllib.request.urlopen(urllib.request.Request(url, headers={"User-Agent": "rotation/1.0"}), timeout=40) as r:
            return json.loads(r.read().decode("utf-8", "replace"))
    except Exception as e:
        logger.warning("eodhd GET %s failed: %s", path[:40], e)
        return None


def import_eodhd_news(tickers=None, days=400, per=1000, max_pages=1000, sleep=0.02):
    """Pull EODHD news + sentiment → NewsItem, PAGINATED (EODHD returns newest-first, capped at
    `limit` per call, so we page with `offset` to reach deep history). `days` sets how far back
    (400≈13mo incremental; ~2000 ≈ 5.5y deep backfill). bulk_create(ignore_conflicts) dedupes on the
    `uid` unique key AND preserves any existing row's LLM classification (it skips, never overwrites).
    Returns net-new inserted count."""
    import hashlib
    from django.utils.dateparse import parse_datetime
    from core.models import NewsItem
    from seq_fundamental_study import build_universe
    if not os.environ.get("EODHD_API_KEY"):
        return {"error": "EODHD_API_KEY not set"}
    tickers = tickers or build_universe()
    frm = (date.today() - timedelta(days=days)).isoformat()
    to = date.today().isoformat()
    before = NewsItem.objects.count()
    seen_uids = set()
    for ti, tk in enumerate(tickers):
        sym = _eodhd_sym(tk)
        for page in range(max_pages):
            rows = _eodhd_get("news", s=sym, **{"from": frm, "to": to}, limit=per, offset=page * per) or []
            if not rows:
                break
            batch = []
            for a in rows:
                d = a.get("date"); title = (a.get("title") or "")[:512]
                if not d:
                    continue
                uid = hashlib.md5(f"{tk}|{d}|{title}".encode()).hexdigest()
                if uid in seen_uids:
                    continue
                seen_uids.add(uid)
                sen = a.get("sentiment") or {}
                tags = a.get("tags") or []
                if not isinstance(tags, list):
                    tags = []
                batch.append(NewsItem(
                    uid=uid, ticker=tk, dt=parse_datetime(d) or d, title=title,
                    sentiment=sen.get("polarity"), pos=sen.get("pos"), neg=sen.get("neg"), neu=sen.get("neu"),
                    tags=[str(t)[:48] for t in tags][:12], url=(a.get("link") or "")[:1024]))
            if batch:
                NewsItem.objects.bulk_create(batch, ignore_conflicts=True, batch_size=500)
            if len(rows) < per:
                break
            time.sleep(sleep)
        if ti % 100 == 0:
            logger.info("import_eodhd_news: %d/%d tickers, net +%d so far",
                        ti, len(tickers), NewsItem.objects.count() - before)
    saved = NewsItem.objects.count() - before
    logger.info("import_eodhd_news: net-new %d items (days=%d)", saved, days)
    return {"saved": saved}


def import_eodhd_earnings(tickers=None):
    """Pull EODHD Earnings::History (dates + EPS surprise) → EarningsEvent."""
    from core.models import EarningsEvent
    from seq_fundamental_study import build_universe
    if not os.environ.get("EODHD_API_KEY"):
        return {"error": "EODHD_API_KEY not set"}
    tickers = tickers or build_universe()
    saved = 0
    for tk in tickers:
        d = _eodhd_get(f"fundamentals/{_eodhd_sym(tk)}", filter="Earnings::History")
        if not isinstance(d, dict):
            continue
        for _, e in d.items():
            rd = e.get("reportDate")
            if not rd:
                # No announcement date — approximate PIT availability as the
                # fiscal period-end + 45d (typical 10-Q filing deadline). Using
                # the period-end itself (e["date"], ~4-6wk earlier) would let a
                # backtest see the report before it was actually filed.
                pe = e.get("date")
                if not pe:
                    continue
                try:
                    rd = (pd.Timestamp(pe) + pd.Timedelta(days=45)).date().isoformat()
                except Exception:
                    continue
            try:
                EarningsEvent.objects.update_or_create(ticker=tk, report_date=rd, defaults=dict(
                    eps_actual=e.get("epsActual"), eps_estimate=e.get("epsEstimate"),
                    eps_surprise_pct=e.get("surprisePercent"), before_after=(e.get("beforeAfterMarket") or "")[:16]))
                saved += 1
            except Exception:
                pass
    logger.info("import_eodhd_earnings: %d events", saved)
    return {"saved": saved}


def import_eodhd_estimates(tickers=None):
    """Pull EODHD Earnings::Trend (estimate revisions: current vs 7d/30d ago) → EstimateRevision."""
    from core.models import EstimateRevision
    from seq_fundamental_study import build_universe
    if not os.environ.get("EODHD_API_KEY"):
        return {"error": "EODHD_API_KEY not set"}
    tickers = tickers or build_universe()
    today = date.today(); saved = 0
    for tk in tickers:
        d = _eodhd_get(f"fundamentals/{_eodhd_sym(tk)}", filter="Earnings::Trend")
        if not isinstance(d, dict):
            continue
        for period, e in d.items():
            if not period or period == "0000-00-00":
                continue
            try:
                EstimateRevision.objects.update_or_create(ticker=tk, period=period, asof=today, defaults=dict(
                    period_label=(e.get("period") or "")[:8],
                    eps_current=e.get("epsTrendCurrent"), eps_7d_ago=e.get("epsTrend7daysAgo"),
                    eps_30d_ago=e.get("epsTrend30daysAgo"), revenue_avg=e.get("revenueEstimateAvg")))
                saved += 1
            except Exception:
                pass
    logger.info("import_eodhd_estimates: %d rows", saved)
    return {"saved": saved}


def import_eodhd_analyst_ratings(tickers=None):
    """Pull the EODHD fundamentals AnalystRatings block → Fundamental analyst distribution columns
    (StrongBuy/Buy/Hold/Sell/StrongSell counts + 1–5 rating mean + target). Updates the latest
    Fundamental row per ticker (leaves ticker + everything else untouched)."""
    from core.models import Fundamental
    from seq_fundamental_study import build_universe
    if not os.environ.get("EODHD_API_KEY"):
        return {"error": "EODHD_API_KEY not set"}
    tickers = tickers or build_universe()
    saved = 0
    for tk in tickers:
        sym = _eodhd_sym(tk)
        if not sym:
            continue
        d = _eodhd_get(f"fundamentals/{sym}", filter="AnalystRatings")
        if not isinstance(d, dict) or not d:
            continue
        row = Fundamental.objects.filter(ticker=tk).order_by("-date").first()
        if not row:
            continue

        def _i(x):
            try:
                return int(round(float(x)))
            except (TypeError, ValueError):
                return None

        def _f(x):
            try:
                return float(x)
            except (TypeError, ValueError):
                return None
        row.analyst_strong_buy = _i(d.get("StrongBuy"))
        row.analyst_buy = _i(d.get("Buy"))
        row.analyst_hold = _i(d.get("Hold"))
        row.analyst_sell = _i(d.get("Sell"))
        row.analyst_strong_sell = _i(d.get("StrongSell"))
        row.analyst_rating_mean = _f(d.get("Rating"))
        if row.analyst_target is None:
            row.analyst_target = _f(d.get("TargetPrice"))
        row.save(update_fields=["analyst_strong_buy", "analyst_buy", "analyst_hold", "analyst_sell",
                                "analyst_strong_sell", "analyst_rating_mean", "analyst_target"])
        saved += 1
    logger.info("import_eodhd_analyst_ratings: %d rows", saved)
    return {"saved": saved}


def import_eodhd_fundamentals(tickers=None, only_missing=True, yf_fallback=True):
    """Backfill historical QUARTERLY financials into FinancialReport from EODHD
    `fundamentals/{sym}` (global coverage), for tickers SEC EDGAR can't serve — foreign filers
    (RMS.PA, 6367.T, …) plus any US names without a CIK match. EDGAR stays PRIMARY for US; this
    only fills the hole so the point-in-time fundamentals layer covers the whole universe.

    only_missing=True → skip tickers that already have (EDGAR-sourced) FinancialReport rows, so we
    never clobber the authoritative US history and we spend one EODHD call only on the gap tickers.
    EODHD stamps each period with `filing_date` → avail_date is the real point-in-time (fallback:
    period_end + 45d if EODHD omits it). Idempotent via update_or_create(ticker, period_end)."""
    from core.models import FinancialReport, Fundamental
    from seq_fundamental_study import build_universe
    if not os.environ.get("EODHD_API_KEY"):
        return {"error": "EODHD_API_KEY not set"}
    tickers = tickers or build_universe()
    if only_missing:
        have = set(FinancialReport.objects.values_list("ticker", flat=True).distinct())
        tickers = [t for t in tickers if t not in have]
    logger.info("import_eodhd_fundamentals: %d target tickers (only_missing=%s)", len(tickers), only_missing)

    def _num(x):
        if x in (None, "", "None", "0000-00-00"):
            return None
        try:
            return float(x)
        except (TypeError, ValueError):
            return None

    def _int(x):
        v = _num(x)
        return int(v) if v is not None else None

    saved = 0; tks_done = 0; gap = []; syms = {}
    for tk in tickers:
        sym = _eodhd_sym(tk)
        if sym is None:
            # No EODHD fundamentals for this symbol (e.g. futures =F) — skip entirely,
            # don't send to the yfinance fallback either (it has no fundamentals for these).
            continue
        # Record the EODHD symbol used onto any existing Fundamental row(s) for this ticker,
        # so global fills are traceable (does NOT mutate the plain `ticker` value).
        syms[tk] = sym
        try:
            Fundamental.objects.filter(ticker=tk).update(eodhd_symbol=sym)
        except Exception:
            pass
        d = _eodhd_get(f"fundamentals/{sym}", filter="Financials")
        tks_done += 1
        if not isinstance(d, dict):
            gap.append(tk)
            continue
        s0 = saved
        inc = (d.get("Income_Statement") or {}).get("quarterly") or {}
        bal = (d.get("Balance_Sheet") or {}).get("quarterly") or {}
        cfl = (d.get("Cash_Flow") or {}).get("quarterly") or {}
        for pe in (set(inc) | set(bal)):
            i = inc.get(pe) or {}; b = bal.get(pe) or {}; c = cfl.get(pe) or {}
            try:
                pe_d = date.fromisoformat(pe[:10])
            except Exception:
                continue
            filed = i.get("filing_date") or b.get("filing_date") or c.get("filing_date")
            try:
                avail = date.fromisoformat(filed[:10]) if filed and filed != "0000-00-00" else pe_d + timedelta(days=45)
            except Exception:
                avail = pe_d + timedelta(days=45)
            tot_debt = _num(b.get("shortLongTermDebtTotal"))
            if tot_debt is None:
                lt = _num(b.get("longTermDebt")); st = _num(b.get("shortTermDebt"))
                if lt is not None or st is not None:
                    tot_debt = (lt or 0) + (st or 0)
            ocf = _num(c.get("totalCashFromOperatingActivities"))
            fcf = _num(c.get("freeCashFlow"))
            if fcf is None:
                capex = _num(c.get("capitalExpenditures"))
                if ocf is not None and capex is not None:
                    fcf = ocf - abs(capex)
            try:
                FinancialReport.objects.update_or_create(
                    ticker=tk, period_end=pe_d, defaults=dict(
                        avail_date=avail,
                        revenue=_int(i.get("totalRevenue")),
                        net_income=_int(i.get("netIncome")),
                        operating_income=_int(i.get("operatingIncome")),
                        gross_profit=_int(i.get("grossProfit")),
                        cost_of_revenue=_int(i.get("costOfRevenue")),
                        rd_expense=_int(i.get("researchDevelopment")),
                        eps_diluted=None,   # EODHD income statement carries no diluted EPS; leave null
                        total_equity=_int(b.get("totalStockholderEquity")),
                        total_debt=_int(tot_debt) if tot_debt is not None else None,
                        current_assets=_int(b.get("totalCurrentAssets")),
                        current_liabilities=_int(b.get("totalCurrentLiabilities")),
                        total_assets=_int(b.get("totalAssets")),
                        inventory=_int(b.get("inventory")),
                        cash_and_equivalents=_int(b.get("cashAndEquivalents") or b.get("cash")),
                        shares_outstanding=_int(b.get("commonStockSharesOutstanding")),
                        operating_cash_flow=_int(ocf) if ocf is not None else None,
                        free_cash_flow=_int(fcf) if fcf is not None else None,
                    ))
                saved += 1
            except Exception:
                pass
        if saved == s0:
            gap.append(tk)
        if tks_done % 50 == 0:
            logger.info("import_eodhd_fundamentals: %d/%d tickers, %d rows", tks_done, len(tickers), saved)
    yf_res = None
    if yf_fallback and gap:
        logger.info("import_eodhd_fundamentals: %d tickers missed by EODHD (404/empty) -> yfinance fallback", len(gap))
        yf_res = import_yf_fundamentals(tickers=gap, only_missing=False)
    logger.info("import_eodhd_fundamentals: %d rows across %d tickers (%d gap -> yf)", saved, tks_done, len(gap))
    return {"saved": saved, "tickers": tks_done, "eodhd_gap": len(gap), "yf_fallback": yf_res, "eodhd_symbols": syms}


def import_yf_fundamentals(tickers=None, only_missing=True):
    """Snapshot fundamentals per ticker from yfinance `.info` into core.models.Fundamental
    (update_or_create keyed on ticker+today's date). This is the SELF-HEALING FALLBACK for names
    that EODHD 404s on (US miners like AEM/GOLD/KGC/FNV/CCJ, small/mid-caps, etc.). EODHD/EDGAR
    stay PRIMARY for historical point-in-time financials; this only fills the Fundamental snapshot
    hole so the fundamentals layer covers the whole universe.

    only_missing=True -> universe = distinct Candle tickers that have NO existing Fundamental row.
    Futures (ticker ends in '=F') are skipped — commodity futures have no equity fundamentals.
    Idempotent (update_or_create), each ticker guarded so one bad symbol never aborts the run."""
    from core.models import Fundamental
    if tickers is None:
        tickers = list(Candle.objects.values_list("ticker", flat=True).distinct())
    tickers = [t for t in tickers if not str(t).endswith("=F")]
    if only_missing:
        have = set(Fundamental.objects.values_list("ticker", flat=True).distinct())
        tickers = [t for t in tickers if t not in have]
    logger.info("import_yf_fundamentals: %d target tickers (only_missing=%s)", len(tickers), only_missing)

    def _g(info, key):
        """float or None (drops NaN)."""
        v = info.get(key)
        if v is None:
            return None
        try:
            f = float(v)
            return f if f == f else None
        except (TypeError, ValueError):
            return None

    def _gi(info, key):
        """int or None (drops NaN)."""
        v = info.get(key)
        if v is None:
            return None
        try:
            f = float(v)
            return int(f) if f == f else None
        except (TypeError, ValueError):
            return None

    today = date.today()
    saved = 0; errors = 0; no_data = 0; done = 0
    for tk in tickers:
        done += 1
        try:
            info = yf.Ticker(tk).info or {}
            if info.get("marketCap") is None and info.get("trailingEps") is None and info.get("totalRevenue") is None:
                no_data += 1
                continue
            fields = {
                "dividend_yield": _g(info, "dividendYield"),
                "pe_ratio": _g(info, "trailingPE"),
                "forward_pe": _g(info, "forwardPE"),
                "pb_ratio": _g(info, "priceToBook"),
                "ps_ratio": _g(info, "priceToSalesTrailing12Months"),
                "peg_ratio": _g(info, "pegRatio"),
                "market_cap": _gi(info, "marketCap"),
                "enterprise_value": _gi(info, "enterpriseValue"),
                "eps": _g(info, "trailingEps"),
                "forward_eps": _g(info, "forwardEps"),
                "annual_revenue": _gi(info, "totalRevenue"),
                "revenue_growth": _g(info, "revenueGrowth"),
                "earnings_growth": _g(info, "earningsGrowth"),
                "profit_margin": _g(info, "profitMargins"),
                "operating_margin": _g(info, "operatingMargins"),
                "shares_outstanding": _gi(info, "sharesOutstanding"),
                "float_shares": _gi(info, "floatShares"),
                "short_ratio": _g(info, "shortRatio"),
                "short_pct_float": _g(info, "shortPercentOfFloat"),
                "insider_pct": _g(info, "heldPercentInsiders"),
                "institution_pct": _g(info, "heldPercentInstitutions"),
                "analyst_rating": info.get("recommendationKey"),
                "analyst_target": _g(info, "targetMeanPrice"),
                "analyst_count": _gi(info, "numberOfAnalystOpinions"),
                "total_cash": _gi(info, "totalCash"),
                "total_debt": _gi(info, "totalDebt"),
                "debt_to_equity": _g(info, "debtToEquity"),
                "current_ratio": _g(info, "currentRatio"),
                "book_value": _g(info, "bookValue"),
                "free_cash_flow": _gi(info, "freeCashflow"),
                "operating_cash_flow": _gi(info, "operatingCashflow"),
                "beta_5y": _g(info, "beta"),
                "fifty_two_wk_high": _g(info, "fiftyTwoWeekHigh"),
                "fifty_two_wk_low": _g(info, "fiftyTwoWeekLow"),
                "avg_volume": _gi(info, "averageVolume"),
                "sector": info.get("sector"),
                "industry": info.get("industry"),
            }
            Fundamental.objects.update_or_create(ticker=tk, date=today, defaults=fields)
            saved += 1
        except Exception as e:
            errors += 1
            logger.warning("import_yf_fundamentals: %s error %s", tk, e)
        # yfinance throttles after many sequential .info calls
        if done % 20 == 0:
            time.sleep(2)
            logger.info("import_yf_fundamentals: %d/%d (%d saved)", done, len(tickers), saved)
    logger.info("import_yf_fundamentals: %d saved, %d no-data, %d errors", saved, no_data, errors)
    return {"saved": saved, "no_data": no_data, "errors": errors, "tickers": done}


def _vix_regime(vx):
    """Current VIX regime — the single strongest entry-timing filter we found. Distress bought in a
    SPIKING VIX (market-wide panic) = +27%/78%wr/5.8%-disaster; the SAME signal in a CALM VIX =
    -0.3%/47%wr/31.6%-disaster (idiosyncratic knife). Returns calm/elevated/spiking/peaking."""
    if vx is None or len(vx) < 25:
        return None
    v = float(vx.iloc[-1]); m = float(vx.rolling(20).mean().iloc[-1]); hi = float(vx.rolling(10).max().iloc[-1])
    if v > 20 and v < 0.88 * hi:
        return "peaking"          # came off a spike — panic fading
    if v > 1.10 * m and v > 15:
        return "spiking"          # fear rising — the golden window for buying distress
    if v < 16:
        return "calm"             # quiet market — a crashing stock here is idiosyncratic (knife)
    return "elevated"


def _news_pop_abn(df, spy_close, news_ts):
    """Signed day-1 ABNORMAL move (%) around a news date: close(t-1)→close(t+1), β-adjusted
    (rolling 60d β vs SPY, clipped [0,3]) — same construction as news_drift_material.py's
    day-1 reaction. Returns None if the window isn't available. +value = a POP."""
    try:
        import numpy as np
        idx = df.index; close = df["Close"].values; n = len(close)
        pos = int(idx.searchsorted(pd.Timestamp(news_ts).tz_localize(None).normalize()))
        rfrom, rto = pos - 1, pos + 1
        if rfrom < 61 or rto >= n:
            return None
        mkt = spy_close.reindex(idx).ffill().values
        if mkt[rfrom] <= 0 or mkt[rto] <= 0:
            return None
        r = df["Close"].pct_change()
        m = pd.Series(mkt, index=idx).pct_change()
        both = pd.concat([r.rename("s"), m.rename("m")], axis=1)
        b = (both["s"].rolling(60).cov(both["m"]) / both["m"].rolling(60).var()).values[rfrom]
        bc = min(max(b if np.isfinite(b) else 1.0, 0.0), 3.0)
        v = ((close[rto] / close[rfrom] - 1.0) - bc * (mkt[rto] / mkt[rfrom] - 1.0)) * 100
        return float(v) if np.isfinite(v) else None
    except Exception:
        return None


def _risk_rating(c):
    """Risk rating CALIBRATED to the historical DISASTER RATE (% of signal events down >25% at
    126d) of our own signals — not hand-picked. Key measured facts (base disaster rate 7.3%):
      • market cap is the dominant, survivorship-ROBUST risk factor: >$10B = 4.9% disaster vs
        ~11-12% for $500M-10B → cap gets the biggest weight.
      • sector IN vs OUT is risk-NEUTRAL per trade (7.3% either way) — its value is portfolio-level,
        so only a small tilt here.
      • smart money does NOT reduce risk (7.7% vs 4.7% — it clusters on distressed names); it's a
        CONVICTION flag, shown but NOT scored.
      • knife (repeat-fire) is a modest real penalty (loss-rate 38% vs 33%).
    Higher score = lower risk. Returns {score, level, drivers}."""
    s = 50.0
    drv = []
    # setup type — Mode A/B are the funnel core (start higher); discretionary surfaces start lower
    if c.get("mode") == "A":
        s += 13; drv.append("+ Mode A: capitulation + accumulation (highest per-trade return)")
    elif c.get("deep_dd"):
        s += 2; drv.append("+ deep-drawdown opportunity screen (discretionary)")
    elif c.get("below_trend"):
        s += 2; drv.append("+ below-trend dip being accumulated (discretionary)")
    else:
        s += 9; drv.append("+ Mode B: dip in an uptrend")
    # MARKET CAP — the dominant measured risk factor (disaster rate: >$10B 4.9% vs mid/small ~11-12%)
    mc = c.get("market_cap")
    if mc is None:
        s -= 8; drv.append("− market cap unknown")
    elif mc >= 10e9:
        s += 10; drv.append("+ mega-cap (>$10B): lowest tail risk — 4.9% disaster rate (survivorship-robust)")
    elif mc >= 2e9:
        s -= 5; drv.append("− mid-cap ($2–10B): elevated tail risk (~11% disaster rate)")
    elif mc >= 5e8:
        s -= 8; drv.append("− small-cap ($0.5–2B): highest tail risk (~12% disaster rate)")
    else:
        s -= 14; drv.append("− micro-cap (<$500M): survivorship-lethal, treat as speculative")
    if (c.get("last_close") or 99) < 10:
        s -= 4; drv.append("− low price (<$10)")
    # sector — per-trade risk-neutral, so only a small tilt (real value is portfolio-level)
    st = c.get("sector_state")
    if c.get("off_gate") or st == "OUT":
        s -= 6; drv.append("− sector OUT (portfolio-drag / discretionary; per-trade risk ≈ neutral)")
    elif st == "IN":
        s += 4; drv.append("+ sector rotating IN")
    elif st in ("LEADER", "STRONG"):
        s += 2; drv.append("+ sector strong")
    # discretionary downside flags (no trend support)
    if c.get("below_trend"):
        s -= 16; drv.append("− below 200dMA: no trend support — needs a catalyst / good news")
    if c.get("deep_dd"):
        s -= 8; drv.append("− deep drawdown: cap term governs (mega-cap reliable, small-cap = knife)")
    # knife — modest, scaled to the measured loss-rate lift (~+5pts per repeat-fire)
    if c.get("knife"):
        pen = min(20, 6 * (c.get("prior_fires") or 1)) + (4 if c.get("knife_fell") else 0)
        s -= pen
        drv.append(f"− KNIFE: this signal already fired {c.get('prior_fires')}× in ~9 months"
                   + (" at higher prices & it kept falling" if c.get("knife_fell") else "")
                   + " — accumulation has been early; take only on a confirmed turn")
    # VIX REGIME at entry — the strongest timing filter (disaster rate: calm-distress 31.6% vs
    # spiking-distress 5.8%). Distress in a panicking market recovers; distress in a quiet market
    # is an idiosyncratic knife. Weighted heavily for Mode A (buys the crash), lighter for Mode B.
    vr = c.get("vix_regime")
    if vr and c.get("mode") == "A":
        if vr == "calm":
            s -= 18; drv.append("− VIX CALM: distress in a quiet market = idiosyncratic knife (31.6% disaster hist)")
        elif vr == "elevated":
            s -= 3; drv.append("· VIX elevated")
        elif vr == "spiking":
            s += 15; drv.append("+ VIX SPIKING: market-wide panic = the golden distress window (+27%/78% hist)")
    elif vr and c.get("mode") == "B":
        if vr == "spiking":
            s += 8; drv.append("+ VIX spiking: dips bought in fear rebound harder (+9.7%/70% hist)")
        elif vr == "calm":
            s -= 2
    # recent EARNINGS MISS — validated downside PEAD (miss_drift_study.py): a stock that just
    # missed keeps underperforming ABNORMALLY for ~1-3mo, and it scales with miss size (mild≈beats,
    # severe/huge ≈ −3% abn/63d) and INVERSELY with cap (small/mid drift most, mega-cap ~−0.9%).
    mp = c.get("recent_miss_pct")
    if mp is not None and mp < 0:
        a = abs(mp)
        pen = 2 if a < 5 else 6 if a < 15 else 10       # mild / moderate / severe+
        if (c.get("market_cap") or 0) >= 10e9:
            pen = round(pen * 0.4)                        # mega-cap miss drift is small
        s -= pen
        drv.append(f"− recent earnings MISS ({mp}% surprise, {c.get('recent_miss_days')}d ago): "
                   "validated downside drift — names keep underperforming ~1–3mo post-miss "
                   "(worse for smaller caps); wait it out, don't buy the dip yet")
    # fresh strong BULLISH-news POP in a mid/small cap — validated size-conditioned FADE
    # (news_drift_robust.py): mid-cap ≈ −12% abn / 90d, small worse; near-zero in mega-caps (so the
    # attach step only flags <$10B). Good news over-extends in less-liquid names and gives back.
    pp = c.get("fresh_bull_pop_pct")
    if pp is not None:
        pen = 9 if (c.get("market_cap") or 0) < 2e9 else 6    # small (<2B) fades hardest, then mid
        s -= pen
        drv.append(f"− fresh strong bullish-news POP (+{pp}% β-adj, {c.get('fresh_bull_pop_days')}d ago) in a "
                   "mid/small cap: validated to GIVE BACK abnormally over ~1–3mo (sell-the-news / "
                   "over-extension) — don't chase the pop, expect a pullback")
    # smart money — shown as CONVICTION, not scored (data: it does not reduce risk)
    if (c.get("insider_buy_180d") or 0) or c.get("recent_13d") or c.get("recent_13g"):
        drv.append("· smart money present (conviction flag — note: not a risk reducer; it clusters on distressed names)")
    wr = c.get("hist_win_rate")
    if wr:
        adj = max(-6.0, min(6.0, (wr - 64) / 3.0)); s += adj
    s = max(0.0, min(100.0, s))
    level = "Low" if s >= 70 else "Medium" if s >= 55 else "High" if s >= 40 else "Very High"
    return {"score": round(s), "level": level, "drivers": drv}


# Commodity themes: trend anchor (ETF, or 'BASKET' = avg of proxies when no clean ETF) + proxy
# stocks to actually buy. Commodities are TREND plays (ride strength), not dip-buys.
COMMODITY_THEMES = {
    "Gold": ("GLD", ["NEM", "GOLD", "AEM", "KGC", "FNV", "WPM", "RGLD"]),
    "Silver": ("SLV", ["CDE", "PAAS", "HL", "WPM"]),
    "Copper": ("COPX", ["FCX", "SCCO", "TECK", "ERO"]),
    "Uranium": ("URA", ["CCJ", "UEC", "DNN", "UUUU", "NXE"]),
    "Oil/Energy": ("XLE", ["XOM", "CVX", "COP", "OXY", "DVN", "EOG", "FANG", "MPC", "SLB"]),
    "NatGas": ("UNG", ["LNG", "EQT", "AR", "RRC", "EXE"]),
    "Coal": ("BASKET", ["BTU", "ARCH", "HCC", "AMR", "CEIX", "ARLP"]),
    "Steel": ("SLX", ["NUE", "STLD", "X", "CLF", "MT", "RS"]),
    "Aluminum": ("BASKET", ["AA", "CENX", "KALU"]),
    "Lithium": ("LIT", ["ALB", "SQM", "LAC", "SGML"]),
    "Agriculture": ("DBA", ["ADM", "BG", "NTR", "MOS", "CF", "CTVA", "DE"]),
    "Platinum/PGM": ("PPLT", ["SBSW", "PLG"]),
    "RareEarth": ("REMX", ["MP"]),
    "Lumber/Timber": ("LBS=F", ["WY", "RYN", "LPX", "PCH"]),
}


def compute_commodity_board(recent=10):
    """Commodity themes treated as first-class sectors. Trend state per theme (anchor ETF /
    underlying future / proxy-basket vs its 200dMA & vs SPY), and each proxy stock scanned with
    BOTH Mode A (capitulation + accum-divergence) and Mode B (dip near a 52wk-high) — gated by
    the commodity's trend. Returns {board, candidates}."""
    import pandas as pd
    import numpy as np
    from seq_fundamental_study import load_candles
    from studies import SIGNALS
    from pit_fundamentals import _ad_state, bucket_ad
    from all_on_all_study import _prepare_indicators
    from core.models import Candle

    anchors = [a for a, _ in COMMODITY_THEMES.values() if a != "BASKET"]
    proxies = sorted({p for _, ps in COMMODITY_THEMES.values() for p in ps})
    have = set(Candle.objects.filter(ticker__in=anchors + proxies).values_list("ticker", flat=True).distinct())
    cd = load_candles(sorted((set(anchors) | set(proxies)) & have) + ["SPY"])
    spy = cd["SPY"]["Close"] if "SPY" in cd else None
    spy63 = spy.pct_change(63) if spy is not None else None
    board, candidates = [], []
    for theme, (anchor, ps) in COMMODITY_THEMES.items():
        if anchor != "BASKET" and anchor in cd:
            ts = cd[anchor]["Close"]; atag = anchor
        else:  # no ETF/future data -> fall back to a proxy basket
            cols = [cd[p]["Close"] for p in ps if p in cd]
            ts = pd.concat(cols, axis=1).mean(axis=1) if cols else None
            atag = "basket"
        if ts is None or len(ts) < 200:
            continue
        abs_up = bool(ts.iloc[-1] > ts.rolling(200).mean().iloc[-1])
        rs = float(ts.pct_change(63).iloc[-1] - spy63.iloc[-1]) * 100 if spy is not None else None
        state = "LEADER" if (abs_up and (rs or 0) > 0) else ("STRONG" if abs_up else ("TURNING" if (rs or 0) > 0 else "OUT"))
        prox = []
        for p in ps:
            if p not in cd or len(cd[p]) < 200:
                continue
            df = cd[p]; pc = df["Close"]
            prox.append({"ticker": p, "last": round(float(pc.iloc[-1]), 2),
                         "uptrend": bool(pc.iloc[-1] > pc.rolling(200).mean().iloc[-1]),
                         "ret_63d": round(float(pc.pct_change(63).iloc[-1]) * 100, 1)})
            if state == "OUT" or len(df) < 260:
                continue
            # scan the proxy with BOTH Mode A and Mode B (commodities can be either), gated by trend
            _prepare_indicators(df)
            cl = df["Close"].values; st = _ad_state(df).values
            if pd.isna(st[-1]) or cl[-1] < 5:
                continue
            hi = pd.Series(cl).rolling(252).max().iloc[-1]; lo = pd.Series(cl).rolling(252).min().iloc[-1]
            posn = (cl[-1] - lo) / (hi - lo) if hi > lo else None
            adlab = bucket_ad(float(st[-1]))
            fires = []
            for sk in AD_CAPIT_SIGNALS:
                try:
                    rr = SIGNALS[sk][1](df).fillna(False).iloc[-recent:].tolist()
                except Exception:
                    continue
                if any(rr):
                    fires.append((sk, next(i for i, v in enumerate(reversed(rr)) if v)))
            is_a = bool(fires) and adlab == "accum divergence"
            # Mode B — "dip in an uptrend" (merged old near-52wk-high B + commodity momentum C into
            # ONE concept): the proxy is above its own 200dMA (uptrend) AND RSI(10) dipped <35 within
            # the window (pullback), gated by the commodity theme being LEADER/STRONG. Commodities
            # trend, so this is the volume engine; deep-oversold rarely fires here.
            rsi = df["_rsi"]
            sma200 = pd.Series(cl).rolling(200).mean().iloc[-1]
            up200 = bool(cl[-1] > sma200) if pd.notna(sma200) else False
            rsi_recent = rsi.iloc[-recent:]
            b_days = next((i for i, v in enumerate(reversed((rsi_recent < 30).tolist())) if v), None)
            rsi_now = float(rsi.iloc[-1]) if pd.notna(rsi.iloc[-1]) else 99
            is_b = state in ("LEADER", "STRONG") and up200 and b_days is not None and rsi_now < 45
            if is_a or is_b:
                mode = "A" if is_a else "B"
                hist = ({"hist_avg_return": None, "hist_win_rate": None, "hist_trades": None} if is_a
                        else {"hist_avg_return": 11.4, "hist_win_rate": 64, "hist_trades": 898})
                candidates.append({
                    "ticker": p, "mode": mode, "sector": theme, "etf": atag,
                    "sector_state": state, "last_close": round(float(cl[-1]), 2),
                    "trigger": (", ".join(SIGNALS[f[0]][0] for f in fires) if is_a
                                else "Dip in uptrend (RSI<30, >200dMA)"),
                    "days_ago": (min(f[1] for f in fires) if is_a else b_days),
                    "ad_state": adlab, "pct_52w": round(posn * 100, 0) if posn is not None else None,
                    "market_cap": None, "pe_ratio": None, **hist,
                    "insider_buy_180d": None, "recent_13d": 0, "recent_13g": 0,
                    "is_commodity": True})
        prox.sort(key=lambda x: -x["ret_63d"])
        board.append({"theme": theme, "anchor": atag, "state": state, "rs_63d": round(rs, 1) if rs is not None else None,
                      "abs_uptrend": abs_up, "n_proxies": len(prox), "proxies": prox})
    board.sort(key=lambda x: -(x["rs_63d"] or -999))
    return {"board": board, "candidates": candidates}


# GICS sector -> SPDR sector ETF, for the sector-rotation gate (ETF vs SPY momentum).
GICS2ETF = {
    "Technology": "XLK", "Healthcare": "XLV", "Energy": "XLE", "Financial Services": "XLF",
    "Consumer Cyclical": "XLY", "Consumer Defensive": "XLP", "Industrials": "XLI",
    "Basic Materials": "XLB", "Real Estate": "XLRE", "Communication Services": "XLC",
    "Utilities": "XLU",
}


def compute_strategy_forward(horizon=90, save=True):
    """Average forward PATH (day 1..H from entry) of the two-mode, sector-gated strategy:
    Mode A = capitulation (new-low/oversold) + accum-divergence; Mode B = oversold near the
    52-wk high; both gated on the stock's GICS sector ETF outperforming SPY over 63 bars.
    Answers 'where is the trade, on average, N days after we bought it'. Saves a JSON the
    dashboard reads. Median is the honest central line (avg is right-tail heavy)."""
    import numpy as np
    import pandas as pd
    from pathlib import Path
    from django.utils import timezone
    from seq_fundamental_study import build_universe, load_candles
    from studies import SIGNALS
    from all_on_all_study import _prepare_indicators
    from pit_fundamentals import _ad_state
    from core.models import Fundamental

    tks = build_universe()
    secmap = {r["ticker"]: r["sector"] for r in
              Fundamental.objects.filter(ticker__in=tks).values("ticker", "sector")}
    t2etf = {t: GICS2ETF[s] for t, s in secmap.items() if s in GICS2ETF}
    mkt = load_candles(sorted(set(t2etf.values())) + ["SPY"])
    r63 = {e: mkt[e]["Close"].pct_change(63) for e in mkt}
    spy = r63.get("SPY")
    if spy is None:
        return {"error": "no SPY candles"}

    def cap(df):
        return (SIGNALS["new_52low"][1](df).fillna(False).values
                | SIGNALS["rsi_oversold20"][1](df).fillna(False).values)

    from seq_fundamental_study import load_financial_reports
    reports = load_financial_reports(tks)   # PIT: shares + TTM net income known as-of entry
    MIN_MCAP = 300e6
    H = horizon
    paths = {"A_gated": [], "either_gated": []}
    cd = load_candles([t for t in tks if t in t2etf])
    for tk, df in cd.items():
        if len(df) < 260:
            continue
        # Build point-in-time step-series from quarterly filings: shares_outstanding and
        # TTM net income, each keyed by the date it became public (avail_date). Everything
        # used to gate a trade is knowable strictly before the entry bar.
        rep = reports.get(tk)
        if rep is None or not len(rep):
            continue
        r2 = rep.dropna(subset=["avail_date", "shares_outstanding"]).sort_values("avail_date")
        if not len(r2):
            continue
        pit_dates = pd.DatetimeIndex(pd.to_datetime(r2["avail_date"]).to_numpy())
        pit_sh = r2["shares_outstanding"].to_numpy(dtype=float)
        _prepare_indicators(df)
        close = df["Close"].values
        n = len(close)
        idx = df.index
        st = _ad_state(df).values
        hi = pd.Series(close).rolling(252).max().values
        lo = pd.Series(close).rolling(252).min().values
        posn = (close - lo) / np.where(hi - lo == 0, np.nan, hi - lo)
        e = r63.get(t2etf[tk])
        A = cap(df) & (st == 2)
        sma200 = pd.Series(close).rolling(200).mean().values
        # Mode B = dip in an uptrend: above the 200dMA + RSI(10)<30. Merges the old near-52wk-high
        # B and commodity momentum C into one rule (posn retained only for the %52w display).
        Bm = (df["_rsi"].values < 30) & (close > sma200)
        for p in range(252, n - 2):
            if not (A[p] or Bm[p]):
                continue
            if close[p] < 5:  # penny check is point-in-time (price at entry)
                continue
            j = int(pit_dates.searchsorted(idx[p], "right")) - 1  # last filing public before entry
            if j < 0:
                continue
            sh = pit_sh[j]
            if np.isnan(sh) or sh * close[p] < MIN_MCAP:
                continue  # micro-cap AT ENTRY (point-in-time market cap)
            # Profitability is NOT a gate (user's call): it excluded unprofitable-at-entry
            # turnarounds, which are some of the best distress trades. Kept PIT-clean by dropping.
            d = idx[p]
            rv = e.asof(d)
            sv = spy.asof(d)
            if not (pd.notna(rv) and pd.notna(sv) and rv > sv):
                continue
            row = np.full(H, np.nan)
            for k in range(1, H + 1):
                if p + k < n:
                    row[k - 1] = (close[p + k] - close[p]) / close[p] * 100
            if A[p]:
                paths["A_gated"].append(row)
            paths["either_gated"].append(row)

    out = {"computed_at": timezone.now().isoformat(), "horizon": H,
           "note": "avg/median forward path from entry; day-H = held to the cap", "modes": {}}
    for key, rows in paths.items():
        if not rows:
            continue
        M = np.array(rows)
        mask = ~np.isnan(M)                       # only trades that have reached day k
        avg = np.nanmean(M, axis=0)
        win = ((M > 0) & mask).sum(axis=0) / np.maximum(mask.sum(axis=0), 1) * 100
        med = np.nanmedian(M, axis=0)
        peak = int(np.nanargmax(avg)) + 1
        out["modes"][key] = {
            "n": len(rows),
            "curve": [{"day": i + 1, "avg": round(float(avg[i]), 2),
                       "win": round(float(win[i]), 1), "median": round(float(med[i]), 2)}
                      for i in range(H)],
            "day_final": {"day": H, "avg": round(float(avg[-1]), 2),
                          "win": round(float(win[-1]), 1), "median": round(float(med[-1]), 2)},
            "peak": {"day": peak, "avg": round(float(avg[peak - 1]), 2)},
        }
    if save:
        d = Path(__file__).resolve().parent.parent / ".data" / "studies"
        d.mkdir(parents=True, exist_ok=True)
        (d / "strategy_forward.json").write_text(json.dumps(out, indent=2))
        logger.info("compute_strategy_forward: saved %d modes", len(out["modes"]))
    return out


def compute_research(save=True):
    """RESEARCH/LAB: one shared trade-gen → all the strategy comparisons we ran, cached for
    the app: trigger×exit matrix, entry-timeframe (daily/weekly/mix), regime, cap-band risk,
    and MPT allocation. All portfolio metrics use the SPY-overlay + SPY<200dMA-cash sim.
    Heavy (~few min). Saves research.json."""
    import pandas as pd
    import numpy as np
    import ta
    from pathlib import Path
    from django.utils import timezone
    from seq_fundamental_study import build_universe, load_candles, load_financial_reports
    from studies import SIGNALS, EXITS, _exit_sort_above
    from all_on_all_study import _prepare_indicators
    from pit_fundamentals import _ad_state
    from core.models import Fundamental

    COST, CAP0, MAXPOS = 0.003, 100_000.0, 8
    EXK = ["sort_gt1", "rsi_80", "trail_20"]
    tks = build_universe()
    sec = {r["ticker"]: r["sector"] for r in Fundamental.objects.filter(ticker__in=tks).values("ticker", "sector")}
    t2etf = {t: GICS2ETF[s] for t, s in sec.items() if s in GICS2ETF}
    reports = load_financial_reports(tks)
    mkt = load_candles(sorted(set(t2etf.values())) + ["SPY"])
    spy = mkt["SPY"]["Close"]; spy63 = spy.pct_change(63); spy200 = spy.rolling(200).mean(); spret = spy.pct_change()
    try:
        import rates as _R
        _rd = _R.get_rates("5y")
        curve = None
        if "rate_3m" in _rd.columns:
            longc = next((c for c in _rd.columns if c not in ("rate_3m",) and _rd[c].dropna().mean() > _rd["rate_3m"].dropna().mean()), None)
            if longc:
                curve = (_rd[longc] - _rd["rate_3m"])
    except Exception:
        curve = None

    def exits_for(df, p, close, n, idx):
        out = {}
        for k in EXK:
            try:
                xi = EXITS[k][1](df, p)
            except Exception:
                xi = min(p + 90, n - 1)
            xi = min(max(xi if xi else p + 1, p + 1), n - 1)
            out[k] = (float(close[xi]), idx[xi], pd.Series(close[p:xi + 1], index=idx[p:xi + 1]))
        return out

    daily, weekly, mix, trig_recs = [], [], [], []
    cd = load_candles([t for t in tks if t in t2etf])
    for tk, df in cd.items():
        if len(df) < 260:
            continue
        rep = reports.get(tk)
        if rep is None or not len(rep):
            continue
        r2 = rep.dropna(subset=["avail_date", "shares_outstanding"]).sort_values("avail_date")
        if not len(r2):
            continue
        pdd = pd.DatetimeIndex(pd.to_datetime(r2["avail_date"]).to_numpy()); shv = r2["shares_outstanding"].to_numpy(dtype=float)
        _prepare_indicators(df)
        close = df["Close"].values; n = len(close); idx = df.index
        st = _ad_state(df).values
        e = mkt[t2etf[tk]]["Close"]; e63 = e.pct_change(63)
        hi = pd.Series(close).rolling(252).max().values; lo = pd.Series(close).rolling(252).min().values
        pos = (close - lo) / np.where(hi - lo == 0, np.nan, hi - lo)
        cap = (SIGNALS["new_52low"][1](df).fillna(False).values | SIGNALS["rsi_oversold20"][1](df).fillna(False).values)
        xc = SIGNALS["seq_rsi20_rsi_10d"][1](df).fillna(False).values
        so30 = SIGNALS["rsi_oversold30"][1](df).fillna(False).values
        # raw trigger flags (no accumulation baked in) for the trigger x accumulation grid
        s52r = SIGNALS["new_52low"][1](df).fillna(False).values
        rsi10 = df["_rsi"].values if "_rsi" in df.columns else ta.momentum.rsi(df["Close"], 10).values
        o20v = rsi10 < 20; o25v = rsi10 < 25
        r14 = ta.momentum.rsi(df["Close"], 14); s14 = r14.rolling(10).mean()
        crs14 = ((r14 > s14) & (r14.shift(1) <= s14.shift(1))).fillna(False).values
        b30_14 = (r14 < 30).rolling(10, min_periods=1).max().fillna(0).astype(bool).values
        new14 = crs14 & b30_14
        wdf = df.resample("W-FRI").agg({"Open": "first", "High": "max", "Low": "min", "Close": "last", "Volume": "sum"}).dropna()
        wrsi = ta.momentum.rsi(wdf["Close"], 10)
        wlow = wdf["Close"] == wdf["Close"].rolling(52).min()
        wpos = (wdf["Close"] - wdf["Close"].rolling(52).min()) / (wdf["Close"].rolling(52).max() - wdf["Close"].rolling(52).min())
        wst = pd.Series(_ad_state(wdf).values, index=wdf.index)
        wA = ((wrsi < 20) | wlow) & (wst == 2); wB = (wrsi < 30) & (wpos >= 0.75)
        wrsi_d = wrsi.reindex(idx, method="ffill")

        def gq(p):
            d = idx[p]
            if not (pd.notna(e63.asof(d)) and pd.notna(spy63.asof(d)) and e63.asof(d) > spy63.asof(d)):
                return None
            j = int(pdd.searchsorted(d, "right")) - 1
            if j < 0 or np.isnan(shv[j]) or shv[j] * close[p] < 300e6 or close[p] < 5:
                return None
            return shv[j] * close[p]

        lastd = lastw = lastm = -99
        for p in range(252, n - 2):
            d = idx[p]
            al = bool(cap[p] and st[p] == 2); ac = bool(xc[p] and st[p] == 2); b = bool(so30[p] and pos[p] >= 0.75)
            # light record for the trigger x accumulation grid (superset, sort_gt1 return only)
            if s52r[p] or o20v[p] or o25v[p] or xc[p] or new14[p] or b:
                mc2 = gq(p)
                if mc2:
                    xi2 = _exit_sort_above(df, p, 1); xi2 = min(max(xi2, p + 1), n - 1)
                    trig_recs.append({"tk": tk, "p": p, "s52": bool(s52r[p]), "o20": bool(o20v[p]),
                                      "o25": bool(o25v[p]), "cross20": bool(xc[p]), "new14": bool(new14[p]),
                                      "accum": bool(st[p] == 2), "b": b,
                                      "ret": (close[xi2] - close[p]) / close[p] * 100})
            if (al or ac or b) and p - lastd >= 10:
                mc = gq(p)
                if mc:
                    lastd = p
                    daily.append({"entry": d, "conv": "A" if (al or ac) else "B", "ep": float(close[p]),
                                  "a_level": al, "a_cross": ac, "b": b, "mcap": mc,
                                  "q": str(d.to_period("Q")), "exits": exits_for(df, p, close, n, idx)})
            wr = wrsi_d.iloc[p]
            mA = (cap[p] and st[p] == 2) and pd.notna(wr) and wr < 40
            mB = (so30[p] and pos[p] >= 0.75) and pd.notna(wr) and wr < 55
            if (mA or mB) and p - lastm >= 10:
                mc = gq(p)
                if mc:
                    lastm = p
                    mix.append({"entry": d, "conv": "A" if mA else "B", "ep": float(close[p]), "exits": exits_for(df, p, close, n, idx)})
        for wd in wdf.index[52:]:
            if not (bool(wA.get(wd, False)) or bool(wB.get(wd, False))):
                continue
            loc = idx.searchsorted(wd)
            if loc >= n - 2 or loc < 252 or loc - lastw < 10:
                continue
            mc = gq(loc)
            if mc:
                lastw = loc
                weekly.append({"entry": idx[loc], "conv": "A" if bool(wA.get(wd, False)) else "B",
                               "ep": float(close[loc]), "exits": exits_for(df, loc, close, n, idx)})

    if not daily:
        logger.warning("compute_research: no daily entries qualified — skipping (empty result)")
        return {"error": "no daily entries", "matrices": {}}
    cal = spy.index[spy.index >= min(x["entry"] for x in daily)]
    oos = cal[cal >= "2025-01-01"]

    def sim(items, exit_key, cal_slice, regime="spy"):
        be = {}
        for en in items:
            be.setdefault(en["entry"], []).append(en)
        base = CAP0; mode = "spy"; op = []; eq = []
        for d in cal_slice:
            if regime == "none":
                on = True
            elif regime == "curve" and curve is not None:
                nowc = curve.reindex([d], method="ffill").iloc[0]; pc = curve.reindex([d - pd.Timedelta(days=63)], method="ffill").iloc[0]
                on = (pd.notna(spy200.asof(d)) and spy.asof(d) > spy200.asof(d)) and pd.notna(nowc) and pd.notna(pc) and nowc >= pc
            else:
                on = pd.notna(spy200.asof(d)) and spy.asof(d) > spy200.asof(d)
            base *= (1 + (spret.asof(d) if mode == "spy" else 0.04 / 252)) if pd.notna(spret.asof(d)) else 1
            want = "spy" if on else "cash"
            if want != mode:
                base *= (1 - COST); mode = want
            keep = []
            for o in op:
                if d >= o["xd"]:
                    base += o["sh"] * o["xp"] * (1 - COST)
                else:
                    keep.append(o)
            op = keep
            if on:
                for en in sorted(be.get(d, []), key=lambda x: 0 if x["conv"] == "A" else 1):
                    if len(op) >= MAXPOS:
                        break
                    xp, xd, path = en["exits"][exit_key]
                    stk = sum(o["sh"] * float(o["path"].asof(d)) for o in op)
                    size = min(base, (base + stk) / MAXPOS)
                    if size < 100:
                        continue
                    base -= size
                    op.append({"sh": size * (1 - COST) / en["ep"], "xp": xp, "xd": xd, "path": path})
            eq.append(base + sum(o["sh"] * float(o["path"].asof(d)) for o in op))
        return pd.Series(eq, index=cal_slice)

    def mtr(eq):
        r = eq.pct_change().dropna(); yrs = (eq.index[-1] - eq.index[0]).days / 365.25
        return {"cagr": round(float((eq.iloc[-1] / eq.iloc[0]) ** (1 / yrs) - 1) * 100, 1),
                "dd": round(float((eq / eq.cummax() - 1).min()) * 100, 1),
                "sharpe": round(float(eq.pct_change().dropna().mean() / eq.pct_change().dropna().std() * np.sqrt(252)), 2) if r.std() > 0 else 0.0,
                "oos_cagr": round(float((eq.reindex(oos).iloc[-1] / eq.reindex(oos).iloc[0]) ** (365.25 / (oos[-1] - oos[0]).days) - 1) * 100, 1) if len(oos) > 30 else None}

    TR = {"level": lambda x: x["a_level"], "+cross": lambda x: x["a_level"] or x["a_cross"], "cross-only": lambda x: x["a_cross"]}
    matrix = []
    for tn, tf in TR.items():
        sub = [x for x in daily if tf(x) or x["b"]]
        for ek in EXK:
            m = mtr(sim(sub, ek, cal))
            matrix.append({"trigger": tn, "exit": ek, **m, "n": len(sub)})
    timeframe = []
    for nm, items in [("daily", daily), ("weekly", weekly), ("mix", mix)]:
        m = mtr(sim(items, "sort_gt1", cal))
        timeframe.append({"tf": nm, "n": len(items), **m})
    regimes = []
    for nm in (["none", "spy", "curve"] if curve is not None else ["none", "spy"]):
        m = mtr(sim(daily, "sort_gt1", cal, regime=nm))
        regimes.append({"regime": {"none": "no regime", "spy": "cash when SPY<200dMA", "curve": "SPY + curve-steepening"}[nm], **m})
    # cap-band risk + MPT (sort_gt1 return per daily entry)
    for x in daily:
        xp = x["exits"]["sort_gt1"][0]; x["_ret"] = (xp - x["ep"]) / x["ep"] * 100
    BANDS = [(300e6, 2e9, "small $0.3-2B"), (2e9, 10e9, "mid $2-10B"), (10e9, 50e9, "large $10-50B"), (50e9, 1e15, "mega ≥$50B")]
    capband = []
    for lo_, hi_, nm in BANDS:
        rr = np.array([x["_ret"] for x in daily if lo_ <= x["mcap"] < hi_])
        if len(rr) >= 5:
            capband.append({"band": nm, "n": len(rr), "win": round(float((rr > 0).mean() * 100), 0),
                            "avg": round(float(rr.mean()), 1), "hit50": round(float((rr >= 50).mean() * 100), 1)})
    # MPT: quarterly return per band -> corr + inverse-vol weights
    qdf = pd.DataFrame([{"q": x["q"], "band": next((nm for lo_, hi_, nm in BANDS if lo_ <= x["mcap"] < hi_), None), "r": x["_ret"]} for x in daily])
    qdf = qdf.dropna()
    Q = qdf.groupby(["q", "band"])["r"].mean().unstack("band")
    mpt = None
    if Q.shape[1] >= 2 and len(Q) >= 4:
        mu = Q.mean(); sd = Q.std(); invv = (1 / sd) / (1 / sd).sum()
        mpt = {"bands": list(Q.columns), "inverse_vol_weights": [round(float(w) * 100, 0) for w in invv],
               "sharpe_per_band": {c: round(float(Q[c].mean() / Q[c].std()), 2) if Q[c].std() > 0 else None for c in Q.columns}}

    # ⑥ Trigger × accumulation grid (trade-level; sort_gt1 return, Mode A only, no Mode B)
    DEFS = [("new_52low", lambda r: r["s52"]), ("RSI(10)<20", lambda r: r["o20"]),
            ("RSI(10)<25", lambda r: r["o25"]), ("RSI(10)<20 then cross", lambda r: r["cross20"]),
            ("RSI(14)<30 then cross", lambda r: r["new14"]),
            ("52low OR RSI<20", lambda r: r["s52"] or r["o20"]),
            ("52low OR RSI<20 OR cross", lambda r: r["s52"] or r["o20"] or r["cross20"])]
    srt = sorted(trig_recs, key=lambda z: (z["tk"], z["p"]))
    triggers = []
    for name, fn in DEFS:
        for req in [True, False]:
            last = {}; rr = []
            for r in srt:
                if not fn(r) or (req and not r["accum"]):
                    continue
                if r["p"] - last.get(r["tk"], -99) < 10:
                    continue
                last[r["tk"]] = r["p"]; rr.append(r["ret"])
            if len(rr) >= 10:
                a = np.array(rr)
                triggers.append({"trigger": name, "accum": req, "n": len(rr),
                                 "win": round(float((a > 0).mean() * 100), 0),
                                 "avg": round(float(a.mean()), 1), "median": round(float(np.median(a)), 1)})

    out = {"computed_at": timezone.now().isoformat(), "n_daily": len(daily), "n_weekly": len(weekly), "n_mix": len(mix),
           "matrix": matrix, "timeframe": timeframe, "regimes": regimes, "capband": capband, "mpt": mpt,
           "triggers": triggers}
    if save:
        d_ = Path(__file__).resolve().parent.parent / ".data" / "studies"
        d_.mkdir(parents=True, exist_ok=True)
        (d_ / "research.json").write_text(json.dumps(out))
        logger.info("compute_research: %d daily / %d weekly / %d mix entries; %d matrix rows",
                    len(daily), len(weekly), len(mix), len(matrix))
    return out


def compute_playbook(recent=10, save=True):
    """LIVE end-to-end playbook: the sector board (which of the 11 GICS/SPDR sectors are
    rotating IN / TURNING / OUT) + today's ranked candidates that pass the full funnel
    (Mode A distress+accum-divergence or Mode B uptrend-dip, in a non-OUT sector, quality
    ≥$300M / ≥$5, with the trigger + historical edge + smart-money flags). Trailing-20%
    exit plan attached. Saves JSON the Playbook page reads."""
    import pandas as pd
    import numpy as np
    from pathlib import Path
    from datetime import date, timedelta
    from django.db.models import Sum, Count
    from django.utils import timezone
    from seq_fundamental_study import build_universe, load_candles, load_fundamentals
    from pit_fundamentals import _ad_state, bucket_ad
    from studies import SIGNALS
    from core.models import Fundamental, InsiderBuy, SecFiling, EarningsEvent, NewsItem

    etf2gics = {v: k for k, v in GICS2ETF.items()}
    etfs = sorted(set(GICS2ETF.values()))
    mkt = load_candles(etfs + ["SPY", "^VIX"])
    spy = mkt["SPY"]["Close"]
    spy_riskoff = bool(len(spy) >= 200 and spy.iloc[-1] < spy.rolling(200).mean().iloc[-1])  # market bear?
    vix_regime = _vix_regime(mkt["^VIX"]["Close"] if "^VIX" in mkt else None)
    vix_level = round(float(mkt["^VIX"]["Close"].iloc[-1]), 1) if "^VIX" in mkt and len(mkt["^VIX"]) else None

    # ── Step 1: sector board ──
    # A sector is tradeable if it's rotating in (relative RS>0) OR in its own absolute uptrend
    # (ETF > its 200dMA). The absolute check is what catches bear-market leaders like Energy /
    # Materials in 2022 — up on their own even while SPY is below its 200dMA.
    sectors, etf_state = [], {}
    for e in etfs:
        if e not in mkt:
            continue
        ec = mkt[e]["Close"]
        spa = spy.reindex(ec.index).ffill()
        rs_lag = float(ec.pct_change(63).iloc[-1] - spa.pct_change(63).iloc[-1]) * 100
        rs = ec / spa
        rs_up = float(rs.iloc[-1] - rs.rolling(20).mean().iloc[-1]) > 0
        abs_up = bool(len(ec) >= 200 and ec.iloc[-1] > ec.rolling(200).mean().iloc[-1])  # own uptrend
        try:
            accum = float(_ad_state(mkt[e]).iloc[-1]) == 2
        except Exception:
            accum = False
        if rs_lag > 0:
            state = "IN"
        elif abs_up:
            state = "LEADER" if spy_riskoff else "STRONG"   # up on its own (bear-market leader)
        elif rs_up or accum:
            state = "TURNING"
        else:
            state = "OUT"
        etf_state[e] = state
        sectors.append({"sector": etf2gics.get(e, e), "etf": e, "rs_63d": round(rs_lag, 1),
                        "rs_turning_up": rs_up, "sector_accum": accum, "abs_uptrend": abs_up, "state": state})
    sectors.sort(key=lambda s: -s["rs_63d"])

    # ── Step 3: candidates ──
    tks = build_universe()
    sec = {r["ticker"]: r["sector"] for r in Fundamental.objects.filter(ticker__in=tks).values("ticker", "sector")}
    funds = load_fundamentals(tks)
    edges = _ad_best_edges(AD_CAPIT_SIGNALS)
    # include OUT sectors too: off-gate Mode A is surfaced (flagged); Mode B still needs non-OUT
    pool = [t for t in tks if GICS2ETF.get(sec.get(t))]
    cd = load_candles(pool)
    cands = []
    for tk, df in cd.items():
        if len(df) < 260:
            continue
        gics = sec.get(tk); etf = GICS2ETF.get(gics)
        f = funds.get(tk, {})
        price = float(df["Close"].iloc[-1]); mcap = f.get("market_cap")
        if price < 5 or (mcap is not None and mcap < 300e6):
            continue  # tails in (small→large) but no penny-microcaps
        st = _ad_state(df).values
        if pd.isna(st[-1]):
            continue
        adlabel = bucket_ad(float(st[-1]))
        close = df["Close"].values
        hi = pd.Series(close).rolling(252).max().iloc[-1]; lo = pd.Series(close).rolling(252).min().iloc[-1]
        pos = (price - lo) / (hi - lo) if hi > lo else None
        sma200 = pd.Series(close).rolling(200).mean().iloc[-1]
        import ta
        rsi_recent = ta.momentum.rsi(df["Close"], window=10).iloc[-recent:]
        # Mode A: capitulation fired within `recent` bars AND in accum-divergence now
        fires = []
        sig_full = {}   # full signal series, computed ONCE per ticker, reused by the knife check below
        for skk in AD_CAPIT_SIGNALS:
            try:
                s = SIGNALS[skk][1](df).fillna(False)
            except Exception:
                continue
            sig_full[skk] = s
            r = s.iloc[-recent:].tolist()
            if any(r):
                fires.append({"key": skk, "name": SIGNALS[skk][0],
                              "days_ago": next(i for i, v in enumerate(reversed(r)) if v)})
        is_a = bool(fires) and adlabel == "accum divergence"
        # Mode B = "dip in an uptrend": price above its 200dMA (uptrend) + RSI(10) dipped <35 in
        # the window (pullback). MERGES the old near-52wk-high B and commodity momentum-pullback C
        # into one rule — dropping the 52wk-high requirement for >200dMA gave ~5x the trades at
        # equal win rate / average return (universe backtest, PIT + sector-gated).
        up200 = bool(price > sma200) if pd.notna(sma200) else False
        rsi_b_days = next((i for i, v in enumerate(reversed((rsi_recent < 30).tolist())) if v), None)
        rsi35_days = next((i for i, v in enumerate(reversed((rsi_recent < 35).tolist())) if v), None)
        rsi_now = float(rsi_recent.iloc[-1]) if len(rsi_recent) and pd.notna(rsi_recent.iloc[-1]) else 99
        state = etf_state.get(etf)
        st_now = int(st[-1]) if not pd.isna(st[-1]) else 0
        # require the dip to still be LIVE (RSI not yet recovered) so the playbook lists
        # actionable-today setups, not ones that already bounced days ago.
        is_b = up200 and rsi_b_days is not None and rsi_now < 45
        # BELOW-TREND DIP (user: "a good stock in a good sector with good news, below its 200dMA,
        # can still perform"). We can't read news, so we surface a below-200 dip ONLY when the other
        # ingredients are present — a non-OUT sector AND smart money already ACCUMULATING it (A/D
        # state 1 or 2). You supply the catalyst read. Rated High (no trend support); not in the
        # backtested core (below-200 dips underperformed as a rule — RKLB-July was one).
        below_trend = ((not up200) and state != "OUT" and rsi35_days is not None
                       and rsi_now < 45 and st_now in (1, 2))
        # DEEP DRAWDOWN FROM ATH (vol-adjusted) — discretionary opportunity screen. "Unusually deep
        # FOR THIS STOCK's volatility": threshold scales with the stock's own annualized vol so a
        # habitual 50%-swinger (RKLB) needs a deeper fall to qualify than a calm name. Study: at
        # >$10B this recovers ~+25%/yr reliably (median≈avg, no survivorship tail); at <$2B it's a
        # knife (negative at 6mo) — so the risk rating (cap-weighted) does the separating. Only for
        # a RECENT ATH (peak within ~1y), else it's ancient history, not a fresh opportunity.
        athv = float(np.maximum.accumulate(close)[-1]); ath_i = int(np.argmax(close))
        dd_now = (price / athv - 1.0) if athv > 0 else 0.0
        annv = float(pd.Series(close[-252:]).pct_change().std() * (252 ** 0.5)) if len(close) >= 252 else 0.5
        dd_thr = min(0.65, max(0.30, 1.3 * annv))
        deep_dd = (dd_now <= -dd_thr and (len(close) - 1 - ath_i) <= 252
                   and not (is_a or is_b or below_trend))
        if not (is_a or is_b or below_trend or deep_dd):
            continue
        # Sector gate is OPTIONAL, not a wall. Mode B (a plain dip) still REQUIRES a non-OUT
        # sector. But Mode A (capitulation + accumulation) is self-confirming, so we still SURFACE
        # it even in an OUT sector — flagged off_gate so it can be taken discretionarily (this is
        # how PODD-in-June gets caught). Backtest note: making A ungated by DEFAULT hurt the book
        # (+43%→+28% CAGR, deeper DD) — most distress in weak sectors is a falling knife — so it's
        # surfaced-and-flagged, not auto-included in the ranked core.
        # Sector is a MENTION, not a wall (user directive): every Mode A/B candidate is surfaced
        # regardless of sector; the sector STATE is shown as the trade's POTENTIAL (IN/LEADER =
        # strong tailwind; OUT = weak, discretionary). off_gate flags the OUT ones so they sink to
        # the bottom and read as speculative. NOTE: the backtested strategy still GATES on sector —
        # ungating drops it to ~+11% CAGR — so OUT-sector picks are discretionary, not core.
        off_gate = bool(state == "OUT")
        best = None
        for fr in fires:
            e2 = edges.get(fr["key"])
            if e2 and (best is None or (e2["hist_avg_return"] or -1e9) > (best["hist_avg_return"] or -1e9)):
                best = e2
        B_EDGE = {"hist_avg_return": 11.4, "hist_win_rate": 64, "hist_trades": 898}  # >200dMA+RSI<30, ~6mo hold
        if is_a:
            trig = ", ".join(fr["name"] for fr in fires); dago = min((fr["days_ago"] for fr in fires), default=None); edge = best
        elif is_b:
            trig = "Dip in uptrend (RSI<30, >200dMA)"; dago = rsi_b_days; edge = B_EDGE
        elif below_trend:
            trig = "Below-trend dip (below 200dMA · being accumulated · needs a catalyst)"; dago = rsi35_days; edge = None
        else:  # deep_dd — discretionary opportunity, edge is cap-dependent (rating handles it)
            trig = f"Deep drawdown {round(dd_now * 100)}% from ATH (unusual for its volatility)"; dago = None; edge = None
        # KNIFE check: has this SAME capitulation signal fired before at HIGHER prices and the stock
        # kept falling? Then the accumulation has been early/wrong (PODD: fired $216 Mar → $139 Jun →
        # $133 Aug, still dropping). Counts distinct prior Mode-A episodes in the last ~9 months
        # (excluding today's cluster) that fired above the current price.
        try:
            def _full(k):  # reuse the series already computed in the fires loop when present
                s = sig_full.get(k)
                if s is None:
                    s = SIGNALS[k][1](df).fillna(False)
                return s.values
            _capA = ((_full("new_52low") | _full("rsi_oversold20")) & (st == 2))
        except Exception:
            _capA = np.zeros(len(close), bool)
        prior_fires = 0; _fell = False; _last = -99
        for _i in range(max(252, len(close) - 190), len(close) - 4):
            if _capA[_i]:
                if _i - _last > 10:            # a distinct prior capitulation episode (any price)
                    prior_fires += 1
                    if close[_i] > price:      # ...and it fired higher then kept falling
                        _fell = True
                _last = _i
        knife = prior_fires >= 1; knife_fell = _fell
        cands.append({
            "ticker": tk, "mode": "A" if is_a else "B", "sector": gics, "etf": etf,
            "sector_state": state, "last_close": round(price, 2),
            "trigger": trig, "days_ago": dago, "knife": knife, "prior_fires": prior_fires, "knife_fell": knife_fell,
            "ad_state": adlabel, "pct_52w": round(pos * 100, 0) if pos is not None else None,
            "dd_from_ath": round(dd_now * 100), "market_cap": mcap, "pe_ratio": f.get("pe_ratio"),
            "hist_avg_return": (edge or {}).get("hist_avg_return"),
            "hist_win_rate": (edge or {}).get("hist_win_rate"),
            "hist_trades": (edge or {}).get("hist_trades"),
            "off_gate": off_gate, "below_trend": bool(below_trend and not is_a and not is_b),
            "deep_dd": bool(deep_dd),
        })

    firing_tks = [c["ticker"] for c in cands]
    today = date.today()
    ins = dict(InsiderBuy.objects.filter(ticker__in=firing_tks, filed_date__gte=today - timedelta(days=180))
               .values_list("ticker").annotate(s=Sum("buy_value")))
    sc = {}
    for r in (SecFiling.objects.filter(ticker__in=firing_tks, form_group__in=["13D", "13G"],
                                       filed_date__gte=today - timedelta(days=180))
              .values("ticker", "form_group").annotate(n=Count("id"))):
        sc.setdefault(r["ticker"], {})[r["form_group"]] = r["n"]
    # most-recent earnings event per candidate within the ~1-3mo drift window — for the validated
    # downside-PEAD penalty (a stock that just MISSED keeps underperforming abnormally for weeks).
    miss_lookup = {}
    for r in (EarningsEvent.objects.filter(ticker__in=firing_tks, eps_surprise_pct__isnull=False,
                                           report_date__gte=today - timedelta(days=95))
              .order_by("ticker", "-report_date").values("ticker", "report_date", "eps_surprise_pct")):
        miss_lookup.setdefault(r["ticker"], r)   # first per ticker = most recent (ordered desc)
    # most-recent BULLISH high-impact LLM-classified headline per candidate in the last ~15d — for the
    # validated size-conditioned FADE (news_drift_robust.py): a fresh strong bullish-news POP in a
    # MID/SMALL cap gives back abnormally over ~1-3mo; near-zero in mega-caps.
    # LOCAL labels (local_impact/local_rating from news_llm_category.py) — the edge was re-validated on
    # them (news_drift_robust_local.py: STRONG-bull mid 2-10B -12.0% vs Haiku -12.1%, large ~flat).
    news_lookup = {}
    for r in (NewsItem.objects.filter(ticker__in=firing_tks, local_impact__gte=2, local_rating__gt=0,
                                      dt__gte=today - timedelta(days=15))
              .order_by("ticker", "-dt").values("ticker", "dt", "local_rating")):
        news_lookup.setdefault(r["ticker"], r)   # first per ticker = most recent bullish high-impact
    for c in cands:
        c["insider_buy_180d"] = ins.get(c["ticker"])
        c["recent_13d"] = sc.get(c["ticker"], {}).get("13D", 0)
        c["recent_13g"] = sc.get(c["ticker"], {}).get("13G", 0)
        me = miss_lookup.get(c["ticker"])
        if me and (me["eps_surprise_pct"] or 0) < 0:   # only a recent MISS penalizes (beats don't drift)
            c["recent_miss_pct"] = round(me["eps_surprise_pct"], 1)
            c["recent_miss_days"] = (today - me["report_date"]).days
        nw = news_lookup.get(c["ticker"])
        if nw and 0 < (c.get("market_cap") or 0) < 10e9 and c["ticker"] in cd:   # mid/small only
            pop = _news_pop_abn(cd[c["ticker"]], spy, nw["dt"])
            if pop is not None and pop >= 6.0:        # a real bullish POP (β-adjusted day-1 ≥ +6%)
                c["fresh_bull_pop_pct"] = round(pop, 1)
                c["fresh_bull_pop_days"] = (today - pd.Timestamp(nw["dt"]).tz_localize(None).normalize().date()).days
        c["is_commodity"] = False

    # Highest-beta (vs SPY, ~1y) quality name per sector — the most LEVERAGED way to play a pond
    # once it gives you an entry. Informational: high beta amplifies BOTH the up-move and the
    # drawdown, so it's a watchlist, not a blind buy. Beta = cov(stock,SPY)/var(SPY) on daily rets.
    spy_ret = spy.pct_change()
    beta_by_sector = {}
    for tk, df in cd.items():
        gics = sec.get(tk)
        if not gics or GICS2ETF.get(gics) is None or len(df) < 130:
            continue
        f = funds.get(tk, {})
        price = float(df["Close"].iloc[-1]); mcap = f.get("market_cap")
        if price < 5 or (mcap is not None and mcap < 300e6):
            continue
        j = df["Close"].pct_change().rename("s").to_frame().join(spy_ret.rename("m"), how="inner").dropna().tail(252)
        if len(j) < 60 or j["m"].var() == 0:
            continue
        beta = float(j["s"].cov(j["m"]) / j["m"].var())
        cur = beta_by_sector.get(gics)
        if cur is None or beta > cur[1]:
            beta_by_sector[gics] = (tk, beta)
    for s in sectors:
        tb = beta_by_sector.get(s["sector"])
        s["top_beta"] = {"ticker": tb[0], "beta": round(tb[1], 2)} if tb else None

    # Commodities as first-class sectors: merge their board into the sector board and their
    # (Mode A + Mode B) proxy candidates into the candidate list.
    try:
        cb = compute_commodity_board(recent=recent)
        comm_board, comm_cands = cb["board"], cb["candidates"]
    except Exception as ex:
        logger.warning("commodity board failed: %s", ex); comm_board, comm_cands = [], []
    have_tk = {c["ticker"] for c in cands}
    for cc in comm_cands:
        if cc["ticker"] not in have_tk:
            cands.append(cc); have_tk.add(cc["ticker"])
    for s in sectors:
        s["kind"] = "sector"
    comm_sec = [{"sector": c["theme"], "etf": c["anchor"], "rs_63d": c["rs_63d"], "abs_uptrend": c["abs_uptrend"],
                 "rs_turning_up": False, "sector_accum": False, "state": c["state"], "kind": "commodity",
                 "proxies": c["proxies"]} for c in comm_board]
    sectors_all = sectors + comm_sec

    for c in cands:
        c["vix_regime"] = vix_regime  # market fear state at entry — the strongest timing filter
        c["risk"] = _risk_rating(c)   # data-driven Low/Medium/High/Very-High, with drivers

    state_rank = {"IN": 0, "STRONG": 1, "LEADER": 1, "TURNING": 2}
    def _smart(c):  # conviction: insider buying / 13D / 13G present
        return 1 if (c.get("insider_buy_180d") or c.get("recent_13d") or c.get("recent_13g")) else 0
    # Rank: Mode A first (rare, big-win), then IN/LEADER sectors, then smart-money conviction,
    # then freshest dip. Since all Mode B share one generic edge now, conviction + sector strength
    # are what separate the cream.
    cands.sort(key=lambda c: (1 if (c.get("off_gate") or c.get("below_trend") or c.get("deep_dd")) else 0,
                              0 if c["mode"] == "A" else 1,
                              state_rank.get(c["sector_state"], 2), -_smart(c),
                              -(c.get("risk", {}).get("score") or 0),
                              c["days_ago"] if c["days_ago"] is not None else 99,
                              -(c["hist_avg_return"] or 0)))
    n_a_full = sum(1 for c in cands if c["mode"] == "A")
    n_b_full = sum(1 for c in cands if c["mode"] == "B")
    n_offgate_full = sum(1 for c in cands if c.get("off_gate"))
    # Merged Mode B (>200dMA + RSI<30) fires broadly, especially on market-wide pullback days.
    # Serve the top slice of gated candidates so the playbook stays actionable — but ALWAYS keep
    # the off-gate Mode A setups (the whole point is not to hide opportunities outside a leading
    # sector). Totals disclosed so the cap is never silent.
    CAP = 40
    def _disc(c):  # discretionary bucket: OUT-sector / below-trend / deep-drawdown (shown, rated, not core)
        return c.get("off_gate") or c.get("below_trend") or c.get("deep_dd")
    core_cands = [c for c in cands if not _disc(c)][:CAP]                   # gated core (potential)
    disc_cands = [c for c in cands if _disc(c)][:25]                        # discretionary, shown not hidden
    cands = core_cands + disc_cands

    out = {"computed_at": timezone.now().isoformat(), "recent_window": recent,
           "exit_plan": "Per-mode, thesis-symmetric. Mode A: you bought because smart money was ACCUMULATING the crash — exit the moment the A/D line flips to DISTRIBUTION (smart money leaving; ~18d avg, 30% trailing + 189d backstops). Best of all A exits tested: +44.7% CAGR / Sharpe 1.69. Mode B: RIDE the uptrend ~6 months (126 trading days) — don't cut momentum short.",
           "market_regime": "risk-off (SPY < 200dMA)" if spy_riskoff else "risk-on (SPY > 200dMA)",
           "spy_riskoff": spy_riskoff, "vix_regime": vix_regime, "vix_level": vix_level,
           "sectors": sectors_all, "candidates": cands,
           "n_in": sum(1 for s in sectors if s["state"] == "IN"),
           "n_leader": sum(1 for s in sectors_all if s["state"] in ("LEADER", "STRONG")),
           "n_turning": sum(1 for s in sectors_all if s["state"] == "TURNING"),
           "n_commodity": len(comm_sec),
           "n_a": n_a_full, "n_b": n_b_full, "n_shown": len(cands),
           "n_offgate": n_offgate_full,
           "n_candidates_total": n_a_full + n_b_full}
    if save:
        import math
        def _f(v):
            return None if (isinstance(v, float) and not math.isfinite(v)) else v
        out["candidates"] = [{k: _f(v) for k, v in c.items()} for c in cands]
        d = Path(__file__).resolve().parent.parent / ".data" / "studies"
        d.mkdir(parents=True, exist_ok=True)
        (d / "playbook.json").write_text(json.dumps(out, indent=2))
        logger.info("compute_playbook: %d IN / %d TURNING sectors, %d Mode-A / %d Mode-B candidates",
                    out["n_in"], out["n_turning"], out["n_a"], out["n_b"])
    return out


def update_paper_trades(max_new=25):
    """Forward paper-trading of Playbook picks: 'buy' each current candidate the day it first
    appears, mark open positions to market, and close them on the sort_gt1 exit. Builds a live
    out-of-sample track record. Idempotent — safe to run daily."""
    import pandas as pd
    import json as _json
    from pathlib import Path
    from datetime import date
    from django.utils import timezone
    from core.models import PaperTrade
    from seq_fundamental_study import load_candles
    from all_on_all_study import _prepare_indicators
    from pit_fundamentals import _ad_state

    pbf = Path(__file__).resolve().parent.parent / ".data" / "studies" / "playbook.json"
    cands = []
    if pbf.exists():
        try:
            cands = _json.loads(pbf.read_text()).get("candidates", [])[:max_new]
        except Exception:
            cands = []

    open_qs = list(PaperTrade.objects.filter(status="open"))
    open_tks = {p.ticker for p in open_qs}
    cand_tks = {c["ticker"] for c in cands}
    dfs = load_candles(sorted(open_tks | cand_tks))
    now = timezone.now()

    # 1) update / close existing open positions
    for p in open_qs:
        df = dfs.get(p.ticker)
        if df is None or not len(df):
            continue
        try:
            _prepare_indicators(df)
            entry_ts = pd.Timestamp(p.entry_date)
            if entry_ts not in df.index:
                continue
            ei = df.index.get_loc(entry_ts)
            close = df["Close"].values
            # per-mode exit (matches the backtested strategy): Mode A exits when accumulation
            # flips to DISTRIBUTION (smart money leaves; 30% trail + 189d backstops); Mode B
            # rides the uptrend ~6 months (126 trading days).
            if (p.mode or "A") == "B":
                xi = min(ei + 126, len(df) - 1)
            else:
                xi = _exit_mode_a(_ad_state(df).values, close, ei, len(df))
            p.peak_price = round(float(close[ei:].max()), 2)
            p.last_price = round(float(close[-1]), 2)
            if xi < len(df) - 1:  # exit condition triggered on a past bar -> closed
                p.status = "closed"; p.exit_date = df.index[xi].date()
                p.exit_price = round(float(close[xi]), 2)
                p.ret_pct = round((close[xi] - p.entry_price) / p.entry_price * 100, 1)
            else:
                p.ret_pct = round((close[-1] - p.entry_price) / p.entry_price * 100, 1)  # unrealized
            p.updated_at = now
            p.save()
        except Exception:
            continue

    # 2) open new positions for fresh candidates
    opened = 0
    for c in cands:
        tk = c["ticker"]
        if tk in open_tks:
            continue
        df = dfs.get(tk)
        if df is None or not len(df):
            continue
        ed = df.index[-1].date()
        if PaperTrade.objects.filter(ticker=tk, entry_date=ed).exists():
            continue
        px = round(float(df["Close"].iloc[-1]), 2)
        PaperTrade.objects.create(ticker=tk, mode=c.get("mode", ""), sector=c.get("sector", ""),
                                  entry_date=ed, entry_price=px, peak_price=px, last_price=px,
                                  status="open", ret_pct=0.0, hist_avg_return=c.get("hist_avg_return"),
                                  opened_at=now, updated_at=now)
        opened += 1
    logger.info("update_paper_trades: %d open updated, %d newly opened", len(open_qs), opened)
    return {"open": PaperTrade.objects.filter(status="open").count(),
            "closed": PaperTrade.objects.filter(status="closed").count(), "new": opened}


def _exit_mode_a(st, close, p, n, trail=0.30, maxhold=189):
    """Mode A exit — thesis-symmetric: A is entered when smart money ACCUMULATES the crash
    (A/D accum-divergence, state==2), so it exits when the A/D line flips to DISTRIBUTION
    (state==-1, smart money leaving). Backstops: a trailing stop + a max-hold cap so a position
    can't ride a collapse or drift forever. Portfolio-best of all A exits tested (+44.7% CAGR /
    Sharpe 1.69 vs +25.5%/1.24 for the old Sortino>1 exit, same -20% drawdown)."""
    import numpy as np
    peak = close[p]
    for t in range(p + 1, n):
        if close[t] > peak:
            peak = close[t]
        if st[t] == -1 or close[t] <= peak * (1 - trail) or (t - p) >= maxhold:
            return t
    return n - 1


def compute_equity_curve(save=True):
    """Full capital-constrained portfolio backtest of the best config (Mode A+B, sector-gated,
    PIT quality, sort_gt1 exit, idle cash parked in SPY, 100% cash when SPY<200dMA earning the
    T-bill, 0.3% round-trip cost). Saves the daily equity curve vs SPY + metrics (CAGR/maxDD/
    Sharpe, full + out-of-sample). This is the 'does it beat the market' evidence."""
    import pandas as pd
    import numpy as np
    from pathlib import Path
    from django.utils import timezone
    from seq_fundamental_study import build_universe, load_candles, load_financial_reports
    from studies import SIGNALS, _exit_sort_above
    from all_on_all_study import _prepare_indicators
    from pit_fundamentals import _ad_state
    from core.models import Fundamental

    COST, CAP0, MAXPOS = 0.003, 100_000.0, 8
    try:
        import rates as _R
        _rd = _R.get_rates("5y")
        _rc = min(_rd.columns, key=lambda c: _rd[c].dropna().mean())
        rff = (_rd[_rc] / 100.0) if _rd[_rc].dropna().mean() > 1 else _rd[_rc]
    except Exception:
        rff = None

    tks = build_universe()
    sec = {r["ticker"]: r["sector"] for r in Fundamental.objects.filter(ticker__in=tks).values("ticker", "sector")}
    t2etf = {t: GICS2ETF[s] for t, s in sec.items() if s in GICS2ETF}
    reports = load_financial_reports(tks)
    mkt = load_candles(sorted(set(t2etf.values())) + ["SPY", "^VIX"])
    spy = mkt["SPY"]["Close"]; spy63 = spy.pct_change(63); spy200 = spy.rolling(200).mean(); spret = spy.pct_change()
    vxc = mkt["^VIX"]["Close"] if "^VIX" in mkt else None  # VIX-scaled sizing (bet bigger into fear)

    def capsig(df):
        return (SIGNALS["new_52low"][1](df).fillna(False).values | SIGNALS["rsi_oversold20"][1](df).fillna(False).values)

    spy21 = spy.pct_change(21)
    # Two selectable SECTOR GATES (per-ETF boolean series, checked at entry via asof):
    #  aggressive = sector out-performed SPY over 63d (max backtested return, +43% CAGR)
    #  defensive  = sector out-performed SPY over 21d AND above its own 200dMA (more responsive,
    #               far lower drawdown -15%/-12% OOS, best OOS Sharpe 2.0, ~40% less return)
    egate = {}
    for etf in sorted(set(t2etf.values())):
        e = mkt[etf]["Close"]
        egate[etf] = {
            "aggressive": (e.pct_change(63) - spy63.reindex(e.index)) > 0,
            "defensive": ((e.pct_change(21) - spy21.reindex(e.index)) > 0) & (e > e.rolling(200).mean()),
        }

    # precompute per-ticker arrays once (indicators are the expensive part); gate applied per mode
    PT = []
    cd = load_candles([t for t in tks if t in t2etf])
    for tk, df in cd.items():
        if len(df) < 260:
            continue
        rep = reports.get(tk)
        if rep is None or not len(rep):
            continue
        r2 = rep.dropna(subset=["avail_date", "shares_outstanding"]).sort_values("avail_date")
        if not len(r2):
            continue
        pdd = pd.DatetimeIndex(pd.to_datetime(r2["avail_date"]).to_numpy()); sh = r2["shares_outstanding"].to_numpy(dtype=float)
        _prepare_indicators(df)
        close = df["Close"].values; n = len(close); idx = df.index
        st = _ad_state(df).values
        sma200 = pd.Series(close).rolling(200).mean().values
        A = capsig(df) & (st == 2)
        # Mode B = dip in an uptrend (>200dMA + RSI(10)<30); merged old near-high B + commodity C.
        B = (df["_rsi"].values < 30) & (close > sma200)
        PT.append(dict(etf=t2etf[tk], close=close, n=n, idx=idx, st=st, A=A, B=B, pdd=pdd, sh=sh))

    HOLD_B = 126  # per-mode exit: A exits when accumulation flips to DISTRIBUTION; B rides ~6mo

    def gen_trades(gate_mode):
        trades = []
        for P in PT:
            g = egate[P["etf"]][gate_mode]
            A, B, close, n, idx, st, pdd, sh = P["A"], P["B"], P["close"], P["n"], P["idx"], P["st"], P["pdd"], P["sh"]
            le = -1
            for p in range(252, n - 2):
                if p <= le or not (A[p] or B[p]) or close[p] < 5:
                    continue
                d = idx[p]; gv = g.asof(d)
                if not (pd.notna(gv) and bool(gv)):
                    continue
                j = int(pdd.searchsorted(d, "right")) - 1
                if j < 0 or np.isnan(sh[j]) or sh[j] * close[p] < 300e6:
                    continue
                conv = "A" if A[p] else "B"
                xi = _exit_mode_a(st, close, p, n) if conv == "A" else min(p + HOLD_B, n - 1)
                xi = min(max(xi, p + 1), n - 1)
                trades.append({"entry": d, "conv": conv, "ep": close[p], "xp": close[xi],
                               "exit": idx[xi], "path": pd.Series(close[p:xi + 1], index=idx[p:xi + 1])})
                le = xi
        return trades

    def rff_d(d):
        if rff is None:
            return 0.04 / 252
        v = rff.reindex([d], method="ffill").iloc[0]
        return (v if pd.notna(v) else 0.04) / 252

    def run_portfolio(trades):
        by_entry = {}
        for t in trades:
            by_entry.setdefault(t["entry"], []).append(t)
        cal = spy.index[spy.index >= min(t["entry"] for t in trades)]
        base = CAP0; mode = "spy"; op = []; eq = []
        for d in cal:
            on = pd.notna(spy200.asof(d)) and spy.asof(d) > spy200.asof(d)
            base *= (1 + (spret.asof(d) if mode == "spy" else rff_d(d))) if pd.notna(spret.asof(d)) else 1
            want = "spy" if on else "cash"
            if want != mode:
                base *= (1 - COST); mode = want
            keep = []
            for o in op:
                if d >= o["exit"]:
                    base += o["shares"] * o["xp"] * (1 - COST)
                else:
                    keep.append(o)
            op = keep
            if on:
                # VIX-scaled sizing: bet bigger into fear (validated +37.4%/-20.1%/1.43 vs equal
                # +35.4%/-21%/1.39 — better on all three, since you deploy into the washout).
                vm = 1.0
                if vxc is not None:
                    _v = vxc.asof(d)
                    if pd.notna(_v):
                        vm = float(np.clip(_v / 18.0, 0.7, 1.8))
                for t in sorted(by_entry.get(d, []), key=lambda x: {"A": 0, "B": 1}[x["conv"]]):
                    if len(op) >= MAXPOS:
                        break
                    stk = sum(o["shares"] * float(o["path"].asof(d)) for o in op)
                    size = min(base, (base + stk) / MAXPOS * vm)
                    if size < 100:
                        continue
                    base -= size
                    op.append({"shares": size * (1 - COST) / t["ep"], "xp": t["xp"], "exit": t["exit"], "path": t["path"]})
            eq.append(base + sum(o["shares"] * float(o["path"].asof(d)) for o in op))
        return pd.Series(eq, index=cal)

    def metrics(e):
        r = e.pct_change().dropna(); yrs = (e.index[-1] - e.index[0]).days / 365.25
        return {"cagr": round(float((e.iloc[-1] / e.iloc[0]) ** (1 / yrs) - 1) * 100, 1),
                "maxdd": round(float((e / e.cummax() - 1).min()) * 100, 1),
                "sharpe": round(float(r.mean() / r.std() * np.sqrt(252)), 2) if r.std() > 0 else 0.0}

    GATE_LABEL = {
        "aggressive": "Aggressive — sector outperformed SPY over 63d (max return)",
        "defensive": "Defensive — sector outperformed SPY over 21d AND above its 200dMA (smoother, lower drawdown)",
    }
    modes_out = {}
    for gm in ("aggressive", "defensive"):
        trades = gen_trades(gm)
        if not trades:
            continue
        eqs = run_portfolio(trades); cal = eqs.index
        spc = spy.reindex(cal).ffill()
        oos = cal[cal >= "2025-01-01"]
        strat_n = (eqs / eqs.iloc[0] * 100); spy_n = (spc / spc.iloc[0] * 100)
        wk = cal[::5]
        modes_out[gm] = {
            "label": GATE_LABEL[gm], "n_trades": len(trades),
            "dates": [d.strftime("%Y-%m-%d") for d in wk],
            "strategy": [round(float(strat_n.asof(d)), 1) for d in wk],
            "spy": [round(float(spy_n.asof(d)), 1) for d in wk],
            "full": {"strategy": metrics(eqs), "spy": metrics(spc)},
            "oos": {"strategy": metrics(eqs.reindex(oos)), "spy": metrics(spc.reindex(oos))} if len(oos) > 30 else None,
        }
    if not modes_out:
        return {"error": "no trades"}
    agg = modes_out.get("aggressive") or next(iter(modes_out.values()))
    out = {"computed_at": timezone.now().isoformat(),
           "config": "Mode A (capit+accum, exit on A/D DISTRIBUTION) + Mode B (>200dMA dip RSI<30, ride ~6mo) · PIT quality · SPY idle-overlay · cash when SPY<200dMA · VIX-scaled sizing (bet bigger into fear) · 0.3% costs · selectable sector gate",
           "gate_modes": modes_out, "default_gate": "aggressive",
           # back-compat: aggressive mode mirrored at top-level for any older reader
           "n_trades": agg["n_trades"], "dates": agg["dates"], "strategy": agg["strategy"],
           "spy": agg["spy"], "full": agg["full"], "oos": agg["oos"]}
    if save:
        dd = Path(__file__).resolve().parent.parent / ".data" / "studies"
        dd.mkdir(parents=True, exist_ok=True)
        (dd / "equity_curve.json").write_text(json.dumps(out))
        _dfn = modes_out.get("defensive", {})
        logger.info("compute_equity_curve: aggressive %d tr CAGR %s | defensive %s tr CAGR %s",
                    agg["n_trades"], agg["full"]["strategy"]["cagr"],
                    _dfn.get("n_trades"), (_dfn.get("full") or {}).get("strategy", {}).get("cagr"))
    return out


def _ad_best_edges(signal_keys, min_trades=100):
    """Best (highest avg_return) StockStudy exit per capitulation signal — the historical
    edge to attach to a live firing hit."""
    from core.models import StockStudy
    best = {}
    for r in (StockStudy.objects.filter(signal_key__in=signal_keys, total_trades__gte=min_trades)
              .order_by("-avg_return")
              .values_list("signal_key", "exit_key", "avg_return", "win_rate", "total_trades")):
        sk, ek, avg, wr, tr = r
        if sk not in best:  # first seen = highest avg_return for that signal
            best[sk] = {"best_exit_key": ek, "hist_avg_return": avg,
                        "hist_win_rate": wr, "hist_trades": tr}
    return best


def compute_ad_divergence(recent_window=10, save_db=True):
    """Scan the individual-stock universe for stocks whose Accumulation/Distribution LINE is in
    'accum divergence' state on the LATEST bar (price flat/down while the ADL rises — read as
    slope+divergence, never sign). Flag those ALSO firing a capitulation signal (new_52low /
    rsi_oversold20 within the last N bars) as `primed` and join the signal's historical edge,
    fundamentals, sectors, and smart-money. Writes AdDivergenceSignal rows.

    In-process (fast: one A/D-state calc + 2 signal checks per ticker, no exit loop)."""
    import pandas as pd
    from datetime import date, timedelta
    from django.db.models import Sum, Count
    from django.utils import timezone
    from django.db import transaction
    from core.models import AdDivergenceSignal, InsiderBuy, SecFiling
    from seq_fundamental_study import build_universe, load_candles, load_fundamentals, DIMENSIONS
    from pit_fundamentals import _ad_state, bucket_ad
    from studies import SIGNALS

    tickers = build_universe()
    candles = load_candles(tickers)
    funds = load_fundamentals(tickers)
    edges = _ad_best_edges(AD_CAPIT_SIGNALS)

    hits = []
    for tk, df in candles.items():
        if len(df) < 60:
            continue
        state = _ad_state(df)
        if len(state) == 0 or pd.isna(state.iloc[-1]):
            continue
        if bucket_ad(float(state.iloc[-1])) != "accum divergence":
            continue
        firing = []
        fires_60d = 0
        for sk in AD_CAPIT_SIGNALS:
            try:
                sig = SIGNALS[sk][1](df).fillna(False)
            except Exception:
                continue
            fires_60d += int(sig.iloc[-60:].sum())  # serial new-lows = falling knife
            recent = sig.iloc[-recent_window:].tolist()
            if any(recent):
                days_ago = next(i for i, v in enumerate(reversed(recent)) if v)
                firing.append({"signal_key": sk, "signal_name": SIGNALS[sk][0], "days_ago": days_ago})
        close = df["Close"].values
        low60 = float(close[-60:].min())
        pct_above_low = round((float(close[-1]) - low60) / low60 * 100, 1) if low60 > 0 else None
        # Falling knife: firing repeatedly AND still pinned near the low (divergence hasn't
        # resolved into a bounce yet) — the "you'd have lost money holding it" case.
        knife = fires_60d >= 4 and pct_above_low is not None and pct_above_low < 5.0
        hits.append({"ticker": tk, "last_close": round(float(df["Close"].iloc[-1]), 2),
                     "firing": firing, "fires_60d": fires_60d,
                     "pct_above_low": pct_above_low, "knife": knife})

    firing_tks = [h["ticker"] for h in hits]

    def bkts(tk):  # live snapshot -> only snapshot (pit=False) dims are meaningful
        f = funds.get(tk, {})
        return {dim: bfn(f.get(field)) for (dim, field, bfn, _o, pit) in DIMENSIONS if not pit}

    today = date.today()
    ins = dict(InsiderBuy.objects.filter(ticker__in=firing_tks, filed_date__gte=today - timedelta(days=180))
               .values_list("ticker").annotate(s=Sum("buy_value")))
    sec = {}
    for r in (SecFiling.objects.filter(ticker__in=firing_tks, filed_date__gte=today - timedelta(days=180))
              .values("ticker", "form_group").annotate(n=Count("id"))):
        sec.setdefault(r["ticker"], {})[r["form_group"]] = r["n"]

    rows = []
    for h in hits:
        tk = h["ticker"]; f = funds.get(tk, {}); firing = h["firing"]
        best_fr = None; best_e = None
        for fr in firing:  # headline edge = firing signal with best historical avg_return
            e = edges.get(fr["signal_key"])
            if e and (best_e is None or (e["hist_avg_return"] or -1e9) > (best_e["hist_avg_return"] or -1e9)):
                best_e = e; best_fr = fr
        rows.append({
            "ticker": tk, "last_close": h["last_close"], "primed": bool(firing),
            "firing": firing, "min_days_ago": min((fr["days_ago"] for fr in firing), default=None),
            "fires_60d": h["fires_60d"], "pct_above_low": h["pct_above_low"], "knife": h["knife"],
            "best_signal_key": best_fr["signal_key"] if best_fr else "",
            "best_signal_name": best_fr["signal_name"] if best_fr else "",
            "best_exit_key": (best_e or {}).get("best_exit_key", ""),
            "hist_avg_return": (best_e or {}).get("hist_avg_return"),
            "hist_win_rate": (best_e or {}).get("hist_win_rate"),
            "hist_trades": (best_e or {}).get("hist_trades"),
            "market_cap": f.get("market_cap"), "pe_ratio": f.get("pe_ratio"),
            "forward_pe": f.get("forward_pe"), "profit_margin": f.get("profit_margin"),
            "low_quality": is_low_quality(f.get("market_cap"), h["last_close"], f.get("profit_margin")),
            "fund_buckets": bkts(tk),
            "sectors": sector_holdings.get_sectors_for_ticker(tk) if sector_holdings else [],
            "insider_buy_90d": ins.get(tk), "recent_13d": sec.get(tk, {}).get("13D", 0),
            "recent_13g": sec.get(tk, {}).get("13G", 0),
        })

    n_primed = sum(1 for r in rows if r["primed"])
    logger.info("compute_ad_divergence: %d in accum-divergence (%d primed) of %d tickers",
                len(rows), n_primed, len(tickers))

    if save_db:
        now = timezone.now()
        with transaction.atomic():
            for r in rows:
                defaults = {k: v for k, v in r.items() if k != "ticker"}
                defaults["computed_at"] = now
                AdDivergenceSignal.objects.update_or_create(ticker=r["ticker"], defaults=defaults)
            AdDivergenceSignal.objects.exclude(computed_at=now).delete()
        logger.info("compute_ad_divergence: saved %d rows, cleared stale", len(rows))
    return {"n": len(rows), "n_primed": n_primed, "universe": len(tickers)}


def run_financial_history_task(jobs=None):
    """Backfill quarterly financials + dividends into FinancialReport/DividendHistory as a
    clean SUBPROCESS (same spawn reasoning as run_stock_studies_task). Feeds the
    point-in-time fundamental buckets the nightly sweep computes."""
    import subprocess
    script = "/app/fetch_financial_history.py"
    if not os.path.exists(script):
        logger.error("run_financial_history_task: %s not found (mount it in docker-compose)", script)
        return
    cmd = ["python", "-u", script, "--db"]
    if jobs:
        cmd += ["--jobs", str(jobs)]
    logger.info("run_financial_history_task: launching %s", " ".join(cmd))
    proc = subprocess.run(cmd, cwd="/app", capture_output=True, text=True)
    if proc.returncode != 0:
        logger.error("financial history backfill failed (rc=%s): %s", proc.returncode, proc.stderr[-2000:])
    else:
        logger.info("financial history backfill done: %s", proc.stdout[-1000:])
    return proc.returncode


def run_insider_task(start_year=2020):
    """Backfill insider open-market transactions (SEC bulk Form 345) as a SUBPROCESS.
    Idempotent upsert on (ticker, filed_date) — no duplicates on re-run."""
    import subprocess
    script = "/app/fetch_insider.py"
    if not os.path.exists(script):
        logger.error("run_insider_task: %s not found (mount it in docker-compose)", script)
        return
    cmd = ["python", "-u", script, "--db", "--start-year", str(start_year)]
    logger.info("run_insider_task: launching %s", " ".join(cmd))
    proc = subprocess.run(cmd, cwd="/app", capture_output=True, text=True)
    if proc.returncode != 0:
        logger.error("insider backfill failed (rc=%s): %s", proc.returncode, proc.stderr[-2000:])
    else:
        logger.info("insider backfill done: %s", proc.stdout[-1000:])
    return proc.returncode


def run_dim_intersection_task(signal, exit_key, dims, jobs=16):
    """Run the targeted dimension-intersection study as a SUBPROCESS; writes
    .data/studies/dim_intersection.json which DimIntersectionView serves."""
    import subprocess
    script = "/app/dim_intersection_study.py"
    if not os.path.exists(script):
        logger.error("run_dim_intersection_task: %s not found", script)
        return
    cmd = ["python", "-u", script, "--db", "--signal", signal, "--exit", exit_key,
           "--dims", ",".join(dims), "--jobs", str(jobs)]
    logger.info("run_dim_intersection_task: %s", " ".join(cmd))
    proc = subprocess.run(cmd, cwd="/app", capture_output=True, text=True)
    if proc.returncode != 0:
        logger.error("dim intersection failed (rc=%s): %s", proc.returncode, proc.stderr[-2000:])
    return proc.returncode


def run_sec_events_task(jobs=None):
    """Backfill 13D/13G filings (SEC submissions) as a SUBPROCESS. Idempotent upsert on
    accession — no duplicates on re-run."""
    import subprocess
    script = "/app/fetch_sec_events.py"
    if not os.path.exists(script):
        logger.error("run_sec_events_task: %s not found (mount it in docker-compose)", script)
        return
    cmd = ["python", "-u", script, "--db"]
    if jobs:
        cmd += ["--jobs", str(jobs)]
    logger.info("run_sec_events_task: launching %s", " ".join(cmd))
    proc = subprocess.run(cmd, cwd="/app", capture_output=True, text=True)
    if proc.returncode != 0:
        logger.error("13D/13G backfill failed (rc=%s): %s", proc.returncode, proc.stderr[-2000:])
    else:
        logger.info("13D/13G backfill done: %s", proc.stdout[-1000:])
    return proc.returncode


def run_studies_task(rebuild_trades=True):
    """Generate and run all studies that haven't been computed yet.

    `rebuild_trades=False` skips the per-study Trade/StudySectorResult delete+rebuild — use it for
    a pure aggregate/MAE backfill where the signal/exit logic is unchanged, so the existing Trade
    rows are already correct. Rebuilding them is ~43M row deletes+inserts and is the whole
    bottleneck; the Study aggregates (incl. avg_mae/clean_pct) come from the in-memory loop, not
    the DB, so skipping the writes changes nothing but speed. Leave True for the nightly (new
    studies genuinely need their trades built)."""
    if config is None:
        return

    from studies import (SIGNALS, EXITS, generate_studies, CLEAN_MAE_THRESH, trade_mae,
                         _episode_starts, _tstat_from_returns)

    study_defs = generate_studies()
    logger.info(f"Generated {len(study_defs)} study definitions")

    # Create Study records for new ones
    for sd in study_defs:
        Study.objects.get_or_create(
            signal_key=sd["signal"], exit_key=sd["exit"],
            defaults={
                "name": sd["name"],
                "signal_name": sd["signal_name"],
                "exit_name": sd["exit_name"],
                "category": sd["category"],
            }
        )

    # Find uncomputed studies
    uncomputed = Study.objects.filter(is_computed=False)
    if not uncomputed.exists():
        logger.info("All studies already computed")
        return

    logger.info(f"Running {uncomputed.count()} uncomputed studies")

    # Load all sector data once — ONE bulk query for every ETF + the benchmark (was an N+1:
    # _get_df per sector).
    sectors = list(Sector.objects.all())
    dfs = _get_dfs([config.BENCHMARK] + [s.etf for s in sectors])
    all_dfs = {}
    for sector in sectors:
        df = dfs.get(sector.etf)
        if df is not None and len(df) >= 60:
            all_dfs[sector.etf] = (sector, df)

    spy_df = dfs.get(config.BENCHMARK)

    def _run_signal_group(signal_key, group):
        """Compute this signal's entry series ONCE per sector (signals are ~99% of study compute),
        then run all its exit-studies off the cached entries. The old per-study path recomputed the
        same signal 70× (once per exit). Exit calc is negligible, so no exit memoization needed.
        Numerics are unchanged — only the redundant signal recompute is hoisted out."""
        if signal_key not in SIGNALS:
            return 0
        _, sig_fn = SIGNALS[signal_key]
        needs_spy = signal_key in ("rsi_x_pos_updn", "rsi_sup10_x_dd50_mkt", "rsi_sup10_x_mkt")
        # entries[etf] = list of (entry_date, idx) — computed once for this signal
        entries = {}
        for etf, (sector, df) in all_dfs.items():
            try:
                if needs_spy and spy_df is not None:
                    sig = sig_fn(df, spy_close=spy_df["Close"]).fillna(False)
                else:
                    sig = sig_fn(df).fillna(False)
            except Exception:
                continue
            entries[etf] = [(ed, df.index.get_loc(ed)) for ed in sig[sig].index]
        # Independent-episode bars per etf (overlap-dedup for the significance stat).
        episode_by_etf = {etf: _episode_starts([idx for _, idx in ents])
                          for etf, ents in entries.items()}

        n = 0
        for study in group:
            if study.exit_key not in EXITS:
                continue
            _, exit_fn = EXITS[study.exit_key]
            sector_results = []
            trades = []
            total_wins = total_trades = 0
            total_ret = 0
            total_hold = 0
            total_mae = 0.0
            total_clean = 0
            eff = []   # one return per independent episode, pooled across etfs → significance

            for etf, (sector, df) in all_dfs.items():
                ents = entries.get(etf)
                if not ents:
                    continue
                epi = episode_by_etf.get(etf, set())
                wins = losses = strades = 0
                sret = 0
                shold = 0
                max_g = max_l = None   # seed from first trade, not 0 (one-sided sectors)
                smae = 0.0        # sum of per-trade MAE (%)
                sclean = 0        # count of clean (barely-dipped) entries
                close_arr = df["Close"].values
                low_arr = df["Low"].values

                for entry_date, idx in ents:
                    exit_idx = exit_fn(df, idx)
                    if exit_idx is None or exit_idx <= idx:
                        continue

                    ep = float(close_arr[idx])
                    xp = float(close_arr[exit_idx])
                    ret = (xp - ep) / ep * 100
                    hold = exit_idx - idx
                    mae = trade_mae(ep, low_arr[idx + 1:exit_idx + 1])

                    strades += 1
                    sret += ret
                    shold += hold
                    smae += mae
                    if mae >= CLEAN_MAE_THRESH: sclean += 1
                    if ret > 0: wins += 1
                    else: losses += 1
                    if idx in epi:
                        eff.append(ret)
                    max_g = ret if max_g is None else max(max_g, ret)
                    max_l = ret if max_l is None else min(max_l, ret)

                    if rebuild_trades:   # skip building ~43M Trade objects on a MAE-only backfill
                        trades.append(Trade(
                            study=study, sector=sector, etf=etf,
                            entry_date=entry_date.date() if hasattr(entry_date, 'date') else entry_date,
                            exit_date=df.index[exit_idx].date() if hasattr(df.index[exit_idx], 'date') else df.index[exit_idx],
                            entry_price=round(ep, 2), exit_price=round(xp, 2),
                            return_pct=round(ret, 3), hold_days=hold,
                        ))

                if strades > 0:
                    sector_results.append(StudySectorResult(
                        study=study, sector=sector,
                        trades=strades, avg_return=round(sret/strades, 3),
                        total_return=round(sret, 2), win_rate=round(wins/strades*100, 1),
                        wins=wins, losses=losses, avg_hold=round(shold/strades, 1),
                        max_gain=round(max_g, 2), max_loss=round(max_l, 2),
                    ))
                    total_wins += wins
                    total_trades += strades
                    total_ret += sret
                    total_hold += shold
                    total_mae += smae
                    total_clean += sclean

            # Save (Trade/StudySectorResult rebuild skipped on a MAE-only backfill — unchanged rows)
            if rebuild_trades:
                if trades:
                    Trade.objects.filter(study=study).delete()
                    Trade.objects.bulk_create(trades, batch_size=5000)
                if sector_results:
                    StudySectorResult.objects.filter(study=study).delete()
                    StudySectorResult.objects.bulk_create(sector_results)

            study.total_trades = total_trades
            study.eff_trades = len(eff)
            study.t_stat = _tstat_from_returns(eff)
            study.avg_return = round(total_ret / total_trades, 3) if total_trades else 0
            study.win_rate = round(total_wins / total_trades * 100, 1) if total_trades else 0
            study.avg_hold = round(total_hold / total_trades, 1) if total_trades else 0
            study.avg_mae = round(total_mae / total_trades, 2) if total_trades else 0
            study.clean_pct = round(total_clean / total_trades * 100, 1) if total_trades else 0
            study.is_computed = True
            study.computed_at = timezone.now()
            study.save()
            n += 1
        return n

    # Group uncomputed studies by signal_key so each signal is computed once (not once per exit).
    from collections import defaultdict
    groups = defaultdict(list)
    for s in uncomputed:
        groups[s.signal_key].append(s)
    logger.info("Running %d studies across %d signal groups", uncomputed.count(), len(groups))

    with ThreadPoolExecutor(max_workers=4) as pool:
        futures = {pool.submit(_run_signal_group, sk, grp): sk for sk, grp in groups.items()}
        done_groups = done_studies = 0
        for f in as_completed(futures):
            done_groups += 1
            try:
                done_studies += f.result() or 0
            except Exception as e:
                logger.warning("signal group %s failed: %s", futures[f], str(e)[:120])
            if done_groups % 20 == 0:
                logger.info("Signal groups: %d/%d done (%d studies)", done_groups, len(groups), done_studies)

    logger.info(f"All studies complete")


# ── Fresh signal watcher + Slack alert ──

FRESH_STATE_PATH = Path("/app/.data/fresh_state.json")
FRESH_WINDOW = 14


def _post_slack(webhook, text):
    """POST a message to a Slack (or Discord/generic) incoming webhook."""
    import urllib.request
    payload = json.dumps({"text": text}).encode()
    req = urllib.request.Request(
        webhook, data=payload, headers={"Content-Type": "application/json"}
    )
    urllib.request.urlopen(req, timeout=10)


def _current_fresh_map(interval="1d"):
    """Compute the sectors that are FRESH right now, from DB candles.

    FRESH = Sortino>0 + RSI crossover + RSI-of-Sortino crossover + green MACD,
    all within the last FRESH_WINDOW (14) trading days.
    """
    # Bulk-load benchmark + every sector ETF in ONE query (was an N+1: _get_df per sector).
    sectors = list(Sector.objects.all())
    dfs = _get_dfs([config.BENCHMARK] + [s.etf for s in sectors], interval)

    spy_df = dfs.get(config.BENCHMARK)
    if spy_df is None or len(spy_df) < 30:
        return {}

    current = {}
    for sector in sectors:
        etf_df = dfs.get(sector.etf)
        if etf_df is None or len(etf_df) < 60:
            continue
        try:
            metrics = indicators.compute_all_risk_metrics(etf_df, spy_df, config.SORTINO_WINDOW)
            macd = indicators.compute_macd(etf_df)
            fc = indicators.compute_fresh_crossovers(etf_df, window=FRESH_WINDOW, sortino_timeframe="W")
        except Exception as e:
            logger.warning(f"Fresh calc failed for {sector.name}: {e}")
            continue

        rsi_x, rs_x = fc.get("fresh_rsi_x_days"), fc.get("fresh_rs_x_days")
        # FRESH = 3 conditions (weekly Sortino>0 + RSI crossover + RSI-of-Sortino crossover) within
        # the window. MACD was removed from the signal rules 2026-07-24; it must NOT gate the alert
        # (that made the daily alert silently under-fire vs the dashboard). Kept as info-only below.
        conds = [fc.get("fresh_sortino_pos"), rsi_x is not None, rs_x is not None]
        if all(conds):
            # Gate on the EARLIER crossover (both must be inside the window),
            # but the composite only COMPLETES on the LATER (more recent)
            # crossover, so the displayed age is the smaller days-ago value.
            if max(rsi_x, rs_x) <= FRESH_WINDOW:
                fd = min(rsi_x, rs_x)
                current[sector.etf] = {
                    "sector": sector.name,
                    "fresh_days": fd,
                    "fresh_since": str(etf_df.index[-1 - fd].date()),
                    "macd_great": macd.get("macd_great", False),
                    "sortino": metrics.get("sortino"),
                    "omega": metrics.get("omega"),
                }
    return current


def compute_fresh_and_alert(interval="1d"):
    """Detect sectors that became FRESH since the last run and alert to Slack."""
    if indicators is None:
        logger.error("indicators module not found")
        return

    current = _current_fresh_map(interval)

    prev = {}
    if FRESH_STATE_PATH.exists():
        try:
            prev = json.loads(FRESH_STATE_PATH.read_text())
        except Exception:
            prev = {}
    prev_fresh = set(prev.get("fresh", {}).keys())
    cur_fresh = set(current.keys())

    newly = sorted(cur_fresh - prev_fresh, key=lambda e: current[e]["fresh_days"])
    dropped = sorted(prev_fresh - cur_fresh)

    import os
    webhook = os.environ.get("FRESH_ALERT_WEBHOOK")

    if newly:
        lines = []
        for e in newly:
            d = current[e]
            star = " *(MACD>0)*" if d["macd_great"] else ""
            lines.append(
                f"• {d['sector']} ({e}) — fresh since {d['fresh_since']} "
                f"({d['fresh_days']}d){star}; Sortino {d['sortino']}, Omega {d['omega']}"
            )
        text = f"*[Sector Rotation] {len(newly)} newly FRESH ({date.today().isoformat()})*\n" + "\n".join(lines)
        if dropped:
            text += "\nNo longer fresh: " + ", ".join(
                prev["fresh"][e]["sector"] for e in dropped if e in prev.get("fresh", {})
            )
        logger.info(text)
        if webhook:
            try:
                _post_slack(webhook, text)
            except Exception as e:
                logger.warning(f"Slack post failed: {e}")
        else:
            logger.warning("FRESH_ALERT_WEBHOOK not set — alert logged only")
    else:
        logger.info(f"No newly fresh sectors. Currently fresh: {len(cur_fresh)}")

    FRESH_STATE_PATH.parent.mkdir(parents=True, exist_ok=True)
    FRESH_STATE_PATH.write_text(json.dumps(
        {"as_of": date.today().isoformat(),
         "run_at": datetime.now().isoformat(timespec="seconds"),
         "fresh": current}, indent=2))
    return {"newly": newly, "dropped": dropped, "fresh_count": len(cur_fresh)}
