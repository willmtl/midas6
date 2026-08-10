#!/usr/bin/env python3
"""
Stock Market Trend Bot - API Server

Flask API serving sector scan and drill-down results as JSON.
Caches scan and drilldown results to .data/results/ as JSON.
"""

import warnings
warnings.filterwarnings("ignore")

import os
import json
import glob
from datetime import datetime, timedelta
from pathlib import Path

from flask import Flask, Blueprint, jsonify, request, send_from_directory
from flask_cors import CORS

import config
import trend_analyzer
import indicators
import data_fetcher
import sector_holdings

FRONTEND_DIR = os.path.join(os.path.dirname(__file__), 'frontend', 'build')
RESULTS_DIR = Path(__file__).parent / ".data" / "results"
CACHE_MAX_AGE_HOURS = 4

# API blueprint — registered first so /api/* routes take priority
api = Blueprint('api', __name__, url_prefix='/api')


def _results_path(name: str) -> Path:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    return RESULTS_DIR / f"{name}.json"


def _cache_valid(path: Path) -> bool:
    if not path.exists():
        return False
    age = datetime.now() - datetime.fromtimestamp(path.stat().st_mtime)
    return age < timedelta(hours=CACHE_MAX_AGE_HOURS)


def _load_cache(name: str):
    path = _results_path(name)
    if _cache_valid(path):
        with open(path, "r") as f:
            return json.load(f)
    return None


def _save_cache(name: str, data):
    path = _results_path(name)
    with open(path, "w") as f:
        json.dump(data, f)


@api.route("/scan")
def scan():
    period = request.args.get("period", config.DEFAULT_PERIOD)
    interval = request.args.get("interval", config.DEFAULT_INTERVAL)
    force = request.args.get("force", "false") == "true"
    cache_key = f"scan_{period}_{interval}"

    if not force:
        cached = _load_cache(cache_key)
        if cached:
            return jsonify(cached)

    results = trend_analyzer.analyze(period=period, interval=interval)
    for r in results:
        r.pop("omega_series", None)

    response = {
        "spy_sortino": results[0]["spy_sortino"] if results else 0,
        "spy_omega": results[0]["spy_omega"] if results else 0,
        "spy_cvar": results[0]["spy_cvar"] if results else 0,
        "spy_ulcer": results[0]["spy_ulcer"] if results else 0,
        "total": len(results),
        "bullish": sum(1 for r in results if r.get("bullish")),
        "sectors": results,
        "cached_at": datetime.now().isoformat(),
    }

    _save_cache(cache_key, response)
    return jsonify(response)


@api.route("/drilldown/<sector_name>")
def drilldown(sector_name):
    period = request.args.get("period", config.DEFAULT_PERIOD)
    force = request.args.get("force", "false") == "true"

    matched = None
    for name in config.SECTOR_ETFS:
        if name.lower() == sector_name.lower() or sector_name.lower() in name.lower():
            matched = name
            break

    if not matched:
        return jsonify({"error": f"Sector '{sector_name}' not found"}), 404

    cache_key = f"drill_{matched.replace(' ', '_')}_{period}"

    if not force:
        cached = _load_cache(cache_key)
        if cached:
            return jsonify(cached)

    holdings = sector_holdings.get_holdings(matched)
    if not holdings:
        response = {"sector": matched, "etf": config.SECTOR_ETFS.get(matched), "stocks": [], "cached_at": datetime.now().isoformat()}
        _save_cache(cache_key, response)
        return jsonify(response)

    # Fetch ETF weights
    etf_ticker = config.SECTOR_ETFS.get(matched)
    weights = {}
    try:
        import yfinance as yf
        etf_obj = yf.Ticker(etf_ticker)
        h = etf_obj.funds_data.top_holdings
        if h is not None:
            for sym, row in h.iterrows():
                weights[str(sym)] = round(float(row.get("Holding Percent", 0)) * 100, 2)
    except Exception:
        pass

    us_tickers = [t for t in holdings if "." not in t][:20]
    stock_data = data_fetcher.fetch_tickers(us_tickers, period)

    results = []
    for ticker in us_tickers:
        df = stock_data.get(ticker)
        if df is None or df.empty or len(df) < 25:
            continue
        try:
            rsi_data = indicators.compute_rsi_crossover(df)
            if rsi_data.get("rsi") is None:
                continue

            gap_data = indicators.detect_gap(df)

            # Sortino + Beta for individual stock
            stk_ret = indicators.daily_returns(df)
            spy_data_single = data_fetcher.fetch_tickers(["SPY"], period)
            spy_df_single = spy_data_single.get("SPY")
            stk_sortino_val = None
            stk_beta_val = None
            if spy_df_single is not None:
                spy_ret_single = indicators.daily_returns(spy_df_single)
                stk_sortino_s = indicators.rolling_sortino(stk_ret, config.SORTINO_WINDOW)
                if not stk_sortino_s.dropna().empty:
                    stk_sortino_val = round(float(stk_sortino_s.dropna().iloc[-1]), 3)
                stk_beta_s = indicators.rolling_beta(stk_ret, spy_ret_single, config.SORTINO_WINDOW)
                if not stk_beta_s.dropna().empty:
                    stk_beta_val = round(float(stk_beta_s.dropna().iloc[-1]), 2)

            close = float(df["Close"].iloc[-1])
            ret_1w = float((df["Close"].iloc[-1] / df["Close"].iloc[-6] - 1) * 100) if len(df) > 6 else 0
            ret_1m = float((df["Close"].iloc[-1] / df["Close"].iloc[-22] - 1) * 100) if len(df) > 22 else 0

            rsi = rsi_data["rsi"]
            rsi_sma = rsi_data["rsi_sma"] or 0
            rsi_above = rsi_data.get("rsi_above_sma", False)
            crossover = rsi_data.get("rsi_crossover", False)

            if rsi_above:
                signal = "ROTATE IN" if crossover else "BULLISH"
            else:
                signal = "BEARISH"

            results.append({
                "ticker": ticker,
                "weight": weights.get(ticker),
                "price": round(close, 2),
                "return_1w": round(ret_1w, 1),
                "return_1m": round(ret_1m, 1),
                "rsi": rsi,
                "rsi_sma": rsi_sma,
                "rsi_spread": round(rsi - rsi_sma, 2),
                "rsi_above_sma": rsi_above,
                "rsi_crossover": crossover,
                "crossover_days_ago": rsi_data.get("crossover_days_ago"),
                "gap": gap_data.get("gap", False),
                "gap_dir": gap_data.get("gap_dir"),
                "gap_days_ago": gap_data.get("gap_days_ago"),
                "gap_pct": gap_data.get("gap_pct"),
                "sortino": stk_sortino_val,
                "beta": stk_beta_val,
                "signal": signal,
            })
        except Exception:
            continue

    results.sort(key=lambda x: x["rsi_spread"], reverse=True)

    response = {
        "sector": matched,
        "etf": config.SECTOR_ETFS.get(matched),
        "stocks": results,
        "cached_at": datetime.now().isoformat(),
    }

    _save_cache(cache_key, response)
    return jsonify(response)


@api.route("/chart/<ticker>")
def chart(ticker):
    period = request.args.get("period", config.DEFAULT_PERIOD)
    interval = request.args.get("interval", config.DEFAULT_INTERVAL)
    sector_etf = request.args.get("sector_etf", None)
    import ta as ta_lib

    # Fetch ticker + SPY + optional sector ETF
    fetch_list = [ticker.upper(), "SPY"]
    if sector_etf and sector_etf.upper() != ticker.upper() and sector_etf.upper() != "SPY":
        fetch_list.append(sector_etf.upper())
    data = data_fetcher.fetch_tickers(fetch_list, period, interval)

    df = data.get(ticker.upper())
    if df is None:
        return jsonify({"error": "Ticker not found"}), 404

    spy_df = data.get("SPY")
    sect_df = data.get(sector_etf.upper()) if sector_etf else None

    w = config.SORTINO_WINDOW
    rsi = ta_lib.momentum.rsi(df["Close"], window=10)
    rsi_sma = rsi.rolling(10).mean()
    ret = indicators.daily_returns(df)

    sortino = indicators.rolling_sortino(ret, w).reindex(df.index)
    omega = indicators.rolling_omega(ret, w).reindex(df.index)

    # RSI of Sortino
    rsi_sort_data = indicators.compute_rsi_of_sortino(df, w)
    rsi_sort_s = rsi_sort_data.get("rsi_sort_series")
    rsi_sort_sma_s = rsi_sort_data.get("rsi_sort_sma_series")
    cvar = indicators.rolling_cvar(ret, w).reindex(df.index)
    ulcer = indicators.rolling_ulcer(df["Close"], w)

    # SPY metrics
    spy_norm = None
    spy_sortino_s = None
    spy_omega_s = None
    beta_s = None
    up_cap_s = None
    dn_cap_s = None
    if spy_df is not None:
        spy_ret = indicators.daily_returns(spy_df)
        beta_s = indicators.rolling_beta(ret, spy_ret, w).reindex(df.index)
        up_cap_s, dn_cap_s = indicators.rolling_updown_capture(ret, spy_ret, w)
        up_cap_s = up_cap_s.reindex(df.index)
        dn_cap_s = dn_cap_s.reindex(df.index)
        spy_sortino_s = indicators.rolling_sortino(spy_ret, w).reindex(df.index)
        spy_omega_s = indicators.rolling_omega(spy_ret, w).reindex(df.index)
        # Normalized SPY price (base = first common date)
        spy_close = spy_df["Close"].reindex(df.index)
        first_valid = spy_close.first_valid_index()
        if first_valid is not None:
            spy_norm = (spy_close / spy_close.loc[first_valid] * df["Close"].loc[first_valid])

    # Sector ETF metrics
    sect_norm = None
    sect_sortino_s = None
    sect_omega_s = None
    if sect_df is not None:
        sect_ret = indicators.daily_returns(sect_df)
        sect_sortino_s = indicators.rolling_sortino(sect_ret, w).reindex(df.index)
        sect_omega_s = indicators.rolling_omega(sect_ret, w).reindex(df.index)
        sect_close = sect_df["Close"].reindex(df.index)
        first_valid = sect_close.first_valid_index()
        if first_valid is not None:
            sect_norm = (sect_close / sect_close.loc[first_valid] * df["Close"].loc[first_valid])

    def _val(series, i):
        if series is None or i >= len(series):
            return None
        v = series.iloc[i]
        return round(float(v), 3) if v == v else None

    candles = []
    for i, (date, row) in enumerate(df.iterrows()):
        rec = {
            "date": str(date)[:10],
            "open": round(float(row.get("Open", 0)), 2),
            "high": round(float(row.get("High", 0)), 2),
            "low": round(float(row.get("Low", 0)), 2),
            "close": round(float(row.get("Close", 0)), 2),
            "volume": int(row.get("Volume", 0)),
        }
        # Gap detection per candle
        if i > 0:
            prev = df.iloc[i - 1]
            cur_open = row.get("Open", 0)
            prev_high = prev.get("High", 0)
            prev_low = prev.get("Low", 0)
            if prev_high > 0 and prev_low > 0:
                gap_up = (cur_open - prev_high) / prev_high * 100
                gap_dn = (cur_open - prev_low) / prev_low * 100
                if gap_up >= 0.5:
                    rec["gap"] = round(float(gap_up), 2)
                elif gap_dn <= -0.5:
                    rec["gap"] = round(float(gap_dn), 2)
        if i < len(rsi) and rsi.iloc[i] == rsi.iloc[i]:
            rec["rsi"] = round(float(rsi.iloc[i]), 2)
        if i < len(rsi_sma) and rsi_sma.iloc[i] == rsi_sma.iloc[i]:
            rec["rsi_sma"] = round(float(rsi_sma.iloc[i]), 2)

        for key, series in [("sortino", sortino), ("omega", omega), ("cvar", cvar),
                            ("ulcer", ulcer), ("beta", beta_s), ("spy_sortino", spy_sortino_s),
                            ("spy_omega", spy_omega_s), ("sect_sortino", sect_sortino_s),
                            ("sect_omega", sect_omega_s),
                            ("rsi_sort", rsi_sort_s), ("rsi_sort_sma", rsi_sort_sma_s)]:
            v = _val(series, i)
            if v is not None: rec[key] = v

        v = _val(up_cap_s, i)
        if v is not None: rec["up_capture"] = round(v, 1)
        v = _val(dn_cap_s, i)
        if v is not None: rec["dn_capture"] = round(v, 1)
        v = _val(spy_norm, i)
        if v is not None: rec["spy_price"] = round(v, 2)
        v = _val(sect_norm, i)
        if v is not None: rec["sect_price"] = round(v, 2)

        candles.append(rec)

    return jsonify({
        "ticker": ticker.upper(),
        "sector_etf": sector_etf.upper() if sector_etf else None,
        "candles": candles,
    })


@api.route("/studies")
def studies_list():
    """Return all study results."""
    path = Path(__file__).parent / ".data" / "studies" / "results.json"
    if not path.exists():
        return jsonify({"error": "No studies found. Run: python studies.py"}), 404
    with open(path) as f:
        return jsonify(json.load(f))


@api.route("/studies/<int:study_id>/trades")
def study_trades(study_id):
    """Get trades for a study by ID. Also supports ?signal=X&exit=Y params."""
    import studies as studies_mod

    path = Path(__file__).parent / ".data" / "studies" / "results.json"
    sig_key = request.args.get("signal")
    exit_key = request.args.get("exit")
    sector_filter = request.args.get("sector")

    # If signal/exit provided directly, use those
    if not sig_key or not exit_key:
        if not path.exists():
            return jsonify({"error": "No studies found"}), 404
        with open(path) as f:
            data = json.load(f)
        study = None
        for s in data["studies"]:
            if s["id"] == study_id:
                study = s
                break
        if not study:
            return jsonify({"error": f"Study {study_id} not found"}), 404
        sig_key = study["signal"]
        exit_key = study["exit"]
    else:
        study = {"name": f"{sig_key} -> {exit_key}"}

    if sig_key not in studies_mod.SIGNALS or exit_key not in studies_mod.EXITS:
        return jsonify({"error": "Invalid signal/exit"}), 400

    _, sig_fn = studies_mod.SIGNALS[sig_key]
    _, exit_fn = studies_mod.EXITS[exit_key]

    all_data = data_fetcher.fetch_all()
    etf_to_sector = {v: k for k, v in config.SECTOR_ETFS.items()}

    trades = []
    for ticker, df in all_data.items():
        if ticker == config.BENCHMARK or len(df) < 60:
            continue
        sector = etf_to_sector.get(ticker, ticker)
        if sector_filter and sector != sector_filter and ticker != sector_filter:
            continue
        try:
            signals = sig_fn(df).fillna(False)
        except Exception:
            continue

        for entry_date in signals[signals].index:
            idx = df.index.get_loc(entry_date)
            exit_idx = exit_fn(df, idx)
            if exit_idx is None or exit_idx <= idx:
                continue

            entry_price = float(df["Close"].iloc[idx])
            exit_price = float(df["Close"].iloc[exit_idx])
            ret = (exit_price - entry_price) / entry_price * 100

            trades.append({
                "sector": sector,
                "etf": ticker,
                "entry_date": str(entry_date)[:10],
                "exit_date": str(df.index[exit_idx])[:10],
                "entry_price": round(entry_price, 2),
                "exit_price": round(exit_price, 2),
                "return": round(ret, 3),
                "hold_days": exit_idx - idx,
            })

    trades.sort(key=lambda x: x["entry_date"], reverse=True)

    return jsonify({
        "study_id": study_id,
        "study_name": study.get("name", ""),
        "signal": sig_key,
        "exit": exit_key,
        "total_trades": len(trades),
        "trades": trades[:200],
    })


@api.route("/regime")
def current_regime():
    """Return current market regime data."""
    try:
        import rates as rates_mod
        import market_regime
        r = rates_mod.get_rates()
        m = market_regime.get_market_data()
        last_r = r.iloc[-1] if len(r) > 0 else {}
        last_m = m.iloc[-1] if len(m) > 0 else {}
        return jsonify({
            "rate_3m": round(float(last_r.get("rate_3m", 0)), 2),
            "rate_10y": round(float(last_r.get("rate_10y", 0)), 2),
            "regime": str(last_r.get("regime", "?")),
            "curve": str(last_r.get("curve", "?")),
            "rate_trend": str(last_r.get("rate_trend", "?")),
            "vix": round(float(last_m.get("vix", 0)), 1),
            "vix_regime": str(last_m.get("vix_regime", "?")),
            "spy_trend": str(last_m.get("spy_trend", "?")),
            "season": str(last_m.get("sell_in_may", "?")),
        })
    except Exception as e:
        return jsonify({"error": str(e)})


@api.route("/sectors")
def sectors():
    return jsonify([{"name": k, "etf": v} for k, v in config.SECTOR_ETFS.items()])


@api.route("/refresh")
def refresh():
    for f in glob.glob(str(RESULTS_DIR / "*.json")):
        os.remove(f)
    return jsonify({"status": "cleared"})


# Create app, register API blueprint, then add frontend catch-all
app = Flask(__name__, static_folder=None)
CORS(app)
app.register_blueprint(api)


@app.route('/', defaults={'path': ''})
@app.route('/<path:path>')
def serve_frontend(path):
    file_path = os.path.join(FRONTEND_DIR, path)
    if path and os.path.isfile(file_path):
        return send_from_directory(FRONTEND_DIR, path)
    return send_from_directory(FRONTEND_DIR, 'index.html')


if __name__ == "__main__":
    app.run(debug=False, port=5001)
