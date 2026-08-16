#!/usr/bin/env python3
"""FX layer — convert foreign-listed (local-currency) candle prices/volumes to USD so returns and the $-volume
liquidity floor are honest for a USD investor. Two facts (verified): yfinance `USD{CUR}=X` returns LOCAL UNITS
PER USD (uniform: usd = local / rate), and `fast_info.currency` gives each ticker's exact quote currency incl.
`GBp` (London pence) / `ZAc` / `ILA` (minor units needing an extra x100). P/B, D/E, trap are ratios/signs =>
FX-invariant, so ONLY price-for-returns and dollar-volume need conversion.

Caches (both under /app/.data): fx_currency_cache.json {ticker: ccy}, fx_rates_cache.json {base: {date: rate}}.
"""
import os, json, time, warnings
warnings.filterwarnings("ignore")
import numpy as np, pandas as pd

CCY_CACHE = "/app/.data/fx_currency_cache.json"
RATE_CACHE = "/app/.data/fx_rates_cache.json"
# minor-unit quote currencies -> (base ISO, extra divisor to reach the major unit)
MINOR = {"GBp": ("GBP", 100.0), "GBX": ("GBP", 100.0), "ZAc": ("ZAR", 100.0),
         "ILA": ("ILS", 100.0), "ILs": ("ILS", 100.0)}


def _load(path):
    try:
        return json.load(open(path))
    except Exception:
        return {}


def _yf_close(sym, period="6y"):
    import yfinance as yf
    d = yf.download(sym, period=period, progress=False, auto_adjust=True)
    if d is None or d.empty:
        return None
    s = d["Close"]
    if hasattr(s, "columns"):
        s = s.iloc[:, 0]
    s.index = pd.to_datetime(s.index).tz_localize(None)
    return s.dropna()


def get_currencies(tickers, refresh=False):
    """{ticker: quote-currency}. US-style tickers assumed USD (no lookup); dotted/numeric looked up once & cached."""
    import re
    cache = _load(CCY_CACHE)
    us_re = re.compile(r"^[A-Z]{1,5}(-[A-Z])?$")
    out, todo = {}, []
    for t in tickers:
        if us_re.match(t):
            out[t] = "USD"
        elif t in cache and not refresh:
            out[t] = cache[t]
        else:
            todo.append(t)
    if todo:
        import yfinance as yf
        print(f"fx: resolving currency for {len(todo)} foreign tickers...", flush=True)
        for i, t in enumerate(todo):
            ccy = "USD"
            try:
                ccy = getattr(yf.Ticker(t).fast_info, "currency", None) or "USD"
            except Exception:
                ccy = cache.get(t, "USD")
            cache[t] = ccy; out[t] = ccy
            if (i + 1) % 50 == 0:
                json.dump(cache, open(CCY_CACHE, "w")); print(f"  {i+1}/{len(todo)}", flush=True); time.sleep(0.1)
        json.dump(cache, open(CCY_CACHE, "w"))
    return out


def load_usd_rates(currencies, refresh=False):
    """{ccy: daily Series of LOCAL-UNITS-PER-USD} for each distinct quote currency (USD -> constant 1.0).
    Minor units (GBp/ZAc/ILA) get the base rate x100. Cached raw per base currency."""
    cache = _load(RATE_CACHE)
    bases = {}
    for c in set(currencies):
        if c == "USD":
            continue
        base, mult = MINOR.get(c, (c, 1.0))
        bases.setdefault(base, [])
        bases[base].append((c, mult))
    for base in bases:
        if base in cache and not refresh:
            continue
        s = _yf_close(f"USD{base}=X")
        if s is None:
            print(f"fx: WARN no rate for USD{base}=X (leaving names in local ccy)", flush=True); continue
        cache[base] = {str(d.date()): float(v) for d, v in s.items()}
    json.dump(cache, open(RATE_CACHE, "w"))

    rates = {"USD": None}
    for c in set(currencies):
        if c == "USD":
            continue
        base, mult = MINOR.get(c, (c, 1.0))
        raw = cache.get(base)
        if not raw:
            rates[c] = None; continue
        ser = pd.Series({pd.Timestamp(k): v for k, v in raw.items()}).sort_index() * mult
        rates[c] = ser
    return rates


def to_usd_factor(ticker_ccy, rate_series, daily_index):
    """Series of USD-per-local multipliers aligned to daily_index (usd = local * factor = local / rate).
    USD or missing-rate -> 1.0 (unchanged)."""
    if ticker_ccy == "USD" or rate_series is None:
        return pd.Series(1.0, index=daily_index)
    r = rate_series.reindex(rate_series.index.union(daily_index)).ffill().reindex(daily_index)
    f = 1.0 / r
    return f.fillna(method="ffill").fillna(method="bfill").fillna(1.0)


def convert_candles_to_usd(daily, currencies=None):
    """In-place-safe: return a NEW {ticker: df} with Open/High/Low/Close in USD (Volume kept = share count, so
    Close*Volume dollar-volume is then USD). Ratios (P/B etc.) computed downstream are FX-invariant either way."""
    tickers = list(daily.keys())
    if currencies is None:
        currencies = get_currencies(tickers)
    rates = load_usd_rates(set(currencies.values()))
    out = {}
    n_conv = 0
    for t, df in daily.items():
        ccy = currencies.get(t, "USD")
        if ccy == "USD" or rates.get(ccy) is None:
            out[t] = df; continue
        f = to_usd_factor(ccy, rates[ccy], df.index)
        g = df.copy()
        for col in ("Open", "High", "Low", "Close"):
            if col in g.columns:
                g[col] = g[col] * f
        out[t] = g; n_conv += 1
    print(f"fx: converted {n_conv}/{len(tickers)} tickers to USD", flush=True)
    return out
