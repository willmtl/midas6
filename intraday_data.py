#!/usr/bin/env python3
"""Shared intraday (1h→4h) data layer for the H4 studies: EODHD 1h fetch, resample, parquet cache,
and liquid-universe selection. NO django.setup() at import — DB access is lazy inside liquid_universe
so the pure functions import cleanly in tests."""
import os, time
from pathlib import Path
import pandas as pd

EOD = os.environ.get("EODHD_API_KEY", "")
DATA = Path(__file__).resolve().parent / ".data" / "intraday"
MIN_BARS = 120


def resample_ohlc(df, hours, from_1h=False):
    """Resample to `hours`-bar OHLCV. `from_1h`: raw EODHD lower-cased cols; else Title-cased 4h frame."""
    cols = ["open", "high", "low", "close", "volume"] if from_1h else ["Open", "High", "Low", "Close", "Volume"]
    agg = {c: f for c, f in zip(cols, ["first", "max", "min", "last", "sum"])}
    out = df[cols].resample(f"{hours}h").agg(agg).dropna()
    out.columns = ["Open", "High", "Low", "Close", "Volume"]
    return out


def fetch_1h(sym, years):
    """Paginated EODHD 1h intraday back ~`years` (120-day windows). UTC-indexed OHLCV df, or None."""
    end = int(time.time())
    floor = end - int(years * 365.25 * 86400)
    frames, cur_to = [], end
    for _ in range(80):
        cur_from = max(floor, cur_to - 120 * 86400)
        u = (f"https://eodhd.com/api/intraday/{sym}?interval=1h&from={cur_from}&to={cur_to}"
             f"&api_token={EOD}&fmt=json")
        try:
            import requests
            r = requests.get(u, timeout=30)
            if r.status_code != 200:
                break
            j = r.json()
        except Exception:
            break
        if isinstance(j, list) and j:
            frames.append(pd.DataFrame(j))
            earliest = min(x["timestamp"] for x in j)
            if cur_from <= floor or earliest <= floor:
                break
        else:
            if cur_from <= floor:
                break
        cur_to = cur_from
        time.sleep(0.25)
    if not frames:
        return None
    df = pd.concat(frames, ignore_index=True).drop_duplicates("timestamp")
    df["dt"] = pd.to_datetime(df["timestamp"], unit="s", utc=True)
    df = (df.set_index("dt").sort_index()[["open", "high", "low", "close", "volume"]]
          .apply(pd.to_numeric, errors="coerce").dropna())
    return df


def get_4h(ticker, years=5, allow_fetch=True):
    """Cache-or-fetch the 4h frame for `ticker` at .data/intraday/4h/<ticker>.parquet (idempotent)."""
    p = DATA / "4h" / f"{ticker}.parquet"
    p.parent.mkdir(parents=True, exist_ok=True)
    if p.exists():
        try:
            return pd.read_parquet(p)
        except Exception:
            pass
    if not allow_fetch or not EOD:
        return None
    raw = fetch_1h(f"{ticker}.US", years)
    if raw is None or raw.empty:
        return None
    df = resample_ohlc(raw, 4, from_1h=True)
    if len(df) < MIN_BARS:
        return None
    try:
        df.to_parquet(p)
    except Exception:
        pass
    return df


def liquid_universe(n=250, months=12):
    """Top-`n` US stocks by MEDIAN daily dollar-volume (close*volume, USD) over the trailing `months`,
    from DB daily Candle. Restricted to tickers that have a Fundamental row, are NOT sector ETFs, and
    are US-listed (no exchange suffix like .KS/.T/.HK) — foreign listings are priced in local currency
    (contaminating the dollar-volume rank) and EODHD serves them under a non-.US suffix so the 1h fetch
    would fail. Lazy DB import so this module imports without Django. Respects the Postgres /dev/shm
    limit by disabling parallel gather for the aggregate scan."""
    import django
    os.environ.setdefault("DJANGO_SETTINGS_MODULE", "rotation.settings")
    django.setup()
    from django.db import connection
    from core.models import Sector, Fundamental
    etfs = set(Sector.objects.values_list("etf", flat=True))
    funda = set(Fundamental.objects.values_list("ticker", flat=True))
    with connection.cursor() as cur:
        cur.execute("SET LOCAL max_parallel_workers_per_gather = 0")
        cur.execute(
            """
            SELECT ticker, percentile_cont(0.5) WITHIN GROUP (ORDER BY close * volume) AS mdv
            FROM core_candle
            WHERE interval = '1d' AND date >= (CURRENT_DATE - INTERVAL '%s months')
            GROUP BY ticker
            """, [months])
        rows = cur.fetchall()
    ranked = sorted(
        ((tk, float(mdv or 0)) for tk, mdv in rows
         if tk in funda and tk not in etfs and "." not in tk and mdv),
        key=lambda x: -x[1])
    return [tk for tk, _ in ranked[:n]]
