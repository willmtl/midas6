#!/usr/bin/env python3
"""RSI(14) crossover on the 4-HOUR timeframe — fetch, cache, backtest -> BacktestResult + JSON.

There are no intraday candles in the DB (all 1d), so we fetch 1-HOUR bars from EODHD for the 93
sector ETFs, resample to 4h, and cache each to .data/intraday/4h/<ticker>.parquet (idempotent —
re-runs skip the fetch). Then we backtest the RSI(14)-crosses-above-its-SMA(14) entry (the 4h
analogue of the daily rsi_x_above_sma) across a bar-based exit ladder, episode-deduped, and run the
SAME signal on the DAILY candles as a side-by-side benchmark.

Bars: 4h RTH ≈ 2 bars/trading-day, so a "10-bar" hold ≈ 1 trading week. Exit labels are in BARS
(with an approx wall-clock). NO fees; entry at the signal bar's close.

Run:  docker exec rotation-backend-1 python -u /app/rsi_4h_study.py            # fetch + backtest
Opts: --limit N (first N ETFs)  --no-fetch (use only cached)  --years Y (default 5)
"""
import warnings
warnings.filterwarnings("ignore")
import os, sys, json, time, argparse
from pathlib import Path
import numpy as np
import pandas as pd
os.environ.setdefault("DJANGO_SETTINGS_MODULE", "rotation.settings")
import django
django.setup()

import ta
from studies import _episode_starts, _tstat_from_returns

RSI_P = 14
SMA_P = 14
GAP = 3                       # episode-dedup gap (bars) between independent entries
MIN_BARS = 120
DATA = Path(__file__).resolve().parent / ".data" / "intraday"
DATA.mkdir(parents=True, exist_ok=True)
STUD = Path(__file__).resolve().parent / ".data" / "studies"
EOD = os.environ.get("EODHD_API_KEY", "")

TF_HOURS = {"4h": 4, "8h": 8, "12h": 12}   # 8h/12h derive from the cached 4h (aligned bin boundaries)
RTH_HOURS = 6.5

# Exit ladder in BARS. Day approximations are filled per-timeframe (bars_per_day ≈ RTH/tf_hours).
FIXED_BARS = [1, 2, 3, 5, 10, 20, 40]


def fetch_1h(sym, years):
    """Paginated EODHD 1h intraday back ~`years` (120-day windows). Returns a UTC-indexed OHLCV df."""
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


def resample_ohlc(df, hours, from_1h=False):
    """Resample to `hours`-bar OHLCV. Works from 1h bars (raw EODHD, lower-cased cols) or from an
    already-4h frame (Title-cased cols) — 8h/12h bin boundaries align with 4h, so it's exact."""
    cols = ["open", "high", "low", "close", "volume"] if from_1h else ["Open", "High", "Low", "Close", "Volume"]
    agg = {c: f for c, f in zip(cols, ["first", "max", "min", "last", "sum"])}
    out = df[cols].resample(f"{hours}h").agg(agg).dropna()
    out.columns = ["Open", "High", "Low", "Close", "Volume"]
    return out


def get_tf(ticker, tf, years, allow_fetch):
    """4h is fetched (1h→4h) and cached; 8h/12h are resampled UP from the cached 4h (no re-fetch)."""
    hours = TF_HOURS[tf]
    p = DATA / tf / f"{ticker}.parquet"
    p.parent.mkdir(parents=True, exist_ok=True)
    if p.exists():
        try:
            return pd.read_parquet(p)
        except Exception:
            pass
    if tf == "4h":
        if not allow_fetch or not EOD:
            return None
        raw = fetch_1h(f"{ticker}.US", years)
        if raw is None or raw.empty:
            return None
        df = resample_ohlc(raw, 4, from_1h=True)
    else:
        four = get_tf(ticker, "4h", years, allow_fetch)   # reuse the cached 4h
        if four is None:
            return None
        df = resample_ohlc(four, hours, from_1h=False)
    if len(df) < MIN_BARS:
        return None
    try:
        df.to_parquet(p)
    except Exception:
        pass
    return df


def day_label(bars, tf):
    days = bars * TF_HOURS[tf] / RTH_HOURS
    return f"~{days:.1f}d" if days >= 1 else "~½ day"


def fixed_for(tf):
    return [(f"{b}b", b, day_label(b, tf)) for b in FIXED_BARS]


def _entries_exits(close):
    rsi = ta.momentum.rsi(close, window=RSI_P)
    sma = rsi.rolling(SMA_P).mean()
    up = (rsi > sma) & (rsi.shift(1) <= sma.shift(1))       # RSI crosses above its SMA
    dn = (rsi < sma) & (rsi.shift(1) >= sma.shift(1))       # exit trigger: crosses back below
    return up.fillna(False).values, dn.fillna(False).values


def backtest_df(df, fixed):
    """Return {exit_key: [returns]} for one instrument (episode-deduped entries)."""
    close = df["Close"].reset_index(drop=True)
    n = len(close)
    up, dn = _entries_exits(close)
    idxs = [i for i in range(n) if up[i]]
    idxs = sorted(_episode_starts(idxs, gap=GAP))
    cvals = close.values
    res = {k: [] for k, _, _ in fixed}
    res["rsi_x_dn"] = []      # hold until RSI crosses back below its SMA
    for i in idxs:
        ep = float(cvals[i])
        if ep <= 0:
            continue
        for k, bars, _ in fixed:
            j = i + bars
            if j < n:
                res[k].append((cvals[j] - ep) / ep * 100)
        j = next((q for q in range(i + 1, n) if dn[q]), None)   # RSI cross-down exit
        if j is not None:
            res["rsi_x_dn"].append((cvals[j] - ep) / ep * 100)
    return res


def agg_rows(pool, fixed):
    rows = []
    order = [k for k, _, _ in fixed] + ["rsi_x_dn"]
    label = {k: f"Hold {k} ({d})" for k, _, d in fixed}
    label["rsi_x_dn"] = "Till RSI crosses back below SMA"
    for k in order:
        r = pool.get(k, [])
        if len(r) < 20:
            continue
        a = np.array(r)
        rows.append({"exit": k, "name": label[k], "trades": len(r),
                     "avg_pct": round(float(a.mean()), 3),
                     "win_pct": round(float((a > 0).mean() * 100), 1),
                     "t": _tstat_from_returns(list(r))})
    rows.sort(key=lambda x: -x["avg_pct"])
    return rows


def daily_backtest(tickers, fixed):
    """Same RSI(14) crossover on DAILY DB candles, same exit bars, as a benchmark."""
    from seq_fundamental_study import load_candles
    pool = {}
    daily = load_candles(tickers)
    n_used = 0
    for tk, df in daily.items():
        if len(df) < MIN_BARS:
            continue
        n_used += 1
        r = backtest_df(df, fixed)
        for k, v in r.items():
            pool.setdefault(k, []).extend(v)
    return agg_rows(pool, fixed), n_used


def run_tf(tf, etfs, years, allow_fetch):
    fixed = fixed_for(tf)
    pool = {}
    got, spans = 0, []
    for i, tk in enumerate(etfs):
        df = get_tf(tk, tf, years, allow_fetch)
        if df is None or len(df) < MIN_BARS:
            continue
        got += 1
        spans.append((tk, str(df.index[0].date()), str(df.index[-1].date())))
        r = backtest_df(df, fixed)
        for k, v in r.items():
            pool.setdefault(k, []).extend(v)
        if (i + 1) % 20 == 0:
            print(f"  [{tf}] ...{i + 1}/{len(etfs)} ({got} with data)", flush=True)
    rows = agg_rows(pool, fixed)
    rows_d, n_daily = daily_backtest([s[0] for s in spans] or etfs, fixed)
    earliest = min((s[1] for s in spans), default=None)
    latest = max((s[2] for s in spans), default=None)
    payload = {
        "computed_at": pd.Timestamp.utcnow().isoformat(),
        "params": {"rsi_period": RSI_P, "sma_period": SMA_P, "timeframe": tf,
                   "universe": "93 sector ETFs", "n_with_data": got, "n_daily": n_daily,
                   "episode_gap_bars": GAP, "history": {"from": earliest, "to": latest}},
        "backtest_tf": rows, "backtest_daily": rows_d,
        "note": (f"EODHD 1h resampled to {tf} (8h/12h derived from cached 4h). Entry at the crossover "
                 "bar's close; episode-deduped; NO fees. Exit holds are in BARS. Daily column runs the "
                 "identical RSI(14) crossover on DB daily candles."),
    }
    (STUD / f"rsi_{tf}_backtest.json").write_text(json.dumps(payload, indent=2, default=str))
    try:
        from core.models import BacktestResult
        from django.utils import timezone
        BacktestResult.objects.update_or_create(
            kind=f"rsi_{tf}_backtest",
            defaults={"payload": json.loads(json.dumps(payload, default=str)),
                      "computed_at": timezone.now()})
    except Exception as e:
        print(f"DB save failed [{tf}]:", e, flush=True)
    print(f"\n=== RSI(14) crossover — {tf.upper()} ({got} ETFs, {earliest}→{latest}) ===", flush=True)
    for r in rows:
        print(f"  {r['exit']:8} {r['name']:22} n={r['trades']:>6} avg {r['avg_pct']:>+6.2f}% "
              f"win {r['win_pct']:>5}% t={r['t']}", flush=True)
    return payload


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tf", default="4h,8h,12h", help="comma list of 4h/8h/12h")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--no-fetch", action="store_true")
    ap.add_argument("--years", type=float, default=5)
    args = ap.parse_args()

    from core.models import Sector
    etfs = sorted(set(Sector.objects.values_list("etf", flat=True)))
    if args.limit:
        etfs = etfs[:args.limit]
    tfs = [t.strip() for t in args.tf.split(",") if t.strip() in TF_HOURS]
    print(f"{len(etfs)} sector ETFs | RSI({RSI_P})×SMA({SMA_P}) crossover | timeframes {tfs} | "
          f"fetch={'off' if args.no_fetch else 'on'}", flush=True)
    for tf in tfs:
        run_tf(tf, etfs, args.years, allow_fetch=not args.no_fetch)


if __name__ == "__main__":
    main()
