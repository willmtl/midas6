#!/usr/bin/env python3
"""Does RSI(14) suppressed-then-cross BEAT THE MARKET? The alpha/beta gate.

For every entry, compare the stock's forward return to SPY's forward return over the
IDENTICAL window (matched dates). Report:
  raw_avg   = mean stock fwd return
  spy_avg   = mean SPY fwd return over the same windows
  excess    = raw_avg - spy_avg  (the only number that answers "beats the market")
  beat%     = share of trades where the stock beat SPY
Split by REGIME at entry (SPY above/below its 200d SMA = bull/bear). A capitulation-bounce
signal that is really just long beta shows ~0 excess and lives only in the bull bucket; a
real edge holds positive excess in the BEAR bucket too.

Same universe + signal as rsi14_suppressed_largecap.py. -> BacktestResult[rsi14_beat_market].
Run: MSYS_NO_PATHCONV=1 docker exec ... python -u /app/rsi14_beat_market.py
"""
import os, json, warnings
warnings.filterwarnings("ignore")
import numpy as np, pandas as pd, ta
os.environ.setdefault("DJANGO_SETTINGS_MODULE", "rotation.settings")
import django
django.setup()
from pathlib import Path
from seq_fundamental_study import build_universe, load_candles, load_financial_reports

OUT = Path(__file__).resolve().parent / ".data" / "studies" / "rsi14_beat_market.json"
MIN_STREAK, LEVEL = 10, 50
HOLDS = {"6m": 126, "12m": 252}
PRICE_FLOOR, DVOL_FLOOR, TOP_N = 5.0, 5_000_000.0, 500


def suppressed_then_cross(close, win):
    """Return (fire_bool, streak_len) where streak_len[t] = consecutive suppressed days
    ending at t-1 (the length of the coiled spring, valid on fire bars)."""
    rsi = ta.momentum.rsi(close, window=win)
    sma = rsi.rolling(win).mean()
    cross = (rsi > sma) & (rsi.shift(1) <= sma.shift(1))
    suppressed = (rsi < LEVEL) & (rsi < sma)
    grp = (~suppressed).cumsum()
    streak = suppressed.groupby(grp).cumsum()
    prior = streak.shift(1)                                  # streak ending the bar before the cross
    fire = (cross & (prior >= MIN_STREAK)).fillna(False)
    return fire, prior.fillna(0)


def episode_dedup(idxs, gap):
    out, last = [], -10**9
    for i in sorted(idxs):
        if i - last >= gap:
            out.append(i); last = i
    return out


STREAK_BUCKETS = [(10, 14), (15, 19), (20, 29), (30, 10**9)]


def gate(candles, tickers, win, hold, spy_close, spy_bull):
    """Return excess-over-SPY stats overall + by regime + by suppression-streak length."""
    exc, beat, raw, spy_r, regime, streaks = [], [], [], [], [], []
    spy = spy_close
    for t in tickers:
        df = candles.get(t)
        if df is None or len(df) < 60 + win:
            continue
        close = df["Close"].values
        idx = df.index
        dvol = (df["Close"] * df["Volume"]).rolling(20).median().values
        n = len(close)
        fire, streak_len = suppressed_then_cross(df["Close"], win)
        fire = fire.values; slen = streak_len.values
        entries = episode_dedup([i for i in range(n) if fire[i]], hold)
        for i in entries:
            j = i + hold
            if j >= n:
                continue
            ep = close[i]
            if ep < PRICE_FLOOR or not np.isfinite(dvol[i]) or dvol[i] < DVOL_FLOOR:
                continue
            d0, d1 = idx[i], idx[j]
            s0 = spy.asof(d0); s1 = spy.asof(d1)
            if not (np.isfinite(s0) and np.isfinite(s1) and s0 > 0):
                continue
            r = (close[j] - ep) / ep * 100
            sr = (s1 / s0 - 1) * 100
            if not (np.isfinite(r) and np.isfinite(sr)):
                continue
            raw.append(r); spy_r.append(sr); exc.append(r - sr); beat.append(1 if r > sr else 0)
            regime.append("bull" if bool(spy_bull.asof(d0)) else "bear")
            streaks.append(float(slen[i]))
    raw, spy_r, exc, beat = map(np.array, (raw, spy_r, exc, beat))
    regime = np.array(regime); streaks = np.array(streaks)

    def summ(mask):
        e = exc[mask]
        if len(e) == 0:
            return {"n": 0}
        t_stat = float(e.mean() / (e.std(ddof=1) / np.sqrt(len(e)))) if len(e) > 1 and e.std() > 0 else None
        return {"n": int(len(e)), "raw_avg": round(float(raw[mask].mean()), 2),
                "spy_avg": round(float(spy_r[mask].mean()), 2),
                "excess_pp": round(float(e.mean()), 2),
                "beat_pct": round(float(beat[mask].mean() * 100), 1),
                "excess_t": round(t_stat, 2) if t_stat is not None else None}

    full = np.ones(len(exc), dtype=bool)
    out = {"all": summ(full), "bull": summ(regime == "bull"), "bear": summ(regime == "bear")}
    # Streak-length correlation + buckets (does a longer coiled spring → bigger bounce?)
    by_streak = {}
    for lo, hi in STREAK_BUCKETS:
        m = (streaks >= lo) & (streaks <= hi)
        label = f"{lo}-{hi if hi < 10**8 else '+'}"
        s = summ(m)
        if s.get("n", 0):
            s["avg_streak"] = round(float(streaks[m].mean()), 1)
        by_streak[label] = s
    if len(streaks) > 2 and streaks.std() > 0:
        out["corr_streak_raw"] = round(float(np.corrcoef(streaks, raw)[0, 1]), 4)
        out["corr_streak_excess"] = round(float(np.corrcoef(streaks, exc)[0, 1]), 4)
    out["by_streak"] = by_streak
    return out


def main():
    import time
    from django.db import connection

    def robust_load(chunk, tries=6):
        for a in range(tries):
            try:
                return load_candles(chunk)
            except Exception:
                connection.close(); time.sleep(5 * (a + 1))
        out = {}
        for t in chunk:
            try:
                out.update(load_candles([t]))
            except Exception:
                connection.close(); time.sleep(2)
        return out

    import pickle
    CACHE = Path("/tmp/beat_candles.pkl")

    def light_universe():
        """Universe from the small Fundamental table — avoids build_universe's parallel
        DISTINCT over the Candle hypertable, which OOM-crashes this DB (/dev/shm limit)."""
        from core.models import Fundamental, Sector
        for a in range(6):
            try:
                etfs = set(Sector.objects.values_list("etf", flat=True)) | {"SPY", "QQQ"}
                fund = set(Fundamental.objects.values_list("ticker", flat=True))
                return sorted(fund - etfs)
            except Exception:
                connection.close(); time.sleep(5 * (a + 1))
        raise RuntimeError("could not load universe")

    if CACHE.exists():
        print("loading candles from cache...", flush=True)
        candles = pickle.loads(CACHE.read_bytes())
        tks = sorted(candles)
        print(f"  {len(candles)} tickers from cache", flush=True)
    else:
        tks = light_universe()
        print(f"universe {len(tks)} (from Fundamental); loading candles + SPY...", flush=True)
        candles = {}
        CH = 25
        for k in range(0, len(tks), CH):
            candles.update(robust_load(tks[k:k + CH]))
            connection.close()
            if (k // CH) % 10 == 0:
                print(f"  {min(k + CH, len(tks))}/{len(tks)}", flush=True)
        CACHE.write_bytes(pickle.dumps(candles))
        print(f"  cached {len(candles)} tickers -> {CACHE}", flush=True)
    tks = sorted(candles)
    spy_df = robust_load(["SPY"]).get("SPY")
    spy_close = spy_df["Close"]
    spy_bull = (spy_close > spy_close.rolling(200).mean())
    print(f"loaded {len(candles)} tickers; SPY bars {len(spy_close)}", flush=True)

    reps = None
    for a in range(6):
        try:
            reps = load_financial_reports(tks); break
        except Exception:
            connection.close(); time.sleep(5 * (a + 1))
    if reps is None:
        reps = {}
    mcap = {}
    for t in tks:
        df = candles.get(t); rep = reps.get(t)
        if df is None or len(df) == 0 or rep is None or not len(rep):
            continue
        r2 = rep.dropna(subset=["shares_outstanding"]).sort_values("avail_date")
        if len(r2):
            sh, px = float(r2["shares_outstanding"].iloc[-1]), float(df["Close"].iloc[-1])
            if sh > 0 and px > 0:
                mcap[t] = sh * px
    largecap = sorted(mcap, key=lambda t: mcap[t], reverse=True)[:TOP_N]
    print(f"large-cap = top {len(largecap)}\n", flush=True)

    results = {}
    for win in (14, 10):
        for hk, hold in HOLDS.items():
            results[f"largecap|rsi{win}|{hk}"] = gate(candles, largecap, win, hold, spy_close, spy_bull)

    print("=== BEATS THE MARKET? excess = stock fwd − SPY fwd, matched windows ===", flush=True)
    print(f"{'key':22} {'bucket':6} {'n':>6} {'raw%':>7} {'spy%':>7} {'excess':>8} {'beat%':>6} {'exc_t':>6}", flush=True)
    for key in results:
        for b in ("all", "bull", "bear"):
            s = results[key][b]
            if s.get("n", 0) == 0:
                print(f"{key:22} {b:6} {'0':>6}", flush=True); continue
            print(f"{key:22} {b:6} {s['n']:>6} {s['raw_avg']:>+7.2f} {s['spy_avg']:>+7.2f} "
                  f"{s['excess_pp']:>+8.2f} {s['beat_pct']:>6.1f} {str(s['excess_t']):>6}", flush=True)

    print("\n=== STREAK LENGTH vs FORWARD RETURN (does a longer coil → bigger bounce?) ===", flush=True)
    for key in results:
        r = results[key]
        print(f"\n{key}   corr(streak,raw)={r.get('corr_streak_raw')}  "
              f"corr(streak,excess)={r.get('corr_streak_excess')}", flush=True)
        print(f"  {'streak':8} {'n':>6} {'avg_streak':>10} {'raw%':>7} {'excess':>8} {'beat%':>6}", flush=True)
        for lab, s in r["by_streak"].items():
            if s.get("n", 0) == 0:
                print(f"  {lab:8} {'0':>6}", flush=True); continue
            print(f"  {lab:8} {s['n']:>6} {s.get('avg_streak'):>10} {s['raw_avg']:>+7.2f} "
                  f"{s['excess_pp']:>+8.2f} {s['beat_pct']:>6.1f}", flush=True)

    payload = {"computed_at": pd.Timestamp.utcnow().isoformat(),
               "params": {"min_streak": MIN_STREAK, "top_n": TOP_N},
               "results": results,
               "caveat": "Matched-window excess over SPY, in-sample, no fees. Entries cluster in "
                         "market-wide selloffs so excess_t is still overstated by cross-sectional "
                         "overlap. Not a portfolio book (ignores capital/concurrency). Bear-bucket "
                         "excess is the alpha tell."}
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(payload, indent=2, default=str))
    try:
        from core.models import BacktestResult
        from django.utils import timezone
        BacktestResult.objects.update_or_create(
            kind="rsi14_beat_market",
            defaults={"payload": json.loads(json.dumps(payload, default=str)), "computed_at": timezone.now()})
        print("\nSaved BacktestResult[rsi14_beat_market]", flush=True)
    except Exception as e:
        print("DB save failed:", e, flush=True)


if __name__ == "__main__":
    main()
