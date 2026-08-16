# H4 Short-Horizon Studies Engine — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build an H4 (4-hour) research engine — the intraday analogue of `studies.py` — that sweeps 5 signal families × a bar-based exit ladder over the liquid top-250 stock universe to find which H4 signals produce real tradeable 0–3 day edge, persisted to `BacktestResult` + a `/research/h4` tab.

**Architecture:** A new shared `intraday_data.py` owns the EODHD 1h→4h fetch/cache/universe logic (extracted from `rsi_4h_study.py`, which is refactored to import it with its numbers preserved). A new `h4_study.py` defines a signal registry × exit ladder, backtests each cached 4h frame, buckets by magnitude (tail-not-average), splits by daily trend, benchmarks vs daily candles, and saves to `BacktestResult(kind="h4_study")`. The API exposes it via a `_StudyResultView` subclass; the frontend adds an `H4StudiesPage` sub-tab under Research reusing existing `RsiLadderTable`/`RsiBucketTable`/`TailStrip` components.

**Tech Stack:** Python 3, pandas, numpy, `ta`, Django ORM (TimescaleDB), Docker Compose bind mounts, React (CRA prod build).

## Global Constraints

- **Tail-not-average:** never surface a signal by its blended mean; every signal is bucketed by magnitude so the tail is visible. (project memory, hard rule)
- **Test signals individually:** each signal is a standalone selector; the liquid universe supplies the dollar-volume floor. (hard rule)
- **Always save backtests:** every run persists to `BacktestResult` (+ JSON under `.data/studies/`); no throwaway prints. (hard rule)
- **Return priority:** rank/surface by absolute return; never trade return for lower drawdown. (hard preference)
- **Do not break existing work:** all wiring is ADDITIVE — new `kind`, new URL route, new sub-tab, new mount lines. The only edit to existing behavior is the `rsi_4h_study.py` refactor, which MUST reproduce its prior numbers (Task 2 verifies this).
- **Ports/run:** backend host 8001, frontend 3001; run scripts via `MSYS_NO_PATHCONV=1 docker exec rotation-backend-1 python -u /app/<script>.py`. `/app` = `./backend`; root scripts are bind-mounted individually.
- **Mount gotcha:** a root file must EXIST before `docker compose up`, else Docker creates both `./x.py` and `./backend/x.py` as directories and the mount fails.
- **No fees / entry-at-close** convention, matching `rsi_4h_study.py`.

---

## File Structure

- **Create `intraday_data.py`** (repo root) — shared: `fetch_1h`, `resample_ohlc`, `get_4h`, `liquid_universe`. No `django.setup()` at import; DB access lazy inside `liquid_universe`.
- **Create `h4_study.py`** (repo root) — the engine: pure signal registry + `backtest_ticker` + `agg_rows` (all importable without Django), plus `main()`/CLI that calls `django.setup()` and saves to DB. Copies the two tiny pure stat helpers from `studies.py` (`_tstat_from_returns`, `_episode_starts`) so signal/agg code stays Django-free and unit-testable.
- **Create `test_h4_study.py`** (repo root) — argv-selectable plain-assert test script (also pytest-compatible), run in-container against the live bind mount.
- **Modify `rsi_4h_study.py`** — import `fetch_1h`/`resample_ohlc`/`get_4h` from `intraday_data`; delete the now-duplicated local copies. Behavior unchanged.
- **Modify `docker-compose.yml`** — add 3 bind-mount lines under the `&backend-volumes` anchor (after line 66).
- **Modify `backend/api/views.py`** — add `H4StudyView(_StudyResultView)`.
- **Modify `backend/api/urls.py`** — add one `path(...)` route.
- **Modify `frontend/src/App.js`** — add `H4StudiesPage` component + one entry in the Research `HubTabs` array (line ~5487).

---

## Task 1: Scaffold — shared module stub, engine stub, test file, mounts (establish the live-edit loop)

Bind mounts are LIVE: once mounted, editing `./h4_study.py` on the host instantly updates `/app/h4_study.py` in the running container. So we mount all three new files ONCE here, then all later tasks iterate with `docker exec` — no further rebuilds.

**Files:**
- Create: `intraday_data.py`, `h4_study.py`, `test_h4_study.py`
- Modify: `docker-compose.yml` (after line 66)

**Interfaces:**
- Produces: `intraday_data.resample_ohlc(df, hours, from_1h=False) -> DataFrame`; `h4_study` importable without Django; `test_h4_study.py` runnable via `python -u /app/test_h4_study.py [check ...]`.

- [ ] **Step 1: Create `intraday_data.py` with the resample function and stubs**

```python
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
    """Top-`n` stocks by MEDIAN daily dollar-volume (close*volume) over the trailing `months`, from DB
    daily Candle, restricted to tickers that have a Fundamental row and are NOT sector ETFs. Lazy DB
    import so this module imports without Django. Respects the Postgres /dev/shm limit by disabling
    parallel gather for the aggregate scan."""
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
         if tk in funda and tk not in etfs and mdv),
        key=lambda x: -x[1])
    return [tk for tk, _ in ranked[:n]]
```

- [ ] **Step 2: Create `h4_study.py` as an importable stub (no Django at import)**

```python
#!/usr/bin/env python3
"""H4 (4-hour) short-horizon studies engine. See docs/superpowers/specs/2026-08-16-h4-short-horizon-engine-design.md.
Signal registry × bar-based exit ladder over the liquid top-250 universe; magnitude-bucketed
(tail-not-average); daily-trend split; daily-candle benchmark; saved to BacktestResult[h4_study].
Pure signal/agg code imports without Django; main() calls django.setup()."""
import numpy as np
import pandas as pd

RTH_HOURS = 6.5
TF_HOURS = 4
GAP = 3                                   # episode-dedup gap in bars
FIXED_BARS = [1, 2, 3, 4, 5, 6, 8, 10]    # 0-3 day focus (~½ day → ~6 days on 4h)


def day_label(bars):
    days = bars * TF_HOURS / RTH_HOURS
    return f"~{days:.1f}d" if days >= 1 else "~½ day"
```

- [ ] **Step 3: Create `test_h4_study.py` (argv-selectable, plain asserts)**

```python
#!/usr/bin/env python3
"""Unit tests for the pure H4 engine logic. Run in-container against the live bind mount:
  docker exec rotation-backend-1 python -u /app/test_h4_study.py            # all
  docker exec rotation-backend-1 python -u /app/test_h4_study.py resample   # one check
Also collected by pytest (test_* functions)."""
import sys
import numpy as np
import pandas as pd


def _idx(n, start="2023-01-02 09:00", freq="1h"):
    return pd.date_range(start, periods=n, freq=freq, tz="UTC")


def test_resample():
    from intraday_data import resample_ohlc
    df = pd.DataFrame({
        "open":   [1, 2, 3, 4, 5, 6, 7, 8],
        "high":   [2, 3, 4, 5, 6, 7, 8, 9],
        "low":    [0, 1, 2, 3, 4, 5, 6, 7],
        "close":  [1.5, 2.5, 3.5, 4.5, 5.5, 6.5, 7.5, 8.5],
        "volume": [10, 10, 10, 10, 20, 20, 20, 20],
    }, index=_idx(8))
    out = resample_ohlc(df, 4, from_1h=True)
    assert list(out.columns) == ["Open", "High", "Low", "Close", "Volume"]
    assert len(out) == 2
    assert out.iloc[0]["Open"] == 1 and out.iloc[0]["High"] == 5 and out.iloc[0]["Low"] == 0
    assert out.iloc[0]["Close"] == 4.5 and out.iloc[0]["Volume"] == 40
    print("test_resample OK")


CHECKS = {"resample": test_resample}


def main():
    names = [a for a in sys.argv[1:] if a in CHECKS] or list(CHECKS)
    for nm in names:
        CHECKS[nm]()
    print(f"\n{len(names)} check(s) passed.")


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Guard against stale stub directories from a prior failed mount**

Run:
```bash
for f in intraday_data.py h4_study.py test_h4_study.py; do
  if [ -d "backend/$f" ]; then rmdir "backend/$f"; fi
done
ls -la intraday_data.py h4_study.py test_h4_study.py
```
Expected: the three files listed as regular files; no `backend/*.py` directories remain.

- [ ] **Step 5: Add the three bind mounts to `docker-compose.yml`**

Insert after line 66 (`  - ./rsi_4h_study.py:/app/rsi_4h_study.py`), inside the `&backend-volumes` anchor:

```yaml
  - ./intraday_data.py:/app/intraday_data.py
  - ./h4_study.py:/app/h4_study.py
  - ./test_h4_study.py:/app/test_h4_study.py
```

- [ ] **Step 6: Recreate the backend so the new mounts take effect**

Run:
```bash
MSYS_NO_PATHCONV=1 docker compose up -d backend
```
Expected: backend recreated (compose detects the changed volume list). No "not a directory" mount error.

- [ ] **Step 7: Verify the test harness runs in-container against the live mount**

Run:
```bash
MSYS_NO_PATHCONV=1 docker exec rotation-backend-1 python -u /app/test_h4_study.py resample
```
Expected: `test_resample OK` then `1 check(s) passed.`

- [ ] **Step 8: Commit**

```bash
git add intraday_data.py h4_study.py test_h4_study.py docker-compose.yml
git commit -m "feat(h4): scaffold shared intraday_data + h4_study engine + live-mount test loop"
```

---

## Task 2: Refactor `rsi_4h_study.py` onto `intraday_data` without changing its numbers

**Files:**
- Modify: `rsi_4h_study.py` (remove local `fetch_1h`/`resample_ohlc`; keep `get_tf` for 8h/12h but base 4h on `intraday_data.get_4h`)
- Uses: `intraday_data.fetch_1h`, `intraday_data.resample_ohlc`, `intraday_data.get_4h`

**Interfaces:**
- Consumes: `intraday_data.{fetch_1h, resample_ohlc, get_4h}` (Task 1).
- Produces: unchanged `rsi_4h_study.run_tf` behavior and `BacktestResult[rsi_4h_backtest]` payload.

- [ ] **Step 1: Capture the current 4h numbers as a regression baseline (before editing)**

Run:
```bash
MSYS_NO_PATHCONV=1 docker exec rotation-backend-1 python -u /app/rsi_4h_study.py --tf 4h --no-fetch --limit 20
```
Expected: prints the `=== RSI(14) crossover — 4H ... ===` ladder. Copy the `4h` ladder rows (exit, n, avg, win, t) into a scratch note — Step 5 must reproduce them byte-for-byte.

Note: if `--no-fetch` yields "0 with data" (empty cache), first run without `--no-fetch --limit 20` once to populate a small cache, then capture the baseline.

- [ ] **Step 2: Point `rsi_4h_study.py` at the shared module**

In `rsi_4h_study.py`, delete the local `fetch_1h` (lines ~45–78) and `resample_ohlc` (lines ~81–88) definitions, and add near the top imports (after `import ta`):

```python
from intraday_data import fetch_1h, resample_ohlc, get_4h
```

Then in `get_tf`, replace the `tf == "4h"` fetch branch body so 4h delegates to the shared cache/fetch (8h/12h logic unchanged — they still resample up from the cached 4h):

```python
    if tf == "4h":
        return get_4h(ticker, years, allow_fetch)
    else:
        four = get_tf(ticker, "4h", years, allow_fetch)   # reuse the cached 4h
        if four is None:
            return None
        df = resample_ohlc(four, hours, from_1h=False)
```

Keep the existing `MIN_BARS`/parquet-write logic for the 8h/12h branch exactly as-is. (Note `get_4h` already applies the `MIN_BARS` guard and cache write for 4h, so the 4h path needs no extra handling.)

- [ ] **Step 3: Confirm the module still imports**

Run:
```bash
MSYS_NO_PATHCONV=1 docker exec rotation-backend-1 python -c "import rsi_4h_study; print('import ok')"
```
Expected: `import ok` (no ImportError, no NameError for the removed functions).

- [ ] **Step 4: Re-run the 4h study on the SAME cached universe**

Run:
```bash
MSYS_NO_PATHCONV=1 docker exec rotation-backend-1 python -u /app/rsi_4h_study.py --tf 4h --no-fetch --limit 20
```
Expected: the `4h` ladder rows match the Step 1 baseline exactly (same n / avg / win / t). Any difference is a refactor regression — fix before continuing.

- [ ] **Step 5: Commit**

```bash
git add rsi_4h_study.py
git commit -m "refactor(h4): rsi_4h_study uses shared intraday_data (numbers unchanged)"
```

---

## Task 3: Signal registry — the five families (pure functions)

Each signal is `fn(df) -> (entry: bool ndarray, mag: float ndarray)` aligned to `df` rows; `entry[i]` True means enter at bar `i`'s close; `mag[i]` is the bucketing magnitude (NaN where not an entry). The registry `SIGNALS[key] = {name, family, fn, buckets, exit_fn}` where `buckets` is a list of `(label, lo, hi)` ranges over `mag`, and `exit_fn` is optional (`None` for most).

**Files:**
- Modify: `h4_study.py` (add indicators, buckets, signal fns, `SIGNALS` registry)
- Modify: `test_h4_study.py` (add signal tests)

**Interfaces:**
- Consumes: `day_label`, `FIXED_BARS`, `GAP` (Task 1).
- Produces: `SIGNALS` dict; `bucket_of(mag_value, buckets) -> label|None`; `FAMILIES` dict `family -> [signal_key,...]`.

- [ ] **Step 1: Write failing tests for three representative signals + the bucketer**

Add to `test_h4_study.py` (and register in `CHECKS`):

```python
def _ohlc_from_close(close, idx=None):
    close = np.asarray(close, float)
    n = len(close)
    idx = idx if idx is not None else _idx(n, freq="4h")
    return pd.DataFrame({
        "Open": close, "High": close * 1.001, "Low": close * 0.999,
        "Close": close, "Volume": np.full(n, 1000.0),
    }, index=idx)


def test_bucket_of():
    from h4_study import bucket_of
    b = [("a", -100, -3), ("b", -3, -2), ("c", -2, 0)]
    assert bucket_of(-5, b) == "a"
    assert bucket_of(-2.5, b) == "b"
    assert bucket_of(-1, b) == "c"
    assert bucket_of(5, b) is None
    print("test_bucket_of OK")


def test_sig_volshock_dn():
    from h4_study import sig_mr_volshock_dn
    close = np.concatenate([np.full(40, 100.0), [90.0], np.full(5, 90.0)])  # -10% shock at idx 40
    entry, mag = sig_mr_volshock_dn(_ohlc_from_close(close))
    assert entry[40] and mag[40] < -2
    assert not entry[:40].any()
    print("test_sig_volshock_dn OK")


def test_sig_ndown():
    from h4_study import sig_mr_ndown
    close = np.array([100, 99, 98, 97, 96, 97, 98], float)  # 4 consecutive downs ending idx 4
    entry, mag = sig_mr_ndown(_ohlc_from_close(close))
    assert entry[4] and mag[4] >= 4
    assert not entry[5]
    print("test_sig_ndown OK")


def test_sig_gap_dn():
    from h4_study import sig_mr_gap_dn
    df = _ohlc_from_close(np.full(30, 100.0))
    df.iloc[20, df.columns.get_loc("Open")] = 96.0   # -4% gap-down open at idx 20
    df.iloc[20, df.columns.get_loc("Close")] = 96.0
    entry, mag = sig_mr_gap_dn(df)
    assert entry[20] and mag[20] < -2
    print("test_sig_gap_dn OK")
```

Add each to `CHECKS`:
```python
CHECKS = {"resample": test_resample, "bucket": test_bucket_of,
          "volshock": test_sig_volshock_dn, "ndown": test_sig_ndown, "gap": test_sig_gap_dn}
```

- [ ] **Step 2: Run the new tests to verify they fail**

Run:
```bash
MSYS_NO_PATHCONV=1 docker exec rotation-backend-1 python -u /app/test_h4_study.py bucket volshock ndown gap
```
Expected: FAIL — `ImportError: cannot import name 'bucket_of'` (functions not defined yet).

- [ ] **Step 3: Implement indicators, bucketer, and signal functions in `h4_study.py`**

Add after `day_label`:

```python
import ta


def bucket_of(v, buckets):
    """Map a magnitude to its bucket label, or None if non-finite / out of range."""
    if v is None or not np.isfinite(v):
        return None
    for label, lo, hi in buckets:
        if lo <= v < hi:
            return label
    return None


def _fresh(cond):
    """True only on the bar a boolean condition first becomes True (rising edge)."""
    c = np.asarray(cond, bool)
    prev = np.concatenate([[False], c[:-1]])
    return c & ~prev


# ── bucket schemes ───────────────────────────────────────────────────────────
RSI_BUCKETS = [("<25", 0, 25), ("25-35", 25, 35), ("35-45", 35, 45), ("45-55", 45, 55), ("55+", 55, 200)]
Z_DN_BUCKETS = [("z<-3", -100, -3), ("-3..-2.5", -3, -2.5), ("-2.5..-2", -2.5, -2)]
GAP_DN_BUCKETS = [("<-4%", -100, -4), ("-4..-3%", -4, -3), ("-3..-2%", -3, -2)]
GAP_UP_BUCKETS = [("2..3%", 2, 3), ("3..4%", 3, 4), (">4%", 4, 100)]
PCTB_BUCKETS = [("<-0.2", -100, -0.2), ("-0.2..-0.1", -0.2, -0.1), ("-0.1..0", -0.1, 0)]
DEPTH_DN_BUCKETS = [("<-6%", -100, -6), ("-6..-3%", -6, -3), ("-3..0%", -3, 0)]
UP_BUCKETS = [("2..4%", 2, 4), ("4..7%", 4, 7), (">7%", 7, 200)]
KDOWN_BUCKETS = [("3", 3, 4), ("4", 4, 5), ("5+", 5, 100)]
RSI_OB_BUCKETS = [("60-70", 60, 70), ("70-80", 70, 80), ("80+", 80, 200)]
BREAK_BUCKETS = [("0..1%", 0, 1), ("1..3%", 1, 3), (">3%", 3, 200)]
DIV_BUCKETS = [("<-6%", -100, -6), ("-6..-3%", -6, -3), ("-3..0%", -3, 0)]
MAPULL_BUCKETS = [("<-2%", -100, -2), ("-2..-0.5%", -2, -0.5), ("-0.5..0.5%", -0.5, 0.5)]


# ── mean-reversion ───────────────────────────────────────────────────────────
def sig_mr_rsi_os(df):
    """RSI(14) crosses above its SMA(14); bucket by RSI level at the cross (mirrors rsi_4h_study)."""
    close = df["Close"]
    rsi = ta.momentum.rsi(close, window=14)
    sma = rsi.rolling(14).mean()
    up = (rsi > sma) & (rsi.shift(1) <= sma.shift(1))
    entry = up.fillna(False).values
    mag = np.where(entry, rsi.values, np.nan)
    return entry, mag


def sig_mr_volshock_dn(df):
    """Vol-normalized down shock: z = ret / trailing-vol(20) <= -2."""
    close = df["Close"]
    ret = close.pct_change()
    vol = ret.rolling(20).std()
    z = (ret / vol).replace([np.inf, -np.inf], np.nan)
    entry = (z <= -2).fillna(False).values
    mag = np.where(entry, z.values, np.nan)
    return entry, mag


def sig_mr_gap_dn(df):
    """Bar opens >=2% below the prior bar's close (gap-down)."""
    gap = (df["Open"] / df["Close"].shift(1) - 1) * 100
    entry = (gap <= -2).fillna(False).values
    mag = np.where(entry, gap.values, np.nan)
    return entry, mag


def sig_mr_bb_low(df):
    """Fresh close below the lower Bollinger(20,2) band (%B < 0)."""
    close = df["Close"]
    bb = ta.volatility.BollingerBands(close, window=20, window_dev=2)
    pctb = (close - bb.bollinger_lband()) / (bb.bollinger_hband() - bb.bollinger_lband())
    entry = _fresh((pctb < 0).fillna(False).values)
    mag = np.where(entry, pctb.values, np.nan)
    return entry, mag


def _newlow(df, n):
    close = df["Close"]
    prior_min = close.shift(1).rolling(n).min()
    fresh_low = _fresh((close < prior_min).fillna(False).values)
    depth = (close / prior_min - 1) * 100
    mag = np.where(fresh_low, depth.values, np.nan)
    return fresh_low, mag


def sig_mr_newlow30(df):
    """Fresh new 30-bar low; bucket by depth below the prior 30-bar low."""
    return _newlow(df, 30)


def sig_mr_newlow60(df):
    """Fresh new 60-bar low; bucket by depth below the prior 60-bar low."""
    return _newlow(df, 60)


def sig_mr_ndown(df):
    """K>=3 consecutive down bars (enter on the bar completing the run); bucket by K."""
    close = df["Close"].values
    down = np.concatenate([[False], close[1:] < close[:-1]])
    run = np.zeros(len(close))
    for i in range(1, len(close)):
        run[i] = run[i - 1] + 1 if down[i] else 0
    entry = _fresh(run >= 3)
    mag = np.where(entry, run, np.nan)
    return entry, mag


# ── momentum / breakout ──────────────────────────────────────────────────────
def sig_mo_burst(df):
    """2-bar cumulative up-burst >= +4%; bucket by burst magnitude."""
    close = df["Close"]
    two = (close / close.shift(2) - 1) * 100
    entry = _fresh((two >= 4).fillna(False).values)
    mag = np.where(entry, two.values, np.nan)
    return entry, mag


def sig_mo_break_hi(df):
    """Fresh new 30-bar high breakout; bucket by distance above the prior high."""
    close = df["Close"]
    prior_max = close.shift(1).rolling(30).max()
    entry = _fresh((close > prior_max).fillna(False).values)
    dist = (close / prior_max - 1) * 100
    mag = np.where(entry, dist.values, np.nan)
    return entry, mag


def sig_mo_rsi_ob(df):
    """RSI(14) crosses above 60 (momentum regime flip up); bucket by RSI level."""
    rsi = ta.momentum.rsi(df["Close"], window=14)
    entry = ((rsi > 60) & (rsi.shift(1) <= 60)).fillna(False).values
    mag = np.where(entry, rsi.values, np.nan)
    return entry, mag


def sig_mo_gap_up(df):
    """Bar opens >=2% above the prior bar's close (gap-up)."""
    gap = (df["Open"] / df["Close"].shift(1) - 1) * 100
    entry = (gap >= 2).fillna(False).values
    mag = np.where(entry, gap.values, np.nan)
    return entry, mag


# ── event-driven (price-based) ───────────────────────────────────────────────
def sig_ev_open_gap(df):
    """First 4h bar of each session; bucket by the signed overnight gap vs the prior session's close."""
    dates = df.index.normalize()
    first_bar = np.concatenate([[True], dates[1:] != dates[:-1]])
    gap = (df["Open"] / df["Close"].shift(1) - 1) * 100
    entry = first_bar & np.isfinite(gap.values)
    mag = np.where(entry, gap.values, np.nan)
    return entry, mag


# ── trend / structure ────────────────────────────────────────────────────────
def sig_st_ad_div(df):
    """A/D (ta.accdist / ADL) rising over 10 bars while price falls over 10 bars — bullish divergence.
    Bucket by how far price fell (deeper drop w/ rising ADL = stronger). Mirrors the daily _ad_rising edge."""
    adl = ta.volume.acc_dist_index(df["High"], df["Low"], df["Close"], df["Volume"])
    price_chg = (df["Close"] / df["Close"].shift(10) - 1) * 100
    adl_chg = adl - adl.shift(10)
    cond = (adl_chg > 0) & (price_chg < 0)
    entry = _fresh(cond.fillna(False).values)
    mag = np.where(entry, price_chg.values, np.nan)
    return entry, mag


def sig_st_ma_pull(df):
    """Pullback to a rising MA(20): price crosses down to <= MA while MA is rising and price was above."""
    close = df["Close"]
    ma = close.rolling(20).mean()
    rising = ma > ma.shift(5)
    cross_dn = (close <= ma) & (close.shift(1) > ma.shift(1))
    entry = (cross_dn & rising).fillna(False).values
    dist = (close / ma - 1) * 100
    mag = np.where(entry, dist.values, np.nan)
    return entry, mag


# ── registry ─────────────────────────────────────────────────────────────────
SIGNALS = {
    "mr_rsi_os":     {"name": "RSI(14) cross-up (oversold buckets)", "family": "mean_reversion",
                      "fn": sig_mr_rsi_os, "buckets": RSI_BUCKETS, "exit_fn": "rsi_x_dn"},
    "mr_volshock_dn":{"name": "Vol-normalized down shock (z<=-2)", "family": "mean_reversion",
                      "fn": sig_mr_volshock_dn, "buckets": Z_DN_BUCKETS, "exit_fn": None},
    "mr_gap_dn":     {"name": "Gap-down bar (>=2%)", "family": "mean_reversion",
                      "fn": sig_mr_gap_dn, "buckets": GAP_DN_BUCKETS, "exit_fn": None},
    "mr_bb_low":     {"name": "Below lower Bollinger (%B<0)", "family": "mean_reversion",
                      "fn": sig_mr_bb_low, "buckets": PCTB_BUCKETS, "exit_fn": None},
    "mr_newlow30":   {"name": "New 30-bar low", "family": "mean_reversion",
                      "fn": sig_mr_newlow30, "buckets": DEPTH_DN_BUCKETS, "exit_fn": None},
    "mr_newlow60":   {"name": "New 60-bar low", "family": "mean_reversion",
                      "fn": sig_mr_newlow60, "buckets": DEPTH_DN_BUCKETS, "exit_fn": None},
    "mr_ndown":      {"name": "K consecutive down bars (>=3)", "family": "mean_reversion",
                      "fn": sig_mr_ndown, "buckets": KDOWN_BUCKETS, "exit_fn": None},
    "mo_burst":      {"name": "2-bar up-burst (>=4%)", "family": "momentum",
                      "fn": sig_mo_burst, "buckets": UP_BUCKETS, "exit_fn": None},
    "mo_break_hi":   {"name": "New 30-bar high breakout", "family": "momentum",
                      "fn": sig_mo_break_hi, "buckets": BREAK_BUCKETS, "exit_fn": None},
    "mo_rsi_ob":     {"name": "RSI(14) cross above 60", "family": "momentum",
                      "fn": sig_mo_rsi_ob, "buckets": RSI_OB_BUCKETS, "exit_fn": None},
    "mo_gap_up":     {"name": "Gap-up bar (>=2%)", "family": "momentum",
                      "fn": sig_mo_gap_up, "buckets": GAP_UP_BUCKETS, "exit_fn": None},
    "ev_open_gap":   {"name": "Session-open gap reaction", "family": "event",
                      "fn": sig_ev_open_gap, "buckets": GAP_DN_BUCKETS + GAP_UP_BUCKETS, "exit_fn": None},
    "st_ad_div":     {"name": "A/D bullish divergence (10-bar)", "family": "structure",
                      "fn": sig_st_ad_div, "buckets": DIV_BUCKETS, "exit_fn": None},
    "st_ma_pull":    {"name": "Pullback to rising MA(20)", "family": "structure",
                      "fn": sig_st_ma_pull, "buckets": MAPULL_BUCKETS, "exit_fn": None},
}

FAMILIES = {}
for _k, _m in SIGNALS.items():
    FAMILIES.setdefault(_m["family"], []).append(_k)
```

- [ ] **Step 4: Run the tests to verify they pass**

Run:
```bash
MSYS_NO_PATHCONV=1 docker exec rotation-backend-1 python -u /app/test_h4_study.py bucket volshock ndown gap
```
Expected: `test_bucket_of OK`, `test_sig_volshock_dn OK`, `test_sig_ndown OK`, `test_sig_gap_dn OK`, then `4 check(s) passed.`

- [ ] **Step 5: Sanity-check the registry imports and every fn runs on a real cached frame**

Run:
```bash
MSYS_NO_PATHCONV=1 docker exec rotation-backend-1 python -c "
import h4_study as H, glob, pandas as pd
p = sorted(glob.glob('/app/.data/intraday/4h/*.parquet'))
df = pd.read_parquet(p[0]); print('bars', len(df))
for k,m in H.SIGNALS.items():
    e,g = m['fn'](df); print(k, int(e.sum()), 'entries')
"
```
Expected: every signal prints a non-negative entry count without raising. (If no parquet exists yet, run one small fetch first: `python -u /app/rsi_4h_study.py --tf 4h --limit 5`.)

- [ ] **Step 6: Commit**

```bash
git add h4_study.py test_h4_study.py
git commit -m "feat(h4): five signal families + magnitude buckets (pure, unit-tested)"
```

---

## Task 4: Backtest, bucketing, daily-trend split, aggregation

**Files:**
- Modify: `h4_study.py` (add `_tstat_from_returns`, `_episode_starts`, `backtest_ticker`, `agg_rows`)
- Modify: `test_h4_study.py` (add backtest + agg tests)

**Interfaces:**
- Consumes: `SIGNALS`, `FIXED_BARS`, `GAP`, `bucket_of`, `day_label` (Task 3).
- Produces:
  - `backtest_ticker(df, dtrend=None) -> {sig: {"flat": {exit:[ret]}, "by_bucket": {blabel:{exit:[ret]}}, "by_dtrend": {"up":{exit:[ret]}, "dn":{exit:[ret]}}}}`
  - `agg_rows(pool, exit_keys) -> [{exit,name,trades,avg_pct,median_pct,win_pct,t}]` sorted by `-avg_pct`
  - `EXITS -> [(key, bars, day_label)]` and `exit_keys_for(sig) -> [key,...]`

- [ ] **Step 1: Write failing tests for the backtest + aggregation**

Add to `test_h4_study.py` and `CHECKS`:

```python
def test_backtest_and_agg():
    from h4_study import backtest_ticker, agg_rows, EXITS
    # monotonic ramp: every entry's forward return is strictly positive
    close = 100 * (1.01 ** np.arange(80))
    df = _ohlc_from_close(close)
    res = backtest_ticker(df)
    assert "mo_break_hi" in res
    flat = res["mo_break_hi"]["flat"]
    assert "1b" in flat and len(flat["1b"]) > 0
    assert all(r > 0 for r in flat["1b"])           # ramp ⇒ positive forward returns
    rows = agg_rows({k: v for k, v in flat.items()}, [e[0] for e in EXITS])
    if rows:
        assert set(["exit", "name", "trades", "avg_pct", "median_pct", "win_pct", "t"]) <= set(rows[0])
    print("test_backtest_and_agg OK")


def test_dtrend_split():
    from h4_study import backtest_ticker
    close = 100 * (1.01 ** np.arange(80))
    df = _ohlc_from_close(close)
    dtrend = {d.date(): "up" for d in df.index}      # all days flagged up-trend
    res = backtest_ticker(df, dtrend=dtrend)
    any_up = any(res[s]["by_dtrend"]["up"].get("1b") for s in res)
    assert any_up
    print("test_dtrend_split OK")
```
```python
CHECKS.update({"backtest": test_backtest_and_agg, "dtrend": test_dtrend_split})
```

- [ ] **Step 2: Run to verify they fail**

Run:
```bash
MSYS_NO_PATHCONV=1 docker exec rotation-backend-1 python -u /app/test_h4_study.py backtest dtrend
```
Expected: FAIL — `ImportError: cannot import name 'backtest_ticker'`.

- [ ] **Step 3: Implement the exit ladder, helpers, backtest, and aggregation**

Add to `h4_study.py`:

```python
# Pure stat helpers copied from studies.py so this module stays Django-free (unit-testable).
def _tstat_from_returns(returns):
    arr = np.asarray(returns, dtype=float)
    arr = arr[np.isfinite(arr)]
    if len(arr) < 3:
        return None
    sd = arr.std(ddof=1)
    if not (sd > 0):
        return None
    return round(float(arr.mean() / (sd / np.sqrt(len(arr)))), 2)


def _episode_starts(entry_idxs, gap=GAP):
    starts, last = set(), -10 ** 9
    for i in entry_idxs:
        if i - last >= gap:
            starts.add(i)
            last = i
    return starts


EXITS = [(f"{b}b", b, day_label(b)) for b in FIXED_BARS]
EXIT_LABEL = {k: f"Hold {k} ({d})" for k, b, d in EXITS}
EXIT_LABEL["rsi_x_dn"] = "Till RSI crosses back below SMA"


def exit_keys_for(sig):
    keys = [k for k, _, _ in EXITS]
    if SIGNALS[sig].get("exit_fn") == "rsi_x_dn":
        keys = keys + ["rsi_x_dn"]
    return keys


def _rsi_x_dn_exit(df):
    """Exit bars where RSI(14) crosses back below its SMA(14) — the native MR exit for RSI signals."""
    rsi = ta.momentum.rsi(df["Close"], window=14)
    sma = rsi.rolling(14).mean()
    dn = (rsi < sma) & (rsi.shift(1) >= sma.shift(1))
    return dn.fillna(False).values


def _empty_exit_pool(sig):
    return {k: [] for k in exit_keys_for(sig)}


def backtest_ticker(df, dtrend=None):
    """Backtest every signal on one 4h frame. dtrend: {date -> 'up'|'dn'} daily-trend map (optional)."""
    close = df["Close"].values
    n = len(close)
    dates = df.index.normalize()
    out = {}
    for sig, meta in SIGNALS.items():
        entry, mag = meta["fn"](df)
        buckets = meta["buckets"]
        idxs = sorted(_episode_starts([i for i in range(n) if entry[i]], gap=GAP))
        flat = _empty_exit_pool(sig)
        by_bucket = {b[0]: _empty_exit_pool(sig) for b in buckets}
        by_dtrend = {"up": _empty_exit_pool(sig), "dn": _empty_exit_pool(sig)}
        dn_exit = _rsi_x_dn_exit(df) if meta.get("exit_fn") == "rsi_x_dn" else None
        for i in idxs:
            ep = float(close[i])
            if ep <= 0:
                continue
            blab = bucket_of(mag[i], buckets)
            dstate = None
            if dtrend is not None:
                dstate = dtrend.get(dates[i].date())
            for k, bars, _ in EXITS:
                j = i + bars
                if j < n:
                    r = (close[j] - ep) / ep * 100
                    flat[k].append(r)
                    if blab is not None:
                        by_bucket[blab][k].append(r)
                    if dstate in ("up", "dn"):
                        by_dtrend[dstate][k].append(r)
            if dn_exit is not None:
                j = next((q for q in range(i + 1, n) if dn_exit[q]), None)
                if j is not None:
                    r = (close[j] - ep) / ep * 100
                    flat["rsi_x_dn"].append(r)
                    if blab is not None:
                        by_bucket[blab]["rsi_x_dn"].append(r)
                    if dstate in ("up", "dn"):
                        by_dtrend[dstate]["rsi_x_dn"].append(r)
        out[sig] = {"flat": flat, "by_bucket": by_bucket, "by_dtrend": by_dtrend}
    return out


def agg_rows(pool, exit_keys, min_trades=20):
    """Aggregate {exit_key: [returns]} into sorted ladder rows (n>=min_trades)."""
    rows = []
    for k in exit_keys:
        r = pool.get(k, [])
        if len(r) < min_trades:
            continue
        a = np.array(r, dtype=float)
        a = a[np.isfinite(a)]
        if len(a) < min_trades:
            continue
        rows.append({"exit": k, "name": EXIT_LABEL.get(k, k), "trades": int(len(a)),
                     "avg_pct": round(float(a.mean()), 3), "median_pct": round(float(np.median(a)), 3),
                     "win_pct": round(float((a > 0).mean() * 100), 1),
                     "t": _tstat_from_returns(list(a))})
    rows.sort(key=lambda x: -x["avg_pct"])
    return rows
```

- [ ] **Step 4: Run the tests to verify they pass**

Run:
```bash
MSYS_NO_PATHCONV=1 docker exec rotation-backend-1 python -u /app/test_h4_study.py backtest dtrend
```
Expected: `test_backtest_and_agg OK`, `test_dtrend_split OK`, then `2 check(s) passed.`

- [ ] **Step 5: Run the FULL suite (regression) and commit**

Run:
```bash
MSYS_NO_PATHCONV=1 docker exec rotation-backend-1 python -u /app/test_h4_study.py
```
Expected: all checks pass (`8 check(s) passed.` or current count).

```bash
git add h4_study.py test_h4_study.py
git commit -m "feat(h4): exit-ladder backtest + magnitude/daily-trend buckets + aggregation"
```

---

## Task 5: Universe fetch, cross-ticker pooling, daily benchmark, DB save, CLI

**Files:**
- Modify: `h4_study.py` (add `_daily_trend_map`, `daily_benchmark`, `run`, `main`)
- Uses: `intraday_data.{get_4h, liquid_universe}`, `seq_fundamental_study.load_candles`, `core.models.BacktestResult`

**Interfaces:**
- Consumes: `backtest_ticker`, `agg_rows`, `SIGNALS`, `FAMILIES`, `exit_keys_for` (Task 4); `liquid_universe`, `get_4h` (Task 1).
- Produces: `BacktestResult(kind="h4_study")` payload + `.data/studies/h4_study.json`; CLI `--limit/--no-fetch/--jobs/--years/--n/--families`.

- [ ] **Step 1: Implement the daily-trend map, benchmark, pooling `run`, and `main`**

Add to `h4_study.py`:

```python
def _daily_trend_map(daily_df):
    """{date -> 'up'|'dn'} from a daily OHLCV frame: 'up' when Close >= SMA(50), else 'dn'."""
    if daily_df is None or len(daily_df) < 50:
        return {}
    sma = daily_df["Close"].rolling(50).mean()
    state = np.where(daily_df["Close"] >= sma, "up", "dn")
    return {d.date(): s for d, s in zip(daily_df.index, state)}


def _merge_into(dst, src):
    for k, v in src.items():
        dst.setdefault(k, []).extend(v)


def daily_benchmark(tickers):
    """Same signals on DAILY DB candles as a scale benchmark. Returns {sig: [agg rows]}."""
    from seq_fundamental_study import load_candles
    daily = load_candles(tickers)
    pools = {s: {} for s in SIGNALS}
    for tk, df in daily.items():
        if len(df) < 120:
            continue
        res = backtest_ticker(df)
        for s in SIGNALS:
            _merge_into(pools[s], res[s]["flat"])
    return {s: agg_rows(pools[s], exit_keys_for(s)) for s in SIGNALS}


def _accumulate(agg_flat, agg_bucket, agg_dtrend, res):
    for s in SIGNALS:
        _merge_into(agg_flat[s], res[s]["flat"])
        for b, d in res[s]["by_bucket"].items():
            for k, v in d.items():
                agg_bucket[s][b].setdefault(k, []).extend(v)
        for dstate in ("up", "dn"):
            for k, v in res[s]["by_dtrend"][dstate].items():
                agg_dtrend[s][dstate].setdefault(k, []).extend(v)


def run(tickers, years, allow_fetch, jobs):
    from seq_fundamental_study import load_candles
    agg_flat = {s: {} for s in SIGNALS}
    agg_bucket = {s: {b[0]: {} for b in SIGNALS[s]["buckets"]} for s in SIGNALS}
    agg_dtrend = {s: {"up": {}, "dn": {}} for s in SIGNALS}
    daily_all = load_candles(tickers)                       # for daily-trend maps (parent-side, no MP DB)
    got, spans = 0, []

    def _one(tk):
        df = get_4h(tk, years, allow_fetch)
        if df is None or len(df) < 120:
            return None
        dtrend = _daily_trend_map(daily_all.get(tk))
        return tk, str(df.index[0].date()), str(df.index[-1].date()), backtest_ticker(df, dtrend=dtrend)

    if jobs and jobs > 1:
        from multiprocessing import Pool
        # Fetch must be single-threaded (rate limit); pre-fetch serially, then compute in parallel.
        for tk in tickers:
            get_4h(tk, years, allow_fetch)
        payloads = [(tk, get_4h(tk, years, False)) for tk in tickers]
        payloads = [(tk, df) for tk, df in payloads if df is not None and len(df) >= 120]
        with Pool(jobs) as pool:
            results = pool.starmap(_bt_worker, [(tk, df, _daily_trend_map(daily_all.get(tk)))
                                                for tk, df in payloads])
        for item in results:
            if item is None:
                continue
            tk, d0, d1, res = item
            got += 1
            spans.append((tk, d0, d1))
            _accumulate(agg_flat, agg_bucket, agg_dtrend, res)
    else:
        for i, tk in enumerate(tickers):
            item = _one(tk)
            if item is None:
                continue
            tk_, d0, d1, res = item
            got += 1
            spans.append((tk_, d0, d1))
            _accumulate(agg_flat, agg_bucket, agg_dtrend, res)
            if (i + 1) % 25 == 0:
                print(f"  ...{i + 1}/{len(tickers)} ({got} with data)", flush=True)

    bench = daily_benchmark([s[0] for s in spans] or tickers)
    signals_payload = {}
    for s in SIGNALS:
        eks = exit_keys_for(s)
        border = [b[0] for b in SIGNALS[s]["buckets"]]
        signals_payload[s] = {
            "name": SIGNALS[s]["name"], "family": SIGNALS[s]["family"],
            "all": agg_rows(agg_flat[s], eks),
            "by_bucket": {b: agg_rows(agg_bucket[s][b], eks) for b in border},
            "bucket_order": border,
            "by_dtrend": {d: agg_rows(agg_dtrend[s][d], eks) for d in ("up", "dn")},
            "daily": bench[s],
        }
    earliest = min((s[1] for s in spans), default=None)
    latest = max((s[2] for s in spans), default=None)
    return {
        "families": FAMILIES,
        "signals": signals_payload,
        "params": {"timeframe": "4h", "universe": "liquid top stocks", "n_with_data": got,
                   "exit_bars": FIXED_BARS, "episode_gap_bars": GAP,
                   "history": {"from": earliest, "to": latest}},
    }


def _bt_worker(tk, df, dtrend):
    return tk, str(df.index[0].date()), str(df.index[-1].date()), backtest_ticker(df, dtrend=dtrend)


def main():
    import os, sys, json, argparse
    os.environ.setdefault("DJANGO_SETTINGS_MODULE", "rotation.settings")
    import django
    django.setup()
    import pandas as pd
    from pathlib import Path
    from intraday_data import liquid_universe

    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=250, help="universe size (top-N by dollar volume)")
    ap.add_argument("--limit", type=int, default=None, help="cap tickers actually processed")
    ap.add_argument("--no-fetch", action="store_true")
    ap.add_argument("--jobs", type=int, default=1)
    ap.add_argument("--years", type=float, default=5)
    ap.add_argument("--families", default=None, help="comma list to restrict SIGNALS by family")
    args = ap.parse_args()

    if args.families:
        keep = set(args.families.split(","))
        for k in [k for k, m in SIGNALS.items() if m["family"] not in keep]:
            del SIGNALS[k]
        FAMILIES.clear()
        for k, m in SIGNALS.items():
            FAMILIES.setdefault(m["family"], []).append(k)

    tickers = liquid_universe(n=args.n)
    if args.limit:
        tickers = tickers[:args.limit]
    print(f"{len(tickers)} liquid tickers | {len(SIGNALS)} signals | jobs={args.jobs} | "
          f"fetch={'off' if args.no_fetch else 'on'}", flush=True)

    payload = run(tickers, args.years, allow_fetch=not args.no_fetch, jobs=args.jobs)
    payload["computed_at"] = pd.Timestamp.utcnow().isoformat()
    payload["note"] = ("EODHD 1h resampled to 4h; entry at signal-bar close; episode-deduped; NO fees. "
                       "Exits are in BARS (0-3 day focus). Every signal bucketed by magnitude (tail-not-"
                       "average) and split by the stock's DAILY trend. 'daily' column = same signal on DB "
                       "daily candles. Caveats: EODHD 1h depth varies; current-liquid universe = survivorship "
                       "bias; NewsItem not backfilled pre-2025 so the event family is price-based only.")

    STUD = Path(__file__).resolve().parent / ".data" / "studies"
    STUD.mkdir(parents=True, exist_ok=True)
    (STUD / "h4_study.json").write_text(json.dumps(payload, indent=2, default=str))
    try:
        from core.models import BacktestResult
        from django.utils import timezone
        BacktestResult.objects.update_or_create(
            kind="h4_study",
            defaults={"payload": json.loads(json.dumps(payload, default=str)),
                      "computed_at": timezone.now()})
        print("saved BacktestResult[h4_study]", flush=True)
    except Exception as e:
        print("DB save failed:", e, flush=True)

    # Console summary: best exit per signal by avg return.
    for fam, sigs in FAMILIES.items():
        print(f"\n=== {fam} ===", flush=True)
        for s in sigs:
            rows = payload["signals"][s]["all"]
            if rows:
                b = rows[0]
                print(f"  {s:16} best {b['exit']:4} avg {b['avg_pct']:+.2f}% win {b['win_pct']}% "
                      f"t={b['t']} n={b['trades']}", flush=True)
            else:
                print(f"  {s:16} (no exit reached n>=20)", flush=True)


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Smoke-run the engine on a small slice (fetch a handful, serial)**

Run:
```bash
MSYS_NO_PATHCONV=1 docker exec rotation-backend-1 python -u /app/h4_study.py --n 250 --limit 8 --jobs 1
```
Expected: prints "N liquid tickers | 14 signals | ...", per-family summaries with best exits, and `saved BacktestResult[h4_study]`. No traceback.

- [ ] **Step 3: Verify the DB row and payload shape**

Run:
```bash
MSYS_NO_PATHCONV=1 docker exec rotation-backend-1 python manage.py shell -c "
from core.models import BacktestResult
r = BacktestResult.objects.get(kind='h4_study')
p = r.payload
print('families', list(p['families']))
s = p['signals']['mr_rsi_os']
print('keys', sorted(s.keys()))
print('n_with_data', p['params']['n_with_data'])
"
```
Expected: families listed; `keys` includes `all, by_bucket, by_dtrend, bucket_order, daily, name, family`; `n_with_data` > 0.

- [ ] **Step 4: Verify `--jobs` parallel path matches serial (determinism)**

Run:
```bash
MSYS_NO_PATHCONV=1 docker exec rotation-backend-1 python -u /app/h4_study.py --limit 8 --no-fetch --jobs 4
```
Expected: same per-family best-exit numbers as the Step 2 serial run (same cached frames, order-independent pooling).

- [ ] **Step 5: Commit**

```bash
git add h4_study.py
git commit -m "feat(h4): universe fetch, cross-ticker pooling, daily benchmark, DB save + CLI"
```

---

## Task 6: API endpoint (`/api/h4-study`)

**Files:**
- Modify: `backend/api/views.py` (add `H4StudyView`)
- Modify: `backend/api/urls.py` (add route)

**Interfaces:**
- Consumes: `_StudyResultView` (existing base, `backend/api/views.py:1786`), `BacktestResult(kind="h4_study")` (Task 5).
- Produces: `GET /api/h4-study` → payload JSON; `POST /api/h4-study` → background recompute.

- [ ] **Step 1: Add the view (mirrors `VolShockStudyView`)**

In `backend/api/views.py`, after `VolShockStudyView` (ends ~line 1856), add:

```python
class H4StudyView(_StudyResultView):
    """H4 short-horizon studies engine: 5 signal families × bar-based exit ladder over the liquid
    top-250 stocks, magnitude-bucketed (tail-not-average), daily-trend split, daily benchmark.
    GET reads BacktestResult[h4_study]; POST recomputes (fetch may take a while)."""
    kind = "h4_study"
    script = "h4_study.py"
    json_path = "/app/.data/studies/h4_study.json"
```

Note: the base `post()` runs `python -u h4_study.py` with default args (`--n 250 --jobs 1`, fetch on). That is the intended recompute.

- [ ] **Step 2: Add the URL route**

In `backend/api/urls.py`, after the `vol-shock-study` line (line 63), add:

```python
    path("h4-study", views.H4StudyView.as_view(), name="h4-study"),
```

- [ ] **Step 3: Reload backend and verify GET returns the saved payload**

Run:
```bash
MSYS_NO_PATHCONV=1 docker exec rotation-backend-1 python -c "import backend" 2>/dev/null; \
curl -s http://localhost:8001/api/h4-study | python -c "import sys,json; d=json.load(sys.stdin); print('computed', d.get('computed')); print('families', list((d.get('families') or {}).keys()))"
```
Expected: `computed True` and the family list. (Django dev/gunicorn autoreloads on the mounted views.py edit; if not, `MSYS_NO_PATHCONV=1 docker compose restart backend` first.)

- [ ] **Step 4: Commit**

```bash
git add backend/api/views.py backend/api/urls.py
git commit -m "feat(h4): /api/h4-study endpoint (GET reads, POST recomputes)"
```

---

## Task 7: Frontend — `H4StudiesPage` sub-tab under Research

**Files:**
- Modify: `frontend/src/App.js` (add `H4StudiesPage` component near `RsiIntradayPage` ~line 6424; add HubTabs entry at ~line 5487)

**Interfaces:**
- Consumes: existing `apiFetch`, `LastUpdatedChip`, `TailStrip`, `RsiLadderTable`, `RsiBucketTable`, `ErrorBanner`, `HubTabs` (all defined in App.js); `GET/POST /h4-study` (Task 6). The backend ladder rows use keys `{exit,name,trades,avg_pct,win_pct,t}` — identical to what `RsiLadderTable`/`RsiBucketTable` already render, so those components are reused verbatim.
- Produces: `/research/h4` tab.

- [ ] **Step 1: Add the `H4StudiesPage` component (place immediately after `RsiIntradayPage`, ~line 6424)**

```jsx
// ---- H4 Short-Horizon Studies -----------------------------------------------
// Reads GET /h4-study. Five signal families × bar-based exit ladder over the liquid top-250, held
// 0-3 days. Every signal bucketed by magnitude (tail-not-average) + split by the stock's daily trend,
// with the same signal on daily candles as a scale benchmark. Mirrors RsiIntradayPage's rendering.
const H4_FAMILY_LABEL = {
  mean_reversion: 'Mean-reversion (snap-back)', momentum: 'Momentum / breakout',
  event: 'Event-driven (price-based)', structure: 'Trend / structure',
};

function H4SignalCard({ sig }) {
  const [open, setOpen] = useState(false);
  const border = sig.bucket_order || [];
  const strip = border.map(b => {
    const rows = sig.by_bucket?.[b] || [];
    const r = rows.find(x => x.exit === '3b') || rows[rows.length - 1];
    return { label: b, value: r ? r.avg_pct : null, n: r ? r.trades : null };
  });
  return (
    <div className="darkpool-card">
      <div className="darkpool-card-head" style={{ cursor: 'pointer' }} onClick={() => setOpen(o => !o)}>
        <h2>{open ? '▾' : '▸'} {sig.name} <span className="dim" style={{ fontSize: 12 }}>({sig.key})</span></h2>
      </div>
      <div style={{ maxWidth: 360 }}><TailStrip buckets={strip} /></div>
      {open && (
        <>
          <h3 style={{ marginTop: 12 }}>Exit ladder by magnitude bucket</h3>
          {border.filter(b => (sig.by_bucket?.[b] || []).length).map(b => (
            <RsiBucketTable key={b} bucket={b} rows={sig.by_bucket[b]} />
          ))}
          <h3 style={{ marginTop: 12 }}>All entries (no bucket)</h3>
          <RsiLadderTable rows={sig.all || []} />
          <h3 style={{ marginTop: 12 }}>Split by daily trend</h3>
          <div className="dim" style={{ fontSize: 12, marginBottom: 4 }}>Daily up-trend</div>
          <RsiLadderTable rows={sig.by_dtrend?.up || []} />
          <div className="dim" style={{ fontSize: 12, margin: '8px 0 4px' }}>Daily down-trend</div>
          <RsiLadderTable rows={sig.by_dtrend?.dn || []} />
          <h3 style={{ marginTop: 12 }}>Daily benchmark (same signal, daily bars)</h3>
          <RsiLadderTable rows={sig.daily || []} />
        </>
      )}
    </div>
  );
}

function H4StudiesPage() {
  const [data, setData] = useState(null);
  const [err, setErr] = useState(null);
  const [running, setRunning] = useState(false);

  const load = () => { setErr(null); apiFetch('/h4-study').then(setData).catch(e => setErr(e.message)); };
  useEffect(() => { load(); }, []);
  const run = () => {
    setRunning(true);
    apiFetch('/h4-study', { method: 'POST' }).then(() => {
      const t = setInterval(() => apiFetch('/h4-study').then(d => {
        if (d && d.computed) { clearInterval(t); setRunning(false); setData(d); }
      }).catch(() => {}), 10000);
    }).catch(e => { setErr(e.message); setRunning(false); });
  };

  if (err) return <div className="darkpool-page"><h1>H4 Short-Horizon Studies</h1><ErrorBanner message={err} onRetry={load} onDismiss={() => setErr(null)} /></div>;
  if (!data) return <div className="loading">Loading H4 studies...</div>;
  if (!data.computed) return (
    <div className="darkpool-page"><h1>H4 Short-Horizon Studies</h1>
      <div className="empty-state" style={{ padding: '40px 0' }}>
        <p>{data.note || data.message || 'Not computed yet.'}</p>
        <button className="refresh-btn" onClick={run} disabled={running}>{running ? 'Running (fetch + backtest)…' : 'Run study'}</button>
      </div>
    </div>
  );

  const fams = data.families || {};
  const sigs = data.signals || {};
  const hist = (data.params || {}).history || {};
  return (
    <div className="darkpool-page">
      <h1>H4 Short-Horizon Studies <LastUpdatedChip value={data.last_updated} /></h1>
      <p className="subtitle darkpool-muted">
        5 signal families on the 4-hour bar, held <b>0–3 days</b>, over the liquid top-{(data.params || {}).universe ? '250' : '250'} stocks.
        Every signal is bucketed by magnitude — the edge lives in the tail, not the average.
        {hist.from ? ` History ${hist.from} → ${hist.to}.` : ''}
        {(data.params || {}).n_with_data != null ? ` ${data.params.n_with_data} names with 4h data.` : ''}
        <button className="refresh-btn" style={{ marginLeft: 10 }} onClick={run} disabled={running}>{running ? 'Running…' : 'Recompute'}</button>
      </p>
      {Object.keys(fams).map(fam => (
        <div key={fam}>
          <h2 style={{ marginTop: 18 }}>{H4_FAMILY_LABEL[fam] || fam}</h2>
          {(fams[fam] || []).map(k => sigs[k] && <H4SignalCard key={k} sig={{ ...sigs[k], key: k }} />)}
        </div>
      ))}
      {data.note && <p className="subtitle" style={{ marginTop: 10, fontStyle: 'italic' }}>{data.note}</p>}
    </div>
  );
}
```

- [ ] **Step 2: Register the sub-tab in the Research `HubTabs` array (line ~5487)**

After the `rsiintraday` entry, add:

```jsx
        { key: 'h4', label: 'H4 Short-Horizon', hash: '/research/h4', match: ['h4'], el: <H4StudiesPage /> },
```

- [ ] **Step 3: Rebuild ONLY the frontend (do not recreate backend)**

Run:
```bash
MSYS_NO_PATHCONV=1 docker compose up -d --build --no-deps frontend
```
Expected: frontend image rebuilds and restarts; backend untouched. If the build serves stale JS, add `--force-recreate` (note: that also recreates backend — only use if needed).

- [ ] **Step 4: Verify the tab renders**

Open `http://localhost:3001/#/research/h4`. Expected: the "H4 Short-Horizon" tab shows family sections, each signal a card with a magnitude TailStrip; expanding a card shows the bucket ladders, all-entries ladder, daily-trend split, and daily benchmark. Confirm no console errors and the "Recompute" button is present.

- [ ] **Step 5: Commit**

```bash
git add frontend
git commit -m "feat(h4): H4 Short-Horizon research sub-tab (magnitude buckets + daily-trend split)"
```

---

## Task 8 (optional): Nightly recompute via Celery

Only do this if you want the H4 study to refresh automatically. Skippable for v1 — the tab's Recompute button and manual `docker exec` runs already work.

**Files:**
- Modify: `backend/api/celery_tasks.py` (add `@shared_task`)
- Modify: `backend/rotation/celery.py` (add a Beat schedule entry)

**Interfaces:**
- Consumes: the `h4_study.py` script (Task 5), run as a subprocess (matches `run_stock_studies` pattern).
- Produces: a scheduled nightly `BacktestResult[h4_study]` refresh.

- [ ] **Step 1: Add the task (mirror an existing subprocess study task in `celery_tasks.py`)**

```python
@shared_task
def run_h4_study():
    """Nightly: H4 short-horizon studies engine → BacktestResult[h4_study] (fetch + backtest, ~minutes)."""
    import subprocess
    subprocess.run(["python", "-u", "/app/h4_study.py", "--n", "250", "--jobs", "4"],
                   cwd="/app", timeout=7200)
```

- [ ] **Step 2: Schedule it in `backend/rotation/celery.py` Beat (pick an off-peak slot, e.g. 22:30)**

Add to the `beat_schedule` dict (follow the exact dict style already there):

```python
    "h4-study-nightly": {
        "task": "api.celery_tasks.run_h4_study",
        "schedule": crontab(hour=22, minute=30),
    },
```

- [ ] **Step 3: Restart the celery worker + beat (required — no autoreload)**

Run:
```bash
MSYS_NO_PATHCONV=1 docker compose restart celery-worker celery-beat
```
Expected: both restart cleanly. Confirm the task registers:
```bash
MSYS_NO_PATHCONV=1 docker exec rotation-celery-worker-1 python -c "from api import celery_tasks; print(hasattr(celery_tasks,'run_h4_study'))"
```
Expected: `True`.

- [ ] **Step 4: Commit**

```bash
git add backend/api/celery_tasks.py backend/rotation/celery.py
git commit -m "feat(h4): nightly celery recompute of the H4 studies engine"
```

---

## Self-Review

**Spec coverage:**
- Shared `intraday_data.py` (fetch/resample/cache/`liquid_universe`) → Task 1. ✓
- `rsi_4h_study.py` refactor preserving numbers → Task 2. ✓
- Exit ladder `[1,2,3,4,5,6,8,10]` bars w/ day labels → Task 4 (`EXITS`). ✓
- All 5 families (MR/MO/event/structure/cross-timeframe) → Task 3 signals + Task 4 `by_dtrend` (cross-timeframe as a reporting split, per spec). ✓
- Native RSI-cross-back exit for RSI signal → Task 4 (`exit_fn`/`_rsi_x_dn_exit`). ✓
- Magnitude bucketing (tail-not-average) → Task 3 buckets + Task 4 `by_bucket`. ✓
- Stats n/avg/median/win/t + daily benchmark → Task 4 `agg_rows` + Task 5 `daily_benchmark`. ✓
- MP `--jobs` → Task 5 (`_bt_worker`, Pool). ✓
- Persist to `BacktestResult` + JSON → Task 5. ✓
- `/research/h4` tab reusing existing components + last-updated chip → Task 7. ✓
- Caveats (EODHD depth, survivorship, NewsItem) in payload/tab → Task 5 `note`, Task 7. ✓
- v1 scope cuts (4h-only, no live scanner, price-based events, stops/targets deferred) → honored; live scanner absent, no 8h/12h in `h4_study`. ✓
- Docker mount gotcha + Postgres /dev/shm limit → Task 1 (rmdir guard, `SET LOCAL max_parallel_workers_per_gather=0`). ✓

**Placeholder scan:** No TBD/TODO; every code step has full bodies; every command has expected output. ✓

**Type consistency:** `backtest_ticker` return shape (`flat`/`by_bucket`/`by_dtrend`) consumed identically in Task 5 `_accumulate`/`run`. `agg_rows` output keys (`exit,name,trades,avg_pct,median_pct,win_pct,t`) match `RsiLadderTable`/`RsiBucketTable` inputs in Task 7. `exit_keys_for` used consistently in Tasks 4–5. `SIGNALS`/`FAMILIES` shapes consistent across Tasks 3–5 and the frontend payload. ✓

**Non-breaking check:** Only additive edits except the Task 2 refactor, which has an explicit before/after numeric regression gate (Task 2 Steps 1 & 4). Frontend rebuild uses `--no-deps` to avoid recreating the backend. ✓
