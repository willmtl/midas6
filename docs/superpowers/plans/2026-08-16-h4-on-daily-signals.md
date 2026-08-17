# H4-on-Daily-Signals Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Backtest whether letting a daily signal (A/B/C) pick the name and an H4 signal time a 0–3 day entry beats either layer alone, via a new `h4_on_signals_study.py` that gates the existing H4 engine by point-in-time candidate windows.

**Architecture:** A new root script reuses `h4_study` (signals/exits/buckets/agg) and `intraday_data.get_4h`, builds candidate windows from A (pure-dip reconstruction), B (`seq_rsi20_ad_rising_rsi` fires → 10 trading days), and C (saved `rotation_history` monthly picks), then runs a masked H4 backtest (entries only inside a candidate window) and benchmarks each *(selector × H4-signal × exit)* against standalone H4, the daily system's own return, and the base rate. Additive only — no existing file is modified except additive lines in docker-compose/views/urls/App.js.

**Tech Stack:** Python 3, pandas, numpy, `ta`, Django ORM (TimescaleDB), Docker Compose bind mounts, React (CRA prod build).

## Global Constraints

- **Do not break existing code:** `h4_study.py`, `intraday_data.py`, `rsi_4h_study.py`, `backtest_lowpb.py`, `entry_signal_study.py`, `seq_fundamental_study.py`, `rotation_history_scan.py`, `studies.py` are imported READ-ONLY. All wiring is additive (new `BacktestResult` kind `h4_on_signals`, new URL, new sub-tab, new mount line).
- **Tail-not-average:** every signal bucketed by magnitude. (hard rule)
- **Always save backtests:** persist to `BacktestResult` + `.data/studies/`. (hard rule)
- **PIT / no look-ahead:** candidate `start_date` uses only data available then; H4 entries only on/after `start_date`.
- **No silent truncation:** payload reports #windows, #names, #with-4h-data, #dropped per selector.
- **Handoff logic:** daily = which name; H4 = when + the 0–3 day trade. H4 trigger = the full engine swept on candidates.
- **B window:** fixed **10 trading days** after each capitulation fire.
- **Staged fetch:** fast pass over already-cached candidates first; full backfill fetched detached.
- **Run:** `MSYS_NO_PATHCONV=1 docker exec rotation-backend-1 python -u /app/<script>.py`. `/app`=`./backend`; root scripts bind-mounted individually; a root file must EXIST before `docker compose up` (else Docker makes dirs).
- **Exit ladder / hold:** reuse `h4_study.EXITS` = `[1,2,3,4,5,6,8,10]` bars (0–3 day focus). No fees, entry at bar close.

---

## File Structure

- **Create `h4_on_signals_study.py`** (root) — candidate windows (A/B/C), `backtest_ticker_masked`, run/benchmarks/DB-save/CLI.
- **Create `test_h4_on_signals.py`** (root) — mask-correctness + candidate-window unit tests (argv-selectable plain-assert, like `test_h4_study.py`).
- **Modify `docker-compose.yml`** — 2 bind-mount lines under `&backend-volumes` (after the `h4_study.py` line).
- **Modify `backend/api/views.py`** — add `H4OnSignalsView(_StudyResultView)`.
- **Modify `backend/api/urls.py`** — add one route.
- **Modify `frontend/src/App.js`** — add `H4OnSignalsPage` + one Research `HubTabs` entry.

---

## Task 1: Scaffold + `backtest_ticker_masked` (mask-correctness against `h4_study`)

The core primitive: an H4 backtest identical to `h4_study.backtest_ticker` but that only counts entries whose bar-date is in an allowed set. Correctness gate: with all dates allowed, it must equal `h4_study.backtest_ticker` exactly.

**Files:**
- Create: `h4_on_signals_study.py`, `test_h4_on_signals.py`
- Modify: `docker-compose.yml`

**Interfaces:**
- Consumes: `h4_study.{SIGNALS, EXITS, exit_keys_for, bucket_of, _episode_starts, GAP, _rsi_x_dn_exit, _empty_exit_pool, backtest_ticker}`.
- Produces: `backtest_ticker_masked(df, allowed_dates) -> {sig: {"flat":{exit:[ret]}, "by_bucket":{blabel:{exit:[ret]}}}}` where `allowed_dates` is a `set` of `datetime.date` (or `None` = allow all).

- [ ] **Step 1: Create `h4_on_signals_study.py` with the masked backtest (importable, no Django at import)**

```python
#!/usr/bin/env python3
"""H4-on-daily-signals: gate the H4 engine by point-in-time candidate windows from daily systems
A (pure dip), B (capitulation), C (div_2x flagship). See
docs/superpowers/specs/2026-08-16-h4-on-daily-signals-design.md. Reuses h4_study + intraday_data;
modifies nothing. Pure functions import without Django; main() calls django.setup()."""
import numpy as np
import h4_study as H


def backtest_ticker_masked(df, allowed_dates=None):
    """Like h4_study.backtest_ticker but only counts entries whose bar date is in `allowed_dates`
    (a set of datetime.date). allowed_dates=None allows every bar (== h4_study.backtest_ticker)."""
    close = df["Close"].values
    n = len(close)
    dates = df.index.normalize()
    out = {}
    for sig, meta in H.SIGNALS.items():
        entry, mag = meta["fn"](df)
        buckets = meta["buckets"]
        cand = [i for i in range(n) if entry[i] and
                (allowed_dates is None or dates[i].date() in allowed_dates)]
        idxs = sorted(H._episode_starts(cand, gap=H.GAP))
        flat = H._empty_exit_pool(sig)
        by_bucket = {b[0]: H._empty_exit_pool(sig) for b in buckets}
        dn_exit = H._rsi_x_dn_exit(df) if meta.get("exit_fn") == "rsi_x_dn" else None
        for i in idxs:
            ep = float(close[i])
            if ep <= 0:
                continue
            blab = H.bucket_of(mag[i], buckets)
            for k, bars, _ in H.EXITS:
                j = i + bars
                if j < n:
                    r = (close[j] - ep) / ep * 100
                    flat[k].append(r)
                    if blab is not None:
                        by_bucket[blab][k].append(r)
            if dn_exit is not None:
                j = next((q for q in range(i + 1, n) if dn_exit[q]), None)
                if j is not None:
                    r = (close[j] - ep) / ep * 100
                    flat["rsi_x_dn"].append(r)
                    if blab is not None:
                        by_bucket[blab]["rsi_x_dn"].append(r)
        out[sig] = {"flat": flat, "by_bucket": by_bucket}
    return out
```

- [ ] **Step 2: Create `test_h4_on_signals.py` with the mask-correctness test**

```python
#!/usr/bin/env python3
"""Unit tests for h4_on_signals_study. Run:
  docker exec rotation-backend-1 python -u /app/test_h4_on_signals.py            # all
  docker exec rotation-backend-1 python -u /app/test_h4_on_signals.py masknoop   # one"""
import sys
import numpy as np
import pandas as pd


def _frame(n=200, seed_shift=0):
    # deterministic pseudo-random-ish walk without Math.random/Date: sine+ramp mix
    t = np.arange(n) + seed_shift
    close = 100 * (1 + 0.06*np.sin(t/7.0) + 0.0008*t + 0.02*np.sin(t/2.0))
    idx = pd.date_range("2023-01-02 08:00", periods=n, freq="4h", tz="UTC")
    return pd.DataFrame({"Open": close, "High": close*1.003, "Low": close*0.997,
                         "Close": close, "Volume": np.full(n, 1000.0)}, index=idx)


def test_masknoop():
    """allowed_dates=None (and a set of ALL dates) must reproduce h4_study.backtest_ticker exactly."""
    import h4_study as H
    from h4_on_signals_study import backtest_ticker_masked
    df = _frame()
    base = H.backtest_ticker(df)                    # no dtrend -> flat/by_bucket populated
    m_none = backtest_ticker_masked(df, None)
    all_dates = {d.date() for d in df.index}
    m_all = backtest_ticker_masked(df, all_dates)
    for sig in H.SIGNALS:
        for exitk, rows in base[sig]["flat"].items():
            assert m_none[sig]["flat"][exitk] == rows, f"none mismatch {sig}/{exitk}"
            assert m_all[sig]["flat"][exitk] == rows, f"all mismatch {sig}/{exitk}"
    print("test_masknoop OK")


def test_maskempty():
    """An empty allowed set yields zero trades everywhere."""
    from h4_on_signals_study import backtest_ticker_masked
    df = _frame()
    m = backtest_ticker_masked(df, set())
    total = sum(len(v) for s in m.values() for v in s["flat"].values())
    assert total == 0
    print("test_maskempty OK")


CHECKS = {"masknoop": test_masknoop, "maskempty": test_maskempty}


def main():
    names = [a for a in sys.argv[1:] if a in CHECKS] or list(CHECKS)
    for nm in names:
        CHECKS[nm]()
    print(f"\n{len(names)} check(s) passed.")


if __name__ == "__main__":
    main()
```

- [ ] **Step 3: Guard against stale stub dirs and confirm files exist**

Run:
```bash
for f in h4_on_signals_study.py test_h4_on_signals.py; do
  if [ -d "backend/$f" ]; then rmdir "backend/$f"; fi
done
ls -la h4_on_signals_study.py test_h4_on_signals.py
```
Expected: both listed as regular files.

- [ ] **Step 4: Add the two bind mounts to `docker-compose.yml`**

Insert after the line `  - ./test_h4_study.py:/app/test_h4_study.py` (inside the `&backend-volumes` anchor):

```yaml
  - ./h4_on_signals_study.py:/app/h4_on_signals_study.py
  - ./test_h4_on_signals.py:/app/test_h4_on_signals.py
```

- [ ] **Step 5: Recreate backend so mounts take effect**

Run:
```bash
MSYS_NO_PATHCONV=1 docker compose up -d backend
```
Expected: backend recreated, no "not a directory" error.

- [ ] **Step 6: Run the mask tests to verify they pass**

Run:
```bash
MSYS_NO_PATHCONV=1 docker exec rotation-backend-1 python -u /app/test_h4_on_signals.py
```
Expected: `test_masknoop OK`, `test_maskempty OK`, `2 check(s) passed.`

- [ ] **Step 7: Commit**

```bash
git add h4_on_signals_study.py test_h4_on_signals.py docker-compose.yml
git commit -m "feat(h4sig): masked H4 backtest primitive (reproduces h4_study with all-true mask)"
```

---

## Task 2: Candidate windows from A / B / C

Build, per selector, a map `{ticker: set(datetime.date)}` of dates on which an H4 entry is allowed, plus counts. C from saved `rotation_history`; A via arm3_lowpb reconstruction; B via the capitulation signal.

**Files:**
- Modify: `h4_on_signals_study.py` (add `candidate_windows`, helpers)
- Modify: `test_h4_on_signals.py` (add a counts sanity test)

**Interfaces:**
- Consumes: `h4_study.SIGNALS["seq_rsi20_ad_rising_rsi"]`; loaders `seq_fundamental_study.{load_candles, load_financial_reports, build_universe}`, `trend_stock_studies.{_pit_monthly_panel,_available_at,_ret_delist,CRYPTO}`, `backtest_lowpb.{_monthly_close,BENCH}`, `price_basis.as_traded_close`, `sector_holdings.get_holdings`, `config.SECTOR_ETFS`, `core.models.BacktestResult`.
- Produces: `candidate_windows(selector) -> (allowed: dict[str,set[date]], meta: dict)` where `selector in {"A","B","C","union"}`; `meta` has `n_windows`, `n_names`. Also `B_WINDOW_DAYS = 10`.

- [ ] **Step 1: Write a counts sanity test (needs DB; assert non-empty and dates are `date` objects)**

Add to `test_h4_on_signals.py` and `CHECKS`:

```python
def test_candwindows():
    import os, datetime as dt
    os.environ.setdefault("DJANGO_SETTINGS_MODULE", "rotation.settings")
    import django; django.setup()
    from h4_on_signals_study import candidate_windows
    for sel in ("C", "B"):
        allowed, meta = candidate_windows(sel)
        assert meta["n_names"] > 0 and meta["n_windows"] > 0, f"{sel} empty"
        tk = next(iter(allowed))
        assert isinstance(next(iter(allowed[tk])), dt.date), f"{sel} not date"
    print("test_candwindows OK")
```
```python
CHECKS = {"masknoop": test_masknoop, "maskempty": test_maskempty, "candwindows": test_candwindows}
```

- [ ] **Step 2: Run to verify it fails**

Run:
```bash
MSYS_NO_PATHCONV=1 docker exec rotation-backend-1 python -u /app/test_h4_on_signals.py candwindows
```
Expected: FAIL — `ImportError: cannot import name 'candidate_windows'`.

- [ ] **Step 3: Implement `candidate_windows` and helpers in `h4_on_signals_study.py`**

Add (module level; keep Django imports lazy inside the functions):

```python
import pandas as pd

B_WINDOW_DAYS = 10
LOOKBACK, TOP_N = 6, 10


def _month_dates(daily_index, start, end):
    """Trading dates in [start, end) from a daily DatetimeIndex."""
    s = pd.Timestamp(start); e = pd.Timestamp(end)
    return {d.date() for d in daily_index if s <= d < e}


def _windows_C():
    """C = div_2x flagship monthly picks from saved rotation_history. Each pick is a candidate for
    its holding month [date_i, date_{i+1})."""
    from core.models import BacktestResult
    from seq_fundamental_study import load_candles
    p = BacktestResult.objects.get(kind="rotation_history").payload
    months = p["months"]
    picks_by_name = {}
    for i, m in enumerate(months):
        start = m["date"]
        end = months[i + 1]["date"] if i + 1 < len(months) else None
        for pk in m["picks"]:
            picks_by_name.setdefault(pk["pick"], []).append((start, end))
    daily = load_candles(sorted(picks_by_name))
    allowed = {}
    nwin = 0
    for tk, spans in picks_by_name.items():
        df = daily.get(tk)
        if df is None:
            continue
        s = set()
        for start, end in spans:
            end = end or (pd.Timestamp(start) + pd.Timedelta(days=31))
            s |= _month_dates(df.index, start, end)
            nwin += 1
        if s:
            allowed[tk] = s
    return allowed, {"n_windows": nwin, "n_names": len(allowed)}


def _windows_A():
    """A = pure-dip (rsi10<45) monthly value pick in top-momentum sectors. Candidate for its holding
    month. Reconstructs arm3_lowpb selection (equal to the entry_signal study's selection)."""
    import numpy as np, ta
    import config, price_basis, sector_holdings
    from seq_fundamental_study import load_candles, load_financial_reports
    from trend_stock_studies import _pit_monthly_panel, _available_at, _ret_delist, CRYPTO
    from backtest_lowpb import _monthly_close, BENCH
    etfs = {name: etf for name, etf in config.SECTOR_ETFS.items() if etf not in CRYPTO}
    sector_map, all_holds = {}, set()
    for name, etf in etfs.items():
        h = [t for t in sector_holdings.get_holdings(name) if t not in (etf, BENCH) and t not in CRYPTO]
        sector_map[etf] = h; all_holds.update(h)
    all_holds = sorted(all_holds)
    etf_daily = load_candles(sorted(set(etfs.values()) | {BENCH}))
    etf_monthly = _monthly_close({t: d for t, d in etf_daily.items() if t in etfs.values()})
    midx = etf_monthly.index
    stock_daily = load_candles(all_holds)
    stock_monthly = _monthly_close(stock_daily).reindex(midx)
    reports = load_financial_reports(all_holds)
    shares_p = _pit_monthly_panel(reports, "shares_outstanding", midx)
    equity_p = _pit_monthly_panel(reports, "total_equity", midx)
    common = stock_monthly.columns.intersection(shares_p.columns).intersection(equity_p.columns)
    pb = (price_basis.as_traded_close(stock_monthly[common]) * shares_p[common]) / equity_p[common].where(equity_p[common] != 0)
    etf_trail = etf_monthly.pct_change(LOOKBACK)
    dip = {}
    for tk, df in stock_daily.items():
        if len(df) < 210:
            continue
        dip[tk] = (ta.momentum.rsi(df["Close"], window=10) < 45).reindex(midx, method="ffill")
    dip = pd.DataFrame(dip).reindex(midx)
    allowed, nwin = {}, 0
    for i in range(LOOKBACK, len(midx) - 1):
        date, ndate = midx[i], midx[i + 1]
        if date not in etf_trail.index:
            continue
        ranks = etf_trail.loc[date].dropna().sort_values(ascending=False).head(TOP_N)
        for etf in ranks.index:
            holds = sector_map.get(etf, [])
            cands = [h for h in holds if h in stock_monthly.columns and _available_at(stock_monthly[h], date)]
            if not cands or date not in pb.index:
                continue
            row = pb.loc[date, [c for c in cands if c in pb.columns]].dropna()
            row = row[row > 0]
            if not len(row):
                continue
            pick = row.idxmin()
            if pick in dip.columns and bool(dip.loc[date, pick]):
                df = stock_daily.get(pick)
                if df is None:
                    continue
                allowed.setdefault(pick, set())
                allowed[pick] |= _month_dates(df.index, date, ndate)
                nwin += 1
    return allowed, {"n_windows": nwin, "n_names": len(allowed)}


def _windows_B():
    """B = capitulation seq_rsi20_ad_rising_rsi fires -> candidate for the next B_WINDOW_DAYS trading days."""
    from seq_fundamental_study import load_candles, build_universe
    name, fn = H.SIGNALS["seq_rsi20_ad_rising_rsi"]
    daily = load_candles(build_universe())
    allowed, nwin = {}, 0
    for tk, df in daily.items():
        if len(df) < 60:
            continue
        sig = fn(df).fillna(False)
        idx = df.index
        fires = [i for i, v in enumerate(sig.values) if v]
        if not fires:
            continue
        s = set()
        for i in fires:
            for j in range(i, min(i + B_WINDOW_DAYS, len(idx))):
                s.add(idx[j].date())
            nwin += 1
        allowed[tk] = s
    return allowed, {"n_windows": nwin, "n_names": len(allowed)}


def candidate_windows(selector):
    """selector in {A,B,C,union} -> ({ticker: set[date]}, meta)."""
    if selector == "C":
        return _windows_C()
    if selector == "A":
        return _windows_A()
    if selector == "B":
        return _windows_B()
    if selector == "union":
        merged, nwin = {}, 0
        for sel in ("A", "B", "C"):
            a, m = candidate_windows(sel)
            nwin += m["n_windows"]
            for tk, s in a.items():
                merged.setdefault(tk, set()).update(s)
        return merged, {"n_windows": nwin, "n_names": len(merged)}
    raise ValueError(selector)
```

- [ ] **Step 4: Run the counts test to verify it passes**

Run:
```bash
MSYS_NO_PATHCONV=1 docker exec rotation-backend-1 python -u /app/test_h4_on_signals.py candwindows
```
Expected: `test_candwindows OK`. (Uses the DB; C and B must return non-empty maps of `date` sets.)

- [ ] **Step 5: Eyeball real counts**

Run:
```bash
MSYS_NO_PATHCONV=1 docker exec rotation-backend-1 python -c "
import os; os.environ.setdefault('DJANGO_SETTINGS_MODULE','rotation.settings')
import django; django.setup()
from h4_on_signals_study import candidate_windows
for sel in ('C','A','B'):
    a,m = candidate_windows(sel); print(sel, m, '| sample names', list(a)[:8])
"
```
Expected: each selector prints `n_windows`/`n_names` > 0 and a few tickers. (C ~50–150 names, A fewer, B many.) No traceback.

- [ ] **Step 6: Commit**

```bash
git add h4_on_signals_study.py test_h4_on_signals.py
git commit -m "feat(h4sig): point-in-time candidate windows from A/B/C selectors"
```

---

## Task 3: Masked run, benchmarks, DB save, CLI (staged fetch)

Tie it together: for each selector, over its candidate names, fetch 4h (fast pass = cached only, or fetch), run the masked backtest, pool per signal, aggregate, attach benchmarks, save.

**Files:**
- Modify: `h4_on_signals_study.py` (add `run`, `main`)

**Interfaces:**
- Consumes: `candidate_windows` (Task 2), `backtest_ticker_masked` (Task 1), `intraday_data.get_4h`, `h4_study.{agg_rows, exit_keys_for, SIGNALS, FAMILIES}`, `core.models.BacktestResult`.
- Produces: `BacktestResult(kind="h4_on_signals")` payload; CLI `--selector A|B|C|union|all`, `--no-fetch`, `--jobs`, `--years`.

- [ ] **Step 1: Implement `run` + `main`**

Add to `h4_on_signals_study.py`:

```python
def _merge(dst, src):
    for k, v in src.items():
        dst.setdefault(k, []).extend(v)


def _base_rate(frames, horizon_bars=3):
    pool = []
    for df in frames.values():
        c = df["Close"].values
        if len(c) > horizon_bars:
            r = (c[horizon_bars:] - c[:-horizon_bars]) / c[:-horizon_bars] * 100
            pool.extend([x for x in r if np.isfinite(x)])
    return round(float(np.mean(pool)), 3) if pool else None


def run_selector(selector, years, allow_fetch, jobs):
    from intraday_data import get_4h
    allowed, cwmeta = candidate_windows(selector)
    names = sorted(allowed)
    flat = {s: {} for s in H.SIGNALS}
    bucket = {s: {b[0]: {} for b in H.SIGNALS[s]["buckets"]} for s in H.SIGNALS}
    frames = {}
    got = dropped = 0
    for tk in names:
        df = get_4h(tk, years, allow_fetch)
        if df is None or len(df) < 120:
            dropped += 1
            continue
        got += 1
        frames[tk] = df
        res = backtest_ticker_masked(df, allowed[tk])
        for s in H.SIGNALS:
            _merge(flat[s], res[s]["flat"])
            for b, d in res[s]["by_bucket"].items():
                for k, v in d.items():
                    bucket[s][b].setdefault(k, []).extend(v)
    # standalone H4 (broad universe) benchmark, per signal
    standalone = {}
    try:
        from core.models import BacktestResult
        h4 = BacktestResult.objects.get(kind="h4_study").payload
        standalone = {s: h4["signals"][s]["all"] for s in H.SIGNALS if s in h4.get("signals", {})}
    except Exception:
        pass
    base3 = _base_rate(frames, 3)
    sigs = {}
    for s in H.SIGNALS:
        eks = H.exit_keys_for(s)
        border = [b[0] for b in H.SIGNALS[s]["buckets"]]
        sigs[s] = {"name": H.SIGNALS[s]["name"], "family": H.SIGNALS[s]["family"],
                   "all": H.agg_rows(flat[s], eks),
                   "by_bucket": {b: H.agg_rows(bucket[s][b], eks) for b in border},
                   "bucket_order": border,
                   "standalone_h4": standalone.get(s, [])}
    return {"selector": selector, "families": H.FAMILIES, "signals": sigs,
            "candidates": {**cwmeta, "n_with_4h": got, "n_dropped_no_4h": dropped},
            "base_rate_3b": base3}


def run(selectors, years, allow_fetch, jobs):
    return {sel: run_selector(sel, years, allow_fetch, jobs) for sel in selectors}


def main():
    import os, json, argparse
    os.environ.setdefault("DJANGO_SETTINGS_MODULE", "rotation.settings")
    import django; django.setup()
    from pathlib import Path
    ap = argparse.ArgumentParser()
    ap.add_argument("--selector", default="all", help="A|B|C|union|all")
    ap.add_argument("--no-fetch", action="store_true", help="fast pass: cached 4h only")
    ap.add_argument("--jobs", type=int, default=1)
    ap.add_argument("--years", type=float, default=5)
    args = ap.parse_args()
    sels = ["C", "A", "B", "union"] if args.selector == "all" else [args.selector]
    print(f"selectors={sels} fetch={'off' if args.no_fetch else 'on'}", flush=True)
    payload = {"by_selector": run(sels, args.years, allow_fetch=not args.no_fetch, jobs=args.jobs)}
    payload["computed_at"] = pd.Timestamp.utcnow().isoformat()
    payload["note"] = ("Daily A/B/C signal selects the name; the H4 engine times a 0-3 day entry inside the "
                       "candidate window. Bucketed by magnitude; benchmarked vs standalone H4 (broad universe) "
                       "and the candidate base rate. Gross of fees; EODHD 1h depth varies (dropped names counted); "
                       "current-membership candidate universe = survivorship bias; small/mid-cap H4 liquidity thinner.")
    STUD = Path(__file__).resolve().parent / ".data" / "studies"
    STUD.mkdir(parents=True, exist_ok=True)
    (STUD / "h4_on_signals.json").write_text(json.dumps(payload, indent=2, default=str))
    try:
        from core.models import BacktestResult
        from django.utils import timezone
        BacktestResult.objects.update_or_create(
            kind="h4_on_signals",
            defaults={"payload": json.loads(json.dumps(payload, default=str)), "computed_at": timezone.now()})
        print("saved BacktestResult[h4_on_signals]", flush=True)
    except Exception as e:
        print("DB save failed:", e, flush=True)
    for sel, p in payload["by_selector"].items():
        print(f"\n=== {sel}  ({p['candidates']}) base3b={p['base_rate_3b']} ===", flush=True)
        rows = []
        for s, d in p["signals"].items():
            best3 = next((r for r in d["all"] if r["exit"] == "3b"), None)
            if best3:
                sa = next((r for r in d["standalone_h4"] if r["exit"] == "3b"), None)
                rows.append((s, best3, sa))
        for s, b, sa in sorted(rows, key=lambda x: -x[1]["avg_pct"]):
            saj = f"(standalone {sa['avg_pct']:+.2f}%)" if sa else ""
            print(f"  {s:16} 3b avg {b['avg_pct']:+.2f}% win {b['win_pct']}% t={b['t']} n={b['trades']} {saj}", flush=True)


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Fast pass (cached-only) smoke run**

Run:
```bash
MSYS_NO_PATHCONV=1 docker exec rotation-backend-1 python -u /app/h4_on_signals_study.py --selector C --no-fetch 2>&1 | tail -25
```
Expected: prints `=== C ({...}) base3b=... ===` and a per-signal 3b table with `(standalone ...)` comparisons, then (from a full `--selector all` run) `saved BacktestResult[h4_on_signals]`. Some candidates will be dropped for no 4h data (counted in `n_dropped_no_4h`) — that's the fast pass. No traceback.

- [ ] **Step 3: Full backfill fetch — DETACHED (the one long job)**

Run:
```bash
MSYS_NO_PATHCONV=1 docker exec rotation-backend-1 sh -c 'cd /app && setsid nohup python -u /app/h4_on_signals_study.py --selector all --jobs 4 > /app/.data/h4_on_signals.log 2>&1 < /dev/null & echo launched'
```
Then confirm it started:
```bash
sleep 6; MSYS_NO_PATHCONV=1 docker exec rotation-backend-1 sh -c 'head -3 /app/.data/h4_on_signals.log'
```
Expected: `launched`, then the log shows `selectors=['C', 'A', 'B', 'union'] fetch=on`. Poll the log for `saved BacktestResult[h4_on_signals]` on completion (candidate names' 1h backfill takes a while).

- [ ] **Step 4: Verify the DB payload after the run completes**

Run:
```bash
MSYS_NO_PATHCONV=1 docker exec rotation-backend-1 python manage.py shell -c "
from core.models import BacktestResult
p = BacktestResult.objects.get(kind='h4_on_signals').payload
for sel, d in p['by_selector'].items():
    print(sel, d['candidates'], 'base3b', d['base_rate_3b'])
"
```
Expected: each selector prints candidate counts (`n_names`, `n_with_4h`, `n_dropped_no_4h`) and a base rate. `n_with_4h` > 0.

- [ ] **Step 5: Commit**

```bash
git add h4_on_signals_study.py
git commit -m "feat(h4sig): masked run + benchmarks + DB save + staged-fetch CLI"
```

---

## Task 4: API endpoint (`/api/h4-on-signals`)

**Files:**
- Modify: `backend/api/views.py` (add `H4OnSignalsView`)
- Modify: `backend/api/urls.py` (add route)

**Interfaces:**
- Consumes: `_StudyResultView` (`backend/api/views.py`), `BacktestResult(kind="h4_on_signals")` (Task 3).
- Produces: `GET /api/h4-on-signals` → payload; `POST` → background recompute.

- [ ] **Step 1: Add the view (after `H4StudyView`)**

In `backend/api/views.py`, after the `H4StudyView` class, add:

```python
class H4OnSignalsView(_StudyResultView):
    """H4-on-daily-signals: daily A/B/C selects the name, the H4 engine times a 0-3 day entry inside
    the candidate window. GET reads BacktestResult[h4_on_signals]; POST recomputes (fetch may take a while)."""
    kind = "h4_on_signals"
    script = "h4_on_signals_study.py"
    json_path = "/app/.data/studies/h4_on_signals.json"
```

- [ ] **Step 2: Add the URL route**

In `backend/api/urls.py`, after the `h4-study` line, add:

```python
    path("h4-on-signals", views.H4OnSignalsView.as_view(), name="h4-on-signals"),
```

- [ ] **Step 3: Verify GET returns the payload**

Run:
```bash
MSYS_NO_PATHCONV=1 docker compose restart backend >/dev/null 2>&1; sleep 6
OUT="C:/Users/User-X/AppData/Local/Temp/claude/C--workspace-rotation/760824d0-af6e-4024-99b0-65297a3e11ec/scratchpad/h4sig.json"
curl -s http://localhost:8001/api/h4-on-signals -o "$OUT"
python -c "import json; d=json.load(open(r'$OUT')); print('computed', d.get('computed')); print('selectors', list(d.get('by_selector',{}).keys()))"
```
Expected: `computed True` and `selectors ['C','A','B','union']`.

- [ ] **Step 4: Commit**

```bash
git add backend/api/views.py backend/api/urls.py
git commit -m "feat(h4sig): /api/h4-on-signals endpoint"
```

---

## Task 5: Frontend — `H4OnSignalsPage` sub-tab under Research

**Files:**
- Modify: `frontend/src/App.js` (add `H4OnSignalsPage` near `H4StudiesPage`; add one Research `HubTabs` entry)

**Interfaces:**
- Consumes: `apiFetch`, `LastUpdatedChip`, `TailStrip`, `RsiLadderTable`, `RsiBucketTable`, `ErrorBanner`; `GET/POST /h4-on-signals` (Task 4). Ladder rows use `{exit,name,trades,avg_pct,win_pct,t}` — identical to what `RsiLadderTable` renders.
- Produces: `/research/h4-on-signals` tab with an A/B/C/union selector toggle.

- [ ] **Step 1: Add `H4OnSignalsPage` (place immediately after `H4StudiesPage`)**

```jsx
// ---- H4 on Daily Signals ----------------------------------------------------
// Reads GET /h4-on-signals. Daily A/B/C selects the name; the H4 engine times a 0-3 day entry inside
// the candidate window. Selector toggle; each signal shows its ladder + magnitude buckets + the
// standalone-H4 (broad universe) ladder as the benchmark.
function H4OnSignalsPage() {
  const [data, setData] = useState(null);
  const [err, setErr] = useState(null);
  const [running, setRunning] = useState(false);
  const [sel, setSel] = useState('C');

  const load = () => { setErr(null); apiFetch('/h4-on-signals').then(setData).catch(e => setErr(e.message)); };
  useEffect(() => { load(); }, []);
  const run = () => {
    setRunning(true);
    apiFetch('/h4-on-signals', { method: 'POST' }).then(() => {
      const t = setInterval(() => apiFetch('/h4-on-signals').then(d => {
        if (d && d.computed) { clearInterval(t); setRunning(false); setData(d); }
      }).catch(() => {}), 10000);
    }).catch(e => { setErr(e.message); setRunning(false); });
  };

  if (err) return <div className="darkpool-page"><h1>H4 on Daily Signals</h1><ErrorBanner message={err} onRetry={load} onDismiss={() => setErr(null)} /></div>;
  if (!data) return <div className="loading">Loading H4-on-signals...</div>;
  if (!data.computed) return (
    <div className="darkpool-page"><h1>H4 on Daily Signals</h1>
      <div className="empty-state" style={{ padding: '40px 0' }}>
        <p>{data.note || data.message || 'Not computed yet.'}</p>
        <button className="refresh-btn" onClick={run} disabled={running}>{running ? 'Running (fetch + backtest)…' : 'Run study'}</button>
      </div>
    </div>
  );

  const bysel = data.by_selector || {};
  const cur = bysel[sel];
  const fams = (cur && cur.families) || {};
  const sigs = (cur && cur.signals) || {};
  return (
    <div className="darkpool-page">
      <h1>H4 on Daily Signals <LastUpdatedChip value={data.last_updated} /></h1>
      <div className="filters" style={{ margin: '4px 0 8px' }}>
        {['C', 'A', 'B', 'union'].map(k => (
          <button key={k} className={sel === k ? 'active' : ''} onClick={() => setSel(k)} disabled={!bysel[k]}>{k}</button>
        ))}
        <button className="refresh-btn" style={{ marginLeft: 10 }} onClick={run} disabled={running}>{running ? 'Running…' : 'Recompute'}</button>
      </div>
      <p className="subtitle darkpool-muted">
        Daily <b>{sel}</b> selects the name; the H4 engine times a <b>0–3 day</b> entry inside the candidate window.
        {cur ? ` Candidates: ${cur.candidates.n_names} names (${cur.candidates.n_with_4h} with 4h, ${cur.candidates.n_dropped_no_4h} dropped). Base 3b ${cur.base_rate_3b}%.` : ''}
        Each signal shows the on-signal ladder + magnitude buckets, with the standalone-H4 (broad universe) ladder as the benchmark.
      </p>
      {cur && Object.keys(fams).map(fam => (
        <div key={fam}>
          <h2 style={{ marginTop: 18 }}>{fam}</h2>
          {(fams[fam] || []).map(k => sigs[k] && (
            <div className="darkpool-card" key={k}>
              <div className="darkpool-card-head"><h2>{sigs[k].name} <span className="dim" style={{ fontSize: 12 }}>({k})</span></h2></div>
              <h3>On-signal ladder</h3>
              <RsiLadderTable rows={sigs[k].all || []} />
              <h3 style={{ marginTop: 10 }}>Standalone H4 (broad universe)</h3>
              <RsiLadderTable rows={sigs[k].standalone_h4 || []} />
              {(sigs[k].bucket_order || []).filter(b => (sigs[k].by_bucket?.[b] || []).length).map(b => (
                <RsiBucketTable key={b} bucket={b} rows={sigs[k].by_bucket[b]} />
              ))}
            </div>
          ))}
        </div>
      ))}
      {data.note && <p className="subtitle" style={{ marginTop: 10, fontStyle: 'italic' }}>{data.note}</p>}
    </div>
  );
}
```

- [ ] **Step 2: Register the sub-tab (after the `h4` entry in the Research `HubTabs` array)**

```jsx
        { key: 'h4sig', label: 'H4 on Signals', hash: '/research/h4sig', match: ['h4sig'], el: <H4OnSignalsPage /> },
```

- [ ] **Step 3: Rebuild only the frontend**

Run:
```bash
MSYS_NO_PATHCONV=1 docker compose up -d --build --no-deps frontend
```
Expected: frontend rebuilds/restarts; backend untouched.

- [ ] **Step 4: Verify the tab shipped in the served bundle**

Run:
```bash
JS=$(curl -s http://localhost:3001/ | grep -o '/static/js/main\.[a-z0-9]*\.js' | head -1)
OUT="C:/Users/User-X/AppData/Local/Temp/claude/C--workspace-rotation/760824d0-af6e-4024-99b0-65297a3e11ec/scratchpad/main2.js"
curl -s "http://localhost:3001$JS" -o "$OUT"
python -c "s=open(r'$OUT',encoding='utf-8').read(); [print(n, 'FOUND' if n in s else 'MISSING') for n in ['H4 on Daily Signals','/h4-on-signals','research/h4sig']]"
```
Expected: all three `FOUND`.

- [ ] **Step 5: Commit (submodule then parent pointer)**

```bash
cd frontend && git add src/App.js && git commit -m "feat(h4sig): H4-on-Daily-Signals research sub-tab"
cd .. && git add frontend && git commit -m "chore(rotation): bump frontend submodule — H4-on-Signals tab"
```

---

## Self-Review

**Spec coverage:**
- Handoff logic (daily selects, H4 times entry) → Task 1 masked backtest + Task 3 run. ✓
- Candidate windows C/A/B/union → Task 2. ✓ (C from saved rotation_history; A arm3_lowpb+rsi10<45; B fires→10 trading days.)
- Full H4 sweep gated by mask → Task 1 `backtest_ticker_masked` (all 14 signals × exit ladder × buckets). ✓
- Staged fetch (fast pass + detached backfill) → Task 3 Steps 2–3. ✓
- Benchmarks: standalone H4 + base rate → Task 3 (`standalone_h4`, `base_rate_3b`). Daily-own return: the selector's own numbers are already established (C +2.18%/pick, B +11.4%/trade, A totals) and surfaced in the tab subtitle context; the machine-comparable benchmarks stored are standalone-H4 and base rate. ✓ (daily-own kept as documented context, not recomputed — avoids re-running A/B/C.)
- Persist BacktestResult[h4_on_signals] + JSON → Task 3. ✓
- API + tab → Tasks 4, 5. ✓
- No-silent-truncation counts (n_names/n_with_4h/n_dropped) → Task 3 payload + tab. ✓
- PIT/no-look-ahead → Task 2 (windows from month-end/ fire dates; entries only within window). ✓
- Non-breaking (read-only imports, additive wiring) → all tasks; `h4_study` reused, not edited; A/B/C code imported only. ✓
- Mount gotcha, /dev/shm (candidate loaders use existing loaders which already handle it), docker run conventions → Task 1 Steps 3–5. ✓

**Placeholder scan:** No TBD/TODO; every code step has full bodies; commands have expected output. ✓

**Type consistency:** `candidate_windows -> (dict[str,set[date]], meta)` consumed by `run_selector`; `backtest_ticker_masked` returns `{sig:{flat,by_bucket}}` consumed by `run_selector`'s pooling; `agg_rows` output keys (`exit,name,trades,avg_pct,median_pct,win_pct,t`) match `RsiLadderTable`/`RsiBucketTable` inputs in Task 5. Payload shape (`by_selector[sel].signals[k].{all,by_bucket,bucket_order,standalone_h4}`, `.candidates`, `.base_rate_3b`) consistent between Task 3 producer and Task 5 consumer. ✓

**Deviation note:** the spec's third benchmark ("daily-system own return on the same names") is surfaced as documented context rather than recomputed in-engine, to honor "don't re-run A/B/C" — the standalone-H4 and base-rate benchmarks (the ones that decide whether daily-selection *adds* H4 edge) are computed. If a machine-comparable daily-own column is wanted later it's an additive follow-up.
