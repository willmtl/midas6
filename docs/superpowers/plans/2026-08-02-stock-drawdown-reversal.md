# "30% off ATH + RSI Reversal" Stock Strategy — Implementation Plan (Core + Backtest)

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a `drawdown_from_high` indicator and a `dd30_rsi_reversal` entry signal (≥30% below running ATH AND RSI(10) crosses above its SMA), then a per-stock backtest over the 858-ticker sector-holdings universe that proves whether the strategy works.

**Architecture:** The signal is the shared unit, registered in `studies.py`'s `SIGNALS` dict so its definition is canonical. A new `stock_drawdown_study.py` applies that signal to each stock's own OHLC (the existing stock runners are ETF-signal-driven and are not reused), backtests across all 70 `EXITS`, aggregates overall + by sector + top stocks, and writes `.data/studies/stock_drawdown.json`.

**Tech Stack:** Python, pandas, numpy, `ta`. No pytest — tests are standalone runnable scripts. Not a git repo — no commit steps.

## Global Constraints

- Drawdown: `df["Close"] / df["Close"].cummax() - 1.0` (≤ 0; `-0.30` = 30% below running ATH). ATH = 5y running high (no new data fetch).
- Reversal: `rsi = ta.momentum.rsi(df["Close"], window=10)`, `sma = rsi.rolling(10).mean()`, cross = `(rsi > sma) & (rsi.shift(1) <= sma.shift(1))`.
- Entry signal fires when `drawdown <= -DD_THRESHOLD` AND the RSI cross is true on that bar. `DD_THRESHOLD = 0.30` (tunable module constant in `studies.py`).
- Universe: distinct US holdings across all `config.SECTOR_ETFS` via `sector_holdings.get_holdings(sector)`; drop tickers containing `.`; each ticker assigned to the FIRST sector that lists it (no double-count).
- `EXITS[key] = (name, exit_fn)`; `exit_fn(df, entry_idx) -> exit_idx or None`. Trade return = `(close[exit_idx] - close[entry_idx]) / close[entry_idx] * 100`; hold = `exit_idx - entry_idx`.
- `data_fetcher.fetch_tickers(tickers: list[str]) -> dict[str, pd.DataFrame]`.
- Every signal Series is reindexed to `df.index` and `.fillna(False)`.
- `studies.py` does NOT currently import `indicators` — add `import indicators`.

---

### Task 1: `drawdown_from_high` indicator

**Files:**
- Modify: `indicators.py` — add function after `rolling_correlation` (added earlier in this session).
- Create: `test_stock_drawdown.py` (project root).

**Interfaces:**
- Consumes: `pandas` (imported in `indicators.py`).
- Produces (used by Task 2 & 3): `indicators.drawdown_from_high(df: pd.DataFrame) -> pd.Series` — fractional drop from running max close, indexed like `df`, `0.0` at a new high, `-0.30` at 30% below ATH.

- [ ] **Step 1: Write the failing test** — create `test_stock_drawdown.py`:

```python
"""Standalone tests for the 30%-off-ATH + RSI reversal strategy.
Run: python test_stock_drawdown.py"""
import numpy as np
import pandas as pd
import indicators
import studies as S


def _mk_df(closes):
    idx = pd.bdate_range("2022-01-03", periods=len(closes))
    c = pd.Series(closes, index=idx, dtype=float)
    return pd.DataFrame({"Open": c, "High": c * 1.005, "Low": c * 0.995,
                         "Close": c, "Volume": 1e6}, index=idx)


def test_drawdown_from_high_basic():
    df = _mk_df([100, 110, 120, 90, 78])   # peak 120, last = 78 -> 78/120-1 = -0.35
    dd = indicators.drawdown_from_high(df)
    assert dd.index.equals(df.index)
    assert abs(dd.iloc[-1] - (78 / 120 - 1)) < 1e-9
    assert dd.iloc[2] == 0.0                # at the running high -> 0
    assert (dd <= 1e-12).all()              # never positive


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in fns:
        fn()
        print(f"PASS {fn.__name__}")
    print(f"\nAll {len(fns)} tests passed.")
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python test_stock_drawdown.py`
Expected: FAIL — `AttributeError: module 'indicators' has no attribute 'drawdown_from_high'`.

- [ ] **Step 3: Add the indicator** to `indicators.py` (after `rolling_correlation`):

```python
def drawdown_from_high(df: pd.DataFrame) -> pd.Series:
    """Fractional drop from the running all-time-high close.
    0.0 at a new high; -0.30 means 30% below the running ATH."""
    close = df["Close"]
    return close / close.cummax() - 1.0
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python test_stock_drawdown.py`
Expected: `PASS test_drawdown_from_high_basic`, ending `All 1 tests passed.`

---

### Task 2: `sig_dd30_rsi_reversal` signal in `studies.py`

**Files:**
- Modify: `studies.py` — add `import indicators`; add `DD_THRESHOLD` + `sig_dd30_rsi_reversal`; register in `SIGNALS`.
- Modify: `test_stock_drawdown.py` — add signal tests.

**Interfaces:**
- Consumes: `indicators.drawdown_from_high` (Task 1), `ta`.
- Produces (used by Task 3): `studies.sig_dd30_rsi_reversal(df) -> pd.Series[bool]` on `df.index`; `studies.DD_THRESHOLD = 0.30`; `SIGNALS["dd30_rsi_reversal"]`.

- [ ] **Step 1: Write the failing test** — append to `test_stock_drawdown.py` (before `__main__`):

```python
def _rise_fall_recover():
    """Rise to a peak, fall >35%, then turn up enough for RSI to cross its SMA."""
    up = list(np.linspace(50, 120, 40))     # peak 120
    down = list(np.linspace(120, 74, 30))   # -38% from peak (74/120-1)
    recover = list(np.linspace(74, 88, 20))  # turn up while still >30% down
    return _mk_df(up + down + recover)


def test_signal_fires_on_reversal_while_down():
    df = _rise_fall_recover()
    sig = S.sig_dd30_rsi_reversal(df)
    assert sig.index.equals(df.index) and sig.dtype == bool
    fire = sig[sig].index
    assert len(fire) >= 1                      # fires at least once
    dd = indicators.drawdown_from_high(df)
    for d in fire:                             # every fire is >=30% below ATH
        assert dd.loc[d] <= -0.30 + 1e-9


def test_no_fire_while_falling():
    df = _rise_fall_recover()
    sig = S.sig_dd30_rsi_reversal(df)
    # during the straight decline (bars 41..68) momentum is down -> no fire
    assert sig.iloc[41:68].sum() == 0


def test_no_fire_when_shallow_drawdown():
    # peak 100 -> trough 78 = only 22% down, then recover; must NOT fire (<30%)
    df = _mk_df(list(np.linspace(60, 100, 40)) + list(np.linspace(100, 78, 20))
                + list(np.linspace(78, 92, 20)))
    sig = S.sig_dd30_rsi_reversal(df)
    dd = indicators.drawdown_from_high(df)
    assert dd.min() > -0.30                    # never reached 30% down
    assert sig.sum() == 0


def test_registered_in_signals():
    assert "dd30_rsi_reversal" in S.SIGNALS
    assert S.SIGNALS["dd30_rsi_reversal"][1] is S.sig_dd30_rsi_reversal
    assert S.DD_THRESHOLD == 0.30
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python test_stock_drawdown.py`
Expected: FAIL — `AttributeError: module 'studies' has no attribute 'sig_dd30_rsi_reversal'`.

- [ ] **Step 3a: Add the import** — in `studies.py`, change:

```python
import config
import data_fetcher
```
to:
```python
import config
import data_fetcher
import indicators
```

- [ ] **Step 3b: Add constant + signal** — in `studies.py`, immediately after `sig_macd_great` (near line 154, before the market-signal constants added earlier):

```python
DD_THRESHOLD = 0.30  # min drawdown from running ATH to qualify (0.30 = 30% below high)


def sig_dd30_rsi_reversal(df):
    """>=30% below running ATH AND RSI(10) crosses above its SMA(10) on that bar."""
    dd = indicators.drawdown_from_high(df)
    rsi = ta.momentum.rsi(df["Close"], window=10)
    sma = rsi.rolling(10).mean()
    cross = (rsi > sma) & (rsi.shift(1) <= sma.shift(1))
    return ((dd <= -DD_THRESHOLD) & cross).fillna(False).astype(bool)
```

- [ ] **Step 3c: Register in `SIGNALS`** — add before the closing `}` of the `SIGNALS` dict (right after the `spy_qqq_rsi_x_*` market entries added earlier, near line 1774):

```python
    "dd30_rsi_reversal": ("30% off ATH + RSI Reversal", sig_dd30_rsi_reversal),
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python test_stock_drawdown.py`
Expected: all PASS, ending `All 5 tests passed.`

- [ ] **Step 5: Verify no import cycle / regression**

Run: `python -c "import studies; print('ok', studies.DD_THRESHOLD, 'dd30_rsi_reversal' in studies.SIGNALS)"`
Expected: `ok 0.3 True`
(If an ImportError about a circular import between `studies` and `indicators` appears: `indicators.py` must not import `studies` — confirm with `grep -n "import studies" indicators.py` returning nothing.)

---

### Task 3: `stock_drawdown_study.py` per-stock backtest runner

**Files:**
- Create: `stock_drawdown_study.py` (project root).
- Modify: `test_stock_drawdown.py` — add a study-runner unit test using a fake data loader.

**Interfaces:**
- Consumes: `studies.SIGNALS["dd30_rsi_reversal"]`, `studies.EXITS`, `sector_holdings.get_holdings`, `config.SECTOR_ETFS`, `data_fetcher.fetch_tickers`.
- Produces:
  - `build_universe() -> tuple[list[str], dict[str, str]]` — `(tickers, ticker_to_sector)`; each ticker assigned to the first sector listing it; `.`-tickers dropped.
  - `run_one_exit(exit_key, stock_data, ticker_to_sector) -> dict` — aggregate result for one exit; keys: `exit_key, exit_name, total_trades, avg_return, win_rate, avg_hold, sector_count, best_sectors, worst_sectors, top_stocks`.
  - `run_all(limit: int | None = None) -> list[dict]` — loads universe data, runs all `EXITS`, writes `.data/studies/stock_drawdown.json`, returns the results list.

- [ ] **Step 1: Write the failing test** — append to `test_stock_drawdown.py` (before `__main__`):

```python
def test_run_one_exit_aggregates_trades(monkeypatch=None):
    import stock_drawdown_study as sd
    # Two synthetic stocks that each produce a dd30 reversal entry
    a = _rise_fall_recover()
    b = _rise_fall_recover()
    stock_data = {"AAA": a, "BBB": b}
    t2s = {"AAA": "Tech", "BBB": "Energy"}
    res = sd.run_one_exit("1w", stock_data, t2s)
    assert res["exit_key"] == "1w"
    assert res["total_trades"] >= 2            # at least one entry per stock
    assert isinstance(res["avg_return"], float)
    assert res["sector_count"] >= 1
    assert isinstance(res["best_sectors"], list)


def test_build_universe_dedupes_and_drops_dotted():
    import stock_drawdown_study as sd
    tickers, t2s = sd.build_universe()
    assert len(tickers) == len(set(tickers))   # deduped
    assert all("." not in t for t in tickers)  # no foreign tickers
    assert all(t in t2s for t in tickers)      # every ticker mapped to a sector
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python test_stock_drawdown.py`
Expected: FAIL — `ModuleNotFoundError: No module named 'stock_drawdown_study'`.

- [ ] **Step 3: Create `stock_drawdown_study.py`:**

```python
#!/usr/bin/env python3
"""Backtest: buy stocks >=30% below their running ATH when RSI(10) crosses above its SMA.
Applies the `dd30_rsi_reversal` signal to each stock in the sector-holdings universe,
across all EXITS. Saves .data/studies/stock_drawdown.json.  Run: python stock_drawdown_study.py
"""
import warnings
warnings.filterwarnings("ignore")

import json
from pathlib import Path

import config
import data_fetcher
import sector_holdings
from studies import SIGNALS, EXITS

STUDIES_DIR = Path(__file__).parent / ".data" / "studies"
STUDIES_DIR.mkdir(parents=True, exist_ok=True)

_SIG_NAME, _SIG_FN = SIGNALS["dd30_rsi_reversal"]
MIN_BARS = 60
MIN_STOCK_TRADES = 3  # min trades for a ticker to appear in top_stocks


def build_universe():
    """Distinct US holdings; each ticker -> first sector that lists it."""
    tickers, t2s = [], {}
    for sector in config.SECTOR_ETFS:
        for t in (sector_holdings.get_holdings(sector) or []):
            if "." in t or t in t2s:
                continue
            t2s[t] = sector
            tickers.append(t)
    return tickers, t2s


def _entries(sdf):
    """Entry integer-indices for a single stock df (empty if none / too short)."""
    if sdf is None or len(sdf) < MIN_BARS:
        return []
    try:
        sig = _SIG_FN(sdf).fillna(False)
    except Exception:
        return []
    return [sdf.index.get_loc(d) for d in sig[sig].index]


def run_one_exit(exit_key, stock_data, ticker_to_sector):
    exit_name, exit_fn = EXITS[exit_key]
    rets, holds = [], []
    sectors_hit = set()
    per_sector = {}   # sector -> list[ret]
    per_stock = {}    # ticker -> list[ret]

    for ticker, sdf in stock_data.items():
        close = sdf["Close"].values
        n = len(close)
        for idx in _entries(sdf):
            exit_idx = exit_fn(sdf, idx)
            if exit_idx is None or exit_idx <= idx or exit_idx >= n:
                continue
            entry_p = float(close[idx])
            if entry_p <= 0:
                continue
            ret = (float(close[exit_idx]) - entry_p) / entry_p * 100
            rets.append(ret)
            holds.append(exit_idx - idx)
            sec = ticker_to_sector.get(ticker, "?")
            sectors_hit.add(sec)
            per_sector.setdefault(sec, []).append(ret)
            per_stock.setdefault(ticker, []).append(ret)

    def _agg(name, lst):
        return {"name": name, "trades": len(lst),
                "win_rate": round(sum(1 for r in lst if r > 0) / len(lst) * 100, 1),
                "avg_return": round(sum(lst) / len(lst), 3)}

    sector_aggs = [_agg(s, l) for s, l in per_sector.items()]
    sector_aggs.sort(key=lambda a: a["avg_return"], reverse=True)
    stock_aggs = [_agg(t, l) for t, l in per_stock.items() if len(l) >= MIN_STOCK_TRADES]
    stock_aggs.sort(key=lambda a: a["avg_return"], reverse=True)

    total = len(rets)
    return {
        "exit_key": exit_key,
        "exit_name": exit_name,
        "total_trades": total,
        "avg_return": round(sum(rets) / total, 3) if total else 0.0,
        "win_rate": round(sum(1 for r in rets if r > 0) / total * 100, 1) if total else 0.0,
        "avg_hold": round(sum(holds) / len(holds), 1) if holds else 0.0,
        "sector_count": len(sectors_hit),
        "best_sectors": sector_aggs[:5],
        "worst_sectors": sector_aggs[-5:][::-1],
        "top_stocks": stock_aggs[:10],
    }


def run_all(limit=None):
    tickers, t2s = build_universe()
    if limit:
        tickers = tickers[:limit]
    print(f"Universe: {len(tickers)} tickers. Loading data...")
    stock_data = data_fetcher.fetch_tickers(tickers)
    print(f"Loaded {len(stock_data)} tickers. Running {len(EXITS)} exits...")

    results = []
    for i, exit_key in enumerate(EXITS, 1):
        results.append(run_one_exit(exit_key, stock_data, t2s))
        if i % 10 == 0:
            print(f"  [{i}/{len(EXITS)}]")

    out = STUDIES_DIR / "stock_drawdown.json"
    out.write_text(json.dumps(results, indent=2))
    print(f"\nSaved {len(results)} exit-results to {out}")

    ranked = sorted(results, key=lambda r: r["avg_return"], reverse=True)
    print("\nTop 8 exits by avg return:")
    for r in ranked[:8]:
        print(f"  {r['exit_key']:12s} avg={r['avg_return']:+.3f}%  wr={r['win_rate']:.0f}%  "
              f"hold={r['avg_hold']:.0f}d  trades={r['total_trades']}")
    return results


if __name__ == "__main__":
    run_all()
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python test_stock_drawdown.py`
Expected: all PASS, ending `All 7 tests passed.`

- [ ] **Step 5: Smoke-run on a small real subset**

Run: `python -c "import stock_drawdown_study as sd; r=sd.run_one_exit('1w', __import__('data_fetcher').fetch_tickers(sd.build_universe()[0][:5]), sd.build_universe()[1]); print('trades', r['total_trades'], 'avg', r['avg_return'], 'sectors', r['sector_count'])"`
Expected: prints a trade count (≥0) and a numeric avg; no exception.

---

### Task 4: Full backtest run + read the findings

**Files:** none modified.

- [ ] **Step 1: Run the full study**

Run: `python stock_drawdown_study.py`
Expected: logs `Universe: 858 tickers` (approx), `Loaded N tickers`, progress `[10/70]…[70/70]`, then `Saved 70 exit-results` and a Top-8 table. No exception.

- [ ] **Step 2: Verify the JSON output**

Run:
```bash
python -c "import json; d=json.load(open('.data/studies/stock_drawdown.json')); print('exits:', len(d)); best=max(d,key=lambda r:r['avg_return']); print('best exit:', best['exit_key'], best['avg_return'], 'wr', best['win_rate'], 'trades', best['total_trades']); print('best sectors:', [s['name'] for s in best['best_sectors']])"
```
Expected: `exits: 70`; a best exit with a numeric avg_return, a plausible win rate, a non-zero trade count, and a list of sector names.

- [ ] **Step 3: Summarize findings**

Report: overall trade count, best/worst exit horizons, best/worst sectors, top individual stocks, and whether the strategy shows a positive edge (compare best exits' avg_return and win_rate). State caveats (small-sample sectors/stocks).

---

## Self-Review notes

- **Spec coverage:** indicator (Task 1), signal + `SIGNALS` registration + `DD_THRESHOLD` + `import indicators` (Task 2), per-stock runner over deduped 858 universe with by-sector + top-stock aggregation + JSON output (Task 3), full run + findings (Task 4). All spec sections mapped.
- **Placeholder scan:** none — all code shown in full; exact commands with expected output.
- **Type consistency:** `drawdown_from_high(df)->Series`, `sig_dd30_rsi_reversal(df)->bool Series`, `EXITS[key]=(name, fn)`, `exit_fn(df, idx)->idx|None`, `fetch_tickers(list)->dict`, `build_universe()->(list, dict)` used consistently across tasks and tests.
- **Boundary correctness:** entries use `sdf.index.get_loc(date)` positional indices against `close = sdf["Close"].values`; exit guarded by `exit_idx >= n`. `_agg` never divides by zero (only called on non-empty lists; `top_stocks` filtered by `MIN_STOCK_TRADES`).
- **No pytest / no git:** standalone test script; no commit steps (environment is not a git repo).
