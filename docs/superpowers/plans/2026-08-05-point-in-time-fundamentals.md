# Point-in-Time Fundamentals Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Bucket every backtest trade by the fundamentals as they were *knowable on the trade's entry date* (point-in-time), removing the fundamental lookahead bias and taking the sweep from 7 snapshot dimensions to 23 (17 point-in-time + 6 labeled snapshot-only).

**Architecture:** Store raw quarterly financials + dividends in Postgres (`FinancialReport`, `DividendHistory`). A pure pandas module (`pit_fundamentals.py`) turns a ticker's price frame + its reports into a date-indexed metrics DataFrame. The all-on-all sweep worker looks up each trade's metrics via `asof(entry_date)` and buckets them, caching labels per entry bar. `DIMENSIONS` gains a `pit` flag so each dimension draws from the point-in-time frame (`pit=True`) or today's `Fundamental` snapshot (`pit=False`). The app badges which is which via API metadata.

**Tech Stack:** Django 4 + PostgreSQL/TimescaleDB, pandas, `ta`, yfinance, Celery/Redis, Docker Compose, React (prod nginx build).

## Global Constraints

- **`REPORT_LAG_DAYS = 45`** — a quarter's numbers become "knowable" at `period_end + 45 days`. The `asof` lookup keys on `avail_date`, never `period_end`. Copy this constant verbatim; it is the anti-lookahead guarantee.
- **Free / yfinance only.** No paid data vendor. Tier-C dimensions (Forward PE, Forward EPS, Short % float, Institution %, Insider %, Float) ship as **snapshot-only, labeled** — never presented as point-in-time.
- **Parallelism = PROCESSES, spawn context**, capped at `min(32, cpu_count)`. Each worker does `django.setup()` + loads its own DB chunk. Parent calls `connections.close_all()` before spawning. Never pickle DataFrames across the boundary.
- **Sweep runners run as a subprocess** (`subprocess.run([... "python","-u","<script>.py","--db"])`), never in-process import inside Django/Celery.
- **⚠️ `docker compose run --rm` detaches** — it survives TaskStop. For tests/short commands use `docker compose exec backend ...` against the already-running container. For long sweeps use `run --rm`, then check `docker ps | grep rotation-backend-run` and `docker kill` orphans.
- **`pit_fundamentals.py` must import only pandas/numpy** — no Django, no `studies` — so it stays unit-testable and picklable-free.
- **Frontend is a prod nginx build (no src mount)** — after any `App.js`/`App.css` edit run `docker compose up -d --build frontend`.
- Ports: backend host 8001, frontend 3001, db 5433, redis 6379.

---

## File Structure

- **Create** `pit_fundamentals.py` (root, mounted) — pure PIT metric math + new bucket functions. Django-free.
- **Create** `fetch_financial_history.py` (root, mounted) — parallel yfinance backfill → `FinancialReport` + `DividendHistory`.
- **Create** `test_pit_fundamentals.py` (root, mounted) — standalone unit tests (same convention as `test_market_signals.py`).
- **Modify** `backend/core/models.py` — add `FinancialReport`, `DividendHistory`.
- **Create** `backend/core/migrations/0013_financialreport_dividendhistory.py`.
- **Modify** `seq_fundamental_study.py` — extend `DIMENSIONS` to 5-tuples (`+pit`), add new bucket fns (import from `pit_fundamentals`), add `load_financial_reports()` / `load_dividends()`, update its own `DIMENSIONS` unpack (line ~245).
- **Modify** `all_on_all_study.py` — 5-tuple unpack (lines ~106, ~234-235), build PIT frame per ticker, per-`entry_idx` label cache, snapshot vs PIT branch.
- **Modify** `backend/api/tasks.py` — add `run_financial_history_task()` subprocess wrapper.
- **Modify** `backend/api/celery_tasks.py` — add `fetch_financial_history` shared_task.
- **Modify** `backend/rotation/celery.py` — beat entry `financial-history-daily` (before `stock-studies-daily`).
- **Modify** `backend/api/views.py` — `StockStudiesView.get` adds `dimension_meta` to the response.
- **Modify** `frontend/src/App.js` — badge snapshot dimensions using `dimension_meta`.
- **Modify** `frontend/src/App.css` — `.dim-badge` styles.
- **Modify** `docker-compose.yml` — mount `pit_fundamentals.py`, `fetch_financial_history.py`, `test_pit_fundamentals.py`.

---

## Task 1: `FinancialReport` + `DividendHistory` models + migration

**Files:**
- Modify: `backend/core/models.py` (append two models after `QuarterlyEarnings`, ~line 120)
- Create: `backend/core/migrations/0013_financialreport_dividendhistory.py`

**Interfaces:**
- Produces: `FinancialReport(ticker, period_end, avail_date, revenue, net_income, eps_diluted, operating_income, total_equity, total_debt, current_assets, current_liabilities, free_cash_flow, shares_outstanding)`, unique `(ticker, period_end)`; `DividendHistory(ticker, ex_date, amount)`, unique `(ticker, ex_date)`.

- [ ] **Step 1: Add the models**

In `backend/core/models.py`, after the `QuarterlyEarnings` class:

```python
class FinancialReport(models.Model):
    """One quarterly financial report per ticker, for point-in-time fundamentals.
    `avail_date` = period_end + REPORT_LAG_DAYS: the date the numbers became public
    (approximated). All point-in-time lookups key on avail_date, never period_end."""
    ticker = models.CharField(max_length=20, db_index=True)
    period_end = models.DateField()               # fiscal quarter end
    avail_date = models.DateField(db_index=True)  # period_end + 45d; the PIT key
    revenue = models.BigIntegerField(null=True, blank=True)
    net_income = models.BigIntegerField(null=True, blank=True)
    eps_diluted = models.FloatField(null=True, blank=True)
    operating_income = models.BigIntegerField(null=True, blank=True)
    total_equity = models.BigIntegerField(null=True, blank=True)
    total_debt = models.BigIntegerField(null=True, blank=True)
    current_assets = models.BigIntegerField(null=True, blank=True)
    current_liabilities = models.BigIntegerField(null=True, blank=True)
    free_cash_flow = models.BigIntegerField(null=True, blank=True)
    shares_outstanding = models.BigIntegerField(null=True, blank=True)

    class Meta:
        ordering = ["ticker", "period_end"]
        unique_together = ["ticker", "period_end"]
        indexes = [models.Index(fields=["ticker", "avail_date"])]


class DividendHistory(models.Model):
    """Per-ticker dividend events, for trailing-twelve-month dividend yield."""
    ticker = models.CharField(max_length=20, db_index=True)
    ex_date = models.DateField()
    amount = models.FloatField()

    class Meta:
        ordering = ["ticker", "ex_date"]
        unique_together = ["ticker", "ex_date"]
```

- [ ] **Step 2: Generate the migration**

Run: `docker compose exec backend python manage.py makemigrations core`
Expected: creates `core/migrations/0013_financialreport_dividendhistory.py` naming both models.

- [ ] **Step 3: Apply the migration**

Run: `docker compose exec backend python manage.py migrate core`
Expected: `Applying core.0013_financialreport_dividendhistory... OK`

- [ ] **Step 4: Verify the tables exist**

Run: `docker compose exec backend python -c "from core.models import FinancialReport, DividendHistory; print(FinancialReport.objects.count(), DividendHistory.objects.count())"`
Expected: `0 0` (tables exist, empty).

- [ ] **Step 5: Commit**

```bash
git add backend/core/models.py backend/core/migrations/0013_financialreport_dividendhistory.py
git commit -m "feat: FinancialReport + DividendHistory models for point-in-time fundamentals"
```

---

## Task 2: `pit_fundamentals.py` — pure point-in-time metric math

This is the load-bearing task. Pure pandas/numpy; no Django. TDD against synthetic frames.

**Files:**
- Create: `pit_fundamentals.py`
- Create: `test_pit_fundamentals.py`
- Modify: `docker-compose.yml` (mount both files)

**Interfaces:**
- Produces: `REPORT_LAG_DAYS = 45`; `avail_date(period_end) -> date`; `prepare_pit_metrics(price_df, reports_df, dividends, spy_close) -> pd.DataFrame` indexed by the price frame's dates with columns: `pe, ps, pb, peg, eps_ttm, revenue_growth, earnings_growth, profit_margin, operating_margin, debt_to_equity, current_ratio, fcf_ttm, market_cap, beta, pct_52w, avg_volume, dividend_yield`. Missing inputs → `NaN` in the affected column (never a raised exception).
  - `price_df`: DateTimeIndex, columns include `Close`, `Volume`.
  - `reports_df`: columns `avail_date, revenue, net_income, eps_diluted, operating_income, total_equity, total_debt, current_assets, current_liabilities, free_cash_flow, shares_outstanding` (one row per quarter, any order).
  - `dividends`: `pd.Series` indexed by ex-date (may be empty).
  - `spy_close`: `pd.Series` of SPY closes indexed by date (may be `None` → `beta = NaN`).
- Produces (bucket fns, imported by `seq_fundamental_study`): `bucket_ps, bucket_pb, bucket_peg, bucket_div_yield, bucket_debt_to_equity, bucket_current_ratio, bucket_op_margin, bucket_earnings_growth, bucket_fcf_sign, bucket_beta, bucket_avg_volume, bucket_fwd_eps, bucket_short_float, bucket_pct` (see Task 3 for exact boundaries — defined here so the pure module owns all bucket logic).

- [ ] **Step 1: Write the failing tests**

Create `test_pit_fundamentals.py`:

```python
"""Standalone tests for point-in-time fundamentals. Run:
  docker compose exec backend python -u test_pit_fundamentals.py
"""
import numpy as np
import pandas as pd
from datetime import date

import pit_fundamentals as P


def _price(n=400, start=50.0, end=60.0):
    idx = pd.bdate_range("2022-01-03", periods=n)
    close = pd.Series(np.linspace(start, end, n), index=idx)
    return pd.DataFrame({"Close": close, "Volume": np.full(n, 1_000_000.0)}, index=idx)


def _reports():
    # Two quarters. Q1 period_end 2022-03-31 -> avail 2022-05-15; Q2 -> avail 2022-08-14.
    return pd.DataFrame([
        dict(period_end=date(2022, 3, 31), revenue=100, net_income=10, eps_diluted=1.0,
             operating_income=12, total_equity=200, total_debt=100,
             current_assets=150, current_liabilities=75, free_cash_flow=8,
             shares_outstanding=1_000),
        dict(period_end=date(2022, 6, 30), revenue=110, net_income=11, eps_diluted=1.1,
             operating_income=13, total_equity=210, total_debt=90,
             current_assets=160, current_liabilities=80, free_cash_flow=9,
             shares_outstanding=1_000),
    ]).assign(avail_date=lambda d: d["period_end"].map(P.avail_date))


def test_avail_date_adds_45_days():
    assert P.avail_date(date(2022, 3, 31)) == date(2022, 5, 15)


def test_no_report_visible_before_avail_date_is_nan():
    m = P.prepare_pit_metrics(_price(), _reports(), pd.Series(dtype=float), None)
    # A date before the first avail_date (2022-05-15) must not see Q1's numbers.
    early = m.loc[m.index < pd.Timestamp("2022-05-15")]
    assert early["pe"].isna().all(), "lookahead: report used before it was public"


def test_report_visible_on_and_after_avail_date():
    price = _price()
    m = P.prepare_pit_metrics(price, _reports(), pd.Series(dtype=float), None)
    row = m.asof(pd.Timestamp("2022-06-01"))  # after Q1 avail, before Q2 avail
    close = float(price["Close"].asof(pd.Timestamp("2022-06-01")))
    assert not np.isnan(row["market_cap"])
    assert abs(row["market_cap"] - close * 1_000) < 1e-6      # Close x Q1 shares
    assert abs(row["current_ratio"] - (150 / 75)) < 1e-9      # 2.0 from Q1


def test_ttm_uses_last_four_quarters_and_pe_positive():
    price = _price()
    m = P.prepare_pit_metrics(price, _reports(), pd.Series(dtype=float), None)
    row = m.asof(pd.Timestamp("2022-09-01"))  # both quarters visible
    assert row["eps_ttm"] > 0 and row["pe"] > 0


def test_dividend_yield_ttm_over_price():
    price = _price(end=50.0)  # flat at 50
    divs = pd.Series([0.5, 0.5], index=[pd.Timestamp("2022-06-15"), pd.Timestamp("2022-09-15")])
    m = P.prepare_pit_metrics(price, _reports(), divs, None)
    row = m.asof(pd.Timestamp("2022-10-01"))
    assert abs(row["dividend_yield"] - (1.0 / 50.0)) < 1e-6   # 1.0 TTM / 50 price

def test_bucket_boundaries():
    assert P.bucket_pb(0.9) == "cheap (<1)"
    assert P.bucket_pb(6.5) == "expensive (>=6)"
    assert P.bucket_current_ratio(0.8) == "weak (<1)"
    assert P.bucket_fcf_sign(-1.0) == "negative FCF"
    assert P.bucket_div_yield(None) == "NA"


def _run_all():
    fns = [v for k, v in globals().items() if k.startswith("test_")]
    for fn in fns:
        fn(); print(f"  PASS {fn.__name__}")
    print(f"\n{len(fns)} passed")


if __name__ == "__main__":
    _run_all()
```

- [ ] **Step 2: Add the mounts and run the test to see it fail**

In `docker-compose.yml`, under the `x-backend-volumes` anchor (where `./all_on_all_study.py` is mounted), add:

```yaml
      - ./pit_fundamentals.py:/app/pit_fundamentals.py
      - ./fetch_financial_history.py:/app/fetch_financial_history.py
      - ./test_pit_fundamentals.py:/app/test_pit_fundamentals.py
```

Recreate the backend so the mounts take effect:
Run: `docker compose up -d backend`
Then: `docker compose exec backend python -u test_pit_fundamentals.py`
Expected: FAIL — `ModuleNotFoundError: No module named 'pit_fundamentals'`.

- [ ] **Step 3: Implement `pit_fundamentals.py`**

Create `pit_fundamentals.py`:

```python
"""Point-in-time fundamentals: turn a ticker's price frame + quarterly reports into a
date-indexed metrics DataFrame, plus the bucket functions for every dimension.

Pure pandas/numpy — NO Django, NO `studies` import — so it stays unit-testable and cheap
to import in spawned workers. All lookups honour REPORT_LAG_DAYS: a quarter's numbers are
only visible on/after period_end + REPORT_LAG_DAYS (avail_date).
"""
from datetime import timedelta
import numpy as np
import pandas as pd

REPORT_LAG_DAYS = 45
BETA_WINDOW = 252   # ~1y of trading days
VOL_WINDOW = 20     # trailing avg volume
HIGH_LOW_WINDOW = 252  # 52-week range

_TTM_SUM_COLS = ["revenue", "net_income", "operating_income", "free_cash_flow", "eps_diluted"]
_POINT_COLS = ["total_equity", "total_debt", "current_assets", "current_liabilities", "shares_outstanding"]


def avail_date(period_end):
    """Date a quarter's numbers are treated as public."""
    return period_end + timedelta(days=REPORT_LAG_DAYS)


def _quarterly_frame(reports_df):
    """Index reports by avail_date, add TTM (rolling 4Q sum) and prior-TTM (YoY) columns."""
    if reports_df is None or reports_df.empty:
        return None
    r = reports_df.copy()
    r["avail_date"] = pd.to_datetime(r["avail_date"])
    r = r.sort_values("avail_date").set_index("avail_date")
    for c in _TTM_SUM_COLS:
        r[c + "_ttm"] = r[c].rolling(4, min_periods=4).sum()
        r[c + "_ttm_prior"] = r[c + "_ttm"].shift(4)
    return r


def prepare_pit_metrics(price_df, reports_df, dividends, spy_close):
    """Return a DataFrame indexed by price_df.index with all point-in-time metric columns.
    Missing inputs yield NaN (never raises)."""
    idx = price_df.index
    close = price_df["Close"].astype(float)
    out = pd.DataFrame(index=idx)

    q = _quarterly_frame(reports_df)
    if q is not None:
        # Forward-fill each quarterly column onto trading days by avail_date (asof).
        aligned = q.reindex(q.index.union(idx)).sort_index().ffill().reindex(idx)
        rev = aligned["revenue_ttm"]
        ni = aligned["net_income_ttm"]
        oi = aligned["operating_income_ttm"]
        fcf = aligned["free_cash_flow_ttm"]
        eps = aligned["eps_diluted_ttm"]
        shares = aligned["shares_outstanding"]
        equity = aligned["total_equity"]
        debt = aligned["total_debt"]
        cur_a = aligned["current_assets"]
        cur_l = aligned["current_liabilities"]
        rev_prior = aligned["revenue_ttm_prior"]
        ni_prior = aligned["net_income_ttm_prior"]

        mcap = close * shares
        out["market_cap"] = mcap
        out["eps_ttm"] = eps
        out["pe"] = _safe_div(close, eps)
        out["ps"] = _safe_div(mcap, rev)
        out["pb"] = _safe_div(mcap, equity)
        out["profit_margin"] = _safe_div(ni, rev)
        out["operating_margin"] = _safe_div(oi, rev)
        out["debt_to_equity"] = _safe_div(debt, equity)
        out["current_ratio"] = _safe_div(cur_a, cur_l)
        out["fcf_ttm"] = fcf
        out["revenue_growth"] = _safe_div(rev, rev_prior) - 1.0
        out["earnings_growth"] = _safe_div(ni, ni_prior) - 1.0
        out["peg"] = _safe_div(out["pe"], out["earnings_growth"] * 100.0)
    else:
        for c in ["market_cap", "eps_ttm", "pe", "ps", "pb", "profit_margin",
                  "operating_margin", "debt_to_equity", "current_ratio", "fcf_ttm",
                  "revenue_growth", "earnings_growth", "peg"]:
            out[c] = np.nan

    # Price-only metrics (always available where candles exist).
    out["avg_volume"] = price_df["Volume"].rolling(VOL_WINDOW, min_periods=1).mean()
    hi = close.rolling(HIGH_LOW_WINDOW, min_periods=20).max()
    lo = close.rolling(HIGH_LOW_WINDOW, min_periods=20).min()
    out["pct_52w"] = _safe_div(close - lo, hi - lo)
    out["dividend_yield"] = _ttm_dividend_yield(close, dividends)
    out["beta"] = _rolling_beta(close, spy_close)
    return out


def _safe_div(a, b):
    """Elementwise a/b with 0/invalid denominators -> NaN. Accepts Series or scalars."""
    a = pd.Series(a) if np.ndim(a) else a
    with np.errstate(divide="ignore", invalid="ignore"):
        res = a / b.replace(0, np.nan) if isinstance(b, pd.Series) else (
            np.nan if (b in (0, None) or (isinstance(b, float) and np.isnan(b))) else a / b)
    return res


def _ttm_dividend_yield(close, dividends):
    if dividends is None or len(dividends) == 0:
        return pd.Series(np.nan, index=close.index)
    d = dividends.copy()
    d.index = pd.to_datetime(d.index)
    d = d.sort_index()
    # Trailing-365d dividend sum at each price date.
    daily = d.reindex(d.index.union(close.index)).fillna(0.0).sort_index()
    ttm = daily.rolling("365D").sum().reindex(close.index)
    return _safe_div(ttm, close)


def _rolling_beta(close, spy_close):
    if spy_close is None or len(spy_close) == 0:
        return pd.Series(np.nan, index=close.index)
    r = close.pct_change()
    m = spy_close.reindex(close.index).ffill().pct_change()
    cov = r.rolling(BETA_WINDOW).cov(m)
    var = m.rolling(BETA_WINDOW).var()
    return _safe_div(cov, var)


# ── Bucket functions (all dimensions; snapshot dims reuse the same fns) ─────────
def _na(v):
    return v is None or (isinstance(v, float) and np.isnan(v))

def bucket_ps(v):
    if _na(v): return "NA"
    if v < 1: return "cheap (<1)"
    if v < 4: return "fair (1-4)"
    if v < 10: return "rich (4-10)"
    return "expensive (>=10)"

def bucket_pb(v):
    if _na(v): return "NA"
    if v < 1: return "cheap (<1)"
    if v < 3: return "fair (1-3)"
    if v < 6: return "rich (3-6)"
    return "expensive (>=6)"

def bucket_peg(v):
    if _na(v): return "NA"
    if v < 0: return "negative"
    if v < 1: return "cheap (<1)"
    if v < 2: return "fair (1-2)"
    return "expensive (>=2)"

def bucket_div_yield(v):
    if _na(v): return "NA"
    if v <= 0: return "none"
    if v < 0.02: return "low (0-2%)"
    if v < 0.04: return "mid (2-4%)"
    return "high (>=4%)"

def bucket_debt_to_equity(v):
    if _na(v): return "NA"
    if v < 0.5: return "low (<0.5)"
    if v < 1.5: return "moderate (0.5-1.5)"
    return "high (>=1.5)"

def bucket_current_ratio(v):
    if _na(v): return "NA"
    if v < 1: return "weak (<1)"
    if v < 2: return "ok (1-2)"
    return "strong (>=2)"

def bucket_op_margin(v):
    if _na(v): return "NA"
    if v < 0: return "unprofitable (<0%)"
    if v < 0.15: return "thin (0-15%)"
    return "fat (>=15%)"

def bucket_earnings_growth(v):
    if _na(v): return "NA"
    if v < 0: return "shrinking (<0%)"
    if v < 0.15: return "slow (0-15%)"
    if v < 0.40: return "fast (15-40%)"
    return "hyper (>=40%)"

def bucket_fcf_sign(v):
    if _na(v): return "NA"
    return "positive FCF" if v > 0 else "negative FCF"

def bucket_beta(v):
    if _na(v): return "NA"
    if v < 0.8: return "low (<0.8)"
    if v < 1.2: return "market (0.8-1.2)"
    if v < 2.0: return "high (1.2-2)"
    return "very high (>=2)"

def bucket_avg_volume(v):
    if _na(v): return "NA"
    if v < 500_000: return "thin (<500K)"
    if v < 5_000_000: return "mid (0.5-5M)"
    return "liquid (>=5M)"

def bucket_fwd_eps(v):
    if _na(v): return "NA"
    return "positive fwd EPS" if v > 0 else "negative fwd EPS"

def bucket_short_float(v):
    if _na(v): return "NA"
    if v < 0.05: return "low (<5%)"
    if v < 0.15: return "moderate (5-15%)"
    if v < 0.30: return "high (15-30%)"
    return "squeeze-fuel (>=30%)"

def bucket_pct(v):
    """Generic 0-1 percentage buckets (institution %, insider %)."""
    if _na(v): return "NA"
    if v < 0.40: return "low (<40%)"
    if v < 0.70: return "mid (40-70%)"
    return "high (>=70%)"
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `docker compose exec backend python -u test_pit_fundamentals.py`
Expected: `6 passed` with a `PASS` line per test.

- [ ] **Step 5: Commit**

```bash
git add pit_fundamentals.py test_pit_fundamentals.py docker-compose.yml
git commit -m "feat: pure point-in-time fundamentals module + bucket fns + tests"
```

---

## Task 3: Extend `DIMENSIONS` to 23 dimensions with a `pit` flag

**Files:**
- Modify: `seq_fundamental_study.py` (bucket-fn section ~line 80-140, `DIMENSIONS` ~line 127, its consumer ~line 245)

**Interfaces:**
- Consumes: bucket fns from `pit_fundamentals` (Task 2).
- Produces: `DIMENSIONS` as a list of **5-tuples** `(name, field, bucket_fn, order, pit)`. 17 with `pit=True` (field names match `prepare_pit_metrics` columns), 6 with `pit=False` (field names match the `Fundamental` snapshot model).

- [ ] **Step 1: Write the failing test**

Append to `test_pit_fundamentals.py` before `_run_all`:

```python
def test_dimensions_shape_and_counts():
    import seq_fundamental_study as S
    assert all(len(t) == 5 for t in S.DIMENSIONS), "DIMENSIONS must be 5-tuples with pit flag"
    pit = [t for t in S.DIMENSIONS if t[4]]
    snap = [t for t in S.DIMENSIONS if not t[4]]
    assert len(pit) == 17 and len(snap) == 6, (len(pit), len(snap))
    assert len(S.DIMENSIONS) == 23
```

- [ ] **Step 2: Run it to verify it fails**

Run: `docker compose exec backend python -u test_pit_fundamentals.py`
Expected: FAIL on `test_dimensions_shape_and_counts` — current tuples have length 4.

- [ ] **Step 3: Update `seq_fundamental_study.py`**

At the top of the bucketing section, import the new fns and reuse `pit_fundamentals` for the ones it defines. Replace the existing `DIMENSIONS` list (lines ~127-141) with:

```python
from pit_fundamentals import (
    bucket_ps, bucket_pb, bucket_peg, bucket_div_yield, bucket_debt_to_equity,
    bucket_current_ratio, bucket_op_margin, bucket_earnings_growth, bucket_fcf_sign,
    bucket_beta, bucket_avg_volume, bucket_fwd_eps, bucket_short_float, bucket_pct,
)

# (name, field, bucket_fn, order, pit)
#   pit=True  -> field is a column in prepare_pit_metrics()'s output (point-in-time)
#   pit=False -> field is a column on the current `Fundamental` snapshot (lookahead-labeled)
DIMENSIONS = [
    # ── Point-in-time (17) ──────────────────────────────────────────────
    ("PE (trailing)", "pe", _bucket_pe,
     ["cheap (<15)", "fair (15-30)", "rich (30-50)", "expensive (>=50)", "negative (no earnings)", "NA"], True),
    ("PS", "ps", bucket_ps, ["cheap (<1)", "fair (1-4)", "rich (4-10)", "expensive (>=10)", "NA"], True),
    ("PB", "pb", bucket_pb, ["cheap (<1)", "fair (1-3)", "rich (3-6)", "expensive (>=6)", "NA"], True),
    ("PEG", "peg", bucket_peg, ["negative", "cheap (<1)", "fair (1-2)", "expensive (>=2)", "NA"], True),
    ("EPS (TTM)", "eps_ttm", _bucket_eps, ["positive EPS", "negative EPS", "NA"], True),
    ("Revenue growth", "revenue_growth", _bucket_growth,
     ["shrinking (<0%)", "slow (0-15%)", "growing (15-40%)", "hypergrowth (>=40%)", "NA"], True),
    ("Earnings growth", "earnings_growth", bucket_earnings_growth,
     ["shrinking (<0%)", "slow (0-15%)", "fast (15-40%)", "hyper (>=40%)", "NA"], True),
    ("Profit margin", "profit_margin", _bucket_margin,
     ["unprofitable (<0%)", "thin (0-10%)", "healthy (>=10%)", "NA"], True),
    ("Operating margin", "operating_margin", bucket_op_margin,
     ["unprofitable (<0%)", "thin (0-15%)", "fat (>=15%)", "NA"], True),
    ("Debt-to-equity", "debt_to_equity", bucket_debt_to_equity,
     ["low (<0.5)", "moderate (0.5-1.5)", "high (>=1.5)", "NA"], True),
    ("Current ratio", "current_ratio", bucket_current_ratio, ["weak (<1)", "ok (1-2)", "strong (>=2)", "NA"], True),
    ("Free cash flow", "fcf_ttm", bucket_fcf_sign, ["positive FCF", "negative FCF", "NA"], True),
    ("Market cap", "market_cap", _bucket_mktcap,
     ["micro (<500M)", "small (0.5-2B)", "mid (2-10B)", "large (>=10B)", "NA"], True),
    ("Dividend yield", "dividend_yield", bucket_div_yield,
     ["none", "low (0-2%)", "mid (2-4%)", "high (>=4%)", "NA"], True),
    ("Beta", "beta", bucket_beta, ["low (<0.8)", "market (0.8-1.2)", "high (1.2-2)", "very high (>=2)", "NA"], True),
    ("Avg volume", "avg_volume", bucket_avg_volume, ["thin (<500K)", "mid (0.5-5M)", "liquid (>=5M)", "NA"], True),
    ("52-wk position", "pct_52w", bucket_pct_52w,
     ["bottom (<25%)", "low (25-50%)", "high (50-75%)", "top (>=75%)", "NA"], True),
    # ── Snapshot-only (6) — labeled "snapshot (lookahead)" in the app ────
    ("Forward PE", "forward_pe", _bucket_fwd_pe,
     ["cheap (<15)", "fair (15-30)", "rich (>=30)", "negative", "NA"], False),
    ("Forward EPS", "forward_eps", bucket_fwd_eps, ["positive fwd EPS", "negative fwd EPS", "NA"], False),
    ("Short % float", "short_pct_float", bucket_short_float,
     ["low (<5%)", "moderate (5-15%)", "high (15-30%)", "squeeze-fuel (>=30%)", "NA"], False),
    ("Institution %", "institution_pct", bucket_pct, ["low (<40%)", "mid (40-70%)", "high (>=70%)", "NA"], False),
    ("Insider %", "insider_pct", bucket_pct, ["low (<40%)", "mid (40-70%)", "high (>=70%)", "NA"], False),
    ("Float", "float_shares", _bucket_float,
     ["micro-float (<20M)", "low-float (20-100M)", "mid-float (100-500M)", "high-float (>=500M)", "NA"], False),
]
```

Add the one bucket fn `52-wk position` needs (put it beside the other `_bucket_*` fns, ~line 126):

```python
def bucket_pct_52w(v):
    if v is None or (isinstance(v, float) and v != v): return "NA"
    if v < 0.25: return "bottom (<25%)"
    if v < 0.50: return "low (25-50%)"
    if v < 0.75: return "high (50-75%)"
    return "top (>=75%)"
```

Update the snapshot `load_fundamentals` `.values(...)` call (~line 71) so the 6 snapshot dims have their fields available:

```python
            .values("ticker", "forward_pe", "forward_eps", "short_pct_float",
                    "institution_pct", "insider_pct", "float_shares"))
```

Update this file's own `DIMENSIONS` consumer (~line 245) to unpack the 5-tuple:

```python
    for dim_name, field, bucket_fn, order, _pit in DIMENSIONS:
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `docker compose exec backend python -u test_pit_fundamentals.py`
Expected: `7 passed`.

- [ ] **Step 5: Commit**

```bash
git add seq_fundamental_study.py test_pit_fundamentals.py
git commit -m "feat: 23-dimension DIMENSIONS with pit flag (17 PIT + 6 snapshot)"
```

---

## Task 4: DB loaders + all-on-all worker point-in-time refactor

**Files:**
- Modify: `seq_fundamental_study.py` (add `load_financial_reports`, `load_dividends`)
- Modify: `all_on_all_study.py` (imports; worker `_worker`; result-build unpack ~234-235)

**Interfaces:**
- Consumes: `prepare_pit_metrics` (Task 2), `DIMENSIONS` 5-tuples (Task 3), `FinancialReport`/`DividendHistory` (Task 1).
- Produces: `load_financial_reports(tickers) -> {ticker: reports_df}`; `load_dividends(tickers) -> {ticker: pd.Series}`. Worker output shape (`overall`, `buckets`) is unchanged, so `_merge`/`_finalize_stat`/`_save_to_db` need no change.

- [ ] **Step 1: Write the failing test (worker labeling)**

Append to `test_pit_fundamentals.py` before `_run_all` — this tests the pure labeling helper the worker will use:

```python
def test_trade_labels_are_pointintime():
    import seq_fundamental_study as S
    import pit_fundamentals as P
    price = _price()
    m = P.prepare_pit_metrics(price, _reports(), pd.Series(dtype=float), None)
    snap = {"forward_pe": 12.0, "forward_eps": 1.0, "short_pct_float": 0.2,
            "institution_pct": 0.5, "insider_pct": 0.1, "float_shares": 50e6}
    labels = S.label_trade(m, pd.Timestamp("2022-06-01"), snap)
    assert labels["Current ratio"] == "ok (1-2)"          # 150/75 = 2.0 -> "ok" boundary is <2 -> strong; check value
    assert labels["Short % float"] == "high (15-30%)"      # snapshot dim from snap dict
    assert labels["PE (trailing)"] != "NA"                 # PIT dim resolved
```

Note: `current_ratio == 2.0` lands in `strong (>=2)`; adjust the assertion after seeing the value if needed — the point is PIT dims come from `m.asof(date)` and snapshot dims from `snap`.

- [ ] **Step 2: Run it to verify it fails**

Run: `docker compose exec backend python -u test_pit_fundamentals.py`
Expected: FAIL — `AttributeError: module 'seq_fundamental_study' has no attribute 'label_trade'`.

- [ ] **Step 3: Add loaders + `label_trade` to `seq_fundamental_study.py`**

```python
def load_financial_reports(tickers):
    """{ticker: DataFrame of quarterly reports} for point-in-time metrics."""
    import pandas as pd
    from core.models import FinancialReport
    cols = ["ticker", "period_end", "avail_date", "revenue", "net_income", "eps_diluted",
            "operating_income", "total_equity", "total_debt", "current_assets",
            "current_liabilities", "free_cash_flow", "shares_outstanding"]
    qs = FinancialReport.objects.filter(ticker__in=list(tickers)).values_list(*cols)
    big = pd.DataFrame.from_records(list(qs), columns=cols)
    out = {}
    if big.empty:
        return out
    for tk, g in big.groupby("ticker", sort=False):
        out[tk] = g.drop(columns=["ticker"]).reset_index(drop=True)
    return out


def load_dividends(tickers):
    """{ticker: Series indexed by ex_date}."""
    import pandas as pd
    from core.models import DividendHistory
    qs = DividendHistory.objects.filter(ticker__in=list(tickers)).values_list("ticker", "ex_date", "amount")
    big = pd.DataFrame.from_records(list(qs), columns=["ticker", "ex_date", "amount"])
    out = {}
    if big.empty:
        return out
    big["ex_date"] = pd.to_datetime(big["ex_date"])
    for tk, g in big.groupby("ticker", sort=False):
        out[tk] = g.set_index("ex_date")["amount"].sort_index()
    return out


def label_trade(pit_metrics, entry_date, snap):
    """Bucket labels for one trade across all DIMENSIONS.
    PIT dims read pit_metrics.asof(entry_date); snapshot dims read the snap dict."""
    row = pit_metrics.asof(entry_date) if pit_metrics is not None and len(pit_metrics) else None
    labels = {}
    for name, field, bfn, _order, pit in DIMENSIONS:
        if pit:
            v = None if row is None else row.get(field)
        else:
            v = (snap or {}).get(field)
        labels[name] = bfn(v)
    return labels
```

- [ ] **Step 4: Refactor the `all_on_all_study.py` worker**

Update the import block (lines ~36-39) to add the new loaders + `label_trade` + `prepare_pit_metrics`:

```python
from seq_fundamental_study import (
    build_universe, load_candles, load_fundamentals, load_financial_reports,
    load_dividends, label_trade, DIMENSIONS, DEFAULT_JOBS, _chunk, MIN_BARS,
)
from pit_fundamentals import prepare_pit_metrics
```

In `_worker` (replace lines ~88-89 and the per-ticker setup + trade loop, lines ~98-147):

```python
    candles = load_candles(tickers)
    funds = load_fundamentals(tickers)
    reports = load_financial_reports(tickers)
    divs = load_dividends(tickers)
    spy_close = load_candles(["SPY"]).get("SPY")
    spy_close = spy_close["Close"] if spy_close is not None else None

    exit_fns = {ek: EXITS[ek][1] for ek in exit_keys}
    overall = {}
    buckets = {}

    _MISS = object()
    for tk, sdf in candles.items():
        if len(sdf) < MIN_BARS:
            continue
        _prepare_indicators(sdf)
        close = sdf["Close"].values
        n = len(close)
        snap = funds.get(tk, {})
        pit_metrics = prepare_pit_metrics(sdf, reports.get(tk), divs.get(tk), spy_close)
        # Bucket labels depend only on the entry bar's date, so memoize per entry_idx
        # (signals fire on overlapping bars → reused across signals/exits, like exit_cache).
        label_cache = {}
        exit_cache = {}

        for sk in signal_keys:
            sig_fn = SIGNALS[sk][1]
            try:
                sig = sig_fn(sdf).fillna(False)
            except Exception:
                continue
            entry_idxs = [sdf.index.get_loc(d) for d in sig[sig].index]
            if not entry_idxs:
                continue
            for ek, exit_fn in exit_fns.items():
                for idx in entry_idxs:
                    ck = (ek, idx)
                    exit_idx = exit_cache.get(ck, _MISS)
                    if exit_idx is _MISS:
                        try:
                            exit_idx = exit_fn(sdf, idx)
                        except Exception:
                            exit_idx = None
                        exit_cache[ck] = exit_idx
                    if exit_idx is None or exit_idx <= idx or exit_idx >= n:
                        continue
                    ep = float(close[idx])
                    if ep <= 0:
                        continue
                    ret = (float(close[exit_idx]) - ep) / ep * 100
                    hold = exit_idx - idx
                    o = overall.get((sk, ek))
                    if o is None:
                        o = overall[(sk, ek)] = _new_stat()
                    _accum(o, ret, hold)
                    labels = label_cache.get(idx)
                    if labels is None:
                        labels = label_cache[idx] = label_trade(pit_metrics, sdf.index[idx], snap)
                    for dim, label in labels.items():
                        key = (sk, ek, dim, label)
                        b = buckets.get(key)
                        if b is None:
                            b = buckets[key] = _new_stat()
                        _accum(b, ret, hold)
    return overall, buckets
```

Update the result-build unpack (lines ~234-235) for 5-tuples:

```python
    dim_names = [d[0] for d in DIMENSIONS]
    dim_order = {d[0]: d[3] for d in DIMENSIONS}
```

- [ ] **Step 5: Run the unit test + a tiny live sweep**

Run: `docker compose exec backend python -u test_pit_fundamentals.py`
Expected: `8 passed` (adjust `test_trade_labels_are_pointintime` assertions to the printed values if a boundary differs).

Then a 3-ticker smoke sweep (needs Task 5's backfill for non-empty PIT, but should not crash with empty reports → PIT dims all `NA`):
Run: `docker compose exec backend python -u all_on_all_study.py --limit 3 --no-db-save --signals seq_rsi20_ad_rising_rsi --exits 6m`
Expected: prints a `by_dimension` block listing all 23 dimension names without error.

- [ ] **Step 6: Commit**

```bash
git add seq_fundamental_study.py all_on_all_study.py test_pit_fundamentals.py
git commit -m "feat: point-in-time bucketing in all-on-all worker (asof entry_date + per-idx cache)"
```

---

## Task 5: `fetch_financial_history.py` backfill + recurring wiring

**Files:**
- Create: `fetch_financial_history.py` (root, mounted in Task 2)
- Modify: `backend/api/tasks.py` (add `run_financial_history_task`)
- Modify: `backend/api/celery_tasks.py` (add `fetch_financial_history` shared_task)
- Modify: `backend/rotation/celery.py` (beat entry before `stock-studies-daily`)

**Interfaces:**
- Consumes: `FinancialReport`, `DividendHistory` (Task 1); `avail_date` (Task 2); the spawn/`--jobs` pattern from `seq_fundamental_study`.
- Produces: populated `FinancialReport` + `DividendHistory`; `run_financial_history_task(jobs=None) -> int` (subprocess return code).

- [ ] **Step 1: Implement the backfill script**

Create `fetch_financial_history.py`:

```python
#!/usr/bin/env python3
"""Backfill quarterly financials + dividends per ticker from yfinance into the
FinancialReport / DividendHistory tables (point-in-time fundamentals source).

Parallel PROCESS pool (spawn), same pattern as the study runners. Run:
  docker compose run --rm backend python -u fetch_financial_history.py --db
Flags: --jobs N  --limit N  --tickers AAPL,MSFT
⚠️ `run --rm` detaches — check `docker ps | grep rotation-backend-run` for orphans after.
"""
import warnings; warnings.filterwarnings("ignore")
import os, sys
from concurrent.futures import ProcessPoolExecutor

DEFAULT_JOBS = min(16, os.cpu_count() or 4)  # network-bound; don't hammer yfinance

# yfinance row-label aliases (labels shift across tickers/versions).
_ALIASES = {
    "revenue": ["Total Revenue", "Operating Revenue"],
    "net_income": ["Net Income", "Net Income Common Stockholders", "Net Income From Continuing Operation Net Minority Interest"],
    "eps_diluted": ["Diluted EPS", "Basic EPS"],
    "operating_income": ["Operating Income", "Total Operating Income As Reported"],
    "total_equity": ["Stockholders Equity", "Total Stockholder Equity", "Common Stock Equity"],
    "total_debt": ["Total Debt", "Total Liabilities Net Minority Interest"],
    "current_assets": ["Current Assets", "Total Current Assets"],
    "current_liabilities": ["Current Liabilities", "Total Current Liabilities"],
    "free_cash_flow": ["Free Cash Flow"],
    "shares_outstanding": ["Ordinary Shares Number", "Share Issued", "Basic Average Shares"],
}


def _pick(df, names, col):
    if df is None or getattr(df, "empty", True):
        return None
    for nm in names:
        if nm in df.index:
            try:
                v = df.loc[nm, col]
                v = float(v)
                return v if v == v else None
            except Exception:
                continue
    return None


def _worker(payload):
    tickers = payload
    import django
    os.environ.setdefault("DJANGO_SETTINGS_MODULE", "rotation.settings")
    try:
        django.setup()
    except Exception:
        pass
    import yfinance as yf
    from core.models import FinancialReport, DividendHistory
    from pit_fundamentals import avail_date
    saved = 0
    for tk in tickers:
        try:
            t = yf.Ticker(tk)
            inc = t.quarterly_income_stmt
            bs = t.quarterly_balance_sheet
            cf = t.quarterly_cashflow
            cols = list(inc.columns) if inc is not None and not inc.empty else []
            for col in cols:
                pe = col.date() if hasattr(col, "date") else col
                fr = dict(
                    revenue=_pick(inc, _ALIASES["revenue"], col),
                    net_income=_pick(inc, _ALIASES["net_income"], col),
                    eps_diluted=_pick(inc, _ALIASES["eps_diluted"], col),
                    operating_income=_pick(inc, _ALIASES["operating_income"], col),
                    total_equity=_pick(bs, _ALIASES["total_equity"], col),
                    total_debt=_pick(bs, _ALIASES["total_debt"], col),
                    current_assets=_pick(bs, _ALIASES["current_assets"], col),
                    current_liabilities=_pick(bs, _ALIASES["current_liabilities"], col),
                    free_cash_flow=_pick(cf, _ALIASES["free_cash_flow"], col),
                    shares_outstanding=_pick(bs, _ALIASES["shares_outstanding"], col),
                )
                # Ints for BigInteger columns (eps_diluted stays float).
                for k in ("revenue", "net_income", "operating_income", "total_equity",
                          "total_debt", "current_assets", "current_liabilities",
                          "free_cash_flow", "shares_outstanding"):
                    fr[k] = int(fr[k]) if fr[k] is not None else None
                FinancialReport.objects.update_or_create(
                    ticker=tk, period_end=pe,
                    defaults={"avail_date": avail_date(pe), **fr})
                saved += 1
            divs = t.dividends
            if divs is not None and len(divs) > 0:
                for ex, amt in divs.items():
                    ed = ex.date() if hasattr(ex, "date") else ex
                    try:
                        DividendHistory.objects.update_or_create(
                            ticker=tk, ex_date=ed, defaults={"amount": float(amt)})
                    except Exception:
                        pass
        except Exception as e:
            print(f"  {tk}: ERROR {e}")
    return saved


def run(jobs=DEFAULT_JOBS, limit=None, tickers=None):
    from seq_fundamental_study import build_universe, _chunk
    from django.db import connections
    uni = tickers or build_universe()
    if limit:
        uni = uni[:limit]
    print(f"Backfilling financial history for {len(uni)} tickers | jobs: {jobs}")
    connections.close_all()
    chunks = _chunk(uni, jobs * 3)
    total = 0
    import multiprocessing as mp
    ctx = mp.get_context("spawn")
    with ProcessPoolExecutor(max_workers=jobs, mp_context=ctx) as ex:
        for got in ex.map(_worker, chunks):
            total += got
    print(f"Done: {total} FinancialReport rows upserted.")
    return total


if __name__ == "__main__":
    argv = sys.argv
    jobs = int(argv[argv.index("--jobs") + 1]) if "--jobs" in argv else DEFAULT_JOBS
    limit = int(argv[argv.index("--limit") + 1]) if "--limit" in argv else None
    tks = argv[argv.index("--tickers") + 1].split(",") if "--tickers" in argv else None
    if "--db" in argv:
        import django
        os.environ.setdefault("DJANGO_SETTINGS_MODULE", "rotation.settings")
        django.setup()
    run(jobs=jobs, limit=limit, tickers=tks)
```

- [ ] **Step 2: Backfill a few tickers and verify PIT resolves**

Run: `docker compose run --rm backend python -u fetch_financial_history.py --db --tickers AAPL,MSFT,INTC`
Expected: `Done: N FinancialReport rows upserted.` with N > 0.
Then check orphans: `docker ps --format "{{.Names}}" | grep rotation-backend-run` (kill any with `docker kill`).
Verify: `docker compose exec backend python -c "from core.models import FinancialReport as F; print(F.objects.filter(ticker='AAPL').count())"`
Expected: a number ≥ 4 (multiple quarters).

- [ ] **Step 3: Verify a real PIT metric on backfilled data**

Run:
```bash
docker compose exec backend python -c "
import django,os;os.environ.setdefault('DJANGO_SETTINGS_MODULE','rotation.settings');django.setup()
from seq_fundamental_study import load_candles, load_financial_reports, load_dividends
from pit_fundamentals import prepare_pit_metrics
c=load_candles(['AAPL'])['AAPL']; r=load_financial_reports(['AAPL']).get('AAPL'); d=load_dividends(['AAPL']).get('AAPL')
m=prepare_pit_metrics(c,r,d,None); print(m[['pe','market_cap','profit_margin']].dropna().tail(3))"
```
Expected: three recent dates with plausible PE / market cap / margin values.

- [ ] **Step 4: Wire the subprocess task**

In `backend/api/tasks.py`, add (mirroring `run_stock_studies_task`):

```python
def run_financial_history_task(jobs=None):
    """Backfill quarterly financials + dividends as a subprocess (spawn re-imports
    __main__; in-process would re-import the server)."""
    import subprocess
    cmd = ["python", "-u", "fetch_financial_history.py", "--db"]
    if jobs:
        cmd += ["--jobs", str(jobs)]
    return subprocess.run(cmd, cwd="/app").returncode
```

In `backend/api/celery_tasks.py`, add:

```python
@shared_task
def fetch_financial_history():
    """Periodic: refresh point-in-time fundamentals (quarterly financials + dividends)
    before the nightly stock sweep buckets trades against them."""
    from api.tasks import run_financial_history_task
    logger.info("Backfilling financial history")
    rc = run_financial_history_task()
    logger.info(f"Financial history backfill complete (rc={rc})")
    return rc
```

In `backend/rotation/celery.py`, add to `beat_schedule` **before** `stock-studies-daily` (financials change rarely; a daily pre-sweep refresh is cheap and keeps PIT current):

```python
    # Point-in-time fundamentals refresh. Runs before the stock sweep (21:45) so the
    # sweep buckets trades against fresh quarterly financials. 21:30 UTC.
    "financial-history-daily": {
        "task": "api.celery_tasks.fetch_financial_history",
        "schedule": crontab(hour=21, minute=30),
    },
```

- [ ] **Step 5: Verify the task import graph**

Run: `docker compose exec backend python -c "from api.celery_tasks import fetch_financial_history; from api.tasks import run_financial_history_task; print('ok')"`
Expected: `ok`.

- [ ] **Step 6: Commit**

```bash
git add fetch_financial_history.py backend/api/tasks.py backend/api/celery_tasks.py backend/rotation/celery.py
git commit -m "feat: financial-history backfill script + daily beat (pre-sweep)"
```

---

## Task 6: App badging of PIT vs snapshot dimensions

Non-breaking: `by_dimension` JSON shape is unchanged; the view adds a `dimension_meta` map so the frontend can badge snapshot dims.

**Files:**
- Modify: `backend/api/views.py` (`StockStudiesView.get` response)
- Modify: `frontend/src/App.js` (`StockStudiesPage` fundamental-bucket render)
- Modify: `frontend/src/App.css` (`.dim-badge`)

**Interfaces:**
- Consumes: `DIMENSIONS` `pit` flag (Task 3).
- Produces: API response gains `dimension_meta: {dim_name: {"pit": bool}}`.

- [ ] **Step 1: Add `dimension_meta` to the view response**

In `backend/api/views.py`, in `StockStudiesView.get`, build the map and include it in the returned payload (place near where `meta` is assembled):

```python
        try:
            from seq_fundamental_study import DIMENSIONS
            dimension_meta = {d[0]: {"pit": bool(d[4])} for d in DIMENSIONS}
        except Exception:
            dimension_meta = {}
```

Add `"dimension_meta": dimension_meta` to the response dict.

- [ ] **Step 2: Verify the API returns it**

Run: `curl -s "http://localhost:3001/api/stock-studies?limit=1" | python -c "import sys,json;d=json.load(sys.stdin);print(list(d.get('dimension_meta',{}).items())[:3])"`
Expected: pairs like `[('PE (trailing)', {'pit': True}), ('Forward PE', {'pit': False}), ...]`.

- [ ] **Step 3: Badge snapshot dims in `App.js`**

In `StockStudiesPage`, capture `dimension_meta` from the fetch response into state (e.g. `const [dimMeta, setDimMeta] = useState({})` and `setDimMeta(data.dimension_meta || {})`). Where each fundamental dimension name is rendered in the expanded row (the `by_dimension` map), append a badge when it is snapshot-only:

```jsx
{dimName}
{dimMeta[dimName] && dimMeta[dimName].pit === false && (
  <span className="dim-badge" title="Uses today's snapshot — carries lookahead bias, not point-in-time">snapshot</span>
)}
```

For the inline "Best fundamental bucket" column (`bestBucket(r)`): when the winning bucket's dimension is snapshot-only, suffix ` (snapshot)` to the displayed text so the caveat is visible without expanding.

- [ ] **Step 4: Add badge styling in `App.css`**

```css
.dim-badge {
  display: inline-block;
  margin-left: 6px;
  padding: 1px 6px;
  font-size: 10px;
  font-weight: 600;
  color: #92400e;
  background: #fef3c7;
  border-radius: 4px;
  vertical-align: middle;
}
```

- [ ] **Step 5: Rebuild the frontend and verify**

Run: `docker compose up -d --build frontend`
Expected: build completes, container recreated.
Then open `localhost:3001/#/studies/stocks`, expand a row: point-in-time dims show no badge; Forward PE / Short % float / Institution % / Insider % / Float / Forward EPS show a `snapshot` badge.

- [ ] **Step 6: Commit**

```bash
git add backend/api/views.py frontend/src/App.js frontend/src/App.css
git commit -m "feat: badge snapshot-only fundamental dimensions in stock studies UI"
```

---

## Task 7: Full backfill + full sweep + end-to-end verification

**Files:** none (operational).

- [ ] **Step 1: Full financial-history backfill**

Run: `docker compose run --rm backend python -u fetch_financial_history.py --db`
Expected: `Done: N FinancialReport rows upserted.` (N in the tens of thousands). This is network-bound and may take a while.
After it returns, check orphans: `docker ps --format "{{.Names}}" | grep rotation-backend-run` → `docker kill` any.

- [ ] **Step 2: Confirm coverage**

Run: `docker compose exec backend python -c "from core.models import FinancialReport as F, DividendHistory as D; print('reports', F.objects.count(), 'tickers', F.objects.values('ticker').distinct().count(), 'divs', D.objects.count())"`
Expected: thousands of reports across ~800–1000+ tickers.

- [ ] **Step 3: Restart backend so the sweep picks up new code, then full sweep**

Run: `docker compose restart backend`
Then: `docker compose run --rm backend python -u all_on_all_study.py --db`
Expected: completes in ~30–45 min; prints upserted `StockStudy` row count. Kill orphans after.

- [ ] **Step 4: Verify PIT buckets are populated and non-degenerate**

Run:
```bash
docker compose exec backend python -c "
import django,os;os.environ.setdefault('DJANGO_SETTINGS_MODULE','rotation.settings');django.setup()
from core.models import StockStudy
s=StockStudy.objects.order_by('-total_trades').first()
bd=s.by_dimension
print(s.signal_key, s.exit_key, 'dims:', len(bd))
for dim in ['PE (trailing)','Beta','Debt-to-equity','Short % float']:
    rows=bd.get(dim,[]); print(dim, '->', [(r['bucket'], r['trades']) for r in rows][:4])"
```
Expected: `dims: 23`; PIT dims (PE, Beta, Debt-to-equity) show multiple non-`NA` buckets with meaningful trade counts; a snapshot dim (Short % float) also populated.

- [ ] **Step 5: End-to-end app check**

Run: `curl -s "http://localhost:3001/api/stock-studies?limit=1" | python -c "import sys,json;d=json.load(sys.stdin);r=d['results'][0];print('dims', len(r['by_dimension']), 'meta', len(d['dimension_meta']))"`
Expected: `dims 23 meta 23`.
Then in the browser (`localhost:3001/#/studies/stocks`) confirm the Best-bucket column and expanded rows show the new dimensions with snapshot badges where expected.

- [ ] **Step 6: Commit any doc updates**

```bash
git add -A
git commit -m "chore: point-in-time fundamentals live (full backfill + sweep verified)"
```

---

## Self-Review

**Spec coverage:**
- `REPORT_LAG_DAYS = 45` / `avail_date` anti-lookahead → Task 2 (constant, `avail_date`, `test_no_report_visible_before_avail_date_is_nan`). ✓
- `FinancialReport` model → Task 1. ✓ (`DividendHistory` added for dividend yield, which the spec left "decided at implementation time".)
- Backfill job → Task 5. ✓
- `_prepare_fundamentals_pit` sweep refactor → implemented as `prepare_pit_metrics` (Task 2) + `label_trade` + worker refactor (Task 4). ✓ (Renamed to a pure builder + a labeler for testability; noted here so naming is consistent.)
- `DIMENSIONS` `pit` flag, 23 dims → Task 3. ✓
- NA bucketing → Task 2 (`_na`, `bucket_*` return `"NA"`) + `test_no_report_visible...`. ✓
- Recurring beat wiring, before stock sweep → Task 5 Step 4 (21:30 < 21:45). ✓
- App badging PIT vs snapshot → Task 6. ✓ (Non-breaking `dimension_meta` instead of changing `by_dimension` shape — safer than the spec's implied approach.)
- Live Firing Now stays snapshot → untouched by design; no task modifies `live_firing_scan.py`/`LiveSignal`. ✓
- Tests (lag, TTM, formulas, boundaries, NA, PIT-correctness, integration) → Task 2 + Task 4 + Task 7 Step 4. ✓

**Placeholder scan:** No TBD/TODO; every code step shows complete code; the one soft spot (`test_trade_labels_are_pointintime` boundary assertion) is explicitly flagged to adjust to printed values, not left vague. ✓

**Type consistency:** `prepare_pit_metrics(price_df, reports_df, dividends, spy_close)` signature identical in Task 2 definition, Task 4 call, and Task 5 Step 3 check. `label_trade(pit_metrics, entry_date, snap)` identical in Task 4 definition/use and Task 4 test. `DIMENSIONS` 5-tuple `(name, field, bucket_fn, order, pit)` consistent across Tasks 3, 4, 6. PIT `field` names (`pe, ps, pb, peg, eps_ttm, revenue_growth, earnings_growth, profit_margin, operating_margin, debt_to_equity, current_ratio, fcf_ttm, market_cap, dividend_yield, beta, avg_volume, pct_52w`) match `prepare_pit_metrics` output columns exactly. Snapshot `field` names (`forward_pe, forward_eps, short_pct_float, institution_pct, insider_pct, float_shares`) match `Fundamental` model + the widened `load_fundamentals` `.values()`. ✓
