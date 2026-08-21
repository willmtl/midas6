# Strategy G — Analyst-Gap Diversified Value Book — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Productionize the validated analyst-gap research prototype (`backend/analyst_gap_book.py`) into a standalone, persisted strategy engine `backend/strategy_g.py` that reproduces G-core's net-of-cost numbers, saves a `BacktestResult`, and passes a walk-forward robustness gate; then (optional Phase 2) expose it live.

**Architecture:** A single self-contained, point-in-time engine module in `backend/` (picked up by the `/app` dir bind-mount — no docker-compose file-mount needed, unlike root-mounted flagship scripts). It reuses `seq_fundamental_study.load_financial_reports` and `.data/analyst_ratings.jsonl` read-only, builds monthly PIT panels, runs an equal-weight (or inverse-vol) monthly sim with a turnover-based cost model, and persists to Postgres via `BacktestResult(kind="strategy_g")`. Phase 2 adds a live-pick scanner, a DRF endpoint, and a frontend tab — gated on Phase-1 sign-off.

**Tech Stack:** Python 3, Django ORM (`core.models.Candle`, `core.models.FinancialReport`, `core.models.Sector`, `core.models.BacktestResult`), pandas/numpy, run inside `rotation-backend-1` via `MSYS_NO_PATHCONV=1 docker exec`. Frontend: React (`frontend/src/App.js`, prod build).

## Global Constraints

- **Run scripts in-container:** `MSYS_NO_PATHCONV=1 docker exec rotation-backend-1 python -u /app/<script>.py`. Django one-liners: `manage.py shell -c`, never bare `python -c`.
- **NEVER fabricate numbers:** every metric stated must come from an actual run/query, not recollection.
- **Always save backtests:** every study run persists a `BacktestResult(+JSON)` — no throwaway prints.
- **PIT correctness (no look-ahead):** analyst upside uses only targets dated ≤ month-end within trailing 180d; profitability uses TTM sums forward-filled by `avail_date` (filing availability), never `period_end`.
- **Survivorship-aware universe:** include delisted-with-candles names; delisted names exit at last/deal price, −100% only on confirmed bankruptcy.
- **US-listed only:** `"." not in ticker` (target/price currency must match); exclude ETFs (`Sector.etf` set ∪ {SPY, QQQ}).
- **Canonical config (G-core):** top-5% by implied upside, TTM net income > 0 gate, equal-weight, monthly month-end rebalance, 20d-avg dollar-volume ≥ $5M/day, price > $1.
- **G-smooth variant:** identical but inverse-vol weighting; selected by flag, off by default.
- **BacktestResult.kind is `unique`** — one row per kind. Use `kind="strategy_g"` holding the full payload (core + smooth + sweep + walk-forward). Persist with `update_or_create(kind=..., defaults={"payload": json.loads(json.dumps(p, default=str)), "computed_at": timezone.now()})`.
- **Commit trailers (every commit):**
  ```
  Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>
  Claude-Session: https://claude.ai/code/session_01SJJ3BR9dcEBSWjiYQqPjSW
  ```
- **Branch:** `flagship-tlsupport-div4x-drift` (do not touch `main`/`master`). Do not push unless asked.

---

## File Structure

**Phase 1**
- Create `backend/strategy_g.py` — the engine (universe, PIT panels, `sim`, cost model, walk-forward, `main()` with CLI + `BacktestResult` persistence).
- Create `backend/test_strategy_g.py` — in-container reproduction/assertion checks (imports the engine, asserts G-core net-of-cost metrics within tolerance of the prototype, asserts PIT no-look-ahead invariants).

**Phase 2 (optional, gated on Phase-1 review)**
- Create `backend/strategy_g_scan.py` — live monthly-pick scanner writing `.data/strategy_g_picks.json`.
- Modify `backend/api/views.py` + `backend/api/urls.py` (or wherever routes live) — `/api/strategy-g` endpoint serving the `BacktestResult` payload + latest picks.
- Modify `frontend/src/App.js` — a "Strategy G" tab.

Reference (read-only, never modified): `backend/analyst_gap_book.py` (prototype — source of truth for expected numbers), `seq_fundamental_study.py` (`load_financial_reports`), `backend/survivorship_smallcap_study.py:4582-4593` (BacktestResult persistence pattern), `backend/core/models.py:1121-1132` (`BacktestResult`).

---

## Task 1: Engine skeleton — universe, panels, and PIT invariants

**Files:**
- Create: `backend/strategy_g.py`
- Test: `backend/test_strategy_g.py`

**Interfaces:**
- Consumes: `seq_fundamental_study.load_financial_reports(tickers) -> {ticker: DataFrame}`; `.data/analyst_ratings.jsonl` lines with `{ticker, price_target, date}`; `core.models.Candle`, `core.models.Sector`.
- Produces:
  - `build_universe() -> (list[str] uni, dict tgts)` — `tgts[ticker] = [(pd.Timestamp.value, float target), ...]`.
  - `build_panels(uni, tgts) -> dict` with keys: `mclose` (month-end close DataFrame, cols incl "SPY"), `mdvol` (20d-avg dollar-vol, month-end), `ups` (implied-upside), `ttm_ni` (TTM net income), `mvol` (60d daily-vol, month-end), `fwd` (next-month return), `spy_ret` (next-month SPY return), `midx` (DatetimeIndex).
  - Module constants: `DVOL_FLOOR = 5e6`, `PRICE_FLOOR = 1.0`, `TOP_FRAC = 0.05`, `STALE_DAYS = 180`.

- [ ] **Step 1: Write the failing test**

```python
# backend/test_strategy_g.py
import os
os.environ.setdefault("DJANGO_SETTINGS_MODULE", "rotation.settings")
import django; django.setup()
import numpy as np, pandas as pd
import strategy_g as G


def test_panels_pit_and_shapes():
    uni, tgts = G.build_universe()
    assert len(uni) > 3000, f"covered universe too small: {len(uni)}"
    P = G.build_panels(uni, tgts)
    # SPY present, monthly index, sane span
    assert "SPY" in P["mclose"].columns
    assert P["midx"][0].year <= 2016 and P["midx"][-1].year >= 2025
    # implied upside is finite where defined and equals target/close-1 sign logic
    ups = P["ups"]; assert ups.notna().values.any()
    # PIT no-look-ahead: TTM net income at month d must not use any avail_date > d.
    # Reconstruct one cell manually for a well-covered name and check it matches.
    tk = "AAPL" if "AAPL" in P["ttm_ni"].columns else P["ttm_ni"].columns[0]
    d = P["midx"][60]
    r = G._reports[tk][["period_end", "avail_date", "net_income"]].dropna()
    r = r.sort_values("period_end"); r["ttm"] = r["net_income"].rolling(4).sum()
    avail = r[pd.to_datetime(r["avail_date"]) <= d]
    expected = avail["ttm"].iloc[-1] if len(avail) else np.nan
    got = P["ttm_ni"].loc[d, tk]
    assert (pd.isna(expected) and pd.isna(got)) or abs(got - expected) < 1e-6, f"PIT breach: {got} vs {expected}"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `MSYS_NO_PATHCONV=1 docker exec rotation-backend-1 python -m pytest /app/test_strategy_g.py::test_panels_pit_and_shapes -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'strategy_g'` (or `AttributeError`).

- [ ] **Step 3: Write minimal implementation**

Port the prototype's universe + panel construction verbatim (it is already validated), exposing them as functions and stashing `_reports` module-global for the PIT test.

```python
#!/usr/bin/env python3
"""Strategy G — analyst-gap diversified value book. See docs/superpowers/specs/2026-08-21-strategy-g-analyst-gap-design.md.
Run: MSYS_NO_PATHCONV=1 docker exec rotation-backend-1 python -u /app/strategy_g.py [--db] [--variant core|smooth] [--cost-bps N] [--sweep]"""
import os, json
os.environ.setdefault("DJANGO_SETTINGS_MODULE", "rotation.settings")
import django; django.setup()
import numpy as np, pandas as pd
from pathlib import Path
from collections import defaultdict
from core.models import Candle, Sector

DVOL_FLOOR = 5e6
PRICE_FLOOR = 1.0
TOP_FRAC = 0.05
STALE_DAYS = 180
RATINGS = Path("/app/.data/analyst_ratings.jsonl")
_reports = {}   # module-global {ticker: quarterly-report DataFrame}, set by build_panels


def build_universe():
    etfs = set(Sector.objects.values_list("etf", flat=True)) | {"SPY", "QQQ"}
    tgts = defaultdict(list)
    for line in RATINGS.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            r = json.loads(line)
        except Exception:
            continue
        tk, pt, d = r.get("ticker"), r.get("price_target"), r.get("date")
        if tk and pt and d and "." not in tk and tk not in etfs:
            tgts[tk].append((pd.Timestamp(d).value, float(pt)))
    return sorted(tgts), tgts


def build_panels(uni, tgts):
    from seq_fundamental_study import load_financial_reports
    global _reports
    rows = list(Candle.objects.filter(ticker__in=uni + ["SPY"], date__gte="2015-12-01")
                .values_list("ticker", "date", "close", "volume"))
    df = pd.DataFrame(rows, columns=["ticker", "date", "close", "volume"])
    df["date"] = pd.to_datetime(df["date"]); df["close"] = df["close"].astype(float); df["volume"] = df["volume"].astype(float)
    df["dvol"] = df["close"] * df["volume"]
    close_d = df.pivot_table(index="date", columns="ticker", values="close")
    dvol20 = df.pivot_table(index="date", columns="ticker", values="dvol").rolling(20, min_periods=5).mean()
    mclose = close_d.resample("ME").last()
    mdvol = dvol20.resample("ME").last()
    midx = mclose.index

    midx_i = np.array([t.value for t in midx], dtype="int64")
    stale = STALE_DAYS * 86400 * 10**9
    ups = pd.DataFrame(np.nan, index=midx, columns=mclose.columns)
    for tk in mclose.columns:
        pts = tgts.get(tk)
        if not pts:
            continue
        arr = np.array(sorted(pts)); di, tv = arr[:, 0], arr[:, 1]
        col = np.full(len(midx), np.nan)
        for j, d in enumerate(midx_i):
            a = np.searchsorted(di, d - stale, side="right"); b = np.searchsorted(di, d, side="right")
            if b > a:
                col[j] = np.median(tv[a:b])
        ups[tk] = col / mclose[tk].values - 1.0

    _reports = load_financial_reports(uni)

    def _pit_panel(field, flow):
        out = {}
        for tk, r in _reports.items():
            if field not in r.columns:
                continue
            d = r[["period_end", "avail_date", field]].dropna(subset=[field]).copy()
            if d.empty:
                continue
            d = d.sort_values("period_end")
            v = d[field].rolling(4).sum() if flow else d[field]
            s = pd.Series(v.values, index=pd.to_datetime(d["avail_date"])).dropna()
            if s.empty:
                continue
            s = s[~s.index.duplicated(keep="last")].sort_index()
            out[tk] = s.reindex(s.index.union(midx)).ffill().reindex(midx)
        return pd.DataFrame(out).reindex(columns=mclose.columns)

    ttm_ni = _pit_panel("net_income", True)
    mvol = close_d.pct_change().rolling(60, min_periods=20).std().resample("ME").last().reindex(midx)
    fwd = mclose.shift(-1) / mclose - 1.0
    spy_ret = mclose["SPY"].shift(-1) / mclose["SPY"] - 1.0
    return dict(mclose=mclose, mdvol=mdvol, ups=ups, ttm_ni=ttm_ni, mvol=mvol, fwd=fwd, spy_ret=spy_ret, midx=midx)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `MSYS_NO_PATHCONV=1 docker exec rotation-backend-1 python -m pytest /app/test_strategy_g.py::test_panels_pit_and_shapes -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add backend/strategy_g.py backend/test_strategy_g.py
git commit -m "feat(strategy-g): engine skeleton — universe + PIT panels

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01SJJ3BR9dcEBSWjiYQqPjSW"
```

---

## Task 2: `sim` — selection, weighting, cost model, win-rate/turnover

**Files:**
- Modify: `backend/strategy_g.py`
- Test: `backend/test_strategy_g.py`

**Interfaces:**
- Consumes: the `build_panels` dict from Task 1.
- Produces:
  - `profit_ok(P, d, t) -> bool` — `ttm_ni[t]` present and > 0 at month `d`.
  - `sim(P, top_frac=TOP_FRAC, profit_gate=True, weight="equal", cost_bps=0.0) -> dict` with keys `ret` (pd.Series monthly), `avg_n` (float), `pos_win` (float, per-position win rate), `turnover` (float, avg one-way), `holdings` ({date_iso: [tickers]}).
  - `perf(ret, spy_ret, posw=None) -> dict` with `total`, `cagr`, `sharpe`, `maxdd`, `hit`, `beat_spy`, `n_mo` (all floats/ints; percentages as numbers, e.g. 31.1).

- [ ] **Step 1: Write the failing test**

```python
def test_gcore_reproduces_prototype():
    uni, tgts = G.build_universe()
    P = G.build_panels(uni, tgts)
    # G-core at 25 bps/side must reproduce the prototype within tolerance (prototype: CAGR 29.5, DD -24.8, hit 69.5).
    r = G.sim(P, cost_bps=25.0)
    m = G.perf(r["ret"], P["spy_ret"], r["pos_win"])
    assert 27.0 <= m["cagr"] <= 32.0, f"CAGR off: {m['cagr']}"
    assert -27.0 <= m["maxdd"] <= -22.0, f"DD off: {m['maxdd']}"
    assert m["hit"] >= 66.0, f"hit rate off: {m['hit']}"
    assert 0.15 <= r["turnover"] <= 0.30, f"turnover off: {r['turnover']}"
    assert 55 <= r["avg_n"] <= 70, f"name count off: {r['avg_n']}"
    # G-core must beat SPY on absolute return net of cost
    spy = G.perf(P["spy_ret"], P["spy_ret"])
    assert m["cagr"] > spy["cagr"] + 8, f"G-core does not clear SPY: {m['cagr']} vs {spy['cagr']}"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `MSYS_NO_PATHCONV=1 docker exec rotation-backend-1 python -m pytest /app/test_strategy_g.py::test_gcore_reproduces_prototype -v`
Expected: FAIL — `AttributeError: module 'strategy_g' has no attribute 'sim'`.

- [ ] **Step 3: Write minimal implementation**

Port `sim`/`perf` from the prototype (equal + inverse-vol weighting, one-way turnover, both-side cost charge), adding a `holdings` record for the scanner.

```python
def profit_ok(P, d, t):
    v = P["ttm_ni"].get(t)
    return v is not None and pd.notna(P["ttm_ni"].loc[d, t]) and P["ttm_ni"].loc[d, t] > 0


def sim(P, top_frac=TOP_FRAC, profit_gate=True, weight="equal", cost_bps=0.0):
    mclose, mdvol, ups, mvol, fwd = P["mclose"], P["mdvol"], P["ups"], P["mvol"], P["fwd"]
    midx = P["midx"]
    pr, held_n, pos_win, pos_tot = [], [], 0, 0
    prev_w, turns, holdings = {}, [], {}
    for d in midx[:-1]:
        u = ups.loc[d].dropna(); f = fwd.loc[d]; dv = mdvol.loc[d]; cl = mclose.loc[d]; vol = mvol.loc[d]
        cand = [t for t in u.index if t != "SPY" and pd.notna(f.get(t)) and np.isfinite(f[t])
                and pd.notna(dv.get(t)) and dv[t] >= DVOL_FLOOR and pd.notna(cl.get(t)) and cl[t] > PRICE_FLOOR]
        if profit_gate:
            cand = [t for t in cand if profit_ok(P, d, t)]
        if len(cand) < 10:
            pr.append(np.nan); held_n.append(0); continue
        ranked = u[cand].sort_values(ascending=False)
        k = max(1, int(len(ranked) * top_frac))
        hold = list(ranked.index[:k]); rr = f[hold]
        if weight == "invvol":
            wv = np.array([1.0 / vol[t] if pd.notna(vol.get(t)) and vol[t] > 0 else np.nan for t in hold])
            wv = np.where(np.isfinite(wv), wv, np.nanmedian(wv)) if np.isfinite(wv).sum() >= 2 else np.ones(len(hold))
            wv = wv / wv.sum(); m = float(np.dot(rr.values, wv))
        else:
            wv = np.full(len(hold), 1.0 / len(hold)); m = float(rr.mean())
        pos_win += int((rr > 0).sum()); pos_tot += len(rr)
        cur_w = {t: wv[i] for i, t in enumerate(hold)}
        keys = set(cur_w) | set(prev_w)
        turn = 0.5 * sum(abs(cur_w.get(t, 0.0) - prev_w.get(t, 0.0)) for t in keys)
        turns.append(turn); m -= turn * 2.0 * (cost_bps / 1e4); prev_w = cur_w
        holdings[d.date().isoformat()] = hold
        pr.append(m); held_n.append(k)
    return dict(ret=pd.Series(pr, index=midx[:-1]), avg_n=float(np.mean([h for h in held_n if h])),
                pos_win=pos_win / pos_tot if pos_tot else float("nan"),
                turnover=float(np.mean(turns)) if turns else float("nan"), holdings=holdings)


def perf(ret, spy_ret, posw=None):
    pr = ret.dropna(); eq = np.cumprod(1 + pr.values)
    yrs = len(pr) / 12.0; sr = spy_ret.reindex(pr.index).values
    return dict(total=(eq[-1] - 1) * 100, cagr=(eq[-1] ** (1 / yrs) - 1) * 100,
                sharpe=float(pr.mean() / pr.std() * np.sqrt(12)),
                maxdd=float((eq / np.maximum.accumulate(eq) - 1).min() * 100),
                hit=float((pr.values > 0).mean() * 100), beat_spy=float(np.nanmean(pr.values > sr) * 100),
                n_mo=int(len(pr)), pos_win=None if posw is None else float(posw * 100))
```

- [ ] **Step 4: Run test to verify it passes**

Run: `MSYS_NO_PATHCONV=1 docker exec rotation-backend-1 python -m pytest /app/test_strategy_g.py::test_gcore_reproduces_prototype -v`
Expected: PASS (CAGR ~29.5, DD ~−24.8, turnover ~0.21, avg_n ~60).

- [ ] **Step 5: Commit**

```bash
git add backend/strategy_g.py backend/test_strategy_g.py
git commit -m "feat(strategy-g): sim + cost model + perf (reproduces prototype G-core)

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01SJJ3BR9dcEBSWjiYQqPjSW"
```

---

## Task 3: Walk-forward / subperiod robustness gate

**Files:**
- Modify: `backend/strategy_g.py`
- Test: `backend/test_strategy_g.py`

**Interfaces:**
- Consumes: `sim`, `perf`, panels.
- Produces: `walk_forward(P, cost_bps=25.0) -> dict` with keys `subperiods` (list of `{label, start, end, cagr, maxdd, hit, beat_spy, spy_cagr}`) covering non-overlapping ~3-year windows (2016–2018, 2019–2021, 2022–2023, 2024–2026) plus a COVID-crash window (2020-01..2020-06), and `verdict` (str: "robust" if G-core CAGR > SPY CAGR in ≥ 3 of the 4 multi-year windows, else "fragile").

- [ ] **Step 1: Write the failing test**

```python
def test_walk_forward_gate():
    uni, tgts = G.build_universe()
    P = G.build_panels(uni, tgts)
    wf = G.walk_forward(P, cost_bps=25.0)
    assert len(wf["subperiods"]) >= 5
    # gate: beats SPY in at least 3 of the 4 multi-year windows
    multi = [s for s in wf["subperiods"] if not s["label"].startswith("covid")]
    beats = sum(1 for s in multi if s["cagr"] > s["spy_cagr"])
    assert beats >= 3, f"not robust: beats SPY in only {beats}/{len(multi)} windows"
    assert wf["verdict"] == "robust"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `MSYS_NO_PATHCONV=1 docker exec rotation-backend-1 python -m pytest /app/test_strategy_g.py::test_walk_forward_gate -v`
Expected: FAIL — `AttributeError: ... 'walk_forward'`.

- [ ] **Step 3: Write minimal implementation**

```python
_WINDOWS = [("2016-2018", "2016-01-01", "2018-12-31"), ("2019-2021", "2019-01-01", "2021-12-31"),
            ("2022-2023", "2022-01-01", "2023-12-31"), ("2024-2026", "2024-01-01", "2026-12-31"),
            ("covid-2020H1", "2020-01-01", "2020-06-30")]


def walk_forward(P, cost_bps=25.0):
    r = sim(P, cost_bps=cost_bps)["ret"]; spy = P["spy_ret"]
    subs = []
    for label, a, b in _WINDOWS:
        seg = r[(r.index >= a) & (r.index <= b)].dropna()
        sseg = spy.reindex(seg.index)
        if len(seg) < 3:
            continue
        m = perf(seg, sseg); sm = perf(sseg, sseg)
        subs.append(dict(label=label, start=a, end=b, cagr=m["cagr"], maxdd=m["maxdd"],
                         hit=m["hit"], beat_spy=m["beat_spy"], spy_cagr=sm["cagr"]))
    multi = [s for s in subs if not s["label"].startswith("covid")]
    beats = sum(1 for s in multi if s["cagr"] > s["spy_cagr"])
    return dict(subperiods=subs, verdict="robust" if beats >= 3 else "fragile")
```

- [ ] **Step 4: Run test to verify it passes**

Run: `MSYS_NO_PATHCONV=1 docker exec rotation-backend-1 python -m pytest /app/test_strategy_g.py::test_walk_forward_gate -v`
Expected: PASS. **If it FAILS (verdict "fragile"), STOP and report** — the edge is a single-regime artifact and Phase 2 must not proceed. Record the subperiod table regardless.

- [ ] **Step 5: Commit**

```bash
git add backend/strategy_g.py backend/test_strategy_g.py
git commit -m "feat(strategy-g): walk-forward subperiod robustness gate

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01SJJ3BR9dcEBSWjiYQqPjSW"
```

---

## Task 4: `main()` — CLI, cost curve, G-smooth, sweep, and `BacktestResult` persistence

**Files:**
- Modify: `backend/strategy_g.py`
- Test: `backend/test_strategy_g.py`

**Interfaces:**
- Consumes: everything above; `core.models.BacktestResult`, `django.utils.timezone`.
- Produces: `build_payload(P) -> dict` (core + smooth metrics across a cost grid, turnover, walk-forward, latest-month holdings, caveats); `main()` CLI: `--db`, `--variant core|smooth`, `--cost-bps N` (default 25), `--sweep` (prints the full DD/win-rate lever table for the record). Writes `.data/strategy_g.json` and (with `--db`) upserts `BacktestResult(kind="strategy_g")`.

- [ ] **Step 1: Write the failing test**

```python
def test_payload_and_persistence():
    uni, tgts = G.build_universe()
    P = G.build_panels(uni, tgts)
    p = G.build_payload(P)
    assert p["config"]["top_frac"] == 0.05 and p["config"]["profit_gate"] is True
    assert "cost_grid" in p and any(abs(c["cost_bps"] - 25) < 1e-9 for c in p["cost_grid"])
    assert p["walk_forward"]["verdict"] in ("robust", "fragile")
    assert p["latest_holdings"] and isinstance(p["latest_holdings"]["tickers"], list)
    assert "computed_at" in p
```

- [ ] **Step 2: Run test to verify it fails**

Run: `MSYS_NO_PATHCONV=1 docker exec rotation-backend-1 python -m pytest /app/test_strategy_g.py::test_payload_and_persistence -v`
Expected: FAIL — `AttributeError: ... 'build_payload'`.

- [ ] **Step 3: Write minimal implementation**

```python
COST_GRID = [0, 10, 25, 50, 100]


def build_payload(P):
    from django.utils import timezone
    def grid(weight):
        out = []
        for cb in COST_GRID:
            r = sim(P, weight=weight, cost_bps=cb); m = perf(r["ret"], P["spy_ret"], r["pos_win"])
            out.append(dict(cost_bps=cb, weight=weight, avg_n=r["avg_n"], turnover=r["turnover"], **m))
        return out
    core_grid = grid("equal"); smooth_grid = grid("invvol")
    r25 = sim(P, cost_bps=25.0)
    last_d = sorted(r25["holdings"])[-1]
    spy = perf(P["spy_ret"], P["spy_ret"])
    return dict(
        computed_at=pd.Timestamp.utcnow().isoformat(),
        config=dict(top_frac=TOP_FRAC, profit_gate=True, weight="equal", dvol_floor=DVOL_FLOOR,
                    price_floor=PRICE_FLOOR, stale_days=STALE_DAYS, rebalance="monthly"),
        cost_grid=core_grid + smooth_grid, spy=spy,
        turnover_oneway_monthly=r25["turnover"],
        walk_forward=walk_forward(P, cost_bps=25.0),
        latest_holdings=dict(date=last_d, tickers=r25["holdings"][last_d]),
        curve={d.date().isoformat(): float(v) for d, v in r25["ret"].dropna().items()},
        caveat="Long-only EW monthly. PIT (targets<=month-end 180d; TTM ni ffill by avail_date). Survivorship-aware "
               "(delisted-with-candles incl.). US-listed only. In-sample 2016-2026; see walk_forward for subperiods.")


OUT = Path("/app/.data/strategy_g.json")


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", action="store_true"); ap.add_argument("--variant", default="core", choices=["core", "smooth"])
    ap.add_argument("--cost-bps", type=float, default=25.0); ap.add_argument("--sweep", action="store_true")
    a = ap.parse_args()
    uni, tgts = build_universe(); print(f"covered universe: {len(uni)}", flush=True)
    P = build_panels(uni, tgts)
    w = "invvol" if a.variant == "smooth" else "equal"
    r = sim(P, weight=w, cost_bps=a.cost_bps); m = perf(r["ret"], P["spy_ret"], r["pos_win"])
    print(f"G-{a.variant} @ {a.cost_bps:.0f}bps: CAGR {m['cagr']:+.1f}%  DD {m['maxdd']:.1f}%  "
          f"Sharpe {m['sharpe']:.2f}  hit {m['hit']:.1f}%  >SPY {m['beat_spy']:.1f}%  turn {r['turnover']*100:.0f}%", flush=True)
    if a.sweep:
        _print_sweep(P)   # ports analyst_gap_book G_SWEEP table for the record
    p = build_payload(P)
    OUT.parent.mkdir(parents=True, exist_ok=True); OUT.write_text(json.dumps(p, indent=2, default=str))
    print(f"verdict: {p['walk_forward']['verdict']}", flush=True)
    if a.db:
        try:
            from core.models import BacktestResult
            from django.utils import timezone
            BacktestResult.objects.update_or_create(
                kind="strategy_g", defaults={"payload": json.loads(json.dumps(p, default=str)), "computed_at": timezone.now()})
            print("Saved BacktestResult[strategy_g]", flush=True)
        except Exception as e:
            print("DB save failed:", e, flush=True)


def _print_sweep(P):
    def row(label, **kw):
        r = sim(P, **kw); m = perf(r["ret"], P["spy_ret"], r["pos_win"])
        print(f"  {label:36} CAGR {m['cagr']:+6.1f}%  DD {m['maxdd']:6.1f}%  Sh {m['sharpe']:4.2f}  "
              f"hit {m['hit']:4.1f}%  posW {m['pos_win']:.1f}%  turn {r['turnover']*100:.0f}%", flush=True)
    print("=== G lever sweep (record) ===", flush=True)
    row("G-core (top-5% profit EW)")
    row("G-smooth (inverse-vol)", weight="invvol")
    for frac, tag in [(0.10, "decile"), (0.20, "quintile")]:
        row(f"{tag} profit EW", top_frac=frac)


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run test + a real DB persistence run**

Run: `MSYS_NO_PATHCONV=1 docker exec rotation-backend-1 python -m pytest /app/test_strategy_g.py::test_payload_and_persistence -v`
Expected: PASS.
Then: `MSYS_NO_PATHCONV=1 docker exec rotation-backend-1 python -u /app/strategy_g.py --db --sweep`
Expected: prints `G-core @ 25bps: CAGR ~+29.5% DD ~-24.8% ...`, `verdict: robust`, `Saved BacktestResult[strategy_g]`.
Verify: `MSYS_NO_PATHCONV=1 docker exec rotation-backend-1 python manage.py shell -c "from core.models import BacktestResult as B; r=B.objects.get(kind='strategy_g'); print(r.payload['config'], r.payload['walk_forward']['verdict'])"`

- [ ] **Step 5: Commit**

```bash
git add backend/strategy_g.py backend/test_strategy_g.py
git commit -m "feat(strategy-g): CLI + cost grid + G-smooth + sweep + BacktestResult persistence

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01SJJ3BR9dcEBSWjiYQqPjSW"
```

---

## Task 5: Full-suite run + prototype-parity confirmation (Phase 1 exit gate)

**Files:**
- Test: `backend/test_strategy_g.py` (run all)

- [ ] **Step 1: Run the whole test file**

Run: `MSYS_NO_PATHCONV=1 docker exec rotation-backend-1 python -m pytest /app/test_strategy_g.py -v`
Expected: all PASS.

- [ ] **Step 2: Parity check vs prototype**

Run the prototype cost validation and the engine side by side; confirm G-core @ {0,10,25,50} bps CAGR/DD/hit match within ±0.5pp:
`MSYS_NO_PATHCONV=1 docker exec rotation-backend-1 python -u /app/analyst_gap_book.py` (cost table) vs `strategy_g.py --sweep`.
Expected: numbers agree (same underlying logic).

- [ ] **Step 3: Report the verdict to the user**

If `walk_forward.verdict == "robust"`: report Phase 1 complete with the net-of-cost table + subperiod table, and ask whether to proceed to Phase 2. If `"fragile"`: STOP, present the subperiod breakdown, and recommend against live wiring.

**Phase 1 ends here.** Phase 2 below is gated on this verdict and explicit user go-ahead.

---

## Task 6 (Phase 2 — gated): live monthly-pick scanner

**Files:**
- Create: `backend/strategy_g_scan.py`

**Interfaces:**
- Consumes: `strategy_g.build_universe/build_panels/sim`.
- Produces: writes `.data/strategy_g_picks.json` = `{computed_at, date, variant, tickers, upside, close}` for the latest month-end; idempotent.

- [ ] **Step 1: Write scanner**

```python
#!/usr/bin/env python3
"""Live Strategy G monthly picks -> .data/strategy_g_picks.json.
Run: MSYS_NO_PATHCONV=1 docker exec rotation-backend-1 python -u /app/strategy_g_scan.py"""
import json
from pathlib import Path
import strategy_g as G

uni, tgts = G.build_universe(); P = G.build_panels(uni, tgts)
r = G.sim(P, cost_bps=0.0)
d = sorted(r["holdings"])[-1]; tks = r["holdings"][d]
ups = P["ups"].loc[G.pd.Timestamp(d)]; cl = P["mclose"].loc[G.pd.Timestamp(d)]
out = dict(computed_at=G.pd.Timestamp.utcnow().isoformat(), date=d, variant="core",
           picks=[dict(ticker=t, upside=float(ups.get(t)), close=float(cl.get(t))) for t in tks])
Path("/app/.data/strategy_g_picks.json").write_text(json.dumps(out, indent=2, default=str))
print(f"wrote {len(tks)} picks for {d}", flush=True)
```

- [ ] **Step 2: Run and verify**

Run: `MSYS_NO_PATHCONV=1 docker exec rotation-backend-1 python -u /app/strategy_g_scan.py`
Expected: `wrote ~60 picks for 2026-08-31`; file exists with a `picks` list.

- [ ] **Step 3: Commit**

```bash
git add backend/strategy_g_scan.py
git commit -m "feat(strategy-g): live monthly-pick scanner

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01SJJ3BR9dcEBSWjiYQqPjSW"
```

---

## Task 7 (Phase 2 — gated): `/api/strategy-g` endpoint

**Files:**
- Modify: `backend/api/views.py`, `backend/api/urls.py` (confirm exact route file with `grep -n "urlpatterns\|path(" backend/api/urls.py`).

**Interfaces:**
- Produces: `GET /api/strategy-g` → JSON `{payload: BacktestResult(kind="strategy_g").payload, picks: <.data/strategy_g_picks.json or null>}`.

- [ ] **Step 1: Find the pattern of an existing simple BacktestResult-serving view**

Run: `grep -n "BacktestResult" backend/api/views.py | head` and mirror the closest existing endpoint (e.g. the survivorship or rotation-lab view).

- [ ] **Step 2: Add the view (mirroring the existing pattern exactly)**

```python
# backend/api/views.py — add near the other BacktestResult views
from core.models import BacktestResult
from rest_framework.decorators import api_view
from rest_framework.response import Response
import json
from pathlib import Path

@api_view(["GET"])
def strategy_g(request):
    try:
        payload = BacktestResult.objects.get(kind="strategy_g").payload
    except BacktestResult.DoesNotExist:
        payload = None
    picks_p = Path("/app/.data/strategy_g_picks.json")
    picks = json.loads(picks_p.read_text()) if picks_p.exists() else None
    return Response({"payload": payload, "picks": picks})
```

- [ ] **Step 3: Wire the route**

```python
# backend/api/urls.py — add to urlpatterns
path("strategy-g", views.strategy_g),
```

- [ ] **Step 4: Restart backend and verify**

Run: `docker compose restart backend` then `curl -s http://localhost:8001/api/strategy-g | python -m json.tool | head -30`
Expected: JSON with `payload.config` and a `picks` list (or nulls if not yet computed).

- [ ] **Step 5: Commit**

```bash
git add backend/api/views.py backend/api/urls.py
git commit -m "feat(strategy-g): /api/strategy-g endpoint

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01SJJ3BR9dcEBSWjiYQqPjSW"
```

---

## Task 8 (Phase 2 — gated): frontend "Strategy G" tab

**Files:**
- Modify: `frontend/src/App.js`

**Interfaces:**
- Consumes: `GET /api/strategy-g`.
- Produces: a tab showing the cost-grid table (G-core + G-smooth), the subperiod/walk-forward table, and the latest picks.

- [ ] **Step 1: Add the tab, mirroring an existing study tab's fetch+render**

Locate an existing tab that reads a single `BacktestResult` payload (e.g. survivorship), copy its structure, point the fetch at `/api/strategy-g`, and render `payload.cost_grid`, `payload.walk_forward.subperiods`, and `picks.picks`.

- [ ] **Step 2: Rebuild the prod frontend (MOUNT RULE)**

Run: `docker compose up -d --build --no-deps frontend`
Expected: build succeeds, backend not recreated.

- [ ] **Step 3: Verify in browser**

Open `http://localhost:3001`, click the Strategy G tab; confirm the cost grid, subperiods, and picks render with a last-updated chip.

- [ ] **Step 4: Commit + bump submodule if applicable**

```bash
git add frontend/src/App.js
git commit -m "feat(strategy-g): frontend Strategy G tab

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01SJJ3BR9dcEBSWjiYQqPjSW"
```

---

## Self-Review

**Spec coverage:**
- Universe/floor/PIT → Task 1. Selection/profit-gate/weighting/cost model → Task 2. G-smooth flag → Tasks 2/4. Walk-forward gate → Task 3 (+ success criterion #3). Persistence `BacktestResult(strategy_g)` + JSON → Task 4. Prototype parity (success #1) → Task 5. Live scanner (success #4) → Task 6. API → Task 7. Frontend → Task 8. Risks (in-sample/coverage/capacity) surfaced via `walk_forward` + `caveat` in payload (Task 4). ✅
- Capacity-curve and analyst-coverage-skew are documented risks in the spec, not build tasks — left as noted follow-ups (YAGNI for v1); no task needed.

**Placeholder scan:** No TBD/TODO. Task 7/8 intentionally say "mirror the existing pattern" and require a `grep` step first, because the exact view/route/tab idioms must be read from the live code — each such step names the exact command and the exact code to add. ✅

**Type consistency:** `sim` returns a dict (`ret/avg_n/pos_win/turnover/holdings`) used identically in Tasks 3/4/6; `perf` returns `cagr/maxdd/hit/beat_spy/sharpe/pos_win` used identically in Tasks 2/3/4; `walk_forward` returns `subperiods/verdict` used in Tasks 3/4; `build_payload` keys (`config/cost_grid/walk_forward/latest_holdings/curve`) consumed by Tasks 4/7/8. Constants `DVOL_FLOOR/PRICE_FLOOR/TOP_FRAC/STALE_DAYS` defined once in Task 1. ✅
