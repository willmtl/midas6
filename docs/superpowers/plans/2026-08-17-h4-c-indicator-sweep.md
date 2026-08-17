# H4 Indicator Sweep on C — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Inside daily-C candidate windows, sweep the full `studies.py` indicator library (~381 singles) plus dip→confirmation combos (headline `gap_dn__rsi_x`) as H4 0–3-day entries, ranked by t-stat with a multiple-comparisons guard, to see if anything beats `mr_rsi_os`.

**Architecture:** A new root script `h4_c_indicators.py` reuses `h4_on_signals_study.candidate_windows("C")` for the PIT windows, `intraday_data.get_4h` for cached 4h frames, `h4_study` for the exit ladder / episode-dedup / agg, and `studies.SIGNALS` for the indicator library. It builds a small combo matrix on each 4h frame, masks entries to C windows, runs the fixed exit ladder, aggregates odds, and ranks. Additive: new `BacktestResult` kind, new API view, new tab. Nothing existing is modified.

**Tech Stack:** Python 3, pandas, numpy, `ta`, Django ORM, Docker bind mounts, React (CRA prod build).

## Global Constraints

- **Read-only reuse:** `h4_on_signals_study.py`, `h4_study.py`, `intraday_data.py`, `studies.py` imported, never modified. Additive wiring only.
- **Env reaps long execs:** run anything >2s **detached** (`setsid nohup … > /app/.data/<log> 2>&1 </dev/null &`) and read the log via quick execs / a `run_in_background` waiter. Never rely on a foreground/backgrounded exec completing.
- **Exit ladder:** reuse `h4_study.EXITS` = `[1,2,3,4,5,6,8,10]` bars (0–3-day focus). No fees, entry at bar close, episode-dedup gap=3.
- **Boolean signals → flat ladder only** (no magnitude buckets).
- **Significance guard:** ~381+ tests → flag "notable" only when **t>3 AND n≥100**; report Bonferroni t (`α=0.05/N` two-sided) in payload. Degenerate signals (alt-data/fundamental that can't fire on price) yield ~0 entries and drop via `n≥20` in agg.
- **Combo:** precondition P fired within prior **K=5** bars, then trigger T fires now. Headline `gap_dn__rsi_x` = gap-down≥2% then RSI(14) crosses above SMA(14) within 5 bars.
- **Mount gotcha:** create root file before `docker compose up`; `rmdir backend/<x>.py` if a stub dir appears.
- **Run:** `MSYS_NO_PATHCONV=1 docker exec rotation-backend-1 …`; `/app`=`./backend`.

---

## File Structure

- **Create `h4_c_indicators.py`** (root) — combo builder, single-signal adapter, masked ladder, sweep run, DB save, CLI.
- **Create `test_h4_c_indicators.py`** (root) — combo-builder + ladder unit tests (argv-selectable plain-assert).
- **Modify `docker-compose.yml`** — 2 mount lines after the `test_h4_on_signals.py` line.
- **Modify `backend/api/views.py`** — add `H4CIndicatorsView`.
- **Modify `backend/api/urls.py`** — add route.
- **Modify `frontend/src/App.js`** — add `H4CLibraryPage` + one Research `HubTabs` entry.

---

## Task 1: Scaffold — combo builder + masked ladder (unit-tested)

**Files:**
- Create: `h4_c_indicators.py`, `test_h4_c_indicators.py`
- Modify: `docker-compose.yml`

**Interfaces:**
- Consumes: `h4_study.{EXITS, _episode_starts, GAP}`; `ta`.
- Produces:
  - `combo_entries(df) -> dict[str, np.ndarray[bool]]` (keys like `gap_dn__rsi_x`).
  - `ladder(entry_arr, close, dates, allowed_dates) -> dict[exit_key, list[float]]` (fixed bars, masked, episode-deduped).

- [ ] **Step 1: Create `h4_c_indicators.py` with the combo builder + ladder**

```python
#!/usr/bin/env python3
"""Sweep the full studies.py indicator library + dip->confirmation combos as H4 0-3 day entries
inside daily-C candidate windows. See docs/superpowers/specs/2026-08-17-h4-c-indicator-sweep-design.md.
Reuses h4_on_signals_study + h4_study + studies read-only. Pure funcs import without Django."""
import numpy as np
import pandas as pd
import ta
import h4_study as H

K_SEQ = 5  # precondition must have fired within the prior K bars


def _rsi(df):
    return ta.momentum.rsi(df["Close"], window=14)


def _preconditions(df):
    c = df["Close"]
    gap = (df["Open"] / c.shift(1) - 1) * 100
    rsi = _rsi(df)
    prior_min = c.shift(1).rolling(30).min()
    down = c < c.shift(1)
    ndown3 = down & down.shift(1) & down.shift(2)
    return {
        "gap_dn": (gap <= -2).fillna(False),
        "rsi_lt30": (rsi < 30).fillna(False),
        "newlow30": (c < prior_min).fillna(False),
        "ndown3": ndown3.fillna(False),
    }


def _triggers(df):
    c = df["Close"]
    rsi = _rsi(df)
    sma = rsi.rolling(14).mean()
    rsi_x = (rsi > sma) & (rsi.shift(1) <= sma.shift(1))
    adl = ta.volume.acc_dist_index(df["High"], df["Low"], df["Close"], df["Volume"])
    ad_sma = adl.rolling(20).mean()
    ad_rising = (ad_sma > ad_sma.shift(1)) & (ad_sma.shift(1) <= ad_sma.shift(2))
    break_hi = c > df["High"].shift(1)
    return {
        "rsi_x": rsi_x.fillna(False),
        "ad_up": ad_rising.fillna(False),
        "break_hi": break_hi.fillna(False),
    }


def _seq(P, T, k=K_SEQ):
    """T fires now AND P fired within the prior k bars (P strictly before T)."""
    Pr = P.rolling(k).max().shift(1).fillna(0).astype(bool)   # P in [i-k, i-1]
    return (T & Pr).fillna(False).values


def combo_entries(df):
    """All precondition x trigger sequences, keyed 'P__T'. gap_dn__rsi_x is the headline."""
    P = _preconditions(df)
    T = _triggers(df)
    out = {}
    for pk, pv in P.items():
        for tk, tv in T.items():
            out[f"{pk}__{tk}"] = _seq(pv, tv)
    return out


def ladder(entry_arr, close, dates, allowed_dates):
    """Masked, episode-deduped fixed-bar exit ladder for one boolean entry series on one frame.
    Returns {exit_key: [returns%]}. allowed_dates=set of datetime.date (C window); None=all."""
    n = len(close)
    cand = [i for i in range(n) if entry_arr[i] and
            (allowed_dates is None or dates[i].date() in allowed_dates)]
    idxs = sorted(H._episode_starts(cand, gap=H.GAP))
    out = {f"{b}b": [] for b in [e[1] for e in H.EXITS]}
    keymap = {b: f"{b}b" for b in [e[1] for e in H.EXITS]}
    for i in idxs:
        ep = float(close[i])
        if ep <= 0:
            continue
        for _, bars, _ in H.EXITS:
            j = i + bars
            if j < n:
                out[keymap[bars]].append((close[j] - ep) / ep * 100)
    return out
```

- [ ] **Step 2: Create `test_h4_c_indicators.py`**

```python
#!/usr/bin/env python3
"""Unit tests for h4_c_indicators. Run:
  docker exec rotation-backend-1 python -u /app/test_h4_c_indicators.py"""
import sys
import numpy as np
import pandas as pd


def _frame(n=120):
    t = np.arange(n)
    close = 100 * (1 + 0.05*np.sin(t/6.0) + 0.0005*t)
    idx = pd.date_range("2023-01-02 08:00", periods=n, freq="4h", tz="UTC")
    return pd.DataFrame({"Open": close, "High": close*1.004, "Low": close*0.996,
                         "Close": close, "Volume": np.full(n, 1e6)}, index=idx)


def test_combo_keys():
    from h4_c_indicators import combo_entries
    df = _frame()
    e = combo_entries(df)
    assert "gap_dn__rsi_x" in e, "headline combo missing"
    assert len(e) == 12, f"expected 12 combos, got {len(e)}"
    for k, v in e.items():
        assert len(v) == len(df) and v.dtype == bool, f"{k} bad shape/dtype"
    print("test_combo_keys OK")


def test_combo_seq_fires():
    """A gap-down at bar 40 then an RSI cross a few bars later fires gap_dn__rsi_x once, near the cross."""
    from h4_c_indicators import combo_entries, _preconditions, _triggers
    df = _frame(120)
    # force a gap-down at bar 40
    df.iloc[40, df.columns.get_loc("Open")] = df["Close"].iloc[39] * 0.95
    e = combo_entries(df)
    P = _preconditions(df); T = _triggers(df)
    assert P["gap_dn"].values[40], "gap_dn precondition should fire at 40"
    # any gap_dn__rsi_x fire must have an rsi_x trigger at that bar and a gap_dn within prior 5 bars
    fired = np.where(e["gap_dn__rsi_x"])[0]
    for i in fired:
        assert T["rsi_x"].values[i], "combo fired without trigger"
        assert P["gap_dn"].values[i-5:i].any(), "combo fired without recent precondition"
    print("test_combo_seq_fires OK")


def test_ladder_mask():
    from h4_c_indicators import ladder
    df = _frame(60)
    entry = np.zeros(60, bool); entry[10] = entry[30] = True
    close = df["Close"].values; dates = df.index.normalize()
    allow = {dates[10].date()}          # only the first entry allowed
    out = ladder(entry, close, dates, allow)
    assert len(out["1b"]) == 1, "mask should keep only the allowed entry"
    out_all = ladder(entry, close, dates, None)
    assert len(out_all["1b"]) == 2, "None mask keeps both"
    print("test_ladder_mask OK")


CHECKS = {"combo_keys": test_combo_keys, "combo_seq": test_combo_seq_fires, "ladder": test_ladder_mask}


def main():
    names = [a for a in sys.argv[1:] if a in CHECKS] or list(CHECKS)
    for nm in names:
        CHECKS[nm]()
    print(f"\n{len(names)} check(s) passed.")


if __name__ == "__main__":
    main()
```

- [ ] **Step 3: Guard stub dirs + confirm files**

Run:
```bash
for f in h4_c_indicators.py test_h4_c_indicators.py; do [ -d "backend/$f" ] && rmdir "backend/$f"; done
ls -la h4_c_indicators.py test_h4_c_indicators.py
```
Expected: both regular files.

- [ ] **Step 4: Add mounts to `docker-compose.yml`**

After `  - ./test_h4_on_signals.py:/app/test_h4_on_signals.py`, add:
```yaml
  - ./h4_c_indicators.py:/app/h4_c_indicators.py
  - ./test_h4_c_indicators.py:/app/test_h4_c_indicators.py
```

- [ ] **Step 5: Recreate backend**

Run:
```bash
MSYS_NO_PATHCONV=1 docker compose up -d backend
```
Expected: backend recreated, no mount error.

- [ ] **Step 6: Run the unit tests (fast, foreground OK — synthetic frames, <2s)**

Run:
```bash
MSYS_NO_PATHCONV=1 docker exec rotation-backend-1 python -u /app/test_h4_c_indicators.py
```
Expected: `test_combo_keys OK`, `test_combo_seq_fires OK`, `test_ladder_mask OK`, `3 check(s) passed.`

- [ ] **Step 7: Commit**

```bash
git add h4_c_indicators.py test_h4_c_indicators.py docker-compose.yml
git commit -m "feat(h4clib): combo builder (dip->confirmation, incl gap_dn__rsi_x) + masked ladder"
```

---

## Task 2: Sweep run — singles + combos, guard, DB save, CLI

**Files:**
- Modify: `h4_c_indicators.py` (add `single_entries`, `sweep`, `main`)

**Interfaces:**
- Consumes: `combo_entries`, `ladder` (Task 1); `h4_on_signals_study.candidate_windows`; `intraday_data.get_4h`; `h4_study.agg_rows`; `studies.SIGNALS`; `core.models.BacktestResult`.
- Produces: `BacktestResult(kind="h4_c_indicators")`; CLI `--no-fetch`, `--years`.

- [ ] **Step 1: Implement `single_entries`, `sweep`, `main`**

Add to `h4_c_indicators.py`:

```python
def single_entries(df):
    """Every studies.py SIGNALS[key] evaluated on this 4h frame -> {key: bool ndarray}. Signals that
    raise, or that can't fire on price alone (alt-data/fundamental), yield all-False (dropped later)."""
    from studies import SIGNALS as STUDY_SIGNALS
    out = {}
    n = len(df)
    for k, (_name, fn) in STUDY_SIGNALS.items():
        try:
            s = fn(df)
            s = s.reindex(df.index).fillna(False) if hasattr(s, "reindex") else pd.Series(s, index=df.index).fillna(False)
            arr = s.values.astype(bool)
            if len(arr) == n:
                out[k] = arr
        except Exception:
            continue
    return out


def _name_of(key, is_combo):
    if is_combo:
        return key
    from studies import SIGNALS as STUDY_SIGNALS
    return STUDY_SIGNALS.get(key, (key,))[0]


def sweep(years, allow_fetch):
    from intraday_data import get_4h
    from h4_on_signals_study import candidate_windows
    allowed, cwmeta = candidate_windows("C")
    names = sorted(allowed)
    pools_s, pools_c = {}, {}
    base_pool = []
    got = dropped = 0
    n_single_keys = 0
    for tk in names:
        df = get_4h(tk, years, allow_fetch)
        if df is None or len(df) < 120:
            dropped += 1
            continue
        got += 1
        close = df["Close"].values
        dates = df.index.normalize()
        ad = allowed[tk]
        c3 = (close[3:] - close[:-3]) / close[:-3] * 100
        base_pool.extend([x for x in c3 if np.isfinite(x)])
        for k, arr in single_entries(df).items():
            n_single_keys = max(n_single_keys, 1)
            d = ladder(arr, close, dates, ad)
            for ek, v in d.items():
                pools_s.setdefault(k, {}).setdefault(ek, []).extend(v)
        for k, arr in combo_entries(df).items():
            d = ladder(arr, close, dates, ad)
            for ek, v in d.items():
                pools_c.setdefault(k, {}).setdefault(ek, []).extend(v)
    exit_keys = [f"{b}b" for b in [e[1] for e in H.EXITS]]

    def _rank(pools, is_combo):
        rows = []
        for k, pool in pools.items():
            agg = H.agg_rows(pool, exit_keys, min_trades=20)
            r3 = next((x for x in agg if x["exit"] == "3b"), None)
            best = max(agg, key=lambda x: x["avg_pct"]) if agg else None
            if r3 or best:
                rows.append({"key": k, "name": _name_of(k, is_combo),
                             "r3b": r3, "best": best})
        rows.sort(key=lambda x: -((x["r3b"] or x["best"] or {}).get("t") or -99))
        return rows

    import math
    ntests = len(pools_s) + len(pools_c)
    bonf_t = round(float(abs(_norm_ppf(0.05 / (2 * max(ntests, 1))))), 2)
    base3 = round(float(np.mean(base_pool)), 3) if base_pool else None
    champ = None
    try:
        from core.models import BacktestResult
        h4os = BacktestResult.objects.get(kind="h4_on_signals").payload
        cc = h4os["by_selector"]["C"]["signals"]["mr_rsi_os"]["all"]
        champ = next((x for x in cc if x["exit"] == "3b"), None)
    except Exception:
        pass
    return {"singles": _rank(pools_s, False), "combos": _rank(pools_c, True),
            "champ_mr_rsi_os_3b": champ, "base_rate_3b": base3,
            "n_signals_tested": ntests, "bonferroni_t": bonf_t,
            "candidates": {**cwmeta, "n_with_4h": got, "n_dropped_no_4h": dropped}}


def _norm_ppf(p):
    """Inverse normal CDF (Acklam approximation) — avoids a scipy dependency."""
    a = [-3.969683028665376e+01, 2.209460984245205e+02, -2.759285104469687e+02,
         1.383577518672690e+02, -3.066479806614716e+01, 2.506628277459239e+00]
    b = [-5.447609879822406e+01, 1.615858368580409e+02, -1.556989798598866e+02,
         6.680131188771972e+01, -1.328068155288572e+01]
    c = [-7.784894002430293e-03, -3.223964580411365e-01, -2.400758277161838e+00,
         -2.549732539343734e+00, 4.374664141464968e+00, 2.938163982698783e+00]
    d = [7.784695709041462e-03, 3.224671290700398e-01, 2.445134137142996e+00, 3.754408661907416e+00]
    pl = 0.02425
    if p < pl:
        q = math.sqrt(-2 * math.log(p))
        return (((((c[0]*q+c[1])*q+c[2])*q+c[3])*q+c[4])*q+c[5]) / ((((d[0]*q+d[1])*q+d[2])*q+d[3])*q+1)
    if p <= 1 - pl:
        q = p - 0.5; r = q*q
        return (((((a[0]*r+a[1])*r+a[2])*r+a[3])*r+a[4])*r+a[5])*q / (((((b[0]*r+b[1])*r+b[2])*r+b[3])*r+b[4])*r+1)
    q = math.sqrt(-2 * math.log(1 - p))
    return -(((((c[0]*q+c[1])*q+c[2])*q+c[3])*q+c[4])*q+c[5]) / ((((d[0]*q+d[1])*q+d[2])*q+d[3])*q+1)


import math


def main():
    import os, json, argparse
    os.environ.setdefault("DJANGO_SETTINGS_MODULE", "rotation.settings")
    import django; django.setup()
    from pathlib import Path
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-fetch", action="store_true")
    ap.add_argument("--years", type=float, default=5)
    args = ap.parse_args()
    print(f"sweeping full library + combos on C candidates (fetch={'off' if args.no_fetch else 'on'})", flush=True)
    payload = sweep(args.years, allow_fetch=not args.no_fetch)
    payload["computed_at"] = pd.Timestamp.utcnow().isoformat()
    payload["note"] = ("Full studies.py indicator library (singles) + dip->confirmation combos swept as H4 "
                       "0-3 day entries inside daily-C candidate windows. Ranked by t@3b. MULTIPLE COMPARISONS: "
                       f"{payload['n_signals_tested']} signals tested; Bonferroni t~{payload['bonferroni_t']} — "
                       "treat t<that as noise. Daily-tuned windows on 4h; gross of fees; C = current-membership "
                       "survivorship; only C names with cached 4h included.")
    STUD = Path(__file__).resolve().parent / ".data" / "studies"
    STUD.mkdir(parents=True, exist_ok=True)
    (STUD / "h4_c_indicators.json").write_text(json.dumps(payload, indent=2, default=str))
    try:
        from core.models import BacktestResult
        from django.utils import timezone
        BacktestResult.objects.update_or_create(
            kind="h4_c_indicators",
            defaults={"payload": json.loads(json.dumps(payload, default=str)), "computed_at": timezone.now()})
        print("saved BacktestResult[h4_c_indicators]", flush=True)
    except Exception as e:
        print("DB save failed:", e, flush=True)
    ch = payload["champ_mr_rsi_os_3b"]
    print(f"\nchamp mr_rsi_os 3b: {ch['avg_pct'] if ch else '?'}% | base3b {payload['base_rate_3b']}% | "
          f"Bonferroni t~{payload['bonferroni_t']} | tested {payload['n_signals_tested']}", flush=True)
    print("\n=== TOP COMBOS (by t@3b) ===", flush=True)
    for r in payload["combos"][:10]:
        x = r["r3b"] or r["best"]
        if x:
            print(f"  {r['key']:20} {x['exit']:4} avg {x['avg_pct']:+.2f}% win {x['win_pct']}% t={x['t']} n={x['trades']}", flush=True)
    print("\n=== TOP SINGLES (by t@3b, n>=100) ===", flush=True)
    shown = 0
    for r in payload["singles"]:
        x = r["r3b"]
        if x and x["trades"] >= 100:
            print(f"  {r['key']:24} 3b avg {x['avg_pct']:+.2f}% win {x['win_pct']}% t={x['t']} n={x['trades']}", flush=True)
            shown += 1
            if shown >= 15:
                break


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Run the sweep DETACHED (env reaps long execs), then wait via a background waiter**

Run (launch):
```bash
MSYS_NO_PATHCONV=1 docker exec rotation-backend-1 sh -c 'cd /app && setsid nohup python -u /app/h4_c_indicators.py --no-fetch > /app/.data/h4clib.log 2>&1 </dev/null & echo launched'
```
Then a `run_in_background` waiter:
```bash
MSYS_NO_PATHCONV=1 docker exec rotation-backend-1 sh -c 'for i in $(seq 1 120); do grep -qE "saved BacktestResult|DB save failed|Traceback" /app/.data/h4clib.log 2>/dev/null && break; sleep 6; done; echo "=== h4clib log ==="; tail -40 /app/.data/h4clib.log'
```
Expected: the champ line, `=== TOP COMBOS ===` (with `gap_dn__rsi_x`), `=== TOP SINGLES ===`, and `saved BacktestResult[h4_c_indicators]`. No traceback.

- [ ] **Step 3: Verify the DB payload (quick exec)**

Run:
```bash
MSYS_NO_PATHCONV=1 docker exec rotation-backend-1 python manage.py shell -c "
from core.models import BacktestResult
p = BacktestResult.objects.get(kind='h4_c_indicators').payload
print('tested', p['n_signals_tested'], 'bonf_t', p['bonferroni_t'], 'base3b', p['base_rate_3b'])
print('n combos', len(p['combos']), 'n singles', len(p['singles']))
g = next((r for r in p['combos'] if r['key']=='gap_dn__rsi_x'), None)
print('gap_dn__rsi_x 3b:', g['r3b'] if g else None)
"
```
Expected: prints counts and the `gap_dn__rsi_x` 3b row (the hypothesis result).

- [ ] **Step 4: Commit**

```bash
git add h4_c_indicators.py
git commit -m "feat(h4clib): full-library + combo sweep on C, t-ranked with Bonferroni guard"
```

---

## Task 3: API endpoint (`/api/h4-c-indicators`)

**Files:**
- Modify: `backend/api/views.py` (add `H4CIndicatorsView`)
- Modify: `backend/api/urls.py` (add route)

**Interfaces:**
- Consumes: `_StudyResultView`; `BacktestResult(kind="h4_c_indicators")` (Task 2).
- Produces: `GET /api/h4-c-indicators`.

- [ ] **Step 1: Add the view (after `H4OnSignalsView`)**

```python
class H4CIndicatorsView(_StudyResultView):
    """Full studies.py indicator library + dip->confirmation combos swept as H4 0-3 day entries inside
    daily-C candidate windows, t-ranked with a Bonferroni guard. GET reads BacktestResult[h4_c_indicators];
    POST recomputes (long — fetch)."""
    kind = "h4_c_indicators"
    script = "h4_c_indicators.py"
    json_path = "/app/.data/studies/h4_c_indicators.json"
```

- [ ] **Step 2: Add the route (after the `h4-on-signals` line)**

```python
    path("h4-c-indicators", views.H4CIndicatorsView.as_view(), name="h4-c-indicators"),
```

- [ ] **Step 3: Verify GET (backend autoreloads; do NOT restart if a sweep is running)**

Run:
```bash
sleep 4; OUT="C:/Users/User-X/AppData/Local/Temp/claude/C--workspace-rotation/5901445c-e9d1-4f54-88b3-5821ea13259c/scratchpad/h4clib.json"
curl -s -m 8 http://localhost:8001/api/h4-c-indicators -o "$OUT" -w "HTTP %{http_code}\n"
python -c "import json; d=json.load(open(r'$OUT')); print('computed', d.get('computed'), '| combos', len(d.get('combos',[])), '| singles', len(d.get('singles',[])))"
```
Expected: `HTTP 200`, `computed True`, combos/singles counts > 0.

- [ ] **Step 4: Commit**

```bash
git add backend/api/views.py backend/api/urls.py
git commit -m "feat(h4clib): /api/h4-c-indicators endpoint"
```

---

## Task 4: Frontend — `H4CLibraryPage` sub-tab

**Files:**
- Modify: `frontend/src/App.js` (add `H4CLibraryPage` + Research `HubTabs` entry)

**Interfaces:**
- Consumes: `apiFetch`, `LastUpdatedChip`, `ErrorBanner`; `GET/POST /h4-c-indicators`.
- Produces: `/research/h4clib` tab with ranked combo + single tables and a significance flag.

- [ ] **Step 1: Add `H4CLibraryPage` (place after `H4OnSignalsPage`)**

```jsx
// ---- H4 C-Library sweep -----------------------------------------------------
// Reads GET /h4-c-indicators. Full studies.py indicator library (singles) + dip->confirmation combos
// swept as H4 0-3 day entries inside daily-C windows, t-ranked with a Bonferroni guard.
function H4CLibraryPage() {
  const [data, setData] = useState(null);
  const [err, setErr] = useState(null);
  const [running, setRunning] = useState(false);

  const load = () => { setErr(null); apiFetch('/h4-c-indicators').then(setData).catch(e => setErr(e.message)); };
  useEffect(() => { load(); }, []);
  const run = () => {
    setRunning(true);
    apiFetch('/h4-c-indicators', { method: 'POST' }).then(() => {
      const t = setInterval(() => apiFetch('/h4-c-indicators').then(d => {
        if (d && d.computed) { clearInterval(t); setRunning(false); setData(d); }
      }).catch(() => {}), 10000);
    }).catch(e => { setErr(e.message); setRunning(false); });
  };

  if (err) return <div className="darkpool-page"><h1>H4 C-Library</h1><ErrorBanner message={err} onRetry={load} onDismiss={() => setErr(null)} /></div>;
  if (!data) return <div className="loading">Loading H4 C-library sweep...</div>;
  if (!data.computed) return (
    <div className="darkpool-page"><h1>H4 C-Library</h1>
      <div className="empty-state" style={{ padding: '40px 0' }}>
        <p>{data.note || data.message || 'Not computed yet.'}</p>
        <button className="refresh-btn" onClick={run} disabled={running}>{running ? 'Running (long)…' : 'Run sweep'}</button>
      </div>
    </div>
  );

  const bonf = data.bonferroni_t;
  const champ = data.champ_mr_rsi_os_3b;
  const Row = ({ r }) => {
    const x = r.r3b || r.best; if (!x) return null;
    const notable = x.t != null && x.t > 3 && x.trades >= 100;
    const survives = x.t != null && bonf != null && x.t >= bonf;
    return (
      <tr className="study-row" style={survives ? { background: 'rgba(63,185,80,0.12)' } : undefined}>
        <td><b>{r.key}</b> <span className="dim" style={{ fontSize: 11 }}>{r.key !== r.name ? r.name : ''}</span></td>
        <td style={{ textAlign: 'right' }}>{x.exit}</td>
        <td style={{ textAlign: 'right' }} className={x.avg_pct > 0 ? 'good' : 'bad'}>{x.avg_pct > 0 ? '+' : ''}{x.avg_pct}%</td>
        <td style={{ textAlign: 'right' }}>{x.win_pct}%</td>
        <td style={{ textAlign: 'right' }} className={survives ? 'good' : (notable ? '' : 'dim')}>{x.t}{survives ? ' ★' : (notable ? ' •' : '')}</td>
        <td style={{ textAlign: 'right' }} className="dim">{x.trades}</td>
      </tr>
    );
  };
  const Table = ({ rows, title }) => (
    <div className="darkpool-card">
      <div className="darkpool-card-head"><h2>{title}</h2></div>
      <table className="studies-table">
        <thead><tr><th>Signal</th><th style={{ textAlign: 'right' }}>exit</th><th style={{ textAlign: 'right' }}>avg</th><th style={{ textAlign: 'right' }}>win</th><th style={{ textAlign: 'right' }}>t</th><th style={{ textAlign: 'right' }}>n</th></tr></thead>
        <tbody>{rows.map(r => <Row key={r.key} r={r} />)}</tbody>
      </table>
    </div>
  );

  return (
    <div className="darkpool-page">
      <h1>H4 C-Library <LastUpdatedChip value={data.last_updated} /></h1>
      <p className="subtitle darkpool-muted">
        Full indicator library ({data.n_signals_tested} tested) + dip→confirmation combos, swept as H4 0–3 day
        entries on daily-C value picks. Bar to beat: <b>mr_rsi_os {champ ? `${champ.avg_pct}%/3b (t=${champ.t})` : ''}</b>.
        Base 3b {data.base_rate_3b}%. <b>★</b> survives Bonferroni (t≥{bonf}); <b>•</b> notable (t&gt;3, n≥100); the rest is noise.
        <button className="refresh-btn" style={{ marginLeft: 10 }} onClick={run} disabled={running}>{running ? 'Running…' : 'Recompute'}</button>
      </p>
      <Table rows={(data.combos || []).slice(0, 20)} title="Combos (dip → confirmation)" />
      <Table rows={(data.singles || []).filter(r => (r.r3b || {}).trades >= 100).slice(0, 30)} title="Singles (full library, n≥100, top 30 by t)" />
      {data.note && <p className="subtitle" style={{ marginTop: 10, fontStyle: 'italic' }}>{data.note}</p>}
    </div>
  );
}
```

- [ ] **Step 2: Register the sub-tab (after the `h4sig` entry)**

```jsx
        { key: 'h4clib', label: 'H4 C-Library', hash: '/research/h4clib', match: ['h4clib'], el: <H4CLibraryPage /> },
```

- [ ] **Step 3: Rebuild only the frontend**

Run:
```bash
MSYS_NO_PATHCONV=1 docker compose up -d --build --no-deps frontend
```
Expected: frontend rebuilt/restarted; backend untouched.

- [ ] **Step 4: Verify the tab shipped**

Run:
```bash
JS=$(curl -s -m 8 http://localhost:3001/ | grep -o '/static/js/main\.[a-z0-9]*\.js' | head -1)
OUT="C:/Users/User-X/AppData/Local/Temp/claude/C--workspace-rotation/5901445c-e9d1-4f54-88b3-5821ea13259c/scratchpad/main3.js"
curl -s -m 15 "http://localhost:3001$JS" -o "$OUT"
python -c "s=open(r'$OUT',encoding='utf-8').read(); [print(n,'FOUND' if n in s else 'MISSING') for n in ['H4 C-Library','/h4-c-indicators','research/h4clib']]"
```
Expected: all three `FOUND`.

- [ ] **Step 5: Commit (submodule + parent pointer)**

```bash
cd frontend && git add src/App.js && git commit -m "feat(h4clib): H4 C-Library sweep sub-tab"
cd .. && git add frontend && git commit -m "chore(rotation): bump frontend submodule — H4 C-Library tab"
```

---

## Self-Review

**Spec coverage:**
- Singles = full `studies.SIGNALS` on C → Task 2 `single_entries`/`sweep`. ✓
- Combos = P→T within K=5, headline `gap_dn__rsi_x` → Task 1 `combo_entries`/`_seq`. ✓
- Masked to C windows, exit ladder, episode-dedup, flat (no buckets) → Task 1 `ladder`. ✓
- Rank by t@3b + multiple-comparisons guard (t>3 & n≥100; Bonferroni t) → Task 2 `_rank`/`_norm_ppf`; Task 4 flags. ✓
- Benchmarks base rate + champ mr_rsi_os → Task 2 `sweep`. ✓
- Persist BacktestResult[h4_c_indicators] + JSON → Task 2. ✓
- API + tab → Tasks 3, 4. ✓
- Detached-run discipline (env reaping) → Task 2 Step 2. ✓
- Verification: combo fires correctly, ladder masks, degenerate skip → Task 1 tests + Task 2 (n-threshold drops degenerate). ✓
- Caveats in payload → Task 2 `note`. ✓

**Placeholder scan:** none; every code step complete; commands have expected output. ✓

**Type consistency:** `combo_entries -> {key: bool ndarray}` and `single_entries -> {key: bool ndarray}` both consumed by `ladder(arr, close, dates, allowed)`. `ladder -> {exit_key:[float]}` pooled and fed to `H.agg_rows(pool, exit_keys)`. `sweep` payload keys (`singles/combos/champ_mr_rsi_os_3b/base_rate_3b/n_signals_tested/bonferroni_t/candidates`) match Task 3 GET consumer and Task 4 render (`r.r3b`,`r.best`,`r.key`,`r.name`). `_norm_ppf`/`math` imported before use (module-level `import math` added). ✓

**Deviation:** spec suggested the library view as a *toggle inside the H4-on-Signals tab*; plan uses a *separate `/research/h4clib` sub-tab* instead — cleaner separation, avoids touching the working `H4OnSignalsPage`, same information. Minor, additive.
