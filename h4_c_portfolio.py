#!/usr/bin/env python3
"""Portfolio-level drawdown controls on the H4-on-C dip-buy — sequential simulator.

The -31% drawdown is CORRELATED CLUSTERING (h4_c_regime: 12 losses in 5 days, one macro shock hits every C
name at once), not a filterable entry. Per-trade stops and entry filters can't fix it. This tests three
PORTFOLIO-level controls that act on the running book:
  1. CONCURRENT CAP  — hold at most N dip trades at once; when more fire, take the highest-conviction N.
  2. SELLOFF GATE    — skip NEW adds on a hard-down market bar (SPY 4h <= -X%): don't buy into the air-pocket.
  3. DD CIRCUIT BREAKER — halve position sizing while the book is in a >15% drawdown (restore at -8%),
     like the macro-liquidity halving overlay.
Plus all-three combined. Simulated bar-by-bar with ABSOLUTE (cash-aware) sizing so de-risking actually cuts
exposure. Sizing = steep_2x conviction weight x UNIT of capital; idle capital is cash (0). -> BacktestResult.
Run: MSYS_NO_PATHCONV=1 docker exec rotation-backend-1 python -u /app/h4_c_portfolio.py
"""
import os, json, warnings
from collections import deque, defaultdict
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
import h4_study as H
import h4_c_conviction as C
from h4_c_upside import load_targets, upside_asof, bucket_upside

UNIT = 0.15            # capital per conviction-unit (w=1 -> 15%, w=2 -> 30%, ...); idle = cash
MAX_GROSS = 1.0        # hard exposure cap (no leverage) — a fair 1x baseline
CAP_N = 6              # concurrent-position cap
SELLOFF_THR = 1.5      # skip new adds when SPY is down more than this on the entry DAY (%)
BREAKER_ON = -15.0     # enter half-size mode at this book drawdown (%)
BREAKER_OFF = -8.0     # exit half-size mode when the book recovers to here (%)
BREAKER_MULT = 0.5


def _spy_daily_ret(years=5, allow_fetch=False):
    """SPY daily % return keyed by date (from the DB; robust — no 4h-cache dependency). Macro air-pocket
    days (e.g. -3.4% on 2022-08-26) are the correlated-cluster trigger the gate targets."""
    from seq_fundamental_study import load_candles
    spy = load_candles(["SPY"]).get("SPY")
    if spy is None or len(spy) < 2:
        return {}
    c = spy["Close"]
    r = (c / c.shift(1) - 1) * 100
    idx = pd.to_datetime(spy.index)
    return {ts.date(): float(v) for ts, v in zip(idx, r.values) if pd.notna(v)}


def collect(years, allow_fetch):
    """Eligible steep_2x dip trades: {entry_ts, weight, sched:[(ts_k, close-to-close ret) for k=1..H]}."""
    from intraday_data import get_4h
    from h4_on_signals_study import candidate_windows
    store = load_targets()
    steep = C.SCHEMES["steep_2x"]
    allowed, cwmeta = candidate_windows("C")
    trades = []
    for tk in sorted(allowed):
        df = get_4h(tk, years, allow_fetch)
        if df is None or len(df) < 120:
            continue
        close = df["Close"].values
        ts = df.index
        n = len(close)
        ad = allowed[tk]
        for i in C._entries(df, C.COMBINED):
            if ts[i].date() not in ad:
                continue
            ep = float(close[i])
            if ep <= 0 or i + C.HOLD_BARS >= n:
                continue
            w = steep.get(bucket_upside(upside_asof(store, tk, ts[i].date(), ep)), 0)
            if w <= 0:
                continue
            sched = [(ts[i + b], float(close[i + b] / close[i + b - 1] - 1)) for b in range(1, C.HOLD_BARS + 1)]
            trades.append({"entry_ts": ts[i], "weight": w, "sched": sched})
    return trades, cwmeta


def simulate(trades, spy_ret, cap=False, gate=False, breaker=False):
    entries_by_ts = defaultdict(list)
    all_ts = set()
    for tr in trades:
        entries_by_ts[tr["entry_ts"]].append(tr)
        all_ts.add(tr["entry_ts"])
        for (tk, _) in tr["sched"]:
            all_ts.add(tk)
    timeline = sorted(all_ts)
    equity = 1.0; peak = 1.0; in_breaker = False
    open_pos = []
    eq_ts, eq_val, grosses = [], [], []
    n_taken = n_skipped_cap = n_skipped_gate = 0
    for t in timeline:
        # 1) realize this bar's returns from open positions, capping effective gross exposure at MAX_GROSS
        gross_now = sum(p["size"] for p in open_pos)
        cap_scale = min(1.0, MAX_GROSS / gross_now) if gross_now > 0 else 1.0
        contrib = 0.0; still = []
        for p in open_pos:
            if p["sched"] and p["sched"][0][0] == t:
                contrib += p["size"] * cap_scale * p["sched"].popleft()[1]
            if p["sched"]:
                still.append(p)
        open_pos = still
        if contrib:
            equity *= (1 + contrib)
        peak = max(peak, equity)
        dd = (equity / peak - 1) * 100
        eq_ts.append(t); eq_val.append(equity)
        # 2) breaker state (hysteresis) + size multiplier for new adds
        if breaker:
            if not in_breaker and dd <= BREAKER_ON:
                in_breaker = True
            elif in_breaker and dd >= BREAKER_OFF:
                in_breaker = False
        mult = BREAKER_MULT if (breaker and in_breaker) else 1.0
        # 3) new entries at t, subject to controls
        news = entries_by_ts.get(t, [])
        if gate and news and spy_ret.get(pd.Timestamp(t).date(), 0.0) <= -SELLOFF_THR:
            n_skipped_gate += len(news); news = []
        news = sorted(news, key=lambda x: -x["weight"])
        if cap:
            room = max(0, CAP_N - len(open_pos))
            if len(news) > room:
                n_skipped_cap += len(news) - room
                news = news[:room]
        for tr in news:
            open_pos.append({"size": tr["weight"] * UNIT * mult, "sched": deque(tr["sched"])})
            n_taken += 1
        grosses.append(min(MAX_GROSS, sum(p["size"] for p in open_pos)))
    eq = np.array(eq_val)
    rets = eq[1:] / eq[:-1] - 1
    total = round((eq[-1] - 1) * 100, 1)
    sd = rets.std(ddof=1) if len(rets) > 1 else 0
    sharpe = round(float(rets.mean() / sd * np.sqrt(C.BARS_PER_YEAR)), 2) if sd > 0 else 0.0
    peaks = np.maximum.accumulate(eq)
    dd_series = (eq - peaks) / peaks
    trough_i = int(dd_series.argmin())
    peak_i = int(np.argmax(eq[:trough_i + 1])) if trough_i > 0 else 0
    maxdd = round(float(dd_series.min() * 100), 1)
    gr = np.array(grosses)
    return {"total_return_pct": total, "sharpe": sharpe, "max_dd_pct": maxdd,
            "n_taken": n_taken, "n_skipped_cap": n_skipped_cap, "n_skipped_gate": n_skipped_gate,
            "avg_gross": round(float(gr[gr > 0].mean()) if (gr > 0).any() else 0.0, 2),
            "max_gross": round(float(gr.max()) if len(gr) else 0.0, 2),
            "dd_start": str(eq_ts[peak_i])[:10], "dd_end": str(eq_ts[trough_i])[:10]}


def main():
    os.environ.setdefault("DJANGO_SETTINGS_MODULE", "rotation.settings")
    import django; django.setup()
    from pathlib import Path
    trades, cwmeta = collect(5, False)
    spy_ret = _spy_daily_ret(5, False)
    variants = {
        "baseline": simulate(trades, spy_ret),
        "cap6": simulate(trades, spy_ret, cap=True),
        "selloff_gate": simulate(trades, spy_ret, gate=True),
        "breaker15": simulate(trades, spy_ret, breaker=True),
        "all_three": simulate(trades, spy_ret, cap=True, gate=True, breaker=True),
    }
    base = variants["baseline"]
    rows = []
    for name, m in variants.items():
        rows.append({"variant": name, **m,
                     "ret_vs_base_pp": round(m["total_return_pct"] - base["total_return_pct"], 1),
                     "dd_vs_base_pp": round(m["max_dd_pct"] - base["max_dd_pct"], 1)})
    payload = {
        "computed_at": pd.Timestamp.utcnow().isoformat(),
        "scheme": "steep_2x", "signals": C.COMBINED, "hold_bars": C.HOLD_BARS,
        "params": {"unit": UNIT, "max_gross": MAX_GROSS, "cap_n": CAP_N, "selloff_thr_pct": SELLOFF_THR,
                   "breaker_on_pct": BREAKER_ON, "breaker_off_pct": BREAKER_OFF, "breaker_mult": BREAKER_MULT},
        "n_eligible_trades": len(trades), "rows": rows,
        "note": ("Portfolio-level drawdown controls on the H4-on-C dip-buy (steep_2x/combined), simulated "
                 "bar-by-bar with absolute cash-aware sizing (size = conviction weight x UNIT of capital, idle "
                 "= cash, gross capped at 1x — no leverage). CAP = max concurrent positions (highest-conviction "
                 "win); GATE = skip adds when SPY is down > X% on the entry day; BREAKER = half size while book "
                 "DD < -15% (restore at -8%). Absolute %s differ from the normalized conviction backtest — read "
                 "the RELATIVE return/DD tradeoff. Gross of fees."),
    }
    STUD = Path(__file__).resolve().parent / ".data" / "studies"
    STUD.mkdir(parents=True, exist_ok=True)
    (STUD / "h4_c_portfolio.json").write_text(json.dumps(payload, indent=2, default=str))
    try:
        from core.models import BacktestResult
        from django.utils import timezone
        BacktestResult.objects.update_or_create(
            kind="h4_c_portfolio",
            defaults={"payload": json.loads(json.dumps(payload, default=str)), "computed_at": timezone.now()})
        print("saved BacktestResult[h4_c_portfolio]", flush=True)
    except Exception as e:
        print("DB save failed:", e, flush=True)
    print(f"\n=== PORTFOLIO DD CONTROLS (steep_2x/combined, {len(trades)} eligible trades) ===", flush=True)
    print(f"  {'variant':14} {'total%':>9} {'maxDD%':>8} {'sharpe':>7} {'taken':>6} {'avgGr':>6} {'maxGr':>6} "
          f"{'Δret':>8} {'ΔDD':>7}", flush=True)
    for r in rows:
        print(f"  {r['variant']:14} {r['total_return_pct']:>9} {r['max_dd_pct']:>8} {r['sharpe']:>7} "
              f"{r['n_taken']:>6} {r['avg_gross']:>6} {r['max_gross']:>6} {r['ret_vs_base_pp']:>+8} "
              f"{r['dd_vs_base_pp']:>+7}", flush=True)
        print(f"     skip cap/gate: {r['n_skipped_cap']}/{r['n_skipped_gate']}  worst DD {r['dd_start']}->{r['dd_end']}", flush=True)


if __name__ == "__main__":
    main()
