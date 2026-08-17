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


def _spy_daily(years=5, allow_fetch=False):
    """SPY daily stress features keyed by date: 1-day return AND trailing 3-day return (the multi-day
    cascade measure). Macro air-pockets (2022-08-26 -3.4%; the 5-day Powell grind) trigger the gate."""
    from seq_fundamental_study import load_candles
    spy = load_candles(["SPY"]).get("SPY")
    if spy is None or len(spy) < 4:
        return {}
    c = spy["Close"]
    r1 = (c / c.shift(1) - 1) * 100
    r3 = (c / c.shift(3) - 1) * 100
    idx = pd.to_datetime(spy.index)
    out = {}
    for ts, a, b in zip(idx, r1.values, r3.values):
        out[ts.date()] = {"ret1d": float(a) if pd.notna(a) else 0.0,
                          "ret3d": float(b) if pd.notna(b) else 0.0}
    return out


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


def simulate(trades, spy, cap=False, gate=None, breaker=False):
    """gate: None, or a dict of stress rules — {day: %, stress3d: %, cooldown_days: int}. `day` skips new
    adds when SPY's 1-day return <= -day; `stress3d` skips while the trailing 3-day return <= -stress3d
    (the multi-day cascade); `cooldown_days` keeps the gate shut for N days AFTER a trigger (stops re-entering
    into the aftermath). Any rule firing blocks adds that bar. Legacy: gate=True -> {day: SELLOFF_THR}."""
    if gate is True:
        gate = {"day": SELLOFF_THR}
    import datetime as _dt
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
    taken_pnl = []                 # (entry_ts, realized 3-bar return) for the loss-streak metric
    cooldown_until = None
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
        d = pd.Timestamp(t).date()
        if gate and news:
            info = spy.get(d, {})
            trig = ((gate.get("day") and info.get("ret1d", 0.0) <= -gate["day"]) or
                    (gate.get("stress3d") and info.get("ret3d", 0.0) <= -gate["stress3d"]))
            if trig and gate.get("cooldown_days"):
                cooldown_until = d + _dt.timedelta(days=int(gate["cooldown_days"]))
            if trig or (cooldown_until is not None and d <= cooldown_until):
                n_skipped_gate += len(news); news = []
        news = sorted(news, key=lambda x: -x["weight"])
        if cap:
            room = max(0, CAP_N - len(open_pos))
            if len(news) > room:
                n_skipped_cap += len(news) - room
                news = news[:room]
        for tr in news:
            open_pos.append({"size": tr["weight"] * UNIT * mult, "sched": deque(tr["sched"])})
            taken_pnl.append((tr["entry_ts"], float(np.prod([1 + x for _, x in tr["sched"]]) - 1)))
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
    # longest run of consecutive LOSING taken trades (the sequential-cascade metric) + its date span
    streak = best = 0; span = start = None
    for (ets, r) in sorted(taken_pnl, key=lambda x: x[0]):
        if r <= 0:
            if streak == 0:
                start = ets
            streak += 1
            if streak > best:
                best = streak; span = (str(start)[:10], str(ets)[:10])
        else:
            streak = 0
    return {"total_return_pct": total, "sharpe": sharpe, "max_dd_pct": maxdd,
            "n_taken": n_taken, "n_skipped_cap": n_skipped_cap, "n_skipped_gate": n_skipped_gate,
            "max_loss_streak": best, "streak_span": span,
            "avg_gross": round(float(gr[gr > 0].mean()) if (gr > 0).any() else 0.0, 2),
            "max_gross": round(float(gr.max()) if len(gr) else 0.0, 2),
            "dd_start": str(eq_ts[peak_i])[:10], "dd_end": str(eq_ts[trough_i])[:10]}


def main():
    os.environ.setdefault("DJANGO_SETTINGS_MODULE", "rotation.settings")
    import django; django.setup()
    from pathlib import Path
    trades, cwmeta = collect(5, False)
    spy = _spy_daily(5, False)
    variants = {
        "baseline": simulate(trades, spy),
        "cap6": simulate(trades, spy, cap=True),
        "selloff_gate": simulate(trades, spy, gate={"day": SELLOFF_THR}),
        "breaker15": simulate(trades, spy, breaker=True),
        "all_three": simulate(trades, spy, cap=True, gate={"day": SELLOFF_THR}, breaker=True),
    }
    base = variants["baseline"]
    # STRESS-PERSISTENCE GATE SWEEP — the fix for SEQUENTIAL cascades: detect the multi-day slide and stop
    # re-entering until it clears (fast re-engage, so calm-bear-market trades survive).
    gate_defs = {
        "day1.5 (winner so far)": {"day": 1.5},
        "day1.5 + 3d cooldown": {"day": 1.5, "cooldown_days": 3},
        "stress3d<=-4%": {"stress3d": 4.0},
        "day1.5 OR stress3d<=-4%": {"day": 1.5, "stress3d": 4.0},
        "day1.5 OR stress3d, +3d cooldown": {"day": 1.5, "stress3d": 4.0, "cooldown_days": 3},
        "day1.5 OR stress3d<=-3%, +5d cooldown": {"day": 1.5, "stress3d": 3.0, "cooldown_days": 5},
    }
    gate_rows = [{"gate": "none (baseline)", **base,
                  "ret_vs_base_pp": 0.0, "dd_vs_base_pp": 0.0}]
    for nm, gd in gate_defs.items():
        m = simulate(trades, spy, gate=gd)
        gate_rows.append({"gate": nm, **m,
                          "ret_vs_base_pp": round(m["total_return_pct"] - base["total_return_pct"], 1),
                          "dd_vs_base_pp": round(m["max_dd_pct"] - base["max_dd_pct"], 1)})
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
        "n_eligible_trades": len(trades), "rows": rows, "gate_sweep": gate_rows,
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
    print(f"\n=== STRESS-PERSISTENCE GATE SWEEP (the fix for sequential cascades) ===", flush=True)
    print(f"  {'gate':38} {'total%':>9} {'maxDD%':>8} {'sharpe':>7} {'lossStrk':>8} {'skip':>6} {'Δret':>8} {'ΔDD':>7}", flush=True)
    for r in gate_rows:
        print(f"  {r['gate']:38} {r['total_return_pct']:>9} {r['max_dd_pct']:>8} {r['sharpe']:>7} "
              f"{r['max_loss_streak']:>8} {r['n_skipped_gate']:>6} {r['ret_vs_base_pp']:>+8} {r['dd_vs_base_pp']:>+7}", flush=True)
        if r.get("streak_span"):
            print(f"     worst loss streak span: {r['streak_span'][0]}..{r['streak_span'][1]}  worst DD {r['dd_start']}->{r['dd_end']}", flush=True)


if __name__ == "__main__":
    main()
