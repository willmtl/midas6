#!/usr/bin/env python3
"""Enhancement sweep on the H4-on-C dip-buy — six levers to push return / cut drawdown, all obeying the
one rule that held all session: NEVER blanket-reduce exposure — size up on the best, skip bad entries, or
HEDGE the correlated component (don't sit in cash).

Levers tested (each vs the current best baseline = steep_2x, hold 3b, gate day1.5-OR-stress3d):
  1. conviction curve   — steeper / continuous size = f(analyst upside)
  2. conditional hold    — let high-upside dips run past 3 bars
  3. more entries        — fold the gap_dn->rsi_x combo into the pool
  4. SPY hedge in stress — stay long the dips, short SPY on stress days (neutralize macro beta)
  5. calm-regime leverage— gross > 1x when the tape is calm (the only exposure move that fits the rule)
  6. per-sector cap      — cap concurrent names per sector (attack the idiosyncratic streak)
Cash-aware sequential sim (its own copy of the h4_c_portfolio engine so the committed one is untouched).
-> BacktestResult[h4_c_enhance]. Run: MSYS_NO_PATHCONV=1 docker exec rotation-backend-1 python -u /app/h4_c_enhance.py
"""
import os, json, warnings, datetime as dt
from collections import deque, defaultdict
warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
import h4_study as H
import h4_c_conviction as C
from h4_c_upside import load_targets, upside_asof, bucket_upside

UNIT = 0.15
BASE_GROSS = 1.0
MAXHOLD = 10
BASE_SIGS = ["mr_rsi_os", "mr_newlow60", "mr_ndown"]
SELLOFF_THR, STRESS3D_THR = 1.5, 4.0
WEIGHTS = {
    "steep_2x": {"<0%": 0, "0-25%": 1, "25-50%": 2, "50-100%": 4, ">100%": 8, "no_target": 1},
    "steep_3x": {"<0%": 0, "0-25%": 1, "25-50%": 3, "50-100%": 9, ">100%": 27, "no_target": 1},
    "steep_4x": {"<0%": 0, "0-25%": 1, "25-50%": 4, "50-100%": 16, ">100%": 64, "no_target": 1},
    "linear":   {"<0%": 0, "0-25%": 1, "25-50%": 2, "50-100%": 3, ">100%": 4, "no_target": 1},
}
HOLD_FIXED = {b: 3 for b in ["<0%", "0-25%", "25-50%", "50-100%", ">100%", "no_target"]}
HOLD_COND = {"<0%": 3, "0-25%": 3, "25-50%": 3, "50-100%": 6, ">100%": 10, "no_target": 3}


def _sector_map():
    from core.models import BacktestResult
    try:
        p = BacktestResult.objects.get(kind="rotation_history").payload
    except Exception:
        return {}
    m = {}
    for mo in p["months"]:
        for pk in mo["picks"]:
            if pk.get("pick") and pk.get("sector"):
                m[pk["pick"]] = pk["sector"]
    return m


def _cont_weight(upside):
    if upside is None:
        return 1.0
    return float(max(0.0, min(8.0, upside / 12.5)))       # 25%->2, 50%->4, 100%->8; <0 -> 0


def collect(sigs, years=5, allow_fetch=False):
    from intraday_data import get_4h
    from h4_on_signals_study import candidate_windows
    from h4_c_indicators import combo_entries
    store = load_targets()
    sectors = _sector_map()
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
        fire = np.zeros(n, dtype=bool)
        for s in sigs:
            if s == "gap_dn__rsi_x":
                fire |= np.asarray(combo_entries(df).get("gap_dn__rsi_x"), dtype=bool)
            else:
                e, _ = H.SIGNALS[s]["fn"](df); fire |= np.asarray(e, dtype=bool)
        idxs = sorted(H._episode_starts([i for i in range(n) if fire[i]], gap=H.GAP))
        for i in idxs:
            if ts[i].date() not in ad or i + 1 >= n:
                continue
            ep = float(close[i])
            if ep <= 0:
                continue
            up = upside_asof(store, tk, ts[i].date(), ep)
            sched = [(ts[i + b], float(close[i + b] / close[i + b - 1] - 1))
                     for b in range(1, MAXHOLD + 1) if i + b < n]
            if not sched:
                continue
            trades.append({"entry_ts": ts[i], "bucket": bucket_upside(up), "upside": up,
                           "sector": sectors.get(tk, tk), "sched": sched})
    return trades, cwmeta


def _spy(years=5):
    """SPY daily stress (1d,3d) keyed by date, and SPY 4h bar returns keyed by ts (for the hedge)."""
    from seq_fundamental_study import load_candles
    from intraday_data import get_4h
    spy = load_candles(["SPY"]).get("SPY")
    daily = {}
    if spy is not None and len(spy) > 4:
        c = spy["Close"]; r1 = (c / c.shift(1) - 1) * 100; r3 = (c / c.shift(3) - 1) * 100
        for tsx, a, b in zip(pd.to_datetime(spy.index), r1.values, r3.values):
            daily[tsx.date()] = {"r1": float(a) if pd.notna(a) else 0.0, "r3": float(b) if pd.notna(b) else 0.0}
    s4 = get_4h("SPY", years, True)                       # fetch 4h for the hedge leg
    bar = {}
    if s4 is not None:
        cc = s4["Close"]
        for tsx, v in ((cc / cc.shift(1) - 1)).items():
            if pd.notna(v):
                bar[tsx] = float(v)
    return daily, bar


TARGET_VOL = 60.0        # vol-parity target (% annualized): size = TARGET_VOL / name_vol, capped
VOLPARITY_CAP = 4.0


def _stressed(daily, d):
    info = daily.get(d, {})
    return info.get("r1", 0.0) <= -SELLOFF_THR or info.get("r3", 0.0) <= -STRESS3D_THR


def _weight_of(tr, mode, wmap):
    """Position weight for a trade under a sizing mode."""
    if mode == "continuous":
        return _cont_weight(tr["upside"])
    if mode == "vol_parity":                     # risk-parity: down-size the wildest names
        v = tr.get("vol")
        return 0.0 if (not v or v <= 0) else float(min(VOLPARITY_CAP, TARGET_VOL / v))
    if mode == "vol_parity_x_conv":              # risk-parity scaled by conviction (upside)
        v = tr.get("vol")
        base = 0.0 if (not v or v <= 0) else float(min(VOLPARITY_CAP, TARGET_VOL / v))
        return base * (WEIGHTS["steep_2x"].get(tr["bucket"], 0) / 2.0)
    return wmap.get(tr["bucket"], 0)


def simulate(trades, daily, spybar, cfg):
    """cfg: weight('steep_2x'|...|'continuous'), hold(dict), gate(bool), hedge_frac(float),
    calm_lev(float|None gross cap when calm), sector_cap(int|None)."""
    mode = cfg["weight"]
    wmap = WEIGHTS.get(mode)                      # None for continuous / vol_parity modes
    hold = cfg["hold"]
    ebt = defaultdict(list); allts = set()
    for tr in trades:
        ebt[tr["entry_ts"]].append(tr); allts.add(tr["entry_ts"])
        for (tk, _) in tr["sched"]:
            allts.add(tk)
    timeline = sorted(allts)
    equity = peak = 1.0
    open_pos = []
    eq_ts, eq_val, taken_pnl = [], [], []
    n_taken = 0
    cost_rate = cfg.get("cost_bps", 0) / 1e4      # per-side transaction cost (bps of notional)
    prev_stressed = False
    for t in timeline:
        d = pd.Timestamp(t).date()
        stressed = _stressed(daily, d)
        gross_now = sum(p["size"] for p in open_pos)
        gcap = BASE_GROSS
        if cfg.get("calm_lev") and not stressed:
            gcap = cfg["calm_lev"]
        scale = min(1.0, gcap / gross_now) if gross_now > 0 else 1.0
        contrib = 0.0; still = []
        exit_notional = 0.0
        for p in open_pos:
            if p["sched"] and p["sched"][0][0] == t:
                contrib += p["size"] * scale * p["sched"].popleft()[1]
                if not p["sched"]:
                    exit_notional += p["size"]        # position fully closes this bar -> exit cost
            if p["sched"]:
                still.append(p)
        open_pos = still
        # SPY/QQQ hedge leg: short hedge_frac of (capped) gross on stress days
        if cfg.get("hedge_frac") and stressed and t in spybar:
            eff_gross = min(gross_now, gcap)
            contrib += -cfg["hedge_frac"] * eff_gross * spybar[t]
        # transaction costs: dip-buy exits this bar + hedge turnover when stress toggles (put on / take off)
        cost = exit_notional * cost_rate
        if cost_rate and cfg.get("hedge_frac") and (stressed != prev_stressed):
            cost += cfg["hedge_frac"] * min(gross_now, gcap) * cost_rate
        prev_stressed = stressed
        if contrib or cost:
            equity *= (1 + contrib - cost)
        peak = max(peak, equity)
        eq_ts.append(t); eq_val.append(equity)
        # new entries
        news = ebt.get(t, [])
        if cfg.get("gate") and news and stressed:
            news = []
        news = sorted(news, key=lambda x: -_weight_of(x, mode, wmap))
        if cfg.get("sector_cap"):
            per = defaultdict(int)
            for p in open_pos:
                per[p["sector"]] += 1
        open_notional = 0.0
        for tr in news:
            w = _weight_of(tr, mode, wmap)
            if w <= 0:
                continue
            if cfg.get("sector_cap"):
                if per[tr["sector"]] >= cfg["sector_cap"]:
                    continue
                per[tr["sector"]] += 1
            h = hold.get(tr["bucket"], 3)
            sz = w * UNIT
            open_pos.append({"size": sz, "sched": deque(tr["sched"][:h]), "sector": tr["sector"]})
            taken_pnl.append((tr["entry_ts"], float(np.prod([1 + x for _, x in tr["sched"][:h]]) - 1)))
            open_notional += sz
            n_taken += 1
        if cost_rate and open_notional:               # entry cost of new positions
            equity *= (1 - open_notional * cost_rate)
            eq_val[-1] = equity
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
    dd_start = str(eq_ts[peak_i])[:10] if eq_ts else None
    dd_end = str(eq_ts[trough_i])[:10] if eq_ts else None
    span_start = str(eq_ts[0])[:10] if eq_ts else None      # earliest bar in the sim (data span)
    # loss streak
    best = strk = 0
    for _, r in sorted(taken_pnl):
        strk = strk + 1 if r <= 0 else 0
        best = max(best, strk)
    return {"total_return_pct": total, "sharpe": sharpe, "max_dd_pct": maxdd,
            "n_taken": n_taken, "max_loss_streak": best,
            "dd_start": dd_start, "dd_end": dd_end, "data_start": span_start}


def main():
    os.environ.setdefault("DJANGO_SETTINGS_MODULE", "rotation.settings")
    import django; django.setup()
    from pathlib import Path
    daily, spybar = _spy(5)
    base_trades, cw = collect(BASE_SIGS, 5, False)
    exp_trades, _ = collect(BASE_SIGS + ["gap_dn__rsi_x"], 5, False)
    GATE = True
    B = {"weight": "steep_2x", "hold": HOLD_FIXED, "gate": GATE}
    variants = {
        "baseline (steep_2x, 3b, gate)": (base_trades, B),
        "1a conviction steep_3x":        (base_trades, {**B, "weight": "steep_3x"}),
        "1b conviction steep_4x":        (base_trades, {**B, "weight": "steep_4x"}),
        "1c conviction continuous":      (base_trades, {**B, "weight": "continuous"}),
        "2 conditional hold (hi runs)":  (base_trades, {**B, "hold": HOLD_COND}),
        "3 + gap_dn__rsi_x entries":     (exp_trades,  B),
        "4a SPY hedge 25% (no gate)":    (base_trades, {**B, "gate": False, "hedge_frac": 0.25}),
        "4b SPY hedge 50% (no gate)":    (base_trades, {**B, "gate": False, "hedge_frac": 0.5}),
        "4c SPY hedge 50% + gate":       (base_trades, {**B, "hedge_frac": 0.5}),
        "5 calm leverage 1.5x":          (base_trades, {**B, "calm_lev": 1.5}),
        "6 per-sector cap 2":            (base_trades, {**B, "sector_cap": 2}),
    }
    rows = []
    base_m = None
    for name, (trs, cfg) in variants.items():
        m = simulate(trs, daily, spybar, cfg)
        if name.startswith("baseline"):
            base_m = m
        rows.append({"variant": name, **m})
    # SMART STACK = the winners only (drop the refuted cond-hold; hedge REPLACES the gate).
    rows.append({"variant": "SMART: steep_4x + hedge50 (no gate)",
                 **simulate(base_trades, daily, spybar, {"weight": "steep_4x", "hold": HOLD_FIXED, "gate": False, "hedge_frac": 0.5})})
    rows.append({"variant": "SMART+lev: steep_4x + hedge50 + calm1.5",
                 **simulate(base_trades, daily, spybar, {"weight": "steep_4x", "hold": HOLD_FIXED, "gate": False, "hedge_frac": 0.5, "calm_lev": 1.5})})
    rows.append({"variant": "SMART-cons: steep_3x + hedge50 (no gate)",
                 **simulate(base_trades, daily, spybar, {"weight": "steep_3x", "hold": HOLD_FIXED, "gate": False, "hedge_frac": 0.5})})
    for r in rows:
        r["ret_vs_base_pp"] = round(r["total_return_pct"] - base_m["total_return_pct"], 1)
        r["dd_vs_base_pp"] = round(r["max_dd_pct"] - base_m["max_dd_pct"], 1)
    payload = {"computed_at": pd.Timestamp.utcnow().isoformat(), "rows": rows,
               "n_base_trades": len(base_trades), "n_exp_trades": len(exp_trades), "candidates": cw,
               "spy_4h_bars": len(spybar),
               "note": ("Six enhancement levers on the H4-on-C dip-buy vs the current best baseline (steep_2x, "
                        "3b hold, day1.5-OR-stress3d gate), cash-aware 1x sim. Hedge = short SPY on stress days "
                        "(4h leg). Calm leverage lifts gross to 1.5x when not stressed. Gross of fees; "
                        "current-membership; cached-4h. STACK combines the levers.")}
    Path("/app/.data/studies").mkdir(parents=True, exist_ok=True)
    Path("/app/.data/studies/h4_c_enhance.json").write_text(json.dumps(payload, indent=2, default=str))
    try:
        from core.models import BacktestResult
        from django.utils import timezone
        BacktestResult.objects.update_or_create(kind="h4_c_enhance",
            defaults={"payload": json.loads(json.dumps(payload, default=str)), "computed_at": timezone.now()})
        print("saved BacktestResult[h4_c_enhance]", flush=True)
    except Exception as e:
        print("DB save failed:", e, flush=True)
    print(f"\n=== ENHANCEMENT SWEEP (base {len(base_trades)} / exp {len(exp_trades)} trades, spy4h {len(spybar)}) ===", flush=True)
    print(f"  {'variant':44}{'total%':>9}{'maxDD%':>8}{'sharpe':>7}{'strk':>5}{'Δret':>8}{'ΔDD':>7}", flush=True)
    for r in rows:
        print(f"  {r['variant']:44}{r['total_return_pct']:>9}{r['max_dd_pct']:>8}{r['sharpe']:>7}"
              f"{r['max_loss_streak']:>5}{r['ret_vs_base_pp']:>+8}{r['dd_vs_base_pp']:>+7}", flush=True)


if __name__ == "__main__":
    main()
