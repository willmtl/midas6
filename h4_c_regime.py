#!/usr/bin/env python3
"""Investigate the H4-on-C dip-buy's LOSING STREAKS and test filter rules to cut them.

The conviction backtest's max drawdown (~-31%) comes from clusters of ~10 losing dip-buys in a row — a
per-trade stop can't cap that (it's a portfolio equity concept). Hypothesis (from prior research): the
losers cluster in (a) MARKET DOWNTRENDS — buying dips into a falling tape = falling knives — and (b) the
EXTREME-DIP tail. This tags every trade by SPY trend + dip extremity to SHOW where the losses live, then
tests candidate FILTER RULES (skip the dip in a bear tape / skip falling knives) on the winning
steep_2x / combined config. -> BacktestResult[h4_c_regime].
Run: MSYS_NO_PATHCONV=1 docker exec rotation-backend-1 python -u /app/h4_c_regime.py
"""
import os, json, warnings
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
import ta
import h4_study as H
import h4_c_conviction as C
from h4_c_upside import load_targets, upside_asof, bucket_upside


def _spy_regime():
    """SPY daily regime as-of each date: above 200d MA, 50d MA rising, 20d return. Sorted-date asof lookup."""
    from seq_fundamental_study import load_candles
    spy = load_candles(["SPY"]).get("SPY")
    if spy is None or len(spy) < 210:
        return None
    c = spy["Close"]
    df = pd.DataFrame({
        "above200": (c > c.rolling(200).mean()),
        "ma50_up": (c.rolling(50).mean() > c.rolling(50).mean().shift(10)),
        "ret20": (c / c.shift(20) - 1) * 100,
    })
    df.index = pd.to_datetime(spy.index).normalize()
    return df


def _asof(regdf, d):
    """Most-recent SPY regime row on/before date d (a python date)."""
    ts = pd.Timestamp(d)
    sub = regdf.loc[:ts]
    if len(sub) == 0:
        return None
    return sub.iloc[-1]


def collect(years, allow_fetch):
    """Trades with C._replay's path fields PLUS tags: entry_date, r3, market (bull/bear), extremity."""
    from intraday_data import get_4h
    from h4_on_signals_study import candidate_windows
    store = load_targets()
    regdf = _spy_regime()
    allowed, cwmeta = candidate_windows("C")
    trades, span = [], []
    for tk in sorted(allowed):
        df = get_4h(tk, years, allow_fetch)
        if df is None or len(df) < 120:
            continue
        close = df["Close"].values
        low = df["Low"].values if "Low" in df else close
        openv = df["Open"].values if "Open" in df else close
        ts = df.index
        n = len(close)
        ad = allowed[tk]
        rsi = ta.momentum.rsi(df["Close"], window=14).values
        span.append(ts[0]); span.append(ts[-1])
        for i in C._entries(df, C.COMBINED):
            if ts[i].date() not in ad:
                continue
            ep = float(close[i])
            if ep <= 0 or i + C.HOLD_BARS >= n:
                continue
            u = upside_asof(store, tk, ts[i].date(), ep)
            gap = (openv[i] / close[i - 1] - 1) * 100 if i >= 1 else 0.0
            nd = 0
            k = i
            while k >= 1 and close[k] < close[k - 1]:
                nd += 1; k -= 1
            knife = bool(gap <= -4 or nd >= 5)          # falling-knife def (h4-on-signals memory)
            reg = _asof(regdf, ts[i].date()) if regdf is not None else None
            bull = bool(reg["above200"]) if reg is not None else None
            ma50_up = bool(reg["ma50_up"]) if reg is not None else None
            path = [(ts[i + b], float(close[i + b - 1]), float(close[i + b]), float(low[i + b]))
                    for b in range(1, C.HOLD_BARS + 1)]
            r3 = (close[i + C.HOLD_BARS] - ep) / ep * 100
            trades.append({"bucket": bucket_upside(u), "ep": ep, "path": path,
                           "entry_date": str(ts[i].date()), "r3": r3, "rsi": None if np.isnan(rsi[i]) else round(float(rsi[i]), 1),
                           "gap": round(gap, 2), "ndown": nd, "knife": knife,
                           "bull": bull, "ma50_up": ma50_up})
    return trades, span, cwmeta


def _grp(trades, pred):
    sub = [t for t in trades if pred(t)]
    if not sub:
        return {"n": 0, "win_pct": None, "avg_r3": None}
    r = np.array([t["r3"] for t in sub])
    return {"n": len(sub), "win_pct": round(float((r > 0).mean() * 100), 1), "avg_r3": round(float(r.mean()), 3)}


def _max_loss_streak(trades):
    """Longest run of consecutive losing trades in entry-date order + the run's date span."""
    ss = sorted(trades, key=lambda t: t["entry_date"])
    best = cur = 0
    best_span = cur_start = None
    for t in ss:
        if t["r3"] <= 0:
            if cur == 0:
                cur_start = t["entry_date"]
            cur += 1
            if cur > best:
                best = cur; best_span = (cur_start, t["entry_date"])
        else:
            cur = 0
    return best, best_span


def main():
    os.environ.setdefault("DJANGO_SETTINGS_MODULE", "rotation.settings")
    import django; django.setup()
    from pathlib import Path
    trades, span, cwmeta = collect(5, False)
    steep = C.SCHEMES["steep_2x"]
    eligible = [t for t in trades if steep.get(t["bucket"], 0) > 0]   # what steep_2x actually trades

    # 1) where do the losses live?
    diag = {
        "by_market": {
            "bull (SPY>200MA)": _grp(eligible, lambda t: t["bull"] is True),
            "bear (SPY<200MA)": _grp(eligible, lambda t: t["bull"] is False),
        },
        "by_ma50": {
            "ma50 rising": _grp(eligible, lambda t: t["ma50_up"] is True),
            "ma50 falling": _grp(eligible, lambda t: t["ma50_up"] is False),
        },
        "by_extremity": {
            "normal dip": _grp(eligible, lambda t: not t["knife"]),
            "falling knife (gap<-4% or >=5 down)": _grp(eligible, lambda t: t["knife"]),
        },
    }

    # 2) filter rules on steep_2x / combined
    FILTERS = {
        "none (baseline)": lambda t: True,
        "bull_only (SPY>200MA)": lambda t: t["bull"] is True,
        "ma50_up": lambda t: t["ma50_up"] is True,
        "not_knife": lambda t: not t["knife"],
        "bull_and_not_knife": lambda t: (t["bull"] is True) and (not t["knife"]),
    }
    rows = []
    for name, pred in FILTERS.items():
        sub = [t for t in eligible if pred(t)]
        m = C._replay(sub, steep, None)
        streak, sspan = _max_loss_streak(sub)
        rows.append({"filter": name, **m, "max_loss_streak": streak, "streak_span": sspan})
    rows.sort(key=lambda r: -r["total_return_pct"])

    payload = {
        "computed_at": pd.Timestamp.utcnow().isoformat(),
        "scheme": "steep_2x", "signals": C.COMBINED, "hold_bars": C.HOLD_BARS,
        "n_eligible_trades": len(eligible), "diagnostics": diag, "filters": rows,
        "note": ("Where the H4-on-C dip-buy loses, and filters to cut the losing streaks. Trades = the "
                 "steep_2x/combined config; tagged by SPY 200d-MA regime + 50d-MA slope + dip extremity "
                 "(falling knife = gap<-4% or >=5 consecutive down bars). Filters re-run the weighted "
                 "portfolio on the surviving trades. Gross of fees; current-membership; cached-4h C names."),
    }
    STUD = Path(__file__).resolve().parent / ".data" / "studies"
    STUD.mkdir(parents=True, exist_ok=True)
    (STUD / "h4_c_regime.json").write_text(json.dumps(payload, indent=2, default=str))
    try:
        from core.models import BacktestResult
        from django.utils import timezone
        BacktestResult.objects.update_or_create(
            kind="h4_c_regime",
            defaults={"payload": json.loads(json.dumps(payload, default=str)), "computed_at": timezone.now()})
        print("saved BacktestResult[h4_c_regime]", flush=True)
    except Exception as e:
        print("DB save failed:", e, flush=True)

    print(f"\n=== WHERE THE LOSSES LIVE (steep_2x/combined, {len(eligible)} trades) ===", flush=True)
    for grp, d in diag.items():
        print(f"  {grp}:", flush=True)
        for k, v in d.items():
            print(f"     {k:40} n={v['n']:>5}  win {v['win_pct']}%  avg {v['avg_r3']}%", flush=True)
    print(f"\n=== FILTER RULES (ranked by total return) ===", flush=True)
    print(f"  {'filter':26} {'total%':>9} {'sharpe':>7} {'maxDD%':>8} {'trades':>7} {'streak':>7}  worst-streak span", flush=True)
    for r in rows:
        sp = f"{r['streak_span'][0]}..{r['streak_span'][1]}" if r["streak_span"] else "-"
        print(f"  {r['filter']:26} {r['total_return_pct']:>9} {r['sharpe']:>7} {r['max_dd_pct']:>8} "
              f"{r['n_trades']:>7} {r['max_loss_streak']:>7}  {sp}", flush=True)
        if r["dd_start"]:
            print(f"     worst drawdown window: {r['dd_start'][:10]} -> {r['dd_end'][:10]}", flush=True)


if __name__ == "__main__":
    main()
