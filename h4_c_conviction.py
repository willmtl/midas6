#!/usr/bin/env python3
"""Backtest CONVICTION-WEIGHTED position sizing (+ STOP-LOSSES) on the H4-on-C dip-buy.

The h4_c_upside finding: the H4 oversold dip on C value names is monotone in analyst implied-upside
(+0.26%/3b at 0-25% upside -> +1.07% at >100%). So instead of trading every dip the same size, SIZE THE
POSITION by conviction (the upside bucket) and see if it lifts total return — the same div_2x logic the
daily flagship already uses, applied to the intraday add.

Method: take the H4 oversold entries inside the point-in-time C windows (h4_on_signals candidate_windows),
precompute each trade's bar path, then replay under (a) each weighting scheme and (b) each stop-loss level,
building a conviction-weighted portfolio equity curve (at each 4h bar the return is the weight-weighted
average of the currently-open dip trades). Stops are intrabar: if a bar's LOW breaches entry*(1-stop) the
trade exits at the stop price that bar. Compare on TOTAL RETURN (return-priority) + Sharpe / max-DD.
-> BacktestResult[h4_c_conviction] + JSON.
Run: MSYS_NO_PATHCONV=1 docker exec rotation-backend-1 python -u /app/h4_c_conviction.py [--fetch]
"""
import os, json, warnings
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
import h4_study as H
from h4_c_upside import load_targets, upside_asof, bucket_upside, UPSIDE_BUCKETS

HOLD_BARS = 3                          # the studied 0-3 day sweet spot
BARS_PER_YEAR = 500                    # ~2 4h bars/day x ~252 US sessions
PRIMARY_SIG = "mr_rsi_os"              # the clean monotone entry
COMBINED = ["mr_rsi_os", "mr_newlow60", "mr_ndown"]   # the monotone oversold cluster

# Weighting schemes: bucket-label -> position weight. Buckets in UPSIDE_BUCKETS order + no_target.
BUCKETS = [b[0] for b in UPSIDE_BUCKETS] + ["no_target"]
SCHEMES = {
    "equal":     {"<0%": 1, "0-25%": 1, "25-50%": 1, "50-100%": 1, ">100%": 1, "no_target": 1},
    "skip_neg":  {"<0%": 0, "0-25%": 1, "25-50%": 1, "50-100%": 1, ">100%": 1, "no_target": 1},
    "linear":    {"<0%": 0, "0-25%": 1, "25-50%": 2, "50-100%": 3, ">100%": 4, "no_target": 1},
    "steep_2x":  {"<0%": 0, "0-25%": 1, "25-50%": 2, "50-100%": 4, ">100%": 8, "no_target": 1},
    "hi_only":   {"<0%": 0, "0-25%": 0, "25-50%": 0, "50-100%": 1, ">100%": 1, "no_target": 0},
}
# Stop-loss levels swept on the winning scheme (fraction below entry); None = fixed-bar hold, no stop.
STOP_LEVELS = [None, 0.10, 0.08, 0.05, 0.03, 0.02]
WIN_SCHEME = "steep_2x"


def _entries(df, sigs):
    """Union of the given oversold signals' entry bars on one 4h frame, episode-deduped (gap=3)."""
    n = len(df)
    fire = np.zeros(n, dtype=bool)
    for s in sigs:
        e, _mag = H.SIGNALS[s]["fn"](df)
        fire |= np.asarray(e, dtype=bool)
    cand = [i for i in range(n) if fire[i]]
    return sorted(H._episode_starts(cand, gap=H.GAP))


def _collect(years, allow_fetch, sigs):
    """One pass over the C names: precompute each dip trade's bar path so schemes/stops replay in memory.
    trade = {bucket, ep, path:[(ts, prev_close, close, low)]}. Returns (trades, span_ts, n_names)."""
    from intraday_data import get_4h
    from h4_on_signals_study import candidate_windows
    store = load_targets()
    allowed, cwmeta = candidate_windows("C")
    trades, span_ts, got = [], [], 0
    for tk in sorted(allowed):
        df = get_4h(tk, years, allow_fetch)
        if df is None or len(df) < 120:
            continue
        got += 1
        close = df["Close"].values
        low = df["Low"].values if "Low" in df else close
        ts = df.index
        n = len(close)
        ad = allowed[tk]
        span_ts.append(ts[0]); span_ts.append(ts[-1])
        for i in _entries(df, sigs):
            if ts[i].date() not in ad:            # only while the name was actually a C pick (point-in-time)
                continue
            ep = float(close[i])
            if ep <= 0 or i + HOLD_BARS >= n:
                continue
            u = upside_asof(store, tk, ts[i].date(), ep)
            path = [(ts[i + b], float(close[i + b - 1]), float(close[i + b]), float(low[i + b]))
                    for b in range(1, HOLD_BARS + 1)]
            trades.append({"bucket": bucket_upside(u), "ep": ep, "path": path})
    return trades, span_ts, got, cwmeta


def _replay(trades, wmap, stop):
    """Replay precomputed trades under a weight map + stop level -> portfolio metrics.
    Stop is intrabar: if a bar's low <= ep*(1-stop), exit at the stop price that bar."""
    bar_map = {}                 # ts -> [sum_w_ret, sum_w]
    realized = []                # (weight, per-trade realized return %)
    n_stopped = n_used = 0
    for tr in trades:
        w = wmap.get(tr["bucket"], 0)
        if w <= 0:
            continue
        n_used += 1
        ep = tr["ep"]
        stop_px = ep * (1 - stop) if stop else None
        exit_ret = None
        for (ts_k, prev_close, close_k, low_k) in tr["path"]:
            if stop_px is not None and low_k <= stop_px:
                br = stop_px / prev_close - 1          # partial bar down to the stop fill
                cur = bar_map.get(ts_k)
                if cur is None:
                    bar_map[ts_k] = [w * br, w]
                else:
                    cur[0] += w * br; cur[1] += w
                exit_ret = (stop_px / ep - 1) * 100
                n_stopped += 1
                break
            br = close_k / prev_close - 1              # close-to-close
            cur = bar_map.get(ts_k)
            if cur is None:
                bar_map[ts_k] = [w * br, w]
            else:
                cur[0] += w * br; cur[1] += w
            exit_ret = (close_k / ep - 1) * 100        # keep last -> full-hold exit if never stopped
        if exit_ret is not None:
            realized.append((w, exit_ret))
    # metrics
    if bar_map:
        tsx = sorted(bar_map)
        rets = np.array([bar_map[t][0] / bar_map[t][1] for t in tsx if bar_map[t][1] > 0])
    else:
        rets = np.array([])
    if len(rets):
        eq = np.cumprod(1 + rets)
        total = round((eq[-1] - 1) * 100, 1)
        sd = rets.std(ddof=1) if len(rets) > 1 else 0
        sharpe = round(float(rets.mean() / sd * np.sqrt(BARS_PER_YEAR)), 2) if sd > 0 else 0.0
        peak = np.maximum.accumulate(eq)
        maxdd = round(float(((eq - peak) / peak).min() * 100), 1)
    else:
        total, sharpe, maxdd = 0.0, 0.0, 0.0
    wnum = sum(w * r for w, r in realized); wden = sum(w for w, _ in realized)
    wmean = round(wnum / wden, 3) if wden else 0.0
    pct_stopped = round(100 * n_stopped / n_used, 1) if n_used else 0.0
    return {"total_return_pct": total, "sharpe": sharpe, "max_dd_pct": maxdd,
            "n_trades": n_used, "n_active_bars": len(rets), "weighted_mean_3b": wmean,
            "pct_stopped": pct_stopped}


def _spy(years, allow_fetch, span_ts):
    from intraday_data import get_4h
    try:
        spy = get_4h("SPY", years, allow_fetch)
        if spy is not None and span_ts:
            lo, hi = min(span_ts), max(span_ts)
            s = spy[(spy.index >= lo) & (spy.index <= hi)]["Close"]
            if len(s) > 1:
                return round((float(s.iloc[-1]) / float(s.iloc[0]) - 1) * 100, 1)
    except Exception:
        pass
    return None


def variant(years, allow_fetch, sigs, label):
    trades, span_ts, got, cwmeta = _collect(years, allow_fetch, sigs)
    rows = []
    for sc, wmap in SCHEMES.items():
        m = _replay(trades, wmap, None)
        rows.append({"scheme": sc, "weights": wmap, **m})
    rows.sort(key=lambda r: -r["total_return_pct"])
    # stop-loss sweep on the winning scheme
    stop_rows = []
    for stop in STOP_LEVELS:
        m = _replay(trades, SCHEMES[WIN_SCHEME], stop)
        stop_rows.append({"stop_pct": (round(stop * 100, 1) if stop else None), **m})
    return {"label": label, "signals": sigs, "hold_bars": HOLD_BARS, "rows": rows,
            "stop_sweep": {"scheme": WIN_SCHEME, "rows": stop_rows},
            "spy_buyhold_pct": _spy(years, allow_fetch, span_ts),
            "n_c_names_with_4h": got, "n_total_entries": len(trades), "candidates": cwmeta}


def main():
    import argparse
    os.environ.setdefault("DJANGO_SETTINGS_MODULE", "rotation.settings")
    import django; django.setup()
    from pathlib import Path
    ap = argparse.ArgumentParser()
    ap.add_argument("--fetch", action="store_true", help="allow 4h fetch (default cached)")
    ap.add_argument("--years", type=float, default=5)
    args = ap.parse_args()
    variants = {
        "mr_rsi_os": variant(args.years, args.fetch, [PRIMARY_SIG], "RSI-oversold only (the monotone entry)"),
        "combined": variant(args.years, args.fetch, COMBINED, "Combined oversold cluster (rsi_os + newlow60 + ndown)"),
    }
    payload = {
        "computed_at": pd.Timestamp.utcnow().isoformat(),
        "variants": variants, "hold_bars": HOLD_BARS, "scheme_defs": SCHEMES,
        "stop_levels_pct": [round(s * 100, 1) if s else None for s in STOP_LEVELS], "win_scheme": WIN_SCHEME,
        "note": ("Conviction-weighted position sizing (+ stop-loss sweep) on the H4-on-C dip-buy: each oversold "
                 "dip inside the point-in-time C window is sized by its analyst implied-upside bucket; a weight-"
                 "weighted portfolio equity curve is compared across schemes and stop levels. Stops are intrabar "
                 "(bar low breaches entry*(1-stop) -> exit at stop price). Ranked by TOTAL RETURN. Gross of fees; "
                 "C = current-membership; only cached-4h C names; on-deployed-capital return (idle bars in cash). "
                 "Gaps through the stop fill at the stop price (slightly optimistic)."),
    }
    STUD = Path(__file__).resolve().parent / ".data" / "studies"
    STUD.mkdir(parents=True, exist_ok=True)
    (STUD / "h4_c_conviction.json").write_text(json.dumps(payload, indent=2, default=str))
    try:
        from core.models import BacktestResult
        from django.utils import timezone
        BacktestResult.objects.update_or_create(
            kind="h4_c_conviction",
            defaults={"payload": json.loads(json.dumps(payload, default=str)), "computed_at": timezone.now()})
        print("saved BacktestResult[h4_c_conviction]", flush=True)
    except Exception as e:
        print("DB save failed:", e, flush=True)
    for key, v in variants.items():
        print(f"\n=== {v['label']} — {v['n_total_entries']} entries, {v['n_c_names_with_4h']} names ===", flush=True)
        print(f"  {'scheme':10} {'total%':>9} {'sharpe':>7} {'maxDD%':>8} {'trades':>7} {'wmean3b':>8}", flush=True)
        for r in v["rows"]:
            print(f"  {r['scheme']:10} {r['total_return_pct']:>9} {r['sharpe']:>7} {r['max_dd_pct']:>8} "
                  f"{r['n_trades']:>7} {r['weighted_mean_3b']:>8}", flush=True)
        print(f"  -- stop-loss sweep on {v['stop_sweep']['scheme']} --", flush=True)
        print(f"  {'stop':>6} {'total%':>9} {'sharpe':>7} {'maxDD%':>8} {'stopped%':>9} {'wmean3b':>8}", flush=True)
        for r in v["stop_sweep"]["rows"]:
            lbl = f"{r['stop_pct']}%" if r["stop_pct"] else "none"
            print(f"  {lbl:>6} {r['total_return_pct']:>9} {r['sharpe']:>7} {r['max_dd_pct']:>8} "
                  f"{r['pct_stopped']:>9} {r['weighted_mean_3b']:>8}", flush=True)


if __name__ == "__main__":
    main()
