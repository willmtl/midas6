#!/usr/bin/env python3
"""Backtest CONVICTION-WEIGHTED position sizing on the H4-on-C dip-buy.

The h4_c_upside finding: the H4 oversold dip on C value names is monotone in analyst implied-upside
(+0.26%/3b at 0-25% upside -> +1.07% at >100%). So instead of trading every dip the same size, SIZE THE
POSITION by conviction (the upside bucket) and see if it lifts total return — the same div_2x logic the
daily flagship already uses, applied to the intraday add.

Method: take the H4 oversold entries inside the point-in-time C windows (h4_on_signals candidate_windows),
size each by a weighting scheme, and build a conviction-weighted portfolio equity curve (at each 4h bar the
return is the weight-weighted average of the currently-open dip trades). Compare schemes on TOTAL RETURN
(return-priority), plus Sharpe / max-DD / trade count. -> BacktestResult[h4_c_conviction] + JSON.
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
# These map to the live scanner's conviction tiers (LOW/MED/HIGH) but at bucket granularity.
BUCKETS = [b[0] for b in UPSIDE_BUCKETS] + ["no_target"]
SCHEMES = {
    # every dip the same size — the null hypothesis (what the scanner does today)
    "equal":     {"<0%": 1, "0-25%": 1, "25-50%": 1, "50-100%": 1, ">100%": 1, "no_target": 1},
    # just drop the value traps (negative implied-upside), everything else equal
    "skip_neg":  {"<0%": 0, "0-25%": 1, "25-50%": 1, "50-100%": 1, ">100%": 1, "no_target": 1},
    # linear conviction ladder by upside bucket
    "linear":    {"<0%": 0, "0-25%": 1, "25-50%": 2, "50-100%": 3, ">100%": 4, "no_target": 1},
    # steep (doubling) conviction — the concentration finding: steeper conviction monotonically helps
    "steep_2x":  {"<0%": 0, "0-25%": 1, "25-50%": 2, "50-100%": 4, ">100%": 8, "no_target": 1},
    # only trade the high-conviction deep-value dips (50%+ upside)
    "hi_only":   {"<0%": 0, "0-25%": 0, "25-50%": 0, "50-100%": 1, ">100%": 1, "no_target": 0},
}


def _entries(df, sigs):
    """Union of the given oversold signals' entry bars on one 4h frame, episode-deduped (gap=3)."""
    n = len(df)
    fire = np.zeros(n, dtype=bool)
    for s in sigs:
        e, _mag = H.SIGNALS[s]["fn"](df)
        fire |= np.asarray(e, dtype=bool)
    cand = [i for i in range(n) if fire[i]]
    return sorted(H._episode_starts(cand, gap=H.GAP))


def _metrics(bar_map):
    """bar_map: {timestamp -> (sum_w_ret, sum_w)}. Returns (total_ret%, sharpe, maxdd%, n_bars)."""
    if not bar_map:
        return 0.0, 0.0, 0.0, 0
    ts = sorted(bar_map)
    rets = np.array([bar_map[t][0] / bar_map[t][1] for t in ts if bar_map[t][1] > 0])
    if len(rets) == 0:
        return 0.0, 0.0, 0.0, 0
    eq = np.cumprod(1 + rets)
    total = (eq[-1] - 1) * 100
    sd = rets.std(ddof=1) if len(rets) > 1 else 0
    sharpe = float(rets.mean() / sd * np.sqrt(BARS_PER_YEAR)) if sd > 0 else 0.0
    peak = np.maximum.accumulate(eq)
    maxdd = float(((eq - peak) / peak).min() * 100)
    return round(total, 1), round(sharpe, 2), round(maxdd, 1), len(rets)


def _weight_mean_3b(trades, wmap):
    """Weighted mean per-trade 3b return (%) under a weight map, and effective trade count."""
    num = den = ntr = 0.0
    for r3, bucket in trades:
        w = wmap.get(bucket, 0)
        if w > 0:
            num += w * r3; den += w; ntr += 1
    return (round(num / den, 3) if den else 0.0), int(ntr)


def backtest(years, allow_fetch, sigs, label):
    from intraday_data import get_4h
    from h4_on_signals_study import candidate_windows
    store = load_targets()
    allowed, cwmeta = candidate_windows("C")
    names = sorted(allowed)

    # trade[bucket] returns + per-scheme timeline of (weight, next-bar return)
    bar_maps = {sc: {} for sc in SCHEMES}
    trades = []            # (r3b, bucket) for the weighted-mean summary
    span_ts = []
    got = 0
    for tk in names:
        df = get_4h(tk, years, allow_fetch)
        if df is None or len(df) < 120:
            continue
        got += 1
        close = df["Close"].values
        ts = df.index
        n = len(close)
        barret = np.empty(n); barret[0] = 0.0
        barret[1:] = close[1:] / close[:-1] - 1
        ad = allowed[tk]
        span_ts.append(ts[0]); span_ts.append(ts[-1])
        idxs = [i for i in _entries(df, sigs) if ts[i].date() in ad]
        for i in idxs:
            ep = float(close[i])
            if ep <= 0 or i + HOLD_BARS >= n:
                continue
            u = upside_asof(store, tk, ts[i].date(), ep)
            bucket = bucket_upside(u)
            r3 = (close[i + HOLD_BARS] - ep) / ep * 100
            trades.append((r3, bucket))
            for b in range(1, HOLD_BARS + 1):
                k = i + b
                r = barret[k]
                for sc, wmap in SCHEMES.items():
                    w = wmap.get(bucket, 0)
                    if w <= 0:
                        continue
                    cur = bar_maps[sc].get(ts[k])
                    if cur is None:
                        bar_maps[sc][ts[k]] = [w * r, w]
                    else:
                        cur[0] += w * r; cur[1] += w

    # SPY buy-hold over the same span, for context
    spy_total = None
    try:
        spy = get_4h("SPY", years, allow_fetch)
        if spy is not None and span_ts:
            lo, hi = min(span_ts), max(span_ts)
            s = spy[(spy.index >= lo) & (spy.index <= hi)]["Close"]
            if len(s) > 1:
                spy_total = round((float(s.iloc[-1]) / float(s.iloc[0]) - 1) * 100, 1)
    except Exception:
        pass

    rows = []
    for sc, wmap in SCHEMES.items():
        total, sharpe, maxdd, nbars = _metrics(bar_maps[sc])
        wmean, ntr = _weight_mean_3b(trades, wmap)
        rows.append({"scheme": sc, "weights": wmap, "total_return_pct": total, "sharpe": sharpe,
                     "max_dd_pct": maxdd, "n_trades": ntr, "n_active_bars": nbars,
                     "weighted_mean_3b": wmean})
    rows.sort(key=lambda r: -r["total_return_pct"])
    return {"label": label, "signals": sigs, "hold_bars": HOLD_BARS, "rows": rows,
            "spy_buyhold_pct": spy_total, "n_c_names_with_4h": got, "n_total_entries": len(trades),
            "candidates": cwmeta}


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
        "mr_rsi_os": backtest(args.years, args.fetch, [PRIMARY_SIG], "RSI-oversold only (the monotone entry)"),
        "combined": backtest(args.years, args.fetch, COMBINED, "Combined oversold cluster (rsi_os + newlow60 + ndown)"),
    }
    payload = {
        "computed_at": pd.Timestamp.utcnow().isoformat(),
        "variants": variants,
        "hold_bars": HOLD_BARS,
        "scheme_defs": SCHEMES,
        "note": ("Conviction-weighted position sizing on the H4-on-C dip-buy: each oversold dip inside the "
                 "point-in-time C window is sized by its analyst implied-upside bucket, and a weight-weighted "
                 "portfolio equity curve (weighted avg of concurrently-open 3-bar trades) is compared to "
                 "equal-weight. Ranked by TOTAL RETURN (return-priority). Weights map to the live scanner's "
                 "conviction tiers. Gross of fees; C = current-membership; only cached-4h C names; on-deployed-"
                 "capital return (idle bars in cash)."),
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
        print(f"\n=== {v['label']} — {v['n_total_entries']} entries, {v['n_c_names_with_4h']} names "
              f"(SPY buy-hold over span {v['spy_buyhold_pct']}%) ===", flush=True)
        print(f"  {'scheme':10} {'total%':>9} {'sharpe':>7} {'maxDD%':>8} {'trades':>7} {'wmean3b':>8}", flush=True)
        for r in v["rows"]:
            print(f"  {r['scheme']:10} {r['total_return_pct']:>9} {r['sharpe']:>7} {r['max_dd_pct']:>8} "
                  f"{r['n_trades']:>7} {r['weighted_mean_3b']:>8}", flush=True)


if __name__ == "__main__":
    main()
