#!/usr/bin/env python3
"""Sweep the full studies.py indicator library + dip->confirmation combos as H4 0-3 day entries
inside daily-C candidate windows. See docs/superpowers/specs/2026-08-17-h4-c-indicator-sweep-design.md.
Reuses h4_on_signals_study + h4_study + studies read-only. Pure funcs import without Django."""
import math
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
    bars_list = [e[1] for e in H.EXITS]
    out = {f"{b}b": [] for b in bars_list}
    keymap = {b: f"{b}b" for b in bars_list}
    for i in idxs:
        ep = float(close[i])
        if ep <= 0:
            continue
        for _, bars, _ in H.EXITS:
            j = i + bars
            if j < n:
                out[keymap[bars]].append((close[j] - ep) / ep * 100)
    return out


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


def sweep(years, allow_fetch):
    from intraday_data import get_4h
    from h4_on_signals_study import candidate_windows
    allowed, cwmeta = candidate_windows("C")
    names = sorted(allowed)
    pools_s, pools_c = {}, {}
    base_pool = []
    got = dropped = 0
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
                rows.append({"key": k, "name": _name_of(k, is_combo), "r3b": r3, "best": best})
        rows.sort(key=lambda x: -((x["r3b"] or x["best"] or {}).get("t") or -99))
        return rows

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
