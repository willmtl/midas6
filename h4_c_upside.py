#!/usr/bin/env python3
"""Bucket the H4-on-C dip-buy by PIT analyst implied-upside (target/price-1) from the dated Benzinga
archive. See docs/superpowers/specs/2026-08-17-h4-c-upside-design.md. Reuses h4_on_signals_study +
h4_study + h4_c_indicators read-only. Pure funcs import without Django."""
import json
import bisect
import datetime as dt
import numpy as np
import pandas as pd
import h4_study as H

ARCHIVE = "/app/.data/analyst_ratings.jsonl"
UPSIDE_BUCKETS = [("<0%", -1e9, 0), ("0-25%", 0, 25), ("25-50%", 25, 50),
                  ("50-100%", 50, 100), (">100%", 100, 1e9)]


def load_targets(path=ARCHIVE):
    """{ticker: (sorted_dates[list[date]], targets[list[float]])} from the jsonl archive. Drops
    null/<=0 targets; keeps last target per (ticker,date)."""
    tmp = {}
    with open(path) as f:
        for line in f:
            try:
                r = json.loads(line)
            except Exception:
                continue
            tk = r.get("ticker"); pt = r.get("price_target"); ds = r.get("date")
            if not tk or not ds or pt in (None, 0) or pt is None:
                continue
            try:
                d = dt.date.fromisoformat(ds); pt = float(pt)
            except Exception:
                continue
            if pt <= 0:
                continue
            tmp.setdefault(tk, {})[d] = pt      # last write per date wins
    store = {}
    for tk, dmap in tmp.items():
        ds = sorted(dmap)
        store[tk] = (ds, [dmap[d] for d in ds])
    return store


def upside_asof(store, tk, d, close):
    """Most recent target on/before date d / close - 1 (percent). None if no prior target / no coverage."""
    rec = store.get(tk)
    if not rec or close is None or close <= 0:
        return None
    ds, pts = rec
    i = bisect.bisect_right(ds, d) - 1        # last index with ds[i] <= d
    if i < 0:
        return None
    return (pts[i] / close - 1) * 100


def bucket_upside(u):
    if u is None:
        return "no_target"
    for label, lo, hi in UPSIDE_BUCKETS:
        if lo <= u < hi:
            return label
    return None


def _entry_series(df):
    """Entry series to bucket: the 14 h4_study SIGNALS + the gap_dn__rsi_x combo. {key: entry_bool_array}."""
    from h4_c_indicators import combo_entries
    out = {}
    for k, meta in H.SIGNALS.items():
        e, _mag = meta["fn"](df)
        out[k] = e
    ce = combo_entries(df)
    out["gap_dn__rsi_x"] = ce["gap_dn__rsi_x"]
    return out


def sweep(years, allow_fetch):
    from intraday_data import get_4h
    from h4_on_signals_study import candidate_windows
    store = load_targets()
    allowed, cwmeta = candidate_windows("C")
    names = sorted(allowed)
    bucket_order = [b[0] for b in UPSIDE_BUCKETS] + ["no_target"]
    pools = {}
    exit_keys = [f"{b}b" for b in [e[1] for e in H.EXITS]]
    covered = set()
    n_no_target = 0
    got = dropped = 0
    for tk in names:
        df = get_4h(tk, years, allow_fetch)
        if df is None or len(df) < 120:
            dropped += 1
            continue
        got += 1
        if tk in store:
            covered.add(tk)
        close = df["Close"].values
        dates = df.index.normalize()
        ad = allowed[tk]
        for sig, entry in _entry_series(df).items():
            n = len(close)
            cand = [i for i in range(n) if entry[i] and dates[i].date() in ad]
            idxs = sorted(H._episode_starts(cand, gap=H.GAP))
            sp = pools.setdefault(sig, {b: {} for b in bucket_order + ["ALL"]})
            for i in idxs:
                ep = float(close[i])
                if ep <= 0:
                    continue
                u = upside_asof(store, tk, dates[i].date(), ep)
                b = bucket_upside(u)
                if b == "no_target":
                    n_no_target += 1
                for _, bars, _ in H.EXITS:
                    j = i + bars
                    if j < n:
                        r = (close[j] - ep) / ep * 100
                        ek = f"{bars}b"
                        sp[b].setdefault(ek, []).append(r)
                        sp["ALL"].setdefault(ek, []).append(r)
    signals = {}
    for sig, sp in pools.items():
        name = sig if sig == "gap_dn__rsi_x" else H.SIGNALS[sig]["name"]
        signals[sig] = {"name": name,
                        "all": H.agg_rows(sp["ALL"], exit_keys),
                        "by_upside": {b: H.agg_rows(sp[b], exit_keys) for b in bucket_order}}
    return {"signals": signals, "bucket_order": bucket_order,
            "coverage": {"n_c_names": cwmeta["n_names"], "n_with_4h": got, "n_dropped_no_4h": dropped,
                         "n_covered_by_archive": len(covered), "n_no_target_entries": n_no_target}}


def main():
    import os, json, argparse
    os.environ.setdefault("DJANGO_SETTINGS_MODULE", "rotation.settings")
    import django; django.setup()
    from pathlib import Path
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-fetch", action="store_true")
    ap.add_argument("--years", type=float, default=5)
    args = ap.parse_args()
    print(f"h4-on-C bucketed by analyst upside (fetch={'off' if args.no_fetch else 'on'})", flush=True)
    payload = sweep(args.years, allow_fetch=not args.no_fetch)
    payload["computed_at"] = pd.Timestamp.utcnow().isoformat()
    payload["note"] = ("H4-on-C 0-3 day entries bucketed by PIT analyst implied-upside (target/price-1, dated "
                       "Benzinga archive). Tests if the dip-buy pays more on high-upside (deep-value) C names. "
                       "n<30 per-bucket cells are noise; gross of fees; 114/132 C names covered; dated-PIT.")
    STUD = Path(__file__).resolve().parent / ".data" / "studies"
    STUD.mkdir(parents=True, exist_ok=True)
    (STUD / "h4_c_upside.json").write_text(json.dumps(payload, indent=2, default=str))
    try:
        from core.models import BacktestResult
        from django.utils import timezone
        BacktestResult.objects.update_or_create(
            kind="h4_c_upside",
            defaults={"payload": json.loads(json.dumps(payload, default=str)), "computed_at": timezone.now()})
        print("saved BacktestResult[h4_c_upside]", flush=True)
    except Exception as e:
        print("DB save failed:", e, flush=True)
    print(f"\ncoverage: {payload['coverage']}", flush=True)
    for sig in ["mr_rsi_os", "mr_ndown", "mr_gap_dn", "gap_dn__rsi_x", "mr_newlow60"]:
        d = payload["signals"].get(sig)
        if not d:
            continue
        print(f"\n=== {sig} — 3b by analyst upside ===", flush=True)
        for b in payload["bucket_order"]:
            r = next((x for x in d["by_upside"][b] if x["exit"] == "3b"), None)
            if r:
                print(f"  {b:10} avg {r['avg_pct']:+.2f}% win {r['win_pct']}% t={r['t']} n={r['trades']}", flush=True)


if __name__ == "__main__":
    main()
