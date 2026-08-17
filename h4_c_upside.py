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
