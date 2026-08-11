#!/usr/bin/env python
"""Survivorship-bias AUDIT (analysis only — no model, no migration).

Quantifies how survivorship-biased our study universe is by comparing the
current (survivor-only) universe against the newly-imported DelistedCompany
list from EODHD.

Run inside the backend container:
    docker exec rotation-backend-1 python -u /app/delisted_survivorship.py
Optional: --top N  (depth of exchange/type breakdowns, default 10)

Writes /app/.data/studies/delisted_survivorship.json and prints a summary.
"""
import os
import sys
import json
import argparse
from collections import Counter
from datetime import datetime, timezone

import django

os.environ.setdefault("DJANGO_SETTINGS_MODULE", "rotation.settings")
django.setup()

from core.models import DelistedCompany, Candle, Fundamental, Sector  # noqa: E402
from seq_fundamental_study import build_universe  # noqa: E402

OUT_PATH = "/app/.data/studies/delisted_survivorship.json"


def _norm(t):
    return (t or "").strip().upper()


def run_audit(top=10):
    # ---- 1. Survivor universe sizes -------------------------------------
    universe = set(build_universe())
    candle_tickers = set(
        _norm(t) for t in Candle.objects.values_list("ticker", flat=True).distinct()
    )
    fundamental_tickers = set(
        _norm(t) for t in Fundamental.objects.values_list("ticker", flat=True).distinct()
    )
    universe_norm = set(_norm(t) for t in universe)

    # ---- 2. Delisted list + overlap -------------------------------------
    delisted_rows = list(
        DelistedCompany.objects.values_list("ticker", "type", "exchange")
    )
    delisted_tickers = set(_norm(t) for t, _typ, _ex in delisted_rows if t)
    delisted_total = len(delisted_rows)

    overlap_universe = sorted(delisted_tickers & universe_norm)
    overlap_candles = sorted(delisted_tickers & candle_tickers)

    # ---- 3. Core bias statement -----------------------------------------
    # Delisted US common stock = the "dead names" that a survivor-only
    # universe silently excludes.
    delisted_common = set(
        _norm(t)
        for t, typ, _ex in delisted_rows
        if t and typ and "common stock" in typ.lower()
    )
    delisted_common_n = len(delisted_common)
    # Dead names we never even tracked (existed & died, absent from candles):
    dead_never_tracked = delisted_common - candle_tickers
    dead_never_tracked_n = len(dead_never_tracked)

    universe_n = len(universe_norm)
    denom = universe_n + delisted_common_n
    bias_ratio_common = (delisted_common_n / denom) if denom else 0.0
    # Stricter variant: only count dead names we never tracked.
    denom2 = universe_n + dead_never_tracked_n
    bias_ratio_never_tracked = (dead_never_tracked_n / denom2) if denom2 else 0.0

    # ---- 4. Breakdowns ---------------------------------------------------
    type_counter = Counter((typ or "(none)").strip() for _t, typ, _ex in delisted_rows)
    exch_counter = Counter((ex or "(none)").strip() for _t, _typ, ex in delisted_rows)
    type_breakdown = dict(type_counter.most_common(top))
    exch_breakdown = dict(exch_counter.most_common(top))

    # ---- 5. Implications -------------------------------------------------
    implications = [
        "Backtests over the current universe are UPWARD-biased: names that "
        "delisted (bankruptcies, buyouts-under-water, forced removals) are "
        "absent, so the loser tail of the historical opportunity set is "
        "systematically missing from every universe-ranking study.",
        "The current study universe is built from TODAY's names "
        "(Candle INTERSECT Fundamental, minus ETFs/benchmarks) — by "
        "construction it contains ONLY survivors as of today.",
        "The $300M+ liquidity floor the user set MITIGATES but does NOT "
        "remove this: large-cap survivors are more stable, yet large names "
        "still go to zero (e.g. fraud/bankruptcy) and those episodes are "
        "excluded from the survivor snapshot.",
        "Event / point-in-time studies keyed on report or entry dates "
        "(EDGAR PIT fundamentals, news-event AR studies, insider/13D "
        "signals) are LESS affected than universe-ranking studies, because "
        "they evaluate a name only around a dated event rather than "
        "ranking a fixed survivor set across all of history.",
        "delisted_date is mostly NULL from this EODHD endpoint, so we "
        "cannot yet reconstruct a point-in-time live universe per date; "
        "the bias magnitude below is a rough COUNT-based upper bound, not "
        "a return-weighted estimate.",
    ]

    result = {
        "computed_at": datetime.now(timezone.utc).isoformat(),
        "top": top,
        "universe": {
            "universe_size": universe_n,
            "distinct_candle_tickers": len(candle_tickers),
            "distinct_fundamental_tickers": len(fundamental_tickers),
        },
        "delisted": {
            "total_rows": delisted_total,
            "distinct_tickers": len(delisted_tickers),
            "common_stock_count": delisted_common_n,
        },
        "overlap": {
            "delisted_in_current_universe": len(overlap_universe),
            "delisted_in_current_universe_sample": overlap_universe[:25],
            "delisted_in_candle_set": len(overlap_candles),
            "delisted_in_candle_set_sample": overlap_candles[:25],
            "data_quality_note": (
                "Overlap with the CURRENT universe should be ~0 — delisted "
                "names should not appear as current survivors. A non-zero "
                "count flags either recently-delisted names still carried in "
                "our tables or ticker-reuse collisions."
            ),
        },
        "bias_magnitude": {
            "delisted_common": delisted_common_n,
            "dead_names_never_tracked": dead_never_tracked_n,
            "ratio_delisted_common_over_universe_plus_delisted_common": round(
                bias_ratio_common, 4
            ),
            "ratio_never_tracked_over_universe_plus_never_tracked": round(
                bias_ratio_never_tracked, 4
            ),
            "interpretation": (
                "Rough UPPER BOUND on the fraction of the historical common-"
                "stock opportunity set excluded by using a survivor-only "
                "universe. ~{:.0%} of (survivors + dead common names) are "
                "dead names our studies never see.".format(bias_ratio_common)
            ),
        },
        "breakdown_by_type": type_breakdown,
        "breakdown_by_exchange": exch_breakdown,
        "implications": implications,
    }
    return result


def print_summary(r):
    u = r["universe"]
    d = r["delisted"]
    o = r["overlap"]
    b = r["bias_magnitude"]
    print("=" * 70)
    print("SURVIVORSHIP-BIAS AUDIT")
    print("computed_at:", r["computed_at"])
    print("=" * 70)
    print("\n[1] SURVIVOR UNIVERSE")
    print("    build_universe() size .......", u["universe_size"])
    print("    distinct Candle tickers .....", u["distinct_candle_tickers"])
    print("    distinct Fundamental tickers ", u["distinct_fundamental_tickers"])

    print("\n[2] DELISTED LIST")
    print("    total rows ..................", d["total_rows"])
    print("    distinct tickers ............", d["distinct_tickers"])
    print("    common-stock rows ...........", d["common_stock_count"])

    print("\n[2b] OVERLAP / DATA QUALITY")
    print("    delisted in CURRENT universe ", o["delisted_in_current_universe"],
          "(expect ~0)")
    if o["delisted_in_current_universe_sample"]:
        print("       e.g.", ", ".join(o["delisted_in_current_universe_sample"][:15]))
    print("    delisted in Candle set ......", o["delisted_in_candle_set"])

    print("\n[3] BIAS MAGNITUDE")
    print("    delisted common stock .......", b["delisted_common"])
    print("    dead names never tracked ....", b["dead_names_never_tracked"])
    print("    ratio (common / uni+common) .",
          b["ratio_delisted_common_over_universe_plus_delisted_common"])
    print("    ratio (never-tracked variant)",
          b["ratio_never_tracked_over_universe_plus_never_tracked"])
    print("    ->", b["interpretation"])

    print("\n[4] BREAKDOWN BY TYPE (top {}):".format(r["top"]))
    for k, v in r["breakdown_by_type"].items():
        print("    {:<28} {}".format(k[:28], v))
    print("\n[4b] BREAKDOWN BY EXCHANGE (top {}):".format(r["top"]))
    for k, v in r["breakdown_by_exchange"].items():
        print("    {:<28} {}".format(k[:28], v))

    print("\n[5] IMPLICATIONS")
    for i, s in enumerate(r["implications"], 1):
        print("    ({}) {}".format(i, s))
    print("=" * 70)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--top", type=int, default=10,
                    help="depth of type/exchange breakdowns")
    args = ap.parse_args()

    r = run_audit(top=args.top)

    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
    with open(OUT_PATH, "w") as f:
        json.dump(r, f, indent=2)

    try:   # persist to Postgres like every other study
        from core.models import BacktestResult
        from django.utils import timezone
        payload = json.loads(json.dumps(r, default=str))
        BacktestResult.objects.update_or_create(
            kind="delisted_survivorship", defaults={"payload": payload, "computed_at": timezone.now()})
        print("[delisted] saved to DB (BacktestResult kind=delisted_survivorship)")
    except Exception as e:
        print("[delisted] DB save failed:", e)

    print_summary(r)
    print("\nWrote", OUT_PATH)


if __name__ == "__main__":
    main()
