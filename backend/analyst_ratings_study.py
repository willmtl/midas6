#!/usr/bin/env python3
"""BENZINGA ANALYST-RATINGS SIGNAL STUDY. Turns the backfilled structured archive
(.data/analyst_ratings.jsonl, 218k dated rows / 925 tickers 2011->2026, from `backfill_analyst_ratings.py`)
into a forward-return event study. Cleaner than the headline-scraped `news_price_target.py`: dated rating
ACTIONS (upgrades/downgrades/initiates) + explicit price_target vs previous_price_target.

Tests, on names we have daily candles for:
  (1) forward 21d/63d return after each rating ACTION (upgrade / downgrade / initiate / maintain)
  (2) forward return by PRICE-TARGET change (raise vs cut, PT vs previous_PT)
  (3) forward return by IMPLIED-UPSIDE quintile (price_target / close_at_event - 1)
Prior (from news_price_target): targets LAG price (analysts chase) -> weak/contrarian. This quantifies the
structured feed. -> .data/studies/analyst_ratings_study.json + BacktestResult[analyst_ratings_study].
Run: MSYS_NO_PATHCONV=1 docker exec rotation-backend-1 python -u /app/analyst_ratings_study.py
"""
import os, json, warnings
warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
os.environ.setdefault("DJANGO_SETTINGS_MODULE", "rotation.settings")
import django
django.setup()
from pathlib import Path
from django.db import connection

SRC = Path("/app/.data/analyst_ratings.jsonl")
OUT = Path("/app/.data/studies/analyst_ratings_study.json")


def main():
    with connection.cursor() as c:
        c.execute("SET max_parallel_workers_per_gather=0")
    from core.models import Candle, BacktestResult
    from django.utils import timezone

    rows = [json.loads(l) for l in SRC.read_text(encoding="utf-8").splitlines() if l.strip()]
    print(f"ratings rows: {len(rows):,}", flush=True)
    tickers = sorted({r["ticker"] for r in rows})

    # daily close series per ticker
    cand = {}
    for tk in tickers:
        q = list(Candle.objects.filter(ticker=tk, interval="1d").order_by("date").values_list("date", "close"))
        if len(q) > 80:
            cand[tk] = pd.Series({pd.Timestamp(d): float(c) for d, c in q}).sort_index()
    print(f"tickers with candles: {len(cand)}", flush=True)

    recs = []
    for r in rows:
        tk, ds = r.get("ticker"), r.get("date")
        s = cand.get(tk)
        if s is None or not ds:
            continue
        dt = pd.Timestamp(ds)
        past = s[s.index <= dt]
        if len(past) < 2:
            continue
        px0 = float(past.iloc[-1])
        if px0 <= 0:
            continue
        fut = s[s.index > dt]
        f21 = (float(fut.iloc[20]) / px0 - 1) if len(fut) > 20 else np.nan
        f63 = (float(fut.iloc[62]) / px0 - 1) if len(fut) > 62 else np.nan
        pt, ppt = r.get("price_target"), r.get("previous_price_target")
        upside = (pt / px0 - 1) if (pt and px0) else np.nan
        pt_dir = 0
        if pt and ppt:
            pt_dir = 1 if pt > ppt * 1.001 else (-1 if pt < ppt * 0.999 else 0)
        recs.append(dict(ticker=tk, date=ds, action=(r.get("rating_action") or "").strip(),
                         pt_dir=pt_dir, upside=upside, f21=f21, f63=f63))
    df = pd.DataFrame(recs)
    df = df[np.isfinite(df["f21"]) | np.isfinite(df["f63"])]
    print(f"events with forward data: {len(df):,}\n", flush=True)

    def agg(g):
        return dict(n=int(len(g)), f21=round(float(g["f21"].mean()) * 100, 2),
                    f63=round(float(g["f63"].mean()) * 100, 2), win21=round(float((g["f21"] > 0).mean()) * 100, 1))

    payload = {"computed_at": pd.Timestamp.utcnow().isoformat(), "n_rows": len(rows), "n_events": int(len(df))}

    # (1) by rating action
    print("=== forward return by RATING ACTION ===", flush=True)
    by_act = {}
    for act, g in df.groupby("action"):
        if len(g) < 50 or not act:
            continue
        a = agg(g); by_act[act] = a
        print(f"  {act:22} n={a['n']:>6}  f21={a['f21']:+6.2f}%  f63={a['f63']:+6.2f}%  win21={a['win21']:.0f}%", flush=True)
    payload["by_action"] = by_act

    # (2) by PT change direction
    print("\n=== forward return by PRICE-TARGET change ===", flush=True)
    by_pt = {}
    lab = {1: "RAISE", -1: "CUT", 0: "unchanged"}
    for d, g in df[df["pt_dir"] != 0].groupby("pt_dir"):
        a = agg(g); by_pt[lab[d]] = a
        print(f"  {lab[d]:10} n={a['n']:>6}  f21={a['f21']:+6.2f}%  f63={a['f63']:+6.2f}%  win21={a['win21']:.0f}%", flush=True)
    payload["by_pt_change"] = by_pt

    # (3) by implied-upside quintile
    print("\n=== forward return by IMPLIED-UPSIDE quintile (Q0=lowest .. Q4=highest) ===", flush=True)
    du = df[np.isfinite(df["upside"])].copy()
    du["upside"] = du["upside"].clip(-0.9, 5)
    by_q = {}
    if len(du) > 100:
        du["q"] = pd.qcut(du["upside"], 5, labels=False, duplicates="drop")
        for q, g in du.groupby("q"):
            a = agg(g); a["avg_upside"] = round(float(g["upside"].mean()) * 100, 1)
            by_q[f"Q{int(q)}"] = a
            print(f"  Q{int(q)}  upside={a['avg_upside']:+6.1f}%  n={a['n']:>6}  f21={a['f21']:+6.2f}%  f63={a['f63']:+6.2f}%", flush=True)
        if 0 in du["q"].values and 4 in du["q"].values:
            spread = by_q["Q4"]["f63"] - by_q["Q0"]["f63"]
            payload["q4_minus_q0_f63"] = round(spread, 2)
            print(f"\n  Q4-Q0 fwd63 spread: {spread:+.2f}%  (+ = high-target-upside outperforms = momentum; − = contrarian/lag)", flush=True)
    payload["by_upside_quintile"] = by_q

    # verdict
    up = by_act.get("upgrades", {}); dn = by_act.get("downgrades", {})
    parts = []
    if up and dn:
        parts.append(f"upgrades f63 {up['f63']:+.1f}% vs downgrades {dn['f63']:+.1f}% "
                     f"(spread {up['f63']-dn['f63']:+.1f}pp)")
    if "RAISE" in by_pt and "CUT" in by_pt:
        parts.append(f"PT-raise {by_pt['RAISE']['f63']:+.1f}% vs PT-cut {by_pt['CUT']['f63']:+.1f}%")
    payload["verdict"] = "; ".join(parts) if parts else "insufficient events"
    print("\nVERDICT:", payload["verdict"], flush=True)

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(payload, indent=2, default=str))
    try:
        BacktestResult.objects.update_or_create(kind="analyst_ratings_study",
            defaults={"payload": json.loads(json.dumps(payload, default=str)), "computed_at": timezone.now()})
        print("Saved BacktestResult[analyst_ratings_study]", flush=True)
    except Exception as e:
        print("DB save failed:", e, flush=True)
    print("DONE_ANALYST", flush=True)


if __name__ == "__main__":
    main()
