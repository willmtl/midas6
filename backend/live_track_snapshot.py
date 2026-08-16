#!/usr/bin/env python3
"""LIVE TRACK RECORD — the only true out-of-sample. Snapshots the CURRENT live flagship basket
(BacktestResult[rotation_picks]) once per month, stores entry prices, then marks EVERY past snapshot to
market from current candles and chains them into a live equity curve vs SPY. Accumulates a real, honest
track record from today forward — no backtest, no lookahead, just what the deployed picks actually did.

No-migration by design: the record lives in BacktestResult[live_track] as an append-log (adding a Django
model or a @shared_task would force a celery-worker restart, which would KILL the dark-pool backfill running
there). Idempotent per calendar month. Run monthly at/after the month-end rebalance:
  MSYS_NO_PATHCONV=1 docker exec rotation-backend-1 python -u /app/live_track_snapshot.py
"""
import os, json, warnings
warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
os.environ.setdefault("DJANGO_SETTINGS_MODULE", "rotation.settings")
import django
django.setup()

from core.models import BacktestResult, Candle
from django.utils import timezone

BENCH = "SPY"


def _close_on(ticker, on_date=None):
    """Latest close for ticker on/<= on_date (or the very latest if on_date is None)."""
    qs = Candle.objects.filter(ticker=ticker)
    if on_date is not None:
        qs = qs.filter(date__lte=on_date)
    row = qs.order_by("-date").values("date", "close").first()
    return (float(row["close"]), row["date"]) if row else (None, None)


def _load_track():
    r = BacktestResult.objects.filter(kind="live_track").first()
    if r and isinstance(r.payload, dict) and "snapshots" in r.payload:
        return r.payload
    return {"snapshots": []}


def main():
    picks_res = BacktestResult.objects.filter(kind="rotation_picks").first()
    if not picks_res:
        print("no rotation_picks basket to snapshot", flush=True); return
    payload = picks_res.payload
    picks = payload.get("picks", [])
    # latest candle date = 'as of' for this snapshot
    latest_close, asof = _close_on(BENCH)
    if asof is None:
        print("no SPY candle", flush=True); return
    month = f"{asof.year:04d}-{asof.month:02d}"

    track = _load_track()
    snaps = track["snapshots"]
    existing_months = {s["month"] for s in snaps}

    if month not in existing_months:
        rows = []
        for p in picks:
            tk = p.get("pick")
            if not tk or p.get("is_etf_proxy"):
                continue
            entry, _ = _close_on(tk, asof)
            if entry is None:
                entry = p.get("last_close")
            rows.append({"ticker": tk, "sector": p.get("sector"), "pb": p.get("pb_ratio"),
                         "acceleration": p.get("acceleration"), "entry_close": entry})
        snaps.append({"month": month, "snapshot_date": str(asof), "spy_close": latest_close, "picks": rows})
        print(f"+ snapshot {month}: {len(rows)} picks entered (SPY {latest_close})", flush=True)
    else:
        print(f"snapshot for {month} already exists — re-marking only", flush=True)

    snaps.sort(key=lambda s: s["month"])

    # ---- mark every snapshot to market; chain into a live monthly-rebalanced curve ----
    track_rows = []
    live_chain, spy_chain = [], []
    for i, s in enumerate(snaps):
        # holding period end = next snapshot's date (closed) or latest candle (open)
        if i + 1 < len(snaps):
            end_date = pd.to_datetime(snaps[i + 1]["snapshot_date"]).date()
            open_period = False
        else:
            _, end_date = _close_on(BENCH)
            open_period = True
        rets = []
        for p in s["picks"]:
            end_px, _ = _close_on(p["ticker"], end_date)
            if end_px and p.get("entry_close"):
                rets.append(end_px / p["entry_close"] - 1)
        basket_ret = float(np.mean(rets)) if rets else 0.0
        spy_end, _ = _close_on(BENCH, end_date)
        spy_ret = (spy_end / s["spy_close"] - 1) if (spy_end and s.get("spy_close")) else 0.0
        live_chain.append(basket_ret); spy_chain.append(spy_ret)
        track_rows.append({"month": s["month"], "entry": s["snapshot_date"], "end": str(end_date),
                           "open": open_period, "n": len(s["picks"]),
                           "basket_ret_pct": round(basket_ret * 100, 2), "spy_ret_pct": round(spy_ret * 100, 2),
                           "alpha_pct": round((basket_ret - spy_ret) * 100, 2)})

    live_total = float(np.prod([1 + r for r in live_chain]) - 1) * 100
    spy_total = float(np.prod([1 + r for r in spy_chain]) - 1) * 100
    alphas = [tr["alpha_pct"] for tr in track_rows]
    hit = float(np.mean([a > 0 for a in alphas]) * 100) if alphas else 0.0
    summary = {"live_months": len(snaps), "live_total_pct": round(live_total, 2), "spy_total_pct": round(spy_total, 2),
               "vs_spy_pct": round(live_total - spy_total, 2), "avg_monthly_alpha_pct": round(float(np.mean(alphas)), 2) if alphas else 0,
               "months_beat_spy": f"{sum(a>0 for a in alphas)}/{len(alphas)}", "beat_rate_pct": round(hit, 1),
               "first_month": snaps[0]["month"], "latest_month": snaps[-1]["month"]}

    out = {"computed_at": timezone.now().isoformat(), "snapshots": snaps, "track": track_rows, "summary": summary,
           "note": ("True out-of-sample: live flagship basket snapshotted monthly, marked to market, chained vs SPY. "
                    "Starts today and ACCUMULATES — one data point now, a real track record over the coming months. "
                    "Compare the live vs_spy to the backtest expectation (~+4%/mo median, ~55% monthly beat).")}
    BacktestResult.objects.update_or_create(kind="live_track", defaults={"payload": out, "computed_at": timezone.now()})

    print("\n=== LIVE TRACK RECORD ===", flush=True)
    for tr in track_rows:
        flag = " (open)" if tr["open"] else ""
        print(f"  {tr['month']}  basket {tr['basket_ret_pct']:>7}%  SPY {tr['spy_ret_pct']:>7}%  alpha {tr['alpha_pct']:>7}%  (n{tr['n']}){flag}", flush=True)
    print(f"\n  live {summary['live_total_pct']}%  vs SPY {summary['spy_total_pct']}%  = {summary['vs_spy_pct']:+}pp   "
          f"| beat {summary['months_beat_spy']} months | {summary['live_months']} month(s) tracked", flush=True)
    print("Saved BacktestResult[live_track]", flush=True)


if __name__ == "__main__":
    main()
