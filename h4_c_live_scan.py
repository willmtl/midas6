#!/usr/bin/env python3
"""LIVE H4 dip-buy scanner on the daily-C value basket.

Combines the two research wins into one actionable signal:
  1. WHAT to hold -> the live div_2x C basket (rotation_pick_scan.build(): cheapest-P/B value pick in each
     accelerating sector). This is the flagship's current names.
  2. WHEN to add  -> an OVERSOLD H4 (4-hour) dip firing NOW on the pick's own 4h bars (h4_study
     mean-reversion signals + the gap_dn->rsi_x combo). On C names the H4 edge is MEAN-REVERSION.
  3. HOW HARD     -> gated/ranked by point-in-time ANALYST implied-upside (target/price-1). The H4 dip-buy
     is monotone in upside: +0.26%/3b at 0-25% upside rising to +0.90% (50-100%) and +1.07% (>100%)
     — the dip pays hardest on high-upside deep-value C names (h4_c_upside study).

So a row FIRES when a current C name is in an H4 oversold dip AND carries analyst upside; conviction is the
study's own expected 3b return for that signal x upside-bucket. -> BacktestResult[h4_c_live] + JSON.
NOT a backtest — a live scan; directional, gross of fees, monthly-rebalance basket + intraday timing.
Run: MSYS_NO_PATHCONV=1 docker exec rotation-backend-1 python -u /app/h4_c_live_scan.py [--no-refresh]
"""
import os, json, warnings
warnings.filterwarnings("ignore")
os.environ.setdefault("DJANGO_SETTINGS_MODULE", "rotation.settings")
import django
django.setup()
import numpy as np
import pandas as pd
import ta
import h4_study as H
import intraday_data as ID
from h4_c_upside import load_targets, upside_asof, bucket_upside

# H4 oversold-dip signals that carry the mean-reversion edge on C names (h4-on-signals + h4_c_indicators).
OVERSOLD_SIGS = ["mr_rsi_os", "mr_newlow60", "mr_ndown", "mr_gap_dn"]
COMBO_SIG = "gap_dn__rsi_x"          # top dip->confirmation combo (h4_c_indicators)
RECENT_BARS = 2                       # "firing now" = triggered within the last N 4h bars (~1 trading day)
HIGH_3B, MED_3B = 0.8, 0.4            # conviction thresholds on the study's expected 3b return (%)


def _refresh_4h(ticker, years=1):
    """Force a fresh 1h->4h pull for one name and overwrite its parquet (the small C basket only).
    Falls back to the cached frame if egress/fetch is unavailable."""
    if ID.EOD:
        try:
            raw = ID.fetch_1h(f"{ticker}.US", years)
            if raw is not None and not raw.empty:
                df = ID.resample_ohlc(raw, 4, from_1h=True)
                if len(df) >= ID.MIN_BARS:
                    try:
                        df.to_parquet(ID.DATA / "4h" / f"{ticker}.parquet")
                    except Exception:
                        pass
                    return df
        except Exception:
            pass
    return ID.get_4h(ticker, years=years, allow_fetch=False)      # cached fallback


def _expected_lookup():
    """{sig: {bucket: {'avg_3b','n_3b','avg_6b','n_6b'}}} from BacktestResult[h4_c_upside] (the study),
    so live conviction is grounded in the backtested per-bucket edge. Empty if the study hasn't run."""
    out = {}
    try:
        from core.models import BacktestResult
        row = BacktestResult.objects.filter(kind="h4_c_upside").first()
        if not row:
            return out
        for sig, d in (row.payload.get("signals") or {}).items():
            bu = d.get("by_upside") or {}
            m = {}
            for bucket, rows in bu.items():
                r3 = next((x for x in rows if x.get("exit") == "3b"), None)
                r6 = next((x for x in rows if x.get("exit") == "6b"), None)
                m[bucket] = {"avg_3b": r3 and r3.get("avg_pct"), "n_3b": r3 and r3.get("trades"),
                             "avg_6b": r6 and r6.get("avg_pct"), "n_6b": r6 and r6.get("trades")}
            out[sig] = m
    except Exception:
        pass
    return out


def _fired(df, sig):
    """(fired_recently, bars_ago) for one signal on the 4h frame. bars_ago=0 is the most recent bar."""
    if sig == COMBO_SIG:
        from h4_c_indicators import combo_entries
        entry = combo_entries(df).get(COMBO_SIG)
    else:
        entry, _mag = H.SIGNALS[sig]["fn"](df)
    if entry is None or len(entry) == 0:
        return False, None
    tail = entry[-RECENT_BARS:]
    if not np.any(tail):
        return False, None
    off = int(np.argmax(tail[::-1]))            # most-recent True within the window
    return True, off


def build(refresh=True):
    import rotation_pick_scan
    rp = rotation_pick_scan.build()             # current live C basket (div_2x picks)
    picks = [p for p in rp.get("picks", []) if p.get("pick") and not p.get("is_etf_proxy")]
    store = load_targets()
    exp = _expected_lookup()
    all_sigs = OVERSOLD_SIGS + [COMBO_SIG]

    rows, firing = [], 0
    for p in picks:
        t = p["pick"]
        df = _refresh_4h(t) if refresh else ID.get_4h(t, years=1, allow_fetch=False)
        row = {"ticker": t, "sector": p.get("sector"), "pb_ratio": p.get("pb_ratio"),
               "pct_alloc": p.get("pct_alloc"), "selection_basis": p.get("selection_basis"),
               "daily_close": p.get("last_close")}
        if df is None or len(df) < 60:
            row.update({"has_4h": False, "state": "no 4h data"})
            rows.append(row)
            continue
        close = df["Close"]
        last_close = float(close.iloc[-1])
        bar_time = df.index[-1]
        rsi4h = ta.momentum.rsi(close, window=14).iloc[-1]
        rsi4h = round(float(rsi4h), 1) if pd.notna(rsi4h) else None
        upside = upside_asof(store, t, bar_time.date(), last_close)
        bucket = bucket_upside(upside)
        target = None
        rec = store.get(t)
        if rec:
            ds, pts = rec
            import bisect
            i = bisect.bisect_right(ds, bar_time.date()) - 1
            target = pts[i] if i >= 0 else None

        fired_sigs, best_exp = [], None
        for sig in all_sigs:
            ok, ago = _fired(df, sig)
            if not ok:
                continue
            e = (exp.get(sig, {}) or {}).get(bucket, {})
            avg3, n3 = e.get("avg_3b"), e.get("n_3b")
            nm = sig if sig == COMBO_SIG else H.SIGNALS[sig]["name"]
            fired_sigs.append({"key": sig, "name": nm, "bars_ago": ago,
                               "exp_3b": avg3, "exp_n": n3})
            if avg3 is not None and (best_exp is None or avg3 > best_exp):
                best_exp = avg3

        is_firing = bool(fired_sigs)
        if is_firing:
            firing += 1
        conviction = "—"
        if is_firing and best_exp is not None:
            conviction = "HIGH" if best_exp >= HIGH_3B else ("MED" if best_exp >= MED_3B else "LOW")
        elif is_firing:
            conviction = "MED" if (upside is not None and upside >= 25) else "LOW"
        row.update({
            "has_4h": True, "last_4h_close": round(last_close, 2), "bar_time": bar_time.isoformat(),
            "rsi_4h": rsi4h, "analyst_target": round(float(target), 2) if target else None,
            "upside_pct": round(upside, 1) if upside is not None else None,
            "upside_bucket": bucket, "fired_signals": fired_sigs, "is_firing": is_firing,
            "expected_3b": best_exp, "conviction": conviction,
            "state": ("DIP FIRING — add" if is_firing else "no H4 dip (hold)"),
        })
        rows.append(row)

    # rank: firing first, then by conviction (expected 3b), then by analyst upside
    conv_rank = {"HIGH": 3, "MED": 2, "LOW": 1, "—": 0}
    rows.sort(key=lambda r: (r.get("is_firing", False), conv_rank.get(r.get("conviction"), 0),
                             r.get("expected_3b") if r.get("expected_3b") is not None else -9,
                             r.get("upside_pct") if r.get("upside_pct") is not None else -9),
              reverse=True)

    return {
        "computed_at": pd.Timestamp.utcnow().isoformat(),
        "n_basket": len(rows), "n_firing": firing,
        "rows": rows,
        "params": {"oversold_signals": OVERSOLD_SIGS, "combo": COMBO_SIG, "recent_bars": RECENT_BARS,
                   "conviction": f"study expected 3b return: HIGH>= {HIGH_3B}% / MED>= {MED_3B}%",
                   "rule": ("live C basket (rotation_pick_scan div_2x) -> for each name, an H4 oversold dip "
                            "firing within the last %d 4h bars, ranked by the h4_c_upside study's expected 3b "
                            "return for that signal x analyst-upside bucket." % RECENT_BARS)},
        "note": ("Live scan (not a backtest). C basket = current div_2x picks; H4 dip = 4h mean-reversion "
                 "signal firing now; conviction = the h4_c_upside study's per-bucket expected 3b return "
                 "(edge is monotone in analyst upside). Directional, gross of fees. Cached-4h fallback if "
                 "no egress. The C basket itself is monthly-rebalance; this only times intraday adds."),
    }


def main():
    import argparse
    from pathlib import Path
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-refresh", action="store_true", help="use cached 4h parquet, skip live fetch")
    args = ap.parse_args()
    payload = build(refresh=not args.no_refresh)
    out = Path(__file__).resolve().parent / ".data" / "studies" / "h4_c_live.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2, default=str))
    try:
        from core.models import BacktestResult
        from django.utils import timezone
        BacktestResult.objects.update_or_create(
            kind="h4_c_live",
            defaults={"payload": json.loads(json.dumps(payload, default=str)), "computed_at": timezone.now()})
        print("Saved BacktestResult[h4_c_live]", flush=True)
    except Exception as e:
        print("DB save failed:", e, flush=True)
    print(f"\n=== H4 DIP-BUY (live C basket) — {payload['n_firing']}/{payload['n_basket']} firing ===", flush=True)
    for r in payload["rows"]:
        if not r.get("has_4h"):
            print(f"  {r['ticker']:8} {r['sector']:22} — {r['state']}", flush=True)
            continue
        up = f"{r['upside_pct']:>+6.1f}%" if r.get("upside_pct") is not None else "   n/a"
        sigs = ",".join(s["key"] for s in r.get("fired_signals", [])) or "-"
        exp = f"{r['expected_3b']:+.2f}%" if r.get("expected_3b") is not None else " n/a "
        print(f"  {r['ticker']:8} {r['sector']:20} RSI4h {str(r['rsi_4h']):>5}  upside {up} [{r['upside_bucket']:>8}]  "
              f"{r['conviction']:>4} exp3b {exp}  {r['state']:20} {sigs}", flush=True)


if __name__ == "__main__":
    main()
