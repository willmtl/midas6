#!/usr/bin/env python3
"""LIVE rotation-pick scanner — the ONLY sector-rotation strategy with real alpha, as a live signal.

The backtest verdict: rotating sector ETFs LOSES to SPY. The edge is using the rotation as a SECTOR-SELECTION
FILTER feeding a VALUE small-cap stock-pick: rank the sector ETFs by momentum ACCELERATION, take the top-10
inflecting sectors, and in each buy the cheapest DRIFT-P/B small-cap value name (with a tl_support entry),
div_4x conviction-weighted.

SINGLE SOURCE OF TRUTH (2026-08-19): this scanner no longer re-implements the selection. It DRIVES the
validated flagship engine (survivorship_smallcap_study.py) in LIVE_PICK mode and reshapes its output for the
dashboard. That engine IS the 112,950%/Sharpe 1.93/DD-23.4% honest backtest (2016-2026, survivorship-de-biased,
USD incl. FX, point-in-time), so the live picks match the backtest BY CONSTRUCTION — no dual implementation to
drift out of sync (an earlier hand-ported copy silently diverged: top-7 vs top-10, raw-P/B + analyst blend vs
drift-P/B + tl_support). The engine also runs a built-in RECONCILE (live picks == non-live backtest on the last
common month); we surface `reconciled` in the payload.

Flagship rules (all inside the engine): rank ~93 sector ETFs by 6mo-momentum ACCELERATION -> top-10; in each,
from the US+CA GICS pool, gate on positive book / $5M/day dollar-volume / value-trap filter / <$2B small-cap
preference; value metric = stale-book-DRIFT P/B (book nowcast: accrue TTM earnings since last filing);
among the 5 cheapest, tl_support entry (rising 9mo trendline, dipped below it); weight = div_4x if the name
shows A/D divergence (accumulation into weakness), renormalized; skip a sleeve with no qualifying US/CA name.

-> BacktestResult[rotation_picks] + JSON. Directional / no fees; monthly-rebalance basket, not intraday.
Run: docker exec rotation-backend-1 python -u /app/rotation_pick_scan.py
"""
import os, sys, json, subprocess, warnings
warnings.filterwarnings("ignore")
os.environ.setdefault("DJANGO_SETTINGS_MODULE", "rotation.settings")
import django
django.setup()
import pandas as pd
from pathlib import Path
import sector_holdings
from seq_fundamental_study import load_candles

HERE = Path(__file__).resolve().parent
ENGINE = str(HERE / "survivorship_smallcap_study.py")
LIVE_JSON = HERE / ".data" / "studies" / "live_flagship_picks.json"
LOOKBACK_D = 126          # ~6 trading months, for the sector's own 6mo-momentum display column
OUT = None


def _dvol(dfp):
    """20-day average dollar volume ($). None if no volume/data."""
    if dfp is None or "Volume" not in dfp or len(dfp) < 5:
        return None
    dv = (dfp["Close"] * dfp["Volume"]).tail(20).mean()
    return float(dv) if pd.notna(dv) else None


def _stock_accel(dfp):
    """Per-stock momentum ACCELERATION (3mo-now minus 3mo-3ago), display only. We FADE this at the stock
    level — the value pick reverts hardest on deeper weakness (lower/negative is the better quadrant)."""
    if dfp is None or len(dfp) < LOOKBACK_D + 1:
        return None
    c = dfp["Close"]
    m3n = c.iloc[-1] / c.iloc[-1 - LOOKBACK_D // 2] - 1
    m3p = c.iloc[-1 - LOOKBACK_D // 2] / c.iloc[-1 - LOOKBACK_D] - 1
    return round(float((m3n - m3p) * 100), 1)


def _mom6(dfp):
    """Sector ETF's own 6mo momentum % (display column; the ranking signal is acceleration, from the engine)."""
    if dfp is None or len(dfp) < LOOKBACK_D + 1:
        return None
    c = dfp["Close"]
    return round(float((c.iloc[-1] / c.iloc[-1 - LOOKBACK_D] - 1) * 100), 1)


def _alt_signals(tickers):
    """Insider / Congress BUYING confirmation flags per ticker (informational overlay, NOT a selection driver:
    a cheap value pick that ALSO has insiders or a legislator buying is higher-conviction, 56-62% win)."""
    from core.models import InsiderBuy, CongressTrade
    from django.db.models import Sum, Count
    from datetime import date, timedelta
    out = {t: {"insider_buying": False, "congress_buying": False} for t in tickers}
    if not tickers:
        return out
    ins_cut = date.today() - timedelta(days=90)
    cong_cut = date.today() - timedelta(days=180)
    for r in (InsiderBuy.objects.filter(ticker__in=tickers, filed_date__gte=ins_cut)
              .values("ticker").annotate(b=Sum("buy_value"), s=Sum("sell_value"))):
        if (r["b"] or 0) > (r["s"] or 0):
            out[r["ticker"]]["insider_buying"] = True
    for r in (CongressTrade.objects.filter(ticker__in=tickers, transaction_type="buy", report_date__gte=cong_cut)
              .values("ticker").annotate(n=Count("id"))):
        if r["n"] > 0:
            out[r["ticker"]]["congress_buying"] = True
    return out


def _run_engine_live():
    """Drive the validated flagship engine in LIVE_PICK mode (single source of truth). It writes
    live_flagship_picks.json with the current-month picks + the full sector trace + a reconcile flag."""
    env = dict(os.environ, LIVE_PICK="1")
    print("running flagship engine (LIVE_PICK) — single source of truth, this takes a few minutes...", flush=True)
    r = subprocess.run([sys.executable, "-u", ENGINE], env=env)   # inherit stdout: engine progress streams live
    if r.returncode != 0:
        raise RuntimeError(f"flagship engine LIVE_PICK failed (rc={r.returncode}); see output above")
    if not LIVE_JSON.exists():
        raise RuntimeError(f"engine did not write {LIVE_JSON}")
    return json.loads(LIVE_JSON.read_text())


def build():
    live = _run_engine_live()
    date = live.get("date")
    reconciled = live.get("reconciled")
    recon_month = live.get("reconcile_month")
    tps = live.get("top_sectors", [])                     # the top-10 ranked sleeves this month (accel + etf)
    accel_by_etf = {s["etf"]: s.get("accel") for s in tps}
    rank_by_etf = {s["etf"]: i for i, s in enumerate(tps)}   # 0-based accel rank (for top-5 conviction flag)
    raw_picks = [p for p in live.get("picks", []) if p.get("ticker")]
    print(f"flagship LIVE picks for {date}: {len(raw_picks)} names "
          f"(reconcile vs backtest on {recon_month}: {'OK ✓' if reconciled else 'DIFFERS ✗'})", flush=True)

    tickers = [p["ticker"] for p in raw_picks]
    etfs = [p["etf"] for p in raw_picks if p.get("etf")]
    px = load_candles(sorted({*tickers, *etfs}))
    alt = _alt_signals(tickers)
    from profitability_guard import guard_flags
    gflags = guard_flags(tickers) if tickers else {}

    picks = []
    for rank, p in enumerate(raw_picks, 1):
        t = p["ticker"]; etf = p.get("etf") or ""
        dfp = px.get(t); dv = _dvol(dfp); g = gflags.get(t, {})
        pb = p.get("pb")                                   # DRIFT-P/B (book nowcast) — the flagship value metric
        # Display conviction 0-5 (derived from the flagship's OWN fields — NOT a selection input, just a UI badge):
        flags = []
        if pb is not None and pb < 1:                flags.append("deep value (drift-P/B<1)")
        if (p.get("ni") or 0) > 0:                   flags.append("profitable (TTM NI>0)")
        if p.get("de") is not None and p["de"] < 0.5: flags.append("low debt (D/E<0.5)")
        if p.get("conviction"):                      flags.append("A/D accumulation (4x weight)")
        if rank_by_etf.get(etf, 99) < 5:             flags.append("top-5 sector")
        picks.append({
            "rank": rank, "sector": p.get("sector"), "etf": etf,
            "momentum_6m": _mom6(px.get(etf)), "acceleration": accel_by_etf.get(etf),
            "pick": t, "company": p.get("company"), "is_etf_proxy": False,
            "pb_ratio": (round(pb, 2) if pb is not None else None),
            "selection_basis": "drift_pb_tl_support",       # top-5 cheapest drift-P/B -> tl_support entry
            "guard_status": g.get("status"), "margin_pct": g.get("margin"),
            "debt_to_equity": (round(p["de"], 2) if p.get("de") is not None else g.get("debt_to_equity")),
            "net_income": p.get("ni"), "improving": g.get("improving"),
            "last_close": round(float(dfp["Close"].iloc[-1]), 2) if dfp is not None and len(dfp) else None,
            "dollar_vol_m": round(dv / 1e6, 1) if dv else None,
            "stock_acceleration": _stock_accel(dfp),
            "accumulating": bool(p.get("conviction")),      # A/D divergence -> div_4x conviction weight
            "market_cap": p.get("mktcap_usd"),              # USD (engine already FX-converted)
            "pe_ratio": p.get("pe"), "forward_pe": None,
            "profit_margin": None, "revenue_growth": p.get("rev_g"),
            "conviction": len(flags), "conviction_flags": flags,
            "insider_buying": alt.get(t, {}).get("insider_buying", False),
            "congress_buying": alt.get(t, {}).get("congress_buying", False),
            "_weight": p.get("weight") or 0.0,              # engine's div_4x weight (renormalized below)
            "pick_sectors": sector_holdings.get_sectors_for_ticker(t),
        })

    # Alloc % = the engine's div_4x conviction weight (A/D-divergence names already carry 4x), renormalized to 100.
    tot_w = sum(p["_weight"] for p in picks) or 1.0
    for p in picks:
        p["pct_alloc"] = round(p.pop("_weight") / tot_w * 100, 1)
    n_accum = sum(1 for p in picks if p["accumulating"])

    picked_etfs = {p["etf"] for p in picks}
    skipped = [{"rank": rank_by_etf.get(s["etf"], 99) + 1, "sector": s.get("sector"), "etf": s["etf"],
                "acceleration": s.get("accel")}
               for s in tps if s["etf"] not in picked_etfs]
    deactivated_now = live.get("deactivated", []) or []
    if skipped:
        print(f"skipped {len(skipped)} sleeve(s) with no qualifying US/CA value stock: "
              f"{[s['etf'] for s in skipped]}", flush=True)

    payload = {
        "computed_at": pd.Timestamp.utcnow().isoformat(),
        "as_of_month": date,
        "reconciled": reconciled, "reconcile_month": recon_month,
        "n_accumulating": n_accum,
        "params": {
            "lookback_days": LOOKBACK_D, "top_n_sectors": 10, "flagship": "usca_small drift+div4x+tl_support",
            "universe": ("US+Canada GICS pool per top core-sector ETF (survivorship-de-biased incl. delisted, "
                         "USD incl. FX, point-in-time) — the exact engine universe, not a live re-derivation"),
            "sector_signal": "momentum ACCELERATION (3mo-now minus 3mo-3ago) — catches the inflection, not the 6mo run",
            "size_tilt": "prefer cheapest names <$2B USD (small-cap size premium is the biggest lever); $5M/day dvol floor",
            "conviction_weighting": ("div_4x: equal-weight basket, but A/D-divergence names (ADL rising ~3mo while "
                                     "price fell ~3mo = accumulation into weakness) get 4x weight, renormalized; "
                                     "pct_alloc per pick is the deployable weight."),
            "value_metric": ("stale-book-DRIFT P/B: market cap ÷ book NOWCAST (accrue TTM earnings since the last "
                             "filing so the multiple reflects today's price vs an up-to-date book, not a 4-month-"
                             "stale one). Among the 5 cheapest, tl_support entry: the name in a rising 9-month "
                             "trendline that has dipped below it."),
            "rule": ("rank sectors by momentum ACCELERATION -> top-10 -> within each, from the US+CA GICS pool, "
                     "candidates pass positive-book + value-trap filter + $5M/day dvol; size-tilt to <$2B; value "
                     "pick = cheapest DRIFT-P/B among the 5, tl_support entry; A/D-divergence names get 4x weight; "
                     "if a sleeve has NO qualifying US/CA value stock (pure commodity/bond/foreign) SKIP it and "
                     "renormalize the rest (proxy-holding is strongly negative)."),
            "backtest": ("usca_small flagship (div4x + stale-book drift + tl_support): +112,950% total / Sharpe 1.93 "
                         "/ DD -23.4% / honest 2016-2026 (survivorship-de-biased incl. delisted, USD incl. FX, "
                         "point-in-time; survivorship_smallcap_study.py). Live picks are produced by that SAME "
                         "engine (LIVE_PICK) — they match the backtest by construction."),
        },
        "picks": picks,
        "n_sectors_skipped": len(skipped),
        "skipped_sectors": skipped,
        "deactivated_sectors": deactivated_now,
        "note": (("⚠️ live/backtest reconcile MISMATCH on " + str(recon_month) + " — investigate before trusting. ")
                 if reconciled is False else "")
                 + ("Value picks are the alpha; sleeves with no qualifying US/CA value stock are SKIPPED, not held "
                    "via ETF. Weights are the engine's div_4x conviction weights, renormalized. Picks come from the "
                    "validated flagship engine (LIVE_PICK) so they match the honest backtest by construction. "
                    "Monthly-rebalance; directional, no fees."),
    }
    return payload


def main():
    global OUT
    OUT = HERE / ".data" / "studies" / "rotation_picks.json"
    payload = build()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(payload, indent=2, default=str))
    try:
        from core.models import BacktestResult
        from django.utils import timezone
        BacktestResult.objects.update_or_create(
            kind="rotation_picks",
            defaults={"payload": json.loads(json.dumps(payload, default=str)),
                      "computed_at": timezone.now()})
        print("Saved BacktestResult[rotation_picks]", flush=True)
    except Exception as e:
        print("DB save failed:", e, flush=True)
    rec = payload.get("reconciled")
    print(f"\n=== ROTATION PICKS {payload.get('as_of_month')} (drift-P/B + tl_support + div_4x; SAME engine as "
          f"backtest) — {payload['n_accumulating']} accumulating | reconcile "
          f"{'OK ✓' if rec else ('MISMATCH ✗' if rec is False else '?')} ===", flush=True)
    for p in payload["picks"]:
        pb = f"{p['pb_ratio']:>5}" if p.get("pb_ratio") is not None else "  n/a"
        acc = "  🔵 ACCUMULATING (4x)" if p.get("accumulating") else ""
        _sec = (p.get("sector") or "?")[:22]
        _ac = f"{p['acceleration']:>+6.1f}" if p.get("acceleration") is not None else "   n/a"
        print(f"  #{p['rank']:>2} {_sec:22} accel {_ac}  ->  "
              f"{p['pick']:8} driftP/B {pb}  ${p.get('last_close')}  alloc {p.get('pct_alloc')}%{acc}", flush=True)


if __name__ == "__main__":
    main()
