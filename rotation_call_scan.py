#!/usr/bin/env python3
"""THE HEADLINE ROTATION CALL — the flagship signal that reflects what actually beats SPY.

Assembled from the three things that survived the whole rotation research:
  1. WHICH sectors  -> macro-REGIME leaders (regime_scan). Price-momentum has ~0 lift; regime
     conditioning is the only prospective angle with real leadership lift. (IN-SAMPLE caveat.)
  2. WHICH stock    -> cheapest positive-P/B holding in each leader sector (the value pick; the ONLY
     equity alpha with t>2 in backtest_lowpb: arm3_lowpb +154% vs SPY).
  3. WHEN to enter  -> an OVERSOLD DIP on the pick's OWN absolute price (entry_signal_study winner
     rsi10_lt_45: buying the value name into weakness ADDS ~+5%/pick; buying strength/confirmation
     SUBTRACTS). RSI(10) < 45 = enter zone; < 35 = deep-dip strong entry.

This is explicitly NOT the sector "ROTATE IN / TREND TURN" alert (that backtests as the WORST
beat-SPY signal). -> BacktestResult[rotation_call] + JSON. Directional / monthly-rebalance / no fees.
Run: MSYS_NO_PATHCONV=1 docker exec rotation-backend-1 python -u /app/rotation_call_scan.py
"""
import os, json, warnings
warnings.filterwarnings("ignore")
os.environ.setdefault("DJANGO_SETTINGS_MODULE", "rotation.settings")
import django
django.setup()
import pandas as pd
import ta
import sector_holdings
import regime_scan
from seq_fundamental_study import load_candles

TOP_N_SECTORS = 10
ENTER_RSI = 45      # entry_signal_study winner (rsi10_lt_45): dip-entry zone
DEEP_RSI = 35       # best risk-adjusted (rsi10_lt_35): deep-dip strong entry


def _entry_state(rsi):
    if rsi is None:
        return ("unknown", None)
    if rsi < DEEP_RSI:
        return ("STRONG DIP — enter", "deep")          # deepest oversold, best per-pick lift
    if rsi < ENTER_RSI:
        return ("DIP — enter (winning zone)", "enter")  # the study's winning entry gate
    if rsi < 60:
        return ("neutral — wait for a pullback", "wait")
    return ("extended — wait for a pullback", "extended")


def build():
    # 1) regime leaders NOW (loads only ETF/SPY/TLT/TIP candles -> light on the box)
    reg = regime_scan.build()
    leaders = reg["leaders_now"][:TOP_N_SECTORS]
    now_labels = reg["now_labels"]

    from core.models import Fundamental
    holds_by_etf, name_by_etf = {}, {}
    for r in leaders:
        etf, name = r["etf"], r["sector"]
        name_by_etf[etf] = name
        holds_by_etf[etf] = [t for t in sector_holdings.get_holdings(name)
                             if t not in (etf, "SPY", "QQQ")]
    univ = sorted({t for hs in holds_by_etf.values() for t in hs})

    # cheapest positive P/B per sector (P/B on Fundamental.pb_ratio, latest row per ticker)
    funds = {}
    for r in (Fundamental.objects.filter(ticker__in=univ).order_by("ticker", "-date")
              .values("ticker", "pb_ratio", "market_cap", "pe_ratio", "forward_pe",
                      "profit_margin", "revenue_growth")):
        funds.setdefault(r["ticker"], r)
    px = load_candles(univ + [r["etf"] for r in leaders])
    # Profitability guard (ex_trap_turn): drop cheap-P/B value traps, keep turnarounds (backtested win).
    from profitability_guard import guard_flags
    gflags = guard_flags(univ)

    def _rsi_state(dfp):
        rsi, last = None, None
        if dfp is not None and len(dfp) >= 15:
            c = dfp["Close"]
            last = float(c.iloc[-1])
            rv = ta.momentum.rsi(c, window=10).iloc[-1]
            rsi = round(float(rv), 1) if pd.notna(rv) else None
        return rsi, last

    picks, ready = [], 0
    for r in leaders:
        etf, name = r["etf"], r["sector"]
        cands = [(t, funds.get(t, {}).get("pb_ratio")) for t in holds_by_etf.get(etf, [])]
        cands = [(t, pb) for t, pb in cands if pb is not None and pb > 0]
        guarded = [(t, pb) for t, pb in cands if not gflags.get(t, {}).get("trap")]
        use = guarded if guarded else cands
        row = {"sector": name, "etf": etf,
               "regime_score_pct": r.get("regime_score_pct"),
               "combo_hit_pct": r.get("combo_hit_pct"), "combo_mean_pct": r.get("combo_mean_pct"),
               "combo_n": r.get("combo_n"), "n_candidates": len(cands), "n_after_guard": len(guarded)}
        if use:
            t, pb = min(use, key=lambda x: x[1])
            f = funds.get(t, {})
            g = gflags.get(t, {})
            rsi, last = _rsi_state(px.get(t))
            state_txt, state_key = _entry_state(rsi)
            if state_key in ("enter", "deep"):
                ready += 1
            row.update({
                "pick": t, "is_etf_proxy": False, "pb_ratio": round(pb, 2),
                "guard_status": g.get("status"), "margin_pct": g.get("margin"),
                "net_income": g.get("net_income"), "improving": g.get("improving"),
                "last_close": round(last, 2) if last else None,
                "rsi10": rsi, "entry_state": state_txt, "entry_key": state_key,
                "market_cap": f.get("market_cap"), "pe_ratio": f.get("pe_ratio"),
                "forward_pe": f.get("forward_pe"), "profit_margin": f.get("profit_margin"),
                "revenue_growth": f.get("revenue_growth"),
                "pick_sectors": sector_holdings.get_sectors_for_ticker(t)})
        else:
            # commodity ETF (no stock) or foreign market whose holdings lack P/B -> hold the ETF itself,
            # treated as the position; still time the entry on the ETF's own oversold dip.
            rsi, last = _rsi_state(px.get(etf))
            state_txt, state_key = _entry_state(rsi)
            if state_key in ("enter", "deep"):
                ready += 1
            row.update({
                "pick": etf, "is_etf_proxy": True, "pb_ratio": None,
                "last_close": round(last, 2) if last else None,
                "rsi10": rsi, "entry_state": state_txt, "entry_key": state_key,
                "market_cap": None, "pe_ratio": None, "forward_pe": None,
                "profit_margin": None, "revenue_growth": None, "pick_sectors": [name]})
        picks.append(row)

    payload = {
        "computed_at": pd.Timestamp.utcnow().isoformat(),
        "regime": {"date": reg["now"]["date"], **now_labels},
        "params": {"top_n_sectors": TOP_N_SECTORS, "enter_rsi": ENTER_RSI, "deep_rsi": DEEP_RSI,
                   "rule": ("regime-leader sectors -> cheapest positive-P/B holding -> enter on an "
                            "oversold dip (RSI(10) < 45; < 35 = deep-dip strong entry)")},
        "picks": picks,
        "ready_to_enter": ready,
        "components": {
            "sectors": "macro-regime leadership (Regime tab) — the only prospective sector angle with lift",
            "stock": "cheapest positive-P/B value pick (arm3_lowpb, +154% vs SPY, t2.09)",
            "entry": "oversold dip on the pick's absolute price (entry_signal_study: dip adds, strength subtracts)",
        },
        "caveat": ("Regime leadership is IN-SAMPLE (hypothesis generator, not walk-forward validated). "
                   "Value pick + oversold entry are backtested but directional/no-fees/survivorship over "
                   "~5y (one regime). This is the best-reasoned rotation call, not a guarantee. It is NOT "
                   "the sector TREND TURN alert, which backtests as the worst beat-SPY signal."),
    }
    return payload


def main():
    from pathlib import Path
    out = Path(__file__).resolve().parent / ".data" / "studies" / "rotation_call.json"
    payload = build()
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2, default=str))
    try:
        from core.models import BacktestResult
        from django.utils import timezone
        BacktestResult.objects.update_or_create(
            kind="rotation_call",
            defaults={"payload": json.loads(json.dumps(payload, default=str)), "computed_at": timezone.now()})
        print("Saved BacktestResult[rotation_call]", flush=True)
    except Exception as e:
        print("DB save failed:", e, flush=True)
    rg = payload["regime"]
    print(f"\n=== ROTATION CALL — regime {rg['rates']}/{rg['inflation']}/{rg['market']} ({rg['date']}) ===", flush=True)
    print(f"{payload['ready_to_enter']} pick(s) at an oversold-dip entry now\n", flush=True)
    for p in payload["picks"]:
        pb = f"{p['pb_ratio']:>5}" if p.get("pb_ratio") is not None else "  ETF"
        tag = " [ETF held as position]" if p.get("is_etf_proxy") else ""
        print(f"  {p['sector']:24} score {p['regime_score_pct']:>+6}%  ->  {p['pick']:8} "
              f"P/B {pb}  RSI {p['rsi10']}  [{p['entry_state']}]{tag}", flush=True)


if __name__ == "__main__":
    main()
