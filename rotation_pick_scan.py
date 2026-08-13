#!/usr/bin/env python3
"""LIVE rotation-pick scanner — the ONLY sector-rotation strategy with real alpha, as a live signal.

The backtest verdict: rotating sector ETFs LOSES to SPY (-27% to -82% vs SPY). The edge is using the
rotation as a SECTOR-SELECTION FILTER feeding a VALUE stock-pick: rank the 93 sector ETFs by trailing
6-month return, take the top-N strengthening sectors, and in EACH pick the CHEAPEST positive-P/B stock
among its holdings. That arm (arm3_lowpb in backtest_lowpb.py) did +237% total / +154% vs SPY / t=2.09
— the only t>2 rotation result. This surfaces that basket live.

-> BacktestResult[rotation_picks] + JSON. Directional / no fees; monthly-rebalance basket, not intraday.
Run: docker exec rotation-backend-1 python -u /app/rotation_pick_scan.py
"""
import os, json, warnings
warnings.filterwarnings("ignore")
os.environ.setdefault("DJANGO_SETTINGS_MODULE", "rotation.settings")
import django
django.setup()
import pandas as pd
import sector_holdings
from seq_fundamental_study import load_candles

LOOKBACK_D = 126          # ~6 trading months (matches backtest LOOKBACK=6 monthly)
TOP_N_SECTORS = 10
OUT = None


def build():
    from core.models import Sector
    sectors = [(n, e) for n, e in Sector.objects.values_list("name", "etf") if e]
    etfs = [e for _, e in sectors]
    name_by_etf = {e: n for n, e in sectors}

    etf_daily = load_candles(etfs)
    mom = {}
    for _, etf in sectors:
        df = etf_daily.get(etf)
        if df is None or len(df) < LOOKBACK_D + 1:
            continue
        c = df["Close"]
        mom[etf] = float((c.iloc[-1] / c.iloc[-1 - LOOKBACK_D] - 1) * 100)
    ranked = sorted(mom.items(), key=lambda kv: -kv[1])[:TOP_N_SECTORS]
    print(f"{len(mom)}/{len(etfs)} sectors ranked; top {len(ranked)} strengthening", flush=True)

    holds_by_etf = {etf: [t for t in sector_holdings.get_holdings(name_by_etf.get(etf, etf))
                          if t not in (etf, "SPY", "QQQ")] for etf, _ in ranked}
    univ = sorted({t for hs in holds_by_etf.values() for t in hs})
    # P/B lives on Fundamental.pb_ratio (not in load_fundamentals' field set); latest row per ticker.
    from core.models import Fundamental
    funds = {}
    for r in (Fundamental.objects.filter(ticker__in=univ).order_by("ticker", "-date")
              .values("ticker", "pb_ratio", "market_cap", "pe_ratio", "forward_pe",
                      "profit_margin", "revenue_growth")):
        funds.setdefault(r["ticker"], r)
    px = load_candles(univ + [e for e, _ in ranked])
    # Profitability guard (ex_trap_turn): exclude cheap-P/B value traps (unprofitable + eroding book +
    # not improving); keep turnarounds. Backtested +231.7% vs +214.7% baseline, better t/Sharpe/DD.
    from profitability_guard import guard_flags
    gflags = guard_flags(univ)

    picks = []
    for rank, (etf, m) in enumerate(ranked, 1):
        name = name_by_etf.get(etf, etf)
        cands = [(t, funds.get(t, {}).get("pb_ratio")) for t in holds_by_etf[etf]]
        cands = [(t, pb) for t, pb in cands if pb is not None and pb > 0]
        # drop traps; if that empties the sector, fall back to the unguarded set (don't lose the sector)
        guarded = [(t, pb) for t, pb in cands if not gflags.get(t, {}).get("trap")]
        use = guarded if guarded else cands
        row = {"rank": rank, "sector": name, "etf": etf, "momentum_6m": round(m, 1),
               "n_candidates": len(cands), "n_after_guard": len(guarded)}
        if use:
            t, pb = min(use, key=lambda x: x[1])
            f = funds.get(t, {})
            g = gflags.get(t, {})
            dfp = px.get(t)
            row.update({
                "pick": t, "is_etf_proxy": False, "pb_ratio": round(pb, 2),
                "guard_status": g.get("status"), "margin_pct": g.get("margin"),
                "net_income": g.get("net_income"), "improving": g.get("improving"),
                "last_close": round(float(dfp["Close"].iloc[-1]), 2) if dfp is not None and len(dfp) else None,
                "market_cap": f.get("market_cap"), "pe_ratio": f.get("pe_ratio"),
                "forward_pe": f.get("forward_pe"), "profit_margin": f.get("profit_margin"),
                "revenue_growth": f.get("revenue_growth"),
                "pick_sectors": sector_holdings.get_sectors_for_ticker(t)})
        else:
            # No positive-P/B stock (pure commodity ETF, or a market whose holdings lack P/B like
            # foreign listings) -> HOLD THE ETF ITSELF, treated as the position. This is a trend/
            # momentum sleeve, not a value pick -- flagged is_etf_proxy so the app can badge it.
            dfe = px.get(etf)
            row.update({
                "pick": etf, "is_etf_proxy": True, "pb_ratio": None,
                "last_close": round(float(dfe["Close"].iloc[-1]), 2) if dfe is not None and len(dfe) else None,
                "market_cap": None, "pe_ratio": None, "forward_pe": None,
                "profit_margin": None, "revenue_growth": None, "pick_sectors": [name]})
        picks.append(row)

    payload = {
        "computed_at": pd.Timestamp.utcnow().isoformat(),
        "params": {"lookback_days": LOOKBACK_D, "top_n_sectors": TOP_N_SECTORS,
                   "rule": ("top trailing-6mo-momentum sectors -> cheapest positive-P/B holding; if a "
                            "sector has no positive-P/B stock (pure commodity ETF, or foreign market "
                            "whose holdings lack P/B) HOLD THE ETF itself, treated as the position"),
                   "backtest": ("value picks: arm3_lowpb +237%/+154% vs SPY t2.09; with commodity/market "
                                "ETFs held as positions (fallback) ~+156% vs SPY — see Entry Signal tab")},
        "picks": picks,
        "note": ("Value picks are the alpha; commodity/market sleeves with no P/B stock are held via their "
                 "ETF (trend sleeve, not value — badged in the app). Monthly-rebalance; directional, no fees."),
    }
    return payload


def main():
    global OUT
    from pathlib import Path
    OUT = Path(__file__).resolve().parent / ".data" / "studies" / "rotation_picks.json"
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
    print("\n=== ROTATION PICKS (cheapest-P/B in each strengthening sector) ===", flush=True)
    for p in payload["picks"]:
        pb = f"{p['pb_ratio']:>5}" if p.get("pb_ratio") is not None else "  ETF"
        tag = "  [ETF held as position]" if p.get("is_etf_proxy") else ""
        print(f"  #{p['rank']:>2} {p['sector']:22} mom6 {p['momentum_6m']:>+6.1f}%  ->  "
              f"{p['pick']:8} P/B {pb}  ${p['last_close']}{tag}", flush=True)


if __name__ == "__main__":
    main()
