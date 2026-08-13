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
    px = load_candles(univ)

    picks = []
    for rank, (etf, m) in enumerate(ranked, 1):
        name = name_by_etf.get(etf, etf)
        cands = [(t, funds.get(t, {}).get("pb_ratio")) for t in holds_by_etf[etf]]
        cands = [(t, pb) for t, pb in cands if pb is not None and pb > 0]
        row = {"rank": rank, "sector": name, "etf": etf, "momentum_6m": round(m, 1),
               "n_candidates": len(cands)}
        if cands:
            t, pb = min(cands, key=lambda x: x[1])
            f = funds.get(t, {})
            dfp = px.get(t)
            row.update({
                "pick": t, "pb_ratio": round(pb, 2),
                "last_close": round(float(dfp["Close"].iloc[-1]), 2) if dfp is not None and len(dfp) else None,
                "market_cap": f.get("market_cap"), "pe_ratio": f.get("pe_ratio"),
                "forward_pe": f.get("forward_pe"), "profit_margin": f.get("profit_margin"),
                "revenue_growth": f.get("revenue_growth"),
                "pick_sectors": sector_holdings.get_sectors_for_ticker(t)})
        else:
            row["pick"] = None
        picks.append(row)

    payload = {
        "computed_at": pd.Timestamp.utcnow().isoformat(),
        "params": {"lookback_days": LOOKBACK_D, "top_n_sectors": TOP_N_SECTORS,
                   "rule": "top trailing-6mo-momentum sectors -> cheapest positive-P/B holding",
                   "backtest": "arm3_lowpb: +237% total / +154% vs SPY / t=2.09 (55 monthly periods)"},
        "picks": picks,
        "note": ("The rotation is a SELECTION FILTER, not an ETF trade (ETF rotation loses to SPY). "
                 "Monthly-rebalance value basket; directional, no fees; stock-universe survivorship."),
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
        if p.get("pick"):
            print(f"  #{p['rank']:>2} {p['sector']:22} mom6 {p['momentum_6m']:>+6.1f}%  ->  "
                  f"{p['pick']:6} P/B {p['pb_ratio']:>5}  ${p['last_close']}", flush=True)
        else:
            print(f"  #{p['rank']:>2} {p['sector']:22} mom6 {p['momentum_6m']:>+6.1f}%  ->  (no positive-P/B holding)", flush=True)


if __name__ == "__main__":
    main()
