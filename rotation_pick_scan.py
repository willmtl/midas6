#!/usr/bin/env python3
"""LIVE rotation-pick scanner — the ONLY sector-rotation strategy with real alpha, as a live signal.

The backtest verdict: rotating sector ETFs LOSES to SPY (-27% to -82% vs SPY). The edge is using the
rotation as a SECTOR-SELECTION FILTER feeding a VALUE stock-pick: rank the sector ETFs by momentum
ACCELERATION, take the top-N inflecting sectors, and in EACH pick the CHEAPEST positive-P/B stock,
profit-guard + low-debt, size-tilted small, div_2x conviction-weighted.

FLAGSHIP = usca_small (survivorship-de-biased, USD-incl-FX, point-in-time, no-penny, $50M-pharma floor,
delisting-audited): +790.4% total / Sharpe 1.49 / DD -20.5% / t3.34 (survivorship_smallcap_study.py).
The core edge is US+CA cheapest-value SMALL-CAP in accelerating sectors. To express that live the candidate
pool is BROADENED from ETF holdings (large-cap biased) to the full live US+CA GICS universe (every stock
whose Fundamental.sector maps to a top core-sector ETF) — the same construction the backtest uses.

CAP-AWARE VALUE METRIC (2026-08-16, reconciled): in SMALL-CAP the earnings multiple beats the book multiple
(P/E>P/B is small-cap-specific — book is unreliable for asset-light/distressed small names, and unprofitable
small-caps are traps). So within the small-cap tier the pick is TWO-STAGE: top-5 cheapest raw P/B -> cheapest
trailing P/E among profitable. Backtest +196pp (790->986%), survivorship + walk-forward robust (survivors-only
small-cap = the live universe: P/E +302pp vs P/B). LARGE-CAP fallback keeps cheapest raw P/B (P/E LOSES in
large-cap: cheap-P/E = cyclical earnings peak that mean-reverts).
(Deliberately NOT the speculative-sector 'skip_wide' variant: 1281% in-sample but REFUTED out-of-sample
by walk-forward — see delisted-survivorship memory.)

-> BacktestResult[rotation_picks] + JSON. Directional / no fees; monthly-rebalance basket, not intraday.
Run: docker exec rotation-backend-1 python -u /app/rotation_pick_scan.py
"""
import os, json, warnings
warnings.filterwarnings("ignore")
os.environ.setdefault("DJANGO_SETTINGS_MODULE", "rotation.settings")
import django
django.setup()
import numpy as np
import pandas as pd
import sector_holdings
from seq_fundamental_study import load_candles

LOOKBACK_D = 126          # ~6 trading months (matches backtest LOOKBACK=6 monthly)
TOP_N_SECTORS = 10
MIN_DOLLAR_VOL = 5e6      # tradeability floor: drop OTC/thin names (<$5M/day). Backtest sweet spot: removes
                         # untradeable pink sheets (e.g. $0/day) AND the net-loser thin band -> +298% honest
                         # tradeable vs +376% inflated by unbuyable pops; higher floors delete real small-cap winners.
# ── usca_small flagship rules (2026-08-16 survivorship study; honest backtest 790%/Sh1.49/t3.34) ──
MIN_PB = 0.1              # reject sub-0.1 book multiples = corrupt/near-zero-equity fundamentals (not real value)
SMALL_CAP_USD = 2e9       # prefer cheapest-P/B among <$2B USD names (size premium is the biggest lever)
PHARMA_ETFS = {"XLV", "XBI", "ARKG"}
MICRO_PHARMA_MIN = 5e7    # bar sub-$50M pharma/biotech = nano binary-blowup lottery tickets
CAD_USD = 0.73            # rough CAD->USD for the size bucket (US+Canada universe; only Canada is non-USD)
# GICS-sector -> core sector-ETF map (mirror of survivorship_smallcap_study.GIC_TO_ETF). Used to BROADEN each
# top core-sector pick's candidate pool from "current ETF holdings" (large-cap biased) to "every live US+CA
# stock in that GICS sector" — the same construction the 790% backtest uses. Fundamental.sector supplies the
# GICS label. Only CORE sectors have a broad bucket; thematic ETFs (SMH/XBI/TAN/...) keep ETF-holdings-only,
# exactly as in the backtest (GICS never maps to a thematic ETF).
GIC_TO_ETF = {
    "Technology": "XLK", "Information Technology": "XLK", "Financials": "XLF", "Financial Services": "XLF",
    "Health Care": "XLV", "Healthcare": "XLV", "Energy": "XLE", "Consumer Cyclical": "XLY",
    "Consumer Discretionary": "XLY", "Consumer Defensive": "XLP", "Consumer Staples": "XLP",
    "Industrials": "XLI", "Basic Materials": "XLB", "Materials": "XLB", "Real Estate": "XLRE",
    "Communication Services": "XLC", "Utilities": "XLU",
}
OUT = None


def _is_usca(t):
    """US (no exchange suffix) or Canada (.TO / .V) — the honest flagship universe (cleanest data, no FX drag)."""
    return ("." not in t) or t.rsplit(".", 1)[1] in ("TO", "V")


def _usd_mcap(t, funds):
    """Market cap in USD (convert Canadian listings from CAD). None if unknown."""
    mc = funds.get(t, {}).get("market_cap")
    if mc is None:
        return None
    return mc * CAD_USD if (t.endswith(".TO") or t.endswith(".V")) else mc


def _dvol(dfp):
    """20-day average dollar volume ($). None if no volume/data."""
    if dfp is None or "Volume" not in dfp or len(dfp) < 5:
        return None
    dv = (dfp["Close"] * dfp["Volume"]).tail(20).mean()
    return float(dv) if pd.notna(dv) else None


def _stock_accel(dfp):
    """Per-stock momentum ACCELERATION (3mo-now minus 3mo-3ago), as an INFO indicator. NOTE: at the stock
    level we FADE this — backtest shows the value pick does BETTER on FADING stocks (+3.6% vs +1.6%): buy
    the cheap laggard that hasn't turned up yet, not the one already accelerating. (Sectors = follow accel;
    stocks = fade accel.)"""
    if dfp is None or len(dfp) < LOOKBACK_D + 1:
        return None
    c = dfp["Close"]
    m3n = c.iloc[-1] / c.iloc[-1 - LOOKBACK_D // 2] - 1
    m3p = c.iloc[-1 - LOOKBACK_D // 2] / c.iloc[-1 - LOOKBACK_D] - 1
    return round(float((m3n - m3p) * 100), 1)


def _ad_divergence(dfp, half):
    """A/D DIVERGENCE flag on the held instrument: the Accumulation/Distribution Line (ADL = cumsum of
    MoneyFlowMultiplier*Volume, TradingView ta.accdist) rising over ~3mo WHILE price fell over ~3mo = quiet
    accumulation into weakness. Mirrors the backtested volume-conviction signal exactly (ad_slope3>0 AND
    px_ret3<0). Returns (is_div, ad_slope_3m, px_ret_3m_pct)."""
    if dfp is None or not {"High", "Low", "Close", "Volume"}.issubset(dfp.columns) or len(dfp) < half + 1:
        return False, None, None
    h, l, c, v = dfp["High"], dfp["Low"], dfp["Close"], dfp["Volume"]
    rng = (h - l).replace(0, np.nan)
    mfm = ((c - l) - (h - c)) / rng
    adl = (mfm.fillna(0) * v).cumsum()
    ad_slope = float(adl.iloc[-1] - adl.iloc[-1 - half])
    px_ret = float(c.iloc[-1] / c.iloc[-1 - half] - 1)
    is_div = bool(ad_slope > 0 and px_ret < 0)
    return is_div, round(ad_slope, 1), round(px_ret * 100, 1)


def _alt_signals(tickers):
    """Insider / Congress BUYING confirmation flags per ticker (informational — a cheap value pick that
    ALSO has insiders or a legislator buying is higher-conviction: 56-62% win vs ~54%). NOT a selection
    driver (tilting toward them hurts) — a confidence overlay only."""
    from core.models import InsiderBuy, CongressTrade
    from django.db.models import Sum, Count
    from datetime import date, timedelta
    out = {t: {"insider_buying": False, "congress_buying": False} for t in tickers}
    ins_cut = date.today() - timedelta(days=90)          # net insider buying, trailing ~1 quarter
    cong_cut = date.today() - timedelta(days=180)         # congress discloses with lag -> wider window
    for r in (InsiderBuy.objects.filter(ticker__in=tickers, filed_date__gte=ins_cut)
              .values("ticker").annotate(b=Sum("buy_value"), s=Sum("sell_value"))):
        if (r["b"] or 0) > (r["s"] or 0):
            out[r["ticker"]]["insider_buying"] = True
    for r in (CongressTrade.objects.filter(ticker__in=tickers, transaction_type="buy", report_date__gte=cong_cut)
              .values("ticker").annotate(n=Count("id"))):
        if r["n"] > 0:
            out[r["ticker"]]["congress_buying"] = True
    return out


def build():
    from core.models import Sector
    sectors = [(n, e) for n, e in Sector.objects.values_list("name", "etf") if e]
    etfs = [e for _, e in sectors]
    name_by_etf = {e: n for n, e in sectors}

    etf_daily = load_candles(etfs)
    HALF = LOOKBACK_D // 2          # ~3 trading months, for the acceleration signal
    mom, mom3, accel_val = {}, {}, {}
    for _, etf in sectors:
        df = etf_daily.get(etf)
        if df is None or len(df) < LOOKBACK_D + 1:
            continue
        c = df["Close"]
        mom[etf] = float((c.iloc[-1] / c.iloc[-1 - LOOKBACK_D] - 1) * 100)
        m3_now = c.iloc[-1] / c.iloc[-1 - HALF] - 1
        m3_prev = c.iloc[-1 - HALF] / c.iloc[-1 - 2 * HALF] - 1
        mom3[etf] = round(float(m3_now * 100), 1)
        accel_val[etf] = float((m3_now - m3_prev) * 100)   # ACCELERATION of 3mo momentum
    # SECTOR SIGNAL = momentum ACCELERATION (3mo-now minus 3mo-3mo-ago). Split-corrected honest flagship
    # +313.2% total / +213pp vs SPY / Sharpe 1.50 / DD -15.5%. Replaces 6mo-momentum LEVEL, which ranked
    # sectors by what already ran (late) — acceleration catches the move at its inflection (turning up NOW).
    ranked = sorted(accel_val.items(), key=lambda kv: -kv[1])[:TOP_N_SECTORS]
    print(f"{len(mom)}/{len(etfs)} sectors ranked by ACCELERATION; top {len(ranked)} (inflecting up)", flush=True)

    holds_by_etf = {etf: [t for t in sector_holdings.get_holdings(name_by_etf.get(etf, etf))
                          if t not in (etf, "SPY", "QQQ")] for etf, _ in ranked}
    from core.models import Fundamental
    # ── BROADEN candidate pool to the full live US+CA GICS universe (match the 790% backtest engine) ──
    # ETF holdings alone are large-cap biased (XLV/XLI hold big names) so the small-cap edge — the flagship's
    # biggest lever — can't be expressed. Pull EVERY live US+CA stock whose Fundamental.sector maps to a top
    # CORE-sector ETF and add it to that sector's pool (thematic ETFs stay holdings-only, as in the backtest).
    etf_to_gics = {}
    for gic, e in GIC_TO_ETF.items():
        etf_to_gics.setdefault(e, set()).add(gic)
    needed_gics = set().union(*(etf_to_gics.get(etf, set()) for etf, _ in ranked)) if ranked else set()
    gics_by_ticker = {}
    if needed_gics:
        for r in (Fundamental.objects.filter(sector__in=needed_gics)
                  .order_by("ticker", "-date").values("ticker", "sector")):
            gics_by_ticker.setdefault(r["ticker"], r["sector"])
    n_added = 0
    for etf, _ in ranked:
        gset = etf_to_gics.get(etf)
        if not gset:
            continue                                   # thematic ETF: no broad GICS bucket -> holdings only
        cur = set(holds_by_etf[etf])
        extra = [t for t, s in gics_by_ticker.items()
                 if s in gset and _is_usca(t) and t not in cur and t not in (etf, "SPY", "QQQ")]
        n_added += len(extra)
        holds_by_etf[etf] = sorted(cur | set(extra))
    print(f"GICS broadening: +{n_added} live US+CA names beyond ETF holdings across "
          f"{sum(1 for e,_ in ranked if e in etf_to_gics)} core-sector picks", flush=True)

    univ = sorted({t for hs in holds_by_etf.values() for t in hs})
    # P/B lives on Fundamental.pb_ratio (not in load_fundamentals' field set); latest row per ticker.
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
    alt = _alt_signals(univ)          # insider / congress buying confirmation flags

    picks = []
    held = set()          # cross-sector dedup: a name can sit in multiple GICS/ETF pools (e.g. INSP in Medtech
                          # AND Healthcare) — never hold it twice; sectors picked in accel rank order (matches backtest).
    skipped = []          # sectors dropped for having no qualifying US/CA value stock (commodity/bond/foreign)
    for rank, (etf, acc) in enumerate(ranked, 1):
        name = name_by_etf.get(etf, etf)
        cands = [(t, funds.get(t, {}).get("pb_ratio")) for t in holds_by_etf[etf] if t not in held]
        cands = [(t, pb) for t, pb in cands if pb is not None and pb > MIN_PB]              # P/B sanity floor (0.1)
        cands = [(t, pb) for t, pb in cands if _is_usca(t)]                                 # US + Canada only
        cands = [(t, pb) for t, pb in cands if (_dvol(px.get(t)) or 0) >= MIN_DOLLAR_VOL]   # drop OTC/thin (untradeable)
        if etf in PHARMA_ETFS:                                                              # no sub-$50M pharma nano-blowups
            cands = [(t, pb) for t, pb in cands if (_usd_mcap(t, funds) or 0) >= MICRO_PHARMA_MIN]
        # Factor Lab winner = guard + low_debt: drop traps, then prefer low-debt (debt/equity<1) names;
        # layered fallback so a sector is never lost.
        guarded = [(t, pb) for t, pb in cands if not gflags.get(t, {}).get("trap")]
        lowdebt = [(t, pb) for t, pb in guarded if gflags.get(t, {}).get("low_debt")]
        use = lowdebt if lowdebt else (guarded if guarded else cands)
        # small-cap preference: among the surviving candidates, restrict to <$2B USD if any qualify (the size
        # premium is the flagship's biggest lever); fall back to all if a sector has no small name.
        small = [(t, pb) for t, pb in use if (_usd_mcap(t, funds) or 9e18) < SMALL_CAP_USD]
        is_small_tier = bool(small)
        use = small if small else use
        row = {"rank": rank, "sector": name, "etf": etf, "momentum_6m": round(mom.get(etf, 0), 1),
               "momentum_3m": mom3.get(etf), "acceleration": round(acc, 1),
               "n_candidates": len(cands), "n_after_guard": len(guarded), "n_low_debt": len(lowdebt)}
        if use:
            # CAP-AWARE selection (2026-08-16, reconciled): in SMALL-CAP, earnings-multiple beats book-multiple
            # (P/E>P/B is small-cap-specific: book unreliable for asset-light/distressed small names, and
            # unprofitable small-caps are traps). Two-stage: top-5 cheapest raw P/B -> cheapest trailing P/E
            # among PROFITABLE (pe>0); fall back to cheapest P/B if none profitable. Backtest +196pp (790->986%),
            # survivorship + walk-forward robust (survivors-only-small = live universe: P/E +302pp). LARGE-CAP
            # fallback keeps cheapest raw P/B (P/E LOSES in large-cap: cheap-P/E = cyclical earnings peak).
            if is_small_tier:
                top5 = sorted(use, key=lambda x: x[1])[:5]
                prof = [(tk, pbv) for tk, pbv in top5 if (funds.get(tk, {}).get("pe_ratio") or 0) > 0]
                if prof:
                    t, pb = min(prof, key=lambda x: funds.get(x[0], {}).get("pe_ratio"))
                    sel_basis = "pe_smallcap"
                else:
                    t, pb = top5[0]
                    sel_basis = "pb_smallcap_noprofit"
            else:
                t, pb = min(use, key=lambda x: x[1])
                sel_basis = "pb_largecap"
            held.add(t)
            f = funds.get(t, {})
            g = gflags.get(t, {})
            dfp = px.get(t)
            dv = _dvol(dfp)
            row.update({
                "pick": t, "is_etf_proxy": False, "pb_ratio": round(pb, 2),
                "selection_basis": sel_basis,
                "guard_status": g.get("status"), "margin_pct": g.get("margin"),
                "debt_to_equity": g.get("debt_to_equity"),
                "net_income": g.get("net_income"), "improving": g.get("improving"),
                "last_close": round(float(dfp["Close"].iloc[-1]), 2) if dfp is not None and len(dfp) else None,
                "dollar_vol_m": round(dv / 1e6, 1) if dv else None,
                "stock_acceleration": _stock_accel(dfp),
                "accumulating": (_div := _ad_divergence(dfp, HALF))[0],
                "ad_slope_3m": _div[1], "price_ret_3m": _div[2],
                "insider_buying": alt.get(t, {}).get("insider_buying", False),
                "congress_buying": alt.get(t, {}).get("congress_buying", False),
                "market_cap": f.get("market_cap"), "pe_ratio": f.get("pe_ratio"),
                "forward_pe": f.get("forward_pe"), "profit_margin": f.get("profit_margin"),
                "revenue_growth": f.get("revenue_growth"),
                "pick_sectors": sector_holdings.get_sectors_for_ticker(t)})
        else:
            # No qualifying US/CA positive-P/B value stock (pure commodity/bond/crypto sector, or a
            # foreign/index sleeve whose holdings lack P/B). SKIP THE SLOT — do NOT hold the raw ETF.
            # Backtest (usca_small_proxy vs usca_small): holding these as ETF proxies SUBTRACTS -468.3pp
            # total (322.1% vs 790.4%), Sharpe 1.34 vs 1.49 — the alpha is entirely in the value pick, a
            # raw beta sleeve only dilutes it. Remaining picks renormalize to 100% in the conviction step.
            skipped.append({"rank": rank, "sector": name, "etf": etf, "acceleration": round(acc, 1)})
            print(f"  #{rank:>2} {name:22} ({etf}): no qualifying US/CA value stock -> SLOT SKIPPED "
                  f"(proxy-hold backtests -468pp)", flush=True)
            continue
        picks.append(row)

    # div_2x CONVICTION WEIGHTING: keep the full basket (breadth intact — same cheapest-P/B pick per sector),
    # just OVERWEIGHT names showing A/D divergence (accumulation into weakness) 2x, then renormalize to 100%.
    # Backtested (split-corrected, honest) +313.2% vs +246% equal-weight / Sharpe 1.50 vs 1.36 / DD -15.5% —
    # the first orthogonal overlay to improve the flagship on all axes (still beats equal in both halves). 2x not 3x
    # ON PURPOSE: divergence fires on only ~1 pick/mo, so 3x concentrates ~25% into a single name; 2x caps that.
    CONVICTION_MULT = 2.0
    for p in picks:
        p["conviction_weight"] = CONVICTION_MULT if p.get("accumulating") else 1.0
    tot_w = sum(p["conviction_weight"] for p in picks) or 1.0
    for p in picks:
        p["pct_alloc"] = round(p["conviction_weight"] / tot_w * 100, 1)
    n_accum = sum(1 for p in picks if p.get("accumulating"))

    payload = {
        "computed_at": pd.Timestamp.utcnow().isoformat(),
        "n_accumulating": n_accum,
        "params": {"lookback_days": LOOKBACK_D, "top_n_sectors": TOP_N_SECTORS,
                   "flagship": "usca_small",
                   "universe": ("US+Canada, GICS-BROADENED: candidates = ETF holdings ∪ every live US/CA stock "
                                "whose Fundamental.sector maps to the top core-sector ETF (surfaces small-caps "
                                "the big sector ETFs don't hold — the flagship's biggest lever)"),
                   "sector_signal": "momentum ACCELERATION (3mo-now minus 3mo-3ago) — catches the inflection, not the 6mo run",
                   "size_tilt": f"prefer cheapest-P/B among <${SMALL_CAP_USD/1e9:g}B USD names (small-cap size premium); fall back to all-cap if none",
                   "conviction_weighting": ("div_2x: equal-weight basket, but A/D-divergence names (ADL rising ~3mo "
                                            "while price fell ~3mo = accumulation into weakness) get 2x weight, "
                                            "renormalized; pct_alloc per pick is the deployable weight."),
                   "value_metric": ("CAP-AWARE: small-cap tier = top-5 cheapest raw P/B then cheapest trailing "
                                    "P/E among profitable (P/E>P/B is small-cap-specific, +196pp backtest, "
                                    "survivorship+walk-forward robust); large-cap fallback = cheapest raw P/B "
                                    "(P/E loses in large-cap). selection_basis per pick: pe_smallcap / "
                                    "pb_smallcap_noprofit / pb_largecap"),
                   "rule": ("rank sectors by momentum ACCELERATION -> top-10 -> within each, from the US+CA "
                            "GICS pool, candidates pass profit-guard + low-debt + $5M dvol (+ $50M floor for "
                            "pharma sectors); size-tilt to <$2B, then CAP-AWARE value pick (small-cap: top-5 "
                            "cheapest P/B -> cheapest P/E among profitable; large-cap: cheapest P/B); if a "
                            "sector has NO qualifying US/CA value stock (pure commodity/bond/foreign) SKIP the "
                            "slot and renormalize the remaining picks (proxy-holding backtests -468pp)"),
                   "backtest": ("usca_small flagship: +790.4% total / Sharpe 1.49 / DD -20.5% / t3.34 "
                                "(survivorship-de-biased, USD incl. FX, point-in-time; survivorship_smallcap_study). "
                                "NOTE: live GICS universe ≈ but not identical to the backtest's, so live basket "
                                "differs; the RULES match")},
        "picks": picks,
        "n_sectors_skipped": len(skipped),
        "skipped_sectors": skipped,
        "note": ("Value picks are the alpha; sectors with no qualifying US/CA value stock (pure commodity/"
                 "bond/foreign sleeves) are SKIPPED, not held via ETF — backtest: proxy-holding subtracts "
                 "-468pp (322% vs 790%). Weights renormalize over the remaining picks. Monthly-rebalance; "
                 "directional, no fees."),
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
    print(f"\n=== ROTATION PICKS (cheapest-P/B per sector, div_2x conviction weight) — "
          f"{payload['n_accumulating']} accumulating ===", flush=True)
    for p in payload["picks"]:
        pb = f"{p['pb_ratio']:>5}" if p.get("pb_ratio") is not None else "  ETF"
        tag = "  [ETF held as position]" if p.get("is_etf_proxy") else ""
        acc = "  🔵 ACCUMULATING (2x)" if p.get("accumulating") else ""
        print(f"  #{p['rank']:>2} {p['sector']:22} mom6 {p['momentum_6m']:>+6.1f}%  ->  "
              f"{p['pick']:8} P/B {pb}  ${p['last_close']}  alloc {p.get('pct_alloc')}%{tag}{acc}", flush=True)


if __name__ == "__main__":
    main()
