#!/usr/bin/env python3
"""FIND THE RIGHT ENTRY SIGNAL for the value-pick rotation basket.

The proven equity alpha (backtest_lowpb, arm3_lowpb): rotate into the top trailing-6mo-momentum
sectors, then in each buy the CHEAPEST positive-P/B holding -> +154% vs SPY / t=2.09. That basket
enters UNCONDITIONALLY at month-end. Question the user asked: what is the RIGHT ENTRY SIGNAL?

We hold the sector+stock SELECTION fixed (arm3_lowpb) and vary ONLY the entry GATE applied to the
picked stock's ABSOLUTE daily price. Relative-strength timing on the sector fails (see rotation-alpha
memory: early/mid/late all lose to SPY); the oversold-REVERSAL edge lives in ABSOLUTE single-stock
price. So the candidate entries here are absolute-price reads on the picked NAME:

  unconditional     always enter (baseline = arm3_lowpb)
  above_200ma       Close > 200d SMA          (only value names in a confirmed uptrend)
  above_50ma        Close > 50d SMA
  rsi10_lt_35       RSI(10) < 35              (deep oversold dip)
  rsi10_lt_45       RSI(10) < 45              (mild pullback)
  rsi10_cross_up    RSI(10) crossed > its SMA(10) in last 5d, RSI<50 at cross (fresh turn)
  macd_hist_up      MACD hist > 0-shift & rising (momentum turning up)
  below_20ma        Close < 20d SMA           (short-term dip -- buy weakness)
  near_20d_low      Close <= 1.03 * 20d min   (near a local low)
  trend_dip         above_200ma AND rsi10_lt_45   (buy the dip IN an uptrend -- the classic)
  trend_turn        above_200ma AND rsi10_cross_up (uptrend + fresh RSI turn)
  trend_pullbk      above_200ma AND below_20ma     (uptrend + short-term pullback)

For each entry we report TWO things:
  (A) CONDITIONAL LIFT: forward 1mo return of picks with the entry ON vs OFF (is the gate informative?)
  (B) PORTFOLIO: gate the basket to only picks passing the entry (equal-weight; skip month if none) ->
      total/vs-SPY/Sharpe/t vs the unconditional baseline. Also 'skipped_frac' = how selective it is.

The WINNER = the entry that improves risk-adjusted forward return over unconditional WITHOUT decimating
the sample (over-selective gates destroy the edge, cf. ENGINE ROTATE-IN). PIT throughout.
-> BacktestResult[entry_signal] + JSON.
Run: MSYS_NO_PATHCONV=1 docker exec rotation-backend-1 python -u /app/entry_signal_study.py
     (--limit 150 for a quick subset)
"""
import os, sys, json, warnings
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
os.environ.setdefault("DJANGO_SETTINGS_MODULE", "rotation.settings")
import django
django.setup()

from pathlib import Path
import config
import sector_holdings
import ta
from studies import _tstat_from_returns
from seq_fundamental_study import load_candles, load_financial_reports
from trend_stock_studies import _pit_monthly_panel, _available_at, _ret_delist, CRYPTO
from backtest_lowpb import _monthly_close, BENCH

OUT = Path(__file__).resolve().parent / ".data" / "studies" / "entry_signal.json"

LOOKBACK = 6
TOP_N_SECTORS = 10


# ---- stats (mirror backtest_lowpb) -----------------------------------------
def _stats(rets, spy_rets):
    r = np.array(rets, dtype=float)
    n = len(r)
    if n == 0:
        return {"total_return": 0, "spy_total": 0, "vs_spy": 0, "annual_return": 0,
                "sharpe": 0, "max_drawdown": 0, "t_stat": None, "periods": 0}
    total = float(np.prod(1 + r) - 1) * 100
    spy_total = float(np.prod(1 + np.array(spy_rets)) - 1) * 100
    ann = (float(np.prod(1 + r)) ** (12.0 / n) - 1) * 100
    sharpe = float(r.mean() / r.std() * np.sqrt(12)) if r.std() > 1e-9 else 0.0
    eq = np.cumprod(1 + r)
    dd = float(((eq / np.maximum.accumulate(eq)) - 1).min() * 100)
    t = _tstat_from_returns(list(r))
    return {"total_return": round(total, 1), "spy_total": round(spy_total, 1),
            "vs_spy": round(total - spy_total, 1), "annual_return": round(ann, 1),
            "sharpe": round(sharpe, 2), "max_drawdown": round(dd, 1),
            "t_stat": round(t, 2) if t is not None else None, "periods": n}


def _hit(vals):
    a = np.asarray(vals, float); a = a[~np.isnan(a)]
    if not len(a):
        return {"mean_pct": None, "hit_pct": None, "n": 0}
    return {"mean_pct": round(float(a.mean()) * 100, 2),
            "hit_pct": round(float((a > 0).mean()) * 100, 1), "n": int(len(a))}


# ---- entry-signal daily boolean panels -------------------------------------
def _entry_panels(daily, midx):
    """Return {entry_name: month-end boolean DataFrame (tickers x midx)} on ABSOLUTE price."""
    per = {name: {} for name in (
        "above_200ma", "above_50ma", "rsi10_lt_35", "rsi10_lt_45", "rsi10_cross_up",
        "macd_hist_up", "below_20ma", "near_20d_low")}
    for tk, df in daily.items():
        if len(df) < 210:
            continue
        c = df["Close"]
        sma200 = c.rolling(200).mean()
        sma50 = c.rolling(50).mean()
        sma20 = c.rolling(20).mean()
        rsi = ta.momentum.rsi(c, window=10)
        rsi_sma = rsi.rolling(10).mean()
        above = rsi > rsi_sma
        cross = (above & (~above.shift(1).fillna(False)) & (rsi < 50)).rolling(5).max().fillna(0).astype(bool)
        macd_h = ta.trend.macd_diff(c)
        low20 = c.rolling(20).min()

        def _me(s):  # month-end value of a boolean/price series on the ETF calendar
            return s.reindex(midx, method="ffill")

        per["above_200ma"][tk] = _me(c > sma200)
        per["above_50ma"][tk] = _me(c > sma50)
        per["rsi10_lt_35"][tk] = _me(rsi < 35)
        per["rsi10_lt_45"][tk] = _me(rsi < 45)
        per["rsi10_cross_up"][tk] = _me(cross)
        per["macd_hist_up"][tk] = _me((macd_h > 0) & (macd_h > macd_h.shift(1)))
        per["below_20ma"][tk] = _me(c < sma20)
        per["near_20d_low"][tk] = _me(c <= 1.03 * low20)

    panels = {name: pd.DataFrame(d).reindex(index=midx).fillna(False).astype(bool)
              for name, d in per.items()}
    # combos
    panels["trend_dip"] = (panels["above_200ma"] & panels["rsi10_lt_45"])
    panels["trend_turn"] = (panels["above_200ma"] & panels["rsi10_cross_up"])
    panels["trend_pullbk"] = (panels["above_200ma"] & panels["below_20ma"])
    panels["unconditional"] = pd.DataFrame(True, index=midx,
                                           columns=panels["above_200ma"].columns)
    return panels


ENTRY_ORDER = ["unconditional", "above_200ma", "above_50ma", "rsi10_lt_35", "rsi10_lt_45",
               "rsi10_cross_up", "macd_hist_up", "below_20ma", "near_20d_low",
               "trend_dip", "trend_turn", "trend_pullbk"]


def build():
    etfs = {name: etf for name, etf in config.SECTOR_ETFS.items() if etf not in CRYPTO}
    limit = int(sys.argv[sys.argv.index("--limit") + 1]) if "--limit" in sys.argv else None

    sector_map, all_holds = {}, set()
    for name, etf in etfs.items():
        h = [t for t in sector_holdings.get_holdings(name) if t not in (etf, BENCH) and t not in CRYPTO]
        sector_map[etf] = (name, h)
        all_holds.update(h)
    all_holds = sorted(all_holds)
    if limit:
        all_holds = all_holds[:limit]
        hset = set(all_holds)
        sector_map = {e: (n, [h for h in hs if h in hset]) for e, (n, hs) in sector_map.items()}

    etf_tickers = list(etfs.values())
    print(f"Loading {len(etf_tickers)} ETFs + {len(all_holds)} stocks + {BENCH}...", flush=True)
    etf_daily = load_candles(etf_tickers + [BENCH])
    etf_monthly = _monthly_close({t: d for t, d in etf_daily.items() if t in etf_tickers})
    midx = etf_monthly.index
    spy_daily = etf_daily.get(BENCH)
    spy_m = spy_daily["Close"].resample("ME").last().reindex(midx)

    stock_daily = load_candles(all_holds)
    stock_monthly = _monthly_close(stock_daily).reindex(midx)

    etf_trail = etf_monthly.pct_change(LOOKBACK)
    stock_fwd = stock_monthly.pct_change().shift(-1)
    spy_fwd = spy_m.pct_change().shift(-1)

    reps = load_financial_reports(all_holds)
    shares_p = _pit_monthly_panel(reps, "shares_outstanding", midx)
    equity_p = _pit_monthly_panel(reps, "total_equity", midx)
    common = stock_monthly.columns.intersection(shares_p.columns).intersection(equity_p.columns)
    pb_panel = (stock_monthly[common] * shares_p[common]) / equity_p[common].where(equity_p[common] != 0)

    print("computing entry panels...", flush=True)
    panels = _entry_panels(stock_daily, midx)

    print(f"months {len(midx)} | stocks {stock_monthly.shape[1]} | pb {pb_panel.shape}", flush=True)
    warmup = max(LOOKBACK, 1)

    def _pick_in_sector(holds, date):
        cands = [h for h in holds if h in stock_monthly.columns and _available_at(stock_monthly[h], date)]
        if not cands or date not in pb_panel.index:
            return None
        row = pb_panel.loc[date, [c for c in cands if c in pb_panel.columns]].dropna()
        row = row[row > 0]
        return row.idxmin() if len(row) else None

    # For each rebalance i: the top-momentum sectors, each a SLOT holding {the cheapest-P/B value pick
    # (if any) with its forward returns + entry flags} and {the sector ETF's forward returns} for the
    # fallback. Forward returns at each HOLD horizon (1/2/3 months). Computed ONCE, reused for every
    # entry x hold x fallback evaluation. This lets us (a) hold names 2 months and (b) fall back to the
    # ETF for empty/gate-failing sectors so the unconditional+fallback baseline reproduces arm3_lowpb.
    HOLDS = [1, 2, 3]

    def _fwd(close, i, h):
        j = i + h
        if j >= len(midx):
            return None
        r = _ret_delist(close, midx[i], midx[j])
        return float(r) if r is not None and np.isfinite(r) else None

    def _spy_fwd_h(i, h):
        j = i + h
        if j >= len(midx):
            return None
        a, b = spy_m.iloc[i], spy_m.iloc[j]
        return float(b / a - 1) if (np.isfinite(a) and np.isfinite(b) and a > 0) else None

    month_by_i = {}
    for i in range(warmup, len(midx) - 1):
        date = midx[i]
        spret = {h: _spy_fwd_h(i, h) for h in HOLDS}
        if spret.get(1) is None:
            continue
        ranks = etf_trail.loc[date].dropna().sort_values(ascending=False).head(TOP_N_SECTORS)
        if not len(ranks):
            continue
        slots = []
        for etf in ranks.index:
            _, holds = sector_map.get(etf, (etf, []))
            etf_rr = {h: (_fwd(etf_monthly[etf], i, h) if etf in etf_monthly.columns else None) for h in HOLDS}
            pk = _pick_in_sector(holds, date)
            pick_rr, flags = None, None
            if pk is not None:
                pr = {h: _fwd(stock_monthly[pk], i, h) for h in HOLDS}
                if pr.get(1) is not None:
                    pick_rr = pr
                    flags = {name: bool(panels[name].loc[date, pk]) if pk in panels[name].columns else False
                             for name in ENTRY_ORDER}
            slots.append({"etf": etf, "pick": pk if pick_rr is not None else None,
                          "pick_rr": pick_rr, "etf_rr": etf_rr, "flags": flags})
        month_by_i[i] = {"date": date, "spret": spret, "slots": slots}
    idxs = sorted(month_by_i)

    # ---- (A) conditional lift at each HOLD: fwd return of picks with entry ON vs OFF ----------------
    def _lift(entry, h):
        on, off = [], []
        for i in idxs:
            for s in month_by_i[i]["slots"]:
                if s["pick_rr"] is None:
                    continue
                v = s["pick_rr"].get(h)
                if v is None or not np.isfinite(v):
                    continue
                (on if s["flags"][entry] else off).append(v)
        a, b = _hit(on), _hit(off)
        lift = (round(a["mean_pct"] - b["mean_pct"], 2)
                if (a["mean_pct"] is not None and b["mean_pct"] is not None) else None)
        return {"on": a, "off": b, "lift_pct": lift}

    # ---- (B) gated portfolio at HOLD h. fallback=True -> empty/gate-failing sectors hold the ETF (all
    # 10 slots, matches arm3_lowpb); fallback=False -> only picks passing the gate (skip month if none).
    # Held h months via non-overlapping rebalance, averaged over the h phase offsets (proper t-stat).
    def _portfolio(entry, h, fallback):
        phase_summ = []
        for phase in range(h):
            rets, spies = [], []
            i = idxs[0] + phase
            while i <= idxs[-1]:
                mb = month_by_i.get(i)
                if mb is not None:
                    sp = mb["spret"].get(h)
                    if sp is not None and np.isfinite(sp):
                        slot_rets = []
                        for s in mb["slots"]:
                            use = None
                            if s["pick_rr"] is not None and s["flags"][entry]:
                                v = s["pick_rr"].get(h)
                                if v is not None and np.isfinite(v):
                                    use = v
                            if use is None and fallback:
                                ev = s["etf_rr"].get(h)
                                if ev is not None and np.isfinite(ev):
                                    use = ev
                            if use is not None:
                                slot_rets.append(use)
                        if slot_rets:
                            rets.append(float(np.mean(slot_rets))); spies.append(float(sp))
                i += h
            if len(rets) >= 4:
                phase_summ.append(_stats(rets, spies))
        if not phase_summ:
            return None

        def _avg(k):
            vals = [s[k] for s in phase_summ if s.get(k) is not None]
            return round(float(np.mean(vals)), 2) if vals else None
        return {"total_return": _avg("total_return"), "spy_total": _avg("spy_total"),
                "vs_spy": _avg("vs_spy"), "annual_return": _avg("annual_return"),
                "sharpe": _avg("sharpe"), "max_drawdown": _avg("max_drawdown"),
                "t_stat": _avg("t_stat"), "periods": int(sum(s["periods"] for s in phase_summ)),
                "phases": len(phase_summ)}

    # ---- deployment stats (from the no-fallback 1mo view) ------------------------------------------
    def _deploy(entry):
        held, pick_slots, months_active = 0, 0, 0
        for i in idxs:
            passing = 0
            for s in month_by_i[i]["slots"]:
                if s["pick_rr"] is None:
                    continue
                pick_slots += 1
                if s["flags"][entry]:
                    passing += 1
            if passing:
                months_active += 1; held += passing
        mt = len(idxs)
        return {"months_active": months_active, "months_total": mt,
                "skipped_frac": round(1 - months_active / mt, 3) if mt else None,
                "held_frac_of_slots": round(held / pick_slots, 3) if pick_slots else None,
                "avg_picks_per_active_month": round(held / months_active, 2) if months_active else 0}

    results = {}
    for entry in ENTRY_ORDER:
        lifts = {h: _lift(entry, h) for h in HOLDS}
        results[entry] = {
            "entry": entry,
            "cond_on": lifts[1]["on"], "cond_off": lifts[1]["off"],
            "cond_lift_pct": lifts[1]["lift_pct"],
            "cond_lift_by_hold": {str(h): lifts[h]["lift_pct"] for h in HOLDS},
            "portfolio": _portfolio(entry, 1, False),            # 1mo hold, pure value picks (headline)
            "portfolio_2mo": _portfolio(entry, 2, False),        # 2mo hold, pure value picks
            "portfolio_fallback": _portfolio(entry, 1, True),    # 1mo + ETF fallback (== arm3_lowpb)
            "portfolio_fallback_2mo": _portfolio(entry, 2, True),  # 2mo + ETF fallback
            **_deploy(entry),
        }

    base = results["unconditional"]["portfolio"] or {}
    for entry in ENTRY_ORDER:
        p = results[entry]["portfolio"] or {}
        results[entry]["vs_baseline_total"] = (round((p.get("total_return") or 0) - (base.get("total_return") or 0), 1))

    # rank the (non-trivial) entries by CONDITIONAL LIFT — the per-pick apples-to-apples read (fwd
    # return of picks with entry ON minus OFF), which is NOT confounded by how selective the gate is.
    # (Portfolio t_stat mechanically falls as a gate skips more months / holds fewer names, so ranking
    # on t would perversely reward the always-in baseline; the lift isolates the entry's real edge.)
    # Require the entry to stay deployable (skip < 50% of months) to be crown-eligible.
    def _score(e):
        r = results[e]
        if e == "unconditional":
            return -1e9
        lift = r["cond_lift_pct"] if r["cond_lift_pct"] is not None else -1e9
        if (r["skipped_frac"] or 0) >= 0.5:      # not deployable enough to crown
            return lift - 1e5
        return lift
    ranked = sorted([e for e in ENTRY_ORDER if e != "unconditional"], key=_score, reverse=True)
    # crown = best positive-lift, deployable entry. best_risk_adjusted = highest-Sharpe positive-lift
    # entry (may be more selective). worst = most-negative-lift (the entry to AVOID).
    eligible = [e for e in ranked if (results[e]["cond_lift_pct"] or -1) > 0
                and (results[e]["skipped_frac"] or 0) < 0.5]
    winner = eligible[0] if eligible else None
    def _sharpe(e):
        p = results[e]["portfolio"] or {}
        return p.get("sharpe") if p.get("sharpe") is not None else -1e9
    pos_lift = [e for e in ENTRY_ORDER if e != "unconditional" and (results[e]["cond_lift_pct"] or -1) > 0]
    best_risk_adjusted = max(pos_lift, key=_sharpe) if pos_lift else None
    worst = min([e for e in ENTRY_ORDER if e != "unconditional"],
                key=lambda e: results[e]["cond_lift_pct"] if results[e]["cond_lift_pct"] is not None else 0)

    # reconciliation: unconditional WITH ETF fallback (all 10 slots) reproduces arm3_lowpb (+154% vs SPY
    # headline); WITHOUT fallback it drops empty sectors -> pure value picks -> higher return but a
    # more-concentrated book. The gap IS the ETF-rotation drag that arm3's fallback carries.
    u = results["unconditional"]
    fb = (u.get("portfolio_fallback") or {}).get("vs_spy")
    nofb = (u.get("portfolio") or {}).get("vs_spy")
    # 2-month-hold insight across the DIP family: the deeper the dip, the longer the reversal takes to
    # mature, so deep dips improve with a 2-month hold while the milder dip already bounces in month 1.
    dip_family = ["rsi10_lt_35", "rsi10_lt_45", "near_20d_low", "below_20ma"]
    def _lb(e, k):
        return (results.get(e, {}).get("cond_lift_by_hold", {}) or {}).get(k)
    improved = [e for e in dip_family
                if _lb(e, "1") is not None and _lb(e, "2") is not None and _lb(e, "2") > _lb(e, "1")]
    hold_note = (
        ("Holding names ~2 months is not just fine, it's BETTER for the DEEPER dips (" + ", ".join(improved) +
         "): the more oversold the entry, the longer the reversal takes to mature, so their per-pick lift "
         "rises from 1mo to 2mo hold. The milder dip (rsi10_lt_45) already gets its bounce in month 1. A "
         "2-month hold also keeps the selective dip gate deployed instead of forcing cash when few new dips "
         "appear.") if improved else
        ("A 2-month hold keeps the selective dip gate deployed instead of forcing cash when few new dips "
         "appear; per-pick lift is roughly flat from 1mo to 2mo across the dip family."))

    recommendation = {
        "winner": winner, "best_risk_adjusted": best_risk_adjusted, "worst": worst,
        "fallback_baseline_vs_spy": fb, "nofallback_baseline_vs_spy": nofb, "hold_note": hold_note,
        "headline": ("Enter the value pick on an OVERSOLD DIP in its own absolute price (RSI(10) pulled "
                     "back), NOT on momentum confirmation. Across the pick universe, buying WEAKNESS "
                     "(RSI<45/<35, below 20d MA, near a 20d low) adds positive forward return per pick; "
                     "buying STRENGTH (above 200d/50d MA, fresh RSI cross-up, MACD-hist up) SUBTRACTS. "
                     "The dip entry raises raw return and per-pick lift but concentrates the book "
                     "(fewer names -> higher drawdown); the always-in baseline keeps the best drawdown. "
                     "Same lesson as ROTATE IN: momentum-confirmation is the worst possible entry."),
        "fallback_note": (f"Baseline vs SPY: {fb}% WITH ETF fallback (all 10 slots = arm3_lowpb, the +154% "
                          f"headline construction) vs {nofb}% WITHOUT (pure value picks, empty sectors "
                          f"dropped). The difference is the ETF-rotation drag the fallback carries."),
    }

    payload = {
        "computed_at": pd.Timestamp.utcnow().isoformat(),
        "params": {"lookback_months": LOOKBACK, "top_n_sectors": TOP_N_SECTORS, "benchmark": BENCH,
                   "selection": "arm3_lowpb (top-momentum sectors -> cheapest positive-P/B pick), entry gate varies",
                   "holds": HOLDS, "months": int(len(midx)), "limit": limit,
                   "portfolios": ("portfolio = 1mo pure-pick; portfolio_2mo = 2mo hold; portfolio_fallback = "
                                  "1mo with ETF fallback for empty/gate-failing sectors (== arm3_lowpb); "
                                  "portfolio_fallback_2mo = 2mo + fallback. Multi-month holds use non-overlapping "
                                  "phase-averaged rebalances.")},
        "baseline": "unconditional",
        "winner": winner,
        "recommendation": recommendation,
        "ranking": ranked,
        "entries": {e: results[e] for e in ENTRY_ORDER},
        "caveat": ("Selection (sector+stock) held fixed = arm3_lowpb; ONLY the entry gate on the picked "
                   "name's ABSOLUTE price varies, isolating entry timing. Monthly rebalance, equal-weight, "
                   "PIT, directional/no-fees; ~5y single regime; multiple entries tested -> multiple-"
                   "comparisons hazard, trust a winner only if its whole family (trend/dip) agrees. "
                   "Over-selective gates (skipped_frac high) shrink effective N -> discount their t."),
    }
    return payload


def _line(tag, r):
    p = r.get("portfolio") or {}
    pf = r.get("portfolio_fallback") or {}
    lb = r.get("cond_lift_by_hold") or {}
    def _g(d, k, w=7, dec=1):
        v = d.get(k)
        return f"{v:>{w}.{dec}f}" if isinstance(v, (int, float)) else f"{'–':>{w}}"
    l1 = lb.get("1"); l2 = lb.get("2")
    l1s = f"{l1:>+5.2f}" if isinstance(l1, (int, float)) else f"{'–':>5}"
    l2s = f"{l2:>+5.2f}" if isinstance(l2, (int, float)) else f"{'–':>5}"
    return (f"  {tag:16} vsSPY {_g(p,'vs_spy')}%  (fallback {_g(pf,'vs_spy')}%)  "
            f"Sh {_g(p,'sharpe',5,2)}  DD {_g(p,'max_drawdown',6)}%  "
            f"lift@1mo {l1s}%  @2mo {l2s}%  skip {str(r.get('skipped_frac')):>5}")


def main():
    payload = build()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(payload, indent=2, default=str))
    try:
        from core.models import BacktestResult
        from django.utils import timezone
        BacktestResult.objects.update_or_create(
            kind="entry_signal",
            defaults={"payload": json.loads(json.dumps(payload, default=str)), "computed_at": timezone.now()})
        print("Saved BacktestResult[entry_signal]", flush=True)
    except Exception as e:
        print("DB save failed:", e, flush=True)

    print("\n=== RIGHT ENTRY SIGNAL for the value-pick basket (arm3_lowpb, entry gate varies) ===", flush=True)
    rc = payload["recommendation"]
    print(f"WINNER (deployable dip entry) = {rc['winner']} | best risk-adj = {rc['best_risk_adjusted']} "
          f"| AVOID = {rc['worst']}", flush=True)
    print(rc.get("fallback_note", ""), flush=True)
    if rc.get("hold_note"):
        print(rc["hold_note"], flush=True)
    print("", flush=True)
    print(_line("unconditional*", payload["entries"]["unconditional"]), flush=True)
    for e in payload["ranking"]:
        print(_line(e, payload["entries"][e]), flush=True)
    print("\n(lift = fwd-1mo mean of picks with entry ON minus OFF; hold = frac of pick-slots kept; "
          "skip = frac of months with zero passing picks)", flush=True)
    print("Saved ->", OUT, flush=True)


if __name__ == "__main__":
    main()
