#!/usr/bin/env python3
"""EARNINGS-GATED HOLD — the splitter study showed the 1mo rotation clips winners (AVGO +6% booked ->
+146% if held) but correctly cuts value-traps (EDPR). The user's hypothesis: a NEWS signal could tell
them apart. Raw headlines aren't backfilled at the decision dates, but EARNINGS ARE (EarningsEvent, full
history, eps_surprise + grounded beat/guidance label). So test the honest version:

Across EVERY pick the deployed flagship makes (accel top-10 -> cheapest as-traded-P/B guard low-debt $5M
div_2x, monthly), tag each pick by its EARNINGS SIGNAL as of the pick date (most recent report within
100d): POS (grounded_score>0 i.e. beat & guided up), NEG (<0), or NONE. Then compare the forward-return
curve (1/3/6/12mo) for each group.

If POS picks keep climbing at 3-6-12mo while NEG/NONE decay, a "hold the beats, rotate the rest" gate has
signal -> we could ride winners without the blanket hold-3mo that tanks the book (+53%/Sh0.53). We also
score a simple event-level gate policy vs always-1mo (same opportunity-cost caveat as splitter_hold).
-> BacktestResult[earnings_hold] + JSON.
Run: MSYS_NO_PATHCONV=1 docker exec rotation-backend-1 python -u /app/earnings_hold_study.py
"""
import os, json, warnings
from datetime import timedelta
warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
os.environ.setdefault("DJANGO_SETTINGS_MODULE", "rotation.settings")
import django
django.setup()

from pathlib import Path
import config, sector_holdings, price_basis
from seq_fundamental_study import load_candles, load_financial_reports
from trend_stock_studies import _pit_monthly_panel, _available_at, _ret_delist, CRYPTO
from backtest_lowpb import _monthly_close, BENCH

TOP_N = 10
MIN_DVOL = 5e6
HORIZONS = [1, 2, 3, 6, 12]
EARN_LOOKBACK_D = 100          # a fresh print within ~1 quarter before the pick
OUT = Path(__file__).resolve().parent / ".data" / "studies" / "earnings_hold.json"


def _fwd(px_col, midx, i, h):
    j = min(i + h, len(midx) - 1)
    if j <= i:
        return None
    r = _ret_delist(px_col, midx[i], midx[j])
    return float(r) if r is not None and np.isfinite(r) else None


def load_earnings(tickers):
    """ticker -> sorted list of (report_date(ts), grounded_score, eps_surprise_pct)."""
    from core.models import EarningsEvent
    out = {}
    qs = EarningsEvent.objects.filter(ticker__in=list(tickers)).values(
        "ticker", "report_date", "grounded_score", "eps_surprise_pct")
    for r in qs:
        if not r["report_date"]:
            continue
        out.setdefault(r["ticker"], []).append(
            (pd.Timestamp(r["report_date"]), r["grounded_score"], r["eps_surprise_pct"]))
    for t in out:
        out[t].sort(key=lambda x: x[0])
    return out


def _classify(gs, eps):
    if gs is not None:
        return ("POS" if gs > 0 else "NEG" if gs < 0 else "NONE"), float(gs)
    if eps is not None:
        return ("POS" if eps > 1.0 else "NEG" if eps < -1.0 else "NONE"), float(eps)
    return "NONE", None


def earn_signal(events, date):
    """POS / NEG / NONE using the most recent report within EARN_LOOKBACK_D before `date` (at-pick)."""
    if not events:
        return "NONE", None
    lo = date - timedelta(days=EARN_LOOKBACK_D)
    recent = [e for e in events if lo <= e[0] <= date]
    if not recent:
        return "NONE", None
    rd, gs, eps = recent[-1]
    return _classify(gs, eps)


def earn_in_hold(events, date, hold_days=35):
    """'News during the hold window' (buy -> rotate): earnings landing in (date, date+hold_days]."""
    if not events:
        return "NONE", None
    hi = date + timedelta(days=hold_days)
    inw = [e for e in events if date < e[0] <= hi]
    if not inw:
        return "NONE", None
    rd, gs, eps = inw[-1]
    return _classify(gs, eps)


def _curve(events_list, horizons):
    """avg forward return by hold across a list of pick-event dicts."""
    out = {}
    for h in horizons + ["end"]:
        vals = [e["fwd"].get(h) for e in events_list if e["fwd"].get(h) is not None]
        out[str(h)] = round(float(np.mean(vals)) * 100, 2) if vals else None
    out["n"] = len(events_list)
    return out


def build():
    etfs = {n: e for n, e in config.SECTOR_ETFS.items() if e not in CRYPTO}
    sector_map, all_holds = {}, set()
    for n, e in etfs.items():
        h = [t for t in sector_holdings.get_holdings(n) if t not in (e, BENCH) and t not in CRYPTO]
        sector_map[e] = (n, h); all_holds.update(h)
    all_holds = sorted(all_holds)
    etf_tk = list(etfs.values())
    print(f"Loading {len(etf_tk)} ETFs + {len(all_holds)} stocks + {BENCH}...", flush=True)
    etf_daily = load_candles(etf_tk + [BENCH])
    etf_m = _monthly_close({t: d for t, d in etf_daily.items() if t in etf_tk})
    midx = etf_m.index
    accel = etf_m.pct_change(3) - etf_m.pct_change(3).shift(3)
    stock_daily = load_candles(all_holds)
    stock_m = _monthly_close(stock_daily).reindex(midx)
    reps = load_financial_reports(all_holds)
    sh, eq, ni, dt = (_pit_monthly_panel(reps, f, midx) for f in
                      ("shares_outstanding", "total_equity", "net_income", "total_debt"))
    common = stock_m.columns.intersection(sh.columns).intersection(eq.columns)
    R = lambda p: p.reindex(index=midx, columns=common)
    px = stock_m[common]; sh, eq, ni, dt = R(sh), R(eq), R(ni), R(dt)
    pb = (price_basis.as_traded_close(px) * sh) / eq.where(eq != 0)
    trap = (ni < 0) & (~(eq >= eq.shift(12))) & (~(ni > ni.shift(4)))
    low = (dt / eq.where(eq != 0)) < 1.0
    dvol = {}
    for t in common:
        d = stock_daily.get(t)
        if d is None or "Volume" not in d or len(d) < 90:
            continue
        dvol[t] = (d["Close"] * d["Volume"]).rolling(20).mean().resample("ME").last().reindex(midx)
    dvol = pd.DataFrame(dvol).reindex(index=midx, columns=common)
    earn = load_earnings(common)
    n_have_earn = sum(1 for t in common if earn.get(t))
    print(f"months {len(midx)} | stocks {len(common)} | with earnings history {n_have_earn}", flush=True)

    # walk the flagship, record EVERY pick with its forward returns + earnings signal at pick
    all_picks = []
    for i in range(9, len(midx) - 1):
        date = midx[i]
        top = accel.loc[date].dropna().sort_values(ascending=False).head(TOP_N).index
        for etf in top:
            name, holds = sector_map.get(etf, (etf, []))
            c = [h for h in holds if h in px.columns and _available_at(px[h], date)
                 and pd.notna(pb.loc[date, h]) and pb.loc[date, h] > 0 and not bool(trap.loc[date, h])
                 and pd.notna(dvol.loc[date, h]) and dvol.loc[date, h] >= MIN_DVOL]
            g = [x for x in c if bool(low.loc[date, x])] or c
            if not g:
                continue
            pick = min(g, key=lambda h: pb.loc[date, h])
            fwd = {h: _fwd(px[pick], midx, i, h) for h in HORIZONS}
            fwd["end"] = _fwd(px[pick], midx, i, len(midx) - 1 - i)
            if fwd.get(1) is None:
                continue
            sig, val = earn_signal(earn.get(pick), date)
            hsig, hval = earn_in_hold(earn.get(pick), date)
            all_picks.append({"ticker": pick, "date": str(date.date()), "i": i, "fwd": fwd,
                              "earn_sig": sig, "earn_val": val, "hold_sig": hsig, "hold_val": hval})

    groups = {g: [p for p in all_picks if p["earn_sig"] == g] for g in ("POS", "NEG", "NONE")}
    curves = {g: _curve(v, HORIZONS) for g, v in groups.items()}
    # user's refinement: NEWS DURING THE HOLD WINDOW (earnings that land between buy and rotate)
    hgroups = {g: [p for p in all_picks if p["hold_sig"] == g] for g in ("POS", "NEG", "NONE")}
    hcurves = {g: _curve(v, HORIZONS) for g, v in hgroups.items()}
    curve_all = _curve(all_picks, HORIZONS)

    # event-level policy scoring (avg per-event return; opportunity cost NOT modeled -- directional):
    #   base    : always hold 1mo (the deployed rotation)
    #   gate_H  : if earnings POS at pick, hold H months; else 1mo
    def policy_gate(H, key):
        rr = []
        for p in all_picks:
            r = p["fwd"].get(H) if p[key] == "POS" and p["fwd"].get(H) is not None else p["fwd"].get(1)
            if r is not None:
                rr.append(r)
        return round(float(np.mean(rr)) * 100, 2)
    base_avg = round(float(np.mean([p["fwd"][1] for p in all_picks])) * 100, 2)
    gate = {f"gate_hold{H}mo_if_beat_atpick": policy_gate(H, "earn_sig") for H in (3, 6, 12)}
    gate_hold = {f"gate_hold{H}mo_if_beat_inhold": policy_gate(H, "hold_sig") for H in (3, 6, 12)}

    # how often does the signal even fire, and is POS-6mo separation real?
    pos6 = curves["POS"]["6"]; non6 = curves["NONE"]["6"]; neg6 = curves["NEG"]["6"]
    verdict = (
        f"Across {len(all_picks)} flagship picks: POS-earnings picks (n={curves['POS']['n']}) fwd 6mo "
        f"{pos6}% vs NONE {non6}% vs NEG {neg6}%. "
        + ("Beat-picks DO keep climbing while the rest lag -> an earnings gate on the HOLD has signal. "
           if (pos6 is not None and non6 is not None and pos6 > non6 + 3) else
           "Beat vs non-beat 6mo returns are NOT separated -> earnings gate does NOT cleanly identify who to ride. ")
        + f"Event-level: always-1mo avg {base_avg}%/pick; gate at-pick(hold6mo if beat) "
          f"{gate['gate_hold6mo_if_beat_atpick']}%; gate in-hold(hold6mo if beat lands) "
          f"{gate_hold['gate_hold6mo_if_beat_inhold']}%."
    )
    def _tbl(cv, title):
        print(f"\n=== {title} ===", flush=True)
        print(f"  {'group':<6}{'n':>5}   1mo    2mo    3mo    6mo    12mo   end", flush=True)
        for g in ("POS", "NEG", "NONE"):
            c = cv[g]
            print(f"  {g:<6}{c['n']:>5}  " + "  ".join(f"{c[str(h)]}" if c[str(h)] is not None else "  NA " for h in [1,2,3,6,12]) + f"   {c['end']}", flush=True)
    _tbl(curves, "forward return by hold, grouped by earnings signal AT PICK")
    _tbl(hcurves, "forward return by hold, grouped by earnings landing DURING THE HOLD (buy->rotate)")
    c = curve_all
    print(f"\n  {'ALL':<6}{c['n']:>5}  " + "  ".join(f"{c[str(h)]}" for h in [1,2,3,6,12]) + f"   {c['end']}", flush=True)
    print(f"\n  event-level avg return/pick:", flush=True)
    print(f"    always-1mo (deployed) {base_avg}%", flush=True)
    print(f"    gate at-pick : " + " | ".join(f"{k.split('_')[2]} {v}%" for k, v in gate.items()), flush=True)
    print(f"    gate in-hold : " + " | ".join(f"{k.split('_')[2]} {v}%" for k, v in gate_hold.items()), flush=True)
    print("\n" + verdict, flush=True)

    return {
        "computed_at": pd.Timestamp.utcnow().isoformat(),
        "params": {"top_n": TOP_N, "min_dvol": MIN_DVOL, "earn_lookback_days": EARN_LOOKBACK_D,
                   "horizons_months": HORIZONS, "benchmark": BENCH, "months": int(len(midx)),
                   "signal": "grounded_score>0 (beat & guided up) within 100d of pick, else eps_surprise"},
        "n_picks": len(all_picks), "curves_by_earn_signal_at_pick": curves,
        "curves_by_earn_signal_in_hold": hcurves, "curve_all": curve_all,
        "event_level_avg_return": {"always_1mo": base_avg, **gate, **gate_hold},
        "verdict": verdict,
        "caveat": "Forward returns = single-name buy-and-hold from pick date (delist-aware); event-level policy "
                  "averages per-pick returns and does NOT model opportunity cost of the capital the rotation would "
                  "redeploy -- directional, same caveat as splitter_hold. grounded_score is PIT (grounded at report). "
                  "as-traded P/B, PIT, no fees, present-day-holdings survivorship. NewsItem headlines NOT used "
                  "(not backfilled at pre-2025 decision dates -- separate history gap to fix).",
    }


def main():
    p = build()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(p, indent=2, default=str))
    try:
        from core.models import BacktestResult
        from django.utils import timezone
        BacktestResult.objects.update_or_create(
            kind="earnings_hold", defaults={"payload": json.loads(json.dumps(p, default=str)),
                                            "computed_at": timezone.now()})
        print("Saved BacktestResult[earnings_hold]", flush=True)
    except Exception as e:
        print("DB save failed:", e, flush=True)


if __name__ == "__main__":
    main()
