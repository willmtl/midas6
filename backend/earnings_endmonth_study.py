#!/usr/bin/env python3
"""EARNINGS TIMING WITHIN THE HOLD — does WHERE in the month the print lands change whether we should ride it?
Hypothesis (user): a beat landing at the END of the month (just before month-end rebalance) gets CLIPPED — we
rotate out before the post-earnings drift plays out (like AVGO). A beat EARLY in the hold lets us capture the
drift within the month, so rotating normally is fine. Symmetric for misses: a miss LATE means we exit right
after (dodge the drift); a miss EARLY means we eat the drift before month-end.

For every flagship pick (accel top-10 -> cheapest as-traded-P/B guard low-debt $5M div_2x, monthly), find the
earnings event inside the hold window (pick date -> next rebalance) and bucket by DAYS-TO-REBALANCE:
  EARLY  print >10 trading-ish days before rebalance (drift captured within the month)
  LATE   print in the last ~10 days before rebalance (drift happens AFTER we've rotated out)
Compare forward 1/3/6mo returns. If LATE beats show low 1mo but high 3/6mo (drift we DIDN'T capture), then a
rule "beat in the last ~10 days -> EXTEND the hold" has signal. Same read for LATE vs EARLY misses.
-> BacktestResult[earnings_endmonth] + JSON.
Run: MSYS_NO_PATHCONV=1 docker exec rotation-backend-1 python -u /app/earnings_endmonth_study.py
"""
import os, json, warnings
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
LATE_DAYS = 10                 # print within this many days of rebalance = "end of month"
OUT = Path(__file__).resolve().parent / ".data" / "studies" / "earnings_endmonth.json"


def _fwd(px_col, midx, i, h):
    j = min(i + h, len(midx) - 1)
    if j <= i:
        return None
    r = _ret_delist(px_col, midx[i], midx[j])
    return float(r) if r is not None and np.isfinite(r) else None


def load_events(tickers):
    from core.models import EarningsEvent
    out = {}
    for r in EarningsEvent.objects.filter(ticker__in=list(tickers)).values(
            "ticker", "report_date", "grounded_score", "eps_surprise_pct"):
        if r["report_date"] is None:
            continue
        gs, eps = r["grounded_score"], r["eps_surprise_pct"]
        if gs is not None:
            d = 1 if gs > 0 else -1 if gs < 0 else 0
        elif eps is not None:
            d = 1 if eps > 1.0 else -1 if eps < -1.0 else 0
        else:
            d = 0
        out.setdefault(r["ticker"], []).append((pd.Timestamp(r["report_date"]), d))
    for t in out:
        out[t].sort()
    return out


def event_in_hold(events, date, ndate):
    """(direction, days_to_rebalance) for the first earnings event in (date, ndate); (None, None) if none."""
    if not events:
        return None, None
    for rd, d in events:
        if date < rd < ndate:
            return d, int((ndate - rd).days)
        if rd >= ndate:
            break
    return None, None


def _curve(rows):
    out = {}
    for h in HORIZONS + ["end"]:
        vals = [e["fwd"].get(h) for e in rows if e["fwd"].get(h) is not None]
        out[str(h)] = round(float(np.mean(vals)) * 100, 2) if vals else None
    out["n"] = len(rows)
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
    ev = load_events(common)
    print(f"months {len(midx)} | stocks {len(common)}", flush=True)

    picks = []
    for i in range(9, len(midx) - 1):
        date, ndate = midx[i], midx[i + 1]
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
            d, dtr = event_in_hold(ev.get(pick), date, ndate)
            picks.append({"ticker": pick, "date": str(date.date()), "fwd": fwd,
                          "dir": d, "days_to_rebal": dtr})

    def grp(direction, when):
        rows = []
        for p in picks:
            if p["dir"] != direction or p["days_to_rebal"] is None:
                continue
            late = p["days_to_rebal"] <= LATE_DAYS
            if (when == "late") == late:
                rows.append(p)
        return rows

    out = {}
    for lbl, dcode in (("beat", 1), ("miss", -1)):
        out[lbl] = {"early": _curve(grp(dcode, "early")), "late": _curve(grp(dcode, "late"))}
    out["no_earnings"] = _curve([p for p in picks if p["dir"] is None])
    out["all"] = _curve(picks)

    print(f"\n=== forward return by hold, split by WHEN in the month the print lands (LATE = last {LATE_DAYS}d) ===", flush=True)
    print(f"  {'group':<14}{'n':>4}    1mo    3mo    6mo    12mo    end", flush=True)
    for lbl in ("beat", "miss"):
        for when in ("early", "late"):
            c = out[lbl][when]
            print(f"  {lbl+' '+when:<14}{c['n']:>4}  " +
                  "  ".join(f"{c[str(h)]!s:>6}" for h in (1, 3, 6, 12)) + f"  {c['end']!s:>6}", flush=True)
    c = out["no_earnings"]
    print(f"  {'no-earnings':<14}{c['n']:>4}  " + "  ".join(f"{c[str(h)]!s:>6}" for h in (1, 3, 6, 12)) + f"  {c['end']!s:>6}", flush=True)

    bl, be = out["beat"]["late"], out["beat"]["early"]
    ml, me = out["miss"]["late"], out["miss"]["early"]
    # "clip" = how much drift a LATE beat leaves on the table = 6mo minus 1mo (captured within month)
    beat_late_clip = (bl["6"] - bl["1"]) if (bl["6"] is not None and bl["1"] is not None) else None
    beat_early_clip = (be["6"] - be["1"]) if (be["6"] is not None and be["1"] is not None) else None
    verdict = (
        f"LATE beat (print in last {LATE_DAYS}d): 1mo {bl['1']}% vs 6mo {bl['6']}% (uncaptured drift ~{beat_late_clip}pp). "
        f"EARLY beat: 1mo {be['1']}% vs 6mo {be['6']}% (~{beat_early_clip}pp). "
        + ("LATE beats leave MORE drift uncaptured than EARLY -> a beat at month-end is worth EXTENDING the hold. "
           if (beat_late_clip is not None and beat_early_clip is not None and beat_late_clip > beat_early_clip + 3)
           else "LATE vs EARLY beats leave similar drift -> timing within the month does NOT change the ride decision. ")
        + f"LATE miss 1mo {ml['1']}% vs EARLY miss 1mo {me['1']}% "
        + ("(late miss dodges the in-month drop -> the month-end cut already helps)."
           if (ml['1'] is not None and me['1'] is not None and ml['1'] > me['1']) else
           "(late miss not clearly better in-month).")
    )
    print("\n" + verdict, flush=True)
    return {
        "computed_at": pd.Timestamp.utcnow().isoformat(),
        "params": {"top_n": TOP_N, "min_dvol": MIN_DVOL, "late_days": LATE_DAYS, "horizons": HORIZONS,
                   "benchmark": BENCH, "months": int(len(midx)), "pb_basis": "as-traded"},
        "n_picks": len(picks), "groups": out,
        "beat_late_uncaptured_drift_pp": round(beat_late_clip, 2) if beat_late_clip is not None else None,
        "beat_early_uncaptured_drift_pp": round(beat_early_clip, 2) if beat_early_clip is not None else None,
        "verdict": verdict,
        "caveat": "Forward returns = single-name buy-and-hold from pick date (delist-aware); does NOT model the "
                  "opportunity cost of the capital the monthly rotation would redeploy. days_to_rebal uses calendar "
                  "days to the next month-end index. grounded_score PIT. as-traded P/B, PIT, no fees, present-day holdings.",
    }


def main():
    p = build()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(p, indent=2, default=str))
    try:
        from core.models import BacktestResult
        from django.utils import timezone
        BacktestResult.objects.update_or_create(
            kind="earnings_endmonth", defaults={"payload": json.loads(json.dumps(p, default=str)),
                                                 "computed_at": timezone.now()})
        print("Saved BacktestResult[earnings_endmonth]", flush=True)
    except Exception as e:
        print("DB save failed:", e, flush=True)


if __name__ == "__main__":
    main()
