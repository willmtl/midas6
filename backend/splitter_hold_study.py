#!/usr/bin/env python3
"""SPLITTER HOLD ANALYSIS — for the 18 future-splitter names the buggy adjusted-close P/B basis
bought, would we have made MORE by holding them LONGER than the 1-month rotation hold, and WHEN did
we actually rotate out of them?

Reproduces the DEPLOYED flagship monthly (accel top-10 -> cheapest positive AS-TRADED-P/B, guard,
low-debt, $5M floor, div_2x, month-end, hold 1mo). Every month a TARGET name is the pick, we record:
  - the booked 1-month return (what the rotation actually earned)
  - forward cumulative returns if we had instead HELD 1/2/3/6/12 months and to-end
  - "runs": consecutive months a name stayed the pick (= how long we actually held before rotating out)

Answers: (a) does extending the hold on these names beat the 1mo rotation? (b) entry/exit dates per run.
Uses as-traded P/B for RANKING (honest), adjusted close for RETURNS (correct total return).
-> BacktestResult[splitter_hold] + JSON.
Run: MSYS_NO_PATHCONV=1 docker exec rotation-backend-1 python -u /app/splitter_hold_study.py
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

TARGETS = ["300757.SZ", "AMZN", "ANET", "AVGO", "CMG", "EDPR.LS", "GOLD", "GOOGL", "IP",
           "LRCX", "LREN3.SA", "MLI", "MTH", "NFLX", "NVDA", "SBSP3.SA", "SRE", "WIT"]
TOP_N = 10
CONVICTION_MULT = 2.0
MIN_DVOL = 5e6
HORIZONS = [1, 2, 3, 6, 12]        # months held (1 = actual rotation hold)
OUT = Path(__file__).resolve().parent / ".data" / "studies" / "splitter_hold.json"


def _fwd(px_col, midx, i, h):
    """Cumulative return holding from month i to i+h (delist-aware). None if unavailable."""
    j = min(i + h, len(midx) - 1)
    if j <= i:
        return None
    r = _ret_delist(px_col, midx[i], midx[j])
    return float(r) if r is not None and np.isfinite(r) else None


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
    # dvol + A/D divergence for div_2x weighting (weight doesn't affect single-name forward returns,
    # but we keep the exact pick logic so picks match the flagship)
    dvol = {}
    adl_m = {}
    for t in common:
        d = stock_daily.get(t)
        if d is None or "Volume" not in d or len(d) < 90:
            continue
        v = d["Volume"]
        dvol[t] = (d["Close"] * v).rolling(20).mean().resample("ME").last().reindex(midx)
        if {"High", "Low", "Close"}.issubset(d.columns):
            rng = (d["High"] - d["Low"]).replace(0, np.nan)
            mfm = ((d["Close"] - d["Low"]) - (d["High"] - d["Close"])) / rng
            adl_m[t] = (mfm.fillna(0) * v).cumsum().resample("ME").last().reindex(midx)
    dvol = pd.DataFrame(dvol).reindex(index=midx, columns=common)
    adl = pd.DataFrame(adl_m).reindex(index=midx, columns=common)
    ad_slope3 = adl - adl.shift(3); px_ret3 = px.pct_change(3)
    present = [t for t in TARGETS if t in common]
    missing = [t for t in TARGETS if t not in common]
    print(f"months {len(midx)} | stocks {len(common)} | targets present {len(present)}/{len(TARGETS)} "
          f"(missing: {missing})", flush=True)

    # walk the flagship, record every month each target is THE pick for its sector
    picks_by_name = {t: [] for t in present}   # list of dicts per pick-month
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
            if pick not in picks_by_name:
                continue
            fwd = {h: _fwd(px[pick], midx, i, h) for h in HORIZONS}
            fwd["end"] = _fwd(px[pick], midx, i, len(midx) - 1 - i)
            picks_by_name[pick].append({
                "i": i, "date": str(date.date()), "sector": name, "etf": etf,
                "pb": round(float(pb.loc[date, pick]), 3),
                "dvol_m": round(float(dvol.loc[date, pick]) / 1e6, 1),
                "fwd": {k: (round(v * 100, 2) if v is not None else None) for k, v in fwd.items()},
            })

    # runs = maximal consecutive pick-months (how long we actually held before rotating out)
    def make_runs(events):
        runs, cur = [], []
        for e in events:
            if cur and e["i"] == cur[-1]["i"] + 1:
                cur.append(e)
            else:
                if cur:
                    runs.append(cur)
                cur = [e]
        if cur:
            runs.append(cur)
        out = []
        for run in runs:
            # actual booked return = compounding each month's 1-mo forward while continuously held
            booked = np.prod([1 + (e["fwd"].get(1) or 0) / 100 for e in run]) - 1
            out.append({
                "entry": run[0]["date"], "exit_after": run[-1]["date"], "months_held": len(run),
                "sector": run[0]["sector"], "entry_pb": run[0]["pb"],
                "booked_return_pct": round(booked * 100, 2),
            })
        return out

    # aggregate: for each name, avg forward return by horizon across its pick-events (independent entries)
    per_name = {}
    all_events = []
    for t, ev in picks_by_name.items():
        if not ev:
            per_name[t] = {"n_picks": 0}
            continue
        all_events += [(t, e) for e in ev]
        agg = {}
        for h in [1] + [x for x in HORIZONS if x != 1] + ["end"]:
            vals = [e["fwd"].get(h) for e in ev if e["fwd"].get(h) is not None]
            agg[str(h)] = round(float(np.mean(vals)), 2) if vals else None
        per_name[t] = {
            "n_picks": len(ev),
            "avg_fwd_by_hold": agg,
            "runs": make_runs(ev),
            "events": ev,
        }

    # pooled across ALL target pick-events: does a longer hold beat the 1mo rotation?
    pooled = {}
    for h in [1] + [x for x in HORIZONS if x != 1] + ["end"]:
        vals = [e["fwd"].get(h) for _, e in all_events if e["fwd"].get(h) is not None]
        pooled[str(h)] = {"avg_ret_pct": round(float(np.mean(vals)), 2) if vals else None,
                          "n": len(vals)} if vals else {"avg_ret_pct": None, "n": 0}
    # annualize each horizon so holds are comparable (a 6mo +X% vs 12 monthly rolls)
    for h, d in pooled.items():
        a = d.get("avg_ret_pct")
        if a is not None and h not in ("end",):
            months = int(h)
            d["annualized_pct"] = round(((1 + a / 100) ** (12.0 / months) - 1) * 100, 1)

    base1 = pooled["1"]["avg_ret_pct"]
    best_h = max([h for h in pooled if h != "end" and pooled[h]["avg_ret_pct"] is not None],
                 key=lambda h: pooled[h].get("annualized_pct", -999), default="1")
    verdict = (
        f"Across {len(all_events)} pick-events of the 18 splitter names, avg forward return by hold: "
        + ", ".join(f"{h}mo {pooled[h]['avg_ret_pct']}%" for h in ["1", "2", "3", "6", "12"] if pooled[h]['avg_ret_pct'] is not None)
        + f". Annualized, the best hold is {best_h}mo "
          f"({pooled[best_h].get('annualized_pct')}%/yr vs 1mo {pooled['1'].get('annualized_pct')}%/yr). "
        + ("Holding LONGER than the 1mo rotation would have earned MORE on these names — the rotation was clipping winners."
           if pooled[best_h].get("annualized_pct", 0) > pooled["1"].get("annualized_pct", 0) + 2 else
           "Holding longer does NOT beat the 1mo rotation on an annualized basis — the monthly rotate-out was right.")
    )
    print("\n=== POOLED forward return by hold (18 splitter names) ===", flush=True)
    for h in ["1", "2", "3", "6", "12", "end"]:
        d = pooled[h]
        print(f"  hold {h:>3}mo: avg {d['avg_ret_pct']}%  (n={d['n']})"
              + (f"  ~{d.get('annualized_pct')}%/yr" if 'annualized_pct' in d else ""), flush=True)
    print("\n=== per-name (n picks, avg fwd 1mo/3mo/6mo/12mo, held runs) ===", flush=True)
    for t in present:
        pn = per_name[t]
        if not pn.get("n_picks"):
            print(f"  {t:<10} never picked by the honest engine", flush=True); continue
        a = pn["avg_fwd_by_hold"]
        runs = "; ".join(f"{r['entry']}->{r['exit_after']}({r['months_held']}mo,{r['booked_return_pct']}%)" for r in pn["runs"])
        print(f"  {t:<10} picks={pn['n_picks']:>2}  1mo {a.get('1')}%  3mo {a.get('3')}%  "
              f"6mo {a.get('6')}%  12mo {a.get('12')}%  end {a.get('end')}%", flush=True)
        print(f"             runs: {runs}", flush=True)
    print("\n" + verdict, flush=True)

    return {
        "computed_at": pd.Timestamp.utcnow().isoformat(),
        "params": {"top_n": TOP_N, "min_dvol": MIN_DVOL, "horizons_months": HORIZONS,
                   "benchmark": BENCH, "months": int(len(midx)), "pb_basis": "as-traded (honest)"},
        "targets": TARGETS, "present": present, "missing": missing,
        "pooled_forward_by_hold": pooled, "per_name": per_name, "verdict": verdict,
        "caveat": "Forward returns are single-name buy-and-hold from each pick date (delist-aware), NOT re-selecting. "
                  "'1mo' = the actual rotation hold. Longer holds ignore what the rotation would have picked instead "
                  "(opportunity cost not modeled). as-traded P/B ranking, PIT, no fees, present-day-holdings survivorship.",
    }


def main():
    p = build()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(p, indent=2, default=str))
    try:
        from core.models import BacktestResult
        from django.utils import timezone
        BacktestResult.objects.update_or_create(
            kind="splitter_hold", defaults={"payload": json.loads(json.dumps(p, default=str)),
                                             "computed_at": timezone.now()})
        print("Saved BacktestResult[splitter_hold]", flush=True)
    except Exception as e:
        print("DB save failed:", e, flush=True)


if __name__ == "__main__":
    main()
