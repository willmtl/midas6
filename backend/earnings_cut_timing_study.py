#!/usr/bin/env python3
"""EARNINGS-MISS CUT — EXIT TIMING. The earnings_cut engine quit at MONTH-END (and detected the miss a month
late), so it ate the full post-miss drift. The right question (user): does quitting on the EXACT DAY of the
miss beat waiting for month-end? Post-earnings-announcement drift says negative surprises keep sliding for
weeks, so exact-day exit should help.

Same position-tracking flagship (accel top-10 sectors, one per sector, cheapest as-traded-P/B guard low-debt,
$5M floor, div_2x, hold up to MAX_HOLD months). A miss (grounded_score<0, else eps_surprise<-1) is detected
in the month it lands. Three exit timings compared, holding selection identical:

  rotate_1mo          baseline monthly churn (no hold, no cut).
  cut_monthend_maxH   on a miss during the hold, earn the FULL month then exit at month-end (lag).
  cut_exactday_maxH   on a miss during the hold, exit at the close the day AFTER the print; cash to month-end.

If cut_exactday > cut_monthend, the drift is real and timing matters; either way we see if precise cutting can
close the gap to the monthly baseline.
-> BacktestResult[earnings_cut_timing] + JSON.
Run: MSYS_NO_PATHCONV=1 docker exec rotation-backend-1 python -u /app/earnings_cut_timing_study.py
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
from backtest_lowpb import _monthly_close, _tstat_from_returns, BENCH

TOP_N = 10
CONVICTION_MULT = 2.0
MIN_DVOL = 5e6
MAX_HOLDS = [3, 6]
OUT = Path(__file__).resolve().parent / ".data" / "studies" / "earnings_cut_timing.json"


def load_miss(tickers):
    from core.models import EarningsEvent
    out = {}
    for r in EarningsEvent.objects.filter(ticker__in=list(tickers)).values(
            "ticker", "report_date", "grounded_score", "eps_surprise_pct"):
        if r["report_date"] is None:
            continue
        gs, eps = r["grounded_score"], r["eps_surprise_pct"]
        is_miss = (gs is not None and gs < 0) or (gs is None and eps is not None and eps < -1.0)
        if is_miss:
            out.setdefault(r["ticker"], []).append(pd.Timestamp(r["report_date"]))
    for t in out:
        out[t].sort()
    return out


def _perf(r, spy):
    r = np.asarray(r, float); n = len(r)
    tot = float(np.prod(1 + r) - 1) * 100
    sp = float(np.prod(1 + np.asarray(spy)) - 1) * 100
    sh = float(r.mean() / r.std() * np.sqrt(12)) if r.std() > 1e-9 else 0.0
    eqc = np.cumprod(1 + r); dd = float(((eqc / np.maximum.accumulate(eqc)) - 1).min() * 100)
    t = _tstat_from_returns(list(r))
    return dict(total=round(tot, 1), vs_spy=round(tot - sp, 1), sharpe=round(sh, 2), dd=round(dd, 1),
                t_stat=round(t, 2) if t is not None else None, months=n, spy_total=round(sp, 1))


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
    spy_m = etf_daily[BENCH]["Close"].resample("ME").last().reindex(midx)
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
    dvol, adl_m = {}, {}
    DCL = {}   # daily close per name (for exact-day exit)
    for t in common:
        d = stock_daily.get(t)
        if d is None or "Volume" not in d or len(d) < 90:
            continue
        v = d["Volume"]; DCL[t] = d["Close"]
        dvol[t] = (d["Close"] * v).rolling(20).mean().resample("ME").last().reindex(midx)
        if {"High", "Low", "Close"}.issubset(d.columns):
            rng = (d["High"] - d["Low"]).replace(0, np.nan)
            mfm = ((d["Close"] - d["Low"]) - (d["High"] - d["Close"])) / rng
            adl_m[t] = (mfm.fillna(0) * v).cumsum().resample("ME").last().reindex(midx)
    dvol = pd.DataFrame(dvol).reindex(index=midx, columns=common)
    adl = pd.DataFrame(adl_m).reindex(index=midx, columns=common)
    ad_slope3 = adl - adl.shift(3); px_ret3 = px.pct_change(3)
    miss = load_miss(common)
    print(f"months {len(midx)} | stocks {len(common)} | names with a miss on record {len(miss)}", flush=True)

    def eligible(etf, date, held_names):
        _, holds = sector_map.get(etf, (etf, []))
        c = [h for h in holds if h in px.columns and h not in held_names and _available_at(px[h], date)
             and pd.notna(pb.loc[date, h]) and pb.loc[date, h] > 0 and not bool(trap.loc[date, h])
             and pd.notna(dvol.loc[date, h]) and dvol.loc[date, h] >= MIN_DVOL]
        g = [x for x in c if bool(low.loc[date, x])] or c
        return min(g, key=lambda h: pb.loc[date, h]) if g else None

    def divweight(name, date):
        a, p = ad_slope3.loc[date].get(name), px_ret3.loc[date].get(name)
        return CONVICTION_MULT if (pd.notna(a) and pd.notna(p) and a > 0 and p < 0) else 1.0

    def miss_day(name, date, ndate):
        """first miss report in (date, ndate); None otherwise."""
        for rd in miss.get(name, []):
            if date < rd < ndate:
                return rd
            if rd >= ndate:
                break
        return None

    def realize(name, date, ndate, timing):
        """(month return, exited_this_month, cut_type). exact-day exits the close AFTER the print."""
        md = miss_day(name, date, ndate) if timing != "none" else None
        if md is None:
            r = _ret_delist(px[name], date, ndate)
            return (float(r) if r is not None and np.isfinite(r) else None), False, None
        if timing == "monthend":
            r = _ret_delist(px[name], date, ndate)          # eat the full month, then exit
            return (float(r) if r is not None and np.isfinite(r) else None), True, "monthend"
        # exact-day: exit at the first daily close strictly after the report date
        s = DCL.get(name)
        if s is None or len(s) == 0:
            r = _ret_delist(px[name], date, ndate)
            return (float(r) if r is not None and np.isfinite(r) else None), True, "fallback"
        k = s.index.searchsorted(md, side="right")
        if k >= len(s) or s.index[k] > ndate:
            r = _ret_delist(px[name], date, ndate)
            return (float(r) if r is not None and np.isfinite(r) else None), True, "no_exit_day"
        p0 = s.asof(date)
        p1 = float(s.iloc[k])
        if not (np.isfinite(p0) and p0 > 0 and np.isfinite(p1) and p1 > 0):
            r = _ret_delist(px[name], date, ndate)
            return (float(r) if r is not None and np.isfinite(r) else None), True, "bad_px"
        return (p1 / p0 - 1), True, "exactday"

    def run(policy, max_hold, timing):
        held = {}
        rets, spies, turns, hold_lens = [], [], [], []
        cut_ct = maxhold_ct = 0
        prev = set()
        for i in range(9, len(midx) - 1):
            date, ndate = midx[i], midx[i + 1]
            sp = spy_m.iloc[i + 1] / spy_m.iloc[i] - 1
            if not np.isfinite(sp):
                continue
            survivors = {}
            for t, pos in held.items():
                hm = i - pos["entry_i"]
                if policy == "rotate_1mo":
                    hold_lens.append(hm); continue
                if hm >= max_hold:
                    maxhold_ct += 1; hold_lens.append(hm); continue
                survivors[t] = pos                          # miss handled in realize()
            rep = {p["etf"] for p in survivors.values()}
            ranks = accel.loc[date].dropna().sort_values(ascending=False).head(TOP_N).index
            for etf in ranks:
                if len(survivors) >= TOP_N:
                    break
                if etf in rep:
                    continue
                pick = eligible(etf, date, set(survivors.keys()))
                if pick is None:
                    continue
                survivors[pick] = {"entry_i": i, "entry_date": date, "etf": etf}
                rep.add(etf)
            held = survivors
            wsum = rr = 0.0; keep = {}
            for t, pos in held.items():
                r, exited, ctype = realize(t, date, ndate, timing if policy == "cut_on_miss" else "none")
                if r is None:
                    continue
                w = divweight(t, date)
                wsum += w; rr += w * r
                if exited:
                    cut_ct += 1
                else:
                    keep[t] = pos
            held = keep
            if wsum <= 0:
                continue
            rets.append(rr / wsum); spies.append(float(sp))
            cur = set(held.keys())
            turns.append(1.0 - len(prev & cur) / max(len(cur), 1)); prev = cur
        perf = _perf(rets, spies)
        perf["avg_turnover_pct"] = round(float(np.mean(turns)) * 100, 1) if turns else None
        perf["miss_cuts"] = cut_ct; perf["maxhold_cuts"] = maxhold_ct
        return perf

    results = {"rotate_1mo": run("rotate_1mo", 1, "none")}
    for H in MAX_HOLDS:
        results[f"cut_monthend_max{H}mo"] = run("cut_on_miss", H, "monthend")
        results[f"cut_exactday_max{H}mo"] = run("cut_on_miss", H, "exactday")

    base = results["rotate_1mo"]
    print(f"\n=== EXIT TIMING on the earnings-miss cut ===", flush=True)
    order = ["rotate_1mo"] + [k for H in MAX_HOLDS for k in (f"cut_monthend_max{H}mo", f"cut_exactday_max{H}mo")]
    for k in order:
        r = results[k]
        print(f"  {k:<22} {r['total']:>7}%  vsSPY {r['vs_spy']:>7}  Sh {r['sharpe']:>5}  DD {r['dd']:>6}%  "
              f"t {r['t_stat']}  turn {r['avg_turnover_pct']}%  [miss-cuts {r['miss_cuts']}]", flush=True)

    deltas = {H: round(results[f"cut_exactday_max{H}mo"]["total"] - results[f"cut_monthend_max{H}mo"]["total"], 1)
              for H in MAX_HOLDS}
    verdict = (
        f"Exact-day vs month-end miss exit (total-return delta): "
        + ", ".join(f"{H}mo {d:+.0f}pp" for H, d in deltas.items()) + ". "
        + ("Cutting on the exact day BEATS waiting for month-end -> post-miss drift is real, timing matters. "
           if all(d > 0 for d in deltas.values()) else
           "Cutting on the exact day does NOT reliably beat month-end -> the intra-month drift is noise here. ")
        + f"Best cut variant still {'BELOW' if max(results[k]['sharpe'] for k in order if k!='rotate_1mo') < base['sharpe'] else 'AT/ABOVE'} "
          f"the monthly baseline (Sh {base['sharpe']}, {base['total']}%)."
    )
    print("\n" + verdict, flush=True)
    return {
        "computed_at": pd.Timestamp.utcnow().isoformat(),
        "params": {"top_n": TOP_N, "min_dvol": MIN_DVOL, "max_holds": MAX_HOLDS, "benchmark": BENCH,
                   "months": int(len(midx)), "miss_rule": "grounded_score<0 (else eps_surprise<-1%)",
                   "exact_day_exit": "close of first trading day AFTER report_date; cash to month-end"},
        "results": results, "exactday_minus_monthend_pp": deltas, "verdict": verdict,
        "caveat": "Exit at close the day after the print (not intraday); cash (0%) for the rest of the month, "
                  "slot refilled next rebalance. report_date used as the event day (before/after-market not modeled). "
                  "PIT, no fees, present-day-holdings survivorship. as-traded P/B, div_2x.",
    }


def main():
    p = build()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(p, indent=2, default=str))
    try:
        from core.models import BacktestResult
        from django.utils import timezone
        BacktestResult.objects.update_or_create(
            kind="earnings_cut_timing", defaults={"payload": json.loads(json.dumps(p, default=str)),
                                                   "computed_at": timezone.now()})
        print("Saved BacktestResult[earnings_cut_timing]", flush=True)
    except Exception as e:
        print("DB save failed:", e, flush=True)


if __name__ == "__main__":
    main()
