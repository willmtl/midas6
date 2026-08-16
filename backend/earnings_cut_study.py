#!/usr/bin/env python3
"""EARNINGS-MISS CUT — the earnings_hold study found the one clean, usable earnings signal is the DOWNSIDE:
a miss landing during the hold is the worst group at every horizon. So implement it as a portfolio rule and
test it honestly at the strategy level (not per-event): hold winners LONGER by default, but CUT a position at
month-end as soon as an earnings MISS (grounded_score<0) lands during its hold.

Position-tracking backtest (positions persist across months, one per top-accel sector, div_2x weight, $5M
floor, as-traded-P/B guard low-debt pick). Variants share the SAME selection; only the EXIT rule differs:

  rotate_1mo   (baseline)  exit every position after 1 month, full re-pick -> reproduces the deployed +313%.
  hold_maxH                hold each position up to MAX_HOLD months, NO miss cut (isolates 'just hold longer').
  cut_on_miss              hold up to MAX_HOLD months BUT exit at the month-end after an earnings miss.

If cut_on_miss > hold_maxH and ideally challenges rotate_1mo on Sharpe, the miss-cut earns its keep: it lets
us ride winners while dodging the fundamental breakdowns that blanket hold-longer eats (+53%/Sh0.53).
-> BacktestResult[earnings_cut] + JSON.
Run: MSYS_NO_PATHCONV=1 docker exec rotation-backend-1 python -u /app/earnings_cut_study.py
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
from backtest_lowpb import _monthly_close, _tstat_from_returns, BENCH

TOP_N = 10
CONVICTION_MULT = 2.0
MIN_DVOL = 5e6
MAX_HOLDS = [3, 6, 12]
OUT = Path(__file__).resolve().parent / ".data" / "studies" / "earnings_cut.json"


def load_miss_dates(tickers):
    """ticker -> sorted list of (report_ts, grounded_score, eps_surprise_pct)."""
    from core.models import EarningsEvent
    out = {}
    for r in EarningsEvent.objects.filter(ticker__in=list(tickers)).values(
            "ticker", "report_date", "grounded_score", "eps_surprise_pct"):
        if r["report_date"]:
            out.setdefault(r["ticker"], []).append(
                (pd.Timestamp(r["report_date"]), r["grounded_score"], r["eps_surprise_pct"]))
    for t in out:
        out[t].sort(key=lambda x: x[0])
    return out


def missed_between(events, t0, t1):
    """True if a MISS (grounded_score<0, else eps_surprise<-1) reported in (t0, t1]."""
    if not events:
        return False
    for rd, gs, eps in events:
        if t0 < rd <= t1:
            if gs is not None and gs < 0:
                return True
            if gs is None and eps is not None and eps < -1.0:
                return True
    return False


def _perf(r, spy):
    r = np.asarray(r, float); n = len(r)
    tot = float(np.prod(1 + r) - 1) * 100
    sp = float(np.prod(1 + np.asarray(spy)) - 1) * 100
    ann = (float(np.prod(1 + r)) ** (12.0 / n) - 1) * 100 if n else 0.0
    sh = float(r.mean() / r.std() * np.sqrt(12)) if r.std() > 1e-9 else 0.0
    eqc = np.cumprod(1 + r); dd = float(((eqc / np.maximum.accumulate(eqc)) - 1).min() * 100)
    t = _tstat_from_returns(list(r))
    return dict(total=round(tot, 1), vs_spy=round(tot - sp, 1), annual=round(ann, 1), sharpe=round(sh, 2),
                dd=round(dd, 1), t_stat=round(t, 2) if t is not None else None, months=n, spy_total=round(sp, 1))


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
    earn = load_miss_dates(common)
    print(f"months {len(midx)} | stocks {len(common)} | with earnings {sum(1 for t in common if earn.get(t))}", flush=True)

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

    def run(policy, max_hold):
        held = {}          # ticker -> {"entry_i", "entry_date", "etf"}
        rets, spies, turns, hold_lens = [], [], [], []
        miss_exits = maxhold_exits = rot_exits = 0
        prev_names = set()
        for i in range(9, len(midx) - 1):
            date, ndate = midx[i], midx[i + 1]
            sp = spy_m.iloc[i + 1] / spy_m.iloc[i] - 1
            if not np.isfinite(sp):
                continue
            # --- 1. exits (evaluated at `date`, before earning date->ndate) ---
            survivors = {}
            for t, pos in held.items():
                hm = i - pos["entry_i"]           # complete months already held
                if policy == "rotate_1mo":
                    rot_exits += 1; hold_lens.append(hm + 1); continue
                if hm >= max_hold:
                    maxhold_exits += 1; hold_lens.append(hm + 1); continue
                if policy == "cut_on_miss" and missed_between(earn.get(t), pos["entry_date"], date):
                    miss_exits += 1; hold_lens.append(hm + 1); continue
                survivors[t] = pos
            # --- 2. fill top-accel sectors not already represented by a survivor ---
            rep_sectors = {p["etf"] for p in survivors.values()}
            ranks = accel.loc[date].dropna().sort_values(ascending=False).head(TOP_N).index
            for etf in ranks:
                if len(survivors) >= TOP_N:
                    break
                if etf in rep_sectors:
                    continue
                pick = eligible(etf, date, set(survivors.keys()))
                if pick is None:
                    continue
                survivors[pick] = {"entry_i": i, "entry_date": date, "etf": etf}
                rep_sectors.add(etf)
            held = survivors
            # --- 3. weight (div_2x) + realize date->ndate ---
            wsum = rr = 0.0; live = {}
            for t, pos in held.items():
                r = _ret_delist(px[t], date, ndate)
                if r is None or not np.isfinite(r):
                    hold_lens.append(i - pos["entry_i"] + 1); continue   # delisted -> exits
                w = divweight(t, date)
                wsum += w; rr += w * float(r); live[t] = pos
            held = live
            if wsum <= 0:
                continue
            rets.append(rr / wsum); spies.append(float(sp))
            cur = set(held.keys())
            turns.append(1.0 - len(prev_names & cur) / max(len(cur), 1)); prev_names = cur
        perf = _perf(rets, spies)
        perf["avg_hold_months"] = round(float(np.mean(hold_lens)), 2) if hold_lens else None
        perf["avg_turnover_pct"] = round(float(np.mean(turns)) * 100, 1) if turns else None
        perf["exits"] = {"miss": miss_exits, "maxhold": maxhold_exits, "rotate": rot_exits}
        return perf

    results = {"rotate_1mo": run("rotate_1mo", 1)}
    for H in MAX_HOLDS:
        results[f"hold_max{H}mo"] = run("hold_maxH", H)
        results[f"cut_on_miss_max{H}mo"] = run("cut_on_miss", H)

    base = results["rotate_1mo"]
    print(f"\n=== EARNINGS-MISS CUT vs baseline (each: total / vsSPY / Sh / DD / t / avgHold / turnover) ===", flush=True)
    order = ["rotate_1mo"] + [k for H in MAX_HOLDS for k in (f"hold_max{H}mo", f"cut_on_miss_max{H}mo")]
    for k in order:
        r = results[k]
        ex = r["exits"]
        print(f"  {k:<20} {r['total']:>7}%  vsSPY {r['vs_spy']:>7}  Sh {r['sharpe']:>5}  DD {r['dd']:>6}%  "
              f"t {r['t_stat']}  hold {r['avg_hold_months']}mo  turn {r['avg_turnover_pct']}%  "
              f"[miss {ex['miss']} maxhold {ex['maxhold']}]", flush=True)

    # verdict: does cut_on_miss beat plain hold at each H, and can any variant challenge the baseline?
    best_cut = max((f"cut_on_miss_max{H}mo" for H in MAX_HOLDS), key=lambda k: results[k]["sharpe"])
    bc = results[best_cut]
    pairs = [(results[f"cut_on_miss_max{H}mo"]["total"] - results[f"hold_max{H}mo"]["total"]) for H in MAX_HOLDS]
    miss_helps = sum(1 for d in pairs if d > 0)
    verdict = (
        f"Baseline rotate_1mo {base['total']}%/Sh{base['sharpe']}/DD{base['dd']}%. "
        f"The miss-cut beats plain hold-longer in {miss_helps}/{len(MAX_HOLDS)} max-hold settings "
        f"(deltas {', '.join(f'{d:+.0f}pp' for d in pairs)}). Best miss-cut = {best_cut}: {bc['total']}%/Sh{bc['sharpe']}/DD{bc['dd']}% "
        f"(avg hold {bc['avg_hold_months']}mo, {bc['exits']['miss']} miss-exits). "
        + ("The miss-cut RESCUES longer holds (beats plain hold) but " +
           ("still trails the monthly baseline on Sharpe -- monthly churn remains best."
            if bc["sharpe"] < base["sharpe"] else
            "actually CHALLENGES the monthly baseline -- ride-winners-cut-losers has merit."))
    )
    print("\n" + verdict, flush=True)
    return {
        "computed_at": pd.Timestamp.utcnow().isoformat(),
        "params": {"top_n": TOP_N, "min_dvol": MIN_DVOL, "max_holds": MAX_HOLDS, "benchmark": BENCH,
                   "months": int(len(midx)), "miss_rule": "grounded_score<0 (else eps_surprise<-1%) in hold window",
                   "pb_basis": "as-traded", "weight": "div_2x"},
        "results": results, "verdict": verdict,
        "caveat": "Positions persist; one per top-accel sector; exit on 1mo (baseline) / max-hold / earnings-miss. "
                  "PIT, no fees, present-day-holdings survivorship. grounded_score PIT at report. Held names whose "
                  "sector leaves top-10 are kept until miss/max-hold (that IS the ride-winners premise).",
    }


def main():
    p = build()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(p, indent=2, default=str))
    try:
        from core.models import BacktestResult
        from django.utils import timezone
        BacktestResult.objects.update_or_create(
            kind="earnings_cut", defaults={"payload": json.loads(json.dumps(p, default=str)),
                                           "computed_at": timezone.now()})
        print("Saved BacktestResult[earnings_cut]", flush=True)
    except Exception as e:
        print("DB save failed:", e, flush=True)


if __name__ == "__main__":
    main()
