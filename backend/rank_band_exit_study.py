#!/usr/bin/env python3
"""RANK-BAND (HYSTERESIS) EXIT — the flagship is BINARY: a sector is in the portfolio iff its momentum-accel rank
is in the top-10, else we rotate the name out at month-end. That clipped big winners (AVGO Aug-2024 booked +6.3%
vs +145.9% keep-to-end) because the SECTOR left the top-10 the very next month. User question: when we rotated
those out, HOW FAR had the sector actually fallen? If it only slipped to rank ~12-15, a HOLD-BAND (enter on
top-10, but only EXIT when the sector falls past rank K>10) would have kept us in the winner. If it collapsed to
rank ~50, nothing would have saved it -- the cut was correct.

Two parts, one engine (accel top-10 entry -> cheapest as-traded-P/B guard low-debt $5M pick, div_2x weight):
  (A) DIAGNOSTIC: run the binary baseline; every time a position rotates out, record the sector's accel RANK at
      that exit rebalance. Distribution of rank-at-exit answers "shallow slip vs deep collapse". Plus the full
      rank trajectory of the clipped-winner names (AVGO/MTH/GOLD/GOOGL/ANET/NVDA) from entry to exit.
  (B) BACKTEST: hold-band variants. Enter a sector only if rank<=TOP_N (10) AND a slot is free; HOLD THE SAME
      NAME as long as its sector rank<=EXIT_K; exit when rank>EXIT_K (or delist). Sweep EXIT_K in {10,15,20,25,
      30,40}. K=10 == binary baseline (holding the same name while top-10). Compare total/vsSPY/Sharpe/DD/turnover.
      If a wider band lifts return without wrecking Sharpe/DD, the binary exit is leaving winner-drift on the table.
-> BacktestResult[rank_band_exit] + JSON.
Run: MSYS_NO_PATHCONV=1 docker exec rotation-backend-1 python -u /app/rank_band_exit_study.py
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

TOP_N = 10                                   # entry threshold + max simultaneous positions
CONVICTION_MULT = 2.0
MIN_DVOL = 5e6
EXIT_BANDS = [10, 15, 20, 25, 30, 40]        # K: exit when sector accel-rank > K (10 == binary baseline)
TARGETS = ["AVGO", "MTH", "GOLD", "GOOGL", "ANET", "NVDA"]   # the clipped winners
OUT = Path(__file__).resolve().parent / ".data" / "studies" / "rank_band_exit.json"


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
    tk_to_etf = {}
    for n, e in etfs.items():
        h = [t for t in sector_holdings.get_holdings(n) if t not in (e, BENCH) and t not in CRYPTO]
        sector_map[e] = (n, h); all_holds.update(h)
        for t in h:
            tk_to_etf.setdefault(t, e)
    all_holds = sorted(all_holds)
    etf_tk = list(etfs.values())
    print(f"Loading {len(etf_tk)} ETFs + {len(all_holds)} stocks + {BENCH}...", flush=True)
    etf_daily = load_candles(etf_tk + [BENCH])
    etf_m = _monthly_close({t: d for t, d in etf_daily.items() if t in etf_tk})
    midx = etf_m.index
    accel = etf_m.pct_change(3) - etf_m.pct_change(3).shift(3)
    rank = accel.rank(axis=1, ascending=False)            # 1 = strongest accel; NaN where accel NaN
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
    print(f"months {len(midx)} | stocks {len(common)}", flush=True)

    def sector_rank(etf, date):
        r = rank.loc[date].get(etf)
        return float(r) if pd.notna(r) else None

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

    def run(exit_k, collect=False):
        """Hold the SAME name while its sector rank<=exit_k; exit when rank>exit_k or delist.
        Entry only when rank<=TOP_N and a slot (<TOP_N) is free."""
        held = {}          # ticker -> {"entry_i","entry_date","etf","ranks":[...]}
        rets, spies, turns, hold_lens = [], [], [], []
        exits = []         # collected exit records (diagnostic)
        traces = {}        # target -> list of {date, rank, held}
        prev_names = set()
        for i in range(9, len(midx) - 1):
            date, ndate = midx[i], midx[i + 1]
            sp = spy_m.iloc[i + 1] / spy_m.iloc[i] - 1
            if not np.isfinite(sp):
                continue
            # --- 1. exits: sector fell past the band, or the name delisted ---
            survivors = {}
            for t, pos in held.items():
                sr = sector_rank(pos["etf"], date)
                if sr is None or sr > exit_k:
                    hm = i - pos["entry_i"]
                    hold_lens.append(hm)
                    if collect:
                        entry_ret = _ret_delist(px[t], pos["entry_date"], date)
                        exits.append({"ticker": t, "etf": pos["etf"], "entry": str(pos["entry_date"].date()),
                                      "exit": str(date.date()), "months_held": int(hm),
                                      "rank_at_exit": None if sr is None else round(sr, 0),
                                      "peak_rank_in_hold": (max(pos["ranks"]) if pos["ranks"] else None),
                                      "booked_ret_pct": round(float(entry_ret) * 100, 1)
                                      if entry_ret is not None and np.isfinite(entry_ret) else None})
                    continue
                survivors[t] = pos
            # --- 2. fill free slots from top-N entry sectors not already held ---
            rep_sectors = {p["etf"] for p in survivors.values()}
            entry_order = rank.loc[date].dropna().sort_values().index          # rank asc => best first
            for etf in entry_order:
                if len(survivors) >= TOP_N:
                    break
                if sector_rank(etf, date) is None or sector_rank(etf, date) > TOP_N or etf in rep_sectors:
                    continue
                pick = eligible(etf, date, set(survivors.keys()))
                if pick is None:
                    continue
                survivors[pick] = {"entry_i": i, "entry_date": date, "etf": etf, "ranks": []}
                rep_sectors.add(etf)
            held = survivors
            # --- 3. record rank, weight (div_2x), realize date->ndate ---
            for t, pos in held.items():
                sr = sector_rank(pos["etf"], date)
                if sr is not None:
                    pos["ranks"].append(sr)
                if collect and t in TARGETS:
                    traces.setdefault(t, []).append({"date": str(date.date()),
                                                     "rank": None if sr is None else round(sr, 0)})
            wsum = rr = 0.0; live = {}
            for t, pos in held.items():
                r = _ret_delist(px[t], date, ndate)
                if r is None or not np.isfinite(r):
                    hold_lens.append(i - pos["entry_i"]); continue
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
        if collect:
            return perf, exits, traces
        return perf

    # --- (A) diagnostic on the binary baseline (exit_k = TOP_N) ---
    base_perf, exits, traces = run(TOP_N, collect=True)
    ranks_at_exit = [e["rank_at_exit"] for e in exits if e["rank_at_exit"] is not None]
    ra = np.asarray(ranks_at_exit, float)
    # how many exits were a SHALLOW slip (rank 11..K) recoverable by a band, vs deep/gone
    def frac_within(k):
        return round(float(np.mean((ra > TOP_N) & (ra <= k))) * 100, 1) if len(ra) else None
    shallow = {f"rank_11_to_{k}": frac_within(k) for k in (15, 20, 30)}
    gone = round(float(np.mean(np.isnan(np.array([e["rank_at_exit"] for e in exits], float)))) * 100, 1) if exits else None
    # winners that were clipped: booked vs how far the sector had fallen
    clipped = [e for e in exits if e["ticker"] in TARGETS]

    print(f"\n=== (A) RANK-AT-EXIT on the binary baseline ({len(exits)} rotate-outs) ===", flush=True)
    if len(ra):
        print(f"  rank-at-exit: median {np.median(ra):.0f}  mean {ra.mean():.1f}  "
              f"p25 {np.percentile(ra,25):.0f}  p75 {np.percentile(ra,75):.0f}  max {ra.max():.0f}", flush=True)
    for k in (15, 20, 30):
        print(f"  exited at rank 11..{k} (a band K={k} would have KEPT it another month): {frac_within(k)}%", flush=True)
    print(f"  exited because sector rank went N/A (data gap / truly gone): {gone}%", flush=True)
    print("\n  clipped-winner exits (booked vs rank the sector had fallen to at exit):", flush=True)
    for e in sorted(clipped, key=lambda x: x["entry"]):
        print(f"    {e['ticker']:<6} entry {e['entry']} held {e['months_held']}mo -> exit {e['exit']} "
              f"at rank {e['rank_at_exit']}  (best rank in hold {e['peak_rank_in_hold']})  booked {e['booked_ret_pct']}%",
              flush=True)

    # --- (B) hold-band sweep ---
    results = {}
    for k in EXIT_BANDS:
        results[f"band_K{k}"] = run(k)
    print(f"\n=== (B) HOLD-BAND SWEEP (enter top-{TOP_N}, exit when sector rank > K; K=10 == binary) ===", flush=True)
    print(f"  {'variant':<12}{'total':>9}{'vsSPY':>9}{'Sharpe':>8}{'DD':>8}{'t':>6}{'hold':>7}{'turn':>7}", flush=True)
    for k in EXIT_BANDS:
        r = results[f"band_K{k}"]
        print(f"  band_K{k:<7}{r['total']:>8}%{r['vs_spy']:>9}{r['sharpe']:>8}{r['dd']:>7}%{str(r['t_stat']):>6}"
              f"{str(r['avg_hold_months']):>7}{str(r['avg_turnover_pct'])+'%':>7}", flush=True)

    base = results["band_K10"]
    best_k = max(EXIT_BANDS, key=lambda k: results[f"band_K{k}"]["total"])
    best_sh_k = max(EXIT_BANDS, key=lambda k: results[f"band_K{k}"]["sharpe"])
    bt, bs = results[f"band_K{best_k}"], results[f"band_K{best_sh_k}"]
    shallow15 = frac_within(15)
    verdict = (
        f"Binary baseline (K=10) {base['total']}%/Sh{base['sharpe']}/DD{base['dd']}%. "
        f"Of {len(exits)} rotate-outs, {shallow15}% exited at rank 11-15 (a shallow slip a band would recover) "
        f"and median rank-at-exit was {np.median(ra):.0f}. "
        f"Best total = band_K{best_k} ({bt['total']}%, {bt['vs_spy']:+}pp vs SPY, Sh{bt['sharpe']}, DD{bt['dd']}%); "
        f"best Sharpe = band_K{best_sh_k} (Sh{bs['sharpe']}, {bs['total']}%). "
        + ("Widening the exit band LIFTS return over the binary cut -- the flagship is rotating winners out one "
           "notch too early." if bt["total"] > base["total"] + 10 else
           "Widening the exit band does NOT beat the binary cut -- sectors that leave the top-10 keep decaying, "
           "so the crisp rotation is right.")
        + (f" (Sharpe-best band K{best_sh_k} trades some return for a smoother ride.)"
           if best_sh_k != best_k else "")
    )
    print("\n" + verdict, flush=True)
    return {
        "computed_at": pd.Timestamp.utcnow().isoformat(),
        "params": {"top_n": TOP_N, "min_dvol": MIN_DVOL, "exit_bands": EXIT_BANDS, "conviction_mult": CONVICTION_MULT,
                   "benchmark": BENCH, "months": int(len(midx)), "pb_basis": "as-traded", "weight": "div_2x",
                   "hold_rule": "hold the SAME name while sector accel-rank<=K; entry only when rank<=top_n & slot free"},
        "diagnostic": {
            "n_exits": len(exits),
            "rank_at_exit": {"median": float(np.median(ra)) if len(ra) else None,
                             "mean": round(float(ra.mean()), 1) if len(ra) else None,
                             "p25": float(np.percentile(ra, 25)) if len(ra) else None,
                             "p75": float(np.percentile(ra, 75)) if len(ra) else None,
                             "max": float(ra.max()) if len(ra) else None},
            "shallow_slip_pct": shallow, "gone_na_pct": gone,
            "clipped_winner_exits": sorted(clipped, key=lambda x: x["entry"]),
            "target_rank_traces": traces,
        },
        "results": results, "best_total_band": f"band_K{best_k}", "best_sharpe_band": f"band_K{best_sh_k}",
        "verdict": verdict,
        "caveat": "Holds the SAME name once bought (rides the winner) while its sector stays in the exit band; a "
                  "wider band means slots stay occupied by rank 11-K sectors, so fewer fresh top-10 entries -- that "
                  "opportunity cost IS reflected in the returns. PIT, no fees, present-day-holdings survivorship. "
                  "as-traded P/B for ranking, adjusted close for returns. K=10 holds the same name (deployed strat "
                  "re-picks monthly), so band_K10 is ~ but not identical to the live +313%.",
    }


def main():
    p = build()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(p, indent=2, default=str))
    try:
        from core.models import BacktestResult
        from django.utils import timezone
        BacktestResult.objects.update_or_create(
            kind="rank_band_exit", defaults={"payload": json.loads(json.dumps(p, default=str)),
                                             "computed_at": timezone.now()})
        print("Saved BacktestResult[rank_band_exit]", flush=True)
    except Exception as e:
        print("DB save failed:", e, flush=True)


if __name__ == "__main__":
    main()
