#!/usr/bin/env python3
"""FRESHNESS-FILTERED strategy — does avoiding the stale-tail sectors (5+ months in the top-10 and/or
fading momentum) actually improve the value-pick engine? Same engine (top-10 by 6mo momentum -> cheapest-
P/B guarded low-debt), variants change ONLY which sectors are eligible:

  baseline               top-10 by 6mo momentum (all)
  drop_stale5            drop sectors in the top-10 for 5+ consecutive months (fewer names)
  drop_fading            drop sectors whose 3mo momentum is fading (fewer names)
  drop_stale_and_fading  drop only the worst quadrant: stale (5+mo) AND fading (surgical)
  fill_fresh             rebuild 10 sectors skipping stale ones, filling from rank 11+ (keep breadth)
Reports vsSPY / Sharpe / DD / per-pick win rate / avg names per month (breadth cost).
-> BacktestResult[freshness_filter] + JSON.
Run: MSYS_NO_PATHCONV=1 docker exec rotation-backend-1 python -u /app/freshness_filter_study.py
"""
import os, json, warnings
warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
os.environ.setdefault("DJANGO_SETTINGS_MODULE", "rotation.settings")
import django
django.setup()

from pathlib import Path
import config, sector_holdings, price_basis
from studies import _tstat_from_returns
from seq_fundamental_study import load_candles, load_financial_reports
from trend_stock_studies import _pit_monthly_panel, _available_at, _ret_delist, CRYPTO
from backtest_lowpb import _monthly_close, BENCH

OUT = Path(__file__).resolve().parent / ".data" / "studies" / "freshness_filter.json"
TOP_N, STALE = 10, 5


def _stats(r, spy):
    r = np.asarray(r, float); n = len(r)
    if n == 0:
        return dict(total_return=0, vs_spy=0, sharpe=0, max_drawdown=0, t_stat=None, periods=0)
    tot = float(np.prod(1 + r) - 1) * 100
    sp = float(np.prod(1 + np.asarray(spy)) - 1) * 100
    sh = float(r.mean() / r.std() * np.sqrt(12)) if r.std() > 1e-9 else 0.0
    eq = np.cumprod(1 + r); dd = float(((eq / np.maximum.accumulate(eq)) - 1).min() * 100)
    t = _tstat_from_returns(list(r))
    return dict(total_return=round(tot, 1), vs_spy=round(tot - sp, 1), sharpe=round(sh, 2),
                max_drawdown=round(dd, 1), t_stat=round(t, 2) if t is not None else None, periods=n)


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
    spy_m = etf_daily[BENCH]["Close"].resample("ME").last().reindex(midx)
    mom6 = etf_m.pct_change(6)
    mom3, mom3_prev = etf_m.pct_change(3), etf_m.pct_change(3).shift(3)
    stock_daily = load_candles(all_holds)
    stock_m = _monthly_close(stock_daily).reindex(midx)

    reps = load_financial_reports(all_holds)
    P = lambda f: _pit_monthly_panel(reps, f, midx)
    shares, equity, ni, debt = P("shares_outstanding"), P("total_equity"), P("net_income"), P("total_debt")
    common = stock_m.columns.intersection(shares.columns).intersection(equity.columns)

    def Rf(p):
        return p.reindex(index=midx, columns=common)
    px = stock_m[common]
    shares, equity, ni, debt = map(Rf, (shares, equity, ni, debt))
    pb = (price_basis.as_traded_close(px) * shares) / equity.where(equity != 0)
    trap = (ni < 0) & (~(equity >= equity.shift(12))) & (~(ni > ni.shift(4)))
    low_debt = (debt / equity.where(equity != 0)) < 1.0
    print(f"months {len(midx)} | stocks {len(common)}", flush=True)
    warmup = 12

    # precompute per-month: baseline ranked sectors, run_len (consec months in top-10), accel flag
    run_len = {e: 0 for e in etf_tk}
    ranked_by, runlen_by, accel_by = {}, {}, {}
    for i in range(len(midx)):
        date = midx[i]
        row = mom6.loc[date].dropna().sort_values(ascending=False)
        ranked = list(row.index)
        top10 = set(ranked[:TOP_N])
        for e in etf_tk:
            run_len[e] = run_len[e] + 1 if e in top10 else 0
        ranked_by[i] = ranked
        runlen_by[i] = dict(run_len)
        accel_by[i] = {e: (pd.notna(mom3.loc[date, e]) and pd.notna(mom3_prev.loc[date, e])
                           and mom3.loc[date, e] > mom3_prev.loc[date, e]) for e in etf_tk}

    def pick_in(etf, date, ndate):
        _, holds = sector_map.get(etf, (etf, []))
        cands = [h for h in holds if h in px.columns and _available_at(px[h], date)
                 and pd.notna(pb.loc[date, h]) and pb.loc[date, h] > 0 and not bool(trap.loc[date, h])]
        ld = [c for c in cands if bool(low_debt.loc[date, c])]
        use = ld or cands
        if not use:
            return None
        pick = pb.loc[date, use].idxmin()
        r = _ret_delist(px[pick], date, ndate)
        return float(r) if (r is not None and np.isfinite(r)) else None

    def select(i, mode):
        ranked, rl, ac = ranked_by[i], runlen_by[i], accel_by[i]
        top = ranked[:TOP_N]
        if mode == "baseline":
            return top
        if mode == "drop_stale5":
            return [e for e in top if rl.get(e, 0) < STALE]
        if mode == "drop_fading":
            return [e for e in top if ac.get(e, True)]
        if mode == "drop_stale_and_fading":
            return [e for e in top if not (rl.get(e, 0) >= STALE and not ac.get(e, True))]
        if mode == "fill_fresh":
            out = []
            for e in ranked:
                if rl.get(e, 0) < STALE:
                    out.append(e)
                if len(out) >= TOP_N:
                    break
            return out
        return top

    MODES = ["baseline", "drop_stale5", "drop_fading", "drop_stale_and_fading", "fill_fresh"]

    def run(mode):
        rets, spies, names, pk = [], [], [], []
        for i in range(warmup, len(midx) - 1):
            date, ndate = midx[i], midx[i + 1]
            sp = spy_m.iloc[i + 1] / spy_m.iloc[i] - 1
            if not np.isfinite(sp):
                continue
            sel = select(i, mode)
            slot = [pick_in(e, date, ndate) for e in sel]
            slot = [x for x in slot if x is not None]
            if slot:
                rets.append(float(np.mean(slot))); spies.append(float(sp)); names.append(len(slot)); pk += slot
        s = _stats(rets, spies)
        s["avg_names"] = round(float(np.mean(names)), 1) if names else 0
        s["pick_win_pct"] = round(float((np.array(pk) > 0).mean() * 100), 1) if pk else 0
        return s

    results = {m: run(m) for m in MODES}
    base = results["baseline"]
    print("\n=== FRESHNESS FILTER (avoid stale-tail sectors) ===", flush=True)
    for m in MODES:
        s = results[m]
        d = "" if m == "baseline" else f"  ({'+' if s['vs_spy']-base['vs_spy']>=0 else ''}{round(s['vs_spy']-base['vs_spy'],1)}pp)"
        print(f"  {m:22} vsSPY {s['vs_spy']:>7}%  Sh {s['sharpe']}  DD {s['max_drawdown']}%  "
              f"pick-win {s['pick_win_pct']}%  names {s['avg_names']}{d}", flush=True)

    best = max(MODES, key=lambda m: results[m]["vs_spy"])
    best_wr = max(MODES, key=lambda m: results[m]["pick_win_pct"])
    helped = results[best]["vs_spy"] > base["vs_spy"] + 5 or results[best_wr]["pick_win_pct"] > base["pick_win_pct"] + 1.5
    payload = {
        "computed_at": pd.Timestamp.utcnow().isoformat(),
        "params": {"top_n": TOP_N, "stale_months": STALE, "benchmark": BENCH, "months": int(len(midx))},
        "results": results, "best_return": best, "best_winrate": best_wr,
        "verdict": (f"Freshness filter HELPS: best return={best} ({results[best]['vs_spy']}% vs baseline {base['vs_spy']}%), "
                    f"best win-rate={best_wr} ({results[best_wr]['pick_win_pct']}% vs {base['pick_win_pct']}%). Avoiding "
                    "the stale/fading sector tail adds return and/or win rate." if helped else
                    "Freshness filter does NOT clearly help — the value pick already neutralizes stale-sector timing; "
                    "the per-pick freshness gap doesn't survive at the portfolio level (breadth loss offsets it)."),
        "caveat": "In-sample, no fees, ~5y, 12mo warmup shortens window (baseline vsSPY < the +229% headline). "
                  "Drop variants reduce names/month (breadth); fill_fresh keeps ~10.",
    }
    return payload


def main():
    p = build()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(p, indent=2, default=str))
    try:
        from core.models import BacktestResult
        from django.utils import timezone
        BacktestResult.objects.update_or_create(
            kind="freshness_filter", defaults={"payload": json.loads(json.dumps(p, default=str)),
                                               "computed_at": timezone.now()})
        print("Saved BacktestResult[freshness_filter]", flush=True)
    except Exception as e:
        print("DB save failed:", e, flush=True)
    print("\n" + p["verdict"], flush=True)


if __name__ == "__main__":
    main()
