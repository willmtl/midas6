#!/usr/bin/env python3
"""WALK-FORWARD the freshness variants — does each one's edge PERSIST out-of-sample, or is it carried by
one lucky (concentrated) stretch? Runs every freshness selection variant, then slices its monthly return
series into first/second half, thirds, per-year, reporting vs-SPY + win rate per slice. A robust edge
beats SPY in BOTH halves and most years; a concentration mirage shows a huge half and a weak/negative one.
Settles drop_fading (+319% concentrated, 5 names) vs drop_stale_and_fading (+214%, Sharpe 1.45, 8 names).
-> BacktestResult[freshness_walkforward] + JSON.
Run: MSYS_NO_PATHCONV=1 docker exec rotation-backend-1 python -u /app/freshness_walkforward.py
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

OUT = Path(__file__).resolve().parent / ".data" / "studies" / "freshness_walkforward.json"
TOP_N, STALE = 10, 5
MODES = ["baseline", "drop_stale5", "drop_fading", "drop_stale_and_fading"]


def _seg(rets, spies):
    r = np.asarray(rets, float); s = np.asarray(spies, float); n = len(r)
    if n == 0:
        return dict(vs_spy=0, strat=0, t_stat=None, win_pct=0, months=0)
    strat = float(np.prod(1 + r) - 1) * 100
    spy = float(np.prod(1 + s) - 1) * 100
    t = _tstat_from_returns(list(r - s))
    return dict(vs_spy=round(strat - spy, 1), strat=round(strat, 1),
                t_stat=round(t, 2) if t is not None else None,
                win_pct=round((r > s).mean() * 100, 1), months=n)


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

    run_len = {e: 0 for e in etf_tk}
    ranked_by, runlen_by, accel_by = {}, {}, {}
    for i in range(len(midx)):
        date = midx[i]
        ranked = list(mom6.loc[date].dropna().sort_values(ascending=False).index)
        top10 = set(ranked[:TOP_N])
        for e in etf_tk:
            run_len[e] = run_len[e] + 1 if e in top10 else 0
        ranked_by[i] = ranked; runlen_by[i] = dict(run_len)
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
        r = _ret_delist(px[pb.loc[date, use].idxmin()], date, ndate)
        return float(r) if (r is not None and np.isfinite(r)) else None

    def select(i, mode):
        ranked, rl, ac = ranked_by[i], runlen_by[i], accel_by[i]
        top = ranked[:TOP_N]
        if mode == "drop_stale5":
            return [e for e in top if rl.get(e, 0) < STALE]
        if mode == "drop_fading":
            return [e for e in top if ac.get(e, True)]
        if mode == "drop_stale_and_fading":
            return [e for e in top if not (rl.get(e, 0) >= STALE and not ac.get(e, True))]
        return top

    def series(mode):
        rets, spies, dates = [], [], []
        for i in range(warmup, len(midx) - 1):
            date, ndate = midx[i], midx[i + 1]
            sp = spy_m.iloc[i + 1] / spy_m.iloc[i] - 1
            if not np.isfinite(sp):
                continue
            slot = [pick_in(e, date, ndate) for e in select(i, mode)]
            slot = [x for x in slot if x is not None]
            if slot:
                rets.append(float(np.mean(slot))); spies.append(float(sp)); dates.append(ndate)
        return np.array(rets), np.array(spies), pd.to_datetime(dates)

    results = {}
    for mode in MODES:
        r, s, d = series(mode)
        n = len(r); h = n // 2
        full = _seg(r, s)
        halves = {"first": _seg(r[:h], s[:h]), "second": _seg(r[h:], s[h:])}
        years = {}
        for y in sorted(set(d.year)):
            m = d.year == y
            if m.sum() >= 3:
                years[str(y)] = _seg(r[m], s[m])
        both_pos = halves["first"]["vs_spy"] > 0 and halves["second"]["vs_spy"] > 0
        yb = sum(1 for v in years.values() if v["vs_spy"] > 0)
        # OOS decay: how much of the full edge shows in the WEAKER half (robustness)
        weak = min(halves["first"]["vs_spy"], halves["second"]["vs_spy"])
        results[mode] = {"full": full, "halves": halves, "years": years,
                         "both_halves_positive": bool(both_pos), "years_beat": f"{yb}/{len(years)}",
                         "weaker_half_vs_spy": weak, "robust": bool(both_pos and yb >= 0.6 * len(years))}

    print("\n=== FRESHNESS WALK-FORWARD (which edge is REAL out-of-sample) ===", flush=True)
    print(f"  {'variant':22} {'FULL':>7} {'1st-half':>9} {'2nd-half':>9} {'weakHalf':>9}  {'yrs':>5} robust", flush=True)
    for m in MODES:
        R = results[m]
        print(f"  {m:22} {R['full']['vs_spy']:>6}% {R['halves']['first']['vs_spy']:>8}% "
              f"{R['halves']['second']['vs_spy']:>8}% {R['weaker_half_vs_spy']:>8}%  {R['years_beat']:>5} "
              f"{'YES' if R['robust'] else 'no'}", flush=True)
    print("\n  per-year vs SPY:", flush=True)
    for m in MODES:
        ys = results[m]["years"]
        print(f"    {m:22} " + "  ".join(f"{y}:{v['vs_spy']:+.0f}" for y, v in ys.items()), flush=True)

    # verdict: compare drop_fading vs surgical on weaker-half robustness
    df, sf = results["drop_fading"], results["drop_stale_and_fading"]
    payload = {
        "computed_at": pd.Timestamp.utcnow().isoformat(),
        "params": {"top_n": TOP_N, "stale_months": STALE, "benchmark": BENCH, "months": int(len(midx))},
        "results": results,
        "verdict": (f"drop_fading full +{df['full']['vs_spy']}% but weaker-half +{df['weaker_half_vs_spy']}% "
                    f"(both halves+ {df['both_halves_positive']}); drop_stale_and_fading full +{sf['full']['vs_spy']}% "
                    f"weaker-half +{sf['weaker_half_vs_spy']}% (both+ {sf['both_halves_positive']}). The one with the "
                    "stronger WEAKER-half and both-halves-positive is the real edge; a big full number with a weak/"
                    "negative half = concentration mirage."),
        "caveat": "In-sample split (no true holdout); rules-based so this is subperiod stability. 12mo warmup "
                  "shortens window. drop_fading runs ~5 names -> higher variance -> noisier subperiod numbers.",
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
            kind="freshness_walkforward", defaults={"payload": json.loads(json.dumps(p, default=str)),
                                                    "computed_at": timezone.now()})
        print("Saved BacktestResult[freshness_walkforward]", flush=True)
    except Exception as e:
        print("DB save failed:", e, flush=True)
    print("\n" + p["verdict"], flush=True)


if __name__ == "__main__":
    main()
