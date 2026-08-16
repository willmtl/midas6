#!/usr/bin/env python3
"""TOP-5 vs TOP-10 CONCENTRATION — answer: is the top-5 flagship's extra return driven by 1-2 monster picks, or
is the average pick just better? And are the picks LIQUID? Same validated engine as rotation_history_scan.py
(momentum-ACCEL top-N sectors -> cheapest positive-P/B guarded low-debt pick, $5M dvol floor, div_2x A/D weight),
run for TOP_N in {5,10}. For each config we record EVERY pick-month's realized return AND its 20d dollar-volume
(liquidity), then:
  (A) headline: total/annual/Sharpe/DD + per-CALENDAR-YEAR returns (lumpiness).
  (B) per-pick distribution: mean/median/std of monthly pick returns, %>+25% 'monster' months, best 15 winners.
  (C) drop-the-best decomposition: recompute cumulative return after REMOVING the k best pick-months (renormalize
      that month's basket). If total collapses toward the benchmark when k is small, the edge is concentration.
  (D) liquidity: $-volume distribution of picks (min/p10/median), smallest-10 picks by $vol, count under $10M/$25M.
-> BacktestResult[top5_concentration]. Run: MSYS_NO_PATHCONV=1 docker exec rotation-backend-1 python -u /app/top5_concentration_study.py
"""
import os, json, warnings
warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
os.environ.setdefault("DJANGO_SETTINGS_MODULE", "rotation.settings")
import django
django.setup()

import config, sector_holdings
import price_basis
from seq_fundamental_study import load_candles, load_financial_reports
from trend_stock_studies import _pit_monthly_panel, _available_at, _ret_delist, CRYPTO
from backtest_lowpb import _monthly_close, BENCH

CONVICTION_MULT = 2.0
MIN_DVOL = 5e6


def _perf(port_rets, spy_rets):
    r = np.asarray(port_rets, float)
    tot = float(np.prod(1 + r) - 1) * 100
    sp = float(np.prod(1 + np.asarray(spy_rets)) - 1) * 100
    n = len(r)
    ann = (float(np.prod(1 + r)) ** (12.0 / n) - 1) * 100 if n else 0.0
    sh = float(r.mean() / r.std() * np.sqrt(12)) if r.std() > 1e-9 else 0.0
    eqc = np.cumprod(1 + r); dd = float(((eqc / np.maximum.accumulate(eqc)) - 1).min() * 100)
    return dict(total=round(tot, 1), vs_spy=round(tot - sp, 1), annual=round(ann, 1),
                sharpe=round(sh, 2), dd=round(dd, 1), months=n, spy_total=round(sp, 1))


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
    adl_m, dvol = {}, {}
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

    def run(top_n):
        monthly = []          # per month: {date, picks:[{name,ret,weight,dvol,accum}], port_ret, spy_ret}
        for i in range(9, len(midx) - 1):
            date, ndate = midx[i], midx[i + 1]
            sp = spy_m.iloc[i + 1] / spy_m.iloc[i] - 1
            if not np.isfinite(sp):
                continue
            top = accel.loc[date].dropna().sort_values(ascending=False).head(top_n).index
            adsl, pr3 = ad_slope3.loc[date], px_ret3.loc[date]
            picks = []
            for etf in top:
                _, holds = sector_map.get(etf, (etf, []))
                c = [h for h in holds if h in px.columns and _available_at(px[h], date) and pd.notna(pb.loc[date, h])
                     and pb.loc[date, h] > 0 and not bool(trap.loc[date, h])
                     and pd.notna(dvol.loc[date, h]) and dvol.loc[date, h] >= MIN_DVOL]
                g = [x for x in c if bool(low.loc[date, x])] or c
                if not g:
                    continue
                pick = min(g, key=lambda h: pb.loc[date, h])
                r = _ret_delist(px[pick], date, ndate)
                if r is None or not np.isfinite(r):
                    continue
                is_div = bool(pd.notna(adsl.get(pick)) and pd.notna(pr3.get(pick))
                              and adsl.get(pick) > 0 and pr3.get(pick) < 0)
                w = CONVICTION_MULT if is_div else 1.0
                picks.append({"name": pick, "date": date.strftime("%Y-%m-%d"), "ret": float(r),
                              "weight": w, "dvol": float(dvol.loc[date, pick]), "accum": is_div})
            if not picks:
                continue
            tw = sum(p["weight"] for p in picks)
            port = sum(p["weight"] * p["ret"] for p in picks) / tw
            monthly.append({"date": date, "picks": picks, "port_ret": port, "spy_ret": float(sp)})
        return monthly

    def cum_after_dropping(monthly, drop_names_dates):
        """Recompute compounded total (%) after removing specific (name,date) pick-months (renormalize month)."""
        eq = 1.0
        for m in monthly:
            keep = [p for p in m["picks"] if (p["name"], p["date"]) not in drop_names_dates]
            if not keep:
                continue                       # month with nothing left -> flat (cash), skip compounding
            tw = sum(p["weight"] for p in keep)
            pr = sum(p["weight"] * p["ret"] for p in keep) / tw
            eq *= (1 + pr)
        return round((eq - 1) * 100, 1)

    out = {}
    for top_n in (5, 10):
        monthly = run(top_n)
        port_rets = [m["port_ret"] for m in monthly]
        spy_rets = [m["spy_ret"] for m in monthly]
        perf = _perf(port_rets, spy_rets)

        # per-calendar-year (lumpiness)
        yr = {}
        for m in monthly:
            y = m["date"].year; yr.setdefault(y, [1.0, 1.0])
            yr[y][0] *= (1 + m["port_ret"]); yr[y][1] *= (1 + m["spy_ret"])
        per_year = {int(y): {"port": round((v[0] - 1) * 100, 1), "spy": round((v[1] - 1) * 100, 1)}
                    for y, v in sorted(yr.items())}

        # per-pick distribution
        all_picks = [p for m in monthly for p in m["picks"]]
        rets = np.array([p["ret"] for p in all_picks]) * 100
        winners = sorted(all_picks, key=lambda p: p["ret"], reverse=True)[:15]
        pick_dist = {
            "n_pick_months": len(all_picks), "mean_ret": round(float(rets.mean()), 2),
            "median_ret": round(float(np.median(rets)), 2), "std_ret": round(float(rets.std()), 2),
            "pct_monster_gt25": round(float((rets > 25).mean()) * 100, 1),
            "pct_gt50": round(float((rets > 50).mean()) * 100, 1),
            "best15": [{"name": p["name"], "month": p["date"][:7], "ret_pct": round(p["ret"] * 100, 1),
                        "dvol_m": round(p["dvol"] / 1e6, 1), "accum": p["accum"]} for p in winners],
        }

        # drop-the-best-k decomposition (by single pick-month realized return)
        ranked = sorted(all_picks, key=lambda p: p["weight"] * p["ret"], reverse=True)
        drop = {}
        for k in (0, 1, 2, 3, 5, 10):
            ds = set((p["name"], p["date"]) for p in ranked[:k])
            drop[k] = cum_after_dropping(monthly, ds)

        # liquidity
        dv = np.array([p["dvol"] for p in all_picks]) / 1e6   # $M/day
        smalls = sorted(all_picks, key=lambda p: p["dvol"])[:10]
        liq = {
            "min_dvol_m": round(float(dv.min()), 1), "p10_dvol_m": round(float(np.percentile(dv, 10)), 1),
            "median_dvol_m": round(float(np.median(dv)), 1), "p90_dvol_m": round(float(np.percentile(dv, 90)), 1),
            "n_under_10m": int((dv < 10).sum()), "n_under_25m": int((dv < 25).sum()), "n_total": len(dv),
            "smallest10": [{"name": p["name"], "month": p["date"][:7], "dvol_m": round(p["dvol"] / 1e6, 2),
                            "ret_pct": round(p["ret"] * 100, 1)} for p in smalls],
        }
        out[f"top{top_n}"] = {"perf": perf, "per_year": per_year, "pick_dist": pick_dist,
                              "drop_best": drop, "liquidity": liq}
        print(f"\n=== TOP-{top_n} div_2x: total {perf['total']}% vs SPY {perf['spy_total']}% | Sh {perf['sharpe']} DD {perf['dd']}% | {perf['months']}mo ===", flush=True)
        print(f"  per-year: " + "  ".join(f"{y}:{d['port']:+.0f}/{d['spy']:+.0f}" for y, d in per_year.items()), flush=True)
        print(f"  pick mean {pick_dist['mean_ret']}% median {pick_dist['median_ret']}% std {pick_dist['std_ret']}% | monster>+25%: {pick_dist['pct_monster_gt25']}%", flush=True)
        print(f"  DROP-BEST total%: k=0 {drop[0]} | k=1 {drop[1]} | k=2 {drop[2]} | k=3 {drop[3]} | k=5 {drop[5]} | k=10 {drop[10]}", flush=True)
        print(f"  liquidity $M/day: min {liq['min_dvol_m']} p10 {liq['p10_dvol_m']} median {liq['median_dvol_m']} | under$10M {liq['n_under_10m']}/{liq['n_total']} under$25M {liq['n_under_25m']}", flush=True)
        print(f"  best picks: " + ", ".join(f"{w['name']} {w['date'][:7]} {w['ret']*100:+.0f}%(${w['dvol']/1e6:.0f}M)" for w in winners[:6]), flush=True)

    t5, t10 = out["top5"]["perf"], out["top10"]["perf"]
    d5 = out["top5"]["drop_best"]
    # how much of top5's ADVANTAGE over top10 survives removing its 2 best months?
    adv_full = t5["total"] - t10["total"]
    adv_drop2 = d5[2] - t10["total"]
    verdict = (
        f"TOP-5 {t5['total']}% (Sh{t5['sharpe']},DD{t5['dd']}%) vs TOP-10 {t10['total']}% (Sh{t10['sharpe']},DD{t10['dd']}%). "
        f"Top-5 mean pick {out['top5']['pick_dist']['mean_ret']}% vs top-10 {out['top10']['pick_dist']['mean_ret']}%. "
        f"Drop top-5's 2 best pick-months: {t5['total']}%->{d5[2]}% (advantage over top10 {adv_full:+.0f}pp -> {adv_drop2:+.0f}pp). "
        f"Liquidity: {out['top5']['liquidity']['n_under_10m']}/{out['top5']['liquidity']['n_total']} picks under $10M/day, "
        f"median ${out['top5']['liquidity']['median_dvol_m']}M/day. " + (
        "CONCENTRATION-DRIVEN: a couple of monster months carry top-5's extra return." if adv_drop2 < adv_full * 0.4 else
        "BROAD: top-5's edge is a higher AVERAGE pick, not 1-2 winners."))
    return {"computed_at": pd.Timestamp.utcnow().isoformat(),
            "params": {"benchmark": BENCH, "conviction_mult": CONVICTION_MULT, "min_dvol": MIN_DVOL,
                       "months": len(midx)},
            "results": out, "verdict": verdict,
            "caveat": "PIT, no fees, stock-universe survivorship (today's holdings). Month-end rebalance, hold 1mo. In-sample."}


def main():
    p = build()
    from pathlib import Path
    Path("/app/.data/studies/top5_concentration.json").write_text(json.dumps(p, indent=2, default=str))
    try:
        from core.models import BacktestResult
        from django.utils import timezone
        BacktestResult.objects.update_or_create(
            kind="top5_concentration", defaults={"payload": json.loads(json.dumps(p, default=str)), "computed_at": timezone.now()})
        print("Saved BacktestResult[top5_concentration]", flush=True)
    except Exception as e:
        print("DB save failed:", e, flush=True)
    print("\n" + p["verdict"], flush=True)


if __name__ == "__main__":
    main()
