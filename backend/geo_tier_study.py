#!/usr/bin/env python3
"""GEO-TIER comparison — the backtest universe has 286/1174 (24%) FOREIGN-listed names whose yfinance prices &
volumes are in LOCAL CURRENCY: the $5M "dollar"-vol floor is not currency-normalized and returns are NOT
FX-adjusted to a USD investor (a Japanese name +20% in JPY while JPY fell 15% is only ~+2% in USD, but the
backtest books +20%). This inflates the headline (foreign picks avg +4.44%/mo vs domestic +2.61%). Run the
DEPLOYED flagship (accel top-10 -> cheapest positive-P/B guard low-debt, $5M floor, div_2x) over 3 universe
tiers and compare, to size how much of the edge rides on the currency-corrupted names:
  us     = US-listed only  (^[A-Z]{1,5}(-[A-Z])? — keeps US ADRs BABA/NIO/VALE in USD; excludes .KS/.T/.L, ITUB4, 00939)
  us_ca  = us + Canada (.TO/.V)   [CAD still ~local-currency; minor distortion, included per request]
  full   = everything (current, corrupted)
-> BacktestResult[geo_tier]. Run: MSYS_NO_PATHCONV=1 docker exec rotation-backend-1 python -u /app/geo_tier_study.py
"""
import os, re, json, warnings
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

TOP_N = 10
CONVICTION_MULT = 2.0
MIN_DVOL = 5e6
US_RE = re.compile(r"^[A-Z]{1,5}(-[A-Z])?$")   # US-listed common/ADR (BRK-B ok); excludes dotted + digit-bearing


def tier_ok(t, tier):
    us = bool(US_RE.match(t))
    if tier == "us":
        return us
    if tier == "us_ca":
        return us or t.endswith(".TO") or t.endswith(".V")
    return True


def _perf(port_rets, spy_rets):
    r = np.asarray(port_rets, float); n = len(r)
    tot = float(np.prod(1 + r) - 1) * 100
    sp = float(np.prod(1 + np.asarray(spy_rets)) - 1) * 100
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
    is_us_col = {t: bool(US_RE.match(t)) for t in common}
    n_for = sum(1 for t in common if not is_us_col[t])
    print(f"months {len(midx)} | stocks {len(common)} | foreign {n_for} ({round(n_for/len(common)*100,1)}%)", flush=True)

    def run(tier):
        monthly = []
        for i in range(9, len(midx) - 1):
            date, ndate = midx[i], midx[i + 1]
            sp = spy_m.iloc[i + 1] / spy_m.iloc[i] - 1
            if not np.isfinite(sp):
                continue
            top = accel.loc[date].dropna().sort_values(ascending=False).head(TOP_N).index
            adsl, pr3 = ad_slope3.loc[date], px_ret3.loc[date]
            picks = []
            for etf in top:
                _, holds = sector_map.get(etf, (etf, []))
                c = [h for h in holds if tier_ok(h, tier) and h in px.columns and _available_at(px[h], date)
                     and pd.notna(pb.loc[date, h]) and pb.loc[date, h] > 0 and not bool(trap.loc[date, h])
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
                picks.append({"name": pick, "date": date, "ret": float(r), "weight": w,
                              "foreign": not is_us_col.get(pick, True)})
            if not picks:
                continue
            tw = sum(p["weight"] for p in picks)
            port = sum(p["weight"] * p["ret"] for p in picks) / tw
            monthly.append({"date": date, "picks": picks, "port_ret": port, "spy_ret": float(sp)})
        port_rets = [m["port_ret"] for m in monthly]; spy_rets = [m["spy_ret"] for m in monthly]
        perf = _perf(port_rets, spy_rets)
        yr = {}
        for m in monthly:
            y = m["date"].year; yr.setdefault(y, [1.0, 1.0])
            yr[y][0] *= (1 + m["port_ret"]); yr[y][1] *= (1 + m["spy_ret"])
        per_year = {int(y): {"port": round((v[0] - 1) * 100, 1), "spy": round((v[1] - 1) * 100, 1)}
                    for y, v in sorted(yr.items())}
        allp = [p for m in monthly for p in m["picks"]]
        nf = sum(1 for p in allp if p["foreign"])
        avg_pick = round(float(np.mean([p["ret"] for p in allp])) * 100, 2) if allp else 0.0
        avg_dom = round(float(np.mean([p["ret"] for p in allp if not p["foreign"]])) * 100, 2) if allp else 0.0
        avg_for = round(float(np.mean([p["ret"] for p in allp if p["foreign"]])) * 100, 2) if nf else None
        return {"perf": perf, "per_year": per_year, "n_pick_months": len(allp),
                "n_foreign_picks": nf, "foreign_share_pct": round(nf / len(allp) * 100, 1) if allp else 0,
                "avg_pick_ret": avg_pick, "avg_domestic_ret": avg_dom, "avg_foreign_ret": avg_for,
                "avg_picks_per_month": round(len(allp) / len(monthly), 2) if monthly else 0}

    out = {}
    for tier in ("us", "us_ca", "full"):
        out[tier] = run(tier)
        r = out[tier]; p = r["perf"]
        print(f"\n=== {tier.upper():6} | total {p['total']}% vs SPY {p['spy_total']}% | Sh {p['sharpe']} DD {p['dd']}% | {p['months']}mo | picks/mo {r['avg_picks_per_month']} | foreign {r['n_foreign_picks']}/{r['n_pick_months']} ({r['foreign_share_pct']}%) ===", flush=True)
        print(f"  avg pick {r['avg_pick_ret']}% (dom {r['avg_domestic_ret']}% / for {r['avg_foreign_ret']}%)", flush=True)
        print(f"  per-year: " + "  ".join(f"{y}:{d['port']:+.0f}/{d['spy']:+.0f}" for y, d in r["per_year"].items()), flush=True)

    us, ca, fu = out["us"]["perf"], out["us_ca"]["perf"], out["full"]["perf"]
    verdict = (
        f"US-only {us['total']}% (Sh{us['sharpe']},DD{us['dd']}%) | US+CA {ca['total']}% (Sh{ca['sharpe']},DD{ca['dd']}%) | "
        f"FULL {fu['total']}% (Sh{fu['sharpe']},DD{fu['dd']}%). Full has {out['full']['foreign_share_pct']}% foreign picks "
        f"(avg {out['full']['avg_foreign_ret']}%/mo vs domestic {out['full']['avg_domestic_ret']}%). "
        f"Foreign/CAD names add {fu['total']-us['total']:+.0f}pp of (currency-uncorrected) headline return over US-only.")
    return {"computed_at": pd.Timestamp.utcnow().isoformat(),
            "params": {"top_n": TOP_N, "benchmark": BENCH, "conviction_mult": CONVICTION_MULT, "min_dvol": MIN_DVOL,
                       "months": len(midx), "foreign_in_universe": n_for},
            "results": out, "verdict": verdict,
            "caveat": "Foreign prices/volumes in LOCAL currency (not FX-adjusted); us_ca CAD also local. US tier = clean "
                      "USD (incl US ADRs). PIT, no fees, present-day-holdings survivorship. Month-end rebalance, hold 1mo."}


def main():
    p = build()
    from pathlib import Path
    Path("/app/.data/studies/geo_tier.json").write_text(json.dumps(p, indent=2, default=str))
    try:
        from core.models import BacktestResult
        from django.utils import timezone
        BacktestResult.objects.update_or_create(
            kind="geo_tier", defaults={"payload": json.loads(json.dumps(p, default=str)), "computed_at": timezone.now()})
        print("Saved BacktestResult[geo_tier]", flush=True)
    except Exception as e:
        print("DB save failed:", e, flush=True)
    print("\n" + p["verdict"], flush=True)


if __name__ == "__main__":
    main()
