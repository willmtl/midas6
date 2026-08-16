#!/usr/bin/env python3
"""GEO-TIER with FX CORRECTION — same deployed flagship (accel top-10 -> cheapest positive-P/B guard low-debt,
$5M floor, div_2x) across us / us_ca / full tiers, but now foreign local-currency candles are CONVERTED TO USD
(fx.py) so returns and the $-volume floor are honest. P/B is computed from LOCAL price (FX-invariant, already
correct); RETURNS and DOLLAR-VOLUME use the USD-converted price. Compares against the uncorrected geo_tier to
show how much of the international 'edge' was just un-FX-adjusted local-currency drift.
-> BacktestResult[geo_tier_fx]. Run: MSYS_NO_PATHCONV=1 docker exec rotation-backend-1 python -u /app/geo_tier_fx_study.py
"""
import os, re, json, warnings
warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
os.environ.setdefault("DJANGO_SETTINGS_MODULE", "rotation.settings")
import django
django.setup()

import config, sector_holdings, fx
from seq_fundamental_study import load_candles, load_financial_reports
from trend_stock_studies import _pit_monthly_panel, _available_at, _ret_delist, CRYPTO
from backtest_lowpb import _monthly_close, BENCH

TOP_N = 10
CONVICTION_MULT = 2.0
MIN_DVOL = 5e6
US_RE = re.compile(r"^[A-Z]{1,5}(-[A-Z])?$")


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
    # LOCAL monthly close -> P/B (FX-invariant); USD-converted daily -> returns + dollar-volume
    stock_m_local = _monthly_close(stock_daily).reindex(midx)
    currencies = fx.get_currencies(all_holds)
    usd_daily = fx.convert_candles_to_usd(stock_daily, currencies)
    stock_m_usd = _monthly_close(usd_daily).reindex(midx)

    reps = load_financial_reports(all_holds)
    sh, eq, ni, dt = (_pit_monthly_panel(reps, f, midx) for f in
                      ("shares_outstanding", "total_equity", "net_income", "total_debt"))
    common = stock_m_local.columns.intersection(stock_m_usd.columns).intersection(sh.columns).intersection(eq.columns)
    R = lambda p: p.reindex(index=midx, columns=common)
    px_local = stock_m_local[common]; px_usd = stock_m_usd[common]
    sh, eq, ni, dt = R(sh), R(eq), R(ni), R(dt)
    pb = (px_local * sh) / eq.where(eq != 0)          # LOCAL price -> correct FX-invariant P/B
    trap = (ni < 0) & (~(eq >= eq.shift(12))) & (~(ni > ni.shift(4)))
    low = (dt / eq.where(eq != 0)) < 1.0
    adl_m, dvol = {}, {}
    for t in common:
        d = usd_daily.get(t)                          # USD candles -> USD dollar-volume + A/D
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
    ad_slope3 = adl - adl.shift(3); px_ret3 = px_usd.pct_change(3)   # USD price ret for A/D divergence
    is_us_col = {t: bool(US_RE.match(t)) for t in common}
    n_for = sum(1 for t in common if not is_us_col[t])
    print(f"months {len(midx)} | stocks {len(common)} | foreign {n_for}", flush=True)

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
                c = [h for h in holds if tier_ok(h, tier) and h in px_local.columns and _available_at(px_usd[h], date)
                     and pd.notna(pb.loc[date, h]) and pb.loc[date, h] > 0 and not bool(trap.loc[date, h])
                     and pd.notna(dvol.loc[date, h]) and dvol.loc[date, h] >= MIN_DVOL]
                g = [x for x in c if bool(low.loc[date, x])] or c
                if not g:
                    continue
                pick = min(g, key=lambda h: pb.loc[date, h])
                r = _ret_delist(px_usd[pick], date, ndate)         # USD return
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
        avg_dom = round(float(np.mean([p["ret"] for p in allp if not p["foreign"]])) * 100, 2) if allp else 0.0
        avg_for = round(float(np.mean([p["ret"] for p in allp if p["foreign"]])) * 100, 2) if nf else None
        return {"perf": perf, "per_year": per_year, "n_pick_months": len(allp),
                "n_foreign_picks": nf, "foreign_share_pct": round(nf / len(allp) * 100, 1) if allp else 0,
                "avg_domestic_ret": avg_dom, "avg_foreign_ret_USD": avg_for}

    out = {}
    for tier in ("us", "us_ca", "full"):
        out[tier] = run(tier)
        r = out[tier]; p = r["perf"]
        print(f"\n=== {tier.upper():6} (FX-USD) | total {p['total']}% vs SPY {p['spy_total']}% | Sh {p['sharpe']} DD {p['dd']}% | foreign {r['n_foreign_picks']}/{r['n_pick_months']} ({r['foreign_share_pct']}%) ===", flush=True)
        print(f"  avg pick USD: dom {r['avg_domestic_ret']}% / foreign {r['avg_foreign_ret_USD']}% (was +4.44% uncorrected)", flush=True)
        print(f"  per-year: " + "  ".join(f"{y}:{d['port']:+.0f}/{d['spy']:+.0f}" for y, d in r["per_year"].items()), flush=True)

    us, ca, fu = out["us"]["perf"], out["us_ca"]["perf"], out["full"]["perf"]
    verdict = (
        f"FX-CORRECTED: US {us['total']}%(Sh{us['sharpe']}) | US+CA {ca['total']}%(Sh{ca['sharpe']}) | FULL {fu['total']}%(Sh{fu['sharpe']}). "
        f"Foreign picks now avg {out['full']['avg_foreign_ret_USD']}%/mo in USD vs domestic {out['full']['avg_domestic_ret']}% "
        f"(was +4.44% uncorrected). Compare to uncorrected geo_tier full +491.7%.")
    return {"computed_at": pd.Timestamp.utcnow().isoformat(),
            "params": {"top_n": TOP_N, "benchmark": BENCH, "conviction_mult": CONVICTION_MULT, "min_dvol": MIN_DVOL,
                       "months": len(midx), "fx_corrected": True},
            "results": out, "verdict": verdict,
            "caveat": "Foreign candles converted to USD via yfinance USD{CUR}=X (GBp/ZAc/ILA handled). P/B from local "
                      "price (FX-invariant); returns + $vol floor in USD. PIT, no fees, present-day-holdings survivorship."}


def main():
    p = build()
    from pathlib import Path
    Path("/app/.data/studies/geo_tier_fx.json").write_text(json.dumps(p, indent=2, default=str))
    try:
        from core.models import BacktestResult
        from django.utils import timezone
        BacktestResult.objects.update_or_create(
            kind="geo_tier_fx", defaults={"payload": json.loads(json.dumps(p, default=str)), "computed_at": timezone.now()})
        print("Saved BacktestResult[geo_tier_fx]", flush=True)
    except Exception as e:
        print("DB save failed:", e, flush=True)
    print("\n" + p["verdict"], flush=True)


if __name__ == "__main__":
    main()
