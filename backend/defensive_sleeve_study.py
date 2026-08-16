#!/usr/bin/env python3
"""DEFENSIVE SLEEVE — the flagship ranks ALL sectors by momentum-acceleration but can only HOLD a sector if it
has an eligible stock (cheapest-P/B pick). Bond ETFs (TLT/AGG/HYG/TIP) and pure-commodity ETFs (GLD/SLV/USO/...)
have ZERO stock holdings, so when they lead the ranking (classic risk-off / flight-to-safety) the engine SKIPS
them and hands the slot to the next equity sector -> the book is structurally ~100% equities, no defensive
sleeve. User: "for the one that has no stock, just BUY THE ETF (if possible)".

Test exactly that. Same flagship (accel top-10 -> cheapest as-traded-P/B guard low-debt $5M div_2x, monthly),
but when a top-10 sector has no eligible STOCK, hold the ETF ITSELF (its own monthly return, weight 1.0):
  baseline           skip no-stock sectors (deployed behaviour).
  bonds_sleeve       hold the ETF for no-stock BOND sectors only (TLT/AGG/HYG/TIP).
  defensive_sleeve   hold the ETF for ANY no-stock sector (bonds + pure commodities: GLD/SLV/USO/UNG/...).
Watch DRAWDOWN + Sharpe (expect the sleeve to help in 2022 when treasuries/commodities led and equities fell).
-> BacktestResult[defensive_sleeve] + JSON.
Run: MSYS_NO_PATHCONV=1 docker exec rotation-backend-1 python -u /app/defensive_sleeve_study.py
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

TOP_N = 10; CONV = 2.0; MIN_DVOL = 5e6
BONDS = {"TLT", "AGG", "HYG", "TIP"}
OUT = Path(__file__).resolve().parent / ".data" / "studies" / "defensive_sleeve.json"


def _perf(r, spy):
    r = np.asarray(r, float); n = len(r)
    tot = float(np.prod(1 + r) - 1) * 100
    sp = float(np.prod(1 + np.asarray(spy)) - 1) * 100
    sh = float(r.mean() / r.std() * np.sqrt(12)) if r.std() > 1e-9 else 0.0
    eqc = np.cumprod(1 + r); dd = float(((eqc / np.maximum.accumulate(eqc)) - 1).min() * 100)
    t = _tstat_from_returns(list(r))
    return dict(total=round(tot, 1), vs_spy=round(tot - sp, 1), sharpe=round(sh, 2), dd=round(dd, 1),
                t_stat=round(t, 2) if t is not None else None, months=n)


def build():
    etfs = {n: e for n, e in config.SECTOR_ETFS.items() if e not in CRYPTO}
    sector_map, all_holds = {}, set()
    for n, e in etfs.items():
        h = [t for t in sector_holdings.get_holdings(n) if t not in (e, BENCH) and t not in CRYPTO]
        sector_map[e] = (n, set(h)); all_holds.update(h)
    all_holds = sorted(all_holds)
    etf_tk = list(etfs.values())
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

    # which sectors STRUCTURALLY have no eligible stock (holdings never intersect the fundamental universe)
    nostock = {e for e in etf_tk if not (sector_map[e][1] & set(common))}
    print(f"no-stock sectors ({len(nostock)}): "
          f"bonds={sorted(nostock & BONDS)}  other={sorted(nostock - BONDS)}", flush=True)

    def pick(etf, date, held):
        _, holds = sector_map.get(etf, (etf, set()))
        c = [h for h in holds if h in px.columns and h not in held and _available_at(px[h], date)
             and pd.notna(pb.loc[date, h]) and pb.loc[date, h] > 0 and not bool(trap.loc[date, h])
             and pd.notna(dvol.loc[date, h]) and dvol.loc[date, h] >= MIN_DVOL]
        g = [x for x in c if bool(low.loc[date, x])] or c
        return min(g, key=lambda h: pb.loc[date, h]) if g else None

    def wt(name, date):
        a, p = ad_slope3.loc[date].get(name), px_ret3.loc[date].get(name)
        return CONV if (pd.notna(a) and pd.notna(p) and a > 0 and p < 0) else 1.0

    def run(etf_hold_set):
        rets, spies = [], []
        etf_holds = 0
        for i in range(9, len(midx) - 1):
            date, ndate = midx[i], midx[i + 1]
            sp = spy_m.iloc[i + 1] / spy_m.iloc[i] - 1
            if not np.isfinite(sp):
                continue
            top = accel.loc[date].dropna().sort_values(ascending=False).head(TOP_N).index
            held = set(); wsum = rr = 0.0
            for etf in top:
                p = pick(etf, date, held)
                if p is not None:
                    held.add(p)
                    r = _ret_delist(px[p], date, ndate)
                    if r is None or not np.isfinite(r):
                        continue
                    w = wt(p, date); wsum += w; rr += w * float(r)
                elif etf in etf_hold_set:
                    # hold the ETF itself (its own month return)
                    e0, e1 = etf_m.loc[date, etf], etf_m.loc[ndate, etf]
                    if pd.notna(e0) and pd.notna(e1) and e0 > 0:
                        r = e1 / e0 - 1
                        wsum += 1.0; rr += 1.0 * float(r); etf_holds += 1
            if wsum <= 0:
                continue
            rets.append(rr / wsum); spies.append(float(sp))
        perf = _perf(rets, spies); perf["etf_holdings"] = etf_holds
        return perf

    results = {
        "baseline": run(set()),
        "bonds_sleeve": run(nostock & BONDS),
        "defensive_sleeve": run(nostock),
    }
    print(f"\n=== DEFENSIVE SLEEVE — hold the ETF when a top-{TOP_N} sector has no stock (total/vsSPY/Sh/DD) ===", flush=True)
    for k, r in results.items():
        print(f"  {k:<18} {r['total']:>7}%  vsSPY {r['vs_spy']:>7}  Sh {r['sharpe']:>5}  DD {r['dd']:>6}%  "
              f"t {r['t_stat']}  (ETF-holds {r['etf_holdings']})", flush=True)

    base = results["baseline"]
    best = max(("bonds_sleeve", "defensive_sleeve"), key=lambda k: results[k]["sharpe"])
    b = results[best]
    verdict = (
        f"Baseline {base['total']}%/Sh{base['sharpe']}/DD{base['dd']}%. "
        f"bonds_sleeve {results['bonds_sleeve']['total']}%/Sh{results['bonds_sleeve']['sharpe']}/"
        f"DD{results['bonds_sleeve']['dd']}%; defensive_sleeve {results['defensive_sleeve']['total']}%/"
        f"Sh{results['defensive_sleeve']['sharpe']}/DD{results['defensive_sleeve']['dd']}%. "
        + ("Holding the ETF when a defensive sector leads IMPROVES risk-adjusted return / cuts drawdown -> wire the "
           "sleeve (and it justifies adding more govt-bond ETFs: SHY/IEF/GOVT/TLH)."
           if b["sharpe"] > base["sharpe"] + 0.03 or b["dd"] > base["dd"] + 2 else
           "Holding the ETF does NOT improve Sharpe/DD here -> the defensive sectors didn't lead often enough, or "
           "when they did the equity slot they replaced did just as well. (Only {} ETF-holds occurred.)".format(
               results["defensive_sleeve"]["etf_holdings"]))
    )
    print("\n" + verdict, flush=True)
    return {
        "computed_at": pd.Timestamp.utcnow().isoformat(),
        "params": {"top_n": TOP_N, "min_dvol": MIN_DVOL, "benchmark": BENCH, "months": int(len(midx)),
                   "bonds": sorted(BONDS), "nostock_sectors": sorted(nostock)},
        "results": results, "verdict": verdict,
        "caveat": "ETF held as its own position (weight 1.0, no div_2x) when a top-10 sector has no eligible stock. "
                  "Uses ONLY existing universe ETFs — adding SHY/IEF/GOVT/TLH/VGIT/VGLT needs a candle backfill. "
                  "PIT, no fees, present-day-holdings survivorship. Bond ETFs are total-return via candle adj close.",
    }


def main():
    p = build()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(p, indent=2, default=str))
    try:
        from core.models import BacktestResult
        from django.utils import timezone
        BacktestResult.objects.update_or_create(
            kind="defensive_sleeve", defaults={"payload": json.loads(json.dumps(p, default=str)),
                                               "computed_at": timezone.now()})
        print("Saved BacktestResult[defensive_sleeve]", flush=True)
    except Exception as e:
        print("DB save failed:", e, flush=True)


if __name__ == "__main__":
    main()
