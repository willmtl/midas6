#!/usr/bin/env python3
"""AMPLIFIER TILT — from prior work, micro/small-cap, cheap-PE and hypergrowth names carry fatter return tails.
The flagship gates only on $5M dollar-volume (NO market-cap filter), so these tilts are untested. On the SAME
engine (accel top-10, cheapest-P/B guard low-debt, div_2x, monthly), give EXTRA conviction weight (2x) to the
sector pick when it has an amplifier trait, and renormalize. Return-additive test per [[return-priority]].
  microcap    mktcap < $2B            smallcap    mktcap < $500M
  cheap_pe    0 < P/E < 15            hypergrowth revenue YoY > 40%
  any_amp     any of the above
Also a SELECTION variant: within-sector pick = cheapest-P/B among SMALL-CAP (<$2B) names (else fall back).
Report LEADING WITH TOTAL RETURN. -> BacktestResult[amplifier] + JSON.
Run: MSYS_NO_PATHCONV=1 docker exec rotation-backend-1 python -u /app/amplifier_study.py
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

TOP_N = 10; CONV = 2.0; AMP = 2.0; MIN_DVOL = 5e6
OUT = Path(__file__).resolve().parent / ".data" / "studies" / "amplifier.json"


def _perf(r, spy):
    r = np.asarray(r, float); n = len(r)
    tot = float(np.prod(1 + r) - 1) * 100
    sp = float(np.prod(1 + np.asarray(spy)) - 1) * 100
    ann = (float(np.prod(1 + r)) ** (12.0 / n) - 1) * 100 if n else 0.0
    sh = float(r.mean() / r.std() * np.sqrt(12)) if r.std() > 1e-9 else 0.0
    eqc = np.cumprod(1 + r); dd = float(((eqc / np.maximum.accumulate(eqc)) - 1).min() * 100)
    t = _tstat_from_returns(list(r))
    return dict(total=round(tot, 1), annual=round(ann, 1), vs_spy=round(tot - sp, 1), sharpe=round(sh, 2),
                dd=round(dd, 1), t_stat=round(t, 2) if t is not None else None, months=n)


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
    P = {f: _pit_monthly_panel(reps, f, midx) for f in
         ("shares_outstanding", "total_equity", "net_income", "total_debt", "eps_diluted", "revenue")}
    common = stock_m.columns.intersection(P["shares_outstanding"].columns).intersection(P["total_equity"].columns)
    R = lambda p: p.reindex(index=midx, columns=common)
    px = stock_m[common]; P = {f: R(P[f]) for f in P}
    as_traded = price_basis.as_traded_close(px)
    mktcap = as_traded * P["shares_outstanding"]
    pb = mktcap / P["total_equity"].where(P["total_equity"] != 0)
    pe = as_traded / P["eps_diluted"].where(P["eps_diluted"] > 0)
    rev_yoy = P["revenue"].pct_change(4)
    trap = (P["net_income"] < 0) & (~(P["total_equity"] >= P["total_equity"].shift(12))) & (~(P["net_income"] > P["net_income"].shift(4)))
    low = (P["total_debt"] / P["total_equity"].where(P["total_equity"] != 0)) < 1.0
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

    def eligible(etf, date, held):
        _, holds = sector_map.get(etf, (etf, set()))
        c = [h for h in holds if h in px.columns and h not in held and _available_at(px[h], date)
             and pd.notna(pb.loc[date, h]) and pb.loc[date, h] > 0 and not bool(trap.loc[date, h])
             and pd.notna(dvol.loc[date, h]) and dvol.loc[date, h] >= MIN_DVOL]
        return [x for x in c if bool(low.loc[date, x])] or c

    def trait(name, date, kind):
        mc = mktcap.loc[date, name]; p = pe.loc[date, name]; rg = rev_yoy.loc[date, name]
        if kind == "microcap":
            return pd.notna(mc) and mc < 2e9
        if kind == "smallcap":
            return pd.notna(mc) and mc < 5e8
        if kind == "cheap_pe":
            return pd.notna(p) and 0 < p < 15
        if kind == "hypergrowth":
            return pd.notna(rg) and rg > 0.40
        if kind == "any":
            return any(trait(name, date, k) for k in ("microcap", "cheap_pe", "hypergrowth"))
        return False

    def accumulating(name, date):
        a, p = ad_slope3.loc[date].get(name), px_ret3.loc[date].get(name)
        return pd.notna(a) and pd.notna(p) and a > 0 and p < 0

    def run(mode):
        rets, spies = [], []
        for i in range(9, len(midx) - 1):
            date, ndate = midx[i], midx[i + 1]
            sp = spy_m.iloc[i + 1] / spy_m.iloc[i] - 1
            if not np.isfinite(sp):
                continue
            top = accel.loc[date].dropna().sort_values(ascending=False).head(TOP_N).index
            held = set(); wsum = rr = 0.0
            for etf in top:
                cands = eligible(etf, date, held)
                if not cands:
                    continue
                if mode == "select_smallcap":
                    sc = [h for h in cands if trait(h, date, "microcap")] or cands
                    p = min(sc, key=lambda h: pb.loc[date, h])
                else:
                    p = min(cands, key=lambda h: pb.loc[date, h])
                held.add(p)
                r = _ret_delist(px[p], date, ndate)
                if r is None or not np.isfinite(r):
                    continue
                w = CONV if accumulating(p, date) else 1.0
                if mode.startswith("tilt_") and trait(p, date, mode[5:]):
                    w *= AMP
                wsum += w; rr += w * float(r)
            if wsum <= 0:
                continue
            rets.append(rr / wsum); spies.append(float(sp))
        return _perf(rets, spies)

    modes = ["baseline", "tilt_microcap", "tilt_cheap_pe", "tilt_hypergrowth", "tilt_any", "select_smallcap"]
    results = {m: run(m) for m in modes}
    order = sorted(results, key=lambda k: results[k]["total"], reverse=True)
    print(f"\n=== AMPLIFIER TILT (top10 div2x base; sorted by TOTAL RETURN) ===", flush=True)
    print(f"  {'mode':<16}{'total':>9}{'annual':>8}{'vsSPY':>8}{'Sharpe':>8}{'DD':>8}{'t':>6}", flush=True)
    for k in order:
        r = results[k]
        star = "  <= baseline" if k == "baseline" else ("  <= BEST" if k == order[0] else "")
        print(f"  {k:<16}{r['total']:>8}%{r['annual']:>7}%{r['vs_spy']:>8}{r['sharpe']:>8}{r['dd']:>7}%{str(r['t_stat']):>6}{star}", flush=True)
    best = order[0]; b = results[best]; base = results["baseline"]
    verdict = (f"Baseline {base['total']}% ({base['annual']}%/yr). BEST = {best}: {b['total']}% ({b['annual']}%/yr) "
               f"= {b['total'] - base['total']:+.0f}pp. "
               + ("An amplifier tilt adds return." if b["total"] > base["total"] + 10 and best != "baseline"
                  else "No amplifier tilt beats baseline — size/PE/growth tilts don't add return here."))
    print("\n" + verdict, flush=True)
    return {"computed_at": pd.Timestamp.utcnow().isoformat(),
            "params": {"top_n": TOP_N, "conv": CONV, "amp_mult": AMP, "months": int(base["months"]),
                       "objective": "MAX TOTAL RETURN"},
            "results": results, "best": best, "verdict": verdict,
            "caveat": "tilt_* = 2x extra weight when the cheapest-P/B pick has the trait; select_smallcap changes the "
                      "PICK to cheapest-P/B among <$2B names. mktcap = as-traded price x PIT shares. PIT, no fees, "
                      "survivorship as base (small/micro-cap survivorship bias is WORSE — dead microcaps absent)."}


def main():
    p = build()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(p, indent=2, default=str))
    try:
        from core.models import BacktestResult
        from django.utils import timezone
        BacktestResult.objects.update_or_create(
            kind="amplifier", defaults={"payload": json.loads(json.dumps(p, default=str)), "computed_at": timezone.now()})
        print("Saved BacktestResult[amplifier]", flush=True)
    except Exception as e:
        print("DB save failed:", e, flush=True)


if __name__ == "__main__":
    main()
