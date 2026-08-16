#!/usr/bin/env python3
"""FLOW ACCELERATION — apply the flagship's acceleration operator DIRECTLY to ETF fund flow. The flagship ranks
sectors by PRICE accel = ret3 - ret3.shift(3). Earlier we tested flow LEVEL/momentum (flow_rank) — it just
echoed price. This tests flow ACCELERATION: is net share creation (inflow) growing at an INCREASING rate?
  flow_accel = shares_out.pct_change(3) - shares_out.pct_change(3).shift(3)   (shares_out = cumulative creation)
If accelerating inflow LEADS price, ranking by it (or combining with price-accel) beats price alone.

Arms (same engine: top-10 -> cheapest as-traded-P/B guard low-debt $5M div_2x, monthly; ALL on the SAME window
where flow_accel is defined, for a fair return comparison):
  price_accel      baseline (flagship ranking)
  flow_accel       rank sectors by flow acceleration
  combo_rank       rank by SUM of price-accel rank + flow-accel rank (both must be strong)
  price_x_flowpos  price-accel top-10 among sectors with flow_accel > 0 (flow confirms)
  flow_x_pricepos  flow-accel top-10 among sectors with price_accel > 0
Report LEADING WITH TOTAL RETURN ([[return-priority]]). -> BacktestResult[flow_accel] + JSON.
Run: MSYS_NO_PATHCONV=1 docker exec rotation-backend-1 python -u /app/flow_accel_study.py
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
OUT = Path(__file__).resolve().parent / ".data" / "studies" / "flow_accel.json"


def load_flow_shares(tickers, midx):
    from core.models import ETFFlow
    rows = ETFFlow.objects.filter(ticker__in=list(tickers)).values("ticker", "date", "shares_out")
    if not rows:
        return pd.DataFrame(index=midx)
    df = pd.DataFrame(list(rows)); df["date"] = pd.to_datetime(df["date"])
    piv = df.pivot_table(index="date", columns="ticker", values="shares_out", aggfunc="last").sort_index()
    return piv.resample("ME").last().reindex(midx).ffill(limit=2)


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
    price_accel = etf_m.pct_change(3) - etf_m.pct_change(3).shift(3)
    spy_m = etf_daily[BENCH]["Close"].resample("ME").last().reindex(midx)

    shares = load_flow_shares(etf_tk, midx)
    shares = shares.reindex(columns=etf_m.columns)
    flow_accel = shares.pct_change(3) - shares.pct_change(3).shift(3)
    # first month with enough flow_accel sectors -> common comparison window
    start_i = next((i for i in range(len(midx)) if flow_accel.iloc[i].notna().sum() >= TOP_N), 9)
    start_i = max(start_i, 9)
    rank_p = price_accel.rank(axis=1, ascending=False)
    rank_f = flow_accel.rank(axis=1, ascending=False)
    combo = rank_p + rank_f

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

    def pick(etf, date, held):
        _, holds = sector_map.get(etf, (etf, set()))
        c = [h for h in holds if h in px.columns and h not in held and _available_at(px[h], date)
             and pd.notna(pb.loc[date, h]) and pb.loc[date, h] > 0 and not bool(trap.loc[date, h])
             and pd.notna(dvol.loc[date, h]) and dvol.loc[date, h] >= MIN_DVOL]
        g = [x for x in c if bool(low.loc[date, x])] or c
        return min(g, key=lambda h: pb.loc[date, h]) if g else None

    def accumulating(name, date):
        a, p = ad_slope3.loc[date].get(name), px_ret3.loc[date].get(name)
        return pd.notna(a) and pd.notna(p) and a > 0 and p < 0

    def ordered(date, mode):
        if mode == "price_accel":
            s = price_accel.loc[date].dropna(); return list(s.sort_values(ascending=False).index)
        if mode == "flow_accel":
            s = flow_accel.loc[date].dropna(); return list(s.sort_values(ascending=False).index)
        if mode == "combo_rank":
            s = combo.loc[date].dropna(); return list(s.sort_values().index)          # lowest sum = best
        if mode == "price_x_flowpos":
            fpos = set(flow_accel.loc[date][flow_accel.loc[date] > 0].index)
            s = price_accel.loc[date].dropna(); return [e for e in s.sort_values(ascending=False).index if e in fpos]
        if mode == "flow_x_pricepos":
            ppos = set(price_accel.loc[date][price_accel.loc[date] > 0].index)
            s = flow_accel.loc[date].dropna(); return [e for e in s.sort_values(ascending=False).index if e in ppos]
        return []

    def run(mode):
        rets, spies = [], []
        for i in range(start_i, len(midx) - 1):
            date, ndate = midx[i], midx[i + 1]
            sp = spy_m.iloc[i + 1] / spy_m.iloc[i] - 1
            if not np.isfinite(sp):
                continue
            order = ordered(date, mode)[:TOP_N]
            if not order:
                continue
            held = set(); wsum = rr = 0.0
            for etf in order:
                p = pick(etf, date, held)
                if not p:
                    continue
                held.add(p)
                r = _ret_delist(px[p], date, ndate)
                if r is None or not np.isfinite(r):
                    continue
                w = CONV if accumulating(p, date) else 1.0
                wsum += w; rr += w * float(r)
            if wsum <= 0:
                continue
            rets.append(rr / wsum); spies.append(float(sp))
        return _perf(rets, spies)

    modes = ["price_accel", "flow_accel", "combo_rank", "price_x_flowpos", "flow_x_pricepos"]
    results = {m: run(m) for m in modes}
    order = sorted(results, key=lambda k: results[k]["total"], reverse=True)
    print(f"\n=== FLOW ACCELERATION vs PRICE ACCEL (common window from {midx[start_i].date()}; TOTAL RETURN) ===", flush=True)
    print(f"  {'mode':<16}{'total':>9}{'annual':>8}{'vsSPY':>8}{'Sharpe':>8}{'DD':>8}{'t':>6}{'mo':>5}", flush=True)
    for k in order:
        r = results[k]
        star = "  <= price baseline" if k == "price_accel" else ("  <= BEST" if k == order[0] else "")
        print(f"  {k:<16}{r['total']:>8}%{r['annual']:>7}%{r['vs_spy']:>8}{r['sharpe']:>8}{r['dd']:>7}%"
              f"{str(r['t_stat']):>6}{r['months']:>5}{star}", flush=True)
    best = order[0]; b = results[best]; base = results["price_accel"]
    verdict = (
        f"Price-accel baseline {base['total']}% ({base['annual']}%/yr) over the flow window. "
        f"flow_accel alone {results['flow_accel']['total']}%. BEST = {best}: {b['total']}% ({b['annual']}%/yr) "
        f"= {b['total'] - base['total']:+.0f}pp. "
        + ("Flow acceleration ADDS return over price-accel alone — accelerating inflow leads price." if best != "price_accel" and b["total"] > base["total"] + 10
           else "Flow acceleration does NOT beat price-accel — it doesn't lead price at the monthly sector level.")
    )
    print("\n" + verdict, flush=True)
    return {"computed_at": pd.Timestamp.utcnow().isoformat(),
            "params": {"top_n": TOP_N, "conv": CONV, "window_start": str(midx[start_i].date()),
                       "months": int(base["months"]), "objective": "MAX TOTAL RETURN",
                       "flow_accel": "shares_out.pct_change(3) - .shift(3) [accel of cumulative creation]"},
            "results": results, "best": best, "verdict": verdict,
            "caveat": "Flow (shares_out) monthly from ETFFlow (2021-08+), so flow_accel starts ~2022; ALL arms run on "
                      "the SAME window for fair comparison (shorter than full flagship). Monthly month-end shares, "
                      "ffill<=2. PIT/no-fees/survivorship as base, div_2x, $5M vol."}


def main():
    p = build()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(p, indent=2, default=str))
    try:
        from core.models import BacktestResult
        from django.utils import timezone
        BacktestResult.objects.update_or_create(
            kind="flow_accel", defaults={"payload": json.loads(json.dumps(p, default=str)), "computed_at": timezone.now()})
        print("Saved BacktestResult[flow_accel]", flush=True)
    except Exception as e:
        print("DB save failed:", e, flush=True)


if __name__ == "__main__":
    main()
