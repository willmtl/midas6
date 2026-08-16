#!/usr/bin/env python3
"""DIVIDEND BASIS OF THE SIGNAL — candles are auto_adjust=True (total-return: split+dividend adjusted), so the
momentum-acceleration RANKING partly rewards a sector's dividend YIELD, not just price. High-yield sleeves
(Utilities, REITs/Mortgage-REITs, Staples, Dividend, bonds TLT/AGG/HYG/TIP, MLPs) get a steady total-return
tilt that could nudge them up the ranking vs a pure price-momentum read. Test whether excluding dividends from
the SIGNAL changes selection / performance. Returns stay TOTAL-RETURN (you DO earn dividends -- price-only
returns would just be mechanically lower and less honest); only the ranking basis changes.

price-only (dividend-excluded, split-included) ETF close = yfinance auto_adjust=False raw Close / future-split
factor (same split back-adjust price_basis uses). total-return close = yfinance Adj Close. Both from ONE pull so
the ONLY difference between arms is dividend inclusion. Stock picks + TR stock returns come from the DB, identical
across arms -- this isolates the ranking-basis effect.
-> BacktestResult[dividend_basis] + JSON.
Run: MSYS_NO_PATHCONV=1 docker exec rotation-backend-1 python -u /app/dividend_basis_study.py
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
OUT = Path(__file__).resolve().parent / ".data" / "studies" / "dividend_basis.json"
HIYIELD_HINT = {"XLU", "XLP", "XLRE", "REM", "SCHD", "AMLP", "TLT", "AGG", "HYG", "TIP", "VTV"}


def _perf(r, spy):
    r = np.asarray(r, float); n = len(r)
    tot = float(np.prod(1 + r) - 1) * 100
    sp = float(np.prod(1 + np.asarray(spy)) - 1) * 100
    sh = float(r.mean() / r.std() * np.sqrt(12)) if r.std() > 1e-9 else 0.0
    eqc = np.cumprod(1 + r); dd = float(((eqc / np.maximum.accumulate(eqc)) - 1).min() * 100)
    t = _tstat_from_returns(list(r))
    return dict(total=round(tot, 1), vs_spy=round(tot - sp, 1), sharpe=round(sh, 2), dd=round(dd, 1),
                t_stat=round(t, 2) if t is not None else None, months=n)


def fetch_etf_price_bases(etf_tk):
    """Return (tr_monthly, pr_monthly) DataFrames [month-end x etf] from ONE yfinance pull.
    tr = Adj Close (dividend+split adjusted). pr = raw Close split-adjusted only (dividends excluded)."""
    import yfinance as yf
    splits = price_basis.refresh_splits(etf_tk)
    raw = yf.download(etf_tk, period="6y", auto_adjust=False, progress=False, group_by="ticker", threads=True)
    tr, pr = {}, {}
    for t in etf_tk:
        try:
            sub = raw[t]
        except Exception:
            continue
        if sub is None or "Close" not in sub or "Adj Close" not in sub:
            continue
        adj = sub["Adj Close"].dropna()
        rawc = sub["Close"].dropna()
        if adj.empty or rawc.empty:
            continue
        tr[t] = adj
        # split-adjust the raw close: divide by product of split ratios STRICTLY AFTER each date
        sm = splits.get(t, {})
        if sm:
            items = sorted((pd.Timestamp(d), float(r)) for d, r in sm.items())
            fac = pd.Series(1.0, index=rawc.index)
            for d, r in items:
                fac[rawc.index > d] *= r
            pr[t] = rawc / fac
        else:
            pr[t] = rawc
    tr_df = pd.DataFrame(tr); pr_df = pd.DataFrame(pr)
    tr_m = tr_df.resample("ME").last()
    pr_m = pr_df.resample("ME").last()
    return tr_m, pr_m


def build():
    etfs = {n: e for n, e in config.SECTOR_ETFS.items() if e not in CRYPTO}
    sector_map, all_holds, name_of = {}, set(), {}
    for n, e in etfs.items():
        h = [t for t in sector_holdings.get_holdings(n) if t not in (e, BENCH) and t not in CRYPTO]
        sector_map[e] = (n, set(h)); all_holds.update(h); name_of[e] = n
    all_holds = sorted(all_holds)
    etf_tk = list(etfs.values())

    # ETF price bases (yfinance, one pull) — TR and price-only
    print(f"Fetching TR + price-only closes for {len(etf_tk)} ETFs (yfinance)...", flush=True)
    tr_m, pr_m = fetch_etf_price_bases(etf_tk)
    # common monthly index = the intersection, aligned to month-end
    midx = tr_m.index.intersection(pr_m.index)
    tr_m = tr_m.reindex(midx); pr_m = pr_m.reindex(midx)
    keep = [e for e in etf_tk if e in tr_m.columns and e in pr_m.columns]
    tr_m, pr_m = tr_m[keep], pr_m[keep]
    print(f"  months {len(midx)}  ({midx[0].date()}..{midx[-1].date()})  ETFs {len(keep)}", flush=True)

    accel_tr = tr_m.pct_change(3) - tr_m.pct_change(3).shift(3)
    accel_pr = pr_m.pct_change(3) - pr_m.pct_change(3).shift(3)

    # SPY monthly TR for the benchmark (from the pull if present, else DB)
    if BENCH in tr_m.columns:
        spy_m = tr_m[BENCH]
    else:
        spy_daily = load_candles([BENCH])[BENCH]["Close"]
        spy_m = spy_daily.resample("ME").last().reindex(midx)

    # DB stock panels (TOTAL-RETURN returns; identical across arms) on the SAME month index
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

    def wt(name, date):
        a, p = ad_slope3.loc[date].get(name), px_ret3.loc[date].get(name)
        return CONV if (pd.notna(a) and pd.notna(p) and a > 0 and p < 0) else 1.0

    def run(accel):
        rets, spies, picks_by_month = [], [], {}
        for i in range(9, len(midx) - 1):
            date, ndate = midx[i], midx[i + 1]
            if not (pd.notna(spy_m.iloc[i]) and pd.notna(spy_m.iloc[i + 1])):
                continue
            sp = spy_m.iloc[i + 1] / spy_m.iloc[i] - 1
            if not np.isfinite(sp):
                continue
            top = accel.loc[date].dropna().sort_values(ascending=False).head(TOP_N).index
            held = set(); wsum = rr = 0.0; picked = []
            for etf in top:
                p = pick(etf, date, held)
                if not p:
                    continue
                held.add(p); picked.append((etf, p))
                r = _ret_delist(px[p], date, ndate)
                if r is None or not np.isfinite(r):
                    continue
                w = wt(p, date); wsum += w; rr += w * float(r)
            if wsum <= 0:
                continue
            rets.append(rr / wsum); spies.append(float(sp)); picks_by_month[str(date.date())] = picked
        return _perf(rets, spies), picks_by_month

    perf_tr, picks_tr = run(accel_tr)
    perf_pr, picks_pr = run(accel_pr)

    # selection overlap + which sectors the price-only ranking pushed up/down
    months = sorted(set(picks_tr) & set(picks_pr))
    same = tot = 0
    for m in months:
        s1 = {e for e, _ in picks_tr[m]}; s2 = {e for e, _ in picks_pr[m]}
        same += len(s1 & s2); tot += len(s1 | s2)
    overlap = round(100 * same / tot, 1) if tot else None
    # avg rank shift (TR rank - PR rank); +ve => price-only ranks it WORSE (dividend was helping it)
    rank_tr = accel_tr.rank(axis=1, ascending=False)
    rank_pr = accel_pr.rank(axis=1, ascending=False)
    shift = (rank_pr - rank_tr).mean().dropna().sort_values()
    top_up = shift.head(6)     # price-only ranks BETTER (was penalised by... none; low yield)
    top_dn = shift.tail(6)     # price-only ranks WORSE => dividend was lifting it

    print("\n=== SIGNAL BASIS: total-return vs price-only (returns TR in both) ===", flush=True)
    for lbl, p in (("rank on TOTAL-RETURN (baseline)", perf_tr), ("rank on PRICE-ONLY", perf_pr)):
        print(f"  {lbl:<32} {p['total']:>7}%  vsSPY {p['vs_spy']:>7}  Sh {p['sharpe']:>5}  DD {p['dd']:>6}%  t {p['t_stat']}",
              flush=True)
    print(f"\n  selection overlap (same sectors in top-10): {overlap}%", flush=True)
    print(f"\n  sectors most LIFTED by dividends in the signal (price-only ranks them worse, +avg rank):", flush=True)
    for e, s in top_dn.items():
        tag = " [high-yield]" if e in HIYIELD_HINT else ""
        print(f"    {name_of.get(e, e):<26} ({e})  +{s:.1f} ranks worse without divs{tag}", flush=True)

    d_tot = perf_pr["total"] - perf_tr["total"]
    verdict = (
        f"Rank on total-return {perf_tr['total']}%/Sh{perf_tr['sharpe']}/DD{perf_tr['dd']}% vs rank on price-only "
        f"{perf_pr['total']}%/Sh{perf_pr['sharpe']}/DD{perf_pr['dd']}% ({d_tot:+.1f}pp). Selection overlap {overlap}%. "
        + ("Excluding dividends from the signal barely changes anything -- the acceleration read is dominated by "
           "price, dividend yield is second-order over a 3-month window, so the current TR basis is fine."
           if abs(d_tot) < 15 and (overlap or 0) > 85 else
           "Excluding dividends from the signal MATERIALLY changes selection/return -- the momentum ranking was "
           "partly a yield tilt; worth switching the signal to price-only while keeping TR returns.")
    )
    print("\n" + verdict, flush=True)
    return {
        "computed_at": pd.Timestamp.utcnow().isoformat(),
        "params": {"top_n": TOP_N, "min_dvol": MIN_DVOL, "conviction_mult": CONV, "benchmark": BENCH,
                   "months": int(len(midx)), "returns_basis": "total-return (both arms)",
                   "signal_bases": ["total-return", "price-only(split-adj, div-excluded)"]},
        "perf_rank_total_return": perf_tr, "perf_rank_price_only": perf_pr,
        "selection_overlap_pct": overlap, "delta_total_pp": round(d_tot, 1),
        "sectors_lifted_by_dividends": {name_of.get(e, e): round(float(s), 2) for e, s in top_dn.items()},
        "verdict": verdict,
        "caveat": "ETF price bases from a fresh yfinance pull (6y, auto_adjust=False); TR=Adj Close, price-only=raw "
                  "Close split-adjusted via cached splits. Stock picks + returns from DB (total-return), identical "
                  "across arms, so this isolates the SIGNAL basis only. Not a price-only-RETURNS test (that would "
                  "just subtract the picks' yield and is less honest for performance). PIT, no fees, survivorship.",
    }


def main():
    p = build()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(p, indent=2, default=str))
    try:
        from core.models import BacktestResult
        from django.utils import timezone
        BacktestResult.objects.update_or_create(
            kind="dividend_basis", defaults={"payload": json.loads(json.dumps(p, default=str)),
                                             "computed_at": timezone.now()})
        print("Saved BacktestResult[dividend_basis]", flush=True)
    except Exception as e:
        print("DB save failed:", e, flush=True)


if __name__ == "__main__":
    main()
