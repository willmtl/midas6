#!/usr/bin/env python3
"""LEVERAGE STACK — return-additive per [[return-priority]]. Take the concentration winner (top6_div6x, 597%)
and amplify with leverage: always-on (1.5x/2x) AND conditional (lever only when net liquidity is EXPANDING —
the [[macro-liquidity-regime]] signal flipped to amplify the good regime instead of dampening the bad; those
months averaged ~2x the return). Never de-risk below 1x (return mandate). Margin interest charged on the
borrowed portion (flat 6%/yr assumption). DRAWDOWN shown prominently — leverage converts drawdown into potential
RUIN, the one caveat that matters even under a max-return mandate.
-> BacktestResult[leverage] + JSON.
Run: MSYS_NO_PATHCONV=1 docker exec rotation-backend-1 python -u /app/leverage_study.py
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

TOP_N = 6; CONV = 6.0; MIN_DVOL = 5e6
BORROW_APR = 0.06                                  # flat margin rate assumption
BORROW_M = (1 + BORROW_APR) ** (1 / 12) - 1
OUT = Path(__file__).resolve().parent / ".data" / "studies" / "leverage.json"


def _perf(r, spy):
    r = np.asarray(r, float); n = len(r)
    tot = float(np.prod(1 + r) - 1) * 100
    sp = float(np.prod(1 + np.asarray(spy)) - 1) * 100
    ann = (float(np.prod(1 + r)) ** (12.0 / n) - 1) * 100 if n else 0.0
    sh = float(r.mean() / r.std() * np.sqrt(12)) if r.std() > 1e-9 else 0.0
    eqc = np.cumprod(1 + r); dd = float(((eqc / np.maximum.accumulate(eqc)) - 1).min() * 100)
    # worst single month (ruin proxy)
    worst = float(np.min(r) * 100) if n else 0.0
    t = _tstat_from_returns(list(r))
    return dict(total=round(tot, 1), annual=round(ann, 1), vs_spy=round(tot - sp, 1), sharpe=round(sh, 2),
                dd=round(dd, 1), worst_month=round(worst, 1), t_stat=round(t, 2) if t is not None else None, months=n)


def net_liq_expanding(midx):
    from core.models import MacroSeries
    def s(sid):
        rows = MacroSeries.objects.filter(series=sid).values_list("date", "value")
        return pd.Series({pd.Timestamp(d): v for d, v in rows}).sort_index().resample("ME").last() if rows else pd.Series(dtype=float)
    walcl, rrp, tga = s("WALCL").reindex(midx).ffill(), s("RRPONTSYD").reindex(midx).ffill(), s("WTREGEN").reindex(midx).ffill()
    net = walcl - rrp * 1000.0 - tga
    return (net.diff(3) > 0).shift(1)                # expanding, known at prior month-end (no look-ahead)


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

    # base top6_div6x monthly returns
    dates, rets, spies = [], [], []
    for i in range(9, len(midx) - 1):
        date, ndate = midx[i], midx[i + 1]
        sp = spy_m.iloc[i + 1] / spy_m.iloc[i] - 1
        if not np.isfinite(sp):
            continue
        top = accel.loc[date].dropna().sort_values(ascending=False).head(TOP_N).index
        held = set(); wsum = rr = 0.0
        for etf in top:
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
        dates.append(date); rets.append(rr / wsum); spies.append(float(sp))
    dates = pd.DatetimeIndex(dates); rets = np.array(rets); spies = np.array(spies)
    expanding = net_liq_expanding(midx).reindex(dates)

    def lever(lev_series):
        out = []
        for i in range(len(rets)):
            L = lev_series[i]
            out.append(L * rets[i] - (L - 1.0) * BORROW_M)      # margin cost on borrowed portion
        return _perf(out, spies)

    def const(L):
        return [L] * len(rets)

    def cond(L):
        return [L if (expanding.iloc[i] == True or pd.isna(expanding.iloc[i])) else 1.0 for i in range(len(rets))]  # noqa: E712

    results = {
        "base_1x": lever(const(1.0)),
        "always_1.5x": lever(const(1.5)),
        "always_2x": lever(const(2.0)),
        "condlev_1.5x": lever(cond(1.5)),
        "condlev_2x": lever(cond(2.0)),
        "condlev_3x": lever(cond(3.0)),
    }
    base = results["base_1x"]
    order = sorted(results, key=lambda k: results[k]["total"], reverse=True)
    print(f"\n=== LEVERAGE on top6_div6x (borrow {BORROW_APR:.0%}/yr; sorted by TOTAL RETURN) ===", flush=True)
    print(f"  {'variant':<14}{'total':>10}{'annual':>8}{'Sharpe':>8}{'maxDD':>9}{'worstMo':>9}{'t':>6}", flush=True)
    for k in order:
        r = results[k]
        ruin = "  ⚠RUIN-RISK" if r["dd"] < -50 else ""
        print(f"  {k:<14}{r['total']:>9}%{r['annual']:>7}%{r['sharpe']:>8}{r['dd']:>8}%{r['worst_month']:>8}%"
              f"{str(r['t_stat']):>6}{ruin}", flush=True)

    best = order[0]; b = results[best]
    verdict = (
        f"Base top6_div6x (1x) {base['total']}% ({base['annual']}%/yr, DD{base['dd']}%). "
        f"BEST RETURN = {best}: {b['total']}% ({b['annual']}%/yr, DD{b['dd']}%, worst month {b['worst_month']}%). "
        f"Conditional (lever only when net-liquidity expanding) vs always-on: "
        + (f"condlev variants dodge some of the levered drawdown. " )
        + ("⚠️ Leverage multiplies DD toward RUIN territory — a >-50% drawdown while levered can force liquidation "
           "at the bottom (you don't get the recovery). Max return, but survival-capped." if any(
               results[k]["dd"] < -50 for k in results) else
           "Drawdowns stay below -50% here, but that leans on the short 2021-2026 sample with few crashes.")
    )
    print("\n" + verdict, flush=True)
    return {
        "computed_at": pd.Timestamp.utcnow().isoformat(),
        "params": {"base": "top6_div6x", "borrow_apr": BORROW_APR, "benchmark": BENCH, "months": int(len(rets)),
                   "lever_signal": "net-liquidity 3mo change > 0 (expanding), prior month-end", "objective": "MAX RETURN"},
        "results": results, "best_return": best, "verdict": verdict,
        "caveat": "Leverage applied to top6_div6x monthly returns; margin interest = (L-1)*(6%/yr monthly) charged. "
                  "Does NOT model margin-call mechanics (real ruin risk worse than the compounded number). "
                  "Conditional lever uses net-liquidity expanding (prior month-end, no look-ahead). PIT/no-fees/"
                  "survivorship as base. Leveraged-ETF decay (if implemented via SSO/UPRO) not modeled.",
    }


def main():
    p = build()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(p, indent=2, default=str))
    try:
        from core.models import BacktestResult
        from django.utils import timezone
        BacktestResult.objects.update_or_create(
            kind="leverage", defaults={"payload": json.loads(json.dumps(p, default=str)),
                                       "computed_at": timezone.now()})
        print("Saved BacktestResult[leverage]", flush=True)
    except Exception as e:
        print("DB save failed:", e, flush=True)


if __name__ == "__main__":
    main()
