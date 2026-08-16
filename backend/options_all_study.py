#!/usr/bin/env python3
"""BACKTEST ALL OPTIONS — every structure we've tested, consolidated as PORTFOLIOS (not just per-trade),
so total-return / Sharpe / drawdown are comparable to the stock strategy. On PERFECT plays (conviction>=4)
with options coverage, priced via Black-Scholes on real historical ATM IV.

Structures:
  naked_call_35     ATM call, 35 DTE, hold to expiry (theta victim)
  vert_exp_35       ATM/+10% call debit spread, 35 DTE, hold to expiry (lottery)
  vert_managed_45   ... 45 DTE, daily-repriced, TAKE PROFIT +50% / STOP -50%, else close at rebalance
  itm20_90          20% ITM call, 90 DTE, close at rebalance (stock replacement, ~4x lev)
  itm10_90          10% ITM, 90 DTE
  atm_90            ATM, 90 DTE (more lev, more theta)
  leaps_itm20_180   20% ITM, 180 DTE
Portfolio: each month deploy fraction F=0.25 of capital into premium split across that month's perfect
plays (rest in T-bills @ RF); compound -> total/Sharpe/DD. Compared to stock perfect-plays (100% invested)
and 1.5x margin. -> BacktestResult[options_all] + JSON.
Run: MSYS_NO_PATHCONV=1 docker exec rotation-backend-1 python -u /app/options_all_study.py
"""
import os, json, warnings
from math import log, sqrt, exp
from statistics import NormalDist
warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
os.environ.setdefault("DJANGO_SETTINGS_MODULE", "rotation.settings")
import django
django.setup()

from pathlib import Path
import config, sector_holdings
from studies import _tstat_from_returns
from seq_fundamental_study import load_candles, load_financial_reports
from trend_stock_studies import _pit_monthly_panel, _available_at, _ret_delist, CRYPTO
from backtest_lowpb import _monthly_close, BENCH
import price_basis
from core.models import OptionSnapshot

OUT = Path(__file__).resolve().parent / ".data" / "studies" / "options_all.json"
LOOKBACK, TOP_N, R_F, RF_M, F_DEPLOY = 6, 10, 0.04, 0.0033, 0.25
_N = NormalDist().cdf


def bs(S, K, T, sig):
    if T <= 0 or sig <= 0 or S <= 0:
        return max(S - K, 0.0)
    d1 = (log(S / K) + (R_F + sig * sig / 2) * T) / (sig * sqrt(T)); d2 = d1 - sig * sqrt(T)
    return S * _N(d1) - K * exp(-R_F * T) * _N(d2)


def vspread(S, K1, K2, T, iv):
    return bs(S, K1, T, iv) - bs(S, K2, T, iv)


def _rsi(close, n=10):
    d = close.diff(); up = d.clip(lower=0).rolling(n).mean(); dn = (-d.clip(upper=0)).rolling(n).mean()
    return (100 - 100 / (1 + up / dn.replace(0, np.nan)))


def _pstats(rets, spies):
    r = np.asarray(rets, float); n = len(r)
    if n == 0:
        return dict(total_return=0, vs_spy=0, sharpe=0, max_drawdown=0, periods=0)
    tot = float(np.prod(1 + r) - 1) * 100
    sp = float(np.prod(1 + np.asarray(spies)) - 1) * 100
    sh = float(r.mean() / r.std() * np.sqrt(12)) if r.std() > 1e-9 else 0.0
    eq = np.cumprod(1 + r); dd = float(((eq / np.maximum.accumulate(eq)) - 1).min() * 100)
    return dict(total_return=round(tot, 1), vs_spy=round(tot - sp, 1), sharpe=round(sh, 2),
                max_drawdown=round(dd, 1), periods=n)


def _tstats(x):
    a = np.array(x, float)
    return None if not len(a) else dict(mean_pct=round(a.mean() * 100, 1),
                                        median_pct=round(float(np.median(a)) * 100, 1),
                                        win_pct=round((a > 0).mean() * 100, 1))


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
    etf_trail = etf_m.pct_change(LOOKBACK)
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
    vlow_debt = (debt / equity.where(equity != 0)) < 0.5
    rsi = pd.DataFrame({t: _rsi(stock_daily[t]["Close"]).resample("ME").last().reindex(midx)
                        for t in common if t in stock_daily and len(stock_daily[t]) > 20}).reindex(index=midx, columns=common)

    ivm = {}
    for tk, d, iv in OptionSnapshot.objects.filter(ticker__in=list(common), atm_iv__isnull=False).values_list("ticker", "date", "atm_iv"):
        ivm.setdefault(tk, {})[pd.Timestamp(d)] = (iv / 100.0 if iv > 1.5 else iv)
    iv_ser = {tk: pd.Series(v).sort_index() for tk, v in ivm.items()}
    iv_daily = {tk: (iv_ser[tk].reindex(stock_daily[tk].index.union(iv_ser[tk].index)).sort_index().ffill()
                     if tk in stock_daily else iv_ser[tk]) for tk in iv_ser}

    def iv_at(tk, date):
        s = iv_ser.get(tk)
        if s is None:
            return None
        s2 = s[s.index <= date]
        return None if (not len(s2) or (date - s2.index[-1]).days > 10) else float(s2.iloc[-1])

    STRUCTS = ["naked_call_35", "vert_exp_35", "vert_managed_45", "itm20_90", "itm10_90", "atm_90", "leaps_itm20_180"]
    trades = {s: [] for s in STRUCTS}                 # per-trade returns (perfect plays)
    monthly = {s: {} for s in STRUCTS}                # month -> [returns] (perfect plays w/ coverage)
    stock_m_ret = {}                                  # month -> [stock returns] (perfect plays)
    print(f"months {len(midx)} | IV tickers {len(iv_ser)}", flush=True)
    warmup = max(LOOKBACK, 1)

    for i in range(warmup, len(midx) - 1):
        date, ndate = midx[i], midx[i + 1]
        if not np.isfinite(spy_m.iloc[i + 1] / spy_m.iloc[i] - 1):
            continue
        hold_days = (ndate - date).days
        ranks = list(etf_trail.loc[date].dropna().sort_values(ascending=False).head(TOP_N).index)
        top5 = set(ranks[:5]); picks = []
        for etf in ranks:
            _, holds = sector_map.get(etf, (etf, []))
            cands = [h for h in holds if h in px.columns and _available_at(px[h], date)
                     and pd.notna(pb.loc[date, h]) and pb.loc[date, h] > 0 and not bool(trap.loc[date, h])]
            ld = [c for c in cands if bool(low_debt.loc[date, c])]
            use = ld or cands
            if not use:
                continue
            pick = pb.loc[date, use].idxmin()
            r = _ret_delist(px[pick], date, ndate)
            if r is not None and np.isfinite(r):
                picks.append((etf, pick, float(r)))
        if not picks:
            continue
        pbs = sorted(pb.loc[date, [p for _, p, _ in picks]].values)
        cc = pbs[max(0, len(pbs) // 3 - 1)] if pbs else 0
        for etf, pick, ret in picks:
            sc = (int(pb.loc[date, pick] <= cc) + int(pd.notna(ni.loc[date, pick]) and ni.loc[date, pick] > 0)
                  + int(bool(vlow_debt.loc[date, pick])) + int(pd.notna(rsi.loc[date, pick]) and rsi.loc[date, pick] < 45)
                  + int(etf in top5))
            if sc < 4:
                continue
            iv0 = iv_at(pick, date)
            S0 = px.loc[date, pick]
            if iv0 is None or not (S0 and S0 > 0):
                continue
            iv1 = iv_at(pick, ndate) or iv0
            ST = S0 * (1 + ret)
            stock_m_ret.setdefault(date, []).append(ret)
            out = {}
            # naked call 35d, expiry intrinsic
            c = bs(S0, S0, 35 / 365.0, iv0)
            out["naked_call_35"] = (max(ST - S0, 0) - c) / c if c > 1e-6 else None
            # vertical 35d expiry
            d35 = vspread(S0, S0, S0 * 1.10, 35 / 365.0, iv0)
            out["vert_exp_35"] = (max(min(ST, S0 * 1.10) - S0, 0) - d35) / d35 if d35 > 1e-6 else None
            # vertical 45d managed TP50/SL50
            d45 = vspread(S0, S0, S0 * 1.10, 45 / 365.0, iv0)
            if d45 > 1e-6:
                exp45 = date + pd.Timedelta(days=45); ivd = iv_daily.get(pick); dd = stock_daily.get(pick)
                res = None
                if ivd is not None and dd is not None:
                    seg = dd["Close"]; hd = seg.index[(seg.index > date) & (seg.index <= min(exp45, ndate))]
                    last = d45
                    for d in hd:
                        Sd = float(seg.loc[d]); ivd_ = ivd.get(d); ivd_ = float(ivd_) if pd.notna(ivd_) else iv0
                        Td = max((exp45 - d).days, 0) / 365.0; val = vspread(Sd, S0, S0 * 1.10, Td, ivd_); last = val
                        if val >= 1.5 * d45:
                            res = 0.5; break
                        if val <= 0.5 * d45:
                            res = -0.5; break
                    if res is None:
                        res = (last - d45) / d45
                out["vert_managed_45"] = res
            else:
                out["vert_managed_45"] = None
            # ITM / LEAPS held to rebalance
            for lab, mny, dte in (("itm20_90", 0.80, 90), ("itm10_90", 0.90, 90), ("atm_90", 1.00, 90), ("leaps_itm20_180", 0.80, 180)):
                prem = bs(S0, S0 * mny, dte / 365.0, iv0)
                out[lab] = (bs(ST, S0 * mny, max(dte - hold_days, 1) / 365.0, iv1) - prem) / prem if prem > 1e-6 else None
            for s in STRUCTS:
                if out[s] is not None and np.isfinite(out[s]):
                    trades[s].append(out[s]); monthly[s].setdefault(date, []).append(out[s])

    # ---- portfolios: F=0.25 premium deploy across perfect plays, rest RF ----
    months = sorted(stock_m_ret.keys())
    spy_ret = {midx[i]: float(spy_m.iloc[i + 1] / spy_m.iloc[i] - 1) for i in range(warmup, len(midx) - 1)
               if np.isfinite(spy_m.iloc[i + 1] / spy_m.iloc[i] - 1)}
    port = {}
    for s in STRUCTS + ["stock", "margin_1p5"]:
        rets, sps = [], []
        for m in months:
            sp = spy_ret.get(m)
            if sp is None:
                continue
            if s == "stock":
                mr = float(np.mean(stock_m_ret[m]))
            elif s == "margin_1p5":
                mr = 1.5 * float(np.mean(stock_m_ret[m])) - 0.5 * RF_M
            else:
                o = monthly[s].get(m)
                mr = F_DEPLOY * float(np.mean(o)) + (1 - F_DEPLOY) * RF_M if o else RF_M
            rets.append(mr); sps.append(sp)
        port[s] = _pstats(rets, sps)

    tstat = {s: _tstats(trades[s]) for s in STRUCTS}
    print("\n=== ALL OPTIONS — per-trade (perfect plays) & PORTFOLIO (F=0.25 premium) ===", flush=True)
    print(f"  {'structure':16} {'trade mean/med/win':22} {'PORT total':>10} {'Sh':>5} {'DD':>8}", flush=True)
    for s in ["stock", "margin_1p5"] + STRUCTS:
        t = tstat.get(s); p = port[s]
        tf = f"{t['mean_pct']}/{t['median_pct']}/{t['win_pct']}%" if t else ("stock/1x" if s == "stock" else "1.5x lev")
        print(f"  {s:16} {tf:22} {p['total_return']:>9}% {p['sharpe']:>5} {p['max_drawdown']:>7}%", flush=True)

    ranked = sorted(STRUCTS, key=lambda s: port[s]["total_return"], reverse=True)
    best = ranked[0]
    payload = {
        "computed_at": pd.Timestamp.utcnow().isoformat(),
        "params": {"deploy_fraction": F_DEPLOY, "rf_monthly": RF_M, "benchmark": BENCH, "months": int(len(midx)),
                   "note": "perfect plays only; F=25% of capital in premium, rest T-bills; stock=100% invested"},
        "per_trade": tstat, "portfolio": port, "ranked": ranked, "best_portfolio": best,
        "verdict": (f"Best options portfolio = {best}: {port[best]['total_return']}% total / Sh {port[best]['sharpe']} "
                    f"/ DD {port[best]['max_drawdown']}% (vs stock {port['stock']['total_return']}%/Sh{port['stock']['sharpe']}, "
                    f"1.5x margin {port['margin_1p5']['total_return']}%/Sh{port['margin_1p5']['sharpe']}). At F=25% deploy, "
                    "options portfolios trade lower total return (most capital idle) for defined risk; deep-ITM leads."),
        "caveat": "Real ATM IV, ATM used for all strikes (deep-ITM real IV lower -> real a touch better), no fees/"
                  "spreads/slippage/assignment, ~Sept2022+, ~47% coverage. Sizing F=25% is a choice; higher F scales "
                  "return AND drawdown. Perfect-play sample small.",
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
            kind="options_all", defaults={"payload": json.loads(json.dumps(p, default=str)),
                                          "computed_at": timezone.now()})
        print("Saved BacktestResult[options_all]", flush=True)
    except Exception as e:
        print("DB save failed:", e, flush=True)
    print("\n" + p["verdict"], flush=True)


if __name__ == "__main__":
    main()
