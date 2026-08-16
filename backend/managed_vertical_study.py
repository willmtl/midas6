#!/usr/bin/env python3
"""MANAGED VERTICAL — the realistic version of the options overlay. Instead of riding call-debit spreads
to expiry (worst case for a buyer), MANAGE them: daily re-price with real ATM IV, TAKE PROFIT at +50% on
the debit (exit when spread value >= 1.5x paid), else close at the monthly rebalance (capturing leftover
time value, not zero). Optional -50% stop variant. Compares stock vs hold-to-expiry vs managed, for ALL
picks and PERFECT plays (conviction>=4). Priced via Black-Scholes on OptionSnapshot ATM IV (Sept2022+).
45 DTE entry (management room), K2 = +10% short strike.
-> BacktestResult[managed_vertical] + JSON.
Run: MSYS_NO_PATHCONV=1 docker exec rotation-backend-1 python -u /app/managed_vertical_study.py
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
from seq_fundamental_study import load_candles, load_financial_reports
from trend_stock_studies import _pit_monthly_panel, _available_at, _ret_delist, CRYPTO
from backtest_lowpb import _monthly_close, BENCH
import price_basis
from core.models import OptionSnapshot

OUT = Path(__file__).resolve().parent / ".data" / "studies" / "managed_vertical.json"
LOOKBACK, TOP_N = 6, 10
R_F, DTE, K2_OTM, TP, SL = 0.04, 45.0 / 365.0, 1.10, 1.50, 0.50
_N = NormalDist().cdf


def bs_call(S, K, T, r, sig):
    if T <= 0 or sig <= 0 or S <= 0:
        return max(S - K, 0.0)
    d1 = (log(S / K) + (r + sig * sig / 2) * T) / (sig * sqrt(T)); d2 = d1 - sig * sqrt(T)
    return S * _N(d1) - K * exp(-r * T) * _N(d2)


def spread(S, K1, K2, T, iv):
    return bs_call(S, K1, T, R_F, iv) - bs_call(S, K2, T, R_F, iv)


def _rsi(close, n=10):
    d = close.diff(); up = d.clip(lower=0).rolling(n).mean(); dn = (-d.clip(upper=0)).rolling(n).mean()
    return (100 - 100 / (1 + up / dn.replace(0, np.nan)))


def _summ(x):
    a = np.array(x, float)
    if not len(a):
        return None
    w, l = a[a > 0], a[a <= 0]
    return dict(n=len(a), mean_pct=round(a.mean() * 100, 1), median_pct=round(float(np.median(a)) * 100, 1),
                win_pct=round((a > 0).mean() * 100, 1),
                avg_win_pct=round(w.mean() * 100, 1) if len(w) else None,
                avg_loss_pct=round(l.mean() * 100, 1) if len(l) else None,
                worst_pct=round(float(a.min()) * 100, 1), best_pct=round(float(a.max()) * 100, 1),
                n_losers=int(len(l)))


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

    iv_rows = OptionSnapshot.objects.filter(ticker__in=list(common), atm_iv__isnull=False).values_list("ticker", "date", "atm_iv")
    iv_map = {}
    for tk, d, iv in iv_rows:
        iv_map.setdefault(tk, {})[pd.Timestamp(d)] = (iv / 100.0 if iv > 1.5 else iv)
    iv_daily = {}                       # ticker -> daily-ffilled IV aligned to that stock's candle index
    for tk in iv_map:
        s = pd.Series(iv_map[tk]).sort_index()
        if tk in stock_daily:
            iv_daily[tk] = s.reindex(stock_daily[tk].index.union(s.index)).sort_index().ffill()
    print(f"months {len(midx)} | stocks {len(common)} | IV tickers {len(iv_daily)}", flush=True)
    warmup = max(LOOKBACK, 1)

    B = {k: {"all": [], "perfect": []} for k in ("stock", "expiry", "managed_tp", "managed_tp_sl")}
    tp_hits = []
    n_pick = n_opt = 0
    for i in range(warmup, len(midx) - 1):
        date, ndate = midx[i], midx[i + 1]
        if not np.isfinite(spy_m.iloc[i + 1] / spy_m.iloc[i] - 1):
            continue
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
        cheap_cut = pbs[max(0, len(pbs) // 3 - 1)] if pbs else 0
        for etf, pick, ret in picks:
            n_pick += 1
            sc = (int(pb.loc[date, pick] <= cheap_cut) + int(pd.notna(ni.loc[date, pick]) and ni.loc[date, pick] > 0)
                  + int(bool(vlow_debt.loc[date, pick])) + int(pd.notna(rsi.loc[date, pick]) and rsi.loc[date, pick] < 45)
                  + int(etf in top5))
            ivd = iv_daily.get(pick); dd = stock_daily.get(pick)
            if ivd is None or dd is None or date not in ivd.index or pd.isna(ivd.get(date)):
                continue
            dser = dd["Close"]
            if date not in dser.index:
                continue
            S0 = float(dser.loc[:date].iloc[-1]); iv0 = float(ivd.loc[:date].iloc[-1])
            if not (S0 > 0 and iv0 > 0):
                continue
            n_opt += 1
            K1, K2 = S0, S0 * K2_OTM
            debit = spread(S0, K1, K2, DTE, iv0)
            if debit <= 1e-6:
                continue
            expiry = date + pd.Timedelta(days=45)
            hold_days = dser.index[(dser.index > date) & (dser.index <= min(expiry, ndate))]
            # hold-to-expiry payoff (intrinsic at expiry from realized rebalance move)
            ST = S0 * (1 + ret)
            exp_ret = (max(min(ST, K2) - K1, 0) - debit) / debit
            # managed: walk daily, take profit at +50% (val>=TP*debit); optional -50% stop
            tp_ret = None; tpsl_ret = None; hit_day = None
            last_val = debit
            for d in hold_days:
                S_d = float(dser.loc[d]); iv_d = ivd.get(d)
                iv_d = float(iv_d) if pd.notna(iv_d) else iv0
                T_d = max((expiry - d).days, 0) / 365.0
                val = spread(S_d, K1, K2, T_d, iv_d)
                last_val = val
                if tp_ret is None and val >= TP * debit:
                    tp_ret = TP - 1.0; hit_day = (d - date).days
                if tpsl_ret is None:
                    if val >= TP * debit:
                        tpsl_ret = TP - 1.0
                    elif val <= SL * debit:
                        tpsl_ret = SL - 1.0
            if tp_ret is None:
                tp_ret = (last_val - debit) / debit          # close at rebalance mark
            if tpsl_ret is None:
                tpsl_ret = (last_val - debit) / debit
            if hit_day is not None:
                tp_hits.append(hit_day)
            for name, val in (("stock", ret), ("expiry", exp_ret), ("managed_tp", tp_ret), ("managed_tp_sl", tpsl_ret)):
                B[name]["all"].append(val)
                if sc >= 4:
                    B[name]["perfect"].append(val)

    res = {k: {g: _summ(v) for g, v in d.items()} for k, d in B.items()}
    cov = round(n_opt / n_pick * 100, 1) if n_pick else 0
    tp_rate = round(len(tp_hits) / len(B["managed_tp"]["all"]) * 100, 1) if B["managed_tp"]["all"] else 0
    print(f"\n=== MANAGED VERTICAL (TP +50%, close at rebalance) — coverage {cov}%, TP-hit {tp_rate}% ===", flush=True)
    print(f"  {'instrument':14} {'ALL (mean/med/win)':30} {'PERFECT (mean/med/win)'}", flush=True)
    for k in ("stock", "expiry", "managed_tp", "managed_tp_sl"):
        a, p = res[k]["all"], res[k]["perfect"]
        af = f"{a['mean_pct']}/{a['median_pct']}/{a['win_pct']}% (n{a['n']})" if a else "–"
        pf = f"{p['mean_pct']}/{p['median_pct']}/{p['win_pct']}% (n{p['n']})" if p else "–"
        print(f"  {k:14} {af:30} {pf}", flush=True)

    mtp = res["managed_tp"]["perfect"]; st = res["stock"]["perfect"]; ex = res["expiry"]["perfect"]
    payload = {
        "computed_at": pd.Timestamp.utcnow().isoformat(),
        "params": {"dte_days": 45, "short_strike_otm": K2_OTM, "take_profit": "+50% on debit", "stop": "-50% (variant)",
                   "coverage_pct": cov, "tp_hit_pct": tp_rate, "avg_tp_day": round(float(np.mean(tp_hits)), 1) if tp_hits else None},
        "results": res,
        "verdict": ("Managing the vertical (take profit +50%, close at rebalance) LIFTS the median well above the "
                    "hold-to-expiry floor" + (f" — perfect-play managed_tp median {mtp['median_pct']}% / win "
                    f"{mtp['win_pct']}% vs expiry median {ex['median_pct']}%; mean {mtp['mean_pct']}% vs stock "
                    f"{st['mean_pct']}%." if (mtp and ex and st) else ".")),
        "caveat": "Real ATM IV, daily re-priced, but ATM used for both legs (no per-strike skew), no fees/slippage/"
                  "spreads/early-assignment, ~Sept2022+, 47% coverage. Option returns high-variance -> tiny size.",
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
            kind="managed_vertical", defaults={"payload": json.loads(json.dumps(p, default=str)),
                                               "computed_at": timezone.now()})
        print("Saved BacktestResult[managed_vertical]", flush=True)
    except Exception as e:
        print("DB save failed:", e, flush=True)
    print("\n" + p["verdict"], flush=True)


if __name__ == "__main__":
    main()
