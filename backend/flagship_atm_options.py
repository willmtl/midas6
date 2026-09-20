#!/usr/bin/env python3
"""FLAGSHIP ATM-OPTIONS OVERLAY (user: "try options at the money"). Instead of buying the flagship's stock pick, buy a
~1-month AT-THE-MONEY call (the most liquid strike) for leverage+convexity. FRICTIONLESS UPPER BOUND: price the ATM
call via Black-Scholes from OptionSnapshot.atm_iv (no bid/ask spread modeled — so this is optimistic; if it fails here
it's dead, since real option spreads only hurt). Hold to ~expiry (payoff = intrinsic). Compare an EW portfolio of ATM
calls vs the SAME picks held as stock, 2022-09+ (options data starts then). Reports total/CAGR/Sharpe/DD + optionable
coverage + how often calls expire worthless. Saves BacktestResult[flagship_atm_options].
Run: MSYS_NO_PATHCONV=1 docker exec rotation-backend-1 python -u /app/flagship_atm_options.py"""
import os, json, math
os.environ.setdefault("DJANGO_SETTINGS_MODULE", "rotation.settings")
import django; django.setup()
import numpy as np, pandas as pd
from core.models import Candle, OptionSnapshot

R = 0.04


def ncdf(x):
    return 0.5 * (1 + math.erf(x / math.sqrt(2)))


def bs_atm_call(S, sigma, T):
    """ATM (K=S) BS call premium. sigma decimal, T years."""
    if S <= 0 or sigma <= 0 or T <= 0:
        return None
    d1 = (R + sigma * sigma / 2) * T / (sigma * math.sqrt(T))
    d2 = d1 - sigma * math.sqrt(T)
    return S * ncdf(d1) - S * math.exp(-R * T) * ncdf(d2)


def load_close(tickers):
    out = {}
    tickers = list(tickers)
    for i in range(0, len(tickers), 200):
        rows = Candle.objects.filter(ticker__in=tickers[i:i + 200], interval="1d", date__gte="2022-01-01"
                                     ).values_list("ticker", "date", "close")
        df = pd.DataFrame(list(rows), columns=["ticker", "date", "close"])
        if df.empty:
            continue
        df["date"] = pd.to_datetime(df["date"]); df["close"] = df["close"].astype(float)
        for tk, g in df.groupby("ticker", sort=False):
            out[tk] = g.sort_values("date").set_index("date")["close"]
    return out


def main():
    fj = json.load(open("/app/.data/studies/flagship_history.json"))
    months = [m for m in fj.get("months", []) if m.get("date", "") >= "2022-08-01"]
    # build (entry_date, exit_date, [tickers]) from consecutive trace months
    seq = []
    for i in range(len(months) - 1):
        d0 = pd.Timestamp(months[i]["date"]); d1 = pd.Timestamp(months[i + 1]["date"])
        tks = [p["ticker"] for p in months[i].get("picks", []) if p.get("ticker")]
        if tks:
            seq.append((d0, d1, tks))
    all_tks = {t for _, _, ts in seq for t in ts}
    close = load_close(all_tks)

    # atm_iv panel: ticker -> Series(date->atm_iv)
    iv = {}
    for tk, d, v in OptionSnapshot.objects.filter(ticker__in=list(all_tks)).exclude(atm_iv=None
            ).values_list("ticker", "date", "atm_iv"):
        iv.setdefault(tk, {})[pd.Timestamp(d)] = float(v)
    iv = {tk: pd.Series(s).sort_index() for tk, s in iv.items()}

    def spot_at(tk, d):
        s = close.get(tk)
        if s is None:
            return None
        p = s.index.searchsorted(d, side="right") - 1
        return float(s.iloc[p]) if 0 <= p < len(s) else None

    def iv_at(tk, d):
        s = iv.get(tk)
        if s is None:
            return None
        w = s[(s.index >= d - pd.Timedelta(days=7)) & (s.index <= d + pd.Timedelta(days=3))]
        return float(w.iloc[(w.index - d).map(lambda x: abs(x.days)).argmin()]) if len(w) else None

    stock_m, call_m = [], []          # monthly EW returns
    n_pick = n_opt = n_worthless = 0
    ivs = []
    for d0, d1, tks in seq:
        T = max((d1 - d0).days, 1) / 365.0
        s_rets, c_rets = [], []
        for tk in tks:
            n_pick += 1
            s0 = spot_at(tk, d0); s1 = spot_at(tk, d1)
            if not s0 or not s1 or s0 <= 0:
                continue
            sr = s1 / s0 - 1.0
            s_rets.append(sr)
            ivv = iv_at(tk, d0)
            if ivv is None:
                continue                                  # un-optionable pick -> no call leg
            sig = ivv / 100.0 if ivv > 3 else ivv         # atm_iv stored in % (e.g. 51.8)
            prem = bs_atm_call(s0, sig, T)
            if not prem or prem <= 0:
                continue
            n_opt += 1; ivs.append(sig)
            payoff = max(s1 - s0, 0.0)                     # hold ~to expiry -> intrinsic
            cr = payoff / prem - 1.0
            if payoff <= 0:
                n_worthless += 1
            c_rets.append(cr)
        if s_rets:
            stock_m.append((d0, float(np.mean(s_rets))))
        if c_rets:
            call_m.append((d0, float(np.mean(c_rets))))

    def stat(ser):
        r = pd.Series({d: v for d, v in ser}).sort_index()
        eq = (1 + r).prod(); yrs = len(r) / 12.0
        return {"n_mo": len(r), "total_pct": round(float((eq - 1) * 100)),
                "cagr_pct": round(float((eq ** (1 / yrs) - 1) * 100), 1) if eq > 0 and yrs > 0 else None,
                "sharpe": round(float(r.mean() / r.std() * math.sqrt(12)) if r.std() > 0 else 0.0, 2),
                "maxdd_pct": round(float(((1 + r).cumprod() / (1 + r).cumprod().cummax() - 1).min() * 100), 1)}

    st_stock, st_call = stat(stock_m), stat(call_m)
    res = {"window": f"{seq[0][0].date()}..{seq[-1][1].date()}", "n_picks": n_pick, "n_optionable": n_opt,
           "opt_coverage_pct": round(100 * n_opt / max(1, n_pick), 0), "pct_calls_worthless": round(100 * n_worthless / max(1, n_opt), 0),
           "avg_atm_iv_pct": round(float(np.mean(ivs) * 100), 0) if ivs else None,
           "stock": st_stock, "atm_call_frictionless": st_call}
    print(f"=== FLAGSHIP ATM-OPTIONS OVERLAY (frictionless upper bound, {res['window']}) ===", flush=True)
    print(f"  picks {n_pick} | optionable {n_opt} ({res['opt_coverage_pct']:.0f}%) | avg ATM IV {res['avg_atm_iv_pct']}% | "
          f"calls expiring worthless {res['pct_calls_worthless']:.0f}%", flush=True)
    print(f"  STOCK (same picks):     total {st_stock['total_pct']:>7,}%  CAGR {st_stock['cagr_pct']}%  Sh {st_stock['sharpe']}  DD {st_stock['maxdd_pct']}%", flush=True)
    print(f"  ATM CALLS (no spread):  total {st_call['total_pct']:>7,}%  CAGR {st_call['cagr_pct']}%  Sh {st_call['sharpe']}  DD {st_call['maxdd_pct']}%", flush=True)
    verdict = "BEATS stock even frictionless -> worth modeling spreads" if (st_call["total_pct"] or -1) > (st_stock["total_pct"] or 0) else "LOSES even frictionless -> DEAD (spreads only worse)"
    res["verdict"] = verdict
    print(f"  VERDICT: {verdict}", flush=True)

    json.dump(res, open("/app/.data/studies/flagship_atm_options.json", "w"), indent=2, default=str)
    try:
        from core.models import BacktestResult
        from django.utils import timezone
        BacktestResult.objects.update_or_create(kind="flagship_atm_options", defaults={"payload": res, "computed_at": timezone.now()})
        print("Saved BacktestResult[flagship_atm_options]", flush=True)
    except Exception as e:
        print("save skipped:", e, flush=True)


if __name__ == "__main__":
    main()
