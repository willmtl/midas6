#!/usr/bin/env python3
"""FLAGSHIP OPTION STRATEGIES — full menu (user: "try all the option strategy not just at the money"). For each pick
(2022-09+, optionable), BS-price every structure from OptionSnapshot.atm_iv (FRICTIONLESS: no bid/ask, no skew -> an
optimistic upper bound; a structure that fails here is dead). Hold ~1mo to expiry (terminal payoff). Build an EW
portfolio of each and compare vs holding the STOCK.
LONG-LEVERAGE (return on cash outlay/premium): deep-ITM/ITM/ATM/OTM calls, bull call spread, LEAPS-style (3mo).
OTHER: risk reversal (synthetic long), covered call (caps tail), cash-secured put (income).
Saves BacktestResult[flagship_option_strategies]. NOTE: ignoring skew flatters the short-put structures (RR/CSP) most.
Run: MSYS_NO_PATHCONV=1 docker exec rotation-backend-1 python -u /app/flagship_option_strategies.py"""
import os, json, math
os.environ.setdefault("DJANGO_SETTINGS_MODULE", "rotation.settings")
import django; django.setup()
import numpy as np, pandas as pd
from core.models import Candle, OptionSnapshot

R = 0.04


def ncdf(x):
    return 0.5 * (1 + math.erf(x / math.sqrt(2)))


def bs(S, K, sig, T, call=True):
    if S <= 0 or K <= 0 or sig <= 0 or T <= 0:
        return None
    d1 = (math.log(S / K) + (R + sig * sig / 2) * T) / (sig * math.sqrt(T))
    d2 = d1 - sig * math.sqrt(T)
    if call:
        return S * ncdf(d1) - K * math.exp(-R * T) * ncdf(d2)
    return K * math.exp(-R * T) * ncdf(-d2) - S * ncdf(-d1)


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
    seq = []
    for i in range(len(months) - 1):
        seq.append((pd.Timestamp(months[i]["date"]), pd.Timestamp(months[i + 1]["date"]),
                    [p["ticker"] for p in months[i].get("picks", []) if p.get("ticker")]))
    all_tks = {t for _, _, ts in seq for t in ts}
    close = load_close(all_tks)
    iv = {}
    for tk, d, v in OptionSnapshot.objects.filter(ticker__in=list(all_tks)).exclude(atm_iv=None
            ).values_list("ticker", "date", "atm_iv"):
        iv.setdefault(tk, {})[pd.Timestamp(d)] = float(v)
    iv = {tk: pd.Series(s).sort_index() for tk, s in iv.items()}

    def spot(tk, d):
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
        v = float(w.iloc[(w.index - d).map(lambda x: abs(x.days)).argmin()]) if len(w) else None
        return (v / 100.0 if v and v > 3 else v)

    # strategy -> function(S, S1, sig, T) -> per-position return on capital (None if inapplicable)
    def r_call(K_mult):
        def f(S, S1, sig, T):
            K = K_mult * S; C = bs(S, K, sig, T, True)
            return None if not C or C <= 0 else (max(S1 - K, 0) - C) / C
        return f

    def r_bullspread(kl, ks):
        def f(S, S1, sig, T):
            Cl, Cs = bs(S, kl * S, sig, T, True), bs(S, ks * S, sig, T, True)
            deb = (Cl or 0) - (Cs or 0)
            if deb <= 0:
                return None
            payoff = max(S1 - kl * S, 0) - max(S1 - ks * S, 0)
            return (payoff - deb) / deb
        return f

    def r_riskrev(S, S1, sig, T):          # long ATM call + short ATM put = synthetic long; return on NOTIONAL S
        return (S1 - S) / S

    def r_covered(ks):
        def f(S, S1, sig, T):              # long stock + short OTM call, return on S
            Cs = bs(S, ks * S, sig, T, True) or 0
            return (min(S1, ks * S) - S + Cs) / S
        return f

    def r_csp(S, S1, sig, T):              # short ATM put, return on strike collateral S
        P = bs(S, S, sig, T, False) or 0
        return (P - max(S - S1, 0)) / S

    STRATS = {
        "stock": lambda S, S1, sig, T: S1 / S - 1.0,
        "call_deepITM(0.80)": r_call(0.80), "call_ITM(0.90)": r_call(0.90),
        "call_ATM(1.00)": r_call(1.00), "call_OTM(1.10)": r_call(1.10), "call_farOTM(1.20)": r_call(1.20),
        "bull_spread(1.0/1.15)": r_bullspread(1.0, 1.15), "bull_spread(1.0/1.30)": r_bullspread(1.0, 1.30),
        "risk_reversal(synthlong)": r_riskrev, "covered_call(1.15)": r_covered(1.15), "cash_sec_put(ATM)": r_csp,
    }

    monthly = {k: [] for k in STRATS}
    n_opt = 0
    for d0, d1, tks in seq:
        T = max((d1 - d0).days, 1) / 365.0
        rr = {k: [] for k in STRATS}
        for tk in tks:
            S, S1, sig = spot(tk, d0), spot(tk, d1), iv_at(tk, d0)
            if not S or not S1 or S <= 0 or sig is None:
                continue
            n_opt += 1
            for k, f in STRATS.items():
                v = f(S, S1, sig, T)
                if v is not None and np.isfinite(v):
                    rr[k].append(v)
        for k in STRATS:
            if rr[k]:
                monthly[k].append((d0, float(np.mean(rr[k]))))

    def stat(ser):
        r = pd.Series({d: v for d, v in ser}).sort_index()
        if len(r) < 12:
            return {"n": len(r)}
        eq = (1 + r).prod(); yrs = len(r) / 12.0
        dd = float(((1 + r).cumprod() / (1 + r).cumprod().cummax() - 1).min())
        def _s(v):
            return None if (v is None or not np.isfinite(v)) else v
        return {"n_mo": len(r), "total_pct": _s(round(float((eq - 1) * 100))),
                "cagr_pct": _s(round(float((eq ** (1 / yrs) - 1) * 100), 1)) if eq > 0 else -100.0,
                "sharpe": _s(round(float(r.mean() / r.std() * math.sqrt(12)) if r.std() > 0 else 0.0, 2)),
                "maxdd_pct": _s(round(dd * 100, 1))}

    res = {"n_optionable_positions": n_opt, "strategies": {}}
    st_stock = stat(monthly["stock"])
    print(f"=== FLAGSHIP OPTION STRATEGIES (frictionless upper bound, no skew; 2022-09..2026) ===", flush=True)
    print(f"  optionable positions: {n_opt}", flush=True)
    print(f"{'strategy':>24} {'total%':>12} {'CAGR':>7} {'Sh':>6} {'maxDD':>8}", flush=True)
    for k in STRATS:
        s = stat(monthly[k]); res["strategies"][k] = s
        flag = "  <-- stock" if k == "stock" else ("  BEATS stock" if (s.get("total_pct") or -1e9) > (st_stock.get("total_pct") or 0) else "")
        g = lambda key: (s.get(key) if s.get(key) is not None else float("nan"))
        print(f"{k:>24} {g('total_pct'):>12,.0f} {g('cagr_pct'):>6.1f}% {g('sharpe'):>6.2f} {g('maxdd_pct'):>7.1f}%{flag}", flush=True)
    winners = [k for k in STRATS if k != "stock" and (res["strategies"][k].get("total_pct") or -1e9) > (st_stock.get("total_pct") or 0)]
    res["beats_stock"] = winners
    print(f"\nstructures beating the STOCK (even frictionless/no-skew): {winners or 'NONE'}", flush=True)

    json.dump(res, open("/app/.data/studies/flagship_option_strategies.json", "w"), indent=2, default=str)
    try:
        from core.models import BacktestResult
        from django.utils import timezone
        BacktestResult.objects.update_or_create(kind="flagship_option_strategies", defaults={"payload": res, "computed_at": timezone.now()})
        print("Saved BacktestResult[flagship_option_strategies]", flush=True)
    except Exception as e:
        print("save skipped:", e, flush=True)


if __name__ == "__main__":
    main()
