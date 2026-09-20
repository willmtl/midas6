#!/usr/bin/env python3
"""FLAGSHIP OPTIONS — LONGER-DATED, SOLD BEFORE EXPIRY (user: "what expiration were you buying?"). The prior tests
bought a ~1mo option and HELD TO EXPIRY = 100% theta decay (worst case; 42% went to -100%). Fairer: buy a longer-dated
call (45/60/90d), SELL after the ~30d hold, repricing at exit via BS with the REMAINING time (exit IV = atm_iv at
exit) so losers keep residual time value instead of zeroing. Test ATM + deep-ITM x several tenors, frictionless (no
spread/skew = optimistic). EW portfolio vs stock, 2022-09+. Saves BacktestResult[flagship_options_roll].
Run: MSYS_NO_PATHCONV=1 docker exec rotation-backend-1 python -u /app/flagship_options_roll.py"""
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
        return max(S - K, 0.0) if call else max(K - S, 0.0)     # at/after expiry -> intrinsic
    d1 = (math.log(S / K) + (R + sig * sig / 2) * T) / (sig * math.sqrt(T)); d2 = d1 - sig * math.sqrt(T)
    return S * ncdf(d1) - K * math.exp(-R * T) * ncdf(d2) if call else K * math.exp(-R * T) * ncdf(-d2) - S * ncdf(-d1)


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
    seq = [(pd.Timestamp(months[i]["date"]), pd.Timestamp(months[i + 1]["date"]),
            [p["ticker"] for p in months[i].get("picks", []) if p.get("ticker")]) for i in range(len(months) - 1)]
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
        w = s[(s.index >= d - pd.Timedelta(days=10)) & (s.index <= d + pd.Timedelta(days=5))]
        v = float(w.iloc[(w.index - d).map(lambda x: abs(x.days)).argmin()]) if len(w) else None
        return (v / 100.0 if v and v > 3 else v)

    # (label, strike_mult, tenor_days) — hold ~ the rebalance gap, sell with remaining time
    ARMS = [("stock", None, None),
            ("ATM 30d hold-to-exp", 1.00, 30), ("ATM 60d sold@~30d", 1.00, 60), ("ATM 90d sold@~30d", 1.00, 90),
            ("ITM0.9 60d sold@~30d", 0.90, 60), ("deepITM0.8 60d sold", 0.80, 60), ("deepITM0.8 90d sold", 0.80, 90),
            ("deepITM0.7 90d sold", 0.70, 90)]

    def stat(ser):
        r = pd.Series({d: v for d, v in ser}).sort_index()
        if len(r) < 12:
            return {"n": len(r)}
        eq = (1 + r).prod(); yrs = len(r) / 12.0
        dd = float(((1 + r).cumprod() / (1 + r).cumprod().cummax() - 1).min())
        f = lambda v: (None if not np.isfinite(v) else v)
        return {"n_mo": len(r), "total_pct": f(round(float((eq - 1) * 100))),
                "cagr_pct": f(round(float((eq ** (1 / yrs) - 1) * 100), 1)) if eq > 0 else -100.0,
                "sharpe": f(round(float(r.mean() / r.std() * math.sqrt(12)) if r.std() > 0 else 0.0, 2)),
                "maxdd_pct": f(round(dd * 100, 1))}

    monthly = {a[0]: [] for a in ARMS}
    for d0, d1, tks in seq:
        hold = max((d1 - d0).days, 1)
        for tk in tks:
            S, S1, sig0 = spot(tk, d0), spot(tk, d1), iv_at(tk, d0)
            if not S or not S1 or S <= 0:
                continue
            sig1 = iv_at(tk, d1) or sig0
            for lab, km, tenor in ARMS:
                if km is None:
                    monthly[lab].append((d0, S1 / S - 1.0)); continue
                if sig0 is None:
                    continue
                K = km * S
                T0 = tenor / 365.0
                Trem = max(tenor - hold, 0) / 365.0                # remaining life at sale
                entry = bs(S, K, sig0, T0, True)
                if not entry or entry <= 0:
                    continue
                exit_ = bs(S1, K, sig1, Trem, True)                # sold with residual time value (or intrinsic if 0)
                monthly[lab].append((d0, exit_ / entry - 1.0))
    for lab in monthly:
        monthly[lab] = [(d, r) for d, r in monthly[lab]]

    res = {"arms": {}}
    st_stock = stat([(d, r) for d, r in monthly["stock"]])
    # aggregate to monthly EW
    def ew(rows):
        g = {}
        for d, r in rows:
            g.setdefault(d, []).append(r)
        return [(d, float(np.mean(v))) for d, v in sorted(g.items())]
    print("=== FLAGSHIP OPTIONS: longer-dated, SOLD before expiry (frictionless; 2022-09+) ===", flush=True)
    print(f"{'arm':>24} {'total%':>12} {'CAGR':>7} {'Sh':>6} {'maxDD':>8}", flush=True)
    stock_ew = ew(monthly["stock"]); sstock = stat(stock_ew)
    for lab, *_ in ARMS:
        s = stat(ew(monthly[lab])); res["arms"][lab] = s
        g = lambda k: (s.get(k) if s.get(k) is not None else float("nan"))
        flag = "  <-- stock" if lab == "stock" else ("  BEATS stock" if (s.get("total_pct") or -1e9) > (sstock.get("total_pct") or 0) else "")
        print(f"{lab:>24} {g('total_pct'):>12,.0f} {g('cagr_pct'):>6.1f}% {g('sharpe'):>6.2f} {g('maxdd_pct'):>7.1f}%{flag}", flush=True)
    winners = [l for l, *_ in ARMS if l != "stock" and (res["arms"][l].get("total_pct") or -1e9) > (sstock.get("total_pct") or 0)]
    res["beats_stock"] = winners
    print(f"\nbeats stock (frictionless): {winners or 'NONE'}", flush=True)

    json.dump(res, open("/app/.data/studies/flagship_options_roll.json", "w"), indent=2, default=str)
    try:
        from core.models import BacktestResult
        from django.utils import timezone
        BacktestResult.objects.update_or_create(kind="flagship_options_roll", defaults={"payload": res, "computed_at": timezone.now()})
        print("Saved BacktestResult[flagship_options_roll]", flush=True)
    except Exception as e:
        print("save skipped:", e, flush=True)


if __name__ == "__main__":
    main()
