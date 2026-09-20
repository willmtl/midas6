#!/usr/bin/env python3
"""FLAGSHIP VOL-AS-SELECTOR, FULL HISTORY (confirms the 2022+ IV finding with REALIZED vol, available for every pick
over 11y). For each flagship pick, compute trailing 63-day realized vol at entry (from candles) and the forward 1mo
return. Tercile split + corr — does higher vol => higher return hold on full history (not just the recent optionable
subset)? Also a BETA check: split by market direction (SPY up/down months) — if high-vol only wins when SPY is up it's
beta torque; if it wins in both it's more than beta. Saves BacktestResult[flagship_vol_selection].
Run: MSYS_NO_PATHCONV=1 docker exec rotation-backend-1 python -u /app/flagship_vol_selection.py"""
import os, json, math
os.environ.setdefault("DJANGO_SETTINGS_MODULE", "rotation.settings")
import django; django.setup()
import numpy as np, pandas as pd
from core.models import Candle


def tstat(x):
    x = pd.Series(x).dropna()
    return float(x.mean() / (x.std(ddof=1) / math.sqrt(len(x)))) if len(x) > 2 and x.std(ddof=1) > 0 else 0.0


def load_close(tickers):
    out = {}
    tickers = list(tickers)
    for i in range(0, len(tickers), 200):
        rows = Candle.objects.filter(ticker__in=tickers[i:i + 200], interval="1d", date__gte="2014-06-01"
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
    months = fj.get("months", [])
    seq = []
    for i in range(len(months) - 1):
        seq.append((pd.Timestamp(months[i]["date"]), pd.Timestamp(months[i + 1]["date"]),
                    [p["ticker"] for p in months[i].get("picks", []) if p.get("ticker")]))
    all_tks = {t for _, _, ts in seq for t in ts}
    close = load_close(all_tks | {"SPY"})
    spy = close.get("SPY")

    def spot_at(tk, d):
        s = close.get(tk)
        if s is None:
            return None
        p = s.index.searchsorted(d, side="right") - 1
        return float(s.iloc[p]) if 0 <= p < len(s) else None

    def rvol_at(tk, d):
        s = close.get(tk)
        if s is None:
            return None
        p = s.index.searchsorted(d, side="right")           # up to entry
        w = s.iloc[max(0, p - 63):p]
        if len(w) < 30:
            return None
        lr = np.log(w / w.shift(1)).dropna()
        return float(lr.std() * math.sqrt(252)) if len(lr) > 10 else None

    def spy_ret(d0, d1):
        a, b = spot_at("SPY", d0), spot_at("SPY", d1)
        return (b / a - 1.0) if a and b and a > 0 else 0.0

    rows = []
    for d0, d1, tks in seq:
        sret = spy_ret(d0, d1)
        for tk in tks:
            s0, s1 = spot_at(tk, d0), spot_at(tk, d1)
            rv = rvol_at(tk, d0)
            if not s0 or not s1 or s0 <= 0 or rv is None:
                continue
            rows.append({"date": d0, "ret": s1 / s0 - 1.0, "rvol": rv, "spy_up": sret > 0})
    df = pd.DataFrame(rows)
    print(f"pick-months with realized-vol: {len(df)} ({df['date'].min().date()}..{df['date'].max().date()})", flush=True)

    def line(x):
        v = pd.Series(x).dropna()
        return f"n={len(v):4} mean {v.mean()*100:>+6.2f}% med {v.median()*100:>+6.2f}% win {(v>0).mean()*100:>4.0f}% t {tstat(v):>5.2f} top10% {v.quantile(0.9)*100:>+6.1f}%"

    df = df.sort_values("rvol")
    df["q"] = pd.qcut(df["rvol"], 3, labels=["lowVol", "midVol", "highVol"])
    print("\n=== forward 1mo return by trailing-63d realized-vol tercile (FULL HISTORY) ===", flush=True)
    seg = {}
    for q in ["lowVol", "midVol", "highVol"]:
        sub = df[df["q"] == q]
        seg[q] = {"n": int(len(sub)), "vol_range": [round(float(sub.rvol.min())*100), round(float(sub.rvol.max())*100)],
                  "mean_pct": round(float(sub.ret.mean()*100), 2), "win_pct": round(float((sub.ret > 0).mean()*100)),
                  "t": round(tstat(sub.ret), 2)}
        print(f"  {q:8} vol[{seg[q]['vol_range'][0]}-{seg[q]['vol_range'][1]}%]  " + line(sub.ret), flush=True)
    corr = float(df["rvol"].corr(df["ret"]))
    print(f"\ncorr(realized vol, forward return) = {corr:+.3f}", flush=True)

    # BETA check: highVol vs lowVol in SPY-up vs SPY-down months
    print("\n=== BETA check: highVol−lowVol mean return, by SPY direction ===", flush=True)
    beta = {}
    for updn, lab in [(True, "SPY UP"), (False, "SPY DOWN")]:
        hi = df[(df.q == "highVol") & (df.spy_up == updn)].ret
        lo = df[(df.q == "lowVol") & (df.spy_up == updn)].ret
        d = (hi.mean() - lo.mean()) * 100 if len(hi) and len(lo) else float("nan")
        beta[lab] = round(float(d), 2)
        print(f"  {lab:9} highVol {hi.mean()*100:>+6.2f}% (n{len(hi)}) − lowVol {lo.mean()*100:>+6.2f}% (n{len(lo)}) = {d:>+6.2f}pp", flush=True)

    res = {"n": len(df), "vol_terciles": seg, "corr_vol_ret": round(corr, 3), "beta_check": beta,
           "note": "trailing 63d realized vol; full history; confirms/【refutes】 the 2022+ IV finding"}
    json.dump(res, open("/app/.data/studies/flagship_vol_selection.json", "w"), indent=2, default=str)
    try:
        from core.models import BacktestResult
        from django.utils import timezone
        BacktestResult.objects.update_or_create(kind="flagship_vol_selection", defaults={"payload": res, "computed_at": timezone.now()})
        print("Saved BacktestResult[flagship_vol_selection]", flush=True)
    except Exception as e:
        print("save skipped:", e, flush=True)


if __name__ == "__main__":
    main()
