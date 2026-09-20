#!/usr/bin/env python3
"""FLAGSHIP IV-AS-SELECTOR (user: "use IV as a selection mechanism"). Does a pick's ATM implied vol (OptionSnapshot
.atm_iv) predict its forward 1-month return? Split the flagship's actual picks by entry ATM-IV tercile and compare
forward stock return (mean/median/win/t + tail). Also: are UN-optionable picks (no options at all — often the highest-
torque microcaps) better or worse than optionable ones? If IV ranks returns, it's a usable tilt/gate. ⚠️ only 2022-09+
(options data start) and only the optionable subset — small, recent sample. Saves BacktestResult[flagship_iv_selection].
Run: MSYS_NO_PATHCONV=1 docker exec rotation-backend-1 python -u /app/flagship_iv_selection.py"""
import os, json, math
os.environ.setdefault("DJANGO_SETTINGS_MODULE", "rotation.settings")
import django; django.setup()
import numpy as np, pandas as pd
from core.models import Candle, OptionSnapshot


def tstat(x):
    x = pd.Series(x).dropna()
    return float(x.mean() / (x.std(ddof=1) / math.sqrt(len(x)))) if len(x) > 2 and x.std(ddof=1) > 0 else 0.0


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

    rows = []
    for d0, d1, tks in seq:
        for tk in tks:
            s0, s1 = spot_at(tk, d0), spot_at(tk, d1)
            if not s0 or not s1 or s0 <= 0:
                continue
            r = s1 / s0 - 1.0
            ivv = iv_at(tk, d0)
            rows.append({"tk": tk, "date": d0, "ret": r, "iv": (ivv / 100.0 if ivv and ivv > 3 else ivv)})
    df = pd.DataFrame(rows)
    opt = df[df["iv"].notna()].copy()
    unopt = df[df["iv"].isna()].copy()
    print(f"pick-months {len(df)} | optionable {len(opt)} ({100*len(opt)/len(df):.0f}%) | un-optionable {len(unopt)}", flush=True)

    def line(lbl, x):
        v = pd.Series(x).dropna()
        return f"{lbl:22} n={len(v):4} mean {v.mean()*100:>+6.2f}% med {v.median()*100:>+6.2f}% win {(v>0).mean()*100:>4.0f}% t {tstat(v):>5.2f} top10%avg {v.quantile(0.9)*100:>+6.1f}%"

    print("\n=== optionable vs un-optionable picks (forward 1mo stock return) ===", flush=True)
    print("  " + line("optionable", opt["ret"]), flush=True)
    print("  " + line("UN-optionable", unopt["ret"]), flush=True)

    # IV terciles within optionable
    opt = opt.sort_values("iv")
    opt["ivq"] = pd.qcut(opt["iv"], 3, labels=["lowIV", "midIV", "highIV"])
    print("\n=== forward return by entry ATM-IV tercile (optionable picks) ===", flush=True)
    seg = {}
    for q in ["lowIV", "midIV", "highIV"]:
        sub = opt[opt["ivq"] == q]
        seg[q] = {"n": int(len(sub)), "iv_range": [round(float(sub["iv"].min())*100, 0), round(float(sub["iv"].max())*100, 0)],
                  "mean_pct": round(float(sub["ret"].mean()*100), 2), "win_pct": round(float((sub["ret"] > 0).mean()*100), 0),
                  "t": round(tstat(sub["ret"]), 2)}
        print(f"  {q:8} IV[{seg[q]['iv_range'][0]:.0f}-{seg[q]['iv_range'][1]:.0f}%]  " + line("", sub["ret"]), flush=True)
    corr = float(opt["iv"].corr(opt["ret"]))
    print(f"\ncorr(entry IV, forward return) = {corr:+.3f}", flush=True)

    res = {"n_pickmonths": len(df), "optionable_pct": round(100*len(opt)/len(df), 0),
           "optionable_mean_pct": round(float(opt["ret"].mean()*100), 2),
           "unoptionable_mean_pct": round(float(unopt["ret"].mean()*100), 2) if len(unopt) else None,
           "iv_terciles": seg, "corr_iv_ret": round(corr, 3),
           "note": "2022-09+ only, optionable subset; small recent sample"}
    json.dump(res, open("/app/.data/studies/flagship_iv_selection.json", "w"), indent=2, default=str)
    try:
        from core.models import BacktestResult
        from django.utils import timezone
        BacktestResult.objects.update_or_create(kind="flagship_iv_selection", defaults={"payload": res, "computed_at": timezone.now()})
        print("Saved BacktestResult[flagship_iv_selection]", flush=True)
    except Exception as e:
        print("save skipped:", e, flush=True)


if __name__ == "__main__":
    main()
