#!/usr/bin/env python3
"""LOSS-SHAPE of the gap-down overnight book — user: "what about the 40% it doesn't hit?".
+16bps/night is an AVERAGE over ~60% winners and ~40% losers. The risk isn't the loser COUNT,
it's the SHAPE (small losses vs rare catastrophic) and whether losers CLUSTER into a drawdown.
Build the two deployable books (>$100M/day; gap<-3% and gap<-5%), buy-close/sell-open equal
weight, and report the full nightly net-return distribution + equity-curve drawdown.
-> BacktestResult[shortterm_gapdown_dist]. Cache /tmp/beat_candles.pkl.
"""
import os, json, warnings, pickle
warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
os.environ.setdefault("DJANGO_SETTINGS_MODULE", "rotation.settings")
import django
django.setup()
from pathlib import Path

CACHE = Path("/tmp/beat_candles.pkl")
LIQ = 100e6


def main():
    candles = pickle.loads(CACHE.read_bytes())
    print(f"{len(candles)} tickers", flush=True)
    E_date, E_on, E_gap = [], [], []
    for tk, df in candles.items():
        if df is None or len(df) < 40:
            continue
        o = df["Open"].values.astype(float); c = df["Close"].values.astype(float)
        idx = df.index; n = len(c)
        dvol = (df["Close"] * df["Volume"]).rolling(20).median().values
        for t in range(20, n - 1):
            if not (np.isfinite(dvol[t]) and dvol[t] >= LIQ and c[t] > 5 and o[t] > 0 and o[t + 1] > 0 and c[t - 1] > 0):
                continue
            gap = o[t] / c[t - 1] - 1.0
            if gap >= -0.03:
                continue
            on = o[t + 1] / c[t] - 1.0
            if abs(on) > 0.5:
                continue
            E_date.append(idx[t]); E_on.append(on); E_gap.append(gap)
    E_date = pd.DatetimeIndex(E_date); E_on = np.array(E_on); E_gap = np.array(E_gap)

    out = {}
    for glabel, gthr in (("gap<-3%", -0.03), ("gap<-5%", -0.05)):
        m = E_gap < gthr
        bybn = {}
        for i in np.where(m)[0]:
            bybn.setdefault(E_date[i], []).append(E_on[i])
        nights = sorted(bybn)
        gross = np.array([np.mean(bybn[d]) for d in nights])
        for cost in (5, 10):
            net = gross - cost / 1e4
            up = net[net > 0]; dn = net[net <= 0]
            eq = np.cumprod(1 + net)
            dd = (eq / np.maximum.accumulate(eq) - 1)
            maxdd = float(dd.min() * 100)
            # longest losing streak (consecutive down nights)
            streak = mx = 0
            for x in net:
                streak = streak + 1 if x <= 0 else 0
                mx = max(mx, streak)
            order = np.argsort(net)
            worst = [(nights[order[k]].date().isoformat(), round(float(net[order[k]]) * 100, 2)) for k in range(8)]
            sk = float(pd.Series(net).skew())
            key = f"{glabel}@{cost}bps"
            out[key] = {"nights": len(nights), "win%": round(len(up) / len(net) * 100, 1),
                        "mean_bps": round(float(net.mean()) * 1e4, 1),
                        "avg_up_bps": round(float(up.mean()) * 1e4, 1) if len(up) else None,
                        "avg_down_bps": round(float(dn.mean()) * 1e4, 1) if len(dn) else None,
                        "skew": round(sk, 2), "max_drawdown%": round(maxdd, 1),
                        "longest_losing_streak_nights": mx,
                        "worst8_nights": worst}
            print(f"\n=== {key}  ({len(nights)} nights) ===", flush=True)
            print(f"  win {out[key]['win%']}%  | avg UP {out[key]['avg_up_bps']}bps  avg DOWN {out[key]['avg_down_bps']}bps  "
                  f"| net mean {out[key]['mean_bps']}bps  skew {sk:.2f}", flush=True)
            print(f"  equity-curve MAX DRAWDOWN {maxdd:.1f}%   longest losing streak {mx} nights", flush=True)
            print(f"  worst 8 nights: " + ", ".join(f"{d} {r}%" for d, r in worst), flush=True)

    payload = {"computed_at": pd.Timestamp.utcnow().isoformat(), "dist": out, "liq_min": LIQ,
               "caveat": "Nightly equal-weight net returns, flat quoted-spread cost. Overnight tail = gap-down "
                         "names that gap down AGAIN on continued bad news (neg skew). No news/halt filter applied."}
    try:
        from core.models import BacktestResult
        from django.utils import timezone
        BacktestResult.objects.update_or_create(
            kind="shortterm_gapdown_dist",
            defaults={"payload": json.loads(json.dumps(payload, default=str)), "computed_at": timezone.now()})
        print("\nSaved BacktestResult[shortterm_gapdown_dist]", flush=True)
    except Exception as e:
        print("DB save failed:", e, flush=True)


if __name__ == "__main__":
    main()
