#!/usr/bin/env python3
"""Do commodities offer something SPY/sector-rotation doesn't — return, or diversification?

For each of the 93 sector ETFs (5y): total return vs SPY, daily-return correlation & beta to SPY, ann vol.
Commodity instruments (pure commodity ETFs/futures) are tagged. The question: do commodities BEAT SPY
(a second return source) or just DIVERSIFY it (low correlation, lower absolute return)?
Run: MSYS_NO_PATHCONV=1 docker exec rotation-backend-1 python -u /app/commodity_check.py
"""
import os, warnings
warnings.filterwarnings("ignore")
os.environ.setdefault("DJANGO_SETTINGS_MODULE", "rotation.settings")
import django
django.setup()
import numpy as np, pandas as pd
import config
from seq_fundamental_study import load_candles

BENCH = "SPY"
KW = ("gold", "silver", "oil", "gas", "copper", "wheat", "agri", "commodit", "lithium",
      "uranium", "steel", "lumber", "coal", "platinum", "palladium", "corn", "energy", "rare earth")


def is_commodity(name):
    n = name.lower()
    # producer/miner equities correlate with stocks; tag only the resource sleeves by keyword
    return any(k in n for k in KW)


def main():
    etfs = {n: e for n, e in config.SECTOR_ETFS.items() if e}
    data = load_candles(list(etfs.values()) + [BENCH])
    spy = data[BENCH]["Close"]
    spy_ret = np.log(spy / spy.shift(1)).dropna()
    spy_5y = float(spy.iloc[-1] / spy.iloc[0] - 1) * 100

    rows = []
    for name, e in etfs.items():
        df = data.get(e)
        if df is None or len(df) < 250:
            continue
        c = df["Close"]
        r = np.log(c / c.shift(1)).dropna()
        al = pd.DataFrame({"e": r, "s": spy_ret}).dropna()
        if len(al) < 250:
            continue
        corr = float(al["e"].corr(al["s"]))
        beta = float(np.cov(al["e"], al["s"])[0, 1] / np.var(al["s"]))
        rows.append({"name": name, "etf": e, "commodity": is_commodity(name),
                     "ret5y": round(float(c.iloc[-1] / c.iloc[0] - 1) * 100, 1),
                     "vs_spy": round(float(c.iloc[-1] / c.iloc[0] - 1) * 100 - spy_5y, 1),
                     "corr": round(corr, 2), "beta": round(beta, 2),
                     "ann_vol": round(float(r.std() * np.sqrt(252)) * 100, 1)})

    comm = [r for r in rows if r["commodity"]]
    print(f"SPY 5y total return: {spy_5y:+.1f}%   |   {len(rows)} ETFs, {len(comm)} commodity-tagged\n", flush=True)
    print("=== COMMODITY SLEEVES (sorted by correlation to SPY — lowest = best diversifier) ===", flush=True)
    print(f"{'sector':26} {'etf':7} {'ret5y%':>7} {'vsSPY%':>7} {'corr':>5} {'beta':>5} {'vol%':>6}", flush=True)
    for r in sorted(comm, key=lambda x: x["corr"]):
        print(f"{r['name']:26} {r['etf']:7} {r['ret5y']:>7} {r['vs_spy']:>7} {r['corr']:>5} {r['beta']:>5} {r['ann_vol']:>6}", flush=True)

    n_beat = sum(1 for r in comm if r["vs_spy"] > 0)
    avg_corr = np.mean([r["corr"] for r in comm])
    avg_ret = np.mean([r["ret5y"] for r in comm])
    print(f"\ncommodity sleeves beating SPY: {n_beat}/{len(comm)} | avg corr {avg_corr:.2f} | "
          f"avg 5y ret {avg_ret:+.1f}% vs SPY {spy_5y:+.1f}%", flush=True)
    # broad-universe context
    n_beat_all = sum(1 for r in rows if r["vs_spy"] > 0)
    print(f"ALL sectors beating SPY: {n_beat_all}/{len(rows)}", flush=True)


if __name__ == "__main__":
    main()
