#!/usr/bin/env python
"""Empirical fidelity: run the SAME port on (a) TradingView's own feed and (b) EODHD candles, with the
TRUE Pine settings (ULCER_MIN=10), and measure how close the indicator values + BUY/SELL signals are.
port-on-TV == what pine.txt plots on the chart; port-on-EODHD == the scanner."""
import os, json
os.environ.setdefault("DJANGO_SETTINGS_MODULE", "rotation.settings")
import django; django.setup()
import numpy as np, pandas as pd
import sortino_dd_scanner as S

S.ULCER_MIN = 10.0                                  # mirror the TRUE Pine ulcerMinimum
WARM = max(S.SORT_WIN + S.SORT_SMOOTH + S.SORT_RSI + S.SORT_EMA, S.DD_DD, 200)
PAIRS = [("RKLB", "/app/.data/tvfeed/RKLB.json"), ("AAPL", "/app/.data/tvfeed/AAPL.json")]


def tv_df(path):
    j = json.load(open(path)); b = j["bars"]
    df = pd.DataFrame(b); df["date"] = pd.to_datetime(df["date"])
    df = df.set_index("date")[["open", "high", "low", "close"]]
    df.columns = ["Open", "High", "Low", "Close"]
    return df


def indics(o, h, l, c, tk):
    close = c.to_frame(tk); ohlc4 = ((o + h + l + c) / 4.0).to_frame(tk)
    ind = S.compute_indicators(close, ohlc4)
    g = lambda k: ind[k][tk]
    return dict(srsi=g("srsi"), srsi_ema=g("srsi_ema"), nrsi=g("nrsi"),
                nrsi_ema=g("nrsi_ema"), dd=g("dd_score"), ulcer=g("ulcer"), close=c)


def signals(ind):
    idx = ind["close"].index
    args = [ind[k].to_numpy() for k in ("srsi", "srsi_ema", "nrsi", "nrsi_ema", "dd", "ulcer")]
    valid = ind["close"].notna().to_numpy()
    trips = S.run_state_machine(0, ind["close"].to_numpy(), *args, valid, WARM)
    buys = [idx[t[0]].date() for t in trips]
    sells = [idx[t[1]].date() for t in trips if t[1] >= 0]
    return buys, sells


for tk, path in PAIRS:
    tv = tv_df(path)
    cd = S.load_candles([tk]).get(tk)
    eo = pd.DataFrame({k: cd[k] for k in ("Open", "High", "Low", "Close")}).sort_index()
    eo = eo[eo.Close > 0]
    itv = indics(tv.Open, tv.High, tv.Low, tv.Close, tk)
    ieo = indics(eo.Open, eo.High, eo.Low, eo.Close, tk)
    common = itv["srsi"].index.intersection(ieo["srsi"].index)
    common = common[common >= common[-1] - pd.Timedelta(days=500)]        # last ~2y overlap
    print(f"\n================= {tk} =================")
    print(f"TV bars {len(tv)} ({tv.index[0].date()}..{tv.index[-1].date()})  |  "
          f"EODHD bars {len(eo)} ({eo.index[0].date()}..{eo.index[-1].date()})  |  common {len(common)}")
    # correlation + mean abs diff of indicator series over common window
    print(f"{'indicator':>10}{'corr':>8}{'mean|Δ|':>10}{'TV last':>10}{'EOD last':>10}{'Δ last':>9}")
    for k, dec in [("srsi", 1), ("srsi_ema", 1), ("dd", 3), ("ulcer", 2), ("nrsi", 1)]:
        a = itv[k].reindex(common); b = ieo[k].reindex(common)
        m = a.notna() & b.notna()
        corr = float(a[m].corr(b[m])) if m.sum() > 5 else float("nan")
        mad = float((a[m] - b[m]).abs().mean())
        print(f"{k:>10}{corr:>8.3f}{mad:>10.3f}{a.dropna().iloc[-1]:>10.3f}{b.dropna().iloc[-1]:>10.3f}"
              f"{a.dropna().iloc[-1]-b.dropna().iloc[-1]:>+9.3f}")
    # signal agreement (true Pine settings)
    btv, stv = signals({**itv, "close": itv["close"]})
    beo, seo = signals({**ieo, "close": ieo["close"]})
    lo = (common[0].date())
    btv = [d for d in btv if d >= lo]; beo = [d for d in beo if d >= lo]
    def match(x, y, tol=2):
        y2 = list(y); hit = 0
        for d in x:
            for e in list(y2):
                if abs((pd.Timestamp(d) - pd.Timestamp(e)).days) <= tol:
                    hit += 1; y2.remove(e); break
        return hit
    mm = match(btv, beo)
    print(f"BUY signals (ULCER_MIN=10, last ~2y):  TV(=Pine)={len(btv)}  EODHD(scanner)={len(beo)}  "
          f"matched within 2 bars={mm}")
    print(f"  TV/Pine BUY dates : {btv}")
    print(f"  EODHD/scan BUY dts: {beo}")
