#!/usr/bin/env python3
"""8-12h (2-3 4h-bar) intraday edge — SEGMENTED by market-cap bucket AND sector (never averaged across all).
Curated momentum + oversold signals on the 1732 4h names (ETFs excluded). Cross-sectionally demeaned vs the 4h
EW-universe index, winsorized, bull/bear split, LONG. Cap via PIT market cap (shares_outstanding x price), sector
via Fundamental. ⚠️ 4h universe is survivor-selected + inherently large-cap (only liquid names have intraday data).
Run: MSYS_NO_PATHCONV=1 docker exec rotation-backend-1 python -u /app/h4_segmented_8h.py"""
import os, json, math, glob
os.environ.setdefault("DJANGO_SETTINGS_MODULE", "rotation.settings")
import django; django.setup()
import numpy as np, pandas as pd
from collections import defaultdict
from studies import SIGNALS
from seq_fundamental_study import load_financial_reports
from core.models import Candle, Sector, Fundamental

HBARS = [1, 2, 3]                          # ~4h, ~8h, ~12h
WINS = (-30.0, 30.0)                       # intraday winsorize (tighter — no multi-day tails)
CURATED = ["gap_up_med", "gap_up_large", "rsi_x_above_sma", "rsi_x50_up", "vol_shock_up",
           "rsi_sup10_x_wk", "rsi_oversold30", "gap_down_large", "vol_shock_dn"]
CURATED = [s for s in CURATED if s in SIGNALS]


def cap_bucket(mc):
    if mc is None or not np.isfinite(mc) or mc <= 0:
        return "cap?"
    b = mc / 1e9
    return "mega>200B" if b >= 200 else "large10-200" if b >= 10 else "mid2-10" if b >= 2 else "small<2B"


etfs = set(Sector.objects.values_list("etf", flat=True)) | {"SPY", "QQQ"}
frames = {}
for f in sorted(glob.glob("/app/.data/intraday/4h/*.parquet")):
    tk = f.split("/")[-1][:-8]
    if tk in etfs:
        continue
    try:
        df = pd.read_parquet(f)
    except Exception:
        continue
    if len(df) >= 200 and "Close" in df.columns:
        frames[tk] = df[~df.index.duplicated(keep="last")].sort_index()
names = list(frames)
print(f"4h stock names (ETFs excl): {len(names)}", flush=True)

fund = {r["ticker"]: (r["market_cap"], r["sector"]) for r in
        Fundamental.objects.filter(ticker__in=names).values("ticker", "market_cap", "sector")}
reps = load_financial_reports(names)


def shares_ff(tk, dates):
    r = reps.get(tk)
    if r is None or "shares_outstanding" not in r.columns:
        return None
    d = r[["avail_date", "shares_outstanding"]].dropna().sort_values("avail_date")
    if d.empty:
        return None
    s = pd.Series(d["shares_outstanding"].values, index=pd.to_datetime(d["avail_date"]))
    s = s[~s.index.duplicated(keep="last")].sort_index()
    return s.reindex(s.index.union(dates)).ffill().reindex(dates)


# 4h EW index + daily-SPY regime
rets = {tk: frames[tk]["Close"].pct_change().pipe(lambda r: r[(r > -0.5) & (r < 0.5)]) for tk in names}
ew = (1.0 + pd.DataFrame(rets).mean(axis=1).fillna(0.0)).cumprod()
spy = pd.DataFrame(list(Candle.objects.filter(ticker="SPY", date__gte="2014-06-01").values_list("date", "close")),
                   columns=["date", "close"]); spy["date"] = pd.to_datetime(spy["date"])
spy_c = spy.set_index("date")["close"].astype(float).sort_index()
bull_by_date = {d.date(): bool(v) for d, v in (spy_c > spy_c.rolling(200).mean()).items()}


def _w(x):
    return float(min(max(x, WINS[0]), WINS[1]))


# acc[(sk,bars,reg,segtype,segval)] = [n, sum_dem, wins]
acc = defaultdict(lambda: [0, 0.0, 0])
for ni, tk in enumerate(names):
    df = frames[tk]; c = df["Close"]; close = c.values; n = len(close)
    bench = ew.reindex(df.index).values
    sh = shares_ff(tk, df.index)
    sec = (fund.get(tk) or (None, None))[1] or "sec?"
    mc_static = (fund.get(tk) or (None, None))[0]
    for sk in CURATED:
        try:
            sig = SIGNALS[sk][1](df).fillna(False)
        except Exception:
            continue
        for i in np.where(sig.values)[0]:
            if close[i] <= 5 or not np.isfinite(bench[i]) or bench[i] <= 0:
                continue
            mc = (sh.iloc[i] * close[i]) if (sh is not None and pd.notna(sh.iloc[i])) else mc_static
            cb = cap_bucket(mc)
            d0 = df.index[i]; reg = "bull" if bull_by_date.get(d0.date(), False) else "bear"
            for h in HBARS:
                j = i + h
                if j >= n or not np.isfinite(bench[j]) or bench[j] <= 0:
                    continue
                dem = _w((close[j] - close[i]) / close[i] * 100.0 - (bench[j] / bench[i] - 1.0) * 100.0)
                for st, sv in (("all", "all"), ("cap", cb), ("sec", sec)):
                    a = acc[(sk, h, reg, st, sv)]
                    a[0] += 1; a[1] += dem; a[2] += 1 if dem > 0 else 0
    if (ni + 1) % 400 == 0:
        print(f"  {ni+1}/{len(names)}", flush=True)


def stat(key):
    a = acc.get(key)
    if not a or a[0] < 60:
        return None
    return a[0], a[1] / a[0], a[2] / a[0] * 100


print("\n=== 8-12h edge, SEGMENTED (demeaned% / win% / n), LONG. bars: 1~4h 2~8h 3~12h ===", flush=True)
for sk in CURATED:
    printed = False
    for reg in ("bull", "bear"):
        # only show a signal/regime if the 'all' cell at 2-3 bars is non-trivial
        base = [stat((sk, h, reg, "all", "all")) for h in (2, 3)]
        if not any(base):
            continue
        printed = True
        allrow = " | ".join(f"{h}bar {('%+.2f' % stat((sk,h,reg,'all','all'))[1]) if stat((sk,h,reg,'all','all')) else '  -  '}" for h in HBARS)
        print(f"\n{sk}  [{reg}]   ALL: {allrow}", flush=True)
        for st, label in (("cap", "by MARKET CAP"), ("sec", "by SECTOR")):
            segs = sorted({k[4] for k in acc if k[0] == sk and k[2] == reg and k[3] == st})
            rows = []
            for sv in segs:
                s2 = stat((sk, 2, reg, st, sv))   # focus 2-bar (~8h)
                if s2:
                    rows.append((sv, s2))
            if rows:
                rows.sort(key=lambda r: -r[1][1])
                print(f"    {label} (~8h hold):", flush=True)
                for sv, (nn, dem, win) in rows:
                    print(f"      {sv:14} dem {dem:+6.2f}%  win {win:4.1f}%  n {nn}", flush=True)
