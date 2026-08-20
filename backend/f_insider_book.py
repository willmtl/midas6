#!/usr/bin/env python3
"""F = INSIDER-BUY BOOK, time-stepped (the D/E/F/G winner promoted). Enter on the filed_date bar (PIT), hold N
trading days, equal-weight across concurrent positions, gross capped 1x (cash when none firing), net 10bps
turnover. Variants: net-buyer / cluster(>=3 insiders) / big-$ (top-quartile buy_value). REGIME CHECK: insider
buys cluster at market bottoms — split daily P&L by SPY>200dMA to see if the drift is real alpha or bear-bottom
beta. Benchmark vs SPY over the same window. Run: MSYS_NO_PATHCONV=1 docker exec rotation-backend-1 python -u /app/f_insider_book.py"""
import os, bisect, warnings
warnings.filterwarnings("ignore")
os.environ.setdefault("DJANGO_SETTINGS_MODULE", "rotation.settings")
import django
django.setup()
import numpy as np, pandas as pd
from django.db import connection
with connection.cursor() as cur:
    cur.execute("SET max_parallel_workers_per_gather = 0")
from seq_fundamental_study import load_candles
import h4_on_signals_study as S
from core.models import InsiderBuy

FEE, DVOL_FLOOR = 1e-3, 5e6
HOLDS = [10, 15, 20]

# ---- events ----
ev = []
for tk, fd, bv, sv, bc in InsiderBuy.objects.values_list("ticker", "filed_date", "buy_value", "sell_value", "buy_count"):
    ev.append((tk, fd, bv or 0.0, sv or 0.0, bc or 0))
tickers = sorted({e[0] for e in ev})
print(f"insider events={len(ev)} tickers={len(tickers)}", flush=True)
daily = load_candles(tickers + ["SPY"])
close = pd.DataFrame({t: daily[t]["Close"] for t in daily if t != "SPY"}).sort_index()
volp = pd.DataFrame({t: daily[t]["Volume"] for t in daily if t != "SPY"}).reindex_like(close)
R = close.pct_change().fillna(0.0)
dvol = (close * volp).rolling(20).mean()
idx = close.index
dates = [d.date() for d in idx]
col = {t: i for i, t in enumerate(close.columns)}
bvs = np.array([e[2] for e in ev if e[2] > 0]); bv_q75 = np.nanpercentile(bvs, 75)

spy = daily["SPY"]["Close"].reindex(idx, method="ffill")
regime_up = (spy > spy.rolling(200).mean()).fillna(False).values     # bull if SPY>200dMA
spy_ret = spy.pct_change().fillna(0.0)


def entry_matrix(kind):
    E = np.zeros(close.shape)
    for tk, fd, bv, sv, bc in ev:
        if tk not in col:
            continue
        if not (bv > sv):                       # net buyer only
            continue
        if kind == "cluster" and bc < 3:
            continue
        if kind == "big" and bv < bv_q75:
            continue
        j = bisect.bisect_left(dates, fd)       # first bar on/after filed_date
        if j >= len(dates):
            continue
        c = col[tk]
        if not np.isfinite(dvol.iat[j, c]) or dvol.iat[j, c] < DVOL_FLOOR:
            continue
        E[j, c] = 1.0
    return E


def run(kind, hold):
    E = pd.DataFrame(entry_matrix(kind), index=idx, columns=close.columns)
    held = (E.shift(1).rolling(hold, min_periods=1).max().fillna(0) > 0)
    hs = held.sum(axis=1)
    W = held.div(hs.where(hs > 0, np.nan), axis=0).fillna(0.0)     # equal-weight, gross 1x when any held else cash
    port = (W.shift(1) * R).sum(axis=1)
    turn = (W - W.shift(1)).abs().sum(axis=1)
    net = port - turn * FEE
    eq = (1 + net).cumprod()
    yrs = (idx[-1] - idx[0]).days / 365.25
    tot = (eq.iloc[-1] - 1) * 100
    cagr = ((eq.iloc[-1]) ** (1 / yrs) - 1) * 100
    dd = ((eq / eq.cummax()) - 1).min() * 100
    sh = net.mean() / net.std() * np.sqrt(252) if net.std() > 0 else 0
    inv = (hs > 0).mean() * 100
    # regime attribution
    up = net[regime_up]; dn = net[~pd.Series(regime_up, index=idx)]
    return dict(tot=tot, cagr=cagr, dd=dd, sh=sh, inv=inv, avgn=float(hs[hs > 0].mean()),
                up_mean=up.mean() * 1e4, dn_mean=dn.mean() * 1e4,
                up_days=int((regime_up).sum()), dn_days=int((~regime_up).sum()),
                ntrades=int(E.values.sum()))


# SPY benchmark
spy_eq = (1 + spy_ret).cumprod()
spy_tot = (spy_eq.iloc[-1] - 1) * 100
spy_dd = ((spy_eq / spy_eq.cummax()) - 1).min() * 100
print(f"window {idx[0].date()}..{idx[-1].date()}  SPY {spy_tot:+.0f}% DD {spy_dd:.1f}%\n", flush=True)
print(f"  {'variant':14}{'hold':>5}{'trades':>8}{'total%':>9}{'CAGR%':>7}{'maxDD%':>8}{'Sharpe':>7}{'inv%':>6}{'avgN':>6}   regime bp/day (bull|bear)", flush=True)
for kind in ("net", "cluster", "big"):
    for hold in HOLDS:
        r = run(kind, hold)
        print(f"  {kind:14}{hold:>5}{r['ntrades']:>8}{r['tot']:>9.0f}{r['cagr']:>7.1f}{r['dd']:>8.1f}"
              f"{r['sh']:>7.2f}{r['inv']:>6.0f}{r['avgn']:>6.1f}   {r['up_mean']:>7.1f} | {r['dn_mean']:>6.1f}"
              f"  ({r['up_days']}u/{r['dn_days']}d)", flush=True)
