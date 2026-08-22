#!/usr/bin/env python3
"""Cost-aware BOOK validation for an RSI(14) bullish-DIVERGENCE bounce strategy (price 20d lower-low while RSI(14)
makes a higher low). Weekly EW book, survivorship-aware (covered ∪ delisted, through delisting), split-adjusted,
$5M/day + >$5. The edge is +~0.6%/wk (bear) / flat (bull), gone by 1mo -> so the make-or-break is TURNOVER/COSTS.
Reports GROSS + NET (10/25/50 bps/side) equity curves, turnover, by regime (all/bear/bull), 1-wk & 2-wk holds, vs SPY.
Run: MSYS_NO_PATHCONV=1 docker exec rotation-backend-1 python -u /app/strategy_divergence_book.py"""
import os, json, math
os.environ.setdefault("DJANGO_SETTINGS_MODULE", "rotation.settings")
import django; django.setup()
import numpy as np, pandas as pd
from pathlib import Path
from collections import defaultdict
from core.models import Candle, Sector, DelistedCompany

DVOL_FLOOR = 5e6; PRICE_FLOOR = 5.0

etfs = set(Sector.objects.values_list("etf", flat=True)) | {"SPY", "QQQ"}
covered = set()
for line in Path("/app/.data/analyst_ratings.jsonl").read_text(encoding="utf-8").splitlines():
    if line.strip():
        try:
            tk = json.loads(line).get("ticker")
            if tk and "." not in tk and tk not in etfs:
                covered.add(tk)
        except Exception:
            pass
delisted = set(DelistedCompany.objects.exclude(delisted_date=None).values_list("ticker", flat=True))
universe = sorted((covered | delisted) - etfs)
print(f"universe {len(universe)} ({len(delisted)} delisted)", flush=True)

spy = pd.DataFrame(list(Candle.objects.filter(ticker="SPY", date__gte="2014-06-01").values_list("date", "close")),
                   columns=["date", "close"]); spy["date"] = pd.to_datetime(spy["date"])
spy_c = spy.set_index("date")["close"].astype(float).sort_index()
spy_w = spy_c.resample("W-FRI").last(); widx = spy_w.index
spy_bull_w = (spy_c > spy_c.rolling(200).mean()).resample("W-FRI").last().reindex(widx).fillna(False)
spy_fwd1w = spy_w.shift(-1) / spy_w - 1.0


def rsi(s, n=14):
    d = s.diff(); up = d.clip(lower=0).rolling(n).mean(); dn = (-d.clip(upper=0)).rolling(n).mean()
    return 100 - 100 / (1 + up / dn.replace(0, np.nan))


FIRED, WCL = {}, {}
for i in range(0, len(universe), 300):
    rows = Candle.objects.filter(ticker__in=universe[i:i + 300], date__gte="2014-06-01").values_list("ticker", "date", "close", "volume")
    df = pd.DataFrame(list(rows), columns=["ticker", "date", "close", "volume"])
    if df.empty:
        continue
    df["date"] = pd.to_datetime(df["date"]); df["close"] = df["close"].astype(float); df["volume"] = df["volume"].astype(float)
    for tk, g in df.groupby("ticker", sort=False):
        s = g.sort_values("date").set_index("date")
        c = s["close"]
        if c.notna().sum() < 300:
            continue
        r = rsi(c, 14)
        bulldiv = (c <= c.rolling(20, min_periods=15).min()) & (r > r.shift(10))
        dvol20 = (c * s["volume"]).rolling(20, min_periods=10).mean()
        liq = (dvol20 >= DVOL_FLOOR) & (c > PRICE_FLOOR)
        fired_w = (bulldiv & liq).resample("W-FRI").max().reindex(widx).fillna(False)
        WCL[tk] = c.ffill(limit=252).resample("W-FRI").last().reindex(widx)   # through-delisting weekly close
        FIRED[tk] = fired_w
    print(f"  loaded {min(i+300,len(universe))}/{len(universe)}", flush=True)

FIRED = pd.DataFrame(FIRED).fillna(False); WCL = pd.DataFrame(WCL).reindex(columns=FIRED.columns)
fwd1 = WCL.shift(-1) / WCL - 1.0; fwd2 = WCL.shift(-2) / WCL - 1.0
print(f"weeks {widx[0].date()}..{widx[-1].date()}  names {FIRED.shape[1]}", flush=True)


def run_book(regime, hold_w, cost_bps):
    fwd = fwd1 if hold_w == 1 else fwd2
    pr, turns, held, prev = [], [], [], set()
    for d in widx[:-hold_w]:
        if regime == "bear" and bool(spy_bull_w.get(d, False)):
            pr.append(0.0); held.append(0); continue
        if regime == "bull" and not bool(spy_bull_w.get(d, False)):
            pr.append(0.0); held.append(0); continue
        row = FIRED.loc[d].astype(bool); frow = fwd.loc[d]
        names = [t for t in row.index if bool(row[t]) and pd.notna(frow[t])]
        if len(names) < 5:
            pr.append(0.0); held.append(0); prev = set(); continue
        r = fwd.loc[d, names].clip(-1.0, 1.5).mean()
        cur = set(names); turn = len(cur ^ prev) / (2 * max(len(cur), 1))   # one-way turnover of the book
        pr.append(float(r) - turn * 2 * cost_bps / 1e4 / hold_w); turns.append(turn); prev = cur
        held.append(len(names))
    return pd.Series(pr, index=widx[:-hold_w]), (np.mean(turns) if turns else 0), (np.mean([h for h in held if h]) if any(held) else 0)


def perf(pr, label, turn=None, nnm=None):
    pr = pr.dropna()
    if len(pr) < 26:
        print(f"  {label:40} (too few)"); return
    eq = np.cumprod(1 + pr.values); tot = (eq[-1] - 1) * 100; yrs = len(pr) / 52.0
    cagr = (eq[-1] ** (1 / yrs) - 1) * 100 if eq[-1] > 0 else -100
    sh = pr.mean() / pr.std() * np.sqrt(52) if pr.std() > 0 else 0
    dd = (eq / np.maximum.accumulate(eq) - 1).min() * 100; hit = (pr.values > 0).mean() * 100
    extra = f"  turn {turn*100:3.0f}%/wk  n~{nnm:.0f}" if turn is not None else ""
    print(f"  {label:40} CAGR {cagr:+6.1f}%  Sh {sh:4.2f}  DD {dd:6.1f}%  hit {hit:4.1f}%{extra}", flush=True)


print("\n=== RSI(14) bullish-divergence WEEKLY book — GROSS then NET of costs ===", flush=True)
for hold_w in (1, 2):
    print(f"-- hold {hold_w}wk --", flush=True)
    for regime in ("all", "bear", "bull"):
        s, tn, nm = run_book(regime, hold_w, 0)
        perf(s, f"{regime:4} GROSS (0 bps)", tn, nm)
        for cb in (10, 25, 50):
            s2, _, _ = run_book(regime, hold_w, cb)
            perf(s2, f"{regime:4} NET {cb} bps/side")
perf(spy_fwd1w, "SPY buy&hold (weekly)")
