#!/usr/bin/env python3
"""Intraday 8-12h GAP-UP CONTINUATION strategy — the core, cost-tested in ABSOLUTE terms (deployment cares about
net absolute return, not demeaned edge). Mega/large-cap only (that's where the edge + tight spreads live).
Per-trade: enter at the gap-up 4h bar close, exit +1/+2/+3 bars (~4/8/12h). Reports GROSS + NET absolute per-trade
return at 5/10/20 bps round-trip (mega-caps are cheap to trade), win%, n, by regime and sector. ETFs excluded.
Run: MSYS_NO_PATHCONV=1 docker exec rotation-backend-1 python -u /app/strategy_intraday_gap.py"""
import os, glob
os.environ.setdefault("DJANGO_SETTINGS_MODULE", "rotation.settings")
import django; django.setup()
import numpy as np, pandas as pd
from collections import defaultdict
from studies import SIGNALS
from seq_fundamental_study import load_financial_reports
from core.models import Candle, Sector, Fundamental

HB = [1, 2, 3]                       # ~4h, ~8h, ~12h
CAP_MIN = 10e9                       # mega+large only (>=$10B): where continuation + tight spreads live
COSTS = [5, 10, 20]                  # round-trip bps
SIGS = [s for s in ("gap_up_med", "gap_up_large") if s in SIGNALS]

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
fund = {r["ticker"]: (r["market_cap"], r["sector"]) for r in
        Fundamental.objects.filter(ticker__in=names).values("ticker", "market_cap", "sector")}
reps = load_financial_reports(names)
spy = pd.DataFrame(list(Candle.objects.filter(ticker="SPY", date__gte="2014-06-01").values_list("date", "close")),
                   columns=["date", "close"]); spy["date"] = pd.to_datetime(spy["date"])
spy_c = spy.set_index("date")["close"].astype(float).sort_index()
bull_by_date = {d.date(): bool(v) for d, v in (spy_c > spy_c.rolling(200).mean()).items()}
print(f"4h stock names: {len(names)}", flush=True)


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


# trades[(sig,hz,regime,seg)] = list of GROSS per-trade returns (%)
trades = defaultdict(list)
for tk in names:
    df = frames[tk]; c = df["Close"].values; n = len(c)
    sh = shares_ff(tk, df.index); mc_static = (fund.get(tk) or (None, None))[0]
    sec = (fund.get(tk) or (None, None))[1] or "sec?"
    for sk in SIGS:
        try:
            sig = SIGNALS[sk][1](df).fillna(False)
        except Exception:
            continue
        for i in np.where(sig.values)[0]:
            if c[i] <= 5:
                continue
            mc = (sh.iloc[i] * c[i]) if (sh is not None and pd.notna(sh.iloc[i])) else mc_static
            if mc is None or not np.isfinite(mc) or mc < CAP_MIN:      # mega/large-cap only
                continue
            reg = "bull" if bull_by_date.get(df.index[i].date(), False) else "bear"
            for h in HB:
                j = i + h
                if j >= n:
                    continue
                r = (c[j] - c[i]) / c[i] * 100.0
                if not np.isfinite(r):
                    continue
                for seg in ("all", f"sec:{sec}"):
                    trades[(sk, h, reg, seg)].append(r)


def summ(lst):
    a = np.array(lst, float); a = a[np.isfinite(a)]
    if len(a) < 60:
        return None
    return len(a), a.mean(), (a > 0).mean() * 100


print(f"\n=== mega/large-cap (>=$10B) GAP-UP continuation — per-trade ABSOLUTE return, GROSS then NET ===", flush=True)
print(f"(net = gross - round-trip cost; ~8h = 2 bars. n>=60 shown)", flush=True)
for sk in SIGS:
    for reg in ("bull", "bear", ):
        print(f"\n{sk} [{reg}]  (regime by SPY 200d)", flush=True)
        print(f"  {'hz':>4}{'n':>7}{'gross%':>8}{'win%':>7}" + "".join(f"{'net'+str(cb):>8}" for cb in COSTS), flush=True)
        for h in HB:
            s = summ(trades[(sk, h, reg, "all")])
            if not s:
                continue
            nn, g, w = s
            nets = "".join(f"{g - cb/100.0:>8.3f}" for cb in COSTS)
            print(f"  {h:>4}{nn:>7}{g:>8.3f}{w:>7.1f}{nets}", flush=True)
        # sector breakdown at ~8h (2 bars)
        secs = sorted({k[3] for k in trades if k[0] == sk and k[2] == reg and k[3].startswith("sec:")})
        rows = []
        for sv in secs:
            s = summ(trades[(sk, 2, reg, sv)])
            if s:
                rows.append((sv[4:], s))
        if rows:
            rows.sort(key=lambda r: -r[1][1])
            print(f"  -- by sector @~8h: " + "  ".join(f"{sv}:{g:+.2f}%/{w:.0f}%(n{nn})" for sv, (nn, g, w) in rows[:6]), flush=True)
