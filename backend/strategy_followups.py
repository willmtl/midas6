#!/usr/bin/env python3
"""Three follow-ups on the daily discovery result, as deployable BOOK equity curves (not per-trade stats):
  #2 SHORT book — monthly EW short of deep-oversold / new-52w-low firers (the t=12.6 edge). Survivorship-aware
     (a short of a name that delists->0 WINS +100%); net a flat borrow cost.
  #3 BEAR-regime oversold-bounce — monthly EW long of rsi_oversold30 firers, ACTIVE ONLY when SPY<200d MA.
  #1 PAIR-GATE flip test — does gating the (negative-long) oversold entry by a QUALITY/TREND/A-D filter flip its
     6-month long alpha positive? (the only plausible route to a new LONG strategy.)
Survivorship-aware universe (covered ∪ delisted), split-adjusted, market-demeaned where noted, through-delisting.
Run: MSYS_NO_PATHCONV=1 docker exec rotation-backend-1 python -u /app/strategy_followups.py"""
import os, json
os.environ.setdefault("DJANGO_SETTINGS_MODULE", "rotation.settings")
import django; django.setup()
import numpy as np, pandas as pd
from pathlib import Path
from collections import defaultdict
from core.models import Candle, Sector, DelistedCompany
from seq_fundamental_study import load_financial_reports

DVOL_FLOOR = 5e6; PRICE_FLOOR = 5.0; BORROW_BPS_YR = 300  # 3%/yr short borrow

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
spy_m = spy_c.resample("ME").last(); midx = spy_m.index
spy_bull = (spy_c > spy_c.rolling(200).mean()).resample("ME").last().reindex(midx).fillna(False)
spy_fwd1 = spy_m.shift(-1) / spy_m - 1.0

reps = load_financial_reports(universe)


def _ttm_ni_monthly(tk):
    r = reps.get(tk)
    if r is None or "net_income" not in r.columns:
        return None
    d = r[["period_end", "avail_date", "net_income"]].dropna().sort_values("period_end")
    if len(d) < 4:
        return None
    d["ttm"] = d["net_income"].rolling(4).sum()
    s = pd.Series(d["ttm"].values, index=pd.to_datetime(d["avail_date"])).dropna()
    if s.empty:
        return None
    s = s[~s.index.duplicated(keep="last")].sort_index()
    return s.reindex(s.index.union(midx)).ffill().reindex(midx)


# monthly panels accumulated per name (memory-light: we only keep monthly booleans/returns)
FWD, LOW52, OSOLD, DD50X, PROF, TREND, ADRISE, FWD6 = ({} for _ in range(8))
BATCH = 300
for i in range(0, len(universe), BATCH):
    batch = universe[i:i + BATCH]
    rows = Candle.objects.filter(ticker__in=batch, date__gte="2014-06-01").values_list("ticker", "date", "high", "low", "close", "volume")
    df = pd.DataFrame(list(rows), columns=["ticker", "date", "high", "low", "close", "volume"])
    if df.empty:
        continue
    for cc in ("high", "low", "close", "volume"):
        df[cc] = df[cc].astype(float)
    df["date"] = pd.to_datetime(df["date"])
    ttm = None
    for tk, g in df.groupby("ticker", sort=False):
        s = g.sort_values("date").set_index("date")
        c = s["close"]
        if c.notna().sum() < 260:
            continue
        cff = c.ffill(limit=252)
        cm = c.resample("ME").last()
        dvolm = (c * s["volume"]).rolling(20, min_periods=10).mean().resample("ME").last()
        # daily signals -> "fired during month"
        d = c.diff(); up = d.clip(lower=0).rolling(14).mean(); dn = (-d.clip(upper=0)).rolling(14).mean()
        rsi = 100 - 100 / (1 + up / dn.replace(0, np.nan))
        os30 = ((rsi < 30) & (rsi.shift(1) >= 30))
        low52 = (c <= c.rolling(252, min_periods=60).min())
        dd60 = c / c.rolling(60, min_periods=20).max() - 1.0
        rsix = (rsi > 30) & (rsi.shift(1) <= 30)             # rsi cross up from oversold
        dd50x = (dd60 < -0.50) & rsix                        # deep-drawdown + rsi turn (the negative-long entry)
        sma200 = c.rolling(200).mean()
        # A/D (accum/dist) rising: ADL cumsum of money-flow-mult * vol; rising slope over ~20d
        mfm = ((s["close"] - s["low"]) - (s["high"] - s["close"])) / (s["high"] - s["low"]).replace(0, np.nan)
        adl = (mfm.fillna(0) * s["volume"]).cumsum()
        adrise = adl > adl.shift(20)
        firem = lambda b: b.resample("ME").max().reindex(midx).fillna(False).astype(bool)
        liqm = (dvolm >= DVOL_FLOOR).reindex(midx).fillna(False) & (cm > PRICE_FLOOR).reindex(midx).fillna(False)
        fwd1 = (cff.resample("ME").last().shift(-1) / cm - 1.0).reindex(midx)
        fwd6 = (cff.resample("ME").last().shift(-6) / cm - 1.0).reindex(midx)
        LOW52[tk] = firem(low52) & liqm
        OSOLD[tk] = firem(os30) & liqm
        DD50X[tk] = firem(dd50x) & liqm
        TREND[tk] = (cm > sma200.resample("ME").last()).reindex(midx).fillna(False)
        ADRISE[tk] = firem(adrise)
        FWD[tk] = fwd1; FWD6[tk] = fwd6
        t = _ttm_ni_monthly(tk)
        PROF[tk] = (t > 0).reindex(midx).fillna(False) if t is not None else pd.Series(False, index=midx)
    print(f"  loaded {min(i+BATCH,len(universe))}/{len(universe)}", flush=True)

FWD = pd.DataFrame(FWD); FWD6 = pd.DataFrame(FWD6)
LOW52 = pd.DataFrame(LOW52).reindex(columns=FWD.columns).fillna(False)
OSOLD = pd.DataFrame(OSOLD).reindex(columns=FWD.columns).fillna(False)
DD50X = pd.DataFrame(DD50X).reindex(columns=FWD.columns).fillna(False)
TREND = pd.DataFrame(TREND).reindex(columns=FWD.columns).fillna(False)
ADRISE = pd.DataFrame(ADRISE).reindex(columns=FWD.columns).fillna(False)
PROF = pd.DataFrame(PROF).reindex(columns=FWD.columns).fillna(False)


def perf(pr, label, ann=12):
    pr = pr.dropna()
    if len(pr) < 12:
        print(f"  {label:42} (too few months)"); return
    eq = np.cumprod(1 + pr.values); tot = (eq[-1] - 1) * 100; cagr = (eq[-1] ** (ann / len(pr)) - 1) * 100
    sh = pr.mean() / pr.std() * np.sqrt(ann) if pr.std() > 0 else 0
    dd = (eq / np.maximum.accumulate(eq) - 1).min() * 100; hit = (pr.values > 0).mean() * 100
    print(f"  {label:42} TOT {tot:+9.1f}%  CAGR {cagr:+6.1f}%  Sh {sh:4.2f}  DD {dd:6.1f}%  hit {hit:4.1f}%  n {len(pr)}", flush=True)


def book(fire_df, direction="long", bear_only=False, hold_fwd=FWD, borrow=False):
    """Monthly EW book. direction long/short. bear_only: hold cash unless SPY<200dMA. Returns monthly series."""
    pr = []
    for d in midx[:-1]:
        if bear_only and bool(spy_bull.get(d, False)):
            pr.append(0.0); continue
        names = [t for t in fire_df.columns if bool(fire_df.loc[d, t]) and pd.notna(hold_fwd.loc[d, t])]
        if len(names) < 5:
            pr.append(0.0); continue
        r = hold_fwd.loc[d, names].clip(-1.0, 1.5).mean()
        m = -r if direction == "short" else r
        if borrow:
            m -= BORROW_BPS_YR / 1e4 / 12.0
        pr.append(float(m))
    return pd.Series(pr, index=midx[:-1])


print("\n=== #2 SHORT BOOK (monthly EW short, survivorship-aware, 3%/yr borrow) ===", flush=True)
perf(book(LOW52, "short", borrow=True), "short new_52low firers")
perf(book(DD50X, "short", borrow=True), "short deep-DD + RSI-turn firers")
perf(-spy_fwd1, "(ref) short SPY buy&hold")

print("\n=== #3 BEAR-REGIME oversold-bounce (long rsi<30 firers, active only SPY<200dMA) ===", flush=True)
perf(book(OSOLD, "long", bear_only=True), "bear-only oversold-bounce book")
perf(book(OSOLD, "long", bear_only=False), "(ref) same book ALL regimes")
perf(spy_fwd1, "(ref) SPY buy&hold")

print("\n=== #1 PAIR-GATE flip test: oversold-entry 6mo LONG alpha, ungated vs gated ===", flush=True)
spy_fwd6 = (spy_m.shift(-6) / spy_m - 1.0).reindex(midx)
def gate_alpha(entry, gate, label):
    dem, wins, n = [], 0, 0
    mo = defaultdict(list)
    for d in midx[:-6]:
        sf = spy_fwd6.get(d)
        if not np.isfinite(sf):
            continue
        names = [t for t in entry.columns if bool(entry.loc[d, t]) and (gate is None or bool(gate.loc[d, t]))
                 and pd.notna(FWD6.loc[d, t])]
        for t in names:
            a = float(np.clip(FWD6.loc[d, t], -1.0, 1.5)) - float(sf) * 100 / 100  # demeaned (both fractions)
            dem.append(a); wins += 1 if a > 0 else 0; n += 1; mo[f"{d.year}-{d.month:02d}"].append(a)
    if n < 100:
        print(f"  {label:46} (n={n}, too few)"); return
    mm = [np.mean(v) for v in mo.values() if v]
    t = np.mean(mm) / (np.std(mm, ddof=1) / np.sqrt(len(mm))) if len(mm) > 5 else float("nan")
    print(f"  {label:46} n {n:>6}  demeaned6m {np.mean(dem)*100:+6.2f}%  win {wins/n*100:4.1f}%  t {t:+5.2f}", flush=True)

gate_alpha(OSOLD, None, "rsi<30 (ungated)")
gate_alpha(OSOLD, PROF, "rsi<30 & TTM-profitable")
gate_alpha(OSOLD, TREND, "rsi<30 & above 200d SMA")
gate_alpha(OSOLD, ADRISE, "rsi<30 & A/D rising")
gate_alpha(OSOLD, (PROF & TREND), "rsi<30 & profitable & above-200SMA")
gate_alpha(OSOLD, (PROF & ADRISE), "rsi<30 & profitable & A/D rising")
