#!/usr/bin/env python3
"""STRATEGY D validation gate — does the CAPITULATION-BOUNCE edge survive honest accounting?

The #1 mirage risk for a bounce thesis is SURVIVORSHIP: capitulation names delist disproportionately, so a
study that silently drops them will show a fake bounce. This includes delisted-with-candles names and computes
forward returns THROUGH delisting (exit at last traded price — the project's verified-neutral convention; a
falling knife into bankruptcy scores its real ≈ last-price loss, NOT dropped). Returns use split-adjusted close.

Sweeps capitulation TRIGGERS x HORIZONS x REGIME (SPY vs 200d MA) x {ALL names vs SURVIVORS-only}, reports
market-DEMEANED bounce (event fwd return minus SPY over the same window), win rate, N, and an HONEST monthly-
panel t-stat (N = calendar months, not overlapping events). The survivors-vs-all GAP is the survivorship estimate.
Run: MSYS_NO_PATHCONV=1 docker exec rotation-backend-1 python -u /app/strategy_d_validate.py"""
import os, json
os.environ.setdefault("DJANGO_SETTINGS_MODULE", "rotation.settings")
import django; django.setup()
import numpy as np, pandas as pd
from pathlib import Path
from core.models import Candle, DelistedCompany, Sector

DVOL_FLOOR = 5e6
START = "2016-01-01"
HORIZONS = [5, 10, 21]

etfs = set(Sector.objects.values_list("etf", flat=True)) | {"SPY", "QQQ"}
covered = set()
for line in Path("/app/.data/analyst_ratings.jsonl").read_text(encoding="utf-8").splitlines():
    if not line.strip():
        continue
    try:
        r = json.loads(line)
    except Exception:
        continue
    tk = r.get("ticker")
    if tk and "." not in tk and tk not in etfs:
        covered.add(tk)
delisted = set(DelistedCompany.objects.exclude(delisted_date=None).values_list("ticker", flat=True)) - etfs
universe = sorted((covered | delisted) - etfs)
print(f"universe {len(universe)} (covered {len(covered)} + delisted {len(delisted)}); survivorship-aware", flush=True)

# common trading calendar + regime from SPY
spy = pd.DataFrame(list(Candle.objects.filter(ticker="SPY", date__gte="2014-06-01").values_list("date", "close")),
                   columns=["date", "close"])
spy["date"] = pd.to_datetime(spy["date"]); spy = spy.set_index("date")["close"].astype(float).sort_index()
cal = spy.index[spy.index >= pd.Timestamp(START)]
spy_bull = (spy > spy.rolling(200).mean()).reindex(cal)          # regime: SPY above its 200d MA
spy_c = spy.reindex(cal)
spy_fwd = {h: spy_c.shift(-h) / spy_c - 1.0 for h in HORIZONS}


def rsi(s, n=14):
    d = s.diff(); up = d.clip(lower=0).rolling(n).mean(); dn = (-d.clip(upper=0)).rolling(n).mean()
    rs = up / dn.replace(0, np.nan)
    return 100 - 100 / (1 + rs)


events = []                                                       # one row per capitulation event
BATCH = 300
for i in range(0, len(universe), BATCH):
    batch = universe[i:i + BATCH]
    rows = Candle.objects.filter(ticker__in=batch, date__gte="2014-06-01").values_list("ticker", "date", "close", "volume")
    df = pd.DataFrame(list(rows), columns=["ticker", "date", "close", "volume"])
    if df.empty:
        continue
    df["date"] = pd.to_datetime(df["date"]); df["close"] = df["close"].astype(float); df["volume"] = df["volume"].astype(float)
    for tk, g in df.groupby("ticker", sort=False):
        s = g.set_index("date").sort_index()
        c = s["close"].reindex(cal)                              # align to common calendar (NaN before list / after delist)
        if c.notna().sum() < 260:
            continue
        v = (s["close"] * s["volume"]).reindex(cal)
        dvol20 = v.rolling(20, min_periods=10).mean()
        ret1 = c.pct_change()
        vol20 = ret1.rolling(20).std()
        z = ret1 / vol20                                          # 1-day return z-score (vol-shock)
        r14 = rsi(c, 14)
        low52 = c <= c.rolling(252, min_periods=60).min()        # new 52-week low
        dd60 = c / c.rolling(60, min_periods=20).max() - 1.0     # drawdown from 60d high
        c_ff = c.ffill(limit=252)                                # exit-at-last-price for delisted names (survivorship-honest)

        def clip(x):
            return np.nan if pd.isna(x) else float(np.clip(x, -1.0, 1.5))   # winsorize +/- kills garbage-bar / seam outliers

        is_delisted = tk in delisted
        state = {"zdrop": z < -3.0, "rsi25": r14 < 25.0, "low52": low52, "dd25": dd60 < -0.25}
        liq = (dvol20 >= DVOL_FLOOR) & (c > 5.0)                  # liquid AND >$5 (kills penny junk + most bad bars)
        up = ret1 > 0
        for name, st in state.items():
            fire = st & (~st.shift(1).fillna(False)) & liq & c.notna()   # EDGE-trigger: the day it crosses into capitulation
            for d in cal[fire.reindex(cal).fillna(False).values]:
                pos = cal.get_loc(d)
                rec = {"ticker": tk, "date": d, "trig": name, "delisted": is_delisted,
                       "bull": bool(spy_bull.get(d, False))}
                # entry A = buy-the-drop (at close of trigger day); entry B = wait-for-turn (first up-day within 5d)
                turn_pos = None
                for k in range(1, 6):
                    if pos + k < len(cal) and bool(up.get(cal[pos + k], False)):
                        turn_pos = pos + k; break
                ok = False
                for h in HORIZONS:
                    base = c.get(d)
                    ev = c_ff.get(cal[pos + h]) if pos + h < len(cal) else np.nan
                    if pd.notna(base) and pd.notna(ev):
                        rec[f"A{h}"] = clip(ev / base - 1.0 - spy_fwd[h].get(d, 0.0)); ok = True
                    else:
                        rec[f"A{h}"] = np.nan
                    if turn_pos is not None and turn_pos + h < len(cal):
                        tb = c.get(cal[turn_pos]); tv = c_ff.get(cal[turn_pos + h])
                        rec[f"B{h}"] = clip(tv / tb - 1.0 - spy_fwd[h].get(cal[turn_pos], 0.0)) if pd.notna(tb) and pd.notna(tv) else np.nan
                    else:
                        rec[f"B{h}"] = np.nan
                if ok:
                    events.append(rec)
    print(f"  processed {min(i+BATCH,len(universe))}/{len(universe)}  events so far {len(events)}", flush=True)

E = pd.DataFrame(events)
print(f"\ntotal capitulation EDGE-events: {len(E)}  ({E['delisted'].mean()*100:.0f}% on names that later delisted)", flush=True)
print("(demeaned = event fwd return minus SPY over same window, winsorized to [-100%,+150%]; entry A=buy-the-drop, B=wait-for-turn)", flush=True)


def report(sub, label):
    print(f"\n--- {label}  (n_events={len(sub)}) ---", flush=True)
    print(f"  {'trigger':7}{'reg':5}{'h':>3}{'entry':>6}{'n':>7}{'mean%':>8}{'median%':>9}{'win%':>7}{'t_mo':>7}", flush=True)
    for name in ["zdrop", "rsi25", "low52", "dd25"]:
        for reg, rmask in [("bull", sub["bull"]), ("bear", ~sub["bull"])]:
            s = sub[(sub["trig"] == name) & rmask]
            if len(s) < 30:
                continue
            for entry in ["A", "B"]:
                for h in HORIZONS:
                    col = f"{entry}{h}"; x = s[[col, "date"]].dropna()
                    if len(x) < 30:
                        continue
                    mo = x.set_index("date")[col].resample("ME").mean().dropna()
                    t = mo.mean() / (mo.std(ddof=1) / np.sqrt(len(mo))) if len(mo) > 2 else np.nan
                    lbl = "drop" if entry == "A" else "turn"
                    print(f"  {name:7}{reg:5}{h:>3}{lbl:>6}{len(x):>7}{x[col].mean()*100:>8.2f}"
                          f"{x[col].median()*100:>9.2f}{(x[col]>0).mean()*100:>7.1f}{t:>7.2f}", flush=True)


report(E, "ALL names (survivorship-aware)")
report(E[~E["delisted"]], "SURVIVORS ONLY (biased — for gap estimate)")
