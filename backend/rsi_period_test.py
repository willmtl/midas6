#!/usr/bin/env python3
"""RSI(14) vs RSI(10) for the oversold-cross bounce — the one honestly-surviving edge (bear-regime, short-horizon).
Same integrity harness as the discovery lab: survivorship-aware (covered ∪ delisted, forward returns through
delisting), split-adjusted, cross-sectionally demeaned vs an equal-weight UNIVERSE index, winsorized, honest
monthly-panel t, bull/bear split. Event = RSI<30 crossing down (prev>=30). Horizons 1d/3d/1w(5)/2w(10)/4w(21).
Run: MSYS_NO_PATHCONV=1 docker exec rotation-backend-1 python -u /app/rsi_period_test.py"""
import os, json, math
os.environ.setdefault("DJANGO_SETTINGS_MODULE", "rotation.settings")
import django; django.setup()
import numpy as np, pandas as pd
from pathlib import Path
from collections import defaultdict
from core.models import Candle, Sector, DelistedCompany

DVOL_FLOOR = 5e6; PRICE_FLOOR = 5.0; WINS = (-100.0, 150.0)
HZ = {"1d": 1, "3d": 3, "1w": 5, "2w": 10, "4w": 21}

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
bull = (spy_c > spy_c.rolling(200).mean())


def rsi(s, n):
    d = s.diff(); up = d.clip(lower=0).rolling(n).mean(); dn = (-d.clip(upper=0)).rolling(n).mean()
    return 100 - 100 / (1 + up / dn.replace(0, np.nan))


def _w(x):
    return float(min(max(x, WINS[0]), WINS[1]))


# pass 1: EW-universe daily index
sums = defaultdict(float); cnts = defaultdict(int)
close_cache = {}
BATCH = 300
for i in range(0, len(universe), BATCH):
    batch = universe[i:i + BATCH]
    rows = Candle.objects.filter(ticker__in=batch, date__gte="2014-06-01").values_list("ticker", "date", "close", "volume")
    df = pd.DataFrame(list(rows), columns=["ticker", "date", "close", "volume"])
    if df.empty:
        continue
    df["date"] = pd.to_datetime(df["date"]); df["close"] = df["close"].astype(float); df["volume"] = df["volume"].astype(float)
    for tk, g in df.groupby("ticker", sort=False):
        s = g.sort_values("date").set_index("date")
        close_cache[tk] = s
        r = s["close"].pct_change()
        r = r[(r > -0.9) & (r < 2.0)]
        for d, v in r.items():
            sums[d] += v; cnts[d] += 1
    if (i + 1) % 1500 == 0:
        print(f"  ew pass {min(i+BATCH,len(universe))}/{len(universe)}", flush=True)
idx = sorted(sums)
ew = (1.0 + pd.Series([sums[d] / cnts[d] if cnts[d] >= 20 else 0.0 for d in idx], index=pd.DatetimeIndex(idx))).cumprod()
print(f"EW index {len(ew)} days; {len(close_cache)} names cached", flush=True)

# pass 2: events for RSI(10) and RSI(14)
acc = defaultdict(lambda: [0, 0.0, 0, 0.0])          # (period,hz,reg,ym)->[n,sum_dem,wins,sum_raw]
for tk, s in close_cache.items():
    c = s["close"]
    if c.notna().sum() < 260:
        continue
    cff = c.ffill(limit=252)
    dvol20 = (c * s["volume"]).rolling(20, min_periods=10).mean()
    bench = ew.reindex(c.index).values
    bull_al = bull.reindex(c.index).values
    cv = cff.values; base = c.values; dv = dvol20.values; n = len(cv)
    for period in (10, 14):
        r = rsi(c, period)
        fire = ((r < 30) & (r.shift(1) >= 30)).values
        for k in np.where(fire)[0]:
            if base[k] <= PRICE_FLOOR or not (dv[k] >= DVOL_FLOOR) or not np.isfinite(bench[k]) or bench[k] <= 0:
                continue
            reg = "bull" if bool(bull_al[k]) else "bear"
            ym = f"{c.index[k].year}-{c.index[k].month:02d}"
            for hl, h in HZ.items():
                j = k + h
                if j >= n or not np.isfinite(bench[j]) or bench[j] <= 0:
                    continue
                raw = (cv[j] - base[k]) / base[k] * 100.0
                dem = _w(raw - (bench[j] / bench[k] - 1.0) * 100.0)
                a = acc[(period, hl, reg, ym)]
                a[0] += 1; a[1] += dem; a[2] += 1 if dem > 0 else 0; a[3] += _w(raw)

grp = defaultdict(dict)
for (p, hl, reg, ym), v in acc.items():
    grp[(p, hl, reg)][ym] = v
print(f"\n{'RSI':>4}{'reg':>5}{'hz':>4}{'n':>8}{'dem%':>8}{'abs%':>8}{'win%':>7}{'t_mo':>7}", flush=True)
for period in (10, 14):
    for reg in ("bear", "bull"):
        for hl in HZ:
            m = grp.get((period, hl, reg))
            if not m:
                continue
            tot_n = sum(v[0] for v in m.values())
            if tot_n < 100:
                continue
            tot_sum = sum(v[1] for v in m.values()); tot_w = sum(v[2] for v in m.values()); tot_raw = sum(v[3] for v in m.values())
            mm = [v[1] / v[0] for v in m.values() if v[0] > 0]
            t = (np.mean(mm) / (np.std(mm, ddof=1) / math.sqrt(len(mm)))) if len(mm) >= 6 and np.std(mm, ddof=1) > 0 else float("nan")
            print(f"  {period:>2}{reg:>5}{hl:>4}{tot_n:>8}{tot_sum/tot_n:>8.2f}{tot_raw/tot_n:>8.2f}{tot_w/tot_n*100:>7.1f}{t:>7.2f}", flush=True)
