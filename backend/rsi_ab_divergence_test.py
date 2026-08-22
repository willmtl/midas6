#!/usr/bin/env python3
"""RSI(10) vs RSI(14) for Strategy-A dip / Strategy-B capitulation entries, PLUS an RSI bullish-DIVERGENCE signal
(price makes a lower low while RSI makes a higher low). Standalone, survivorship-aware (covered ∪ delisted, through
delisting), split-adjusted, cross-sectionally demeaned vs EW-universe, winsorized, honest monthly-panel t, bull/bear,
LONG. Answers: does RSI(14) help A/B, and does bullish divergence have any standalone long edge (at 10 or 14)?
Run: MSYS_NO_PATHCONV=1 docker exec rotation-backend-1 python -u /app/rsi_ab_divergence_test.py"""
import os, json, math
os.environ.setdefault("DJANGO_SETTINGS_MODULE", "rotation.settings")
import django; django.setup()
import numpy as np, pandas as pd
from pathlib import Path
from collections import defaultdict
from core.models import Candle, Sector, DelistedCompany

DVOL_FLOOR = 5e6; PRICE_FLOOR = 5.0; WINS = (-100.0, 150.0)
HZ = {"1w": 5, "2w": 10, "1mo": 21, "3mo": 63, "6mo": 126}

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


sums = defaultdict(float); cnts = defaultdict(int); cache = {}
for i in range(0, len(universe), 300):
    rows = Candle.objects.filter(ticker__in=universe[i:i + 300], date__gte="2014-06-01").values_list("ticker", "date", "close", "volume")
    df = pd.DataFrame(list(rows), columns=["ticker", "date", "close", "volume"])
    if df.empty:
        continue
    df["date"] = pd.to_datetime(df["date"]); df["close"] = df["close"].astype(float); df["volume"] = df["volume"].astype(float)
    for tk, g in df.groupby("ticker", sort=False):
        s = g.sort_values("date").set_index("date"); cache[tk] = s
        r = s["close"].pct_change(); r = r[(r > -0.9) & (r < 2.0)]
        for d, v in r.items():
            sums[d] += v; cnts[d] += 1
idx = sorted(sums)
ew = (1.0 + pd.Series([sums[d] / cnts[d] if cnts[d] >= 20 else 0.0 for d in idx], index=pd.DatetimeIndex(idx))).cumprod()
print(f"EW index {len(ew)} days; {len(cache)} names", flush=True)

acc = defaultdict(lambda: [0, 0.0, 0, 0.0])       # (sigkey,hz,reg,ym)->[n,sum_dem,wins,sum_raw]
for tk, s in cache.items():
    c = s["close"]
    if c.notna().sum() < 300:
        continue
    cff = c.ffill(limit=252); bench = ew.reindex(c.index).values; bull_al = bull.reindex(c.index).values
    dv = (c * s["volume"]).rolling(20, min_periods=10).mean().values
    cv = cff.values; base = c.values; n = len(cv)
    for p in (10, 14):
        r = rsi(c, p)
        events = {
            f"A_dip45_rsi{p}": (r < 45) & (r.shift(1) >= 45),                 # enter mild dip (A proxy)
            f"B_cap20_rsi{p}": (r < 20) & (r.shift(1) >= 20),                 # capitulation entry
            f"B_cap20turn_rsi{p}": (r >= 20) & (r.shift(1) < 20),             # turn up out of <20 (B turn)
            f"bulldiv_rsi{p}": (c <= c.rolling(20, min_periods=15).min()) & (r > r.shift(10)),  # lower low, RSI higher low
        }
        for sk, mask in events.items():
            for k in np.where(mask.fillna(False).values)[0]:
                if base[k] <= PRICE_FLOOR or not (dv[k] >= DVOL_FLOOR) or not np.isfinite(bench[k]) or bench[k] <= 0:
                    continue
                reg = "bull" if bool(bull_al[k]) else "bear"
                ym = f"{c.index[k].year}-{c.index[k].month:02d}"
                for hl, h in HZ.items():
                    j = k + h
                    if j >= n or not np.isfinite(bench[j]) or bench[j] <= 0:
                        continue
                    dem = _w((cv[j] - base[k]) / base[k] * 100.0 - (bench[j] / bench[k] - 1.0) * 100.0)
                    a = acc[(sk, hl, reg, ym)]
                    a[0] += 1; a[1] += dem; a[2] += 1 if dem > 0 else 0; a[3] += _w((cv[j] - base[k]) / base[k] * 100.0)

grp = defaultdict(dict)
for (sk, hl, reg, ym), v in acc.items():
    grp[(sk, hl, reg)][ym] = v
print(f"\n{'signal':22}{'reg':5}{'hz':>5}{'n':>8}{'dem%':>8}{'abs%':>8}{'win%':>7}{'t':>7}", flush=True)
order = [f"{b}_rsi{p}" for b in ("A_dip45", "B_cap20", "B_cap20turn", "bulldiv") for p in (10, 14)]
for sk in order:
    for reg in ("bear", "bull"):
        for hl in HZ:
            m = grp.get((sk, hl, reg))
            if not m:
                continue
            tn = sum(v[0] for v in m.values())
            if tn < 100:
                continue
            ts = sum(v[1] for v in m.values()); tw = sum(v[2] for v in m.values()); tr = sum(v[3] for v in m.values())
            mm = [v[1] / v[0] for v in m.values() if v[0] > 0]
            t = (np.mean(mm) / (np.std(mm, ddof=1) / math.sqrt(len(mm)))) if len(mm) >= 6 and np.std(mm, ddof=1) > 0 else float("nan")
            print(f"  {sk:22}{reg:5}{hl:>5}{tn:>8}{ts/tn:>8.2f}{tr/tn:>8.2f}{tw/tn*100:>7.1f}{t:>7.2f}", flush=True)
