#!/usr/bin/env python3
"""Can we raise FLAGSHIP-B's win rate without gutting frequency? Test filters on the B(capitulation w15)->gap-up
@3b trade set and show, for each: #trades, win%, avg%, and the compounded $10k (all-in sequential, net ~10bps)
so the win-rate-vs-frequency tradeoff on TOTAL RETURN is explicit.
Run: MSYS_NO_PATHCONV=1 docker exec rotation-backend-1 python -u /app/b_winrate.py"""
import os, json, bisect, warnings
warnings.filterwarnings("ignore")
os.environ.setdefault("DJANGO_SETTINGS_MODULE", "rotation.settings")
import django
django.setup()
import numpy as np
import h4_on_signals_study as S
import h4_study as H
from seq_fundamental_study import load_candles
from studies import SIGNALS as STUDY_SIGNALS
from intraday_data import get_4h
from core.models import OptionSnapshot

_n, seq_fn = STUDY_SIGNALS["seq_rsi20_ad_rising_rsi"]
FEE = 0.001   # ~10bps round-trip (big-cap)


def asof(ser, d):
    ds = [x[0] for x in ser]; i = bisect.bisect_right(ds, d) - 1
    return ser[i][1] if i >= 0 else None


# capitulation w15 windows
daily = load_candles(S._stock_universe())
allowed = {}
for tk, df in daily.items():
    if len(df) < 60:
        continue
    sig = seq_fn(df).fillna(False); idx = df.index
    dv = (df["Close"] * df["Volume"]).rolling(20).mean()
    s = set()
    for i in np.flatnonzero(sig.values):
        if dv.iloc[i] < 5e6:
            continue
        for j in range(i, min(i + 15, len(idx))):
            s.add(idx[j].date())
    if s:
        allowed[tk] = s

ups = {}
for line in open("/app/.data/analyst_ratings.jsonl"):
    r = json.loads(line); t, d, pt = r.get("ticker"), r.get("date"), r.get("adjusted_price_target") or r.get("price_target")
    if t and d and pt:
        ups.setdefault(t, []).append((d, float(pt)))
for t in ups:
    ups[t].sort()
skew = {}
for r in OptionSnapshot.objects.filter(iv_skew__isnull=False).values_list("ticker", "date", "iv_skew"):
    skew.setdefault(r[0], []).append((r[1], float(r[2])))
for t in skew:
    skew[t].sort()

rows = []   # (r3, gapmag, prior5, up, sk)
for tk, dates in allowed.items():
    df = get_4h(tk, 5, False)
    if df is None or len(df) < 120:
        continue
    entry, mag = H.SIGNALS["mo_gap_up"]["fn"](df)
    c = df["Close"].values; ts = df.index
    for i in range(5, len(c) - 3):
        if not entry[i] or ts[i].date() not in dates:
            continue
        r3 = (c[i + 3] - c[i]) / c[i] * 100
        pri = c[i] / c[i - 5] - 1.0
        d = ts[i].date().isoformat()
        up = (asof(ups[tk], d) / c[i] - 1.0) if (tk in ups and c[i] > 0 and asof(ups[tk], d)) else None
        sk = asof(skew[tk], ts[i].date()) if tk in skew else None
        rows.append((r3, float(mag[i]), pri, up, sk))

sk_all = sorted(r[4] for r in rows if r[4] is not None)
sk_med = sk_all[len(sk_all) // 2] if sk_all else 0
sk_p60 = sk_all[int(len(sk_all) * 0.6)] if sk_all else 0
gm = sorted(r[1] for r in rows)
gm_med = gm[len(gm) // 2] if gm else 0


def report(lab, keep):
    a = np.array([r[0] for r in rows if keep(r)], float)
    if len(a) < 5:
        print(f"  {lab:30}{'n<5':>7}", flush=True); return
    net = a / 100 - FEE
    fac = np.prod(1 + net)
    print(f"  {lab:30}{len(a):>6}{(a > 0).mean() * 100:>7.0f}%{a.mean():>+8.2f}{10000 * fac:>14,.0f}", flush=True)


print(f"B gap-up@3b entries: {len(rows)} (skew cov {len(sk_all)}, analyst cov {sum(1 for r in rows if r[3] is not None)})\n", flush=True)
print(f"  {'filter':30}{'n':>6}{'win':>8}{'avg%':>8}{'$10k ->':>14}", flush=True)
report("base (all)",               lambda r: True)
report("iv_skew top50%",           lambda r: r[4] is not None and r[4] >= sk_med)
report("iv_skew top40%",           lambda r: r[4] is not None and r[4] >= sk_p60)
report("analyst upside>=25%",      lambda r: r[3] is not None and r[3] >= 0.25)
report("analyst 25-50%",           lambda r: r[3] is not None and 0.25 <= r[3] < 0.50)
report("not-knife (prior5d>-20%)", lambda r: r[2] > -0.20)
report("gap-mag >= median",        lambda r: r[1] >= gm_med)
report("skew top50 & upside>=25",  lambda r: r[4] is not None and r[4] >= sk_med and r[3] is not None and r[3] >= 0.25)
report("skew top50 & not-knife",   lambda r: r[4] is not None and r[4] >= sk_med and r[2] > -0.20)
