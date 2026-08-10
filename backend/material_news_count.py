"""Count news events that DEMONSTRABLY MATTERED — |abnormal reaction| over [d-1,d+1] above
thresholds — so we classify only those (most news is noise). Pure candle math, no LLM."""
import django, os
os.environ.setdefault("DJANGO_SETTINGS_MODULE", "rotation.settings"); django.setup()
import numpy as np, pandas as pd
from collections import defaultdict
from seq_fundamental_study import build_universe, load_candles
from core.models import NewsItem

BETA_WIN = 60; MIN_PRICE = 3.0
THRS = [4, 6, 8, 12]

tks = build_universe()
cds = load_candles(tks + ["SPY"])
spy = cds.get("SPY"); spy_ret = spy["Close"].pct_change()
news_by_tk = defaultdict(set)   # ticker -> set of news calendar dates
for r in NewsItem.objects.filter(title__gt="").values_list("ticker", "dt"):
    news_by_tk[r[0]].add(pd.Timestamp(r[1]).tz_localize(None).normalize())

counts = {t: 0 for t in THRS}; total_days = 0
for tk, df in cds.items():
    if tk == "SPY" or df is None or len(df) < BETA_WIN + 50:
        continue
    days = news_by_tk.get(tk)
    if not days:
        continue
    idx = df.index; close = df["Close"].values; n = len(close)
    mkt = spy["Close"].reindex(idx).ffill().values
    r = df["Close"].pct_change()
    both = pd.concat([r.rename("s"), spy_ret.rename("m")], axis=1).reindex(idx)
    beta = (both["s"].rolling(BETA_WIN).cov(both["m"]) / both["m"].rolling(BETA_WIN).var()).values
    for d in days:
        d0 = int(idx.searchsorted(d))
        rfrom, rto = d0 - 1, d0 + 1
        if rfrom < BETA_WIN or rto >= n or close[rto] < MIN_PRICE or mkt[rfrom] <= 0:
            continue
        b = beta[rfrom]; bc = min(max(b if np.isfinite(b) else 1.0, 0.0), 3.0)
        abn = abs((close[rto] / close[rfrom] - 1.0) - bc * (mkt[rto] / mkt[rfrom] - 1.0)) * 100
        if not np.isfinite(abn):
            continue
        total_days += 1
        for t in THRS:
            if abn >= t:
                counts[t] += 1
print("total ticker-news-days with a valid window:", total_days)
for t in THRS:
    print(f"  |abn reaction| >= {t:>2}% : {counts[t]:>6}  ({counts[t]/max(1,total_days)*100:.1f}% of news-days)")
