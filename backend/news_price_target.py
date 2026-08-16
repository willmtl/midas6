#!/usr/bin/env python3
"""NEWS -> PRICE-TARGET EXTRACTOR + SIGNAL TEST. The only free historical analyst-target source we own: 52k
PRICE-TARGET-tagged NewsItem headlines (2020-08 -> now). Extract the target $ and direction from titles like
'TD Cowen Lowers Cerence (CRNC) PT to $12' / 'Keeps $155 PT', attribute via the (TICKER) in the title, then
test whether the signal predicts forward returns:
  - implied upside = target/close_at_event - 1  -> forward 21d & 63d return by upside quintile
  - RAISE vs CUT events -> forward return (does a target hike/cut lead price?)
NOTE: analyst targets are known to LAG price (analysts chase), so the prior is weak/contrarian. This quantifies it.
-> /app/.data/studies/news_price_target.json + BacktestResult[news_price_target].
Run: MSYS_NO_PATHCONV=1 docker exec rotation-celery-worker-1 python -u /app/news_price_target.py
"""
import os, re, json, warnings
warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
os.environ.setdefault("DJANGO_SETTINGS_MODULE", "rotation.settings")
import django
django.setup()
from pathlib import Path
from django.db import connection
from django.db.models import Q

OUT = Path(__file__).resolve().parent / ".data" / "studies" / "news_price_target.json"

RAISE = re.compile(r"\b(rais|lift|boost|hik|increas|upgrad|bump|up\b|reiterat.*(?:buy|outperform|overweight))", re.I)
CUT = re.compile(r"\b(lower|cut|trim|reduc|slash|downgrad|decreas|de-?rat)", re.I)
# target patterns (calibrated to the real headline formats)
P1 = re.compile(r"(?:PT|price target|target price|price objective|fair value)\s*(?:to|of|at|:|=)?\s*\$\s?([0-9][0-9,]{0,5}(?:\.[0-9]{1,2})?)", re.I)
P2 = re.compile(r"\$\s?([0-9][0-9,]{0,5}(?:\.[0-9]{1,2})?)\s*(?:PT|price target|price objective)", re.I)
TICK = re.compile(r"\(([A-Z]{1,6})\)")   # (CRNC) attribution


def extract(title):
    if not title:
        return None
    m = P1.search(title) or P2.search(title)
    if not m:
        return None
    try:
        tgt = float(m.group(1).replace(",", ""))
    except ValueError:
        return None
    if not (0.5 <= tgt <= 5000):          # plausible per-share target (drops $trillion-club noise)
        return None
    d = 1 if RAISE.search(title) else (-1 if CUT.search(title) else 0)
    return tgt, d


def main():
    with connection.cursor() as c:
        c.execute("SET max_parallel_workers_per_gather=0")
    from core.models import NewsItem, Candle
    pt = NewsItem.objects.filter(Q(tags__contains=["PRICE-TARGET"]) | Q(tags__contains=["PRICE TARGET"])) \
                         .values_list("ticker", "dt", "title")
    rows = list(pt)
    print(f"PRICE-TARGET headlines: {len(rows):,}", flush=True)
    events = []
    for tk, dt, title in rows:
        ex = extract(title)
        if not ex:
            continue
        tgt, d = ex
        # attribution: if the title has a (TICKER), require it to match the row ticker (reduces mislabels)
        tset = set(TICK.findall(title or ""))
        if tset and tk not in tset:
            continue
        events.append((tk, pd.Timestamp(dt).tz_localize(None), tgt, d))
    print(f"extracted target events: {len(events):,} ({100*len(events)/max(len(rows),1):.1f}% yield)  "
          f"distinct tickers: {len({e[0] for e in events})}", flush=True)

    # signal test: implied upside + forward returns, on tickers we have candles for
    tickers = sorted({e[0] for e in events})
    cand = {}
    for tk in tickers:
        q = list(Candle.objects.filter(ticker=tk, interval="1d").order_by("date").values_list("date", "close"))
        if len(q) > 60:
            s = pd.Series({pd.Timestamp(d): float(c) for d, c in q}).sort_index()
            cand[tk] = s
    recs = []
    for tk, dt, tgt, d in events:
        s = cand.get(tk)
        if s is None:
            continue
        past = s[s.index <= dt]
        if len(past) < 2:
            continue
        px0 = past.iloc[-1]
        if px0 <= 0:
            continue
        fut = s[s.index > dt]
        f21 = (fut.iloc[20] / px0 - 1) if len(fut) > 20 else np.nan
        f63 = (fut.iloc[62] / px0 - 1) if len(fut) > 62 else np.nan
        recs.append(dict(ticker=tk, date=str(dt.date()), target=tgt, dir=d,
                         px=px0, upside=tgt / px0 - 1, f21=f21, f63=f63))
    df = pd.DataFrame(recs)
    print(f"\nevents with price + forward data: {len(df):,}", flush=True)
    payload = {"n_headlines": len(rows), "n_events": len(events), "n_with_returns": len(df)}
    if len(df) > 50:
        df["upside"] = df["upside"].clip(-0.9, 5)
        df["q"] = pd.qcut(df["upside"], 5, labels=False, duplicates="drop")
        print("\n=== forward return by IMPLIED-UPSIDE quintile (Q0=lowest upside .. Q4=highest) ===", flush=True)
        print(f"  {'quintile':<10}{'avg upside':>12}{'fwd 21d':>10}{'fwd 63d':>10}{'n':>7}", flush=True)
        qtab = {}
        for q, g in df.groupby("q"):
            print(f"  Q{int(q):<9}{g['upside'].mean()*100:>11.1f}%{g['f21'].mean()*100:>9.2f}%{g['f63'].mean()*100:>9.2f}%{len(g):>7}", flush=True)
            qtab[f"Q{int(q)}"] = dict(upside=round(g['upside'].mean()*100,1), f21=round(g['f21'].mean()*100,2),
                                      f63=round(g['f63'].mean()*100,2), n=len(g))
        print("\n=== forward return by DIRECTION (target raise vs cut) ===", flush=True)
        dtab = {}
        for d, g in df.groupby("dir"):
            lab = {1: "RAISE", -1: "CUT", 0: "maintain"}[d]
            print(f"  {lab:<10} fwd21={g['f21'].mean()*100:+.2f}%  fwd63={g['f63'].mean()*100:+.2f}%  n={len(g)}", flush=True)
            dtab[lab] = dict(f21=round(g['f21'].mean()*100,2), f63=round(g['f63'].mean()*100,2), n=len(g))
        # top-minus-bottom quintile spread (the tradeable signal)
        if 0 in df["q"].values and 4 in df["q"].values:
            spread21 = df[df.q==4]["f21"].mean()*100 - df[df.q==0]["f21"].mean()*100
            print(f"\n  Q4-Q0 fwd21 spread: {spread21:+.2f}%   (positive = high-upside outperforms = momentum; negative = contrarian/lags)", flush=True)
            payload["q4_minus_q0_f21_pct"] = round(spread21, 2)
        payload["by_upside_quintile"] = qtab; payload["by_direction"] = dtab
    else:
        print("Too few events with returns to test a signal.", flush=True)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(payload, indent=2, default=str))
    try:
        from core.models import BacktestResult
        from django.utils import timezone
        BacktestResult.objects.update_or_create(kind="news_price_target",
            defaults={"payload": json.loads(json.dumps(payload, default=str)), "computed_at": timezone.now()})
        print("Saved BacktestResult[news_price_target]", flush=True)
    except Exception as e:
        print("DB save failed:", e, flush=True)
    print("DONE_NPT", flush=True)


if __name__ == "__main__":
    main()
