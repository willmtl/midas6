#!/usr/bin/env python3
"""Does a CATALYST behind the gap-up make momentum continue harder? Buckets the A_plus gap-up momentum entry
(mo_gap_up on the high-vol liquid universe) by three PIT catalysts and reports the 3b + 8b forward bounce:
  1. analyst implied-UPSIDE level (target/price-1, dated Benzinga archive)   -> reuse h4_c_upside
  2. recent analyst CATALYST: an upgrade OR target-raise in the last ~7 days (rating/price_target_action)
  3. recent EARNINGS surprise sign: last EarningsEvent in ~10 days, grounded_score/eps_surprise good vs bad
News HEADLINES are NOT testable here (NewsItem not backfilled pre-2025) -> analyst + earnings are the backtestable
'news-like' catalysts with deep history. Run: MSYS_NO_PATHCONV=1 docker exec rotation-backend-1 python -u /app/a_catalyst.py [--fetch]"""
import os, sys, json, warnings, datetime as dt
warnings.filterwarnings("ignore")
os.environ.setdefault("DJANGO_SETTINGS_MODULE", "rotation.settings")
import django
django.setup()
import numpy as np
from django.db import connection
with connection.cursor() as cur:
    cur.execute("SET max_parallel_workers_per_gather = 0")
import h4_study as H
import h4_c_upside as U
from h4_on_signals_study import candidate_windows
from intraday_data import get_4h

FETCH = "--fetch" in sys.argv
ARCHIVE = "/app/.data/analyst_ratings.jsonl"


def load_events(path=ARCHIVE):
    """{ticker: sorted[(date, is_catalyst)]}; catalyst = analyst UPGRADE or price-target RAISE."""
    ev = {}
    with open(path) as f:
        for line in f:
            try:
                r = json.loads(line)
            except Exception:
                continue
            tk = r.get("ticker"); ds = r.get("date")
            if not tk or not ds:
                continue
            try:
                d = dt.date.fromisoformat(ds)
            except Exception:
                continue
            cat = (r.get("rating_action") == "upgrades") or (r.get("price_target_action") == "raises")
            ev.setdefault(tk, []).append((d, bool(cat)))
    for tk in ev:
        ev[tk].sort()
    return ev


def recent_catalyst(ev, tk, d, days=7):
    rec = ev.get(tk)
    if not rec:
        return "no_cov"
    lo = d - dt.timedelta(days=days)
    return "catalyst" if any(cat for (dd, cat) in rec if lo < dd <= d) else "none"


def load_earn(unis):
    from core.models import EarningsEvent
    e = {}
    for tk, rd, gs, su in EarningsEvent.objects.filter(ticker__in=unis).values_list(
            "ticker", "report_date", "grounded_score", "eps_surprise_pct"):
        e.setdefault(tk, []).append((rd, gs, su))
    for tk in e:
        e[tk].sort()
    return e


def earn_recent(e, tk, d, days=10):
    rec = e.get(tk)
    if not rec:
        return "no_earn"
    lo = d - dt.timedelta(days=days)
    hits = [(gs, su) for (rd, gs, su) in rec if lo < rd <= d]
    if not hits:
        return "no_recent"
    gs, su = hits[-1]
    score = gs if gs is not None else su
    if score is None:
        return "recent_unk"
    return "recent_good" if score > 0 else ("recent_bad" if score < 0 else "recent_flat")


def _t(a):
    a = np.asarray(a, float); a = a[np.isfinite(a)]
    if len(a) < 3:
        return None
    sd = a.std(ddof=1)
    return round(float(a.mean() / (sd / np.sqrt(len(a)))), 2) if sd > 0 else None


def row(a):
    a = np.asarray(a, float); a = a[np.isfinite(a)]
    return f"{len(a):>6} {a.mean():+.2f}% win{(a>0).mean()*100:.0f}% t={_t(a)}" if len(a) else "  (none)"


store = U.load_targets()
ev = load_events()
allowed, meta = candidate_windows("A_plus")
names = sorted(allowed)
earn = load_earn(names)
print(f"A_plus windows={meta['n_windows']} names={meta['n_names']}; analyst-cov names={sum(1 for t in names if t in store)}; "
      f"earn-cov names={sum(1 for t in names if t in earn)}; fetch={'on' if FETCH else 'off'}", flush=True)

# dim -> label -> {bars -> [returns]}
UP = [b[0] for b in U.UPSIDE_BUCKETS] + ["no_target"]
CAT = ["catalyst", "none", "no_cov"]
EAR = ["recent_good", "recent_bad", "recent_flat", "recent_unk", "no_recent", "no_earn"]
pools = {"upside": {b: {} for b in UP}, "catalyst": {b: {} for b in CAT},
         "earn": {b: {} for b in EAR}, "ALL": {}}
got = 0
for tk in names:
    df = get_4h(tk, 5, FETCH)
    if df is None or len(df) < 120:
        continue
    got += 1
    entry, _mag = H.SIGNALS["mo_gap_up"]["fn"](df)
    close = df["Close"].values
    dates = df.index.normalize()
    ad = allowed[tk]
    n = len(close)
    cand = [i for i in range(n) if entry[i] and dates[i].date() in ad]
    for i in sorted(H._episode_starts(cand, gap=H.GAP)):
        ep = float(close[i])
        if ep <= 0:
            continue
        d = dates[i].date()
        ub = U.bucket_upside(U.upside_asof(store, tk, d, ep))
        cb = recent_catalyst(ev, tk, d)
        eb = earn_recent(earn, tk, d)
        for bars in (3, 8):
            j = i + bars
            if j < n:
                r = (close[j] - ep) / ep * 100
                pools["ALL"].setdefault(bars, []).append(r)
                if ub in pools["upside"]:
                    pools["upside"][ub].setdefault(bars, []).append(r)
                pools["catalyst"][cb].setdefault(bars, []).append(r)
                pools["earn"][eb].setdefault(bars, []).append(r)
print(f"names_with_4h={got}\n", flush=True)

for bars in (3, 8):
    print(f"===== gap-up momentum continuation @ {bars}b  (baseline ALL: {row(pools['ALL'].get(bars, []))}) =====", flush=True)
    print("  -- by analyst UPSIDE level --", flush=True)
    for b in UP:
        print(f"    {b:12}{row(pools['upside'][b].get(bars, []))}", flush=True)
    print("  -- by recent analyst CATALYST (upgrade/target-raise <=7d) --", flush=True)
    for b in CAT:
        print(f"    {b:12}{row(pools['catalyst'][b].get(bars, []))}", flush=True)
    print("  -- by recent EARNINGS surprise (<=10d) --", flush=True)
    for b in EAR:
        print(f"    {b:12}{row(pools['earn'][b].get(bars, []))}", flush=True)
    print("", flush=True)
