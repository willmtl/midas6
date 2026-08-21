#!/usr/bin/env python3
"""Rebuild .data/splits_cache.json from the AUTHORITATIVE EODHD splits feed for the full flagship stock universe.
The existing cache is yfinance-built and never refreshed (_load_splits skips already-cached tickers), so it is
stale/incomplete (misses recent + some historical splits — e.g. PIPR 2026 4:1, LTHM 2024 conversion). This
fetches EODHD splits for every stock ticker we have candles for and writes a merged cache (EODHD authoritative,
yfinance kept only where EODHD returns nothing). Backs up the old cache first.
Run DETACHED in the egress container:
  MSYS_NO_PATHCONV=1 docker exec rotation-celery-worker-1 sh -c 'cd /app && setsid nohup python -u refresh_splits_eodhd.py > /app/.data/_splitrefresh.log 2>&1 &'"""
import os, json, time, shutil, urllib.request, urllib.parse
os.environ.setdefault("DJANGO_SETTINGS_MODULE", "rotation.settings")
import django; django.setup()

KEY = os.environ.get("EODHD_API_KEY") or os.environ.get("EODHD_TOKEN") or os.environ.get("EODHD_API_TOKEN")
CACHE = "/app/.data/splits_cache.json"
old = json.load(open(CACHE)) if os.path.exists(CACHE) else {}

# Universe = existing cache keys (built by pb_split_fix_study FROM the flagship universe) ∪ every flagship pick
# name ever recorded. This IS the flagship's stock universe; avoids a DISTINCT scan over the Candle hypertable
# (known /dev/shm DiskFull trap).
picks = set()
for hf in ["flagship_history", "flagship_history_core", "flagship_history_middle", "flagship_history_aggressive"]:
    p = f"/app/.data/studies/{hf}.json"
    if os.path.exists(p):
        J = json.load(open(p))
        for m in J.get("months", []):
            for pk in m.get("picks", []):
                picks.add(pk["ticker"])
# Strategy G universe: every analyst-covered US non-ETF name (same filter as strategy_g.build_universe)
gnames = set()
rp = "/app/.data/analyst_ratings.jsonl"
if os.path.exists(rp):
    from pathlib import Path as _P
    for line in _P(rp).read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            r = json.loads(line)
        except Exception:
            continue
        tk = r.get("ticker")
        if tk and r.get("price_target") and "." not in tk:
            gnames.add(tk)
universe = set(old.keys()) | picks | gnames
todo = sorted(t for t in universe if t not in old)          # only fetch names not already in the (refreshed) cache
print(f"universe {len(universe)} ({len(old)} cached + {len(picks)} picks + {len(gnames)} analyst) -> fetching {len(todo)} new  key={bool(KEY)}", flush=True)
tickers = todo


def eodhd_sym(tk):
    return tk if "." in tk else f"{tk}.US"


def fetch(tk):
    url = f"https://eodhd.com/api/splits/{urllib.parse.quote(eodhd_sym(tk))}?api_token={KEY}&fmt=json"
    try:
        with urllib.request.urlopen(url, timeout=30) as r:
            data = json.loads(r.read().decode())
    except Exception:
        return None
    out = {}
    for row in data or []:
        ds, sp = row.get("date"), row.get("split")
        if not ds or not sp:
            continue
        try:
            a, b = sp.split("/"); out[ds] = float(a) / float(b)
        except Exception:
            continue
    return out


new = dict(old)
fetched = added = 0
for i, tk in enumerate(tickers):
    sp = fetch(tk)
    if sp is None:
        continue                      # network error -> keep whatever we had
    fetched += 1
    before = new.get(tk, {})
    if sp != before:
        added += len(set(sp) - set(before))
    new[tk] = sp                      # EODHD authoritative (empty dict = confirmed no splits)
    if (i + 1) % 100 == 0:
        json.dump(new, open(CACHE + ".tmp", "w"))
        print(f"  {i+1}/{len(tickers)}  fetched={fetched} new_split_dates={added}", flush=True)
        time.sleep(0.1)

shutil.copy(CACHE, CACHE + ".bak-refresh") if os.path.exists(CACHE) else None
json.dump(new, open(CACHE, "w"), indent=0)
print(f"\nDONE: refreshed {fetched}/{len(tickers)} tickers; {added} new split-dates added vs old cache; "
      f"cache now {len(new)} tickers. Backup at splits_cache.json.bak-refresh", flush=True)
