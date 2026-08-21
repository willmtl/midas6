#!/usr/bin/env python3
"""AUTHORITATIVE split-coverage check for Strategy C: pull the EODHD split list for every flagship PICK name
and diff against .data/splits_cache.json (the yfinance-populated, never-refreshed cache the flagship relies on).
A split EODHD reports that the cache LACKS, occurring AT/AFTER a pick month, = a genuinely contaminated pick
(as_traded not un-adjusted -> P/B understated at selection). Writes verdict to .data/pick_splits_audit.json.
Run DETACHED in the egress container:
  MSYS_NO_PATHCONV=1 docker exec rotation-celery-worker-1 sh -c 'cd /app && setsid nohup python -u verify_pick_splits_eodhd.py > /app/.data/_picksplit.log 2>&1 &'"""
import os, json, time, urllib.request, urllib.parse
os.environ.setdefault("DJANGO_SETTINGS_MODULE", "rotation.settings")
import django; django.setup()
import pandas as pd

KEY = os.environ.get("EODHD_API_KEY") or os.environ.get("EODHD_TOKEN") or os.environ.get("EODHD_API_TOKEN")
CACHE = json.load(open("/app/.data/splits_cache.json"))
J = json.load(open("/app/.data/studies/flagship_history.json"))

picks = []
for m in J["months"]:
    d = pd.Timestamp(m["date"])
    for p in m["picks"]:
        picks.append((p["ticker"], d))
names = sorted({t for t, _ in picks})
print(f"pick names: {len(names)}  key set: {bool(KEY)}", flush=True)


def eodhd_sym(tk):
    return tk if "." in tk else f"{tk}.US"          # foreign already carries an exchange suffix


def fetch_splits(tk):
    url = f"https://eodhd.com/api/splits/{urllib.parse.quote(eodhd_sym(tk))}?api_token={KEY}&fmt=json"
    try:
        with urllib.request.urlopen(url, timeout=30) as r:
            data = json.loads(r.read().decode())
    except Exception as e:
        return None, str(e)
    out = {}
    for row in data or []:
        ds = row.get("date"); sp = row.get("split")
        if not ds or not sp:
            continue
        try:
            a, b = sp.split("/"); ratio = float(a) / float(b)
        except Exception:
            continue
        out[ds] = ratio
    return out, None


def cache_has(tk, ds, days=15):
    when = pd.Timestamp(ds)
    for cd in CACHE.get(tk, {}):
        if abs((pd.Timestamp(cd) - when).days) <= days:
            return True
    return False


missing_all, contaminated, errs = [], [], []
for i, tk in enumerate(names):
    sp, err = fetch_splits(tk)
    if err:
        errs.append((tk, err)); continue
    for ds, ratio in sp.items():
        if not cache_has(tk, ds):
            missing_all.append((tk, ds, round(ratio, 4)))
            when = pd.Timestamp(ds)
            pre = sorted({p.date().isoformat() for t, p in picks if t == tk and p <= when})
            if pre:
                contaminated.append(dict(ticker=tk, split_date=ds, ratio=round(ratio, 4), picks_before=pre))
    if (i + 1) % 40 == 0:
        print(f"  {i+1}/{len(names)}", flush=True); time.sleep(0.1)

res = dict(
    checked=len(names), errors=len(errs),
    splits_missing_from_cache=len(missing_all),
    missing_list=[dict(ticker=t, date=d, ratio=r) for t, d, r in missing_all],
    contaminated_picks=contaminated,
    verdict="CLEAN" if not contaminated else "CONTAMINATED",
)
json.dump(res, open("/app/.data/pick_splits_audit.json", "w"), indent=2)
print(f"\n=== EODHD authoritative diff ===", flush=True)
print(f"checked {len(names)}  errors {len(errs)}  splits missing from cache {len(missing_all)}  "
      f"contaminated picks {len(contaminated)}  VERDICT {res['verdict']}", flush=True)
for c in contaminated:
    print(f"  {c['ticker']:6} split {c['split_date']} x{c['ratio']}  picked-before {c['picks_before']}", flush=True)
