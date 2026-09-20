#!/usr/bin/env python3
"""FDA CATALYST — data step 1. Pull novel-drug approval ACTION DATES from openFDA (api.fda.gov/drug/drugsfda, free,
full history) -> /app/.data/fda_approvals.json. Keeps ONLY real catalysts: original NDA/BLA approvals (submission_type
ORIG) + EFFICACY supplements (new indications). Drops generics (ANDA) and non-efficacy supplements (LABELING/MANUF/
REMS). Each event = one (ticker-mappable) FDA action with a dated action_date. Idempotent (overwrites the json).
Run: MSYS_NO_PATHCONV=1 docker exec rotation-backend-1 python -u /app/fetch_openfda_approvals.py"""
import os, json, urllib.request, datetime as dt
os.environ.setdefault("DJANGO_SETTINGS_MODULE", "rotation.settings")
import django; django.setup()

OUT = "/app/.data/fda_approvals.json"
BASE = "https://api.fda.gov/drug/drugsfda.json"


def _get(url, tries=4):
    import time
    for i in range(tries):
        try:
            return json.loads(urllib.request.urlopen(url, timeout=45).read().decode())
        except Exception as e:
            if i == tries - 1:
                raise
            time.sleep(1.5 * (i + 1))


def _date(s):
    try:
        return dt.date.fromisoformat(f"{s[:4]}-{s[4:6]}-{s[6:8]}").isoformat()
    except Exception:
        return None


def fetch_prefix(prefix):
    """Paginate all applications whose application_number starts with prefix (NDA/BLA)."""
    apps = []
    skip = 0
    total = None
    while True:
        url = f"{BASE}?search=application_number:{prefix}*&limit=1000&skip={skip}"
        d = _get(url)
        if total is None:
            total = d.get("meta", {}).get("results", {}).get("total", 0)
            print(f"  {prefix}: total {total} applications", flush=True)
        res = d.get("results", [])
        apps.extend(res)
        skip += len(res)
        if not res or skip >= total or skip >= 25000:
            break
    return apps


def main():
    events = []
    for prefix in ("NDA", "BLA"):
        apps = fetch_prefix(prefix)
        for a in apps:
            appno = a.get("application_number", "")
            sponsor = a.get("sponsor_name", "")
            prods = a.get("products") or []
            brand = prods[0].get("brand_name") if prods else None
            for s in a.get("submissions", []):
                if s.get("submission_status") != "AP":
                    continue
                stype = s.get("submission_type")
                cls = s.get("submission_class_code")
                # keep original approvals + efficacy supplements only
                if stype == "ORIG":
                    kind = "ORIG"
                elif stype == "SUPPL" and cls == "EFFICACY":
                    kind = "sNDA"
                else:
                    continue
                d = _date(s.get("submission_status_date") or "")
                if not d:
                    continue
                events.append({"application_number": appno, "app_type": prefix, "kind": kind,
                               "sponsor_name": sponsor, "brand_name": brand, "action_date": d,
                               "submission_number": s.get("submission_number"),
                               "review_priority": s.get("review_priority"),
                               "class_desc": s.get("submission_class_code_description")})
    events.sort(key=lambda e: e["action_date"])
    json.dump(events, open(OUT, "w"), indent=1)
    n_orig = sum(1 for e in events if e["kind"] == "ORIG")
    n_snda = sum(1 for e in events if e["kind"] == "sNDA")
    sponsors = len({e["sponsor_name"] for e in events})
    print(f"\nWROTE {OUT}: {len(events)} events ({n_orig} ORIG + {n_snda} sNDA), {sponsors} sponsors, "
          f"{events[0]['action_date']}..{events[-1]['action_date']}", flush=True)


if __name__ == "__main__":
    main()
