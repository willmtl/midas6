#!/usr/bin/env python3
"""Backfill 13D (activist 5%+ stake) and 13G (passive institutional 5%+ stake) filings
from the SEC submissions API, indexed by the SUBJECT company's CIK (no CUSIP mapping
needed — that's why we use 13D/13G rather than raw 13F). `filed_date` is point-in-time.

Run: docker compose run --rm backend python -u fetch_sec_events.py --db
Flags: --jobs N  --limit N  --tickers AAPL,MSFT
"""
import warnings; warnings.filterwarnings("ignore")
import os, sys, json, time, urllib.request, urllib.error
from concurrent.futures import ProcessPoolExecutor

DEFAULT_JOBS = 6
UA = {"User-Agent": "webisoft-research william@webisoft.com"}
REQ_SLEEP = 0.12


def _get(url):
    return json.load(urllib.request.urlopen(urllib.request.Request(url, headers=UA), timeout=45))


def _group(form):
    f = (form or "").upper()
    if "13D" in f:
        return "13D"
    if "13G" in f:
        return "13G"
    return None


def _rows_from_arrays(arr):
    """Yield (group, filed_date, accession) from a submissions form/date/accession block."""
    forms = arr.get("form", [])
    dates = arr.get("filingDate", [])
    accs = arr.get("accessionNumber", [])
    for f, d, a in zip(forms, dates, accs):
        g = _group(f)
        if g:
            yield g, d, a


def _worker(payload):
    tickers, cikmap = payload
    import django
    os.environ.setdefault("DJANGO_SETTINGS_MODULE", "rotation.settings")
    try:
        django.setup()
    except Exception:
        pass
    from core.models import SecFiling
    saved = 0
    for tk in tickers:
        cik = cikmap.get(tk.upper())
        if not cik:
            continue
        try:
            sub = _get(f"https://data.sec.gov/submissions/CIK{cik:010d}.json")
            blocks = [sub["filings"]["recent"]]
            for extra in sub["filings"].get("files", []):          # older paginated filings
                try:
                    blocks.append(_get(f"https://data.sec.gov/submissions/{extra['name']}"))
                    time.sleep(REQ_SLEEP)
                except Exception:
                    pass
            for blk in blocks:
                for g, d, acc in _rows_from_arrays(blk):
                    _, created = SecFiling.objects.update_or_create(
                        accession=acc,
                        defaults={"ticker": tk, "form_group": g, "filed_date": d})
                    saved += 1
        except urllib.error.HTTPError as e:
            if e.code != 404:
                print(f"  {tk} (CIK {cik}): HTTP {e.code}")
        except Exception as e:
            print(f"  {tk}: ERROR {e}")
        time.sleep(REQ_SLEEP)
    return saved


def _cik_map():
    m = _get("https://www.sec.gov/files/company_tickers.json")
    return {v["ticker"].upper(): int(v["cik_str"]) for v in m.values()}


def _chunk(seq, n):
    n = max(1, n); k, r = divmod(len(seq), n); out = []; i = 0
    for j in range(n):
        sz = k + (1 if j < r else 0)
        if sz:
            out.append(seq[i:i + sz]); i += sz
    return out


def run(jobs=DEFAULT_JOBS, limit=None, tickers=None):
    from seq_fundamental_study import build_universe
    from django.db import connections
    uni = tickers or build_universe()
    if limit:
        uni = uni[:limit]
    cikmap = _cik_map()
    print(f"13D/13G backfill for {len(uni)} tickers | jobs: {jobs} | CIK map {len(cikmap)}")
    connections.close_all()
    import multiprocessing as mp
    ctx = mp.get_context("spawn")
    payloads = [(c, cikmap) for c in _chunk(uni, jobs * 2)]
    total = 0
    with ProcessPoolExecutor(max_workers=jobs, mp_context=ctx) as ex:
        for got in ex.map(_worker, payloads):
            total += got
    print(f"Done: {total} SecFiling rows upserted.")
    return total


if __name__ == "__main__":
    argv = sys.argv
    jobs = int(argv[argv.index("--jobs") + 1]) if "--jobs" in argv else DEFAULT_JOBS
    limit = int(argv[argv.index("--limit") + 1]) if "--limit" in argv else None
    tks = argv[argv.index("--tickers") + 1].split(",") if "--tickers" in argv else None
    if "--db" in argv:
        import django
        os.environ.setdefault("DJANGO_SETTINGS_MODULE", "rotation.settings")
        django.setup()
    run(jobs=jobs, limit=limit, tickers=tks)
