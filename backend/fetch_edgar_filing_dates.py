#!/usr/bin/env python3
"""AUTHORITATIVE point-in-time filing dates from SEC EDGAR, to REPAIR the unreliable EODHD `avail_date`
in FinancialReport (13.5% of EODHD rows are impossible — avail_date <= period_end — and missing-date rows
fall back to a flat period_end+45d guess that is too early for the fiscal-year-end 10-K).

For each US ticker we resolve its CIK (SEC company_tickers.json) and pull the submissions history
(data.sec.gov/submissions/CIK{cik}.json, incl. older paginated pages). For every PERIODIC report we record
(reportDate -> earliest authoritative filed date). Availability rule (user, 2026-09-21):
  "use the 10-Q unless a 10-K with a better [earlier] date is available"
i.e. per reportDate, availability = EARLIEST `filingDate` among the ORIGINAL periodic forms {10-Q, 10-K,
20-F, 40-F} (amendments 10-K/A, 10-Q/A used only if no original exists). The 10-Q covers interim quarters;
the fiscal-year-end quarter (no 10-Q) resolves to the 10-K; whichever is earlier wins when both report it.

Writes /app/.data/edgar_filing_dates.json = {ticker: {reportDate: {"filed": "YYYY-MM-DD", "form": F}}}.
The flagship engine loads this and prefers the EDGAR date over EODHD's avail_date (falling back to a
frequency-aware floor for names EDGAR can't serve — foreign filers, delisted names without a CIK match).

Idempotent. SEC fair-access: <10 req/s, descriptive User-Agent.
Run: MSYS_NO_PATHCONV=1 docker exec rotation-backend-1 python -u /app/fetch_edgar_filing_dates.py [--only-missing] [--tickers MUX,GNK]"""
import os, sys, json, time, urllib.request, urllib.error, argparse
os.environ.setdefault("DJANGO_SETTINGS_MODULE", "rotation.settings")
import django; django.setup()
from core.models import FinancialReport

UA = {"User-Agent": "rotation-research william2@webisoft.com"}
OUT = "/app/.data/edgar_filing_dates.json"
CIK_CACHE = "/app/.data/sec_ticker_cik.json"
PERIODIC = ("10-Q", "10-K", "20-F", "40-F")          # original periodic forms (interim + annual)
PERIODIC_AMEND = ("10-Q/A", "10-K/A", "20-F/A", "40-F/A")
REQ_GAP = 0.13                                        # ~7.7 req/s, under SEC's 10/s ceiling


def _get(url, tries=4):
    for i in range(tries):
        try:
            req = urllib.request.Request(url, headers=UA)
            with urllib.request.urlopen(req, timeout=45) as r:
                return json.load(r)
        except urllib.error.HTTPError as e:
            if e.code == 404:
                return None
            time.sleep(1.0 + i)                       # 429/5xx backoff
        except Exception:
            time.sleep(1.0 + i)
    return None


def load_cik_map():
    if os.path.exists(CIK_CACHE):
        return json.load(open(CIK_CACHE))
    m = _get("https://www.sec.gov/files/company_tickers.json")
    if not m:
        raise SystemExit("could not fetch SEC company_tickers.json")
    t2c = {v["ticker"].upper(): str(v["cik_str"]).zfill(10) for v in m.values()}
    json.dump(t2c, open(CIK_CACHE, "w"))
    print(f"cached {len(t2c)} ticker->CIK", flush=True)
    return t2c


def _collect(recent, acc, ek):
    """Fold one filings page (parallel arrays) into acc[reportDate]={form: earliest_filed} and, separately,
    collect the filing dates of earnings-release 8-Ks (form 8-K carrying Item 2.02 'Results of Operations')
    into `ek` — those hit EDGAR the same day as the earnings announcement, before the full 10-Q/10-K."""
    forms = recent.get("form", []); fds = recent.get("filingDate", [])
    rds = recent.get("reportDate", []); items = recent.get("items", [])
    for i, form in enumerate(forms):
        fd = fds[i] if i < len(fds) else None
        if not fd:
            continue
        rd = rds[i] if i < len(rds) else None
        if (form in PERIODIC or form in PERIODIC_AMEND) and rd:
            d = acc.setdefault(rd, {})
            if form not in d or fd < d[form]:
                d[form] = fd
        elif form == "8-K":
            it = items[i] if i < len(items) else ""
            if it and "2.02" in it:                                     # Results of Operations = earnings release
                ek.append(fd)


def edgar_report_dates(cik):
    """Return {reportDate: {"filed": earliest_authoritative_public_date, "form": source_form}} for one CIK.
    Availability = the EARNINGS 8-K (Item 2.02) filed between period-end and the full report when one exists
    (user 2026-09-21: "use the 8-K when available" — the condensed statements are public at the release),
    else the earliest 10-Q/10-K (interim vs FYE, amendments only if no original)."""
    s = _get(f"https://data.sec.gov/submissions/CIK{cik}.json")
    if not s:
        return {}
    acc = {}; ek = []
    _collect(s.get("filings", {}).get("recent", {}), acc, ek)
    for pg in s.get("filings", {}).get("files", []):          # older paginated history
        nm = pg.get("name")
        if nm:
            time.sleep(REQ_GAP)
            extra = _get(f"https://data.sec.gov/submissions/{nm}")
            if extra:
                _collect(extra, acc, ek)
    ek = sorted(set(ek))
    out = {}
    for rd, forms in acc.items():
        orig = {f: d for f, d in forms.items() if f in PERIODIC}          # prefer originals
        pool = orig or forms                                             # fall back to amendments
        form, filed = min(pool.items(), key=lambda kv: kv[1])           # earliest 10-Q/10-K = full-statement date
        cand = [d for d in ek if rd < d <= filed]                       # earnings 8-K for THIS period (release <= full)
        if cand and cand[0] < filed:
            filed, form = cand[0], "8-K"                                # earliest legit public disclosure
        out[rd] = {"filed": filed, "form": form}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only-missing", action="store_true", help="skip tickers already in the output cache")
    ap.add_argument("--tickers", default="", help="comma-separated subset (else all no-dot FinancialReport tickers)")
    args = ap.parse_args()

    t2c = load_cik_map()
    if args.tickers:
        universe = [t.strip().upper() for t in args.tickers.split(",") if t.strip()]
    else:
        universe = sorted({t for t in FinancialReport.objects.values_list("ticker", flat=True).distinct()
                           if "." not in t})            # US (no exchange suffix); foreign ADRs stay on EODHD+floor
    existing = json.load(open(OUT)) if os.path.exists(OUT) else {}
    if args.only_missing:
        universe = [t for t in universe if t not in existing]

    mapped = [t for t in universe if t.upper() in t2c]
    print(f"{len(universe)} target tickers, {len(mapped)} have a CIK match (rest fall back to EODHD+floor)", flush=True)

    out = dict(existing)
    done = hit = 0
    for tk in mapped:
        time.sleep(REQ_GAP)
        rd = edgar_report_dates(t2c[tk.upper()])
        if rd:
            out[tk] = rd
            hit += 1
        done += 1
        if done % 100 == 0:
            print(f"  {done}/{len(mapped)}  ({hit} with data)  last={tk}:{len(rd)} periods", flush=True)
            json.dump(out, open(OUT, "w"))             # checkpoint
    json.dump(out, open(OUT, "w"), indent=0)
    from collections import Counter
    forms = Counter(v["form"] for m in out.values() for v in m.values())
    print(f"WROTE {OUT}: {len(out)} tickers ({hit} fetched this run)", flush=True)
    print(f"  availability source-form: {dict(forms)}", flush=True)


if __name__ == "__main__":
    main()
