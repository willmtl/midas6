"""
Scrape the FULL Finviz taxonomy: 11 sectors -> ~145 industries -> ALL stocks.

Finviz classifies every listed name into 11 sectors and ~145 industries (the Morningstar/Yahoo
scheme). We scrape the SCREENER filtered by each of the 11 sectors (f=sec_<slug>), paginating every
page (r=1,21,41,...). Each screener row carries data-boxover-ticker / -company / -industry / -country,
so ONE pass over the 11 sectors tags every stock with BOTH its sector (from the filter) and its
industry (from the row) — no separate 148-industry scrape needed, and it self-maps industry->sector.

Server-side GET with a browser UA gets through Cloudflare from the worker (the headless-Playwright
path was walled; a plain urllib GET is not). Run DETACHED in rotation-celery-worker-1 (has egress):

  MSYS_NO_PATHCONV=1 docker exec -d rotation-celery-worker-1 \
    sh -c 'cd /app && python -u scrape_finviz_universe.py > /app/.data/finviz_universe.log 2>&1'

Output: /app/.data/finviz_universe.json
  { generated, sectors:{Sector:{Industry:{slug,count,tickers:[{t,company,country}]}}},
    by_ticker:{TICKER:{sector,industry,company,country}}, totals:{...} }
"""
import urllib.request
import re
import json
import time
import os

OUT = "/app/.data/finviz_universe.json"
UA = ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
      "(KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36")

SECTORS = [
    ("basicmaterials", "Basic Materials"),
    ("communicationservices", "Communication Services"),
    ("consumercyclical", "Consumer Cyclical"),
    ("consumerdefensive", "Consumer Defensive"),
    ("energy", "Energy"),
    ("financial", "Financial"),
    ("healthcare", "Healthcare"),
    ("industrials", "Industrials"),
    ("realestate", "Real Estate"),
    ("technology", "Technology"),
    ("utilities", "Utilities"),
]

PAGE = 20            # Finviz screener rows per page
MAX_PAGES = 400      # hard backstop (400*20 = 8000 names/sector; no sector is near that)
SLEEP = 0.35         # polite delay between requests


def fetch(url, tries=4):
    last = None
    for i in range(tries):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": UA, "Accept": "text/html"})
            return urllib.request.urlopen(req, timeout=30).read().decode("utf-8", "ignore")
        except Exception as e:  # transient network / rate-limit -> backoff
            last = e
            time.sleep(1.5 * (i + 1))
    raise last


def parse_rows(html):
    """Yield (ticker, company, industry, country) for every screener row on the page (deduped upstream)."""
    out = []
    for m in re.finditer(r'data-boxover-ticker="([^"]+)"([^>]*)>', html):
        t = m.group(1).strip()
        rest = m.group(2)
        comp = re.search(r'data-boxover-company="([^"]*)"', rest)
        ind = re.search(r'data-boxover-industry="([^"]*)"', rest)
        ctry = re.search(r'data-boxover-country="([^"]*)"', rest)
        out.append((t,
                    comp.group(1).strip() if comp else "",
                    ind.group(1).strip() if ind else "",
                    ctry.group(1).strip() if ctry else ""))
    return out


def total_count(html):
    m = re.search(r'(\d[\d,]*)\s*Total', html)
    return int(m.group(1).replace(",", "")) if m else None


def scrape_sector(slug, name):
    """Paginate the whole sector. Returns {ticker: {company, industry, country}}."""
    names = {}
    total = None
    r = 1
    pages = 0
    while pages < MAX_PAGES:
        url = f"https://finviz.com/screener.ashx?v=111&f=sec_{slug}&r={r}"
        html = fetch(url)
        if total is None:
            total = total_count(html)
        rows = parse_rows(html)
        new = 0
        for t, comp, ind, ctry in rows:
            if t not in names:
                names[t] = {"company": comp, "industry": ind, "country": ctry}
                new += 1
        pages += 1
        got = len(names)
        print(f"  [{name}] r={r:<5} page rows={len(rows):<3} new={new:<3} total_so_far={got}"
              f"{'' if total is None else f'/{total}'}", flush=True)
        # stop conditions: reached the reported total, or a page added nothing (end of list)
        if total is not None and got >= total:
            break
        if new == 0 and pages > 1:
            break
        r += PAGE
        time.sleep(SLEEP)
    return names, total


def main():
    t0 = time.time()
    sectors_out = {}
    by_ticker = {}
    print(f"=== Finviz universe scrape :: {len(SECTORS)} sectors ===", flush=True)
    for slug, name in SECTORS:
        print(f"[{name}] starting...", flush=True)
        names, total = scrape_sector(slug, name)
        # group into industries
        inds = {}
        for t, d in names.items():
            ind = d["industry"] or "Unknown"
            inds.setdefault(ind, []).append({"t": t, "company": d["company"], "country": d["country"]})
            # a stock can only be in one sector; if Finviz ever double-lists, first sector wins
            by_ticker.setdefault(t, {"sector": name, "industry": ind,
                                     "company": d["company"], "country": d["country"]})
        sectors_out[name] = {
            ind: {"count": len(lst), "tickers": sorted(lst, key=lambda x: x["t"])}
            for ind, lst in sorted(inds.items())
        }
        rep = "" if total is None else f" (reported {total})"
        print(f"[{name}] DONE: {len(names)} stocks across {len(inds)} industries{rep}\n", flush=True)

    totals = {
        "stocks": len(by_ticker),
        "sectors": len(sectors_out),
        "industries": sum(len(v) for v in sectors_out.values()),
        "by_sector": {s: sum(i["count"] for i in inds.values()) for s, inds in sectors_out.items()},
    }
    out = {
        "generated": time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime()),
        "source": "finviz.com/screener (v=111, per-sector filter, all pages)",
        "totals": totals,
        "sectors": sectors_out,
        "by_ticker": by_ticker,
    }
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w") as f:
        json.dump(out, f, indent=1)
    print("=== SUMMARY ===", flush=True)
    for s, n in totals["by_sector"].items():
        print(f"  {s:<24} {n:>5} stocks  {len(sectors_out[s]):>3} industries", flush=True)
    print(f"TOTAL {totals['stocks']} stocks | {totals['industries']} industry-buckets | "
          f"{time.time()-t0:.0f}s -> {OUT}", flush=True)


if __name__ == "__main__":
    main()
