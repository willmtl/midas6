"""Build a ticker -> company-name cache for the M&A extractor (ma_extract.py).

The local LLM knows major tickers (WDAY=Workday) but NOT obscure small-caps (GNK=Genco Shipping &
Trading) — which is exactly the flagship's universe. Without the company name it can't tell that a
headline about "Genco" concerns GNK, so it misses target/withdrawal events. This pulls the clean
General::Name field from EODHD for the M&A-candidate tickers and caches it, so ma_extract can pass
"Company: <name>" into the prompt. One cheap single-field call per ticker; idempotent (skips cached).

Run in the celery worker (has EODHD egress + isn't reaped):
  docker exec rotation-celery-worker-1 python -u /app/fetch_ticker_names.py
"""
import django, os, json, urllib.request, urllib.parse
os.environ.setdefault("DJANGO_SETTINGS_MODULE", "rotation.settings"); django.setup()
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from django.db import connection
from django.db.models import Q

OUT = Path(__file__).resolve().parent / ".data" / "ticker_names.json"
KEY = os.environ.get("EODHD_API_KEY")
TIGHT = (r"(to be acquired|buyout|takeover|tender offer|to go private|per share in|merger agreement|"
         r"to be taken private|definitive (merger )?agreement|acquisition of [A-Z])")


def eodhd_symbol(tk, eod):
    """Prefer the stored EODHD symbol; else map US/Canada to EODHD's exchange suffix."""
    if eod:
        return eod
    if tk.endswith(".TO") or tk.endswith(".V"):
        return tk                       # EODHD uses .TO / .V for Canada, same as ours
    return f"{tk}.US"


def fetch_name(sym):
    u = (f"https://eodhd.com/api/fundamentals/{urllib.parse.quote(sym)}"
         f"?api_token={KEY}&filter=General::Name&fmt=json")
    try:
        r = urllib.request.urlopen(u, timeout=25).read().decode().strip().strip('"')
        return r if r and r.lower() not in ("null", "na", "none") else None
    except Exception:
        return None


def main():
    if not KEY:
        print("EODHD_API_KEY not set — aborting", flush=True); return
    with connection.cursor() as c:
        c.execute("SET max_parallel_workers_per_gather=0")
    from core.models import NewsItem, Fundamental
    univ = {r["ticker"]: r["eodhd_symbol"] for r in
            Fundamental.objects.values("ticker", "eodhd_symbol").order_by("ticker", "-date")}
    cand = set(NewsItem.objects.filter(
        (Q(title__iregex=TIGHT) | Q(cat_llm="ma") | Q(cat_auto="ma")) & Q(ticker__in=univ.keys())
    ).values_list("ticker", flat=True).distinct())
    cache = json.loads(OUT.read_text()) if OUT.exists() else {}
    todo = [t for t in sorted(cand) if t not in cache]
    print(f"candidate tickers {len(cand)}; cached {len(cache)}; to fetch {len(todo)}", flush=True)

    def work(tk):
        return tk, fetch_name(eodhd_symbol(tk, univ.get(tk)))
    got = 0
    with ThreadPoolExecutor(max_workers=8) as ex:
        for i, (tk, nm) in enumerate(ex.map(work, todo), 1):
            if nm:
                cache[tk] = nm; got += 1
            if i % 100 == 0:
                OUT.write_text(json.dumps(cache, indent=0))
                print(f"  {i}/{len(todo)}  got={got}", flush=True)
    OUT.write_text(json.dumps(cache, indent=0))
    print(f"DONE names cached: {len(cache)} (fetched {got}/{len(todo)}) -> {OUT}", flush=True)
    for t in ["GNK", "FATE", "MHK", "WDAY", "INSP"]:
        print(f"  {t} -> {cache.get(t)}", flush=True)


if __name__ == "__main__":
    main()
