"""M&A EVENT EXTRACTOR — LOCAL LLM (qwen2.5:14b via on-box Ollama, fully offline).

Goal: turn noisy M&A-ish headlines into structured deal events so a backtest can SELL THE VALUE PICK
INTO THE ACQUISITION POP (not at month-end, not via _ret_delist) AND avoid the round-trip when an offer
is later WITHDRAWN. Regex can't do this — one "Workday buyout" article is tagged to WDAY/ADBE/CRM/NOW/SAP
(only WDAY is the target); "Cash Tender Offers for its Senior Notes" is a bond tender, not an M&A; and
"Diana Shipping withdraws Genco bid" (the exact withdrawal case we must catch) all look alike to a keyword.

Per (ticker, headline) the model decides:
  is_target : is THIS ticker's company the one being ACQUIRED / taken private? (false for acquirer, peer
              mention, analyst/thematic piece, or a debt/notes tender offer)
  status    : announced | completed | withdrawn | rumored | none
  acquirer  : buyer name or null
  offer_price: per-share cash offer ($) if stated in the headline, else null
  confidence: 0..1

Candidate set = (tight target-regex  UNION  cat_llm='ma' OR cat_auto='ma')  AND  ticker in Fundamental
universe. ~6.5k headlines / ~630 tickers. Title-only (NewsItem has no body); headlines usually state the
deal + price + a status verb, which is enough for the exit signal.

Idempotent + resumable: appends one JSON line per NewsItem.id to .data/ma_events.jsonl; a re-run skips
ids already present. Concurrent (Ollama serves parallel requests). Nothing leaves the box.

Run (validate first, then full detached in the celery worker which has egress + isn't reaped):
  docker exec rotation-backend-1 python -u /app/ma_extract.py --sample 30
  setsid nohup python -u /app/ma_extract.py --workers 6 > /app/.data/ma_extract.log 2>&1 < /dev/null &
"""
import django, os, sys, json, time, urllib.request
os.environ.setdefault("DJANGO_SETTINGS_MODULE", "rotation.settings"); django.setup()
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from django.db import connection
from django.db.models import Q

OUT = Path(__file__).resolve().parent / ".data" / "ma_events.jsonl"
STATUSES = {"announced", "completed", "withdrawn", "rumored", "none"}

# TARGET-focused regex (ticker as the acquired company / explicit deal terms) — the LLM does the real
# filtering; this only bounds the candidate pool so we don't run the model over all 693k headlines.
TIGHT = (r"(to be acquired|agrees? to be acquired|to acquire [A-Z]|acquisition of [A-Z]|buyout|takeover|"
         r"tender offer|to go private|per share in (an? )?(all-)?cash|all-cash (deal|transaction|"
         r"acquisition)|definitive (merger )?agreement|merger agreement|receives.{0,20}(offer|bid|"
         r"proposal)|to be taken private|to merge with)")

PROMPT = (
    "You extract merger & acquisition events from a stock news HEADLINE.\n"
    "You are given a TICKER, its COMPANY name, and a HEADLINE. Decide whether the headline reports an "
    "M&A deal in which THE GIVEN TICKER'S COMPANY (the one named under 'Company:') IS THE TARGET — the "
    "company being acquired, bought out, or taken private. The headline may refer to it by that company "
    "name rather than the ticker.\n\n"
    "Rules:\n"
    "- is_target = true ONLY if the ticker's company is the one being ACQUIRED / taken private.\n"
    "  false if the ticker is the ACQUIRER/buyer, only a peer or comparison mentioned, an analyst or "
    "thematic 'are mergers back?' piece, OR a DEBT/NOTES tender offer (bonds, senior notes — NOT a "
    "company acquisition).\n"
    "- status: 'announced' = a deal/offer/definitive agreement is announced or a firm bid proposed; "
    "'completed' = deal closed/completed/finalized; 'withdrawn' = offer withdrawn/terminated/rejected/"
    "deal collapsed/called off; 'rumored' = only exploring/considering/reportedly weighing/in talks; "
    "'none' = not a target M&A event.\n"
    "- acquirer: the buying company's name, or null.\n"
    "- offer_price: the per-share cash offer price as a plain number if the headline states one (e.g. "
    "'$25.50 per share' -> 25.5), else null.\n"
    "- confidence: 0.0 to 1.0.\n\n"
    "Return ONLY a JSON object, no prose:\n"
    '{{"is_target": true|false, "status": "announced|completed|withdrawn|rumored|none", '
    '"acquirer": "name or null", "offer_price": number or null, "confidence": 0.0}}\n\n'
    "Ticker: {ticker}\nCompany: {company}\nHeadline: {title}\nJSON:"
)


def _clean(d):
    """Coerce the model's JSON into the canonical shape; tolerate string nulls / stray fields."""
    out = {"is_target": bool(d.get("is_target")),
           "status": str(d.get("status", "none")).strip().lower(),
           "acquirer": d.get("acquirer"), "offer_price": d.get("offer_price"),
           "confidence": d.get("confidence")}
    if out["status"] not in STATUSES:
        out["status"] = "none"
    if isinstance(out["acquirer"], str) and out["acquirer"].strip().lower() in ("null", "none", "n/a", ""):
        out["acquirer"] = None
    try:
        out["offer_price"] = float(out["offer_price"]) if out["offer_price"] not in (None, "null", "") else None
    except (TypeError, ValueError):
        out["offer_price"] = None
    try:
        out["confidence"] = float(out["confidence"])
    except (TypeError, ValueError):
        out["confidence"] = None
    if not out["is_target"]:
        out["status"] = "none"          # not a target event -> force none for a clean downstream filter
    return out


def make_extract(url, model):
    def extract(ticker, title, company=None):
        body = json.dumps({
            "model": model,
            "prompt": PROMPT.format(ticker=(ticker or "?"), company=(company or "(unknown)"),
                                    title=(title or "")[:400]),
            "stream": False, "format": "json",
            "options": {"temperature": 0, "num_predict": 120, "top_p": 0.1},
        }).encode()
        req = urllib.request.Request(url, data=body, headers={"Content-Type": "application/json"})
        r = urllib.request.urlopen(req, timeout=180)
        return _clean(json.loads(json.load(r)["response"]))
    return extract


def candidates():
    from core.models import NewsItem, Fundamental
    univ = set(Fundamental.objects.values_list("ticker", flat=True).distinct())
    q = (Q(title__iregex=TIGHT) | Q(cat_llm="ma") | Q(cat_auto="ma")) & Q(ticker__in=univ)
    return list(NewsItem.objects.filter(q).values_list("id", "ticker", "title", "dt").order_by("-dt"))


def load_done():
    done = set()
    if OUT.exists():
        for line in OUT.read_text(encoding="utf-8").splitlines():
            try:
                done.add(json.loads(line)["id"])
            except Exception:
                pass
    return done


def main(sample=None, workers=6, model="qwen2.5:14b", host="host.docker.internal:11434"):
    with connection.cursor() as c:
        c.execute("SET max_parallel_workers_per_gather=0")     # never spawn the toxic Candle-style fan-out
    url = f"http://{host}/api/generate"
    extract = make_extract(url, model)

    try:
        t0 = time.time()
        print(f"warming {model} @ {host} ...", flush=True)
        demo = extract("KLAC", "Acme Corp to be acquired by BigCo for $42.00 per share in all-cash deal")
        print(f"  warm -> {demo}  ({time.time()-t0:.1f}s)", flush=True)
    except Exception as e:
        print(f"ABORT: cannot reach Ollama at {url}: {e}", flush=True)
        return

    names_path = OUT.parent / "ticker_names.json"
    NAMES = json.loads(names_path.read_text()) if names_path.exists() else {}
    print(f"company-name map loaded: {len(NAMES)} tickers"
          + ("" if NAMES else "  (WARN: run fetch_ticker_names.py first for small-cap recall)"), flush=True)
    rows = candidates()
    done = load_done()
    rows = [r for r in rows if r[0] not in done]
    if sample:
        rows = rows[:sample]
    total = len(rows)
    print(f"candidates to extract: {total} (already done {len(done)}; workers={workers}; sample={sample})",
          flush=True)
    if not total:
        print("nothing to do", flush=True); return

    from collections import Counter
    stat = Counter(); ntarget = 0; done_n = 0; t0 = time.time()
    fh = OUT.open("a", encoding="utf-8")

    def work(row):
        _id, ticker, title, dt = row
        try:
            ev = extract(ticker, title, NAMES.get(ticker))
            ev.update({"id": _id, "ticker": ticker, "company": NAMES.get(ticker),
                       "dt": dt.isoformat() if dt else None, "title": title})
            return ev
        except Exception:
            return None                # transient — leave unwritten so a re-run retries it

    with ThreadPoolExecutor(max_workers=workers) as ex:
        for ev in ex.map(work, rows):
            done_n += 1
            if ev is None:
                continue
            fh.write(json.dumps(ev, default=str) + "\n"); fh.flush()
            stat[ev["status"]] += 1
            if ev["is_target"]:
                ntarget += 1
            if done_n % 200 == 0:
                rate = done_n / (time.time() - t0)
                print(f"  {done_n}/{total}  {rate:.1f}/s  eta {(total-done_n)/rate/60:.0f}m  "
                      f"targets={ntarget}  status={dict(stat)}", flush=True)
    fh.close()
    dt = time.time() - t0
    print(f"DONE {done_n}/{total} in {dt/60:.1f}m ({done_n/dt:.1f}/s)  targets={ntarget}", flush=True)
    print(f"  status dist: {dict(stat)}", flush=True)
    print(f"  -> {OUT}", flush=True)

    if sample:      # show the actual extractions for eyeballing prompt quality
        print("\n=== SAMPLE EXTRACTIONS (is_target only) ===", flush=True)
        for line in OUT.read_text(encoding="utf-8").splitlines()[-sample:]:
            e = json.loads(line)
            if e["is_target"]:
                print(f"  {e['ticker']:6} [{e['status']:9}] ${e['offer_price']} <- {e['acquirer']}  "
                      f"| {e['title'][:70]}", flush=True)


if __name__ == "__main__":
    kw, a = {}, sys.argv
    for i, x in enumerate(a):
        if x == "--sample" and i + 1 < len(a): kw["sample"] = int(a[i + 1])
        elif x == "--workers" and i + 1 < len(a): kw["workers"] = int(a[i + 1])
        elif x == "--model" and i + 1 < len(a): kw["model"] = a[i + 1]
        elif x == "--host" and i + 1 < len(a): kw["host"] = a[i + 1]
    main(**kw)
