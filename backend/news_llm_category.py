"""
News EVENT-CATEGORY + OFF-TICKER classifier — LOCAL LLM (qwen2.5:14b via on-box Ollama), fully offline.

The title-only heuristic (compute_news_category.py -> cat_auto) lands ~69% of headlines in "other".
This refines the category with a local model so NOTHING leaves the machine (headlines are public but
the constraint is "everything runs local / nothing gets out of Docker"). Runs INSIDE the backend
container, calling the host Ollama at host.docker.internal:11435 (a scoped 2nd Ollama pointed at the
F: model store). Writes NewsItem.cat_llm; effective category = cat_llm or cat_auto.

ALSO judges OFF-TICKER: the ticker is passed to the model, which decides whether the headline is
actually about THIS ticker. If not (a market-wide/macro story, or a story about a DIFFERENT company
that merely surfaced in this symbol's feed, e.g. a "| Stock Movers" recap), off_ticker is set True and
the item is dropped from the ticker-specific news views. Both fields are written in one call.

MOVED-FIRST: rows with day_effect=True (the news that actually moved the stock) are classified before
the rest, so the valuable subset is done quickly. Idempotent + resumable: only rows not yet judged
(off_ticker IS NULL) are processed; re-run to continue. Concurrent (ThreadPoolExecutor) — Ollama
serves parallel requests.

Run: docker compose exec -T backend python -u news_llm_category.py [--moved-only] [--limit N]
                                                                   [--workers N] [--model qwen2.5:14b]
                                                                   [--host host.docker.internal:11435]
"""
import django, os, sys, json, time, urllib.request
os.environ.setdefault("DJANGO_SETTINGS_MODULE", "rotation.settings"); django.setup()
from concurrent.futures import ThreadPoolExecutor, as_completed
from core.models import NewsItem

CATS = ["earnings", "guidance", "analyst", "ma", "partnership", "contract", "product",
        "clinical", "legal", "offering", "dividend", "buyback", "insider", "mgmt", "macro", "other"]
CATSET = set(CATS)
# common ways the model phrases a label -> canonical
ALIAS = {
    "m&a": "ma", "m and a": "ma", "merger": "ma", "acquisition": "ma", "buyout": "ma",
    "analyst rating": "analyst", "rating": "analyst", "upgrade": "analyst", "downgrade": "analyst",
    "management": "mgmt", "executive": "mgmt", "personnel": "mgmt",
    "product/launch": "product", "launch": "product", "clinical/fda": "clinical", "fda": "clinical",
    "legal/lawsuit": "legal", "lawsuit": "legal", "offering/dilution": "offering", "dilution": "offering",
    "share buyback": "buyback", "repurchase": "buyback", "stake": "insider", "insider trading": "insider",
    "macroeconomic": "macro", "economy": "macro", "none": "other", "n/a": "other",
}

PROMPT = (
    "You are a financial news classifier. You are given a stock TICKER and a news headline.\n\n"
    "Do TWO things:\n"
    "1) Classify the headline into EXACTLY ONE category from this list:\n"
    "   " + ", ".join(CATS) + "\n"
    "2) Decide whether the headline is specifically ABOUT the given ticker's company. Answer N when it "
    "is a market-wide / macro story (Fed, CPI, tariffs, 'stocks plunge'), OR when it is really about a "
    "DIFFERENT company that just happens to appear in this ticker's feed (e.g. a multi-stock recap or "
    "'| Stock Movers' roundup, or the ticker is only mentioned in passing). Answer Y only when the "
    "company is the clear subject of the headline.\n\n"
    "Definitions:\n"
    "- earnings: quarterly results, EPS, revenue, beats/misses estimates\n"
    "- guidance: forward outlook/forecast raised, cut, or reaffirmed\n"
    "- analyst: analyst rating change, price-target change, initiation of coverage\n"
    "- ma: merger, acquisition, buyout, takeover, going private\n"
    "- partnership: collaboration, joint venture, strategic alliance, teaming up\n"
    "- contract: order/deal/award won, supply agreement, selected as vendor\n"
    "- product: product/service launch, unveiling, new release, availability\n"
    "- clinical: FDA action, clinical trial, drug/device approval or data\n"
    "- legal: lawsuit, investigation, settlement, regulatory charge, recall\n"
    "- offering: stock/debt offering, dilution, convertible notes, private placement\n"
    "- dividend: dividend declared/raised/cut, distribution\n"
    "- buyback: share repurchase program\n"
    "- insider: insider buy/sell, 13D/13G stake, institutional position change\n"
    "- mgmt: executive or board change (CEO/CFO appointed, resigns, hired)\n"
    "- macro: Fed, interest rates, tariffs, inflation, CPI, jobs — market-wide, not company-specific\n"
    "- other: none of the above / vague / opinion\n\n"
    "3) Judge the likely DIRECTION for the stock: up (bullish), flat (neutral/unclear), or down (bearish).\n"
    "4) Judge the IMPACT / materiality for the share price. BE STRICT — the large MAJORITY of headlines "
    "are 0 or 1; a 3 is RARE. Do not inflate.\n"
    "   0 = noise / routine / already-priced (most PR, recaps, awards, reiterated views, minor updates)\n"
    "   1 = minor (small or short-lived price effect)\n"
    "   2 = moderate (a genuine but not dramatic mover)\n"
    "   3 = MAJOR, thesis-changing ONLY (e.g. a merger/buyout, a large earnings surprise, a pivotal "
    "FDA decision, a guidance shock, bankruptcy). If unsure between 2 and 3, choose 2.\n"
    "5) Judge the HORIZON over which it should play out: day, week, month, or 3mo.\n\n"
    "Reply with EXACTLY five tokens separated by single spaces, nothing else:\n"
    "  <category> <up|flat|down> <0-3> <day|week|month|3mo> <Y|N>\n"
    "Example: earnings up 3 3mo Y\n\n"
    "Ticker: {ticker}\nHeadline: {title}\nAnswer:"
)

DIRMAP = {"up": 1, "bull": 1, "bullish": 1, "positive": 1, "pos": 1, "+1": 1, "1": 1,
          "flat": 0, "neutral": 0, "neu": 0, "none": 0, "0": 0,
          "down": -1, "bear": -1, "bearish": -1, "negative": -1, "neg": -1, "-1": -1}
HORIZONS = {"day", "week", "month", "3mo"}
HZALIAS = {"daily": "day", "weekly": "week", "monthly": "month", "quarter": "3mo",
           "quarterly": "3mo", "3month": "3mo", "3months": "3mo", "3-month": "3mo", "90d": "3mo"}


def normalize(resp):
    """Parse '<category> <up|flat|down> <0-3> <day|week|month|3mo> <Y|N>' ->
    (cat, off_ticker_bool, dir[-1..1], impact[0..3], horizon_str). Positional order is a hint only —
    every field is recovered by token scanning so a reordered/short reply still parses. Sensible
    defaults (other / on-ticker / flat / impact 0 / '') so a partial reply never crashes the run."""
    t = (resp or "").strip().lower().strip(".:\"' \n\t")
    toks = [w.strip(".:\"'") for w in t.replace("/", " ").replace(",", " ").split()]
    toks = [w for w in toks if w]

    cat = "other"
    for w in toks:
        if w in CATSET:
            cat = w; break
        if w in ALIAS:
            cat = ALIAS[w]; break

    # direction: first token that maps to a direction word/sign. Skip a bare '0'/'1' that is really the
    # impact digit by preferring explicit words first, then falling back to signs.
    direction = 0
    for w in toks:
        if w in ("up", "bull", "bullish", "down", "bear", "bearish", "flat", "neutral", "+1", "-1"):
            direction = DIRMAP[w]; break

    # impact: first standalone 0-3 digit
    impact = 0
    for w in toks:
        if w in ("0", "1", "2", "3"):
            impact = int(w); break

    horizon = ""
    for w in toks:
        if w in HORIZONS:
            horizon = w; break
        if w in HZALIAS:
            horizon = HZALIAS[w]; break

    # off-ticker = an explicit 'n' / 'no' token (Y is the safe default). Guard against 'n' being part
    # of another token by only accepting a lone n/no/off/false.
    off = any(w in ("n", "no", "off", "false") for w in toks)
    return cat, off, direction, impact, horizon


def make_classify(url, model):
    def classify(ticker, title):
        body = json.dumps({
            "model": model,
            "prompt": PROMPT.format(ticker=(ticker or "?"), title=(title or "")[:400]),
            "stream": False,
            "options": {"temperature": 0, "num_predict": 24, "top_p": 0.1},
        }).encode()
        req = urllib.request.Request(url, data=body, headers={"Content-Type": "application/json"})
        r = urllib.request.urlopen(req, timeout=180)
        return normalize(json.load(r)["response"])
    return classify


UPDATE_FIELDS = ["cat_llm", "off_ticker", "local_dir", "local_impact", "local_horizon", "local_rating"]


def main(moved_only=False, limit=None, workers=4, model="qwen2.5:14b",
         host="host.docker.internal:11435", overlap_only=False):
    url = f"http://{host}/api/generate"
    classify = make_classify(url, model)

    # warm the model once (first call loads weights, ~30s) so worker threads don't all block on load
    try:
        t0 = time.time()
        print(f"warming {model} @ {host} ...", flush=True)
        print(f"  warm label={classify('AAPL', 'Company reports Q3 earnings, beats estimates')} "
              f"({time.time()-t0:.1f}s)", flush=True)
    except Exception as e:
        print(f"ABORT: cannot reach Ollama at {url}: {e}", flush=True)
        return {"updated": 0, "error": str(e)}

    # resume on local_rating IS NULL (rows not yet RICHLY judged). This re-covers any row that got only
    # cat_llm/off_ticker in the earlier category-only run so they pick up dir/impact/horizon too.
    # overlap_only: restrict to rows that ALSO carry an Anthropic label (llm_rating set) — the head-to-
    # head validation set. moved-first otherwise so the valuable subset is done first.
    base = NewsItem.objects.filter(local_rating__isnull=True)
    if overlap_only:
        base = base.filter(llm_rating__isnull=False)
    if moved_only:
        base = base.filter(day_effect=True)
    else:
        # moved-first, then newest. NOTE: Postgres sorts NULLs FIRST on a plain DESC, so a bare
        # order_by("-day_effect") drains the UNSCORED (day_effect IS NULL) rows before the True
        # movers — defeating moved-first. F(...).desc(nulls_last=True) forces True -> False -> NULL.
        from django.db.models import F
        base = base.order_by(F("day_effect").desc(nulls_last=True), "-dt")
    vals = base.values_list("id", "ticker", "title")
    rows = list(vals[:limit] if limit else vals)
    total = len(rows)
    print(f"to classify: {total} (moved_only={moved_only}, overlap_only={overlap_only}, "
          f"workers={workers})", flush=True)
    if not total:
        return {"updated": 0}

    from collections import Counter
    counts = Counter()
    done = 0
    n_off = 0
    rating_dist = Counter()
    batch = []
    t0 = time.time()

    def work(row):
        _id, ticker, title = row
        try:
            cat, off, direction, impact, horizon = classify(ticker, title)
            return _id, cat, off, direction, impact, horizon
        except Exception:
            return _id, None, None, None, None, None  # transient — leave null so a later run retries

    with ThreadPoolExecutor(max_workers=workers) as ex:
        for _id, cat, off, direction, impact, horizon in ex.map(work, rows):
            done += 1
            if cat is None:
                continue
            counts[cat] += 1
            if off:
                n_off += 1
            rating = int(direction) * int(impact)
            rating_dist[rating] += 1
            batch.append(NewsItem(id=_id, cat_llm=cat, off_ticker=bool(off), local_dir=direction,
                                  local_impact=impact, local_horizon=horizon, local_rating=rating))
            if len(batch) >= 200:
                NewsItem.objects.bulk_update(batch, UPDATE_FIELDS, batch_size=200)
                batch = []
            if done % 500 == 0:
                rate = done / (time.time() - t0)
                eta = (total - done) / rate if rate else 0
                print(f"  {done}/{total}  {rate:.1f}/s  eta {eta/60:.0f}m  off-ticker={n_off}  "
                      f"top={dict(counts.most_common(5))}", flush=True)
    if batch:
        NewsItem.objects.bulk_update(batch, UPDATE_FIELDS, batch_size=200)

    dt = time.time() - t0
    print(f"DONE classified={sum(counts.values())}/{total} in {dt/60:.1f}m ({done/dt:.1f}/s) "
          f"off-ticker={n_off}", flush=True)
    print(f"  rating dist (dir*impact): {dict(sorted(rating_dist.items()))}", flush=True)
    for c, n in counts.most_common():
        print(f"  {c:14s} {n:7d}", flush=True)
    return {"updated": sum(counts.values()), "off_ticker": n_off}


if __name__ == "__main__":
    kw = {}
    a = sys.argv
    for i, x in enumerate(a):
        if x == "--moved-only":
            kw["moved_only"] = True
        elif x == "--overlap-only":
            kw["overlap_only"] = True
        elif x == "--limit" and i + 1 < len(a):
            kw["limit"] = int(a[i + 1])
        elif x == "--workers" and i + 1 < len(a):
            kw["workers"] = int(a[i + 1])
        elif x == "--model" and i + 1 < len(a):
            kw["model"] = a[i + 1]
        elif x == "--host" and i + 1 < len(a):
            kw["host"] = a[i + 1]
    main(**kw)
