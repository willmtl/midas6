"""
Classify news headlines with a cheap LLM (Anthropic Haiku) into a real signed rating,
because EODHD's own sentiment is ~all-positive and unusable.

Per headline the model returns: dir (-1 bearish / 0 neutral / +1 bullish), impact (0-3),
cat (fixed taxonomy). We store llm_rating = dir * impact (signed -3..+3) on NewsItem.

No SDK — direct HTTPS to the Anthropic Messages API via urllib (avoids adding a dependency /
image rebuild), same pattern as the EODHD importer. Reads ANTHROPIC_API_KEY from env.

Usage (inside container):
  docker compose exec -T backend python -u manage.py shell -c "from news_classifier import classify_news; classify_news(limit=6000)"
"""
import os, json, time, urllib.request, urllib.error, datetime as dt, random, logging
logger = logging.getLogger(__name__)

MODEL = "claude-haiku-4-5-20251001"
API_URL = "https://api.anthropic.com/v1/messages"
CATS = ("earnings_beat", "earnings_miss", "guidance_up", "guidance_down", "upgrade",
        "downgrade", "mna", "product", "partnership", "legal", "capital", "macro", "other")

SYSTEM = ("You are a precise financial-news classifier. For each headline judge its likely effect "
          "on THAT company's own stock over the next few weeks. Be willing to mark bearish and "
          "neutral — do NOT default to bullish. Reply with ONLY a JSON array, no prose.")

def _prompt(batch):
    lines = [f'{i+1}. [{r["ticker"]}] {r["title"]}' for i, r in enumerate(batch)]
    return (f"Classify these {len(batch)} headlines. For EACH return an object "
            '{"i":<num>,"dir":<-1|0|1>,"impact":<0|1|2|3>,"cat":"<category>"}.\n'
            "dir: 1 bullish for the stock, -1 bearish, 0 neutral/unclear.\n"
            "impact: 0 none, 1 minor, 2 moderate, 3 major (clearly market-moving).\n"
            f"cat one of: {', '.join(CATS)}.\n\nHeadlines:\n" + "\n".join(lines) +
            "\n\nReturn ONLY the JSON array of "
            f"{len(batch)} objects, ordered by i (1..{len(batch)}).")

def _call(prompt, max_tokens, key, retries=4):
    body = json.dumps({"model": MODEL, "max_tokens": max_tokens, "system": SYSTEM,
                       "messages": [{"role": "user", "content": prompt}]}).encode()
    req = urllib.request.Request(API_URL, data=body, method="POST", headers={
        "x-api-key": key, "anthropic-version": "2023-06-01", "content-type": "application/json"})
    for attempt in range(retries):
        try:
            with urllib.request.urlopen(req, timeout=90) as resp:
                j = json.loads(resp.read())
            txt = "".join(b.get("text", "") for b in j.get("content", []) if b.get("type") == "text")
            u = j.get("usage", {})
            return txt, u.get("input_tokens", 0), u.get("output_tokens", 0)
        except urllib.error.HTTPError as e:
            if e.code in (429, 500, 502, 503, 529) and attempt < retries - 1:
                time.sleep(2 * (attempt + 1)); continue
            logger.warning("anthropic HTTP %s: %s", e.code, e.read()[:200]); return None, 0, 0
        except Exception as ex:
            if attempt < retries - 1:
                time.sleep(2 * (attempt + 1)); continue
            logger.warning("anthropic call failed: %s", ex); return None, 0, 0
    return None, 0, 0

def _parse(txt, n):
    if not txt:
        return None
    a, b = txt.find("["), txt.rfind("]")
    if a < 0 or b < 0:
        return None
    try:
        arr = json.loads(txt[a:b + 1])
    except Exception:
        return None
    out = {}
    for o in arr:
        try:
            i = int(o["i"])
            d = int(o.get("dir", 0)); d = -1 if d < 0 else 1 if d > 0 else 0
            imp = max(0, min(3, int(o.get("impact", 0))))
            cat = str(o.get("cat", "other")).lower().strip()
            if cat not in CATS:
                cat = "other"
            if 1 <= i <= n:
                out[i] = (d, imp, cat)
        except Exception:
            continue
    return out

def classify_news(limit=6000, batch=25, min_age_days=45, max_age_days=None, tickers=None,
                  only_unclassified=True, seed=17):
    """Classify up to `limit` NewsItem rows. Default: older than min_age_days (forward room for the
    drift study). Set max_age_days to bound the OTHER side — e.g. min_age_days=0, max_age_days=15
    targets the RECENT window for the live playbook fade-flag. Pass `tickers` to scope to a specific
    set (e.g. today's firing candidates) so the recent-window pass stays cheap. Writes llm_* fields."""
    import django
    os.environ.setdefault("DJANGO_SETTINGS_MODULE", "rotation.settings")
    try:
        django.setup()
    except Exception:
        pass
    from django.utils import timezone
    from core.models import NewsItem
    key = os.environ.get("ANTHROPIC_API_KEY")
    if not key:
        return {"error": "ANTHROPIC_API_KEY not set"}
    cutoff = timezone.now() - dt.timedelta(days=min_age_days)
    qs = NewsItem.objects.filter(title__gt="", dt__lte=cutoff)
    if max_age_days is not None:
        qs = qs.filter(dt__gte=timezone.now() - dt.timedelta(days=max_age_days))
    if tickers:
        qs = qs.filter(ticker__in=list(tickers))
    if only_unclassified:
        qs = qs.filter(classified_at__isnull=True)
    ids = list(qs.values_list("id", flat=True))
    random.seed(seed); random.shuffle(ids)
    ids = ids[:limit]
    print(f"classifying {len(ids)} headlines in batches of {batch} with {MODEL}", flush=True)
    done = 0; tin = 0; tout = 0; dist = {}
    for s in range(0, len(ids), batch):
        chunk_ids = ids[s:s + batch]
        rows = list(NewsItem.objects.filter(id__in=chunk_ids).values("id", "ticker", "title"))
        if not rows:
            continue
        txt, ti, to = _call(_prompt(rows), batch * 22 + 120, key)
        tin += ti; tout += to
        parsed = _parse(txt, len(rows))
        if not parsed:
            print(f"  batch @{s}: parse failed, skipped", flush=True); continue
        now = timezone.now(); upd = []
        for i, r in enumerate(rows, start=1):
            p = parsed.get(i)
            if not p:
                continue
            d, imp, cat = p
            obj = NewsItem(id=r["id"]); obj.llm_dir = d; obj.llm_impact = imp; obj.llm_cat = cat
            obj.llm_rating = d * imp; obj.llm_model = MODEL; obj.classified_at = now
            upd.append(obj); dist[d * imp] = dist.get(d * imp, 0) + 1
        if upd:
            NewsItem.objects.bulk_update(
                upd, ["llm_dir", "llm_impact", "llm_cat", "llm_rating", "llm_model", "classified_at"])
            done += len(upd)
        if (s // batch) % 10 == 0:
            print(f"  {done}/{len(ids)}  (in={tin} out={tout} tok)", flush=True)
        time.sleep(0.05)
    # rough cost: Haiku ~$1/Mtok in, ~$5/Mtok out
    cost = tin / 1e6 * 1.0 + tout / 1e6 * 5.0
    print(f"DONE classified={done} in_tok={tin} out_tok={tout} ~${cost:.2f}", flush=True)
    print("rating distribution (dir*impact):", dict(sorted(dist.items())), flush=True)
    return {"classified": done, "in_tok": tin, "out_tok": tout, "cost_usd": round(cost, 2), "dist": dist}
