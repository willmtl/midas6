"""
News CONTENT-QUALITY classifier — title-only heuristic, no LLM, no candle data.

Flags NewsItem.junk=True when the headline is opinion / clickbait / performance-recap rather than a
discrete news EVENT. These pollute the corpus and the drift study: they carry no fresh information,
they just editorialize about a move that already happened (user flagged:
"Oklo Is Still Down 33% This Year. What Will It Take to Get OKLO Stock Back to $100?").

Four complementary detectors (ANY -> junk):
  - question  : rhetorical / advice headlines — "Is X a Buy?", "Should You Buy...", "Where Will X Be",
                "What Will It Take...", "Why ... stock", any headline that just ends in "?".
  - listicle  : roundups & comparisons — "3 Stocks to Buy", "Better Buy: A vs B", "Best Stock Right Now".
  - perf_recap: recap of a move that ALREADY happened — "Down 33% This Year", "Up 95% in 2026",
                "still down", "back to $100", "year-to-date".
  - opinion   : opinion-mill framings — "Here's Why", "Motley Fool", "Prediction:", "My Top ...",
                "I just bought", "billionaire ...", "buy the dip".

junk is orthogonal to day_suspect (bad candle) — a junk headline can still have moved the stock; we
just don't want to treat editorial noise as an information event. The drift study excludes junk=True;
the dashboard table default-hides it (toggle to show, with a badge).

Writes NewsItem.junk (bool). Idempotent; re-runnable. Title-only -> whole corpus in seconds.
Run: docker compose exec -T backend python -u compute_news_junk.py [--limit N] [--only-null]
"""
import django, os, sys, re
os.environ.setdefault("DJANGO_SETTINGS_MODULE", "rotation.settings"); django.setup()
from core.models import NewsItem

PATTERNS = {
    "question": re.compile(
        r"\?\s*$"
        r"|\bwhat will it take\b|\bshould you (buy|sell|own)\b|\bis it time to\b"
        r"|\bwhere will\b.*\bbe\b|\bis .+ a (buy|sell|good buy|smart buy|value)\b"
        r"|\bwhy .+ (stock|shares|is|are|could|might)\b|\bcan .+ (stock|shares)\b",
        re.I),
    "listicle": re.compile(
        r"^\s*\d+\s+(stocks?|reasons|things|top|best|dividend)\b"
        r"|\bbetter buy\b|\b(vs\.?|versus)\b.*\bwhich\b"
        r"|\bstocks? to (buy|watch|consider|avoid|own)\b"
        r"|\b\d+\s+(top|best|great|growth|dividend|cheap)\b.*\bstocks?\b",
        re.I),
    "perf_recap": re.compile(
        r"\b(down|up|gained?|lost|fell|rose|dropped|surged|plunged|soared|jumped|tanked|slumped)\s+\d+(\.\d+)?%\s+(this year|year to date|ytd|so far|in \d{4}|this (week|month|quarter))\b"
        r"|\bback to \$?\d"
        r"|\bstill (down|up)\b"
        r"|\byear[- ]to[- ]date\b",
        re.I),
    "opinion": re.compile(
        r"\bmotley fool\b|\bshould investors\b|\bhere'?s why\b|\bcould (soar|surge|double|triple|be worth)\b"
        r"|\bmy top\b|\bi (just )?(bought|sold)\b|\bprediction\b|\bforecast for \d{4}\b"
        r"|\bbillionaire\b|\bwall street (thinks|says|is)\b|\bbuy the dip\b",
        re.I),
    # multi-ticker ROUNDUP / recap — the headline is about several stocks, not a discrete event for
    # THIS ticker (user: "Peloton Slides ...; Six Flags Falls on Revenue Miss | Stock Movers ... some
    # news are not really related to the actual ticker"). Aggregator/market-wrap columns.
    "roundup": re.compile(
        r"\|\s*stock movers\b|\bstock movers\b|\bmarket movers\b|\bbiggest movers\b"
        r"|\b(pre-?market|after[- ]hours?|midday|premarket)\s+(movers|gainers|losers|winners)\b"
        r"|\b(gainers?\s+(and|&|/)\s+losers?)\b|\btrending (stocks|tickers)\b"
        r"|\bstocks? (that are )?(moving|to watch)\b.*\b(today|now|premarket)\b"
        r"|\bwhat'?s moving\b|\bmarket wrap\b|\bstocks? making the biggest moves\b"
        r"|\bmovers?:\s|\bmoving the market\b|\bwall street breakfast\b",
        re.I),
    # PR / award / recognition fluff — corporate self-congratulation, no market event
    "prfluff": re.compile(
        r"\brecognized (at|as|by|for|among|with|in|on)\b"
        r"|\b(wins?|receives?|earns?|honored with|takes home)\b[\w\s,'-]*\baward\b"
        r"|\bawards?\b[\w\s,'-]*\b(ceremony|gala|winners?|finalist|honoree|recogni)"
        r"|\borbie\b|\bstevie awards?\b|\bmagic quadrant\b"
        r"|\bnamed (one of|to the|a leader|among|to fortune|to inc\.?|to the list)\b"
        r"|\b(best|top) (places?|companies?) to work\b|\bgreat place to work\b|\btop workplace"
        r"|\bmost admired\b|\bhonored (at|as|for|among|by)\b|\bhall of fame\b"
        r"|\bproud to (announce|sponsor)\b|\bcelebrates? \d+ years\b",
        re.I),
}


def is_junk(title):
    t = title or ""
    return any(p.search(t) for p in PATTERNS.values())


def main(limit=None, only_null=False):
    qs = NewsItem.objects.all()
    if only_null:
        qs = qs.filter(junk__isnull=True)
    rows = list(qs.values_list("id", "title"))
    if limit:
        rows = rows[:limit]
    print(f"scanning {len(rows)} items (only_null={only_null})", flush=True)

    batch, updated, junk = [], 0, 0
    for _id, title in rows:
        j = is_junk(title)
        batch.append(NewsItem(id=_id, junk=j))
        updated += 1
        junk += j
        if len(batch) >= 5000:
            NewsItem.objects.bulk_update(batch, ["junk"], batch_size=2000)
            batch = []
            print(f"  {updated}/{len(rows)} scanned, junk={junk}", flush=True)
    if batch:
        NewsItem.objects.bulk_update(batch, ["junk"], batch_size=2000)

    print(f"DONE scanned={updated} junk={junk} ({(100*junk/updated if updated else 0):.1f}%)", flush=True)
    return {"updated": updated, "junk": junk}


if __name__ == "__main__":
    lim = None; onull = False
    for i, a in enumerate(sys.argv):
        if a == "--limit" and i + 1 < len(sys.argv):
            lim = int(sys.argv[i + 1])
        if a == "--only-null":
            onull = True
    main(limit=lim, only_null=onull)
