"""Dump ALL already-classified headlines (the 12.5k) into batch files for RE-classification with the
enriched schema (signed type + expected horizon). writeback overwrites the existing llm_* fields."""
import django, os, json
os.environ.setdefault("DJANGO_SETTINGS_MODULE", "rotation.settings"); django.setup()
from core.models import NewsItem

B = 400
d = "/app/.data/news_batches"
os.makedirs(d, exist_ok=True)
for f in os.listdir(d):
    if f.endswith(".json"):
        os.remove(os.path.join(d, f))
rows = list(NewsItem.objects.filter(llm_rating__isnull=False).values("id", "ticker", "title").order_by("id"))
nb = 0
for s in range(0, len(rows), B):
    batch = rows[s:s + B]
    data = [{"n": i + 1, "id": r["id"], "ticker": r["ticker"], "title": r["title"][:200]}
            for i, r in enumerate(batch)]
    with open(f"{d}/batch_{nb:03d}.json", "w", encoding="utf-8") as fh:
        json.dump(data, fh, ensure_ascii=False)
    nb += 1
print(f"wrote {nb} batch files, {len(rows)} headlines -> {d}")
