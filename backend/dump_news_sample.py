"""Dump a validation sample of unclassified headlines into batch files for inline (subagent)
classification. Files land in /app/.data/news_batches (= backend/.data/news_batches on host)."""
import django, os, json, random, datetime as dt
os.environ.setdefault("DJANGO_SETTINGS_MODULE", "rotation.settings"); django.setup()
from django.utils import timezone
from core.models import NewsItem

N = 10000      # scaled sample to power the impact>=2 / 90d cells
B = 400        # headlines per batch (one subagent per batch) -> 25 batches
cutoff = timezone.now() - dt.timedelta(days=95)   # need 90d forward room for the drift study
ids = list(NewsItem.objects.filter(title__gt="", dt__lte=cutoff, classified_at__isnull=True)
           .values_list("id", flat=True))
random.seed(17); random.shuffle(ids); ids = ids[:N]
rows = list(NewsItem.objects.filter(id__in=ids).values("id", "ticker", "title"))
outdir = "/app/.data/news_batches"; os.makedirs(outdir, exist_ok=True)
for f in os.listdir(outdir):
    if f.endswith(".json"):
        os.remove(os.path.join(outdir, f))
nb = 0
for s in range(0, len(rows), B):
    batch = rows[s:s + B]
    data = [{"n": i + 1, "id": r["id"], "ticker": r["ticker"], "title": r["title"][:200]}
            for i, r in enumerate(batch)]
    with open(f"{outdir}/batch_{nb:03d}.json", "w", encoding="utf-8") as fh:
        json.dump(data, fh, ensure_ascii=False)
    nb += 1
print(f"wrote {nb} batch files, {len(rows)} headlines -> {outdir}")
