"""Ingest the subagent-written rated_*.json files into NewsItem.llm_* fields."""
import django, os, json
os.environ.setdefault("DJANGO_SETTINGS_MODULE", "rotation.settings"); django.setup()
from django.utils import timezone
from core.models import NewsItem

D = "/app/.data/news_batches"
MODEL = "claude-haiku-4-5 (inline)"
now = timezone.now()
files = sorted(f for f in os.listdir(D) if f.startswith("rated_") and f.endswith(".json"))
print("rated files:", files, flush=True)
HORIZONS = {"day", "week", "month", "3mo"}
upd = []; dist = {}; hz = {}; bad = 0
for fn in files:
    try:
        arr = json.load(open(os.path.join(D, fn), encoding="utf-8-sig"))
    except Exception as e:
        print("  bad file", fn, e); bad += 1; continue
    for o in arr:
        try:
            i = int(o["id"])
            d = int(o.get("dir", 0)); d = -1 if d < 0 else 1 if d > 0 else 0
            imp = max(0, min(3, int(o.get("impact", 0))))
            cat = str(o.get("cat", "other")).lower().strip()[:24]
            hor = str(o.get("horizon", "")).lower().strip()
            if hor not in HORIZONS:
                hor = ""
            obj = NewsItem(id=i)
            obj.llm_dir = d; obj.llm_impact = imp; obj.llm_cat = cat; obj.llm_horizon = hor
            obj.llm_rating = d * imp; obj.llm_model = MODEL; obj.classified_at = now
            upd.append(obj); dist[d * imp] = dist.get(d * imp, 0) + 1; hz[hor] = hz.get(hor, 0) + 1
        except Exception:
            continue
for k in range(0, len(upd), 500):
    NewsItem.objects.bulk_update(
        upd[k:k + 500], ["llm_dir", "llm_impact", "llm_cat", "llm_horizon", "llm_rating", "llm_model", "classified_at"])
print(f"updated {len(upd)} rows (bad files {bad})", flush=True)
print("rating distribution (dir*impact):", dict(sorted(dist.items())), flush=True)
print("horizon distribution:", dict(sorted(hz.items())), flush=True)
