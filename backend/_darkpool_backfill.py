#!/usr/bin/env python3
"""Detached 5-year dark-pool history backfill (Polygon flat files).

Runs the whole weekday range MOST-RECENT-FIRST so useful data lands first, one day at a time
for clean progress + resumability (backfill_darkpool_flatfiles skips days already done). Meant to
be launched detached and left to grind for ~6 days:

  docker compose exec -d backend sh -c "python -u _darkpool_backfill.py >> /app/.data/darkpool_backfill.log 2>&1"

Resumable: if the container restarts, just relaunch — completed days are skipped.
"""
import os, django, datetime
os.environ.setdefault("DJANGO_SETTINGS_MODULE", "rotation.settings")
django.setup()

from api.tasks import backfill_darkpool_flatfiles

YEARS = 5
end = datetime.date.today()
start = end - datetime.timedelta(days=365 * YEARS + 10)

# weekdays only (Mon–Fri), most-recent-first; holidays auto-skip (file 404 → counted 'missing')
days = []
d = end
while d >= start:
    if d.weekday() < 5:
        days.append(d.isoformat())
    d -= datetime.timedelta(days=1)

print(f"darkpool 5y backfill: {len(days)} weekdays {days[-1]} .. {days[0]} (most-recent-first)", flush=True)

tot_days = tot_rows = tot_skip = tot_miss = 0
for i, ds in enumerate(days):
    r = backfill_darkpool_flatfiles([ds])
    tot_days += r.get("days", 0); tot_rows += r.get("rows", 0)
    tot_skip += r.get("skipped", 0); tot_miss += r.get("missing", 0)
    print(f"[{i+1}/{len(days)}] {ds} | +{r.get('rows',0)} rows | "
          f"done={tot_days} skip={tot_skip} miss={tot_miss} rows={tot_rows}", flush=True)

print(f"ALL DONE: processed={tot_days} skipped={tot_skip} missing={tot_miss} rows={tot_rows}", flush=True)
