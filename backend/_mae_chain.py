"""Autonomous MAE + alt-data chain driver — runs DETACHED IN-CONTAINER so it survives the CLI
session (background pollers there kept getting reaped). Single-tasked, one stage at a time:

  1. wait until the sector-study recompute finishes (Study.is_computed all True)
  2. run the all-on-all stock sweep at reduced jobs → fills StockStudy MAE + the new dark-pool
     amplifier dimensions (this is the dark-pool validation surface)
  3. refresh the live-firing scan so the Firing Now tab picks up the fresh StockStudy
  4. launch the local news off_ticker LLM drain at reduced workers (detached; multi-day) — the
     "restart the llm, less threads" step, run only AFTER the heavy compute clears

Logs to /app/.data/mae_chain.log. Launch:
  docker compose exec -d backend sh -c "python -u _mae_chain.py >> /app/.data/mae_chain.log 2>&1"
"""
import os
import time
import subprocess
import django

os.environ.setdefault("DJANGO_SETTINGS_MODULE", "rotation.settings")
django.setup()

from core.models import Study


def log(m):
    print("[mae_chain] " + m, flush=True)


# 1) wait for the sector recompute to drain to zero uncomputed
while True:
    n = Study.objects.filter(is_computed=False).count()
    if n == 0:
        break
    log("waiting on sector recompute: %d uncomputed" % n)
    time.sleep(60)
log("sector recompute DONE (%d studies, %d with MAE)"
    % (Study.objects.count(), Study.objects.exclude(avg_mae=0).count()))

# 2) stock sweep — StockStudy MAE + dark-pool dimensions — at reduced jobs (less threads)
log("starting stock sweep --jobs 6")
from api.tasks import run_stock_studies_task, run_live_firing_task
rc = run_stock_studies_task(jobs=6)
log("STOCK_SWEEP_DONE rc=%s" % rc)

# 3) refresh live firing so Firing Now reflects the fresh StockStudy
try:
    run_live_firing_task()
    log("LIVE_FIRING_DONE")
except Exception as e:
    log("live_firing error: %s" % str(e)[:200])

# 4) launch the local news off_ticker LLM drain at reduced workers, detached (multi-day job)
log("launching LLM off_ticker drain --workers 4 (detached)")
try:
    lf = open("/app/.data/news_llm_drain.log", "a")
    subprocess.Popen(["python", "-u", "news_llm_category.py", "--workers", "4"],
                     cwd="/app", stdout=lf, stderr=subprocess.STDOUT)
    log("CHAIN_DONE (LLM drain launched detached)")
except Exception as e:
    log("drain launch error: %s" % str(e)[:200])
