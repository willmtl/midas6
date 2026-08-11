"""Stock-sweep chain (decoupled from the slow sector recompute): run the all-on-all stock sweep at
reduced jobs (fills StockStudy MAE + the new dark-pool amplifier dimensions) → refresh live firing
so Firing Now reflects it. Does NOT launch the LLM drain (that's gated on the user confirming the
'done' state). Logs /app/.data/sweep_chain.log. Launch:
  docker compose exec -d backend sh -c "python -u _sweep_chain.py >> /app/.data/sweep_chain.log 2>&1"
"""
import os
import django

os.environ.setdefault("DJANGO_SETTINGS_MODULE", "rotation.settings")
django.setup()


def log(m):
    print("[sweep_chain] " + m, flush=True)


from api.tasks import run_stock_studies_task, run_live_firing_task

log("starting stock sweep --jobs 6 (StockStudy MAE + dark-pool dimensions)")
rc = run_stock_studies_task(jobs=6)
log("STOCK_SWEEP_DONE rc=%s" % rc)

try:
    run_live_firing_task()
    log("LIVE_FIRING_DONE")
except Exception as e:
    log("live_firing error: %s" % str(e)[:200])

log("SWEEP_CHAIN_DONE")
