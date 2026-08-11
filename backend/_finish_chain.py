"""Post-sweep finish chain (in-container, detached — survives CLI-session reaping). Single-tasked,
in order:
  1. wait for the stock sweep to finish (SWEEP_CHAIN_DONE in sweep_chain.log)
  2. run the multiprocessing sector recompute (sector_studies_mp.py --jobs 8) — fast MAE backfill
  3. launch the local news off_ticker LLM drain (--workers 4, detached) — the "start the llm when
     you're done, less threads" step, only after all heavy compute clears
Logs /app/.data/finish_chain.log.
"""
import os
import time
import subprocess
import django

os.environ.setdefault("DJANGO_SETTINGS_MODULE", "rotation.settings")
django.setup()


def log(m):
    print("[finish] " + m, flush=True)


def sweep_done():
    try:
        with open("/app/.data/sweep_chain.log") as f:
            return "SWEEP_CHAIN_DONE" in f.read()
    except FileNotFoundError:
        return False


# 1) wait for the stock sweep
while not sweep_done():
    log("waiting for stock sweep to finish...")
    time.sleep(60)
log("stock sweep done — starting MP sector recompute --jobs 8")

# 2) optimized (multiprocessing) sector recompute — subprocess so spawn gets a clean __main__
rc = subprocess.run(["python", "-u", "sector_studies_mp.py", "--jobs", "8"],
                    cwd="/app", capture_output=True, text=True)
log("MP recompute rc=%s tail=%s" % (rc.returncode, (rc.stdout or "")[-300:]))
if rc.returncode != 0:
    log("MP recompute stderr tail: %s" % (rc.stderr or "")[-500:])

from core.models import Study
log("sector studies with MAE now: %d/%d"
    % (Study.objects.exclude(avg_mae=0).count(), Study.objects.count()))

# 3) launch the LLM off_ticker drain (detached, multi-day, gentle)
log("launching LLM off_ticker drain --workers 4 (detached)")
try:
    lf = open("/app/.data/news_llm_drain.log", "a")
    subprocess.Popen(["python", "-u", "news_llm_category.py", "--workers", "4"],
                     cwd="/app", stdout=lf, stderr=subprocess.STDOUT)
    log("FINISH_CHAIN_DONE (LLM drain launched)")
except Exception as e:
    log("drain launch error: %s" % str(e)[:200])
