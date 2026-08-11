"""Focused dark-pool validation — SINGLE-PROCESS (jobs=1, no multiprocessing → no hang risk) after
the full MP sweep kept stalling on this box. Runs the top winner signals across the stock universe
and prints the dark-pool amplifier-dimension breakdown (does high / accumulating dark-pool activity
lift the return?). Non-destructive: save_db=False (won't touch StockStudy). Logs to
/app/.data/darkpool_validation.log.
"""
import os
import django

os.environ.setdefault("DJANGO_SETTINGS_MODULE", "rotation.settings")
django.setup()

from all_on_all_study import run

TOPS = ["obv_div_sort_pos", "new_52low", "rsi_sup10_x_dd70", "rsi_sup10_x_dd50_vol",
        "gap_down_large", "rsi_oversold20", "rsi_sup10_x_dd60", "rsi_sup10_x_dd50_mkt",
        "rsi_sup10_x_dd50", "seq_rsi20_sort_neg_pocket", "activist_13d_os",
        "seq_rsi20_sort_neg_vol_rsi_20d", "seq_rsi20_sort_neg_vol_rsi", "gap_down_med",
        "rsi_sup10_x_wk"]

print("DP validation: %d signals, jobs=1 (single-process, no MP)" % len(TOPS), flush=True)
out = run(jobs=1, signal_keys=TOPS, exit_keys=["6m", "sort_gt1"], min_trades=20, save_db=False)

print("DARKPOOL_VALIDATION_START", flush=True)
for r in out["results"]:
    bd = r.get("by_dimension", {})
    dps = bd.get("Dark-pool share")
    dpt = bd.get("Dark-pool trend")
    if not (dps or dpt):
        continue
    print("SIG %s -> %s | base %.1f%% / %dtr" % (r["signal_key"], r["exit_key"], r["avg_return"], r["trades"]), flush=True)
    for b in (dps or []):
        print("   share %-18s %6.1f%% / %dtr" % (b["bucket"], b["avg_return"], b["trades"]), flush=True)
    for b in (dpt or []):
        print("   trend %-22s %6.1f%% / %dtr" % (b["bucket"], b["avg_return"], b["trades"]), flush=True)
print("DARKPOOL_VALIDATION_DONE", flush=True)
