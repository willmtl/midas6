#!/usr/bin/env python3
"""Multiprocessing sector-study recompute — the fast, truly-parallel replacement for the
thread-based (GIL-bound) inner loop of run_studies_task. IDENTICAL numerics; parallelizes over
signal GROUPS across PROCESSES (each with its own GIL) so the heavy path-dependent exit loops
actually run in parallel instead of serializing on one core.

Run via subprocess (spawn needs a clean script __main__, exactly like all_on_all_study.py):
  docker compose exec -d backend python -u sector_studies_mp.py --jobs 12

Aggregate-only: computes each Study's stats (incl. avg_mae/clean_pct) and UPDATEs the row; it does
NOT rebuild Trade/StudySectorResult (those are unchanged in a MAE backfill — the ~43M-row rebuild
is what made the thread path a ~day-long job). The nightly run_studies_task still handles brand-new
studies (which genuinely need trades built).
"""
import os
import argparse
import time
from concurrent.futures import ProcessPoolExecutor, as_completed

os.environ.setdefault("DJANGO_SETTINGS_MODULE", "rotation.settings")
import django

# Per-process caches (loaded once per worker, reused across its groups).
_DFS = None
_SPY = None
_NEEDS_SPY = ("rsi_x_pos_updn", "rsi_sup10_x_dd50_mkt", "rsi_sup10_x_mkt")


def _ensure_loaded():
    global _DFS, _SPY
    if _DFS is not None:
        return
    django.setup()
    from api.tasks import _get_dfs
    import config
    from core.models import Sector
    sectors = list(Sector.objects.all())
    dfs = _get_dfs([config.BENCHMARK] + [s.etf for s in sectors])
    all_dfs = {}
    for s in sectors:
        d = dfs.get(s.etf)
        if d is not None and len(d) >= 60:
            all_dfs[s.etf] = d
    _DFS = all_dfs
    spy = dfs.get(config.BENCHMARK)
    _SPY = spy["Close"] if spy is not None else None


def _process_group(payload):
    """payload = (signal_key, [(study_id, exit_key), ...]). Computes this signal's entries ONCE,
    then each exit-study's aggregates off them, and UPDATEs the Study row in-process. Returns
    (signal_key, n_done)."""
    signal_key, studies = payload
    _ensure_loaded()
    from studies import (SIGNALS, EXITS, CLEAN_MAE_THRESH, trade_mae,
                         _episode_starts, _tstat_from_returns)
    from core.models import Study
    if signal_key not in SIGNALS:
        return (signal_key, 0)
    _, sig_fn = SIGNALS[signal_key]
    needs_spy = signal_key in _NEEDS_SPY

    entries = {}   # etf -> list of entry idx
    for etf, df in _DFS.items():
        try:
            if needs_spy and _SPY is not None:
                sig = sig_fn(df, spy_close=_SPY).fillna(False)
            else:
                sig = sig_fn(df).fillna(False)
        except Exception:
            continue
        entries[etf] = [df.index.get_loc(d) for d in sig[sig].index]
    # Independent-episode bars per etf (overlap-dedup for the significance stat).
    episode_by_etf = {etf: _episode_starts(idxs) for etf, idxs in entries.items()}

    n_done = 0
    for study_id, exit_key in studies:
        if exit_key not in EXITS:
            continue
        _, exit_fn = EXITS[exit_key]
        tw = tt = 0
        tret = thold = 0.0
        tmae = 0.0
        tclean = 0
        eff = []   # one return per independent episode, pooled across etfs → significance
        for etf, df in _DFS.items():
            ents = entries.get(etf)
            if not ents:
                continue
            epi = episode_by_etf.get(etf, set())
            close_arr = df["Close"].values
            low_arr = df["Low"].values
            for idx in ents:
                exit_idx = exit_fn(df, idx)
                if exit_idx is None or exit_idx <= idx:
                    continue
                ep = float(close_arr[idx])
                if ep <= 0:
                    continue
                xp = float(close_arr[exit_idx])
                ret = (xp - ep) / ep * 100
                mae = trade_mae(ep, low_arr[idx + 1:exit_idx + 1])
                tt += 1
                tret += ret
                thold += (exit_idx - idx)
                tmae += mae
                if mae >= CLEAN_MAE_THRESH:
                    tclean += 1
                if ret > 0:
                    tw += 1
                if idx in epi:
                    eff.append(ret)
        Study.objects.filter(id=study_id).update(
            total_trades=tt,
            eff_trades=len(eff),
            t_stat=_tstat_from_returns(eff),
            avg_return=round(tret / tt, 3) if tt else 0,
            win_rate=round(tw / tt * 100, 1) if tt else 0,
            avg_hold=round(thold / tt, 1) if tt else 0,
            avg_mae=round(tmae / tt, 2) if tt else 0,
            clean_pct=round(tclean / tt * 100, 1) if tt else 0,
            is_computed=True,
        )
        n_done += 1
    return (signal_key, n_done)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--jobs", type=int, default=12)
    ap.add_argument("--all", action="store_true",
                    help="recompute every study (default: only is_computed=False)")
    args = ap.parse_args()
    django.setup()
    from collections import defaultdict
    from core.models import Study
    from django.db import connections
    qs = Study.objects.all() if args.all else Study.objects.filter(is_computed=False)
    groups = defaultdict(list)
    for sid, sk, ek in qs.values_list("id", "signal_key", "exit_key"):
        groups[sk].append((sid, ek))
    payloads = list(groups.items())
    total = sum(len(v) for v in groups.values())
    print("sector MP recompute: %d studies across %d signal groups, jobs=%d"
          % (total, len(payloads), args.jobs), flush=True)
    connections.close_all()   # never share the parent's DB connection across spawned workers
    t0 = time.time()
    done = gdone = 0
    with ProcessPoolExecutor(max_workers=args.jobs) as ex:
        futs = [ex.submit(_process_group, p) for p in payloads]
        for f in as_completed(futs):
            sk, n = f.result()
            done += n
            gdone += 1
            if gdone % 20 == 0:
                print("  %d/%d groups, %d studies, %ds"
                      % (gdone, len(payloads), done, time.time() - t0), flush=True)
    print("SECTORS_MP_DONE %d studies in %ds" % (done, time.time() - t0), flush=True)


if __name__ == "__main__":
    main()
