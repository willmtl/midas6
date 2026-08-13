#!/usr/bin/env python3
"""Per-SIGNAL live firing scan -> SignalFiring.

For EVERY study signal, find which names (stock / ETF / commodity) fired it within the last N bars,
so the grouped Studies table can show a "firing now (3d)" count per study and open a pane with the
list. Reuses live_firing_scan._worker (signals-only, shared indicators via _prepare_indicators, spawn
pool). Market signals (need SPY/QQQ injection) are skipped.

Run: docker exec rotation-backend-1 python -u /app/signal_firing_scan.py --db --jobs 4
Opts: --jobs N  --recent N (bars, default 3)  --no-db-save
"""
import warnings
warnings.filterwarnings("ignore")
import os, sys
from studies import SIGNALS, MARKET_SIGNAL_KEYS
from seq_fundamental_study import DEFAULT_JOBS, _chunk

SKIP = set(MARKET_SIGNAL_KEYS) | {"rsi_x_pos_updn"}
RECENT = 3


def run(jobs, recent=RECENT, save_db=True):
    from core.models import Candle
    from live_firing_scan import _worker
    import sector_holdings
    signal_keys = [k for k in SIGNALS if k not in SKIP]
    tickers = sorted(set(Candle.objects.values_list("ticker", flat=True).distinct()))
    print(f"{len(signal_keys)} signals x {len(tickers)} names, recent {recent} bars, jobs {jobs}",
          flush=True)

    hits = []
    if jobs <= 1:
        # Serial but CHUNKED so peak memory stays bounded (the shared box SIGKILLs memory spikes).
        chunks = _chunk(tickers, 24)
        for i, c in enumerate(chunks):
            hits.extend(_worker((signal_keys, recent, c)))
            print(f"  chunk {i + 1}/{len(chunks)} done — {len(hits)} hits so far", flush=True)
    else:
        import concurrent.futures as cf
        import multiprocessing as mp
        try:
            from django.db import connections
            connections.close_all()
        except Exception:
            pass
        # Finer chunks (smaller per-worker footprint) — 375 signals × full universe is memory-heavy;
        # the box OOM'd at jobs=4/coarse chunks. Keep jobs low (default 2) and chunks small.
        payloads = [(signal_keys, recent, c) for c in _chunk(tickers, max(jobs * 8, 24))]
        ctx = mp.get_context("spawn")
        with cf.ProcessPoolExecutor(max_workers=jobs, mp_context=ctx) as ex:
            for h in ex.map(_worker, payloads):
                hits.extend(h)

    firing = {}
    for tk, sk, days_ago, last_close in hits:
        firing.setdefault(sk, []).append((tk, days_ago, last_close))
    secmap = {tk: sector_holdings.get_sectors_for_ticker(tk) for tk in {h[0] for h in hits}}
    print(f"{len(hits)} firings across {len(firing)} signals", flush=True)

    if save_db:
        _save(signal_keys, firing, secmap, recent)
    for sk in sorted(firing, key=lambda k: -len(firing[k]))[:15]:
        print(f"  {sk:26} {len(firing[sk])} firing", flush=True)


def _save(all_keys, firing, secmap, recent):
    from core.models import SignalFiring
    from django.utils import timezone
    from django.db import transaction
    now = timezone.now()
    with transaction.atomic():
        for sk in all_keys:
            lst = firing.get(sk, [])
            rows = sorted(({"ticker": tk, "days_ago": d, "last_close": round(c, 2),
                            "sectors": secmap.get(tk, [])} for tk, d, c in lst),
                          key=lambda r: (r["days_ago"], r["ticker"]))
            SignalFiring.objects.update_or_create(signal_key=sk, defaults={
                "signal_name": SIGNALS[sk][0], "recent_bars": recent, "n_firing": len(rows),
                "firing": rows, "computed_at": now})
    print(f"DB: upserted {len(all_keys)} SignalFiring rows", flush=True)


if __name__ == "__main__":
    argv = sys.argv

    def _opt(name, default):
        return argv[argv.index(name) + 1] if name in argv else default

    jobs = int(_opt("--jobs", 2))
    recent = int(_opt("--recent", RECENT))
    if "--db" in argv:
        import django
        os.environ.setdefault("DJANGO_SETTINGS_MODULE", "rotation.settings")
        django.setup()
    run(jobs, recent=recent, save_db=("--db" in argv and "--no-db-save" not in argv))
