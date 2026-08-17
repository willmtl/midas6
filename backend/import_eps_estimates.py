# -*- coding: utf-8 -*-
"""IMPORT EODHD FORWARD EPS ESTIMATES — start a POINT-IN-TIME capture series.

EODHD `fundamentals/{sym}?filter=Highlights` returns forward EPS estimates:
  EPSEstimateCurrentYear, EPSEstimateNextYear, EPSEstimateNextQuarter, EPSEstimateCurrentQuarter
plus PERatio. These are a CURRENT SNAPSHOT (EODHD keeps no history), so a single pull can NOT be
backtested historically without look-ahead. The point of importing is to (1) have a live forward
signal for go-live and (2) APPEND one dated snapshot per run so that, run at each rebalance, we build
our OWN point-in-time forward-estimate history from today on.

Each run appends one JSON line per ticker to .data/eps_estimates_snapshots.jsonl, stamped with the
capture date passed in (--asof YYYY-MM-DD; never uses Date.now so it's deterministic/reproducible) and
the current price (latest candle close) so forward P/E = price / EPSEstimateNextYear is computable.

Run detached in the celery worker (EODHD egress):
  docker exec rotation-celery-worker-1 python -u /app/import_eps_estimates.py --asof 2026-08-16
"""
import os, sys, json, time
os.environ.setdefault("DJANGO_SETTINGS_MODULE", "rotation.settings")
import django; django.setup()
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
from api.tasks import _eodhd_get, _eodhd_sym
from seq_fundamental_study import build_universe
from core.models import Candle

OUT = Path("/app/.data/eps_estimates_snapshots.jsonl")
FIELDS = ("EPSEstimateCurrentYear", "EPSEstimateNextYear", "EPSEstimateNextQuarter",
          "EPSEstimateCurrentQuarter")


def _f(x):
    try:
        v = float(x)
        return v if v == v else None      # drop NaN
    except (TypeError, ValueError):
        return None


def _last_close(tk):
    c = Candle.objects.filter(ticker=tk, interval="1d").order_by("-date").values_list("close", flat=True).first()
    return float(c) if c is not None else None


def load_done(asof):
    done = set()
    if OUT.exists():
        for line in OUT.read_text(encoding="utf-8").splitlines():
            try:
                r = json.loads(line)
                if r.get("asof") == asof:
                    done.add(r["ticker"])
            except Exception:
                pass
    return done


def main(asof, workers=6):
    tickers = build_universe()
    done = load_done(asof)
    todo = [t for t in tickers if t not in done]
    print(f"universe {len(tickers)} | already captured for {asof}: {len(done)} | to fetch: {len(todo)}", flush=True)
    fh = OUT.open("a", encoding="utf-8")
    got = withest = 0; t0 = time.time()

    def work(tk):
        sym = _eodhd_sym(tk)
        if not sym:
            return None
        d = _eodhd_get(f"fundamentals/{sym}", filter="Highlights")
        if not isinstance(d, dict) or not d:
            return None
        est = {k: _f(d.get(k)) for k in FIELDS}
        px = _last_close(tk)
        eny = est["EPSEstimateNextYear"]
        fwd_pe = (px / eny) if (px and eny and eny > 0) else None
        return {"asof": asof, "ticker": tk, "sym": sym, "price": px,
                "pe_ratio": _f(d.get("PERatio")), "fwd_pe": fwd_pe, **est}

    with ThreadPoolExecutor(max_workers=workers) as ex:
        for r in ex.map(work, todo):
            if r is None:
                continue
            got += 1
            if r.get("EPSEstimateNextYear") is not None:
                withest += 1
            fh.write(json.dumps(r) + "\n"); fh.flush()
            if got % 200 == 0:
                rate = got / (time.time() - t0)
                print(f"  {got}/{len(todo)}  {rate:.1f}/s  with-estimate={withest} ({100*withest/got:.0f}%)", flush=True)
    fh.close()
    print(f"DONE {got} tickers captured for {asof} in {(time.time()-t0)/60:.1f}m | "
          f"with forward estimate: {withest} ({100*withest/max(got,1):.0f}%)", flush=True)
    print(f"  -> {OUT}", flush=True)


if __name__ == "__main__":
    asof = None; workers = 6
    a = sys.argv
    for i, x in enumerate(a):
        if x == "--asof" and i + 1 < len(a):
            asof = a[i + 1]
        elif x == "--workers" and i + 1 < len(a):
            workers = int(a[i + 1])
    if not asof:
        print("ERROR: pass --asof YYYY-MM-DD (deterministic capture date)", flush=True); sys.exit(1)
    main(asof, workers)
