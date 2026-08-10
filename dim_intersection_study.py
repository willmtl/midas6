#!/usr/bin/env python3
"""Targeted multi-dimension INTERSECTION study.

The all-on-all sweep slices a signal's trades by each dimension INDEPENDENTLY (marginal).
This asks the joint question: do the amplifiers STACK? For one proven signal x exit, bucket
its trades by the *combination* of 2-3 chosen point-in-time dimensions (e.g. micro-cap ∩
cheap-P/B ∩ insider-buying) and report which intersections beat the signal's baseline.

Targeted (one signal, a few dims) on purpose — the brute-force dim cross-product overfits.

Run: docker compose run --rm backend python -u dim_intersection_study.py \
        --signal obv_div_sort_pos --exit 6m --dims "Market cap,PB,Insider buying"
"""
import warnings; warnings.filterwarnings("ignore")
import os, sys, json
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor

DEFAULT_JOBS = 16
MIN_BARS = 60
STUDIES_DIR = Path(__file__).parent / ".data" / "studies"


def _worker(payload):
    signal_key, exit_key, dim_names, tickers = payload
    import django
    os.environ.setdefault("DJANGO_SETTINGS_MODULE", "rotation.settings")
    try:
        django.setup()
    except Exception:
        pass
    from studies import SIGNALS, EXITS
    from seq_fundamental_study import (load_candles, load_fundamentals, load_financial_reports,
                                       load_dividends, load_insider, load_filings, label_trade)
    from pit_fundamentals import prepare_pit_metrics
    from all_on_all_study import _prepare_alt   # attaches _insider_buy/_filed_13d/_filed_13g
    sig_fn = SIGNALS[signal_key][1]
    exit_fn = EXITS[exit_key][1]
    candles = load_candles(tickers)
    funds = load_fundamentals(tickers)
    reports = load_financial_reports(tickers)
    divs = load_dividends(tickers)
    insider = load_insider(tickers)
    filings = load_filings(tickers)
    spy = load_candles(["SPY"]).get("SPY")
    spy = spy["Close"] if spy is not None else None

    inter = {}   # tuple(labels) -> [n, sum_ret, wins]
    base = [0, 0.0, 0]
    for tk, sdf in candles.items():
        if len(sdf) < MIN_BARS:
            continue
        _prepare_alt(sdf, insider.get(tk), filings.get(tk))  # for alt-data signals
        try:
            sig = sig_fn(sdf).fillna(False)
        except Exception:
            continue
        idxs = [sdf.index.get_loc(d) for d in sig[sig].index]
        if not idxs:
            continue
        close = sdf["Close"].values
        n = len(close)
        pit = prepare_pit_metrics(sdf, reports.get(tk), divs.get(tk), spy, insider.get(tk), filings.get(tk))
        snap = funds.get(tk, {})
        label_cache = {}
        for idx in idxs:
            try:
                ex = exit_fn(sdf, idx)
            except Exception:
                ex = None
            if ex is None or ex <= idx or ex >= n:
                continue
            ep = float(close[idx])
            if ep <= 0:
                continue
            ret = (float(close[ex]) - ep) / ep * 100
            base[0] += 1; base[1] += ret; base[2] += 1 if ret > 0 else 0
            labels = label_cache.get(idx)
            if labels is None:
                labels = label_cache[idx] = label_trade(pit, sdf.index[idx], snap)
            key = tuple(labels.get(d, "NA") for d in dim_names)
            s = inter.setdefault(key, [0, 0.0, 0])
            s[0] += 1; s[1] += ret; s[2] += 1 if ret > 0 else 0
    return base, inter


def _chunk(seq, n):
    n = max(1, n); k, r = divmod(len(seq), n); out = []; i = 0
    for j in range(n):
        sz = k + (1 if j < r else 0)
        if sz:
            out.append(seq[i:i + sz]); i += sz
    return out


def run(signal_key, exit_key, dim_names, jobs=DEFAULT_JOBS, limit=None, min_trades=30):
    from seq_fundamental_study import build_universe
    from studies import SIGNALS, EXITS
    from django.db import connections
    uni = build_universe()
    if limit:
        uni = uni[:limit]
    print(f"Signal {signal_key} -> {exit_key} | dims {dim_names} | {len(uni)} stocks | jobs {jobs}")
    connections.close_all()
    import multiprocessing as mp
    ctx = mp.get_context("spawn")
    payloads = [(signal_key, exit_key, dim_names, c) for c in _chunk(uni, jobs * 2)]
    base = [0, 0.0, 0]
    inter = {}
    with ProcessPoolExecutor(max_workers=jobs, mp_context=ctx) as ex:
        for b, im in ex.map(_worker, payloads):
            base[0] += b[0]; base[1] += b[1]; base[2] += b[2]
            for k, s in im.items():
                d = inter.setdefault(k, [0, 0.0, 0])
                d[0] += s[0]; d[1] += s[1]; d[2] += s[2]

    b_avg = base[1] / base[0] if base[0] else 0
    b_wr = base[2] / base[0] * 100 if base[0] else 0
    print(f"\nBASELINE (all trades): {b_avg:+.2f}% avg, {b_wr:.0f}% wr, {base[0]} trades\n")

    rows = [(k, s[0], s[1] / s[0], s[2] / s[0] * 100) for k, s in inter.items() if s[0] >= min_trades]
    rows.sort(key=lambda r: r[2], reverse=True)
    print(f"Intersections (>= {min_trades} trades), by avg return, vs baseline {b_avg:+.2f}%:")
    print(f"  {'  '.join(dim_names):50}  {'avg':>8} {'lift':>7} {'wr':>5} {'tr':>6}")
    for k, n, avg, wr in rows[:25]:
        combo = " ∩ ".join(str(x) for x in k)
        print(f"  {combo:50} {avg:>+7.1f}% {avg-b_avg:>+6.1f}% {wr:>4.0f}% {n:>6}")

    out = {
        "signal": signal_key, "exit": exit_key, "dims": dim_names,
        "baseline": round(b_avg, 2), "baseline_wr": round(b_wr, 1), "baseline_trades": base[0],
        "min_trades": min_trades,
        "rows": [{"combo": list(k), "trades": n, "avg": round(avg, 2),
                  "wr": round(wr, 1), "lift": round(avg - b_avg, 2)} for k, n, avg, wr in rows[:60]],
    }
    STUDIES_DIR.mkdir(parents=True, exist_ok=True)
    (STUDIES_DIR / "dim_intersection.json").write_text(json.dumps(out, indent=2))
    print(f"\nSaved -> {STUDIES_DIR / 'dim_intersection.json'}")
    return out


if __name__ == "__main__":
    argv = sys.argv
    sig = argv[argv.index("--signal") + 1] if "--signal" in argv else "obv_div_sort_pos"
    exk = argv[argv.index("--exit") + 1] if "--exit" in argv else "6m"
    dims = argv[argv.index("--dims") + 1].split(",") if "--dims" in argv else ["Market cap", "PB", "Insider buying"]
    dims = [d.strip() for d in dims]
    jobs = int(argv[argv.index("--jobs") + 1]) if "--jobs" in argv else DEFAULT_JOBS
    limit = int(argv[argv.index("--limit") + 1]) if "--limit" in argv else None
    mt = int(argv[argv.index("--min-trades") + 1]) if "--min-trades" in argv else 30
    if "--db" in argv:
        import django
        os.environ.setdefault("DJANGO_SETTINGS_MODULE", "rotation.settings")
        django.setup()
    run(sig, exk, dims, jobs=jobs, limit=limit, min_trades=mt)
