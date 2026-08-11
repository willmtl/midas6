#!/usr/bin/env python3
"""FIRING NOW scanner.

Scan the top robust signals (auto-derived from the all-on-all StockStudy results) across
the whole stock universe and find every stock currently FIRING one of them (signal True
within the last N bars). Each hit is joined with the signal's historical edge, the stock's
fundamentals (+ which amplifier bucket it lands in), and its sector(s). Answers "what do I
look at today". Writes LiveSignal rows to Postgres + a JSON cache.

Signals-only (no 70-exit loop) → fast. Parallel process pool like all_on_all_study.

Run:  docker compose run --rm backend python -u live_firing_scan.py --db
Opts: --jobs N  --recent N (bars, default 10)  --top N (signals, default 12)  --no-db-save
"""
import warnings
warnings.filterwarnings("ignore")

import json
import os
import sys
from pathlib import Path

from studies import SIGNALS, MARKET_SIGNAL_KEYS
from seq_fundamental_study import (
    build_universe, load_fundamentals, DIMENSIONS, DEFAULT_JOBS, _chunk, MIN_BARS,
)
from all_on_all_study import _prepare_indicators
from seq_fundamental_study import load_candles

STUDIES_DIR = Path(__file__).parent / ".data" / "studies"
STUDIES_DIR.mkdir(parents=True, exist_ok=True)

SKIP_SIGNALS = set(MARKET_SIGNAL_KEYS) | {"rsi_x_pos_updn"}


def top_signals(top_n, min_trades=1000):
    """Best signals by their strongest exit (from StockStudy), most robust first.
    Per signal, prefers a SIGNIFICANT exit (|t_stat|>=2) over a merely-high avg_return one, then
    ranks signals robust-first, then by avg_return — so the docstring's "most robust first" is real
    (the old code sorted by raw -avg_return only). Degrades to pure avg_return when t_stat is null
    (rows predating the significance layer), so behavior is unchanged until the sweep repopulates.
    Returns [(signal_key, best_exit_key, hist_avg, hist_wr, hist_trades, hist_avg_mae, hist_clean_pct)]."""
    from core.models import StockStudy
    best = {}   # signal_key -> (rank_key, row_tuple)
    for sk, ek, avg, wr, tr, mae, clean, ts in (
            StockStudy.objects.filter(total_trades__gte=min_trades)
            .values_list("signal_key", "exit_key", "avg_return", "win_rate", "total_trades",
                         "avg_mae", "clean_pct", "t_stat")):
        if sk in SKIP_SIGNALS:
            continue
        robust = ts is not None and abs(ts) >= 2
        rank_key = (1 if robust else 0, avg if avg is not None else -1e9)
        cur = best.get(sk)
        if cur is None or rank_key > cur[0]:   # best (robust, then avg_return) exit for this signal
            best[sk] = (rank_key, (sk, ek, avg, wr, tr, mae, clean))
    ranked = sorted(best.values(), key=lambda x: x[0], reverse=True)   # robust first, then avg_return
    return [row for _, row in ranked[:top_n]]


def _worker(payload):
    signal_keys, recent_window, tickers = payload
    import django
    os.environ.setdefault("DJANGO_SETTINGS_MODULE", "rotation.settings")
    try:
        django.setup()
    except Exception:
        pass
    candles = load_candles(tickers)
    hits = []  # (ticker, signal_key, days_ago, last_close)
    for tk, sdf in candles.items():
        if len(sdf) < MIN_BARS:
            continue
        _prepare_indicators(sdf)
        close = sdf["Close"].values
        last_close = float(close[-1])
        for sk in signal_keys:
            sig_fn = SIGNALS[sk][1]
            try:
                sig = sig_fn(sdf).fillna(False)
            except Exception:
                continue
            recent = sig.iloc[-recent_window:].tolist()
            if not any(recent):
                continue
            # bars since most recent True (0 = latest bar)
            days_ago = next(i for i, v in enumerate(reversed(recent)) if v)
            hits.append((tk, sk, days_ago, round(last_close, 2)))
    return hits


def run(jobs, recent_window=10, top_n=12, save_db=True):
    import sector_holdings
    sigs = top_signals(top_n)
    if not sigs:
        print("No StockStudy rows yet — run all_on_all_study.py first."); return
    sig_meta = {s[0]: {"best_exit_key": s[1], "hist_avg_return": s[2],
                       "hist_win_rate": s[3], "hist_trades": s[4],
                       "hist_avg_mae": s[5], "hist_clean_pct": s[6]} for s in sigs}
    signal_keys = list(sig_meta)
    print(f"Scanning {len(signal_keys)} top signals (recent {recent_window} bars): {signal_keys}")

    tickers = build_universe()
    funds = load_fundamentals(tickers)
    print(f"Universe: {len(tickers)} stocks | jobs: {jobs}")

    hits = []
    if jobs <= 1:
        hits = _worker((signal_keys, recent_window, tickers))
    else:
        import concurrent.futures as cf
        import multiprocessing as mp
        try:
            from django.db import connections
            connections.close_all()
        except Exception:
            pass
        payloads = [(signal_keys, recent_window, c) for c in _chunk(tickers, jobs * 3)]
        ctx = mp.get_context("spawn")
        with cf.ProcessPoolExecutor(max_workers=jobs, mp_context=ctx) as ex:
            for h in ex.map(_worker, payloads):
                hits.extend(h)

    # Fundamental bucket labels per ticker. Live signals are inherently snapshot (a stock
    # firing *now* has today's fundamentals), so we bucket only the snapshot (pit=False)
    # DIMENSIONS whose fields live on the current Fundamental snapshot. The point-in-time
    # dims read a historical metrics frame that isn't meaningful for a live snapshot.
    def bkts(tk):
        f = funds.get(tk, {})
        return {dim: bfn(f.get(field)) for (dim, field, bfn, _o, pit) in DIMENSIONS if not pit}

    # Smart-money confirmation for the firing tickers (SEC EDGAR): trailing-90d insider
    # open-market buys + recent (180d) 5%+ stake filings.
    from core.models import InsiderBuy, SecFiling
    from django.db.models import Sum, Count
    from datetime import date, timedelta
    firing_tks = list({tk for tk, _, _, _ in hits})
    today = date.today()
    # 180d window (not 90): SEC bulk Form 345 lags ~1 quarter, so a 90d-from-today window
    # is usually empty. 180d catches the most recently published quarter's insider buying.
    ins90 = dict(InsiderBuy.objects.filter(ticker__in=firing_tks, filed_date__gte=today - timedelta(days=180))
                 .values_list("ticker").annotate(s=Sum("buy_value")))
    sec = {}
    for r in (SecFiling.objects.filter(ticker__in=firing_tks, filed_date__gte=today - timedelta(days=180))
              .values("ticker", "form_group").annotate(n=Count("id"))):
        sec.setdefault(r["ticker"], {})[r["form_group"]] = r["n"]

    rows = []
    for tk, sk, days_ago, last_close in hits:
        f = funds.get(tk, {})
        m = sig_meta[sk]
        rows.append({
            "ticker": tk, "signal_key": sk, "signal_name": SIGNALS[sk][0],
            "days_ago": days_ago, "last_close": last_close,
            **m,
            "market_cap": f.get("market_cap"), "pe_ratio": f.get("pe_ratio"),
            "forward_pe": f.get("forward_pe"), "profit_margin": f.get("profit_margin"),
            "fund_buckets": bkts(tk),
            "sectors": sector_holdings.get_sectors_for_ticker(tk),
            "insider_buy_90d": ins90.get(tk),
            "recent_13d": sec.get(tk, {}).get("13D", 0),
            "recent_13g": sec.get(tk, {}).get("13G", 0),
        })
    rows.sort(key=lambda r: (r["days_ago"], -(r["hist_avg_return"] or 0)))

    out = {"recent_window": recent_window, "n_signals": len(signal_keys),
           "universe_size": len(tickers), "n_firing": len(rows), "signals": signal_keys,
           "results": rows}
    path = STUDIES_DIR / "live_firing.json"
    path.write_text(json.dumps(out, indent=2))
    print(f"\n{len(rows)} firing hits -> {path}")
    for r in rows[:25]:
        print(f"  {r['ticker']:8} {r['days_ago']}d  {r['signal_key']:22} hist {r['hist_avg_return']:+.1f}%  "
              f"{','.join(r['sectors'][:2]) or '-'}")

    if save_db:
        _save_to_db(rows)
    return out


def _save_to_db(rows):
    from core.models import LiveSignal
    from django.utils import timezone
    from django.db import transaction
    now = timezone.now()
    keep = set()
    with transaction.atomic():
        for r in rows:
            keep.add((r["ticker"], r["signal_key"]))
            LiveSignal.objects.update_or_create(
                ticker=r["ticker"], signal_key=r["signal_key"],
                defaults={
                    "signal_name": r["signal_name"], "days_ago": r["days_ago"],
                    "last_close": r["last_close"], "best_exit_key": r["best_exit_key"],
                    "hist_avg_return": r["hist_avg_return"], "hist_win_rate": r["hist_win_rate"],
                    "hist_trades": r["hist_trades"],
                    "hist_avg_mae": r.get("hist_avg_mae"), "hist_clean_pct": r.get("hist_clean_pct"),
                    "market_cap": r["market_cap"],
                    "pe_ratio": r["pe_ratio"], "forward_pe": r["forward_pe"],
                    "profit_margin": r["profit_margin"], "fund_buckets": r["fund_buckets"],
                    "sectors": r["sectors"], "computed_at": now,
                    "insider_buy_90d": r.get("insider_buy_90d"),
                    "recent_13d": r.get("recent_13d", 0), "recent_13g": r.get("recent_13g", 0),
                },
            )
    # Clear stale rows (no longer firing).
    LiveSignal.objects.exclude(computed_at=now).delete()
    print(f"DB: upserted {len(keep)} LiveSignal rows, cleared stale.")


if __name__ == "__main__":
    argv = sys.argv

    def _opt(name, default=None):
        return argv[argv.index(name) + 1] if name in argv else default

    jobs = int(_opt("--jobs", DEFAULT_JOBS))
    recent = int(_opt("--recent", 10))
    top_n = int(_opt("--top", 12))
    if "--db" in argv:
        import django
        os.environ.setdefault("DJANGO_SETTINGS_MODULE", "rotation.settings")
        django.setup()
    run(jobs, recent_window=recent, top_n=top_n, save_db=("--db" in argv and "--no-db-save" not in argv))
