#!/usr/bin/env python3
"""EVERYTHING ON EVERYTHING (stock side).

Run EVERY signal × EVERY exit across all ~1035 individual stocks (Candle∩Fundamental
minus ETFs/benchmarks), and attach the fundamental-bucket breakdown (PE, forward PE, EPS,
revenue growth, margin, float, market cap) to each signal×exit result. This is the
alpha-discovery sweep: "try everything, find what rebounds."

Efficiency: each signal is computed ONCE per ticker, then all 70 exits are applied to it
(exits are cheap index lookups). Aggregation happens INSIDE each worker (running sums +
per-bucket sums), so we never hold tens of millions of trade rows in memory — workers
return small stat dicts that the parent just sums together.

⚠️ Same LOOKAHEAD caveat as seq_fundamental_study.py: fundamentals are yfinance's CURRENT
snapshot bucketed onto historical trades — directional, not point-in-time tradable.

Run in the backend container:
  docker compose run --rm backend python -u all_on_all_study.py --db
Options: --jobs N  --limit N (tickers)  --min-trades N (report floor, default 20)
         --exits k1,k2,... (restrict exit set)  --signals k1,k2,... (restrict signal set)
"""
import warnings
warnings.filterwarnings("ignore")

import json
import os
import sys
from pathlib import Path

from studies import (
    SIGNALS, EXITS, MARKET_SIGNAL_KEYS, _categorize,
    _rolling_sortino, _rolling_omega, _rsi_of_sortino,
    CLEAN_MAE_THRESH, trade_mae, _episode_starts,
)
import ta
# Reuse universe / loaders / fundamental bucketing from the single-signal study.
from seq_fundamental_study import (
    build_universe, load_candles, load_fundamentals, load_financial_reports,
    load_dividends, load_insider, load_filings, load_darkpool, load_news, label_trade,
    DIMENSIONS, DEFAULT_JOBS, _chunk, MIN_BARS,
)
from pit_fundamentals import prepare_pit_metrics

STUDIES_DIR = Path(__file__).parent / ".data" / "studies"
STUDIES_DIR.mkdir(parents=True, exist_ok=True)

# Signals that need spy/qqq market series injected — meaningless as pure per-stock df
# signals, so we skip them in the per-stock sweep (they live in the sector engine).
SKIP_SIGNALS = set(MARKET_SIGNAL_KEYS) | {"rsi_x_pos_updn"}


def _prepare_indicators(df):
    """Attach the precomputed indicator columns the signal/exit functions look for.

    CRITICAL for speed: exits like exit_rsi_cross_down do `df["_rsi"] if "_rsi" in
    df.columns else ta.momentum.rsi(...)`. Without these columns every exit call
    recomputes RSI/Sortino on the full series — hundreds of millions of times over the
    full sweep. Computing them ONCE per ticker (matching run_all()'s recipe) turns exits
    into O(1) column reads. Mutates and returns df.
    """
    df["_sortino"] = _rolling_sortino(df)
    df["_omega"] = _rolling_omega(df)
    df["_rsi"] = ta.momentum.rsi(df["Close"], window=10)
    df["_rsi_sma"] = df["_rsi"].rolling(10).mean()
    df["_rsi_sort"] = _rsi_of_sortino(df)
    df["_rsi_sort_sma"] = df["_rsi_sort"].rolling(10).mean()
    return df


def _prepare_alt(df, insider_s, filings_df):
    """Attach alt-data event columns the Phase-D signals read: per-bar insider open-market
    buy $, and 1.0 on bars a 13D/13G was filed. Absent data -> all-zero columns.

    Each event column is SHIFTED one trading bar forward (.shift(1)) so the signal fires — and
    enters at that bar's close — the session AFTER the filing date. SEC Form 4 / 13D / 13G are
    routinely filed after the 16:00 ET close, so close[filed_date] pre-dated the disclosure;
    entering on filed_date's close was a one-session lookahead on these directional events."""
    import pandas as pd
    idx = df.index
    if insider_s is not None and len(insider_s):
        s = insider_s.copy(); s.index = pd.to_datetime(s.index)
        df["_insider_buy"] = s.groupby(s.index).sum().reindex(idx).shift(1).fillna(0.0)
    else:
        df["_insider_buy"] = 0.0
    for grp, col in (("13D", "_filed_13d"), ("13G", "_filed_13g")):
        if filings_df is not None and len(filings_df):
            sub = filings_df[filings_df["form_group"] == grp]
            ev = pd.Series(1.0, index=pd.to_datetime(sub["filed_date"]))
            df[col] = ev.groupby(ev.index).sum().reindex(idx).shift(1).fillna(0.0) if len(ev) else 0.0
        else:
            df[col] = 0.0
    return df


def _new_stat():
    # [n, sum_ret, wins, sum_hold, sum_mae, cleans, eff_n, eff_sum, eff_sumsq]
    # The last three carry the overlap-deduped "independent episode" returns (running sum + sum of
    # squares) so a one-sample t-stat can be finalized without keeping the full return list.
    return [0, 0.0, 0, 0.0, 0.0, 0, 0, 0.0, 0.0]


def _accum(stat, ret, hold, mae, clean, is_episode=False):
    stat[0] += 1
    stat[1] += ret
    stat[2] += 1 if ret > 0 else 0
    stat[3] += hold
    stat[4] += mae
    stat[5] += clean
    if is_episode:
        stat[6] += 1
        stat[7] += ret
        stat[8] += ret * ret


def _worker(payload):
    """One spawned process: load its ticker chunk from the DB, run all signals × exits,
    return nested aggregates. Keys are strings/tuples (pickled back to the parent)."""
    signal_keys, exit_keys, tickers = payload
    import django
    os.environ.setdefault("DJANGO_SETTINGS_MODULE", "rotation.settings")
    try:
        django.setup()
    except Exception:
        pass
    candles = load_candles(tickers)
    funds = load_fundamentals(tickers)
    reports = load_financial_reports(tickers)
    divs = load_dividends(tickers)
    insider = load_insider(tickers)
    filings = load_filings(tickers)
    darkpool = load_darkpool(tickers)
    news = load_news(tickers)
    spy_close = load_candles(["SPY"]).get("SPY")
    spy_close = spy_close["Close"] if spy_close is not None else None

    exit_fns = {ek: EXITS[ek][1] for ek in exit_keys}
    # overall[(sk, ek)] = [n, sum_ret, wins, sum_hold]
    overall = {}
    # buckets[(sk, ek, dim_name, bucket_label)] = [n, sum_ret, wins, sum_hold]
    buckets = {}

    _MISS = object()
    for tk, sdf in candles.items():
        if len(sdf) < MIN_BARS:
            continue
        _prepare_indicators(sdf)  # once per ticker → all exits become column reads
        _prepare_alt(sdf, insider.get(tk), filings.get(tk))  # Phase-D event-signal columns
        close = sdf["Close"].values
        low = sdf["Low"].values
        n = len(close)
        snap = funds.get(tk, {})
        # Point-in-time metrics frame for this ticker (indexed by the price dates).
        pit_metrics = prepare_pit_metrics(sdf, reports.get(tk), divs.get(tk), spy_close,
                                          insider.get(tk), filings.get(tk),
                                          darkpool.get(tk), news.get(tk))
        # Bucket labels depend only on the entry bar's date, so memoize per entry_idx
        # (signals fire on overlapping bars → reused across signals/exits, like exit_cache).
        label_cache = {}
        # Memoize exit_fn by (exit_key, entry_idx): the exit result depends only on
        # (df, idx), so the same entry bar reached by different signals reuses it. Many of
        # the 354 signals fire on overlapping bars → collapses ~10× of exit computation.
        exit_cache = {}

        for sk in signal_keys:
            sig_fn = SIGNALS[sk][1]
            try:
                sig = sig_fn(sdf).fillna(False)
            except Exception:
                continue
            entry_idxs = [sdf.index.get_loc(d) for d in sig[sig].index]
            if not entry_idxs:
                continue
            # Overlap-dedup: fires within EFFECTIVE_GAP bars collapse to one independent episode
            # (computed once per signal — the entry bars are the same across all exits).
            episode = _episode_starts(entry_idxs)
            for ek, exit_fn in exit_fns.items():
                for idx in entry_idxs:
                    ck = (ek, idx)
                    exit_idx = exit_cache.get(ck, _MISS)
                    if exit_idx is _MISS:
                        try:
                            exit_idx = exit_fn(sdf, idx)
                        except Exception:
                            exit_idx = None
                        exit_cache[ck] = exit_idx
                    if exit_idx is None or exit_idx <= idx or exit_idx >= n:
                        continue
                    ep = float(close[idx])
                    if ep <= 0:
                        continue
                    ret = (float(close[exit_idx]) - ep) / ep * 100
                    hold = exit_idx - idx
                    mae = trade_mae(ep, low[idx + 1:exit_idx + 1])
                    clean = 1 if mae >= CLEAN_MAE_THRESH else 0
                    is_ep = idx in episode
                    o = overall.get((sk, ek))
                    if o is None:
                        o = overall[(sk, ek)] = _new_stat()
                    _accum(o, ret, hold, mae, clean, is_ep)
                    labels = label_cache.get(idx)
                    if labels is None:
                        labels = label_cache[idx] = label_trade(pit_metrics, sdf.index[idx], snap)
                    for dim, label in labels.items():
                        key = (sk, ek, dim, label)
                        b = buckets.get(key)
                        if b is None:
                            b = buckets[key] = _new_stat()
                        _accum(b, ret, hold, mae, clean, is_ep)
    return overall, buckets


def _merge(dst, src):
    for k, s in src.items():
        d = dst.get(k)
        if d is None:
            dst[k] = list(s)
        else:
            d[0] += s[0]; d[1] += s[1]; d[2] += s[2]; d[3] += s[3]; d[4] += s[4]; d[5] += s[5]
            d[6] += s[6]; d[7] += s[7]; d[8] += s[8]


def _finalize_stat(s):
    n = s[0]
    if n == 0:
        return None
    out = {"trades": n, "avg_return": round(s[1] / n, 2),
           "win_rate": round(s[2] / n * 100, 1), "avg_hold": round(s[3] / n, 1),
           "avg_mae": round(s[4] / n, 2), "clean_pct": round(s[5] / n * 100, 1)}
    # Significance over the overlap-deduped independent episodes: one-sample t vs 0.
    en, esum, esumsq = s[6], s[7], s[8]
    out["eff_trades"] = int(en)
    t = None
    if en >= 3:
        mean = esum / en
        var = (esumsq - esum * esum / en) / (en - 1)
        if var > 0:
            t = round(mean / (var / en) ** 0.5, 2)
    out["t_stat"] = t
    return out


def _save_to_db(results, universe_size):
    """Upsert ranked results into the StockStudy Postgres table (mirrors sector Study)."""
    from core.models import StockStudy
    from django.utils import timezone
    from django.db import transaction
    now = timezone.now()
    keep = set()
    with transaction.atomic():
        for r in results:
            keep.add((r["signal_key"], r["exit_key"]))
            StockStudy.objects.update_or_create(
                signal_key=r["signal_key"], exit_key=r["exit_key"],
                defaults={
                    "signal_name": r["signal_name"], "exit_name": r["exit_name"],
                    "category": r["category"], "total_trades": r["trades"],
                    "eff_trades": r.get("eff_trades"), "t_stat": r.get("t_stat"),
                    "avg_return": r["avg_return"], "win_rate": r["win_rate"],
                    "avg_hold": r["avg_hold"], "avg_mae": r["avg_mae"],
                    "clean_pct": r["clean_pct"], "universe_size": universe_size,
                    "by_dimension": r["by_dimension"], "computed_at": now,
                },
            )
    # Drop combos that no longer clear the trade floor (keep the table in sync with JSON).
    existing = set(StockStudy.objects.values_list("signal_key", "exit_key"))
    to_delete = existing - keep
    if to_delete:
        from django.db.models import Q
        q = Q()
        for sk, ek in to_delete:
            q |= Q(signal_key=sk, exit_key=ek)
        StockStudy.objects.filter(q).delete()
    print(f"DB: upserted {len(keep)} StockStudy rows, removed {len(to_delete)} stale.")


def run(jobs, limit=None, min_trades=20, signal_keys=None, exit_keys=None, save_db=True):
    signal_keys = [s for s in (signal_keys or list(SIGNALS)) if s not in SKIP_SIGNALS]
    exit_keys = exit_keys or list(EXITS)

    tickers = build_universe()
    if limit:
        tickers = tickers[:limit]
    funds_meta = load_fundamentals(tickers)  # parent copy for reporting only
    print(f"Universe: {len(tickers)} stocks | signals: {len(signal_keys)} | "
          f"exits: {len(exit_keys)} | combos: {len(signal_keys) * len(exit_keys)} | jobs: {jobs}")

    overall, buckets = {}, {}
    if jobs <= 1:
        o, b = _worker((signal_keys, exit_keys, tickers))
        _merge(overall, o); _merge(buckets, b)
    else:
        import concurrent.futures as cf
        import multiprocessing as mp
        try:
            from django.db import connections
            connections.close_all()
        except Exception:
            pass
        chunks = _chunk(tickers, jobs * 3)
        payloads = [(signal_keys, exit_keys, c) for c in chunks]
        ctx = mp.get_context("spawn")
        done = 0
        with cf.ProcessPoolExecutor(max_workers=jobs, mp_context=ctx) as ex:
            for o, b in ex.map(_worker, payloads):
                _merge(overall, o); _merge(buckets, b)
                done += 1
                print(f"  ...{done}/{len(payloads)} chunks merged ({len(overall)} live combos)")

    # Build ranked result rows (only combos clearing the trade floor).
    dim_names = [d[0] for d in DIMENSIONS]
    dim_order = {d[0]: d[3] for d in DIMENSIONS}  # d = (name, field, fn, order, pit)
    results = []
    for (sk, ek), s in overall.items():
        if s[0] < min_trades:
            continue
        fin = _finalize_stat(s)
        by_dim = {}
        for dim in dim_names:
            rows = []
            for label in dim_order[dim]:
                bs = buckets.get((sk, ek, dim, label))
                if bs and bs[0] >= max(10, min_trades // 2):
                    rows.append({"bucket": label, **_finalize_stat(bs)})
            if rows:
                by_dim[dim] = rows
        results.append({
            "signal_key": sk, "signal_name": SIGNALS[sk][0],
            "exit_key": ek, "exit_name": EXITS[ek][0],
            "category": _categorize(sk),
            **fin, "by_dimension": by_dim,
        })
    results.sort(key=lambda r: r["avg_return"], reverse=True)

    out = {
        "universe_size": len(tickers),
        "n_signals": len(signal_keys), "n_exits": len(exit_keys),
        "min_trades": min_trades, "n_results": len(results),
        "results": results,
    }
    path = STUDIES_DIR / "stock_studies_all.json"
    path.write_text(json.dumps(out, indent=2))
    print(f"\nSaved {len(results)} combos (>= {min_trades} trades) -> {path}")

    if save_db:
        _save_to_db(results, len(tickers))

    print("\n=== TOP 25 signal × exit by avg return (>= {} trades) ===".format(min_trades))
    for r in results[:25]:
        print(f"  {r['avg_return']:+7.2f}%  {r['win_rate']:>4.0f}%wr  {r['trades']:>5}tr  "
              f"{r['signal_key']} -> {r['exit_key']}")
    return out


if __name__ == "__main__":
    argv = sys.argv

    def _opt(name, default=None):
        return argv[argv.index(name) + 1] if name in argv else default

    jobs = int(_opt("--jobs", DEFAULT_JOBS))
    limit = int(_opt("--limit")) if "--limit" in argv else None
    min_trades = int(_opt("--min-trades", 20))
    sig_arg = _opt("--signals")
    exit_arg = _opt("--exits")
    signal_keys = sig_arg.split(",") if sig_arg else None
    exit_keys = exit_arg.split(",") if exit_arg else None

    if "--db" in argv:
        import django
        os.environ.setdefault("DJANGO_SETTINGS_MODULE", "rotation.settings")
        django.setup()
    # Persist to Postgres StockStudy by default when --db (Django) is active; --no-db-save opts out.
    save_db = ("--db" in argv) and ("--no-db-save" not in argv)
    run(jobs, limit=limit, min_trades=min_trades,
        signal_keys=signal_keys, exit_keys=exit_keys, save_db=save_db)
