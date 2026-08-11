#!/usr/bin/env python
"""Congressional (legislator) disclosed-BUY forward-edge study — PIT, market-adjusted.

Question: do US congressional disclosed BUYS have forward edge vs SPY?

METHOD (point-in-time, observational):
  - Universe = distinct US tickers in CongressTrade that have candle history.
  - For each trade, ENTER at the first trading day whose date >= report_date (the
    PUBLIC disclosure date — NOT transaction_date). This is the moment the info
    became public, so it is the only tradeable, non-lookahead entry.
  - Forward RAW return over 21 / 63 / 126 trading days (~1m/3m/6m). SPY return over
    the SAME calendar window. Market-adjusted return AR = stock_ret - spy_ret (%).
  - Drop trades too recent to have the full window, or with no post-entry candles.
  - Aggregate AR overall and sliced by transaction_type, chamber, party, amount
    bucket (amount_min), and the top-15 members by trade count. Report n, avg AR%,
    win-rate (%AR>0), t_stat.

This is DIRECTIONAL / OBSERVATIONAL — not a tradeable backtest (no costs, slippage,
position sizing, or overlap control).
"""
import os
import argparse
import json

os.environ.setdefault("DJANGO_SETTINGS_MODULE", "rotation.settings")
import django  # noqa: E402
django.setup()

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from seq_fundamental_study import load_candles  # noqa: E402
from studies import _tstat_from_returns  # noqa: E402
import config  # noqa: E402
from core.models import CongressTrade  # noqa: E402

BENCH = getattr(config, "BENCHMARK", "SPY")


def _is_us_ticker(tk):
    """Skip non-US / composite tickers (foreign exchanges carry a '.' suffix)."""
    if not tk:
        return False
    tk = tk.strip().upper()
    if not tk or "." in tk:
        return False
    # keep plain alnum tickers (allow '-' for BRK-B style)
    return all(c.isalnum() or c == "-" for c in tk)


def _fwd_return(close, entry_idx, horizon):
    """Raw fwd % return from entry_idx to entry_idx+horizon. None if window truncated."""
    exit_idx = entry_idx + horizon
    if exit_idx >= len(close):
        return None
    p0 = close.iloc[entry_idx]
    p1 = close.iloc[exit_idx]
    if not (p0 > 0) or not np.isfinite(p1):
        return None
    return (p1 / p0 - 1.0) * 100.0


def _entry_idx(idx, report_date):
    """First candle position whose date >= report_date (searchsorted, PIT)."""
    pos = idx.searchsorted(pd.Timestamp(report_date), side="left")
    if pos >= len(idx):
        return None
    return int(pos)


def _amount_bucket(amount_min):
    if amount_min is None:
        return "unknown"
    a = float(amount_min)
    if a < 15000:
        return "<15k"
    if a < 50000:
        return "15k-50k"
    if a < 250000:
        return "50k-250k"
    return "250k+"


def _cell(ars):
    """Summary dict for a list of AR values."""
    arr = np.asarray([x for x in ars if x is not None and np.isfinite(x)], dtype=float)
    n = len(arr)
    if n == 0:
        return {"n": 0, "avg_ar": None, "win_rate": None, "t_stat": None}
    return {
        "n": int(n),
        "avg_ar": round(float(arr.mean()), 3),
        "win_rate": round(float((arr > 0).mean() * 100.0), 1),
        "t_stat": _tstat_from_returns(list(arr)),
    }


def compute(limit=None, horizons=(21, 63, 126)):
    # ---- pull trades ----
    qs = (CongressTrade.objects
          .exclude(report_date__isnull=True)
          .values("ticker", "member", "chamber", "party", "state",
                  "transaction_type", "report_date", "amount_min"))
    trades = list(qs)
    print(f"[congress] loaded {len(trades)} disclosed trades (report_date not null)")

    # normalize + filter to US tickers
    clean = []
    counts = {}
    for t in trades:
        tk = (t["ticker"] or "").strip().upper()
        if not _is_us_ticker(tk):
            continue
        t["ticker"] = tk
        clean.append(t)
        counts[tk] = counts.get(tk, 0) + 1

    universe = sorted(counts, key=lambda k: counts[k], reverse=True)
    if limit:
        universe = universe[:limit]
    uni_set = set(universe)
    clean = [t for t in clean if t["ticker"] in uni_set]
    print(f"[congress] {len(uni_set)} US tickers in universe (limit={limit}); "
          f"{len(clean)} trades on them")

    # ---- load candles for universe + bench ----
    candles = load_candles(list(uni_set) + [BENCH])
    bench_df = candles.get(BENCH)
    if bench_df is None or bench_df.empty:
        raise SystemExit(f"[congress] no candles for benchmark {BENCH}")
    bench_close = bench_df["Close"]
    bench_idx = bench_df.index

    have = {tk for tk in uni_set if tk in candles and not candles[tk].empty}
    print(f"[congress] {len(have)}/{len(uni_set)} tickers have candles; "
          f"benchmark {BENCH} candles={len(bench_close)}")

    horizons = list(horizons)
    # records[horizon] = list of (ar, meta_dict)
    records = {h: [] for h in horizons}
    used = 0
    skipped_no_candle = 0
    skipped_no_entry = 0

    for t in clean:
        tk = t["ticker"]
        if tk not in have:
            skipped_no_candle += 1
            continue
        df = candles[tk]
        close = df["Close"]
        idx = df.index
        eidx = _entry_idx(idx, t["report_date"])
        if eidx is None:
            skipped_no_entry += 1
            continue
        entry_date = idx[eidx]
        # bench entry aligned to same calendar day
        beidx = bench_idx.searchsorted(entry_date, side="left")
        if beidx >= len(bench_idx):
            continue

        meta = {
            "transaction_type": (t["transaction_type"] or "").lower(),
            "chamber": t["chamber"] or "unknown",
            "party": t["party"] or "unknown",
            "amount_bucket": _amount_bucket(t["amount_min"]),
            "member": t["member"] or "unknown",
        }
        contributed = False
        for h in horizons:
            sret = _fwd_return(close, eidx, h)
            bret = _fwd_return(bench_close, int(beidx), h)
            if sret is None or bret is None:
                continue
            ar = sret - bret
            records[h].append((ar, sret, bret, meta))
            contributed = True
        if contributed:
            used += 1

    print(f"[congress] used {used} trades; skipped {skipped_no_candle} (no candle), "
          f"{skipped_no_entry} (no post-report candle)")

    # ---- aggregate ----
    def slice_by(recs, keyfn, txn_filter=None):
        """Return {group: cell} of AR summaries."""
        groups = {}
        for ar, sret, bret, meta in recs:
            if txn_filter is not None and meta["transaction_type"] != txn_filter:
                continue
            g = keyfn(meta)
            groups.setdefault(g, []).append(ar)
        return {g: _cell(v) for g, v in groups.items()}

    # top-15 members by trade count (across all horizons' first horizon set)
    mem_counts = {}
    for ar, sret, bret, meta in records[horizons[0]]:
        mem_counts[meta["member"]] = mem_counts.get(meta["member"], 0) + 1
    top_members = set(sorted(mem_counts, key=lambda k: mem_counts[k], reverse=True)[:15])

    out = {
        "computed_at": str(pd.Timestamp.utcnow()),
        "benchmark": BENCH,
        "universe_size": len(have),
        "trades_used": used,
        "horizons": horizons,
        "note": ("DIRECTIONAL / OBSERVATIONAL — PIT entry on report_date (public "
                 "disclosure). Market-adjusted AR = stock_ret - SPY_ret. No costs/"
                 "slippage/overlap control; not a tradeable backtest."),
        "by_horizon": {},
    }

    for h in horizons:
        recs = records[h]
        all_ars = [r[0] for r in recs]
        buy_ars = [r[0] for r in recs if r[3]["transaction_type"] == "buy"]
        sell_ars = [r[0] for r in recs if r[3]["transaction_type"] == "sell"]
        # baseline raw returns for buys vs SPY
        buy_raw = [r[1] for r in recs if r[3]["transaction_type"] == "buy"]
        buy_spy = [r[2] for r in recs if r[3]["transaction_type"] == "buy"]

        hres = {
            "overall": _cell(all_ars),
            "buy": _cell(buy_ars),
            "sell": _cell(sell_ars),
            "baseline_buy_raw_vs_spy": {
                "n": len(buy_raw),
                "avg_buy_raw_ret": round(float(np.mean(buy_raw)), 3) if buy_raw else None,
                "avg_spy_ret_same_window": round(float(np.mean(buy_spy)), 3) if buy_spy else None,
            },
            "buy_by_transaction_type": slice_by(recs, lambda m: m["transaction_type"]),
            "buy_by_chamber": slice_by(recs, lambda m: m["chamber"], txn_filter="buy"),
            "buy_by_party": slice_by(recs, lambda m: m["party"], txn_filter="buy"),
            "buy_by_amount_bucket": slice_by(recs, lambda m: m["amount_bucket"], txn_filter="buy"),
            "sell_by_chamber": slice_by(recs, lambda m: m["chamber"], txn_filter="sell"),
            "sell_by_party": slice_by(recs, lambda m: m["party"], txn_filter="sell"),
            "buy_by_member_top15": {
                m: c for m, c in
                slice_by([r for r in recs if r[3]["member"] in top_members],
                         lambda meta: meta["member"], txn_filter="buy").items()
            },
        }
        out["by_horizon"][str(h)] = hres

    return out


def _fmt_cell(c):
    if not c or c.get("n", 0) == 0:
        return "     n=0"
    t = c["t_stat"]
    tstr = f"{t:+.2f}" if t is not None else "  n/a"
    return f"n={c['n']:>5}  AR={c['avg_ar']:+7.2f}%  win={c['win_rate']:>5.1f}%  t={tstr}"


def print_summary(out):
    hs = out["horizons"]
    headline = str(63) if 63 in hs else str(hs[len(hs) // 2])
    print("\n" + "=" * 78)
    print(f"CONGRESSIONAL BUY FORWARD EDGE vs {out['benchmark']}  (PIT on report_date)")
    print(f"universe={out['universe_size']} tickers   trades_used={out['trades_used']}")
    print(out["note"])
    print("=" * 78)

    for h in hs:
        hr = out["by_horizon"][str(h)]
        star = "  <<< HEADLINE" if str(h) == headline else ""
        label = {21: "~1m", 63: "~3m", 126: "~6m"}.get(h, "")
        print(f"\n--- horizon {h}d {label}{star} ---")
        print(f"  OVERALL   {_fmt_cell(hr['overall'])}")
        print(f"  BUY       {_fmt_cell(hr['buy'])}")
        print(f"  SELL      {_fmt_cell(hr['sell'])}")
        b = hr["baseline_buy_raw_vs_spy"]
        if b["avg_buy_raw_ret"] is not None:
            print(f"  baseline  buys raw {b['avg_buy_raw_ret']:+.2f}%  vs SPY "
                  f"{b['avg_spy_ret_same_window']:+.2f}%  (n={b['n']})")
        print("  BUY by chamber:")
        for g, c in sorted(hr["buy_by_chamber"].items(), key=lambda kv: -(kv[1]["n"] or 0)):
            print(f"    {g:<10} {_fmt_cell(c)}")
        print("  BUY by party:")
        for g, c in sorted(hr["buy_by_party"].items(), key=lambda kv: -(kv[1]["n"] or 0)):
            print(f"    {g:<10} {_fmt_cell(c)}")
        print("  BUY by amount bucket:")
        for g in ["<15k", "15k-50k", "50k-250k", "250k+", "unknown"]:
            if g in hr["buy_by_amount_bucket"]:
                print(f"    {g:<10} {_fmt_cell(hr['buy_by_amount_bucket'][g])}")

    # headline slices best/worst
    hr = out["by_horizon"][headline]
    print(f"\n--- headline ({headline}d) BUY party/chamber ranking ---")
    combined = []
    for g, c in hr["buy_by_party"].items():
        if c["n"]:
            combined.append(("party:" + g, c))
    for g, c in hr["buy_by_chamber"].items():
        if c["n"]:
            combined.append(("chamber:" + g, c))
    combined.sort(key=lambda kv: -(kv[1]["avg_ar"] or -1e9))
    for name, c in combined:
        print(f"    {name:<18} {_fmt_cell(c)}")

    print("\n--- headline BUY top-15 members by trade count ---")
    for m, c in sorted(hr["buy_by_member_top15"].items(),
                       key=lambda kv: -(kv[1]["avg_ar"] or -1e9)):
        print(f"    {m:<28} {_fmt_cell(c)}")


def main():
    ap = argparse.ArgumentParser(description="Congressional buy forward-edge study (PIT, market-adjusted)")
    ap.add_argument("--limit", type=int, default=None,
                    help="cap universe to N most-traded tickers (light test)")
    ap.add_argument("--horizons", type=str, default="21,63,126",
                    help="comma-separated forward trading-day horizons")
    args = ap.parse_args()
    horizons = [int(x) for x in args.horizons.split(",") if x.strip()]

    out = compute(limit=args.limit, horizons=horizons)

    outdir = "/app/.data/studies"
    os.makedirs(outdir, exist_ok=True)
    outpath = os.path.join(outdir, "congress_study.json")
    with open(outpath, "w") as f:
        json.dump(out, f, indent=2, default=str)
    print(f"\n[congress] wrote {outpath}")

    print_summary(out)


if __name__ == "__main__":
    main()
