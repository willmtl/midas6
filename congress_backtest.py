#!/usr/bin/env python
"""Congressional (legislator) trade EQUITY-CURVE backtest — monthly, equal-weight, PIT, vs SPY.

The forward-AR study (congress_study.py) asked "do disclosed BUYS have forward edge?" per
trade. This asks the portfolio question: if you FOLLOWED the disclosures — equal-weight, roll
monthly — what would the equity curve have looked like vs just holding SPY?

METHOD (point-in-time, observational):
  - Universe = distinct US tickers in CongressTrade that have candle history (skip '.'-suffixed
    / non-US / no-candle names).
  - Monthly calendar comes from the candle history (month-end closes).
  - PIT ENTRY on report_date (PUBLIC disclosure) — a buy only becomes actionable once disclosed,
    so a name enters the book at the first month-end >= its report_date, NEVER on transaction_date.
  - HOLD window: a disclosed name stays in the book for `hold_months` months after it becomes
    known, then drops unless re-disclosed. Each rebalance month the book = every name disclosed
    in the trailing `hold_months` months.
  - Equal-weight the book; realize each name's NEXT-month return (delisted / NaN names drop from
    the basket that month); roll monthly. Compare to SPY over the identical months.

Strategies:
  1. follow_all_buys   — every stock with a disclosed BUY in the trailing window.
  2. senate_buys_only  — buys where chamber == 'Senate' (the slice that showed edge in the study).
  3. large_buys_50k    — buys with amount_min >= 50,000.
  4. follow_sells      — contrast: hold (long) names with disclosed SELLS (sells shouldn't predict
                         declines — this basket is expected to NOT beat, sanity-checking the method).

Each strategy reports _stats (total_return, spy_total, vs_spy, annual_return, sharpe,
max_drawdown, t_stat, periods) + a _curve equity curve.

OBSERVATIONAL / DIRECTIONAL — no costs, slippage, position sizing, or overlap control; not a
tradeable strategy. Mirrors the PIT equity-curve machinery of backtest_lowpb.py.

  export MSYS_NO_PATHCONV=1
  docker exec rotation-backend-1 python -u /app/congress_backtest.py --limit 120
  docker exec rotation-backend-1 python -u /app/congress_backtest.py --hold 3
"""
import os
import sys
import json
import argparse
import warnings

warnings.filterwarnings("ignore")

os.environ.setdefault("DJANGO_SETTINGS_MODULE", "rotation.settings")
import django  # noqa: E402
django.setup()

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from pathlib import Path  # noqa: E402

import config  # noqa: E402
from seq_fundamental_study import load_candles  # noqa: E402
from studies import _tstat_from_returns  # noqa: E402
from core.models import CongressTrade  # noqa: E402

BENCH = getattr(config, "BENCHMARK", "SPY")
OUT = Path("/app/.data/studies/congress_backtest.json")


# --- stats (mirror backtest_lowpb._stats / ._curve) --------------------------
def _stats(rets, spy_rets):
    r = np.array(rets, dtype=float)
    n = len(r)
    if n == 0:
        return {"total_return": 0, "spy_total": 0, "vs_spy": 0, "annual_return": 0,
                "sharpe": 0, "max_drawdown": 0, "t_stat": None, "periods": 0}
    total = float(np.prod(1 + r) - 1) * 100
    spy_total = float(np.prod(1 + np.array(spy_rets)) - 1) * 100
    ann = (float(np.prod(1 + r)) ** (12.0 / n) - 1) * 100
    sharpe = float(r.mean() / r.std() * np.sqrt(12)) if r.std() > 1e-9 else 0.0
    eq = np.cumprod(1 + r)
    dd = float(((eq / np.maximum.accumulate(eq)) - 1).min() * 100)
    t = _tstat_from_returns(list(r))
    return {"total_return": round(total, 1), "spy_total": round(spy_total, 1),
            "vs_spy": round(total - spy_total, 1), "annual_return": round(ann, 1),
            "sharpe": round(sharpe, 2), "max_drawdown": round(dd, 1),
            "t_stat": round(t, 2) if t is not None else None, "periods": n}


def _curve(rets, spy_rets, index):
    eq = np.cumprod(1 + np.array(rets)) if rets else []
    seq = np.cumprod(1 + np.array(spy_rets)) if spy_rets else []
    return [{"date": str(pd.Timestamp(d).date()), "strat": round(float(s), 4), "spy": round(float(sp), 4)}
            for d, s, sp in zip(index, eq, seq)]


def _arm(select_fn, index, spy_fwd, warmup):
    """Monthly loop: at each rebalance i, select_fn(i) returns the realized (index[i]->index[i+1])
    equal-weight portfolio return, or None to skip the period (e.g. empty book)."""
    rets, spy_rets, idx = [], [], []
    for i in range(warmup, len(index) - 1):
        sp = spy_fwd.iloc[i]
        if pd.isna(sp):
            continue
        pr = select_fn(i)
        if pr is None:
            continue
        rets.append(float(pr)); spy_rets.append(float(sp)); idx.append(index[i + 1])
    return {"summary": _stats(rets, spy_rets), "curve": _curve(rets, spy_rets, idx)}


# --- data --------------------------------------------------------------------
def _monthly_close(daily, min_len=60):
    """{ticker: daily OHLCV df} -> month-end Close DataFrame (union calendar)."""
    cols = {t: df["Close"].resample("ME").last() for t, df in daily.items() if len(df) > min_len}
    return pd.DataFrame(cols).sort_index()


def _is_us_ticker(tk):
    """Skip non-US / composite tickers (foreign exchanges carry a '.' suffix)."""
    if not tk:
        return False
    tk = tk.strip().upper()
    if not tk or "." in tk:
        return False
    return all(c.isalnum() or c == "-" for c in tk)


# strategy predicate: does this trade qualify for the strategy's book?
STRATEGIES = {
    "follow_all_buys": lambda t: (t["transaction_type"] or "").lower() == "buy",
    "senate_buys_only": lambda t: (t["transaction_type"] or "").lower() == "buy"
    and (t["chamber"] or "").strip().lower() == "senate",
    "large_buys_50k": lambda t: (t["transaction_type"] or "").lower() == "buy"
    and t["amount_min"] is not None and float(t["amount_min"]) >= 50000,
    "follow_sells": lambda t: (t["transaction_type"] or "").lower() == "sell",
}


def build(limit=None, hold_months=3):
    # ---- pull disclosed trades (report_date is the PIT anchor) ----
    qs = (CongressTrade.objects
          .exclude(report_date__isnull=True)
          .values("ticker", "chamber", "transaction_type", "report_date", "amount_min"))
    trades = list(qs)
    print(f"[congress-bt] loaded {len(trades)} disclosed trades (report_date not null)", flush=True)

    clean, counts = [], {}
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
    print(f"[congress-bt] {len(uni_set)} US tickers in universe (limit={limit}); "
          f"{len(clean)} trades on them", flush=True)

    # ---- load candles for universe + benchmark ----
    candles = load_candles(list(uni_set) + [BENCH])
    bench_df = candles.get(BENCH)
    if bench_df is None or bench_df.empty:
        raise SystemExit(f"[congress-bt] no candles for benchmark {BENCH}")

    have = {tk for tk in uni_set if tk in candles and not candles[tk].empty}
    print(f"[congress-bt] {len(have)}/{len(uni_set)} tickers have candles", flush=True)

    # ---- monthly calendar from candle history ----
    stock_monthly = _monthly_close({tk: candles[tk] for tk in have})
    midx = stock_monthly.index
    spy_m = bench_df["Close"].resample("ME").last().reindex(midx)

    # realized NEXT-month returns (index[i] -> index[i+1]); PIT-safe (book uses only past disclosures)
    stock_fwd = stock_monthly.pct_change().shift(-1)
    spy_fwd = spy_m.pct_change().shift(-1)
    print(f"[congress-bt] months={len(midx)} ({midx[0].date()}..{midx[-1].date()}) | "
          f"stocks={stock_monthly.shape[1]}", flush=True)

    # ---- per-strategy: month-position -> set(tickers disclosed & first known that month) ----
    # A trade first becomes actionable at the first month-end >= its report_date.
    def _disc_month(report_date):
        pos = midx.searchsorted(pd.Timestamp(report_date), side="left")
        return int(pos) if pos < len(midx) else None

    disc_by_month = {name: {} for name in STRATEGIES}
    for t in clean:
        if t["ticker"] not in have:
            continue
        j = _disc_month(t["report_date"])
        if j is None:
            continue
        for name, pred in STRATEGIES.items():
            if pred(t):
                disc_by_month[name].setdefault(j, set()).add(t["ticker"])

    def _make_sel(dbm):
        def sel(i):
            book = set()
            for j in range(max(0, i - hold_months + 1), i + 1):
                book |= dbm.get(j, set())
            cols = [tk for tk in book if tk in stock_fwd.columns]
            if not cols:
                return None
            fwd = stock_fwd.loc[midx[i], cols].dropna()
            return float(fwd.mean()) if len(fwd) else None
        return sel

    warmup = max(hold_months, 1)
    strategies = {}
    for name, dbm in disc_by_month.items():
        n_disc = sum(len(v) for v in dbm.values())
        arm = _arm(_make_sel(dbm), midx, spy_fwd, warmup)
        arm["disclosure_ticker_months"] = int(n_disc)
        strategies[name] = arm

    payload = {
        "computed_at": str(pd.Timestamp.utcnow()),
        "params": {"hold_months": hold_months, "benchmark": BENCH, "rebalance": "monthly",
                   "weighting": "equal_weight", "limit": limit,
                   "universe": {"stocks": int(stock_monthly.shape[1]), "months": int(len(midx))}},
        "strategies": strategies,
        "note": ("OBSERVATIONAL / DIRECTIONAL — monthly equal-weight, PIT entry on report_date "
                 "(public disclosure), held ~%d months. Realized next-month returns vs SPY over the "
                 "same months. No costs / slippage / sizing / overlap control; not a tradeable "
                 "strategy." % hold_months),
    }
    return payload


def _line(tag, s):
    return (f"  {tag:20} total {s['total_return']:>8.1f}%  SPY {s['spy_total']:>8.1f}%  "
            f"vs SPY {s['vs_spy']:>7.1f}%  Sharpe {s['sharpe']:>5.2f}  DD {s['max_drawdown']:>7.1f}%  "
            f"t={s['t_stat']}  n={s['periods']}")


def main():
    ap = argparse.ArgumentParser(description="Congressional-trade equity-curve backtest (PIT, vs SPY)")
    ap.add_argument("--limit", type=int, default=None,
                    help="cap universe to N most-traded tickers (light test)")
    ap.add_argument("--hold", type=int, default=3, help="hold window in months (default 3)")
    args = ap.parse_args()

    payload = build(limit=args.limit, hold_months=args.hold)

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(payload, indent=2, default=str))
    print(f"\n[congress-bt] wrote {OUT}", flush=True)

    try:
        from core.models import BacktestResult
        from django.utils import timezone
        db_payload = json.loads(json.dumps(payload, default=str))
        BacktestResult.objects.update_or_create(
            kind="congress_backtest", defaults={"payload": db_payload, "computed_at": timezone.now()})
        print("[congress-bt] saved to DB (BacktestResult kind=congress_backtest)", flush=True)
    except Exception as e:
        print("[congress-bt] DB save failed:", e, flush=True)

    p = payload["params"]
    print("\n" + "=" * 96, flush=True)
    print(f"CONGRESSIONAL-TRADE EQUITY-CURVE BACKTEST vs {p['benchmark']}  "
          f"(monthly, equal-weight, PIT on report_date, hold={p['hold_months']}m)", flush=True)
    print(f"universe={p['universe']['stocks']} stocks   months={p['universe']['months']}", flush=True)
    print(payload["note"], flush=True)
    print("=" * 96, flush=True)
    for name, arm in payload["strategies"].items():
        print(_line(name, arm["summary"]), flush=True)


if __name__ == "__main__":
    main()
