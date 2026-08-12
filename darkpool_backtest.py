#!/usr/bin/env python3
"""Historical DARK-POOL backtest engine -> .data/studies/darkpool_backtest.json + Postgres.

Question: does a stock's OFF-EXCHANGE (ATS / dark-pool) share of volume carry a
tradeable signal? We build a monthly, equal-weight, point-in-time backtest over the
FINRA OTC-Transparency weekly ATS data (`core.models.DarkPoolWeek`) and compare four
baskets vs SPY, plus a bucketed forward-return table (does concentration matter?).

POINT-IN-TIME: every weekly off_pct is gated on `published_date` (the PUBLIC FINRA
disclosure date, which already bakes in FINRA's ~2-4wk lag). For each month-end we take
each stock's MOST-RECENT weekly off_pct whose published_date <= month-end (forward-fill by
publish date; NaN before a stock's first publish). No lookahead.

Strategies (monthly rebalance, equal-weight, next-month realized return vs SPY):
  1. "High dark-pool share"   top-N by PIT off_pct, gated off_pct >= FLOOR.
  2. "Dark-pool accumulation" top-N by RISE in off_pct vs ~LOOKBACK months prior (>0).
  3. "Low dark-pool share"    bottom-N by off_pct (contrast arm).
  4. "All-with-data baseline" equal-weight every stock with a PIT off_pct that month.

Universe = stocks with BOTH DB candles AND DarkPoolWeek rows. Baskets realize via next-month
pct_change (delisted names drop out as NaN). OBSERVATIONAL, not transaction-cost adjusted.

Run in the backend container:
  docker exec rotation-backend-1 python -u /app/darkpool_backtest.py
  docker exec rotation-backend-1 python -u /app/darkpool_backtest.py --limit 150   # quick subset
Options: --limit N (cap universe), --top N (basket size).
"""
import os, sys, json, argparse, warnings
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
os.environ.setdefault("DJANGO_SETTINGS_MODULE", "rotation.settings")
import django
django.setup()

from pathlib import Path
import config
from studies import _tstat_from_returns
from seq_fundamental_study import load_candles

OUT = Path(__file__).resolve().parent / ".data" / "studies" / "darkpool_backtest.json"
BENCH = getattr(config, "BENCHMARK", "SPY")

# --- knobs -------------------------------------------------------------------
TOP_N = 20            # basket size
FLOOR = 0.10          # min PIT off_pct to qualify for the "High dark-pool share" arm
LOOKBACK = 2          # months for the accumulation (rise-in-off_pct) trend
MIN_LEN = 60          # min daily bars for a stock to enter the monthly frame
# forward-return buckets by off_pct level (does concentration matter?)
BUCKETS = [
    ("low (<5%)",       0.00, 0.05),
    ("mid (5-12%)",     0.05, 0.12),
    ("high (12-20%)",   0.12, 0.20),
    ("very-high (>=20%)", 0.20, np.inf),
]


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
    """Monthly loop: at rebalance i, select_fn(i) returns the realized (index[i]->index[i+1])
    equal-weight basket return, or None to skip the period (no candidates)."""
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
def _monthly_close(daily, min_len=MIN_LEN):
    cols = {t: df["Close"].resample("ME").last() for t, df in daily.items() if len(df) > min_len}
    return pd.DataFrame(cols).sort_index()


def _darkpool_universe():
    """Distinct DarkPoolWeek tickers that also have DB candles."""
    from core.models import DarkPoolWeek, Candle
    have_dp = set(DarkPoolWeek.objects.values_list("ticker", flat=True).distinct())
    have_candles = set(Candle.objects.values_list("ticker", flat=True).distinct())
    return sorted(have_dp & have_candles)


def _pit_offpct_panel(tickers, monthly_index):
    """Month x ticker panel of the MOST-RECENT weekly off_pct whose published_date <= month-end
    (forward-fill by PUBLISH date; NaN before first publish). Lookahead-safe."""
    from core.models import DarkPoolWeek
    qs = (DarkPoolWeek.objects.filter(ticker__in=list(tickers), off_pct__isnull=False,
                                      published_date__isnull=False)
          .values_list("ticker", "published_date", "off_pct"))
    big = pd.DataFrame.from_records(list(qs), columns=["ticker", "published_date", "off_pct"])
    out = {}
    if big.empty:
        return pd.DataFrame(out)
    big["published_date"] = pd.to_datetime(big["published_date"])
    for tk, g in big.groupby("ticker", sort=False):
        s = g.set_index("published_date")["off_pct"].sort_index()
        # multiple weeks can share a publish date -> keep the last (most recent week_start order lost
        # here, but same publish date means simultaneously-disclosed; last is a safe tie-break).
        s = s[~s.index.duplicated(keep="last")]
        out[tk] = s.reindex(s.index.union(monthly_index)).ffill().reindex(monthly_index)
    return pd.DataFrame(out)


def build(limit=None, top_n=TOP_N):
    universe = _darkpool_universe()
    if limit:
        universe = universe[:limit]
    print(f"Loading {len(universe)} dark-pool stocks + {BENCH} from DB...", flush=True)

    daily = load_candles(universe + [BENCH])
    spy_daily = daily.get(BENCH)
    stock_monthly = _monthly_close({t: d for t, d in daily.items() if t in universe})
    if stock_monthly.empty:
        raise SystemExit("no monthly stock data")
    midx = stock_monthly.index
    spy_m = (spy_daily["Close"].resample("ME").last().reindex(midx) if spy_daily is not None
             else pd.Series(index=midx, dtype=float))

    # PIT off_pct panel on the same monthly calendar.
    off_panel = _pit_offpct_panel(universe, midx)
    off_panel = off_panel.reindex(columns=[c for c in stock_monthly.columns if c in off_panel.columns])
    off_rise = off_panel - off_panel.shift(LOOKBACK)

    stock_fwd = stock_monthly.pct_change().shift(-1)
    spy_fwd = spy_m.pct_change().shift(-1)

    n_with_dp = int((off_panel.notna().any(axis=0)).sum())
    print(f"months {len(midx)} | stocks {stock_monthly.shape[1]} | with dark-pool PIT {n_with_dp}",
          flush=True)

    warmup = max(LOOKBACK, 1)

    def _basket_ret(date, picks):
        cols = [p for p in picks if p in stock_fwd.columns]
        if not cols:
            return None
        fwd = stock_fwd.loc[date, cols].dropna()
        return float(fwd.mean()) if len(fwd) else None

    # ---- strategy selectors -------------------------------------------------
    def _high(i):
        date = midx[i]
        row = off_panel.loc[date].dropna()
        row = row[row >= FLOOR]
        if not len(row):
            return None
        return _basket_ret(date, row.nlargest(top_n).index)

    def _accumulation(i):
        date = midx[i]
        row = off_rise.loc[date].dropna()
        row = row[row > 0]                       # trending UP only
        if not len(row):
            return None
        return _basket_ret(date, row.nlargest(top_n).index)

    def _low(i):
        date = midx[i]
        row = off_panel.loc[date].dropna()
        if not len(row):
            return None
        return _basket_ret(date, row.nsmallest(top_n).index)

    def _baseline(i):
        date = midx[i]
        row = off_panel.loc[date].dropna()
        if not len(row):
            return None
        return _basket_ret(date, row.index)

    strategies = {
        "High dark-pool share": _arm(_high, midx, spy_fwd, warmup),
        "Dark-pool accumulation": _arm(_accumulation, midx, spy_fwd, warmup),
        "Low dark-pool share": _arm(_low, midx, spy_fwd, warmup),
        "All-with-data baseline": _arm(_baseline, midx, spy_fwd, warmup),
    }

    # ---- bucketed forward-return table --------------------------------------
    # For each month & stock with a PIT off_pct, bucket by level, collect next-month return.
    buckets = {label: {"rets": []} for label, _, _ in BUCKETS}
    for i in range(warmup, len(midx) - 1):
        date = midx[i]
        row = off_panel.loc[date].dropna()
        if not len(row):
            continue
        fwd = stock_fwd.loc[date]
        for tk, ov in row.items():
            r = fwd.get(tk)
            if r is None or pd.isna(r):
                continue
            for label, lo, hi in BUCKETS:
                if lo <= ov < hi:
                    buckets[label]["rets"].append(float(r))
                    break
    bucket_out = {}
    for label, _, _ in BUCKETS:
        rs = buckets[label]["rets"]
        bucket_out[label] = {
            "avg_ret_pct": round(float(np.mean(rs)) * 100, 3) if rs else None,
            "n": len(rs),
        }

    payload = {
        "computed_at": pd.Timestamp.utcnow().isoformat(),
        "params": {"top_n": top_n, "floor": FLOOR, "lookback": LOOKBACK, "benchmark": BENCH,
                   "universe": {"stocks": int(stock_monthly.shape[1]), "months": int(len(midx))}},
        "strategies": strategies,
        "buckets": bucket_out,
        "note": ("Observational, NOT transaction-cost adjusted. PIT on FINRA published_date "
                 "(respects the ~2-4wk disclosure lag). off_pct = ATS share of consolidated volume."),
    }
    return payload


def _line(tag, s):
    return (f"  {tag:26} total {s['total_return']:>8.1f}%  vs SPY {s['vs_spy']:>7.1f}%  "
            f"Sharpe {s['sharpe']:>5.2f}  DD {s['max_drawdown']:>7.1f}%  t={s['t_stat']}  n={s['periods']}")


def _save_db(payload):
    try:
        from core.models import BacktestResult
        from django.utils import timezone
        BacktestResult.objects.update_or_create(
            kind="darkpool_backtest",
            defaults={"payload": json.loads(json.dumps(payload, default=str)),
                      "computed_at": timezone.now()})
        print("Saved BacktestResult[darkpool_backtest] to DB", flush=True)
    except Exception as e:
        print(f"DB save failed for darkpool_backtest: {e}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=None, help="cap universe for a light test")
    ap.add_argument("--top", type=int, default=TOP_N, help="basket size (top-N)")
    args = ap.parse_args()

    payload = build(limit=args.limit, top_n=args.top)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(payload, indent=2, default=str))
    _save_db(payload)

    print("\n=== DARK-POOL BACKTEST (monthly, equal-weight, PIT on published_date) ===", flush=True)
    for name, arm in payload["strategies"].items():
        print(_line(name, arm["summary"]), flush=True)

    print("\n=== OFF_PCT BUCKET -> next-month mean return (does concentration matter?) ===", flush=True)
    for label, _, _ in BUCKETS:
        b = payload["buckets"][label]
        av = f"{b['avg_ret_pct']:>7.3f}%" if b["avg_ret_pct"] is not None else "   n/a "
        print(f"  {label:20} avg {av}   n={b['n']}", flush=True)

    print("\nSaved ->", OUT, flush=True)


if __name__ == "__main__":
    main()
