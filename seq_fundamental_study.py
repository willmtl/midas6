#!/usr/bin/env python3
"""Run the seq WINNER signal on ALL individual stocks (not the 93 sector ETFs), then
bucket the resulting trades by fundamentals (PE, forward PE, EPS, revenue growth, margin,
float, market cap) to see which kinds of companies actually rebound.

Universe = every ticker that has BOTH DB candles AND a Fundamental row, minus the sector
ETFs / benchmarks. Signal defaults to `seq_rsi20_ad_rising_rsi`; exit defaults to `6m`.

⚠️ LOOKAHEAD CAVEAT: fundamentals are yfinance's CURRENT snapshot, not point-in-time. A
trade from 3y ago is bucketed by the company's PE/float TODAY. Read results as directional
(does 'the kind of company that looks good now' rebound better historically), not tradable.

Run in the backend container:
  docker compose run --rm backend python -u seq_fundamental_study.py --db
Options: --signal <key>  --exit <key>  --limit N
"""
import warnings
warnings.filterwarnings("ignore")

import json
import os
import sys
from pathlib import Path

from studies import SIGNALS, EXITS, _episode_starts, _tstat_from_returns

STUDIES_DIR = Path(__file__).parent / ".data" / "studies"
STUDIES_DIR.mkdir(parents=True, exist_ok=True)

MIN_BARS = 60
MIN_BUCKET_TRADES = 10  # don't report a fundamental bucket with fewer trades than this

# Parallelism: this run is CPU-bound (per-ticker pandas/`ta` signal computation), so
# we use PROCESSES (not threads — the GIL makes threads useless for this) capped at the
# core count. Each worker loads its own candle chunk from the DB so we never pickle big
# DataFrames across process boundaries — only the small (ticker, return) trade list comes
# back. Default job count = min(32, cpu_count); override with --jobs N.
DEFAULT_JOBS = min(32, os.cpu_count() or 4)


def build_universe():
    """Distinct DB-candle tickers that also have fundamentals, minus ETFs/benchmarks."""
    from core.models import Candle, Fundamental, Sector
    etfs = set(Sector.objects.values_list("etf", flat=True)) | {"SPY", "QQQ"}
    have_fund = set(Fundamental.objects.values_list("ticker", flat=True))
    have_candles = set(Candle.objects.values_list("ticker", flat=True).distinct())
    return sorted((have_candles & have_fund) - etfs)


def load_candles(tickers, interval="1d"):
    import pandas as pd
    from core.models import Candle
    qs = (Candle.objects.filter(ticker__in=list(tickers), interval=interval)
          .values_list("ticker", "date", "open", "high", "low", "close", "volume"))
    big = pd.DataFrame.from_records(
        list(qs), columns=["ticker", "date", "Open", "High", "Low", "Close", "Volume"])
    if big.empty:
        return {}
    big["date"] = pd.to_datetime(big["date"])
    out = {}
    for tk, g in big.groupby("ticker", sort=False):
        out[tk] = g.sort_values("date").set_index("date")[["Open", "High", "Low", "Close", "Volume"]]
    return out


def load_financial_reports(tickers):
    """{ticker: DataFrame of quarterly reports} for point-in-time metrics."""
    import pandas as pd
    from core.models import FinancialReport
    cols = ["ticker", "period_end", "avail_date", "revenue", "net_income", "eps_diluted",
            "operating_income", "total_equity", "total_debt", "current_assets",
            "current_liabilities", "free_cash_flow", "shares_outstanding",
            "operating_cash_flow", "total_assets", "gross_profit", "cost_of_revenue",
            "rd_expense", "inventory", "cash_and_equivalents"]
    qs = FinancialReport.objects.filter(ticker__in=list(tickers)).values_list(*cols)
    big = pd.DataFrame.from_records(list(qs), columns=cols)
    out = {}
    if big.empty:
        return out
    for tk, g in big.groupby("ticker", sort=False):
        out[tk] = g.drop(columns=["ticker"]).reset_index(drop=True)
    return out


def load_dividends(tickers):
    """{ticker: Series indexed by ex_date}."""
    import pandas as pd
    from core.models import DividendHistory
    qs = DividendHistory.objects.filter(ticker__in=list(tickers)).values_list("ticker", "ex_date", "amount")
    big = pd.DataFrame.from_records(list(qs), columns=["ticker", "ex_date", "amount"])
    out = {}
    if big.empty:
        return out
    big["ex_date"] = pd.to_datetime(big["ex_date"])
    for tk, g in big.groupby("ticker", sort=False):
        out[tk] = g.set_index("ex_date")["amount"].sort_index()
    return out


def load_filings(tickers):
    """{ticker: DataFrame[filed_date, form_group]} of 13D/13G filings."""
    import pandas as pd
    from core.models import SecFiling
    qs = SecFiling.objects.filter(ticker__in=list(tickers)).values_list("ticker", "filed_date", "form_group")
    big = pd.DataFrame.from_records(list(qs), columns=["ticker", "filed_date", "form_group"])
    out = {}
    if big.empty:
        return out
    for tk, g in big.groupby("ticker", sort=False):
        out[tk] = g[["filed_date", "form_group"]].reset_index(drop=True)
    return out


def load_insider(tickers):
    """{ticker: Series of open-market buy $ indexed by filed_date} for insider dimension."""
    import pandas as pd
    from core.models import InsiderBuy
    qs = InsiderBuy.objects.filter(ticker__in=list(tickers)).values_list("ticker", "filed_date", "buy_value")
    big = pd.DataFrame.from_records(list(qs), columns=["ticker", "filed_date", "buy_value"])
    out = {}
    if big.empty:
        return out
    big["filed_date"] = pd.to_datetime(big["filed_date"])
    for tk, g in big.groupby("ticker", sort=False):
        out[tk] = g.set_index("filed_date")["buy_value"].sort_index()
    return out


def load_darkpool(tickers):
    """{ticker: DataFrame[off_pct, published_date]} of FINRA weekly ATS dark-pool volume, for the
    point-in-time dark-pool dimensions (keyed by publish date to respect the reporting lag)."""
    import pandas as pd
    from core.models import DarkPoolWeek
    qs = (DarkPoolWeek.objects.filter(ticker__in=list(tickers), off_pct__isnull=False,
                                      published_date__isnull=False)
          .values_list("ticker", "off_pct", "published_date"))
    big = pd.DataFrame.from_records(list(qs), columns=["ticker", "off_pct", "published_date"])
    out = {}
    if big.empty:
        return out
    for tk, g in big.groupby("ticker", sort=False):
        out[tk] = g[["off_pct", "published_date"]].reset_index(drop=True)
    return out


def load_news(tickers):
    """{ticker: Series of sentiment polarity indexed by news datetime} for the news dimension."""
    import pandas as pd
    from core.models import NewsItem
    qs = (NewsItem.objects.filter(ticker__in=list(tickers), sentiment__isnull=False)
          .values_list("ticker", "dt", "sentiment"))
    big = pd.DataFrame.from_records(list(qs), columns=["ticker", "dt", "sentiment"])
    out = {}
    if big.empty:
        return out
    # Convert to ET and roll news released at/after the 16:00 ET close to the NEXT session
    # before bucketing — UTC-date bucketing folded after-close ET news (16:00-20:00 ET, still
    # the same UTC date) into the bar that had already closed, a same-bar leak into news_sent.
    # (Mirrors api/news_market_study.py's tz_convert+after-close shift.)
    _dt = pd.to_datetime(big["dt"], utc=True).dt.tz_convert("America/New_York")
    _after_close = (_dt.dt.hour >= 16).astype("int64")
    big["dt"] = _dt.dt.tz_localize(None).dt.normalize() + pd.to_timedelta(_after_close, unit="D")
    for tk, g in big.groupby("ticker", sort=False):
        out[tk] = g.set_index("dt")["sentiment"].sort_index()
    return out


def label_trade(pit_metrics, entry_date, snap):
    """Bucket labels for one trade across all DIMENSIONS.
    PIT dims read pit_metrics via a forward-fill lookup at entry_date; snapshot dims read
    the snap dict. NaN (PIT columns that aren't defined yet) is coerced to None so the
    bucket fns' `if v is None` guards route it to the 'NA' bucket."""
    from pit_fundamentals import pit_row
    row = pit_row(pit_metrics, entry_date)
    labels = {}
    for name, field, bfn, _order, pit in DIMENSIONS:
        if pit:
            v = None if row is None else row.get(field)
        else:
            v = (snap or {}).get(field)
        if isinstance(v, float) and v != v:  # NaN -> None
            v = None
        labels[name] = bfn(v)
    return labels


def load_fundamentals(tickers):
    """Latest Fundamental row per ticker -> dict of the fields we bucket on."""
    from core.models import Fundamental
    # Union of fields needed by: the 6 snapshot DIMENSIONS (forward_pe, forward_eps,
    # short_pct_float, institution_pct, insider_pct, float_shares) + the live-firing
    # display (market_cap, pe_ratio, profit_margin). PIT dims read the metrics frame, not
    # this snapshot, so their fields are intentionally absent here.
    rows = (Fundamental.objects.filter(ticker__in=list(tickers))
            .order_by("ticker", "-date")
            .values("ticker", "forward_pe", "forward_eps", "short_pct_float",
                    "institution_pct", "insider_pct", "float_shares",
                    "market_cap", "pe_ratio", "profit_margin"))
    out = {}
    for r in rows:
        if r["ticker"] not in out:  # first = latest (ordered -date)
            out[r["ticker"]] = r
    return out


# ── Fundamental bucketing ──────────────────────────────────────────────────
def _bucket_pe(v):
    if v is None: return "NA"
    if v < 0: return "negative (no earnings)"
    if v < 15: return "cheap (<15)"
    if v < 30: return "fair (15-30)"
    if v < 50: return "rich (30-50)"
    return "expensive (>=50)"

def _bucket_fwd_pe(v):
    if v is None: return "NA"
    if v < 0: return "negative"
    if v < 15: return "cheap (<15)"
    if v < 30: return "fair (15-30)"
    return "rich (>=30)"

def _bucket_eps(v):
    if v is None: return "NA"
    return "positive EPS" if v > 0 else "negative EPS"

def _bucket_growth(v):
    if v is None: return "NA"
    if v < 0: return "shrinking (<0%)"
    if v < 0.15: return "slow (0-15%)"
    if v < 0.40: return "growing (15-40%)"
    return "hypergrowth (>=40%)"

def _bucket_margin(v):
    if v is None: return "NA"
    if v < 0: return "unprofitable (<0%)"
    if v < 0.10: return "thin (0-10%)"
    return "healthy (>=10%)"

def _bucket_float(v):
    if v is None: return "NA"
    if v < 20e6: return "micro-float (<20M)"
    if v < 100e6: return "low-float (20-100M)"
    if v < 500e6: return "mid-float (100-500M)"
    return "high-float (>=500M)"

def _bucket_mktcap(v):
    if v is None: return "NA"
    if v < 500e6: return "micro (<500M)"
    if v < 2e9: return "small (0.5-2B)"
    if v < 10e9: return "mid (2-10B)"
    return "large (>=10B)"

def bucket_pct_52w(v):
    if v is None or (isinstance(v, float) and v != v): return "NA"
    if v < 0.25: return "bottom (<25%)"
    if v < 0.50: return "low (25-50%)"
    if v < 0.75: return "high (50-75%)"
    return "top (>=75%)"

from pit_fundamentals import (
    bucket_ps, bucket_pb, bucket_peg, bucket_div_yield, bucket_debt_to_equity,
    bucket_current_ratio, bucket_op_margin, bucket_earnings_growth, bucket_fcf_sign,
    bucket_beta, bucket_avg_volume, bucket_fwd_eps, bucket_short_float, bucket_pct,
    bucket_gross_margin, bucket_rd_intensity, bucket_asset_turnover, bucket_accruals,
    bucket_fcf_yield, bucket_cash_ratio, bucket_buyback_yield, bucket_insider, bucket_stake,
    bucket_ad, bucket_darkpool_level, bucket_darkpool_trend, bucket_news_sent,
)

# (name, field, bucket_fn, order, pit)
#   pit=True  -> field is a column in prepare_pit_metrics()'s output (point-in-time)
#   pit=False -> field is a column on the current `Fundamental` snapshot (lookahead-labeled)
DIMENSIONS = [
    # ── Point-in-time (17) ──────────────────────────────────────────────
    ("PE (trailing)", "pe", _bucket_pe,
     ["cheap (<15)", "fair (15-30)", "rich (30-50)", "expensive (>=50)", "negative (no earnings)", "NA"], True),
    ("PS", "ps", bucket_ps, ["cheap (<1)", "fair (1-4)", "rich (4-10)", "expensive (>=10)", "NA"], True),
    ("PB", "pb", bucket_pb, ["negative equity", "cheap (<1)", "fair (1-3)", "rich (3-6)", "expensive (>=6)", "NA"], True),
    ("PEG", "peg", bucket_peg, ["negative", "cheap (<1)", "fair (1-2)", "expensive (>=2)", "NA"], True),
    ("EPS (TTM)", "eps_ttm", _bucket_eps, ["positive EPS", "negative EPS", "NA"], True),
    ("Revenue growth", "revenue_growth", _bucket_growth,
     ["shrinking (<0%)", "slow (0-15%)", "growing (15-40%)", "hypergrowth (>=40%)", "NA"], True),
    ("Earnings growth", "earnings_growth", bucket_earnings_growth,
     ["shrinking (<0%)", "slow (0-15%)", "fast (15-40%)", "hyper (>=40%)", "NA"], True),
    ("Profit margin", "profit_margin", _bucket_margin,
     ["unprofitable (<0%)", "thin (0-10%)", "healthy (>=10%)", "NA"], True),
    ("Operating margin", "operating_margin", bucket_op_margin,
     ["unprofitable (<0%)", "thin (0-15%)", "fat (>=15%)", "NA"], True),
    ("Debt-to-equity", "debt_to_equity", bucket_debt_to_equity,
     ["negative equity", "low (<0.5)", "moderate (0.5-1.5)", "high (>=1.5)", "NA"], True),
    ("Current ratio", "current_ratio", bucket_current_ratio, ["weak (<1)", "ok (1-2)", "strong (>=2)", "NA"], True),
    ("Free cash flow", "fcf_ttm", bucket_fcf_sign, ["positive FCF", "negative FCF", "NA"], True),
    ("Market cap", "market_cap", _bucket_mktcap,
     ["micro (<500M)", "small (0.5-2B)", "mid (2-10B)", "large (>=10B)", "NA"], True),
    ("Dividend yield", "dividend_yield", bucket_div_yield,
     ["none", "low (0-2%)", "mid (2-4%)", "high (>=4%)", "NA"], True),
    ("Beta", "beta", bucket_beta, ["low (<0.8)", "market (0.8-1.2)", "high (1.2-2)", "very high (>=2)", "NA"], True),
    ("Avg volume", "avg_volume", bucket_avg_volume, ["thin (<500K)", "mid (0.5-5M)", "liquid (>=5M)", "NA"], True),
    ("52-wk position", "pct_52w", bucket_pct_52w,
     ["bottom (<25%)", "low (25-50%)", "high (50-75%)", "top (>=75%)", "NA"], True),
    # ── Phase A: earnings-quality / efficiency / yield (point-in-time, 7) ──
    ("Gross margin", "gross_margin", bucket_gross_margin,
     ["low (<20%)", "fair (20-40%)", "high (40-60%)", "very high (>=60%)", "NA"], True),
    ("R&D intensity", "rd_intensity", bucket_rd_intensity,
     ["none", "low (0-5%)", "mid (5-15%)", "heavy (>=15%)", "NA"], True),
    ("Asset turnover", "asset_turnover", bucket_asset_turnover,
     ["low (<0.5)", "mid (0.5-1)", "high (>=1)", "NA"], True),
    ("Accruals (quality)", "accruals", bucket_accruals,
     ["negative (best)", "low (0-10%)", "high (>=10%)", "NA"], True),
    ("FCF yield", "fcf_yield", bucket_fcf_yield,
     ["negative", "low (0-4%)", "mid (4-8%)", "high (>=8%)", "NA"], True),
    ("Cash ratio", "cash_ratio", bucket_cash_ratio,
     ["low (<0.25)", "mid (0.25-1)", "high (>=1)", "NA"], True),
    ("Buyback yield", "buyback_yield", bucket_buyback_yield,
     ["dilution (<-2%)", "neutral (-2..2%)", "buyback (2-5%)", "heavy buyback (>=5%)", "NA"], True),
    # ── Phase B: insider open-market buying (point-in-time) ──────────────
    ("Insider buying", "insider_buy_intensity", bucket_insider,
     ["none", "small (<0.1%)", "notable (0.1-1%)", "large cluster (>=1%)", "NA"], True),
    # ── Phase C: 5%+ stake filings (point-in-time, subject-CIK indexed) ──
    ("Activist 13D (1y)", "stake_13d_1y", bucket_stake, ["none", "one (1y)", "multiple (1y)", "NA"], True),
    ("Institutional 13G (1y)", "stake_13g_1y", bucket_stake, ["none", "one (1y)", "multiple (1y)", "NA"], True),
    # ── A/D line state (point-in-time, slope+divergence — not sign) ──────
    ("A/D state", "ad_state", bucket_ad, ["accum divergence", "accum trend-up", "neutral", "distribution", "NA"], True),
    # ── Alt-data amplifiers (point-in-time; validation only, NOT wired to risk rating) ──
    # Dark pool: FINRA weekly ATS share, keyed by publish date (~3-4wk lag → lookahead-safe).
    ("Dark-pool share", "dp_off_pct", bucket_darkpool_level,
     ["low (<5%)", "mid (5-12%)", "high (12-20%)", "very high (>=20%)", "NA"], True),
    ("Dark-pool trend", "dp_trend_z", bucket_darkpool_trend,
     ["distributing (z<=-1)", "steady (-1..1)", "accumulating (1-2)", "surging (>=2)", "NA"], True),
    # News sentiment dimension is PLUMBED (loader/PIT column/bucket in place) but held OUT of the
    # sweep: the stored EODHD polarity is near-constant (median 0.99, 90%+ ~+1.0 → no discriminating
    # power) and history is shallow/uneven (only 127 tickers reach 2021; most start 2025-26). Re-add
    # after re-deriving net sentiment (pos-neg) + a deep news backfill. See News-drift/PEAD follow-up.
    # ("News sentiment", "news_sent", bucket_news_sent,
    #  ["negative (<=-.15)", "neutral (-.15..15)", "positive (.15-.5)", "very positive (>=.5)", "NA"], True),
    # Float, now POINT-IN-TIME: EDGAR share count as-of the filing date (not the yfinance
    # snapshot). A share-count proxy for free float; "Market cap" carries the economic size.
    ("Float", "float_shares", _bucket_float,
     ["micro-float (<20M)", "low-float (20-100M)", "mid-float (100-500M)", "high-float (>=500M)", "NA"], True),
    # NOTE: the other 5 snapshot dims were removed 2026-08-07 as lookahead. Their honest PIT
    # equivalents already exist above: Insider % -> "Insider buying"; Institution % ->
    # "Institutional 13G (1y)" / "Activist 13D (1y)"; Forward PE/EPS -> "PE (trailing)" /
    # "EPS (TTM)". Only "Short % float" lacks a PIT source (needs FINRA historical short interest).
]


def _compute_trades(candles, sig_fn, exit_fn):
    """Core per-ticker backtest loop, shared by serial and parallel paths.
    Returns (ticker, entry_idx, return_pct) — entry_idx is retained so the caller can
    dedup overlapping fires into independent episodes for the eff-N / t-stat significance layer."""
    trades = []  # (ticker, entry_idx, return_pct)
    for tk, sdf in candles.items():
        if len(sdf) < MIN_BARS:
            continue
        try:
            sig = sig_fn(sdf).fillna(False)
        except Exception:
            continue
        close = sdf["Close"].values
        n = len(close)
        for d in sig[sig].index:
            idx = sdf.index.get_loc(d)
            exit_idx = exit_fn(sdf, idx)
            if exit_idx is None or exit_idx <= idx or exit_idx >= n:
                continue
            ep = float(close[idx])
            if ep <= 0:
                continue
            trades.append((tk, idx, (float(close[exit_idx]) - ep) / ep * 100))
    return trades


def _worker(payload):
    """Run in a spawned subprocess: set up Django, load this chunk's candles from the
    DB, and backtest. Returns a small list of (ticker, return_pct) tuples."""
    signal_key, exit_key, tickers = payload
    import django
    os.environ.setdefault("DJANGO_SETTINGS_MODULE", "rotation.settings")
    try:
        django.setup()
    except Exception:
        pass  # already set up (e.g. fork); safe to ignore
    _, sig_fn = SIGNALS[signal_key]
    _, exit_fn = EXITS[exit_key]
    return _compute_trades(load_candles(tickers), sig_fn, exit_fn)


def _chunk(seq, n):
    """Split seq into n roughly-equal contiguous chunks (drops empties)."""
    n = max(1, n)
    k, r = divmod(len(seq), n)
    out, i = [], 0
    for j in range(n):
        size = k + (1 if j < r else 0)
        if size:
            out.append(seq[i:i + size])
        i += size
    return out


def run(signal_key, exit_key, limit=None, jobs=1):
    sig_name, sig_fn = SIGNALS[signal_key]
    exit_name, exit_fn = EXITS[exit_key]

    tickers = build_universe()
    if limit:
        tickers = tickers[:limit]
    print(f"Universe: {len(tickers)} individual stocks (candles + fundamentals, ex-ETFs)")
    funds = load_fundamentals(tickers)
    print(f"Signal: {sig_name}  |  Exit: {exit_name}  |  jobs: {jobs}\n")

    trades = []  # (ticker, return_pct)
    if jobs <= 1:
        trades = _compute_trades(load_candles(tickers), sig_fn, exit_fn)
    else:
        # More chunks than workers -> better load balancing as fast workers steal work.
        import concurrent.futures as cf
        import multiprocessing as mp
        # Parent Django connection must not be inherited by spawned children.
        try:
            from django.db import connections
            connections.close_all()
        except Exception:
            pass
        chunks = _chunk(tickers, jobs * 3)
        payloads = [(signal_key, exit_key, c) for c in chunks]
        ctx = mp.get_context("spawn")
        done = 0
        with cf.ProcessPoolExecutor(max_workers=jobs, mp_context=ctx) as ex:
            for chunk_trades in ex.map(_worker, payloads):
                trades.extend(chunk_trades)
                done += 1
                print(f"  ...{done}/{len(payloads)} chunks done ({len(trades)} trades)")
    print(f"Loaded fundamentals for {len(funds)}\n")

    # Significance layer: overlapping fires on consecutive bars share a forward window and
    # inflate the apparent trade count. Dedup per ticker into independent episodes (>=EFFECTIVE_GAP
    # bars apart), then report eff_trades + a one-sample t-stat on the deduped returns alongside
    # the raw avg/win. Entries are grouped per ticker and episode-flagged ONCE here so every
    # bucket below inherits the same flags.
    from collections import defaultdict
    idxs_by_tk = defaultdict(list)
    for tk, idx, _r in trades:
        idxs_by_tk[tk].append(idx)
    ep_by_tk = {tk: _episode_starts(sorted(ix)) for tk, ix in idxs_by_tk.items()}
    # Augment each trade with is_episode; buckets carry (ret, is_ep) pairs.
    trades = [(tk, idx, r, idx in ep_by_tk[tk]) for tk, idx, r in trades]

    def _agg(pairs):
        # pairs: list of (return_pct, is_episode)
        if not pairs:
            return {"trades": 0, "avg_return": 0.0, "win_rate": 0.0, "eff_trades": 0, "t_stat": None}
        rets = [r for r, _ in pairs]
        eff = [r for r, is_ep in pairs if is_ep]
        return {"trades": len(rets),
                "avg_return": round(sum(rets) / len(rets), 2),
                "win_rate": round(sum(1 for r in rets if r > 0) / len(rets) * 100, 1),
                "eff_trades": len(eff),
                "t_stat": _tstat_from_returns(eff)}

    all_pairs = [(r, is_ep) for _tk, _idx, r, is_ep in trades]
    overall = _agg(all_pairs)
    print(f"=== {signal_key} on ALL STOCKS ({overall['trades']} trades, "
          f"eff {overall['eff_trades']}, t={overall['t_stat']}) ===")
    print(f"    avg {overall['avg_return']:+.2f}%   win {overall['win_rate']:.0f}%\n")

    # Bucket by each fundamental dimension
    by_dimension = {}
    # NOTE: this standalone single-signal path buckets by the current snapshot only, so
    # point-in-time dims (which read from the PIT metrics frame, not `funds`) fall to NA
    # here. The all-on-all engine (all_on_all_study.py) does true point-in-time bucketing.
    for dim_name, field, bucket_fn, order, _pit in DIMENSIONS:
        buckets = {}
        for tk, _idx, r, is_ep in trades:
            fv = funds.get(tk, {}).get(field)
            buckets.setdefault(bucket_fn(fv), []).append((r, is_ep))
        rows = []
        print(f"--- {dim_name} ---")
        for label in order:
            lst = buckets.get(label, [])
            if len(lst) < MIN_BUCKET_TRADES:
                continue
            a = _agg(lst)
            rows.append({"bucket": label, **a})
            print(f"    {label:22} {a['trades']:>5} tr   {a['avg_return']:+7.2f}%   "
                  f"{a['win_rate']:>4.0f}% wr   eff {a['eff_trades']:>4}  t={a['t_stat']}")
        by_dimension[dim_name] = rows
        print()

    out = {
        "signal_key": signal_key, "signal_name": sig_name,
        "exit_key": exit_key, "exit_name": exit_name,
        "universe_size": len(tickers), "overall": overall,
        "by_dimension": by_dimension,
    }
    path = STUDIES_DIR / f"seq_fundamental_{signal_key}.json"
    path.write_text(json.dumps(out, indent=2))
    print(f"Saved -> {path}")
    return out


if __name__ == "__main__":
    argv = sys.argv
    signal_key = argv[argv.index("--signal") + 1] if "--signal" in argv else "seq_rsi20_ad_rising_rsi"
    exit_key = argv[argv.index("--exit") + 1] if "--exit" in argv else "6m"
    limit = int(argv[argv.index("--limit") + 1]) if "--limit" in argv else None
    jobs = int(argv[argv.index("--jobs") + 1]) if "--jobs" in argv else DEFAULT_JOBS
    if "--db" in argv:
        import django
        os.environ.setdefault("DJANGO_SETTINGS_MODULE", "rotation.settings")
        django.setup()
    run(signal_key, exit_key, limit, jobs=jobs)
