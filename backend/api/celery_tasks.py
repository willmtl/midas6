"""
Celery tasks for periodic data updates.
"""

import logging
from celery import shared_task

logger = logging.getLogger(__name__)


@shared_task
def update_candles():
    """
    Hourly task: fetch missing candle data up to current day.
    Current day is always re-fetched (market may still be open).
    """
    from api.tasks import import_candles_task
    logger.info("Starting hourly candle update")
    import_candles_task()
    logger.info("Hourly candle update complete")


@shared_task
def recompute_scan():
    """Recompute sector scan after candle update."""
    from api.tasks import compute_scan
    logger.info("Recomputing daily scan")
    compute_scan("1d")
    logger.info("Daily scan complete")


@shared_task
def run_new_studies():
    """Run any studies that haven't been computed yet."""
    from api.tasks import run_studies_task
    logger.info("Running new studies")
    run_studies_task()
    logger.info("Studies complete")


@shared_task
def run_stock_studies():
    """Periodic: re-run the all-on-all stock sweep (every signal × exit over all stocks
    + fundamental buckets) so the Individual Stocks tab stays current as candles update."""
    from api.tasks import run_stock_studies_task
    logger.info("Running all-on-all stock studies sweep")
    rc = run_stock_studies_task()
    logger.info(f"Stock studies sweep complete (rc={rc})")
    return rc


@shared_task
def fetch_financial_history():
    """Periodic: refresh point-in-time fundamentals (quarterly financials + dividends)
    before the nightly stock sweep buckets trades against them."""
    from api.tasks import run_financial_history_task
    logger.info("Backfilling financial history")
    rc = run_financial_history_task()
    logger.info(f"Financial history backfill complete (rc={rc})")
    return rc


@shared_task
def fetch_insider_data():
    """Periodic: refresh insider open-market transactions (SEC bulk Form 345) before the
    nightly stock sweep buckets trades by insider buying."""
    from api.tasks import run_insider_task
    logger.info("Backfilling insider transactions")
    rc = run_insider_task()
    logger.info(f"Insider backfill complete (rc={rc})")
    return rc


@shared_task
def fetch_sec_events():
    """Periodic: refresh 13D/13G (activist / passive 5%+ stake) filings."""
    from api.tasks import run_sec_events_task
    logger.info("Backfilling 13D/13G filings")
    rc = run_sec_events_task()
    logger.info(f"13D/13G backfill complete (rc={rc})")
    return rc


@shared_task
def run_live_firing():
    """Periodic: refresh the 'firing now' scan so the Firing Now tab shows today's triggers."""
    from api.tasks import run_live_firing_task
    logger.info("Running firing-now scan")
    rc = run_live_firing_task()
    logger.info(f"Firing-now scan complete (rc={rc})")
    return rc


@shared_task
def run_news_horizon():
    """Daily: refresh the news-horizon scan (recent material news → horizon-conditioned drift /
    validated fades) so the News Horizon tab shows what to fade over which window."""
    from api.tasks import run_news_horizon_scan_task
    logger.info("Running news-horizon scan")
    rc = run_news_horizon_scan_task()
    logger.info(f"News-horizon scan complete (rc={rc})")
    return rc


@shared_task
def run_ad_divergence():
    """Periodic: refresh the A/D-divergence scan so the A/D Divergence tab shows today's
    accum-divergence stocks (and which are primed by a capitulation signal)."""
    from api.tasks import compute_ad_divergence
    logger.info("Running A/D-divergence scan")
    result = compute_ad_divergence()
    logger.info(f"A/D-divergence scan complete: {result}")
    return result


@shared_task
def run_playbook():
    """Periodic: refresh the live Playbook (sector board + today's ranked candidates)."""
    from api.tasks import compute_playbook
    logger.info("Running Playbook refresh")
    result = compute_playbook()
    logger.info(f"Playbook refresh complete: {result.get('n_a')} A / {result.get('n_b')} B candidates")
    return result


@shared_task
def run_paper_trades():
    """Periodic: snapshot today's Playbook picks into the forward paper-trade record and mark
    open positions to market / close on exit. Must run AFTER run_playbook."""
    from api.tasks import update_paper_trades
    logger.info("Updating paper-trade record")
    result = update_paper_trades()
    logger.info(f"Paper-trade update complete: {result}")
    return result


@shared_task
def run_equity_curve():
    """Weekly: recompute the portfolio backtest equity curve vs SPY (heavy)."""
    from api.tasks import compute_equity_curve
    logger.info("Recomputing equity-curve backtest")
    result = compute_equity_curve()
    logger.info(f"Equity-curve recompute complete: {result.get('n_trades')} trades")
    return result


@shared_task
def run_research():
    """Weekly: recompute the Research/Lab comparison matrices (heavy)."""
    from api.tasks import compute_research
    logger.info("Recomputing research comparisons")
    result = compute_research()
    logger.info(f"Research recompute complete: {result.get('n_daily')} entries, {len(result.get('matrix', []))} matrix rows")
    return result


@shared_task
def run_options_snapshot():
    """Nightly: snapshot options summary (30d ATM IV, put/call vol & OI, IV skew, GEX) for liquid US
    names. Uses Polygon (real OI + Greeks) when POLYGON_API_KEY is set, else falls back to the free
    yfinance collector. Builds history forward; stores raw ratios (direction validated later)."""
    import os
    logger.info("Collecting nightly options snapshot")
    if os.environ.get("POLYGON_API_KEY"):
        from api.tasks import collect_options_polygon
        r = collect_options_polygon()
    else:
        from api.tasks import collect_options_snapshot
        r = collect_options_snapshot()
    logger.info(f"Options snapshot complete: {r}")
    return r


@shared_task
def run_darkpool_snapshot():
    """Nightly: reconstruct off-exchange (dark-pool) volume % per liquid US name from Polygon's
    trade tape. No-op without POLYGON_API_KEY (Advanced/Stocks tier). See POLYGON_SETUP.md."""
    import os
    if not os.environ.get("POLYGON_API_KEY"):
        logger.info("darkpool snapshot skipped — POLYGON_API_KEY not set")
        return {"skipped": "no key"}
    from api.tasks import collect_darkpool_polygon
    from seq_fundamental_study import build_universe, load_fundamentals
    tks = build_universe(); funds = load_fundamentals(tks)
    us = [t for t in tks if "." not in t and (funds.get(t, {}).get("market_cap") or 0) >= 2e9]
    logger.info("Collecting dark-pool snapshot for %d names", len(us))
    r = collect_darkpool_polygon(us)
    logger.info(f"Dark-pool snapshot complete: {r}")
    return r


@shared_task
def run_eodhd_news():
    """Daily: pull EODHD news + sentiment for the universe → NewsItem. No-op without EODHD_API_KEY."""
    import os
    if not os.environ.get("EODHD_API_KEY"):
        return {"skipped": "no key"}
    from api.tasks import import_eodhd_news
    r = import_eodhd_news(days=10)   # incremental — dedup handles overlap
    logger.info(f"EODHD news import: {r}")
    return r


@shared_task
def classify_recent_news():
    """RETIRED 2026-08-09 — cut over to the on-box LOCAL model (classify_news_local). This task used
    the Anthropic API and sent headlines OUT of the container, against the local-only constraint. It
    is UNSCHEDULED (beat entry removed) and now hard-disabled: it will not run unless someone sets
    NEWS_ALLOW_ANTHROPIC=1 to deliberately re-enable the external call. The playbook fade flag and the
    news-horizon scanner now read the local labels (local_impact/local_rating)."""
    import os, json
    from pathlib import Path
    if os.environ.get("NEWS_ALLOW_ANTHROPIC") != "1":
        logger.info("classify_recent_news DISABLED (local-only cutover) — set NEWS_ALLOW_ANTHROPIC=1 to override")
        return {"skipped": "disabled — local-only cutover; use classify_news_local"}
    if not os.environ.get("ANTHROPIC_API_KEY"):
        logger.info("classify_recent_news skipped — ANTHROPIC_API_KEY not set")
        return {"skipped": "no key"}
    from core.models import LiveSignal
    tickers = set(LiveSignal.objects.values_list("ticker", flat=True))
    try:
        pb = json.loads((Path("/app/.data/studies/playbook.json")).read_text())
        tickers |= {c["ticker"] for c in pb.get("candidates", []) if c.get("ticker")}
    except Exception:
        pass
    if not tickers:
        return {"skipped": "no firing tickers"}
    from news_classifier import classify_news
    r = classify_news(limit=4000, min_age_days=0, max_age_days=16, tickers=tickers)
    logger.info(f"Recent-news classification ({len(tickers)} firing tickers): {r}")
    return r


@shared_task
def classify_news_local(limit=6000, workers=6):
    """Daily (LOCAL LLM — nothing leaves Docker): classify newly-imported news with the on-box
    qwen model — event category (cat_llm) AND the off-ticker guard — so the off-ticker filter and
    categories stay current as the nightly EODHD import adds ~2k rows/day. Incremental + resumable:
    only rows not yet judged (off_ticker IS NULL) are processed, moved-first (day_effect rows win),
    so the high-value subset is always covered even if `limit` is hit. Bounded by `limit` to finish
    inside one nightly window. No-op if the on-box Ollama is unreachable (returns {'error': ...}).

    This is the LOCAL counterpart to classify_recent_news (which uses the Anthropic API and sends
    headlines OUT of the container). Prefer this task when the local-only constraint is in force."""
    import news_llm_category as nlc
    logger.info("Local-LLM news classification (incremental, off_ticker IS NULL, moved-first)")
    r = nlc.main(limit=limit, workers=workers)
    logger.info(f"Local-LLM news classification complete: {r}")
    return r


@shared_task
def run_eodhd_earnings():
    """Daily: pull EODHD earnings dates + EPS surprises for the universe → EarningsEvent."""
    import os
    if not os.environ.get("EODHD_API_KEY"):
        return {"skipped": "no key"}
    from api.tasks import import_eodhd_earnings
    r = import_eodhd_earnings()
    logger.info(f"EODHD earnings import: {r}")
    return r


@shared_task
def run_eodhd_estimates():
    """Weekly: pull EODHD analyst estimate revisions (current vs 7d/30d ago) → EstimateRevision."""
    import os
    if not os.environ.get("EODHD_API_KEY"):
        return {"skipped": "no key"}
    from api.tasks import import_eodhd_estimates
    r = import_eodhd_estimates()
    logger.info(f"EODHD estimates import: {r}")
    return r


@shared_task
def check_fresh_alert():
    """Daily: detect sectors that became FRESH since the last run, alert to Slack."""
    from api.tasks import compute_fresh_and_alert
    logger.info("Checking for newly fresh sectors")
    result = compute_fresh_and_alert("1d")
    logger.info(f"Fresh check complete: {result}")
    return result
