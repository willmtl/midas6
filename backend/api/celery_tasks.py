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
def run_finra_ats():
    """Weekly: pull FINRA OTC-Transparency ATS (dark-pool) weekly volume → DarkPoolWeek. Public API,
    no key needed. only_missing=False so newly-published weeks (FINRA lags ~3-4wk) get picked up;
    idempotent clean-replace per ticker."""
    from api.tasks import import_finra_ats
    r = import_finra_ats(only_missing=False)
    logger.info(f"FINRA ATS weekly complete: {r}")
    return r


@shared_task
def run_news_event_study():
    """Nightly: recompute the market-adjusted news event study (our-model read: dir/impact × beta ×
    IV, market-stripped AR + drift). Run after the local news classifier so local_dir is fresh."""
    from api.news_market_study import run_and_save
    r = run_and_save()
    logger.info(f"news event study: {r}")
    return r


@shared_task
def run_iv_calibration_task():
    """Weekly: recompute the IV-calibration study (implied vs realized next-day move; per-ticker
    over/under-pricing). Slow-moving, so weekly suffices."""
    from api.iv_calibration import run_and_save
    r = run_and_save()
    logger.info(f"iv calibration: {r}")
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
def run_eodhd_analyst_ratings():
    """Weekly: pull the EODHD AnalystRatings distribution (StrongBuy…StrongSell + mean) → Fundamental."""
    import os
    if not os.environ.get("EODHD_API_KEY"):
        return {"skipped": "no key"}
    from api.tasks import import_eodhd_analyst_ratings
    r = import_eodhd_analyst_ratings()
    logger.info(f"EODHD analyst ratings import: {r}")
    return r


@shared_task
def run_backtest_lab():
    """Nightly: regenerate the backtest lab (root backtest_concept.py → .data/studies/backtest_concept.json)
    as a clean SUBPROCESS.

    Why a subprocess and not an in-process import: the backtest scripts parallelize with
    multiprocessing 'spawn', which re-imports the child __main__. Running inside this
    Django/Celery process would make the spawned workers re-import the server's __main__
    (runserver/celery) instead of the script. Launching it as `python backtest_concept.py`
    gives spawn a clean script __main__, exactly like the manual command. Mirrors
    api.tasks.run_stock_studies_task. Scheduled after the fundamentals + stock-studies chain
    so candles and fundamentals are fresh."""
    import os
    import subprocess
    script = "/app/backtest_concept.py"
    if not os.path.exists(script):
        logger.error("run_backtest_lab: %s not found (mount it in docker-compose)", script)
        return
    cmd = ["python", "-u", script]
    logger.info("run_backtest_lab: launching %s", " ".join(cmd))
    proc = subprocess.run(cmd, cwd="/app", capture_output=True, text=True, timeout=1800)
    if proc.returncode != 0:
        logger.error("backtest lab failed (rc=%s): %s", proc.returncode, proc.stderr[-2000:])
    else:
        logger.info("backtest lab done: %s", proc.stdout[-1000:])
    return proc.returncode


@shared_task
def check_fresh_alert():
    """Daily: detect sectors that became FRESH since the last run, alert to Slack."""
    from api.tasks import compute_fresh_and_alert
    logger.info("Checking for newly fresh sectors")
    result = compute_fresh_and_alert("1d")
    logger.info(f"Fresh check complete: {result}")
    return result


# ── EODHD extra data sources (corporate actions, market cap + IDs, delisted, UST rates, congress) ──
# Each runs the standalone /app/fetch_*.py module as a subprocess (matches the run_stock_studies /
# run_backtest_lab pattern; keeps heavy per-ticker loops out of the worker process). Idempotent, so
# nightly re-runs only add new rows. No-op without EODHD_API_KEY.
def _run_fetch(script, *args, timeout=3600):
    import os, subprocess
    if not os.environ.get("EODHD_API_KEY"):
        return {"skipped": "no EODHD_API_KEY"}
    path = f"/app/{script}"
    if not os.path.exists(path):
        logger.error("_run_fetch: %s not found (mount it in docker-compose)", path)
        return {"error": "not mounted"}
    cmd = ["python", "-u", path, *args]
    logger.info("_run_fetch: %s", " ".join(cmd))
    proc = subprocess.run(cmd, cwd="/app", capture_output=True, text=True, timeout=timeout)
    if proc.returncode != 0:
        logger.error("%s failed (rc=%s): %s", script, proc.returncode, proc.stderr[-2000:])
    else:
        logger.info("%s done: %s", script, proc.stdout[-600:])
    return proc.returncode


@shared_task
def run_corp_actions():
    """Daily: splits + dividends → CorporateAction (also feeds the FINRA off_pct split adjustment)."""
    return _run_fetch("fetch_corp_actions.py")


@shared_task
def run_market_cap():
    """Daily: historical market cap → MarketCapHistory, and CUSIP/CIK/ISIN → Fundamental."""
    r1 = _run_fetch("fetch_market_cap.py", "--mcap")
    r2 = _run_fetch("fetch_market_cap.py", "--ids")
    return {"mcap": r1, "ids": r2}


@shared_task
def run_delisted():
    """Weekly: delisted/inactive US tickers → DelistedCompany (survivorship-free reference)."""
    return _run_fetch("fetch_delisted.py", "--run")


@shared_task
def run_ust_rates():
    """Daily: official US Treasury curve → TreasuryRate (feeds the rates.py regime layer)."""
    return _run_fetch("fetch_ust_rates.py", "--run")


@shared_task
def run_congress_trades():
    """Daily: congressional (legislator) trades → CongressTrade (follow-the-politicians alt-data)."""
    return _run_fetch("fetch_congress.py", "--run")


@shared_task
def run_congress_study():
    """Nightly: PIT market-adjusted forward-return study of congressional trades vs SPY →
    BacktestResult[congress_study] (the script persists to DB itself)."""
    import subprocess, os
    if not os.path.exists("/app/congress_study.py"):
        return {"error": "not mounted"}
    proc = subprocess.run(["python", "-u", "/app/congress_study.py"], cwd="/app",
                          capture_output=True, text=True, timeout=3600)
    if proc.returncode != 0:
        logger.error("congress_study failed (rc=%s): %s", proc.returncode, proc.stderr[-2000:])
    return proc.returncode


@shared_task
def run_delisted_survivorship():
    """Weekly: survivorship-bias audit of the universe vs the delisted list →
    BacktestResult[delisted_survivorship]."""
    import subprocess, os
    if not os.path.exists("/app/delisted_survivorship.py"):
        return {"error": "not mounted"}
    proc = subprocess.run(["python", "-u", "/app/delisted_survivorship.py"], cwd="/app",
                          capture_output=True, text=True, timeout=600)
    if proc.returncode != 0:
        logger.error("delisted_survivorship failed (rc=%s): %s", proc.returncode, proc.stderr[-2000:])
    return proc.returncode


@shared_task
def run_darkpool_backtest():
    """Nightly: historical dark-pool equity-curve backtest → BacktestResult[darkpool_backtest]."""
    import subprocess, os
    if not os.path.exists("/app/darkpool_backtest.py"):
        return {"error": "not mounted"}
    proc = subprocess.run(["python", "-u", "/app/darkpool_backtest.py"], cwd="/app",
                          capture_output=True, text=True, timeout=3600)
    if proc.returncode != 0:
        logger.error("darkpool_backtest failed (rc=%s): %s", proc.returncode, proc.stderr[-2000:])
    return proc.returncode


@shared_task
def run_congress_backtest():
    """Nightly: legislator-trade equity-curve backtest → BacktestResult[congress_backtest]."""
    import subprocess, os
    if not os.path.exists("/app/congress_backtest.py"):
        return {"error": "not mounted"}
    proc = subprocess.run(["python", "-u", "/app/congress_backtest.py"], cwd="/app",
                          capture_output=True, text=True, timeout=3600)
    if proc.returncode != 0:
        logger.error("congress_backtest failed (rc=%s): %s", proc.returncode, proc.stderr[-2000:])
    return proc.returncode


@shared_task
def run_vol_shock_study():
    """Nightly: vol-normalized shock continuation study → BacktestResult[vol_shock_study]."""
    import subprocess, os
    if not os.path.exists("/app/vol_shock_study.py"):
        return {"error": "not mounted"}
    proc = subprocess.run(["python", "-u", "/app/vol_shock_study.py"], cwd="/app",
                          capture_output=True, text=True, timeout=3600)
    if proc.returncode != 0:
        logger.error("vol_shock_study failed (rc=%s): %s", proc.returncode, proc.stderr[-2000:])
    return proc.returncode


@shared_task
def run_rotation_picks():
    """Nightly: live rotation-pick basket (cheapest-P/B in each top-momentum sector) → BacktestResult."""
    import subprocess, os
    if not os.path.exists("/app/rotation_pick_scan.py"):
        return {"error": "not mounted"}
    proc = subprocess.run(["python", "-u", "/app/rotation_pick_scan.py"], cwd="/app",
                          capture_output=True, text=True, timeout=1200)
    if proc.returncode != 0:
        logger.error("rotation_pick_scan failed (rc=%s): %s", proc.returncode, proc.stderr[-2000:])
    return proc.returncode


@shared_task
def run_rotation_call():
    """Nightly: the flagship Rotation Call (regime-leader sectors ∩ cheapest-P/B value pick ∩ oversold
    entry; commodity/foreign sleeves pick a producer/foreign name, else the ETF) → BacktestResult."""
    import subprocess, os
    if not os.path.exists("/app/rotation_call_scan.py"):
        return {"error": "not mounted"}
    proc = subprocess.run(["python", "-u", "/app/rotation_call_scan.py"], cwd="/app",
                          capture_output=True, text=True, timeout=1200)
    if proc.returncode != 0:
        logger.error("rotation_call_scan failed (rc=%s): %s", proc.returncode, proc.stderr[-2000:])
    return proc.returncode


@shared_task
def run_profitability_guard():
    """Weekly: profitability-guard study (does excluding cheap-P/B value traps improve the pick?).
    Heavy (loads all candles) + slow-moving → weekly. → BacktestResult[profitability_guard]."""
    import subprocess, os
    if not os.path.exists("/app/profitability_guard_study.py"):
        return {"error": "not mounted"}
    proc = subprocess.run(["python", "-u", "/app/profitability_guard_study.py"], cwd="/app",
                          capture_output=True, text=True, timeout=1800)
    if proc.returncode != 0:
        logger.error("profitability_guard_study failed (rc=%s): %s", proc.returncode, proc.stderr[-2000:])
    return proc.returncode


@shared_task
def run_factor_lab():
    """Weekly: factor-lab sweep (filters/tilts/combos on the value pick, best-return search). Heavy +
    slow-moving → weekly. → BacktestResult[factor_lab]."""
    import subprocess, os
    if not os.path.exists("/app/factor_lab.py"):
        return {"error": "not mounted"}
    proc = subprocess.run(["python", "-u", "/app/factor_lab.py"], cwd="/app",
                          capture_output=True, text=True, timeout=1800)
    if proc.returncode != 0:
        logger.error("factor_lab failed (rc=%s): %s", proc.returncode, proc.stderr[-2000:])
    return proc.returncode


@shared_task
def run_portfolio_blender():
    """Weekly: portfolio blender (mix CORE value + CAPITULATION sleeves). Heavy + slow-moving → weekly.
    → BacktestResult[portfolio_blender]."""
    import subprocess, os
    if not os.path.exists("/app/portfolio_blender.py"):
        return {"error": "not mounted"}
    proc = subprocess.run(["python", "-u", "/app/portfolio_blender.py"], cwd="/app",
                          capture_output=True, text=True, timeout=1800)
    if proc.returncode != 0:
        logger.error("portfolio_blender failed (rc=%s): %s", proc.returncode, proc.stderr[-2000:])
    return proc.returncode


@shared_task
def run_signal_firing():
    """Nightly: per-signal firing scan (all signals × full universe, last 3 bars) → SignalFiring.
    After candles refresh; powers the grouped Studies 'firing now' column."""
    import subprocess, os
    if not os.path.exists("/app/signal_firing_scan.py"):
        return {"error": "not mounted"}
    proc = subprocess.run(["python", "-u", "/app/signal_firing_scan.py", "--db", "--jobs", "2"],
                          cwd="/app", capture_output=True, text=True, timeout=3600)
    if proc.returncode != 0:
        logger.error("signal_firing_scan failed (rc=%s): %s", proc.returncode, proc.stderr[-2000:])
    return proc.returncode


@shared_task
def run_ground_earnings():
    """Nightly: ground the earnings categorization (EPS surprise + forward guidance) on EarningsEvent.
    After the earnings (21:05) + financials (21:30) imports so the inputs are fresh."""
    import subprocess, os
    if not os.path.exists("/app/ground_earnings.py"):
        return {"error": "not mounted"}
    proc = subprocess.run(["python", "-u", "/app/ground_earnings.py"], cwd="/app",
                          capture_output=True, text=True, timeout=3600)
    if proc.returncode != 0:
        logger.error("ground_earnings failed (rc=%s): %s", proc.returncode, proc.stderr[-2000:])
    return proc.returncode


@shared_task
def run_burst_scan():
    """Nightly: short-term burst scan + global confluence → ShortTermSignal + GlobalSignal.
    Runs after the stock sweep (needs StockStudy short-horizon edges) + dark-pool/A-D jobs."""
    import subprocess, os
    if not os.path.exists("/app/burst_scan.py"):
        return {"error": "not mounted"}
    proc = subprocess.run(["python", "-u", "/app/burst_scan.py", "--db", "--jobs", "4"], cwd="/app",
                          capture_output=True, text=True, timeout=3600)
    if proc.returncode != 0:
        logger.error("burst_scan failed (rc=%s): %s", proc.returncode, proc.stderr[-2000:])
    return proc.returncode


@shared_task
def run_backtest_decomp():
    """Nightly: rotation-edge decomposition (pick vs rotation vs both, 200MA both-numbers,
    value×technical) → BacktestResult[decomposition]. Runs AFTER candles/fundamentals refresh."""
    import os, subprocess
    script = "/app/backtest_lowpb.py"
    if not os.path.exists(script):
        logger.error("run_backtest_decomp: %s not found", script)
        return
    proc = subprocess.run(["python", "-u", script], cwd="/app",
                          capture_output=True, text=True, timeout=3600)
    if proc.returncode != 0:
        logger.error("backtest decomp failed (rc=%s): %s", proc.returncode, proc.stderr[-2000:])
    else:
        logger.info("backtest decomp done: %s", proc.stdout[-600:])
    return proc.returncode
