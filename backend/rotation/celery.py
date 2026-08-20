import os
from celery import Celery
from celery.schedules import crontab

os.environ.setdefault("DJANGO_SETTINGS_MODULE", "rotation.settings")

app = Celery("rotation")
app.config_from_object("django.conf:settings", namespace="CELERY")
# The @shared_task definitions live in api/celery_tasks.py, NOT api/tasks.py. Celery's autodiscover
# defaults to a module named "tasks", so pointing it at "api" alone imports api/tasks.py (which has
# NO tasks) and registers nothing — the worker then rejects every scheduled job as "unregistered"
# and the entire beat pipeline silently no-ops. related_name pins it to the real module; it is
# imported lazily on worker finalize (after django.setup()), so there is no circular-import risk.
app.autodiscover_tasks(["api"], related_name="celery_tasks")

# Periodic tasks
app.conf.beat_schedule = {
    "update-candles-hourly": {
        "task": "api.celery_tasks.update_candles",
        "schedule": crontab(minute=0),  # Every hour
    },
    "recompute-scan-hourly": {
        "task": "api.celery_tasks.recompute_scan",
        "schedule": crontab(minute=5),  # 5 min after candle update
    },
    # Daily "suddenly fresh" alert. Runs once/day after candles refresh.
    # crontab is UTC unless CELERY_TIMEZONE is set; 21:15 UTC ≈ after US close.
    "fresh-alert-daily": {
        "task": "api.celery_tasks.check_fresh_alert",
        "schedule": crontab(hour=21, minute=15),
    },
    # Point-in-time fundamentals refresh. Runs before the stock sweep (21:45) so the
    # sweep buckets trades against fresh quarterly financials. 21:30 UTC.
    "financial-history-daily": {
        "task": "api.celery_tasks.fetch_financial_history",
        "schedule": crontab(hour=21, minute=30),
    },
    # Insider open-market transactions refresh (SEC bulk Form 345). 21:35 UTC, after
    # financials (21:30) and before the sweep (21:45).
    "insider-daily": {
        "task": "api.celery_tasks.fetch_insider_data",
        "schedule": crontab(hour=21, minute=35),
    },
    # 13D/13G (activist / passive 5%+ stake) filings. 21:40 UTC, before the sweep.
    "sec-events-daily": {
        "task": "api.celery_tasks.fetch_sec_events",
        "schedule": crontab(hour=21, minute=40),
    },
    # EODHD news + sentiment for the universe → NewsItem. Incremental (last 10d, deduped).
    # No-op without EODHD_API_KEY. 21:00 UTC, before the fundamentals chain.
    "eodhd-news-daily": {
        "task": "api.celery_tasks.run_eodhd_news",
        "schedule": crontab(hour=21, minute=0),
    },
    # EODHD earnings dates + EPS surprises → EarningsEvent (PEAD + earnings-proximity awareness).
    # 21:05 UTC.
    "eodhd-earnings-daily": {
        "task": "api.celery_tasks.run_eodhd_earnings",
        "schedule": crontab(hour=21, minute=5),
    },
    # EODHD analyst estimate revisions (current vs 7d/30d) → EstimateRevision. The Earnings::Trend
    # endpoint returns ONLY the current snapshot — no history, NOT backfillable — so a PIT revision-
    # momentum series can only accrue FORWARD. Weekly (~4 asof/mo) would take years to build a usable
    # backtest; run WEEKDAY-DAILY (~21 asof/mo) to build it at monthly-rebalance resolution in months.
    # ~1174 API calls/run; no-op without EODHD_API_KEY. 21:10 UTC Mon–Fri. [[always-want-history]]
    "eodhd-estimates-daily": {
        "task": "api.celery_tasks.run_eodhd_estimates",
        "schedule": crontab(hour=21, minute=10, day_of_week="1-5"),
    },
    # DISABLED 2026-08-09 (cutover to local LLM): classify_recent_news uses the Anthropic API and
    # sends headlines OUT of Docker, against the local-only constraint. The playbook fade flag +
    # news-horizon scanner now read the LOCAL labels (local_impact/local_rating) produced on the whole
    # corpus by classify_news_local (23:15). The Anthropic task is left DEFINED (api/celery_tasks.py)
    # for manual/emergency use but is no longer scheduled — nothing leaves the container.
    # "classify-recent-news-daily": {
    #     "task": "api.celery_tasks.classify_recent_news",
    #     "schedule": crontab(hour=22, minute=40),
    # },
    # Sector/ETF studies engine (every signal × exit over the 93 ETFs). Incremental —
    # computes only is_computed=False, so this is a no-op most nights and auto-picks up any
    # newly-added signal without a manual POST. Before the fundamentals chain. 21:20 UTC.
    "sector-studies-daily": {
        "task": "api.celery_tasks.run_new_studies",
        "schedule": crontab(hour=21, minute=20),
    },
    # All-on-all stock sweep. Heavy but parallelized (minutes), so run once daily after
    # candles refresh. Keeps the Individual Stocks tab current. 21:45 UTC ≈ after US close.
    "stock-studies-daily": {
        "task": "api.celery_tasks.run_stock_studies",
        "schedule": crontab(hour=21, minute=45),
    },
    # LOCAL-LLM news classification (on-box qwen, nothing leaves Docker): category + off-ticker
    # guard for news rows added by the nightly EODHD import. Incremental (off_ticker IS NULL,
    # moved-first, bounded). Runs late, after the heavy compute chain, so it doesn't contend with
    # the stock sweep / scans for the worker pool. 23:15 UTC.
    "news-classify-local-daily": {
        "task": "api.celery_tasks.classify_news_local",
        "schedule": crontab(hour=23, minute=15),
    },
    # Firing-now scan. Runs after the stock sweep (needs its StockStudy results to pick
    # the top signals). 22:30 UTC.
    "live-firing-daily": {
        "task": "api.celery_tasks.run_live_firing",
        "schedule": crontab(hour=22, minute=30),
    },
    # A/D-divergence scan. Runs after the stock sweep (needs StockStudy for the historical
    # edge join) and after live-firing. 22:40 UTC.
    "ad-divergence-daily": {
        "task": "api.celery_tasks.run_ad_divergence",
        "schedule": crontab(hour=22, minute=40),
    },
    # News-horizon scan (recent material news → horizon-conditioned validated fades). After the
    # recent-news classification (22:40), before the playbook. 22:45 UTC.
    "news-horizon-daily": {
        "task": "api.celery_tasks.run_news_horizon",
        "schedule": crontab(hour=22, minute=45),
    },
    # Options snapshot — IV / put-call / skew / GEX per liquid US name (Polygon if key set, else
    # yfinance). Once/day, 22:15 UTC.
    "options-snapshot-daily": {
        "task": "api.celery_tasks.run_options_snapshot",
        "schedule": crontab(hour=22, minute=15),
    },
    # Dark-pool (off-exchange %) snapshot from Polygon trade tape. No-op without POLYGON_API_KEY.
    # Heavy (trade-level), so once/day after close. 22:20 UTC.
    "darkpool-snapshot-daily": {
        "task": "api.celery_tasks.run_darkpool_snapshot",
        "schedule": crontab(hour=22, minute=20),
    },
    # FINRA ATS (official weekly dark-pool volume). Public API, no key. FINRA publishes weekly with a
    # 2-4wk lag, so a weekly Sunday pull suffices. Full history in minutes; idempotent. 22:15 UTC Sun.
    "finra-ats-weekly": {
        "task": "api.celery_tasks.run_finra_ats",
        "schedule": crontab(hour=22, minute=15, day_of_week=0),
    },
    # Market-adjusted news event study (our-model read). After news-classify-local (23:15) so
    # local_dir/local_impact are fresh. 23:40 UTC.
    "news-event-study-daily": {
        "task": "api.celery_tasks.run_news_event_study",
        "schedule": crontab(hour=23, minute=40),
    },
    # IV calibration (implied vs realized move; per-ticker over/under-pricing). Slow-moving → weekly.
    "iv-calibration-weekly": {
        "task": "api.celery_tasks.run_iv_calibration_task",
        "schedule": crontab(hour=23, minute=50, day_of_week=0),
    },
    # Playbook refresh (sector board + ranked candidates). After A/D-divergence. 22:50 UTC.
    "playbook-daily": {
        "task": "api.celery_tasks.run_playbook",
        "schedule": crontab(hour=22, minute=50),
    },
    # Forward paper-trade record: snapshot today's Playbook picks + mark to market. MUST run
    # after the playbook refresh (reads playbook.json). 23:00 UTC. This accumulates the real
    # out-of-sample track record over time.
    "paper-trades-daily": {
        "task": "api.celery_tasks.run_paper_trades",
        "schedule": crontab(hour=23, minute=0),
    },
    # Portfolio backtest equity curve. Heavy (~minutes) and barely changes day-to-day, so
    # weekly — Sundays 23:30 UTC.
    "equity-curve-weekly": {
        "task": "api.celery_tasks.run_equity_curve",
        "schedule": crontab(hour=23, minute=30, day_of_week=0),
    },
    # Research/Lab comparison matrices. Heavy; weekly — Sundays 23:50 UTC (after equity curve).
    "research-weekly": {
        "task": "api.celery_tasks.run_research",
        "schedule": crontab(hour=23, minute=50, day_of_week=0),
    },
    # Backtest lab (root backtest_concept.py → .data/studies/backtest_concept.json). Runs after the
    # fundamentals (21:30) + stock-studies (21:45) chain so candles + fundamentals are fresh. 22:45 UTC.
    "backtest-lab-nightly": {
        "task": "api.celery_tasks.run_backtest_lab",
        "schedule": crontab(hour=22, minute=45),
    },
    # rotation-edge decomposition (pick vs rotation vs both, 200MA both-numbers, value×technical)
    "backtest-decomp-nightly": {
        "task": "api.celery_tasks.run_backtest_decomp",
        "schedule": crontab(hour=23, minute=5),
    },
    # EODHD extra data sources — split before/after the study window so they're fresh for studies.
    "eodhd-corp-actions": {          # splits + dividends (feeds FINRA split adjustment)
        "task": "api.celery_tasks.run_corp_actions",
        "schedule": crontab(hour=20, minute=40),
    },
    "eodhd-ust-rates": {             # official Treasury curve → rates regime
        "task": "api.celery_tasks.run_ust_rates",
        "schedule": crontab(hour=20, minute=45),
    },
    "eodhd-congress-trades": {       # legislator trades
        "task": "api.celery_tasks.run_congress_trades",
        "schedule": crontab(hour=20, minute=50),
    },
    "eodhd-market-cap": {            # historical market cap + CUSIP/CIK/ISIN
        "task": "api.celery_tasks.run_market_cap",
        "schedule": crontab(hour=20, minute=55),
    },
    "polygon-etf-flows": {           # ETF fund flows (Polygon shares outstanding → creation/redemption)
        "task": "api.celery_tasks.run_etf_flows",
        "schedule": crontab(hour=21, minute=0),
    },
    "fred-macro-daily": {            # FRED macro/liquidity (M2, net liquidity, HY spread, curve)
        "task": "api.celery_tasks.run_fred",
        "schedule": crontab(hour=21, minute=8),
    },
    "eodhd-delisted-weekly": {       # survivorship-free reference (weekly)
        "task": "api.celery_tasks.run_delisted",
        "schedule": crontab(hour=21, minute=5, day_of_week=0),
    },
    "eodhd-analyst-ratings-weekly": {   # analyst buy/hold/sell distribution → Fundamental
        "task": "api.celery_tasks.run_eodhd_analyst_ratings",
        "schedule": crontab(hour=21, minute=12, day_of_week=0),
    },
    "congress-study-nightly": {         # legislator-trade forward-return study (after data refresh)
        "task": "api.celery_tasks.run_congress_study",
        "schedule": crontab(hour=23, minute=20),
    },
    "delisted-survivorship-weekly": {   # survivorship-bias audit
        "task": "api.celery_tasks.run_delisted_survivorship",
        "schedule": crontab(hour=21, minute=20, day_of_week=0),
    },
    "darkpool-backtest-nightly": {      # historical dark-pool equity-curve backtest
        "task": "api.celery_tasks.run_darkpool_backtest",
        "schedule": crontab(hour=23, minute=30),
    },
    "congress-backtest-nightly": {      # legislator-trade equity-curve backtest
        "task": "api.celery_tasks.run_congress_backtest",
        "schedule": crontab(hour=23, minute=35),
    },
    "vol-shock-study-nightly": {        # vol-normalized shock continuation + slices + exit-ladder
        "task": "api.celery_tasks.run_vol_shock_study",
        "schedule": crontab(hour=23, minute=45),
    },
    "burst-scan-nightly": {             # short-term burst + global confluence (after stock sweep/AD)
        "task": "api.celery_tasks.run_burst_scan",
        "schedule": crontab(hour=22, minute=55),
    },
    "ground-earnings-nightly": {        # grounded earnings verdict (EPS+guidance), after financials
        "task": "api.celery_tasks.run_ground_earnings",
        "schedule": crontab(hour=21, minute=55),
    },
    "signal-firing-nightly": {          # per-signal firing scan (all signals × universe) for Studies
        "task": "api.celery_tasks.run_signal_firing",
        "schedule": crontab(hour=22, minute=10),
    },
    "sector-acceleration-nightly": {    # sector-acceleration leaderboard (the validated sector signal) — 22:03
        "task": "api.celery_tasks.run_sector_acceleration",
        "schedule": crontab(hour=22, minute=3),
    },
    # refresh the live blend's data BEFORE the rotation-pick scan (else implied-upside decays to stale/empty
    # and the blend reverts to cheapest-P/B). Fast (~1-2min each); Polygon egress.
    "refresh-analyst-ratings-nightly": {
        "task": "api.celery_tasks.refresh_analyst_ratings",
        "schedule": crontab(hour=21, minute=50),
    },
    "refresh-short-interest-nightly": {
        "task": "api.celery_tasks.refresh_short_interest",
        "schedule": crontab(hour=21, minute=55),
    },
    "rotation-picks-nightly": {         # live rotation-pick basket (blend: cheap-P/B × analyst-upside; bear->FCF)
        "task": "api.celery_tasks.run_rotation_picks",
        "schedule": crontab(hour=22, minute=5),
    },
    "rotation-call-nightly": {          # flagship Rotation Call (regime-leaders ∩ value ∩ oversold entry)
        "task": "api.celery_tasks.run_rotation_call",
        "schedule": crontab(hour=22, minute=7),
    },
    "rotation-picks-regime-nightly": {  # opt-in blend+200MA-regime COPY of the live basket (kept for comparison)
        "task": "api.celery_tasks.run_rotation_picks_regime",
        "schedule": crontab(hour=22, minute=9),
    },
    "live-conviction-nightly": {        # score live basket 0-5, tag perfect plays — after picks — 22:12 UTC
        "task": "api.celery_tasks.run_live_conviction",
        "schedule": crontab(hour=22, minute=12),
    },
    "h4-c-live-nightly": {              # live H4 dip-buy on the C basket (oversold dip × analyst upside) + alert — after picks
        "task": "api.celery_tasks.run_h4_c_live",
        "schedule": crontab(hour=22, minute=25),
    },
    "profitability-guard-weekly": {     # value-trap guard study (slow-moving) — Sundays 23:55 UTC
        "task": "api.celery_tasks.run_profitability_guard",
        "schedule": crontab(hour=23, minute=55, day_of_week=0),
    },
    "factor-lab-weekly": {              # factor sweep (best-return search) — Sundays 23:58 UTC
        "task": "api.celery_tasks.run_factor_lab",
        "schedule": crontab(hour=23, minute=58, day_of_week=0),
    },
    "portfolio-blender-weekly": {       # sleeve-mix study (core value + capitulation) — Sat 23:58 UTC
        "task": "api.celery_tasks.run_portfolio_blender",
        "schedule": crontab(hour=23, minute=58, day_of_week=6),
    },
    "strategy-lab-weekly": {            # rotation-vs-rules decomposition — Sat 23:52 UTC
        "task": "api.celery_tasks.run_strategy_lab",
        "schedule": crontab(hour=23, minute=52, day_of_week=6),
    },
    "value-ranking-weekly": {           # which value metric picks the best name — Sat 23:46 UTC
        "task": "api.celery_tasks.run_value_ranking",
        "schedule": crontab(hour=23, minute=46, day_of_week=6),
    },
    "return-lab-weekly": {              # 4 return levers (concentration/leverage/orthogonal/regime) — Sat 23:40 UTC
        "task": "api.celery_tasks.run_return_lab",
        "schedule": crontab(hour=23, minute=40, day_of_week=6),
    },
    "deep-pool-weekly": {               # more-stocks-per-ETF (top-20 vs full pool) — Sat 23:34 UTC
        "task": "api.celery_tasks.run_deep_pool",
        "schedule": crontab(hour=23, minute=34, day_of_week=6),
    },
    "bear-defense-weekly": {            # dual-momentum bear overlay — Sat 23:30 UTC
        "task": "api.celery_tasks.run_bear_defense",
        "schedule": crontab(hour=23, minute=30, day_of_week=6),
    },
    "v2-strategy-weekly": {             # stacked v2 decomposition — Sat 23:26 UTC
        "task": "api.celery_tasks.run_v2_strategy",
        "schedule": crontab(hour=23, minute=26, day_of_week=6),
    },
    "walk-forward-weekly": {            # subperiod validation — Sat 23:22 UTC
        "task": "api.celery_tasks.run_walk_forward",
        "schedule": crontab(hour=23, minute=22, day_of_week=6),
    },

    # Live ETF universe growth: refresh the EODHD-expanded constituents so the live scanner's pool tracks
    # current ETF membership (backtests stay on the frozen top-20). Weekly Sun 20:05, before the recomputes.
    "expanded-universe-weekly": {
        "task": "api.celery_tasks.refresh_expanded_universe",
        "schedule": crontab(hour=20, minute=5, day_of_week=0),
    },
    # Backfill candles + PIT financials for any NEW constituent from the refresh above, so newly-added ETF
    # members become tradeable by the live scanner (else build_universe() never sees them). Sun 20:10, right
    # after the membership refresh (20:05) and before the recomputes. Idempotent / fetch-only-missing.
    "ensure-universe-data-weekly": {
        "task": "api.celery_tasks.ensure_universe_data",
        "schedule": crontab(hour=20, minute=10, day_of_week=0),
    },

    # ── FINVIZ VERSION pipeline (separate industry-rotation engine) + flagship tearsheet, self-maintaining ──
    # Weekly data refresh (Sun, before the nightly recomputes use it). Universe → fundamentals → candles,
    # in order; each only fetches what's missing so weeks after the first are fast.
    "finviz-universe-weekly": {         # re-scrape 11 sectors → 149 industries → all names
        "task": "api.celery_tasks.refresh_finviz_universe",
        "schedule": crontab(hour=20, minute=15, day_of_week=0),
    },
    "finviz-fundamentals-weekly": {     # EDGAR fundamentals for newly-missing Finviz names
        "task": "api.celery_tasks.refresh_finviz_fundamentals",
        "schedule": crontab(hour=20, minute=25, day_of_week=0),
    },
    "finviz-candles-weekly": {          # EODHD daily candles for newly-missing Finviz names (breadth unlock)
        "task": "api.celery_tasks.refresh_finviz_candles",
        "schedule": crontab(hour=20, minute=40, day_of_week=0),
    },
    "survivorship-flagship-weekly": {   # ETF flagship arms → BacktestResult[survivorship_smallcap] (dashboard)
        "task": "api.celery_tasks.recompute_survivorship_flagship",
        "schedule": crontab(hour=21, minute=10, day_of_week=0),
    },
    # Nightly: recompute the Finviz engine, then rebuild the flagship tearsheet (both tabs). Ordered so the
    # Finviz tab reads a fresh BacktestResult[finviz_rotation].
    "finviz-rotation-nightly": {
        "task": "api.celery_tasks.recompute_finviz_rotation",
        "schedule": crontab(hour=22, minute=40),
    },
    "flagship-doc-nightly": {           # FLAGSHIP_TRACE → enrich → HTML (after finviz recompute)
        "task": "api.celery_tasks.rebuild_flagship_doc",
        "schedule": crontab(hour=22, minute=55),
    },
}
