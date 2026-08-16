from django.urls import path
from api import views

urlpatterns = [
    # Sector scan
    path("scan", views.ScanView.as_view(), name="scan"),
    path("sectors", views.SectorListView.as_view(), name="sectors"),

    # Drill-down
    path("drilldown/<str:sector_name>", views.DrilldownView.as_view(), name="drilldown"),

    # Chart
    path("chart/<str:ticker>", views.ChartView.as_view(), name="chart"),

    # Studies
    path("studies", views.StudyListView.as_view(), name="studies"),
    path("studies/<int:study_id>/trades", views.StudyTradesView.as_view(), name="study-trades"),

    # Stock studies (all-on-all sweep over individual stocks + fundamental buckets)
    path("stock-studies", views.StockStudiesView.as_view(), name="stock-studies"),

    # Sector -> specific-stock drill-down (a sector signal becomes per-stock signals)
    path("sector-drilldown", views.SectorStockDrilldownView.as_view(), name="sector-drilldown"),

    # Firing now: stocks currently triggering the top signals across all sectors
    path("live-signals", views.LiveSignalsView.as_view(), name="live-signals"),
    path("news-event-study", views.NewsEventStudyView.as_view(), name="news-event-study"),
    path("iv-calibration", views.IvCalibrationView.as_view(), name="iv-calibration"),

    # News horizon: recent material news joined to horizon-conditioned drift (validated fades)
    path("news-horizon", views.NewsHorizonSignalsView.as_view(), name="news-horizon"),
    path("news-effect", views.NewsEffectView.as_view(), name="news-effect"),
    path("news-effect/chart", views.NewsEffectChartView.as_view(), name="news-effect-chart"),
    # News clusters: bursts of headlines on one ticker (promotion / "propping" footprint) + fade check
    path("news-clusters", views.NewsClusterView.as_view(), name="news-clusters"),
    # Smart-money detail: individual 13D/13G filings + insider buy/sell behind the badges (popup)
    path("smart-money", views.SmartMoneyView.as_view(), name="smart-money"),

    # A/D divergence: stocks whose Accumulation/Distribution line is in accum-divergence now
    path("ad-divergence", views.AdDivergenceView.as_view(), name="ad-divergence"),
    # Per-ticker price + A/D line series for the divergence chart popup
    path("ad-divergence/chart", views.AdDivergenceChartView.as_view(), name="ad-divergence-chart"),
    # Two-mode sector-gated strategy: average forward path (where the trade is after N days)
    path("strategy-forward", views.StrategyForwardView.as_view(), name="strategy-forward"),
    # Live end-to-end playbook: sector board + today's ranked candidates through the funnel
    path("playbook", views.PlaybookView.as_view(), name="playbook"),
    # Portfolio backtest equity curve vs SPY
    path("equity-curve", views.EquityCurveView.as_view(), name="equity-curve"),
    # Sector-rotation lab: many rotation rules backtested vs SPY + OOS split + top-signals portfolio
    path("backtest-lab", views.BacktestLabView.as_view(), name="backtest-lab"),
    # Rotation-edge decomposition: pick-only vs rotation-only vs rotation+pick, 200MA both-numbers,
    # and value×technical (cheapest-P/B pick)
    path("backtest-decomp", views.BacktestDecompView.as_view(), name="backtest-decomp"),
    # Alt-data validation studies (analysis only, not wired to risk-rating)
    path("congress-study", views.CongressStudyView.as_view(), name="congress-study"),
    path("delisted-survivorship", views.DelistedSurvivorshipView.as_view(), name="delisted-survivorship"),
    # Dark pool: daily Polygon off-% + official weekly FINRA ATS off-% (overlay) + amplifier result
    path("dark-pool", views.DarkPoolView.as_view(), name="dark-pool"),
    # Dark-pool + alt-data equity-curve backtests (DB-first via BacktestResult)
    path("darkpool-backtest", views.DarkPoolBacktestView.as_view(), name="darkpool-backtest"),
    path("congress-backtest", views.CongressBacktestView.as_view(), name="congress-backtest"),
    # Vol-normalized shock continuation study (continuation matrix + slices + exit-ladder backtest)
    path("vol-shock-study", views.VolShockStudyView.as_view(), name="vol-shock-study"),
    # News overreaction detector + reversion backtest (size-bucketed); intraday RSI crossover study
    path("news-overreaction", views.NewsOverreactionView.as_view(), name="news-overreaction"),
    path("rsi-intraday", views.RsiIntradayView.as_view(), name="rsi-intraday"),
    # H4 short-horizon studies engine (5 families × exit ladder, magnitude buckets, 0-3 day holds)
    path("h4-study", views.H4StudyView.as_view(), name="h4-study"),
    # Per-signal live firing (names firing each study signal in the last N bars)
    path("signal-firing", views.SignalFiringView.as_view(), name="signal-firing"),
    # Live rotation-pick scanner (cheapest-P/B in each strengthening sector)
    path("rotation-picks", views.RotationPicksView.as_view(), name="rotation-picks"),
    # Time machine: month-by-month PIT reconstruction of the flagship basket + realized returns
    path("rotation-history", views.RotationHistoryView.as_view(), name="rotation-history"),
    # RS-trend method sweep (~20 selection rules on the ETF/SPY bar, one value pick)
    path("rs-methods", views.RsMethodsView.as_view(), name="rs-methods"),
    # MA crossover run on every synthetic RS candle (mean-reversion diagnostic)
    path("synthetic-ma-cross", views.SyntheticMaCrossView.as_view(), name="synthetic-ma-cross"),
    # Short-term absolute single-stock oversold-reversal entry + live firing list
    path("oversold-bounce", views.OversoldBounceView.as_view(), name="oversold-bounce"),
    # Diversifiers: rank sleeves by correlation to SPY (commodities/Gold)
    path("diversifier", views.DiversifierView.as_view(), name="diversifier"),
    # Macro regime -> sector leadership (rates / inflation / market)
    path("regime", views.RegimeView.as_view(), name="regime"),
    # Right ENTRY signal for the value-pick basket (dip adds, strength subtracts)
    path("entry-signal", views.EntrySignalView.as_view(), name="entry-signal"),
    # THE headline rotation call: regime-leaders ∩ value-pick ∩ oversold entry
    path("rotation-call", views.RotationCallView.as_view(), name="rotation-call"),
    # Profitability guard: does excluding cheap-P/B value traps (unprofitable+eroding book) help?
    path("profitability-guard", views.ProfitabilityGuardView.as_view(), name="profitability-guard"),
    # Factor lab: sweep filters/tilts/combos on the value pick, ranked to find the best return
    path("factor-lab", views.FactorLabView.as_view(), name="factor-lab"),
    # Portfolio blender: mix CORE value + CAPITULATION sleeves (correlation, crisis-alpha, allocation)
    path("portfolio-blender", views.PortfolioBlenderView.as_view(), name="portfolio-blender"),
    # Strategy lab: can A/B beat C without the rotation; do C's rules travel?
    path("strategy-lab", views.StrategyLabView.as_view(), name="strategy-lab"),
    # Value ranking lab: which value metric (P/B, EV/EBIT, FCF-yield, ...) picks the best name
    path("value-ranking", views.ValueRankingView.as_view(), name="value-ranking"),
    path("return-lab", views.ReturnLabView.as_view(), name="return-lab"),
    path("deep-pool", views.DeepPoolView.as_view(), name="deep-pool"),
    path("bear-defense", views.BearDefenseView.as_view(), name="bear-defense"),
    path("v2-strategy", views.V2StrategyView.as_view(), name="v2-strategy"),
    path("walk-forward", views.WalkForwardView.as_view(), name="walk-forward"),
    path("sector-acceleration", views.SectorAccelerationView.as_view(), name="sector-acceleration"),
    # Short-term burst scanner + Global confluence scanner (both from burst_scan.py)
    path("short-term", views.ShortTermView.as_view(), name="short-term"),
    path("global", views.GlobalView.as_view(), name="global"),
    # Forward paper-trade track record of Playbook picks
    path("paper-trades", views.PaperTradesView.as_view(), name="paper-trades"),
    # Research/Lab: trigger×exit / timeframe / regime / cap-band / MPT comparisons
    path("research", views.ResearchView.as_view(), name="research"),

    # Multi-dimension intersection study (do amplifiers stack?)
    path("dim-intersection", views.DimIntersectionView.as_view(), name="dim-intersection"),

    # Market regime
    path("regime", views.RegimeView.as_view(), name="regime"),
    path("regime/history", views.RegimeHistoryView.as_view(), name="regime-history"),

    # Fundamentals
    path("fundamentals", views.FundamentalsListView.as_view(), name="fundamentals-list"),
    path("fundamentals/<str:ticker>", views.FundamentalsView.as_view(), name="fundamentals"),

    # Trend studies
    path("trend-studies", views.TrendStudyListView.as_view(), name="trend-studies"),
    path("trend-studies/<int:study_id>", views.TrendStudyDetailView.as_view(), name="trend-study-detail"),

    # Stock drilldown
    path("stock-drilldown", views.StockDrilldownListView.as_view(), name="stock-drilldown"),

    # Data management
    path("refresh", views.RefreshView.as_view(), name="refresh"),
    path("import/candles", views.ImportCandlesView.as_view(), name="import-candles"),
    path("import/studies", views.RunStudiesView.as_view(), name="run-studies"),
]
