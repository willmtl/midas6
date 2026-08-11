from django.db import models


class Sector(models.Model):
    """A sector/theme with its ETF ticker."""
    name = models.CharField(max_length=100, unique=True)
    etf = models.CharField(max_length=10, unique=True)
    category = models.CharField(max_length=50, blank=True, default="")

    class Meta:
        ordering = ["name"]

    def __str__(self):
        return f"{self.name} ({self.etf})"


class Holding(models.Model):
    """Top holdings within a sector ETF."""
    sector = models.ForeignKey(Sector, on_delete=models.CASCADE, related_name="holdings")
    ticker = models.CharField(max_length=20)
    weight = models.FloatField(null=True, blank=True)
    rank = models.IntegerField(default=0)

    class Meta:
        ordering = ["rank"]
        unique_together = ["sector", "ticker"]

    def __str__(self):
        return f"{self.ticker} in {self.sector.etf}"


class Candle(models.Model):
    """OHLCV candle data. TimescaleDB hypertable partitioned by date."""
    ticker = models.CharField(max_length=20)
    date = models.DateField()
    open = models.FloatField()
    high = models.FloatField()
    low = models.FloatField()
    close = models.FloatField()
    volume = models.BigIntegerField()
    interval = models.CharField(max_length=5, default="1d")

    class Meta:
        managed = False  # We create the table via raw SQL
        ordering = ["-date"]

    def __str__(self):
        return f"{self.ticker} {self.date} C={self.close}"


class Fundamental(models.Model):
    """Fundamental data per ticker."""
    ticker = models.CharField(max_length=20, db_index=True)
    date = models.DateField()
    # Valuation
    dividend_yield = models.FloatField(null=True, blank=True)
    pe_ratio = models.FloatField(null=True, blank=True)
    forward_pe = models.FloatField(null=True, blank=True)
    pb_ratio = models.FloatField(null=True, blank=True)
    ps_ratio = models.FloatField(null=True, blank=True)
    peg_ratio = models.FloatField(null=True, blank=True)
    market_cap = models.BigIntegerField(null=True, blank=True)
    enterprise_value = models.BigIntegerField(null=True, blank=True)
    # Earnings & Revenue
    eps = models.FloatField(null=True, blank=True)
    forward_eps = models.FloatField(null=True, blank=True)
    annual_revenue = models.BigIntegerField(null=True, blank=True)
    revenue_growth = models.FloatField(null=True, blank=True)
    earnings_growth = models.FloatField(null=True, blank=True)
    profit_margin = models.FloatField(null=True, blank=True)
    operating_margin = models.FloatField(null=True, blank=True)
    # Shares & Float
    shares_outstanding = models.BigIntegerField(null=True, blank=True)
    float_shares = models.BigIntegerField(null=True, blank=True)
    short_ratio = models.FloatField(null=True, blank=True)
    short_pct_float = models.FloatField(null=True, blank=True)
    insider_pct = models.FloatField(null=True, blank=True)
    institution_pct = models.FloatField(null=True, blank=True)
    # Analyst
    analyst_rating = models.CharField(max_length=20, null=True, blank=True)
    analyst_target = models.FloatField(null=True, blank=True)
    analyst_count = models.IntegerField(null=True, blank=True)
    # Balance sheet
    total_cash = models.BigIntegerField(null=True, blank=True)
    total_debt = models.BigIntegerField(null=True, blank=True)
    debt_to_equity = models.FloatField(null=True, blank=True)
    current_ratio = models.FloatField(null=True, blank=True)
    book_value = models.FloatField(null=True, blank=True)
    # Cash flow
    free_cash_flow = models.BigIntegerField(null=True, blank=True)
    operating_cash_flow = models.BigIntegerField(null=True, blank=True)
    # Other
    beta_5y = models.FloatField(null=True, blank=True)
    fifty_two_wk_high = models.FloatField(null=True, blank=True)
    fifty_two_wk_low = models.FloatField(null=True, blank=True)
    avg_volume = models.BigIntegerField(null=True, blank=True)
    sector = models.CharField(max_length=100, null=True, blank=True)
    industry = models.CharField(max_length=200, null=True, blank=True)
    # EODHD TICKER.EXCHANGE symbol used to fetch this row (e.g. 7203.TSE, WIPRO.NSE, AAPL.US).
    # Distinct from `ticker` (plain/yahoo form) — recorded so global fundamentals fills are traceable.
    eodhd_symbol = models.CharField(max_length=32, null=True, blank=True)
    # ID mapping (EODHD Symbol-Change / fundamentals General block) — cross-vendor identifiers so a
    # name can be joined to CUSIP/CIK/ISIN feeds (SEC EDGAR uses CIK, dark-pool/13F use CUSIP).
    cusip = models.CharField(max_length=12, null=True, blank=True, db_index=True)
    cik = models.CharField(max_length=12, null=True, blank=True, db_index=True)
    isin = models.CharField(max_length=16, null=True, blank=True, db_index=True)

    class Meta:
        ordering = ["-date"]
        unique_together = ["ticker", "date"]

    def __str__(self):
        return f"{self.ticker} {self.date} eps={self.eps}"


class QuarterlyEarnings(models.Model):
    """Quarterly earnings data for computing historical P/E."""
    ticker = models.CharField(max_length=20, db_index=True)
    date = models.DateField()  # quarter end date
    revenue = models.BigIntegerField(null=True, blank=True)
    earnings = models.BigIntegerField(null=True, blank=True)
    eps = models.FloatField(null=True, blank=True)
    eps_estimate = models.FloatField(null=True, blank=True)
    surprise_pct = models.FloatField(null=True, blank=True)

    class Meta:
        ordering = ["-date"]
        unique_together = ["ticker", "date"]

    def __str__(self):
        return f"{self.ticker} Q{self.date} eps={self.eps}"


class FinancialReport(models.Model):
    """One quarterly financial report per ticker, for point-in-time fundamentals.
    `avail_date` = period_end + REPORT_LAG_DAYS: the date the numbers became public
    (approximated). All point-in-time lookups key on avail_date, never period_end."""
    ticker = models.CharField(max_length=20, db_index=True)
    period_end = models.DateField()               # fiscal quarter end
    avail_date = models.DateField(db_index=True)  # period_end + 45d; the PIT key
    revenue = models.BigIntegerField(null=True, blank=True)
    net_income = models.BigIntegerField(null=True, blank=True)
    eps_diluted = models.FloatField(null=True, blank=True)
    operating_income = models.BigIntegerField(null=True, blank=True)
    total_equity = models.BigIntegerField(null=True, blank=True)
    total_debt = models.BigIntegerField(null=True, blank=True)
    current_assets = models.BigIntegerField(null=True, blank=True)
    current_liabilities = models.BigIntegerField(null=True, blank=True)
    free_cash_flow = models.BigIntegerField(null=True, blank=True)
    shares_outstanding = models.BigIntegerField(null=True, blank=True)
    # Phase A: richer statement lines for earnings-quality / efficiency / yield dims.
    operating_cash_flow = models.BigIntegerField(null=True, blank=True)
    total_assets = models.BigIntegerField(null=True, blank=True)
    gross_profit = models.BigIntegerField(null=True, blank=True)
    cost_of_revenue = models.BigIntegerField(null=True, blank=True)
    rd_expense = models.BigIntegerField(null=True, blank=True)
    inventory = models.BigIntegerField(null=True, blank=True)
    cash_and_equivalents = models.BigIntegerField(null=True, blank=True)

    class Meta:
        ordering = ["ticker", "period_end"]
        unique_together = ["ticker", "period_end"]
        indexes = [models.Index(fields=["ticker", "avail_date"])]


class DividendHistory(models.Model):
    """Per-ticker dividend events, for trailing-twelve-month dividend yield."""
    ticker = models.CharField(max_length=20, db_index=True)
    ex_date = models.DateField()
    amount = models.FloatField()

    class Meta:
        ordering = ["ticker", "ex_date"]
        unique_together = ["ticker", "ex_date"]


class SecFiling(models.Model):
    """Ownership/catalyst filings indexed by the SUBJECT company (SEC submissions API).
    form_group: '13D' (activist 5%+ stake) or '13G' (passive institutional 5%+ stake).
    `filed_date` = public disclosure (point-in-time). Accession is globally unique."""
    ticker = models.CharField(max_length=20, db_index=True)
    form_group = models.CharField(max_length=8, db_index=True)  # '13D' | '13G'
    filed_date = models.DateField(db_index=True)
    accession = models.CharField(max_length=32, unique=True)

    class Meta:
        ordering = ["ticker", "filed_date"]


class InsiderBuy(models.Model):
    """Per-ticker per-filing-day aggregate of insider OPEN-MARKET transactions (SEC Form 4,
    bulk Form 345 datasets). `filed_date` = when it became public (point-in-time). Buys =
    code P (acquired), sells = code S (disposed) — the signal-rich open-market trades."""
    ticker = models.CharField(max_length=20, db_index=True)
    filed_date = models.DateField(db_index=True)
    buy_value = models.BigIntegerField(default=0)    # $ of open-market purchases
    sell_value = models.BigIntegerField(default=0)   # $ of open-market sales
    buy_count = models.IntegerField(default=0)       # # of purchase transactions

    class Meta:
        ordering = ["ticker", "filed_date"]
        unique_together = ["ticker", "filed_date"]


class TrendStudy(models.Model):
    """Sector momentum rotation backtest results.
    hold_mode: what the rotation actually holds in each winning sector —
    'etf' (the sector ETF, original), 'momentum' (top-trailing-return stock),
    'hibeta' (highest-beta stock). The last two 'mix' rotation with stock-picking."""
    lookback_months = models.IntegerField()
    hold_months = models.IntegerField()
    top_n = models.IntegerField()
    hold_mode = models.CharField(max_length=12, default="etf", db_index=True)

    total_return = models.FloatField(default=0)
    annual_return = models.FloatField(default=0)
    spy_total = models.FloatField(default=0)
    alpha = models.FloatField(default=0)
    max_drawdown = models.FloatField(default=0)
    num_trades = models.IntegerField(default=0)   # = number of non-overlapping rebalance periods
    win_rate = models.FloatField(default=0)
    # One-sample t of the per-period returns vs 0 (periods are non-overlapping → no overlap
    # correction). |t|>=2 with num_trades>=12 = "robust"; below that a big total_return is usually a
    # lucky single-name run (esp. top_n=1). Null when <3 periods or zero dispersion.
    t_stat = models.FloatField(null=True, blank=True)

    equity_curve = models.JSONField(null=True, blank=True)
    spy_curve = models.JSONField(null=True, blank=True)
    trade_log = models.JSONField(null=True, blank=True)
    computed_at = models.DateTimeField(null=True, blank=True)

    class Meta:
        ordering = ["-total_return"]
        unique_together = ["lookback_months", "hold_months", "top_n", "hold_mode"]

    def __str__(self):
        return f"Look={self.lookback_months}m Hold={self.hold_months}m Top={self.top_n} Ret={self.total_return:+.1f}%"


class StockDrilldown(models.Model):
    """Stock-level drilldown: buy highest beta stock when sector signal fires."""
    study = models.OneToOneField('Study', on_delete=models.CASCADE, related_name='drilldown')
    # Stock-level results
    stock_trades = models.IntegerField(default=0)
    stock_avg_return = models.FloatField(default=0)
    stock_win_rate = models.FloatField(default=0)
    stock_avg_hold = models.FloatField(default=0)
    stock_max_drawdown = models.FloatField(default=0)
    # Comparison to ETF
    etf_avg_return = models.FloatField(default=0)
    alpha_vs_etf = models.FloatField(default=0)
    # Top stocks
    best_stocks = models.JSONField(null=True, blank=True)
    worst_stocks = models.JSONField(null=True, blank=True)
    computed_at = models.DateTimeField(null=True, blank=True)

    class Meta:
        ordering = ["-stock_avg_return"]

    def __str__(self):
        return f"{self.study.name} stock_avg={self.stock_avg_return:+.2f}%"


class Study(models.Model):
    """A trading study definition + aggregated results."""
    name = models.CharField(max_length=200)
    signal_key = models.CharField(max_length=50)
    signal_name = models.CharField(max_length=100)
    exit_key = models.CharField(max_length=50)
    exit_name = models.CharField(max_length=100)
    category = models.CharField(max_length=50, blank=True)

    # Aggregated results
    total_trades = models.IntegerField(default=0)
    avg_return = models.FloatField(default=0)
    win_rate = models.FloatField(default=0)
    avg_hold = models.FloatField(default=0)
    sector_count = models.IntegerField(default=0)

    # Entry-quality (max adverse excursion). avg_mae = avg worst intraday drawdown from entry
    # over the hold (%, <= 0). clean_pct = % of trades whose MAE never breached CLEAN_MAE_THRESH
    # (≈-2%) — i.e. near-"perfect" entries that barely dipped before working. Together they say
    # "do we usually nail the entry, or usually sit underwater first".
    avg_mae = models.FloatField(default=0)
    clean_pct = models.FloatField(default=0)

    # Significance / independence. eff_trades = trade count after collapsing fires within
    # EFFECTIVE_GAP (=5) bars into one independent observation (level-condition signals fire on
    # consecutive bars → heavily overlapping forward windows, so total_trades overstates the
    # independent sample). t_stat = one-sample t of those deduped per-trade returns vs 0; |t|>=2
    # ≈ a real edge, small |t| = noise despite a big total_trades. Coarse (an overlap proxy, not
    # Newey-West) but flags signals whose apparent robustness is just near-duplicate bars.
    eff_trades = models.IntegerField(null=True, blank=True)
    t_stat = models.FloatField(null=True, blank=True)

    # Peak / curve data (averages)
    peak_day = models.IntegerField(null=True, blank=True)
    peak_avg = models.FloatField(null=True, blank=True)
    ret_90d = models.FloatField(null=True, blank=True)

    # Best case (p90 percentile)
    best_peak_day = models.IntegerField(null=True, blank=True)
    best_peak_ret = models.FloatField(null=True, blank=True)
    best_ret_90d = models.FloatField(null=True, blank=True)

    # Regime performance (JSON: {"LOW": {"trades":N,"avg_return":X,"win_rate":Y}, ...})
    by_regime = models.JSONField(null=True, blank=True)
    by_curve = models.JSONField(null=True, blank=True)
    by_vix = models.JSONField(null=True, blank=True)
    by_spy_trend = models.JSONField(null=True, blank=True)
    by_season = models.JSONField(null=True, blank=True)

    # Best/worst sectors (JSON: [{"sector":"X","avg_return":Y,"trades":N,"win_rate":Z}])
    best_sectors = models.JSONField(null=True, blank=True)
    worst_sectors = models.JSONField(null=True, blank=True)

    computed_at = models.DateTimeField(null=True, blank=True)
    is_computed = models.BooleanField(default=False)

    class Meta:
        ordering = ["-avg_return"]
        unique_together = ["signal_key", "exit_key"]

    def __str__(self):
        return self.name


class StudySectorResult(models.Model):
    """Per-sector results for a study."""
    study = models.ForeignKey(Study, on_delete=models.CASCADE, related_name="sector_results")
    sector = models.ForeignKey(Sector, on_delete=models.CASCADE)
    trades = models.IntegerField(default=0)
    avg_return = models.FloatField(default=0)
    total_return = models.FloatField(default=0)
    win_rate = models.FloatField(default=0)
    wins = models.IntegerField(default=0)
    losses = models.IntegerField(default=0)
    avg_hold = models.FloatField(default=0)
    max_gain = models.FloatField(default=0)
    max_loss = models.FloatField(default=0)

    class Meta:
        unique_together = ["study", "sector"]

    def __str__(self):
        return f"{self.study.name} - {self.sector.name}"


class Trade(models.Model):
    """Individual trade from a study."""
    study = models.ForeignKey(Study, on_delete=models.CASCADE, related_name="trades")
    sector = models.ForeignKey(Sector, on_delete=models.CASCADE)
    etf = models.CharField(max_length=10)
    entry_date = models.DateField()
    exit_date = models.DateField()
    entry_price = models.FloatField()
    exit_price = models.FloatField()
    return_pct = models.FloatField()
    hold_days = models.IntegerField()

    class Meta:
        ordering = ["-entry_date"]
        indexes = [
            models.Index(fields=["study", "entry_date"]),
            models.Index(fields=["study", "sector"]),
        ]

    def __str__(self):
        return f"{self.etf} {self.entry_date} {self.return_pct:+.2f}%"


class ScanResult(models.Model):
    """Cached sector scan results."""
    sector = models.ForeignKey(Sector, on_delete=models.CASCADE)
    interval = models.CharField(max_length=5, default="1d")
    rsi = models.FloatField(null=True)
    rsi_sma = models.FloatField(null=True)
    rsi_spread = models.FloatField(null=True)
    rsi_above_sma = models.BooleanField(default=False)
    rsi_crossover = models.BooleanField(default=False)
    crossover_days_ago = models.IntegerField(null=True)
    sortino = models.FloatField(null=True)
    spy_sortino = models.FloatField(null=True)
    sortino_trend = models.CharField(max_length=5, default="flat")
    omega = models.FloatField(null=True)
    spy_omega = models.FloatField(null=True)
    omega_trend = models.CharField(max_length=5, default="flat")
    cvar = models.FloatField(null=True)
    spy_cvar = models.FloatField(null=True)
    ulcer = models.FloatField(null=True)
    spy_ulcer = models.FloatField(null=True)
    ulcer_trend = models.CharField(max_length=5, default="flat")
    up_capture = models.FloatField(null=True)
    down_capture = models.FloatField(null=True)
    down_capture_trend = models.CharField(max_length=5, default="flat")
    beta = models.FloatField(null=True)
    corr_spy = models.FloatField(null=True)
    corr_qqq = models.FloatField(null=True)
    beta_qqq = models.FloatField(null=True)
    gap = models.BooleanField(default=False)
    gap_dir = models.CharField(max_length=5, null=True)
    gap_days_ago = models.IntegerField(null=True)
    gap_pct = models.FloatField(null=True)
    signal = models.CharField(max_length=20, default="BEARISH")
    bullish = models.BooleanField(default=False)
    computed_at = models.DateTimeField(auto_now=True)

    class Meta:
        unique_together = ["sector", "interval"]

    def __str__(self):
        return f"{self.sector.name} {self.interval} {self.signal}"


class StockStudy(models.Model):
    """Aggregated results of one signal × exit run across the individual-stock universe,
    with fundamental-bucket breakdowns. The stock-side analogue of Study (which is
    sector/ETF). Written by the all-on-all sweep."""
    signal_key = models.CharField(max_length=60)
    signal_name = models.CharField(max_length=120)
    exit_key = models.CharField(max_length=60)
    exit_name = models.CharField(max_length=120)
    category = models.CharField(max_length=50, blank=True)

    total_trades = models.IntegerField(default=0)
    avg_return = models.FloatField(default=0)
    win_rate = models.FloatField(default=0)
    avg_hold = models.FloatField(default=0)
    universe_size = models.IntegerField(default=0)

    # Entry-quality (max adverse excursion) — see Study.avg_mae / Study.clean_pct.
    avg_mae = models.FloatField(default=0)
    clean_pct = models.FloatField(default=0)

    # Significance / independence — see Study.eff_trades / Study.t_stat. eff_trades collapses
    # fires within EFFECTIVE_GAP (=5) bars per ticker into one independent observation; t_stat is
    # the one-sample t of those deduped returns vs 0 (|t|>=2 ≈ real, small |t| = noise). This is
    # what separates the genuinely-robust winners from combos inflated by overlapping trades.
    eff_trades = models.IntegerField(null=True, blank=True)
    t_stat = models.FloatField(null=True, blank=True)

    # Fundamental-bucket breakdown:
    # {"PE (trailing)": [{"bucket":"cheap (<15)","trades":N,"avg_return":X,"win_rate":Y}, ...], ...}
    by_dimension = models.JSONField(null=True, blank=True)

    computed_at = models.DateTimeField(null=True, blank=True)

    class Meta:
        ordering = ["-avg_return"]
        unique_together = ["signal_key", "exit_key"]
        indexes = [
            models.Index(fields=["category"]),
            models.Index(fields=["-avg_return"]),
        ]

    def __str__(self):
        return f"{self.signal_name} -> {self.exit_name} ({self.avg_return:+.1f}%)"


class LiveSignal(models.Model):
    """A stock currently FIRING one of the top signals (within the last N bars), joined
    with the signal's historical edge, the stock's fundamentals, and its sector(s).
    Refreshed by the firing-now scan. Answers 'what do I look at today'."""
    ticker = models.CharField(max_length=20)
    signal_key = models.CharField(max_length=60)
    signal_name = models.CharField(max_length=120)
    days_ago = models.IntegerField(default=0)          # bars since the fire (0 = latest bar)
    last_close = models.FloatField(default=0)

    # Historical edge of this signal (its best exit) from the all-on-all sweep.
    best_exit_key = models.CharField(max_length=60, blank=True)
    hist_avg_return = models.FloatField(null=True, blank=True)
    hist_win_rate = models.FloatField(null=True, blank=True)
    hist_trades = models.IntegerField(null=True, blank=True)
    # Entry quality of this signal's best exit (from the sweep): avg max adverse excursion (%)
    # and % of "clean" (barely-dipped) entries. Tells you whether firing tends to give a clean
    # entry or one you'll sit underwater on first.
    hist_avg_mae = models.FloatField(null=True, blank=True)
    hist_clean_pct = models.FloatField(null=True, blank=True)

    # Fundamental snapshot + which "amplifier" buckets it lands in.
    market_cap = models.FloatField(null=True, blank=True)
    pe_ratio = models.FloatField(null=True, blank=True)
    forward_pe = models.FloatField(null=True, blank=True)
    profit_margin = models.FloatField(null=True, blank=True)
    fund_buckets = models.JSONField(null=True, blank=True)  # {"Market cap":"micro (<500M)", ...}
    sectors = models.JSONField(null=True, blank=True)        # ["Quantum Computing", ...]
    # Smart-money confirmation (SEC EDGAR): insider open-market buying + recent 5% stakes.
    insider_buy_90d = models.BigIntegerField(null=True, blank=True)  # $ trailing 90d
    recent_13d = models.IntegerField(default=0)                      # 13D filings, last 180d
    recent_13g = models.IntegerField(default=0)                      # 13G filings, last 180d

    computed_at = models.DateTimeField(null=True, blank=True)

    class Meta:
        ordering = ["days_ago", "-hist_avg_return"]
        unique_together = ["ticker", "signal_key"]
        indexes = [models.Index(fields=["signal_key"]), models.Index(fields=["days_ago"])]

    def __str__(self):
        return f"{self.ticker} firing {self.signal_key} ({self.days_ago}d ago)"


class NewsHorizonSignal(models.Model):
    """A recent, material, LLM-classified news event joined with the horizon-conditioned drift we
    measured for its TYPE (news_drift_horizon.py / news_horizon_robust.py). Answers: 'this stock
    just had <news type>; over the type's own horizon our data says it tends to <fade|ride>, and
    it's <N> days into a <horizon> window.' Refreshed by the news-horizon scan. The only ROBUST
    (time+size validated) stances are the FADES: earnings-beat & product & strong-bullish-pop,
    concentrated in mid/small caps. Everything else is WATCH (informational, not validated)."""
    ticker = models.CharField(max_length=20)
    news_date = models.DateField()
    cat = models.CharField(max_length=24)              # signed type: earnings_beat, product, ma, ...
    direction = models.SmallIntegerField(default=0)    # -1 bearish / 0 / +1 bullish
    impact = models.SmallIntegerField(default=0)       # 2 moderate / 3 major
    horizon = models.CharField(max_length=8)           # day / week / month / 3mo (the type's own)
    pop_pct = models.FloatField(null=True, blank=True) # day-1 β-adj abnormal move (context)

    market_cap = models.FloatField(null=True, blank=True)
    cap_bucket = models.CharField(max_length=16, blank=True)   # small / mid / large
    exp_drift = models.FloatField(null=True, blank=True)       # historical oriented drift at horizon (%)
    stance = models.CharField(max_length=8, blank=True)        # FADE / RIDE / WATCH
    robust = models.BooleanField(default=False)                # did the type's effect pass robustness?
    days_since = models.IntegerField(default=0)                # bars since the news
    days_left = models.IntegerField(default=0)                 # bars remaining in the horizon window
    last_close = models.FloatField(default=0)
    title = models.CharField(max_length=240, blank=True)
    sectors = models.JSONField(null=True, blank=True)
    computed_at = models.DateTimeField(null=True, blank=True)

    class Meta:
        ordering = ["-robust", "days_left", "-impact"]
        unique_together = ["ticker", "news_date", "cat"]
        indexes = [models.Index(fields=["stance"]), models.Index(fields=["robust"]),
                   models.Index(fields=["cat"])]

    def __str__(self):
        return f"{self.ticker} {self.cat} {self.stance} ({self.days_left}d left in {self.horizon})"


class AdDivergenceSignal(models.Model):
    """A stock whose Accumulation/Distribution LINE is in 'accum divergence' state RIGHT NOW
    (price flat/down while the ADL rises — read as slope+divergence, never sign). The study
    found this slice ~triples the edge on price-capitulation signals, so each row is flagged
    `primed` when a capitulation signal (new_52low / rsi_oversold20) is ALSO firing within the
    last N bars, and joined with that signal's historical edge. Rows without a capitulation
    trigger are the weaker 'watch' version. Refreshed by compute_ad_divergence()."""
    ticker = models.CharField(max_length=20, unique=True)
    last_close = models.FloatField(default=0)
    primed = models.BooleanField(default=False)          # a capitulation signal is also firing
    firing = models.JSONField(null=True, blank=True)      # [{signal_key, signal_name, days_ago}]
    min_days_ago = models.IntegerField(null=True, blank=True)  # freshest capitulation fire (sort key)
    fires_60d = models.IntegerField(default=0)            # capitulation fires in last 60 bars (serial = knife)
    pct_above_low = models.FloatField(null=True, blank=True)  # % current close is above trailing-60 low
    knife = models.BooleanField(default=False)            # falling knife: fires_60d>=4 AND still near low
    low_quality = models.BooleanField(default=False)      # landmine: micro-cap / penny / unprofitable

    # Historical edge of the best-firing capitulation signal (its strongest exit, from StockStudy).
    best_signal_key = models.CharField(max_length=60, blank=True)
    best_signal_name = models.CharField(max_length=120, blank=True)
    best_exit_key = models.CharField(max_length=60, blank=True)
    hist_avg_return = models.FloatField(null=True, blank=True)
    hist_win_rate = models.FloatField(null=True, blank=True)
    hist_trades = models.IntegerField(null=True, blank=True)

    # Fundamental snapshot + which amplifier buckets it lands in.
    market_cap = models.FloatField(null=True, blank=True)
    pe_ratio = models.FloatField(null=True, blank=True)
    forward_pe = models.FloatField(null=True, blank=True)
    profit_margin = models.FloatField(null=True, blank=True)
    fund_buckets = models.JSONField(null=True, blank=True)
    sectors = models.JSONField(null=True, blank=True)
    # Smart-money confirmation (SEC EDGAR): insider open-market buying + recent 5% stakes.
    insider_buy_90d = models.BigIntegerField(null=True, blank=True)
    recent_13d = models.IntegerField(default=0)
    recent_13g = models.IntegerField(default=0)

    computed_at = models.DateTimeField(null=True, blank=True)

    class Meta:
        ordering = ["-primed", "min_days_ago", "-hist_avg_return"]
        indexes = [models.Index(fields=["primed"]), models.Index(fields=["min_days_ago"])]

    def __str__(self):
        return f"{self.ticker} accum-divergence{' PRIMED' if self.primed else ''}"


class PaperTrade(models.Model):
    """Forward paper-trading record of Playbook picks — the real out-of-sample evidence.
    Each Playbook candidate is 'bought' (on paper) the day it first appears, then marked to
    market and closed by the sort_gt1 exit. Builds a live track record over time."""
    ticker = models.CharField(max_length=20)
    mode = models.CharField(max_length=4, blank=True)       # A / B
    sector = models.CharField(max_length=100, blank=True)
    entry_date = models.DateField()
    entry_price = models.FloatField()
    peak_price = models.FloatField(default=0)
    last_price = models.FloatField(default=0)
    status = models.CharField(max_length=8, default="open")  # open / closed
    exit_date = models.DateField(null=True, blank=True)
    exit_price = models.FloatField(null=True, blank=True)
    ret_pct = models.FloatField(null=True, blank=True)       # realized (closed) or unrealized (open)
    hist_avg_return = models.FloatField(null=True, blank=True)
    opened_at = models.DateTimeField(null=True, blank=True)
    updated_at = models.DateTimeField(null=True, blank=True)

    class Meta:
        ordering = ["status", "-opened_at"]
        unique_together = ["ticker", "entry_date"]

    def __str__(self):
        return f"{self.ticker} {self.mode} {self.status} {self.ret_pct}"


class OptionSnapshot(models.Model):
    """Daily options summary per liquid US stock — collected forward from now to BUILD our own
    IV / put-call history (yfinance gives only a current snapshot; no history to backtest). Once
    we have ~6-12 months, we can validate a single-stock IV / put-call signal the way we validated
    everything else. atm_iv = 30d ATM implied vol (%); pc_vol/pc_oi = put/call volume & OI ratios."""
    ticker = models.CharField(max_length=20, db_index=True)
    date = models.DateField(db_index=True)
    spot = models.FloatField(null=True, blank=True)
    atm_iv = models.FloatField(null=True, blank=True)     # 30-day ATM implied volatility, %
    pc_vol = models.FloatField(null=True, blank=True)     # put/call VOLUME ratio (near expiries)
    pc_oi = models.FloatField(null=True, blank=True)      # put/call OPEN-INTEREST ratio
    n_exp = models.IntegerField(default=0)                # # expiries (liquidity proxy)
    iv_skew = models.FloatField(null=True, blank=True)    # put_IV − call_IV near ATM (fear/skew), vol pts
    gex = models.FloatField(null=True, blank=True)        # dealer gamma exposure estimate ($, calls + / puts −)
    source = models.CharField(max_length=16, default="yfinance")  # yfinance / polygon
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        unique_together = ["ticker", "date"]
        indexes = [models.Index(fields=["ticker", "date"])]

    def __str__(self):
        return f"{self.ticker} {self.date} IV={self.atm_iv} pc_vol={self.pc_vol}"


class DarkPoolDay(models.Model):
    """Daily off-exchange (dark-pool + internalizer) volume per stock, reconstructed from Polygon's
    trade tape (TRF-reported prints: exchange==4 with a trf_id). `off_pct` = off-exchange share of
    total volume; `block_off_vol` = off-exchange volume in large prints (≥ block_min shares), a
    proxy for INSTITUTIONAL dark-pool activity (retail internalization is small-lot). NOTE: the tape
    is blended (dark pool + retail) with no per-ATS attribution and no signed side; the block filter
    is the institutional proxy. FINRA ATS is the clean-but-lagged alternative."""
    ticker = models.CharField(max_length=20, db_index=True)
    date = models.DateField(db_index=True)
    total_vol = models.BigIntegerField(default=0)
    off_vol = models.BigIntegerField(default=0)            # off-exchange (TRF) volume
    off_pct = models.FloatField(null=True, blank=True)     # off_vol / total_vol
    block_off_vol = models.BigIntegerField(default=0)      # off-exchange volume in ≥block_min prints
    block_min = models.IntegerField(default=5000)
    source = models.CharField(max_length=16, default="polygon")
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        unique_together = ["ticker", "date"]
        indexes = [models.Index(fields=["ticker", "date"])]

    def __str__(self):
        return f"{self.ticker} {self.date} off%={self.off_pct}"


class DarkPoolWeek(models.Model):
    """Weekly ATS (dark-pool) volume per stock from FINRA OTC Transparency — the OFFICIAL source that
    mandates every ATS venue to report. Unlike DarkPoolDay (a daily BLENDED off-exchange proxy the raw
    tape can't split), this is the canonical per-ticker dark-pool number, cleanly separated from
    wholesaler/internalizer flow. Carries `published_date` — the date FINRA made it public (~3-4wk
    lag) — so studies can gate on known-as-of for lookahead-safe point-in-time analysis. `off_pct` =
    ats_shares / that week's consolidated volume (summed from Candle, Monday-anchored week)."""
    ticker = models.CharField(max_length=20, db_index=True)
    week_start = models.DateField(db_index=True)             # Monday of the reporting week
    ats_shares = models.BigIntegerField(default=0)           # FINRA totalWeeklyShareQuantity
    ats_trades = models.BigIntegerField(default=0)           # FINRA totalWeeklyTradeCount
    ats_notional = models.FloatField(null=True, blank=True)  # FINRA totalNotionalSum ($)
    off_pct = models.FloatField(null=True, blank=True)       # ats_shares / weekly consolidated volume
    tier = models.CharField(max_length=16, blank=True)       # "NMS Tier 1" / "NMS Tier 2" / "OTC"
    published_date = models.DateField(null=True, blank=True)  # FINRA initialPublishedDate (PIT)
    source = models.CharField(max_length=16, default="finra_ats")
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        unique_together = ["ticker", "week_start", "source"]
        indexes = [models.Index(fields=["ticker", "week_start"])]

    def __str__(self):
        return f"{self.ticker} wk{self.week_start} ats={self.ats_shares}"


class NewsEventStudy(models.Model):
    """Latest market-adjusted news event-study result (api.news_market_study.run_news_market_study),
    stored as one JSON blob for the dashboard. AR = R_stock − beta·R_spy, keyed on OUR model's read
    (local_dir/local_impact), with IV/surprise + next-morning gap + short/long beta. Single row
    (label='latest'), refreshed nightly after the local news classifier runs."""
    label = models.CharField(max_length=32, unique=True, default="latest")
    n_events = models.IntegerField(default=0)
    n_tickers = models.IntegerField(default=0)
    data = models.JSONField(default=dict)     # {by_model_dir_beta:[...], by_iv_dir:[...], ...}
    computed_at = models.DateTimeField(auto_now=True)

    def __str__(self):
        return f"NewsEventStudy({self.label}, {self.n_events} events)"


class IvCalibration(models.Model):
    """Latest IV-calibration study (api.iv_calibration): is ATM implied vol a good predictor of the
    next-day move? Aggregate (variance-risk-premium stats) + per-ticker ranking of over/under-priced
    options. Single JSON-blob row (label='latest'), refreshed weekly."""
    label = models.CharField(max_length=32, unique=True, default="latest")
    n_days = models.IntegerField(default=0)
    n_tickers = models.IntegerField(default=0)
    data = models.JSONField(default=dict)     # {aggregate:{...}, per_ticker:[...]}
    computed_at = models.DateTimeField(auto_now=True)

    def __str__(self):
        return f"IvCalibration({self.label}, {self.n_days} days)"


class NewsItem(models.Model):
    """News headlines + sentiment from EODHD, per ticker. Feeds the news under/over-reaction (drift)
    study: a story's sentiment vs the stock's actual price reaction → residual → forward drift."""
    ticker = models.CharField(max_length=20, db_index=True)
    dt = models.DateTimeField(db_index=True)
    title = models.CharField(max_length=512, blank=True)
    sentiment = models.FloatField(null=True, blank=True)    # polarity −1..+1
    pos = models.FloatField(null=True, blank=True)
    neg = models.FloatField(null=True, blank=True)
    neu = models.FloatField(null=True, blank=True)
    tags = models.JSONField(default=list, blank=True)       # EODHD news-type taxonomy, e.g. ["EARNINGS","AI"]
    # LLM classification (EODHD's own sentiment is ~all-positive & unusable; a cheap model gives a
    # real signed read). llm_rating = llm_dir(-1/0/+1) * llm_impact(0-3) -> signed -3..+3.
    llm_dir = models.SmallIntegerField(null=True, blank=True)      # -1 bearish / 0 neutral / +1 bullish
    llm_impact = models.SmallIntegerField(null=True, blank=True)   # 0 none / 1 minor / 2 moderate / 3 major
    llm_cat = models.CharField(max_length=24, blank=True)          # SIGNED type: earnings_beat/earnings_miss,
    #   guidance_up/guidance_down, upgrade/downgrade, ma, product, contract, legal, mgmt, capital, dividend,
    #   macro, clinical, other
    llm_horizon = models.CharField(max_length=8, blank=True)       # expected digestion window: day/week/month/3mo
    # title-based EVENT CATEGORY (no LLM, whole corpus) — a human-readable news type that exists for
    # EVERY item, not just the ~13mo LLM-classified slice. earnings/guidance/analyst/ma/partnership/
    # contract/product/clinical/legal/offering/dividend/buyback/insider/mgmt/macro/other.
    cat_auto = models.CharField(max_length=20, blank=True, db_index=True)
    # LOCAL-LLM refined category (qwen2.5:14b via on-box Ollama) — same taxonomy as cat_auto but far
    # more accurate on nuanced/ambiguous headlines (title heuristic lands ~69% in 'other'). Populated
    # moved-first by news_llm_category.py. Effective category = cat_llm or (fallback) cat_auto.
    cat_llm = models.CharField(max_length=20, blank=True, db_index=True)
    # OFF-TICKER guard (local LLM): the headline is NOT specifically about this ticker — it is a
    # market-wide / macro story, or it is really about a DIFFERENT company that merely surfaced in
    # this symbol's feed (e.g. a "| Stock Movers" recap tagged to one of the movers). Set by
    # news_llm_category.py, which passes the ticker to the model and asks whether the story is about
    # it. When True the item is excluded from the ticker-specific news views (the ticker association
    # is dropped for analysis; the row + its feed provenance are kept, so it's reversible).
    off_ticker = models.BooleanField(null=True, blank=True, db_index=True)
    # Same-day price EFFECT (pure candle math, no LLM): the β-adjusted abnormal move over the news's
    # own reaction session (prior close → close of the first session that trades on the news, so it
    # captures the OVERNIGHT / pre-market gap for after-hours & pre-open news). day_effect gates the
    # drift analysis to "news that actually moved the stock that day".
    day_abn = models.FloatField(null=True, blank=True, db_index=True)  # signed abnormal day-effect %
    day_effect = models.BooleanField(null=True, blank=True, db_index=True)  # |day_abn| >= threshold
    # data-quality guard: the move is likely a BAD CANDLE / artifact, not real news — illiquid or
    # sub-dollar OTC name, or a large single-bar spike that snaps back next session. When suspect,
    # day_effect is forced False so the drift analysis excludes it.
    day_suspect = models.BooleanField(null=True, blank=True, db_index=True)
    # content-quality guard (title-only heuristic, no LLM): the headline is opinion / clickbait /
    # performance-recap, NOT a discrete news event — e.g. "Is X Still a Buy?", "3 Stocks to Buy",
    # "Down 33% This Year". Excluded from the drift study and default-hidden in the dashboard table.
    junk = models.BooleanField(null=True, blank=True, db_index=True)
    # forward "results since" — raw total return from the reaction-session close forward N trading
    # days (~21 / 63 / 252 = 1mo / 3mo / 1yr), signed %. Null when the news is too recent to have the
    # full window yet, or the candle history runs out. Answers "how did the stock do after this news?".
    ret_1m = models.FloatField(null=True, blank=True)
    ret_3m = models.FloatField(null=True, blank=True)
    ret_1y = models.FloatField(null=True, blank=True)
    llm_rating = models.SmallIntegerField(null=True, blank=True, db_index=True)   # dir * impact, -3..+3
    llm_model = models.CharField(max_length=40, blank=True)
    classified_at = models.DateTimeField(null=True, blank=True)
    # LOCAL-LLM rating layer (qwen via on-box Ollama) — the offline counterpart of the llm_* fields
    # above, produced in the SAME call as cat_llm/off_ticker by news_llm_category.py. Kept in separate
    # columns so a head-to-head vs the Anthropic labels is possible before we cut consumers over.
    # local_rating (= local_dir * local_impact) doubles as the "richly classified" resume marker.
    local_dir = models.SmallIntegerField(null=True, blank=True)     # -1 bearish / 0 neutral / +1 bullish
    local_impact = models.SmallIntegerField(null=True, blank=True)  # 0 none / 1 minor / 2 moderate / 3 major
    local_horizon = models.CharField(max_length=8, blank=True)      # day/week/month/3mo
    local_rating = models.SmallIntegerField(null=True, blank=True, db_index=True)  # dir*impact, -3..+3
    url = models.CharField(max_length=1024, blank=True)
    uid = models.CharField(max_length=64, unique=True)      # hash(ticker+dt+title) to dedupe
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        indexes = [models.Index(fields=["ticker", "dt"])]

    def __str__(self):
        return f"{self.ticker} {self.dt:%Y-%m-%d} s={self.sentiment}"


class EarningsEvent(models.Model):
    """Earnings report dates + surprises from EODHD — for post-earnings-announcement drift (PEAD)
    and 'don't buy distress right before earnings' awareness."""
    ticker = models.CharField(max_length=20, db_index=True)
    report_date = models.DateField(db_index=True)
    eps_actual = models.FloatField(null=True, blank=True)
    eps_estimate = models.FloatField(null=True, blank=True)
    eps_surprise_pct = models.FloatField(null=True, blank=True)
    revenue_actual = models.FloatField(null=True, blank=True)
    revenue_estimate = models.FloatField(null=True, blank=True)
    before_after = models.CharField(max_length=16, blank=True)   # BeforeMarket / AfterMarket
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        unique_together = ["ticker", "report_date"]
        indexes = [models.Index(fields=["ticker", "report_date"])]

    def __str__(self):
        return f"{self.ticker} {self.report_date} surp={self.eps_surprise_pct}"


class EstimateRevision(models.Model):
    """Analyst EPS/revenue estimate TREND (EODHD Earnings.Trend) — the estimate moving over time
    (current vs 7d/30d ago). Estimate-revision direction is a robust factor EDGAR can't give."""
    ticker = models.CharField(max_length=20, db_index=True)
    period = models.DateField()                     # the forecast period end
    period_label = models.CharField(max_length=8, blank=True)   # +1q / +1y etc.
    eps_current = models.FloatField(null=True, blank=True)
    eps_7d_ago = models.FloatField(null=True, blank=True)
    eps_30d_ago = models.FloatField(null=True, blank=True)
    revenue_avg = models.FloatField(null=True, blank=True)
    asof = models.DateField(db_index=True)          # snapshot date
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        unique_together = ["ticker", "period", "asof"]
        indexes = [models.Index(fields=["ticker", "asof"])]

    def __str__(self):
        return f"{self.ticker} {self.period_label} cur={self.eps_current}"


class CorporateAction(models.Model):
    """Splits + cash dividends from EODHD Corporate Actions. Clean split factors let us correctly
    split-adjust off-exchange share volume (fixes the known FINRA off_pct split mismatch) and give a
    PIT dividend history. action_type distinguishes the two; ex_date is the effective date."""
    ticker = models.CharField(max_length=20, db_index=True)
    action_type = models.CharField(max_length=12, db_index=True)   # "split" | "dividend"
    ex_date = models.DateField(db_index=True)
    # split: ratio = to/from (a 4-for-1 split -> 4.0); dividend rows leave these null.
    split_ratio = models.FloatField(null=True, blank=True)
    split_from = models.FloatField(null=True, blank=True)
    split_to = models.FloatField(null=True, blank=True)
    # dividend: cash per share + the associated dates.
    dividend = models.FloatField(null=True, blank=True)
    currency = models.CharField(max_length=8, blank=True)
    record_date = models.DateField(null=True, blank=True)
    pay_date = models.DateField(null=True, blank=True)
    declaration_date = models.DateField(null=True, blank=True)
    period = models.CharField(max_length=16, blank=True)           # Quarterly / Annual …
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        unique_together = ["ticker", "action_type", "ex_date"]
        indexes = [models.Index(fields=["ticker", "ex_date"])]

    def __str__(self):
        return f"{self.ticker} {self.action_type} {self.ex_date}"


class MarketCapHistory(models.Model):
    """Point-in-time historical market cap from EODHD (historical-market-cap) — a real PIT size
    factor, more honest than shares×price computed from today's share count."""
    ticker = models.CharField(max_length=20, db_index=True)
    date = models.DateField(db_index=True)
    market_cap = models.BigIntegerField(null=True, blank=True)     # USD
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        unique_together = ["ticker", "date"]
        indexes = [models.Index(fields=["ticker", "date"])]

    def __str__(self):
        return f"{self.ticker} {self.date} mc={self.market_cap}"


class DelistedCompany(models.Model):
    """Delisted / inactive tickers from EODHD (exchange 'delisted' list) — a survivorship-free
    reference so backtests can acknowledge names that later died (delisting = a real exit)."""
    ticker = models.CharField(max_length=20, db_index=True)        # plain/yahoo-style ticker
    eodhd_symbol = models.CharField(max_length=32, blank=True)     # TICKER.EXCHANGE
    name = models.CharField(max_length=255, blank=True)
    exchange = models.CharField(max_length=32, blank=True)
    country = models.CharField(max_length=8, blank=True)
    currency = models.CharField(max_length=8, blank=True)
    type = models.CharField(max_length=40, blank=True)             # Common Stock / ETF / …
    isin = models.CharField(max_length=20, blank=True)
    delisted_date = models.DateField(null=True, blank=True, db_index=True)
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        unique_together = ["ticker", "exchange"]
        indexes = [models.Index(fields=["ticker"])]

    def __str__(self):
        return f"{self.ticker} ({self.exchange}) delisted {self.delisted_date}"


class TreasuryRate(models.Model):
    """Official US Treasury yield curve from EODHD (UST API, beta) — daily par yields, T-bill rates
    and real (TIPS) yields by tenor. Cleaner than the yfinance ^TNX/^IRX proxy for the rates regime,
    and gives the full curve (inversion) + real yields the proxy can't."""
    date = models.DateField(db_index=True)
    series = models.CharField(max_length=16, db_index=True)        # yield | bill | real
    tenor = models.CharField(max_length=8)                         # 1M,3M,6M,1Y,2Y,3Y,5Y,7Y,10Y,20Y,30Y
    rate = models.FloatField(null=True, blank=True)                # percent
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        unique_together = ["date", "series", "tenor"]
        indexes = [models.Index(fields=["series", "date"])]

    def __str__(self):
        return f"{self.date} {self.series} {self.tenor}={self.rate}"


class CongressTrade(models.Model):
    """US Congressional (legislator) stock trades from EODHD (congressional-trades, beta) — the
    'follow the politicians' alt-data signal. Disclosed under the STOCK Act, so report_date lags the
    transaction; PIT studies MUST key on report_date (when it became public), not transaction_date.
    Amounts are disclosed as a RANGE (amount_min..amount_max), not an exact figure."""
    ticker = models.CharField(max_length=20, db_index=True)
    member = models.CharField(max_length=160, db_index=True)
    chamber = models.CharField(max_length=16, blank=True)          # Senate / House
    party = models.CharField(max_length=24, blank=True)
    state = models.CharField(max_length=8, blank=True)
    transaction_type = models.CharField(max_length=16, db_index=True)  # buy / sell / exchange
    transaction_date = models.DateField(null=True, blank=True, db_index=True)
    report_date = models.DateField(null=True, blank=True, db_index=True)   # public disclosure (PIT)
    amount_min = models.BigIntegerField(null=True, blank=True)     # disclosed range low
    amount_max = models.BigIntegerField(null=True, blank=True)     # disclosed range high
    owner = models.CharField(max_length=24, blank=True)            # self / spouse / child / joint
    asset_type = models.CharField(max_length=40, blank=True)
    uid = models.CharField(max_length=64, unique=True)             # hash to dedupe re-imports
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        indexes = [models.Index(fields=["ticker", "report_date"]),
                   models.Index(fields=["member", "report_date"])]

    def __str__(self):
        return f"{self.member} {self.transaction_type} {self.ticker} {self.report_date}"


class BacktestResult(models.Model):
    """Persisted backtest payloads so results live in Postgres like every other study — queryable and
    auto-updated by the nightly recompute — instead of ONLY in a .data JSON file. `payload` is the
    exact JSON the API serves (rotation lab, or the rotation-edge decomposition)."""
    kind = models.CharField(max_length=32, unique=True)   # "rotation_lab" | "decomposition"
    payload = models.JSONField()
    computed_at = models.DateTimeField()
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    def __str__(self):
        return f"{self.kind} @ {self.computed_at:%Y-%m-%d %H:%M}"
