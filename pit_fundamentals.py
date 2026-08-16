"""Point-in-time fundamentals: turn a ticker's price frame + quarterly reports into a
date-indexed metrics DataFrame, plus the bucket functions for every dimension.

Pure pandas/numpy — NO Django, NO `studies` import — so it stays unit-testable and cheap
to import in spawned workers. All lookups honour REPORT_LAG_DAYS: a quarter's numbers are
only visible on/after period_end + REPORT_LAG_DAYS (avail_date).
"""
from datetime import timedelta
import numpy as np
import pandas as pd

REPORT_LAG_DAYS = 45
BETA_WINDOW = 252   # ~1y of trading days
VOL_WINDOW = 20     # trailing avg volume
HIGH_LOW_WINDOW = 252  # 52-week range

# EPS is intentionally NOT summed here: per-share figures are split-sensitive (mixing
# pre/post-split quarters corrupts the sum). We derive TTM EPS from net income / shares
# and PE from market_cap / net_income_ttm instead — both split-agnostic dollar figures.
_TTM_SUM_COLS = ["revenue", "net_income", "operating_income", "free_cash_flow",
                 "operating_cash_flow", "gross_profit", "cost_of_revenue", "rd_expense"]
_POINT_COLS = ["total_equity", "total_debt", "current_assets", "current_liabilities", "shares_outstanding"]


def avail_date(period_end):
    """Date a quarter's numbers are treated as public."""
    return period_end + timedelta(days=REPORT_LAG_DAYS)


def pit_row(metrics, when):
    """Last metrics row on/before `when`, tolerating NaN columns.

    NOT `DataFrame.asof` — that returns the last *fully non-NaN* row, which is almost
    never what we want here (beta/dividend_yield/early-TTM columns are legitimately NaN).
    This does a positional forward-fill lookup and returns that row's Series (or None if
    `when` precedes the frame)."""
    if metrics is None or len(metrics) == 0:
        return None
    pos = metrics.index.get_indexer([pd.Timestamp(when)], method="ffill")[0]
    if pos < 0:
        return None
    return metrics.iloc[pos]


def _quarterly_frame(reports_df):
    """TTM/prior-TTM columns, indexed by avail_date for a forward-fill onto trading days.

    TTM is rolled on the *period_end* axis (chronological, one row per quarter) so the
    rolling-4Q sums are correct. Only afterward do we re-key to avail_date for the ffill;
    if two quarters were filed the same day, the later period_end wins."""
    if reports_df is None or reports_df.empty:
        return None
    r = reports_df.copy()
    r["avail_date"] = pd.to_datetime(r["avail_date"])
    r["period_end"] = pd.to_datetime(r["period_end"])
    r = r.sort_values("period_end").drop_duplicates("period_end")
    for c in _TTM_SUM_COLS:
        if c in r.columns:
            r[c + "_ttm"] = r[c].rolling(4, min_periods=4).sum()
            r[c + "_ttm_prior"] = r[c + "_ttm"].shift(4)
    if "shares_outstanding" in r.columns:
        r["shares_prior"] = r["shares_outstanding"].shift(4)  # ~1y ago, for buyback yield
    r = r.sort_values(["avail_date", "period_end"]).drop_duplicates("avail_date", keep="last")
    return r.set_index("avail_date")


def prepare_pit_metrics(price_df, reports_df, dividends, spy_close, insider=None, filings=None,
                        darkpool=None, news=None):
    """Return a DataFrame indexed by price_df.index with all point-in-time metric columns.
    Missing inputs yield NaN (never raises). `insider` = Series of open-market buy $ indexed
    by filed_date. `filings` = DataFrame[filed_date, form_group] of 13D/13G filings.
    `darkpool` = DataFrame[off_pct, published_date] of FINRA weekly ATS volume. `news` =
    Series of sentiment polarity indexed by news datetime."""
    idx = price_df.index
    close = price_df["Close"].astype(float)
    out = pd.DataFrame(index=idx)

    q = _quarterly_frame(reports_df)
    if q is not None:
        # Forward-fill each quarterly column onto trading days by avail_date (asof).
        aligned = q.reindex(q.index.union(idx)).sort_index().ffill().reindex(idx)
        rev = aligned["revenue_ttm"]
        ni = aligned["net_income_ttm"]
        oi = aligned["operating_income_ttm"]
        fcf = aligned["free_cash_flow_ttm"]
        shares = aligned["shares_outstanding"]
        equity = aligned["total_equity"]
        debt = aligned["total_debt"]
        cur_a = aligned["current_assets"]
        cur_l = aligned["current_liabilities"]
        rev_prior = aligned["revenue_ttm_prior"]
        ni_prior = aligned["net_income_ttm_prior"]

        mcap = close * shares
        out["market_cap"] = mcap
        out["float_shares"] = shares      # PIT share count (EDGAR), a point-in-time float proxy
        out["eps_ttm"] = _safe_div(ni, shares)       # TTM EPS from net income (split-safe)
        out["pe"] = _safe_div(mcap, ni)              # = price / (ni/shares); split-agnostic
        out["ps"] = _safe_div(mcap, rev)
        out["pb"] = _safe_div(mcap, equity)
        out["profit_margin"] = _safe_div(ni, rev)
        out["operating_margin"] = _safe_div(oi, rev)
        out["debt_to_equity"] = _safe_div(debt, equity)
        out["current_ratio"] = _safe_div(cur_a, cur_l)
        out["fcf_ttm"] = fcf
        out["revenue_growth"] = _safe_div(rev, rev_prior) - 1.0
        out["earnings_growth"] = _safe_div(ni, ni_prior) - 1.0
        out["peg"] = _safe_div(out["pe"], out["earnings_growth"] * 100.0)

        # Phase A: earnings-quality / efficiency / yield metrics.
        gp = aligned.get("gross_profit_ttm")
        cogs = aligned.get("cost_of_revenue_ttm")
        rd = aligned.get("rd_expense_ttm")
        ocf = aligned.get("operating_cash_flow_ttm")
        assets = aligned.get("total_assets")
        cash = aligned.get("cash_and_equivalents")
        shares_prior = aligned.get("shares_prior")
        gm = _safe_div(gp, rev) if gp is not None else _nan(idx)
        if cogs is not None:
            gm = gm.fillna(_safe_div(rev - cogs, rev))   # fall back to (rev-COGS)/rev
        out["gross_margin"] = gm
        out["rd_intensity"] = _safe_div(rd, rev) if rd is not None else _nan(idx)
        out["asset_turnover"] = _safe_div(rev, assets) if assets is not None else _nan(idx)
        out["accruals"] = _safe_div(ni - ocf, assets) if (ocf is not None and assets is not None) else _nan(idx)
        out["fcf_yield"] = _safe_div(fcf, mcap)
        out["cash_ratio"] = _safe_div(cash, cur_l) if cash is not None else _nan(idx)
        out["buyback_yield"] = _safe_div(shares_prior - shares, shares_prior) if shares_prior is not None else _nan(idx)
    else:
        for c in ["market_cap", "float_shares", "eps_ttm", "pe", "ps", "pb", "profit_margin",
                  "operating_margin", "debt_to_equity", "current_ratio", "fcf_ttm",
                  "revenue_growth", "earnings_growth", "peg", "gross_margin",
                  "rd_intensity", "asset_turnover", "accruals", "fcf_yield",
                  "cash_ratio", "buyback_yield"]:
            out[c] = np.nan

    # Price-only metrics (always available where candles exist).
    out["avg_volume"] = price_df["Volume"].rolling(VOL_WINDOW, min_periods=1).mean()
    hi = close.rolling(HIGH_LOW_WINDOW, min_periods=20).max()
    lo = close.rolling(HIGH_LOW_WINDOW, min_periods=20).min()
    out["pct_52w"] = _safe_div(close - lo, hi - lo)
    out["dividend_yield"] = _ttm_dividend_yield(close, dividends)
    out["beta"] = _rolling_beta(close, spy_close)
    out["ad_state"] = _ad_state(price_df)
    out["insider_buy_intensity"] = _insider_intensity(close, insider, out["market_cap"])
    out["stake_13d_1y"] = _filings_trailing(close.index, filings, "13D")
    out["stake_13g_1y"] = _filings_trailing(close.index, filings, "13G")
    dp_level, dp_trend = _darkpool_features(idx, darkpool)
    out["dp_off_pct"] = dp_level
    out["dp_trend_z"] = dp_trend
    out["news_sent"] = _news_sentiment(idx, news)
    return out


def _darkpool_features(idx, darkpool, window=8):
    """PIT dark-pool level + trend as of each trading day, keyed by FINRA PUBLISH date so the
    ~3-4wk reporting lag is respected (no lookahead). `darkpool` = DataFrame[off_pct,
    published_date]. Returns (level, trend_z): level = most-recently-published weekly ATS
    off_pct; trend_z = that week's off_pct z-scored vs the trailing `window` published weeks.
    NaN where the ticker has no dark-pool data (so 'no data' stays distinct from 'low')."""
    if darkpool is None or len(darkpool) == 0:
        return _nan(idx), _nan(idx)
    d = darkpool.dropna(subset=["published_date", "off_pct"])
    if d.empty:
        return _nan(idx), _nan(idx)
    off = pd.Series(d["off_pct"].astype(float).values, index=pd.to_datetime(d["published_date"]))
    off = off.groupby(off.index).last().sort_index()      # one value per publish date
    # Z-score this week's off_pct against the TRAILING window (prior weeks only). Including the
    # current week in its own mean/std shrinks |z| toward 0 (each point is 1/window of its own
    # baseline), so a genuine surge under-reads — use shift(1) so the baseline excludes the point
    # being scored.
    prior = off.shift(1)
    rmean = prior.rolling(window, min_periods=3).mean()
    rstd = prior.rolling(window, min_periods=3).std()
    z = _safe_div(off - rmean, rstd)
    level = off.reindex(off.index.union(idx)).sort_index().ffill().reindex(idx)
    trend = z.reindex(z.index.union(idx)).sort_index().ffill().reindex(idx)
    return level, trend


def _news_sentiment(idx, news, window="7D"):
    """PIT trailing-`window` mean news sentiment as of each trading day (only news dt <= bar).
    `news` = Series of polarity (−1..1) indexed by news datetime. Built on a full daily calendar
    so a bar with no news in the window is NaN, not a stale carry-forward. NaN where no news."""
    if news is None or len(news) == 0:
        return _nan(idx)
    ni = pd.to_datetime(news.index)
    if getattr(ni, "tz", None) is not None:
        ni = ni.tz_localize(None)
    s = pd.Series(np.asarray(news.values, dtype=float), index=ni).sort_index()
    daily = s.groupby(s.index.normalize()).mean()
    if daily.empty:
        return _nan(idx)
    idxn = idx.tz_localize(None) if getattr(idx, "tz", None) is not None else idx
    full = pd.date_range(daily.index.min(), max(daily.index.max(), idxn.max()), freq="D")
    roll = daily.reindex(full).rolling(window, min_periods=1).mean()   # trailing mean, skips NaN
    out = roll.reindex(roll.index.union(idxn)).sort_index().ffill().reindex(idxn)
    out.index = idx   # restore caller's original index labels
    return out


def _filings_trailing(idx, filings, group, window="365D"):
    """Count of `group` (13D/13G) filings in the trailing window as of each day. NaN if no
    filing data at all for the ticker (so 'no data' stays distinct from 'none in window')."""
    if filings is None or len(filings) == 0:
        return pd.Series(np.nan, index=idx)
    sub = filings[filings["form_group"] == group]
    ev = pd.Series(1.0, index=pd.to_datetime(sub["filed_date"]))
    if len(ev) == 0:
        return pd.Series(0.0, index=idx)   # had filings, just not this group
    ev = ev.groupby(ev.index).sum().sort_index()
    daily = ev.reindex(ev.index.union(idx)).fillna(0.0).sort_index()
    return daily.rolling(window).sum().reindex(idx)


def _insider_intensity(close, insider, mcap):
    """Trailing-90d insider open-market buy $ as a fraction of market cap. NaN if no
    insider data for the ticker (distinct from 0 = 'had data, no buys in window')."""
    if insider is None or len(insider) == 0:
        return pd.Series(np.nan, index=close.index)
    s = insider.copy()
    s.index = pd.to_datetime(s.index)
    s = s.sort_index()
    daily = s.reindex(s.index.union(close.index)).fillna(0.0).sort_index()
    buy90 = daily.rolling("90D").sum().reindex(close.index)
    return _safe_div(buy90, mcap)


def _nan(idx):
    return pd.Series(np.nan, index=idx)


def _safe_div(a, b):
    """Elementwise a/b with 0/invalid denominators -> NaN. Accepts Series or scalars."""
    if isinstance(a, pd.Series) or isinstance(b, pd.Series):
        a_s = a if isinstance(a, pd.Series) else pd.Series(a, index=b.index)
        b_s = b if isinstance(b, pd.Series) else pd.Series(b, index=a.index)
        with np.errstate(divide="ignore", invalid="ignore"):
            return a_s / b_s.replace(0, np.nan)
    # both scalars
    if b in (0, None) or (isinstance(b, float) and np.isnan(b)):
        return np.nan
    return a / b


def _ttm_dividend_yield(close, dividends):
    if dividends is None or len(dividends) == 0:
        return pd.Series(np.nan, index=close.index)
    d = dividends.copy()
    d.index = pd.to_datetime(d.index)
    d = d.sort_index()
    # Trailing-365d dividend sum at each price date.
    daily = d.reindex(d.index.union(close.index)).fillna(0.0).sort_index()
    ttm = daily.rolling("365D").sum().reindex(close.index)
    return _safe_div(ttm, close)


def _ad_state(price_df, window=20, sma=10):
    """Point-in-time classification of the cumulative Accumulation/Distribution LINE
    (ta.accdist), read as SLOPE + DIVERGENCE vs price — NEVER as sign (project gotcha).
    Codes: 2=accum divergence (price flat/down while ADL rises — the seq-winner's edge),
    1=accum trend-up (ADL above its SMA), 0=neutral, -1=distribution (ADL below SMA & falling).
    """
    if not {"High", "Low", "Close", "Volume"}.issubset(price_df.columns):
        return pd.Series(np.nan, index=price_df.index)
    h, l, c, v = price_df["High"], price_df["Low"], price_df["Close"], price_df["Volume"]
    rng = (h - l).replace(0, np.nan)
    mfm = (((c - l) - (h - c)) / rng).fillna(0.0)
    adl = (mfm * v).cumsum()
    adl_sma = adl.rolling(sma).mean()
    px_chg = c.pct_change(window)
    adl_chg = adl.diff(window)
    code = pd.Series(0.0, index=price_df.index)
    code[(adl < adl_sma) & (adl_chg < 0)] = -1.0          # distribution
    code[adl > adl_sma] = 1.0                              # accumulation trend-up
    code[(px_chg <= 0.02) & (adl_chg > 0)] = 2.0          # bullish divergence (overrides)
    code[adl_sma.isna()] = np.nan                          # not enough history yet
    return code


def _rolling_beta(close, spy_close, window=BETA_WINDOW):
    if spy_close is None or len(spy_close) == 0:
        return pd.Series(np.nan, index=close.index)
    r = close.pct_change()
    m = spy_close.reindex(close.index).ffill().pct_change()
    cov = r.rolling(window).cov(m)
    var = m.rolling(window).var()
    return _safe_div(cov, var)


# ── Bucket functions (all dimensions; snapshot dims reuse the same fns) ─────────
def _na(v):
    return v is None or (isinstance(v, float) and np.isnan(v))


def bucket_darkpool_level(v):
    """FINRA ATS dark-pool share of that week's consolidated volume."""
    if _na(v): return "NA"
    if v < 0.05: return "low (<5%)"
    if v < 0.12: return "mid (5-12%)"
    if v < 0.20: return "high (12-20%)"
    return "very high (>=20%)"


def bucket_darkpool_trend(v):
    """Z-score of the latest published dark-pool week vs its trailing 8-week baseline."""
    if _na(v): return "NA"
    if v <= -1.0: return "distributing (z<=-1)"
    if v < 1.0: return "steady (-1..1)"
    if v < 2.0: return "accumulating (1-2)"
    return "surging (>=2)"


def bucket_news_sent(v):
    """Trailing-7d mean news sentiment polarity (−1..1) as of entry."""
    if _na(v): return "NA"
    if v <= -0.15: return "negative (<=-.15)"
    if v < 0.15: return "neutral (-.15..15)"
    if v < 0.50: return "positive (.15-.5)"
    return "very positive (>=.5)"

def bucket_ps(v):
    if _na(v): return "NA"
    if v < 1: return "cheap (<1)"
    if v < 4: return "fair (1-4)"
    if v < 10: return "rich (4-10)"
    return "expensive (>=10)"

def bucket_pb(v):
    if _na(v): return "NA"
    if v < 0: return "negative equity"
    if v < 1: return "cheap (<1)"
    if v < 3: return "fair (1-3)"
    if v < 6: return "rich (3-6)"
    return "expensive (>=6)"

def bucket_peg(v):
    if _na(v): return "NA"
    if v < 0: return "negative"
    if v < 1: return "cheap (<1)"
    if v < 2: return "fair (1-2)"
    return "expensive (>=2)"

def bucket_div_yield(v):
    if _na(v): return "NA"
    if v <= 0: return "none"
    if v < 0.02: return "low (0-2%)"
    if v < 0.04: return "mid (2-4%)"
    return "high (>=4%)"

def bucket_debt_to_equity(v):
    if _na(v): return "NA"
    if v < 0: return "negative equity"
    if v < 0.5: return "low (<0.5)"
    if v < 1.5: return "moderate (0.5-1.5)"
    return "high (>=1.5)"

def bucket_current_ratio(v):
    if _na(v): return "NA"
    if v < 1: return "weak (<1)"
    if v < 2: return "ok (1-2)"
    return "strong (>=2)"

def bucket_op_margin(v):
    if _na(v): return "NA"
    if v < 0: return "unprofitable (<0%)"
    if v < 0.15: return "thin (0-15%)"
    return "fat (>=15%)"

def bucket_earnings_growth(v):
    if _na(v): return "NA"
    if v < 0: return "shrinking (<0%)"
    if v < 0.15: return "slow (0-15%)"
    if v < 0.40: return "fast (15-40%)"
    return "hyper (>=40%)"

def bucket_fcf_sign(v):
    if _na(v): return "NA"
    return "positive FCF" if v > 0 else "negative FCF"

def bucket_beta(v):
    if _na(v): return "NA"
    if v < 0.8: return "low (<0.8)"
    if v < 1.2: return "market (0.8-1.2)"
    if v < 2.0: return "high (1.2-2)"
    return "very high (>=2)"

def bucket_avg_volume(v):
    if _na(v): return "NA"
    if v < 500_000: return "thin (<500K)"
    if v < 5_000_000: return "mid (0.5-5M)"
    return "liquid (>=5M)"

def bucket_fwd_eps(v):
    if _na(v): return "NA"
    return "positive fwd EPS" if v > 0 else "negative fwd EPS"

def bucket_short_float(v):
    if _na(v): return "NA"
    if v < 0.05: return "low (<5%)"
    if v < 0.15: return "moderate (5-15%)"
    if v < 0.30: return "high (15-30%)"
    return "squeeze-fuel (>=30%)"

def bucket_pct(v):
    """Generic 0-1 percentage buckets (institution %, insider %)."""
    if _na(v): return "NA"
    if v < 0.40: return "low (<40%)"
    if v < 0.70: return "mid (40-70%)"
    return "high (>=70%)"

# ── Phase A: earnings-quality / efficiency / yield buckets ──────────────────
def bucket_gross_margin(v):
    if _na(v): return "NA"
    if v < 0.20: return "low (<20%)"
    if v < 0.40: return "fair (20-40%)"
    if v < 0.60: return "high (40-60%)"
    return "very high (>=60%)"

def bucket_rd_intensity(v):
    if _na(v): return "NA"
    if v <= 0: return "none"
    if v < 0.05: return "low (0-5%)"
    if v < 0.15: return "mid (5-15%)"
    return "heavy (>=15%)"

def bucket_asset_turnover(v):
    if _na(v): return "NA"
    if v < 0.5: return "low (<0.5)"
    if v < 1.0: return "mid (0.5-1)"
    return "high (>=1)"

def bucket_accruals(v):
    """Sloan accruals = (NI - OCF)/assets. LOW/negative = higher earnings quality."""
    if _na(v): return "NA"
    if v < 0: return "negative (best)"
    if v < 0.10: return "low (0-10%)"
    return "high (>=10%)"

def bucket_fcf_yield(v):
    if _na(v): return "NA"
    if v < 0: return "negative"
    if v < 0.04: return "low (0-4%)"
    if v < 0.08: return "mid (4-8%)"
    return "high (>=8%)"

def bucket_cash_ratio(v):
    if _na(v): return "NA"
    if v < 0.25: return "low (<0.25)"
    if v < 1.0: return "mid (0.25-1)"
    return "high (>=1)"

def bucket_buyback_yield(v):
    """(shares_1y_ago - shares_now)/shares_1y_ago. Positive = net buyback."""
    if _na(v): return "NA"
    if v < -0.02: return "dilution (<-2%)"
    if v < 0.02: return "neutral (-2..2%)"
    if v < 0.05: return "buyback (2-5%)"
    return "heavy buyback (>=5%)"

def bucket_insider(v):
    """Trailing-90d insider open-market buy $ / market cap. 0 = no buys (had data)."""
    if _na(v): return "NA"
    if v <= 0: return "none"
    if v < 0.001: return "small (<0.1%)"
    if v < 0.01: return "notable (0.1-1%)"
    return "large cluster (>=1%)"

def bucket_stake(v):
    """Count of 13D/13G filings in trailing 1y. 0 = none (had data)."""
    if _na(v): return "NA"
    if v <= 0: return "none"
    if v < 2: return "one (1y)"
    return "multiple (1y)"

def bucket_ad(v):
    """A/D-line state code (slope + divergence, never sign)."""
    if _na(v): return "NA"
    if v >= 2: return "accum divergence"
    if v >= 1: return "accum trend-up"
    if v <= -1: return "distribution"
    return "neutral"
