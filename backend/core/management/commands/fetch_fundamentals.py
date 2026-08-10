"""
Fetch fundamental data for all sector ETFs and their holdings from yfinance.
Usage: python manage.py fetch_fundamentals
"""

import time
from datetime import date
from django.core.management.base import BaseCommand
from core.models import Fundamental, QuarterlyEarnings
import config
import sector_holdings


class Command(BaseCommand):
    help = "Fetch fundamental data from yfinance for all tickers"

    def add_arguments(self, parser):
        parser.add_argument('--etfs-only', action='store_true', help='Only fetch ETF data')
        parser.add_argument('--ticker', type=str, help='Fetch single ticker')

    def handle(self, *args, **options):
        import yfinance as yf

        tickers = set()

        if options.get('ticker'):
            tickers.add(options['ticker'].upper())
        elif options.get('etfs_only'):
            tickers.update(config.SECTOR_ETFS.values())
        else:
            # All ETFs + all holdings
            tickers.update(config.SECTOR_ETFS.values())
            for sector_name, data in sector_holdings.HOLDINGS.items():
                for holding in data.get("holdings", []):
                    tickers.add(holding)

        tickers = sorted(tickers)
        self.stdout.write(f"Fetching fundamentals for {len(tickers)} tickers...")

        today = date.today()
        saved = 0
        errors = 0

        for i, ticker in enumerate(tickers):
            try:
                info = yf.Ticker(ticker).info
                if not info or info.get("trailingPegRatio") is None and info.get("marketCap") is None and info.get("trailingEps") is None:
                    self.stdout.write(f"  [{i+1}/{len(tickers)}] {ticker}: no data")
                    continue

                fields = {
                    "dividend_yield": _g(info, "dividendYield"),
                    "pe_ratio": _g(info, "trailingPE"),
                    "forward_pe": _g(info, "forwardPE"),
                    "pb_ratio": _g(info, "priceToBook"),
                    "ps_ratio": _g(info, "priceToSalesTrailing12Months"),
                    "peg_ratio": _g(info, "pegRatio"),
                    "market_cap": _gi(info, "marketCap"),
                    "enterprise_value": _gi(info, "enterpriseValue"),
                    "eps": _g(info, "trailingEps"),
                    "forward_eps": _g(info, "forwardEps"),
                    "annual_revenue": _gi(info, "totalRevenue"),
                    "revenue_growth": _g(info, "revenueGrowth"),
                    "earnings_growth": _g(info, "earningsGrowth"),
                    "profit_margin": _g(info, "profitMargins"),
                    "operating_margin": _g(info, "operatingMargins"),
                    "shares_outstanding": _gi(info, "sharesOutstanding"),
                    "float_shares": _gi(info, "floatShares"),
                    "short_ratio": _g(info, "shortRatio"),
                    "short_pct_float": _g(info, "shortPercentOfFloat"),
                    "insider_pct": _g(info, "heldPercentInsiders"),
                    "institution_pct": _g(info, "heldPercentInstitutions"),
                    "analyst_rating": info.get("recommendationKey"),
                    "analyst_target": _g(info, "targetMeanPrice"),
                    "analyst_count": _gi(info, "numberOfAnalystOpinions"),
                    "total_cash": _gi(info, "totalCash"),
                    "total_debt": _gi(info, "totalDebt"),
                    "debt_to_equity": _g(info, "debtToEquity"),
                    "current_ratio": _g(info, "currentRatio"),
                    "book_value": _g(info, "bookValue"),
                    "free_cash_flow": _gi(info, "freeCashflow"),
                    "operating_cash_flow": _gi(info, "operatingCashflow"),
                    "beta_5y": _g(info, "beta"),
                    "fifty_two_wk_high": _g(info, "fiftyTwoWeekHigh"),
                    "fifty_two_wk_low": _g(info, "fiftyTwoWeekLow"),
                    "avg_volume": _gi(info, "averageVolume"),
                    "sector": info.get("sector"),
                    "industry": info.get("industry"),
                }

                Fundamental.objects.update_or_create(
                    ticker=ticker,
                    date=today,
                    defaults=fields,
                )
                saved += 1
                cap = f"${info.get('marketCap', 0)/1e9:.1f}B" if info.get('marketCap') else '-'
                self.stdout.write(f"  [{i+1}/{len(tickers)}] {ticker}: EPS={fields['eps']} P/E={fields['pe_ratio']} Cap={cap}")

            except Exception as e:
                errors += 1
                self.stdout.write(f"  [{i+1}/{len(tickers)}] {ticker}: ERROR {e}")

            # Rate limit: yfinance throttles after many requests
            if (i + 1) % 20 == 0:
                time.sleep(2)

        self.stdout.write(f"\nDone: {saved} saved, {errors} errors, {Fundamental.objects.count()} total in DB")

        # Fetch quarterly earnings history
        self.stdout.write("\nFetching quarterly earnings history...")
        q_saved = 0
        for i, ticker in enumerate(tickers):
            try:
                t = yf.Ticker(ticker)
                # Try quarterly earnings
                qe = getattr(t, 'quarterly_earnings', None)
                if qe is not None and hasattr(qe, 'iterrows') and len(qe) > 0:
                    for qdate, row in qe.iterrows():
                        q_date = qdate.date() if hasattr(qdate, 'date') else qdate
                        QuarterlyEarnings.objects.update_or_create(
                            ticker=ticker,
                            date=q_date,
                            defaults={
                                "revenue": _gi_val(row.get("Revenue")),
                                "earnings": _gi_val(row.get("Earnings")),
                            }
                        )
                        q_saved += 1

                # Try quarterly income statement for EPS
                qi = getattr(t, 'quarterly_income_stmt', None)
                if qi is not None and hasattr(qi, 'columns') and len(qi.columns) > 0:
                    for col in qi.columns:
                        q_date = col.date() if hasattr(col, 'date') else col
                        eps_val = None
                        if 'Basic EPS' in qi.index:
                            eps_val = _g_val(qi.loc['Basic EPS', col])
                        elif 'Diluted EPS' in qi.index:
                            eps_val = _g_val(qi.loc['Diluted EPS', col])
                        if eps_val is not None:
                            QuarterlyEarnings.objects.update_or_create(
                                ticker=ticker,
                                date=q_date,
                                defaults={"eps": eps_val}
                            )
            except Exception:
                pass

            if (i + 1) % 20 == 0:
                time.sleep(2)
                self.stdout.write(f"  [{i+1}/{len(tickers)}] quarterly earnings...")

        self.stdout.write(f"Quarterly earnings: {q_saved} records, {QuarterlyEarnings.objects.count()} total")


def _g(info, key):
    """Get float value, return None if missing or NaN."""
    v = info.get(key)
    if v is None:
        return None
    try:
        f = float(v)
        return f if f == f else None  # NaN check
    except (TypeError, ValueError):
        return None


def _gi(info, key):
    """Get int value."""
    v = info.get(key)
    if v is None:
        return None
    try:
        return int(v)
    except (TypeError, ValueError):
        return None


def _g_val(v):
    """Get float from a direct value."""
    if v is None:
        return None
    try:
        f = float(v)
        return f if f == f else None
    except (TypeError, ValueError):
        return None


def _gi_val(v):
    """Get int from a direct value."""
    if v is None:
        return None
    try:
        i = int(float(v))
        return i if float(v) == float(v) else None
    except (TypeError, ValueError):
        return None
