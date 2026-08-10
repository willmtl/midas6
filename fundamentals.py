"""
Fundamentals Data Fetcher

Downloads dividend yield, revenue, analyst ratings for all stock holdings.
Saves to PostgreSQL via Django ORM or to JSON cache.
"""

import warnings
warnings.filterwarnings("ignore")

import json
import time
from pathlib import Path
from datetime import datetime, date
from concurrent.futures import ThreadPoolExecutor, as_completed

import yfinance as yf

import sector_holdings

CACHE_DIR = Path(__file__).parent / ".data" / "fundamentals"
CACHE_DIR.mkdir(parents=True, exist_ok=True)


def fetch_ticker_fundamentals(ticker):
    """Fetch fundamental data for a single ticker from yfinance."""
    try:
        t = yf.Ticker(ticker)
        info = t.info or {}

        result = {
            "ticker": ticker,
            "date": date.today().isoformat(),
            "dividend_yield": info.get("dividendYield"),
            "trailing_div_yield": info.get("trailingAnnualDividendYield"),
            "forward_div_yield": info.get("forwardAnnualDividendYield"),
            "annual_revenue": info.get("totalRevenue"),
            "revenue_growth": info.get("revenueGrowth"),
            "pe_ratio": info.get("trailingPE"),
            "forward_pe": info.get("forwardPE"),
            "market_cap": info.get("marketCap"),
            "analyst_rating": info.get("recommendationKey"),  # buy/hold/sell/strong_buy
            "analyst_target": info.get("targetMeanPrice"),
            "analyst_count": info.get("numberOfAnalystOpinions"),
            "current_price": info.get("currentPrice") or info.get("regularMarketPrice"),
            "sector": info.get("sector"),
            "industry": info.get("industry"),
            "beta": info.get("beta"),
            "52w_high": info.get("fiftyTwoWeekHigh"),
            "52w_low": info.get("fiftyTwoWeekLow"),
            "avg_volume": info.get("averageVolume"),
        }

        # Get dividend history
        try:
            divs = t.dividends
            if divs is not None and len(divs) > 0:
                # Annual dividend yields from history
                yearly_divs = {}
                for dt, amount in divs.items():
                    yr = dt.year
                    yearly_divs[yr] = yearly_divs.get(yr, 0) + float(amount)
                result["dividend_history"] = yearly_divs
            else:
                result["dividend_history"] = {}
        except Exception:
            result["dividend_history"] = {}

        return result
    except Exception as e:
        return {"ticker": ticker, "error": str(e)}


def fetch_all_fundamentals(max_workers=8):
    """Fetch fundamentals for all unique stock tickers across all sectors."""
    all_tickers = sector_holdings.get_all_unique_tickers()
    # Filter to US tickers only
    us_tickers = [t for t in all_tickers if "." not in t and len(t) <= 5]

    print(f"Fetching fundamentals for {len(us_tickers)} tickers...")
    results = {}

    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = {pool.submit(fetch_ticker_fundamentals, t): t for t in us_tickers}
        done = 0
        for f in as_completed(futures):
            done += 1
            r = f.result()
            ticker = r.get("ticker", "?")
            if "error" not in r:
                results[ticker] = r
            if done % 50 == 0:
                print(f"  {done}/{len(us_tickers)}")

    print(f"Got fundamentals for {len(results)} tickers")

    # Save to JSON cache
    cache_file = CACHE_DIR / "all_fundamentals.json"
    with open(cache_file, "w") as f:
        json.dump({
            "fetched_at": datetime.now().isoformat(),
            "count": len(results),
            "tickers": results,
        }, f)

    print(f"Saved to {cache_file}")
    return results


def load_fundamentals():
    """Load fundamentals from JSON cache."""
    cache_file = CACHE_DIR / "all_fundamentals.json"
    if not cache_file.exists():
        return {}
    with open(cache_file) as f:
        data = json.load(f)
    return data.get("tickers", {})


def import_to_db():
    """Import fundamentals into Django Fundamental model."""
    import django
    import os
    os.environ.setdefault("DJANGO_SETTINGS_MODULE", "rotation.settings")
    django.setup()
    from core.models import Fundamental

    data = load_fundamentals()
    if not data:
        print("No fundamentals data. Run fetch_all_fundamentals() first.")
        return

    bulk = []
    for ticker, info in data.items():
        bulk.append(Fundamental(
            ticker=ticker,
            date=info.get("date", date.today().isoformat()),
            dividend_yield=info.get("dividend_yield") or info.get("trailing_div_yield"),
            annual_revenue=info.get("annual_revenue"),
            revenue_growth=info.get("revenue_growth"),
            analyst_rating=info.get("analyst_rating"),
            analyst_target=info.get("analyst_target"),
            analyst_count=info.get("analyst_count"),
            pe_ratio=info.get("pe_ratio"),
            market_cap=info.get("market_cap"),
        ))

    Fundamental.objects.all().delete()
    Fundamental.objects.bulk_create(bulk, batch_size=500)
    print(f"Imported {len(bulk)} fundamentals into DB")


if __name__ == "__main__":
    results = fetch_all_fundamentals()

    # Show summary
    has_div = sum(1 for r in results.values() if r.get("dividend_yield"))
    has_rev = sum(1 for r in results.values() if r.get("annual_revenue"))
    has_analyst = sum(1 for r in results.values() if r.get("analyst_rating"))
    print(f"\nWith dividend yield: {has_div}")
    print(f"With revenue: {has_rev}")
    print(f"With analyst rating: {has_analyst}")

    # Top dividends
    by_div = sorted([r for r in results.values() if r.get("dividend_yield")],
                    key=lambda x: x["dividend_yield"], reverse=True)
    print(f"\nTop 10 dividend yields:")
    for r in by_div[:10]:
        print(f"  {r['ticker']:6s}  div={r['dividend_yield']*100:.2f}%  rev={r.get('annual_revenue',0)/1e9:.1f}B  analyst={r.get('analyst_rating','?')}")
