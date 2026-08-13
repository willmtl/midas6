"""One-off: fetch P/B (+ a few fields) for foreign country-sleeve holdings via yfinance so the country
sleeves pick a real cheap stock instead of holding the ETF. Korea (.KS) yfinance-P/B is null -> tries
EODHD as a fallback if a token is configured. Run: docker exec rotation-backend-1 python /app/_fetch_foreign_pb.py
"""
import os, time, json
os.environ.setdefault("DJANGO_SETTINGS_MODULE", "rotation.settings")
import django
django.setup()
from datetime import date
import yfinance as yf
import sector_holdings
from core.models import Fundamental, Candle

COUNTRY_SLEEVES = [
    "Emerging Markets", "International (EAFE)", "China", "India", "Japan", "South Korea",
    "Latin America", "Brazil", "Mexico", "Europe", "Germany", "UK", "Australia", "Africa",
]


def _g(info, k):
    v = info.get(k)
    try:
        f = float(v); return f if f == f else None
    except (TypeError, ValueError):
        return None


def _gi(info, k):
    v = info.get(k)
    try:
        return int(v)
    except (TypeError, ValueError):
        return None


def main():
    tickers = []
    seen = set()
    for s in COUNTRY_SLEEVES:
        for t in sector_holdings.get_holdings(s):
            if t not in seen:
                seen.add(t); tickers.append(t)
    print(f"{len(tickers)} unique foreign holdings across {len(COUNTRY_SLEEVES)} sleeves", flush=True)

    today = date.today()
    got_pb, no_pb, no_candle = 0, [], []
    for i, t in enumerate(tickers):
        if not Candle.objects.filter(ticker=t).exists():
            no_candle.append(t)
        try:
            info = yf.Ticker(t).info or {}
        except Exception as e:
            no_pb.append(t);
            print(f"  [{i+1}/{len(tickers)}] {t}: ERR {str(e)[:30]}", flush=True); continue
        pb = _g(info, "priceToBook")
        fields = {
            "pb_ratio": pb, "pe_ratio": _g(info, "trailingPE"), "forward_pe": _g(info, "forwardPE"),
            "market_cap": _gi(info, "marketCap"), "profit_margin": _g(info, "profitMargins"),
            "revenue_growth": _g(info, "revenueGrowth"), "shares_outstanding": _gi(info, "sharesOutstanding"),
            "book_value": _g(info, "bookValue"), "eps": _g(info, "trailingEps"),
            "sector": info.get("sector"), "industry": info.get("industry"),
        }
        if any(v is not None for v in fields.values()):
            Fundamental.objects.update_or_create(ticker=t, date=today, defaults=fields)
        if pb is not None and pb > 0:
            got_pb += 1
        else:
            no_pb.append(t)
        if (i + 1) % 20 == 0:
            time.sleep(2)
            print(f"  [{i+1}/{len(tickers)}] ... {got_pb} with P/B so far", flush=True)

    print(f"\nDONE: {got_pb} tickers got positive P/B; {len(no_pb)} without P/B", flush=True)
    print(f"no P/B (needs alt source, mostly .KS Korea): {sorted(no_pb)[:40]}", flush=True)
    if no_candle:
        print(f"no candle ({len(no_candle)}): {sorted(no_candle)[:20]}", flush=True)


if __name__ == "__main__":
    main()
