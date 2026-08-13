"""Fallback P/B via EODHD for foreign holdings yfinance can't price (chiefly Korea .KS -> EODHD .KO).
EODHD fundamentals give Valuation.PriceBookMRQ + Highlights.MarketCapitalization/PERatio for exchanges
yfinance leaves null. Stored under the ticker we hold (e.g. 005930.KS) so it matches candles/holdings.
Run: docker exec rotation-backend-1 python /app/_fetch_eodhd_pb.py [SUFFIX_MAP defaults to KS->KO]
"""
import os, time, json, urllib.request
os.environ.setdefault("DJANGO_SETTINGS_MODULE", "rotation.settings")
import django
django.setup()
from datetime import date
import sector_holdings
from core.models import Fundamental, Candle

TOKEN = os.environ.get("EODHD_API_KEY") or os.environ.get("EODHD_TOKEN") or os.environ.get("EODHD_API_TOKEN")
# yfinance-suffix -> EODHD-suffix for exchanges yfinance can't price P/B for.
SUFFIX_MAP = {".KS": ".KO"}   # Korea KOSPI. (KOSDAQ would be .KQ; large-cap EWY holdings are KOSPI.)
SLEEVES = ["South Korea", "Emerging Markets"]   # sleeves holding .KS names


def _num(x):
    try:
        f = float(x); return f if f == f else None
    except (TypeError, ValueError):
        return None


def eodhd_fund(sym):
    url = f"https://eodhd.com/api/fundamentals/{sym}?api_token={TOKEN}&fmt=json"
    with urllib.request.urlopen(url, timeout=25) as r:
        return json.load(r)


def main():
    if not TOKEN:
        print("no EODHD token"); return
    targets = set()
    for s in SLEEVES:
        for t in sector_holdings.get_holdings(s):
            for ys, es in SUFFIX_MAP.items():
                if t.endswith(ys):
                    targets.add((t, t[: -len(ys)] + es))
    targets = sorted(targets)
    print(f"{len(targets)} foreign tickers to price via EODHD", flush=True)

    today = date.today()
    got = 0
    for i, (store, sym) in enumerate(targets):
        try:
            d = eodhd_fund(sym)
        except Exception as e:
            print(f"  {store} ({sym}): ERR {str(e)[:40]}", flush=True); continue
        val = d.get("Valuation", {}) if isinstance(d, dict) else {}
        hl = d.get("Highlights", {}) if isinstance(d, dict) else {}
        pb = _num(val.get("PriceBookMRQ"))
        fields = {
            "pb_ratio": pb, "pe_ratio": _num(hl.get("PERatio")),
            "market_cap": int(_num(hl.get("MarketCapitalization"))) if _num(hl.get("MarketCapitalization")) else None,
            "profit_margin": _num(hl.get("ProfitMargin")), "eps": _num(hl.get("EarningsShare")),
            "book_value": _num(hl.get("BookValue")),
            "sector": (d.get("General", {}) or {}).get("Sector"),
            "industry": (d.get("General", {}) or {}).get("Industry"),
        }
        if any(v is not None for v in fields.values()):
            Fundamental.objects.update_or_create(ticker=store, date=today, defaults=fields)
        if pb and pb > 0:
            got += 1
            print(f"  {store}: P/B {pb}  cap {fields['market_cap']}  candle={Candle.objects.filter(ticker=store).exists()}", flush=True)
        time.sleep(0.3)
    print(f"\nDONE: {got}/{len(targets)} priced with positive P/B via EODHD", flush=True)


if __name__ == "__main__":
    main()
