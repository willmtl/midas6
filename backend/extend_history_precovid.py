"""Extend candle history BEFORE COVID (~2015) for the FLAGSHIP BACKTEST universe only (sector-ETF holdings
+ delisted names with fundamentals), so the strategy launches ~2016 and trades THROUGH the 2020 crash
instead of starting at its bottom — removing the #1 return inflator. Run in celery-worker (EODHD egress):

  MSYS_NO_PATHCONV=1 docker exec -d rotation-celery-worker-1 sh -c \
    'cd /app && CANDLE_YEARS=11 setsid nohup python -u extend_history_precovid.py \
     > /app/.data/extend_precovid.log 2>&1 < /dev/null &'
"""
import os
import django
os.environ.setdefault("DJANGO_SETTINGS_MODULE", "rotation.settings")
django.setup()

import sector_holdings
from core.models import FinancialReport
from fetch_candles_eodhd import backfill, FROM

# the backtest universe: every sector-ETF holding + the sector ETFs themselves + benchmarks
tickers = set(sector_holdings.get_all_unique_tickers())
for name, d in sector_holdings.HOLDINGS.items():
    tickers.add(d["etf"])
tickers |= {"SPY", "QQQ", "IVV", "DIA", "VTI"}
# + delisted names that have fundamentals (they're in the backtest via GICS mapping)
have_fund = set(FinancialReport.objects.values_list("ticker", flat=True).distinct())
tickers |= {t for t in have_fund}                     # ensure every fundamentals name has deep candles too
tickers = sorted(t for t in tickers if t)

print(f"extending candles to FROM={FROM} for {len(tickers)} backtest-universe tickers", flush=True)
backfill(tickers, jobs=8)
print("EXTEND COMPLETE", flush=True)
