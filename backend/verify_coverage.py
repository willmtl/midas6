#!/usr/bin/env python3
"""VERIFY the 2015-extended C backtest actually HAS the data (user: 'you likely did it wrong or did not have
all the data'). Reports (1) per-year candle coverage: distinct tickers + row counts, earliest/latest date;
(2) how many of C's ACTUAL small-cap value universe (USCA holdings, <$2B, in Fundamental) were tradable each
year -- the real question is whether 2015-2019 had a deep enough pool or the -60.8%/12929% came from a thin,
survivorship-biased early sample. Run: MSYS_NO_PATHCONV=1 docker exec rotation-backend-1 python -u /app/verify_coverage.py"""
import os, warnings
warnings.filterwarnings("ignore")
os.environ.setdefault("DJANGO_SETTINGS_MODULE", "rotation.settings")
import django
django.setup()
from django.db import connection

with connection.cursor() as cur:
    cur.execute("SET max_parallel_workers_per_gather = 0")

    # (1) overall span + per-year distinct tickers / rows (stocks only: interval='1d')
    cur.execute("SELECT MIN(date), MAX(date), COUNT(*) FROM core_candle WHERE interval='1d'")
    lo, hi, tot = cur.fetchone()
    print(f"CANDLE span: {lo} .. {hi}   total 1d rows={tot:,}\n", flush=True)

    print(f"  {'year':>6}{'distinct_tickers':>18}{'rows':>14}{'median_rows/tk':>16}", flush=True)
    cur.execute("""
        SELECT EXTRACT(YEAR FROM date)::int yr,
               COUNT(DISTINCT ticker) ntk,
               COUNT(*) rows
        FROM core_candle WHERE interval='1d'
        GROUP BY 1 ORDER BY 1
    """)
    for yr, ntk, rows in cur.fetchall():
        print(f"  {yr:>6}{ntk:>18,}{rows:>14,}{rows/max(1,ntk):>16.0f}", flush=True)

# (2) C's real tradable universe per year: USCA holdings that are (a) in Fundamental, (b) have candles that year
print("\n--- C's tradable small-cap universe by year (USCA holdings ∩ Fundamental ∩ candles) ---", flush=True)
import config, sector_holdings
from trend_stock_studies import CRYPTO
from backtest_lowpb import BENCH
from core.models import Fundamental

etfs = {n: e for n, e in config.SECTOR_ETFS.items() if e not in CRYPTO}
holds = set()
for name, etf in etfs.items():
    for t in sector_holdings.get_holdings(name):
        if t not in (etf, BENCH) and t not in CRYPTO:
            holds.add(t)
holds = sorted(holds)
print(f"total mapped USCA holdings across {len(etfs)} sectors: {len(holds)}", flush=True)

fund_tk = set(Fundamental.objects.values_list("ticker", flat=True).distinct())
universe = [t for t in holds if t in fund_tk]
print(f"of those, with ANY Fundamental row: {len(universe)}", flush=True)

# per-year: how many of `universe` had >=200 trading days that year (a real, tradable name)
with connection.cursor() as cur:
    cur.execute("SET max_parallel_workers_per_gather = 0")
    ph = ",".join(["%s"] * len(universe))
    cur.execute(f"""
        SELECT yr, COUNT(*) FROM (
            SELECT ticker, EXTRACT(YEAR FROM date)::int yr, COUNT(*) d
            FROM core_candle
            WHERE interval='1d' AND ticker IN ({ph})
            GROUP BY ticker, yr
            HAVING COUNT(*) >= 200
        ) s GROUP BY yr ORDER BY yr
    """, universe)
    print(f"\n  {'year':>6}{'tradable_names(>=200d)':>24}", flush=True)
    for yr, n in cur.fetchall():
        print(f"  {yr:>6}{n:>24,}", flush=True)
