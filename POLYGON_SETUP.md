# Polygon / Massive integration — setup & runbook

Everything is **key-driven** via `POLYGON_API_KEY`. Nothing calls Polygon until that env var is set,
so the code is safe to ship un-subscribed (all collectors no-op cleanly without the key).

## Plan to buy — options & stocks are SEPARATE subscriptions (post-rebrand)
All tiers below Advanced are 15-min delayed, which is **fine for our EOD/positional use** — skip the
$199 real-time tiers.
- **Options** (IV / OI / Greeks / skew / GEX → `collect_options_polygon`, snapshot endpoint):
  - **Options Starter $29** — Greeks, IV, daily OI, flat files, snapshot; **2yr** history.
  - **Options Developer $79** — same + trades; **4yr** history (recommended for backtesting).
- **Dark pool** (off-exchange trades → `collect_darkpool_polygon`, `/v3/trades` + flat files):
  - **Stocks Developer $79** — cheapest with trade-level data + flat files; **10yr** history.
- **"Both" ≈ $158/mo** (Options Developer + Stocks Developer). Budget: Options Starter $29 +
  Stocks Developer $79 = **$108/mo**. Options-only to start: **$29** (Starter).

## Set the key
Add to the backend service env (docker-compose `environment:` or a `.env`), then restart backend:
```
POLYGON_API_KEY=your_key_here
# for the dark-pool FLAT-FILE backfill only (from the Polygon dashboard → Flat Files / S3):
POLYGON_S3_KEY=...
POLYGON_S3_SECRET=...
```

## What runs automatically (once the key is set)
| Celery beat | task | what it does |
|---|---|---|
| 22:15 UTC | `run_options_snapshot` | Polygon option-chain snapshot → `OptionSnapshot` (atm_iv, pc_oi, pc_vol, **iv_skew**, **gex**). Falls back to yfinance if no key. |
| 22:20 UTC | `run_darkpool_snapshot` | Polygon trade tape → `DarkPoolDay` (off_pct, block_off_vol). No-op without key. |

## Data model
- **`OptionSnapshot`** (ticker, date, spot, atm_iv, pc_vol, pc_oi, **iv_skew**, **gex**, source) — daily options summary.
- **`DarkPoolDay`** (ticker, date, total_vol, off_vol, **off_pct**, **block_off_vol**, block_min, source) — daily off-exchange share.

## First-run verification (do this the day you subscribe)
The REST field shapes are coded to the public docs; confirm them against a live response:
```bash
docker compose exec backend python -u -c "\
import django,os; os.environ.setdefault('DJANGO_SETTINGS_MODULE','rotation.settings'); django.setup(); \
from api.tasks import collect_options_polygon, collect_darkpool_polygon; \
print(collect_options_polygon(limit=3)); \
print(collect_darkpool_polygon(['AAPL'], day='2026-08-07'))"
```
Check the saved rows look sane (IV 0–200%, pc_oi ~0.3–2, off_pct ~0.3–0.6). If a field name differs
(e.g. `greeks.gamma`, `open_interest`, `day.volume`, trade `exchange`/`trf_id`), fix in
`collect_options_polygon` / `collect_darkpool_polygon` and re-run.

## Historical backfill (the backtest enabler — finalize on subscribe)
- **Options history** → `backfill_options_polygon(tickers, start, end)` (stub). Polygon's snapshot is
  current-only, so history = per-contract daily **aggregates** (have per-day *volume*) + **Black-Scholes
  IV/skew** computed from option close + underlying + risk-free. Historical **OI** is not reliably in
  Polygon REST → OI-based history (pc_oi, GEX) may need the S3 flat files. Wire this against live data.
- **Dark-pool history** → `backfill_darkpool_flatfiles(dates, tickers)` (stub). Pull daily trade flat
  files from S3 (`s3://flatfiles/us_stocks_sip/trades_v1/YYYY/MM/YYYY-MM-DD.csv.gz`), filter
  `exchange==4 & trf_id`, aggregate per ticker/day. Needs `POLYGON_S3_KEY/SECRET` + boto3.

## Then: validate before trusting (same as VIX)
Once history is loaded, run the disaster-rate validation:
- Does high **put/call** predict up (contrarian) or down (informed)?  ← the open question
- Do single-stock **IV spikes** improve Mode A (the VIX effect per name)?
- Does **put-skew** / **GEX** separate knife from bounce?

Nothing gets wired into the risk rating until it survives that test.
