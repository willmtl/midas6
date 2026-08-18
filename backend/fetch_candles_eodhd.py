#!/usr/bin/env python3
"""EODHD CANDLE RE-IMPORTER — repair for the destructive yfinance updater. `import_candles_task` deleted candle
windows then re-downloaded from yfinance; yfinance has NO egress from this container, so the refetch returned
empty and US price history got wiped to 2021-07..2023-05 (foreign tickers, sourced elsewhere, survived to 2026).
This rebuilds full daily history from EODHD (egress works) and is NON-DESTRUCTIVE: a ticker's rows are replaced
ONLY after a non-empty EODHD fetch, so a bad response never destroys data.

Storage matches the existing auto_adjust=True convention: close = EODHD adjusted_close (split+div adjusted);
open/high/low = raw * (adjusted_close/close); volume = raw (see caveat). interval='1d'.
CLI:
  python -u fetch_candles_eodhd.py --probe                 # 5 tickers, report spans, save nothing
  python -u fetch_candles_eodhd.py --run [--limit N] [--jobs 8]
Run: MSYS_NO_PATHCONV=1 docker exec rotation-backend-1 python -u /app/fetch_candles_eodhd.py --run --jobs 8
"""
import os, argparse, datetime as dt
from concurrent.futures import ThreadPoolExecutor, as_completed
os.environ.setdefault("DJANGO_SETTINGS_MODULE", "rotation.settings")
import django  # noqa: E402
django.setup()

import config  # noqa: E402
from api.tasks import _eodhd_get, _eodhd_sym  # noqa: E402
from core.models import Candle  # noqa: E402

YEARS = int(os.environ.get("CANDLE_YEARS", "11"))    # 2026-08-18: default 11y (~2015) so the backtest trades
                                                     # THROUGH the 2020 crash, not from its bottom (the #1 return
                                                     # inflator). The re-importer is delete+reinsert per ticker, so
                                                     # a shallower FROM would TRUNCATE this deep history — keep 11.
FROM = (dt.date.today() - dt.timedelta(days=int(YEARS * 365.25))).isoformat()


def _rows(tk):
    """Fetch EODHD daily EOD for ticker -> list of Candle(...) (adjusted). [] if unavailable."""
    sym = _eodhd_sym(tk)
    if sym is None:
        return []
    resp = _eodhd_get(f"eod/{sym}", **{"from": FROM, "period": "d"})
    if not isinstance(resp, list) or not resp:
        return []
    out = []
    for r in resp:
        d = r.get("date")
        cl = r.get("close")
        adj = r.get("adjusted_close", cl)
        if not d or cl in (None, "") or adj in (None, ""):
            continue
        try:
            cl = float(cl); adj = float(adj)
            fac = adj / cl if cl else 1.0
            o = float(r.get("open") or cl) * fac
            h = float(r.get("high") or cl) * fac
            lo = float(r.get("low") or cl) * fac
            vol = int(float(r.get("volume") or 0))
        except (TypeError, ValueError):
            continue
        if adj <= 0:
            continue
        out.append(Candle(ticker=tk, date=d, interval="1d", open=o, high=h, low=lo, close=adj, volume=vol))
    return out


def universe():
    tks = set(config.SECTOR_ETFS.values()) | {getattr(config, "BENCHMARK", "SPY"), "SPY"}
    tks |= set(Candle.objects.filter(interval="1d").values_list("ticker", flat=True).distinct())
    try:
        from seq_fundamental_study import build_universe
        tks |= set(build_universe())
    except Exception as e:
        print("build_universe failed:", e, flush=True)
    # skip futures / vix (EODHD symbol mapping differs); keep equities/ETFs/crypto
    return sorted(t for t in tks if "=F" not in t and not t.startswith("^"))


def backfill(tickers, jobs=8):
    print(f"EODHD candle rebuild: {len(tickers)} tickers from {FROM} (non-destructive)", flush=True)
    done = restored = skipped = 0
    results = {}
    with ThreadPoolExecutor(max_workers=jobs) as ex:
        futs = {ex.submit(_rows, tk): tk for tk in tickers}
        for f in as_completed(futs):
            tk = futs[f]
            try:
                results[tk] = f.result()
            except Exception:
                results[tk] = []
    for tk in tickers:
        objs = results.get(tk) or []
        done += 1
        if not objs:
            skipped += 1
            continue
        # replace only after a good fetch (safe): delete this ticker's rows, insert fresh full history
        Candle.objects.filter(ticker=tk, interval="1d").delete()
        Candle.objects.bulk_create(objs, ignore_conflicts=True, batch_size=5000)
        restored += 1
        if restored % 100 == 0:
            print(f"  ...{restored} restored / {done} processed", flush=True)
    print(f"DONE: {restored} tickers restored, {skipped} skipped (no EODHD data), {done} processed", flush=True)


def _probe():
    for tk in ("SPY", "XLK", "AAPL", "NVDA", "TLT"):
        r = _rows(tk)
        if r:
            print(f"  {tk}: {len(r)} bars  {r[0].date} -> {r[-1].date}  (last close {r[-1].close:.2f})", flush=True)
        else:
            print(f"  {tk}: NO DATA (sym={_eodhd_sym(tk)})", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--probe", action="store_true")
    ap.add_argument("--run", action="store_true")
    ap.add_argument("--jobs", type=int, default=8)
    ap.add_argument("--limit", type=int, default=None)
    a = ap.parse_args()
    if not (a.run or a.probe):
        a.probe = True
    if not os.environ.get("EODHD_API_KEY"):
        print("EODHD_API_KEY not set — aborting", flush=True); return
    if a.probe:
        _probe()
    if a.run:
        tks = universe()
        if a.limit:
            tks = tks[: a.limit]
        backfill(tks, jobs=a.jobs)


if __name__ == "__main__":
    main()
