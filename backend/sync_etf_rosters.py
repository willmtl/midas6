#!/usr/bin/env python3
"""AUTO-SYNC ETF ROSTERS (user: "how will we add stocks automatically as they get added to our ETFs"). For each CORE
US-EQUITY sleeve, pull the ETF's CURRENT constituents from EODHD (ETF_Data.Holdings) and UNION new US/CA names into
.data/roster_overlay.json (merged by sector_holdings.get_holdings, default-on). UNION-ONLY: never removes a name (a
constituent dropped today was valid historically -> keeping it avoids survivorship bias; the engine filters by PIT
candles+fundamentals anyway). New names get candles + PIT FinancialReport fetched so they're actually pickable.

SCOPE = core equity sleeves only (S&P sectors + clearly-US-equity themes). Commodity / foreign-country / bond / crypto
sleeves are DELIBERATELY excluded (memory: adds there HURT — foreign −56/−82%, commodity sub-sleeves −34%). Manual
curation preserved via BLOCKLIST (SBSW = ZAR-book currency bug; ticker-rot impostors). Idempotent.
Run: MSYS_NO_PATHCONV=1 docker exec rotation-backend-1 python -u /app/sync_etf_rosters.py [--run] [--fetch]
"""
import os, json, argparse
os.environ.setdefault("DJANGO_SETTINGS_MODULE", "rotation.settings")
import django; django.setup()
import config
import sector_holdings as sh
from api.tasks import _eodhd_get
from core.models import Candle, FinancialReport

# core US-equity sleeve ETFs (GICS sectors + clearly-US-equity themes). NOT commodity/foreign/bond/crypto.
CORE_EQUITY_ETFS = {
    "XLK", "XLF", "XLV", "XLY", "XLP", "XLI", "XLB", "XLRE", "XLC", "XLE", "XLU",   # GICS sectors
    "KRE", "XBI", "IGV", "SMH", "IHI", "ITA", "XHB", "XRT", "IYT", "PAVE", "IAK",   # US-equity themes
    "CIBR", "FDN", "SKYY", "FINX", "IDNA", "SOCL", "IBUY",
}
BLOCKLIST = {"SBSW", "GOLD", "ABX"}          # ZAR-book ADR + reused/rotated tickers -> keep manual
OVERLAY_PATH = "/app/.data/roster_overlay.json"


def mapt(sym):
    """EODHD holding symbol -> our ticker convention, or None to drop (foreign non-CA / physical)."""
    if not sym or sym == "." or sym.strip() == "":
        return None
    if "." not in sym:
        return sym
    base, ex = sym.rsplit(".", 1)
    if ex == "US":
        return base
    if ex in ("TO", "V"):          # Canadian — allowed by the country gate
        return sym
    return None                    # other foreign — drop (foreign relaxation refuted)


def etf_holdings(etf):
    f = _eodhd_get(f"fundamentals/{etf}.US")
    if not isinstance(f, dict):
        return []
    ed = f.get("ETF_Data") or {}
    h = ed.get("Holdings") or ed.get("Top_10_Holdings") or {}
    return [k for k in h.keys()]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", action="store_true", help="write roster_overlay.json")
    ap.add_argument("--fetch", action="store_true", help="fetch candles+fundamentals for new names")
    a = ap.parse_args()

    name_by_etf = {e: n for n, e in config.SECTOR_ETFS.items()}
    overlay = {}
    all_new = set()
    print("=== SYNC ETF ROSTERS (core equity sleeves, union-only) ===", flush=True)
    for etf in sorted(CORE_EQUITY_ETFS):
        name = name_by_etf.get(etf)
        if not name:
            print(f"  {etf}: not in SECTOR_ETFS, skip", flush=True); continue
        raw = etf_holdings(etf)
        mapped = [t for t in (mapt(s) for s in raw) if t and t not in BLOCKLIST]
        base = set(sh.get_holdings(name, expanded=False))       # curated base only (no overlay) for the diff
        # union: everything the ETF holds now, minus what's already in the curated roster
        add = [t for t in dict.fromkeys(mapped) if t not in base]
        if add:
            overlay[name] = add
            all_new.update(add)
        print(f"  {name[:26]:26} {etf:5} holdings={len(raw):3} mapped={len(mapped):3} NEW={len(add):3}  {add[:8]}", flush=True)

    print(f"\nTotal NEW names across core sleeves: {len(all_new)}", flush=True)
    # which new names lack data (would be inert until fetched)
    have_c = set(Candle.objects.filter(ticker__in=list(all_new), interval="1d").values_list("ticker", flat=True).distinct())
    have_f = set(FinancialReport.objects.filter(ticker__in=list(all_new)).values_list("ticker", flat=True).distinct())
    need_c = sorted(all_new - have_c)
    need_f = sorted(all_new - have_f)
    print(f"  new names missing candles: {len(need_c)} | missing FinancialReport: {len(need_f)}", flush=True)

    if a.fetch and (need_c or need_f):
        from fetch_candles_eodhd import backfill
        from fetch_delisted_fundamentals import _fetch
        if need_c:
            print(f"fetching candles for {len(need_c)} names...", flush=True)
            backfill(need_c, jobs=8)
        if need_f:
            print(f"fetching FinancialReport for {len(need_f)} names...", flush=True)
            got = 0
            for tk in need_f:
                try:
                    r = _fetch(tk, f"{tk}.US")
                except Exception:
                    continue
                if r and r.get("rows"):
                    good = [x for x in r["rows"] if x.get("total_equity") is not None or x.get("net_income") is not None]
                    if good:
                        FinancialReport.objects.filter(ticker=tk).delete()
                        FinancialReport.objects.bulk_create([FinancialReport(**row) for row in good],
                                                            ignore_conflicts=True, batch_size=2000)
                        got += 1
            print(f"  got FinancialReport for {got}/{len(need_f)}", flush=True)

    if a.run:
        # merge with any existing overlay (union-only, never drop) so re-runs accumulate
        prev = {}
        try:
            prev = json.load(open(OVERLAY_PATH))
        except Exception:
            pass
        for name, add in overlay.items():
            merged = list(dict.fromkeys((prev.get(name) or []) + add))
            prev[name] = merged
        json.dump(prev, open(OVERLAY_PATH, "w"), indent=2)
        print(f"\nWROTE {OVERLAY_PATH}: {len(prev)} sleeves, {sum(len(v) for v in prev.values())} overlay names", flush=True)
        try:
            from core.models import BacktestResult
            from django.utils import timezone
            BacktestResult.objects.update_or_create(kind="roster_sync_log", defaults={
                "payload": {"sleeves": {k: v for k, v in overlay.items()}, "n_new": len(all_new),
                            "fetched_candles": a.fetch and len(need_c), "fetched_fund": a.fetch and len(need_f)},
                "computed_at": timezone.now()})
        except Exception as e:
            print("log skipped:", e, flush=True)
    else:
        print("\n(dry-run; pass --run to write overlay, --fetch to pull data)", flush=True)


if __name__ == "__main__":
    main()
