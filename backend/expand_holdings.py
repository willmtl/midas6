#!/usr/bin/env python3
"""EXPAND UNIVERSE step 2a: for every EQUITY sector, pull the full current constituent list from EODHD
(ETF_Data.Holdings, US names only), union with our hardcoded top-20, and write the NEW tickers to
/app/.data/expanded_holdings.json. Sizes the expansion and produces the merge file that get_holdings()
will union in. Read-only vs the DB (writes one JSON). Backfill of candles/fundamentals for the new
names is the next step.
Run: MSYS_NO_PATHCONV=1 docker exec rotation-backend-1 python -u /app/expand_holdings.py [--limit N]
"""
import os, sys, json, warnings
warnings.filterwarnings("ignore")
os.environ.setdefault("DJANGO_SETTINGS_MODULE", "rotation.settings")
import django
django.setup()

from pathlib import Path
import sector_holdings
from api.tasks import _eodhd_get

OUT = Path("/app/.data/expanded_holdings.json")
TOP = 60          # keep top-N constituents by weight per ETF
limit = int(sys.argv[sys.argv.index("--limit") + 1]) if "--limit" in sys.argv else None


def eod_to_yahoo(code, exch):
    # US sector ETFs hold US names; EODHD "BRK.B" -> yahoo "BRK-B". Keep US only.
    if exch and exch.upper() not in ("US", "NYSE", "NASDAQ", "BATS", "NYSE ARCA", "AMEX", "NYSE MKT"):
        return None
    return code.replace(".", "-").upper()


expanded, tot_new, tot_full = {}, 0, 0
items = list(sector_holdings.HOLDINGS.items())
if limit:
    items = items[:limit]

for name, meta in items:
    etf = meta.get("etf")
    ours = [t for t in meta.get("holdings", []) if t]
    if not etf or not ours:              # skip commodity/bond/crypto sleeves (no equity holdings)
        continue
    data = _eodhd_get(f"fundamentals/{etf}.US")
    if not isinstance(data, dict) or not isinstance(data.get("ETF_Data"), dict):
        continue
    hd = data["ETF_Data"].get("Holdings") or {}
    rows = []
    for k, v in hd.items():
        try:
            w = float(v.get("Assets_%", 0) or 0)
        except (TypeError, ValueError):
            w = 0.0
        y = eod_to_yahoo(v.get("Code") or k.split(".")[0], v.get("Exchange") or "")
        if y:
            rows.append((y, w))
    rows.sort(key=lambda x: -x[1])
    full = [y for y, _ in rows[:TOP]]
    ours_up = {t.replace("-", "-").upper() for t in ours}
    added = [y for y in full if y not in ours_up and y not in {t.upper() for t in ours}]
    if not full:
        continue
    expanded[name] = {"etf": etf, "existing": len(ours), "eodhd_seen": len(hd),
                      "added": added, "full_union": sorted(set([t.upper() for t in ours]) | set(full))}
    tot_new += len(added); tot_full += len(expanded[name]["full_union"])
    print(f"  {name:26} {etf:6} ours {len(ours):>2}  eodhd {len(hd):>3}  +{len(added):>2} new", flush=True)

OUT.parent.mkdir(parents=True, exist_ok=True)
OUT.write_text(json.dumps(expanded, indent=1))
uniq_new = sorted({t for s in expanded.values() for t in s["added"]})
print(f"\nequity sectors expanded: {len(expanded)} | total new (with dupes): {tot_new} | "
      f"UNIQUE new tickers: {len(uniq_new)}", flush=True)
print("wrote", OUT, flush=True)
