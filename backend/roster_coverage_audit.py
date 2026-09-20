#!/usr/bin/env python3
"""ROSTER COVERAGE AUDIT (user: "what else can we add?"). For EVERY SECTOR_ETFS sleeve, report how many roster names
are actually PICKABLE = US/CA-listed (flagship country gate: US [no dot] or .TO/.V) AND have candles AND have PIT
FinancialReport (value-rankable). Surfaces the same gap we just fixed on metals: empty rosters, foreign-only rosters,
or rosters whose tickers lack data while US variants exist. Flags each sleeve by TYPE so we can tell a legit fixable
equity sector from a KNOWN-REFUTED one (commodity-futures / foreign-country adds all lose — see memory). No writes.
Run: MSYS_NO_PATHCONV=1 docker exec rotation-backend-1 python -u /app/roster_coverage_audit.py"""
import os
os.environ.setdefault("DJANGO_SETTINGS_MODULE", "rotation.settings")
import django; django.setup()
import config, sector_holdings as sh
from core.models import Candle, FinancialReport


def usca(tk):
    return ("." not in tk) or tk.rsplit(".", 1)[1] in ("TO", "V")


def main():
    have_c = set(Candle.objects.values_list("ticker", flat=True).distinct())
    have_f = set(FinancialReport.objects.values_list("ticker", flat=True).distinct())
    rows = []
    for name, etf in config.SECTOR_ETFS.items():
        try:
            hold = sh.get_holdings(name) or []
        except Exception:
            hold = []
        n = len(hold)
        n_usca = sum(1 for t in hold if usca(t))
        pickable = [t for t in hold if usca(t) and t in have_c and t in have_f]
        rows.append((name, etf, n, n_usca, len(pickable), pickable[:6]))
    # sort: worst coverage first (0 pickable = the gaps)
    rows.sort(key=lambda r: (r[4], r[3]))
    print(f"{'sleeve':>30} {'etf':>6} {'roster':>6} {'usca':>4} {'PICKABLE':>8}  sample", flush=True)
    for name, etf, n, nu, npick, samp in rows:
        flag = " <-- GAP" if npick == 0 else ("" if npick >= 3 else " <-- thin")
        print(f"{name[:30]:>30} {etf:>6} {n:>6} {nu:>4} {npick:>8}  {','.join(samp)}{flag}", flush=True)
    gaps = [r for r in rows if r[4] == 0]
    thin = [r for r in rows if 0 < r[4] < 3]
    print(f"\nGAP sleeves (0 pickable): {len(gaps)}", flush=True)
    print(f"THIN sleeves (1-2 pickable): {len(thin)}", flush=True)


if __name__ == "__main__":
    main()
