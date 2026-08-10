import os, django
os.environ.setdefault("DJANGO_SETTINGS_MODULE", "rotation.settings")
django.setup()

import numpy as np, pandas as pd
from seq_fundamental_study import build_universe, load_candles, load_fundamentals, load_financial_reports

tks = build_universe()
funds = load_fundamentals(tks)
reports = load_financial_reports(tks)
cd = load_candles(tks[:400])

print("SANITY: PIT market cap (latest shares x latest price) vs current-snapshot market cap")
n_ok = 0
checked = 0
for tk in tks:
    rep = reports.get(tk); df = cd.get(tk)
    cur = (funds.get(tk) or {}).get("market_cap")
    if rep is None or df is None or cur is None or not len(rep):
        continue
    r2 = rep.dropna(subset=["avail_date", "shares_outstanding"]).sort_values("avail_date")
    if not len(r2):
        continue
    sh = r2["shares_outstanding"].to_numpy()[-1]
    pit = sh * df["Close"].values[-1]
    checked += 1
    ratio = pit / cur if cur else 0
    if 0.4 < ratio < 2.5:
        n_ok += 1
    if checked <= 8:
        print(f"  {tk:8} PIT=${pit/1e9:8.2f}B  current=${cur/1e9:8.2f}B  ratio={ratio:.2f}")
print(f"  within 0.4-2.5x of current: {n_ok}/{checked} ({n_ok/max(checked,1)*100:.0f}%) -> units look {'OK' if n_ok/max(checked,1)>0.7 else 'WRONG'}")

print("\nRe-running compute_strategy_forward with PIT market-cap filter...")
from api.tasks import compute_strategy_forward
r = compute_strategy_forward()
for k, m in r["modes"].items():
    print(f"  {k}: n={m['n']} day90 median={m['day_final']['median']}% avg={m['day_final']['avg']}% win={m['day_final']['win']}%")
