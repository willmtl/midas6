"""Scan the whole individual-stock universe for the CURRENT A/D-line state on the
latest bar, and flag which stocks are also firing a capitulation signal
(new_52low / rsi_oversold20) — the setups where accum-divergence tripled the edge.

Run: docker compose exec -T backend python -u ad_state_now.py
"""
import os
import django
os.environ.setdefault("DJANGO_SETTINGS_MODULE", "rotation.settings")
django.setup()

import pandas as pd  # noqa: E402
from seq_fundamental_study import build_universe, load_candles  # noqa: E402
from pit_fundamentals import _ad_state, bucket_ad  # noqa: E402
from studies import SIGNALS  # noqa: E402

CAPIT = ["new_52low", "rsi_oversold20"]
RECENT = 10  # bars

tickers = build_universe()
print(f"Universe: {len(tickers)} stocks\n")
candles = load_candles(tickers)

rows = []
for tk, df in candles.items():
    if len(df) < 60:
        continue
    state = _ad_state(df)
    if len(state) == 0 or pd.isna(state.iloc[-1]):
        continue
    label = bucket_ad(float(state.iloc[-1]))
    firing = []
    for sk in CAPIT:
        try:
            sig = SIGNALS[sk][1](df).fillna(False)
        except Exception:
            continue
        recent = sig.iloc[-RECENT:].tolist()
        if any(recent):
            days_ago = next(i for i, v in enumerate(reversed(recent)) if v)
            firing.append(f"{sk}({days_ago}d)")
    rows.append((tk, label, round(float(df["Close"].iloc[-1]), 2), ", ".join(firing)))

accum = [r for r in rows if r[1] == "accum divergence"]
print(f"IN 'accum divergence' RIGHT NOW: {len(accum)} of {len(rows)} scanned\n")

# The actionable subset: accum-divergence AND firing a capitulation signal.
hot = sorted([r for r in accum if r[3]], key=lambda r: r[0])
print(f"=== ACCUM-DIVERGENCE + firing a capitulation signal (the edge): {len(hot)} ===")
for tk, lbl, px, fir in hot:
    print(f"  {tk:8s} ${px:<10} {fir}")

print(f"\n=== All {len(accum)} in accum-divergence (tickers only) ===")
print(", ".join(sorted(r[0] for r in accum)))
