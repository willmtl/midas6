#!/usr/bin/env python3
"""Call RATIO BACKSPREAD on B/A — the last untested aligned options structure (long the tail, unlike the bull
spread). Sell 1 call at K1=ATM, buy 2 at K2=entry*(1+m*sigma). Profits on a BIG up move; small loss if flat;
max loss at K2. Hypothesis: our ~2% move is TOO SMALL to reach the payoff zone (the short ATM loses before the
2 OTM longs kick in). P&L reported as fraction of underlying (comparable to the stock's move). BS-priced from
PIT atm_iv, 7 & 45 DTE, IV flat + crush. Run: MSYS_NO_PATHCONV=1 docker exec rotation-backend-1 python -u /app/b_backspread.py"""
import os, math, bisect, warnings
warnings.filterwarnings("ignore")
os.environ.setdefault("DJANGO_SETTINGS_MODULE", "rotation.settings")
import django
django.setup()
import numpy as np
import h4_study as H
import h4_on_signals_study as S
from intraday_data import get_4h
from core.models import OptionSnapshot

HOLD_DAYS = 2 / 365.0
HC = 0.03


def bs_call(Sp, K, T, iv):
    if T <= 0 or iv <= 0 or Sp <= 0 or K <= 0:
        return max(Sp - K, 0.0)
    d1 = (math.log(Sp / K) + 0.5 * iv * iv * T) / (iv * math.sqrt(T)); d2 = d1 - iv * math.sqrt(T)
    return Sp * 0.5 * (1 + math.erf(d1 / math.sqrt(2))) - K * 0.5 * (1 + math.erf(d2 / math.sqrt(2)))


allowed, meta = S.candidate_windows("B_plus")
names = sorted(allowed)
opt = {}
for r in OptionSnapshot.objects.filter(ticker__in=names).values("ticker", "date", "atm_iv").order_by("ticker", "date"):
    rec = opt.setdefault(r["ticker"], ([], [])); rec[0].append(r["date"]); rec[1].append(r["atm_iv"])


def iv_asof(tk, d):
    rec = opt.get(tk)
    if not rec:
        return None
    i = bisect.bisect_right(rec[0], d) - 1
    return rec[1][i] if i >= 0 else None


trades = []
for tk in names:
    df = get_4h(tk, 5, False)
    if df is None or len(df) < 120:
        continue
    entry, _m = H.SIGNALS["mo_gap_up"]["fn"](df)
    c = df["Close"].values; ts = df.index; n = len(c)
    ad = allowed[tk]
    for i in sorted(H._episode_starts([j for j in range(n) if entry[j] and ts[j].date() in ad], gap=H.GAP)):
        if i + 3 >= n or c[i] <= 0:
            continue
        iv0 = iv_asof(tk, ts[i].date())
        if iv0 and iv0 > 0:
            trades.append((float(c[i]), float(c[i + 3]), float(iv0) / 100.0))
print(f"B trades with IV: {len(trades)}; stock avg { (np.mean([(s1-s0)/s0 for s0,s1,_ in trades])*100):+.2f}%\n", flush=True)


def backspread_pnl(S0, S1, iv0, iv1, K2, Tent, Text):
    # sell 1 ATM, buy 2 at K2 ; P&L per underlying share, incl haircut (sell@bid/buy@ask in, reverse out)
    a0 = bs_call(S0, S0, Tent, iv0); b0 = bs_call(S0, K2, Tent, iv0)
    a1 = bs_call(S1, S0, Text, iv1); b1 = bs_call(S1, K2, Text, iv1)
    entry_cost = 2 * b0 * (1 + HC) - a0 * (1 - HC)     # buy 2 longs@ask, sell 1 short@bid
    exit_val = 2 * b1 * (1 - HC) - a1 * (1 + HC)       # sell 2 longs@bid, buy back short@ask
    return (exit_val - entry_cost) / S0 * 100          # as % of underlying, comparable to stock move


def summ(a):
    a = np.array([x for x in a if x is not None], float)
    return f"n={len(a):>4} avg {a.mean():+.2f}%  med {np.median(a):+.2f}%  win {(a>0).mean()*100:.0f}%" if len(a) else "(none)"


for scen, mult in (("iv_flat", 1.0), ("iv_crush", 0.85)):
    print(f"===== {scen} (exit iv={mult}×) =====", flush=True)
    for dte in (7, 45):
        Tent = dte / 365.0; Text = (dte - HOLD_DAYS * 365) / 365.0
        for m in (1.0, 1.5, 2.0):
            pnl = [backspread_pnl(s0, s1, iv, iv * mult, s0 * (1 + m * iv * math.sqrt(HOLD_DAYS)), Tent, Text)
                   for s0, s1, iv in trades]
            print(f"  {dte:>2}-DTE  K2=+{m}σ   {summ(pnl)}", flush=True)
    print("", flush=True)
