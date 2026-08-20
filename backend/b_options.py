#!/usr/bin/env python3
"""Does trading FLAGSHIP-B via OPTIONS beat the stock? B = capitulation (seq_rsi20_ad_rising_rsi, $5M floor)
-> gap-up entry -> exit 3 bars (~2 calendar days), +1.96%/68% on big-cap liquid names. Compare, on B's actual
trades: STOCK vs LONG CALL (long the tail) vs BULL CALL SPREAD with the short strike placed by HOW BIG WE THINK
THE MOVE IS — K_short = entry*(1 + m*sigma_hold), sigma_hold = atm_iv*sqrt(hold) (the IV-implied move over the
hold), swept over m. Black-Scholes-priced from PIT atm_iv (OptionSnapshot, 2022-09+), with a weekly-option
spread haircut and an IV-crush scenario (bounce -> vol comes in, hurts long premium).
Run: MSYS_NO_PATHCONV=1 docker exec rotation-backend-1 python -u /app/b_options.py"""
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

T_ENTRY, HOLD_DAYS = 7 / 365.0, 2 / 365.0        # buy a ~weekly; hold ~2 calendar days (3 four-hour bars)
T_EXIT = T_ENTRY - HOLD_DAYS
M_GRID = [0.5, 1.0, 1.5, 2.0]                    # bull-spread cap = m * IV-implied hold-move
HAIRCUTS = [0.0, 0.03, 0.05]                     # per-leg half-spread (fraction of premium)
IV_SCEN = {"iv_flat": 1.0, "iv_crush": 0.85}     # exit IV multiplier


def bs_call(Sp, K, T, iv):
    if T <= 0 or iv <= 0 or Sp <= 0 or K <= 0:
        return max(Sp - K, 0.0)
    d1 = (math.log(Sp / K) + 0.5 * iv * iv * T) / (iv * math.sqrt(T))
    d2 = d1 - iv * math.sqrt(T)
    nd1 = 0.5 * (1 + math.erf(d1 / math.sqrt(2)))
    nd2 = 0.5 * (1 + math.erf(d2 / math.sqrt(2)))
    return Sp * nd1 - K * nd2


def lc_ret(S0, S1, iv0, iv1, K, hc, Tent, Text):
    c0 = bs_call(S0, K, Tent, iv0); c1 = bs_call(S1, K, Text, iv1)
    cost = c0 * (1 + hc)
    if cost <= 0:
        return None
    return (c1 * (1 - hc) - cost) / cost


def bs_ret(S0, S1, iv0, iv1, Klong, Kshort, hc, Tent, Text):
    l0 = bs_call(S0, Klong, Tent, iv0); s0 = bs_call(S0, Kshort, Tent, iv0)
    l1 = bs_call(S1, Klong, Text, iv1);  s1 = bs_call(S1, Kshort, Text, iv1)
    debit = l0 * (1 + hc) - s0 * (1 - hc)              # buy long@ask, sell short@bid
    if debit <= 0:
        return None
    value = l1 * (1 - hc) - s1 * (1 + hc)              # sell long@bid, buy back short@ask
    return (value - debit) / debit


# ---- options as-of store (atm_iv) ----
allowed, meta = S.candidate_windows("B_plus")
names = sorted(allowed)
opt = {}
for r in OptionSnapshot.objects.filter(ticker__in=names).values("ticker", "date", "atm_iv").order_by("ticker", "date"):
    rec = opt.setdefault(r["ticker"], ([], []))
    rec[0].append(r["date"]); rec[1].append(r["atm_iv"])


def iv_asof(tk, d):
    rec = opt.get(tk)
    if not rec:
        return None
    i = bisect.bisect_right(rec[0], d) - 1
    return rec[1][i] if i >= 0 else None


# ---- collect B trades with IV coverage ----
trades = []      # (S0, S1, iv0)
n_names = covered = n_fire = 0
for tk in names:
    df = get_4h(tk, 5, False)
    if df is None or len(df) < 120:
        continue
    n_names += 1
    entry, _m = H.SIGNALS["mo_gap_up"]["fn"](df)
    c = df["Close"].values; ts = df.index; n = len(c)
    ad = allowed[tk]
    cov = tk in opt
    if cov:
        covered += 1
    for i in sorted(H._episode_starts([j for j in range(n) if entry[j] and ts[j].date() in ad], gap=H.GAP)):
        if i + 3 >= n or c[i] <= 0:
            continue
        n_fire += 1
        iv0 = iv_asof(tk, ts[i].date())
        if iv0 is None or iv0 <= 0:
            continue
        trades.append((float(c[i]), float(c[i + 3]), float(iv0) / 100.0))   # atm_iv stored in PERCENT -> decimal

print(f"B_plus names={meta['n_names']} with-4h={n_names} opt-covered={covered}; B fires={n_fire} with-IV={len(trades)}", flush=True)
if trades:
    ivs = np.array([t[2] for t in trades])
    sig_hold = ivs * math.sqrt(HOLD_DAYS)
    print(f"atm_iv median {np.median(ivs)*100:.0f}%; IV-implied 1σ hold-move median {np.median(sig_hold)*100:.1f}% "
          f"(so 1σ cap ≈ entry×(1+{np.median(sig_hold)*100:.1f}%))\n", flush=True)


def summ(rets):
    a = np.array([r for r in rets if r is not None], float)
    if not len(a):
        return "  (none)"
    return f"n={len(a):>4} avg {a.mean()*100:+7.1f}%  med {np.median(a)*100:+6.1f}%  win {(a>0).mean()*100:4.0f}%  wipeout {(a<=-0.99).mean()*100:3.0f}%"


stock = [(S1 - S0) / S0 for (S0, S1, _iv) in trades]
print(f"  STOCK (2-day hold)   {summ(stock)}", flush=True)

# ===== DTE SWEEP: buy 1-2mo out, HOLD ~2 days, sell — vs the 7-DTE weekly (worst-case theta) =====
DTE_GRID = [7, 30, 45, 60]     # calendar days to expiry AT ENTRY; hold = HOLD_DAYS then sell
HC = 0.03                       # central spread haircut (monthlies are TIGHTER than weeklies in reality)
for scen, mult in IV_SCEN.items():
    print(f"\n===== exit-IV: {scen} (exit iv={mult}×), spread haircut {HC*100:.0f}%/leg =====", flush=True)
    for dte in DTE_GRID:
        Tent = dte / 365.0; Text = (dte - HOLD_DAYS * 365) / 365.0
        lc_atm = [lc_ret(S0, S1, iv, iv * mult, S0, HC, Tent, Text) for (S0, S1, iv) in trades]
        lc_itm = [lc_ret(S0, S1, iv, iv * mult, 0.95 * S0, HC, Tent, Text) for (S0, S1, iv) in trades]
        print(f"  --- {dte:>2}-DTE ---", flush=True)
        print(f"    LONG CALL atm      {summ(lc_atm)}", flush=True)
        print(f"    LONG CALL 5%-ITM   {summ(lc_itm)}", flush=True)
        for m in (1.0, 2.0):
            rets = [bs_ret(S0, S1, iv, iv * mult, S0, S0 * (1 + m * iv * math.sqrt(HOLD_DAYS)), HC, Tent, Text)
                    for (S0, S1, iv) in trades]
            print(f"    BULL SPREAD m={m:>3}σ    {summ(rets)}", flush=True)
