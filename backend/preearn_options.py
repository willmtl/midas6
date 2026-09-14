#!/usr/bin/env python3
"""PRE-EARNINGS RUN-UP via OPTIONS + portfolio sizing. 'How much can we make using options with the run-up?'
The play: on option-liquid names ($ADV>=LIQ_MIN), buy ~35-DTE ATM call at close of day t-10, SELL at close t-1
(the day BEFORE the earnings print). This rides the pre-earnings drift AND the IV RAMP UP into earnings (long
vega while vol rises), and EXITS before the IV crush + the binary gap. First-pass pricing: Black-Scholes from
PIT OptionSnapshot.atm_iv (as-of entry AND exit dates -> captures the ramp), per-leg spread haircut. Also a
bull-call-spread variant (caps upside, cuts theta/vega cost). Then a PORTFOLIO sim: each event sized at f% of
NAV, trades in date order, overlap allowed up to full deployment -> equity curve vs SPY. Coverage limited to
OptionSnapshot (2022-09+). Confirm later with real OPRA contract prices.
Run: MSYS_NO_PATHCONV=1 docker exec rotation-backend-1 python -u /app/preearn_options.py"""
import os, sys, json, math, bisect
sys.path.insert(0, "/app")
os.environ.setdefault("DJANGO_SETTINGS_MODULE", "rotation.settings")
import django; django.setup()
import numpy as np, pandas as pd
from collections import defaultdict
from pathlib import Path
from seq_fundamental_study import load_candles, MIN_BARS, _chunk
from signal_discovery import _universe, PRICE_FLOOR

K_RUNUP = 10
LIQ_MIN = 100e6                 # option-liquid: >=$100M/day 20d avg dollar volume
DTE = 35                        # days-to-expiry of the call at entry (monthly bracketing earnings)
HAIRCUT = 0.03                  # per-leg half-spread (fraction of premium) for liquid names
OTM_SPREAD = 0.07               # bull-call-spread short strike = entry*(1+OTM_SPREAD)
F_GRID = [0.02, 0.05, 0.10]     # % of NAV risked per position (portfolio sizing sweep)


def bs_call(Sp, K, T, iv):
    if T <= 0 or iv <= 0 or Sp <= 0 or K <= 0:
        return max(Sp - K, 0.0)
    d1 = (math.log(Sp / K) + 0.5 * iv * iv * T) / (iv * math.sqrt(T))
    d2 = d1 - iv * math.sqrt(T)
    nd1 = 0.5 * (1 + math.erf(d1 / math.sqrt(2))); nd2 = 0.5 * (1 + math.erf(d2 / math.sqrt(2)))
    return Sp * nd1 - K * nd2


def call_ret(S0, S1, iv0, iv1, K, Tent, Text):
    c0 = bs_call(S0, K, Tent, iv0); c1 = bs_call(S1, K, Text, iv1)
    cost = c0 * (1 + HAIRCUT)
    return None if cost <= 0 else (c1 * (1 - HAIRCUT) - cost) / cost


def spread_ret(S0, S1, iv0, iv1, Kl, Ks, Tent, Text):
    l0 = bs_call(S0, Kl, Tent, iv0); s0 = bs_call(S0, Ks, Tent, iv0)
    l1 = bs_call(S1, Kl, Text, iv1); s1 = bs_call(S1, Ks, Text, iv1)
    debit = l0 * (1 + HAIRCUT) - s0 * (1 - HAIRCUT)
    return None if debit <= 0 else (l1 * (1 - HAIRCUT) - s1 * (1 + HAIRCUT) - debit) / debit


def main():
    universe, delisted = _universe()
    from core.models import EarningsEvent, OptionSnapshot, Candle
    have_e = set(EarningsEvent.objects.values_list("ticker", flat=True).distinct())
    have_o = set(OptionSnapshot.objects.filter(atm_iv__isnull=False).values_list("ticker", flat=True).distinct())
    names = [t for t in universe if t in have_e and t in have_o]
    print(f"names with earnings dates AND option IV: {len(names)}", flush=True)

    # IV as-of store (atm_iv stored in PERCENT)
    ivs = defaultdict(lambda: ([], []))
    for r in OptionSnapshot.objects.filter(ticker__in=names, atm_iv__isnull=False).values(
            "ticker", "date", "atm_iv").order_by("ticker", "date"):
        ivs[r["ticker"]][0].append(r["date"]); ivs[r["ticker"]][1].append(r["atm_iv"] / 100.0)

    def iv_asof(tk, d):
        rec = ivs.get(tk)
        if not rec or not rec[0]:
            return None
        i = bisect.bisect_right(rec[0], d) - 1
        return rec[1][i] if i >= 0 else None

    edates = defaultdict(list)
    for tk, rd in EarningsEvent.objects.filter(ticker__in=names).values_list("ticker", "report_date"):
        edates[tk].append(pd.Timestamp(rd))

    spy = pd.DataFrame(list(Candle.objects.filter(ticker="SPY").values_list("date", "close")), columns=["d", "c"])
    spy["d"] = pd.to_datetime(spy["d"]); spy_c = spy.set_index("d")["c"].astype(float).sort_index()

    trades = []   # dict(entry_date, exit_date, stock_r, call_r, spread_r)
    done = 0
    for ch in _chunk(names, 40):
        candles = load_candles(ch)
        for tk, sdf in candles.items():
            if len(sdf) < MIN_BARS:
                continue
            dates = sdf.index; close = sdf["Close"].values; vol = sdf["Volume"].values; n = len(close)
            dvol20 = pd.Series(close * vol).rolling(20, min_periods=10).mean().values
            di = dates.values
            for rd in edates.get(tk, []):
                t = int(np.searchsorted(di, np.datetime64(rd), side="left"))
                if t >= n:
                    continue
                a = t - K_RUNUP; b = t - 1
                if a < 0 or b <= a:
                    continue
                S0 = float(close[a]); S1 = float(close[b])
                if S0 <= PRICE_FLOOR or not np.isfinite(S0) or not np.isfinite(S1):
                    continue
                if not (np.isfinite(dvol20[a]) and dvol20[a] >= LIQ_MIN):
                    continue
                da = dates[a].date(); db = dates[b].date()
                iv0 = iv_asof(tk, da); iv1 = iv_asof(tk, db)
                if iv0 is None or iv1 is None:
                    continue
                cal_gap = (dates[b] - dates[a]).days
                Tent = DTE / 365.0; Text = max(DTE - cal_gap, 1) / 365.0
                cr = call_ret(S0, S1, iv0, iv1, S0, Tent, Text)                 # ATM call, K=S0
                sr = spread_ret(S0, S1, iv0, iv1, S0, S0 * (1 + OTM_SPREAD), Tent, Text)
                if cr is None:
                    continue
                trades.append(dict(entry=dates[a], exit=dates[b], stock=(S1 - S0) / S0,
                                   call=cr, spread=sr if sr is not None else np.nan, iv0=iv0, iv1=iv1))
        done += len(ch)
        print(f"  scanned {done}/{len(names)}  trades {len(trades)}", flush=True)

    tf = pd.DataFrame(trades).sort_values("entry").reset_index(drop=True)
    print(f"\nTOTAL run-up option trades (2022-09+, $ADV>={LIQ_MIN/1e6:.0f}M): {len(tf)}", flush=True)
    print(f"  window {tf['entry'].min().date()} .. {tf['exit'].max().date()}  |  IV ramp mean "
          f"{ (tf['iv1']-tf['iv0']).mean()*100:+.2f} vol pts (entry->exit)", flush=True)

    def desc(col):
        a = tf[col].dropna().values * 100
        return dict(n=len(a), mean=round(a.mean(), 2), med=round(float(np.median(a)), 2),
                    win=round((a > 0).mean() * 100, 1), p10=round(float(np.percentile(a, 10)), 1),
                    p90=round(float(np.percentile(a, 90)), 1))
    print("\n=== PER-TRADE returns (%/trade, spread-haircut incl.) ===", flush=True)
    for col in ("stock", "call", "spread"):
        d = desc(col)
        print(f"  {col:8} n{d['n']:>6}  mean {d['mean']:+7.2f}%  med {d['med']:+6.2f}%  win {d['win']:4.1f}%  "
              f"p10 {d['p10']:+6.1f}  p90 {d['p90']:+6.1f}", flush=True)

    # ---- PORTFOLIO sim: size each trade at f% of NAV, date-ordered, overlap up to full deployment ----
    def portfolio(col, f):
        nav = 1.0; deployed = 0.0
        open_pos = []      # (exit_date, pnl_at_exit_fraction_of_nav_at_entry_scaled)
        curve = {}; last = None
        # process by day across union of entry/exit dates
        all_days = sorted(set(tf["entry"]) | set(tf["exit"]))
        ti = 0; tf_sorted = tf.sort_values("entry").reset_index(drop=True)
        by_entry = defaultdict(list)
        for _, row in tf_sorted.iterrows():
            r = row[col]
            if not np.isfinite(r):
                continue
            by_entry[row["entry"]].append((row["exit"], r))
        open_book = []     # list of (exit_date, stake, r)
        for day in all_days:
            # close exits first
            still = []
            for exitd, stake, r in open_book:
                if exitd <= day:
                    nav += stake * r
                    deployed -= stake
                else:
                    still.append((exitd, stake, r))
            open_book = still
            # open new (cap deployment at 100% of NAV)
            for exitd, r in by_entry.get(day, []):
                stake = f * nav
                if deployed + stake > nav:      # portfolio-management cap: never lever past 1x
                    continue
                deployed += stake
                open_book.append((exitd, stake, r))
            curve[day] = nav
        s = pd.Series(curve).sort_index()
        dret = s.pct_change().dropna()
        if len(dret) < 30:
            return None
        yrs = (s.index[-1] - s.index[0]).days / 365.25
        cagr = (s.iloc[-1] ** (1 / yrs) - 1) * 100 if s.iloc[-1] > 0 else -100
        eq = s.values; dd = (eq / np.maximum.accumulate(eq) - 1).min() * 100
        sh = dret.mean() / dret.std() * math.sqrt(252) if dret.std() > 0 else 0
        return dict(f=f, total=round((s.iloc[-1] - 1) * 100, 1), cagr=round(cagr, 1),
                    maxdd=round(dd, 1), sharpe=round(sh, 2), start=str(s.index[0].date()), end=str(s.index[-1].date()))

    # SPY same window
    w0, w1 = tf["entry"].min(), tf["exit"].max()
    spy_w = spy_c[(spy_c.index >= w0) & (spy_c.index <= w1)]
    spy_tot = (spy_w.iloc[-1] / spy_w.iloc[0] - 1) * 100
    spy_cagr = ((spy_w.iloc[-1] / spy_w.iloc[0]) ** (365.25 / (spy_w.index[-1] - spy_w.index[0]).days) - 1) * 100
    print(f"\n  SPY buy-hold same window: total {spy_tot:+.1f}%  CAGR {spy_cagr:+.1f}%", flush=True)

    out = {"per_trade": {c: desc(c) for c in ("stock", "call", "spread")},
           "spy_same_window": dict(total=round(spy_tot, 1), cagr=round(spy_cagr, 1)), "portfolio": {}}
    print("\n=== PORTFOLIO equity curve (size f% NAV/trade, capped at 1x deployment) ===", flush=True)
    for col in ("call", "spread"):
        print(f"  -- instrument: {col} --", flush=True)
        for f in F_GRID:
            p = portfolio(col, f)
            if not p:
                continue
            out["portfolio"][f"{col}|f{int(f*100)}"] = p
            print(f"    f={int(f*100):>3}%  total {p['total']:+9.1f}%  CAGR {p['cagr']:+7.1f}%  "
                  f"maxDD {p['maxdd']:+6.1f}%  Sharpe {p['sharpe']:.2f}  ({p['start']}..{p['end']})", flush=True)

    payload = dict(computed_at=pd.Timestamp.utcnow().isoformat(), method="BSM from OptionSnapshot atm_iv (as-of "
                   "entry+exit), first-pass", k_runup=K_RUNUP, liq_min=LIQ_MIN, dte=DTE, haircut=HAIRCUT,
                   otm_spread=OTM_SPREAD, results=out,
                   caveat="First-pass: BSM-priced from stored ATM IV (2022-09+ only, ~4yr). Captures IV ramp via "
                   "as-of entry+exit IV. Confirm with real OPRA contract fill prices for the definitive number.")
    Path("/app/.data/studies/preearn_options.json").write_text(json.dumps(payload, default=str))
    try:
        from core.models import BacktestResult
        from django.utils import timezone
        BacktestResult.objects.update_or_create(kind="preearn_options",
            defaults={"payload": json.loads(json.dumps(payload, default=str)), "computed_at": timezone.now()})
        print("\nSaved BacktestResult[preearn_options]", flush=True)
    except Exception as e:
        print("DB save failed:", e, flush=True)


if __name__ == "__main__":
    main()
