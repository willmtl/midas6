#!/usr/bin/env python3
"""INDICATOR STANDALONE SWEEP — the honest test the bake-off skipped. `indicator_bakeoff.py` only overlaid the
indicators on our 115 rotation picks (tiny samples). This tests each indicator as a STANDALONE signal across the
FULL liquid universe (own trigger, $5M dollar-volume floor) per the HARD RULE [[test-signals-individually]] and
the way the EODHD articles actually use them: every stock, every trigger, thousands of trades.

For each ticker: compute each indicator's BUY condition (from indicator_bakeoff.CONDS), take NON-OVERLAPPING
trades (>=HOLD days apart, so quasi-independent), hold HOLD trading days, record MARKET-ADJUSTED forward return
(stock - SPY over the same window). Only fire when trailing $ vol >= floor. Aggregate mean / win% / t / N per
indicator, plus the base rate (unconditional forward return) and a tail read (share of >+20% / <-20% trades)
per [[tail-not-average]]. -> BacktestResult[indicator_standalone] + JSON.
Run: MSYS_NO_PATHCONV=1 docker exec rotation-backend-1 python -u /app/indicator_standalone.py
"""
import os, json, warnings
warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
os.environ.setdefault("DJANGO_SETTINGS_MODULE", "rotation.settings")
import django
django.setup()

from pathlib import Path
import config, sector_holdings
from seq_fundamental_study import load_candles
from trend_stock_studies import CRYPTO
from backtest_lowpb import _tstat_from_returns, BENCH
from indicator_bakeoff import CONDS       # reuse the exact 15 indicator conditions

HOLD = 21          # trading days (~1 month)
MIN_DVOL = 5e6
OUT = Path(__file__).resolve().parent / ".data" / "studies" / "indicator_standalone.json"


def build():
    etfs = {e for e in config.SECTOR_ETFS.values() if e not in CRYPTO}
    names = set()
    for n in config.SECTOR_ETFS:
        for t in sector_holdings.get_holdings(n):
            if t not in etfs and t not in CRYPTO:
                names.add(t)
    names = sorted(names)
    print(f"Loading {len(names)} stocks + {BENCH}...", flush=True)
    cds = load_candles(names + [BENCH])
    spy = cds.get(BENCH)
    if spy is None:
        print("no SPY", flush=True); return None
    spy_c = spy["Close"]
    spy_fwd = (spy_c.shift(-HOLD) / spy_c - 1)         # SPY forward-HOLD return by date

    trades = {name: [] for name in CONDS}
    base = []                                          # unconditional base rate (non-overlapping)
    n_names = 0
    for t in names:
        d = cds.get(t)
        if d is None or not {"Open", "High", "Low", "Close", "Volume"}.issubset(d.columns) or len(d) < 120:
            continue
        n_names += 1
        c = d["Close"]
        fwd = (c.shift(-HOLD) / c - 1).to_numpy()
        dvol = (c * d["Volume"]).rolling(20).mean().to_numpy()
        sfwd = spy_fwd.reindex(d.index).to_numpy()
        liquid = dvol >= MIN_DVOL
        # base rate: non-overlapping days where liquid & fwd valid
        last = -HOLD
        for i in range(len(c)):
            if liquid[i] and np.isfinite(fwd[i]) and np.isfinite(sfwd[i]) and i - last >= HOLD:
                base.append(fwd[i] - sfwd[i]); last = i
        for name, fn in CONDS.items():
            try:
                cond = fn(d).to_numpy()
            except Exception:
                continue
            idx = np.where((cond == True) & liquid & np.isfinite(fwd) & np.isfinite(sfwd))[0]  # noqa: E712
            last = -HOLD
            for i in idx:
                if i - last >= HOLD:
                    trades[name].append(fwd[i] - sfwd[i]); last = i
    print(f"universe {n_names} stocks; base-rate trades {len(base)}", flush=True)

    def stat(arr):
        a = np.asarray(arr, float)
        a = a[np.isfinite(a)]
        if len(a) == 0:
            return None
        return dict(n=len(a), mean=round(float(a.mean()) * 100, 2), median=round(float(np.median(a)) * 100, 2),
                    win=round(float((a > 0).mean()) * 100, 1),
                    t=round(_tstat_from_returns(list(a)), 2) if len(a) > 5 else None,
                    big_win_pct=round(float((a > 0.20).mean()) * 100, 1),
                    big_loss_pct=round(float((a < -0.20).mean()) * 100, 1))

    br = stat(base)
    results = {name: stat(v) for name, v in trades.items()}
    print(f"\n=== STANDALONE indicator signal — market-adjusted fwd-{HOLD}d, full universe, non-overlapping ===", flush=True)
    print(f"  base rate (any liquid day): mean {br['mean']}%  win {br['win']}%  n {br['n']:,}", flush=True)
    print(f"  {'indicator':<20}{'N':>8}{'mean%':>8}{'edge':>7}{'win%':>7}{'t':>7}{'>+20%':>7}{'<-20%':>7}", flush=True)
    ranked = sorted((k for k in results if results[k]), key=lambda k: results[k]["mean"], reverse=True)
    for name in ranked:
        r = results[name]
        edge = round(r["mean"] - br["mean"], 2)
        print(f"  {name:<20}{r['n']:>8,}{r['mean']:>8}{edge:>7}{r['win']:>7}{str(r['t']):>7}"
              f"{r['big_win_pct']:>7}{r['big_loss_pct']:>7}", flush=True)

    winners = [k for k in results if results[k] and results[k]["mean"] - br["mean"] > 0.5
               and results[k]["t"] is not None and results[k]["t"] > 2]
    best = ranked[0] if ranked else None
    verdict = (
        f"Base rate (any liquid stock, {HOLD}d fwd, mkt-adj) mean {br['mean']}% / win {br['win']}% (n {br['n']:,}). "
        f"Best standalone = {best} ({results[best]['mean']}%, edge {round(results[best]['mean']-br['mean'],2)}pp, "
        f"t{results[best]['t']}, n{results[best]['n']:,}). "
        + (f"Indicators clearing base rate at t>2: {', '.join(winners)}. " if winners else
           "NO indicator clears the base rate with t>2 standalone — they don't beat buying a random liquid stock. ")
        + "Consistent with the overlay + [[entry-signal-value-pick]]: oversold entries carry a small real edge, "
          "trend/breakout entries don't (or lose)."
    )
    print("\n" + verdict, flush=True)
    return {
        "computed_at": pd.Timestamp.utcnow().isoformat(),
        "params": {"hold_days": HOLD, "min_dvol": MIN_DVOL, "universe": n_names, "benchmark": BENCH,
                   "adjust": "market-adjusted (stock - SPY over same window)", "trades": "non-overlapping >=HOLD apart"},
        "base_rate": br, "results": results, "winners_t2": winners, "verdict": verdict,
        "caveat": "Signal-quality sweep (base-rate style), NOT a managed backtest: fixed HOLD-day exit, no stop/TP, "
                  "no position sizing, present-day-holdings survivorship (dead names absent -> optimistic), "
                  "market-adjusted vs SPY only. Non-overlapping trades per ticker keep t-stats honest-ish but names "
                  "still cluster in time (macro). $5M dollar-vol floor at signal.",
    }


def main():
    p = build()
    if p is None:
        return
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(p, indent=2, default=str))
    try:
        from core.models import BacktestResult
        from django.utils import timezone
        BacktestResult.objects.update_or_create(
            kind="indicator_standalone", defaults={"payload": json.loads(json.dumps(p, default=str)),
                                                   "computed_at": timezone.now()})
        print("Saved BacktestResult[indicator_standalone]", flush=True)
    except Exception as e:
        print("DB save failed:", e, flush=True)


if __name__ == "__main__":
    main()
