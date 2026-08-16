#!/usr/bin/env python3
"""Does a PROFITABILITY GUARD improve the cheapest-P/B value pick? Test the value-trap thesis:
cheap P/B is only a bargain if book equity is real and durable; a money-loser's book erodes
(melting ice cube), so `cheapest positive P/B` can buy a falling knife.

Same selection as arm3_lowpb (top trailing-6mo-momentum sectors -> within each, a value pick from
the positive-P/B candidates) but vary the FILTER applied to the candidate set before choosing the
cheapest (all PIT from FinancialReport net_income / revenue / total_equity):

  baseline            positive P/B only (current strategy)
  profitable          + net_income > 0
  book_stable         + total_equity >= its value ~1yr ago (book not shrinking)
  profitable_book     + net_income > 0 AND book not shrinking (both guards)
  ex_trap             DROP only the trap: net_income < 0 AND book shrinking (keeps turnarounds:
                      unprofitable-but-book-growing names stay eligible)
  ex_trap_turn        ex_trap, but ALSO keep an unprofitable name if its net_income is IMPROVING
                      (turnaround) even if book slipped

If a sector has no candidate passing the filter, the slot is dropped (pure value picks). We also
report each variant WITH ETF fallback (empty sector -> hold the ETF) for the realistic number.
-> BacktestResult[profitability_guard] + JSON.
Run: MSYS_NO_PATHCONV=1 docker exec rotation-backend-1 python -u /app/profitability_guard_study.py
     (--limit 150 quick subset)
"""
import os, sys, json, warnings
warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
os.environ.setdefault("DJANGO_SETTINGS_MODULE", "rotation.settings")
import django
django.setup()

from pathlib import Path
import config, sector_holdings
import price_basis
from studies import _tstat_from_returns
from seq_fundamental_study import load_candles, load_financial_reports
from trend_stock_studies import _pit_monthly_panel, _available_at, _ret_delist, CRYPTO
from backtest_lowpb import _monthly_close, BENCH

OUT = Path(__file__).resolve().parent / ".data" / "studies" / "profitability_guard.json"
LOOKBACK, TOP_N = 6, 10
VARIANTS = ["baseline", "profitable", "book_stable", "profitable_book", "ex_trap", "ex_trap_turn"]


def _stats(rets, spy_rets):
    r = np.array(rets, float); n = len(r)
    if n == 0:
        return {"total_return": 0, "vs_spy": 0, "sharpe": 0, "max_drawdown": 0, "t_stat": None, "periods": 0}
    total = float(np.prod(1 + r) - 1) * 100
    spy = float(np.prod(1 + np.array(spy_rets)) - 1) * 100
    sharpe = float(r.mean() / r.std() * np.sqrt(12)) if r.std() > 1e-9 else 0.0
    eq = np.cumprod(1 + r); dd = float(((eq / np.maximum.accumulate(eq)) - 1).min() * 100)
    t = _tstat_from_returns(list(r))
    return {"total_return": round(total, 1), "vs_spy": round(total - spy, 1), "sharpe": round(sharpe, 2),
            "max_drawdown": round(dd, 1), "t_stat": round(t, 2) if t is not None else None, "periods": n}


def build():
    etfs = {n: e for n, e in config.SECTOR_ETFS.items() if e not in CRYPTO}
    limit = int(sys.argv[sys.argv.index("--limit") + 1]) if "--limit" in sys.argv else None
    sector_map, all_holds = {}, set()
    for n, e in etfs.items():
        h = [t for t in sector_holdings.get_holdings(n) if t not in (e, BENCH) and t not in CRYPTO]
        sector_map[e] = (n, h); all_holds.update(h)
    all_holds = sorted(all_holds)
    if limit:
        all_holds = all_holds[:limit]; hs = set(all_holds)
        sector_map = {e: (n, [h for h in hh if h in hs]) for e, (n, hh) in sector_map.items()}

    etf_tk = list(etfs.values())
    print(f"Loading {len(etf_tk)} ETFs + {len(all_holds)} stocks + {BENCH}...", flush=True)
    etf_daily = load_candles(etf_tk + [BENCH])
    etf_m = _monthly_close({t: d for t, d in etf_daily.items() if t in etf_tk})
    midx = etf_m.index
    spy_m = etf_daily[BENCH]["Close"].resample("ME").last().reindex(midx)
    etf_trail = etf_m.pct_change(LOOKBACK)
    stock_daily = load_candles(all_holds)
    stock_m = _monthly_close(stock_daily).reindex(midx)

    reps = load_financial_reports(all_holds)
    shares = _pit_monthly_panel(reps, "shares_outstanding", midx)
    equity = _pit_monthly_panel(reps, "total_equity", midx)
    ni = _pit_monthly_panel(reps, "net_income", midx)
    rev = _pit_monthly_panel(reps, "revenue", midx)
    common = stock_m.columns.intersection(shares.columns).intersection(equity.columns)
    pb = (price_basis.as_traded_close(stock_m[common]) * shares[common]) / equity[common].where(equity[common] != 0)
    # PIT guards (aligned to pb columns/index)
    ni = ni.reindex(index=midx, columns=common)
    equity = equity.reindex(index=midx, columns=common)
    book_stable = equity >= equity.shift(12)                 # book not shrinking vs ~1yr ago
    profitable = ni > 0
    ni_improving = ni > ni.shift(4)                          # net income better than ~2 quarters ago
    trap = (ni < 0) & (~book_stable)                         # unprofitable AND book shrinking

    print(f"months {len(midx)} | stocks {stock_m.shape[1]} | pb {pb.shape}", flush=True)
    warmup = max(LOOKBACK, 1)

    def _candidates(holds, date):
        return [h for h in holds if h in stock_m.columns and _available_at(stock_m[h], date)
                and h in pb.columns and pd.notna(pb.loc[date, h]) and pb.loc[date, h] > 0]

    def _pass(variant, date, t):
        if variant == "baseline":
            return True
        prof = bool(profitable.loc[date, t]) if t in profitable.columns else False
        stab = bool(book_stable.loc[date, t]) if t in book_stable.columns else False
        tr = bool(trap.loc[date, t]) if t in trap.columns else False
        imp = bool(ni_improving.loc[date, t]) if t in ni_improving.columns else False
        if variant == "profitable":
            return prof
        if variant == "book_stable":
            return stab
        if variant == "profitable_book":
            return prof and stab
        if variant == "ex_trap":
            return not tr
        if variant == "ex_trap_turn":
            return (not tr) or imp
        return True

    def _pick(variant, holds, date):
        cands = [t for t in _candidates(holds, date) if _pass(variant, date, t)]
        if not cands:
            return None
        row = pb.loc[date, cands]
        return row.idxmin()

    def _etf_ret(etf, date, ndate):
        if etf in etf_m.columns:
            r = _ret_delist(etf_m[etf], date, ndate)
            return r if r is not None and np.isfinite(r) else None
        return None

    def _run(variant, fallback):
        rets, spies, dropped = [], [], 0
        for i in range(warmup, len(midx) - 1):
            date, ndate = midx[i], midx[i + 1]
            sp = spy_m.iloc[i + 1] / spy_m.iloc[i] - 1
            if not np.isfinite(sp):
                continue
            ranks = etf_trail.loc[date].dropna().sort_values(ascending=False).head(TOP_N).index
            slot = []
            for etf in ranks:
                _, holds = sector_map.get(etf, (etf, []))
                pk = _pick(variant, holds, date)
                if pk is not None:
                    r = _ret_delist(stock_m[pk], date, ndate)
                    if r is not None and np.isfinite(r):
                        slot.append(float(r)); continue
                if fallback:
                    er = _etf_ret(etf, date, ndate)
                    if er is not None:
                        slot.append(er); continue
                dropped += 1
            if slot:
                rets.append(float(np.mean(slot))); spies.append(float(sp))
        s = _stats(rets, spies); s["dropped_slots"] = dropped
        return s

    results = {}
    for v in VARIANTS:
        results[v] = {"no_fallback": _run(v, False), "fallback": _run(v, True)}
        print(f"  {v:16} no-fb vsSPY {results[v]['no_fallback']['vs_spy']:>7}%  "
              f"t={results[v]['no_fallback']['t_stat']}  Sh {results[v]['no_fallback']['sharpe']}  "
              f"DD {results[v]['no_fallback']['max_drawdown']}%  | fb vsSPY {results[v]['fallback']['vs_spy']}%", flush=True)

    base = results["baseline"]["no_fallback"]
    for v in VARIANTS:
        results[v]["vs_baseline_vs_spy"] = round(results[v]["no_fallback"]["vs_spy"] - base["vs_spy"], 1)
    # best guard = highest no-fallback vs_spy among the guards (not baseline)
    guards = [v for v in VARIANTS if v != "baseline"]
    best = max(guards, key=lambda v: results[v]["no_fallback"]["vs_spy"])
    helps = results[best]["no_fallback"]["vs_spy"] > base["vs_spy"]

    payload = {
        "computed_at": pd.Timestamp.utcnow().isoformat(),
        "params": {"lookback_months": LOOKBACK, "top_n_sectors": TOP_N, "benchmark": BENCH,
                   "months": int(len(midx)), "limit": limit,
                   "note": "same arm3_lowpb selection; profitability filter varies the candidate set (PIT)"},
        "variants": results, "baseline": "baseline", "best_guard": best, "guard_helps": bool(helps),
        "recommendation": (
            f"Best guard '{best}' {'BEATS' if helps else 'does NOT beat'} the unguarded baseline "
            f"({results[best]['no_fallback']['vs_spy']}% vs {base['vs_spy']}% vs SPY, no-fallback). "
            "Guards drop cheap-but-unprofitable/eroding-book value traps; ex_trap_turn keeps genuine "
            "negative-EPS turnarounds. If a guard beats baseline on vs-SPY AND drawdown, wire it live."),
        "caveat": ("PIT from FinancialReport (net_income/total_equity, avail_date-gated). Directional/"
                   "no-fees; ~5y single regime. 'book_stable' uses total_equity vs ~12mo prior; "
                   "'ni_improving' vs ~2 quarters prior (rough turnaround proxy)."),
    }
    return payload


def main():
    payload = build()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(payload, indent=2, default=str))
    try:
        from core.models import BacktestResult
        from django.utils import timezone
        BacktestResult.objects.update_or_create(
            kind="profitability_guard",
            defaults={"payload": json.loads(json.dumps(payload, default=str)), "computed_at": timezone.now()})
        print("Saved BacktestResult[profitability_guard]", flush=True)
    except Exception as e:
        print("DB save failed:", e, flush=True)
    print("\n" + payload["recommendation"], flush=True)


if __name__ == "__main__":
    main()
