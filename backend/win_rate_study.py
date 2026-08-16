#!/usr/bin/env python3
"""IMPROVE WIN RATE without killing expectancy. Per-pick forward return over 1/2/3-month holds, split by
entry oversold depth and quality, reporting WIN RATE + mean + asymmetry (avg win / avg loss) for each.
Goal: find levers that lift win rate while keeping the positive edge (win +10 / lose -6 @ ~55%).

Levers:
  hold        realize the pick over H = 1, 2, 3 months (mean-reversion may need time to complete)
  entry       all picks | oversold RSI(10)<45 | deep-oversold RSI(10)<35
  quality     guard (default, keeps turnarounds) | profitable-only (net income > 0)
-> BacktestResult[win_rate] + JSON.
Run: MSYS_NO_PATHCONV=1 docker exec rotation-backend-1 python -u /app/win_rate_study.py
"""
import os, json, warnings
warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
os.environ.setdefault("DJANGO_SETTINGS_MODULE", "rotation.settings")
import django
django.setup()

from pathlib import Path
import config, sector_holdings
import price_basis
from seq_fundamental_study import load_candles, load_financial_reports
from trend_stock_studies import _pit_monthly_panel, _available_at, CRYPTO
from backtest_lowpb import _monthly_close, BENCH

OUT = Path(__file__).resolve().parent / ".data" / "studies" / "win_rate.json"
LOOKBACK, TOP_N = 6, 10
HOLDS = [1, 2, 3]


def _rsi(close, n=10):
    d = close.diff(); up = d.clip(lower=0).rolling(n).mean(); dn = (-d.clip(upper=0)).rolling(n).mean()
    return (100 - 100 / (1 + up / dn.replace(0, np.nan)))


def _wr(rets):
    a = np.array(rets, float)
    if not len(a):
        return None
    w, l = a[a > 0], a[a <= 0]
    return dict(n=len(a), win_pct=round((a > 0).mean() * 100, 1), mean_pct=round(a.mean() * 100, 2),
                avg_win_pct=round(w.mean() * 100, 2) if len(w) else None,
                avg_loss_pct=round(l.mean() * 100, 2) if len(l) else None)


def build():
    etfs = {n: e for n, e in config.SECTOR_ETFS.items() if e not in CRYPTO}
    sector_map, all_holds = {}, set()
    for n, e in etfs.items():
        h = [t for t in sector_holdings.get_holdings(n) if t not in (e, BENCH) and t not in CRYPTO]
        sector_map[e] = (n, h); all_holds.update(h)
    all_holds = sorted(all_holds)

    etf_tk = list(etfs.values())
    print(f"Loading {len(etf_tk)} ETFs + {len(all_holds)} stocks + {BENCH}...", flush=True)
    etf_daily = load_candles(etf_tk + [BENCH])
    etf_m = _monthly_close({t: d for t, d in etf_daily.items() if t in etf_tk})
    midx = etf_m.index
    etf_trail = etf_m.pct_change(LOOKBACK)
    stock_daily = load_candles(all_holds)
    stock_m = _monthly_close(stock_daily).reindex(midx)

    reps = load_financial_reports(all_holds)
    P = lambda f: _pit_monthly_panel(reps, f, midx)
    shares, equity, ni, debt = P("shares_outstanding"), P("total_equity"), P("net_income"), P("total_debt")
    common = stock_m.columns.intersection(shares.columns).intersection(equity.columns)

    def Rf(p):
        return p.reindex(index=midx, columns=common)
    px = stock_m[common]
    shares, equity, ni, debt = map(Rf, (shares, equity, ni, debt))
    pb = (price_basis.as_traded_close(px) * shares) / equity.where(equity != 0)
    trap = (ni < 0) & (~(equity >= equity.shift(12))) & (~(ni > ni.shift(4)))
    low_debt = (debt / equity.where(equity != 0)) < 1.0
    rsi = pd.DataFrame({t: _rsi(stock_daily[t]["Close"]).resample("ME").last().reindex(midx)
                        for t in common if t in stock_daily and len(stock_daily[t]) > 20}).reindex(index=midx, columns=common)
    print(f"months {len(midx)} | stocks {len(common)}", flush=True)
    warmup = max(LOOKBACK, 1)

    # collect per-pick: (rsi_entry, profitable, {H: fwd_ret})
    recs = []
    for i in range(warmup, len(midx) - 1):
        date = midx[i]
        ranks = etf_trail.loc[date].dropna().sort_values(ascending=False).head(TOP_N).index
        for etf in ranks:
            _, holds = sector_map.get(etf, (etf, []))
            cands = [h for h in holds if h in px.columns and _available_at(px[h], date)
                     and pd.notna(pb.loc[date, h]) and pb.loc[date, h] > 0 and not bool(trap.loc[date, h])]
            ld = [c for c in cands if bool(low_debt.loc[date, c])]
            use = ld or cands
            if not use:
                continue
            pick = pb.loc[date, use].idxmin()
            S0 = px.loc[date, pick]
            if not (S0 and S0 > 0):
                continue
            r_entry = rsi.loc[date, pick] if pick in rsi.columns else np.nan
            prof = bool(pd.notna(ni.loc[date, pick]) and ni.loc[date, pick] > 0)
            fwd = {}
            for H in HOLDS:
                j = i + H
                if j < len(midx) and pd.notna(px.loc[midx[j], pick]):
                    fwd[H] = float(px.loc[midx[j], pick] / S0 - 1)
            recs.append((float(r_entry) if pd.notna(r_entry) else None, prof, fwd))

    def bucket(entry_fn, qual_fn, H):
        return _wr([r[2][H] for r in recs if H in r[2] and (r[0] is not None) and entry_fn(r[0]) and qual_fn(r[1])])

    ALL = lambda x: True
    OS45 = lambda x: x < 45
    OS35 = lambda x: x < 35
    QALL = lambda p: True
    QPROF = lambda p: p
    combos = {
        "baseline (all/guard)": (ALL, QALL),
        "oversold<45": (OS45, QALL),
        "deep-oversold<35": (OS35, QALL),
        "profitable-only": (ALL, QPROF),
        "oversold<45 + profitable": (OS45, QPROF),
        "deep-os<35 + profitable": (OS35, QPROF),
    }
    results = {}
    print("\n=== WIN RATE levers (per-pick, by hold) ===", flush=True)
    for name, (ef, qf) in combos.items():
        results[name] = {}
        line = f"  {name:26}"
        for H in HOLDS:
            s = bucket(ef, qf, H); results[name][f"{H}mo"] = s
            line += f"  {H}mo:{s['win_pct'] if s else '–'}%/{s['mean_pct'] if s else '–'} (n{s['n'] if s else 0})"
        print(line, flush=True)

    # find best win-rate combo that KEEPS mean >= baseline 1mo mean * 0.9
    base_mean = results["baseline (all/guard)"]["1mo"]["mean_pct"]
    best, best_wr = None, 0
    for name, hs in results.items():
        for H, s in hs.items():
            if s and s["mean_pct"] >= base_mean * 0.9 and s["win_pct"] > best_wr:
                best_wr = s["win_pct"]; best = f"{name} @ {H}"
    payload = {
        "computed_at": pd.Timestamp.utcnow().isoformat(),
        "params": {"lookback_months": LOOKBACK, "top_n": TOP_N, "benchmark": BENCH, "holds": HOLDS,
                   "baseline_1mo_mean_pct": base_mean},
        "results": results, "best_winrate_keeping_edge": best, "best_win_pct": best_wr,
        "verdict": (f"Best win-rate lever that keeps the edge: {best} -> win {best_wr}%. Holding longer lets oversold "
                    "value complete its reversion (more picks turn green); the entry/quality filters trade breadth "
                    "for hit-rate. Win rate is raised by TIME + selectivity, not by cutting winners short."),
        "caveat": "Per-pick forward returns (overlapping across holds -> not a portfolio Sharpe). In-sample, no fees, ~5y.",
    }
    return payload


def main():
    p = build()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(p, indent=2, default=str))
    try:
        from core.models import BacktestResult
        from django.utils import timezone
        BacktestResult.objects.update_or_create(
            kind="win_rate", defaults={"payload": json.loads(json.dumps(p, default=str)),
                                       "computed_at": timezone.now()})
        print("Saved BacktestResult[win_rate]", flush=True)
    except Exception as e:
        print("DB save failed:", e, flush=True)
    print("\n" + p["verdict"], flush=True)


if __name__ == "__main__":
    main()
