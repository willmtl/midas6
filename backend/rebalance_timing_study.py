#!/usr/bin/env python3
"""BEST REBALANCE DAY — does WHEN in the month you switch matter? First-of-month vs mid-month vs month-end?
Same accel-sector value engine, only the monthly rebalance ANCHOR changes: the trading day nearest the
1st / 8th / 15th / 22nd / month-end. Reports vs SPY / Sharpe / DD for each anchor. A 'turn-of-the-month'
effect (or any consistent best day) shows up as a spread across anchors.
-> BacktestResult[rebalance_timing] + JSON.
Run: MSYS_NO_PATHCONV=1 docker exec rotation-backend-1 python -u /app/rebalance_timing_study.py
"""
import os, json, warnings
warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
os.environ.setdefault("DJANGO_SETTINGS_MODULE", "rotation.settings")
import django
django.setup()

from pathlib import Path
import config, sector_holdings, price_basis
from studies import _tstat_from_returns
from seq_fundamental_study import load_candles, load_financial_reports
from trend_stock_studies import _pit_monthly_panel, _available_at, _ret_delist, CRYPTO
from backtest_lowpb import BENCH

OUT = Path(__file__).resolve().parent / ".data" / "studies" / "rebalance_timing.json"
LOOKBACK, TOP_N = 6, 10
ANCHORS = {"day~1 (start)": 1, "day~8": 8, "day~15 (mid)": 15, "day~22": 22, "month-end": 99}


def _stats(r, spy):
    r = np.asarray(r, float); n = len(r)
    if n == 0:
        return dict(vs_spy=0, sharpe=0, max_drawdown=0, t_stat=None, periods=0)
    tot = float(np.prod(1 + r) - 1) * 100
    sp = float(np.prod(1 + np.asarray(spy)) - 1) * 100
    sh = float(r.mean() / r.std() * np.sqrt(12)) if r.std() > 1e-9 else 0.0
    eq = np.cumprod(1 + r); dd = float(((eq / np.maximum.accumulate(eq)) - 1).min() * 100)
    t = _tstat_from_returns(list(r))
    return dict(vs_spy=round(tot - sp, 1), sharpe=round(sh, 2), max_drawdown=round(dd, 1),
                t_stat=round(t, 2) if t is not None else None, periods=n)


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
    stock_daily = load_candles(all_holds)
    # daily close panels
    etf_close = pd.DataFrame({t: etf_daily[t]["Close"] for t in etf_tk if t in etf_daily}).sort_index()
    stock_close = pd.DataFrame({t: stock_daily[t]["Close"] for t in all_holds if t in stock_daily}).sort_index()
    spy_close = etf_daily[BENCH]["Close"]
    di = spy_close.dropna().index                         # trading-day calendar
    reps = load_financial_reports(all_holds)

    def to_monthly(c, D):
        """Per-ticker: one value per month = close on the trading day nearest day-of-month D. D>=28 = last
        (== resample('ME').last(), the validated month-end); D<=1 = first. Labelled by month-start so all
        tickers align; delisting handled naturally (ticker's series ends -> months stop)."""
        out = {}
        for p, s in c.groupby(c.index.to_period("M")):
            s = s.dropna()
            if not len(s):
                continue
            if D >= 28:
                v = s.iloc[-1]; lbl = p.to_timestamp("M")          # month-end calendar label (== _monthly_close)
            elif D <= 1:
                v = s.iloc[0]; lbl = p.to_timestamp()               # month-start
            else:
                tgt = pd.Timestamp(p.year, p.month, min(D, 28))
                i = int(np.argmin(np.abs((s.index.values - tgt.to_datetime64()).astype("timedelta64[D]").astype(int))))
                v = s.iloc[i]; lbl = tgt                            # label = the Dth, so PIT aligns with the price
            out[lbl] = float(v)
        return pd.Series(out).sort_index()

    def run(D):
        etf_m = pd.DataFrame({t: to_monthly(etf_daily[t]["Close"], D) for t in etf_tk if t in etf_daily}).sort_index()
        midx = etf_m.index
        spy_m = to_monthly(spy_close, D).reindex(midx)
        stk_m = pd.DataFrame({t: to_monthly(stock_daily[t]["Close"], D) for t in all_holds
                              if t in stock_daily and len(stock_daily[t]) > 60}).reindex(midx)   # match _monthly_close min-history
        etf_accel = etf_m.pct_change(3) - etf_m.pct_change(3).shift(3)
        shares = _pit_monthly_panel(reps, "shares_outstanding", midx)
        equity = _pit_monthly_panel(reps, "total_equity", midx)
        ni = _pit_monthly_panel(reps, "net_income", midx)
        debt = _pit_monthly_panel(reps, "total_debt", midx)
        common = stk_m.columns.intersection(shares.columns).intersection(equity.columns)
        px = stk_m[common]
        R = lambda p: p.reindex(index=midx, columns=common)
        shares, equity, ni, debt = R(shares), R(equity), R(ni), R(debt)
        pb = (price_basis.as_traded_close(px) * shares) / equity.where(equity != 0)
        trap = (ni < 0) & (~(equity >= equity.shift(12))) & (~(ni > ni.shift(4)))
        low_debt = (debt / equity.where(equity != 0)) < 1.0
        rets, spies = [], []
        for i in range(9, len(midx) - 1):
            date, ndate = midx[i], midx[i + 1]
            sp = spy_m.iloc[i + 1] / spy_m.iloc[i] - 1
            if not np.isfinite(sp):
                continue
            ranks = etf_accel.loc[date].dropna().sort_values(ascending=False).head(TOP_N).index
            slot = []
            for etf in ranks:
                _, holds = sector_map.get(etf, (etf, []))
                cands = [h for h in holds if h in px.columns and _available_at(px[h], date)
                         and pd.notna(pb.loc[date, h]) and pb.loc[date, h] > 0 and not bool(trap.loc[date, h])]
                guarded = [c for c in cands if bool(low_debt.loc[date, c])] or cands
                if not guarded:
                    continue
                r = _ret_delist(px[pb.loc[date, guarded].idxmin()], date, ndate)
                if r is not None and np.isfinite(r):
                    slot.append(float(r))
            if slot:
                rets.append(float(np.mean(slot))); spies.append(float(sp))
        return _stats(rets, spies)

    results = {}
    for label, D in ANCHORS.items():
        results[label] = run(D)
        print(f"  {label:15} vsSPY {results[label]['vs_spy']:>7}%  Sh {results[label]['sharpe']}  "
              f"DD {results[label]['max_drawdown']}%  t={results[label]['t_stat']}", flush=True)

    best = max(results, key=lambda k: results[k]["vs_spy"])
    worst = min(results, key=lambda k: results[k]["vs_spy"])
    spread = round(results[best]["vs_spy"] - results[worst]["vs_spy"], 1)
    payload = {
        "computed_at": pd.Timestamp.utcnow().isoformat(),
        "params": {"top_n": TOP_N, "benchmark": BENCH, "anchors": list(ANCHORS.keys())},
        "results": results, "best": best, "worst": worst, "spread_pp": spread,
        "verdict": (f"Best rebalance = {best} (+{results[best]['vs_spy']}%), worst = {worst} (+{results[worst]['vs_spy']}%), "
                    f"spread {spread}pp. " + ("Rebalance timing MATTERS — a consistent best window exists." if spread > 60
                    else "Rebalance day barely matters (spread small / noise) — pick a consistent day and stick to it; "
                    "the edge is the selection, not the calendar. First-of-month vs mid-month is a wash.")),
        "caveat": "In-sample, no fees, ~5y, 9mo warmup. Anchors = trading day nearest the target day-of-month. "
                  "5 anchors tested -> small spreads are noise (multiple-comparisons).",
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
            kind="rebalance_timing", defaults={"payload": json.loads(json.dumps(p, default=str)),
                                               "computed_at": timezone.now()})
        print("Saved BacktestResult[rebalance_timing]", flush=True)
    except Exception as e:
        print("DB save failed:", e, flush=True)
    print("\n" + p["verdict"], flush=True)


if __name__ == "__main__":
    main()
