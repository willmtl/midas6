#!/usr/bin/env python3
"""Does adding an RSI CROSSOVER (daily or weekly) help the value pick? Our prior work found momentum-
CONFIRMATION entries subtract (ROTATE IN = worst), while oversold DIPS help — so test, don't assume.

For every monthly value pick (rotation top-10 -> cheapest-P/B guarded low-debt), tag 4 crossover signals
measured on the pick's OWN price at the selection date, and report conditional forward-return LIFT
(mean fwd return WITH the signal minus WITHOUT) + hit rates:
  d_cross      daily  RSI(10) crossed above its SMA(10) in the last 5 trading days
  w_cross      weekly RSI(10) crossed above its SMA(10) in the last 2 weeks
  d_cross_os   daily  cross-up AND RSI was oversold (<35) within the last 10 days  (the tail-reversal form)
  w_cross_os   weekly cross-up from oversold
Positive lift -> worth adding (as a conviction point / entry gate). ~0 or negative -> confirms it doesn't
belong. -> BacktestResult[rsi_crossover] + JSON.
Run: MSYS_NO_PATHCONV=1 docker exec rotation-backend-1 python -u /app/rsi_crossover_test.py
"""
import os, json, warnings
warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
os.environ.setdefault("DJANGO_SETTINGS_MODULE", "rotation.settings")
import django
django.setup()

from pathlib import Path
import config, sector_holdings
from seq_fundamental_study import load_candles, load_financial_reports
from trend_stock_studies import _pit_monthly_panel, _available_at, _ret_delist, CRYPTO
from backtest_lowpb import _monthly_close, BENCH
import price_basis

OUT = Path(__file__).resolve().parent / ".data" / "studies" / "rsi_crossover.json"
LOOKBACK, TOP_N = 6, 10


def _rsi(close, n=10):
    d = close.diff()
    up = d.clip(lower=0).rolling(n).mean()
    dn = (-d.clip(upper=0)).rolling(n).mean()
    rs = up / dn.replace(0, np.nan)
    return (100 - 100 / (1 + rs))


def _cross_panels(daily, midx):
    """Per-ticker -> monthly booleans: daily/weekly RSI-crossed-above-signal recently, plain + from-oversold."""
    d_c, w_c, d_co, w_co = {}, {}, {}, {}
    for t, df in daily.items():
        c = df["Close"]
        if len(c) < 40:
            continue
        r = _rsi(c); s = r.rolling(10).mean()
        cu = ((r.shift(1) <= s.shift(1)) & (r > s)).fillna(False)
        os_recent = (r.rolling(10).min() < 35)
        d_c[t] = cu.rolling(5).max().resample("ME").last().reindex(midx).fillna(0).astype(bool)
        d_co[t] = (cu & os_recent).rolling(5).max().resample("ME").last().reindex(midx).fillna(0).astype(bool)
        cw = c.resample("W").last()
        rw = _rsi(cw); sw = rw.rolling(10).mean()
        cuw = ((rw.shift(1) <= sw.shift(1)) & (rw > sw)).fillna(False)
        osw = (rw.rolling(10).min() < 35)
        w_c[t] = cuw.rolling(2).max().resample("ME").last().reindex(midx).fillna(0).astype(bool)
        w_co[t] = (cuw & osw).rolling(2).max().resample("ME").last().reindex(midx).fillna(0).astype(bool)
    F = lambda d: pd.DataFrame(d).reindex(index=midx)
    return {"d_cross": F(d_c), "w_cross": F(w_c), "d_cross_os": F(d_co), "w_cross_os": F(w_co)}


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
    spy_m = etf_daily[BENCH]["Close"].resample("ME").last().reindex(midx)
    etf_trail = etf_m.pct_change(LOOKBACK)
    stock_daily = load_candles(all_holds)
    stock_m = _monthly_close(stock_daily).reindex(midx)

    reps = load_financial_reports(all_holds)
    P = lambda f: _pit_monthly_panel(reps, f, midx)
    shares, equity, ni, debt = P("shares_outstanding"), P("total_equity"), P("net_income"), P("total_debt")
    common = stock_m.columns.intersection(shares.columns).intersection(equity.columns)

    def R(p):
        return p.reindex(index=midx, columns=common)
    px = stock_m[common]
    shares, equity, ni, debt = map(R, (shares, equity, ni, debt))
    pb = (price_basis.as_traded_close(px) * shares) / equity.where(equity != 0)
    trap = (ni < 0) & (~(equity >= equity.shift(12))) & (~(ni > ni.shift(4)))
    low_debt = (debt / equity.where(equity != 0)) < 1.0
    print("building RSI crossover panels...", flush=True)
    panels = _cross_panels({t: d for t, d in stock_daily.items() if t in set(common)}, midx)
    print(f"months {len(midx)} | stocks {len(common)}", flush=True)
    warmup = max(LOOKBACK, 1)

    sig_on = {k: [] for k in panels}          # fwd rets where signal True for the pick
    sig_off = {k: [] for k in panels}
    for i in range(warmup, len(midx) - 1):
        date, ndate = midx[i], midx[i + 1]
        if not np.isfinite(spy_m.iloc[i + 1] / spy_m.iloc[i] - 1):
            continue
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
            r = _ret_delist(px[pick], date, ndate)
            if r is None or not np.isfinite(r):
                continue
            for k, pan in panels.items():
                on = bool(pan.loc[date, pick]) if (pick in pan.columns and pd.notna(pan.loc[date, pick])) else False
                (sig_on[k] if on else sig_off[k]).append(float(r))

    results = {}
    for k in panels:
        on, off = np.array(sig_on[k]), np.array(sig_off[k])
        lift = (on.mean() - off.mean()) * 100 if len(on) and len(off) else None
        results[k] = {"n_on": len(on), "n_off": len(off),
                      "avg_on_%": round(on.mean() * 100, 2) if len(on) else None,
                      "avg_off_%": round(off.mean() * 100, 2) if len(off) else None,
                      "lift_pp": round(lift, 2) if lift is not None else None,
                      "hit_on_%": round((on > 0).mean() * 100, 1) if len(on) else None}
    print("\n=== RSI CROSSOVER — conditional forward-return lift on the value pick ===", flush=True)
    for k, v in results.items():
        print(f"  {k:11} on n={v['n_on']:>4} avg {str(v['avg_on_%']):>6}%  | off avg {str(v['avg_off_%']):>6}%  "
              f"| LIFT {str(v['lift_pp']):>6}pp  hit {v['hit_on_%']}%", flush=True)
    best = max(results, key=lambda k: results[k]["lift_pp"] if results[k]["lift_pp"] is not None else -99)
    helps = results[best]["lift_pp"] is not None and results[best]["lift_pp"] > 1.0

    payload = {
        "computed_at": pd.Timestamp.utcnow().isoformat(),
        "params": {"lookback_months": LOOKBACK, "top_n": TOP_N, "benchmark": BENCH, "months": int(len(midx))},
        "results": results, "best": best, "helps": bool(helps),
        "verdict": (f"{best} adds +{results[best]['lift_pp']}pp/pick — worth adding as a conviction point / entry "
                    "signal." if helps else "No RSI-crossover variant adds meaningful lift (>1pp) to the value "
                    "pick — confirms momentum-confirmation entries don't help; the oversold DIP (RSI<45) remains "
                    "the entry edge, not the crossover."),
        "caveat": "RSI(10) vs its own SMA(10) signal line. In-sample, no fees, ~5y. Conditional lift per pick "
                  "(not a full portfolio) — a signal that helps per-pick can still shrink the book if used as a gate.",
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
            kind="rsi_crossover", defaults={"payload": json.loads(json.dumps(p, default=str)),
                                            "computed_at": timezone.now()})
        print("Saved BacktestResult[rsi_crossover]", flush=True)
    except Exception as e:
        print("DB save failed:", e, flush=True)
    print("\n" + p["verdict"], flush=True)


if __name__ == "__main__":
    main()
