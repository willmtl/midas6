#!/usr/bin/env python3
"""IS STOCK-LEVEL ACCELERATION a signal? Sector acceleration validated (+422%). Now test the SAME idea on
individual stocks: accel = (stock 3mo return) minus (its prior 3mo return). Two questions:

  A. STANDALONE — does buying high-acceleration stocks beat SPY? (pure, +quality guard+low_debt, +value tilt)
  B. ON THE VALUE PICK — does the rotation value pick do BETTER when the picked stock is itself accelerating,
     or does it HURT (momentum-confirmation vs the oversold-value 'buy weakness' thesis, like RSI crossover)?

Monthly, PIT, vs SPY. -> BacktestResult[stock_acceleration] + JSON.
Run: MSYS_NO_PATHCONV=1 docker exec rotation-backend-1 python -u /app/stock_acceleration_study.py
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
from studies import _tstat_from_returns
from seq_fundamental_study import load_candles, load_financial_reports
from trend_stock_studies import _pit_monthly_panel, _available_at, _ret_delist, CRYPTO
from backtest_lowpb import _monthly_close, BENCH

OUT = Path(__file__).resolve().parent / ".data" / "studies" / "stock_acceleration.json"
LOOKBACK, TOP_N, TOP_STK = 6, 10, 20


def _stats(r, spy):
    r = np.asarray(r, float); n = len(r)
    if n == 0:
        return dict(total_return=0, vs_spy=0, sharpe=0, max_drawdown=0, t_stat=None, win_pct=0, periods=0)
    tot = float(np.prod(1 + r) - 1) * 100
    sp = float(np.prod(1 + np.asarray(spy)) - 1) * 100
    sh = float(r.mean() / r.std() * np.sqrt(12)) if r.std() > 1e-9 else 0.0
    eq = np.cumprod(1 + r); dd = float(((eq / np.maximum.accumulate(eq)) - 1).min() * 100)
    t = _tstat_from_returns(list(r))
    return dict(total_return=round(tot, 1), vs_spy=round(tot - sp, 1), sharpe=round(sh, 2),
                max_drawdown=round(dd, 1), t_stat=round(t, 2) if t is not None else None, periods=n)


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
    accel = etf_m.pct_change(3) - etf_m.pct_change(3).shift(3)     # sector momentum ACCELERATION (flagship)
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
    stock_accel = px.pct_change(3) - px.pct_change(3).shift(3)     # stock momentum ACCELERATION
    dvol = {}                                                      # $5M dollar-volume floor (flagship)
    for t in common:
        d = stock_daily.get(t)
        if d is None or "Volume" not in d or len(d) < 90:
            continue
        dvol[t] = (d["Close"] * d["Volume"]).rolling(20).mean().resample("ME").last().reindex(midx)
    dvol = pd.DataFrame(dvol).reindex(index=midx, columns=common)
    print(f"months {len(midx)} | stocks {len(common)}", flush=True)
    warmup = 9

    # ---- A. standalone stock-acceleration strategies ----
    def run_standalone(mode):
        rets, spies = [], []
        for i in range(warmup, len(midx) - 1):
            date, ndate = midx[i], midx[i + 1]
            sp = spy_m.iloc[i + 1] / spy_m.iloc[i] - 1
            if not np.isfinite(sp):
                continue
            ok = [c for c in common if _available_at(px[c], date) and pd.notna(stock_accel.loc[date, c])]
            if mode in ("quality", "value"):
                ok = [c for c in ok if pd.notna(pb.loc[date, c]) and pb.loc[date, c] > 0 and not bool(trap.loc[date, c])
                      and bool(low_debt.loc[date, c])
                      and pd.notna(dvol.loc[date, c]) and dvol.loc[date, c] >= 5e6]
            if mode == "value":                       # cheapest-P/B among ACCELERATING quality stocks
                acc = [c for c in ok if stock_accel.loc[date, c] > 0]
                picks = list(pb.loc[date, acc].nsmallest(TOP_STK).index) if acc else []
            else:                                     # top by acceleration (pure / quality)
                picks = list(stock_accel.loc[date, ok].nlargest(TOP_STK).index)
            slot = [_ret_delist(px[p], date, ndate) for p in picks]
            slot = [float(x) for x in slot if x is not None and np.isfinite(x)]
            if slot:
                rets.append(float(np.mean(slot))); spies.append(float(sp))
        return _stats(rets, spies)

    standalone = {m: run_standalone(m) for m in ("pure", "quality", "value")}

    # ---- B. conditional lift on the rotation value pick (accelerating stock vs fading) ----
    on, off = [], []
    base_pick = []
    for i in range(warmup, len(midx) - 1):
        date, ndate = midx[i], midx[i + 1]
        if not np.isfinite(spy_m.iloc[i + 1] / spy_m.iloc[i] - 1):
            continue
        ranks = accel.loc[date].dropna().sort_values(ascending=False).head(TOP_N).index
        for etf in ranks:
            _, holds = sector_map.get(etf, (etf, []))
            cands = [h for h in holds if h in px.columns and _available_at(px[h], date)
                     and pd.notna(pb.loc[date, h]) and pb.loc[date, h] > 0 and not bool(trap.loc[date, h])
                     and pd.notna(dvol.loc[date, h]) and dvol.loc[date, h] >= 5e6]
            ld = [c for c in cands if bool(low_debt.loc[date, c])]
            use = ld or cands
            if not use:
                continue
            pick = pb.loc[date, use].idxmin()
            r = _ret_delist(px[pick], date, ndate)
            if r is None or not np.isfinite(r):
                continue
            r = float(r); base_pick.append(r)
            a = stock_accel.loc[date, pick]
            if pd.notna(a):
                (on if a > 0 else off).append(r)

    def _lift(x):
        a = np.array(x, float)
        return None if not len(a) else dict(n=len(a), mean_pct=round(a.mean() * 100, 2),
                                            win_pct=round((a > 0).mean() * 100, 1))
    cond = {"pick_accelerating": _lift(on), "pick_fading": _lift(off), "all_picks": _lift(base_pick)}
    lift = (cond["pick_accelerating"]["mean_pct"] - cond["pick_fading"]["mean_pct"]) if (on and off) else None

    print("\n=== A. STANDALONE stock acceleration (vs SPY) ===", flush=True)
    for m, s in standalone.items():
        print(f"  {m:9} vsSPY {s['vs_spy']:>7}%  Sh {s['sharpe']}  DD {s['max_drawdown']}%  t={s['t_stat']}", flush=True)
    print("\n=== B. ON THE VALUE PICK — does stock acceleration help? ===", flush=True)
    for k, v in cond.items():
        print(f"  {k:18} mean {v['mean_pct']}%  win {v['win_pct']}%  (n{v['n']})" if v else f"  {k}: –", flush=True)
    print(f"  LIFT (accelerating minus fading): {round(lift, 2) if lift is not None else '–'}pp", flush=True)

    helps_pick = lift is not None and lift > 1.0
    payload = {
        "computed_at": pd.Timestamp.utcnow().isoformat(),
        "params": {"lookback_months": LOOKBACK, "top_n_sectors": TOP_N, "top_stocks": TOP_STK,
                   "benchmark": BENCH, "months": int(len(midx)),
                   "accel": "stock 3mo return minus prior 3mo return"},
        "standalone": standalone, "on_value_pick": cond, "value_pick_lift_pp": round(lift, 2) if lift is not None else None,
        "verdict": (("STANDALONE stock acceleration is a real signal" + (" and it HELPS the value pick (+lift)."
                     if helps_pick else " but on the VALUE PICK it does NOT help (momentum-confirmation vs buy-"
                     "weakness) — show it as an INFO indicator, don't gate the value pick on it."))
                    if max(s["vs_spy"] for s in standalone.values()) > 30 else
                    "Stock acceleration is weak even standalone; informational indicator only."),
        "caveat": "In-sample, no fees, ~5y, 9mo warmup. Stock accel is a per-name momentum 2nd-derivative — good "
                  "as a displayed INDICATOR regardless; whether it should DRIVE selection is what this tests.",
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
            kind="stock_acceleration", defaults={"payload": json.loads(json.dumps(p, default=str)),
                                                 "computed_at": timezone.now()})
        print("Saved BacktestResult[stock_acceleration]", flush=True)
    except Exception as e:
        print("DB save failed:", e, flush=True)
    print("\n" + p["verdict"], flush=True)


if __name__ == "__main__":
    main()
