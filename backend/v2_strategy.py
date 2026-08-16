#!/usr/bin/env python3
"""V2 — stack the three levers that genuinely helped, decomposed additively vs the validated baseline.

  baseline   top-20 pool, top-10 sectors, equal-weight, no gate            (= validated +229% vs SPY)
  +deep      full expanded pool (more stocks per ETF)                       (deep_pool_study)
  +conc      ... then top-5 sectors, inverse-vol weight                     (return_lab winner)
  v2         ... then slow 12mo absolute-momentum bear gate (fail->cash)    (bear_defense winner)

Same PIT engine throughout (rotation by 6mo momentum -> cheapest-P/B guarded low-debt). Shows how much
each lever adds and the fully-stacked v2. NOTE the deep-pool leg carries survivorship inflation (current
expanded constituents over history) -> v2's level is optimistic; the DECOMPOSITION shows lever ordering.
-> BacktestResult[v2_strategy] + JSON.
Run: MSYS_NO_PATHCONV=1 docker exec rotation-backend-1 python -u /app/v2_strategy.py
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

OUT = Path(__file__).resolve().parent / ".data" / "studies" / "v2_strategy.json"
LOOKBACK, RF_M = 6, 0.0033


def _stats(r, spy):
    r = np.asarray(r, float); n = len(r)
    if n == 0:
        return dict(total_return=0, vs_spy=0, sharpe=0, max_drawdown=0, t_stat=None, periods=0)
    tot = float(np.prod(1 + r) - 1) * 100
    sp = float(np.prod(1 + np.asarray(spy)) - 1) * 100
    sh = float(r.mean() / r.std() * np.sqrt(12)) if r.std() > 1e-9 else 0.0
    eq = np.cumprod(1 + r); dd = float(((eq / np.maximum.accumulate(eq)) - 1).min() * 100)
    t = _tstat_from_returns(list(r))
    return dict(total_return=round(tot, 1), vs_spy=round(tot - sp, 1), sharpe=round(sh, 2),
                max_drawdown=round(dd, 1), t_stat=round(t, 2) if t is not None else None, periods=n)


def build():
    etfs = {n: e for n, e in config.SECTOR_ETFS.items() if e not in CRYPTO}
    top20_map, full_map, all_holds = {}, {}, set()
    for n, e in etfs.items():
        b = [t for t in sector_holdings.HOLDINGS.get(n, {}).get("holdings", []) if t not in (e, BENCH) and t not in CRYPTO]
        f = [t for t in sector_holdings.get_holdings(n, expanded=True) if t not in (e, BENCH) and t not in CRYPTO]
        top20_map[e] = (n, b); full_map[e] = (n, f); all_holds.update(f)
    all_holds = sorted(all_holds)

    etf_tk = list(etfs.values())
    print(f"Loading {len(etf_tk)} ETFs + {len(all_holds)} stocks + {BENCH}...", flush=True)
    etf_daily = load_candles(etf_tk + [BENCH])
    etf_m = _monthly_close({t: d for t, d in etf_daily.items() if t in etf_tk})
    midx = etf_m.index
    spy_m = etf_daily[BENCH]["Close"].resample("ME").last().reindex(midx)
    etf_t6 = etf_m.pct_change(LOOKBACK)
    etf_t12 = etf_m.pct_change(12)
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
    volp = pd.DataFrame({t: stock_daily[t]["Close"].pct_change().rolling(60).std().resample("ME").last().reindex(midx)
                         for t in common if t in stock_daily and len(stock_daily[t]) > 70}).reindex(index=midx, columns=common)
    print(f"months {len(midx)} | stocks {len(common)}", flush=True)
    warmup = 12

    def run(pool_map, top_n, weight, gate):
        rets, spies = [], []
        for i in range(warmup, len(midx) - 1):
            date, ndate = midx[i], midx[i + 1]
            sp = spy_m.iloc[i + 1] / spy_m.iloc[i] - 1
            if not np.isfinite(sp):
                continue
            ranks = etf_t6.loc[date].dropna().sort_values(ascending=False).head(top_n).index
            picks, ws = [], []
            for etf in ranks:
                _, holds = pool_map.get(etf, (etf, []))
                cands = [h for h in holds if h in px.columns and _available_at(px[h], date)
                         and pd.notna(pb.loc[date, h]) and pb.loc[date, h] > 0 and not bool(trap.loc[date, h])]
                ld = [c for c in cands if bool(low_debt.loc[date, c])]
                use = ld or cands
                gated_out = gate and not (pd.notna(etf_t12.loc[date, etf]) and etf_t12.loc[date, etf] > 0)
                if not use or gated_out:
                    if gate:
                        picks.append(RF_M); ws.append(1.0)      # cash slot under bear gate
                    continue
                pick = pb.loc[date, use].idxmin()
                r = _ret_delist(px[pick], date, ndate)
                if r is None or not np.isfinite(r):
                    continue
                w = 1.0
                if weight == "invvol" and pd.notna(volp.loc[date, pick]) and volp.loc[date, pick] > 0:
                    w = 1.0 / volp.loc[date, pick]
                picks.append(float(r)); ws.append(float(w))
            if picks:
                w = np.array(ws); w = w / w.sum()
                rets.append(float(np.dot(w, picks))); spies.append(float(sp))
        return _stats(rets, spies)

    configs = {
        "baseline": run(top20_map, 10, "equal", False),
        "+deep": run(full_map, 10, "equal", False),
        "+deep+conc": run(full_map, 5, "invvol", False),
        "v2": run(full_map, 5, "invvol", True),
    }
    print("\n=== V2 — stacked levers (additive) ===", flush=True)
    prev = None
    for k, s in configs.items():
        add = "" if prev is None else f"  ({'+' if s['vs_spy']-prev>=0 else ''}{round(s['vs_spy']-prev,1)}pp)"
        print(f"  {k:12} vsSPY {s['vs_spy']:>7}%  total {s['total_return']}%  t={s['t_stat']}  "
              f"Sh {s['sharpe']}  DD {s['max_drawdown']}%{add}", flush=True)
        prev = s["vs_spy"]

    payload = {
        "computed_at": pd.Timestamp.utcnow().isoformat(),
        "params": {"lookback_months": LOOKBACK, "rf_monthly": RF_M, "benchmark": BENCH, "months": int(len(midx)),
                   "warmup": warmup},
        "configs": configs,
        "verdict": (f"V2 (deep pool + top-5 inverse-vol + slow-mom bear gate) = {configs['v2']['vs_spy']}% vs SPY "
                    f"(Sharpe {configs['v2']['sharpe']}, DD {configs['v2']['max_drawdown']}%) vs baseline "
                    f"{configs['baseline']['vs_spy']}%. Levers stack, but the deep-pool leg is survivorship-inflated "
                    "-> treat v2's LEVEL as optimistic; the decomposition shows each lever's contribution."),
        "caveat": "Deep-pool leg = current expanded constituents over history (survivorship). In-sample, no fees, "
                  "~5y single regime, 12mo warmup shortens window. NEEDS walk-forward before trusting.",
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
            kind="v2_strategy", defaults={"payload": json.loads(json.dumps(p, default=str)),
                                          "computed_at": timezone.now()})
        print("Saved BacktestResult[v2_strategy]", flush=True)
    except Exception as e:
        print("DB save failed:", e, flush=True)
    print("\n" + p["verdict"], flush=True)


if __name__ == "__main__":
    main()
