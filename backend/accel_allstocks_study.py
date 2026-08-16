#!/usr/bin/env python3
"""IS THE ACCELERATION EDGE IN THE SECTOR SIGNAL, OR STILL THE VALUE PICK? Rotate into the top-10
ACCELERATION sectors (the validated signal), then hold the stocks DIFFERENT ways — no value filter at
all — to see where the +422% comes from:

  etf_only     hold the 10 accel-sector ETFs (pure sector-acceleration rotation, no stocks)
  all_stocks   hold ALL holdings of the 10 sectors, equal-weight (no selection)
  all_guarded  all holdings passing guard+low_debt (quality, still no value)
  mom_stock    highest-6mo-momentum stock per sector (momentum pick, not value)
  value_pick   cheapest-P/B guarded low-debt (the baseline, +422%)
If etf_only / all_stocks already beat SPY big, the acceleration SIGNAL carries it; if only value_pick
does, the value selection is still the alpha and acceleration just improved the sector timing.
-> BacktestResult[accel_allstocks] + JSON.
Run: MSYS_NO_PATHCONV=1 docker exec rotation-backend-1 python -u /app/accel_allstocks_study.py
"""
import os, json, warnings
warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
os.environ.setdefault("DJANGO_SETTINGS_MODULE", "rotation.settings")
import django
django.setup()

from pathlib import Path
import config, sector_holdings
from studies import _tstat_from_returns
from seq_fundamental_study import load_candles, load_financial_reports
from trend_stock_studies import _pit_monthly_panel, _available_at, _ret_delist, CRYPTO
from backtest_lowpb import _monthly_close, BENCH
import price_basis

OUT = Path(__file__).resolve().parent / ".data" / "studies" / "accel_allstocks.json"
LOOKBACK, TOP_N = 6, 10


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
    etf_accel = etf_m.pct_change(3) - etf_m.pct_change(3).shift(3)
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
    s_mom6 = px.pct_change(6)
    print(f"months {len(midx)} | stocks {len(common)}", flush=True)
    warmup = 9

    def run(mode):
        rets, spies = [], []
        for i in range(warmup, len(midx) - 1):
            date, ndate = midx[i], midx[i + 1]
            sp = spy_m.iloc[i + 1] / spy_m.iloc[i] - 1
            if not np.isfinite(sp):
                continue
            ranks = etf_accel.loc[date].dropna().sort_values(ascending=False).head(TOP_N).index
            slot = []
            for etf in ranks:
                if mode == "etf_only":
                    r = _ret_delist(etf_m[etf], date, ndate) if etf in etf_m.columns else None
                    if r is not None and np.isfinite(r):
                        slot.append(float(r))
                    continue
                _, holds = sector_map.get(etf, (etf, []))
                avail = [h for h in holds if h in px.columns and _available_at(px[h], date)]
                if mode == "all_stocks":
                    picks = avail
                elif mode == "all_guarded":
                    picks = [h for h in avail if not bool(trap.loc[date, h]) and bool(low_debt.loc[date, h])]
                elif mode == "mom_stock":
                    a = s_mom6.loc[date, [h for h in avail if pd.notna(s_mom6.loc[date, h])]]
                    picks = [a.idxmax()] if len(a) else []
                else:  # value_pick
                    cands = [h for h in avail if pd.notna(pb.loc[date, h]) and pb.loc[date, h] > 0 and not bool(trap.loc[date, h])]
                    ld = [c for c in cands if bool(low_debt.loc[date, c])]
                    use = ld or cands
                    picks = [pb.loc[date, use].idxmin()] if use else []
                rr = [_ret_delist(px[p], date, ndate) for p in picks]
                rr = [float(x) for x in rr if x is not None and np.isfinite(x)]
                if rr:
                    slot.append(float(np.mean(rr)))     # equal-weight within sector
            if slot:
                rets.append(float(np.mean(slot))); spies.append(float(sp))
        return _stats(rets, spies)

    MODES = ["etf_only", "all_stocks", "all_guarded", "mom_stock", "value_pick"]
    results = {m: run(m) for m in MODES}
    print("\n=== ACCELERATION sectors — where's the edge? (no-value vs value) ===", flush=True)
    for m in MODES:
        s = results[m]
        print(f"  {m:12} vsSPY {s['vs_spy']:>7}%  Sh {s['sharpe']}  DD {s['max_drawdown']}%  t={s['t_stat']}", flush=True)

    ev = results["value_pick"]["vs_spy"]; eo = results["etf_only"]["vs_spy"]; ea = results["all_stocks"]["vs_spy"]
    payload = {
        "computed_at": pd.Timestamp.utcnow().isoformat(),
        "params": {"lookback_months": LOOKBACK, "top_n": TOP_N, "benchmark": BENCH, "months": int(len(midx))},
        "results": results,
        "verdict": (f"Accel sectors: etf_only +{eo}%, all_stocks +{ea}%, value_pick +{ev}%. "
                    + ("The acceleration SECTOR signal already beats SPY without any stock selection — but the "
                       "VALUE PICK still adds large alpha on top (best). Both matter: accel times the sector, value "
                       "picks the stock." if ev > max(eo, ea) + 40 else
                       "The acceleration signal carries most of the edge even without value selection.")),
        "caveat": "In-sample, no fees, ~5y, 9mo warmup. etf_only = pure sector rotation on the accel signal.",
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
            kind="accel_allstocks", defaults={"payload": json.loads(json.dumps(p, default=str)),
                                              "computed_at": timezone.now()})
        print("Saved BacktestResult[accel_allstocks]", flush=True)
    except Exception as e:
        print("DB save failed:", e, flush=True)
    print("\n" + p["verdict"], flush=True)


if __name__ == "__main__":
    main()
