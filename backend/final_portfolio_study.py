#!/usr/bin/env python3
"""ASSEMBLE THE PORTFOLIO — stitch the validated pieces into one sized, risk-managed book and decompose each
leg's marginal contribution honestly. Uses ONLY cleanly-applicable, walk-forward-survived pieces (skips the
levers memory flags as fragile/survivorship-inflated). The crisis stress-test proved crisis defense must be
an EXPLICIT gate, so the bear gate is now core, not optional.
  CORE      flagship: accel top-10 -> cheapest-P/B guard low-debt -> month-end, $5M floor.
  +BEAR     slow 12mo absolute-momentum gate: SPY trailing-12mo < 0 -> that month to CASH (abs12_cash).
  +DISTRESS 15% satellite sleeve (neg-book unprofitable basket), also bear-gated (it's pro-cyclical).
Same warmup window for all legs (apples-to-apples; baseline total < the +398 headline because of the 12mo
warmup). Reports total/vsSPY/Sharpe/DD/win per leg + the 2022 bear behavior. -> BacktestResult[final_portfolio].
Run: MSYS_NO_PATHCONV=1 docker exec rotation-backend-1 python -u /app/final_portfolio_study.py
"""
import os, json, warnings
warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
os.environ.setdefault("DJANGO_SETTINGS_MODULE", "rotation.settings")
import django
django.setup()

import config, sector_holdings
import price_basis
from seq_fundamental_study import load_candles, load_financial_reports
from trend_stock_studies import _pit_monthly_panel, _available_at, _ret_delist, CRYPTO
from backtest_lowpb import _monthly_close, BENCH

TOP_N = 10
W_DIST = 0.15          # distressed satellite weight
WARMUP = 12            # 12mo for the bear gate


def _stats(r, spy):
    r = np.asarray(r, float); n = len(r)
    if n == 0:
        return dict(total=0, vs_spy=0, sharpe=0, dd=0, win=0)
    tot = float(np.prod(1 + r) - 1) * 100; sp = float(np.prod(1 + np.asarray(spy)) - 1) * 100
    sh = float(r.mean() / r.std() * np.sqrt(12)) if r.std() > 1e-9 else 0.0
    eq = np.cumprod(1 + r); dd = float(((eq / np.maximum.accumulate(eq)) - 1).min() * 100)
    return dict(total=round(tot, 1), vs_spy=round(tot - sp, 1), sharpe=round(sh, 2), dd=round(dd, 1),
                win=round((r > 0).mean() * 100, 1))


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
    spy_close = etf_daily[BENCH]["Close"].resample("ME").last().reindex(midx)
    spy_12mom = spy_close.pct_change(12)                       # trailing 12mo absolute momentum (bear gate)
    accel = etf_m.pct_change(3) - etf_m.pct_change(3).shift(3)
    stock_daily = load_candles(all_holds)
    stock_m = _monthly_close(stock_daily).reindex(midx)
    reps = load_financial_reports(all_holds)
    sh, eq, ni, dt = (_pit_monthly_panel(reps, f, midx) for f in
                      ("shares_outstanding", "total_equity", "net_income", "total_debt"))
    common = stock_m.columns.intersection(sh.columns).intersection(eq.columns)
    R = lambda p: p.reindex(index=midx, columns=common)
    px = stock_m[common]; sh, eq, ni, dt = R(sh), R(eq), R(ni), R(dt)
    pb = (price_basis.as_traded_close(px) * sh) / eq.where(eq != 0)
    trap = (ni < 0) & (~(eq >= eq.shift(12))) & (~(ni > ni.shift(4)))
    low = (dt / eq.where(eq != 0)) < 1.0
    dvol = pd.DataFrame({t: (stock_daily[t]["Close"] * stock_daily[t]["Volume"]).rolling(20).mean()
                         .resample("ME").last().reindex(midx) for t in common
                         if t in stock_daily and "Volume" in stock_daily[t]}).reindex(index=midx, columns=common)
    print(f"months {len(midx)} | stocks {len(common)}", flush=True)

    core, dist, spies, gate, dates = [], [], [], [], []
    for i in range(WARMUP, len(midx) - 1):
        date, ndate = midx[i], midx[i + 1]
        sp = spy_close.iloc[i + 1] / spy_close.iloc[i] - 1
        if not np.isfinite(sp):
            continue
        top = accel.loc[date].dropna().sort_values(ascending=False).head(TOP_N).index
        f_slot = []
        for etf in top:
            _, holds = sector_map.get(etf, (etf, []))
            c = [h for h in holds if h in px.columns and _available_at(px[h], date) and pd.notna(pb.loc[date, h])
                 and pb.loc[date, h] > 0 and not bool(trap.loc[date, h])
                 and pd.notna(dvol.loc[date, h]) and dvol.loc[date, h] >= 5e6]
            g = [x for x in c if bool(low.loc[date, x])] or c
            if g:
                r = _ret_delist(px[pb.loc[date, g].idxmin()], date, ndate)
                if r is not None and np.isfinite(r):
                    f_slot.append(float(r))
        if not f_slot:
            continue
        d_slot = []
        for h in common:
            if not _available_at(px[h], date):
                continue
            if not (pd.notna(dvol.loc[date, h]) and dvol.loc[date, h] >= 5e6):
                continue
            e_ = eq.loc[date, h]
            if pd.isna(e_) or e_ >= 0 or not (pd.notna(ni.loc[date, h]) and ni.loc[date, h] <= 0):
                continue
            r = _ret_delist(px[h], date, ndate)
            if r is not None and np.isfinite(r):
                d_slot.append(float(r))
        core.append(float(np.mean(f_slot)))
        dist.append(float(np.mean(d_slot)) if d_slot else 0.0)
        spies.append(float(sp)); dates.append(ndate)
        gate.append(0.0 if (pd.notna(spy_12mom.iloc[i]) and spy_12mom.iloc[i] < 0) else 1.0)   # 0=cash

    core, dist, spies, gate = np.array(core), np.array(dist), np.array(spies), np.array(gate)
    dts = pd.DatetimeIndex(dates)
    n_cash = int((gate == 0).sum())

    leg_core = core
    leg_bear = core * gate                                    # cash in bear months
    leg_full = (1 - W_DIST) * (core * gate) + W_DIST * (dist * gate)   # bear-gated blend

    legs = {"CORE flagship": _stats(leg_core, spies),
            "+ BEAR gate": _stats(leg_bear, spies),
            f"+ {int(W_DIST*100)}% DISTRESS (final)": _stats(leg_full, spies)}
    spy_ref = _stats(spies, spies)
    print(f"\n=== ASSEMBLED PORTFOLIO — additive decomposition ({len(core)} months, {n_cash} cash-gated) ===", flush=True)
    for k, s in legs.items():
        print(f"  {k:26} total {s['total']:>7}%  vsSPY {s['vs_spy']:>7}  Sh {s['sharpe']:>5}  DD {s['dd']:>7}%  win {s['win']}%", flush=True)
    print(f"  {'SPY (buy-hold)':26} total {spy_ref['total']:>7}%  {'':>14} DD {spy_ref['dd']}%", flush=True)

    # 2022 bear behavior (only in-sample crisis)
    m22 = (dts >= "2022-01-01") & (dts <= "2022-12-31")
    b22 = {}
    if m22.any():
        for name, series in (("core", leg_core), ("bear_gated", leg_bear), ("final", leg_full), ("spy", spies)):
            s = series[m22]
            b22[name] = dict(ret=round(float(np.prod(1 + s) - 1) * 100, 1),
                             dd=round(float(((np.cumprod(1 + s) / np.maximum.accumulate(np.cumprod(1 + s))) - 1).min() * 100), 1))
        print(f"\n=== 2022 BEAR (the only in-sample crisis) ===", flush=True)
        for k, v in b22.items():
            print(f"  {k:12} return {v['ret']:>7}%  maxDD {v['dd']:>7}%", flush=True)

    fin = legs[f"+ {int(W_DIST*100)}% DISTRESS (final)"]; c = legs["CORE flagship"]
    payload = {
        "computed_at": pd.Timestamp.utcnow().isoformat(),
        "params": {"top_n": TOP_N, "distress_weight": W_DIST, "warmup": WARMUP, "benchmark": BENCH,
                   "months": int(len(core)), "cash_gated_months": n_cash},
        "legs": legs, "spy": spy_ref, "bear_2022": b22,
        "spec": {"core": "accel top-10 -> cheapest-P/B, profit-guard (not trap), low-debt D/E<1, $5M/day floor, "
                          "equal-weight, hold to month-end rebalance",
                 "bear_gate": "SPY trailing-12mo absolute momentum < 0 -> full portfolio to cash that month",
                 "distressed_sleeve": f"{int(W_DIST*100)}% in equal-weight neg-book+unprofitable $5M-liquid basket, "
                                      "bear-gated with the core"},
        "verdict": (f"Assembled book: {fin['total']}% total ({fin['vs_spy']:+.0f} vs SPY), Sharpe {fin['sharpe']}, "
                    f"maxDD {fin['dd']}% vs core-only Sharpe {c['sharpe']}/DD {c['dd']}%. Bear gate "
                    f"{'improves' if legs['+ BEAR gate']['dd'] > c['dd'] else 'does not improve'} drawdown; distressed "
                    f"tilt adds {fin['vs_spy']-legs['+ BEAR gate']['vs_spy']:+.0f}pp vs SPY. This is the deployable spec."),
        "caveat": "12mo-warmup window so CORE total < the +398 headline (fewer months). Bear gate has ONE in-sample "
                  "bear (2022) — crisis value is INSURANCE-logic (crisis-test showed the signal alone gives no "
                  "protection), not a backtested edge. Distressed leg pro-cyclical/crisis-untested. No fees.",
    }
    return payload


def main():
    p = build()
    from pathlib import Path
    OUT = Path("/app/.data/studies/final_portfolio.json")
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(p, indent=2, default=str))
    try:
        from core.models import BacktestResult
        from django.utils import timezone
        BacktestResult.objects.update_or_create(
            kind="final_portfolio", defaults={"payload": json.loads(json.dumps(p, default=str)), "computed_at": timezone.now()})
        print("Saved BacktestResult[final_portfolio]", flush=True)
    except Exception as e:
        print("DB save failed:", e, flush=True)
    print("\n" + p["verdict"], flush=True)


if __name__ == "__main__":
    main()
