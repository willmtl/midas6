#!/usr/bin/env python3
"""ESTIMATE MOMENTUM — the historically-testable analyst-estimate signal. True day-to-day revisions aren't
backfillable (EODHD Trend is current-only), but EODHD Earnings::History gives the CONSENSUS EPS ESTIMATE at
every past earnings back decades (already in EarningsEvent). Build 'estimate momentum' = is a stock's consensus
EPS estimate TRENDING UP across its last few reported quarters, using ONLY reports with report_date <= rebalance
date (clean PIT). Test as a flagship overlay: (1) conditional forward-return lift of the value pick when
estimates are rising vs falling; (2) a conviction TILT toward rising-estimate names. Does analyst-estimate
momentum add to the flagship, or (like earnings-surprise) invert on deep-value picks?
-> BacktestResult[estimate_momentum]. Run: MSYS_NO_PATHCONV=1 docker exec rotation-backend-1 python -u /app/estimate_momentum_study.py
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
LOOKBACK_Q = 4        # compare latest consensus estimate vs this many reported quarters ago (YoY-ish)


def _stats(r, spy):
    r = np.asarray(r, float); n = len(r)
    if n == 0:
        return dict(total=0, vs_spy=0, sharpe=0, dd=0, win=0)
    tot = float(np.prod(1 + r) - 1) * 100; sp = float(np.prod(1 + np.asarray(spy)) - 1) * 100
    sh = float(r.mean() / r.std() * np.sqrt(12)) if r.std() > 1e-9 else 0.0
    eqc = np.cumprod(1 + r); dd = float(((eqc / np.maximum.accumulate(eqc)) - 1).min() * 100)
    return dict(total=round(tot, 1), vs_spy=round(tot - sp, 1), sharpe=round(sh, 2), dd=round(dd, 1),
                win=round((r > 0).mean() * 100, 1))


def _est_series(universe):
    """Per-ticker time-series of (report_date, eps_estimate) from EarningsEvent, sorted, estimate not null."""
    from core.models import EarningsEvent
    out = {}
    for r in (EarningsEvent.objects.filter(ticker__in=universe, eps_estimate__isnull=False)
              .values("ticker", "report_date", "eps_estimate").order_by("ticker", "report_date")):
        out.setdefault(r["ticker"], []).append((pd.Timestamp(r["report_date"]), float(r["eps_estimate"])))
    return out


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
    print("loading EODHD historical consensus estimates (EarningsEvent)...", flush=True)
    est = _est_series(list(common))
    have = sum(1 for t in common if len(est.get(t, [])) > LOOKBACK_Q)
    print(f"months {len(midx)} | stocks {len(common)} | stocks w/ >{LOOKBACK_Q}q estimates: {have}", flush=True)

    def est_mom(ticker, date):
        """PIT estimate momentum: latest consensus estimate (report<=date) vs LOOKBACK_Q reports earlier.
        +1 rising, -1 falling, None if insufficient/degenerate. Uses only reports known by `date`."""
        s = est.get(ticker)
        if not s:
            return None
        past = [(d, v) for d, v in s if d <= date]
        if len(past) <= LOOKBACK_Q:
            return None
        cur = past[-1][1]; prev = past[-1 - LOOKBACK_Q][1]
        if prev is None or cur is None or abs(prev) < 1e-6:
            return None
        chg = (cur - prev) / abs(prev)
        return chg

    # conditional forward-return of the flagship pick, split by estimate momentum sign; + tilt variants
    up_rets, dn_rets, na_rets = [], [], []
    base_port, tilt_port, convict_port, spies = [], [], [], []
    n_up = n_dn = n_na = 0
    for i in range(9, len(midx) - 1):
        date, ndate = midx[i], midx[i + 1]
        sp = spy_m.iloc[i + 1] / spy_m.iloc[i] - 1
        if not np.isfinite(sp):
            continue
        top = accel.loc[date].dropna().sort_values(ascending=False).head(TOP_N).index
        picks = []
        for etf in top:
            _, holds = sector_map.get(etf, (etf, []))
            c = [h for h in holds if h in px.columns and _available_at(px[h], date) and pd.notna(pb.loc[date, h])
                 and pb.loc[date, h] > 0 and not bool(trap.loc[date, h])
                 and pd.notna(dvol.loc[date, h]) and dvol.loc[date, h] >= 5e6]
            g = [x for x in c if bool(low.loc[date, x])] or c
            if not g:
                continue
            pick = min(g, key=lambda h: pb.loc[date, h])
            r = _ret_delist(px[pick], date, ndate)
            if r is None or not np.isfinite(r):
                continue
            em = est_mom(pick, date)
            picks.append((float(r), em))
            if em is None:
                na_rets.append(float(r)); n_na += 1
            elif em > 0:
                up_rets.append(float(r)); n_up += 1
            else:
                dn_rets.append(float(r)); n_dn += 1
        if not picks:
            continue
        rs = np.array([r for r, _ in picks])
        base_port.append(float(rs.mean()))
        # TILT: replace the basket's picks? No — keep breadth, WEIGHT toward rising-estimate names (2x), like div_2x
        wc = np.array([2.0 if (em is not None and em > 0) else 1.0 for _, em in picks])
        convict_port.append(float(np.sum(wc * rs) / wc.sum()))
        # pure TILT (drop falling-estimate picks entirely if any riser exists) — the aggressive version
        keep = [r for r, em in picks if not (em is not None and em < 0)] or list(rs)
        tilt_port.append(float(np.mean(keep)))
        spies.append(float(sp))

    def mean(x):
        return round(float(np.mean(x)) * 100, 2) if x else None

    cond = {"est_rising": {"avg_ret": mean(up_rets), "n": n_up, "win": round(np.mean(np.array(up_rets) > 0) * 100, 1) if up_rets else None},
            "est_falling": {"avg_ret": mean(dn_rets), "n": n_dn, "win": round(np.mean(np.array(dn_rets) > 0) * 100, 1) if dn_rets else None},
            "no_estimate": {"avg_ret": mean(na_rets), "n": n_na, "win": round(np.mean(np.array(na_rets) > 0) * 100, 1) if na_rets else None}}
    base = _stats(base_port[:len(spies)], spies)
    convict = _stats(convict_port[:len(spies)], spies)
    tilt = _stats(tilt_port[:len(spies)], spies)
    lift = (cond["est_rising"]["avg_ret"] - cond["est_falling"]["avg_ret"]) if (cond["est_rising"]["avg_ret"] is not None and cond["est_falling"]["avg_ret"] is not None) else None

    print(f"\n=== ESTIMATE MOMENTUM on the flagship value pick ({len(spies)} months) ===", flush=True)
    print(f"  conditional next-month return of the pick:", flush=True)
    for k, v in cond.items():
        print(f"     {k:12} avg {v['avg_ret']}%  win {v['win']}%  (n={v['n']})", flush=True)
    print(f"  rising-minus-falling lift: {lift} pp", flush=True)
    print(f"  portfolios: base {base['total']}%/Sh{base['sharpe']}  |  convict_2x_rising {convict['total']}%/Sh{convict['sharpe']}  |  tilt(drop-falling) {tilt['total']}%/Sh{tilt['sharpe']}", flush=True)

    helps = (lift is not None and lift > 1.0) and convict["sharpe"] > base["sharpe"] + 0.03 and convict["vs_spy"] > base["vs_spy"]
    payload = {
        "computed_at": pd.Timestamp.utcnow().isoformat(),
        "params": {"top_n": TOP_N, "lookback_quarters": LOOKBACK_Q, "benchmark": BENCH, "months": int(len(spies)),
                   "source": "EODHD Earnings::History consensus epsEstimate (EarningsEvent), PIT report_date<=date"},
        "conditional": cond, "rising_minus_falling_pp": lift,
        "portfolios": {"base": base, "convict_2x_rising": convict, "tilt_drop_falling": tilt},
        "verdict": (
            f"Estimate momentum lift (rising−falling) = {lift}pp on the value pick. " + (
            f"ESTIMATE MOMENTUM ADDS — rising-consensus names outperform and a 2x conviction tilt lifts the portfolio "
            f"({convict['total']}%/Sh{convict['sharpe']} vs base {base['total']}%/Sh{base['sharpe']}). A real orthogonal overlay."
            if helps else
            f"Estimate momentum does NOT cleanly add on the deep-value pick (convict {convict['total']}%/Sh{convict['sharpe']} "
            f"vs base {base['total']}%/Sh{base['sharpe']}; tilt {tilt['total']}%/Sh{tilt['sharpe']}). "
            "Likely the same inversion as earnings-surprise: within oversold value picks, already-rising estimates = "
            "already-popped = less reversion left. Historical estimate data now proven usable; this particular signal isn't the edge.")),
        "caveat": "Estimate MOMENTUM (consensus level trending across reported quarters), NOT day-to-day revisions "
                  "(those now accrue daily). PIT: report_date<=date only. In-sample ~5y, no fees.",
    }
    return payload


def main():
    p = build()
    from pathlib import Path
    Path("/app/.data/studies/estimate_momentum.json").write_text(json.dumps(p, indent=2, default=str))
    try:
        from core.models import BacktestResult
        from django.utils import timezone
        BacktestResult.objects.update_or_create(
            kind="estimate_momentum", defaults={"payload": json.loads(json.dumps(p, default=str)), "computed_at": timezone.now()})
        print("Saved BacktestResult[estimate_momentum]", flush=True)
    except Exception as e:
        print("DB save failed:", e, flush=True)
    print("\n" + p["verdict"], flush=True)


if __name__ == "__main__":
    main()
