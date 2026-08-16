#!/usr/bin/env python3
"""VOLATILITY REGIME OVERLAY — three of the article links (SMA/EWMA vol, GARCH) collapse to one question: does a
volatility estimate improve the flagship as a SIZING / REGIME-GATE overlay? Our prior ([[factor-lab]]): hard vol
FILTERS hurt (over-select), so vol belongs in EXPOSURE, not selection. Estimate next-month market vol three ways
on SPY daily returns — SMA(21) realized, EWMA (RiskMetrics lambda=0.94), GARCH(1,1) 1-step (needs `arch`; skipped
if absent) — all as-of month-end (no look-ahead), then overlay on the flagship's realized monthly returns:

  baseline        always fully invested (the deployed flagship).
  vol_target      scale exposure = target_vol / forecast_vol, capped [0,1] (cash earns 0); target = median fcast.
  risk_off_gate   exposure 0 (to cash) when forecast vol > its trailing median, else 1.
Plus a DIAGNOSTIC: flagship monthly return split by high vs low vol regime (are high-vol months the bad ones?).
If vol-target lifts Sharpe / cuts DD at acceptable return cost, it's worth wiring as a sizing layer.
-> BacktestResult[volatility_regime] + JSON.
Run: MSYS_NO_PATHCONV=1 docker exec rotation-backend-1 python -u /app/volatility_regime_study.py
"""
import os, json, warnings
warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
os.environ.setdefault("DJANGO_SETTINGS_MODULE", "rotation.settings")
import django
django.setup()

from pathlib import Path
import config, sector_holdings, price_basis
from seq_fundamental_study import load_candles, load_financial_reports
from trend_stock_studies import _pit_monthly_panel, _available_at, _ret_delist, CRYPTO
from backtest_lowpb import _monthly_close, _tstat_from_returns, BENCH

TOP_N = 10; CONV = 2.0; MIN_DVOL = 5e6
OUT = Path(__file__).resolve().parent / ".data" / "studies" / "volatility_regime.json"

try:
    from arch import arch_model
    HAVE_ARCH = True
except Exception:
    HAVE_ARCH = False


def _perf(r, spy):
    r = np.asarray(r, float); n = len(r)
    tot = float(np.prod(1 + r) - 1) * 100
    sp = float(np.prod(1 + np.asarray(spy)) - 1) * 100
    sh = float(r.mean() / r.std() * np.sqrt(12)) if r.std() > 1e-9 else 0.0
    eqc = np.cumprod(1 + r); dd = float(((eqc / np.maximum.accumulate(eqc)) - 1).min() * 100)
    t = _tstat_from_returns(list(r))
    return dict(total=round(tot, 1), vs_spy=round(tot - sp, 1), sharpe=round(sh, 2), dd=round(dd, 1),
                t_stat=round(t, 2) if t is not None else None, months=n)


def flagship_monthly():
    """(dates, port_rets, spy_rets) for the deployed flagship."""
    etfs = {n: e for n, e in config.SECTOR_ETFS.items() if e not in CRYPTO}
    sector_map, all_holds = {}, set()
    for n, e in etfs.items():
        h = [t for t in sector_holdings.get_holdings(n) if t not in (e, BENCH) and t not in CRYPTO]
        sector_map[e] = (n, set(h)); all_holds.update(h)
    all_holds = sorted(all_holds)
    etf_tk = list(etfs.values())
    etf_daily = load_candles(etf_tk + [BENCH])
    etf_m = _monthly_close({t: d for t, d in etf_daily.items() if t in etf_tk})
    midx = etf_m.index
    accel = etf_m.pct_change(3) - etf_m.pct_change(3).shift(3)
    spy_daily = etf_daily[BENCH]["Close"]
    spy_m = spy_daily.resample("ME").last().reindex(midx)
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
    dvol, adl_m = {}, {}
    for t in common:
        d = stock_daily.get(t)
        if d is None or "Volume" not in d or len(d) < 90:
            continue
        v = d["Volume"]
        dvol[t] = (d["Close"] * v).rolling(20).mean().resample("ME").last().reindex(midx)
        if {"High", "Low", "Close"}.issubset(d.columns):
            rng = (d["High"] - d["Low"]).replace(0, np.nan)
            mfm = ((d["Close"] - d["Low"]) - (d["High"] - d["Close"])) / rng
            adl_m[t] = (mfm.fillna(0) * v).cumsum().resample("ME").last().reindex(midx)
    dvol = pd.DataFrame(dvol).reindex(index=midx, columns=common)
    adl = pd.DataFrame(adl_m).reindex(index=midx, columns=common)
    ad_slope3 = adl - adl.shift(3); px_ret3 = px.pct_change(3)

    def pick(etf, date, held):
        _, holds = sector_map.get(etf, (etf, set()))
        c = [h for h in holds if h in px.columns and h not in held and _available_at(px[h], date)
             and pd.notna(pb.loc[date, h]) and pb.loc[date, h] > 0 and not bool(trap.loc[date, h])
             and pd.notna(dvol.loc[date, h]) and dvol.loc[date, h] >= MIN_DVOL]
        g = [x for x in c if bool(low.loc[date, x])] or c
        return min(g, key=lambda h: pb.loc[date, h]) if g else None

    def wt(name, date):
        a, p = ad_slope3.loc[date].get(name), px_ret3.loc[date].get(name)
        return CONV if (pd.notna(a) and pd.notna(p) and a > 0 and p < 0) else 1.0

    dates, rets, spies = [], [], []
    for i in range(9, len(midx) - 1):
        date, ndate = midx[i], midx[i + 1]
        sp = spy_m.iloc[i + 1] / spy_m.iloc[i] - 1
        if not np.isfinite(sp):
            continue
        top = accel.loc[date].dropna().sort_values(ascending=False).head(TOP_N).index
        held = set(); wsum = rr = 0.0
        for etf in top:
            p = pick(etf, date, held)
            if not p:
                continue
            held.add(p)
            r = _ret_delist(px[p], date, ndate)
            if r is None or not np.isfinite(r):
                continue
            w = wt(p, date); wsum += w; rr += w * float(r)
        if wsum <= 0:
            continue
        dates.append(date); rets.append(rr / wsum); spies.append(float(sp))
    return pd.DatetimeIndex(dates), np.array(rets), np.array(spies), spy_daily


def vol_estimates(spy_daily, month_dates):
    """forecast next-month annualized vol as-of each month-end, three ways."""
    r = spy_daily.pct_change().dropna()
    sma = (r.rolling(21).std() * np.sqrt(252))
    # EWMA (RiskMetrics)
    lam = 0.94
    var = r.pow(2).ewm(alpha=1 - lam, adjust=False).mean()
    ewma = (var.pow(0.5) * np.sqrt(252))
    out = {"sma": [], "ewma": [], "garch": []}
    for d in month_dates:
        rr = r[r.index <= d]
        out["sma"].append(float(sma.reindex([sma.index[sma.index <= d][-1]]).iloc[0]) if (sma.index <= d).any() else np.nan)
        out["ewma"].append(float(ewma.reindex([ewma.index[ewma.index <= d][-1]]).iloc[0]) if (ewma.index <= d).any() else np.nan)
        if HAVE_ARCH and len(rr) > 260:
            try:
                res = arch_model(rr * 100, vol="Garch", p=1, q=1, dist="normal").fit(disp="off")
                fc = res.forecast(horizon=21, reindex=False)
                fvar = float(fc.variance.values[-1].sum())      # 21-day % variance
                out["garch"].append(np.sqrt(fvar) / 100 * np.sqrt(252 / 21))
            except Exception:
                out["garch"].append(np.nan)
        else:
            out["garch"].append(np.nan)
    return {k: pd.Series(v, index=month_dates) for k, v in out.items()}


def build():
    dates, rets, spies, spy_daily = flagship_monthly()
    print(f"flagship months {len(dates)} ({dates[0].date()}..{dates[-1].date()}); arch={HAVE_ARCH}", flush=True)
    vols = vol_estimates(spy_daily, dates)
    base = _perf(rets, spies)

    def overlay(fvol, mode):
        """exposure decided at START of month t from fvol as-of month t-1 (shift 1) -> no look-ahead."""
        f = fvol.shift(1)                                  # use prior month-end forecast for this month
        med = f.expanding(min_periods=6).median()
        out = []
        for i, d in enumerate(dates):
            fv, mv = f.iloc[i], med.iloc[i]
            if not np.isfinite(fv) or not np.isfinite(mv):
                exp = 1.0
            elif mode == "target":
                exp = float(np.clip((mv / fv), 0.0, 1.0))  # target vol = trailing median
            else:  # gate
                exp = 0.0 if fv > mv else 1.0
            out.append(exp * rets[i])
        return _perf(out, spies)

    results = {"baseline": base}
    for est in ("sma", "ewma", "garch"):
        if vols[est].notna().any():
            results[f"vol_target_{est}"] = overlay(vols[est], "target")
            results[f"risk_off_gate_{est}"] = overlay(vols[est], "gate")

    # diagnostic: flagship return by vol regime (high = ewma fcast above trailing median)
    f = vols["ewma"].shift(1); med = f.expanding(min_periods=6).median()
    hi = [rets[i] for i in range(len(dates)) if np.isfinite(f.iloc[i]) and np.isfinite(med.iloc[i]) and f.iloc[i] > med.iloc[i]]
    lo = [rets[i] for i in range(len(dates)) if np.isfinite(f.iloc[i]) and np.isfinite(med.iloc[i]) and f.iloc[i] <= med.iloc[i]]
    diag = {"high_vol_mean_pct": round(float(np.mean(hi)) * 100, 2) if hi else None, "high_vol_n": len(hi),
            "low_vol_mean_pct": round(float(np.mean(lo)) * 100, 2) if lo else None, "low_vol_n": len(lo)}

    print(f"\n=== VOL REGIME OVERLAY on the flagship (total / vsSPY / Sharpe / DD) ===", flush=True)
    for k, r in results.items():
        print(f"  {k:<24} {r['total']:>7}%  vsSPY {r['vs_spy']:>7}  Sh {r['sharpe']:>5}  DD {r['dd']:>6}%  t {r['t_stat']}",
              flush=True)
    print(f"\n  diagnostic — flagship monthly mean: HIGH-vol {diag['high_vol_mean_pct']}% (n{diag['high_vol_n']}) "
          f"vs LOW-vol {diag['low_vol_mean_pct']}% (n{diag['low_vol_n']})", flush=True)

    cand = [k for k in results if k != "baseline"]
    best = max(cand, key=lambda k: results[k]["sharpe"]) if cand else None
    verdict = (
        f"Baseline {base['total']}%/Sh{base['sharpe']}/DD{base['dd']}%. "
        + (f"Best vol overlay = {best} ({results[best]['total']}%/Sh{results[best]['sharpe']}/DD{results[best]['dd']}%). "
           if best else "")
        + ("A vol overlay improves Sharpe/DD -> worth wiring as a sizing layer. "
           if best and results[best]["sharpe"] > base["sharpe"] + 0.05 else
           "No vol overlay beats baseline Sharpe -> vol-timing the whole book doesn't help; keep fully invested. ")
        + (f"High-vol months avg {diag['high_vol_mean_pct']}% vs low-vol {diag['low_vol_mean_pct']}% "
           + ("(high-vol IS worse -> a gentle de-risk has basis but the overlays above already test it)."
              if (diag['high_vol_mean_pct'] is not None and diag['low_vol_mean_pct'] is not None
                  and diag['high_vol_mean_pct'] < diag['low_vol_mean_pct']) else
              "(high-vol NOT clearly worse -> nothing to time).") if diag['high_vol_mean_pct'] is not None else "")
    )
    print("\n" + verdict, flush=True)
    return {
        "computed_at": pd.Timestamp.utcnow().isoformat(),
        "params": {"top_n": TOP_N, "min_dvol": MIN_DVOL, "benchmark": BENCH, "months": int(len(dates)),
                   "have_arch_garch": HAVE_ARCH, "estimators": ["sma21", "ewma(0.94)", "garch(1,1)"],
                   "overlay": "exposure from prior-month-end forecast (no look-ahead), cash=0"},
        "results": results, "diagnostic": diag, "verdict": verdict,
        "caveat": "Overlay applied to the flagship's realized monthly returns (vol times the whole book, not per-name). "
                  "Cash earns 0 (no T-bill). GARCH refit expanding each month. PIT, no fees, survivorship as in the "
                  "flagship. Vol-target uses trailing-median vol as the target so avg exposure ~1.",
    }


def main():
    p = build()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(p, indent=2, default=str))
    try:
        from core.models import BacktestResult
        from django.utils import timezone
        BacktestResult.objects.update_or_create(
            kind="volatility_regime", defaults={"payload": json.loads(json.dumps(p, default=str)),
                                                "computed_at": timezone.now()})
        print("Saved BacktestResult[volatility_regime]", flush=True)
    except Exception as e:
        print("DB save failed:", e, flush=True)


if __name__ == "__main__":
    main()
