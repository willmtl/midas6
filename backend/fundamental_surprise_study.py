#!/usr/bin/env python3
"""SARIMA-ON-FUNDAMENTALS (#1, return-additive) — the ONE place SARIMA plays to its strength (seasonality +
autocorrelation), unlike returns. For each flagship PICK, fit SARIMA on the company's quarterly REVENUE history
(PIT — only reports available at the pick date), forecast next quarter, and compute predicted YoY growth. If
picks with HIGH SARIMA-predicted growth outperform, anticipated fundamental momentum is a usable SELECTION tilt.

Bucket the flagship picks by predicted-growth quartile -> mean forward 3-month return per bucket + t-stat, and
corr(predicted growth, fwd return). Bounded (~one SARIMA fit per pick). -> BacktestResult[fundamental_surprise].
Run: MSYS_NO_PATHCONV=1 docker exec rotation-backend-1 python -u /app/fundamental_surprise_study.py
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
from statsmodels.tsa.statespace.sarimax import SARIMAX
from scipy.stats import spearmanr

TOP_N = 10; MIN_DVOL = 5e6; FWD_M = 3
OUT = Path(__file__).resolve().parent / ".data" / "studies" / "fundamental_surprise.json"


def build():
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
    dvol = {}
    for t in common:
        d = stock_daily.get(t)
        if d is None or "Volume" not in d or len(d) < 90:
            continue
        dvol[t] = (d["Close"] * d["Volume"]).rolling(20).mean().resample("ME").last().reindex(midx)
    dvol = pd.DataFrame(dvol).reindex(index=midx, columns=common)

    def pick(etf, date, held):
        _, holds = sector_map.get(etf, (etf, set()))
        c = [h for h in holds if h in px.columns and h not in held and _available_at(px[h], date)
             and pd.notna(pb.loc[date, h]) and pb.loc[date, h] > 0 and not bool(trap.loc[date, h])
             and pd.notna(dvol.loc[date, h]) and dvol.loc[date, h] >= MIN_DVOL]
        g = [x for x in c if bool(low.loc[date, x])] or c
        return min(g, key=lambda h: pb.loc[date, h]) if g else None

    # 1) flagship picks with forward 3-month return
    picks = []
    for i in range(9, len(midx) - 1):
        date = midx[i]
        top = accel.loc[date].dropna().sort_values(ascending=False).head(TOP_N).index
        held = set()
        for etf in top:
            p = pick(etf, date, held)
            if not p:
                continue
            held.add(p)
            r = _ret_delist(px[p], date, midx[min(i + FWD_M, len(midx) - 1)])
            if r is not None and np.isfinite(r):
                picks.append({"date": date, "ticker": p, "fwd": float(r)})
    print(f"{len(picks)} flagship picks; forecasting quarterly revenue (SARIMA) per pick...", flush=True)

    # 2) SARIMA-predicted next-quarter revenue YoY growth for each pick (PIT)
    rev_by = {}
    for t, g in reps.items():
        if "revenue" in g and "avail_date" in g and "period_end" in g:
            rev_by[t] = g[["period_end", "avail_date", "revenue"]].copy()
    done = 0
    for pk in picks:
        pk["pred_yoy"] = None
        g = rev_by.get(pk["ticker"])
        if g is None:
            continue
        av = pd.to_datetime(g["avail_date"], errors="coerce")
        hist = g[av <= pk["date"]].dropna(subset=["revenue"]).sort_values("period_end")
        rev = pd.to_numeric(hist["revenue"], errors="coerce").dropna().values
        if len(rev) < 12:
            continue
        try:
            res = SARIMAX(rev, order=(1, 1, 0), seasonal_order=(1, 1, 0, 4),
                          enforce_stationarity=False, enforce_invertibility=False).fit(disp=0, maxiter=50)
            f = float(res.forecast(steps=1)[0])
            base = rev[-4]                                  # same quarter last year
            if base and np.isfinite(f):
                pk["pred_yoy"] = f / base - 1
        except Exception:
            pass
        done += 1
    have = [pk for pk in picks if pk.get("pred_yoy") is not None and np.isfinite(pk["pred_yoy"])]
    print(f"  SARIMA forecasts on {len(have)}/{len(picks)} picks", flush=True)

    # 3) bucket by predicted growth quartile -> mean fwd return
    if len(have) < 12:
        print("too few forecasts", flush=True); return None
    yoy = np.array([pk["pred_yoy"] for pk in have])
    fwd = np.array([pk["fwd"] for pk in have])
    qs = np.quantile(yoy, [0.25, 0.5, 0.75])
    buckets = {"Q1_low": [], "Q2": [], "Q3": [], "Q4_high": []}
    for pk in have:
        y = pk["pred_yoy"]
        b = "Q1_low" if y <= qs[0] else "Q2" if y <= qs[1] else "Q3" if y <= qs[2] else "Q4_high"
        buckets[b].append(pk["fwd"])
    ic = float(spearmanr(yoy, fwd)[0])

    print(f"\n=== fwd-{FWD_M}mo return by SARIMA-predicted revenue-YoY quartile (n={len(have)}) ===", flush=True)
    out_b = {}
    for b in ("Q1_low", "Q2", "Q3", "Q4_high"):
        v = np.array(buckets[b]); m = round(float(v.mean()) * 100, 2) if len(v) else None
        t = round(_tstat_from_returns(list(v)), 2) if len(v) > 3 else None
        out_b[b] = {"mean_pct": m, "n": len(v), "t": t}
        print(f"  {b:<9} mean {m}%  (n {len(v)}, t {t})", flush=True)
    spread = (out_b["Q4_high"]["mean_pct"] - out_b["Q1_low"]["mean_pct"]) if (out_b["Q4_high"]["mean_pct"] is not None and out_b["Q1_low"]["mean_pct"] is not None) else None
    print(f"  Q4-Q1 spread: {spread}pp   |   IC (Spearman pred_yoy vs fwd): {ic:+.3f}", flush=True)

    verdict = (
        f"SARIMA-predicted revenue-YoY on flagship picks: Q4(high)-Q1(low) fwd-{FWD_M}mo spread {spread}pp, "
        f"IC {ic:+.3f} (n={len(have)}). "
        + ("High predicted-growth picks outperform -> anticipated fundamental momentum is a usable selection tilt."
           if spread is not None and spread > 3 and ic > 0.05 else
           "No clean monotonic payoff -> SARIMA-anticipated revenue growth does NOT sharpen the pick's return here "
           "(revenue growth may already be in the price / P/B; small sample).")
    )
    print("\n" + verdict, flush=True)
    return {"computed_at": pd.Timestamp.utcnow().isoformat(),
            "params": {"top_n": TOP_N, "fwd_months": FWD_M, "n_picks": len(picks), "n_forecast": len(have),
                       "model": "SARIMAX(1,1,0)(1,1,0,4) on quarterly revenue, PIT (avail_date)"},
            "buckets": out_b, "q4_minus_q1_pp": spread, "ic_spearman": round(ic, 4), "verdict": verdict,
            "caveat": "One SARIMA fit per pick on PIT quarterly revenue (>=12 quarters); predicted next-Q revenue / "
                      "same-Q-last-year - 1 = predicted YoY. Bucketed vs fwd-3mo return. Revenue only (not EPS/"
                      "margins); consensus not used (naive YoY reference). Small sample, no fees, survivorship as base."}


def main():
    p = build()
    if p is None:
        return
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(p, indent=2, default=str))
    try:
        from core.models import BacktestResult
        from django.utils import timezone
        BacktestResult.objects.update_or_create(
            kind="fundamental_surprise", defaults={"payload": json.loads(json.dumps(p, default=str)), "computed_at": timezone.now()})
        print("Saved BacktestResult[fundamental_surprise]", flush=True)
    except Exception as e:
        print("DB save failed:", e, flush=True)


if __name__ == "__main__":
    main()
