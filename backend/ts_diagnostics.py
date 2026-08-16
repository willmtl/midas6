#!/usr/bin/env python3
"""TIME-SERIES DIAGNOSTICS — deduce structure (not forecast returns). Three analyses on data we already have:
  (#4) SEASONALITY  — mean sector-ETF return by calendar month; which months/sectors reliably rally.
  (#5) STATIONARITY — ADF on each sector's log-price: reject unit root => MEAN-REVERTING (reversal-suited);
                      fail to reject => TRENDING (momentum-suited). Also ADF on the sector/SPY RATIO (pairs).
  (#2) GRANGER      — does each macro series (USD, HY spread, M2 YoY, net-liquidity) Granger-CAUSE the market
                      return? Puts the [[macro-liquidity-regime]] signal on a causal footing (lead vs coincide).
-> BacktestResult[ts_diagnostics] + JSON.
Run: MSYS_NO_PATHCONV=1 docker exec rotation-backend-1 python -u /app/ts_diagnostics.py
"""
import os, json, warnings
warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
os.environ.setdefault("DJANGO_SETTINGS_MODULE", "rotation.settings")
import django
django.setup()

from pathlib import Path
import config
from seq_fundamental_study import load_candles
from trend_stock_studies import CRYPTO
from backtest_lowpb import _monthly_close, BENCH
from statsmodels.tsa.stattools import adfuller, grangercausalitytests

OUT = Path(__file__).resolve().parent / ".data" / "studies" / "ts_diagnostics.json"


def _series(sid):
    from core.models import MacroSeries
    rows = MacroSeries.objects.filter(series=sid).values_list("date", "value")
    return pd.Series({pd.Timestamp(d): v for d, v in rows}).sort_index() if rows else pd.Series(dtype=float)


def build():
    etfs = {n: e for n, e in config.SECTOR_ETFS.items() if e not in CRYPTO}
    etf_tk = list(etfs.values())
    name_of = {e: n for n, e in etfs.items()}
    daily = load_candles(etf_tk + [BENCH])
    etf_m = _monthly_close({t: d for t, d in daily.items() if t in etf_tk})
    ret_m = etf_m.pct_change()
    spy_m = daily[BENCH]["Close"].resample("ME").last()
    spy_ret = spy_m.pct_change()

    # ---- (#4) SEASONALITY ----
    ret_m2 = ret_m.copy(); ret_m2["__month__"] = ret_m2.index.month
    all_by_month = ret_m.mean(axis=1)          # equal-weight sector return per month
    month_avg = all_by_month.groupby(all_by_month.index.month).mean() * 100
    seasonal = {int(m): round(float(v), 2) for m, v in month_avg.items()}
    # per-sector standout month (highest mean) among sectors with >=3y history
    sector_best = {}
    for e in etf_tk:
        s = ret_m[e].dropna()
        if len(s) < 36:
            continue
        bym = s.groupby(s.index.month).mean() * 100
        m = int(bym.idxmax())
        sector_best[name_of[e]] = {"best_month": m, "avg_ret_pct": round(float(bym.max()), 2)}
    top_seasonal = dict(sorted(sector_best.items(), key=lambda kv: kv[1]["avg_ret_pct"], reverse=True)[:8])

    # ---- (#5) STATIONARITY (ADF on log price; ADF on sector/SPY ratio) ----
    trending, meanrev, ratio_mr = [], [], []
    for e in etf_tk:
        p = etf_m[e].dropna()
        if len(p) < 40:
            continue
        try:
            pval = adfuller(np.log(p), maxlag=6, autolag="AIC")[1]
            (meanrev if pval < 0.05 else trending).append(name_of[e])
        except Exception:
            pass
        try:
            r = (etf_m[e] / spy_m).dropna()
            if len(r) >= 40 and adfuller(r, maxlag=6, autolag="AIC")[1] < 0.05:
                ratio_mr.append(name_of[e])         # sector/SPY ratio mean-reverts => relative-value candidate
        except Exception:
            pass

    # ---- (#2) GRANGER (macro -> market return) ----
    walcl = _series("WALCL").resample("ME").last()
    rrp = _series("RRPONTSYD").resample("ME").last()
    tga = _series("WTREGEN").resample("ME").last()
    net_liq = (walcl - rrp * 1000.0 - tga)
    macros = {
        "USD_chg": _series("DTWEXBGS").resample("ME").last().pct_change(),
        "HYspread_chg": _series("BAMLH0A0HYM2").resample("ME").last().diff(),
        "M2_yoy": _series("M2SL").resample("ME").last().pct_change(12),
        "netliq_chg": net_liq.pct_change(),
        "curve": _series("T10Y2Y").resample("ME").last(),
    }
    granger = {}
    for nm, x in macros.items():
        df = pd.concat([spy_ret.rename("y"), x.rename("x")], axis=1).dropna()
        if len(df) < 30:
            granger[nm] = None; continue
        try:
            res = grangercausalitytests(df[["y", "x"]], maxlag=3, verbose=False)
            pmin = min(res[l][0]["ssr_ftest"][1] for l in res)     # best p across lags 1-3
            best_lag = min(res, key=lambda l: res[l][0]["ssr_ftest"][1])
            granger[nm] = {"min_p": round(float(pmin), 4), "best_lag": int(best_lag),
                           "leads": bool(pmin < 0.05), "n": len(df)}
        except Exception as ex:
            granger[nm] = {"error": str(ex)[:60]}

    print("=== (#4) SEASONALITY — equal-weight sector return by calendar month (%) ===", flush=True)
    print("  " + "  ".join(f"{m}:{seasonal.get(m)}" for m in range(1, 13)), flush=True)
    best_m = max(seasonal, key=seasonal.get); worst_m = min(seasonal, key=seasonal.get)
    print(f"  best month {best_m} ({seasonal[best_m]}%)  worst month {worst_m} ({seasonal[worst_m]}%)", flush=True)
    print(f"  strongest sector-months: {json.dumps(top_seasonal)[:300]}", flush=True)
    print(f"\n=== (#5) STATIONARITY (ADF) ===\n  TRENDING (momentum-suited): {len(trending)} sectors", flush=True)
    print(f"  MEAN-REVERTING (reversal-suited): {len(meanrev)} -> {meanrev}", flush=True)
    print(f"  sector/SPY ratio mean-reverts (relative-value candidates): {len(ratio_mr)} -> {ratio_mr[:15]}", flush=True)
    print(f"\n=== (#2) GRANGER (macro -> SPY return; p<0.05 = leads) ===", flush=True)
    for nm, g in granger.items():
        print(f"  {nm:<14} {g}", flush=True)

    leaders = [nm for nm, g in granger.items() if isinstance(g, dict) and g.get("leads")]
    verdict = (
        f"Seasonality: best month {best_m} ({seasonal[best_m]}%), worst {worst_m} ({seasonal[worst_m]}%). "
        f"ADF: {len(trending)} trending vs {len(meanrev)} mean-reverting sectors; {len(ratio_mr)} sector/SPY "
        f"ratios mean-revert (pairs candidates). Granger: macro series that LEAD market returns (p<0.05): "
        + (", ".join(leaders) if leaders else "NONE at monthly lags 1-3") + ". "
        + ("Confirms a leading macro signal." if leaders else "No macro Granger-causes returns at monthly horizon "
           "(short sample; the regime EDGE was conditional means, not linear causality).")
    )
    print("\n" + verdict, flush=True)
    return {"computed_at": pd.Timestamp.utcnow().isoformat(),
            "seasonality_by_month_pct": seasonal, "strongest_sector_months": top_seasonal,
            "stationarity": {"trending_n": len(trending), "meanrev": meanrev, "ratio_meanrev": ratio_mr},
            "granger": granger, "verdict": verdict,
            "caveat": "Monthly data, ~63 months (short for Granger). ADF on log-price (unit root); sector/SPY ratio "
                      "ADF flags relative-value mean-reversion. Seasonality is in-sample averages (no OOS). Granger "
                      "= linear causality only. Diagnostics, not a backtest."}


def main():
    p = build()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(p, indent=2, default=str))
    try:
        from core.models import BacktestResult
        from django.utils import timezone
        BacktestResult.objects.update_or_create(
            kind="ts_diagnostics", defaults={"payload": json.loads(json.dumps(p, default=str)), "computed_at": timezone.now()})
        print("Saved BacktestResult[ts_diagnostics]", flush=True)
    except Exception as e:
        print("DB save failed:", e, flush=True)


if __name__ == "__main__":
    main()
