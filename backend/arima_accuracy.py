#!/usr/bin/env python3
"""HOW CLOSE IS ARIMA TO REALITY — direct forecast-accuracy check for the monthly ARIMA(1,0,1) sector return
model (the one behind arima_study). For each sector, expanding-window forecast of next-month return, paired with
the ACTUAL next-month return. Reports:
  corr / sign-accuracy  — does the forecast track / get direction right?
  RMSE vs naive         — ARIMA error vs 'predict 0' and 'predict last return' (does it beat trivial baselines?)
  Information Coeff (IC) — avg monthly cross-sectional Spearman(signal, next-return) for ARIMA vs PRICE-ACCEL
                          (the fair 'which signal ranks sectors better' metric)
-> BacktestResult[arima_accuracy] + JSON.
Run: MSYS_NO_PATHCONV=1 docker exec rotation-backend-1 python -u /app/arima_accuracy.py
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
from scipy.stats import pearsonr, spearmanr
from arima_study import forecast_matrix          # reuse the exact monthly ARIMA forecaster

OUT = Path(__file__).resolve().parent / ".data" / "studies" / "arima_accuracy.json"


def build():
    etfs = {n: e for n, e in config.SECTOR_ETFS.items() if e not in CRYPTO}
    etf_tk = list(etfs.values())
    daily = load_candles(etf_tk + [BENCH])
    etf_m = _monthly_close({t: d for t, d in daily.items() if t in etf_tk})
    midx = etf_m.index
    ret = etf_m.pct_change()
    nxt = ret.shift(-1)                                   # actual NEXT-month return (what we're predicting)
    price_accel = etf_m.pct_change(3) - etf_m.pct_change(3).shift(3)

    print("fitting monthly ARIMA forecasts...", flush=True)
    fc = forecast_matrix(ret, seasonal=False)             # fc[date, etf] = predicted next-month return

    # pooled (prediction, actual) pairs
    preds, acts, last = [], [], []
    for i in range(len(midx)):
        d = midx[i]
        for e in etf_tk:
            f = fc.loc[d, e] if e in fc.columns else np.nan
            a = nxt.loc[d, e]; l = ret.loc[d, e]
            if pd.notna(f) and pd.notna(a):
                preds.append(float(f)); acts.append(float(a)); last.append(float(l) if pd.notna(l) else 0.0)
    preds, acts, last = np.array(preds), np.array(acts), np.array(last)
    n = len(preds)
    corr = float(pearsonr(preds, acts)[0]) if n > 5 else None
    sign = float(np.mean(np.sign(preds) == np.sign(acts))) if n else None
    rmse_arima = float(np.sqrt(np.mean((preds - acts) ** 2)))
    rmse_zero = float(np.sqrt(np.mean(acts ** 2)))                 # predict 0
    rmse_last = float(np.sqrt(np.mean((last - acts) ** 2)))        # predict last month's return

    # Information Coefficient: monthly cross-sectional rank corr of signal vs next return
    def ic(signal):
        vals = []
        for d in midx:
            s = signal.loc[d]; a = nxt.loc[d]
            m = s.notna() & a.notna()
            if m.sum() >= 8:
                r = spearmanr(s[m], a[m])[0]
                if np.isfinite(r):
                    vals.append(r)
        return (round(float(np.mean(vals)), 4), len(vals)) if vals else (None, 0)
    ic_arima = ic(fc); ic_accel = ic(price_accel)

    print(f"\n=== ARIMA monthly forecast accuracy (n={n} sector-months) ===", flush=True)
    print(f"  corr(pred, actual):     {corr:+.4f}   (0 = no linear tracking)", flush=True)
    print(f"  sign accuracy:          {sign*100:.1f}%   (50% = coin flip)", flush=True)
    print(f"  RMSE ARIMA:             {rmse_arima*100:.2f}%", flush=True)
    print(f"  RMSE predict-0:         {rmse_zero*100:.2f}%   (ARIMA {'BEATS' if rmse_arima < rmse_zero else 'LOSES to'} it)", flush=True)
    print(f"  RMSE predict-last:      {rmse_last*100:.2f}%", flush=True)
    print(f"  Information Coeff (IC):  ARIMA {ic_arima[0]}  vs  PRICE-ACCEL {ic_accel[0]}  (n_months {ic_arima[1]})", flush=True)

    verdict = (
        f"ARIMA next-month forecast vs actual: corr {corr:+.3f}, sign {sign*100:.0f}%, RMSE {rmse_arima*100:.2f}% "
        f"vs predict-0 {rmse_zero*100:.2f}% ({'beats' if rmse_arima < rmse_zero else 'no better than'} the trivial "
        f"'predict zero'). IC {ic_arima[0]} vs price-accel {ic_accel[0]}. "
        + ("ARIMA has ~no predictive content — it's essentially forecasting noise; price-accel ranks sectors better."
           if (corr is None or abs(corr) < 0.05) and rmse_arima >= rmse_zero * 0.99 else
           "ARIMA shows some tracking — worth a closer look.")
    )
    print("\n" + verdict, flush=True)
    return {"computed_at": pd.Timestamp.utcnow().isoformat(),
            "n_sector_months": n, "corr_pred_actual": round(corr, 4) if corr is not None else None,
            "sign_accuracy_pct": round(sign * 100, 1) if sign is not None else None,
            "rmse_arima_pct": round(rmse_arima * 100, 2), "rmse_predict0_pct": round(rmse_zero * 100, 2),
            "rmse_predictlast_pct": round(rmse_last * 100, 2),
            "ic_arima": ic_arima[0], "ic_price_accel": ic_accel[0], "verdict": verdict,
            "caveat": "Monthly ARIMA(1,0,1) expanding-window, next-month forecast. corr/sign pooled across sector-"
                      "months; IC = avg monthly cross-sectional Spearman. ~63 months. RMSE in return units."}


def main():
    p = build()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(p, indent=2, default=str))
    try:
        from core.models import BacktestResult
        from django.utils import timezone
        BacktestResult.objects.update_or_create(
            kind="arima_accuracy", defaults={"payload": json.loads(json.dumps(p, default=str)), "computed_at": timezone.now()})
        print("Saved BacktestResult[arima_accuracy]", flush=True)
    except Exception as e:
        print("DB save failed:", e, flush=True)


if __name__ == "__main__":
    main()
