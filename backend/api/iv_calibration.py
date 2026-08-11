"""IV calibration study — is implied vol a good predictor of the next-day move?

For each stock-day carrying an option snapshot, compare the IV-implied 1-sigma daily move
(atm_iv / sqrt(252)) to the ACTUAL realized next-day |move|. The ratio actual/implied reveals the
variance risk premium (IV runs hot: most days move far less than implied) and, per ticker, whose
options are most over-priced (low ratio) vs whose moves blow past IV (high ratio).

Aggregate + per-ticker; light single-pass. Reference: a well-calibrated normal has median
ratio ~0.67 and exceeds 1-sigma ~32% of days.
"""
import numpy as np
import pandas as pd
from collections import defaultdict

_SQRT_252 = np.sqrt(252.0)


def run_iv_calibration(min_days=100):
    """Return {aggregate:{...}, per_ticker:[...], n_tickers}. per_ticker sorted most-over-priced
    (lowest actual/implied ratio) first; only tickers with >= min_days observations."""
    from core.models import OptionSnapshot
    from api.tasks import _get_dfs
    rows = OptionSnapshot.objects.filter(atm_iv__isnull=False).values_list("ticker", "date", "atm_iv")
    ivd = defaultdict(list)
    for tk, d, iv in rows:
        ivd[tk].append((d, iv))
    dfs = _get_dfs(list(ivd.keys()))

    all_r = []
    per = []
    for tk, lst in ivd.items():
        df = dfs.get(tk)
        if df is None or len(df) < 3:
            continue
        close = df["Close"].astype(float)
        realized_next = (close.shift(-1) / close - 1.0).abs()          # |next-day move|, indexed by D
        ser = pd.Series({pd.Timestamp(d): float(iv) for d, iv in lst})
        ser = ser[~ser.index.duplicated()].reindex(close.index)
        implied = ser / 100.0 / _SQRT_252                              # IV-implied 1-sigma daily move
        m = implied.notna() & realized_next.notna() & (implied > 0)
        if not m.any():
            continue
        rr = (realized_next[m] / implied[m]).values
        rr = rr[np.isfinite(rr)]
        if len(rr) == 0:
            continue
        all_r.append(rr)
        if len(rr) >= min_days:
            per.append({
                "ticker": tk, "n": int(len(rr)),
                "median_ratio": round(float(np.median(rr)), 3),
                "pct_1sig": round(100 * float((rr > 1).mean()), 1),
                "avg_iv": round(float(ser[m].mean()), 1),
            })

    r = np.concatenate(all_r) if all_r else np.array([])
    agg = {}
    if len(r):
        agg = {
            "n_days": int(len(r)),
            "median_ratio": round(float(np.median(r)), 3),
            "mean_ratio": round(float(np.mean(r)), 3),
            "pct_1sig": round(100 * float((r > 1).mean()), 1),
            "pct_2sig": round(100 * float((r > 2).mean()), 1),
            "pct_half": round(100 * float((r < 0.5).mean()), 1),
        }
    per.sort(key=lambda x: x["median_ratio"])   # most over-priced (low ratio) first
    return {"aggregate": agg, "per_ticker": per, "n_tickers": len(per)}


def run_and_save(min_days=100):
    """Compute + upsert the single 'latest' IvCalibration row for the dashboard."""
    from core.models import IvCalibration
    res = run_iv_calibration(min_days=min_days)
    IvCalibration.objects.update_or_create(label="latest", defaults={
        "n_days": res["aggregate"].get("n_days", 0), "n_tickers": res["n_tickers"], "data": res})
    return {"n_days": res["aggregate"].get("n_days", 0), "n_tickers": res["n_tickers"]}
