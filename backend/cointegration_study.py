#!/usr/bin/env python3
"""COINTEGRATION (#3) — Engle-Granger test across all sector-ETF pairs to find pairs whose PRICES move together
long-run (a stationary linear combination), i.e. relative-value / pairs-trade candidates. Complements the ADF
finding that 9 sector/SPY ratios mean-revert. NOTE: pairs trading is market-NEUTRAL (long one / short the other),
so it's a LOWER-absolute-return, diversifying source — less aligned with [[return-priority]] — reported as a
diagnostic (the candidate pairs), not wired.
-> BacktestResult[cointegration] + JSON.
Run: MSYS_NO_PATHCONV=1 docker exec rotation-backend-1 python -u /app/cointegration_study.py
"""
import os, json, warnings, itertools
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
from statsmodels.tsa.stattools import coint

OUT = Path(__file__).resolve().parent / ".data" / "studies" / "cointegration.json"


def build():
    etfs = {n: e for n, e in config.SECTOR_ETFS.items() if e not in CRYPTO}
    name_of = {e: n for n, e in etfs.items()}
    etf_tk = list(etfs.values())
    daily = load_candles(etf_tk)
    # use daily log-prices (more points -> a real cointegration test); aligned, full-history only
    px = {}
    for t in etf_tk:
        d = daily.get(t)
        if d is not None and len(d) > 400:
            px[t] = np.log(d["Close"])
    pxdf = pd.DataFrame(px).dropna()
    cols = list(pxdf.columns)
    print(f"cointegration over {len(cols)} sectors x {len(pxdf)} daily bars ({len(cols)*(len(cols)-1)//2} pairs)...", flush=True)

    pairs = []
    for a, b in itertools.combinations(cols, 2):
        try:
            t, p, _ = coint(pxdf[a], pxdf[b])
            if np.isfinite(p):
                corr = float(pxdf[a].corr(pxdf[b]))
                pairs.append({"a": name_of[a], "b": name_of[b], "pval": round(float(p), 4),
                              "tstat": round(float(t), 2), "corr": round(corr, 2)})
        except Exception:
            continue
    pairs.sort(key=lambda x: x["pval"])
    sig = [p for p in pairs if p["pval"] < 0.05]
    strong = [p for p in pairs if p["pval"] < 0.01]

    print(f"\n=== COINTEGRATED SECTOR PAIRS (Engle-Granger p<0.05: {len(sig)}; p<0.01: {len(strong)}) ===", flush=True)
    print(f"  {'sector A':<22}{'sector B':<22}{'pval':>7}{'tstat':>7}{'corr':>6}", flush=True)
    for p in pairs[:18]:
        print(f"  {p['a']:<22}{p['b']:<22}{p['pval']:>7}{p['tstat']:>7}{p['corr']:>6}", flush=True)

    verdict = (
        f"{len(sig)}/{len(pairs)} sector pairs cointegrated at p<0.05 ({len(strong)} at p<0.01). "
        f"Top pair: {pairs[0]['a']} ~ {pairs[0]['b']} (p={pairs[0]['pval']}). "
        "These are relative-value / pairs candidates (long the cheap leg, short the rich leg when the spread "
        "diverges). Market-NEUTRAL -> diversifying but LOWER absolute return; informational under return-priority, "
        "not wired. Multiple-testing caveat: ~4000 pairs tested, expect ~200 false positives at p<0.05."
    )
    print("\n" + verdict, flush=True)
    return {"computed_at": pd.Timestamp.utcnow().isoformat(),
            "n_pairs": len(pairs), "n_sig_p05": len(sig), "n_strong_p01": len(strong),
            "top_pairs": pairs[:25], "verdict": verdict,
            "caveat": "Engle-Granger coint on daily log-prices, full-history-aligned sectors only. ~4000 pairs -> "
                      "MULTIPLE-TESTING: expect ~5% (200) false positives at p<0.05; treat individual pairs as "
                      "candidates needing OOS spread-stability, not confirmed. Pairs trading market-neutral (low "
                      "abs return). No pairs backtest run (diagnostic only)."}


def main():
    p = build()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(p, indent=2, default=str))
    try:
        from core.models import BacktestResult
        from django.utils import timezone
        BacktestResult.objects.update_or_create(
            kind="cointegration", defaults={"payload": json.loads(json.dumps(p, default=str)), "computed_at": timezone.now()})
        print("Saved BacktestResult[cointegration]", flush=True)
    except Exception as e:
        print("DB save failed:", e, flush=True)


if __name__ == "__main__":
    main()
