#!/usr/bin/env python3
"""COMMODITY RULE study. The proxy test proved momentum/ACCELERATION LOSES on commodities (buying UNG/USO/
PPLT when they accelerate = buying a spike that reverts). Commodities are macro/mean-reverting, not trend-
following. So test the OPPOSITE: does buying OVERSOLD commodities (most-negative trailing momentum) bounce?
And is any commodity rule worth a small diversifying sleeve on the flagship?

Ranks the commodity ETFs monthly by trailing momentum; reports forward 1-month return by momentum quintile
(bottom = oversold, top = accelerating) — reversion => bottom quintile has the HIGHER forward return.
-> .data/studies/commodity_rule.json + BacktestResult[commodity_rule].
Run: MSYS_NO_PATHCONV=1 docker exec rotation-backend-1 python -u /app/commodity_rule_study.py
"""
import os, json, warnings
warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
os.environ.setdefault("DJANGO_SETTINGS_MODULE", "rotation.settings")
import django
django.setup()
from pathlib import Path
from django.db import connection
from studies import _tstat_from_returns

OUT = Path("/app/.data/studies/commodity_rule.json")
COMMODITIES = ["USO", "UNG", "PPLT", "GLD", "SLV", "WEAT", "CORN", "DBA", "DBC", "DBO", "BNO", "UGA",
               "PALL", "CPER", "JO", "SGG", "NIB", "CANE", "SOYB", "WOOD", "URA"]


def main():
    with connection.cursor() as c:
        c.execute("SET max_parallel_workers_per_gather=0")
    from core.models import Candle, BacktestResult
    from django.utils import timezone

    cols = {}
    for tk in COMMODITIES:
        q = list(Candle.objects.filter(ticker=tk, interval="1d").order_by("date").values_list("date", "close"))
        if len(q) > 200:
            cols[tk] = pd.Series({pd.Timestamp(d): float(c) for d, c in q}).sort_index().resample("ME").last()
    spy = pd.Series({pd.Timestamp(d): float(c) for d, c in
                     Candle.objects.filter(ticker="SPY", interval="1d").order_by("date").values_list("date", "close")}).sort_index().resample("ME").last()
    px = pd.DataFrame(cols)
    midx = px.index
    print(f"commodity ETFs with data: {list(px.columns)} ({px.shape[1]}) x {len(midx)} months", flush=True)

    mom6 = px.pct_change(6)
    mom3 = px.pct_change(3)
    accel = px.pct_change(3) - px.pct_change(3).shift(3)
    fwd = px.shift(-1) / px - 1                       # forward 1-month return

    # forward return by MOMENTUM quintile (pooled across months)
    def quint_report(signal, label, out):
        rows = {q: [] for q in range(5)}
        for dt in midx:
            s = signal.loc[dt].dropna()
            f = fwd.loc[dt]
            s = s[[t for t in s.index if pd.notna(f.get(t))]]
            if len(s) < 5:
                continue
            qs = pd.qcut(s.rank(method="first"), 5, labels=False)
            for t in s.index:
                rows[int(qs[t])].append(float(f[t]))
        print(f"\n=== forward 1mo return by {label} quintile (Q0=lowest/oversold .. Q4=highest/accelerating) ===", flush=True)
        qo = {}
        for q in range(5):
            r = np.asarray(rows[q], float)
            if len(r) < 10:
                continue
            t = _tstat_from_returns(list(r))
            qo[f"Q{q}"] = {"n": len(r), "fwd": round(float(r.mean()) * 100, 2), "t": round(t, 2) if t else None}
            print(f"  Q{q}  n={len(r):>4}  fwd {r.mean()*100:>+6.2f}%  t {round(t,2) if t else 0}", flush=True)
        out[label] = qo
        if "Q0" in qo and "Q4" in qo:
            spread = qo["Q0"]["fwd"] - qo["Q4"]["fwd"]
            print(f"  Q0-Q4 (oversold − accelerating): {spread:+.2f}pp  -> "
                  f"{'MEAN-REVERSION (buy oversold)' if spread > 0.3 else ('MOMENTUM (buy strength)' if spread < -0.3 else 'flat')}", flush=True)
            out[label + "_Q0_minus_Q4"] = round(spread, 2)

    out = {"computed_at": pd.Timestamp.utcnow().isoformat(), "commodities": list(px.columns), "months": len(midx)}
    quint_report(mom6, "6mo_momentum", out)
    quint_report(mom3, "3mo_momentum", out)
    quint_report(accel, "acceleration", out)

    # simple sleeve backtests: each month hold the single most-oversold vs most-accelerating commodity, equal 1mo
    def sleeve(signal, pick_low):
        rets, spies = [], []
        for i in range(len(midx) - 1):
            dt, nd = midx[i], midx[i + 1]
            s = signal.loc[dt].dropna()
            f = fwd.loc[dt]
            s = s[[t for t in s.index if pd.notna(f.get(t))]]
            if len(s) < 5:
                continue
            picks = s.nsmallest(3).index if pick_low else s.nlargest(3).index    # 3 most oversold / accelerating
            rets.append(float(f[picks].mean()))
            sp = spy.get(nd) / spy.get(dt) - 1 if pd.notna(spy.get(nd)) and pd.notna(spy.get(dt)) else np.nan
            spies.append(float(sp) if np.isfinite(sp) else 0.0)
        r = np.asarray(rets); sp = np.asarray(spies)
        tot = float(np.prod(1 + r) - 1) * 100
        sh = float(r.mean() / r.std() * np.sqrt(12)) if r.std() > 1e-9 else 0
        return round(tot, 1), round(sh, 2), round(float(np.prod(1 + sp) - 1) * 100, 1)
    print("\n=== commodity SLEEVE backtests (hold top-3 by rule, monthly, vs SPY) ===", flush=True)
    for lab, sig, low in [("oversold 6mo (mean-rev)", mom6, True), ("accelerating (momentum)", accel, False),
                          ("oversold accel", accel, True), ("strongest 6mo", mom6, False)]:
        tot, sh, spt = sleeve(sig, low)
        out.setdefault("sleeves", {})[lab] = {"total": tot, "sharpe": sh, "spy": spt}
        print(f"  {lab:26} total {tot:>8.1f}%  Sharpe {sh:>5}  (SPY {spt}%)", flush=True)

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(out, indent=2, default=str))
    try:
        BacktestResult.objects.update_or_create(kind="commodity_rule",
            defaults={"payload": json.loads(json.dumps(out, default=str)), "computed_at": timezone.now()})
        print("\nSaved BacktestResult[commodity_rule]", flush=True)
    except Exception as e:
        print("DB save failed:", e, flush=True)
    print("DONE_COMMODITY", flush=True)


if __name__ == "__main__":
    main()
