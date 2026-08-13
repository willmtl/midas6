#!/usr/bin/env python3
"""MACRO REGIME → SECTOR LEADERSHIP. The one honest angle on "which sectors beat SPY, at the right time":
not price-momentum (which has ~0 lift), but macro REGIME.

Classify each month on three PIT price-based axes:
  - RATES:     TLT 3mo return < 0  -> rising  (bonds down = yields up)
  - INFLATION: TIP/TLT ratio 3mo change > 0 -> rising (TIPS beating nominal = breakevens up)
  - MARKET:    SPY > 200d MA -> risk-on
Then measure each sector's forward 3mo RELATIVE return (vs SPY) conditional on each regime state, rank the
leaders per state, report today's regime + the sectors that historically led in it.

CAVEAT (in payload): this is IN-SAMPLE historical leadership, not a walk-forward validated predictor — a
hypothesis generator, not a signal. Directional; no fees.
-> BacktestResult[regime] + JSON.
Run: MSYS_NO_PATHCONV=1 docker exec rotation-backend-1 python -u /app/regime_scan.py
"""
import os, json, warnings
warnings.filterwarnings("ignore")
os.environ.setdefault("DJANGO_SETTINGS_MODULE", "rotation.settings")
import django
django.setup()
import numpy as np, pandas as pd
import config
from backtest_lowpb import _monthly_close, BENCH, CRYPTO
from seq_fundamental_study import load_candles

FWD = 3          # forward months for the leadership measure
TOPN = 12


def _agg(s):
    a = s.dropna().values
    if not len(a):
        return {"mean_pct": None, "hit_pct": None, "n": 0}
    return {"mean_pct": round(float(a.mean()) * 100, 2), "hit_pct": round(float((a > 0).mean()) * 100, 1), "n": int(len(a))}


def build():
    etfs = [e for e in config.SECTOR_ETFS.values() if e and e not in CRYPTO]
    name_by = {e: n for n, e in config.SECTOR_ETFS.items()}
    data = load_candles(etfs + [BENCH])
    for req in ("TLT", "TIP"):
        if req not in data:
            raise RuntimeError(f"missing {req} for regime axes")
    em = _monthly_close({t: d for t, d in data.items() if t in etfs})
    midx = em.index
    spy = data[BENCH]["Close"].resample("ME").last().reindex(midx)

    # regime axes (monthly, PIT)
    tlt_m = data["TLT"]["Close"].resample("ME").last().reindex(midx)
    tip_m = data["TIP"]["Close"].resample("ME").last().reindex(midx)
    rates_up = tlt_m.pct_change(3) < 0
    infl_up = (tip_m / tlt_m).pct_change(3) > 0
    spy_d = data[BENCH]["Close"]
    bull_series = spy_d > spy_d.rolling(200).mean()
    bull_m = bull_series.reindex(midx, method="ffill").fillna(False).astype(bool)

    # forward 3mo relative return per sector (vs SPY)
    srel = (em.shift(-FWD) / em - 1).sub((spy.shift(-FWD) / spy - 1), axis=0)

    axes = {
        "rates": (rates_up, "rising", "falling"),
        "inflation": (infl_up, "rising", "falling"),
        "market": (bull_m, "risk-on", "risk-off"),
    }
    # valid months = flags known AND forward return known
    valid = rates_up.notna() & infl_up.notna() & srel.notna().any(axis=1)

    def leaders(mask):
        rows = []
        for e in etfs:
            if e not in srel.columns:
                continue
            a = _agg(srel.loc[mask & valid, e])
            if a["n"] >= 4:
                rows.append({"sector": name_by.get(e, e), "etf": e, **a})
        rows.sort(key=lambda r: -(r["mean_pct"] if r["mean_pct"] is not None else -1e9))
        return rows

    by_axis = {}
    for ax, (flag, up, dn) in axes.items():
        by_axis[ax] = {up: leaders(flag == True)[:TOPN], dn: leaders(flag == False)[:TOPN]}

    # current regime + leaders that historically led in ALL three current states (avg of the 3 conditional means)
    now = {"rates": bool(rates_up.iloc[-1]), "inflation": bool(infl_up.iloc[-1]), "market": bool(bull_m.iloc[-1]),
           "date": str(midx[-1].date())}
    m_now = {"rates": (rates_up == now["rates"]), "inflation": (infl_up == now["inflation"]),
             "market": (bull_m == now["market"])}
    scores = []
    for e in etfs:
        if e not in srel.columns:
            continue
        parts = [srel.loc[m & valid, e].mean() for m in m_now.values()]
        parts = [p for p in parts if np.isfinite(p)]
        if len(parts) < 3:
            continue
        hit_all = _agg(srel.loc[m_now["rates"] & m_now["inflation"] & m_now["market"] & valid, e])
        scores.append({"sector": name_by.get(e, e), "etf": e,
                       "regime_score_pct": round(float(np.mean(parts)) * 100, 2),
                       "combo_mean_pct": hit_all["mean_pct"], "combo_hit_pct": hit_all["hit_pct"], "combo_n": hit_all["n"]})
    scores.sort(key=lambda r: -r["regime_score_pct"])

    base = _agg(srel.loc[valid].stack())
    payload = {
        "computed_at": pd.Timestamp.utcnow().isoformat(),
        "params": {"forward_months": FWD, "benchmark": BENCH, "n_months": int(valid.sum()),
                   "axes": {"rates": "TLT 3mo return < 0 = rising", "inflation": "TIP/TLT 3mo change > 0 = rising",
                            "market": "SPY > 200d MA = risk-on"}},
        "now": now,
        "now_labels": {"rates": "rising" if now["rates"] else "falling",
                       "inflation": "rising" if now["inflation"] else "falling",
                       "market": "risk-on" if now["market"] else "risk-off"},
        "leaders_now": scores[:TOPN],
        "by_axis": by_axis,
        "base_rate": base,
        "caveat": ("IN-SAMPLE historical leadership (which sectors led in each regime), NOT a walk-forward "
                   "validated predictor — a hypothesis generator. Forward 3mo relative return vs SPY; the "
                   "base-rate mean is negative because most sectors lag SPY. Directional, no fees."),
    }
    return payload


def main():
    from pathlib import Path
    payload = build()
    out = Path(__file__).resolve().parent / ".data" / "studies" / "regime.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2, default=str))
    try:
        from core.models import BacktestResult
        from django.utils import timezone
        BacktestResult.objects.update_or_create(
            kind="regime", defaults={"payload": json.loads(json.dumps(payload, default=str)),
                                     "computed_at": timezone.now()})
        print("Saved BacktestResult[regime]", flush=True)
    except Exception as e:
        print("DB save failed:", e, flush=True)
    nl = payload["now_labels"]
    print(f"\n=== REGIME NOW ({payload['now']['date']}): rates {nl['rates']} | inflation {nl['inflation']} | market {nl['market']} ===", flush=True)
    b = payload["base_rate"]
    print(f"base rate (all sectors, fwd {payload['params']['forward_months']}mo rel): hit {b['hit_pct']}% mean {b['mean_pct']}%\n", flush=True)
    print("Sectors that historically LED in the current regime combo:", flush=True)
    for r in payload["leaders_now"]:
        print(f"  {r['sector']:26} {r['etf']:7} score {r['regime_score_pct']:>+6}%  combo(hit {r['combo_hit_pct']}% mean {r['combo_mean_pct']}% n{r['combo_n']})", flush=True)


if __name__ == "__main__":
    main()
