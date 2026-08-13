#!/usr/bin/env python3
"""DIVERSIFIERS — rank all 93 sector sleeves by how much they DIVERSIFY SPY (low correlation), not just
by return. Sector rotation can't beat SPY, but an uncorrelated sleeve improves risk-adjusted return /
drawdown. Pure commodities (esp. GOLD) are the standouts: near-zero correlation, and Gold also appreciated.

Per sleeve (5y): total return, vs SPY, daily-return correlation & beta to SPY, ann vol, commodity tag, and a
"good diversifier" flag (corr < 0.35 AND positive 5y return). -> BacktestResult[diversifier] + JSON.
Run: MSYS_NO_PATHCONV=1 docker exec rotation-backend-1 python -u /app/diversifier_scan.py
"""
import os, json, warnings
warnings.filterwarnings("ignore")
os.environ.setdefault("DJANGO_SETTINGS_MODULE", "rotation.settings")
import django
django.setup()
import numpy as np, pandas as pd
import config
from seq_fundamental_study import load_candles

BENCH = "SPY"
KW = ("gold", "silver", "oil", "gas", "copper", "wheat", "agri", "commodit", "lithium",
      "uranium", "steel", "lumber", "coal", "platinum", "palladium", "corn", "energy", "rare earth")
LOWCORR = 0.35


def is_commodity(name):
    n = name.lower()
    return any(k in n for k in KW)


def build():
    etfs = {n: e for n, e in config.SECTOR_ETFS.items() if e}
    data = load_candles(list(etfs.values()) + [BENCH])
    spy = data[BENCH]["Close"]
    spy_ret = np.log(spy / spy.shift(1)).dropna()
    spy_5y = round(float(spy.iloc[-1] / spy.iloc[0] - 1) * 100, 1)

    rows = []
    for name, e in etfs.items():
        df = data.get(e)
        if df is None or len(df) < 250:
            continue
        c = df["Close"]
        r = np.log(c / c.shift(1)).dropna()
        al = pd.DataFrame({"e": r, "s": spy_ret}).dropna()
        if len(al) < 250:
            continue
        corr = round(float(al["e"].corr(al["s"])), 2)
        beta = round(float(np.cov(al["e"], al["s"])[0, 1] / np.var(al["s"])), 2)
        ret5y = round(float(c.iloc[-1] / c.iloc[0] - 1) * 100, 1)
        rows.append({"name": name, "etf": e, "commodity": is_commodity(name),
                     "ret5y": ret5y, "vs_spy": round(ret5y - spy_5y, 1),
                     "corr": corr, "beta": beta,
                     "ann_vol": round(float(r.std() * np.sqrt(252)) * 100, 1),
                     "good_diversifier": bool(corr < LOWCORR and ret5y > 0)})

    rows.sort(key=lambda x: x["corr"])
    comm = [r for r in rows if r["commodity"]]
    divs = [r for r in rows if r["good_diversifier"]]
    payload = {
        "computed_at": pd.Timestamp.utcnow().isoformat(),
        "params": {"benchmark": BENCH, "spy_5y_return": spy_5y, "n_sleeves": len(rows),
                   "lowcorr_threshold": LOWCORR,
                   "rule": "rank by correlation to SPY (low = diversifier); good_diversifier = corr<0.35 AND ret5y>0"},
        "rows": rows,
        "summary": {
            "spy_5y": spy_5y,
            "n_beat_spy": sum(1 for r in rows if r["vs_spy"] > 0),
            "n_sleeves": len(rows),
            "n_commodity": len(comm),
            "commodity_beat_spy": sum(1 for r in comm if r["vs_spy"] > 0),
            "avg_commodity_corr": round(float(np.mean([r["corr"] for r in comm])), 2) if comm else None,
            "good_diversifiers": [r["name"] for r in divs],
        },
        "note": ("Commodities don't reliably beat SPY on return, but PURE ones are uncorrelated — that is the "
                 "value (drawdown control), not outperformance. Gold is the standout (beat SPY AND corr ~0.15). "
                 "Miner/producer ETFs correlate like stocks (beta>1) and are already in the value-pick universe. "
                 "5y daily returns; directional."),
    }
    return payload


def main():
    from pathlib import Path
    payload = build()
    out = Path(__file__).resolve().parent / ".data" / "studies" / "diversifier.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2, default=str))
    try:
        from core.models import BacktestResult
        from django.utils import timezone
        BacktestResult.objects.update_or_create(
            kind="diversifier",
            defaults={"payload": json.loads(json.dumps(payload, default=str)),
                      "computed_at": timezone.now()})
        print("Saved BacktestResult[diversifier]", flush=True)
    except Exception as e:
        print("DB save failed:", e, flush=True)
    s = payload["summary"]
    print(f"\n=== DIVERSIFIERS (SPY 5y {s['spy_5y']:+}%) — {s['n_beat_spy']}/{s['n_sleeves']} beat SPY ===", flush=True)
    print(f"{'sleeve':26} {'etf':7} {'ret5y%':>7} {'vsSPY%':>7} {'corr':>5} {'beta':>5} {'div?':>5}", flush=True)
    for r in payload["rows"][:22]:
        print(f"{r['name']:26} {r['etf']:7} {r['ret5y']:>7} {r['vs_spy']:>7} {r['corr']:>5} {r['beta']:>5} "
              f"{'YES' if r['good_diversifier'] else '':>5}", flush=True)
    print(f"\ngood diversifiers (corr<{LOWCORR}, +ret): {', '.join(s['good_diversifiers'])}", flush=True)


if __name__ == "__main__":
    main()
