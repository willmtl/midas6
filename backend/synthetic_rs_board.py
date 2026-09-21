#!/usr/bin/env python3
"""Build the SYNTHETIC relative-strength series for EVERY flagship sector vs SPY (sector/SPY), plus QQQ/SPY as a
growth-vs-market regime gauge (user 2026-09-21). For each: daily + weekly ratio, RSI(14) on the ratio, current level,
52-week RS percentile, distance vs the ratio's 200d MA, and whether the deep-oversold criterion (RSI<20/<30) or
overbought (>70/80) is firing NOW. Ranked most-oversold-vs-SPY first (the flagship RS-criteria candidates).
Saves /app/.data/studies/synthetic_rs_board.json (current diagnostics + recent 260d of each ratio for charting) +
BacktestResult[synthetic_rs_board]. Run: MSYS_NO_PATHCONV=1 docker exec rotation-backend-1 python -u /app/synthetic_rs_board.py"""
import os, json
os.environ.setdefault("DJANGO_SETTINGS_MODULE", "rotation.settings")
import django; django.setup()
import numpy as np, pandas as pd
from core.models import BacktestResult
import config
from synthetic_ratio_study import wilder_rsi
from seq_fundamental_study import load_candles

OUT = "/app/.data/studies/synthetic_rs_board.json"


def diag(ratio):
    """Current RS diagnostics for one ratio series (daily or weekly bars)."""
    r = ratio.dropna()
    if len(r) < 60:
        return None
    rsi = wilder_rsi(r, 14)
    win = 252 if len(r) >= 252 else len(r)              # ~52wk on daily; on weekly this is 'all' (few yrs)
    lo, hi = float(r.iloc[-win:].min()), float(r.iloc[-win:].max())
    pct = round((float(r.iloc[-1]) - lo) / (hi - lo) * 100, 1) if hi > lo else None
    ma200 = r.rolling(200, min_periods=50).mean().iloc[-1]
    cur = float(rsi.iloc[-1])
    # recent cross-up through oversold within last 10 bars
    def _crossed(thr):
        x = (rsi.shift(1) < thr) & (rsi >= thr)
        return bool(x.iloc[-10:].any())
    return dict(rsi=round(cur, 1),
                state=("oversold<20" if cur < 20 else "oversold<30" if cur < 30 else
                       "overbought>80" if cur > 80 else "overbought>70" if cur > 70 else "neutral"),
                pct52=pct, vs200ma=round((float(r.iloc[-1]) / float(ma200) - 1) * 100, 1) if pd.notna(ma200) and ma200 else None,
                cross20_recent=_crossed(20), cross30_recent=_crossed(30),
                trend=("up" if pd.notna(ma200) and r.iloc[-1] > ma200 else "down"))


def weekly(c):
    return c.resample("W-FRI").last().dropna()


def main():
    pairs = dict(config.SECTOR_ETFS)                    # {name: ETF} — all flagship sleeves
    need = sorted(set(pairs.values()) | {"SPY", "QQQ"})
    raw = load_candles(need)
    px = {t: d["Close"] for t, d in raw.items() if d is not None and len(d) > 260}
    if "SPY" not in px:
        raise SystemExit("no SPY candles")
    spy = px["SPY"]
    asof = str(spy.index[-1].date())
    print(f"loaded {len(px)} series; asof {asof}", flush=True)

    def build(name, etf):
        if etf not in px:
            return None
        c = px[etf]; idx = c.index.intersection(spy.index)
        rd = (c.reindex(idx) / spy.reindex(idx)).dropna()
        if len(rd) < 260:
            return None
        rec = rd.iloc[-260:]                            # recent year of the ratio for charting
        return dict(name=name, etf=etf, daily=diag(rd), weekly=diag(weekly(rd)),
                    series={"dates": [str(d.date()) for d in rec.index], "ratio": [round(float(x), 5) for x in rec.values]})

    sectors = [b for b in (build(n, e) for n, e in pairs.items()) if b and b["daily"]]
    qs = build("QQQ / SPY (growth regime)", "QQQ")
    # rank most-oversold-vs-SPY first (lowest daily RSI on the ratio)
    sectors.sort(key=lambda s: s["daily"]["rsi"])
    out = {"asof": asof, "qqq_spy": qs, "sectors": sectors}
    json.dump(out, open(OUT, "w"))

    def _line(s):
        d = s["daily"]; w = s["weekly"] or {}
        return (f"  {s['name'][:26]:26} {s['etf']:6}  RSI(d)={d['rsi']:>5} {d['state']:<13} "
                f"52pct={str(d['pct52']):>5}  vs200={str(d['vs200ma']):>6}%  RSI(w)={str((w or {}).get('rsi')):>5}  "
                f"{'X20' if d['cross20_recent'] else '   '} {'x30' if d['cross30_recent'] else ''}")
    print(f"\n=== QQQ/SPY regime: RSI(d)={qs['daily']['rsi']} {qs['daily']['state']} 52pct={qs['daily']['pct52']} "
          f"vs200={qs['daily']['vs200ma']}% trend={qs['daily']['trend']} | RSI(w)={qs['weekly']['rsi'] if qs['weekly'] else None} ===", flush=True)
    print(f"\n=== MOST OVERSOLD vs SPY (top 15) — flagship RS-criteria candidates ===", flush=True)
    for s in sectors[:15]:
        print(_line(s), flush=True)
    print(f"\n=== MOST OVERBOUGHT vs SPY (top 8) ===", flush=True)
    for s in sectors[-8:]:
        print(_line(s), flush=True)
    print(f"\n{len(sectors)} sectors. X20/x30 = crossed UP through RSI 20/30 on the ratio within last 10 days.", flush=True)
    try:
        from django.utils import timezone
        BacktestResult.objects.update_or_create(kind="synthetic_rs_board", defaults=dict(payload=out, computed_at=timezone.now()))
        print("saved BacktestResult[synthetic_rs_board] +", OUT, flush=True)
    except Exception as e:
        print("save skipped:", e, flush=True)


if __name__ == "__main__":
    main()
