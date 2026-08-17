#!/usr/bin/env python3
"""Cut the high-vol dip-buy's drawdown to something tradeable (target < ~30%), keeping the return.

The -57% DD is ONE localized event: the 2021-11->2022-05 growth/rate-hike bear (a 6-month grind, not a
quick crash). It's mostly GROWTH BETA — and we were under-hedging (SPY 50%, stress-days only). A high-vol
book is tech/growth-heavy, so hedge with QQQ, CONTINUOUSLY (beta-neutralize), and add a SLOW-regime de-risk
(cut exposure while QQQ is in a sustained downtrend — the one case where de-risking helps, since this DD is
a grind not a V). Base = broad high-vol, vol_parity x conviction sizing. -> BacktestResult[h4_vol_ddfix].
Run: MSYS_NO_PATHCONV=1 docker exec rotation-backend-1 python -u /app/h4_vol_ddfix.py
"""
import os, json, warnings
from collections import deque, defaultdict
warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
import h4_c_enhance as E
import h4_vol_dipbuy as V

UNIT = E.UNIT
BASE_GROSS = 1.0
REGIME_MULT = 0.35        # size multiplier for NEW adds while QQQ is below its 100d MA (sustained downtrend)
BREAKER_ON, BREAKER_OFF, BREAKER_MULT = -20.0, -10.0, 0.5


def _qqq():
    """QQQ 4h bar returns {ts: r} + daily regime {date: above_100dMA bool} + daily 1d/3d stress reuse."""
    from seq_fundamental_study import load_candles
    from intraday_data import get_4h
    q4 = get_4h("QQQ", 5, True)
    bar = {}
    if q4 is not None:
        cc = q4["Close"]
        for ts, r in (cc / cc.shift(1) - 1).items():
            if pd.notna(r):
                bar[ts] = float(r)
    qd = load_candles(["QQQ"]).get("QQQ")
    regime = {}
    if qd is not None and len(qd) > 110:
        c = qd["Close"]; ma = c.rolling(100).mean()
        for ts, cl, m in zip(pd.to_datetime(qd.index), c.values, ma.values):
            if pd.notna(m):
                regime[ts.date()] = bool(cl > m)
    return bar, regime


def sim(trades, hbar, regime, sdaily, hedge_ratio, hedge_mode, regime_derisk, breaker):
    """hedge_mode: 'stress' (short on stress days) | 'always' (continuous beta hedge). regime_derisk: bool
    (shrink new adds while QQQ<100dMA). breaker: bool (half size while book DD < -20%)."""
    ebt = defaultdict(list); allts = set()
    for tr in trades:
        ebt[tr["entry_ts"]].append(tr); allts.add(tr["entry_ts"])
        for (tk, _) in tr["sched"]:
            allts.add(tk)
    timeline = sorted(allts)
    equity = peak = 1.0
    open_pos = []
    eq_ts, eq_val = [], []
    in_breaker = False
    for t in timeline:
        d = pd.Timestamp(t).date()
        stressed = E._stressed(sdaily, d)
        gross_now = sum(p["size"] for p in open_pos)
        scale = min(1.0, BASE_GROSS / gross_now) if gross_now > 0 else 1.0
        eff_gross = min(gross_now, BASE_GROSS)
        contrib = 0.0; still = []
        for p in open_pos:
            if p["sched"] and p["sched"][0][0] == t:
                contrib += p["size"] * scale * p["sched"].popleft()[1]
            if p["sched"]:
                still.append(p)
        open_pos = still
        # QQQ hedge leg
        if hedge_ratio and t in hbar:
            if hedge_mode == "always" or (hedge_mode == "stress" and stressed):
                contrib += -hedge_ratio * eff_gross * hbar[t]
        if contrib:
            equity *= (1 + contrib)
        peak = max(peak, equity)
        dd = (equity / peak - 1) * 100
        eq_ts.append(t); eq_val.append(equity)
        if breaker:
            if not in_breaker and dd <= BREAKER_ON:
                in_breaker = True
            elif in_breaker and dd >= BREAKER_OFF:
                in_breaker = False
        mult = BREAKER_MULT if (breaker and in_breaker) else 1.0
        if regime_derisk and not regime.get(d, True):     # QQQ below 100d MA -> sustained downtrend
            mult *= REGIME_MULT
        for tr in ebt.get(t, []):
            w = E._weight_of(tr, "vol_parity_x_conv", None)
            if w <= 0:
                continue
            open_pos.append({"size": w * UNIT * mult, "sched": deque(tr["sched"][:3])})
    eq = np.array(eq_val)
    rets = eq[1:] / eq[:-1] - 1
    total = round((eq[-1] - 1) * 100, 1)
    sd = rets.std(ddof=1) if len(rets) > 1 else 0
    sharpe = round(float(rets.mean() / sd * np.sqrt(E.C.BARS_PER_YEAR)), 2) if sd > 0 else 0.0
    peaks = np.maximum.accumulate(eq); dds = (eq - peaks) / peaks
    ti = int(dds.argmin()); pi = int(np.argmax(eq[:ti + 1])) if ti > 0 else 0
    return {"total_return_pct": total, "sharpe": sharpe, "max_dd_pct": round(float(dds.min() * 100), 1),
            "dd_start": str(eq_ts[pi])[:10], "dd_end": str(eq_ts[ti])[:10]}


def main():
    os.environ.setdefault("DJANGO_SETTINGS_MODULE", "rotation.settings")
    import django; django.setup()
    from pathlib import Path
    from h4_on_signals_study import candidate_windows
    allowedC, _ = candidate_windows("C")
    store = E.load_targets(); sectors = E._sector_map()
    sdaily, _spybar = E._spy(5)
    hbar, regime = _qqq()
    trades = V.collect("highvol", allowedC, store, sectors)
    print(f"high-vol trades {len(trades)}, QQQ 4h bars {len(hbar)}, regime days {len(regime)}", flush=True)

    VARIANTS = {
        "vp_conv (no new fix)":                dict(hedge_ratio=0.0, hedge_mode="stress", regime_derisk=False, breaker=False),
        "QQQ stress-hedge 0.5":                dict(hedge_ratio=0.5, hedge_mode="stress", regime_derisk=False, breaker=False),
        "QQQ CONTINUOUS 0.5":                  dict(hedge_ratio=0.5, hedge_mode="always", regime_derisk=False, breaker=False),
        "QQQ CONTINUOUS 1.0":                  dict(hedge_ratio=1.0, hedge_mode="always", regime_derisk=False, breaker=False),
        "regime de-risk (QQQ<100dMA)":         dict(hedge_ratio=0.0, hedge_mode="stress", regime_derisk=True, breaker=False),
        "DD breaker -20%":                     dict(hedge_ratio=0.0, hedge_mode="stress", regime_derisk=False, breaker=True),
        "COMBO: QQQ-cont0.75 + regime":        dict(hedge_ratio=0.75, hedge_mode="always", regime_derisk=True, breaker=False),
        "COMBO: QQQ-cont0.5 + regime + break": dict(hedge_ratio=0.5, hedge_mode="always", regime_derisk=True, breaker=True),
    }
    rows = []
    for name, kw in VARIANTS.items():
        m = sim(trades, hbar, regime, sdaily, **kw)
        rows.append({"variant": name, **m})
        print(f"  {name:38} total {m['total_return_pct']:>8}%  DD {m['max_dd_pct']:>7} @{m['dd_start']}->{m['dd_end']}  Sh {m['sharpe']:>5}", flush=True)

    payload = {"computed_at": pd.Timestamp.utcnow().isoformat(), "rows": rows, "n_trades": len(trades),
               "params": {"sizing": "vol_parity_x_conv", "regime_mult": REGIME_MULT,
                          "breaker": [BREAKER_ON, BREAKER_OFF, BREAKER_MULT]},
               "note": ("DD-control sweep on the broad high-vol dip-buy (vol_parity x conviction sizing). QQQ "
                        "hedge (stress vs CONTINUOUS beta-neutral), slow-regime de-risk (shrink adds while QQQ "
                        "< 100d MA), and a -20% DD breaker. Goal: DD < ~30% keeping return. Gross of fees.")}
    Path("/app/.data/studies").mkdir(parents=True, exist_ok=True)
    Path("/app/.data/studies/h4_vol_ddfix.json").write_text(json.dumps(payload, indent=2, default=str))
    try:
        from core.models import BacktestResult
        from django.utils import timezone
        BacktestResult.objects.update_or_create(kind="h4_vol_ddfix",
            defaults={"payload": json.loads(json.dumps(payload, default=str)), "computed_at": timezone.now()})
        print("saved BacktestResult[h4_vol_ddfix]", flush=True)
    except Exception as e:
        print("DB save failed:", e, flush=True)


if __name__ == "__main__":
    print("=== HIGH-VOL DD-CONTROL SWEEP ===", flush=True)
    main()
