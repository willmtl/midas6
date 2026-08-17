#!/usr/bin/env python3
"""Does an EMA trend filter help the H4-on-C dip-buy? — decisive use-it-or-not test.

Question: gate the dip-buy by an EMA (only trade when price/market is on the right side of a moving average)?
Priors say no (the 200MA regime filter HALVED C's return — dip-buys work in bear tapes; entry-signal study:
confirmation SUBTRACTS), but EMA is faster, so test it. Filters each C dip-buy trade by an EMA condition
(name's own 4h EMA level/slope, and QQQ daily EMA regime) and re-runs the same config (steep_4x + hedge).
-> BacktestResult[h4_c_ema]. Run: MSYS_NO_PATHCONV=1 docker exec rotation-backend-1 python -u /app/h4_c_ema.py
"""
import os, json, warnings
warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
import h4_study as H
import h4_c_enhance as E
from h4_c_upside import load_targets, upside_asof, bucket_upside

SIGS = ["mr_rsi_os", "mr_newlow60", "mr_ndown"]


def _qqq_ema_regime(span=50):
    from seq_fundamental_study import load_candles
    qd = load_candles(["QQQ"]).get("QQQ")
    reg = {}
    if qd is not None and len(qd) > span + 5:
        c = qd["Close"]; ema = c.ewm(span=span).mean()
        for ts, cl, m in zip(pd.to_datetime(qd.index), c.values, ema.values):
            if pd.notna(m):
                reg[ts.date()] = bool(cl > m)
    return reg


def collect(allowedC, store, sectors, qqqreg):
    import intraday_data as ID
    trades = []
    for tk in sorted(allowedC):
        df = ID.get_4h(tk, 5, False)
        if df is None or len(df) < 120:
            continue
        close = df["Close"]
        c = close.values
        ts = df.index
        n = len(c)
        ema20 = close.ewm(span=20).mean().values
        ema50 = close.ewm(span=50).mean().values
        ad = allowedC[tk]
        fire = np.zeros(n, dtype=bool)
        for s in SIGS:
            e, _ = H.SIGNALS[s]["fn"](df); fire |= np.asarray(e, dtype=bool)
        idxs = sorted(H._episode_starts([i for i in range(n) if fire[i]], gap=H.GAP))
        for i in idxs:
            if i + 1 >= n or c[i] <= 0 or ts[i].date() not in ad or i < 51:
                continue
            up = upside_asof(store, tk, ts[i].date(), float(c[i]))
            sched = [(ts[i + b], float(c[i + b] / c[i + b - 1] - 1)) for b in range(1, E.MAXHOLD + 1) if i + b < n]
            if not sched:
                continue
            trades.append({"entry_ts": ts[i], "bucket": bucket_upside(up), "upside": up,
                           "sector": sectors.get(tk, tk), "sched": sched, "vol": 60.0,
                           "above_ema20": bool(c[i] > ema20[i]),
                           "ema20_rising": bool(ema20[i] > ema20[i - 3]),
                           "above_ema50": bool(c[i] > ema50[i]),
                           "qqq_up": qqqreg.get(ts[i].date(), True)})
    return trades


def main():
    os.environ.setdefault("DJANGO_SETTINGS_MODULE", "rotation.settings")
    import django; django.setup()
    from pathlib import Path
    from h4_on_signals_study import candidate_windows
    allowedC, _ = candidate_windows("C")
    store = load_targets(); sectors = E._sector_map()
    daily, spybar = E._spy(5)
    from h4_vol_ddfix import _qqq
    qqq_bar, _qreg = _qqq()
    qqqreg = _qqq_ema_regime(50)
    trades = collect(allowedC, store, sectors, qqqreg)
    print(f"C dip trades {len(trades)}", flush=True)

    CFG = {"weight": "steep_4x", "hold": E.HOLD_FIXED, "gate": False, "hedge_frac": 0.5}
    # --- C config: SPY hedge vs QQQ hedge (the user's ask) ---
    print("--- hedge index on C (steep4x, no EMA filter) ---", flush=True)
    hedge_rows = []
    for hn, hbar in [("SPY hedge", spybar), ("QQQ hedge", qqq_bar), ("no hedge", {})]:
        cfg = dict(CFG) if hbar else {"weight": "steep_4x", "hold": E.HOLD_FIXED, "gate": True}
        m = E.simulate(trades, daily, hbar, cfg)
        hedge_rows.append({"hedge": hn, **m})
        print(f"  {hn:12} total {m['total_return_pct']:>8}%  DD {m['max_dd_pct']:>7}  Sh {m['sharpe']:>5}", flush=True)

    HEDGE = qqq_bar          # EMA sweep uses the better (QQQ) hedge
    FILTERS = {
        "no EMA filter (baseline)":     lambda t: True,
        "name price > EMA20 (uptrend)": lambda t: t["above_ema20"],
        "name price < EMA20 (below)":   lambda t: not t["above_ema20"],
        "name EMA20 rising":            lambda t: t["ema20_rising"],
        "name EMA20 falling":           lambda t: not t["ema20_rising"],
        "name price > EMA50":           lambda t: t["above_ema50"],
        "QQQ > EMA50d (mkt uptrend)":   lambda t: t["qqq_up"],
        "QQQ < EMA50d (mkt downtrend)": lambda t: not t["qqq_up"],
    }
    print("--- EMA entry filters (steep4x + QQQ hedge) ---", flush=True)
    rows = []
    for name, pred in FILTERS.items():
        sub = [t for t in trades if pred(t)]
        m = E.simulate(sub, daily, HEDGE, CFG)
        rows.append({"filter": name, "n_pool": len(sub), **m})
        print(f"  {name:30} n{len(sub):>6}  total {m['total_return_pct']:>8}%  DD {m['max_dd_pct']:>7}  Sh {m['sharpe']:>5}  taken {m['n_taken']}", flush=True)

    payload = {"computed_at": pd.Timestamp.utcnow().isoformat(), "rows": rows, "hedge_rows": hedge_rows, "n_trades": len(trades),
               "note": ("EMA trend-filter test on the C value-pick H4 dip-buy (steep_4x + hedge). Each filter "
                        "keeps only trades passing an EMA condition (name 4h EMA20/50 level+slope, QQQ daily "
                        "EMA50 regime) then re-runs the same config. Verdict = does any EMA gate beat 'no "
                        "filter' on return/DD/Sharpe. Gross of fees.")}
    Path("/app/.data/studies").mkdir(parents=True, exist_ok=True)
    Path("/app/.data/studies/h4_c_ema.json").write_text(json.dumps(payload, indent=2, default=str))
    try:
        from core.models import BacktestResult
        from django.utils import timezone
        BacktestResult.objects.update_or_create(kind="h4_c_ema",
            defaults={"payload": json.loads(json.dumps(payload, default=str)), "computed_at": timezone.now()})
        print("saved BacktestResult[h4_c_ema]", flush=True)
    except Exception as e:
        print("DB save failed:", e, flush=True)


if __name__ == "__main__":
    print("=== EMA TREND-FILTER TEST on C dip-buy ===", flush=True)
    main()
