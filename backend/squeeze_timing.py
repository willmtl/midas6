#!/usr/bin/env python3
"""Per-stock LONG/FLAT timing on the LazyBear Squeeze Momentum histogram (user: "try the squeeze momentum
indicator"). Signal = the momentum histogram `Plot`: GREEN (Plot>0) => long, RED (Plot<=0) => cash. Also a
stricter 'rising' variant (long only when Plot>0 AND increasing = lime). Traded per stock, enter the bar AFTER
the flip, 10bps/side. Compared to buy-&-hold each stock; EW portfolio gated to POINT-IN-TIME top-cap membership
(pit_top150_membership.json — no survivorship) vs SPY. Daily & weekly. Saves BacktestResult[squeeze_timing].
Run: MSYS_NO_PATHCONV=1 docker exec rotation-backend-1 python -u /app/squeeze_timing.py"""
import os, json, math, glob
os.environ.setdefault("DJANGO_SETTINGS_MODULE", "rotation.settings")
import django; django.setup()
import numpy as np, pandas as pd
from core.models import Candle

SQ_DIR = "/app/.data/tradingview/squeeze"
OUT = "/app/.data/studies/squeeze_timing.json"
COST = 10.0


def main():
    recs = []
    for f in sorted(glob.glob(SQ_DIR + "/*.json")):
        try:
            recs.append(json.load(open(f)))
        except Exception:
            pass
    tickers = [r["ticker"] for r in recs]
    print(f"loaded {len(recs)} squeeze files", flush=True)
    if not recs:
        print("no squeeze files yet", flush=True); return

    PITMEM = {}
    if os.path.exists("/app/.data/pit_top150_membership.json"):
        PITMEM = {k: set(v) for k, v in json.load(open("/app/.data/pit_top150_membership.json")).items()}

    from django.db import connection
    with connection.cursor() as cur:
        cur.execute("SET max_parallel_workers_per_gather = 0")
    close = {}
    for i in range(0, len(tickers + ['SPY']), 100):
        chunk = (tickers + ['SPY'])[i:i + 100]
        q = Candle.objects.filter(ticker__in=chunk, interval="1d", date__gte="2014-06-01").values_list("ticker", "date", "close")
        df = pd.DataFrame(list(q), columns=["ticker", "date", "close"])
        if df.empty:
            continue
        df["date"] = pd.to_datetime(df["date"]); df["close"] = df["close"].astype(float)
        for tk, g in df.groupby("ticker", sort=False):
            close[tk] = g.sort_values("date").set_index("date")["close"].sort_index()
    spy = close.get("SPY"); spy_ret = spy.pct_change()

    def state_series(rec, tf, mode, cidx):
        ev = rec.get(tf) or []
        dates, vals = [], []
        prev = None
        for row in sorted(ev, key=lambda x: x.get("t", 0)):
            t = row.get("t"); p = row.get("Plot")
            if t is None or p is None or (isinstance(p, float) and (p != p or abs(p) > 1e50)):
                continue
            if mode == "sign":
                s = 1 if p > 0 else -1
            else:  # rising: long only when positive AND increasing
                s = 1 if (p > 0 and prev is not None and p > prev) else -1
            dates.append(pd.Timestamp(int(t), unit="s").normalize()); vals.append(s); prev = p
        if not dates:
            return None
        ser = pd.Series(vals, index=pd.DatetimeIndex(dates))
        ser = ser[~ser.index.duplicated(keep="last")].sort_index()
        if tf == "weekly":
            ser = ser.shift(1)   # weekly bar is stamped at week-OPEN but its value is known only at week-CLOSE;
                                 # lag one full week so we never trade a week on its own (look-ahead) outcome
        return ser.reindex(cidx.union(ser.index)).ffill().reindex(cidx).fillna(-1)

    def pit_daily_mask(tk, cidx):
        if not PITMEM:
            return pd.Series(True, index=cidx)
        m = pd.Series(False, index=cidx)
        # month-end membership -> ffill to daily
        me = {pd.Timestamp(k): (tk in v) for k, v in PITMEM.items()}
        s = pd.Series(me).sort_index()
        return s.reindex(cidx.union(s.index)).ffill().reindex(cidx).fillna(False).astype(bool)

    def annual(r):
        r = r.dropna()
        if len(r) < 252:
            return {}
        eq = (1 + r).prod(); yrs = len(r) / 252.0
        return {"cagr_pct": float((eq ** (1 / yrs) - 1) * 100) if eq > 0 else -100.0,
                "sharpe": float(r.mean() / r.std() * math.sqrt(252)) if r.std() > 0 else 0.0,
                "maxdd_pct": float(((1 + r).cumprod() / (1 + r).cumprod().cummax() - 1).min() * 100)}

    res = {"n_files": len(recs), "cost_bps_per_side": COST, "arms": {}}
    print(f"\n=== Squeeze Momentum per-stock LONG/FLAT timing vs BUY-&-HOLD (PIT-gated portfolio, {COST}bps/flip) ===", flush=True)
    print(f"{'arm':>18} {'winVsBH%':>9} {'medDCAGR':>9} {'portCAGR':>9} {'portSh':>7} {'BHCAGR':>8} {'BHSh':>6} {'timeMkt':>8}", flush=True)
    for tf in ("daily", "weekly"):
        for mode in ("sign", "rising"):
            per = []; SR, BR, IV = [], [], []
            for rec in recs:
                tk = rec["ticker"]; c = close.get(tk)
                if c is None or c.notna().sum() < 500:
                    continue
                ret = c.pct_change()
                st = state_series(rec, tf, mode, c.index)
                if st is None:
                    continue
                pos = (st == 1).astype(float).shift(1).fillna(0.0)
                flips = pos.diff().abs().fillna(0.0)
                sret = pos * ret - flips * (COST / 1e4)
                a_s, a_b = annual(sret), annual(ret)
                if a_s and a_b:
                    per.append((a_s["cagr_pct"], a_b["cagr_pct"]))
                    pm = pit_daily_mask(tk, c.index)
                    SR.append((sret.where(pm)).rename(tk)); BR.append((ret.where(pm)).rename(tk)); IV.append((pos.where(pm)).rename(tk))
            if not per:
                continue
            SRd = pd.concat(SR, axis=1); BRd = pd.concat(BR, axis=1)
            port_s = SRd.mean(axis=1); port_b = BRd.mean(axis=1)
            win = float(np.mean([s > b for s, b in per]) * 100); med = float(np.median([s - b for s, b in per]))
            aps, apb = annual(port_s), annual(port_b)
            tim = float(pd.concat(IV, axis=1).mean(axis=1).mean() * 100)
            lab = f"{tf}/{mode}"
            res["arms"][lab] = {"n_stocks": len(per), "win_vs_bh_pct": round(win, 1), "median_dCAGR_pp": round(med, 2),
                                "port_strat": aps, "port_bh": apb, "time_in_market_pct": round(tim, 1)}
            print(f"{lab:>18} {win:>8.1f}% {med:>+8.2f} {aps.get('cagr_pct',0):>+8.1f}% {aps.get('sharpe',0):>6.2f} "
                  f"{apb.get('cagr_pct',0):>+7.1f}% {apb.get('sharpe',0):>5.2f} {tim:>7.1f}%", flush=True)

    res["spy"] = annual(spy_ret)
    print(f"\n  SPY buy&hold: CAGR {res['spy'].get('cagr_pct',0):+.1f}% / Sharpe {res['spy'].get('sharpe',0):.2f}", flush=True)
    print("  (portCAGR/Sh = EW portfolio of timed strategies, gated to PIT top-150 membership; BHCAGR = same-universe hold)", flush=True)

    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    open(OUT, "w").write(json.dumps(res, indent=2, default=float))
    try:
        from core.models import BacktestResult
        from django.utils import timezone
        BacktestResult.objects.update_or_create(kind="squeeze_timing", defaults={"payload": res, "computed_at": timezone.now()})
        print("Saved BacktestResult[squeeze_timing]", flush=True)
    except Exception as e:
        print("save skipped:", e, flush=True)


if __name__ == "__main__":
    main()
