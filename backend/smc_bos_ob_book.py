#!/usr/bin/env python3
"""MARKET STRUCTURE BREAK (BOS) & ORDER BLOCK (OB) as tradeable signals (user request). From the LuxAlgo SMC
series (PIT-topcap universe). Monthly, look-ahead-safe: a signal FIRES in month M (any day, from the DAILY event
series -> known at month-end), and we go long EW the next month (hold H), PIT-membership-gated, costed 20bps/side,
vs SPY and vs the equal-weight PIT-topcap control. Bullish events for the long test; bearish reported for contrast.
Saves BacktestResult[smc_bos_ob_book]. Run: docker exec rotation-backend-1 python -u /app/smc_bos_ob_book.py"""
import os, json, math, glob
os.environ.setdefault("DJANGO_SETTINGS_MODULE", "rotation.settings")
import django; django.setup()
import numpy as np, pandas as pd
from core.models import Candle

SMC_DIR = "/app/.data/tradingview/smc"
OUT = "/app/.data/studies/smc_bos_ob_book.json"
SIGNALS = {
    "MSB bull (swing BOS)": "Bullish_BOS",
    "MSB bull (internal BOS)": "Internal_Bullish_BOS",
    "OB bull (swing breakout)": "Bullish_Swing_OB_Breakout",
    "OB bull (internal breakout)": "Bullish_Internal_OB_Breakout",
    "MSB bear (swing BOS)": "Bearish_BOS",
    "OB bear (swing breakout)": "Bearish_Swing_OB_Breakout",
}


def nz(v):
    return v is not None and v == v and v != 0 and not (isinstance(v, (int, float)) and abs(v) > 1e50)


def main():
    recs = []
    for f in sorted(glob.glob(SMC_DIR + "/*.json")):
        try:
            recs.append(json.load(open(f)))
        except Exception:
            pass
    tickers = [r["ticker"] for r in recs]
    print(f"loaded {len(recs)} SMC files", flush=True)
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
    close = pd.DataFrame(close).sort_index()
    spy_m = close["SPY"].resample("ME").last()
    midx = spy_m.index[spy_m.index >= "2016-01-01"]
    cm = close.resample("ME").last().reindex(midx)
    fret = cm.shift(-1) / cm - 1.0
    spy_fwd = spy_m.reindex(midx).shift(-1) / spy_m.reindex(midx) - 1.0

    # monthly event-fired flag per name from the DAILY series (any fire in the month = known at month-end)
    def month_flags(ev):
        flags = {}
        for r in recs:
            tk = r["ticker"]
            if tk not in cm.columns:
                continue
            days = [pd.Timestamp(int(row["t"]), unit="s").normalize() for row in (r.get("daily") or []) if row.get("t") is not None and nz(row.get(ev))]
            if not days:
                continue
            s = pd.Series(1, index=pd.DatetimeIndex(days))
            flags[tk] = (s.resample("ME").sum() > 0).reindex(midx).fillna(False)
        return pd.DataFrame(flags, index=midx).reindex(columns=[t for t in tickers if t in cm.columns]).fillna(False)

    def pit_ok(t, d):
        if not PITMEM:
            return True
        k = str(d.date())
        return k in PITMEM and t in PITMEM[k]

    def book(flag_df, hold=1, cost_bps=20.0):
        rets, prev = [], set()
        for i, d in enumerate(midx[:-hold]):
            fr = (cm.shift(-hold) / cm - 1.0).loc[d]
            names = [t for t in flag_df.columns if pd.notna(fr[t]) and pit_ok(t, d) and bool(flag_df.loc[d, t])]
            if len(names) < 5:
                rets.append(np.nan); prev = set(); continue
            cur = set(names); turn = len(cur ^ prev) / (2 * max(len(cur), 1))
            rets.append(float(fr[names].mean()) / hold - turn * 2 * cost_bps / 1e4); prev = cur
        return pd.Series(rets, index=midx[:-hold])

    def ctrl_book(hold=1):
        rets = []
        for i, d in enumerate(midx[:-hold]):
            fr = (cm.shift(-hold) / cm - 1.0).loc[d]
            names = [t for t in cm.columns if t != "SPY" and pd.notna(fr[t]) and pit_ok(t, d)]
            rets.append(float(fr[names].mean()) / hold if len(names) >= 5 else np.nan)
        return pd.Series(rets, index=midx[:-hold])

    def stats(pr, bench=None):
        pr = pr.dropna()
        if len(pr) < 12:
            return {"n": int(len(pr))}
        eq = np.cumprod(1 + pr.values); yrs = len(pr) / 12.0
        o = {"n": int(len(pr)), "cagr_pct": float((eq[-1] ** (1 / yrs) - 1) * 100) if eq[-1] > 0 else -100.0,
             "sharpe": float(pr.mean() / pr.std() * math.sqrt(12)) if pr.std() > 0 else 0.0,
             "maxdd_pct": float((eq / np.maximum.accumulate(eq) - 1).min() * 100), "avg_names": None}
        if bench is not None:
            b = bench.reindex(pr.index).fillna(0.0)
            o["vs_spy_pp"] = float(((eq[-1] - 1) - (np.cumprod(1 + b.values)[-1] - 1)) * 100)
        return o

    res = {"n_files": len(recs), "window": [str(midx[0].date()), str(midx[-1].date())], "arms": {}}
    ctrl = {h: stats(ctrl_book(h), bench=spy_fwd) for h in (1, 3)}
    spq = stats(spy_fwd.reindex(midx[:-1]))
    res["control"] = ctrl; res["spy"] = spq
    print(f"\n=== BOS / OB monthly book (PIT-gated, costed 20bps, {midx[0].date()}..{midx[-1].date()}) ===", flush=True)
    print(f"  SPY CAGR {spq.get('cagr_pct',0):+.1f}%/Sh{spq.get('sharpe',0):.2f} | control(hold1) CAGR {ctrl[1].get('cagr_pct',0):+.1f}%/Sh{ctrl[1].get('sharpe',0):.2f} (vsSPY {ctrl[1].get('vs_spy_pp',0):+.0f})", flush=True)
    print(f"{'signal':>30} {'hold':>4} {'CAGR':>7} {'Sh':>5} {'DD':>7} {'vsSPY':>7} {'vsCtrl':>7}", flush=True)
    for lab, ev in SIGNALS.items():
        fd = month_flags(ev)
        res["arms"][lab] = {}
        for h in (1, 3):
            st = stats(book(fd, hold=h), bench=spy_fwd)
            res["arms"][lab][f"hold{h}"] = st
            vc = st.get("cagr_pct", 0) - ctrl[h].get("cagr_pct", 0)
            print(f"{lab:>30} {h:>4} {st.get('cagr_pct',0):>+6.1f}% {st.get('sharpe',0):>5.2f} {st.get('maxdd_pct',0):>6.1f}% {st.get('vs_spy_pp',0):>+6.0f} {vc:>+6.1f}", flush=True)

    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    open(OUT, "w").write(json.dumps(res, indent=2, default=float))
    try:
        from core.models import BacktestResult
        from django.utils import timezone
        BacktestResult.objects.update_or_create(kind="smc_bos_ob_book", defaults={"payload": res, "computed_at": timezone.now()})
        print("Saved BacktestResult[smc_bos_ob_book]", flush=True)
    except Exception as e:
        print("save skipped:", e, flush=True)


if __name__ == "__main__":
    main()
