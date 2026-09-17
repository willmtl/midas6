#!/usr/bin/env python3
"""CONFOUND-CONTROLLED book test of the LuxAlgo SMC weekly structural state. The event study showed weekly
Bullish BOS/CHoCH -> positive continuation. Here we test whether it's TRADEABLE and, crucially, whether it
adds anything beyond just holding the pilot universe: within the SAME 114 names, does holding only the names
in a BULLISH weekly structure beat equal-weighting ALL 114? (same universe both sides => removes the mega-cap
/ survivorship confound). Also vs SPY. Monthly rebalance, EW, costed 20bps/side, both halves.

State: walk weekly SMC events in time; Bullish_BOS/Bullish_CHoCH -> +1, Bearish_BOS/Bearish_CHoCH -> -1
(swing structure; also reports INTERNAL structure). Forward-filled, sampled at each month-end. Long the +1 names.
Saves BacktestResult[smc_structure_book]. Run: docker exec rotation-backend-1 python -u /app/smc_structure_book.py"""
import os, json, math, glob
os.environ.setdefault("DJANGO_SETTINGS_MODULE", "rotation.settings")
import django; django.setup()
import numpy as np, pandas as pd
from core.models import Candle

SMC_DIR = "/app/.data/tradingview/smc"
OUT = "/app/.data/studies/smc_structure_book.json"
UP = {"swing": ["Bullish_BOS", "Bullish_CHoCH"], "internal": ["Internal_Bullish_BOS", "Internal_Bullish_CHoCH"]}
DN = {"swing": ["Bearish_BOS", "Bearish_CHoCH"], "internal": ["Internal_Bearish_BOS", "Internal_Bearish_CHoCH"]}


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

    # PIT top-cap membership gate (no look-ahead): only trade a name in months it was actually a top-N mega-cap.
    PITMEM = {}
    mf = "/app/.data/pit_top150_membership.json"
    if os.path.exists(mf):
        raw = json.load(open(mf))
        PITMEM = {k: set(v) for k, v in raw.items()}
        cov = len(set().union(*PITMEM.values()) & set(tickers)) if PITMEM else 0
        print(f"PIT membership loaded ({len(PITMEM)} months); {cov}/{len(tickers)} fetched names appear in it", flush=True)

    def pit_ok(t, d):
        if not PITMEM:
            return True
        k = str(d.date())
        return k in PITMEM and t in PITMEM[k]

    # monthly close panel (our split-adj closes) + SPY
    from django.db import connection
    with connection.cursor() as cur:
        cur.execute("SET max_parallel_workers_per_gather = 0")
    close = {}
    allt = tickers + ["SPY"]
    for i in range(0, len(allt), 200):
        rows = Candle.objects.filter(ticker__in=allt[i:i + 200], interval="1d", date__gte="2014-06-01").values_list("ticker", "date", "close")
        df = pd.DataFrame(list(rows), columns=["ticker", "date", "close"])
        if df.empty:
            continue
        df["date"] = pd.to_datetime(df["date"]); df["close"] = df["close"].astype(float)
        for tk, g in df.groupby("ticker", sort=False):
            close[tk] = g.sort_values("date").set_index("date")["close"]
    close = pd.DataFrame(close).sort_index()
    spy_m = close["SPY"].resample("ME").last()
    midx = spy_m.index[spy_m.index >= "2016-01-01"]
    cm = close.resample("ME").last().reindex(midx)
    fret = cm.shift(-1) / cm - 1.0
    spy_fwd = spy_m.reindex(midx).shift(-1) / spy_m.reindex(midx) - 1.0

    def state_panel(kind, tf="weekly"):
        st = {}
        for r in recs:
            tk = r["ticker"]
            wk = r.get(tf) or []
            if not wk or tk not in cm.columns:
                continue
            dates, vals, cur = [], [], 0
            for row in sorted(wk, key=lambda x: x.get("t", 0)):
                t = row.get("t")
                if t is None:
                    continue
                up = any(nz(row.get(e)) for e in UP[kind]); dn = any(nz(row.get(e)) for e in DN[kind])
                if up and not dn:
                    cur = 1
                elif dn and not up:
                    cur = -1
                dates.append(pd.Timestamp(int(t), unit="s").normalize()); vals.append(cur)
            if dates:
                s = pd.Series(vals, index=pd.DatetimeIndex(dates))
                s = s[~s.index.duplicated(keep="last")].sort_index()
                if tf == "weekly":
                    s.index = s.index + pd.Timedelta(days=7)   # weekly bar value known only at week-close; delay
                    s = s.sort_index()                          # a week so month-end sampling can't peek intra-week
                st[tk] = s.reindex(cm.index.union(s.index)).ffill().reindex(midx)
        return pd.DataFrame(st).reindex(columns=[t for t in tickers if t in cm.columns])

    def book(mask_fn, cost_bps=20.0):
        rets, prev = [], set()
        for i, d in enumerate(midx[:-1]):
            fr = fret.loc[d]
            names = [t for t in cm.columns if t != "SPY" and pd.notna(fr[t]) and pit_ok(t, d) and mask_fn(t, d)]
            if len(names) < 5:
                rets.append(np.nan); prev = set(); continue
            cur = set(names); turn = len(cur ^ prev) / (2 * max(len(cur), 1))
            rets.append(float(fr[names].mean()) - turn * 2 * cost_bps / 1e4); prev = cur
        return pd.Series(rets, index=midx[:-1])

    def stats(pr, bench=None):
        pr = pr.dropna()
        if len(pr) < 12:
            return {"n": int(len(pr))}
        eq = np.cumprod(1 + pr.values); yrs = len(pr) / 12.0
        o = {"n": int(len(pr)), "total_pct": float((eq[-1] - 1) * 100),
             "cagr_pct": float((eq[-1] ** (1 / yrs) - 1) * 100) if eq[-1] > 0 else -100.0,
             "sharpe": float(pr.mean() / pr.std() * math.sqrt(12)) if pr.std() > 0 else 0.0,
             "maxdd_pct": float((eq / np.maximum.accumulate(eq) - 1).min() * 100),
             "hit_pct": float((pr.values > 0).mean() * 100)}
        if bench is not None:
            b = bench.reindex(pr.index).fillna(0.0)
            o["vs_spy_pp"] = float(((eq[-1] - 1) - (np.cumprod(1 + b.values)[-1] - 1)) * 100)
        return o

    res = {"n_files": len(recs), "window": [str(midx[0].date()), str(midx[-1].date())], "arms": {}}
    # benchmarks: hold-all-114 EW (the confound control) and SPY
    all_ew = book(lambda t, d: True)
    res["arms"]["hold_all_EW"] = stats(all_ew, bench=spy_fwd)
    res["arms"]["SPY"] = stats(spy_fwd.reindex(midx[:-1]))
    print(f"\n=== SMC structure book — within-pilot (114 mega-caps), monthly, costed 20bps, {midx[0].date()}..{midx[-1].date()} ===", flush=True)
    print(f"{'arm':>34} {'total':>9} {'CAGR':>7} {'Sh':>5} {'DD':>7} {'hit':>5} {'vsSPY':>7} {'vsALL':>7}", flush=True)
    base_all = res["arms"]["hold_all_EW"]["total_pct"]

    def show(lab, st):
        vs_all = st.get("total_pct", 0) - base_all
        print(f"{lab:>34} {st.get('total_pct',0):>8.0f}% {st.get('cagr_pct',0):>6.1f}% {st.get('sharpe',0):>5.2f} "
              f"{st.get('maxdd_pct',0):>6.1f}% {st.get('hit_pct',0):>4.0f}% {st.get('vs_spy_pp',0):>+6.0f} {vs_all:>+6.0f}", flush=True)
    show("SPY", res["arms"]["SPY"])
    show("hold PIT-topcap EW (control)", res["arms"]["hold_all_EW"])

    for tf in ("weekly", "daily"):
        for kind in ("swing", "internal"):
            sp = state_panel(kind, tf)
            for lab, cond in [(f"{tf}/{kind}: long UP", 1), (f"{tf}/{kind}: long DOWN", -1)]:
                def mask(t, d, _sp=sp, _c=cond):
                    return t in _sp.columns and d in _sp.index and _sp.loc[d, t] == _c
                st = stats(book(mask), bench=spy_fwd)
                res["arms"][lab] = st
                show(lab, st)

    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    open(OUT, "w").write(json.dumps(res, indent=2, default=float))
    print(f"\nwrote {OUT}", flush=True)
    try:
        from core.models import BacktestResult
        from django.utils import timezone
        BacktestResult.objects.update_or_create(kind="smc_structure_book",
            defaults={"payload": res, "computed_at": timezone.now()})
        print("Saved BacktestResult[smc_structure_book]", flush=True)
    except Exception as e:
        print("save skipped:", e, flush=True)


if __name__ == "__main__":
    main()
