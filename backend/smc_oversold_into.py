#!/usr/bin/env python3
"""'OVERSOLD INTO X' — SEQUENTIAL entry (user correction: not oversold AND x simultaneously, but oversold THEN
turning up into a bullish trigger = buy the confirmed reversal, not the falling knife). For each name (daily),
the setup = was OVERSOLD recently (RSI(10)<40 in the last 10 bars) AND a bullish trigger FIRES today:
  X=rsi_cross : RSI crosses back above 45
  X=smc_bos   : bullish BOS/CHoCH fires (SMC daily)
  X=sqz_green : Squeeze momentum crosses >0
  X=cmf_cross : CMF crosses >0
Event study of market-adjusted (vs SPY) forward returns at 5/10/20 trading days, on the PIT-topcap universe.
Compares oversold-INTO-X vs the TRIGGER alone and vs OVERSOLD alone — does the sequence add over the pieces?
(⚠ overlapping windows inflate t; read magnitude + hit + n.) Saves BacktestResult[smc_oversold_into].
Run: MSYS_NO_PATHCONV=1 docker exec rotation-backend-1 python -u /app/smc_oversold_into.py"""
import os, json, math, glob
os.environ.setdefault("DJANGO_SETTINGS_MODULE", "rotation.settings")
import django; django.setup()
import numpy as np, pandas as pd
from collections import defaultdict
from core.models import Candle

SMC_DIR = "/app/.data/tradingview/smc"; SQ_DIR = "/app/.data/tradingview/squeeze"
OUT = "/app/.data/studies/smc_oversold_into.json"
UP = ["Bullish_BOS", "Bullish_CHoCH", "Internal_Bullish_BOS", "Internal_Bullish_CHoCH"]
OS_TH = 40.0; OS_WIN = 10; RSI_CROSS = 45.0
HZ = [5, 10, 20]


def nz(v):
    return v is not None and v == v and v != 0 and not (isinstance(v, (int, float)) and abs(v) > 1e50)


def rsi(c, n=10):
    d = c.diff(); up = d.clip(lower=0).ewm(alpha=1 / n, adjust=False).mean(); dn = (-d.clip(upper=0)).ewm(alpha=1 / n, adjust=False).mean()
    return 100 - 100 / (1 + up / dn.replace(0, np.nan))


def main():
    smc = {}; sqz = {}
    for f in glob.glob(SMC_DIR + "/*.json"):
        d = json.load(open(f)); smc[d["ticker"]] = d
    for f in glob.glob(SQ_DIR + "/*.json"):
        d = json.load(open(f)); sqz[d["ticker"]] = d
    tickers = sorted(set(smc) | set(sqz))
    PITMEM = {k: set(v) for k, v in json.load(open("/app/.data/pit_top150_membership.json")).items()} if os.path.exists("/app/.data/pit_top150_membership.json") else {}
    print(f"{len(tickers)} names", flush=True)

    from django.db import connection
    with connection.cursor() as cur:
        cur.execute("SET max_parallel_workers_per_gather = 0")
    data = {}
    for i in range(0, len(tickers + ['SPY']), 100):
        q = Candle.objects.filter(ticker__in=(tickers + ['SPY'])[i:i + 100], interval="1d", date__gte="2014-06-01").values_list("ticker", "date", "high", "low", "close", "volume")
        df = pd.DataFrame(list(q), columns=["ticker", "date", "high", "low", "close", "volume"])
        if df.empty:
            continue
        df["date"] = pd.to_datetime(df["date"])
        for cc in ("high", "low", "close", "volume"):
            df[cc] = df[cc].astype(float)
        for tk, g in df.groupby("ticker", sort=False):
            data[tk] = g.sort_values("date").set_index("date")
    spy = data["SPY"]["close"]
    spyf = {h: spy.shift(-h) / spy - 1.0 for h in HZ}

    def daily_event_dates(rec, keys):
        s = set()
        for row in (rec.get("daily") or []):
            if row.get("t") is not None and any(nz(row.get(k)) for k in keys):
                s.add(pd.Timestamp(int(row["t"]), unit="s").normalize())
        return s

    def sqz_series(rec):
        d, v = [], []
        for row in sorted(rec.get("daily") or [], key=lambda x: x.get("t", 0)):
            t = row.get("t"); p = row.get("Plot")
            if t is None or p is None or (isinstance(p, float) and (p != p or abs(p) > 1e50)):
                continue
            d.append(pd.Timestamp(int(t), unit="s").normalize()); v.append(p)
        return pd.Series(v, index=pd.DatetimeIndex(d)) if d else None

    def pit_ok(tk, dt):
        if not PITMEM:
            return True
        k = str((dt + pd.offsets.MonthEnd(0)).date())
        return k in PITMEM and tk in PITMEM[k]

    acc = defaultdict(lambda: defaultdict(list))   # setup -> horizon -> [excess]

    def add(setup, dt, tk, idx, c):
        if dt not in c.index:
            p = idx.searchsorted(dt)
            if p >= len(idx):
                return
            dt = idx[p]
        if not pit_ok(tk, dt):
            return
        for h in HZ:
            fr = c.shift(-h).get(dt); base = c.get(dt)
            sr = spyf[h].get(dt)
            if fr is not None and base and base > 0 and sr is not None and pd.notna(fr) and pd.notna(sr):
                acc[setup][h].append(float(fr / base - 1 - sr))

    for tk in tickers:
        if tk not in data:
            continue
        c = data[tk]["close"]; idx = c.index
        if c.notna().sum() < 300:
            continue
        r = rsi(c, 10)
        os_recent = (r < OS_TH).rolling(OS_WIN, min_periods=1).max().astype(bool)
        rsi_cross = (r > RSI_CROSS) & (r.shift(1) <= RSI_CROSS)
        smc_bos_dates = daily_event_dates(smc.get(tk, {}), UP)
        smc_bos = pd.Series(idx.isin(smc_bos_dates), index=idx)
        sq = sqz_series(sqz.get(tk, {}))
        if sq is not None:
            sq = sq.reindex(idx).ffill()
            sqz_green = (sq > 0) & (sq.shift(1) <= 0)
        else:
            sqz_green = pd.Series(False, index=idx)
        h_, l_, cl_, v_ = data[tk]["high"], data[tk]["low"], c, data[tk]["volume"]
        rng = (h_ - l_).replace(0, np.nan); mfm = ((cl_ - l_) - (h_ - cl_)) / rng
        cmf = (mfm * v_).fillna(0).rolling(20).sum() / v_.rolling(20).sum().replace(0, np.nan)
        cmf_cross = (cmf > 0) & (cmf.shift(1) <= 0)

        triggers = {"rsi_cross": rsi_cross, "smc_bos": smc_bos, "sqz_green": sqz_green, "cmf_cross": cmf_cross}
        # baselines
        for dt in idx[(r < OS_TH).fillna(False)]:
            add("oversold_only(RSI<40)", dt, tk, idx, c)
        for xname, trig in triggers.items():
            trig = trig.fillna(False)
            for dt in idx[trig]:
                add(f"TRIGGER_only:{xname}", dt, tk, idx, c)
            seq = trig & os_recent.shift(1).fillna(False)   # was oversold in prior 10 bars, trigger fires now
            for dt in idx[seq]:
                add(f"OVERSOLD_INTO:{xname}", dt, tk, idx, c)

    def summ(vals):
        a = np.array(vals, float); a = a[np.isfinite(a)]
        if len(a) < 20:
            return {"n": int(len(a))}
        m = a.mean(); s = a.std(ddof=1)
        return {"n": int(len(a)), "mean_exc_pct": round(m * 100, 3), "hit_pct": round(float((a > 0).mean()) * 100, 1),
                "t": round(float(m / (s / math.sqrt(len(a)))) if s > 0 else 0, 2)}

    res = {"params": {"os_th": OS_TH, "os_win": OS_WIN, "rsi_cross": RSI_CROSS, "hz": HZ}, "setups": {}}
    order = ["oversold_only(RSI<40)"] + [f"TRIGGER_only:{x}" for x in ["rsi_cross", "smc_bos", "sqz_green", "cmf_cross"]] + [f"OVERSOLD_INTO:{x}" for x in ["rsi_cross", "smc_bos", "sqz_green", "cmf_cross"]]
    print(f"\n=== OVERSOLD-INTO-X sequential entry — market-adj fwd returns (PIT univ) ===", flush=True)
    print(f"{'setup':>28} {'h':>3} {'n':>6} {'meanExc%':>9} {'hit%':>6} {'t':>6}", flush=True)
    for setup in order:
        res["setups"][setup] = {}
        for h in HZ:
            st = summ(acc.get(setup, {}).get(h, []))
            res["setups"][setup][h] = st
            if st.get("n", 0) >= 20:
                print(f"{setup:>28} {h:>3} {st['n']:>6} {st['mean_exc_pct']:>+9.3f} {st['hit_pct']:>6.1f} {st['t']:>+6.2f}", flush=True)

    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    open(OUT, "w").write(json.dumps(res, indent=2, default=float))
    try:
        from core.models import BacktestResult
        from django.utils import timezone
        BacktestResult.objects.update_or_create(kind="smc_oversold_into", defaults={"payload": res, "computed_at": timezone.now()})
        print("Saved BacktestResult[smc_oversold_into]", flush=True)
    except Exception as e:
        print("save skipped:", e, flush=True)


if __name__ == "__main__":
    main()
