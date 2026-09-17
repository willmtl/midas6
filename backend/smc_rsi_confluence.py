#!/usr/bin/env python3
"""CONFLUENCE test (user: "combine all of those with RSI"). On the PIT-topcap universe, build 5 look-ahead-safe
monthly signals per name and test whether requiring several to AGREE beats the pieces / the index:
  RSI_OS  = RSI(10) < 45 (oversold dip - the one known edge, CONTRARIAN)
  RSI_STR = RSI(10) > 55 (strength, MOMENTUM)
  MF      = CMF(20) monthly > 0 (money-flow accumulation)
  SMC     = SMC weekly structure bullish (BOS/CHoCH state, lagged 1wk)
  SQZ     = Squeeze Momentum histogram > 0 (weekly, lagged 1wk)
Monthly EW book, hold 1mo, PIT-membership-gated, costed 20bps, vs SPY and the hold-everything control. Reports
avg #names so we can see when an AND-combo thins to nothing. Saves BacktestResult[smc_rsi_confluence].
Run: MSYS_NO_PATHCONV=1 docker exec rotation-backend-1 python -u /app/smc_rsi_confluence.py"""
import os, json, math, glob
os.environ.setdefault("DJANGO_SETTINGS_MODULE", "rotation.settings")
import django; django.setup()
import numpy as np, pandas as pd
from core.models import Candle

SMC_DIR = "/app/.data/tradingview/smc"; SQ_DIR = "/app/.data/tradingview/squeeze"
OUT = "/app/.data/studies/smc_rsi_confluence.json"
UP = ["Bullish_BOS", "Bullish_CHoCH"]; DN = ["Bearish_BOS", "Bearish_CHoCH"]


def nz(v):
    return v is not None and v == v and v != 0 and not (isinstance(v, (int, float)) and abs(v) > 1e50)


def rsi(c, n=10):
    d = c.diff(); up = d.clip(lower=0).ewm(alpha=1 / n, adjust=False).mean(); dn = (-d.clip(upper=0)).ewm(alpha=1 / n, adjust=False).mean()
    return 100 - 100 / (1 + up / dn.replace(0, np.nan))


def main():
    smc = {json.load(open(f))["ticker"]: json.load(open(f)) for f in glob.glob(SMC_DIR + "/*.json")}
    sqz = {json.load(open(f))["ticker"]: json.load(open(f)) for f in glob.glob(SQ_DIR + "/*.json")}
    tickers = sorted(set(smc) | set(sqz))
    print(f"SMC {len(smc)}, squeeze {len(sqz)}, union {len(tickers)}", flush=True)
    PITMEM = {k: set(v) for k, v in json.load(open("/app/.data/pit_top150_membership.json")).items()} if os.path.exists("/app/.data/pit_top150_membership.json") else {}

    from django.db import connection
    with connection.cursor() as cur:
        cur.execute("SET max_parallel_workers_per_gather = 0")
    # need OHLCV for RSI + CMF
    data = {}
    allt = tickers + ["SPY"]
    for i in range(0, len(allt), 100):
        q = Candle.objects.filter(ticker__in=allt[i:i + 100], interval="1d", date__gte="2014-06-01").values_list("ticker", "date", "high", "low", "close", "volume")
        df = pd.DataFrame(list(q), columns=["ticker", "date", "high", "low", "close", "volume"])
        if df.empty:
            continue
        df["date"] = pd.to_datetime(df["date"])
        for cc in ("high", "low", "close", "volume"):
            df[cc] = df[cc].astype(float)
        for tk, g in df.groupby("ticker", sort=False):
            data[tk] = g.sort_values("date").set_index("date")
    spy_m = data["SPY"]["close"].resample("ME").last()
    midx = spy_m.index[spy_m.index >= "2016-01-01"]
    cm = pd.DataFrame({tk: data[tk]["close"].resample("ME").last().reindex(midx) for tk in tickers if tk in data})
    fret = cm.shift(-1) / cm - 1.0
    spy_fwd = spy_m.reindex(midx).shift(-1) / spy_m.reindex(midx) - 1.0

    def smc_state(tk):
        r = smc.get(tk)
        if not r:
            return pd.Series(0, index=midx)
        dates, vals, cur = [], [], 0
        for row in sorted(r.get("weekly") or [], key=lambda x: x.get("t", 0)):
            t = row.get("t")
            if t is None:
                continue
            u = any(nz(row.get(e)) for e in UP); d = any(nz(row.get(e)) for e in DN)
            if u and not d:
                cur = 1
            elif d and not u:
                cur = -1
            dates.append(pd.Timestamp(int(t), unit="s").normalize()); vals.append(cur)
        if not dates:
            return pd.Series(0, index=midx)
        s = pd.Series(vals, index=pd.DatetimeIndex(dates)); s = s[~s.index.duplicated(keep="last")].sort_index()
        s.index = s.index + pd.Timedelta(days=7)      # look-ahead-safe weekly lag
        return s.reindex(cm.index.union(s.index)).ffill().reindex(midx).fillna(0)

    def sqz_bull(tk):
        r = sqz.get(tk)
        if not r:
            return pd.Series(False, index=midx)
        dates, vals = [], []
        for row in sorted(r.get("weekly") or [], key=lambda x: x.get("t", 0)):
            t = row.get("t"); p = row.get("Plot")
            if t is None or p is None or (isinstance(p, float) and (p != p or abs(p) > 1e50)):
                continue
            dates.append(pd.Timestamp(int(t), unit="s").normalize()); vals.append(1 if p > 0 else 0)
        if not dates:
            return pd.Series(False, index=midx)
        s = pd.Series(vals, index=pd.DatetimeIndex(dates)); s = s[~s.index.duplicated(keep="last")].sort_index()
        s.index = s.index + pd.Timedelta(days=7)
        return (s.reindex(cm.index.union(s.index)).ffill().reindex(midx).fillna(0) == 1)

    # signal panels [midx x tk]
    RSI, CMF = {}, {}
    for tk in cm.columns:
        c = data[tk]["close"]; RSI[tk] = rsi(c, 10).resample("ME").last().reindex(midx)
        h, l, cl, v = data[tk]["high"].resample("ME").max(), data[tk]["low"].resample("ME").min(), c.resample("ME").last(), data[tk]["volume"].resample("ME").sum()
        rng = (h - l).replace(0, np.nan); mfm = ((cl - l) - (h - cl)) / rng
        CMF[tk] = ((mfm * v).fillna(0).rolling(20).sum() / v.rolling(20).sum().replace(0, np.nan)).reindex(midx)
    RSI = pd.DataFrame(RSI); CMF = pd.DataFrame(CMF)
    SMCst = pd.DataFrame({tk: smc_state(tk) for tk in cm.columns})
    SQZ = pd.DataFrame({tk: sqz_bull(tk) for tk in cm.columns})

    A_os = RSI < 45; A_str = RSI > 55; B_mf = CMF > 0; C_smc = SMCst == 1; D_sqz = SQZ

    def pit_ok(t, d):
        if not PITMEM:
            return True
        k = str(d.date()); return k in PITMEM and t in PITMEM[k]

    def book(mask, cost_bps=20.0):
        rets, prev, ncount = [], set(), []
        for d in midx[:-1]:
            fr = fret.loc[d]; row = mask.loc[d]
            names = [t for t in mask.columns if pd.notna(fr[t]) and pit_ok(t, d) and bool(row[t])]
            ncount.append(len(names))
            if len(names) < 5:
                rets.append(np.nan); prev = set(); continue
            cur = set(names); turn = len(cur ^ prev) / (2 * max(len(cur), 1))
            rets.append(float(fr[names].mean()) - turn * 2 * cost_bps / 1e4); prev = cur
        return pd.Series(rets, index=midx[:-1]), float(np.mean(ncount))

    def ctrl():
        rets = []
        for d in midx[:-1]:
            fr = fret.loc[d]; names = [t for t in cm.columns if t != "SPY" and pd.notna(fr[t]) and pit_ok(t, d)]
            rets.append(float(fr[names].mean()) if len(names) >= 5 else np.nan)
        return pd.Series(rets, index=midx[:-1])

    def stats(pr, bench=None):
        pr = pr.dropna()
        if len(pr) < 12:
            return {"n": int(len(pr))}
        eq = np.cumprod(1 + pr.values); yrs = len(pr) / 12.0
        o = {"n": int(len(pr)), "cagr_pct": float((eq[-1] ** (1 / yrs) - 1) * 100) if eq[-1] > 0 else -100.0,
             "sharpe": float(pr.mean() / pr.std() * math.sqrt(12)) if pr.std() > 0 else 0.0,
             "maxdd_pct": float((eq / np.maximum.accumulate(eq) - 1).min() * 100)}
        if bench is not None:
            b = bench.reindex(pr.index).fillna(0.0); o["vs_spy_pp"] = float(((eq[-1] - 1) - (np.cumprod(1 + b.values)[-1] - 1)) * 100)
        return o

    score_mom = A_str.astype(int) + B_mf.astype(int) + C_smc.astype(int) + D_sqz.astype(int)
    arms = {
        "RSI oversold (dip)": A_os, "RSI strong": A_str, "money-flow +": B_mf, "SMC bull": C_smc, "squeeze green": D_sqz,
        "dip + SMC bull": A_os & C_smc, "dip + money-flow+": A_os & B_mf, "dip + squeeze green": A_os & D_sqz,
        "dip + ALL 3 momentum": A_os & B_mf & C_smc & D_sqz,
        "momentum score>=2": score_mom >= 2, "momentum score>=3": score_mom >= 3, "momentum ALL 4": score_mom >= 4,
    }
    cst = stats(ctrl(), bench=spy_fwd); spq = stats(spy_fwd.reindex(midx[:-1]))
    res = {"control": cst, "spy": spq, "arms": {}}
    print(f"\n=== RSI x SMC x money-flow x squeeze CONFLUENCE (PIT monthly, costed 20bps) ===", flush=True)
    print(f"  SPY CAGR {spq.get('cagr_pct',0):+.1f}%/Sh{spq.get('sharpe',0):.2f} | control CAGR {cst.get('cagr_pct',0):+.1f}%/Sh{cst.get('sharpe',0):.2f} (vsSPY {cst.get('vs_spy_pp',0):+.0f})", flush=True)
    print(f"{'arm':>24} {'avgN':>5} {'CAGR':>7} {'Sh':>5} {'vsSPY':>7} {'vsCtrl':>7}", flush=True)
    for lab, mask in arms.items():
        pr, avgn = book(mask); st = stats(pr, bench=spy_fwd); st["avg_names"] = round(avgn, 1)
        res["arms"][lab] = st
        vc = st.get("cagr_pct", 0) - cst.get("cagr_pct", 0)
        print(f"{lab:>24} {avgn:>5.0f} {st.get('cagr_pct',0):>+6.1f}% {st.get('sharpe',0):>5.2f} {st.get('vs_spy_pp',0):>+6.0f} {vc:>+6.1f}", flush=True)

    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    open(OUT, "w").write(json.dumps(res, indent=2, default=float))
    try:
        from core.models import BacktestResult
        from django.utils import timezone
        BacktestResult.objects.update_or_create(kind="smc_rsi_confluence", defaults={"payload": res, "computed_at": timezone.now()})
        print("Saved BacktestResult[smc_rsi_confluence]", flush=True)
    except Exception as e:
        print("save skipped:", e, flush=True)


if __name__ == "__main__":
    main()
