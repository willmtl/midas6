#!/usr/bin/env python3
"""RELATIVE STRENGTH vs SPY — the one signal family that hasn't died (cross-sectional momentum ~= the only thing
that survived the SMC/RSI/money-flow arc). RS line = stock_adjclose / SPY_close. Monthly, look-ahead-safe: signal
sampled at month-end t, long EW next month (hold 1), costed 20bps/side, $5M dollar-vol floor, SEGMENTED BY CAP
BUCKET (HARD RULE — never average across the universe). Cap = PIT mktcap (as-traded close x PIT shares).

Arms (each tested WITHIN each cap bucket, vs that bucket's EW control and vs SPY, both halves):
  RS_mom6_topT   : top tercile by 6mo RS momentum (rs/rs[-6]-1)  <- classic relative-strength momentum
  RS_mom12_topT  : top tercile by 12mo RS momentum
  RS_newhigh12   : RS at a trailing-12mo high (leadership breakout)
  RS_gt_MA10     : RS above its own 10mo MA (Mansfield-style RS positive)
  absmom6_topT   : ABSOLUTE 6mo price momentum top tercile (control: does RELATIVE beat ABSOLUTE?)
Saves BacktestResult[rs_vs_spy_book]. Run: MSYS_NO_PATHCONV=1 docker exec rotation-backend-1 python -u /app/rs_vs_spy_book.py"""
import os, json, math
os.environ.setdefault("DJANGO_SETTINGS_MODULE", "rotation.settings")
import django; django.setup()
import numpy as np, pandas as pd
from collections import defaultdict
import price_basis
from core.models import Candle, Sector, DelistedCompany, FinancialReport

OUT = "/app/.data/studies/rs_vs_spy_book.json"
DOLLAR_VOL_FLOOR = 5e6
COST_BPS = 20.0
# cap buckets in $ (PIT mktcap): mega / large / mid / small
BUCKETS = [("mega", 50e9, np.inf), ("large", 10e9, 50e9), ("mid", 2e9, 10e9), ("small", 0.0, 2e9)]


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


def halves(pr, bench):
    pr = pr.dropna()
    if len(pr) < 24:
        return {}
    mid = len(pr) // 2
    return {"H1": stats(pr.iloc[:mid], bench), "H2": stats(pr.iloc[mid:], bench)}


def main():
    etfs = set(Sector.objects.values_list("etf", flat=True)) | {"SPY", "QQQ"}
    covered = set()
    for line in open("/app/.data/analyst_ratings.jsonl", encoding="utf-8"):
        if line.strip():
            try:
                tk = json.loads(line).get("ticker")
                if tk and "." not in tk and tk not in etfs:
                    covered.add(tk)
            except Exception:
                pass
    delisted = set(DelistedCompany.objects.exclude(delisted_date=None).values_list("ticker", flat=True))
    universe = sorted((covered | delisted) - etfs)
    print(f"broad universe {len(universe)} ({len(delisted)} delisted)", flush=True)

    from django.db import connection
    with connection.cursor() as cur:
        cur.execute("SET max_parallel_workers_per_gather = 0")

    spy = pd.DataFrame(list(Candle.objects.filter(ticker="SPY", date__gte="2014-06-01").values_list("date", "close")), columns=["date", "close"])
    spy["date"] = pd.to_datetime(spy["date"]); spy_m = spy.set_index("date")["close"].astype(float).resample("ME").last()
    midx = spy_m.index[spy_m.index >= "2015-06-01"]
    midx_ts = np.array([d.value for d in midx], dtype="int64")

    close_m, dvol_m = {}, {}
    for i in range(0, len(universe), 200):
        q = Candle.objects.filter(ticker__in=universe[i:i + 200], interval="1d", date__gte="2014-06-01").values_list("ticker", "date", "close", "volume")
        df = pd.DataFrame(list(q), columns=["ticker", "date", "close", "volume"])
        if df.empty:
            continue
        df["date"] = pd.to_datetime(df["date"])
        df["close"] = df["close"].astype(float); df["volume"] = df["volume"].astype(float)
        for tk, g in df.groupby("ticker", sort=False):
            g = g.sort_values("date").set_index("date")
            close_m[tk] = g["close"].resample("ME").last().reindex(midx)
            dvol_m[tk] = (g["close"] * g["volume"]).resample("ME").median().reindex(midx)
    close_m = pd.DataFrame(close_m); dvol_m = pd.DataFrame(dvol_m)
    cols = list(close_m.columns)
    print(f"names with monthly closes {len(cols)}", flush=True)

    # PIT mktcap for cap-bucketing
    frs = defaultdict(list)
    for tk, ad, sh in FinancialReport.objects.filter(ticker__in=cols, shares_outstanding__gt=0).values_list("ticker", "avail_date", "shares_outstanding"):
        if ad and sh:
            frs[tk].append((pd.Timestamp(ad).value, float(sh)))
    shares_m = {}
    for tk in cols:
        pts = sorted(frs.get(tk, []))
        if not pts:
            continue
        di = np.array([t for t, _ in pts]); sv = np.array([v for _, v in pts]); o = np.full(len(midx), np.nan)
        for j, dd in enumerate(midx_ts):
            b = np.searchsorted(di, dd, side="right")
            if b > 0:
                o[j] = sv[b - 1]
        shares_m[tk] = pd.Series(o, index=midx)
    shares_m = pd.DataFrame(shares_m)
    mcap = price_basis.as_traded_close(close_m[shares_m.columns]) * shares_m
    mcap = mcap.reindex(columns=cols)
    print(f"names with PIT mktcap {mcap.notna().any().sum()}", flush=True)

    # forward return (adjusted close = total-return proxy) and RS line
    fret = close_m.shift(-1) / close_m - 1.0
    spy_fwd = spy_m.reindex(midx).shift(-1) / spy_m.reindex(midx) - 1.0
    rs = close_m.div(spy_m.reindex(midx), axis=0)     # relative strength line

    rs_mom6 = rs / rs.shift(6) - 1.0
    rs_mom12 = rs / rs.shift(12) - 1.0
    rs_newhigh12 = rs >= rs.rolling(12, min_periods=12).max()
    rs_ma10 = rs > rs.rolling(10, min_periods=10).mean()
    abs_mom6 = close_m / close_m.shift(6) - 1.0

    eligible = (dvol_m >= DOLLAR_VOL_FLOOR) & close_m.notna() & fret.notna()

    def bucket_mask(d, lo, hi):
        row = mcap.loc[d]
        return (row >= lo) & (row < hi)

    def run_arm(kind, bkt):
        """kind in {rs_mom6,rs_mom12,rs_newhigh12,rs_ma10,absmom6,control}. Returns monthly return series."""
        lo, hi = bkt[1], bkt[2]
        rets, prev, ncnt = [], set(), []
        for d in midx[:-1]:
            elig = eligible.loc[d] & bucket_mask(d, lo, hi)
            names = [t for t in cols if elig.get(t, False)]
            if len(names) < 8:
                rets.append(np.nan); prev = set(); ncnt.append(0); continue
            if kind == "control":
                sel = names
            elif kind in ("rs_mom6", "rs_mom12", "absmom6"):
                src = {"rs_mom6": rs_mom6, "rs_mom12": rs_mom12, "absmom6": abs_mom6}[kind].loc[d]
                sc = src[names].dropna()
                if len(sc) < 8:
                    rets.append(np.nan); prev = set(); ncnt.append(0); continue
                thr = sc.quantile(2 / 3)
                sel = list(sc[sc >= thr].index)
            elif kind == "rs_newhigh12":
                sel = [t for t in names if bool(rs_newhigh12.loc[d, t])]
            elif kind == "rs_ma10":
                sel = [t for t in names if bool(rs_ma10.loc[d, t])]
            else:
                sel = []
            if len(sel) < 5:
                rets.append(np.nan); prev = set(); ncnt.append(0); continue
            fr = fret.loc[d]; cur = set(sel)
            turn = len(cur ^ prev) / (2 * max(len(cur), 1))
            rets.append(float(fr[sel].mean()) - turn * 2 * COST_BPS / 1e4)
            prev = cur; ncnt.append(len(sel))
        return pd.Series(rets, index=midx[:-1]), float(np.mean([n for n in ncnt if n > 0]) if any(ncnt) else 0)

    ARMS = ["rs_mom6", "rs_mom12", "rs_newhigh12", "rs_ma10", "absmom6"]
    res = {"params": {"dollar_vol_floor": DOLLAR_VOL_FLOOR, "cost_bps": COST_BPS, "window": [str(midx[0].date()), str(midx[-1].date())]},
           "spy": stats(spy_fwd.reindex(midx[:-1])), "buckets": {}}
    print(f"\n=== RELATIVE STRENGTH vs SPY — cross-sectional, PIT cap-bucketed, costed {COST_BPS:.0f}bps, {midx[0].date()}..{midx[-1].date()} ===", flush=True)
    print(f"  SPY: CAGR {res['spy'].get('cagr_pct',0):+.1f}%/Sh{res['spy'].get('sharpe',0):.2f}", flush=True)
    for bkt in BUCKETS:
        bname = bkt[0]
        ctrl, cn = run_arm("control", bkt)
        cst = stats(ctrl, bench=spy_fwd)
        res["buckets"][bname] = {"control": {**cst, "avg_names": round(cn, 1)}, "arms": {}}
        print(f"\n--- {bname.upper()} cap (${bkt[1]/1e9:.0f}-{bkt[2]/1e9 if np.isfinite(bkt[2]) else '∞'}B)  control avgN {cn:.0f}  CAGR {cst.get('cagr_pct',0):+.1f}%/Sh{cst.get('sharpe',0):.2f} vsSPY {cst.get('vs_spy_pp',0):+.0f} ---", flush=True)
        print(f"{'arm':>16} {'avgN':>5} {'total':>8} {'CAGR':>7} {'Sh':>5} {'DD':>7} {'vsSPY':>7} {'vsCtrl':>7} {'H1sh':>5} {'H2sh':>5}", flush=True)
        for kind in ARMS:
            pr, n = run_arm(kind, bkt)
            st = stats(pr, bench=spy_fwd); hv = halves(pr, spy_fwd)
            st["avg_names"] = round(n, 1); st["halves"] = hv
            res["buckets"][bname]["arms"][kind] = st
            vc = st.get("total_pct", 0) - cst.get("total_pct", 0)
            print(f"{kind:>16} {n:>5.0f} {st.get('total_pct',0):>+7.0f}% {st.get('cagr_pct',0):>+6.1f}% {st.get('sharpe',0):>5.2f} "
                  f"{st.get('maxdd_pct',0):>6.1f}% {st.get('vs_spy_pp',0):>+6.0f} {vc:>+6.0f} "
                  f"{hv.get('H1',{}).get('sharpe',0):>5.2f} {hv.get('H2',{}).get('sharpe',0):>5.2f}", flush=True)

    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    open(OUT, "w").write(json.dumps(res, indent=2, default=float))
    try:
        from core.models import BacktestResult
        from django.utils import timezone
        BacktestResult.objects.update_or_create(kind="rs_vs_spy_book", defaults={"payload": res, "computed_at": timezone.now()})
        print("\nSaved BacktestResult[rs_vs_spy_book]", flush=True)
    except Exception as e:
        print("save skipped:", e, flush=True)


if __name__ == "__main__":
    main()
