#!/usr/bin/env python3
"""Is mega-cap 6mo momentum (the one RS arm that beat SPY) REAL ALPHA or just BETA/bull-market? Recompute the
mega-cap top-tercile 6mo-momentum monthly book (identical selection to rs_vs_spy_book), then: (a) CAPM regress on
SPY -> annualized alpha, beta, t(alpha); (b) per-calendar-year vsSPY; (c) correlation with the analyst-revision
large-cap book (tgt_rev_3m) if its monthly series is reproducible here -> is this a DISTINCT 3rd book or a
re-expression of book #2? Costed 20bps, $5M dvol floor, same PIT cap-bucket machinery.
Run: MSYS_NO_PATHCONV=1 docker exec rotation-backend-1 python -u /app/rs_momentum_decompose.py"""
import os, json, math
os.environ.setdefault("DJANGO_SETTINGS_MODULE", "rotation.settings")
import django; django.setup()
import numpy as np, pandas as pd
from collections import defaultdict
import price_basis
from core.models import Candle, Sector, DelistedCompany, FinancialReport

DOLLAR_VOL_FLOOR = 5e6; COST_BPS = 20.0


def build_panels():
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
    from django.db import connection
    with connection.cursor() as cur:
        cur.execute("SET max_parallel_workers_per_gather = 0")
    spy = pd.DataFrame(list(Candle.objects.filter(ticker="SPY", date__gte="2014-06-01").values_list("date", "close")), columns=["date", "close"])
    spy["date"] = pd.to_datetime(spy["date"]); spy_m = spy.set_index("date")["close"].astype(float).resample("ME").last()
    midx = spy_m.index[spy_m.index >= "2015-06-01"]; midx_ts = np.array([d.value for d in midx], dtype="int64")
    close_m, dvol_m = {}, {}
    for i in range(0, len(universe), 200):
        q = Candle.objects.filter(ticker__in=universe[i:i + 200], interval="1d", date__gte="2014-06-01").values_list("ticker", "date", "close", "volume")
        df = pd.DataFrame(list(q), columns=["ticker", "date", "close", "volume"])
        if df.empty:
            continue
        df["date"] = pd.to_datetime(df["date"]); df["close"] = df["close"].astype(float); df["volume"] = df["volume"].astype(float)
        for tk, g in df.groupby("ticker", sort=False):
            g = g.sort_values("date").set_index("date")
            close_m[tk] = g["close"].resample("ME").last().reindex(midx)
            dvol_m[tk] = (g["close"] * g["volume"]).resample("ME").median().reindex(midx)
    close_m = pd.DataFrame(close_m); dvol_m = pd.DataFrame(dvol_m); cols = list(close_m.columns)
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
    mcap = (price_basis.as_traded_close(close_m[shares_m.columns]) * shares_m).reindex(columns=cols)
    return cols, midx, spy_m, close_m, dvol_m, mcap


def momentum_book(cols, midx, spy_m, close_m, dvol_m, mcap, lo, hi, look=6):
    fret = close_m.shift(-1) / close_m - 1.0
    mom = close_m / close_m.shift(look) - 1.0
    eligible = (dvol_m >= DOLLAR_VOL_FLOOR) & close_m.notna() & fret.notna()
    rets, prev = [], set()
    for d in midx[:-1]:
        row = mcap.loc[d]; elig = eligible.loc[d] & (row >= lo) & (row < hi)
        names = [t for t in cols if elig.get(t, False)]
        if len(names) < 8:
            rets.append(np.nan); prev = set(); continue
        sc = mom.loc[d][names].dropna()
        if len(sc) < 8:
            rets.append(np.nan); prev = set(); continue
        sel = list(sc[sc >= sc.quantile(2 / 3)].index)
        if len(sel) < 5:
            rets.append(np.nan); prev = set(); continue
        fr = fret.loc[d]; cur = set(sel); turn = len(cur ^ prev) / (2 * max(len(cur), 1))
        rets.append(float(fr[sel].mean()) - turn * 2 * COST_BPS / 1e4); prev = cur
    return pd.Series(rets, index=midx[:-1])


def capm(pr, spy_fwd):
    df = pd.concat([pr.rename("p"), spy_fwd.rename("m")], axis=1).dropna()
    if len(df) < 24:
        return {}
    x = df["m"].values; y = df["p"].values
    b, a = np.polyfit(x, y, 1)
    resid = y - (a + b * x); se = resid.std(ddof=2) / (math.sqrt(len(x)) * x.std())
    ta = (b - 0) / se if se > 0 else 0
    # alpha t-stat
    se_a = resid.std(ddof=2) * math.sqrt(1 / len(x) + x.mean() ** 2 / (len(x) * x.var()))
    t_alpha = a / se_a if se_a > 0 else 0
    corr = float(np.corrcoef(x, y)[0, 1])
    return {"alpha_ann_pct": round(a * 12 * 100, 2), "beta": round(float(b), 3),
            "t_alpha": round(float(t_alpha), 2), "corr_spy": round(corr, 2)}


def main():
    cols, midx, spy_m, close_m, dvol_m, mcap = build_panels()
    spy_fwd = spy_m.reindex(midx).shift(-1) / spy_m.reindex(midx) - 1.0
    print(f"panels: {len(cols)} names, {midx[0].date()}..{midx[-1].date()}", flush=True)
    out = {"buckets": {}}
    for bname, lo, hi in [("mega", 50e9, np.inf), ("large", 10e9, 50e9)]:
        for look in (6, 12):
            pr = momentum_book(cols, midx, spy_m, close_m, dvol_m, mcap, lo, hi, look)
            cp = capm(pr, spy_fwd)
            # per-year vsSPY
            yr = {}
            for y in sorted(set(pr.index.year)):
                p = (1 + pr[pr.index.year == y].dropna()).prod() - 1
                s = (1 + spy_fwd.reindex(pr.index)[pr.index.year == y].dropna()).prod() - 1
                yr[int(y)] = round((p - s) * 100, 1)
            key = f"{bname}_mom{look}"
            out["buckets"][key] = {"capm": cp, "yearly_vs_spy_pp": yr}
            print(f"\n{key}: alpha {cp.get('alpha_ann_pct',0):+.1f}%/yr (t{cp.get('t_alpha',0):+.2f})  beta {cp.get('beta',0):.2f}  corrSPY {cp.get('corr_spy',0):.2f}", flush=True)
            print("  yr vsSPY: " + "  ".join(f"{y}:{v:+.0f}" for y, v in yr.items()), flush=True)
    # correlation of mega-mom6 with the analyst-revision book, if its return series is stored
    try:
        from core.models import BacktestResult
        ar = BacktestResult.objects.filter(kind="analyst_revision_live").first()
        series = None
        if ar and isinstance(ar.payload, dict):
            for k in ("monthly_returns", "returns", "book_returns"):
                if k in ar.payload:
                    series = ar.payload[k]; break
        print(f"\nanalyst_revision_live series present: {series is not None}", flush=True)
    except Exception as e:
        print("AR corr skipped:", e, flush=True)
    open("/app/.data/studies/rs_momentum_decompose.json", "w").write(json.dumps(out, indent=2, default=float))
    try:
        from core.models import BacktestResult
        from django.utils import timezone
        BacktestResult.objects.update_or_create(kind="rs_momentum_decompose", defaults={"payload": out, "computed_at": timezone.now()})
        print("Saved BacktestResult[rs_momentum_decompose]", flush=True)
    except Exception as e:
        print("save skipped:", e, flush=True)


if __name__ == "__main__":
    main()
