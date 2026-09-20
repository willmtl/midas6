#!/usr/bin/env python3
"""GROWTH-MOMENTUM BOOK, SURVIVORSHIP-CLEAN (user: "fix that by getting past names" — done, 565 delisted growth-GICS
names fetched). Universe = US/CA stocks in growth GICS (Technology + Healthcare + Communication Services), SURVIVORS
(Fundamental.sector) ∪ DELISTED (delisted_gic.json), so names that went to zero are present during their life. Monthly
top-quintile price-momentum, EW, hold 1mo, costed 20bps. Delisted names exit naturally when their candles end (last-
price basis; no bankruptcy flag available). Compares vs SPY, vs QQQ (growth-beta benchmark), LONG-SHORT spread
(alpha vs beta), both halves — vs the earlier SURVIVORSHIP-BIASED +1436% (current-ETF-holdings) to see how much was
a mirage. Saves BacktestResult[growth_momentum_pit].
Run: MSYS_NO_PATHCONV=1 docker exec rotation-backend-1 python -u /app/growth_momentum_pit.py"""
import os, json, math
os.environ.setdefault("DJANGO_SETTINGS_MODULE", "rotation.settings")
import django; django.setup()
import numpy as np, pandas as pd
import config
from seq_fundamental_study import load_candles
from core.models import Candle, Fundamental

MIN_DVOL = 5e6; COST_BPS = 20.0; SPLIT = "2021-01-01"
GROWTH_GICS = {"Information Technology", "Technology", "Health Care", "Healthcare", "Communication Services"}


def is_usca(tk):
    return ("." not in tk) or tk.rsplit(".", 1)[1] in ("TO", "V")


def stats(r, bench=None, bull=None):
    r = r.dropna()
    if len(r) < 12:
        return {"n": int(len(r))}
    eq = (1 + r).prod(); yrs = len(r) / 12.0
    o = {"n": int(len(r)), "total_pct": float((eq - 1) * 100), "cagr_pct": float((eq ** (1 / yrs) - 1) * 100) if eq > 0 else -100.0,
         "sharpe": float(r.mean() / r.std() * math.sqrt(12)) if r.std() > 0 else 0.0,
         "maxdd_pct": float(((1 + r).cumprod() / (1 + r).cumprod().cummax() - 1).min() * 100)}
    if bench is not None:
        b = bench.reindex(r.index).fillna(0.0)
        o["vs_spy_pp"] = float(((eq - 1) - ((1 + b).prod() - 1)) * 100)
    if bull is not None:
        bl = bull.reindex(r.index).fillna(True)
        o["bull_mean_pct"] = float(r[bl].mean() * 100); o["bear_mean_pct"] = float(r[~bl].mean() * 100)
    return o


def main():
    gic = json.load(open("/app/.data/delisted_gic.json"))
    delisted = [t for t, g in gic.items() if g in GROWTH_GICS and is_usca(t)]
    surv = [t for t in Fundamental.objects.filter(sector__in=list(GROWTH_GICS)).values_list("ticker", flat=True) if is_usca(t)]
    universe = sorted(set(delisted) | set(surv))
    from django.db import connection
    with connection.cursor() as c:
        c.execute("SET max_parallel_workers_per_gather = 0")
    have = set(Candle.objects.filter(ticker__in=universe, interval="1d").values_list("ticker", flat=True).distinct())
    universe = [t for t in universe if t in have]
    print(f"survivorship-clean growth universe: {len(universe)} names ({len(set(delisted)&have)} delisted + {len(set(surv)&have)} survivors, US/CA, growth GICS)", flush=True)

    ed = load_candles(["SPY", "QQQ"])
    spy_m = ed["SPY"]["Close"].resample("ME").last(); qqq_m = ed["QQQ"]["Close"].resample("ME").last()
    midx = spy_m.index[spy_m.index >= "2016-01-01"]
    spy_m = spy_m.reindex(midx); qqq_m = qqq_m.reindex(midx)
    spy_fwd = spy_m.shift(-1) / spy_m - 1.0; qqq_fwd = qqq_m.shift(-1) / qqq_m - 1.0
    spy200 = ed["SPY"]["Close"].rolling(200).mean().resample("ME").last().reindex(midx); bull = spy_m >= spy200

    close_m, dvol_m = {}, {}
    for i in range(0, len(universe), 200):
        rows = Candle.objects.filter(ticker__in=universe[i:i + 200], interval="1d", date__gte="2014-06-01").values_list("ticker", "date", "close", "volume")
        df = pd.DataFrame(list(rows), columns=["ticker", "date", "close", "volume"])
        if df.empty:
            continue
        df["date"] = pd.to_datetime(df["date"]); df["close"] = df["close"].astype(float); df["volume"] = df["volume"].astype(float)
        for tk, g in df.groupby("ticker", sort=False):
            g = g.sort_values("date").set_index("date")
            close_m[tk] = g["close"].resample("ME").last().reindex(midx)
            dvol_m[tk] = (g["close"] * g["volume"]).resample("ME").mean().reindex(midx)
    close_m = pd.DataFrame(close_m); dvol_m = pd.DataFrame(dvol_m)
    tickers = list(close_m.columns)
    fret = close_m.shift(-1) / close_m - 1.0
    liquid = dvol_m >= MIN_DVOL
    SIG = {"mom6": close_m / close_m.shift(6) - 1, "mom12": close_m / close_m.shift(12) - 1,
           "mom12_1": close_m.shift(1) / close_m.shift(12) - 1}

    def book(sig, leg="long", q=0.2):
        rets, prev, ncnt = [], set(), []
        for d in midx[:-1]:
            elig = liquid.loc[d] & close_m.loc[d].notna() & fret.loc[d].notna() & sig.loc[d].notna()
            names = sig.loc[d][elig.reindex(tickers).fillna(False)].dropna()
            if len(names) < 15:
                rets.append(np.nan); prev = set(); continue
            k = max(1, int(len(names) * q)); top = set(names.nlargest(k).index); bot = set(names.nsmallest(k).index)
            fr = fret.loc[d]; ncnt.append(len(top))
            if leg == "ls":
                rets.append(float(fr[list(top)].mean() - fr[list(bot)].mean())); prev = top
            else:
                turn = len(top ^ prev) / max(1, len(top)); rets.append(float(fr[list(top)].mean()) - (COST_BPS / 1e4) * turn); prev = top
        return pd.Series(rets, index=midx[:-1]), float(np.mean(ncnt)) if ncnt else 0

    def halves(r):
        r = r.dropna(); mid = len(r) // 2
        if len(r) < 24:
            return {}
        return {"H1": round(stats(r.iloc[:mid]).get("sharpe", 0), 2), "H2": round(stats(r.iloc[mid:]).get("sharpe", 0), 2)}

    spyv = spy_fwd.reindex(midx[:-1]); qqqv = qqq_fwd.reindex(midx[:-1])
    res = {"universe_n": len(tickers), "spy": stats(spyv), "qqq": stats(qqqv, bench=spyv), "arms": {}}
    print(f"\n=== GROWTH-MOMENTUM SURVIVORSHIP-CLEAN (growth GICS incl delisted, top-quintile, costed {COST_BPS:.0f}bps) ===", flush=True)
    print(f"  SPY {res['spy']['total_pct']:+.0f}%/Sh{res['spy']['sharpe']:.2f} | QQQ {res['qqq']['total_pct']:+.0f}%/Sh{res['qqq']['sharpe']:.2f}", flush=True)
    print(f"  (compare: SURVIVORSHIP-BIASED thematic-holdings run was mom6 +1436%/Sh1.03/LS0.48)", flush=True)
    print(f"{'arm':>10} {'avgN':>5} {'total':>8} {'CAGR':>6} {'Sh':>5} {'DD':>7} {'vsSPY':>7} {'vsQQQ':>7} {'bull/bear':>11} {'LS_Sh':>6} {'H1/H2':>9}", flush=True)
    for sname, sig in SIG.items():
        lg, n = book(sig, "long"); ls, _ = book(sig, "ls")
        st = stats(lg, bench=spyv, bull=bull); ls_st = stats(ls); hv = halves(lg)
        eqb = (1 + lg.dropna()).prod(); vq = float((eqb - 1) * 100 - res["qqq"]["total_pct"])
        st["avg_names"] = round(n, 0); st["vs_qqq_pp"] = vq; st["ls_sharpe"] = ls_st.get("sharpe", 0); st["halves"] = hv
        res["arms"][sname] = st
        print(f"{sname:>10} {n:>5.0f} {st.get('total_pct',0):>+7.0f}% {st.get('cagr_pct',0):>+5.1f}% {st.get('sharpe',0):>5.2f} "
              f"{st.get('maxdd_pct',0):>6.1f}% {st.get('vs_spy_pp',0):>+6.0f} {vq:>+6.0f} {st.get('bull_mean_pct',0):>+5.1f}/{st.get('bear_mean_pct',0):>+4.1f} "
              f"{ls_st.get('sharpe',0):>6.2f} {hv.get('H1',0):>4.2f}/{hv.get('H2',0):>4.2f}", flush=True)

    open("/app/.data/studies/growth_momentum_pit.json", "w").write(json.dumps(res, indent=2, default=float))
    try:
        from core.models import BacktestResult
        from django.utils import timezone
        BacktestResult.objects.update_or_create(kind="growth_momentum_pit", defaults={"payload": res, "computed_at": timezone.now()})
        print("\nSaved BacktestResult[growth_momentum_pit]", flush=True)
    except Exception as e:
        print("save skipped:", e, flush=True)


if __name__ == "__main__":
    main()
