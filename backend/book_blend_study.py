#!/usr/bin/env python3
"""FLAGSHIP + BOOK #2 BLEND — does allocating across the two ~uncorrelated books add ABSOLUTE return (return-priority:
NOT Sharpe/DD)? A fixed-weight monthly-rebalanced blend's return sits BETWEEN the two books UNLESS the rebalancing
bonus (two uncorrelated positive-drift books) lifts COMPOUND return above the better one. Plus the capacity angle:
flagship has a ~$5-10M ceiling, book #2 (tgt_rev_3m large-cap) is high-capacity.

Flagship NET monthly series: from flagship_history.json (FLAGSHIP_TRACE=1 CONFIG=adaptive, perf['monthly'] = deployed
net series). Book #2 NET monthly series: reproduced inline = the VALIDATED PIT-membership costed tgt_rev_3m top-quintile
large-cap book (same code as analyst_revision_book.live()._costed_pit). Aligned on the common window.

Tests: correlation; fixed-weight w*flag + (1-w)*book2, monthly-rebalanced, w in 0..1; compound total / CAGR / Sharpe /
maxDD / vsSPY; diversification return (geom(blend) - weighted-avg geom); the w that MAXIMIZES compound return (corner
=concentrate, interior=blend bonus). Saves BacktestResult[book_blend]. Run: MSYS_NO_PATHCONV=1 docker exec rotation-backend-1 python -u /app/book_blend_study.py"""
import os, json, math
os.environ.setdefault("DJANGO_SETTINGS_MODULE", "rotation.settings")
import django; django.setup()
import numpy as np, pandas as pd
from pathlib import Path
from collections import defaultdict
import price_basis
from seq_fundamental_study import load_candles

ANALYST = "/app/.data/analyst_ratings.jsonl"
FLAG = "/app/.data/studies/flagship_history.json"
MIN_DVOL = 5e6


def book2_series():
    """Reproduce the validated PIT-membership costed tgt_rev_3m top-quintile large-cap book -> net monthly Series."""
    tgt = defaultdict(list)
    for line in Path(ANALYST).read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            r = json.loads(line)
        except Exception:
            continue
        tk, d = r.get("ticker"), r.get("date")
        if tk and d and r.get("price_target"):
            try:
                tgt[tk].append((pd.Timestamp(d), float(r["price_target"])))
            except (TypeError, ValueError):
                pass
    for tk in tgt:
        tgt[tk].sort()
    covered = sorted(tgt)
    from django.db import connection
    from core.models import Candle, FinancialReport
    with connection.cursor() as cur:
        cur.execute("SET max_parallel_workers_per_gather = 0")
    havec = set(Candle.objects.filter(ticker__in=covered, interval="1d").values_list("ticker", flat=True).distinct())
    uni = sorted(set(covered) & havec)
    cand = {}
    for i in range(0, len(uni), 40):
        cand.update(load_candles(uni[i:i + 40]))
    spy = load_candles(["SPY"]).get("SPY")
    spy_m = spy["Close"].resample("ME").last()
    midx = spy_m.index[spy_m.index >= "2016-01-01"]
    midx_ts = np.array([d.value for d in midx], dtype="int64")
    close_m, dvol = {}, {}
    for tk, df in cand.items():
        if df is None or df.empty:
            continue
        close_m[tk] = df["Close"].resample("ME").last().reindex(midx)
        dvol[tk] = (df["Close"] * df["Volume"]).resample("ME").mean().reindex(midx)
    close_m = pd.DataFrame(close_m); dvol = pd.DataFrame(dvol); tickers = list(close_m.columns)

    def consensus_target(tk, win_days=180):
        pts = tgt.get(tk)
        if not pts:
            return pd.Series(np.nan, index=midx)
        di = np.array([t.value for t, _ in pts]); tv = np.array([v for _, v in pts]); w = win_days * 86400 * 10**9
        out = np.full(len(midx), np.nan)
        for j, d in enumerate(midx_ts):
            a = np.searchsorted(di, d - w, side="right"); b = np.searchsorted(di, d, side="right")
            if b > a:
                out[j] = np.median(tv[a:b])
        return pd.Series(out, index=midx)

    ctar = pd.DataFrame({tk: consensus_target(tk) for tk in tickers})
    rev = (ctar - ctar.shift(3)) / close_m.where(close_m > 0)
    liquid = dvol >= MIN_DVOL
    fret = close_m.pct_change().shift(-1)
    frs = defaultdict(list)
    for tk, ad, sh in FinancialReport.objects.filter(ticker__in=tickers, shares_outstanding__gt=0).values_list("ticker", "avail_date", "shares_outstanding"):
        if ad and sh:
            frs[tk].append((pd.Timestamp(ad).value, float(sh)))
    shares_m = {}
    for tk in tickers:
        pts = sorted(frs.get(tk, []))
        if not pts:
            continue
        di = np.array([t for t, _ in pts]); sv = np.array([v for _, v in pts]); o = np.full(len(midx), np.nan)
        for j, dd in enumerate(midx_ts):
            b = np.searchsorted(di, dd, side="right")
            if b > 0:
                o[j] = sv[b - 1]
        shares_m[tk] = pd.Series(o, index=midx)
    shares_m = pd.DataFrame(shares_m).reindex(columns=tickers)
    large_pit = (price_basis.as_traded_close(close_m) * shares_m) >= 2e9
    rets, prev = [], set()
    for dd in midx[:-1]:
        s = rev.loc[dd]; fr = fret.loc[dd]; lq = liquid.loc[dd]; lg = large_pit.loc[dd].fillna(False)
        mm = s.notna() & fr.notna() & lq & lg; names = s[mm]
        if len(names) < 20:
            rets.append(np.nan); prev = set(); continue
        kk = max(1, int(len(names) * 0.2)); top = set(names.nlargest(kk).index)
        turn = len(top ^ prev) / max(1, len(top)); rets.append(float(fr[list(top)].mean()) - (20.0 / 1e4) * turn); prev = top
    b2 = pd.Series(rets, index=midx[:-1]).dropna()
    spyv = spy_m.reindex(midx).pct_change().shift(-1).reindex(b2.index)
    return b2, spyv


def perf(r, spy=None):
    r = r.dropna()
    if len(r) < 12:
        return {"n": int(len(r))}
    eq = (1 + r).prod(); yrs = len(r) / 12.0
    o = {"n": int(len(r)), "total_pct": float((eq - 1) * 100), "cagr_pct": float((eq ** (1 / yrs) - 1) * 100) if eq > 0 else -100.0,
         "sharpe": float(r.mean() / r.std() * math.sqrt(12)) if r.std() > 0 else 0.0,
         "maxdd_pct": float(((1 + r).cumprod() / (1 + r).cumprod().cummax() - 1).min() * 100),
         "geom_mo_pct": float((eq ** (1 / len(r)) - 1) * 100)}
    if spy is not None:
        b = spy.reindex(r.index).fillna(0.0)
        o["vs_spy_pp"] = float(((eq - 1) - ((1 + b).prod() - 1)) * 100)
    return o


def main():
    fj = json.load(open(FLAG))
    fm = fj.get("monthly_net") or []
    flag = pd.Series({pd.Timestamp(d): float(v) for d, v in fm}).sort_index()
    print(f"flagship months {len(flag)} ({flag.index.min().date()}..{flag.index.max().date()}), total {((1+flag).prod()-1)*100:,.0f}%", flush=True)
    b2, spyv = book2_series()
    print(f"book#2 months {len(b2)} ({b2.index.min().date()}..{b2.index.max().date()}), total {((1+b2).prod()-1)*100:,.0f}%", flush=True)

    common = flag.index.intersection(b2.index)
    flag = flag.reindex(common); b2 = b2.reindex(common); spyv = spyv.reindex(common)
    both = pd.concat([flag.rename("flag"), b2.rename("b2")], axis=1).dropna()
    flag = both["flag"]; b2 = both["b2"]; spyv = spyv.reindex(both.index)
    corr = float(flag.corr(b2))
    print(f"\ncommon window {both.index.min().date()}..{both.index.max().date()} ({len(both)}mo)  CORR(flag,book2) = {corr:+.2f}", flush=True)

    pf, pb = perf(flag, spyv), perf(b2, spyv); ps = perf(spyv)
    gflag, gb2 = math.log1p(pf["geom_mo_pct"] / 100), math.log1p(pb["geom_mo_pct"] / 100)
    print(f"\n{'w_flag':>7} {'total':>12} {'CAGR':>7} {'Sh':>5} {'maxDD':>7} {'vsSPY':>9} {'divRet_pp':>9}", flush=True)
    print(f"{'SPY':>7} {ps['total_pct']:>11,.0f}% {ps['cagr_pct']:>6.1f}% {ps['sharpe']:>5.2f} {ps['maxdd_pct']:>6.1f}%", flush=True)
    res = {"corr": corr, "window": [str(both.index.min().date()), str(both.index.max().date())], "n_months": len(both),
           "flagship": pf, "book2": pb, "spy": ps, "blends": {}}
    best_w, best_total = None, -1e9
    for w in [round(x, 2) for x in np.arange(0.0, 1.0001, 0.1)]:
        r = w * flag + (1 - w) * b2
        st = perf(r, spyv)
        # diversification return = geom(blend) - weighted avg of the two geoms (log-space)
        wavg_geom = (math.exp(w * gflag + (1 - w) * gb2) - 1) * 100
        st["div_ret_pp"] = round(st["geom_mo_pct"] - wavg_geom, 4)
        res["blends"][f"{w:.1f}"] = st
        print(f"{w:>7.1f} {st['total_pct']:>11,.0f}% {st['cagr_pct']:>6.1f}% {st['sharpe']:>5.2f} {st['maxdd_pct']:>6.1f}% {st.get('vs_spy_pp',0):>+8.0f} {st['div_ret_pp']:>+9.4f}", flush=True)
        if st["total_pct"] > best_total:
            best_total, best_w = st["total_pct"], w
    res["best_return_w_flag"] = best_w
    interior = 0.0 < best_w < 1.0
    print(f"\nMAX compound return at w_flag={best_w:.1f} ({'INTERIOR -> rebalancing bonus' if interior else 'CORNER -> concentrate in the higher-return book'})", flush=True)
    print(f"  flagship alone {pf['total_pct']:,.0f}% | book#2 alone {pb['total_pct']:,.0f}% | best blend {best_total:,.0f}%", flush=True)
    res["verdict"] = ("interior_blend_bonus" if interior else "concentrate_no_bonus")

    Path("/app/.data/studies/book_blend.json").write_text(json.dumps(res, indent=2, default=float))
    try:
        from core.models import BacktestResult
        from django.utils import timezone
        BacktestResult.objects.update_or_create(kind="book_blend", defaults={"payload": res, "computed_at": timezone.now()})
        print("\nSaved BacktestResult[book_blend]", flush=True)
    except Exception as e:
        print("save skipped:", e, flush=True)


if __name__ == "__main__":
    main()
