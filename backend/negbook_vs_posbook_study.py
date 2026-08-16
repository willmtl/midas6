#!/usr/bin/env python3
"""FAIR FIGHT: is buying DISTRESSED (neg-book unprofitable) actually better than buying POSITIVE-BOOK?
The prior negbook_long run compared a ~14-name distressed BASKET to SPY, but our flagship 'positive book'
is a single concentrated pick — not apples-to-apples. Here every variant is the SAME thing: a monthly
equal-weight BASKET over the whole universe, $5M/day floor (OTC filtered), so the only difference is WHICH
names go in the basket:
  distressed_nb   neg-book + unprofitable          (the +461% group)
  posbook_all     every positive-book name         (the broad market-ish basket)
  posbook_cheap   positive-book with P/B < 1        (sub-book value = closest pos-book analog to distress)
  allbook_cheap   ANY book sign, cheapest by P/S    (deep-value regardless of book)
Isolates: is the edge NEGATIVE BOOK, or just CHEAPNESS/DISTRESS? -> BacktestResult[negbook_vs_posbook].
Run: MSYS_NO_PATHCONV=1 docker exec rotation-backend-1 python -u /app/negbook_vs_posbook_study.py
"""
import os, json, warnings
warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
os.environ.setdefault("DJANGO_SETTINGS_MODULE", "rotation.settings")
import django
django.setup()

import config, sector_holdings
from seq_fundamental_study import load_candles, load_financial_reports
from trend_stock_studies import _pit_monthly_panel, _available_at, _ret_delist, CRYPTO
from backtest_lowpb import _monthly_close, BENCH


def _st(rets, spy):
    r = np.asarray(rets, float); n = len(r)
    if n == 0:
        return dict(total=0, vs_spy=0, sharpe=0, dd=0, win=0)
    tot = float(np.prod(1 + r) - 1) * 100; sp = float(np.prod(1 + np.asarray(spy)) - 1) * 100
    sh = float(r.mean() / r.std() * np.sqrt(12)) if r.std() > 1e-9 else 0.0
    eq = np.cumprod(1 + r); dd = float(((eq / np.maximum.accumulate(eq)) - 1).min() * 100)
    return dict(total=round(tot, 1), vs_spy=round(tot - sp, 1), sharpe=round(sh, 2), dd=round(dd, 1),
                win=round((r > 0).mean() * 100, 1))


def build():
    etfs = {n: e for n, e in config.SECTOR_ETFS.items() if e not in CRYPTO}
    all_holds = set()
    for n, e in etfs.items():
        for t in sector_holdings.get_holdings(n):
            if t not in (e, BENCH) and t not in CRYPTO:
                all_holds.add(t)
    all_holds = sorted(all_holds)
    etf_tk = list(etfs.values())
    print(f"Loading {len(etf_tk)} ETFs + {len(all_holds)} stocks + {BENCH}...", flush=True)
    etf_daily = load_candles(etf_tk + [BENCH])
    midx = _monthly_close({t: d for t, d in etf_daily.items() if t in etf_tk}).index
    spy_m = etf_daily[BENCH]["Close"].resample("ME").last().reindex(midx)
    stock_daily = load_candles(all_holds)
    stock_m = _monthly_close(stock_daily).reindex(midx)
    reps = load_financial_reports(all_holds)
    sh, eq, ni, rev = (_pit_monthly_panel(reps, f, midx) for f in
                       ("shares_outstanding", "total_equity", "net_income", "revenue"))
    common = stock_m.columns.intersection(sh.columns).intersection(eq.columns)
    R = lambda p: p.reindex(index=midx, columns=common)
    px = stock_m[common]; sh, eq, ni, rev = R(sh), R(eq), R(ni), R(rev)
    mktcap = px * sh
    pb = mktcap / eq.where(eq != 0)
    ps = mktcap / rev.where(rev > 0)
    dvol = pd.DataFrame({t: (stock_daily[t]["Close"] * stock_daily[t]["Volume"]).rolling(20).mean()
                         .resample("ME").last().reindex(midx) for t in common
                         if t in stock_daily and "Volume" in stock_daily[t]}).reindex(index=midx, columns=common)
    print(f"months {len(midx)} | stocks {len(common)}", flush=True)

    keys = ["distressed_nb", "posbook_all", "posbook_cheap", "allbook_cheap"]
    port = {k: [] for k in keys}; nn = {k: [] for k in keys}; spies = []
    for i in range(9, len(midx) - 1):
        date, ndate = midx[i], midx[i + 1]
        sp = spy_m.iloc[i + 1] / spy_m.iloc[i] - 1
        if not np.isfinite(sp):
            continue
        buckets = {k: [] for k in keys}
        for h in common:
            if not _available_at(px[h], date):
                continue
            if not (pd.notna(dvol.loc[date, h]) and dvol.loc[date, h] >= 5e6):
                continue                                  # OTC / thin filtered
            e_ = eq.loc[date, h]
            if pd.isna(e_):
                continue
            r = _ret_delist(px[h], date, ndate)
            if r is None or not np.isfinite(r):
                continue
            r = float(r); pbv, psv, n_ = pb.loc[date, h], ps.loc[date, h], ni.loc[date, h]
            if e_ < 0:
                if pd.notna(n_) and n_ <= 0:
                    buckets["distressed_nb"].append(r)
            else:
                buckets["posbook_all"].append(r)
                if pd.notna(pbv) and pbv < 1.0:
                    buckets["posbook_cheap"].append(r)
            if pd.notna(psv):
                buckets["allbook_cheap"].append((psv, r))   # rank later
        # allbook_cheap = cheapest-P/S quartile basket (any book sign)
        ab = buckets["allbook_cheap"]
        if ab:
            ab_sorted = sorted(ab, key=lambda x: x[0])
            cut = max(1, len(ab_sorted) // 4)
            buckets["allbook_cheap"] = [r for _, r in ab_sorted[:cut]]
        any_ = False
        for k in keys:
            if buckets[k]:
                port[k].append(float(np.mean(buckets[k]))); nn[k].append(len(buckets[k])); any_ = True
            else:
                port[k].append(0.0); nn[k].append(0)
        if any_:
            spies.append(float(sp))

    print("\n=== FAIR FIGHT: same equal-weight BASKET harness ($5M floor) ===", flush=True)
    res = {}
    for k in keys:
        s = _st(port[k][:len(spies)], spies); s["avg_names"] = round(float(np.mean(nn[k])), 1)
        res[k] = s
        print(f"  {k:14} total {s['total']:>7}%  vsSPY {s['vs_spy']:>7}%  Sh {s['sharpe']:>5}  DD {s['dd']:>7}%  "
              f"win {s['win']}%  ~{s['avg_names']} names/mo", flush=True)

    d, p = res["distressed_nb"], res["posbook_cheap"]
    payload = {
        "computed_at": pd.Timestamp.utcnow().isoformat(),
        "params": {"benchmark": BENCH, "months": int(len(midx)), "floor_usd": 5e6},
        "results": res,
        "verdict": (f"Apples-to-apples baskets: distressed-negbook {d['total']}% (Sh {d['sharpe']}, DD {d['dd']}%, "
                    f"~{d['avg_names']} names) vs positive-book-cheap {p['total']}% (Sh {p['sharpe']}, DD {p['dd']}%, "
                    f"~{p['avg_names']} names). " + (
                    "Distressed wins on RETURN but check Sharpe/DD/breadth — its edge is fewer names + fatter tail "
                    "(more idiosyncratic risk), not free alpha." if d['total'] > p['total'] else
                    "Positive-book-cheap holds up — the distressed premium shrinks once you compare like-for-like baskets.")),
        "caveat": "In-sample ~5y (speculative-mania regime), no fees. distressed basket is small (~14) = concentrated; "
                  "posbook_all is hundreds = market-like. Neg-book un-rankable by P/B so eq-wt-all.",
    }
    return payload


def main():
    p = build()
    from pathlib import Path
    OUT = Path(__file__).resolve().parent / ".data" / "studies" / "negbook_vs_posbook.json"
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(p, indent=2, default=str))
    try:
        from core.models import BacktestResult
        from django.utils import timezone
        BacktestResult.objects.update_or_create(
            kind="negbook_vs_posbook", defaults={"payload": json.loads(json.dumps(p, default=str)), "computed_at": timezone.now()})
        print("Saved BacktestResult[negbook_vs_posbook]", flush=True)
    except Exception as e:
        print("DB save failed:", e, flush=True)
    print("\n" + p["verdict"], flush=True)


if __name__ == "__main__":
    main()
