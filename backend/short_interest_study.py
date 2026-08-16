#!/usr/bin/env python3
"""SHORT INTEREST as a signal — "how shorted is a stock, does it help?" Polygon short-interest endpoint has
HISTORY (bi-monthly settlements back to 2017): short_interest (shares), avg_daily_volume, days_to_cover. We
fetch it for the universe (cached), build PIT monthly panels with a publication LAG (settlement disseminated
~10 trading days later — only use settlements known by the rebalance date), and test on the flagship value
pick: (A) broad conditional fwd-return by short%-of-shares / days-to-cover quintile; (B) flagship overlay —
does high (squeeze fuel / contrarian) or low short interest help the pick? PIT, market-adjusted.
-> BacktestResult[short_interest]. Run: MSYS_NO_PATHCONV=1 docker exec rotation-backend-1 python -u /app/short_interest_study.py
"""
import os, json, time, warnings, urllib.request
warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
os.environ.setdefault("DJANGO_SETTINGS_MODULE", "rotation.settings")
import django
django.setup()

import config, sector_holdings
import price_basis
from seq_fundamental_study import load_candles, load_financial_reports
from trend_stock_studies import _pit_monthly_panel, _available_at, _ret_delist, CRYPTO
from backtest_lowpb import _monthly_close, BENCH

CACHE = "/app/.data/short_interest_cache.json"
PUB_LAG_D = 12        # trading-day-ish publication lag: only use settlements <= date - PUB_LAG_D (PIT-safe)
TOP_N = 10


def _fetch_short_interest(tickers):
    """Polygon short-interest per ticker -> {ticker: [(settlement_date, short_interest, days_to_cover)]}. Cached."""
    cache = {}
    if os.path.exists(CACHE):
        try:
            cache = json.load(open(CACHE))
        except Exception:
            cache = {}
    key = os.environ.get("POLYGON_API_KEY")
    if not key:
        print("WARNING: POLYGON_API_KEY missing — short-interest fetch will fail; results may be a false null.", flush=True)
    # retry tickers that are MISSING or have an EMPTY cached list (empty = never successfully fetched, don't treat as done)
    todo = [t for t in tickers if not cache.get(t)]
    print(f"short-interest: {sum(1 for v in cache.values() if v)} cached w/ data, fetching {len(todo)}...", flush=True)
    for i, tk in enumerate(todo):
        rows = []
        url = f"https://api.polygon.io/stocks/v1/short-interest?ticker={tk}&limit=1000&apiKey={key}"
        try:
            with urllib.request.urlopen(url, timeout=25) as r:
                d = json.load(r)
            for e in d.get("results", []):
                sd = e.get("settlement_date"); si = e.get("short_interest"); dtc = e.get("days_to_cover")
                if sd and si is not None:
                    rows.append([sd, float(si), float(dtc) if dtc is not None else None])
        except Exception:
            rows = []
        # only persist a ticker once the fetch actually returned rows — never poison the cache with [] on a transient failure
        if rows:
            cache[tk] = rows
        if (i + 1) % 100 == 0:
            print(f"  fetched {i+1}/{len(todo)}", flush=True)
            json.dump(cache, open(CACHE, "w"))
            time.sleep(0.2)
    json.dump(cache, open(CACHE, "w"))
    return {t: cache.get(t, []) for t in tickers}


def build():
    etfs = {n: e for n, e in config.SECTOR_ETFS.items() if e not in CRYPTO}
    sector_map, all_holds = {}, set()
    for n, e in etfs.items():
        h = [t for t in sector_holdings.get_holdings(n) if t not in (e, BENCH) and t not in CRYPTO]
        sector_map[e] = (n, h); all_holds.update(h)
    all_holds = sorted(all_holds)
    etf_tk = list(etfs.values())
    print(f"Loading {len(etf_tk)} ETFs + {len(all_holds)} stocks + {BENCH}...", flush=True)
    etf_daily = load_candles(etf_tk + [BENCH])
    etf_m = _monthly_close({t: d for t, d in etf_daily.items() if t in etf_tk})
    midx = etf_m.index
    spy_m = etf_daily[BENCH]["Close"].resample("ME").last().reindex(midx)
    spy_ret = spy_m.pct_change().shift(-1)
    accel = etf_m.pct_change(3) - etf_m.pct_change(3).shift(3)
    stock_daily = load_candles(all_holds)
    stock_m = _monthly_close(stock_daily).reindex(midx)
    reps = load_financial_reports(all_holds)
    sh, eq, ni, dt = (_pit_monthly_panel(reps, f, midx) for f in
                      ("shares_outstanding", "total_equity", "net_income", "total_debt"))
    common = stock_m.columns.intersection(sh.columns).intersection(eq.columns)
    R = lambda p: p.reindex(index=midx, columns=common)
    px = stock_m[common]; sh, eq, ni, dt = R(sh), R(eq), R(ni), R(dt)
    pb = (price_basis.as_traded_close(px) * sh) / eq.where(eq != 0)
    trap = (ni < 0) & (~(eq >= eq.shift(12))) & (~(ni > ni.shift(4)))
    low = (dt / eq.where(eq != 0)) < 1.0
    dvol = pd.DataFrame({t: (stock_daily[t]["Close"] * stock_daily[t]["Volume"]).rolling(20).mean()
                         .resample("ME").last().reindex(midx) for t in common
                         if t in stock_daily and "Volume" in stock_daily[t]}).reindex(index=midx, columns=common)
    fret = px.pct_change().shift(-1)
    mkt_adj = fret.sub(spy_ret, axis=0)

    si_raw = _fetch_short_interest(list(common))
    # build PIT panels: at each month-end use the latest settlement known by (date - PUB_LAG_D)
    si_sh, dtc = {}, {}
    lag = pd.Timedelta(days=int(PUB_LAG_D * 1.5))     # calendar buffer over trading-day lag
    for tk in common:
        rows = si_raw.get(tk) or []
        if not rows:
            continue
        s = pd.DataFrame(rows, columns=["sd", "si", "dtc"]); s["sd"] = pd.to_datetime(s["sd"])
        s = s.sort_values("sd")
        si_series, dtc_series = [], []
        for d in midx:
            avail = s[s["sd"] <= (d - lag)]
            if len(avail):
                si_series.append(avail["si"].iloc[-1]); dtc_series.append(avail["dtc"].iloc[-1])
            else:
                si_series.append(np.nan); dtc_series.append(np.nan)
        si_sh[tk] = pd.Series(si_series, index=midx); dtc[tk] = pd.Series(dtc_series, index=midx)
    si_sh = pd.DataFrame(si_sh).reindex(index=midx, columns=common)
    dtc = pd.DataFrame(dtc).reindex(index=midx, columns=common)
    short_pct = (si_sh / sh.where(sh > 0)) * 100        # short interest as % of shares outstanding
    have = int((~short_pct.isna()).any().sum())
    print(f"months {len(midx)} | stocks {len(common)} | stocks w/ short data: {have}", flush=True)

    def _buckets(sig, qn=5):
        s = sig.stack(); f = mkt_adj.stack()
        j = pd.concat([s.rename("s"), f.rename("r")], axis=1, join="inner").dropna()
        if len(j) < qn * 30:
            return None, len(j)
        try:
            j["q"] = pd.qcut(j["s"], qn, labels=False, duplicates="drop")
        except Exception:
            return None, len(j)
        return {int(q): dict(avg=round(float(g["r"].mean()) * 100, 2), win=round(float((g["r"] > 0).mean()) * 100, 1),
                             n=int(len(g)), med=round(float(g["s"].median()), 2)) for q, g in j.groupby("q")}, len(j)

    res = {}
    for name, sig in [("short_pct_shares", short_pct), ("days_to_cover", dtc)]:
        b, n = _buckets(sig, 5); res[f"level_{name}"] = {"buckets": b, "n": n}
        if b:
            print(f"\n  {name} quintiles (Q0=low short..Q{max(b)}=high short) market-adj fwd ret:", flush=True)
            for q in sorted(b):
                print(f"     Q{q} (med {b[q]['med']}): {b[q]['avg']:>+6}%  win {b[q]['win']}%  n={b[q]['n']}", flush=True)
            print(f"     spread hi-lo: {round(b[max(b)]['avg']-b[min(b)]['avg'],2)}pp", flush=True)

    # flagship overlay: split the value pick by its short_pct (relative to that month's picks)
    hi, lo, no = [], [], []
    for i in range(9, len(midx) - 1):
        date = midx[i]
        top = accel.loc[date].dropna().sort_values(ascending=False).head(TOP_N).index
        sprow = short_pct.loc[date]; med = sprow.median()
        for etf in top:
            _, holds = sector_map.get(etf, (etf, []))
            c = [h for h in holds if h in px.columns and _available_at(px[h], date) and pd.notna(pb.loc[date, h])
                 and pb.loc[date, h] > 0 and not bool(trap.loc[date, h])
                 and pd.notna(dvol.loc[date, h]) and dvol.loc[date, h] >= 5e6]
            g = [x for x in c if bool(low.loc[date, x])] or c
            if not g:
                continue
            pick = min(g, key=lambda h: pb.loc[date, h]); r = fret.loc[date, pick]; spv = sprow.get(pick)
            if pd.isna(r):
                continue
            if pd.isna(spv) or pd.isna(med):
                no.append(float(r))
            elif spv >= med:
                hi.append(float(r))
            else:
                lo.append(float(r))
    def mean(x):
        return round(float(np.mean(x)) * 100, 2) if x else None
    fl = {"pick_high_short": {"avg": mean(hi), "n": len(hi)}, "pick_low_short": {"avg": mean(lo), "n": len(lo)},
          "pick_no_data": {"avg": mean(no), "n": len(no)}}
    lift = (fl["pick_high_short"]["avg"] - fl["pick_low_short"]["avg"]) if (hi and lo) else None
    res["flagship_pick_split"] = {**fl, "high_minus_low_pp": lift}
    print(f"\n  FLAGSHIP value pick by short interest: high {fl['pick_high_short']['avg']}% (n={len(hi)}) vs "
          f"low {fl['pick_low_short']['avg']}% (n={len(lo)}) -> lift {lift}pp  [no-data {fl['pick_no_data']['avg']}% n={len(no)}]", flush=True)

    sp_b = res.get("level_short_pct_shares", {}).get("buckets")
    spread = round(sp_b[max(sp_b)]["avg"] - sp_b[min(sp_b)]["avg"], 2) if sp_b else None
    payload = {
        "computed_at": pd.Timestamp.utcnow().isoformat(),
        "params": {"top_n": TOP_N, "benchmark": BENCH, "pub_lag_days": PUB_LAG_D, "months": int(len(midx)),
                   "source": "Polygon short-interest (bi-monthly settlements, PIT-lagged)", "stocks_with_data": have},
        "results": res, "short_pct_hi_lo_spread_pp": spread,
        "verdict": (f"Short interest: short%-shares quintile hi-lo spread = {spread}pp; flagship pick high-vs-low short lift = {lift}pp. " + (
            "HIGH short interest predicts LOWER forward returns (justified bearishness dominates squeeze) — a short-side/AVOID "
            "signal, useful as a NEGATIVE screen on the value pick." if (spread is not None and spread < -1.0) else
            "LOW short interest / heavily-shorted contrarian bounce shows up (squeeze) — high short = higher fwd return." if (spread is not None and spread > 1.0) else
            "Short interest shows NO clean monotone edge on the forward return in this universe — like the other alt-data overlays, "
            "it doesn't cleanly separate winners on the value pick.")),
        "caveat": "Polygon short-interest bi-monthly, PIT-lagged ~2wk for dissemination; short%-shares approximates short%-float. "
                  "In-sample, no fees. Foreign tickers lack Polygon short data (NaN).",
    }
    return payload


def main():
    p = build()
    from pathlib import Path
    Path("/app/.data/studies/short_interest.json").write_text(json.dumps(p, indent=2, default=str))
    try:
        from core.models import BacktestResult
        from django.utils import timezone
        BacktestResult.objects.update_or_create(
            kind="short_interest", defaults={"payload": json.loads(json.dumps(p, default=str)), "computed_at": timezone.now()})
        print("Saved BacktestResult[short_interest]", flush=True)
    except Exception as e:
        print("DB save failed:", e, flush=True)
    print("\n" + p["verdict"], flush=True)


if __name__ == "__main__":
    main()
