#!/usr/bin/env python3
"""DAILY SHORT-VOLUME RATIO — the finer cousin of bi-monthly short interest. Polygon short-volume gives daily
short_volume_ratio (% of the day's volume sold short) = intraday shorting PRESSURE/flow, distinct from short
INTEREST (settlement positioning, already tested & failed). Fetch (cached), resample to monthly mean, and test
per the rules: (A) standalone selector ($5M floor) — quintiles + most/least-shorted-flow long portfolios;
(B) flagship overlay conditional lift + 2x conviction weight. Does daily short-flow carry any edge?
-> BacktestResult[short_volume]. Run: MSYS_NO_PATHCONV=1 docker exec rotation-backend-1 python -u /app/short_volume_study.py
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

CACHE = "/app/.data/short_volume_cache.json"
MIN_DVOL = 5e6
TOP_N = 10


def _fetch_short_volume(tickers):
    cache = {}
    if os.path.exists(CACHE):
        try:
            cache = json.load(open(CACHE))
        except Exception:
            cache = {}
    key = os.environ.get("POLYGON_API_KEY")
    todo = [t for t in tickers if t not in cache]
    print(f"short-volume: {len(cache)} cached, fetching {len(todo)}...", flush=True)
    for i, tk in enumerate(todo):
        rows = []
        url = f"https://api.polygon.io/stocks/v1/short-volume?ticker={tk}&limit=1000&apiKey={key}"
        try:
            with urllib.request.urlopen(url, timeout=25) as r:
                d = json.load(r)
            for e in d.get("results", []):
                dt = e.get("date"); sr = e.get("short_volume_ratio")
                if dt and sr is not None:
                    rows.append([dt, float(sr)])
        except Exception:
            rows = []
        cache[tk] = rows
        if (i + 1) % 100 == 0:
            print(f"  fetched {i+1}/{len(todo)}", flush=True); json.dump(cache, open(CACHE, "w")); time.sleep(0.2)
    json.dump(cache, open(CACHE, "w"))
    return cache


def _stats(r, spy):
    r = np.asarray(r, float); n = len(r)
    if n == 0:
        return dict(total=0, vs_spy=0, sharpe=0, dd=0, win=0)
    tot = float(np.prod(1 + r) - 1) * 100; sp = float(np.prod(1 + np.asarray(spy)) - 1) * 100
    sh = float(r.mean() / r.std() * np.sqrt(12)) if r.std() > 1e-9 else 0.0
    eqc = np.cumprod(1 + r); dd = float(((eqc / np.maximum.accumulate(eqc)) - 1).min() * 100)
    return dict(total=round(tot, 1), vs_spy=round(tot - sp, 1), sharpe=round(sh, 2), dd=round(dd, 1),
                win=round((r > 0).mean() * 100, 1))


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
    fret = px.pct_change().shift(-1); mkt_adj = fret.sub(spy_ret, axis=0)

    sv_raw = _fetch_short_volume(list(common))
    svp = {}
    for tk in common:
        rows = sv_raw.get(tk) or []
        if not rows:
            continue
        s = pd.DataFrame(rows, columns=["d", "sr"]); s["d"] = pd.to_datetime(s["d"])
        svp[tk] = s.set_index("d")["sr"].resample("ME").mean().reindex(midx)
    sv = pd.DataFrame(svp).reindex(index=midx, columns=common)
    print(f"months {len(midx)} | stocks w/ short-vol: {int((~sv.isna()).any().sum())}", flush=True)

    res = {}
    # A. broad quintiles
    ss = sv.stack(); ff = mkt_adj.stack()
    j = pd.concat([ss.rename("s"), ff.rename("r")], axis=1, join="inner").dropna()
    if len(j) > 150:
        j["q"] = pd.qcut(j["s"], 5, labels=False, duplicates="drop")
        b = {int(q): dict(avg=round(float(g["r"].mean()) * 100, 2), win=round(float((g["r"] > 0).mean()) * 100, 1),
                          n=int(len(g)), med=round(float(g["s"].median()), 1)) for q, g in j.groupby("q")}
        res["level_quintiles"] = b
        print(f"\n  short-vol-ratio quintiles (Q0=low..Q4=high short flow) market-adj fwd ret:", flush=True)
        for q in sorted(b):
            print(f"     Q{q} (med {b[q]['med']}%): {b[q]['avg']:>+6}%  win {b[q]['win']}%  n={b[q]['n']}", flush=True)
        print(f"     spread hi-lo: {round(b[max(b)]['avg']-b[min(b)]['avg'],2)}pp", flush=True)

    # B. standalone long portfolios + flagship overlay lift
    most, least, spies = [], [], []
    hi_pick, lo_pick = [], []
    for i in range(9, len(midx) - 1):
        date, ndate = midx[i], midx[i + 1]
        spx = spy_m.iloc[i + 1] / spy_m.iloc[i] - 1
        if not np.isfinite(spx):
            continue
        elig = [t for t in common if _available_at(px[t], date) and pd.notna(dvol.loc[date, t]) and dvol.loc[date, t] >= MIN_DVOL
                and pd.notna(sv.loc[date, t])]
        if len(elig) >= 20:
            svv = sv.loc[date, elig]; hc = svv.quantile(0.8); lc = svv.quantile(0.2)
            def pr(names):
                rr = [_ret_delist(px[t], date, ndate) for t in names]
                return [float(x) for x in rr if x is not None and np.isfinite(x)]
            rm = pr([t for t in elig if svv[t] >= hc]); rl = pr([t for t in elig if svv[t] <= lc])
            most.append(float(np.mean(rm)) if rm else 0.0); least.append(float(np.mean(rl)) if rl else 0.0)
            spies.append(float(spx))
        # flagship pick split
        top = accel.loc[date].dropna().sort_values(ascending=False).head(TOP_N).index
        svrow = sv.loc[date]; med = svrow.median()
        for etf in top:
            _, holds = sector_map.get(etf, (etf, []))
            c = [h for h in holds if h in px.columns and _available_at(px[h], date) and pd.notna(pb.loc[date, h])
                 and pb.loc[date, h] > 0 and not bool(trap.loc[date, h]) and pd.notna(dvol.loc[date, h]) and dvol.loc[date, h] >= MIN_DVOL]
            g = [x for x in c if bool(low.loc[date, x])] or c
            if not g:
                continue
            pick = min(g, key=lambda h: pb.loc[date, h]); rr = fret.loc[date, pick]; svv = svrow.get(pick)
            if pd.isna(rr) or pd.isna(svv) or pd.isna(med):
                continue
            (hi_pick if svv >= med else lo_pick).append(float(rr))
    res["standalone"] = {"most_short_flow": _stats(most, spies), "least_short_flow": _stats(least, spies),
                         "spy_total": round(float(np.prod(1 + np.asarray(spies)) - 1) * 100, 1)}
    lift = (round(float(np.mean(hi_pick)) * 100, 2) - round(float(np.mean(lo_pick)) * 100, 2)) if (hi_pick and lo_pick) else None
    res["flagship_pick_lift_pp"] = lift
    print(f"\n  STANDALONE ($5M floor): most-short-flow {res['standalone']['most_short_flow']['total']}%/Sh{res['standalone']['most_short_flow']['sharpe']} "
          f"vs least {res['standalone']['least_short_flow']['total']}%/Sh{res['standalone']['least_short_flow']['sharpe']} (SPY {res['standalone']['spy_total']}%)", flush=True)
    print(f"  FLAGSHIP pick high-vs-low short-flow lift: {lift}pp (n {len(hi_pick)}/{len(lo_pick)})", flush=True)

    b = res.get("level_quintiles"); spread = round(b[max(b)]["avg"] - b[min(b)]["avg"], 2) if b else None
    payload = {
        "computed_at": pd.Timestamp.utcnow().isoformat(),
        "params": {"benchmark": BENCH, "min_dvol": MIN_DVOL, "months": int(len(midx)), "source": "Polygon daily short_volume_ratio"},
        "results": res, "quintile_spread_pp": spread,
        "verdict": (f"Daily short-volume ratio: quintile hi-lo spread {spread}pp; flagship pick lift {lift}pp; standalone most-flow "
                    f"{res['standalone']['most_short_flow']['total']}% vs SPY {res['standalone']['spy_total']}%. " + (
                    "Daily short-flow shows a usable edge." if (spread is not None and abs(spread) > 1.5) else
                    "Daily short-flow shows NO clean edge — like short interest, daily shorting pressure doesn't separate winners "
                    "on the $5M universe or lift the flagship pick. Confirms shorting-based signals don't beat the value pick.")),
        "caveat": "Polygon daily short_volume_ratio, ~last 1000 trading days/ticker; monthly mean; $5M floor. In-sample, no fees.",
    }
    return payload


def main():
    p = build()
    from pathlib import Path
    Path("/app/.data/studies/short_volume.json").write_text(json.dumps(p, indent=2, default=str))
    try:
        from core.models import BacktestResult
        from django.utils import timezone
        BacktestResult.objects.update_or_create(
            kind="short_volume", defaults={"payload": json.loads(json.dumps(p, default=str)), "computed_at": timezone.now()})
        print("Saved BacktestResult[short_volume]", flush=True)
    except Exception as e:
        print("DB save failed:", e, flush=True)
    print("\n" + p["verdict"], flush=True)


if __name__ == "__main__":
    main()
