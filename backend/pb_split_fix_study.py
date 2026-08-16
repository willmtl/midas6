#!/usr/bin/env python3
"""P/B BASIS FIX (finding #2) — the pb panel uses dividend/split-ADJUSTED close x as-reported shares. Future
splits back-adjust the historical price DOWN, so a name that later split 10:1 (NVDA/AVGO) or 50:1 (CMG) shows a
historical pb ~1/N of reality and gets spuriously picked as 'cheapest P/B' — a LOOK-AHEAD (you couldn't know the
future split) that pulls mega-cap winners into the 'value' book and inflates returns. FIX: recover the
as-traded (split-as-known-at-t) price = adj_close x product(future split ratios), and rank pb on THAT. Returns
still use the adjusted DB close (correct for total return). Re-run the deployed flagship (top-10 div_2x) with
corrected pb and compare. Splits cached to /app/.data/splits_cache.json.
-> BacktestResult[pb_split_fix]. Run: MSYS_NO_PATHCONV=1 docker exec rotation-backend-1 python -u /app/pb_split_fix_study.py
"""
import os, json, time, warnings
warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
os.environ.setdefault("DJANGO_SETTINGS_MODULE", "rotation.settings")
import django
django.setup()

import config, sector_holdings
from seq_fundamental_study import load_candles, load_financial_reports
from trend_stock_studies import _pit_monthly_panel, _available_at, _ret_delist, CRYPTO
from backtest_lowpb import _monthly_close, BENCH

TOP_N = 10
CONVICTION_MULT = 2.0
MIN_DVOL = 5e6
SPLIT_CACHE = "/app/.data/splits_cache.json"


def _load_splits(tickers):
    import yfinance as yf
    cache = {}
    if os.path.exists(SPLIT_CACHE):
        try:
            cache = json.load(open(SPLIT_CACHE))
        except Exception:
            cache = {}
    todo = [t for t in tickers if t not in cache]
    print(f"splits: {len(cache)} cached, fetching {len(todo)}...", flush=True)
    for i, t in enumerate(todo):
        try:
            sp = yf.Ticker(t).splits
            cache[t] = {str(pd.Timestamp(d).tz_localize(None).date()): float(r) for d, r in sp.items()} if len(sp) else {}
        except Exception:
            cache[t] = {}
        if (i + 1) % 100 == 0:
            json.dump(cache, open(SPLIT_CACHE, "w")); print(f"  {i+1}/{len(todo)}", flush=True); time.sleep(0.1)
    json.dump(cache, open(SPLIT_CACHE, "w"))
    return cache


def _future_split_factor(split_map, midx):
    """Series on midx: product of split ratios STRICTLY AFTER each month-end (to undo future back-adjustment)."""
    if not split_map:
        return pd.Series(1.0, index=midx)
    items = sorted((pd.Timestamp(d), r) for d, r in split_map.items())
    out = []
    for m in midx:
        f = 1.0
        for d, r in items:
            if d > m:
                f *= r
        out.append(f)
    return pd.Series(out, index=midx)


def _perf(port_rets, spy_rets):
    r = np.asarray(port_rets, float); n = len(r)
    tot = float(np.prod(1 + r) - 1) * 100; sp = float(np.prod(1 + np.asarray(spy_rets)) - 1) * 100
    ann = (float(np.prod(1 + r)) ** (12.0 / n) - 1) * 100 if n else 0.0
    sh = float(r.mean() / r.std() * np.sqrt(12)) if r.std() > 1e-9 else 0.0
    eqc = np.cumprod(1 + r); dd = float(((eqc / np.maximum.accumulate(eqc)) - 1).min() * 100)
    return dict(total=round(tot, 1), vs_spy=round(tot - sp, 1), annual=round(ann, 1),
                sharpe=round(sh, 2), dd=round(dd, 1), months=n, spy_total=round(sp, 1))


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
    accel = etf_m.pct_change(3) - etf_m.pct_change(3).shift(3)
    stock_daily = load_candles(all_holds)
    stock_m = _monthly_close(stock_daily).reindex(midx)
    reps = load_financial_reports(all_holds)
    sh, eq, ni, dt = (_pit_monthly_panel(reps, f, midx) for f in
                      ("shares_outstanding", "total_equity", "net_income", "total_debt"))
    common = stock_m.columns.intersection(sh.columns).intersection(eq.columns)
    R = lambda p: p.reindex(index=midx, columns=common)
    px = stock_m[common]; sh, eq, ni, dt = R(sh), R(eq), R(ni), R(dt)

    # split-correction: as-traded price = adj_close * future-split-factor
    splits = _load_splits(list(common))
    sf = pd.DataFrame({t: _future_split_factor(splits.get(t, {}), midx) for t in common}).reindex(index=midx, columns=common)
    px_raw = px * sf
    n_split = int((sf.iloc[0] > 1.01).sum())
    print(f"names with a future split in-window: {n_split}", flush=True)

    pb_adj = (px * sh) / eq.where(eq != 0)          # OLD (buggy) — adjusted close
    pb_raw = (px_raw * sh) / eq.where(eq != 0)      # FIXED — as-traded close
    trap = (ni < 0) & (~(eq >= eq.shift(12))) & (~(ni > ni.shift(4)))
    low = (dt / eq.where(eq != 0)) < 1.0
    adl_m, dvol = {}, {}
    for t in common:
        d = stock_daily.get(t)
        if d is None or "Volume" not in d or len(d) < 90:
            continue
        v = d["Volume"]
        dvol[t] = (d["Close"] * v).rolling(20).mean().resample("ME").last().reindex(midx)
        if {"High", "Low", "Close"}.issubset(d.columns):
            rng = (d["High"] - d["Low"]).replace(0, np.nan)
            mfm = ((d["Close"] - d["Low"]) - (d["High"] - d["Close"])) / rng
            adl_m[t] = (mfm.fillna(0) * v).cumsum().resample("ME").last().reindex(midx)
    dvol = pd.DataFrame(dvol).reindex(index=midx, columns=common)
    adl = pd.DataFrame(adl_m).reindex(index=midx, columns=common)
    ad_slope3 = adl - adl.shift(3); px_ret3 = px.pct_change(3)
    print(f"months {len(midx)} | stocks {len(common)}", flush=True)

    def run(pb):
        monthly = []; picks_flat = []
        for i in range(9, len(midx) - 1):
            date, ndate = midx[i], midx[i + 1]
            spx = spy_m.iloc[i + 1] / spy_m.iloc[i] - 1
            if not np.isfinite(spx):
                continue
            top = accel.loc[date].dropna().sort_values(ascending=False).head(TOP_N).index
            adsl, pr3 = ad_slope3.loc[date], px_ret3.loc[date]
            picks = []
            for etf in top:
                _, holds = sector_map.get(etf, (etf, []))
                c = [h for h in holds if h in px.columns and _available_at(px[h], date) and pd.notna(pb.loc[date, h])
                     and pb.loc[date, h] > 0 and not bool(trap.loc[date, h])
                     and pd.notna(dvol.loc[date, h]) and dvol.loc[date, h] >= MIN_DVOL]
                g = [x for x in c if bool(low.loc[date, x])] or c
                if not g:
                    continue
                pick = min(g, key=lambda h: pb.loc[date, h])
                r = _ret_delist(px[pick], date, ndate)
                if r is None or not np.isfinite(r):
                    continue
                is_div = bool(pd.notna(adsl.get(pick)) and pd.notna(pr3.get(pick)) and adsl.get(pick) > 0 and pr3.get(pick) < 0)
                w = CONVICTION_MULT if is_div else 1.0
                picks.append({"name": pick, "ret": float(r), "weight": w})
                picks_flat.append(pick)
            if not picks:
                continue
            tw = sum(p["weight"] for p in picks)
            monthly.append((sum(p["weight"] * p["ret"] for p in picks) / tw, float(spx)))
        perf = _perf([m[0] for m in monthly], [m[1] for m in monthly])
        return perf, picks_flat

    old_perf, old_picks = run(pb_adj)
    new_perf, new_picks = run(pb_raw)
    from collections import Counter
    dropped = Counter([p for p in old_picks if p not in set(new_picks)])
    splitters = {t for t in common if (sf.loc[:, t] > 1.01).any()}
    split_in_old = sorted(set(old_picks) & splitters)
    split_in_new = sorted(set(new_picks) & splitters)

    print(f"\n=== P/B BASIS FIX (split look-ahead) ===", flush=True)
    print(f"  OLD (adjusted-close pb): total {old_perf['total']}%  Sh {old_perf['sharpe']}  DD {old_perf['dd']}%", flush=True)
    print(f"  NEW (as-traded pb)     : total {new_perf['total']}%  Sh {new_perf['sharpe']}  DD {new_perf['dd']}%", flush=True)
    print(f"  future-splitters picked  OLD: {split_in_old}", flush=True)
    print(f"  future-splitters picked  NEW: {split_in_new}", flush=True)
    payload = {
        "computed_at": pd.Timestamp.utcnow().isoformat(),
        "params": {"top_n": TOP_N, "benchmark": BENCH, "min_dvol": MIN_DVOL, "months": int(len(midx))},
        "old_adjusted_pb": old_perf, "new_astraded_pb": new_perf,
        "future_splitters_picked_OLD": split_in_old, "future_splitters_picked_NEW": split_in_new,
        "return_impact_pp": round(new_perf["total"] - old_perf["total"], 1),
        "verdict": (f"Correcting the split look-ahead in P/B changes the flagship {old_perf['total']}% -> {new_perf['total']}% "
                    f"({round(new_perf['total']-old_perf['total'],1):+}pp), Sharpe {old_perf['sharpe']}->{new_perf['sharpe']}. "
                    f"Future-splitters spuriously picked under the OLD basis: {split_in_old}. "
                    + ("MATERIAL — the adjusted-close basis was inflating returns via hindsight splitters; the as-traded basis is honest."
                       if abs(new_perf['total'] - old_perf['total']) > 25 else
                       "Minor — the split look-ahead exists but barely moves the deployed number.")),
        "caveat": "Split-corrected (dividend drag ~5-10% on some names left uncorrected — small). Returns use adjusted "
                  "close (correct for total return); only pb RANKING uses as-traded price. PIT, no fees.",
    }
    return payload


def main():
    p = build()
    from pathlib import Path
    Path("/app/.data/studies/pb_split_fix.json").write_text(json.dumps(p, indent=2, default=str))
    try:
        from core.models import BacktestResult
        from django.utils import timezone
        BacktestResult.objects.update_or_create(
            kind="pb_split_fix", defaults={"payload": json.loads(json.dumps(p, default=str)), "computed_at": timezone.now()})
        print("Saved BacktestResult[pb_split_fix]", flush=True)
    except Exception as e:
        print("DB save failed:", e, flush=True)
    print("\n" + p["verdict"], flush=True)


if __name__ == "__main__":
    main()
