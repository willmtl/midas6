#!/usr/bin/env python3
"""VOLUME AS A SIGNAL (not just the $5M liquidity floor) on the flagship pick. The stock-studies engine
already found A/D (accumulation/distribution) DIVERGENCE is 'the edge' in deep-oversold reversals — but we
never tested whether volume/accumulation transfers to the flagship rotation pick. Volume is the overlay most
likely to ADD because it's orthogonal to price/value (unlike news/insider/congress/darkpool/earnings/PE which
all failed). On the flagship pick (cheapest-P/B guard low-debt in accel sectors), PIT:
  (A) conditional forward return by volume state:
      ad_divergence  ADL rising while price fell 3mo (bullish accumulation divergence — the known edge)
      ad_rising      ADL 3mo slope > 0
      vol_surge      20d avg vol / 90d avg vol > 1.15 (interest waking up)
      accumulating   20d $vol rising vs 3mo ago
  (B) strategy: TILT/FILTER the pick toward volume-confirmed names — does it beat the flagship?
-> BacktestResult[volume_overlay]. Run: MSYS_NO_PATHCONV=1 docker exec rotation-backend-1 python -u /app/volume_overlay_study.py
"""
import os, json, warnings
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

TOP_N = 10


def _stats(r, spy):
    r = np.asarray(r, float); n = len(r)
    if n == 0:
        return dict(total=0, vs_spy=0, sharpe=0, dd=0, win=0)
    tot = float(np.prod(1 + r) - 1) * 100; sp = float(np.prod(1 + np.asarray(spy)) - 1) * 100
    sh = float(r.mean() / r.std() * np.sqrt(12)) if r.std() > 1e-9 else 0.0
    eq = np.cumprod(1 + r); dd = float(((eq / np.maximum.accumulate(eq)) - 1).min() * 100)
    return dict(total=round(tot, 1), vs_spy=round(tot - sp, 1), sharpe=round(sh, 2), dd=round(dd, 1),
                win=round((r > 0).mean() * 100, 1))


def _w(x):
    a = np.array(x, float)
    return None if not len(a) else dict(n=len(a), mean_pct=round(a.mean() * 100, 2), win_pct=round((a > 0).mean() * 100, 1))


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
    pb = (price_basis.as_traded_close(px) * sh) / eq.where(eq != 0)
    trap = (ni < 0) & (~(eq >= eq.shift(12))) & (~(ni > ni.shift(4)))
    low = (dt / eq.where(eq != 0)) < 1.0

    # ---- volume feature panels (monthly, from daily OHLCV) ----
    print("building volume/accumulation panels...", flush=True)
    relvol, adl_m, dvol = {}, {}, {}
    for t in common:
        d = stock_daily.get(t)
        if d is None or "Volume" not in d or len(d) < 90:
            continue
        v = d["Volume"]
        relvol[t] = (v.rolling(20).mean() / v.rolling(90).mean()).resample("ME").last().reindex(midx)
        dvol[t] = (d["Close"] * v).rolling(20).mean().resample("ME").last().reindex(midx)
        if {"High", "Low", "Close"}.issubset(d.columns):
            rng = (d["High"] - d["Low"]).replace(0, np.nan)
            mfm = ((d["Close"] - d["Low"]) - (d["High"] - d["Close"])) / rng
            adl_m[t] = (mfm.fillna(0) * v).cumsum().resample("ME").last().reindex(midx)
    relvol = pd.DataFrame(relvol).reindex(index=midx, columns=common)
    dvol = pd.DataFrame(dvol).reindex(index=midx, columns=common)
    adl = pd.DataFrame(adl_m).reindex(index=midx, columns=common)
    ad_slope3 = adl - adl.shift(3)                       # ADL 3mo slope (sign = accumulation/distribution)
    px_ret3 = px.pct_change(3)                           # price 3mo return
    dvol_tr = dvol / dvol.shift(3) - 1                   # $vol accumulation trend
    print(f"months {len(midx)} | stocks {len(common)}", flush=True)

    cond = {"ad_divergence": [], "no_divergence": [], "ad_rising": [], "ad_falling": [],
            "vol_surge": [], "vol_quiet": [], "accumulating": [], "distributing": [], "has_vol": []}
    flag, tilt_div, tilt_surge, filt_div, spies = [], [], [], [], []
    pt = 0
    for i in range(9, len(midx) - 1):
        date, ndate = midx[i], midx[i + 1]
        sp = spy_m.iloc[i + 1] / spy_m.iloc[i] - 1
        if not np.isfinite(sp):
            continue
        top = accel.loc[date].dropna().sort_values(ascending=False).head(TOP_N).index
        rv, adsl, pr3, dtr = relvol.loc[date], ad_slope3.loc[date], px_ret3.loc[date], dvol_tr.loc[date]
        f_slot, td_slot, ts_slot, fd_slot = [], [], [], []
        for etf in top:
            _, holds = sector_map.get(etf, (etf, []))
            c = [h for h in holds if h in px.columns and _available_at(px[h], date) and pd.notna(pb.loc[date, h])
                 and pb.loc[date, h] > 0 and not bool(trap.loc[date, h])
                 and pd.notna(dvol.loc[date, h]) and dvol.loc[date, h] >= 5e6]
            g = [x for x in c if bool(low.loc[date, x])] or c
            if not g:
                continue
            g_sorted = sorted(g, key=lambda h: pb.loc[date, h])
            pick = g_sorted[0]
            r = _ret_delist(px[pick], date, ndate)
            if r is None or not np.isfinite(r):
                continue
            r = float(r); f_slot.append(r); pt += 1
            # conditional buckets on the flagship pick
            adv = pd.notna(adsl.get(pick)) and pd.notna(pr3.get(pick))
            if adv:
                cond["has_vol"].append(r)
                div = (adsl.get(pick) > 0) and (pr3.get(pick) < 0)
                (cond["ad_divergence"] if div else cond["no_divergence"]).append(r)
                (cond["ad_rising"] if adsl.get(pick) > 0 else cond["ad_falling"]).append(r)
            if pd.notna(rv.get(pick)):
                (cond["vol_surge"] if rv.get(pick) > 1.15 else cond["vol_quiet"]).append(r)
            if pd.notna(dtr.get(pick)):
                (cond["accumulating"] if dtr.get(pick) > 0 else cond["distributing"]).append(r)
            # tilt candidates
            div_c = [h for h in g_sorted if pd.notna(adsl.get(h)) and pd.notna(pr3.get(h)) and adsl.get(h) > 0 and pr3.get(h) < 0]
            surge_c = [h for h in g_sorted if pd.notna(rv.get(h)) and rv.get(h) > 1.15]
            for cand_list, slot in ((div_c, td_slot), (surge_c, ts_slot)):
                tp = cand_list[0] if cand_list else pick     # cheapest-P/B among confirmed, else flagship
                tr = _ret_delist(px[tp], date, ndate)
                slot.append(float(tr) if tr is not None and np.isfinite(tr) else r)
            # filter: only take pick if IT has divergence
            if adv and (adsl.get(pick) > 0) and (pr3.get(pick) < 0):
                fd_slot.append(r)
        if f_slot:
            flag.append(float(np.mean(f_slot)))
            tilt_div.append(float(np.mean(td_slot)) if td_slot else float(np.mean(f_slot)))
            tilt_surge.append(float(np.mean(ts_slot)) if ts_slot else float(np.mean(f_slot)))
            filt_div.append(float(np.mean(fd_slot)) if fd_slot else 0.0)
            spies.append(float(sp))

    base = _w(cond["has_vol"])
    print(f"\n=== (A) conditional forward return of FLAGSHIP pick by VOLUME state ===  (has-vol base {base['mean_pct']}%, n{base['n']})", flush=True)
    for k in ("ad_divergence", "no_divergence", "ad_rising", "ad_falling", "vol_surge", "vol_quiet", "accumulating", "distributing"):
        s = _w(cond[k])
        if s:
            print(f"  {k:14} mean {s['mean_pct']:>6}%  win {s['win_pct']}%  (n{s['n']})  ({s['mean_pct']-base['mean_pct']:+.2f}pp)", flush=True)

    fb, td, ts, fd = _stats(flag, spies), _stats(tilt_div, spies), _stats(tilt_surge, spies), _stats(filt_div, spies)
    print(f"\n=== (B) strategy: does a volume signal improve the flagship? ===", flush=True)
    print(f"  flagship (baseline)     total {fb['total']:>7}%  vsSPY {fb['vs_spy']:>7}  Sh {fb['sharpe']}  DD {fb['dd']}%  win {fb['win']}%", flush=True)
    print(f"  + A/D-divergence TILT   total {td['total']:>7}%  vsSPY {td['vs_spy']:>7}  Sh {td['sharpe']}  DD {td['dd']}%  win {td['win']}%  ({td['vs_spy']-fb['vs_spy']:+.1f}pp)", flush=True)
    print(f"  + vol-surge TILT        total {ts['total']:>7}%  vsSPY {ts['vs_spy']:>7}  Sh {ts['sharpe']}  DD {ts['dd']}%  win {ts['win']}%  ({ts['vs_spy']-fb['vs_spy']:+.1f}pp)", flush=True)
    print(f"  + A/D-divergence FILTER total {fd['total']:>7}%  vsSPY {fd['vs_spy']:>7}  Sh {fd['sharpe']}  DD {fd['dd']}%  win {fd['win']}%  ({fd['vs_spy']-fb['vs_spy']:+.1f}pp)", flush=True)

    div, ndiv = _w(cond["ad_divergence"]), _w(cond["no_divergence"])
    payload = {
        "computed_at": pd.Timestamp.utcnow().isoformat(),
        "params": {"top_n": TOP_N, "benchmark": BENCH, "months": int(len(flag))},
        "conditional": {k: _w(v) for k, v in cond.items()},
        "flagship": fb, "ad_div_tilt": td, "vol_surge_tilt": ts, "ad_div_filter": fd,
        "verdict": (f"On the flagship pick: A/D-divergence {div['mean_pct'] if div else 'n/a'}%/{div['win_pct'] if div else '-'}%win vs "
                    f"no-div {ndiv['mean_pct'] if ndiv else 'n/a'}%. Div-TILT {td['vs_spy']-fb['vs_spy']:+.1f}pp, vol-surge-TILT "
                    f"{ts['vs_spy']-fb['vs_spy']:+.1f}pp vs flagship. " + (
                    "VOLUME/ACCUMULATION ADDS — the A/D edge transfers to the rotation pick; first orthogonal overlay to "
                    "improve selection." if (td["vs_spy"] > fb["vs_spy"] + 10 or ts["vs_spy"] > fb["vs_spy"] + 10 or
                    (div and ndiv and div["mean_pct"] - ndiv["mean_pct"] > 1.5)) else
                    "Volume does NOT improve the flagship at portfolio level — like the other overlays, the value+quality "
                    "pick already captures it; A/D divergence is a deep-oversold STOCK-engine edge that doesn't transfer "
                    "to the cheapest-P/B rotation pick. Keep volume as the liquidity FLOOR only.")),
        "caveat": "ADL = cumsum(MFM*Vol) monthly, 3mo slope; PIT. Tilt falls back to flagship when no confirmed candidate. "
                  "In-sample ~5y, no fees.",
    }
    return payload


def main():
    p = build()
    from pathlib import Path
    Path("/app/.data/studies/volume_overlay.json").write_text(json.dumps(p, indent=2, default=str))
    try:
        from core.models import BacktestResult
        from django.utils import timezone
        BacktestResult.objects.update_or_create(
            kind="volume_overlay", defaults={"payload": json.loads(json.dumps(p, default=str)), "computed_at": timezone.now()})
        print("Saved BacktestResult[volume_overlay]", flush=True)
    except Exception as e:
        print("DB save failed:", e, flush=True)
    print("\n" + p["verdict"], flush=True)


if __name__ == "__main__":
    main()
