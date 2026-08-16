#!/usr/bin/env python3
"""EARNINGS-SURPRISE axis — does EPS surprise / grounded-earnings / guidance add to or time the value pick?
The one genuinely unmined signal: PEAD (post-earnings-announcement drift) is a documented anomaly, and we
built `grounded_earnings` (EPS surprise + forward guidance -> grounded_label/score) but never used it for
ROTATION selection. On the flagship pick (cheapest-P/B guard low-debt in accel sectors), PIT on report_date:
  (A) conditional forward return by recent-quarter: EPS beat vs miss; grounded positive vs negative; guidance
      raised vs cut; and a FRESH-beat (earnings in last ~45d = PEAD window) cut.
  (B) strategy variants vs flagship: TILT (prefer grounded-positive candidate in each sector) and FILTER
      (only take picks with a recent grounded-positive quarter).
-> BacktestResult[earnings_overlay].
Run: MSYS_NO_PATHCONV=1 docker exec rotation-backend-1 python -u /app/earnings_overlay_study.py
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
FRESH_DAYS = 45      # PEAD window: earnings within last ~45d


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


def _earnings_panels(tickers, midx):
    """PIT monthly panels (ffill by report_date) of the MOST-RECENT quarter's fields + last report_date."""
    from core.models import EarningsEvent
    qs = (EarningsEvent.objects.filter(ticker__in=list(tickers), report_date__isnull=False)
          .values_list("ticker", "report_date", "eps_surprise_pct", "grounded_score", "guidance_eps_pct"))
    df = pd.DataFrame.from_records(list(qs), columns=["ticker", "rdate", "eps_surp", "gscore", "guid"])
    if df.empty:
        return {}
    df["rdate"] = pd.to_datetime(df["rdate"])
    df = df[(df["rdate"] >= midx.min() - pd.Timedelta(days=200)) & (df["rdate"] <= midx.max())]
    panels = {f: {} for f in ["eps_surp", "gscore", "guid", "rdate_num"]}
    for tk, g in df.groupby("ticker", sort=False):
        g = g.sort_values("rdate").drop_duplicates("rdate", keep="last").set_index("rdate")
        for f in ["eps_surp", "gscore", "guid"]:
            s = pd.to_numeric(g[f], errors="coerce")
            panels[f][tk] = s.reindex(s.index.union(midx)).ffill().reindex(midx)
        rnum = pd.Series(g.index.values.astype("int64"), index=g.index)   # last report_date as ns
        panels["rdate_num"][tk] = rnum.reindex(rnum.index.union(midx)).ffill().reindex(midx)
    return {f: pd.DataFrame(d) for f, d in panels.items()}


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
    dvol = pd.DataFrame({t: (stock_daily[t]["Close"] * stock_daily[t]["Volume"]).rolling(20).mean()
                         .resample("ME").last().reindex(midx) for t in common
                         if t in stock_daily and "Volume" in stock_daily[t]}).reindex(index=midx, columns=common)
    ep = _earnings_panels(list(common), midx)
    surp = ep.get("eps_surp", pd.DataFrame()).reindex(index=midx, columns=common)
    gsc = ep.get("gscore", pd.DataFrame()).reindex(index=midx, columns=common)
    guid = ep.get("guid", pd.DataFrame()).reindex(index=midx, columns=common)
    rnum = ep.get("rdate_num", pd.DataFrame()).reindex(index=midx, columns=common)
    print(f"months {len(midx)} | stocks {len(common)} | earnings coverage {surp.notna().any().mean()*100:.0f}%", flush=True)

    cond = {"beat": [], "miss": [], "grounded_pos": [], "grounded_neg": [], "guide_up": [], "guide_dn": [],
            "fresh_beat": [], "has_eps": []}
    flag, tilt, filt, spies = [], [], [], []
    pt = pw = 0
    for i in range(9, len(midx) - 1):
        date, ndate = midx[i], midx[i + 1]
        sp = spy_m.iloc[i + 1] / spy_m.iloc[i] - 1
        if not np.isfinite(sp):
            continue
        top = accel.loc[date].dropna().sort_values(ascending=False).head(TOP_N).index
        srow, grow, urow, rrow = surp.loc[date], gsc.loc[date], guid.loc[date], rnum.loc[date]
        date_ns = date.value
        f_slot, t_slot, fil_slot = [], [], []
        for etf in top:
            _, holds = sector_map.get(etf, (etf, []))
            c = [h for h in holds if h in px.columns and _available_at(px[h], date) and pd.notna(pb.loc[date, h])
                 and pb.loc[date, h] > 0 and not bool(trap.loc[date, h])
                 and pd.notna(dvol.loc[date, h]) and dvol.loc[date, h] >= 5e6]
            g = [x for x in c if bool(low.loc[date, x])] or c
            if not g:
                continue
            pick = pb.loc[date, g].idxmin()
            r = _ret_delist(px[pick], date, ndate)
            if r is None or not np.isfinite(r):
                continue
            r = float(r); f_slot.append(r); pt += 1
            sv, gv, uv, rv = srow.get(pick), grow.get(pick), urow.get(pick), rrow.get(pick)
            if pd.notna(sv):
                pw += 1; cond["has_eps"].append(r)
                (cond["beat"] if sv > 0 else cond["miss"]).append(r)
                fresh = pd.notna(rv) and (date_ns - int(rv)) / 8.64e13 <= FRESH_DAYS
                if fresh and sv > 0:
                    cond["fresh_beat"].append(r)
            if pd.notna(gv):
                (cond["grounded_pos"] if gv > 0 else cond["grounded_neg"]).append(r)
            if pd.notna(uv):
                (cond["guide_up"] if uv > 0 else cond["guide_dn"]).append(r)
            # TILT: cheapest-P/B among grounded-positive candidates (fallback flagship)
            gp = [x for x in g if pd.notna(grow.get(x)) and grow.get(x) > 0]
            tpick = pb.loc[date, gp].idxmin() if gp else pick
            tr = _ret_delist(px[tpick], date, ndate)
            if tr is not None and np.isfinite(tr):
                t_slot.append(float(tr))
            # FILTER: only take pick if it's grounded-positive
            if pd.notna(gv) and gv > 0:
                fil_slot.append(r)
        if f_slot:
            flag.append(float(np.mean(f_slot)))
            tilt.append(float(np.mean(t_slot)) if t_slot else float(np.mean(f_slot)))
            filt.append(float(np.mean(fil_slot)) if fil_slot else 0.0)
            spies.append(float(sp))

    base = _w(cond["has_eps"])
    print(f"\n=== (A) conditional forward return of FLAGSHIP pick ===  (coverage {pw}/{pt} picks have recent earnings)", flush=True)
    for k in ("beat", "miss", "grounded_pos", "grounded_neg", "guide_up", "guide_dn", "fresh_beat"):
        s = _w(cond[k])
        if s:
            print(f"  {k:13} mean {s['mean_pct']:>6}%  win {s['win_pct']}%  (n{s['n']})  vs has-EPS base {base['mean_pct']}%  ({s['mean_pct']-base['mean_pct']:+.2f}pp)", flush=True)
        else:
            print(f"  {k}: none", flush=True)

    fb, tb, flb = _stats(flag, spies), _stats(tilt, spies), _stats(filt, spies)
    print(f"\n=== (B) strategy: does grounded-earnings improve the flagship? ===", flush=True)
    print(f"  flagship (baseline)      total {fb['total']:>7}%  vsSPY {fb['vs_spy']:>7}  Sh {fb['sharpe']}  DD {fb['dd']}%  win {fb['win']}%", flush=True)
    print(f"  + grounded-pos TILT      total {tb['total']:>7}%  vsSPY {tb['vs_spy']:>7}  Sh {tb['sharpe']}  DD {tb['dd']}%  win {tb['win']}%  ({tb['vs_spy']-fb['vs_spy']:+.1f}pp)", flush=True)
    print(f"  + grounded-pos FILTER    total {flb['total']:>7}%  vsSPY {flb['vs_spy']:>7}  Sh {flb['sharpe']}  DD {flb['dd']}%  win {flb['win']}%  ({flb['vs_spy']-fb['vs_spy']:+.1f}pp)", flush=True)

    beat, miss = _w(cond["beat"]), _w(cond["miss"])
    fresh = _w(cond["fresh_beat"])
    payload = {
        "computed_at": pd.Timestamp.utcnow().isoformat(),
        "params": {"top_n": TOP_N, "fresh_days": FRESH_DAYS, "benchmark": BENCH, "months": int(len(flag)),
                   "coverage_pct": round(pw / max(pt, 1) * 100, 1)},
        "conditional": {k: _w(v) for k, v in cond.items()},
        "flagship": fb, "grounded_tilt": tb, "grounded_filter": flb,
        "tilt_lift_pp": round(tb["vs_spy"] - fb["vs_spy"], 1), "filter_lift_pp": round(flb["vs_spy"] - fb["vs_spy"], 1),
        "verdict": (f"On the flagship pick: EPS beat {beat['mean_pct']}%/{beat['win_pct']}%win vs miss {miss['mean_pct']}%/"
                    f"{miss['win_pct']}% ({beat['mean_pct']-miss['mean_pct']:+.2f}pp); fresh-beat (PEAD) "
                    f"{fresh['mean_pct'] if fresh else 'n/a'}%. Grounded TILT {tb['vs_spy']-fb['vs_spy']:+.1f}pp, "
                    f"FILTER {flb['vs_spy']-fb['vs_spy']:+.1f}pp vs flagship. " + (
                    "Earnings surprise is a REAL confirming/timing overlay — beat>miss and the tilt lifts the flagship; "
                    "first alt-data axis that adds to selection." if (tb["vs_spy"] > fb["vs_spy"] + 10 or
                    (beat and miss and beat["mean_pct"] - miss["mean_pct"] > 1.5)) else
                    "Earnings surprise does NOT improve the flagship at portfolio level — informational like the other "
                    "overlays; value+quality already prices the fundamentals in. PEAD is a standalone anomaly, not "
                    "additive to deep-value rotation.")),
        "caveat": "PIT on report_date (grounded_score derived from reported numbers+guidance, known at report). "
                  "Coverage < 100%. In-sample ~5y, no fees. FILTER cashes months with no grounded-positive pick.",
    }
    return payload


def main():
    p = build()
    from pathlib import Path
    OUT = Path("/app/.data/studies/earnings_overlay.json")
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(p, indent=2, default=str))
    try:
        from core.models import BacktestResult
        from django.utils import timezone
        BacktestResult.objects.update_or_create(
            kind="earnings_overlay", defaults={"payload": json.loads(json.dumps(p, default=str)), "computed_at": timezone.now()})
        print("Saved BacktestResult[earnings_overlay]", flush=True)
    except Exception as e:
        print("DB save failed:", e, flush=True)
    print("\n" + p["verdict"], flush=True)


if __name__ == "__main__":
    main()
