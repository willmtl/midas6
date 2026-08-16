#!/usr/bin/env python3
"""DARK-POOL as an ORTHOGONAL OVERLAY on the flagship — does institutional accumulation confirm the value
pick? Standalone off_pct carries no edge (arms underperform their own universe baseline), but a weak signal
can still help if it's UNCORRELATED with value. Same overlay test we ran for insider/congress: on the
flagship pick (cheapest-P/B guard low-debt in accel sectors), measure
  (A) conditional forward return: picks WITH dark-pool accumulation (off_pct rising, PIT) vs WITHOUT; and
      high vs low off_pct level.
  (B) strategy variants vs flagship: TILT (prefer accumulating candidate in each sector) and FILTER (only
      take picks that are accumulating) — does either beat the flagship?
PIT on FINRA published_date (~2-4wk disclosure lag baked in). -> BacktestResult[darkpool_overlay].
Run: MSYS_NO_PATHCONV=1 docker exec rotation-backend-1 python -u /app/darkpool_overlay_study.py
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
LOOKBACK = 2          # months for the off_pct accumulation trend


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


def _pit_offpct_panel(tickers, monthly_index):
    from core.models import DarkPoolWeek
    qs = (DarkPoolWeek.objects.filter(ticker__in=list(tickers), off_pct__isnull=False, published_date__isnull=False)
          .values_list("ticker", "published_date", "off_pct"))
    big = pd.DataFrame.from_records(list(qs), columns=["ticker", "published_date", "off_pct"])
    out = {}
    if big.empty:
        return pd.DataFrame(out)
    big["published_date"] = pd.to_datetime(big["published_date"])
    for tk, g in big.groupby("ticker", sort=False):
        s = g.set_index("published_date")["off_pct"].sort_index()
        s = s[~s.index.duplicated(keep="last")]
        out[tk] = s.reindex(s.index.union(monthly_index)).ffill().reindex(monthly_index)
    return pd.DataFrame(out)


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
    off = _pit_offpct_panel(list(common), midx).reindex(columns=common)
    off_rise = off - off.shift(LOOKBACK)
    cover = float(off.reindex(columns=common).notna().any(axis=0).mean()) * 100
    print(f"months {len(midx)} | stocks {len(common)} | with dark-pool data {cover:.0f}%", flush=True)

    cond = {"accum": [], "no_accum": [], "high_off": [], "low_off": [], "has_dp": [], "no_dp": []}
    flag, tilt, filt, spies = [], [], [], []
    picks_total = picks_with_dp = 0
    for i in range(9, len(midx) - 1):
        date, ndate = midx[i], midx[i + 1]
        sp = spy_m.iloc[i + 1] / spy_m.iloc[i] - 1
        if not np.isfinite(sp):
            continue
        top = accel.loc[date].dropna().sort_values(ascending=False).head(TOP_N).index
        f_slot, t_slot, fil_slot = [], [], []
        off_row = off.loc[date]; rise_row = off_rise.loc[date]
        med_off = off_row.dropna().median()
        for etf in top:
            _, holds = sector_map.get(etf, (etf, []))
            c = [h for h in holds if h in px.columns and _available_at(px[h], date) and pd.notna(pb.loc[date, h])
                 and pb.loc[date, h] > 0 and not bool(trap.loc[date, h])
                 and pd.notna(dvol.loc[date, h]) and dvol.loc[date, h] >= 5e6]
            g = [x for x in c if bool(low.loc[date, x])] or c
            if not g:
                continue
            pick = pb.loc[date, g].idxmin()                # flagship pick
            r = _ret_delist(px[pick], date, ndate)
            if r is None or not np.isfinite(r):
                continue
            r = float(r); f_slot.append(r)
            picks_total += 1
            # conditional buckets on the flagship pick
            ov, rv = off_row.get(pick), rise_row.get(pick)
            if pd.notna(ov):
                picks_with_dp += 1
                cond["has_dp"].append(r)
                (cond["accum"] if (pd.notna(rv) and rv > 0) else cond["no_accum"]).append(r)
                (cond["high_off"] if ov >= med_off else cond["low_off"]).append(r)
            else:
                cond["no_dp"].append(r)
            # TILT: among candidates, cheapest-P/B that is ACCUMULATING (fallback flagship)
            acc_g = [x for x in g if pd.notna(rise_row.get(x)) and rise_row.get(x) > 0]
            tpick = pb.loc[date, acc_g].idxmin() if acc_g else pick
            tr = _ret_delist(px[tpick], date, ndate)
            if tr is not None and np.isfinite(tr):
                t_slot.append(float(tr))
            # FILTER: only take the pick if IT is accumulating
            if pd.notna(rv) and rv > 0:
                fil_slot.append(r)
        if f_slot:
            flag.append(float(np.mean(f_slot)))
            tilt.append(float(np.mean(t_slot)) if t_slot else float(np.mean(f_slot)))
            filt.append(float(np.mean(fil_slot)) if fil_slot else 0.0)   # cash the month if none accumulating
            spies.append(float(sp))

    print(f"\n=== (A) conditional forward return of the FLAGSHIP pick ===  (coverage {picks_with_dp}/{picks_total} picks have DP)", flush=True)
    base = _w(cond["has_dp"])
    for k in ("accum", "no_accum", "high_off", "low_off"):
        s = _w(cond[k])
        lift = (s["mean_pct"] - base["mean_pct"]) if s else 0
        print(f"  {k:10} mean {s['mean_pct']:>6}%  win {s['win_pct']}%  (n{s['n']})  vs has-DP base {base['mean_pct']}%  ({lift:+.2f}pp)" if s else f"  {k}: none", flush=True)

    fb, tb, flb = _stats(flag, spies), _stats(tilt, spies), _stats(filt, spies)
    print(f"\n=== (B) strategy: does dark-pool accumulation improve the flagship? ===", flush=True)
    print(f"  flagship (baseline)      total {fb['total']:>7}%  vsSPY {fb['vs_spy']:>7}  Sh {fb['sharpe']}  DD {fb['dd']}%  win {fb['win']}%", flush=True)
    print(f"  + accumulation TILT      total {tb['total']:>7}%  vsSPY {tb['vs_spy']:>7}  Sh {tb['sharpe']}  DD {tb['dd']}%  win {tb['win']}%  ({tb['vs_spy']-fb['vs_spy']:+.1f}pp)", flush=True)
    print(f"  + accumulation FILTER    total {flb['total']:>7}%  vsSPY {flb['vs_spy']:>7}  Sh {flb['sharpe']}  DD {flb['dd']}%  win {flb['win']}%  ({flb['vs_spy']-fb['vs_spy']:+.1f}pp)", flush=True)

    acc, nacc = _w(cond["accum"]), _w(cond["no_accum"])
    payload = {
        "computed_at": pd.Timestamp.utcnow().isoformat(),
        "params": {"top_n": TOP_N, "lookback": LOOKBACK, "benchmark": BENCH, "months": int(len(flag)),
                   "coverage_pct": round(picks_with_dp / max(picks_total, 1) * 100, 1)},
        "conditional": {k: _w(v) for k, v in cond.items()},
        "flagship": fb, "accum_tilt": tb, "accum_filter": flb,
        "tilt_lift_pp": round(tb["vs_spy"] - fb["vs_spy"], 1), "filter_lift_pp": round(flb["vs_spy"] - fb["vs_spy"], 1),
        "verdict": (f"On the flagship pick, dark-pool accumulation {acc['mean_pct']}%/{acc['win_pct']}%win vs no-accum "
                    f"{nacc['mean_pct']}%/{nacc['win_pct']}% ({acc['mean_pct']-nacc['mean_pct']:+.2f}pp). Accum-TILT "
                    f"{tb['vs_spy']-fb['vs_spy']:+.1f}pp, FILTER {flb['vs_spy']-fb['vs_spy']:+.1f}pp vs flagship. " + (
                    "Dark-pool accumulation is a REAL confirming overlay — modest positive lift, orthogonal to value."
                    if (tb["vs_spy"] > fb["vs_spy"] + 10 or (acc and nacc and acc["mean_pct"] - nacc["mean_pct"] > 1)) else
                    "Dark-pool accumulation does NOT improve the flagship — like news/insider/congress it's "
                    "informational at best; the value+quality pick already prices it in. Not a selection driver.")),
        "caveat": "PIT on FINRA published_date. Coverage < 100% (not all picks have ATS data). In-sample ~5y, no fees. "
                  "FILTER cashes months with no accumulating pick (breadth cost).",
    }
    return payload


def main():
    p = build()
    from pathlib import Path
    OUT = Path(__file__).resolve().parent / ".data" / "studies" / "darkpool_overlay.json"
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(p, indent=2, default=str))
    try:
        from core.models import BacktestResult
        from django.utils import timezone
        BacktestResult.objects.update_or_create(
            kind="darkpool_overlay", defaults={"payload": json.loads(json.dumps(p, default=str)), "computed_at": timezone.now()})
        print("Saved BacktestResult[darkpool_overlay]", flush=True)
    except Exception as e:
        print("DB save failed:", e, flush=True)
    print("\n" + p["verdict"], flush=True)


if __name__ == "__main__":
    main()
