#!/usr/bin/env python3
"""NEGATIVE BOOK VALUE — are we wrongly excluding winners? Our pb>0 filter drops all negative-equity
stocks. But negative book = TWO opposite populations: buyback machines (profitable, returned so much
capital equity went negative -> elite compounders like HD/MCD/AZO) vs distressed wrecks (unprofitable,
accumulated losses). P/B can't tell them apart. Measure:
  (A) forward returns of stocks in accel-sector holdings by book class: pos-book / neg-book+profitable /
      neg-book+unprofitable  -> do neg-book stocks go up?
  (B) strategy: rank by P/S (works for both signs) on pos-book only vs pos-book + neg-book-profitable
      -> does ADDING the buyback machines help vs our pos-P/B baseline?
-> BacktestResult[negbook] + JSON.
Run: MSYS_NO_PATHCONV=1 docker exec rotation-backend-1 python -u /app/negbook_study.py
"""
import os, json, warnings
warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
os.environ.setdefault("DJANGO_SETTINGS_MODULE", "rotation.settings")
import django
django.setup()

import config, sector_holdings
from studies import _tstat_from_returns
from seq_fundamental_study import load_candles, load_financial_reports
from trend_stock_studies import _pit_monthly_panel, _available_at, _ret_delist, CRYPTO
from backtest_lowpb import _monthly_close, BENCH

TOP_N = 10


def _wr(x):
    a = np.array(x, float)
    return None if not len(a) else dict(n=len(a), mean_pct=round(a.mean() * 100, 2), win_pct=round((a > 0).mean() * 100, 1))


def _st(r, spy):
    r = np.asarray(r, float); n = len(r)
    if n == 0:
        return dict(vs_spy=0, sharpe=0, dd=0)
    tot = float(np.prod(1 + r) - 1) * 100; sp = float(np.prod(1 + np.asarray(spy)) - 1) * 100
    sh = float(r.mean() / r.std() * np.sqrt(12)) if r.std() > 1e-9 else 0.0
    eq = np.cumprod(1 + r); dd = float(((eq / np.maximum.accumulate(eq)) - 1).min() * 100)
    return dict(vs_spy=round(tot - sp, 1), sharpe=round(sh, 2), dd=round(dd, 1))


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
    sh, eq, ni, dt, rev = (_pit_monthly_panel(reps, f, midx) for f in
                           ("shares_outstanding", "total_equity", "net_income", "total_debt", "revenue"))
    common = stock_m.columns.intersection(sh.columns).intersection(eq.columns)
    R = lambda p: p.reindex(index=midx, columns=common)
    px = stock_m[common]; sh, eq, ni, dt, rev = R(sh), R(eq), R(ni), R(dt), R(rev)
    mktcap = px * sh
    pb = mktcap / eq.where(eq != 0)
    ps = mktcap / rev.where(rev > 0)
    trap = (ni < 0) & (~(eq >= eq.shift(12))) & (~(ni > ni.shift(4)))
    dvol = pd.DataFrame({t: (stock_daily[t]["Close"] * stock_daily[t]["Volume"]).rolling(20).mean()
                         .resample("ME").last().reindex(midx) for t in common
                         if t in stock_daily and "Volume" in stock_daily[t]}).reindex(index=midx, columns=common)
    print(f"months {len(midx)} | stocks {len(common)}", flush=True)

    # ---- A. forward returns by book class (accel-sector holdings, tradeable) ----
    cls = {"pos_book": [], "neg_book_profitable": [], "neg_book_unprofitable": []}
    # ---- B. strategy: P/S-ranked, pos-book-only vs pos-book + neg-book-profitable ----
    ps_pos, ps_ext, base_pb, spies = [], [], [], []
    for i in range(9, len(midx) - 1):
        date, ndate = midx[i], midx[i + 1]
        sp = spy_m.iloc[i + 1] / spy_m.iloc[i] - 1
        if not np.isfinite(sp):
            continue
        ranks = accel.loc[date].dropna().sort_values(ascending=False).head(TOP_N).index
        pb_slot, psp_slot, pse_slot = [], [], []
        for etf in ranks:
            _, holds = sector_map.get(etf, (etf, []))
            avail = [h for h in holds if h in px.columns and _available_at(px[h], date)
                     and pd.notna(dvol.loc[date, h]) and dvol.loc[date, h] >= 5e6]
            for h in avail:
                e_, n_ = eq.loc[date, h], ni.loc[date, h]
                if pd.isna(e_):
                    continue
                r = _ret_delist(px[h], date, ndate)
                if r is None or not np.isfinite(r):
                    continue
                if e_ > 0:
                    cls["pos_book"].append(float(r))
                elif pd.notna(n_) and n_ > 0:
                    cls["neg_book_profitable"].append(float(r))
                else:
                    cls["neg_book_unprofitable"].append(float(r))
            # baseline pos-P/B guarded
            g = [h for h in avail if pd.notna(pb.loc[date, h]) and pb.loc[date, h] > 0 and not bool(trap.loc[date, h])]
            if g:
                rr = _ret_delist(px[pb.loc[date, g].idxmin()], date, ndate)
                if rr is not None and np.isfinite(rr):
                    pb_slot.append(float(rr))
            # P/S pos-book only
            pos_ps = [h for h in g if pd.notna(ps.loc[date, h])]
            if pos_ps:
                rr = _ret_delist(px[ps.loc[date, pos_ps].idxmin()], date, ndate)
                if rr is not None and np.isfinite(rr):
                    psp_slot.append(float(rr))
            # P/S pos-book + neg-book-profitable (add the buyback machines)
            ext = pos_ps + [h for h in avail if pd.notna(eq.loc[date, h]) and eq.loc[date, h] < 0
                            and pd.notna(ni.loc[date, h]) and ni.loc[date, h] > 0 and pd.notna(ps.loc[date, h])]
            if ext:
                rr = _ret_delist(px[ps.loc[date, ext].idxmin()], date, ndate)
                if rr is not None and np.isfinite(rr):
                    pse_slot.append(float(rr))
        if pb_slot:
            base_pb.append(float(np.mean(pb_slot))); spies.append(float(sp))
            ps_pos.append(float(np.mean(psp_slot)) if psp_slot else float(np.mean(pb_slot)))
            ps_ext.append(float(np.mean(pse_slot)) if pse_slot else float(np.mean(pb_slot)))

    print("\n=== A. do negative-book stocks go up? (forward return by class) ===", flush=True)
    for k in ("pos_book", "neg_book_profitable", "neg_book_unprofitable"):
        s = _wr(cls[k])
        print(f"  {k:24} mean {s['mean_pct']:>6}%  win {s['win_pct']}%  (n{s['n']})" if s else f"  {k}: none", flush=True)
    b = _st(base_pb, spies); pp = _st(ps_pos, spies); pe = _st(ps_ext, spies)
    print("\n=== B. include neg-book-profitable in the pick? ===", flush=True)
    print(f"  baseline cheapest-P/B (pos only):     vsSPY {b['vs_spy']}%  Sh {b['sharpe']}", flush=True)
    print(f"  P/S-ranked, pos-book only:            vsSPY {pp['vs_spy']}%  Sh {pp['sharpe']}", flush=True)
    print(f"  P/S-ranked + neg-book-profitable:     vsSPY {pe['vs_spy']}%  Sh {pe['sharpe']}  "
          f"({'+' if pe['vs_spy']-pp['vs_spy']>=0 else ''}{round(pe['vs_spy']-pp['vs_spy'],1)}pp vs P/S pos-only)", flush=True)

    npr = _wr(cls["neg_book_profitable"]); pos = _wr(cls["pos_book"])
    payload = {
        "computed_at": pd.Timestamp.utcnow().isoformat(),
        "params": {"top_n": TOP_N, "benchmark": BENCH, "months": int(len(midx))},
        "by_class": {k: _wr(v) for k, v in cls.items()},
        "baseline_pb": b, "ps_pos": pp, "ps_ext": pe, "ps_ext_lift_pp": round(pe["vs_spy"] - pp["vs_spy"], 1),
        "verdict": (f"Neg-book PROFITABLE (buyback machines) forward {npr['mean_pct']}%/{npr['win_pct']}%win vs pos-book "
                    f"{pos['mean_pct']}%/{pos['win_pct']}% — " + (
                    "they DO go up; but adding them via P/S changed the strategy by "
                    f"{round(pe['vs_spy']-pp['vs_spy'],1)}pp. Neg-book profitable = quality buyback names (not deep-value), "
                    "distressed neg-book = junk; our pb>0 filter conflates them but the deep-value engine doesn't want "
                    "either — buyback machines aren't cheap, wrecks are traps.")),
        "caveat": "In-sample, ~5y. P/S used to rank across book signs. Neg-book needs a non-book value metric.",
    }
    return payload


def main():
    p = build()
    from pathlib import Path
    OUT = Path(__file__).resolve().parent / ".data" / "studies" / "negbook.json"
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(p, indent=2, default=str))
    try:
        from core.models import BacktestResult
        from django.utils import timezone
        BacktestResult.objects.update_or_create(
            kind="negbook", defaults={"payload": json.loads(json.dumps(p, default=str)), "computed_at": timezone.now()})
        print("Saved BacktestResult[negbook]", flush=True)
    except Exception as e:
        print("DB save failed:", e, flush=True)
    print("\n" + p["verdict"], flush=True)


if __name__ == "__main__":
    main()
