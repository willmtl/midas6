#!/usr/bin/env python3
"""SECTORS THAT JUST FELL INTO THE WORST-10 — capitulation bounce? The flagship buys FRESH entrants to the
TOP-10 acceleration (fresh > stale). Symmetric question on the BOTTOM: a sector JUST knocked into the worst-10
accel = peak-pessimism / capitulation — does it BOUNCE (contrarian buy, the sector-level analog of buying a
deep-oversold stock) or keep falling? Test both levels:
  (A) ETF forward return: fresh-worst (just entered bottom-10) vs stale-worst (>=3mo in) vs all-worst vs SPY.
  (B) VALUE-PICK sleeve: cheapest-P/B guarded low-debt inside FRESH-worst sectors, equal-weight monthly, vs
      the same pick in stale-worst / all-worst, vs the FLAGSHIP (fresh-top). Is the deepest-capitulation cheap
      name the best reversion buy?
-> BacktestResult[fresh_worst]. Run: MSYS_NO_PATHCONV=1 docker exec rotation-backend-1 python -u /app/fresh_worst_study.py
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
    return None if not len(a) else dict(n=len(a), mean_pct=round(a.mean() * 100, 2),
                                        win_pct=round((a > 0).mean() * 100, 1))


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
    etf_ret = etf_m.pct_change().shift(-1)      # next-month ETF return, aligned at month t
    dvol = pd.DataFrame({t: (stock_daily[t]["Close"] * stock_daily[t]["Volume"]).rolling(20).mean()
                         .resample("ME").last().reindex(midx) for t in common
                         if t in stock_daily and "Volume" in stock_daily[t]}).reindex(index=midx, columns=common)
    print(f"months {len(midx)} | stocks {len(common)}", flush=True)

    # bottom-10 membership run-length
    bottom_hist = {}
    etf_grp = {"fresh_worst": [], "stale_worst": [], "all_worst": [], "fresh_top": [], "spy": []}
    pick = {"fresh_worst": [], "stale_worst": [], "all_worst": [], "flagship_top": []}
    spies = []
    prev_bottom = set()
    run_len = {e: 0 for e in etf_tk}
    for i in range(9, len(midx) - 1):
        date = midx[i]
        sp = spy_m.iloc[i + 1] / spy_m.iloc[i] - 1
        if not np.isfinite(sp):
            continue
        acc = accel.loc[date].dropna()
        if acc.empty:
            continue
        bottom = set(acc.sort_values(ascending=True).head(TOP_N).index)
        top = set(acc.sort_values(ascending=False).head(TOP_N).index)
        for e in etf_tk:
            run_len[e] = run_len[e] + 1 if e in bottom else 0
        fresh_b = {e for e in bottom if e not in prev_bottom}          # JUST entered worst-10
        stale_b = {e for e in bottom if run_len[e] >= 3}               # chronic
        prev_bottom = bottom

        # (A) ETF forward returns
        spies.append(float(sp)); etf_grp["spy"].append(float(sp))
        for e in bottom:
            rr = etf_ret.loc[date, e]
            if pd.notna(rr):
                etf_grp["all_worst"].append(float(rr))
        for e in fresh_b:
            rr = etf_ret.loc[date, e]
            if pd.notna(rr):
                etf_grp["fresh_worst"].append(float(rr))
        for e in stale_b:
            rr = etf_ret.loc[date, e]
            if pd.notna(rr):
                etf_grp["stale_worst"].append(float(rr))

        # (B) value-pick sleeves (cheapest-P/B guard low-debt, $5M floor)
        def pick_in(sectors):
            slot = []
            ndate = midx[i + 1]
            for e in sectors:
                _, holds = sector_map.get(e, (e, []))
                c = [h for h in holds if h in px.columns and _available_at(px[h], date) and pd.notna(pb.loc[date, h])
                     and pb.loc[date, h] > 0 and not bool(trap.loc[date, h])
                     and pd.notna(dvol.loc[date, h]) and dvol.loc[date, h] >= 5e6]
                g = [x for x in c if bool(low.loc[date, x])] or c
                if g:
                    r = _ret_delist(px[pb.loc[date, g].idxmin()], date, ndate)
                    if r is not None and np.isfinite(r):
                        slot.append(float(r))
            return float(np.mean(slot)) if slot else None

        for key, secs in (("fresh_worst", fresh_b), ("stale_worst", stale_b),
                          ("all_worst", bottom), ("flagship_top", top)):
            v = pick_in(secs)
            pick[key].append(v if v is not None else 0.0)

    print("\n=== (A) ETF forward return: does a freshly-cratered sector BOUNCE? ===", flush=True)
    spy_w = _w(etf_grp["spy"])
    print(f"  {'SPY (month avg)':16} mean {spy_w['mean_pct']:>6}%  win {spy_w['win_pct']}%", flush=True)
    for k in ("all_worst", "fresh_worst", "stale_worst", "fresh_top"):
        if k == "fresh_top":
            continue
        s = _w(etf_grp[k])
        print(f"  {k:16} mean {s['mean_pct']:>6}%  win {s['win_pct']}%  (n{s['n']})" if s else f"  {k}: none", flush=True)

    print("\n=== (B) VALUE-PICK sleeve (cheapest-P/B guard low-debt in those sectors) ===", flush=True)
    res = {}
    for k in ("flagship_top", "all_worst", "fresh_worst", "stale_worst"):
        s = _stats(pick[k][:len(spies)], spies); res[k] = s
        print(f"  {k:14} total {s['total']:>7}%  vsSPY {s['vs_spy']:>7}  Sh {s['sharpe']:>5}  DD {s['dd']:>7}%  win {s['win']}%", flush=True)

    fw_etf = _w(etf_grp["fresh_worst"]); aw_etf = _w(etf_grp["all_worst"])
    fw_pick, flag_pick = res["fresh_worst"], res["flagship_top"]
    payload = {
        "computed_at": pd.Timestamp.utcnow().isoformat(),
        "params": {"top_n": TOP_N, "benchmark": BENCH, "months": int(len(spies))},
        "etf_forward": {k: _w(v) for k, v in etf_grp.items() if k != "spy"}, "spy_month_mean": spy_w["mean_pct"],
        "pick_sleeves": res,
        "verdict": (f"Freshly-worst-10 ETF forward {fw_etf['mean_pct']}%/{fw_etf['win_pct']}%win vs all-worst "
                    f"{aw_etf['mean_pct']}% vs SPY {spy_w['mean_pct']}%. Value-pick in fresh-worst sectors "
                    f"{fw_pick['total']}%/Sh{fw_pick['sharpe']} vs flagship (fresh-top) {flag_pick['total']}%/Sh{flag_pick['sharpe']}. "
                    + ("FRESH-WORST BOUNCES and/or the value pick there rivals the flagship — a real contrarian sleeve."
                       if (fw_etf['mean_pct'] > spy_w['mean_pct'] and fw_pick['sharpe'] > 1.0) else
                       "Fresh-worst does NOT reliably bounce as an ETF; only the value pick inside it carries (if any) — "
                       "capitulation timing at the sector level is weak, the stock-selection does the work.")),
        "caveat": "In-sample ~5y. fresh = entered bottom-10 accel this month (not in it last month). Thin per-month "
                  "counts on fresh/stale buckets. $5M floor on picks; no fees.",
    }
    return payload


def main():
    p = build()
    from pathlib import Path
    OUT = Path(__file__).resolve().parent / ".data" / "studies" / "fresh_worst.json"
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(p, indent=2, default=str))
    try:
        from core.models import BacktestResult
        from django.utils import timezone
        BacktestResult.objects.update_or_create(
            kind="fresh_worst", defaults={"payload": json.loads(json.dumps(p, default=str)), "computed_at": timezone.now()})
        print("Saved BacktestResult[fresh_worst]", flush=True)
    except Exception as e:
        print("DB save failed:", e, flush=True)
    print("\n" + p["verdict"], flush=True)


if __name__ == "__main__":
    main()
