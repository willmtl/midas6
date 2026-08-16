#!/usr/bin/env python3
"""P/E as a VETO FILTER (not a ranking metric). We rank/pick by cheapest-P/B; P/E-RANKING already lost badly
(+62% vs P/B +229%, forward_pe_test). Different question: keep ranking by P/B but VETO a pick if it's frothy
on earnings — walk cheapest-P/B first and skip any PROFITABLE name whose trailing P/E > CAP (loss-makers pass
through, the profit-guard already vets them); if a sector has no surviving name, skip that sector (honest
breadth cost of the veto). Sweep the cap. Does trimming the few earnings-expensive names (e.g. VCYT ~P/E32)
help, or does it just cost breadth? PIT trailing-12mo earnings (same _ntm_panel as forward_pe_test).
-> BacktestResult[pe_filter]. Run: MSYS_NO_PATHCONV=1 docker exec rotation-backend-1 python -u /app/pe_filter_study.py
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

TOP_N = 10
CAPS = [None, 50, 40, 30, 25, 20]     # None = baseline flagship (no P/E veto)


def _stats(r, spy):
    r = np.asarray(r, float); n = len(r)
    if n == 0:
        return dict(total=0, vs_spy=0, sharpe=0, dd=0, win=0)
    tot = float(np.prod(1 + r) - 1) * 100; sp = float(np.prod(1 + np.asarray(spy)) - 1) * 100
    sh = float(r.mean() / r.std() * np.sqrt(12)) if r.std() > 1e-9 else 0.0
    eq = np.cumprod(1 + r); dd = float(((eq / np.maximum.accumulate(eq)) - 1).min() * 100)
    return dict(total=round(tot, 1), vs_spy=round(tot - sp, 1), sharpe=round(sh, 2), dd=round(dd, 1),
                win=round((r > 0).mean() * 100, 1))


def _ttm_ni(reps, midx, common, days=365):
    """PIT trailing-12mo net income per ticker (only reports available at the date)."""
    out = {}
    md = list(midx)
    los = [np.datetime64(d - pd.Timedelta(days=days)) for d in md]
    d0 = [np.datetime64(d) for d in md]
    for tk in common:
        df = reps.get(tk)
        if df is None or len(df) == 0 or "period_end" not in df:
            continue
        d = df.dropna(subset=["period_end"]).copy()
        pe = pd.to_datetime(d["period_end"]).values
        ni = pd.to_numeric(d["net_income"], errors="coerce").values.astype(float)
        av = pd.to_datetime(d["avail_date"]).values if "avail_date" in d else pe
        col = np.full(len(md), np.nan)
        for j in range(len(md)):
            m = (pe > los[j]) & (pe <= d0[j]) & (av <= d0[j])
            if m.any() and np.isfinite(ni[m]).any():
                col[j] = np.nansum(ni[m])
        out[tk] = pd.Series(col, index=midx)
    return pd.DataFrame(out).reindex(index=midx, columns=common)


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
    mktcap = px * sh
    pb = mktcap / eq.where(eq != 0)
    trap = (ni < 0) & (~(eq >= eq.shift(12))) & (~(ni > ni.shift(4)))
    low = (dt / eq.where(eq != 0)) < 1.0
    dvol = pd.DataFrame({t: (stock_daily[t]["Close"] * stock_daily[t]["Volume"]).rolling(20).mean()
                         .resample("ME").last().reindex(midx) for t in common
                         if t in stock_daily and "Volume" in stock_daily[t]}).reindex(index=midx, columns=common)
    print("building PIT trailing-12mo earnings...", flush=True)
    ttm = _ttm_ni(reps, midx, list(common), ).reindex(index=midx, columns=common)
    ttm_pe = mktcap / ttm.where(ttm > 0)
    print(f"months {len(midx)} | stocks {len(common)}", flush=True)

    def qual(h, date):
        return (h in px.columns and _available_at(px[h], date) and pd.notna(pb.loc[date, h]) and pb.loc[date, h] > 0
                and not bool(trap.loc[date, h]) and pd.notna(dvol.loc[date, h]) and dvol.loc[date, h] >= 5e6)

    def pe_ok(h, date, cap):
        if cap is None:
            return True
        pev = ttm_pe.loc[date, h]
        return pd.isna(pev) or pev <= cap     # NaN P/E (loss-maker/no data) passes -> guard's call

    ports = {("cap" if c else "base") + (str(c) if c else "line"): [] for c in CAPS}
    keys = list(ports.keys())
    spies = []; nnames = {k: [] for k in keys}; vetoed = 0; base_picks = 0
    for i in range(9, len(midx) - 1):
        date, ndate = midx[i], midx[i + 1]
        sp = spy_m.iloc[i + 1] / spy_m.iloc[i] - 1
        if not np.isfinite(sp):
            continue
        top = accel.loc[date].dropna().sort_values(ascending=False).head(TOP_N).index
        slots = {k: [] for k in keys}
        for etf in top:
            _, holds = sector_map.get(etf, (etf, []))
            c = [h for h in holds if qual(h, date)]
            g = [x for x in c if bool(low.loc[date, x])] or c
            if not g:
                continue
            g_sorted = sorted(g, key=lambda h: pb.loc[date, h])   # cheapest-P/B first
            base = g_sorted[0]
            base_picks += 1
            for c_, k in zip(CAPS, keys):
                pick = next((h for h in g_sorted if pe_ok(h, date, c_)), None)
                if pick is None:
                    continue                                       # sector vetoed entirely -> skip slot
                if c_ is not None and pick != base:
                    vetoed += 1
                r = _ret_delist(px[pick], date, ndate)
                if r is not None and np.isfinite(r):
                    slots[k].append(float(r))
        if slots[keys[0]]:
            for k in keys:
                ports[k].append(float(np.mean(slots[k])) if slots[k] else 0.0)
                nnames[k].append(len(slots[k]))
            spies.append(float(sp))

    print("\n=== P/E VETO FILTER on the cheapest-P/B pick ===", flush=True)
    res = {}
    for c_, k in zip(CAPS, keys):
        s = _stats(ports[k][:len(spies)], spies); s["avg_names"] = round(float(np.mean(nnames[k])), 1)
        res[k] = s
        label = "no cap (flagship)" if c_ is None else f"P/E <= {c_}"
        print(f"  {label:18} total {s['total']:>7}%  vsSPY {s['vs_spy']:>7}  Sh {s['sharpe']:>5}  DD {s['dd']:>7}%  "
              f"win {s['win']}%  ~{s['avg_names']} names/mo", flush=True)
    print(f"\n  base pick vetoed by cap in {vetoed} sector-months (of {base_picks} picks across all caps' checks)", flush=True)

    base_k = keys[0]
    best = max((k for k in keys if k != base_k), key=lambda k: res[k]["sharpe"])
    helps = res[best]["sharpe"] > res[base_k]["sharpe"] + 0.03 and res[best]["vs_spy"] > res[base_k]["vs_spy"] - 10
    payload = {
        "computed_at": pd.Timestamp.utcnow().isoformat(),
        "params": {"top_n": TOP_N, "caps": [c for c in CAPS], "benchmark": BENCH, "months": int(len(spies))},
        "results": res,
        "verdict": (f"Best P/E cap = {best} (Sh {res[best]['sharpe']} vs flagship {res[base_k]['sharpe']}, vsSPY "
                    f"{res[best]['vs_spy']} vs {res[base_k]['vs_spy']}). " + (
                    "A P/E veto HELPS — trimming the frothy names improves risk-adjusted return without much breadth cost."
                    if helps else
                    "A P/E veto does NOT help — it costs return/breadth for no Sharpe gain. The earnings-expensive "
                    "P/B-cheap names (e.g. growth-sector picks) earn their keep; the profit-guard is the right earnings "
                    "screen, not a P/E cap. Leave P/E off entirely (ranking AND filter).")),
        "caveat": "PIT trailing-12mo P/E; loss-makers/no-data pass the veto (guard handles them). Vetoed sectors skip "
                  "the slot (breadth cost). In-sample ~5y, no fees.",
    }
    return payload


def main():
    p = build()
    from pathlib import Path
    Path("/app/.data/studies/pe_filter.json").write_text(json.dumps(p, indent=2, default=str))
    try:
        from core.models import BacktestResult
        from django.utils import timezone
        BacktestResult.objects.update_or_create(
            kind="pe_filter", defaults={"payload": json.loads(json.dumps(p, default=str)), "computed_at": timezone.now()})
        print("Saved BacktestResult[pe_filter]", flush=True)
    except Exception as e:
        print("DB save failed:", e, flush=True)
    print("\n" + p["verdict"], flush=True)


if __name__ == "__main__":
    main()
