#!/usr/bin/env python3
"""WIRE-CANDIDATE: high-short-interest EXCLUSION on the flagship (memory flagged NEXT; return-priority
fit since the SI long-leg lags but high-SI = crowded value-trap risk). Replicates the EXACT deployed
flagship (volatility_regime_study.flagship_monthly) as a baseline SELF-CHECK, then drops candidates
whose PIT short-interest %-of-shares exceeds a threshold and remeasures RETURN.

SI data: /app/.data/short_interest.jsonl (Polygon, bi-monthly ~2017-12+; short_interest shares,
days_to_cover). PIT publication lag: FINRA disseminates ~8-10 business days after settlement_date, so
a reading is only usable at avail = settlement_date + 10 business days. SI% = short_interest / PIT
shares_outstanding. Exclude names with SI% > threshold at the rebalance date.

Baseline MUST reproduce ~3631.8% total / +3273.6 vsSPY / 122 mo (else replication is unfaithful -> stop).
-> BacktestResult[flagship_si_exclusion] + JSON.
Run: MSYS_NO_PATHCONV=1 docker exec rotation-celery-worker-1 python -u /app/flagship_si_exclusion.py
"""
import os, json, warnings
warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
os.environ.setdefault("DJANGO_SETTINGS_MODULE", "rotation.settings")
import django
django.setup()
from pathlib import Path
import config, sector_holdings, price_basis
from seq_fundamental_study import load_candles, load_financial_reports
from trend_stock_studies import _pit_monthly_panel, _available_at, _ret_delist, CRYPTO
from backtest_lowpb import _monthly_close, _tstat_from_returns, BENCH

TOP_N = 10; CONV = 2.0; MIN_DVOL = 5e6
SI_PATH = Path("/app/.data/short_interest.jsonl")
OUT = Path(__file__).resolve().parent / ".data" / "studies" / "flagship_si_exclusion.json"


def _perf(r, spy):
    r = np.asarray(r, float)
    tot = float(np.prod(1 + r) - 1) * 100
    sp = float(np.prod(1 + np.asarray(spy)) - 1) * 100
    sh = float(r.mean() / r.std() * np.sqrt(12)) if r.std() > 1e-9 else 0.0
    eqc = np.cumprod(1 + r); dd = float(((eqc / np.maximum.accumulate(eqc)) - 1).min() * 100)
    t = _tstat_from_returns(list(r))
    return dict(total=round(tot, 1), vs_spy=round(tot - sp, 1), sharpe=round(sh, 2), dd=round(dd, 1),
                t_stat=round(t, 2) if t is not None else None, months=int(len(r)))


def load_si_pct(midx, common, sh_panel):
    """Monthly PIT SI%-of-shares panel aligned to midx x common (NaN where unknown)."""
    if not SI_PATH.exists():
        print("  WARN: no short_interest.jsonl", flush=True)
        return None, None
    rows = {}
    dtc = {}
    with SI_PATH.open() as f:
        for line in f:
            try:
                r = json.loads(line)
            except Exception:
                continue
            t = r.get("ticker"); sd = r.get("settlement_date"); si = r.get("short_interest")
            if t is None or sd is None or si is None or t not in set(common):
                continue
            rows.setdefault(t, []).append((sd, float(si), r.get("days_to_cover")))
    si_cols, dtc_cols = {}, {}
    lag = pd.tseries.offsets.BDay(10)
    for t, recs in rows.items():
        recs.sort()
        avail = pd.DatetimeIndex([pd.Timestamp(sd) + lag for sd, _, _ in recs])
        s_si = pd.Series([v for _, v, _ in recs], index=avail).sort_index()
        s_si = s_si[~s_si.index.duplicated(keep="last")].reindex(midx, method="ffill")
        si_cols[t] = s_si
        s_dtc = pd.Series([(d if d is not None else np.nan) for _, _, d in recs], index=avail).sort_index()
        s_dtc = s_dtc[~s_dtc.index.duplicated(keep="last")].reindex(midx, method="ffill")
        dtc_cols[t] = s_dtc
    si_shares = pd.DataFrame(si_cols).reindex(index=midx, columns=common)
    dtc = pd.DataFrame(dtc_cols).reindex(index=midx, columns=common)
    si_pct = (si_shares / sh_panel.where(sh_panel > 0)) * 100.0
    cov = float(si_pct.notna().mean().mean()) * 100
    print(f"  SI panel coverage: {cov:.0f}% of (month,name) cells have a PIT SI reading", flush=True)
    return si_pct, dtc


def run(exclude_si_pct=None, exclude_dtc=None, si_pct=None, dtc=None, ctx=None):
    (midx, accel, spy_m, px, pb, trap, low, dvol, ad_slope3, px_ret3, sector_map) = ctx
    excl_hits = 0; total_cands = 0

    def pick(etf, date, held):
        nonlocal excl_hits, total_cands
        _, holds = sector_map.get(etf, (etf, set()))
        c = []
        for h in holds:
            if h not in px.columns or h in held or not _available_at(px[h], date):
                continue
            if not (pd.notna(pb.loc[date, h]) and pb.loc[date, h] > 0 and not bool(trap.loc[date, h])):
                continue
            if not (pd.notna(dvol.loc[date, h]) and dvol.loc[date, h] >= MIN_DVOL):
                continue
            total_cands += 1
            if exclude_si_pct is not None and si_pct is not None:
                v = si_pct.loc[date, h] if h in si_pct.columns else np.nan
                if pd.notna(v) and v > exclude_si_pct:
                    excl_hits += 1; continue
            if exclude_dtc is not None and dtc is not None:
                v = dtc.loc[date, h] if h in dtc.columns else np.nan
                if pd.notna(v) and v > exclude_dtc:
                    excl_hits += 1; continue
            c.append(h)
        g = [x for x in c if bool(low.loc[date, x])] or c
        return min(g, key=lambda h: pb.loc[date, h]) if g else None

    def wt(name, date):
        a, p = ad_slope3.loc[date].get(name), px_ret3.loc[date].get(name)
        return CONV if (pd.notna(a) and pd.notna(p) and a > 0 and p < 0) else 1.0

    excl_names = []
    dates, rets, spies = [], [], []
    for i in range(9, len(midx) - 1):
        date, ndate = midx[i], midx[i + 1]
        sp = spy_m.iloc[i + 1] / spy_m.iloc[i] - 1
        if not np.isfinite(sp):
            continue
        top = accel.loc[date].dropna().sort_values(ascending=False).head(TOP_N).index
        held = set(); wsum = rr = 0.0
        for etf in top:
            p = pick(etf, date, held)
            if not p:
                continue
            held.add(p)
            r = _ret_delist(px[p], date, ndate)
            if r is None or not np.isfinite(r):
                continue
            w = wt(p, date); wsum += w; rr += w * float(r)
        if wsum <= 0:
            continue
        dates.append(date); rets.append(rr / wsum); spies.append(float(sp))
    perf = _perf(rets, spies)
    perf["excluded_picks"] = excl_hits
    perf["_dates"] = pd.DatetimeIndex(dates); perf["_rets"] = np.array(rets); perf["_spies"] = np.array(spies)
    return perf


def build_ctx():
    etfs = {n: e for n, e in config.SECTOR_ETFS.items() if e not in CRYPTO}
    sector_map, all_holds = {}, set()
    for n, e in etfs.items():
        h = [t for t in sector_holdings.get_holdings(n) if t not in (e, BENCH) and t not in CRYPTO]
        sector_map[e] = (n, set(h)); all_holds.update(h)
    all_holds = sorted(all_holds)
    etf_tk = list(etfs.values())
    etf_daily = load_candles(etf_tk + [BENCH])
    etf_m = _monthly_close({t: d for t, d in etf_daily.items() if t in etf_tk})
    midx = etf_m.index
    accel = etf_m.pct_change(3) - etf_m.pct_change(3).shift(3)
    spy_m = etf_daily[BENCH]["Close"].resample("ME").last().reindex(midx)
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
    dvol_d, adl_m = {}, {}
    for t in common:
        d = stock_daily.get(t)
        if d is None or "Volume" not in d or len(d) < 90:
            continue
        v = d["Volume"]
        dvol_d[t] = (d["Close"] * v).rolling(20).mean().resample("ME").last().reindex(midx)
        if {"High", "Low", "Close"}.issubset(d.columns):
            rng = (d["High"] - d["Low"]).replace(0, np.nan)
            mfm = ((d["Close"] - d["Low"]) - (d["High"] - d["Close"])) / rng
            adl_m[t] = (mfm.fillna(0) * v).cumsum().resample("ME").last().reindex(midx)
    dvol = pd.DataFrame(dvol_d).reindex(index=midx, columns=common)
    adl = pd.DataFrame(adl_m).reindex(index=midx, columns=common)
    ad_slope3 = adl - adl.shift(3); px_ret3 = px.pct_change(3)
    ctx = (midx, accel, spy_m, px, pb, trap, low, dvol, ad_slope3, px_ret3, sector_map)
    return ctx, common, sh


def main():
    print("building flagship context...", flush=True)
    ctx, common, sh = build_ctx()
    midx = ctx[0]

    base = run(ctx=ctx)
    print(f"\nBASELINE (self-check): {base['total']}% total | vsSPY {base['vs_spy']} | Sh {base['sharpe']} "
          f"| DD {base['dd']}% | t {base['t_stat']} | n{base['months']}", flush=True)
    ok = abs(base["total"] - 3631.8) < 60 and base["months"] == 122
    print(f"  replication {'OK (matches deployed flagship)' if ok else 'MISMATCH — treat deltas with care'}", flush=True)

    si_pct, dtc = load_si_pct(midx, common, sh)

    clean = lambda r: {k: v for k, v in r.items() if not k.startswith("_")}
    results = {"baseline": clean(base), "replication_ok": bool(ok)}
    variants = {}
    if si_pct is not None:
        print("\n=== high-SI EXCLUSION — finer threshold grid (robustness) ===", flush=True)
        print(f"{'variant':22} {'total%':>9} {'vsSPY':>9} {'Sh':>5} {'DD%':>7} {'t':>5} {'nExcl':>6}", flush=True)
        print(f"{'baseline':22} {base['total']:>9} {base['vs_spy']:>9} {base['sharpe']:>5} {base['dd']:>7} "
              f"{str(base['t_stat']):>5} {'-':>6}", flush=True)
        for thr in (30.0, 25.0, 22.0, 20.0, 18.0, 15.0, 10.0):
            r = run(exclude_si_pct=thr, si_pct=si_pct, ctx=ctx)
            variants[f"exclude_si>{thr:g}pct"] = r
            results[f"exclude_si>{thr:g}pct"] = clean(r)
            print(f"{'exclude SI>'+format(thr,'g')+'%':22} {r['total']:>9} {r['vs_spy']:>9} {r['sharpe']:>5} "
                  f"{r['dd']:>7} {str(r['t_stat']):>5} {r['excluded_picks']:>6}", flush=True)

        # ---- concentration / stability of the headline >20% winner ----
        win = variants["exclude_si>20pct"]
        bd = pd.Series(base["_rets"], index=base["_dates"])
        wd = pd.Series(win["_rets"], index=win["_dates"])
        common_idx = bd.index.intersection(wd.index)
        delta = (wd.reindex(common_idx) - bd.reindex(common_idx))            # per-month ret difference
        changed = delta[delta.abs() > 1e-9].sort_values()
        # compounded halves
        h = len(common_idx) // 2
        def sub(idx):
            return _perf(list(wd.reindex(idx)), list(pd.Series(base["_spies"], index=base["_dates"]).reindex(idx))), \
                   _perf(list(bd.reindex(idx)), list(pd.Series(base["_spies"], index=base["_dates"]).reindex(idx)))
        w1, b1 = sub(common_idx[:h]); w2, b2 = sub(common_idx[h:])
        print("\n=== is the >20% win robust or a few names? ===", flush=True)
        print(f"  months where the pick changed: {int((delta.abs()>1e-9).sum())} of {len(common_idx)}", flush=True)
        print(f"  3 biggest POSITIVE monthly deltas (variant-baseline):", flush=True)
        for d, v in changed.tail(3)[::-1].items():
            print(f"    {d.date()}  {v*100:+.2f}pp", flush=True)
        print(f"  3 biggest NEGATIVE monthly deltas:", flush=True)
        for d, v in changed.head(3).items():
            print(f"    {d.date()}  {v*100:+.2f}pp", flush=True)
        print(f"  1st half: variant {w1['total']}% vs base {b1['total']}%  (delta {round(w1['total']-b1['total'],1)}pp)", flush=True)
        print(f"  2nd half: variant {w2['total']}% vs base {b2['total']}%  (delta {round(w2['total']-b2['total'],1)}pp)", flush=True)
        results["robustness_win20"] = {
            "months_changed": int((delta.abs() > 1e-9).sum()), "n_months": len(common_idx),
            "top_pos": [[str(d.date()), round(float(v) * 100, 2)] for d, v in changed.tail(3)[::-1].items()],
            "top_neg": [[str(d.date()), round(float(v) * 100, 2)] for d, v in changed.head(3).items()],
            "half1_delta_pp": round(w1["total"] - b1["total"], 1), "half2_delta_pp": round(w2["total"] - b2["total"], 1)}

        for thr in (10.0, 5.0):
            r = run(exclude_dtc=thr, dtc=dtc, ctx=ctx)
            results[f"exclude_dtc>{thr:g}"] = clean(r)
            print(f"{'exclude DtC>'+format(thr,'g'):22} {r['total']:>9} {r['vs_spy']:>9} {r['sharpe']:>5} "
                  f"{r['dd']:>7} {str(r['t_stat']):>5} {r['excluded_picks']:>6}", flush=True)
        base = clean(base)

    best = max([k for k in results if k.startswith("exclude")],
               key=lambda k: results[k]["total"], default=None) if si_pct is not None else None
    verdict = (f"Baseline {base['total']}% (vsSPY {base['vs_spy']}). "
               + (f"Best SI-exclusion = {best} -> {results[best]['total']}% "
                  f"({'+' if results[best]['total'] >= base['total'] else ''}{round(results[best]['total']-base['total'],1)}pp vs baseline). "
                  + ("High-SI exclusion ADDS return -> wire into flagship pick()." if best and results[best]["total"] > base["total"] + 50
                     else "High-SI exclusion does NOT add meaningful return on the flagship (SI long-leg was already ~neutral; "
                          "exclusion barely moves the value-pick). Not worth wiring as a return lever.")
                  if best else "SI data unavailable."))
    print("\n" + verdict, flush=True)

    payload = {"computed_at": pd.Timestamp.utcnow().isoformat(), "results": results, "verdict": verdict,
               "params": {"top_n": TOP_N, "conv": CONV, "min_dvol": MIN_DVOL, "si_lag_bdays": 10},
               "caveat": "SI% = short_interest / PIT shares_outstanding, FINRA ~10bd publication lag. SI data "
                         "bi-monthly ~2017-12+, so exclusion only binds on the post-2017 portion of the 2016-2026 "
                         "backtest. Same PIT/survivorship as deployed flagship. Return-priority framing (absolute)."}
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(payload, indent=2, default=str))
    try:
        from core.models import BacktestResult
        from django.utils import timezone
        BacktestResult.objects.update_or_create(
            kind="flagship_si_exclusion",
            defaults={"payload": json.loads(json.dumps(payload, default=str)), "computed_at": timezone.now()})
        print("Saved BacktestResult[flagship_si_exclusion]", flush=True)
    except Exception as e:
        print("DB save failed:", e, flush=True)


if __name__ == "__main__":
    main()
