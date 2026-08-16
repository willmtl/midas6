#!/usr/bin/env python3
"""CAN WE DO BETTER — on the HONEST (split-corrected P/B, finding #2) flagship engine. All prior improvement
findings were measured on the inflated pb, so re-confirm the two highest-conviction levers on the corrected
engine and stack them:
  base        = corrected pb, accel top-10, cheapest positive-P/B guard low-debt, $5M floor, div_2x, 1mo hold
  freshness   = base but DROP sectors that are BOTH stale (>=5 consecutive mo in top-10) AND fading (accel now <
                accel 3mo ago); keep >=4 sectors
  hold3       = base but rebalance every 3 months (hold 3mo)
  fresh_hold3 = both
Reports honest total/annual/Sharpe/DD/vs-SPY. -> BacktestResult[improve_corrected].
Run: MSYS_NO_PATHCONV=1 docker exec rotation-backend-1 python -u /app/improve_corrected_study.py
"""
import os, json, warnings
warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
os.environ.setdefault("DJANGO_SETTINGS_MODULE", "rotation.settings")
import django
django.setup()

import config, sector_holdings, price_basis
from seq_fundamental_study import load_candles, load_financial_reports
from trend_stock_studies import _pit_monthly_panel, _available_at, _ret_delist, CRYPTO
from backtest_lowpb import _monthly_close, BENCH

TOP_N = 10
CONVICTION_MULT = 2.0
MIN_DVOL = 5e6
STALE_MO = 5
MIN_SECTORS = 4


def _perf(port_rets, spy_rets, periods_per_year):
    r = np.asarray(port_rets, float); n = len(r)
    tot = float(np.prod(1 + r) - 1) * 100; sp = float(np.prod(1 + np.asarray(spy_rets)) - 1) * 100
    ann = (float(np.prod(1 + r)) ** (periods_per_year / n) - 1) * 100 if n else 0.0
    sh = float(r.mean() / r.std() * np.sqrt(periods_per_year)) if r.std() > 1e-9 else 0.0
    eqc = np.cumprod(1 + r); dd = float(((eqc / np.maximum.accumulate(eqc)) - 1).min() * 100)
    return dict(total=round(tot, 1), vs_spy=round(tot - sp, 1), annual=round(ann, 1),
                sharpe=round(sh, 2), dd=round(dd, 1), periods=n, spy_total=round(sp, 1))


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
    px_at = price_basis.as_traded_close(px, price_basis.refresh_splits(list(common)))
    pb = (px_at * sh) / eq.where(eq != 0)                       # corrected P/B
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

    # precompute top-10 accel membership + run-length (consecutive months in top-10) per etf
    topmap = {}
    for i in range(len(midx)):
        row = accel.iloc[i].dropna()
        topmap[i] = set(row.sort_values(ascending=False).head(TOP_N).index) if len(row) else set()
    runlen = {}  # (i, etf) -> consecutive months in top-10 ending at i
    for e in etf_tk:
        c = 0
        for i in range(len(midx)):
            c = c + 1 if e in topmap[i] else 0
            runlen[(i, e)] = c
    print(f"months {len(midx)} | stocks {len(common)}", flush=True)

    def sectors_at(i, freshness):
        top = list(accel.iloc[i].dropna().sort_values(ascending=False).head(TOP_N).index)
        if not freshness:
            return top
        acc_now = accel.iloc[i]; acc_prev = accel.iloc[i - 3] if i >= 3 else accel.iloc[i]
        kept = []
        for e in top:
            stale = runlen[(i, e)] >= STALE_MO
            fading = pd.notna(acc_now.get(e)) and pd.notna(acc_prev.get(e)) and acc_now.get(e) < acc_prev.get(e)
            if stale and fading:
                continue
            kept.append(e)
        return kept if len(kept) >= MIN_SECTORS else top

    def pick_for(etf, date):
        _, holds = sector_map.get(etf, (etf, []))
        c = [h for h in holds if h in px.columns and _available_at(px[h], date) and pd.notna(pb.loc[date, h])
             and pb.loc[date, h] > 0 and not bool(trap.loc[date, h])
             and pd.notna(dvol.loc[date, h]) and dvol.loc[date, h] >= MIN_DVOL]
        g = [x for x in c if bool(low.loc[date, x])] or c
        if not g:
            return None
        pick = min(g, key=lambda h: pb.loc[date, h])
        adsl, pr3 = ad_slope3.loc[date], px_ret3.loc[date]
        is_div = bool(pd.notna(adsl.get(pick)) and pd.notna(pr3.get(pick)) and adsl.get(pick) > 0 and pr3.get(pick) < 0)
        return pick, (CONVICTION_MULT if is_div else 1.0)

    def run(freshness, hold):
        port_rets, spy_rets = [], []
        i = 9
        while i < len(midx) - hold:
            date = midx[i]; ndate = midx[i + hold]
            spx = spy_m.iloc[i + hold] / spy_m.iloc[i] - 1
            if not np.isfinite(spx):
                i += hold; continue
            picks = []
            for etf in sectors_at(i, freshness):
                pr = pick_for(etf, date)
                if pr is None:
                    continue
                pick, w = pr
                r = _ret_delist(px[pick], date, ndate)     # hold-period return (adjusted close = correct)
                if r is None or not np.isfinite(r):
                    continue
                picks.append((w, float(r)))
            if picks:
                tw = sum(w for w, _ in picks)
                port_rets.append(sum(w * r for w, r in picks) / tw); spy_rets.append(float(spx))
            i += hold
        return _perf(port_rets, spy_rets, 12.0 / hold)

    out = {
        "base": run(False, 1),
        "freshness": run(True, 1),
        "hold3": run(False, 3),
        "fresh_hold3": run(True, 3),
    }
    print("\n=== CAN WE DO BETTER (honest split-corrected engine, div_2x) ===", flush=True)
    print(f"  {'variant':14} {'total':>8} {'vsSPY':>8} {'ann':>7} {'Sharpe':>7} {'DD':>8} {'periods':>8}", flush=True)
    for k, s in out.items():
        print(f"  {k:14} {s['total']:>7}% {s['vs_spy']:>8} {s['annual']:>6}% {s['sharpe']:>7} {s['dd']:>7}% {s['periods']:>8}", flush=True)

    base = out["base"]
    best = max(out, key=lambda k: out[k]["sharpe"])
    verdict = (f"Honest engine base +{base['total']}%/Sh{base['sharpe']}/DD{base['dd']}%. "
               f"freshness +{out['freshness']['total']}%/Sh{out['freshness']['sharpe']}; hold3 +{out['hold3']['total']}%/Sh{out['hold3']['sharpe']}; "
               f"fresh_hold3 +{out['fresh_hold3']['total']}%/Sh{out['fresh_hold3']['sharpe']}. Best Sharpe = {best}. " + (
               f"'{best}' improves on base (Sh {out[best]['sharpe']} vs {base['sharpe']})." if out[best]['sharpe'] > base['sharpe'] + 0.03 else
               "No variant clearly beats base on the corrected engine — the honest engine is near its ceiling."))
    return {"computed_at": pd.Timestamp.utcnow().isoformat(),
            "params": {"top_n": TOP_N, "conviction_mult": CONVICTION_MULT, "min_dvol": MIN_DVOL,
                       "stale_mo": STALE_MO, "pb_basis": "as-traded (split-corrected)"},
            "results": out, "best": best, "verdict": verdict,
            "caveat": "Split-corrected P/B; div_2x; $5M floor; PIT; present-day-holdings survivorship; no fees. In-sample."}


def main():
    p = build()
    from pathlib import Path
    Path("/app/.data/studies/improve_corrected.json").write_text(json.dumps(p, indent=2, default=str))
    try:
        from core.models import BacktestResult
        from django.utils import timezone
        BacktestResult.objects.update_or_create(
            kind="improve_corrected", defaults={"payload": json.loads(json.dumps(p, default=str)), "computed_at": timezone.now()})
        print("Saved BacktestResult[improve_corrected]", flush=True)
    except Exception as e:
        print("DB save failed:", e, flush=True)
    print("\n" + p["verdict"], flush=True)


if __name__ == "__main__":
    main()
