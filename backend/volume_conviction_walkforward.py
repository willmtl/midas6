#!/usr/bin/env python3
"""WALK-FORWARD the volume-conviction edge — is A/D-divergence overweighting distributed across years, or a
one-stretch mirage? Same gate acceleration/distressed/freshness passed. Compare equal-weight vs div_2x vs
div_3x: per-year and per-half, and crucially the div-MINUS-equal ADVANTAGE each period (is the volume lift
present every year?). -> BacktestResult[volume_conviction_wf].
Run: MSYS_NO_PATHCONV=1 docker exec rotation-backend-1 python -u /app/volume_conviction_walkforward.py
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


def _tot(r):
    return float(np.prod(1 + np.asarray(r, float)) - 1) * 100 if len(r) else 0.0


def _sh(r):
    r = np.asarray(r, float)
    return float(r.mean() / r.std() * np.sqrt(12)) if len(r) > 1 and r.std() > 1e-9 else 0.0


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
    print("building A/D panels...", flush=True)
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

    eqw, d2, d3, spies, dts = [], [], [], [], []
    ndiv_month = []
    for i in range(9, len(midx) - 1):
        date, ndate = midx[i], midx[i + 1]
        sp = spy_m.iloc[i + 1] / spy_m.iloc[i] - 1
        if not np.isfinite(sp):
            continue
        top = accel.loc[date].dropna().sort_values(ascending=False).head(TOP_N).index
        adsl, pr3 = ad_slope3.loc[date], px_ret3.loc[date]
        picks = []
        for etf in top:
            _, holds = sector_map.get(etf, (etf, []))
            c = [h for h in holds if h in px.columns and _available_at(px[h], date) and pd.notna(pb.loc[date, h])
                 and pb.loc[date, h] > 0 and not bool(trap.loc[date, h])
                 and pd.notna(dvol.loc[date, h]) and dvol.loc[date, h] >= 5e6]
            g = [x for x in c if bool(low.loc[date, x])] or c
            if not g:
                continue
            pick = min(g, key=lambda h: pb.loc[date, h])
            r = _ret_delist(px[pick], date, ndate)
            if r is None or not np.isfinite(r):
                continue
            is_div = bool(pd.notna(adsl.get(pick)) and pd.notna(pr3.get(pick)) and adsl.get(pick) > 0 and pr3.get(pick) < 0)
            picks.append((float(r), is_div))
        if not picks:
            continue
        rs = np.array([r for r, _ in picks])
        w2 = np.array([2.0 if d else 1.0 for _, d in picks]); w3 = np.array([3.0 if d else 1.0 for _, d in picks])
        eqw.append(float(rs.mean())); d2.append(float(np.sum(w2 * rs) / w2.sum())); d3.append(float(np.sum(w3 * rs) / w3.sum()))
        spies.append(float(sp)); dts.append(ndate); ndiv_month.append(int(sum(d for _, d in picks)))

    df = pd.DataFrame({"date": pd.to_datetime(dts), "eq": eqw, "d2": d2, "d3": d3, "spy": spies, "ndiv": ndiv_month})
    df["yr"] = df["date"].dt.year
    full = {k: dict(total=round(_tot(df[k]), 1), sharpe=round(_sh(df[k]), 2)) for k in ("eq", "d2", "d3")}
    print(f"\n=== WALK-FORWARD volume-conviction (div_3x vs div_2x vs equal) ===", flush=True)
    print(f"  FULL  equal {full['eq']['total']}%/Sh{full['eq']['sharpe']}  d2 {full['d2']['total']}%/Sh{full['d2']['sharpe']}  "
          f"d3 {full['d3']['total']}%/Sh{full['d3']['sharpe']}  (avg {df.ndiv.mean():.1f} divergent picks/mo)", flush=True)
    h = len(df) // 2
    print(f"  HALVES d3: 1st {_tot(df['d3'][:h]):+.1f}% vs eq {_tot(df['eq'][:h]):+.1f}%  |  2nd {_tot(df['d3'][h:]):+.1f}% vs eq {_tot(df['eq'][h:]):+.1f}%", flush=True)

    peryear = {}; adv_pos = 0
    print("  per-year (advantage = div_3x total − equal total):", flush=True)
    for yr, g in df.groupby("yr"):
        e_, t3 = _tot(g["eq"]), _tot(g["d3"]); adv = t3 - e_; peryear[int(yr)] = round(adv, 1)
        adv_pos += adv > 0
        print(f"     {yr}: equal {e_:>+7.1f}%  div_3x {t3:>+7.1f}%  advantage {adv:>+6.1f}pp  ({g['ndiv'].mean():.1f} div/mo)", flush=True)

    h1adv, h2adv = _tot(df['d3'][:h]) - _tot(df['eq'][:h]), _tot(df['d3'][h:]) - _tot(df['eq'][h:])
    robust = (h1adv > 0) and (h2adv > 0) and (adv_pos >= len(peryear) - 1)
    payload = {
        "computed_at": pd.Timestamp.utcnow().isoformat(),
        "params": {"top_n": TOP_N, "benchmark": BENCH, "months": int(len(df))},
        "full": full, "half_advantage_pp": {"h1": round(h1adv, 1), "h2": round(h2adv, 1)},
        "per_year_advantage_pp": peryear, "years_div3x_beats_equal": f"{adv_pos}/{len(peryear)}",
        "avg_divergent_per_mo": round(float(df.ndiv.mean()), 1), "robust": bool(robust),
        "verdict": ("Volume-conviction edge is ROBUST — div_3x beats equal-weight in both halves and "
                    f"{adv_pos}/{len(peryear)} years. The A/D-divergence overweight is distributed, not one-stretch. "
                    "VALIDATED — wire div_2-3x conviction weight + live A/D-divergence flag."
                    if robust else
                    f"Volume-conviction edge is NOT robust across subperiods (halves {h1adv:+.0f}/{h2adv:+.0f}pp, "
                    f"{adv_pos}/{len(peryear)} yrs) — the in-sample all-4-axes improvement leans on a stretch; treat as "
                    "a live conviction FLAG, not a validated sizing rule, until more data."),
        "caveat": "Subperiod split, not a true holdout. Divergence rare (~avg/mo shown); div_3x concentrates into it. "
                  "In-sample ~5y, no fees.",
    }
    return payload


def main():
    p = build()
    from pathlib import Path
    Path("/app/.data/studies/volume_conviction_wf.json").write_text(json.dumps(p, indent=2, default=str))
    try:
        from core.models import BacktestResult
        from django.utils import timezone
        BacktestResult.objects.update_or_create(
            kind="volume_conviction_wf", defaults={"payload": json.loads(json.dumps(p, default=str)), "computed_at": timezone.now()})
        print("Saved BacktestResult[volume_conviction_wf]", flush=True)
    except Exception as e:
        print("DB save failed:", e, flush=True)
    print("\n" + p["verdict"], flush=True)


if __name__ == "__main__":
    main()
