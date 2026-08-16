#!/usr/bin/env python3
"""ROTATION HISTORY (time machine) — reconstruct the flagship basket AS IT STOOD at every month-end over the
last ~5 years, point-in-time (no lookahead), with the realized NEXT-month return of each pick. Same validated
engine as the live scanner + backtest: top-10 momentum-ACCELERATION sectors -> cheapest positive-P/B guarded
low-debt pick, $5M dvol floor, div_2x A/D-conviction weighting. Lets the app page month-by-month (◀ ▶) and
see what you'd have held on the last trading day of each month and how it did. -> BacktestResult[rotation_history].
Run: MSYS_NO_PATHCONV=1 docker exec rotation-backend-1 python -u /app/rotation_history_scan.py
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
CONVICTION_MULT = 2.0     # div_2x: overweight A/D-divergence (accumulation-into-weakness) names


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
    # P/B on AS-TRADED price (undo future-split back-adjustment look-ahead, finding #2); returns stay on adj close
    px_at = price_basis.as_traded_close(px, price_basis.refresh_splits(list(common)))
    pb = (px_at * sh) / eq.where(eq != 0)
    trap = (ni < 0) & (~(eq >= eq.shift(12))) & (~(ni > ni.shift(4)))
    low = (dt / eq.where(eq != 0)) < 1.0
    print("building A/D + dollar-volume panels...", flush=True)
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

    months = []
    cum_port = cum_spy = 1.0
    for i in range(9, len(midx) - 1):
        date, ndate = midx[i], midx[i + 1]
        sp = spy_m.iloc[i + 1] / spy_m.iloc[i] - 1
        if not np.isfinite(sp):
            continue
        top = accel.loc[date].dropna().sort_values(ascending=False).head(TOP_N).index
        adsl, pr3 = ad_slope3.loc[date], px_ret3.loc[date]
        picks = []
        for rank, etf in enumerate(top, 1):
            name, holds = sector_map.get(etf, (etf, []))
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
            is_div = bool(pd.notna(adsl.get(pick)) and pd.notna(pr3.get(pick))
                          and adsl.get(pick) > 0 and pr3.get(pick) < 0)
            picks.append({"rank": rank, "sector": name, "etf": etf, "pick": pick,
                          "pb": round(float(pb.loc[date, pick]), 2), "accumulating": is_div,
                          "ret": round(float(r) * 100, 2)})
        if not picks:
            continue
        # div_2x weighting -> pct_alloc + weighted portfolio return; equal-weight for comparison
        for p in picks:
            p["weight"] = CONVICTION_MULT if p["accumulating"] else 1.0
        tw = sum(p["weight"] for p in picks) or 1.0
        for p in picks:
            p["pct_alloc"] = round(p["weight"] / tw * 100, 1)
        port = sum(p["weight"] * p["ret"] / 100 for p in picks) / tw
        eqr = float(np.mean([p["ret"] / 100 for p in picks]))
        cum_port *= (1 + port); cum_spy *= (1 + sp)
        months.append({
            "date": date.strftime("%Y-%m-%d"), "label": date.strftime("%b %Y"),
            "n_picks": len(picks), "n_accumulating": sum(1 for p in picks if p["accumulating"]),
            "port_ret": round(port * 100, 2), "eq_ret": round(eqr * 100, 2), "spy_ret": round(float(sp) * 100, 2),
            "cum_port": round((cum_port - 1) * 100, 1), "cum_spy": round((cum_spy - 1) * 100, 1),
            "picks": picks,
        })

    beat = sum(1 for m in months if m["port_ret"] > m["spy_ret"])
    payload = {
        "computed_at": pd.Timestamp.utcnow().isoformat(),
        "params": {"top_n": TOP_N, "benchmark": BENCH, "conviction_mult": CONVICTION_MULT,
                   "engine": ("PIT reconstruction of the flagship: momentum-ACCELERATION top-10 sectors -> cheapest "
                              "positive-P/B guarded low-debt pick, $5M dvol floor, div_2x A/D-conviction weight. "
                              "Each month = basket held at that month-end, realized return over the FOLLOWING month.")},
        "n_months": len(months),
        "summary": {"first": months[0]["label"] if months else None, "last": months[-1]["label"] if months else None,
                    "cum_port": months[-1]["cum_port"] if months else 0, "cum_spy": months[-1]["cum_spy"] if months else 0,
                    "months_beat_spy": beat, "pct_beat": round(beat / len(months) * 100, 1) if months else 0},
        "months": months,
        "note": "Point-in-time; no fees; stock-universe survivorship applies. Month-end rebalance, hold one month.",
    }
    return payload


def main():
    p = build()
    from pathlib import Path
    Path("/app/.data/studies/rotation_history.json").write_text(json.dumps(p, indent=2, default=str))
    try:
        from core.models import BacktestResult
        from django.utils import timezone
        BacktestResult.objects.update_or_create(
            kind="rotation_history", defaults={"payload": json.loads(json.dumps(p, default=str)), "computed_at": timezone.now()})
        print("Saved BacktestResult[rotation_history]", flush=True)
    except Exception as e:
        print("DB save failed:", e, flush=True)
    s = p["summary"]
    print(f"\n=== ROTATION HISTORY: {p['n_months']} months {s['first']}..{s['last']} | "
          f"cum {s['cum_port']}% vs SPY {s['cum_spy']}% | beat {s['months_beat_spy']}/{p['n_months']} ({s['pct_beat']}%) ===", flush=True)
    for m in p["months"][-6:]:
        acc = f" {m['n_accumulating']}🔵" if m["n_accumulating"] else ""
        print(f"  {m['label']:9} port {m['port_ret']:>+6.2f}%  spy {m['spy_ret']:>+6.2f}%  cum {m['cum_port']:>+7.1f}%{acc}", flush=True)


if __name__ == "__main__":
    main()
