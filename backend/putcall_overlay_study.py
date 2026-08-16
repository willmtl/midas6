#!/usr/bin/env python3
"""PUT/CALL as a real PORTFOLIO OVERLAY (not just conditional buckets) — the div_2x treatment. Keep the flagship
basket, re-WEIGHT by the put/call signal, measure portfolio total/vsSPY/Sharpe/DD over the options window
(2022-09..2026-08). Schemes (each vs equal-weight over the SAME window so it's apples-to-apples):
  equal                 baseline
  pc_contrarian_2x      overweight 2x the picks with HIGH pc_vol (fear/hedging = contrarian bounce?)
  pc_momentum_2x        overweight 2x the picks with LOW pc_vol (call-heavy = bullish flow)
  skew_2x               overweight 2x high iv_skew (put demand)
  div_2x                our VALIDATED A/D overlay (reference — does put/call beat / add to it?)
  pc_contra_x_div_2x    stack contrarian put/call ON div_2x
High/low = above/below the median pc_vol AMONG THAT MONTH'S picks (relative, self-normalizing).
-> BacktestResult[putcall_overlay]. Run: MSYS_NO_PATHCONV=1 docker exec rotation-backend-1 python -u /app/putcall_overlay_study.py
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


def _panel(field, midx):
    from core.models import OptionSnapshot
    rows = list(OptionSnapshot.objects.exclude(**{f"{field}__isnull": True}).values_list("ticker", "date", field))
    if not rows:
        return pd.DataFrame(index=midx)
    df = pd.DataFrame(rows, columns=["ticker", "date", field]); df["date"] = pd.to_datetime(df["date"])
    return df.pivot_table(index="date", columns="ticker", values=field, aggfunc="last").resample("ME").last().reindex(midx)


def _stats(r, spy):
    r = np.asarray(r, float); n = len(r)
    if n == 0:
        return dict(total=0, vs_spy=0, sharpe=0, dd=0, win=0, n=0)
    tot = float(np.prod(1 + r) - 1) * 100; sp = float(np.prod(1 + np.asarray(spy)) - 1) * 100
    sh = float(r.mean() / r.std() * np.sqrt(12)) if r.std() > 1e-9 else 0.0
    eqc = np.cumprod(1 + r); dd = float(((eqc / np.maximum.accumulate(eqc)) - 1).min() * 100)
    return dict(total=round(tot, 1), vs_spy=round(tot - sp, 1), sharpe=round(sh, 2), dd=round(dd, 1),
                win=round((r > 0).mean() * 100, 1), n=n)


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
    # A/D divergence for the reference div_2x
    adl_m = {}
    for t in common:
        d = stock_daily.get(t)
        if d is None or "Volume" not in d or len(d) < 90 or not {"High", "Low", "Close"}.issubset(d.columns):
            continue
        v = d["Volume"]; rng = (d["High"] - d["Low"]).replace(0, np.nan)
        mfm = ((d["Close"] - d["Low"]) - (d["High"] - d["Close"])) / rng
        adl_m[t] = (mfm.fillna(0) * v).cumsum().resample("ME").last().reindex(midx)
    adl = pd.DataFrame(adl_m).reindex(index=midx, columns=common)
    ad_slope3 = adl - adl.shift(3); px_ret3 = px.pct_change(3)
    print("loading put/call panels...", flush=True)
    pc_vol = _panel("pc_vol", midx).reindex(columns=common)
    iv_skew = _panel("iv_skew", midx).reindex(columns=common)
    win = set(pc_vol.dropna(how="all").index)
    print(f"options months: {len(win)} | first pick eval from month 9", flush=True)

    schemes = ["equal", "pc_contrarian_2x", "pc_momentum_2x", "skew_2x", "div_2x", "pc_contra_x_div_2x"]
    port = {k: [] for k in schemes}; spies = []
    cov = []
    for i in range(9, len(midx) - 1):
        date, ndate = midx[i], midx[i + 1]
        if date not in win:
            continue
        sp = spy_m.iloc[i + 1] / spy_m.iloc[i] - 1
        if not np.isfinite(sp):
            continue
        top = accel.loc[date].dropna().sort_values(ascending=False).head(TOP_N).index
        adsl, pr3, pcrow, skrow = ad_slope3.loc[date], px_ret3.loc[date], pc_vol.loc[date], iv_skew.loc[date]
        picks = []      # (ret, pcv, skew, is_div)
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
            picks.append((float(r), pcrow.get(pick), skrow.get(pick), is_div))
        if not picks:
            continue
        rs = np.array([p[0] for p in picks])
        pcv = np.array([p[1] if pd.notna(p[1]) else np.nan for p in picks])
        skv = np.array([p[2] if pd.notna(p[2]) else np.nan for p in picks])
        divf = np.array([p[3] for p in picks])
        cov.append(float(np.mean(np.isfinite(pcv))))
        pc_med = np.nanmedian(pcv); sk_med = np.nanmedian(skv)
        def W(scheme):
            w = np.ones(len(picks))
            if scheme == "pc_contrarian_2x":
                w = np.where(np.isfinite(pcv) & (pcv >= pc_med), 2.0, 1.0)
            elif scheme == "pc_momentum_2x":
                w = np.where(np.isfinite(pcv) & (pcv < pc_med), 2.0, 1.0)
            elif scheme == "skew_2x":
                w = np.where(np.isfinite(skv) & (skv >= sk_med), 2.0, 1.0)
            elif scheme == "div_2x":
                w = np.where(divf, 2.0, 1.0)
            elif scheme == "pc_contra_x_div_2x":
                w = np.where(divf, 2.0, 1.0) * np.where(np.isfinite(pcv) & (pcv >= pc_med), 2.0, 1.0)
            return w
        for k in schemes:
            w = W(k)
            port[k].append(float(np.sum(w * rs) / w.sum()))
        spies.append(float(sp))

    print(f"\n=== PUT/CALL PORTFOLIO OVERLAY ({len(spies)} option-months, avg pc coverage {round(np.mean(cov)*100,0)}% of picks) ===", flush=True)
    print(f"  {'scheme':22} {'total':>8} {'vsSPY':>8} {'Sh':>5} {'DD':>8} {'win':>6}", flush=True)
    res = {}
    for k in schemes:
        st = _stats(port[k][:len(spies)], spies); res[k] = st
        print(f"  {k:22} {st['total']:>7}% {st['vs_spy']:>8} {st['sharpe']:>5} {st['dd']:>7}% {st['win']:>5}%", flush=True)
    base = res["equal"]
    pcbest = max(("pc_contrarian_2x", "pc_momentum_2x", "skew_2x"), key=lambda k: res[k]["sharpe"])
    helps = res[pcbest]["sharpe"] > base["sharpe"] + 0.05 and res[pcbest]["total"] > base["total"]
    stacks = res["pc_contra_x_div_2x"]["sharpe"] > res["div_2x"]["sharpe"] + 0.05 and res["pc_contra_x_div_2x"]["total"] > res["div_2x"]["total"]
    payload = {
        "computed_at": pd.Timestamp.utcnow().isoformat(),
        "params": {"top_n": TOP_N, "benchmark": BENCH, "option_months": int(len(spies)),
                   "avg_pc_coverage_pct": round(float(np.mean(cov)) * 100, 1),
                   "window": "2022-09..2026-08 (options era only)"},
        "results": res, "best_putcall_scheme": pcbest,
        "verdict": (
            f"Over the options window: equal {base['total']}%/Sh{base['sharpe']}, div_2x {res['div_2x']['total']}%/Sh{res['div_2x']['sharpe']}. "
            f"Best put/call scheme = {pcbest} ({res[pcbest]['total']}%/Sh{res[pcbest]['sharpe']}). " + (
            "PUT/CALL OVERLAY ADDS as a standalone weight. " if helps else "Put/call overlay does NOT beat equal-weight. ") + (
            "And it STACKS on div_2x (contra×div beats div alone)." if stacks else
            "It does NOT add on top of div_2x either — the A/D-divergence overlay already captures the sizing edge; "
            "put/call is redundant/noise. Confirms options sentiment is not a live overlay for the flagship.")),
        "caveat": "Options era only (~47mo, 2022-09+); shorter than the 62mo backtest so totals not comparable to +490 headline. "
                  "pc coverage on the small-cap value PICKS is partial. In-sample, single provider, no fees.",
    }
    return payload


def main():
    p = build()
    from pathlib import Path
    Path("/app/.data/studies/putcall_overlay.json").write_text(json.dumps(p, indent=2, default=str))
    try:
        from core.models import BacktestResult
        from django.utils import timezone
        BacktestResult.objects.update_or_create(
            kind="putcall_overlay", defaults={"payload": json.loads(json.dumps(p, default=str)), "computed_at": timezone.now()})
        print("Saved BacktestResult[putcall_overlay]", flush=True)
    except Exception as e:
        print("DB save failed:", e, flush=True)
    print("\n" + p["verdict"], flush=True)


if __name__ == "__main__":
    main()
