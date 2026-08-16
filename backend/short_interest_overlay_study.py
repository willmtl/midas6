#!/usr/bin/env python3
"""SHORT-INTEREST OVERLAY + WALK-FORWARD — the div_2x treatment for the +1.43pp short-interest pulse. Keep the
flagship basket, re-WEIGHT toward heavily-shorted picks, measure portfolio total/vsSPY/Sharpe/DD over the FULL
62mo window, and WALK-FORWARD (halves + per-year) to see if the pulse survives OOS or is small-sample luck.
Schemes vs equal:
  equal
  short_2x_med      2x the picks above the median short% AMONG that month's picks
  short_2x_hi       2x the picks with short% >= 5% of shares (absolute 'heavily shorted')
  short_3x_hi       3x the >=5% names
  div_2x            our validated A/D overlay (reference)
  short_x_div_2x    stack short-2x-hi ON div_2x (do they add?)
-> BacktestResult[short_interest_overlay]. Run: MSYS_NO_PATHCONV=1 docker exec rotation-backend-1 python -u /app/short_interest_overlay_study.py
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
from short_interest_study import _fetch_short_interest, PUB_LAG_D

TOP_N = 10
HI_SHORT = 5.0        # "heavily shorted" = short interest >= 5% of shares outstanding


def _stats(r, spy):
    r = np.asarray(r, float); n = len(r)
    if n == 0:
        return dict(total=0, vs_spy=0, sharpe=0, dd=0, win=0)
    tot = float(np.prod(1 + r) - 1) * 100; sp = float(np.prod(1 + np.asarray(spy)) - 1) * 100
    sh = float(r.mean() / r.std() * np.sqrt(12)) if r.std() > 1e-9 else 0.0
    eqc = np.cumprod(1 + r); dd = float(((eqc / np.maximum.accumulate(eqc)) - 1).min() * 100)
    return dict(total=round(tot, 1), vs_spy=round(tot - sp, 1), sharpe=round(sh, 2), dd=round(dd, 1),
                win=round((r > 0).mean() * 100, 1))


def _tot(r):
    return float(np.prod(1 + np.asarray(r, float)) - 1) * 100 if len(r) else 0.0


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

    si_raw = _fetch_short_interest(list(common))
    lag = pd.Timedelta(days=int(PUB_LAG_D * 1.5))
    si_sh = {}
    for tk in common:
        rows = si_raw.get(tk) or []
        if not rows:
            continue
        s = pd.DataFrame(rows, columns=["sd", "si", "dtc"]); s["sd"] = pd.to_datetime(s["sd"]); s = s.sort_values("sd")
        ser = []
        for d in midx:
            av = s[s["sd"] <= (d - lag)]
            ser.append(av["si"].iloc[-1] if len(av) else np.nan)
        si_sh[tk] = pd.Series(ser, index=midx)
    si_sh = pd.DataFrame(si_sh).reindex(index=midx, columns=common)
    short_pct = (si_sh / sh.where(sh > 0)) * 100
    print(f"months {len(midx)} | stocks w/ short data: {int((~short_pct.isna()).any().sum())}", flush=True)

    schemes = ["equal", "short_2x_med", "short_2x_hi", "short_3x_hi", "div_2x", "short_x_div_2x"]
    port = {k: [] for k in schemes}; spies = []; dts = []
    for i in range(9, len(midx) - 1):
        date, ndate = midx[i], midx[i + 1]
        sp = spy_m.iloc[i + 1] / spy_m.iloc[i] - 1
        if not np.isfinite(sp):
            continue
        top = accel.loc[date].dropna().sort_values(ascending=False).head(TOP_N).index
        adsl, pr3, sprow = ad_slope3.loc[date], px_ret3.loc[date], short_pct.loc[date]
        picks = []      # (ret, short_pct, is_div)
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
            picks.append((float(r), sprow.get(pick), is_div))
        if not picks:
            continue
        rs = np.array([p[0] for p in picks])
        spv = np.array([p[1] if pd.notna(p[1]) else np.nan for p in picks])
        divf = np.array([p[2] for p in picks])
        med = np.nanmedian(spv)
        def W(k):
            if k == "short_2x_med":
                return np.where(np.isfinite(spv) & (spv >= med), 2.0, 1.0)
            if k == "short_2x_hi":
                return np.where(np.isfinite(spv) & (spv >= HI_SHORT), 2.0, 1.0)
            if k == "short_3x_hi":
                return np.where(np.isfinite(spv) & (spv >= HI_SHORT), 3.0, 1.0)
            if k == "div_2x":
                return np.where(divf, 2.0, 1.0)
            if k == "short_x_div_2x":
                return np.where(divf, 2.0, 1.0) * np.where(np.isfinite(spv) & (spv >= HI_SHORT), 2.0, 1.0)
            return np.ones(len(picks))
        for k in schemes:
            w = W(k); port[k].append(float(np.sum(w * rs) / w.sum()))
        spies.append(float(sp)); dts.append(ndate)

    print(f"\n=== SHORT-INTEREST OVERLAY ({len(spies)} months) ===", flush=True)
    print(f"  {'scheme':18} {'total':>8} {'vsSPY':>8} {'Sh':>5} {'DD':>8} {'win':>6}", flush=True)
    res = {}
    for k in schemes:
        st = _stats(port[k][:len(spies)], spies); res[k] = st
        print(f"  {k:18} {st['total']:>7}% {st['vs_spy']:>8} {st['sharpe']:>5} {st['dd']:>7}% {st['win']:>5}%", flush=True)

    # WALK-FORWARD best short scheme vs equal
    best = max(("short_2x_med", "short_2x_hi", "short_3x_hi"), key=lambda k: res[k]["sharpe"])
    df = pd.DataFrame({"date": pd.to_datetime(dts), "eq": port["equal"], "S": port[best], "spy": spies})
    df["yr"] = df["date"].dt.year
    h = len(df) // 2
    halves = {"eq_1st": round(_tot(df["eq"][:h]), 1), "eq_2nd": round(_tot(df["eq"][h:]), 1),
              "S_1st": round(_tot(df["S"][:h]), 1), "S_2nd": round(_tot(df["S"][h:]), 1)}
    print(f"\n  WALK-FORWARD {best} vs equal (per-year, adv = short − equal):", flush=True)
    peryear = {}; adv_pos = 0
    for yr, g in df.groupby("yr"):
        e_, s_ = _tot(g["eq"]), _tot(g["S"]); adv = s_ - e_; peryear[int(yr)] = round(adv, 1); adv_pos += adv > 0
        print(f"     {yr}: equal {e_:>+7.1f}%  {best} {s_:>+7.1f}%  adv {adv:>+6.1f}pp", flush=True)
    h1adv = halves["S_1st"] - halves["eq_1st"]; h2adv = halves["S_2nd"] - halves["eq_2nd"]
    print(f"  halves adv: 1st {h1adv:+.1f}pp  2nd {h2adv:+.1f}pp", flush=True)
    robust = (h1adv > 0 and h2adv > 0 and adv_pos >= len(peryear) - 1 and res[best]["sharpe"] > res["equal"]["sharpe"])

    stacks = res["short_x_div_2x"]["sharpe"] > res["div_2x"]["sharpe"] + 0.03 and res["short_x_div_2x"]["total"] > res["div_2x"]["total"]
    payload = {
        "computed_at": pd.Timestamp.utcnow().isoformat(),
        "params": {"top_n": TOP_N, "benchmark": BENCH, "hi_short_pct": HI_SHORT, "months": int(len(spies))},
        "results": res, "best_scheme": best, "walk_forward": {"halves": halves, "per_year_adv_pp": peryear,
        "years_short_beats_equal": f"{adv_pos}/{len(peryear)}"}, "robust": bool(robust), "stacks_on_div2x": bool(stacks),
        "verdict": (
            f"Best short overlay = {best}: {res[best]['total']}%/Sh{res[best]['sharpe']}/DD{res[best]['dd']}% vs equal "
            f"{res['equal']['total']}%/Sh{res['equal']['sharpe']}. " + (
            f"SHORT-INTEREST OVERLAY VALIDATES — beats equal both halves ({h1adv:+.0f}/{h2adv:+.0f}pp) and {adv_pos}/{len(peryear)} years. "
            + ("It even STACKS on div_2x. " if stacks else "It does NOT add on top of div_2x (redundant with it). ")
            + "Overweighting heavily-shorted value picks is a real 2nd overlay."
            if robust else
            f"Short-interest overlay does NOT robustly beat equal-weight in walk-forward (halves {h1adv:+.0f}/{h2adv:+.0f}pp, "
            f"{adv_pos}/{len(peryear)} yrs) — the +1.43pp conditional lift was largely in-sample/small-sample. Keep as a weak "
            "FLAG, not a validated sizing rule. div_2x remains the only validated overlay.")),
        "caveat": "Polygon short-interest bi-monthly PIT-lagged; short%-shares approximates short%-float; foreign tickers NaN. "
                  "In-sample subperiod split, no fees.",
    }
    return payload


def main():
    p = build()
    from pathlib import Path
    Path("/app/.data/studies/short_interest_overlay.json").write_text(json.dumps(p, indent=2, default=str))
    try:
        from core.models import BacktestResult
        from django.utils import timezone
        BacktestResult.objects.update_or_create(
            kind="short_interest_overlay", defaults={"payload": json.loads(json.dumps(p, default=str)), "computed_at": timezone.now()})
        print("Saved BacktestResult[short_interest_overlay]", flush=True)
    except Exception as e:
        print("DB save failed:", e, flush=True)
    print("\n" + p["verdict"], flush=True)


if __name__ == "__main__":
    main()
