#!/usr/bin/env python3
"""SHORT-INTEREST VETO on top of the FULL stack (flagship + div_2x). Overweighting shorted names failed; the
standalone test showed heavily-shorted-cheap = VALUE TRAP (worst bucket). So test the OPPOSITE application:
use short interest as a NEGATIVE screen ON TOP of what we already have — when the cheapest-P/B pick is heavily
shorted, SKIP it and take the next-cheapest clean name (still guard+low-debt+$5M floor), then div_2x weight.
Sweep the veto threshold. Does avoiding the shorted value-traps improve the validated flagship?
-> BacktestResult[short_interest_veto]. Run: MSYS_NO_PATHCONV=1 docker exec rotation-backend-1 python -u /app/short_interest_veto_study.py
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
VETOS = [None, 20.0, 15.0, 10.0, 7.0, 5.0]     # None = baseline (no veto); else skip picks with short% >= veto


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
    lag = pd.Timedelta(days=int(PUB_LAG_D * 1.5)); si_sh = {}
    for tk in common:
        rows = si_raw.get(tk) or []
        if not rows:
            continue
        s = pd.DataFrame(rows, columns=["sd", "si", "dtc"]); s["sd"] = pd.to_datetime(s["sd"]); s = s.sort_values("sd")
        ser = [s[s["sd"] <= (d - lag)]["si"].iloc[-1] if len(s[s["sd"] <= (d - lag)]) else np.nan for d in midx]
        si_sh[tk] = pd.Series(ser, index=midx)
    si_sh = pd.DataFrame(si_sh).reindex(index=midx, columns=common)
    short_pct = (si_sh / sh.where(sh > 0)) * 100
    print(f"months {len(midx)} | stocks w/ short data: {int((~short_pct.isna()).any().sum())}", flush=True)

    labels = {v: ("baseline" if v is None else f"veto>={int(v)}%") for v in VETOS}
    port = {v: [] for v in VETOS}; spies = []; dts = []; n_vetoed = {v: 0 for v in VETOS}; n_picks = 0
    for i in range(9, len(midx) - 1):
        date, ndate = midx[i], midx[i + 1]
        sp = spy_m.iloc[i + 1] / spy_m.iloc[i] - 1
        if not np.isfinite(sp):
            continue
        top = accel.loc[date].dropna().sort_values(ascending=False).head(TOP_N).index
        adsl, pr3, sprow = ad_slope3.loc[date], px_ret3.loc[date], short_pct.loc[date]
        slot = {v: [] for v in VETOS}
        for etf in top:
            _, holds = sector_map.get(etf, (etf, []))
            c = [h for h in holds if h in px.columns and _available_at(px[h], date) and pd.notna(pb.loc[date, h])
                 and pb.loc[date, h] > 0 and not bool(trap.loc[date, h])
                 and pd.notna(dvol.loc[date, h]) and dvol.loc[date, h] >= 5e6]
            g = [x for x in c if bool(low.loc[date, x])] or c
            if not g:
                continue
            g_sorted = sorted(g, key=lambda h: pb.loc[date, h])   # cheapest-P/B first
            base = g_sorted[0]
            n_picks += 1
            for v in VETOS:
                if v is None:
                    pick = base
                else:
                    # skip picks heavily shorted (short%>=v); names w/o short data pass (don't penalize foreign)
                    clean = [h for h in g_sorted if not (pd.notna(sprow.get(h)) and sprow.get(h) >= v)]
                    pick = clean[0] if clean else base
                    if pick != base:
                        n_vetoed[v] += 1
                r = _ret_delist(px[pick], date, ndate)
                if r is None or not np.isfinite(r):
                    continue
                is_div = bool(pd.notna(adsl.get(pick)) and pd.notna(pr3.get(pick)) and adsl.get(pick) > 0 and pr3.get(pick) < 0)
                slot[v].append((float(r), is_div))
        if not slot[None]:
            continue
        for v in VETOS:
            picks = slot[v]
            if not picks:
                port[v].append(0.0); continue
            rs = np.array([p[0] for p in picks]); w = np.array([2.0 if p[1] else 1.0 for p in picks])   # div_2x
            port[v].append(float(np.sum(w * rs) / w.sum()))
        spies.append(float(sp)); dts.append(ndate)

    print(f"\n=== SHORT-INTEREST VETO on flagship+div_2x ({len(spies)} months) ===", flush=True)
    print(f"  {'variant':14} {'total':>8} {'vsSPY':>8} {'Sh':>5} {'DD':>8} {'win':>6} {'#vetoed':>8}", flush=True)
    res = {}
    for v in VETOS:
        st = _stats(port[v][:len(spies)], spies); res[labels[v]] = st
        print(f"  {labels[v]:14} {st['total']:>7}% {st['vs_spy']:>8} {st['sharpe']:>5} {st['dd']:>7}% {st['win']:>5}% {n_vetoed[v]:>8}", flush=True)

    base = res["baseline"]
    veto_keys = [labels[v] for v in VETOS if v is not None]
    best = max(veto_keys, key=lambda k: res[k]["sharpe"])
    helps = res[best]["sharpe"] > base["sharpe"] + 0.03 and res[best]["vs_spy"] >= base["vs_spy"] - 5
    # walk-forward the best veto vs baseline
    df = pd.DataFrame({"date": pd.to_datetime(dts), "base": port[None], "best": port[[v for v in VETOS if labels[v] == best][0]], "spy": spies})
    df["yr"] = df["date"].dt.year; h = len(df) // 2
    wf = {"h1_base": round(_tot(df["base"][:h]), 1), "h1_best": round(_tot(df["best"][:h]), 1),
          "h2_base": round(_tot(df["base"][h:]), 1), "h2_best": round(_tot(df["best"][h:]), 1), "per_year": {}}
    yr_pos = 0
    print(f"  walk-forward {best} vs baseline per-year (adv):", flush=True)
    for yr, g in df.groupby("yr"):
        b_, s_ = _tot(g["base"]), _tot(g["best"]); adv = s_ - b_; wf["per_year"][int(yr)] = round(adv, 1); yr_pos += adv > 0
        print(f"     {yr}: base {b_:>+7.1f}%  {best} {s_:>+7.1f}%  adv {adv:>+6.1f}pp", flush=True)
    h1adv = wf["h1_best"] - wf["h1_base"]; h2adv = wf["h2_best"] - wf["h2_base"]
    robust = helps and h1adv > -1 and h2adv > -1 and yr_pos >= len(wf["per_year"]) - 1

    payload = {
        "computed_at": pd.Timestamp.utcnow().isoformat(),
        "params": {"top_n": TOP_N, "vetos": [v for v in VETOS], "benchmark": BENCH, "months": int(len(spies))},
        "results": res, "best_veto": best, "walk_forward": {**wf, "h1_adv_pp": round(h1adv, 1), "h2_adv_pp": round(h2adv, 1),
        "years_beat": f"{yr_pos}/{len(wf['per_year'])}"}, "robust": bool(robust),
        "verdict": (f"Best veto = {best}: {res[best]['total']}%/Sh{res[best]['sharpe']}/DD{res[best]['dd']}% vs baseline "
                    f"flagship+div_2x {base['total']}%/Sh{base['sharpe']}/DD{base['dd']}%. " + (
                    f"VETO HELPS — screening out heavily-shorted value-traps on top of div_2x improves the stack and holds in "
                    f"walk-forward (halves {h1adv:+.0f}/{h2adv:+.0f}pp, {yr_pos}/{len(wf['per_year'])} yrs). Wire the veto."
                    if robust else
                    f"Veto does NOT robustly improve the stack (best Sh {res[best]['sharpe']} vs {base['sharpe']}, halves "
                    f"{h1adv:+.0f}/{h2adv:+.0f}pp, {yr_pos}/{len(wf['per_year'])} yrs). The guard+low-debt gates already screen most "
                    "traps; a short-interest veto rarely fires and doesn't add on top. div_2x stays the only overlay.")),
        "caveat": "Polygon short% PIT-lagged; veto only replaces a pick when a cleaner cheaper name exists (never loses the sector); "
                  "names w/o short data pass. In-sample, no fees.",
    }
    return payload


def main():
    p = build()
    from pathlib import Path
    Path("/app/.data/studies/short_interest_veto.json").write_text(json.dumps(p, indent=2, default=str))
    try:
        from core.models import BacktestResult
        from django.utils import timezone
        BacktestResult.objects.update_or_create(
            kind="short_interest_veto", defaults={"payload": json.loads(json.dumps(p, default=str)), "computed_at": timezone.now()})
        print("Saved BacktestResult[short_interest_veto]", flush=True)
    except Exception as e:
        print("DB save failed:", e, flush=True)
    print("\n" + p["verdict"], flush=True)


if __name__ == "__main__":
    main()
