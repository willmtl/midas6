#!/usr/bin/env python3
"""PUT/CALL RATIO as a SIGNAL — multiple angles. We have pc_vol (put/call by volume), pc_oi (by open interest),
iv_skew, atm_iv in OptionSnapshot but ONLY ever used IV to price option structures — never tested put/call as
a SELECTION/sentiment signal. Test it here. ⚠️ options data ~2022-09..2026-08 (47mo, decent n for pc_vol; pc_oi sparse) → still single-provider, treat as scouting.
Angles (forward 1-mo MARKET-ADJUSTED return, month-end sampled):
  A. LEVEL quintiles of pc_vol / pc_oi / iv_skew (is high put/call contrarian-bullish or bearish?)
  B. EXTREME high pc_vol (top decile = fear/capitulation) — contrarian long?
  C. CHANGE (pc_vol rising vs falling MoM) — is it the move not the level?
  D. FLAGSHIP overlay — conditional lift on the cheapest-P/B value pick when pc signals fear.
-> BacktestResult[putcall]. Run: MSYS_NO_PATHCONV=1 docker exec rotation-backend-1 python -u /app/putcall_study.py
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


def _panel(field, midx):
    """Month-end OptionSnapshot field per ticker -> DataFrame [midx x ticker]."""
    from core.models import OptionSnapshot
    rows = list(OptionSnapshot.objects.exclude(**{f"{field}__isnull": True})
                .values_list("ticker", "date", field))
    if not rows:
        return pd.DataFrame(index=midx)
    df = pd.DataFrame(rows, columns=["ticker", "date", field])
    df["date"] = pd.to_datetime(df["date"])
    piv = df.pivot_table(index="date", columns="ticker", values=field, aggfunc="last")
    return piv.resample("ME").last().reindex(midx)


def _bucketed(sig, fret, qn=5):
    """avg market-adj fwd return + win + n per quantile bucket of sig; pooled across all stock-months."""
    s = sig.stack(); f = fret.stack()
    j = pd.concat([s.rename("sig"), f.rename("ret")], axis=1, join="inner").dropna()
    if len(j) < qn * 20:
        return None, len(j)
    try:
        j["q"] = pd.qcut(j["sig"], qn, labels=False, duplicates="drop")
    except Exception:
        return None, len(j)
    out = {}
    for q, g in j.groupby("q"):
        out[int(q)] = dict(avg=round(float(g["ret"].mean()) * 100, 2), win=round(float((g["ret"] > 0).mean()) * 100, 1),
                           n=int(len(g)), sig_med=round(float(g["sig"].median()), 3))
    return out, len(j)


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
    spy_ret = spy_m.pct_change().shift(-1)               # next-month SPY return, aligned to date
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
    fret = px.pct_change().shift(-1)                     # next-month stock return per name
    mkt_adj = fret.sub(spy_ret, axis=0)                  # market-adjusted next-month return

    print("loading OptionSnapshot put/call panels...", flush=True)
    pc_vol = _panel("pc_vol", midx).reindex(columns=common)
    pc_oi = _panel("pc_oi", midx).reindex(columns=common)
    iv_skew = _panel("iv_skew", midx).reindex(columns=common)
    win_months = pc_vol.dropna(how="all").index
    print(f"options window: {win_months.min().date() if len(win_months) else None} .. "
          f"{win_months.max().date() if len(win_months) else None} ({len(win_months)} months)", flush=True)

    res = {}
    # A. LEVEL quintiles
    for name, sig in [("pc_vol", pc_vol), ("pc_oi", pc_oi), ("iv_skew", iv_skew)]:
        b, n = _bucketed(sig, mkt_adj, 5)
        res[f"level_{name}"] = {"buckets": b, "n": n}
        if b:
            lo, hi = b[min(b)], b[max(b)]
            print(f"\n  LEVEL {name} quintiles (Q0=low..Q{max(b)}=high) market-adj fwd ret:", flush=True)
            for q in sorted(b):
                print(f"     Q{q} (med {b[q]['sig_med']}): {b[q]['avg']:>+6}%  win {b[q]['win']}%  n={b[q]['n']}", flush=True)
            print(f"     spread hi-lo: {round(hi['avg']-lo['avg'],2)}pp", flush=True)
    # B. EXTREME high pc_vol (top decile) vs rest
    s = pc_vol.stack(); f = mkt_adj.stack()
    j = pd.concat([s.rename("s"), f.rename("r")], axis=1, join="inner").dropna()
    if len(j) > 200:
        cut = j["s"].quantile(0.9)
        hi = j[j["s"] >= cut]["r"]; rest = j[j["s"] < cut]["r"]
        res["extreme_pcvol_top10"] = {"top_decile_avg": round(float(hi.mean())*100,2), "top_n": int(len(hi)),
                                      "rest_avg": round(float(rest.mean())*100,2), "spread_pp": round(float(hi.mean()-rest.mean())*100,2)}
        print(f"\n  EXTREME pc_vol top-decile (>= {round(cut,2)}): {round(float(hi.mean())*100,2)}% vs rest {round(float(rest.mean())*100,2)}% "
              f"(spread {round(float(hi.mean()-rest.mean())*100,2)}pp, n_top={len(hi)})", flush=True)
    # C. CHANGE (pc_vol MoM change) quintiles
    b, n = _bucketed(pc_vol.diff(), mkt_adj, 5)
    res["change_pc_vol"] = {"buckets": b, "n": n}
    if b:
        print(f"\n  CHANGE pc_vol (MoM) quintiles market-adj fwd ret:", flush=True)
        for q in sorted(b):
            print(f"     Q{q}: {b[q]['avg']:>+6}%  win {b[q]['win']}%  n={b[q]['n']}", flush=True)
    # D. FLAGSHIP overlay — conditional lift on the value pick when pc_vol is high (fear) vs low
    hi_rets, lo_rets = [], []
    for i in range(9, len(midx) - 1):
        date = midx[i]
        if date not in win_months:
            continue
        top = accel.loc[date].dropna().sort_values(ascending=False).head(10).index
        pcrow = pc_vol.loc[date]
        med = pcrow.median()
        for etf in top:
            _, holds = sector_map.get(etf, (etf, []))
            c = [h for h in holds if h in px.columns and _available_at(px[h], date) and pd.notna(pb.loc[date, h])
                 and pb.loc[date, h] > 0 and not bool(trap.loc[date, h])
                 and pd.notna(dvol.loc[date, h]) and dvol.loc[date, h] >= 5e6]
            g = [x for x in c if bool(low.loc[date, x])] or c
            if not g:
                continue
            pick = min(g, key=lambda h: pb.loc[date, h])
            r = fret.loc[date, pick]
            pcv = pcrow.get(pick)
            if pd.isna(r) or pd.isna(pcv) or pd.isna(med):
                continue
            (hi_rets if pcv >= med else lo_rets).append(float(r))
    fl = {"pick_high_pcvol": {"avg": round(float(np.mean(hi_rets))*100,2) if hi_rets else None, "n": len(hi_rets)},
          "pick_low_pcvol": {"avg": round(float(np.mean(lo_rets))*100,2) if lo_rets else None, "n": len(lo_rets)}}
    lift = (fl["pick_high_pcvol"]["avg"] - fl["pick_low_pcvol"]["avg"]) if (hi_rets and lo_rets) else None
    res["flagship_pick_split"] = {**fl, "high_minus_low_pp": lift}
    print(f"\n  FLAGSHIP value pick: high-pc_vol {fl['pick_high_pcvol']['avg']}% (n={len(hi_rets)}) vs "
          f"low {fl['pick_low_pcvol']['avg']}% (n={len(lo_rets)}) -> lift {lift}pp", flush=True)

    # verdict: is any angle a clean, sizeable (>1.5pp) monotone-ish signal?
    spreads = []
    for k in ("level_pc_vol", "level_pc_oi", "level_iv_skew", "change_pc_vol"):
        b = res.get(k, {}).get("buckets")
        if b:
            spreads.append((k, round(b[max(b)]["avg"] - b[min(b)]["avg"], 2)))
    strongest = max(spreads, key=lambda x: abs(x[1])) if spreads else (None, 0)
    payload = {
        "computed_at": pd.Timestamp.utcnow().isoformat(),
        "params": {"benchmark": BENCH, "options_window": f"{win_months.min().date()}..{win_months.max().date()}" if len(win_months) else None,
                   "n_option_months": int(len(win_months))},
        "results": res, "strongest_angle": {"angle": strongest[0], "hi_lo_spread_pp": strongest[1]},
        "verdict": (f"Strongest put/call angle = {strongest[0]} (hi-lo spread {strongest[1]}pp); extreme top-decile spread "
                    f"{res.get('extreme_pcvol_top10',{}).get('spread_pp')}pp; flagship lift {lift}pp. "
                    "⚠️ EXPLORATORY — 47mo options data (2022-09..2026-08, ~30k stock-months for pc_vol), spanning multiple regimes. "
                    "Suggestive at best; would need multi-year options history to validate as a live overlay."),
        "caveat": "Options data ~14mo single regime; monthly-sampled put/call; market-adjusted fwd returns. NOT validatable "
                  "on this window — a scouting pass to see if put/call deserves a longer-history data pull.",
    }
    return payload


def main():
    p = build()
    from pathlib import Path
    Path("/app/.data/studies/putcall.json").write_text(json.dumps(p, indent=2, default=str))
    try:
        from core.models import BacktestResult
        from django.utils import timezone
        BacktestResult.objects.update_or_create(
            kind="putcall", defaults={"payload": json.loads(json.dumps(p, default=str)), "computed_at": timezone.now()})
        print("Saved BacktestResult[putcall]", flush=True)
    except Exception as e:
        print("DB save failed:", e, flush=True)
    print("\n" + p["verdict"], flush=True)


if __name__ == "__main__":
    main()
